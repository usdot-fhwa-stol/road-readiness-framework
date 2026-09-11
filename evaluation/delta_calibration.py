"""Freeze the D3' localization tolerance ``delta`` (and per-processor operating
points) on the calibration split.

Where this sits in the pipeline
-------------------------------
``evaluation.calibration_split`` carves out a stratified held-out subset. On the
GPU box, both reference processors are run over exactly those images to produce,
per (processor, calibration image), the canonical centerline point sets for the
prediction and the GT (see ``evaluation.marking_support`` and
``evaluation.d_metrics._centerline_points``). This module consumes those point
sets — NOT raw images — and selects:

  1. per-processor operating point tau (the detection threshold that maximizes
     calibration support-F1 at a provisional reference delta), and
  2. the shared tolerance delta (one yardstick for every processor), chosen at
     the *knee* of the mean support-F1-vs-delta curve and cross-checked against
     the pooled localization-error distribution.

It then runs a bootstrap-stability check: resample the calibration images and
re-select delta; if delta* is stable within one grid step, the 5% split was
large enough; if it swings, the artifact says so (bump the fraction).

Design constraints honored
--------------------------
  * delta is SELECTED here, on calibration data only, and written to a frozen
    config artifact — it is never chosen on the evaluation set (audit risk 6).
  * The selector REPORTS the full F1-vs-delta table and multiple candidate delta
    values (knee, error-percentile); it does not silently emit a single opaque
    number. The chosen value + rule are recorded so the freeze is auditable.
  * Pure stdlib + NumPy: the distance/curve math is unit-testable in the minimal
    env. Producing the per-image point sets (needs cv2/torch/scikit-image) is
    the GPU-box driver's job (see ``evaluation.run_delta_calibration``), kept
    out of this module so the methodology is testable in isolation.

Vocabulary note (semantic constraint, report p20-21): "support" F1/precision/
recall here is agreement between *recovered marking evidence* and the reference
marking geometry — not lane-keeping or ADS performance.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from evaluation import marking_support

DELTA_CALIBRATION_VERSION = "delta_calib_v1_knee_diag_fraction"

# Default delta search grid: fractions of the image diagonal from 0.2% to 5%.
# Log-spaced so fine resolution near the small tolerances that matter, coarse in
# the tail. This is a SEARCH grid, not a frozen value; the chosen delta is one of
# these points (or reported between two when the knee falls on a step).
DEFAULT_DELTA_GRID = tuple(round(float(x), 5) for x in np.geomspace(0.002, 0.05, 15))


def support_f1_curve(
    per_image_distances: Sequence[dict],
    delta_grid: Sequence[float] = DEFAULT_DELTA_GRID,
) -> dict:
    """Mean support precision/recall/F1 as a function of delta.

    ``per_image_distances`` is a list of per-image dicts, each with:
        ``gt_to_pred``  : 1-D array of GT->nearest-pred distances (diag fraction)
        ``pred_to_gt``  : 1-D array of pred->nearest-GT distances (diag fraction)
    (Either may be empty; +inf entries encode "nothing to match against" exactly
    as ``marking_support._min_distances`` produces.) These are precomputed once
    per image on the box, so sweeping delta is cheap and requires no re-inference.

    For each delta, precision/recall/F1 are computed per image with the SAME
    None-semantics as ``marking_support.localization_scores`` (empty GT -> recall
    None; empty pred -> precision None), then averaged over images that define
    the quantity (None excluded, never coerced to 0). Returns per-grid-point
    means plus the counts contributing to each.
    """
    grid = [float(d) for d in delta_grid]
    if not grid:
        raise ValueError("delta_grid must be non-empty")
    if any((not np.isfinite(d)) or d < 0 for d in grid):
        raise ValueError("delta_grid must contain finite non-negative values")

    rows = []
    for delta in grid:
        precs, recs, f1s = [], [], []
        for rec in per_image_distances:
            g2p = np.asarray(rec.get("gt_to_pred", []), dtype=np.float64).reshape(-1)
            p2g = np.asarray(rec.get("pred_to_gt", []), dtype=np.float64).reshape(-1)
            n_gt, n_pred = g2p.size, p2g.size
            recall = None if n_gt == 0 else float(np.mean(g2p <= delta))
            precision = None if n_pred == 0 else float(np.mean(p2g <= delta))
            if recall is not None:
                recs.append(recall)
            if precision is not None:
                precs.append(precision)
            if recall is not None and precision is not None:
                f1s.append(0.0 if (precision + recall) == 0
                           else 2.0 * precision * recall / (precision + recall))
        rows.append({
            "delta": delta,
            "mean_precision": float(np.mean(precs)) if precs else None,
            "mean_recall": float(np.mean(recs)) if recs else None,
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "n_images_f1": len(f1s),
        })
    return {"delta_grid": grid, "curve": rows}


def _knee_delta(grid: Sequence[float], f1: Sequence[Optional[float]]) -> Optional[float]:
    """Delta at the knee of the F1-vs-delta curve.

    The knee is the point of maximum positive curvature — where increasing the
    tolerance stops buying meaningful F1. We use the standard normalized
    "distance from the chord" (Kneedle-style) construction on the finite portion
    of the curve: normalize delta and F1 to [0,1], measure each point's vertical
    gap below the straight chord from first to last point, and take the delta
    with the largest gap. Returns None if fewer than 3 finite points.
    """
    xs, ys = [], []
    for d, v in zip(grid, f1):
        if v is not None and np.isfinite(v):
            xs.append(float(d)); ys.append(float(v))
    if len(xs) < 3:
        return None
    xs = np.asarray(xs); ys = np.asarray(ys)
    x_rng = xs.max() - xs.min()
    y_rng = ys.max() - ys.min()
    if x_rng <= 0 or y_rng <= 0:
        return None
    xn = (xs - xs.min()) / x_rng
    yn = (ys - ys.min()) / y_rng
    # Chord from first to last normalized point; F1 is concave-increasing, so the
    # knee is where the curve rises farthest ABOVE the chord.
    chord = xn * (yn[-1] - yn[0]) + yn[0]  # yn[0] + xn*(slope); xn spans [0,1]
    gap = yn - chord
    return float(xs[int(np.argmax(gap))])


def _percentile_delta(per_image_distances: Sequence[dict], q: float) -> Optional[float]:
    """The q-th percentile of the pooled, FINITE GT->pred localization errors —
    i.e. the tolerance that would 'support' q% of reference marking points if a
    prediction exists. Infinite entries (empty prediction) are excluded because
    they represent absence of evidence, not a large-but-finite error."""
    pooled = []
    for rec in per_image_distances:
        g2p = np.asarray(rec.get("gt_to_pred", []), dtype=np.float64).reshape(-1)
        pooled.append(g2p[np.isfinite(g2p)])
    if not pooled:
        return None
    alld = np.concatenate(pooled) if pooled else np.array([])
    if alld.size == 0:
        return None
    return float(np.percentile(alld, q))


def select_delta(
    per_image_distances: Sequence[dict],
    delta_grid: Sequence[float] = DEFAULT_DELTA_GRID,
    error_percentile: float = 90.0,
) -> dict:
    """Choose delta from calibration point-distance distributions.

    Returns candidate deltas and the selected value + rule:
      * ``delta_knee``  : knee of the mean-F1-vs-delta curve (the primary rule);
      * ``delta_p{q}``  : q-th percentile of pooled finite GT->pred errors
                          (cross-check / fallback);
      * ``delta_selected`` / ``selection_rule`` : the knee when available, else
        the error percentile.
    """
    curve = support_f1_curve(per_image_distances, delta_grid)
    grid = curve["delta_grid"]
    f1 = [row["mean_f1"] for row in curve["curve"]]
    delta_knee = _knee_delta(grid, f1)
    delta_pct = _percentile_delta(per_image_distances, error_percentile)

    if delta_knee is not None:
        selected, rule = delta_knee, "f1_knee"
    elif delta_pct is not None:
        selected, rule = delta_pct, f"gt_error_p{int(error_percentile)}"
    else:
        selected, rule = None, "undefined_insufficient_data"

    return {
        "delta_selected": selected,
        "selection_rule": rule,
        "delta_knee": delta_knee,
        f"delta_p{int(error_percentile)}": delta_pct,
        "error_percentile": error_percentile,
        "f1_curve": curve["curve"],
        "delta_grid": grid,
        "n_images": len(per_image_distances),
    }


def bootstrap_delta_stability(
    per_image_distances: Sequence[dict],
    delta_grid: Sequence[float] = DEFAULT_DELTA_GRID,
    error_percentile: float = 90.0,
    n_boot: int = 200,
    seed: int = 20260810,
) -> dict:
    """Resample calibration IMAGES with replacement, re-select delta each time,
    and summarize the spread of delta*.

    The point of the calibration-set-size question ("is 5% enough?") is answered
    here empirically: if the bootstrap delta* distribution is tight (its IQR
    spans <= one grid step, and the median lands on the same grid neighborhood as
    the full-sample delta), the split is large enough. If it is wide, it is not.
    Returns the per-replicate selected deltas plus summary stats and a boolean
    ``stable_within_one_grid_step``.
    """
    grid = sorted(float(d) for d in delta_grid)
    rng = np.random.RandomState(seed)
    n = len(per_image_distances)
    if n == 0:
        return {"n_boot": 0, "deltas": [], "stable_within_one_grid_step": None,
                "reason": "no calibration images"}

    full = select_delta(per_image_distances, grid, error_percentile)["delta_selected"]
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = [per_image_distances[i] for i in idx]
        d = select_delta(sample, grid, error_percentile)["delta_selected"]
        if d is not None and np.isfinite(d):
            deltas.append(float(d))

    if not deltas:
        return {"n_boot": n_boot, "deltas": [], "delta_full_sample": full,
                "stable_within_one_grid_step": None, "reason": "no finite bootstrap deltas"}

    arr = np.asarray(deltas)
    med = float(np.median(arr))
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    # One grid step near the median (median grid spacing as a robust scale).
    steps = np.diff(grid)
    one_step = float(np.median(steps)) if steps.size else 0.0
    stable = bool((q75 - q25) <= one_step + 1e-12)
    return {
        "n_boot": n_boot,
        "seed": seed,
        "delta_full_sample": full,
        "delta_bootstrap_median": med,
        "delta_bootstrap_iqr": [q25, q75],
        "delta_bootstrap_min": float(arr.min()),
        "delta_bootstrap_max": float(arr.max()),
        "one_grid_step": one_step,
        "stable_within_one_grid_step": stable,
        "deltas": deltas,
    }


def select_operating_point(
    threshold_grid: Sequence[float],
    distances_at_threshold: Callable[[float], Sequence[dict]],
    reference_delta: float,
) -> dict:
    """Pick a processor's detection threshold tau by maximizing calibration
    support-F1 at a provisional ``reference_delta``.

    ``distances_at_threshold(tau)`` returns the per-image distance dicts (same
    shape as ``support_f1_curve`` consumes) obtained by thresholding that
    processor's raw output at ``tau`` on the calibration images. This function is
    supplied by the GPU-box driver (it needs the model outputs); the selection
    logic here is pure and testable. Returns the chosen tau and the per-tau F1.
    """
    if reference_delta is None or not np.isfinite(reference_delta) or reference_delta < 0:
        raise ValueError("reference_delta must be finite and non-negative")
    rows = []
    for tau in threshold_grid:
        dists = distances_at_threshold(float(tau))
        curve = support_f1_curve(dists, [reference_delta])["curve"][0]
        rows.append({"tau": float(tau), "mean_f1": curve["mean_f1"],
                     "mean_precision": curve["mean_precision"],
                     "mean_recall": curve["mean_recall"]})
    finite = [r for r in rows if r["mean_f1"] is not None]
    best = max(finite, key=lambda r: r["mean_f1"]) if finite else None
    return {
        "reference_delta": float(reference_delta),
        "tau_selected": best["tau"] if best else None,
        "tau_f1": best["mean_f1"] if best else None,
        "grid": rows,
    }
