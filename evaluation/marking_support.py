"""Marking-support representation and centerline localization metric (D3' basis).

This module implements the audit-approved replacement for the thick-stroke pixel
IoU used by the current D3/D4/D6. The construct measures *where recovered
marking evidence lies* against the reference, on a 1-px-centerline basis, in a
resolution-free canonical coordinate space — NOT how thick either raster stroke
happens to be.

Why this exists (see docs/d_metrics_audit.md, findings 1-2):
  * The live D3/D4/D6 dilate the prediction toward the GT stroke width
    (``d_metrics.standardize_stroke_width``) and then take pixel IoU. That
    inflates overlap for thin outputs and makes the CLRerNet>YOLOPX ordering an
    artifact of stroke handling. This module deliberately does NOT thicken.
  * The reference GT and each processor's output are reduced to centerline point
    sets, normalized, and compared by nearest-neighbor distance at a frozen
    tolerance ``delta``.

Semantic constraint (report p20-21): the points represent *visible longitudinal
lane-line marking evidence* recovered by a frozen reference processor, not lanes,
lane centers, corridors, or vehicle paths.

Coordinate canonicalization
---------------------------
A pixel ``(x, y)`` in an image of size ``W x H`` maps to
``p_bar = (x / D, y / D)`` with ``D = sqrt(W**2 + H**2)`` (the image diagonal).
Dividing BOTH axes by the same ``D`` makes the space:
  * resolution invariant  — scaling the image by ``k`` scales ``x, y, D`` by
    ``k``, leaving ``p_bar`` unchanged;
  * isotropic — a tolerance ``delta`` is the same physical fraction of the
    diagonal horizontally and vertically (unlike ``(x/W, y/H)``, which stretches
    distance differently per axis on non-square images).
So ``delta`` is a dimensionless *fraction of the image diagonal*.

``delta`` is a CALIBRATION parameter. It must be frozen once on a designated
dev/calibration split and then held fixed — never tuned on the evaluation set.
This module takes ``delta`` as an explicit argument and does not choose a value.

The distance math is pure NumPy and unit-tested. The mask -> centerline
extraction depends on scikit-image / OpenCV and is exercised on the inference
box; it is written against those documented APIs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Image-band layout (shared by D4' profile and legacy near-field D4)
# --------------------------------------------------------------------------- #
# Bands are fractions of image HEIGHT measured from the TOP. The lowest 5% of the
# image is deliberately NOT covered by any band: in many datasets the ego-vehicle
# hood occupies the very bottom rows, and marking evidence there is the hood, not
# the road. Excluding it keeps the "near" band a road region rather than sheet
# metal. This is a FIXED, UNCALIBRATED image-region assumption applied identically
# to every dataset and processor — the hood is not truly a fixed fraction across
# cameras, so this is a documented limitation, not a physical-distance claim.
HOOD_IGNORE_FRACTION = 0.05          # bottom 5% ignored (assumed ego-vehicle hood)
NEAR_BAND_FRACTION = 0.15            # the "near"/lower band spans 15% of height,
                                     # sitting ABOVE the hood: y/H in [0.80, 0.95)
# Full profile: upper 40% / middle 40% / lower(near) 15% / hood 5% (ignored).
BAND_EDGES: tuple[tuple[float, float], ...] = (
    (0.00, 0.40),   # upper
    (0.40, 0.80),   # middle
    (0.80, 0.95),   # lower / near (above hood)
)
BAND_LABELS: tuple[str, ...] = ("upper", "middle", "lower")


# --------------------------------------------------------------------------- #
# Coordinate canonicalization
# --------------------------------------------------------------------------- #
def diagonal(width: int, height: int) -> float:
    """Image diagonal ``D = sqrt(W**2 + H**2)`` used as the isotropic scale."""
    return float(np.hypot(float(width), float(height)))


def to_canonical(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map pixel ``(x, y)`` points to canonical ``(x/D, y/D)`` coordinates.

    ``points_xy`` is an ``(N, 2)`` array of pixel coordinates. Returns an
    ``(N, 2)`` float array; an empty input yields an empty ``(0, 2)`` array.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"expected (N,2) points, got shape {pts.shape}")
    d = diagonal(width, height)
    if d <= 0:
        raise ValueError("image diagonal must be positive")
    return pts[:, :2] / d


# --------------------------------------------------------------------------- #
# Nearest-neighbour distances (pure NumPy, unit-tested)
# --------------------------------------------------------------------------- #
def _min_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each point in ``a`` return the Euclidean distance to the nearest
    point in ``b``. ``a`` is ``(N,2)``, ``b`` is ``(M,2)``.

    Returns an ``(N,)`` array. If ``b`` is empty, returns ``+inf`` for every
    point in ``a`` (nothing to match against). If ``a`` is empty, returns an
    empty array.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if b.shape[0] == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float64)
    # Chunk over ``a`` to bound peak memory on dense masks (N,M can be large).
    out = np.empty((a.shape[0],), dtype=np.float64)
    chunk = max(1, int(2_000_000 // max(1, b.shape[0])))
    for start in range(0, a.shape[0], chunk):
        block = a[start:start + chunk]
        # (block, M) squared distances via (u-v)^2 = u^2 - 2uv + v^2
        d2 = (
            np.sum(block * block, axis=1)[:, None]
            - 2.0 * block @ b.T
            + np.sum(b * b, axis=1)[None, :]
        )
        np.maximum(d2, 0.0, out=d2)
        out[start:start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


# --------------------------------------------------------------------------- #
# D3' — marking-support localization (pure NumPy, unit-tested)
# --------------------------------------------------------------------------- #
def localization_scores(
    pred_canonical: np.ndarray,
    gt_canonical: np.ndarray,
    delta: float,
) -> dict:
    """Centerline localization precision/recall/F1 + localization error.

    Both inputs are canonical ``(N,2)`` point sets (see :func:`to_canonical`).
    ``delta`` is the frozen tolerance as a fraction of the image diagonal.

    Definitions (audit spec section 3):
      * recall    = fraction of GT points whose nearest prediction is within
                    ``delta``. Undefined (``None``) when GT is empty.
      * precision = fraction of predicted points whose nearest GT is within
                    ``delta``. Undefined (``None``) when the prediction is empty.
      * f1        = harmonic mean; ``None`` if either side is undefined.
      * loc_err_median / loc_err_p95 = median and 95th percentile of the GT->pred
                    nearest distances (in diagonal fractions). Undefined when GT
                    is empty. Distances are finite iff a prediction exists; when
                    the prediction is empty every GT distance is ``+inf`` and the
                    error percentiles are reported as ``None`` (unbounded) rather
                    than a fabricated large number.

    All ``None`` results are genuine "insufficient evidence" markers and must be
    excluded from aggregation, never coerced to 0.
    """
    if delta is None or not np.isfinite(delta) or delta < 0:
        raise ValueError(f"delta must be a finite non-negative float, got {delta!r}")

    pred = np.asarray(pred_canonical, dtype=np.float64).reshape(-1, 2) if np.size(pred_canonical) else np.zeros((0, 2))
    gt = np.asarray(gt_canonical, dtype=np.float64).reshape(-1, 2) if np.size(gt_canonical) else np.zeros((0, 2))

    n_pred, n_gt = pred.shape[0], gt.shape[0]

    gt_to_pred = _min_distances(gt, pred)      # (n_gt,)  +inf where pred empty
    pred_to_gt = _min_distances(pred, gt)      # (n_pred,) +inf where gt empty

    recall = None if n_gt == 0 else float(np.mean(gt_to_pred <= delta))
    precision = None if n_pred == 0 else float(np.mean(pred_to_gt <= delta))

    if recall is None or precision is None or (precision + recall) == 0:
        f1 = None if (recall is None or precision is None) else 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    if n_gt == 0:
        loc_err_median = loc_err_p95 = None
    elif not np.all(np.isfinite(gt_to_pred)):
        # Prediction empty -> distances unbounded; do not fabricate a magnitude.
        loc_err_median = loc_err_p95 = None
    else:
        loc_err_median = float(np.median(gt_to_pred))
        loc_err_p95 = float(np.percentile(gt_to_pred, 95))

    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "loc_err_median": loc_err_median,
        "loc_err_p95": loc_err_p95,
        "n_pred_points": int(n_pred),
        "n_gt_points": int(n_gt),
        "delta": float(delta),
    }


def band_localization(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    width: int,
    height: int,
    delta: float,
    bands: tuple[tuple[float, float], ...] = BAND_EDGES,
    band_labels: tuple[str, ...] = BAND_LABELS,
) -> dict:
    """Localization scores per horizontal image band (D4' readability profile).

    Points are split by their *original-pixel* ``y`` fraction (``y / H``) into the
    given bands BEFORE canonicalization, so a band is a fixed slice of the image
    regardless of resolution. Each band is scored independently by
    :func:`localization_scores`. This replaces the old "near-field IoU" — with no
    camera calibration in the repo, a band is an image region, NOT a physical
    distance (audit finding: no metric-distance claim is supportable).

    Default layout: upper ``[0.00, 0.40)`` / middle ``[0.40, 0.80)`` / lower/near
    ``[0.80, 0.95)``. The bottom 5% (``y/H >= 0.95``) is covered by NO band and is
    therefore excluded — it is the assumed ego-vehicle hood region (see
    ``HOOD_IGNORE_FRACTION``). Points at or below ``0.95`` fall through every
    ``[lo, hi)`` interval and contribute to no band.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2) if np.size(pred_xy) else np.zeros((0, 2))
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2) if np.size(gt_xy) else np.zeros((0, 2))
    result = {}
    for (lo, hi), label in zip(bands, band_labels):
        y_lo, y_hi = lo * height, hi * height
        p = pred_xy[(pred_xy[:, 1] >= y_lo) & (pred_xy[:, 1] < y_hi)] if pred_xy.shape[0] else pred_xy
        g = gt_xy[(gt_xy[:, 1] >= y_lo) & (gt_xy[:, 1] < y_hi)] if gt_xy.shape[0] else gt_xy
        result[label] = localization_scores(
            to_canonical(p, width, height),
            to_canonical(g, width, height),
            delta,
        )
    return result


# --------------------------------------------------------------------------- #
# Representation extraction
# --------------------------------------------------------------------------- #
def polyline_to_points(polyline_xy: np.ndarray, step_px: float = 1.0) -> np.ndarray:
    """Densify a polyline to ~``step_px``-spaced points (pure NumPy, tested).

    ``polyline_xy`` is an ordered ``(K,2)`` vertex array in pixel coords. Returns
    an ``(M,2)`` array of points sampled uniformly along the polyline so that a
    sparse row-anchored curve (CLRerNet) and a dense mask centerline (YOLOPX)
    become comparable point clouds at the same spatial density. This is the
    marking-support representation for a native polyline — the curve is NOT
    rasterized with a thick stroke.
    """
    pts = np.asarray(polyline_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pts.shape[0] == 1:
        return pts.copy()
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    if total <= 0:
        return pts[:1].copy()
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    n = max(2, int(np.ceil(total / max(step_px, 1e-9))) + 1)
    targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, cum, pts[:, 0])
    ys = np.interp(targets, cum, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def polylines_to_points(polylines, step_px: float = 1.0) -> np.ndarray:
    """Concatenate :func:`polyline_to_points` over a list of polylines.

    Each polyline may be an ``(K,2)`` array or a list of ``{"x","y"}`` dicts /
    ``(x,y)`` pairs. Returns a single ``(M,2)`` point cloud.
    """
    chunks = []
    for lane in polylines or []:
        arr = _coerce_polyline(lane)
        if arr.shape[0] >= 1:
            chunks.append(polyline_to_points(arr, step_px))
    if not chunks:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _coerce_polyline(lane) -> np.ndarray:
    if isinstance(lane, np.ndarray):
        return lane.reshape(-1, 2).astype(np.float64) if lane.size else np.zeros((0, 2))
    rows = []
    for point in lane or []:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        else:
            try:
                x, y = point[:2]
            except (TypeError, ValueError):
                continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xf) and np.isfinite(yf):
            rows.append([xf, yf])
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 2), dtype=np.float64)


def mask_to_points(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary marking mask to 1-px centerline points (x, y).

    This is the marking-support representation for a dense segmentation output
    (YOLOPX): the mask is thinned to a 1-px medial line so the comparison is
    placement, not the model's raster width. Requires scikit-image (preferred)
    or OpenCV ximgproc; raises ``RuntimeError`` if neither is available.

    NOTE: exercised on the inference box (this repo's minimal env has neither
    library). Written against the documented ``skimage.morphology.skeletonize``
    and ``cv2.ximgproc.thinning`` APIs.
    """
    m = (np.asarray(mask) > 0)
    if m.size == 0 or not m.any():
        return np.zeros((0, 2), dtype=np.float64)

    skel = None
    try:  # preferred: scikit-image
        from skimage.morphology import skeletonize as _sk
        skel = _sk(m)
    except Exception:  # noqa: BLE001 - fall back to OpenCV thinning
        try:
            import cv2
            thin = cv2.ximgproc.thinning((m.astype(np.uint8) * 255))
            skel = thin > 0
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "mask_to_points needs scikit-image or cv2.ximgproc for "
                f"skeletonization; neither is available ({exc!r})"
            )
    ys, xs = np.nonzero(skel)
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
