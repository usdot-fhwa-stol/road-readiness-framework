"""Minimal GT-guided lane-width stability (I4) for the tagged pilot subset.

The full engine in ``lane_width_stability.py`` gates so conservatively that it
returns *unavailable* for every image in the tagged subset. This module is the
deliberately minimal classical alternative used for the pilot demos:

1. GT lane boundaries (polylines) are interpolated as ``x(y)``.
2. Adjacent boundaries (sorted by x) with enough vertical overlap form a lane.
3. The horizontal width ``w(y)`` is sampled every few rows. Under perspective a
   constant-width lane has ``w(y)`` linear in ``y``, so a robust (least absolute
   deviation via iteratively reweighted fit) line is the expected profile.
4. Stability = ``1 - clip(median |residual| / (tolerance * expected), 0, 1)``:
   parallel boundaries score high; erratic or diverging boundaries score low.

The overlay draws both boundaries, one width tick per station — green where the
measured width matches the linear perspective model, red where it deviates —
and burns the score into the image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from evaluation.lane_continuity import prepare_guidance_lanes

EPS = 1e-8


@dataclass(frozen=True)
class SimpleWidthConfig:
    row_step_px: int = 10               # sample w(y) every this many rows
    min_overlap_rows: int = 60          # boundaries must share this much y-range
    min_pair_width_px: float = 25.0     # reject implausibly narrow pairings
    max_pair_width_frac: float = 0.7    # reject pairs wider than this * image width
    residual_tolerance: float = 0.22    # relative residual mapped to score 0
    bad_sample_rel_residual: float = 0.10  # tick drawn red above this


def _x_of_y(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ys, xs) sorted by y for interpolation."""

    order = np.argsort(points[:, 1])
    ys = points[order, 1]
    xs = points[order, 0]
    keep = np.concatenate([[True], np.diff(ys) > 1e-6])
    return ys[keep], xs[keep]


def analyze_lane_width_simple(
    lane_geometries: Optional[Iterable[Any]],
    image_shape: tuple[int, int],
    *,
    config: Optional[SimpleWidthConfig] = None,
) -> dict[str, Any]:
    """Compute the minimal I4 width-stability score from GT boundaries."""

    cfg = config or SimpleWidthConfig()
    height, width = int(image_shape[0]), int(image_shape[1])
    prepared = prepare_guidance_lanes(lane_geometries, image_shape)
    boundaries = []
    for lane in prepared:
        ys, xs = _x_of_y(np.asarray(lane["points"]))
        if ys.size >= 2 and (ys[-1] - ys[0]) >= cfg.min_overlap_rows:
            boundaries.append((ys, xs))
    # Sort left-to-right by x at each boundary's lowest (nearest) row.
    boundaries.sort(key=lambda b: b[1][-1])

    pairs = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        y0 = max(left[0][0], right[0][0])
        y1 = min(left[0][-1], right[0][-1])
        if y1 - y0 < cfg.min_overlap_rows:
            continue
        ys = np.arange(y0, y1, cfg.row_step_px, dtype=np.float64)
        xl = np.interp(ys, left[0], left[1])
        xr = np.interp(ys, right[0], right[1])
        w = xr - xl
        ok = (w > cfg.min_pair_width_px) & (w < cfg.max_pair_width_frac * width)
        if np.count_nonzero(ok) < 6:
            continue
        ys, xl, xr, w = ys[ok], xl[ok], xr[ok], w[ok]

        # Robust linear fit of w(y): IRLS with absolute-deviation weights.
        weights = np.ones_like(w)
        for _ in range(3):
            coeffs = np.polyfit(ys, w, 1, w=weights)
            residual = w - np.polyval(coeffs, ys)
            scale = max(float(np.median(np.abs(residual))), EPS)
            weights = 1.0 / np.maximum(np.abs(residual) / scale, 0.5)
        expected = np.polyval(coeffs, ys)
        rel_residual = np.abs(w - expected) / np.maximum(expected, EPS)
        med_rel = float(np.median(rel_residual))
        score = float(np.clip(1.0 - med_rel / cfg.residual_tolerance, 0.0, 1.0))
        pairs.append({
            "ys": ys, "xl": xl, "xr": xr, "w": w, "expected": expected,
            "rel_residual": rel_residual, "score": score,
            "median_rel_residual": med_rel,
            "support_rows": float(ys[-1] - ys[0]),
        })

    if not pairs:
        return {"I4": None, "score": None, "pairs": [], "boundaries": boundaries,
                "unavailable_reason": "no_adjacent_boundary_pair"}
    support = np.asarray([p["support_rows"] for p in pairs])
    support = support / support.sum()
    score = float(np.sum(support * np.asarray([p["score"] for p in pairs])))
    return {"I4": score, "score": score, "pairs": pairs, "boundaries": boundaries,
            "unavailable_reason": None}


def render_width_overlay(
    image_rgb: np.ndarray,
    analysis: dict[str, Any],
    *,
    title: str = "I4 lane width stability",
    config: Optional[SimpleWidthConfig] = None,
) -> np.ndarray:
    """Draw boundaries + width ticks (green = matches linear perspective model,
    red = deviates) with the score burned in. Returns BGR."""

    cfg = config or SimpleWidthConfig()
    bgr = cv2.cvtColor(
        np.asarray(image_rgb)[..., :3].astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR
    ).copy()

    for ys, xs in analysis.get("boundaries", []):
        pts = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(bgr, [pts], False, (255, 200, 40), 2, cv2.LINE_AA)

    for pair in analysis.get("pairs", []):
        for y, xl, xr, rel in zip(pair["ys"], pair["xl"], pair["xr"], pair["rel_residual"]):
            bad = rel > cfg.bad_sample_rel_residual
            colour = (40, 40, 230) if bad else (70, 200, 70)
            cv2.line(bgr, (int(xl), int(y)), (int(xr), int(y)), colour,
                     3 if bad else 1, cv2.LINE_AA)

    score = analysis.get("I4")
    label = "I4: N/A" if score is None else f"I4: {score:.3f}"
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 40), (25, 25, 25), -1)
    cv2.putText(bgr, f"{title}   {label}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, "boundaries=blue  width ok=green  width deviates=red",
                (10, bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


__all__ = ["SimpleWidthConfig", "analyze_lane_width_simple", "render_width_overlay"]
