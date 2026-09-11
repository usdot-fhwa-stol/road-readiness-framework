"""Contour/component-based extraction of lane polylines from a binary mask.

Used to derive an approximate TuSimple-style `lane_json` for datasets whose
ground truth is provided only as a rasterised mask (BDD100K, CULane seg). Each
connected component is treated as one lane instance and sampled at the fixed
h_samples grid by taking the mean x of its pixels in each row.
"""
from __future__ import annotations

import cv2
import numpy as np

from .lanes_to_tusimple import make_h_samples, NO_POINT


def mask_to_lane_json(
    mask: np.ndarray,
    step: int = 10,
    min_points: int = 2,
    min_area: int = 20,
) -> dict:
    """Approximate a TuSimple-style lane_json from a binary lane mask.

    `mask` is [H, W] with nonzero = lane pixels (any dtype). h_samples are
    derived from the mask height so they always match the stored mask. Lanes are
    ordered left-to-right by mean x. Components smaller than `min_area` pixels or
    with fewer than `min_points` sampled rows are dropped as noise.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height = binary.shape[0]
    h_samples = make_h_samples(height, step)

    num_labels, labels = cv2.connectedComponents(binary)

    lanes: list[list[int]] = []
    for lab in range(1, num_labels):  # 0 is background
        comp = labels == lab
        if comp.sum() < min_area:
            continue
        row_xs: list[int] = []
        valid = 0
        for y in h_samples:
            cols = np.flatnonzero(comp[y])
            if cols.size:
                row_xs.append(int(round(float(cols.mean()))))
                valid += 1
            else:
                row_xs.append(NO_POINT)
        if valid >= min_points:
            lanes.append(row_xs)

    # Order lanes left-to-right by their average sampled x (TuSimple convention).
    def _mean_x(row_xs: list[int]) -> float:
        vals = [x for x in row_xs if x != NO_POINT]
        return float(np.mean(vals)) if vals else float("inf")

    lanes.sort(key=_mean_x)

    return {"h_samples": h_samples, "lanes": lanes}
