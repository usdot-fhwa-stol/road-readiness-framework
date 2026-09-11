from __future__ import annotations
from typing import List

import cv2
import numpy as np


def lanes_to_mask(
    lanes: List[np.ndarray],
    height: int,
    width: int,
    thickness: int = 16,
    value: int = 1,
) -> np.ndarray:
    """Rasterise lane polylines into a binary uint8 mask [H, W].

    Each lane is an [N, 2] float array of (x, y) image coords.
    Points outside image bounds are clipped so cv2 does not crash.
    Lanes with fewer than 2 valid points are skipped.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for lane in lanes:
        pts = np.asarray(lane, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        valid = (
            (pts[:, 0] > -width) & (pts[:, 0] < 2 * width) &
            (pts[:, 1] > -height) & (pts[:, 1] < 2 * height)
        )
        pts = pts[valid]
        if len(pts) < 2:
            continue
        pts_clipped = np.clip(pts, [0, 0], [width - 1, height - 1])
        pts_int = pts_clipped.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts_int], isClosed=False, color=value, thickness=thickness)
    return mask


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert any nonzero pixels to 1, return uint8."""
    return (mask > 0).astype(np.uint8)
