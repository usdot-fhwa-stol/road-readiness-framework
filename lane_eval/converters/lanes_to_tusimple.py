from __future__ import annotations
from typing import List

import numpy as np

# Sentinel for "no lane point at this h_sample row", matching the TuSimple
# convention used in the universal manifest format.
NO_POINT = -2


def make_h_samples(height: int, step: int = 10) -> List[int]:
    """Row coordinates at which lane x-positions are sampled.

    Increments of `step` pixels, skipping 0 and the image height itself, e.g.
    height=720 -> [10, 20, ..., 710]. This is the universal manifest grid.
    """
    return list(range(step, height, step))


def polylines_to_lane_json(
    lanes: List[np.ndarray],
    height: int,
    width: int,
    step: int = 10,
) -> dict:
    """Resample free-form lane polylines onto the fixed h_samples grid.

    Each lane is an [N, 2] (x, y) array. For every h_sample row inside the
    lane's vertical span the x is linearly interpolated; rows outside the span
    get NO_POINT (-2). Returns {"h_samples": [...], "lanes": [[x per row], ...]}
    in TuSimple format. The `width` argument is accepted for signature parity
    with the mask converter and to allow future out-of-bounds handling.
    """
    h_samples = make_h_samples(height, step)
    out_lanes: List[List[int]] = []

    for lane in lanes:
        pts = np.asarray(lane, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        # np.interp needs the sample points (ys) sorted ascending.
        order = np.argsort(ys)
        ys_s, xs_s = ys[order], xs[order]
        y_min, y_max = ys_s[0], ys_s[-1]

        row_xs: List[int] = []
        for y in h_samples:
            if y < y_min or y > y_max:
                row_xs.append(NO_POINT)
            else:
                row_xs.append(int(round(float(np.interp(y, ys_s, xs_s)))))
        # Drop lanes that ended up with no in-span samples (e.g. a near-horizontal
        # lane that falls between two grid rows).
        if any(x != NO_POINT for x in row_xs):
            out_lanes.append(row_xs)

    return {"h_samples": h_samples, "lanes": out_lanes}
