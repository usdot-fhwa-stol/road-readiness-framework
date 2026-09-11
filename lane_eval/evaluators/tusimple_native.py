"""TuSimple native lane metric: accuracy + FP + FN via horizontal sampling.

Faithful reimplementation of the official TuSimple benchmark metric
(https://github.com/TuSimple/tusimple-benchmark, `evaluate/lane.py::LaneEval`).
The only deviation is using np.polyfit for the lane-angle slope instead of
sklearn's LinearRegression, which is numerically equivalent and avoids the
extra dependency.

Operates on the universal lane_json representation ({"h_samples", "lanes"}),
so both GT (native TuSimple) and predicted lanes (derived from the YOLOPX mask)
plug straight in. Predictions are resampled onto the GT h_samples if the two
grids differ.
"""
from __future__ import annotations

from typing import List

import numpy as np

PIXEL_THRESH = 20      # base horizontal tolerance in pixels (official)
PT_THRESH = 0.85       # a GT lane counts as matched if its best acc >= this


def _get_angle(xs: np.ndarray, y_samples: np.ndarray) -> float:
    """Lane angle from valid (x, y) points; 0 if fewer than 2 points."""
    valid = xs >= 0
    xv, yv = xs[valid], y_samples[valid]
    if len(xv) > 1:
        k = np.polyfit(yv, xv, 1)[0]   # slope dx/dy (== sklearn LinearRegression coef)
        return float(np.arctan(k))
    return 0.0


def _line_accuracy(pred: np.ndarray, gt: np.ndarray, thresh: float) -> float:
    pred = np.array([p if p >= 0 else -100 for p in pred], dtype=float)
    gt = np.array([g if g >= 0 else -100 for g in gt], dtype=float)
    return float(np.sum(np.abs(pred - gt) < thresh) / len(gt))


def _resample(xs: List[float], src_h: List[int], dst_h: List[int]) -> np.ndarray:
    """Resample a lane's x-positions from src_h onto dst_h (-2 outside span)."""
    xs = np.asarray(xs, dtype=float)
    valid = xs >= 0
    if valid.sum() < 2:
        return np.full(len(dst_h), -2.0)
    sy = np.asarray(src_h, dtype=float)[valid]
    sx = xs[valid]
    order = np.argsort(sy)
    sy, sx = sy[order], sx[order]
    out = []
    for y in dst_h:
        out.append(float(np.interp(y, sy, sx)) if sy[0] <= y <= sy[-1] else -2.0)
    return np.array(out)


def bench(pred: List[np.ndarray], gt: List[np.ndarray], y_samples: List[int]) -> tuple[float, float, float]:
    """Official per-image (accuracy, fp, fn). pred/gt are lists of x-arrays
    aligned to y_samples; -2 marks a missing point."""
    if len(gt) + 2 < len(pred):
        return 0.0, 0.0, 1.0
    ys = np.asarray(y_samples, dtype=float)
    angles = [_get_angle(np.asarray(x_gts, dtype=float), ys) for x_gts in gt]
    threshs = [PIXEL_THRESH / np.cos(a) for a in angles]

    line_accs: List[float] = []
    matched = 0.0
    for x_gts, thresh in zip(gt, threshs):
        accs = [_line_accuracy(np.asarray(x_pred, dtype=float), np.asarray(x_gts, dtype=float), thresh)
                for x_pred in pred]
        max_acc = max(accs) if accs else 0.0
        if max_acc >= PT_THRESH:
            matched += 1
        line_accs.append(max_acc)

    fp = (len(pred) - matched) / len(pred) if len(pred) > 0 else 0.0
    fn = (len(gt) - matched)
    if len(gt) > 4 and fn > 0:
        fn -= 1
    s = sum(line_accs)
    if len(gt) > 4:
        s -= min(line_accs)
    acc = s / max(min(4.0, len(gt)), 1.0)
    fn = fn / max(min(len(gt), 4.0), 1.0)
    return acc, fp, fn


class TuSimpleNativeEvaluator:
    """Accumulate the TuSimple accuracy/FP/FN metric over images.

    Usage:
        ev = TuSimpleNativeEvaluator()
        ev.update(pred_lane_json, gt_lane_json)
        results = ev.compute()   # {accuracy, fp, fn, num_images}
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._acc = 0.0
        self._fp = 0.0
        self._fn = 0.0
        self._n = 0

    def update(self, pred_lane_json: dict, gt_lane_json: dict) -> None:
        gt_h = gt_lane_json["h_samples"]
        gt_lanes = [np.asarray(l, dtype=float) for l in gt_lane_json["lanes"]]

        pred_h = pred_lane_json["h_samples"]
        if pred_h == gt_h:
            pred_lanes = [np.asarray(l, dtype=float) for l in pred_lane_json["lanes"]]
        else:
            pred_lanes = [_resample(l, pred_h, gt_h) for l in pred_lane_json["lanes"]]

        acc, fp, fn = bench(pred_lanes, gt_lanes, gt_h)
        self._acc += acc
        self._fp += fp
        self._fn += fn
        self._n += 1

    def compute(self) -> dict:
        n = max(self._n, 1)
        return {
            "num_images": self._n,
            "accuracy": self._acc / n,
            "fp": self._fp / n,
            "fn": self._fn / n,
        }
