"""CULane native lane metric: F1 from line-IoU matching at a threshold.

Implements the standard CULane evaluation (originally the C++ tool from
https://github.com/XingangPan/SCNN; this matches the widely-used Python port in
LaneATT/CLRNet `culane_metric.py`). Each lane is rasterised as a fixed-width
(default 30px) polyline; predicted and GT lanes are matched by maximum IoU via
the Hungarian algorithm; a match with IoU >= threshold (default 0.5) is a TP.
Precision/recall/F1 are aggregated over the dataset from total TP/FP/FN.

Operates on the universal lane_json representation, converting each lane's
sampled x-positions back into (x, y) polyline points before rasterising.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

LANE_WIDTH = 30        # rasterised lane thickness in px (official CULane setting)
IOU_THRESHOLD = 0.5    # IoU at/above which a matched pair counts as a TP


def _lane_json_to_points(lane_json: dict) -> List[np.ndarray]:
    """Convert {"h_samples", "lanes"} into a list of [N, 2] (x, y) polylines.
    Drops -2 (missing) points; keeps only lanes with >= 2 valid points."""
    h = lane_json["h_samples"]
    out: List[np.ndarray] = []
    for xs in lane_json["lanes"]:
        pts = [[float(x), float(y)] for x, y in zip(xs, h) if x != -2]
        if len(pts) >= 2:
            out.append(np.asarray(pts, dtype=np.float32))
    return out


def _draw_lane(lane: np.ndarray, img_shape, width: int) -> np.ndarray:
    img = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = lane.astype(np.int32)
    for p1, p2 in zip(pts[:-1], pts[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=1, thickness=width)
    return img > 0


def _discrete_cross_iou(preds, gts, img_shape, width) -> np.ndarray:
    pred_masks = [_draw_lane(p, img_shape, width) for p in preds]
    gt_masks = [_draw_lane(g, img_shape, width) for g in gts]
    ious = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            union = (pm | gm).sum()
            ious[i, j] = (pm & gm).sum() / union if union > 0 else 0.0
    return ious


def culane_tp_fp_fn(preds, gts, img_shape, width=LANE_WIDTH,
                    iou_threshold=IOU_THRESHOLD) -> tuple[int, int, int]:
    """Per-image (tp, fp, fn) via Hungarian matching on line-IoU."""
    if len(preds) == 0:
        return 0, 0, len(gts)
    if len(gts) == 0:
        return 0, len(preds), 0
    ious = _discrete_cross_iou(preds, gts, img_shape, width)
    row, col = linear_sum_assignment(1 - ious)
    tp = int((ious[row, col] >= iou_threshold).sum())
    return tp, len(preds) - tp, len(gts) - tp


class CULaneNativeEvaluator:
    """Accumulate CULane line-IoU F1 over images.

    Usage:
        ev = CULaneNativeEvaluator()
        ev.update(pred_lane_json, gt_lane_json, height, width)
        results = ev.compute()   # {precision, recall, f1, tp, fp, fn, num_images}
    """

    def __init__(self, lane_width: int = LANE_WIDTH, iou_threshold: float = IOU_THRESHOLD):
        self.lane_width = lane_width
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._n = 0

    def update(self, pred_lane_json: dict, gt_lane_json: dict, height: int, width: int) -> None:
        preds = _lane_json_to_points(pred_lane_json)
        gts = _lane_json_to_points(gt_lane_json)
        tp, fp, fn = culane_tp_fp_fn(
            preds, gts, (height, width), self.lane_width, self.iou_threshold
        )
        self._tp += tp
        self._fp += fp
        self._fn += fn
        self._n += 1

    def compute(self) -> dict:
        tp, fp, fn = self._tp, self._fp, self._fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "num_images": self._n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
