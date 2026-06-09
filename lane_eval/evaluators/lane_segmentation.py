"""Thin wrapper around YOLOPX's SegmentationMetric.

We deliberately do NOT reimplement the metric math.  Instead we import
SegmentationMetric directly from the YOLOPX repo and expose it through a
small adapter that matches our LanePrediction / LaneTarget types.

The metrics reported are the same ones in the YOLOPX paper:
  lineAccuracy  → "Acc"   (recall of the lane class)
  IoU           → lane-class IoU
  mIoU          → mean IoU over {background, lane}
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np


class LaneSegmentationEvaluator:
    """Wraps YOLOPX's SegmentationMetric for lane-only evaluation.

    Usage
    -----
    evaluator = LaneSegmentationEvaluator(yolopx_repo="/path/to/YOLOPX")
    evaluator.reset()
    for pred_mask, gt_mask in ...:
        evaluator.update(pred_mask, gt_mask)
    results = evaluator.compute()
    """

    def __init__(self, yolopx_repo: str):
        if yolopx_repo not in sys.path:
            sys.path.insert(0, yolopx_repo)
        from lib.core.evaluate import SegmentationMetric  # noqa: F401 — YOLOPX import
        self._SegmentationMetric = SegmentationMetric
        self._metric = SegmentationMetric(2)  # 2 classes: background + lane
        self._n = 0

    def reset(self) -> None:
        self._metric.reset()
        self._n = 0

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        """Add one image pair.

        Both masks must be uint8 / int arrays with values in {0, 1} and the
        same shape.  Raises ValueError on shape mismatch.
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}. "
                "Resize prediction to match GT before calling update()."
            )
        pred = (pred_mask > 0).astype(np.int64)
        gt   = (gt_mask   > 0).astype(np.int64)
        self._metric.addBatch(pred, gt)
        self._n += 1

    def compute(self) -> dict:
        """Return aggregated metrics dict (matching YOLOPX paper terminology)."""
        acc   = float(self._metric.lineAccuracy())
        iou   = float(self._metric.IntersectionOverUnion())
        miou  = float(self._metric.meanIntersectionOverUnion())
        pixel_acc = float(self._metric.pixelAccuracy())
        return {
            "num_images": self._n,
            "lane_accuracy": acc,      # YOLOPX "Acc" — recall of lane class
            "lane_iou": iou,           # YOLOPX "IoU"
            "lane_miou": miou,         # YOLOPX "mIoU"
            "pixel_accuracy": pixel_acc,
        }
