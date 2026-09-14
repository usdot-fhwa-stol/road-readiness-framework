"""Per-image lane metrics + JSON/CSV writers, shared by the YOLOPX and HybridNets
runners.

Each image is scored with a fresh YOLOPX ``SegmentationMetric`` (a 2x2 confusion
matrix) — the SAME metric used for the aggregate — so per-image and aggregate
numbers are computed identically. No new metric math.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PER_IMAGE_FIELDS = [
    "model", "dataset", "split", "image_id", "image_path",
    "gt_px", "pred_px", "lane_iou", "lane_f1", "lane_precision", "lane_recall",
]


def per_image_scores(pred_mask, gt_mask) -> dict:
    """IoU / F1 / Precision / Recall for one image. ``pred_mask`` and ``gt_mask``
    must already be the same HxW shape (resize the GT to the prediction first,
    exactly as the aggregate path does). Requires the YOLOPX repo on sys.path
    (the runners add it before calling this)."""
    from lib.core.evaluate import SegmentationMetric

    pred = (np.asarray(pred_mask) > 0).astype(np.int64)
    gt = (np.asarray(gt_mask) > 0).astype(np.int64)
    m = SegmentationMetric(2)
    m.addBatch(pred, gt)
    recall = float(m.lineAccuracy())                 # TP / (TP + FN)
    precision = float(m.classPixelAccuracy()[1])     # TP / (TP + FP)
    f1 = float(2 * precision * recall / (precision + recall + 1e-12))
    return {
        "gt_px": int(gt.sum()),
        "pred_px": int(pred.sum()),
        "lane_iou": float(m.IntersectionOverUnion()),
        "lane_f1": f1,
        "lane_precision": precision,
        "lane_recall": recall,
    }


def per_image_paths(output_path: str) -> Tuple[str, str]:
    """Derive (json, csv) paths in a ``per_image/`` subdir next to the aggregate
    output — e.g. outputs/results/yolopx_bdd100k_lane.json ->
    outputs/results/per_image/yolopx_bdd100k_lane_per_image.{json,csv}."""
    out = Path(output_path)
    d = out.parent / "per_image"
    return str(d / f"{out.stem}_per_image.json"), str(d / f"{out.stem}_per_image.csv")


def write_per_image(records: List[dict], json_path: Optional[str], csv_path: Optional[str]) -> None:
    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(records))   # compact: one record array, can be large (CULane ~34k)
    if csv_path and records:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fields = [k for k in PER_IMAGE_FIELDS if k in records[0]]
        fields += [k for k in records[0] if k not in fields]  # any extra keys
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
