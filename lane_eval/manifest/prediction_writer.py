"""Write model predictions in the universal manifest format.

This is the output side of the shared, model-agnostic manifest adapter (used by
both YOLOPX and HybridNets). For each processed image it
records both prediction representations the manifest spec asks for: a predicted
lane-mask PNG link and a predicted lane_json (derived from the mask via the
existing mask_to_lanes converter — no new geometry).

Output JSON mirrors the input manifest:
    {
      "metadata": {"model": ..., "dataset": ..., "num_samples": N},
      "samples": [
        {"sample_id", "image_path", "width", "height",
         "prediction": {"mask_path": ..., "lane_json": {"h_samples", "lanes"}}}
      ]
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..converters import mask_to_lane_json


def _safe(name) -> str:
    return str(name).replace("/", "_").replace("\\", "_")


class PredictionManifestWriter:
    def __init__(self, out_path: str, model: str, dataset: str,
                 h_step: int = 10, save_masks: bool = True):
        self.out_path = Path(out_path)
        self.mask_dir = self.out_path.parent / "masks"
        self.model = model
        self.dataset = dataset
        self.h_step = h_step
        self.save_masks = save_masks
        if save_masks:
            self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.samples: list[dict] = []

    def add(self, sample_id, image_path, pred_mask: np.ndarray) -> None:
        pred = (np.asarray(pred_mask) > 0).astype(np.uint8)
        h, w = pred.shape[:2]

        mask_path = None
        if self.save_masks:
            mask_path = str(self.mask_dir / f"{_safe(sample_id)}.png")
            cv2.imwrite(mask_path, pred * 255)

        lane_json = mask_to_lane_json(pred, step=self.h_step)
        self.samples.append({
            "sample_id": sample_id,
            "image_path": image_path,
            "width": w,
            "height": h,
            "prediction": {"mask_path": mask_path, "lane_json": lane_json},
        })

    def write(self) -> Path:
        manifest = {
            "metadata": {
                "model": self.model,
                "dataset": self.dataset,
                "num_samples": len(self.samples),
            },
            "samples": self.samples,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(manifest, f)
        return self.out_path
