"""Write model predictions in the universal manifest format.

This is the output side of the shared, model-agnostic manifest adapter (used by
both YOLOPX and HybridNets). For each processed image it
records a predicted lane-mask PNG link and a compatible lane_json. Native float
polylines are retained when a model provides them; mask-only callers continue
to derive lane_json through the legacy converter.

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

from ..converters import mask_to_lane_json, polylines_to_lane_json


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

    @staticmethod
    def _serialize_polylines(polylines) -> tuple[list[list[dict]], list[np.ndarray]]:
        serialized: list[list[dict]] = []
        arrays: list[np.ndarray] = []
        source = [] if polylines is None else polylines
        for lane in source:
            rows = []
            for point in lane:
                if isinstance(point, dict):
                    x, y = point.get("x"), point.get("y")
                else:
                    try:
                        x, y = point[:2]
                    except (TypeError, ValueError):
                        continue
                try:
                    x_float, y_float = float(x), float(y)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(x_float) and np.isfinite(y_float):
                    rows.append({"x": x_float, "y": y_float})
            if len(rows) >= 2:
                serialized.append(rows)
                arrays.append(np.asarray([[p["x"], p["y"]] for p in rows], dtype=np.float64))
        return serialized, arrays

    def add(
        self,
        sample_id,
        image_path,
        pred_mask: np.ndarray,
        lane_json=None,
        *,
        polylines=None,
        prediction_meta=None,
    ) -> None:
        """Add one prediction while preserving the original three-argument API."""

        pred = (np.asarray(pred_mask) > 0).astype(np.uint8)
        h, w = pred.shape[:2]

        mask_path = None
        if self.save_masks:
            mask_path = str(self.mask_dir / f"{_safe(sample_id)}.png")
            cv2.imwrite(mask_path, pred * 255)

        serialized_polylines, polyline_arrays = self._serialize_polylines(polylines)
        if serialized_polylines:
            geometry_source = "native_polyline"
            if lane_json is None:
                lane_json = polylines_to_lane_json(
                    polyline_arrays,
                    height=h,
                    width=w,
                    step=self.h_step,
                )
        elif lane_json is not None:
            geometry_source = "lane_json"
        else:
            geometry_source = "mask_derived"
            lane_json = mask_to_lane_json(pred, step=self.h_step)

        prediction = {
            "mask_path": mask_path,
            "lane_json": lane_json,
            "geometry_source": geometry_source,
        }
        if serialized_polylines:
            prediction["polylines"] = serialized_polylines
        if prediction_meta:
            prediction["meta"] = dict(prediction_meta)
        self.samples.append({
            "sample_id": sample_id,
            "image_path": image_path,
            "width": w,
            "height": h,
            "prediction": prediction,
        })

    def write(self) -> Path:
        manifest = {
            "metadata": {
                "model": self.model,
                "dataset": self.dataset,
                "num_samples": len(self.samples),
                "prediction_manifest_version": 2,
            },
            "samples": self.samples,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(manifest, f)
        return self.out_path
