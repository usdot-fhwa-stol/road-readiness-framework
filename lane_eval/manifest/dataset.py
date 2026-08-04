"""Read a universal manifest JSON and present it as a dataset adapter.

This is the input side of the shared, model-agnostic manifest adapter: it
replaces the native dataset adapters so inference consumes the dataset-agnostic
manifest produced by lane_eval.manifest.generator. It yields the same LaneSample
objects the rest of the eval pipeline expects, so it slots directly into the
eval runners' _EvalDataset — both run_lane_eval (YOLOPX) and run_hybridnets
(HybridNets) consume it identically via --manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..schema.sample import LaneSample
from ..schema.lane import LanePrediction, LaneTarget


_PATTERN_FIELDS = {
    "style",
    "pattern",
    "lane_style",
    "line_style",
    "marking_type",
    "line_type",
}


class ManifestDataset:
    """Adapter-compatible reader over a manifest_<dataset>.json file.

    GT masks are loaded from each sample's ground_truth.mask_path. The GT
    lane_json and natural_gt indicator are carried in target.meta so a
    lane-point evaluator can use them later.
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path) as f:
            data = json.load(f)
        self.metadata = data.get("metadata", {})
        self.name = self.metadata.get("dataset", self.manifest_path.stem)
        self._samples = data["samples"]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> LaneSample:
        s = self._samples[idx]
        gt = s.get("ground_truth", {})

        mask = None
        mask_path = gt.get("mask_path")
        # Resolve relative mask paths against the manifest's own directory so the
        # manifest can be read from any working directory. (Generated manifests
        # now store absolute paths, so this is a no-op for them — but it keeps
        # any relative-path manifest robust.)
        resolved_mask_path = None
        if mask_path:
            p = Path(mask_path)
            resolved_mask_path = p if p.is_absolute() else (self.manifest_path.parent / p)
            if resolved_mask_path.exists():
                m = cv2.imread(str(resolved_mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mask = (m > 0).astype(np.uint8)

        target_meta = {
            "lane_json": gt.get("lane_json"),
            "natural_gt": gt.get("natural_gt"),
        }
        explicit_meta = gt.get("meta") if isinstance(gt.get("meta"), dict) else {}
        for key in _PATTERN_FIELDS:
            if key in gt:
                target_meta[key] = gt[key]
            elif key in explicit_meta:
                target_meta[key] = explicit_meta[key]

        return LaneSample(
            image_id=s["sample_id"],
            image_path=s["image_path"],
            width=s["width"],
            height=s["height"],
            target=LaneTarget(
                mask=mask,
                mask_path=str(resolved_mask_path) if resolved_mask_path else mask_path,
                meta=target_meta,
            ),
            meta=s.get("meta", {}),
        )

    def iter_samples(self):
        for i in range(len(self)):
            yield self[i]


class PredictionManifestReader:
    """Backward-compatible reader for old and extended prediction manifests."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path) as f:
            data = json.load(f)
        self.metadata = data.get("metadata", {})
        self._samples = data.get("samples", [])
        self._by_id = {str(sample.get("sample_id")): sample for sample in self._samples}

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _native_polylines(prediction: dict) -> list[np.ndarray]:
        lanes = []
        for lane in prediction.get("polylines", []) or []:
            points = []
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
                    points.append([x_float, y_float])
            if len(points) >= 2:
                lanes.append(np.asarray(points, dtype=np.float64))
        return lanes

    def _read_entry(self, entry: dict) -> LanePrediction:
        prediction = entry.get("prediction", {})
        mask = None
        mask_path = prediction.get("mask_path")
        resolved_mask_path = None
        if mask_path:
            path = Path(mask_path)
            resolved_mask_path = path if path.is_absolute() else self.manifest_path.parent / path
            if resolved_mask_path.exists():
                loaded = cv2.imread(str(resolved_mask_path), cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    mask = (loaded > 0).astype(np.uint8)
        lanes = self._native_polylines(prediction)
        geometry_source = prediction.get("geometry_source")
        if geometry_source is None:
            if lanes:
                geometry_source = "native_polyline"
            elif prediction.get("lane_json") is not None:
                geometry_source = "lane_json"
            elif mask is not None:
                geometry_source = "mask_derived"
            else:
                geometry_source = "unavailable"
        meta = dict(prediction.get("meta") or {})
        meta.update(
            {
                "lane_json": prediction.get("lane_json"),
                "geometry_source": geometry_source,
                "mask_path": str(resolved_mask_path) if resolved_mask_path else mask_path,
            }
        )
        return LanePrediction(mask=mask, lanes=lanes or None, meta=meta)

    def __getitem__(self, index: int) -> LanePrediction:
        return self._read_entry(self._samples[index])

    def get(self, sample_id: str) -> Optional[LanePrediction]:
        """Return a prediction by sample id, or None when it is absent."""

        entry = self._by_id.get(str(sample_id))
        return None if entry is None else self._read_entry(entry)
