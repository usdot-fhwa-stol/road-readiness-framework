"""Read a universal manifest JSON and present it as a dataset adapter.

This is the input side of the YOLOPX adapter: it replaces the native dataset
adapters so inference consumes the dataset-agnostic manifest produced by
lane_eval.manifest.generator. It yields the same LaneSample objects the rest of
the eval pipeline expects, so it slots directly into run_lane_eval's
_EvalDataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..schema.sample import LaneSample
from ..schema.lane import LaneTarget


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

        return LaneSample(
            image_id=s["sample_id"],
            image_path=s["image_path"],
            width=s["width"],
            height=s["height"],
            target=LaneTarget(
                mask=mask,
                mask_path=str(resolved_mask_path) if resolved_mask_path else mask_path,
                meta={"lane_json": gt.get("lane_json"), "natural_gt": gt.get("natural_gt")},
            ),
            meta=s.get("meta", {}),
        )

    def iter_samples(self):
        for i in range(len(self)):
            yield self[i]
