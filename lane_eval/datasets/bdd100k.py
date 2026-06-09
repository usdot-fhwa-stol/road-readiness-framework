from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import LaneDatasetAdapter
from .registry import register_dataset
from ..schema.sample import LaneSample
from ..schema.lane import LaneTarget


@register_dataset("bdd100k_lane")
class BDD100KLaneAdapter(LaneDatasetAdapter):
    """Adapter for the YOLOPX/YOLOP-prepared BDD100K lane-mask layout.

    Expected structure (mirrors what YOLOPX was trained on):
        image_root/      *.jpg
        lane_mask_root/  *.png   (pixel values: 0=background, 255=lane)

    Images and masks are paired by basename stem.
    target.lanes is always None — BDD100K GT is provided as pre-rasterised masks.

    Optional: pass det_annotations_root to attach per-image weather/scene/timeofday
    metadata from BDD100K's detection annotation JSONs.

        det_annotations_root/   *.json   (one per image, keyed by stem)
          e.g. /shared/data/bdd100k/det_annotations/val/

    When provided, each sample.meta gains:
        weather   : str  (clear / rainy / overcast / foggy / snowy / partly cloudy / undefined)
        scene     : str  (city street / highway / residential / tunnel / ...)
        timeofday : str  (daytime / night / dawn/dusk / undefined)
    """

    name = "bdd100k_lane"

    def __init__(
        self,
        image_root: str,
        lane_mask_root: str,
        split: str = "val",
        det_annotations_root: Optional[str] = None,
    ):
        self.image_root = Path(image_root)
        self.lane_mask_root = Path(lane_mask_root)
        self.split = split
        self._ann_root = Path(det_annotations_root) if det_annotations_root else None
        self._samples = self._build_samples()

    def _build_samples(self) -> list[tuple[str, str]]:
        samples = []
        for mask_path in sorted(self.lane_mask_root.glob("*.png")):
            img_path = self.image_root / (mask_path.stem + ".jpg")
            if img_path.exists():
                samples.append((str(img_path), str(mask_path)))
        if not samples:
            raise FileNotFoundError(
                f"No paired image/mask files found. "
                f"image_root={self.image_root}  lane_mask_root={self.lane_mask_root}"
            )
        return samples

    def _load_attributes(self, stem: str) -> dict:
        if self._ann_root is None:
            return {}
        ann_path = self._ann_root / f"{stem}.json"
        if not ann_path.exists():
            return {}
        try:
            with open(ann_path) as f:
                return json.load(f).get("attributes", {})
        except Exception:
            return {}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> LaneSample:
        img_path, mask_path = self._samples[idx]
        stem = Path(img_path).stem

        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = 720, 1280  # BDD100K default

        lane_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary = (lane_mask > 0).astype(np.uint8) if lane_mask is not None else np.zeros((h, w), np.uint8)

        attrs = self._load_attributes(stem)

        return LaneSample(
            image_id=stem,
            image_path=img_path,
            width=w,
            height=h,
            target=LaneTarget(mask=binary, mask_path=mask_path),
            meta=attrs,
        )
