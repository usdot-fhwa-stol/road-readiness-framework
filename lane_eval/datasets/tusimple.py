from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .base import LaneDatasetAdapter
from .registry import register_dataset
from ..schema.sample import LaneSample
from ..schema.lane import LaneTarget
from ..converters.lanes_to_mask import lanes_to_mask


@register_dataset("tusimple")
class TuSimpleAdapter(LaneDatasetAdapter):
    """Adapter for TuSimple lane detection dataset.

    Annotation format: JSON-lines files where each line is:
        {"raw_file": "clips/...", "lanes": [[x, ...], ...], "h_samples": [y, ...]}

    x == -2 means missing point for that h_sample; those are filtered out.
    Image size is read from the actual file if available (default: 720x1280).
    """

    name = "tusimple"

    def __init__(
        self,
        root: str,
        split: str = "val",
        annotation_files: Optional[List[str]] = None,
        mask_thickness: int = 16,
    ):
        self.root = Path(root)
        self.mask_thickness = mask_thickness
        self._db = self._load_annotations(split, annotation_files)

    def _load_annotations(self, split: str, annotation_files: Optional[List[str]]) -> list:
        if annotation_files:
            paths = [Path(p) for p in annotation_files]
        else:
            paths = sorted(self.root.rglob(f"*{split}*.json"))
            if not paths:
                paths = sorted(self.root.rglob("*.json"))

        db = []
        for fpath in paths:
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        db.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return db

    def __len__(self) -> int:
        return len(self._db)

    def __getitem__(self, idx: int) -> LaneSample:
        rec = self._db[idx]
        raw_file: str = rec["raw_file"]
        h_samples: list = rec["h_samples"]
        lanes_raw: list = rec["lanes"]

        img_path = self.root / raw_file
        if img_path.exists():
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2] if img is not None else (720, 1280)
        else:
            h, w = 720, 1280

        lanes = []
        for lane_xs in lanes_raw:
            pts = [
                [float(x), float(y)]
                for x, y in zip(lane_xs, h_samples)
                if float(x) >= 0
            ]
            if len(pts) >= 2:
                lanes.append(np.array(pts, dtype=np.float32))

        mask = lanes_to_mask(lanes, h, w, thickness=self.mask_thickness) if lanes else np.zeros((h, w), np.uint8)

        return LaneSample(
            image_id=raw_file.replace("/", "_"),
            image_path=str(img_path),
            width=w,
            height=h,
            target=LaneTarget(
                lanes=lanes,
                mask=mask,
                meta={"raw_file": raw_file, "h_samples": h_samples},
            ),
        )
