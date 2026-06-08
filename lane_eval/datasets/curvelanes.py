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
from ..converters.lanes_to_mask import lanes_to_mask


@register_dataset("curvelanes")
class CurveLanesAdapter(LaneDatasetAdapter):
    """Adapter for CurveLanes dataset.

    Expected structure:
        root/
          {split}/
            images/   *.jpg
            labels/   *.lines.json   ({"Lines": [[{"x": ..., "y": ...}, ...], ...]})
            {split}.txt  (optional list file)

    Labels contain float x/y string values. Image size is typically 1440x2560
    but is read from the actual file to be safe.
    """

    name = "curvelanes"

    def __init__(
        self,
        root: str,
        split: str = "valid",
        list_file: Optional[str] = None,
        mask_thickness: int = 16,
    ):
        self.root = Path(root)
        self.split = split
        self.mask_thickness = mask_thickness
        self._samples = self._build_samples(list_file)

    def _build_samples(self, list_file: Optional[str]) -> list[tuple[str, Optional[str]]]:
        img_dir = self.root / self.split / "images"
        label_dir = self.root / self.split / "labels"

        if list_file:
            with open(list_file) as f:
                stems = [Path(l.strip()).stem for l in f if l.strip()]
            entries = [(str(img_dir / (s + ".jpg")), str(label_dir / (s + ".lines.json"))) for s in stems]
        else:
            entries = []
            for img_path in sorted(img_dir.glob("*.jpg")):
                label_path = label_dir / (img_path.stem + ".lines.json")
                entries.append((str(img_path), str(label_path) if label_path.exists() else None))

        return entries

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> LaneSample:
        img_path, label_path = self._samples[idx]

        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = 1440, 2560  # CurveLanes default resolution

        lanes = []
        if label_path and Path(label_path).exists():
            try:
                with open(label_path) as f:
                    data = json.load(f)
                for lane_pts in data.get("Lines", []):
                    pts = []
                    for pt in lane_pts:
                        try:
                            pts.append([float(pt["x"]), float(pt["y"])])
                        except (KeyError, ValueError):
                            continue
                    if len(pts) >= 2:
                        lanes.append(np.array(pts, dtype=np.float32))
            except Exception:
                pass

        mask = lanes_to_mask(lanes, h, w, thickness=self.mask_thickness) if lanes else np.zeros((h, w), np.uint8)

        return LaneSample(
            image_id=Path(img_path).stem,
            image_path=img_path,
            width=w,
            height=h,
            target=LaneTarget(lanes=lanes, mask=mask),
        )
