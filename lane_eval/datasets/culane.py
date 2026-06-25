from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import LaneDatasetAdapter
from .registry import register_dataset
from ..schema.sample import LaneSample
from ..schema.lane import LaneTarget
from ..converters.lanes_to_mask import lanes_to_mask


@register_dataset("culane")
class CULaneAdapter(LaneDatasetAdapter):
    """Adapter for CULane dataset.

    Expected structure:
        root/
          driver_*_*frame/
            *.MP4/              (directories named like video files, containing frames)
              *.jpg
              *.lines.txt       (one lane per line: x1 y1 x2 y2 ...)
          laneseg_label_w16/    (pre-rasterised GT masks, pixel values 0=bg, 1-4=lane)
            driver_*_*frame/
              *.MP4/
                *.png
          list/
            val.txt / test.txt / train.txt

    GT priority: laneseg_label_w16 PNG masks (pre-rasterised, matches paper setup)
    are used when available; falls back to rasterising .lines.txt polylines.
    Images are required for YOLOPX inference — samples with missing images are
    skipped by the eval loop (width/height fall back to CULane standard 590×1640).
    """

    name = "culane"
    _DEFAULT_HW = (590, 1640)
    # Prefer the standard train/val seg masks; fall back to the test-split masks
    # (CULane ships test GT separately under laneseg_label_w16_test). If neither
    # exists for a frame, __getitem__ rasterises the .lines.txt polylines.
    _SEG_LABEL_DIRS = ("laneseg_label_w16", "laneseg_label_w16_test")

    # Map CULane test-split scenario keywords → condition label used in C5.
    # 'normal' becomes 'clear' so it serves as the clear-weather baseline.
    _SCENARIO_MAP = {
        "normal":  "clear",
        "night":   "night",
        "hlight":  "dazzle",
        "shadow":  "shadow",
        "crowd":   "crowded",
        "noline":  "no_marking",
        "arrow":   "arrow",
        "curve":   "curve",
        "cross":   "cross",
    }

    def __init__(
        self,
        root: str,
        split: str = "val",
        list_file: Optional[str] = None,
        mask_thickness: int = 16,
        strict: bool = False,
    ):
        self.root = Path(root)
        self.mask_thickness = mask_thickness
        self.strict = strict

        list_path = Path(list_file) if list_file else self.root / "list" / f"{split}.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"CULane list file not found: {list_path}")

        self._scenario = self._parse_scenario(list_path)
        self._samples = self._build_samples(list_path)

    @classmethod
    def _parse_scenario(cls, list_path: Path) -> Optional[str]:
        """Infer scenario from list filename, e.g. 'test8_night.txt' → 'night'."""
        stem = list_path.stem  # e.g. 'test8_night'
        for keyword, condition in cls._SCENARIO_MAP.items():
            if keyword in stem:
                return condition
        return None

    def _seg_path_for(self, rel: str) -> str:
        """Resolve a frame's seg-mask path, preferring whichever dir actually has it."""
        name = rel.replace(".jpg", ".png")
        for d in self._SEG_LABEL_DIRS:
            cand = self.root / d / name
            if cand.exists():
                return str(cand)
        # Default to the first dir; missing file triggers the .lines.txt fallback.
        return str(self.root / self._SEG_LABEL_DIRS[0] / name)

    def _build_samples(self, list_path: Path) -> list[tuple[str, str, str]]:
        samples = []
        with open(list_path) as f:
            for line in f:
                rel = line.strip().lstrip("/")
                if not rel:
                    continue
                img_path = str(self.root / rel)
                annot_path = str(self.root / rel.replace(".jpg", ".lines.txt"))
                seg_path = self._seg_path_for(rel)
                samples.append((img_path, annot_path, seg_path))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> LaneSample:
        img_path, annot_path, seg_path = self._samples[idx]

        if Path(img_path).exists():
            img = cv2.imread(img_path)
            h, w = img.shape[:2] if img is not None else self._DEFAULT_HW
        else:
            h, w = self._DEFAULT_HW

        # GT: prefer pre-rasterised segmentation mask (laneseg_label_w16)
        mask = None
        mask_path = None
        lanes = []
        if Path(seg_path).exists():
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if seg is not None:
                mask = (seg > 0).astype(np.uint8)  # binarise: any lane category → 1
                mask_path = seg_path  # expose the on-disk GT so consumers can link it
                h, w = seg.shape[:2]

        if mask is None:
            # Fallback: rasterise from polyline annotations
            annot = Path(annot_path)
            if annot.exists():
                with open(annot) as f:
                    for line in f:
                        vals = line.strip().split()
                        if len(vals) < 4:
                            continue
                        # Skip a single malformed line without aborting the rest.
                        try:
                            coords = list(map(float, vals))
                            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
                            lanes.append(pts)
                        except (ValueError, TypeError):
                            continue
            elif self.strict:
                raise FileNotFoundError(f"Annotation missing: {annot_path}")
            mask = lanes_to_mask(lanes, h, w, thickness=self.mask_thickness) if lanes else np.zeros((h, w), np.uint8)

        image_id = str(Path(img_path).relative_to(self.root)).replace("/", "_").replace(".jpg", "")

        meta = {}
        if self._scenario is not None:
            meta["condition"] = self._scenario
            if self._scenario == "night":
                meta["timeofday"] = "night"

        return LaneSample(
            image_id=image_id,
            image_path=img_path,
            width=w,
            height=h,
            target=LaneTarget(lanes=lanes or None, mask=mask, mask_path=mask_path),
            meta=meta,
        )
