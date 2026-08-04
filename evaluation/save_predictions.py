"""Save per-image predictions for qualitative analysis.

For every image: a binary predicted lane-mask PNG (0/255).
For a strided sample per (model, dataset): a colored overlay JPG showing the
prediction vs GT (TP=yellow, FN=green, FP=red), reusing make_overlay() from
visualize_failures (no new visualization code).

Layout:
    <out_root>/<model>/<dataset>/masks/<image_id>.png
    <out_root>/<model>/<dataset>/overlays/<image_id>.jpg
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _safe(name) -> str:
    return str(name).replace("/", "_").replace("\\", "_")


class PredictionSaver:
    def __init__(self, out_root: str, model: str, dataset: str, total: int,
                 overlay_sample: int = 100, max_side: int = 1280):
        base = Path(out_root) / model / dataset
        self.mask_dir = base / "masks"
        self.overlay_dir = base / "overlays"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        if overlay_sample > 0:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_sample = overlay_sample
        # Stride so the sampled overlays are spread across the dataset, not clustered.
        self.stride = max(1, total // overlay_sample) if overlay_sample > 0 else 0
        self.max_side = max_side
        self.idx = 0
        self.n_mask = 0
        self.n_overlay = 0

    def save(self, image_id, image_path, pred_mask, gt_mask) -> None:
        stem = _safe(image_id)
        pred = (np.asarray(pred_mask) > 0).astype(np.uint8)
        cv2.imwrite(str(self.mask_dir / f"{stem}.png"), pred * 255)
        self.n_mask += 1

        take = (self.overlay_sample > 0 and self.n_overlay < self.overlay_sample
                and self.idx % self.stride == 0)
        self.idx += 1
        if not take:
            return
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return
        from evaluation.visualize_failures import make_overlay, _resize_if_needed
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        gt = np.asarray(gt_mask).astype(np.uint8)
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        p = pred if pred.shape[:2] == (h, w) else cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = _resize_if_needed(make_overlay(img_rgb, gt, p), self.max_side)
        cv2.imwrite(str(self.overlay_dir / f"{stem}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        self.n_overlay += 1
