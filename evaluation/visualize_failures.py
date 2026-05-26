"""Save raw + overlay images for every frame that fails a readiness metric threshold."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Failure thresholds
# Tuple: (operator, value)
#   'lt' → metric < value is a failure
#   'gt' → metric > value is a failure
#   'eq' → metric == value is a failure
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    # Layer 1 — infrastructure
    "I1": ("lt", 50.0),   # continuity < 50 %
    "I2": ("lt", 0.15),   # contrast < 0.15
    "I3": ("lt", 0.30),   # sharpness < 0.30
    "I4": ("gt", 20.0),   # width std > 20 px (unstable)
    "I5": ("gt", 0.10),   # curvature > 0.10 (highly curved)
    "I6": ("eq", 0),      # zero lanes detected
    # Layer 2 — detection (per-image values: D1/D2 are 0 or 100)
    "D1": ("lt", 50.0),   # no detection on this frame
    "D2": ("lt", 50.0),   # wrong lane count
    "D3": ("lt", 0.30),   # IoU < 0.30
    "D4": ("lt", 0.30),   # near-field IoU < 0.30
    "D5": ("gt", 0.30),   # occlusion gap > 0.30
    "D6": ("gt", 0.50),   # > 50 % of GT lane pixels missed
}


def _fails(value: float, threshold: Tuple[str, float]) -> bool:
    op, thresh = threshold
    if op == "lt":
        return value < thresh
    if op == "gt":
        return value > thresh
    if op == "eq":
        return value == thresh
    return False


def make_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Blend GT and prediction masks onto the image.

    Colour coding (alpha = 0.55):
      Yellow  — TP  (correctly detected lane pixel)
      Green   — FN  (GT lane pixel the model missed)
      Red     — FP  (predicted lane pixel with no GT)
    """
    overlay = image_rgb.astype(np.float32)
    alpha = 0.55

    tp = (gt_mask > 0) & (pred_mask > 0)
    fn = (gt_mask > 0) & (pred_mask == 0)
    fp = (gt_mask == 0) & (pred_mask > 0)

    # Yellow (255, 255, 0) for TP
    for c, v in zip([0, 1, 2], [255, 255, 0]):
        overlay[tp, c] = overlay[tp, c] * (1 - alpha) + v * alpha

    # Green (0, 200, 0) for FN — missed lanes
    for c, v in zip([0, 1, 2], [0, 200, 0]):
        overlay[fn, c] = overlay[fn, c] * (1 - alpha) + v * alpha

    # Red (220, 0, 0) for FP — spurious detections
    for c, v in zip([0, 1, 2], [220, 0, 0]):
        overlay[fp, c] = overlay[fp, c] * (1 - alpha) + v * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _resize_if_needed(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn metric name + value into the top-left corner."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 6, 26), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_failure_images(
    samples: list,
    images_rgb: List[np.ndarray],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    per_image_metrics: List[dict],
    output_dir: str,
    thresholds: Optional[Dict[str, Tuple[str, float]]] = None,
    max_side: int = 1280,
) -> Dict[str, int]:
    """
    For every metric in thresholds, scan all images and save the ones that fail.

    Saves two files per failing image per metric:
        {output_dir}/{metric}/{image_id}_raw.jpg      — original frame
        {output_dir}/{metric}/{image_id}_overlay.jpg  — GT/pred colour overlay

    Returns a dict {metric: count_saved}.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    out = Path(output_dir)
    counts: Dict[str, int] = {m: 0 for m in thresholds}

    metrics_in_data = set(per_image_metrics[0].keys()) if per_image_metrics else set()
    active = {m: t for m, t in thresholds.items() if m in metrics_in_data}

    # Pre-create output dirs
    for metric in active:
        (out / metric).mkdir(parents=True, exist_ok=True)

    for idx, (sample, img_rgb, pred, gt, mets) in enumerate(
        zip(samples, images_rgb, pred_masks, gt_masks, per_image_metrics)
    ):
        image_id = getattr(sample, "image_id", f"frame_{idx:05d}")
        # sanitise for filesystem
        image_id = str(image_id).replace("/", "_").replace("\\", "_")

        overlay = None  # build once, reuse across metrics for same image

        for metric, threshold in active.items():
            val = mets.get(metric)
            if val is None or not _fails(float(val), threshold):
                continue

            # Build overlay on first metric failure for this image
            if overlay is None:
                overlay = make_overlay(img_rgb, gt, pred)

            raw_small = _resize_if_needed(img_rgb, max_side)
            ovl_small = _resize_if_needed(overlay, max_side)

            label_text = f"{metric}={val:.2f}" if isinstance(val, float) else f"{metric}={val}"
            ovl_labelled = _put_label(
                cv2.cvtColor(ovl_small, cv2.COLOR_RGB2BGR), label_text
            )
            raw_bgr = cv2.cvtColor(raw_small, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(out / metric / f"{image_id}_raw.jpg"),     raw_bgr)
            cv2.imwrite(str(out / metric / f"{image_id}_overlay.jpg"), ovl_labelled)
            counts[metric] += 1

    return counts
