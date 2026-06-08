"""Save raw + overlay images for frames that fail or excel on each readiness metric."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Thresholds — (operator, value)
#   For failures  : 'lt' → value < thresh is BAD
#                   'gt' → value > thresh is BAD
#                   'eq' → value == thresh is BAD
#   For good cases: 'gt' → value > thresh is GOOD
#                   'lt' → value < thresh is GOOD
#                   'gte'→ value >= thresh is GOOD
# ---------------------------------------------------------------------------

DEFAULT_FAILURE_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    "D3_tolerant_f1_r5": ("lt", 0.45),
    "D3_iou": ("lt", 0.25),
    "D4_near_iou": ("lt", 0.30),
    "D6_missed_marking_ratio": ("gt", 0.50),
    "I2_local_contrast": ("lt", 0.12),
    "I1_pattern_continuity": ("lt", 0.45),
    "I3_boundary_sharpness": ("lt", 0.20),
    "R1_marking_readability_score": ("lt", 60.0),
    "R2_reference_detectability_score": ("lt", 60.0),
    "D5_dark_region_iou_gap": ("gt", 0.30),
}

# Keep the old name as an alias so existing call sites don't break.
DEFAULT_THRESHOLDS = DEFAULT_FAILURE_THRESHOLDS

DEFAULT_GOOD_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    "D3_tolerant_f1_r5": ("gt", 0.80),
    "D3_iou": ("gt", 0.60),
    "D4_near_iou": ("gt", 0.65),
    "D6_missed_marking_ratio": ("lt", 0.15),
    "I2_local_contrast": ("gt", 0.30),
    "I1_pattern_continuity": ("gt", 0.75),
    "I3_boundary_sharpness": ("gt", 0.45),
    "R1_marking_readability_score": ("gt", 80.0),
    "R2_reference_detectability_score": ("gt", 80.0),
}


# ---------------------------------------------------------------------------
# Threshold checks
# ---------------------------------------------------------------------------

def _fails(value: float, threshold: Tuple[str, float]) -> bool:
    op, thresh = threshold
    if op == "lt":  return value < thresh
    if op == "gt":  return value > thresh
    if op == "eq":  return value == thresh
    return False


def _is_good(value: float, threshold: Tuple[str, float]) -> bool:
    op, thresh = threshold
    if op == "gt":  return value > thresh
    if op == "lt":  return value < thresh
    if op == "gte": return value >= thresh
    return False


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def make_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Blend GT and prediction masks onto the image.

    Colour coding (alpha = 0.55):
      Yellow — TP  correctly detected lane pixel
      Green  — FN  GT lane pixel the model missed
      Red    — FP  predicted lane pixel with no GT
    """
    overlay = image_rgb.astype(np.float32)
    alpha = 0.55

    tp = (gt_mask > 0) & (pred_mask > 0)
    fn = (gt_mask > 0) & (pred_mask == 0)
    fp = (gt_mask == 0) & (pred_mask > 0)

    for c, v in zip([0, 1, 2], [255, 255, 0]):   # Yellow TP
        overlay[tp, c] = overlay[tp, c] * (1 - alpha) + v * alpha
    for c, v in zip([0, 1, 2], [0, 200, 0]):     # Green FN
        overlay[fn, c] = overlay[fn, c] * (1 - alpha) + v * alpha
    for c, v in zip([0, 1, 2], [220, 0, 0]):     # Red FP
        overlay[fp, c] = overlay[fp, c] * (1 - alpha) + v * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_if_needed(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str, good: bool = False) -> np.ndarray:
    """Burn metric + value into the top-left corner. Green background for good, black for fail."""
    img = img.copy()
    bg = (0, 80, 0) if good else (0, 0, 0)
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 6, 26), bg, -1)
    cv2.putText(img, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Core save function
# ---------------------------------------------------------------------------

def _save_examples(
    samples: list,
    images_rgb: List[np.ndarray],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    per_image_metrics: List[dict],
    output_dir: str,
    thresholds: Dict[str, Tuple[str, float]],
    check_fn: Callable[[float, Tuple[str, float]], bool],
    good: bool,
    max_side: int,
) -> Dict[str, int]:
    out = Path(output_dir)
    counts: Dict[str, int] = {m: 0 for m in thresholds}

    metrics_in_data = set(per_image_metrics[0].keys()) if per_image_metrics else set()
    active = {m: t for m, t in thresholds.items() if m in metrics_in_data}

    for metric in active:
        (out / metric).mkdir(parents=True, exist_ok=True)

    for idx, (sample, img_rgb, pred, gt, mets) in enumerate(
        zip(samples, images_rgb, pred_masks, gt_masks, per_image_metrics)
    ):
        image_id = str(getattr(sample, "image_id", f"frame_{idx:05d}"))
        image_id = image_id.replace("/", "_").replace("\\", "_")

        overlay = None  # built once, reused across metrics for the same image

        for metric, threshold in active.items():
            val = mets.get(metric)
            if val is None or not check_fn(float(val), threshold):
                continue

            if overlay is None:
                overlay = make_overlay(img_rgb, gt, pred)

            raw_small = _resize_if_needed(img_rgb, max_side)
            ovl_small = _resize_if_needed(overlay, max_side)

            suffix = " [good]" if good else ""
            label_text = f"{metric}={val:.2f}{suffix}" if isinstance(val, float) else f"{metric}={val}{suffix}"
            ovl_labelled = _put_label(cv2.cvtColor(ovl_small, cv2.COLOR_RGB2BGR), label_text, good=good)
            raw_bgr = cv2.cvtColor(raw_small, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(out / metric / f"{image_id}_raw.jpg"),     raw_bgr)
            cv2.imwrite(str(out / metric / f"{image_id}_overlay.jpg"), ovl_labelled)
            counts[metric] += 1

    return counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Save raw + overlay for every frame that fails a metric threshold."""
    return _save_examples(
        samples, images_rgb, pred_masks, gt_masks, per_image_metrics,
        output_dir,
        thresholds=thresholds or DEFAULT_FAILURE_THRESHOLDS,
        check_fn=_fails,
        good=False,
        max_side=max_side,
    )


def save_good_images(
    samples: list,
    images_rgb: List[np.ndarray],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    per_image_metrics: List[dict],
    output_dir: str,
    thresholds: Optional[Dict[str, Tuple[str, float]]] = None,
    max_side: int = 1280,
) -> Dict[str, int]:
    """Save raw + overlay for every frame that excels on a metric (green label)."""
    return _save_examples(
        samples, images_rgb, pred_masks, gt_masks, per_image_metrics,
        output_dir,
        thresholds=thresholds or DEFAULT_GOOD_THRESHOLDS,
        check_fn=_is_good,
        good=True,
        max_side=max_side,
    )
