"""D2/D4/D5/D6 visualization panels matching the i_outputs/i_N/*.jpg house
style: a black header bar (bold title, metric value lines, dataset/image
name, a plain-text legend line) above two side-by-side panels -- RAW image on
the left, annotated overlay on the right -- both at a fixed display width.

No false positives are ever drawn, per the "false negatives only" requirement:
overlays use only TP (green, agreement) and FN (magenta, missed GT), never an
FP color. D2 has no pixel-level TP/FN concept (it's an instance-count ratio)
so it instead shows GT lane instances vs predicted lane instances, mirroring
i_outputs/i_6's GT-lane-count style extended with the prediction side.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PANEL_WIDTH = 760
_TP_COLOR = (60, 220, 60)      # BGR green -- correctly detected GT
_FN_COLOR = (255, 0, 255)      # BGR magenta -- missed GT (false negative)
_GT_COLOR = (255, 0, 255)      # BGR magenta -- GT lane instances (D2)
_PRED_COLOR = (255, 220, 0)    # BGR cyan -- predicted lane instances (D2)

_TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
_BODY_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _fit_mask(mask: np.ndarray, shape_hw) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.shape != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def _thin_overlay(base_bgr: np.ndarray, mask: np.ndarray, color, thickness: int = 2) -> np.ndarray:
    """Draw a mask's contours as thin unfilled strokes (not an alpha-blended
    fill) -- matches the crisp thin-line look of the i_outputs examples,
    rather than the thick blended regions the old D-metric panels used."""
    out = base_bgr.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out


def _draw_polylines(base_bgr: np.ndarray, lanes, color, thickness: int = 2) -> np.ndarray:
    out = base_bgr.copy()
    for lane in lanes or []:
        pts = np.asarray(lane, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
    return out


def _compose(raw_bgr: np.ndarray, annotated_bgr: np.ndarray, right_title: str,
             right_lines: list[str], legend_line: str, panel_width: int = PANEL_WIDTH) -> np.ndarray:
    left = _resize_to_width(raw_bgr, panel_width)
    right = _resize_to_width(annotated_bgr, panel_width)
    h = min(left.shape[0], right.shape[0])
    left, right = left[:h], right[:h]

    line_h = 26
    left_lines = 1  # "Raw Image" title only
    right_n = 1 + len(right_lines) + (1 if legend_line else 0)
    n_lines = max(left_lines, right_n)
    header_h = 20 + line_h * n_lines

    header = np.zeros((header_h, panel_width * 2, 3), dtype=np.uint8)
    cv2.putText(header, "Raw Image", (16, 30), _TITLE_FONT, 0.72, (255, 255, 255), 1, cv2.LINE_AA)

    x2 = panel_width + 16
    cv2.putText(header, right_title, (x2, 30), _TITLE_FONT, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
    y = 30
    for line in right_lines:
        y += line_h
        cv2.putText(header, line, (x2, y), _BODY_FONT, 0.55, (225, 225, 225), 1, cv2.LINE_AA)
    if legend_line:
        y += line_h
        cv2.putText(header, legend_line, (x2, y), _BODY_FONT, 0.55, (225, 225, 225), 1, cv2.LINE_AA)

    body = np.hstack([left, right])
    return np.vstack([header, body])


def _save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_d2(out_path: Path, raw_bgr, gt_lanes, pred_lanes, record: dict, dataset: str, sample_id: str,
              gt_mask=None, pred_mask=None) -> None:
    """GT lane instances (magenta) vs predicted lane instances (cyan).

    Prefers native polylines (gt_lanes from lane_json / pred_lanes from a
    processor like CLRerNet) when available. Mask-only sources (CULane/BDD100K
    GT has no lane_json; YOLOPX has no native polylines) fall back to mask
    contours -- without this fallback nothing would be drawn at all for those
    combos even though the header's counts are non-zero.
    """
    annotated = raw_bgr.copy()
    if gt_lanes:
        annotated = _draw_polylines(annotated, gt_lanes, _GT_COLOR, 2)
    elif gt_mask is not None:
        annotated = _thin_overlay(annotated, _fit_mask(gt_mask, raw_bgr.shape[:2]), _GT_COLOR, 2)
    if pred_lanes:
        annotated = _draw_polylines(annotated, pred_lanes, _PRED_COLOR, 2)
    elif pred_mask is not None:
        annotated = _thin_overlay(annotated, _fit_mask(pred_mask, raw_bgr.shape[:2]), _PRED_COLOR, 2)
    lines = [
        f"D2 Lane Count Ratio: {_fmt(record.get('D2_count_match'))}",
        f"GT Lane Count: {record.get('D2_gt_lane_count')}  |  Predicted Lane Count: {record.get('D2_pred_lane_count')}",
        f"Predicted Count Source: {record.get('D2_pred_count_source')}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    legend = "MAGENTA = GT lane instances | CYAN = predicted lane instances"
    _save(out_path, _compose(raw_bgr, annotated, "D2 Lane Count Metric", lines, legend))


def render_d4(out_path: Path, raw_bgr, gt_mask, pred_mask, record: dict, dataset: str, sample_id: str) -> None:
    h = raw_bgr.shape[0]
    gt = _fit_mask(gt_mask, raw_bgr.shape[:2])
    pred = _fit_mask(pred_mask, raw_bgr.shape[:2])
    tp = ((pred > 0) & (gt > 0)).astype(np.uint8)
    fn = ((pred == 0) & (gt > 0)).astype(np.uint8)
    annotated = _thin_overlay(raw_bgr, tp, _TP_COLOR, 2)
    annotated = _thin_overlay(annotated, fn, _FN_COLOR, 2)
    top = record.get("D4_region_top_row")
    bottom = record.get("D4_region_bottom_row")
    if top is not None and bottom is not None:
        top_s = int(top * raw_bgr.shape[0] / h) if h else top
        bottom_s = int(bottom * raw_bgr.shape[0] / h) if h else bottom
        annotated[:top_s] = (0.35 * annotated[:top_s]).astype(np.uint8)
        annotated[bottom_s:] = (0.35 * annotated[bottom_s:]).astype(np.uint8)
        cv2.line(annotated, (0, top_s), (annotated.shape[1], top_s), (255, 255, 255), 1)
        cv2.line(annotated, (0, bottom_s), (annotated.shape[1], bottom_s), (255, 255, 255), 1)
    lines = [
        f"D4 Upper Band F1: {_fmt(record.get('D4p_upper_f1'))}",
        f"D4 Middle Band F1: {_fmt(record.get('D4p_middle_f1'))}",
        f"D4 Lower Band F1: {_fmt(record.get('D4p_lower_f1'))}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    legend = "GREEN = detected GT | MAGENTA = missed GT (FN); outer bands dimmed"
    _save(out_path, _compose(raw_bgr, annotated, "D4 Image-Band Readability Metric", lines, legend))


def render_d5(out_path: Path, raw_bgr, gt_mask, pred_mask, record: dict, dataset: str, sample_id: str) -> None:
    gt = _fit_mask(gt_mask, raw_bgr.shape[:2])
    pred = _fit_mask(pred_mask, raw_bgr.shape[:2])
    tp = ((pred > 0) & (gt > 0)).astype(np.uint8)
    fn = ((pred == 0) & (gt > 0)).astype(np.uint8)
    annotated = _thin_overlay(raw_bgr, tp, _TP_COLOR, 2)
    annotated = _thin_overlay(annotated, fn, _FN_COLOR, 2)
    lines = [
        f"Visibility Group: {record.get('D5_visibility_group')}",
        f"Observed Marking Visibility Tag: {record.get('visibility_tag')}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    legend = "GREEN = detected GT | MAGENTA = missed GT (FN)"
    _save(out_path, _compose(raw_bgr, annotated, "D5 Marking-Condition Sensitivity Metric", lines, legend))


def render_d6(out_path: Path, raw_bgr, gt_mask, pred_mask, record: dict, dataset: str, sample_id: str) -> None:
    gt = _fit_mask(gt_mask, raw_bgr.shape[:2])
    pred = _fit_mask(pred_mask, raw_bgr.shape[:2])
    fn = ((pred == 0) & (gt > 0)).astype(np.uint8)
    annotated = _thin_overlay(raw_bgr, fn, _FN_COLOR, 2)
    lines = [
        f"D6 Missed-Marking Ratio: {_fmt(record.get('D6_image_missed_ratio'))}",
        f"GT Pixels: {record.get('D6_gt_pixels')}  |  Missed (FN) Pixels: {record.get('D6_fn_pixels')}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    legend = "MAGENTA = missed GT pixels (false negative only)"
    _save(out_path, _compose(raw_bgr, annotated, "D6 Missed-Marking Metric", lines, legend))
