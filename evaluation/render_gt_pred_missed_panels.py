"""D/R-metric visualization panels using solid GT/predicted/missed masks,
color-matched to the I-metrics house style (black header bar, RAW image left /
annotated overlay right, plain-text legend line -- see i_outputs/i_N/*.jpg
and evaluation/render_i_style_d_panels.py).

Unlike the earlier D-metric panel styles (TP/FN/FP jargon, thin contour
lines), this uses exactly three solid colors with no correctness jargon in
the legend:
    BLUE  = ground-truth marking
    GREEN = predicted marking
    RED   = missed GT (ground truth with no matching prediction nearby)
False positives (predicted, no GT) are not separately colored -- they show as
plain GREEN, same as correct predictions, per the "drop FP" simplification.

"Missed" is computed against a stroke-width-standardized copy of the
prediction (evaluation.d_metrics.standardize_stroke_width), the same
common-representation step the D3/D4/D6 metrics themselves use, so the red
region matches what the metrics actually score as a miss -- not a raw
pixel-diff artifact of GT being drawn at a different stroke width than a thin
polyline-rasterized prediction.

Every render_* function downscales the raw image to PANEL_WIDTH first and
does all mask-fitting/drawing/compositing at that final resolution. Drawing
at full source resolution (which can be several thousand px wide) and only
downscaling the finished, already-colored image at the very end (as an
earlier version of this module did) forces a heavy INTER_AREA average over
many thin, closely-spaced, differently-colored strokes -- which silently
blends e.g. blue GT and red missed into a stray magenta fringe that isn't
part of the 3-color legend. Fitting masks directly to the small target
resolution with INTER_NEAREST (see _fit_mask) and drawing hard-edged
(LINE_8) strokes there avoids that blending entirely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from evaluation.d_metrics import (
    NEAR_FIELD_TOP_FRACTION,
    NEAR_FIELD_BOTTOM_FRACTION,
    lanes_from_lane_json,
    standardize_stroke_width,
)
from evaluation.marking_support import mask_to_points

PANEL_WIDTH = 760

_GT_COLOR = (255, 40, 0)      # BGR strong blue
_PRED_COLOR = (0, 200, 0)     # BGR strong green
_MISSED_COLOR = (0, 0, 230)   # BGR strong red

_TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
_BODY_FONT = cv2.FONT_HERSHEY_SIMPLEX

_LEGEND_GT_PRED_MISSED = "BLUE = GT marking | GREEN = predicted marking | RED = missed GT"
_LEGEND_GT_PRED = "BLUE = GT lane instances | GREEN = predicted lane instances"
_LEGEND_PRED_ONLY = "GREEN = predicted marking"


def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _downscale(raw_bgr: np.ndarray, width: int = PANEL_WIDTH) -> tuple[np.ndarray, float]:
    """Downscale the source photo to panel width once, up front (INTER_AREA is
    fine here -- it's a natural photo, not colored line art yet). Returns
    (small_bgr, scale) so callers with coordinates in source-pixel space
    (e.g. D2's native lane polylines) can rescale them to match."""
    h, w = raw_bgr.shape[:2]
    scale = width / w
    small = cv2.resize(raw_bgr, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return small, scale


def _fit_mask(mask: np.ndarray, shape_hw) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.shape != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def _dilate_for_display(mask: np.ndarray, image_h: int, scale: float = 1.0) -> np.ndarray:
    k = max(2, int(round(image_h / 220 * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def _blend(base_bgr: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    out = base_bgr.copy()
    sel = mask > 0
    if not np.any(sel):
        return out
    overlay = np.empty_like(out)
    overlay[:] = color
    out[sel] = (alpha * overlay[sel] + (1 - alpha) * out[sel]).astype(np.uint8)
    return out


def gt_pred_missed_masks(gt_mask: np.ndarray, pred_mask: np.ndarray, shape_hw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (gt, pred, missed) fitted binary masks at shape_hw. ``missed``
    uses stroke-width-standardized pred (same common representation the D3/D4/D6
    metrics use) so it matches what's actually scored as a miss. Standardizing
    at full source resolution before this function's caller downscales would
    change the effective dilation math, so callers pass already-panel-sized
    masks in and this fits+matches them at that same size."""
    gt = _fit_mask(gt_mask, shape_hw)
    pred = _fit_mask(pred_mask, shape_hw)
    matched_pred, _ = standardize_stroke_width(pred, gt)
    missed = ((matched_pred == 0) & (gt > 0)).astype(np.uint8)
    return gt, pred, missed


def _gt_centerline_mask(gt: np.ndarray) -> np.ndarray:
    """1px skeleton of the GT band (reuses marking_support.mask_to_points, the
    same skeletonizer the D3' metric uses for YOLOPX's dense masks), rather
    than the band's own outline.

    At panel display resolution the GT band is often only ~4-8px wide once
    downscaled -- an outline traced around it (even at minimum thickness)
    doesn't leave a visible gap: the top and bottom edges of the stroke
    effectively merge into a solid block and fully occlude the red/green fill
    underneath. A thin line through the middle of the band leaves a visible
    fill margin on both sides, so BLUE = GT, and RED/GREEN = matched/missed,
    are both readable at once instead of blue winning by occlusion."""
    if not np.any(gt):
        return np.zeros_like(gt)
    pts = mask_to_points(gt)
    if pts.size == 0:
        return np.zeros_like(gt)
    out = np.zeros_like(gt)
    xs = np.clip(pts[:, 0].astype(np.int32), 0, gt.shape[1] - 1)
    ys = np.clip(pts[:, 1].astype(np.int32), 0, gt.shape[0] - 1)
    out[ys, xs] = 1
    return out


def _overlay(base_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray, missed: np.ndarray) -> np.ndarray:
    """Predicted (green) and missed (red) are solid fills; GT is a thin blue
    centerline drawn through the middle of its band, on top of the fills (see
    _gt_centerline_mask)."""
    h = base_bgr.shape[0]
    pred_d = _dilate_for_display(pred, h, scale=1.0)
    missed_d = _dilate_for_display(missed, h, scale=1.0)
    out = _blend(base_bgr, pred_d, _PRED_COLOR, alpha=0.85)
    out = _blend(out, missed_d, _MISSED_COLOR, alpha=0.85)
    gt_center_d = _dilate_for_display(_gt_centerline_mask(gt), h, scale=0.7)
    out = _blend(out, gt_center_d, _GT_COLOR, alpha=0.95)
    return out


def _pred_only_overlay(base_bgr: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h = base_bgr.shape[0]
    return _blend(base_bgr, _dilate_for_display(pred, h), _PRED_COLOR, alpha=0.85)


def _compose(raw_small: np.ndarray, annotated_small: np.ndarray, right_title: str,
             right_lines: list[str], legend_line: str) -> np.ndarray:
    """Both inputs are already at PANEL_WIDTH (see _downscale) -- no further
    resizing here, since resizing an already-colored annotation risks
    re-introducing the cross-color blending this module is written to avoid."""
    h = min(raw_small.shape[0], annotated_small.shape[0])
    left, right = raw_small[:h], annotated_small[:h]
    panel_width = left.shape[1]

    line_h = 26
    left_lines = 1
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
        cv2.putText(header, legend_line, (x2, y), _BODY_FONT, 0.5, (225, 225, 225), 1, cv2.LINE_AA)

    body = np.hstack([left, right])
    return np.vstack([header, body])


def _save(path: Path, img: np.ndarray) -> None:
    """PNG, not JPEG: JPEG's chroma subsampling averages color across nearby
    pixels, which visibly blends thin, closely-spaced, differently-colored
    lines (e.g. blue GT next to red missed) into a stray magenta -- confirmed
    by comparing the pre-compression array (no magenta) against a JPEG save
    (magenta appears). Since these panels are read by their exact color, a
    lossless format matters more here than file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def render_raw(out_path: Path, raw_bgr: np.ndarray) -> None:
    small, _ = _downscale(raw_bgr)
    _save(out_path, small)


def render_pred(out_path: Path, raw_bgr: np.ndarray, pred_mask: np.ndarray) -> None:
    small, _ = _downscale(raw_bgr)
    pred = _fit_mask(pred_mask, small.shape[:2])
    _save(out_path, _pred_only_overlay(small, pred))


def render_d1(out_path: Path, raw_bgr: np.ndarray, pred_mask: np.ndarray, record: dict,
              dataset: str, sample_id: str) -> None:
    small, _ = _downscale(raw_bgr)
    pred = _fit_mask(pred_mask, small.shape[:2])
    detected = bool(record.get("D1_detection"))
    annotated = _pred_only_overlay(small, pred)
    if not detected:
        h, w = small.shape[:2]
        cv2.putText(annotated, "NO LANE OUTPUT", (w // 8, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, _MISSED_COLOR, 3, cv2.LINE_AA)
    lines = [
        f"D1 Detection Success: {int(detected)}  ({record.get('D1_pred_pixel_count', 0)} predicted lane px)",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    _save(out_path, _compose(small, annotated, "D1 Detection Success Metric", lines, _LEGEND_PRED_ONLY))


def render_d2(out_path: Path, raw_bgr: np.ndarray, gt_lanes, pred_lanes, record: dict,
              dataset: str, sample_id: str, gt_mask=None, pred_mask=None) -> None:
    # GT is drawn thicker and first, pred thinner and on top -- same
    # guaranteed-visibility trick as _overlay(): a plain same-width overwrite
    # would let a near-identical predicted line fully hide the GT line under
    # it, leaving no blue visible at all. LINE_8 throughout (not LINE_AA):
    # the GT and pred strokes sit directly on top of one another here, and
    # anti-aliased partial-coverage blending between two saturated stroke
    # colors produces a stray cyan/magenta fringe that isn't part of the
    # 3-color legend.
    small, scale = _downscale(raw_bgr)
    annotated = small.copy()
    h = small.shape[0]
    thickness = max(3, h // 160)
    if gt_lanes:
        for lane in gt_lanes:
            pts = np.asarray(np.asarray(lane) * scale, dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) >= 2:
                cv2.polylines(annotated, [pts], False, _GT_COLOR, thickness + 4, cv2.LINE_8)
    elif gt_mask is not None:
        gt_fit = _fit_mask(gt_mask, small.shape[:2])
        gt_center_d = _dilate_for_display(_gt_centerline_mask(gt_fit), h, scale=0.7)
        annotated = _blend(annotated, gt_center_d, _GT_COLOR, alpha=0.95)
    if pred_lanes:
        for lane in pred_lanes:
            pts = np.asarray(np.asarray(lane) * scale, dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) >= 2:
                cv2.polylines(annotated, [pts], False, _PRED_COLOR, thickness, cv2.LINE_8)
    elif pred_mask is not None:
        annotated = _blend(annotated, _dilate_for_display(_fit_mask(pred_mask, small.shape[:2]), h), _PRED_COLOR, 0.85)
    lines = [
        f"D2 Lane Count Ratio: {_fmt(record.get('D2_count_match'))}",
        f"GT Lane Count: {record.get('D2_gt_lane_count')}  |  Predicted Lane Count: {record.get('D2_pred_lane_count')}",
        f"Predicted Count Source: {record.get('D2_pred_count_source')}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    _save(out_path, _compose(small, annotated, "D2 Lane Count Metric", lines, _LEGEND_GT_PRED))


def render_d3(out_path: Path, raw_bgr: np.ndarray, gt_mask, pred_mask, record: dict,
              dataset: str, sample_id: str) -> np.ndarray:
    small, _ = _downscale(raw_bgr)
    gt, pred, missed = gt_pred_missed_masks(gt_mask, pred_mask, small.shape[:2])
    annotated = _overlay(small, gt, pred, missed)
    lines = [
        f"D3 Segmentation IoU: {_fmt(record.get('D3_iou'))}",
        f"Precision: {_fmt(record.get('D3_precision'))}   Recall: {_fmt(record.get('D3_recall'))}   F1: {_fmt(record.get('D3_f1'))}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    out = _compose(small, annotated, "D3 Segmentation IoU Metric", lines, _LEGEND_GT_PRED_MISSED)
    _save(out_path, out)
    return out


def render_d4(out_path: Path, raw_bgr: np.ndarray, gt_mask, pred_mask, record: dict,
              dataset: str, sample_id: str) -> None:
    small, _ = _downscale(raw_bgr)
    h = small.shape[0]
    gt, pred, missed = gt_pred_missed_masks(gt_mask, pred_mask, small.shape[:2])
    annotated = _overlay(small, gt, pred, missed)
    top = record.get("D4_region_top_row")
    bottom = record.get("D4_region_bottom_row")
    if top is None or bottom is None:
        top_f, bottom_f = NEAR_FIELD_TOP_FRACTION, NEAR_FIELD_BOTTOM_FRACTION
    else:
        # top/bottom in the record are source-resolution rows; rescale to
        # this panel's (downscaled) row space via their fraction of height.
        src_h = raw_bgr.shape[0]
        top_f, bottom_f = top / src_h, bottom / src_h
    top = int(round(h * top_f))
    bottom = int(round(h * bottom_f))
    annotated[:top] = (0.35 * annotated[:top]).astype(np.uint8)
    annotated[bottom:] = (0.35 * annotated[bottom:]).astype(np.uint8)
    cv2.line(annotated, (0, top), (annotated.shape[1], top), (255, 255, 255), 1)
    cv2.line(annotated, (0, bottom), (annotated.shape[1], bottom), (255, 255, 255), 1)
    lines = [
        f"D4 Near-Field IoU: {_fmt(record.get('D4_near_iou'))}   (full-image D3 IoU = {_fmt(record.get('D3_iou'))})",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    _save(out_path, _compose(small, annotated, "D4 Near-Field IoU Metric", lines, _LEGEND_GT_PRED_MISSED))


def render_d5(out_path: Path, raw_bgr: np.ndarray, gt_mask, pred_mask, record: dict,
              dataset: str, sample_id: str) -> None:
    small, _ = _downscale(raw_bgr)
    gt, pred, missed = gt_pred_missed_masks(gt_mask, pred_mask, small.shape[:2])
    annotated = _overlay(small, gt, pred, missed)
    lines = [
        f"Visibility Group: {record.get('D5_visibility_group')}",
        f"Observed Marking Visibility Tag: {record.get('visibility_tag')}",
        f"D3 IoU (this image): {_fmt(record.get('D3_iou'))}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    _save(out_path, _compose(small, annotated, "D5 Marking-Condition Sensitivity Metric", lines, _LEGEND_GT_PRED_MISSED))


def render_d6(out_path: Path, raw_bgr: np.ndarray, gt_mask, pred_mask, record: dict,
              dataset: str, sample_id: str) -> None:
    small, _ = _downscale(raw_bgr)
    gt, pred, missed = gt_pred_missed_masks(gt_mask, pred_mask, small.shape[:2])
    annotated = _overlay(small, gt, pred, missed)
    lines = [
        f"D6 Missed-Marking Ratio: {_fmt(record.get('D6_image_missed_ratio'))}",
        f"GT Pixels: {record.get('D6_gt_pixels')}  |  Missed Pixels: {record.get('D6_fn_pixels')}",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    _save(out_path, _compose(small, annotated, "D6 Missed-Marking Metric", lines, _LEGEND_GT_PRED_MISSED))


def render_r2(out_path: Path, raw_bgr: np.ndarray, gt_mask, pred_mask, r2_row: dict, model: str,
              dataset: str, sample_id: str) -> None:
    small, _ = _downscale(raw_bgr)
    gt, pred, missed = gt_pred_missed_masks(gt_mask, pred_mask, small.shape[:2])
    annotated = _overlay(small, gt, pred, missed)

    def v(key):
        val = r2_row.get(key)
        try:
            return None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
        except (TypeError, ValueError):
            return None

    f1 = v(f"{model}_f1")
    iou = v(f"{model}_iou")
    d4 = v(f"{model}_D4_lower")
    d6 = v(f"{model}_D6")
    detected = None if d6 is None else 1.0 - d6
    r2 = v(f"{model}_R2")
    lines = [
        f"R2 Reference-Detectability Score ({model}): {_fmt(r2)}",
        f"0.35*f1({_fmt(f1)}) + 0.20*iou({_fmt(iou)}) + 0.20*D4near({_fmt(d4)}) + 0.20*(1-D6)({_fmt(detected)})",
        f"Dataset Name: {dataset}",
        f"Image Name: {sample_id}.jpg",
    ]
    _save(out_path, _compose(small, annotated, "R2 Reference-Detectability Score", lines, _LEGEND_GT_PRED_MISSED))
