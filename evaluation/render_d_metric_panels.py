"""Side-by-side visualization panels for the draft-aligned D metrics.

For each evaluated image this writes, into <out_dir>/:
    <sid>__RAW.jpg   raw source frame
    <sid>__PRED.jpg  prediction overlaid on the raw frame
    <sid>__D1.jpg .. <sid>__D6.jpg
                     [raw | annotated] panels, score burned into a header bar,
                     with the pixels that drive the score highlighted:
                       D1  predicted lane output (green) or NO-DETECTION banner
                       D2  GT lane instances vs predicted instances, colored per
                           instance, counts in the header
                       D3  TP green / FN red / FP orange
                       D4  same coding, upper (far-field) region dimmed out
                       D5  D3 coding + the image's visibility group, and the
                           group-level gap it contributes to
                       D6  missed GT pixels (FN) in red -- the numerator of the
                           group detection-gap ratio

D7 has no panel: confidence is unavailable for these cached predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from evaluation.d_metrics import NEAR_FIELD_FRACTION, lanes_from_lane_json

_TP_COLOR = (60, 200, 60)     # BGR green
_FN_COLOR = (40, 40, 230)     # BGR red
_FP_COLOR = (30, 150, 255)    # BGR orange
_PRED_COLOR = (230, 200, 40)  # BGR cyan-ish for plain prediction display
_INSTANCE_COLORS = [
    (60, 200, 60), (40, 40, 230), (30, 150, 255), (230, 200, 40),
    (200, 60, 200), (60, 220, 220), (200, 120, 40), (120, 220, 120),
]


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def _fit(mask: np.ndarray, shape_hw) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.shape != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def _dilate_for_display(mask: np.ndarray, image_h: int) -> np.ndarray:
    k = max(3, image_h // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def _blend(base_bgr: np.ndarray, mask: np.ndarray, color, alpha: float = 0.65) -> np.ndarray:
    out = base_bgr.copy()
    sel = mask > 0
    overlay = np.empty_like(out)
    overlay[:] = color
    out[sel] = (alpha * overlay[sel] + (1 - alpha) * out[sel]).astype(np.uint8)
    return out


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _panel(left_bgr: np.ndarray, right_bgr: np.ndarray, header_lines: list[str],
           max_side: int, header_color=(255, 255, 255), legend: Optional[list[tuple[str, tuple]]] = None) -> np.ndarray:
    left = _downscale(left_bgr, max_side)
    right = _downscale(right_bgr, max_side)
    if left.shape[0] != right.shape[0]:
        h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    gap = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    body = np.hstack([left, gap, right])

    line_h = 26
    header_h = 14 + line_h * len(header_lines) + (30 if legend else 0)
    header = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(header_lines):
        cv2.putText(header, line, (10, 28 + line_h * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    header_color if i == 0 else (200, 200, 200), 1, cv2.LINE_AA)
    if legend:
        x = 10
        y = header_h - 10
        for label, color in legend:
            cv2.rectangle(header, (x, y - 14), (x + 18, y), color, -1)
            cv2.putText(header, label, (x + 24, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1, cv2.LINE_AA)
            x += 24 + 12 * len(label) + 24
    return np.vstack([header, body])


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _save(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def render_sample_panels(
    out_dir: Path,
    sample_id: str,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    record: dict,
    group_summary: Optional[dict] = None,
    max_side: int = 640,
    gt_lane_json: Optional[dict] = None,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = sample_id.replace("/", "_")
    base = _to_bgr(image_rgb)
    h, w = base.shape[:2]
    gt = _fit(gt_mask, (h, w))
    pred = _fit(pred_mask, (h, w))
    gt_d = _dilate_for_display(gt, h)
    pred_d = _dilate_for_display(pred, h)
    tp_d = _dilate_for_display(((pred > 0) & (gt > 0)).astype(np.uint8), h)
    fn_d = _dilate_for_display(((pred == 0) & (gt > 0)).astype(np.uint8), h)
    fp_d = _dilate_for_display(((pred > 0) & (gt == 0)).astype(np.uint8), h)
    n = 0

    # RAW + PRED
    _save(out_dir / f"{sid}__RAW.jpg", _downscale(base, max_side)); n += 1
    _save(out_dir / f"{sid}__PRED.jpg", _downscale(_blend(base, pred_d, _PRED_COLOR), max_side)); n += 1

    # D1 -- detection presence
    detected = bool(record.get("D1_detection"))
    d1_img = _blend(base, pred_d, _TP_COLOR if detected else _FN_COLOR)
    if not detected:
        cv2.putText(d1_img, "NO LANE OUTPUT", (w // 6, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 255), 3, cv2.LINE_AA)
    _save(out_dir / f"{sid}__D1.jpg", _panel(
        base, d1_img,
        [f"D1 Detection Success = {int(detected)}   ({record.get('D1_pred_pixel_count', 0)} predicted lane px)",
         "Presence of any lane output; does not measure accuracy."],
        max_side, legend=[("predicted output", _TP_COLOR if detected else _FN_COLOR)])); n += 1

    # D2 -- lane instance counts: GT instances left, predicted instances right
    gt_view = base.copy()
    for i, lane in enumerate(lanes_from_lane_json(gt_lane_json)):
        pts = lane.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(gt_view, [pts], False, _INSTANCE_COLORS[i % len(_INSTANCE_COLORS)],
                      max(3, h // 200), cv2.LINE_AA)
    pred_view = base.copy()
    num, comp = cv2.connectedComponents(pred, connectivity=8)
    shown = 0
    for i in range(1, num):
        sel = comp == i
        if int(np.count_nonzero(sel)) < 30:
            continue
        color = _INSTANCE_COLORS[shown % len(_INSTANCE_COLORS)]
        pred_view = _blend(pred_view, _dilate_for_display(sel.astype(np.uint8), h), color)
        shown += 1
    match = record.get("D2_count_match")
    verdict = "match" if match == 1 else ("MISMATCH" if match == 0 else "not eligible")
    _save(out_dir / f"{sid}__D2.jpg", _panel(
        gt_view, pred_view,
        [f"D2 Lane Count: GT={record.get('D2_gt_lane_count')}  pred={record.get('D2_pred_lane_count')}  -> {verdict}",
         f"pred count source: {record.get('D2_pred_count_source')}"
         + ("  (component counts are approximate)" if record.get("D2_pred_count_source") == "connected_components" else "")],
        max_side,
        header_color=(80, 220, 80) if match == 1 else (60, 60, 255))); n += 1

    # D3 -- segmentation IoU with TP/FN/FP error map
    d3_img = _blend(_blend(_blend(base, tp_d, _TP_COLOR), fn_d, _FN_COLOR), fp_d, _FP_COLOR)
    _save(out_dir / f"{sid}__D3.jpg", _panel(
        base, d3_img,
        [f"D3 Segmentation IoU = {_fmt(record.get('D3_iou'))}",
         f"precision {_fmt(record.get('D3_precision'))}   recall {_fmt(record.get('D3_recall'))}   f1 {_fmt(record.get('D3_f1'))}"],
        max_side,
        legend=[("TP (agree)", _TP_COLOR), ("FN (missed GT)", _FN_COLOR), ("FP (false pred)", _FP_COLOR)])); n += 1

    # D4 -- near-field IoU: same error map, far field dimmed
    top = int(round(h * (1.0 - NEAR_FIELD_FRACTION)))
    d4_img = d3_img.copy()
    d4_img[:top] = (0.30 * d4_img[:top]).astype(np.uint8)
    cv2.line(d4_img, (0, top), (w, top), (255, 255, 255), 2)
    cv2.putText(d4_img, "near-field region", (10, min(h - 10, top + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    _save(out_dir / f"{sid}__D4.jpg", _panel(
        base, d4_img,
        [f"D4 Near-Field IoU = {_fmt(record.get('D4_near_iou'))}   (full-image D3 = {_fmt(record.get('D3_iou'))})",
         f"lower {int(NEAR_FIELD_FRACTION*100)}% of image rows; not a physical distance"],
        max_side,
        legend=[("TP", _TP_COLOR), ("FN", _FN_COLOR), ("FP", _FP_COLOR)])); n += 1

    # D5 -- occlusion/visibility group membership + group gap context
    group = record.get("D5_visibility_group")
    lines = [f"D5 group: {group}   tag: {record.get('visibility_tag')}",
             f"this image's IoU = {_fmt(record.get('D3_iou'))} contributes to the group mean"]
    if group_summary:
        gap = (group_summary.get("D5_visibility_degraded_gap") or {})
        lines.append(
            f"group gap (clear - degraded) = {_fmt(gap.get('gap'))}  "
            f"[clear n={gap.get('n_reference')} mean {_fmt(gap.get('mean_iou_reference'))} | "
            f"degraded n={gap.get('n_comparison')} mean {_fmt(gap.get('mean_iou_comparison'))}]")
    d5_color = {"clear": (80, 220, 80), "degraded": (30, 150, 255),
                "occluded": (60, 60, 255)}.get(group, (200, 200, 200))
    _save(out_dir / f"{sid}__D5.jpg", _panel(base, d3_img, lines, max_side, header_color=d5_color,
                                             legend=[("TP", _TP_COLOR), ("FN", _FN_COLOR), ("FP", _FP_COLOR)])); n += 1

    # D6 -- detection gap: only the missed GT pixels
    d6_img = _blend(_blend(base, gt_d, (110, 110, 110), alpha=0.35), fn_d, _FN_COLOR)
    lines = [f"D6 missed-GT ratio (this image) = {_fmt(record.get('D6_image_missed_ratio'))}   "
             f"({record.get('D6_fn_pixels')}/{record.get('D6_gt_pixels')} GT px missed)"]
    if group_summary:
        lines.append(f"group Detection Gap Ratio (sum FN / sum GT) = {_fmt(group_summary.get('D6_detection_gap_ratio'))}")
    _save(out_dir / f"{sid}__D6.jpg", _panel(
        base, d6_img, lines, max_side,
        legend=[("GT marking", (110, 110, 110)), ("missed (FN)", _FN_COLOR)])); n += 1

    return n
