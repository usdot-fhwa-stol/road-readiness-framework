"""Draft-aligned Detection Performance Indicators (D1-D7).

Implements Table 16 of the FHWA pilot-analysis draft exactly as specified,
replacing the earlier readiness_metrics variants where they diverged:

  D1 Detection Success Rate   -- share of processed images whose standardized
                                 output contains >=1 predicted lane pixel or
                                 lane-boundary instance (no accuracy gating).
  D2 Lane Count Accuracy      -- share of eligible images where the predicted
                                 lane-boundary instance count equals the GT
                                 instance count. Native polylines are used when
                                 available; connected-component counts are
                                 flagged as approximate per the draft caveat.
  D3 Segmentation IoU         -- per-image TP/(TP+FP+FN), summarized over the
                                 image group.
  D4 Near-Field IoU           -- D3 restricted to the lower half of the image
                                 (documented region; no metric-distance claim).
  D5 Occlusion Robustness Gap -- mean IoU(nonoccluded) - mean IoU(occluded)
                                 using the human "Observed Marking Visibility"
                                 tags, NOT dark-pixel thresholds. Reported only
                                 when both groups have >= min_group images; a
                                 broader visibility-degraded variant is also
                                 reported because strictly-occluded tags are
                                 rare in the pilot subset.
  D6 Detection Gap Ratio      -- sum(FN pixels) / sum(GT pixels) across the
                                 image group (pixel-weighted, not mean-of-ratios).
  D7 Confidence Mean          -- mean lane-output confidence where available;
                                 reported as unavailable otherwise.
"""
from __future__ import annotations

from typing import Any, List, Optional

import cv2
import numpy as np

from evaluation.readiness_metrics import (
    _resize_mask_if_needed,
    _valid_lanes,
    safe_binary_mask,
    safe_mean,
)

D_METRIC_VERSION = "d_metrics_v1_draft_table16"

NEAR_FIELD_FRACTION = 0.5  # documented lower-image region for D4: lower half
MIN_COMPONENT_AREA = 30    # px; suppresses speckle when counting mask components
MIN_GROUP_FOR_D5 = 3       # both visibility groups need >= this many images

# "Observed Marking Visibility" tag -> D5 grouping. Tags may be ';'-joined.
_OCCLUDED_TAGS = {"occluded", "partially missing"}
_DEGRADED_TAGS = {"faded / worn / not visible", "low contrast"} | _OCCLUDED_TAGS
_CLEAR_TAGS = {"clearly visible"}


# ---------------------------------------------------------------------------
# per-image computations
# ---------------------------------------------------------------------------

def _aligned_masks(pred_mask: Any, gt_mask: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = safe_binary_mask(pred_mask)
    gt = _resize_mask_if_needed(gt_mask, pred.shape) if pred.size else safe_binary_mask(gt_mask)
    if pred.size == 0 and gt.size != 0:
        pred = np.zeros_like(gt)
    if gt.size == 0 and pred.size != 0:
        gt = np.zeros_like(pred)
    return pred, gt


def compute_d1_detection(pred_mask: Any, pred_lanes: Optional[List[np.ndarray]] = None) -> dict:
    """Draft D1: any lane-related output at all (presence, not correctness)."""
    pred = safe_binary_mask(pred_mask)
    pixel_count = int(np.count_nonzero(pred))
    lane_count = len(_valid_lanes(pred_lanes)) if pred_lanes else 0
    return {
        "D1_detection": int(pixel_count > 0 or lane_count > 0),
        "D1_pred_pixel_count": pixel_count,
    }


def _component_lane_count(mask: Any, min_area: int = MIN_COMPONENT_AREA) -> int:
    m = safe_binary_mask(mask)
    if m.size == 0 or not np.any(m):
        return 0
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return int(sum(1 for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area))


def lanes_from_lane_json(lane_json: Optional[dict]) -> list[np.ndarray]:
    """TuSimple-style lane_json -> list of Nx2 polylines (valid points only)."""
    if not lane_json:
        return []
    h_samples = lane_json.get("h_samples") or []
    out = []
    for xs in lane_json.get("lanes") or []:
        pts = [(float(x), float(y)) for x, y in zip(xs, h_samples) if x is not None and x >= 0]
        if len(pts) >= 2:
            out.append(np.asarray(pts, dtype=np.float64))
    return out


def compute_d2_lane_count(
    pred_mask: Any,
    gt_lane_json: Optional[dict],
    gt_mask: Any = None,
    pred_lanes: Optional[List[np.ndarray]] = None,
) -> dict:
    """Draft D2: exact agreement of lane-boundary instance counts.

    GT count comes from the manifest lane_json instances. Predicted count uses
    native polylines when the processor provides them; otherwise it falls back
    to connected components of the merged mask, which the draft flags as
    approximate -- the fallback is recorded in D2_pred_count_source.
    """
    gt_count = len(lanes_from_lane_json(gt_lane_json))
    if gt_count == 0 and gt_mask is not None:
        gt_count = _component_lane_count(gt_mask)
    native = _valid_lanes(pred_lanes) if pred_lanes else []
    if native:
        pred_count, source = len(native), "native_polyline"
    else:
        pred_count, source = _component_lane_count(pred_mask), "connected_components"
    match = None if gt_count == 0 else int(pred_count == gt_count)
    return {
        "D2_gt_lane_count": gt_count,
        "D2_pred_lane_count": pred_count,
        "D2_pred_count_source": source,
        "D2_count_match": match,
    }


def compute_d3_iou(pred_mask: Any, gt_mask: Any) -> dict:
    """Draft D3: plain pixel IoU = TP / (TP + FP + FN), plus context P/R/F1."""
    pred, gt = _aligned_masks(pred_mask, gt_mask)
    if gt.size == 0 or not np.any(gt):
        return {"D3_iou": None, "D3_tp": 0, "D3_fp": int(np.count_nonzero(pred)), "D3_fn": 0,
                "D3_precision": None, "D3_recall": None, "D3_f1": None}
    tp = int(np.count_nonzero((pred > 0) & (gt > 0)))
    fp = int(np.count_nonzero((pred > 0) & (gt == 0)))
    fn = int(np.count_nonzero((pred == 0) & (gt > 0)))
    denom = tp + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None or recall is not None else None)
    return {
        "D3_iou": float(tp / denom) if denom else None,
        "D3_tp": tp, "D3_fp": fp, "D3_fn": fn,
        "D3_precision": precision, "D3_recall": recall, "D3_f1": f1,
    }


def compute_d4_near_field_iou(pred_mask: Any, gt_mask: Any, fraction: float = NEAR_FIELD_FRACTION) -> dict:
    """Draft D4: the D3 IoU restricted to the documented lower image region."""
    pred, gt = _aligned_masks(pred_mask, gt_mask)
    if pred.size == 0:
        return {"D4_near_iou": None, "D4_region_top_row": None}
    top = int(round(pred.shape[0] * (1.0 - fraction)))
    d3 = compute_d3_iou(pred[top:, :], gt[top:, :])
    return {"D4_near_iou": d3["D3_iou"], "D4_region_top_row": top}


def compute_d6_image_counts(pred_mask: Any, gt_mask: Any) -> dict:
    """Per-image FN / GT pixel counts; the group D6 ratio sums these."""
    pred, gt = _aligned_masks(pred_mask, gt_mask)
    gt_px = int(np.count_nonzero(gt))
    fn_px = int(np.count_nonzero((pred == 0) & (gt > 0)))
    return {
        "D6_gt_pixels": gt_px,
        "D6_fn_pixels": fn_px,
        "D6_image_missed_ratio": float(fn_px / gt_px) if gt_px else None,
    }


def compute_d7_confidence(lane_prob: Optional[np.ndarray], pred_mask: Any) -> dict:
    if lane_prob is None:
        return {"D7_confidence_mean": None}
    prob = np.clip(np.asarray(lane_prob, dtype=np.float32), 0.0, 1.0)
    if prob.size == 0:
        return {"D7_confidence_mean": None}
    pred = _resize_mask_if_needed(pred_mask, prob.shape)
    sel = pred > 0
    return {"D7_confidence_mean": float(np.mean(prob[sel])) if np.any(sel) else 0.0}


def effective_stroke_width(mask: Any) -> float:
    """Mean marking thickness in px, via the distance transform (2x mean
    interior distance). Rough but stable enough to compare representations."""
    m = safe_binary_mask(mask)
    if m.size == 0 or not np.any(m):
        return 0.0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return 2.0 * float(np.mean(dist[m > 0]))


def standardize_stroke_width(pred_mask: Any, gt_mask: Any, tolerance: float = 1.2) -> tuple[np.ndarray, dict]:
    """Chapter-3 common-representation step: if the predicted marking stroke is
    materially thinner than the reference stroke (as with polyline processors
    rasterized at a few px vs the ~16 px reference rendering), dilate the
    prediction to the reference's effective width so pixel IoU compares
    placement rather than rasterization width. Never erodes; applied uniformly
    to every processor."""
    pred = safe_binary_mask(pred_mask)
    info = {"stroke_standardized": False, "pred_stroke_px": None, "gt_stroke_px": None}
    if pred.size == 0 or not np.any(pred):
        return pred, info
    gt = _resize_mask_if_needed(gt_mask, pred.shape)
    wp, wg = effective_stroke_width(pred), effective_stroke_width(gt)
    info["pred_stroke_px"] = round(wp, 2)
    info["gt_stroke_px"] = round(wg, 2)
    if wg <= 0 or wg <= wp * tolerance:
        return pred, info
    k = max(3, int(round(wg - wp)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    info["stroke_standardized"] = True
    info["stroke_dilation_kernel_px"] = k
    return cv2.dilate(pred, kernel), info


def visibility_group(visibility_tag: Optional[str]) -> dict:
    """Map the human 'Observed Marking Visibility' tag to D5 groups."""
    tags = [t.strip().lower() for t in (visibility_tag or "").split(";") if t.strip()]
    occluded = any(t in _OCCLUDED_TAGS for t in tags)
    degraded = any(t in _DEGRADED_TAGS for t in tags)
    clear = any(t in _CLEAR_TAGS for t in tags) and not degraded
    if occluded:
        group = "occluded"
    elif degraded:
        group = "degraded"
    elif clear:
        group = "clear"
    else:
        group = "not_applicable"
    return {"visibility_tag": visibility_tag, "D5_visibility_group": group}


def build_d_record(
    sample_id: str,
    pred_mask: Any,
    gt_mask: Any,
    gt_lane_json: Optional[dict] = None,
    pred_lanes: Optional[List[np.ndarray]] = None,
    lane_prob: Optional[np.ndarray] = None,
    visibility_tag: Optional[str] = None,
    image_path: Optional[str] = None,
    match_stroke_width: bool = True,
) -> dict:
    record: dict = {"sample_id": sample_id, "image_path": image_path}
    # D1/D2 use the raw output; the width standardization below only affects the
    # pixel-agreement measures (D3/D4/D6), matching the chapter-3 representation.
    record.update(compute_d1_detection(pred_mask, pred_lanes))
    record.update(compute_d2_lane_count(pred_mask, gt_lane_json, gt_mask=gt_mask, pred_lanes=pred_lanes))
    if match_stroke_width:
        pixel_pred, stroke_info = standardize_stroke_width(pred_mask, gt_mask)
        record.update(stroke_info)
    else:
        pixel_pred = pred_mask
    record.update(compute_d3_iou(pixel_pred, gt_mask))
    record.update(compute_d4_near_field_iou(pixel_pred, gt_mask))
    record.update(compute_d6_image_counts(pixel_pred, gt_mask))
    record.update(compute_d7_confidence(lane_prob, pred_mask))
    record.update(visibility_group(visibility_tag))
    record["gt_eligible"] = int(record["D6_gt_pixels"] > 0)
    return record


# ---------------------------------------------------------------------------
# group summarization
# ---------------------------------------------------------------------------

def _gap(records: list[dict], reference_group: str, comparison_groups: set[str]) -> dict:
    ref = [r["D3_iou"] for r in records if r["D5_visibility_group"] == reference_group and r["D3_iou"] is not None]
    cmp_ = [r["D3_iou"] for r in records if r["D5_visibility_group"] in comparison_groups and r["D3_iou"] is not None]
    result = {"n_reference": len(ref), "n_comparison": len(cmp_),
              "mean_iou_reference": safe_mean(ref), "mean_iou_comparison": safe_mean(cmp_)}
    if len(ref) >= MIN_GROUP_FOR_D5 and len(cmp_) >= MIN_GROUP_FOR_D5:
        result["gap"] = float(np.mean(ref) - np.mean(cmp_))
    else:
        result["gap"] = None
        result["reason"] = f"insufficient group size (need >= {MIN_GROUP_FOR_D5} per group)"
    return result


def summarize_d_records(records: list[dict]) -> dict:
    eligible = [r for r in records if r["gt_eligible"]]
    d3 = [r["D3_iou"] for r in eligible if r["D3_iou"] is not None]
    d4 = [r["D4_near_iou"] for r in eligible if r["D4_near_iou"] is not None]
    d2 = [r["D2_count_match"] for r in eligible if r["D2_count_match"] is not None]
    gt_px = sum(r["D6_gt_pixels"] for r in eligible)
    fn_px = sum(r["D6_fn_pixels"] for r in eligible)
    d7 = [r["D7_confidence_mean"] for r in records if r["D7_confidence_mean"] is not None]
    n_standardized = sum(1 for r in records if r.get("stroke_standardized"))
    return {
        "metric_version": D_METRIC_VERSION,
        "num_processed_images": len(records),
        "num_annotation_eligible": len(eligible),
        "D1_detection_success_rate": safe_mean([r["D1_detection"] for r in records]),
        "D2_lane_count_accuracy": safe_mean(d2),
        "D2_num_eligible": len(d2),
        "D2_component_fallback_share": safe_mean(
            [1.0 if r["D2_pred_count_source"] == "connected_components" else 0.0 for r in eligible]
        ),
        "D3_mean_iou": safe_mean(d3),
        "D3_iou_std": float(np.std(d3)) if d3 else None,
        "D3_iou_median": float(np.median(d3)) if d3 else None,
        "D4_mean_near_field_iou": safe_mean(d4),
        "D5_occlusion_robustness_gap": _gap(eligible, "clear", {"occluded"}),
        "D5_visibility_degraded_gap": _gap(eligible, "clear", {"occluded", "degraded"}),
        "D6_detection_gap_ratio": float(fn_px / gt_px) if gt_px else None,
        "D6_total_gt_pixels": gt_px,
        "D6_total_fn_pixels": fn_px,
        "D7_confidence_mean": safe_mean(d7) if d7 else None,
        "D7_available": bool(d7),
        "num_stroke_standardized": n_standardized,
        "notes": [
            "D3/D4/D6 are computed on stroke-width-standardized predictions "
            "(thin rasterized outputs dilated to the reference stroke width, "
            "per the chapter-3 common evaluation representation).",
            "D1 counts output presence only; it does not gate on accuracy (draft Table 16).",
            f"D4 near-field region = lower {int(NEAR_FIELD_FRACTION*100)}% of image rows; not a physical distance.",
            "D5 uses human 'Observed Marking Visibility' tags, not dark-pixel thresholds.",
            "D2 counts from merged masks (connected_components source) are approximate per the draft caveat.",
            "D6 is pixel-weighted across the group (sum FN / sum GT), not a mean of per-image ratios.",
        ],
    }
