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
  D4 Near-Field IoU           -- D3 restricted to the near image band (rows
                                 80%-95% down, above the assumed ego-vehicle hood;
                                 documented region; no metric-distance claim).
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

from evaluation import marking_support
from evaluation.readiness_metrics import (
    _resize_mask_if_needed,
    _valid_lanes,
    safe_binary_mask,
    safe_mean,
)

D_METRIC_VERSION = "d_metrics_v1_draft_table16"

# D4 near-field region = the "lower/near" image band, ABOVE the assumed hood.
# y/H in [NEAR_FIELD_TOP_FRACTION, NEAR_FIELD_BOTTOM_FRACTION); the bottom
# HOOD_IGNORE_FRACTION of rows is excluded (assumed ego-vehicle hood). Bounds are
# derived from the shared band layout in marking_support so the legacy near-field
# D4 and the D4' band profile stay consistent.
NEAR_FIELD_TOP_FRACTION = 1.0 - (marking_support.NEAR_BAND_FRACTION + marking_support.HOOD_IGNORE_FRACTION)  # 0.80
NEAR_FIELD_BOTTOM_FRACTION = 1.0 - marking_support.HOOD_IGNORE_FRACTION  # 0.95
MIN_COMPONENT_AREA = 30    # px; suppresses speckle when counting mask components
MIN_GROUP_FOR_D5 = 3       # both visibility groups need >= this many images
D3PRIME_STEP_PX = 1.0      # polyline densification spacing for centerline points

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
    """Draft D2: fractional agreement of lane-boundary instance counts.

    GT count comes from the manifest lane_json instances. Predicted count uses
    native polylines when the processor provides them; otherwise it falls back
    to connected components of the merged mask, which the draft flags as
    approximate -- the fallback is recorded in D2_pred_count_source.

    D2_count_match is a per-image ratio in [0, 1], not a strict exact-match
    flag: min(pred_count, gt_count) / max(pred_count, gt_count). E.g. 3
    predicted lanes vs 4 GT lanes -> 0.75. Symmetric: over-counting (6 pred vs
    4 GT) is penalized the same as under-counting (4 pred vs 6 GT). 1.0 only
    when counts match exactly; None when GT count == 0 (undefined, excluded
    from aggregation rather than coerced to 0 or 1).
    """
    gt_count = len(lanes_from_lane_json(gt_lane_json))
    if gt_count == 0 and gt_mask is not None:
        gt_count = _component_lane_count(gt_mask)
    native = _valid_lanes(pred_lanes) if pred_lanes else []
    if native:
        pred_count, source = len(native), "native_polyline"
    else:
        pred_count, source = _component_lane_count(pred_mask), "connected_components"
    if gt_count == 0:
        match = None
    else:
        denom = max(pred_count, gt_count)
        match = (min(pred_count, gt_count) / denom) if denom else 1.0
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


def compute_d4_near_field_iou(
    pred_mask: Any,
    gt_mask: Any,
    top_fraction: float = NEAR_FIELD_TOP_FRACTION,
    bottom_fraction: float = NEAR_FIELD_BOTTOM_FRACTION,
) -> dict:
    """Draft D4: the D3 IoU restricted to the near image band, above the hood.

    The near band is ``y/H in [top_fraction, bottom_fraction)`` — by default rows
    80%-95% down the image. The bottom ``1 - bottom_fraction`` of rows (assumed
    ego-vehicle hood) is excluded; this is an image region, not a physical distance.
    """
    pred, gt = _aligned_masks(pred_mask, gt_mask)
    if pred.size == 0:
        return {"D4_near_iou": None, "D4_region_top_row": None, "D4_region_bottom_row": None}
    h = pred.shape[0]
    top = int(round(h * top_fraction))
    bottom = int(round(h * bottom_fraction))
    d3 = compute_d3_iou(pred[top:bottom, :], gt[top:bottom, :])
    return {"D4_near_iou": d3["D3_iou"], "D4_region_top_row": top, "D4_region_bottom_row": bottom}


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


def _centerline_points(
    mask: Any,
    lanes: Optional[List[np.ndarray]],
    step_px: float = D3PRIME_STEP_PX,
) -> tuple[np.ndarray, str]:
    """Reduce a marking representation to a centerline point cloud (x, y).

    Prefers native polylines (CLRerNet: densified, NOT thick-rasterized). Falls
    back to skeletonizing a dense mask (YOLOPX). Returns ``(points, source)``.
    """
    valid = _valid_lanes(lanes) if lanes else []
    if valid:
        return marking_support.polylines_to_points(valid, step_px), "native_polyline"
    m = safe_binary_mask(mask)
    if m.size and np.any(m):
        return marking_support.mask_to_points(m), "mask_skeleton"
    return np.zeros((0, 2), dtype=np.float64), "empty"


def compute_d3_prime(
    delta: float,
    width: int,
    height: int,
    pred_mask: Any = None,
    gt_mask: Any = None,
    pred_lanes: Optional[List[np.ndarray]] = None,
    gt_lanes: Optional[List[np.ndarray]] = None,
    step_px: float = D3PRIME_STEP_PX,
    bands: bool = True,
) -> dict:
    """D3' marking-support localization on an un-thickened centerline basis.

    Replaces the dilated pixel IoU of D3/D4/D6 with resolution-free,
    isotropically-normalized nearest-neighbour precision/recall/F1 and
    median/p95 localization error at a frozen tolerance ``delta`` (fraction of
    the image diagonal; see :mod:`evaluation.marking_support`). ``delta`` must be
    frozen on a calibration split, never tuned here.

    The prediction and GT are each reduced to centerline points (native polyline
    when available, else mask skeleton) at original-pixel scale, then
    canonicalized before scoring, so nothing is thickened and no metric-distance
    claim is made. Per-band (upper/middle/lower) scores replace "near-field IoU".
    """
    pred_xy, pred_src = _centerline_points(pred_mask, pred_lanes, step_px)
    gt_xy, gt_src = _centerline_points(gt_mask, gt_lanes, step_px)

    core = marking_support.localization_scores(
        marking_support.to_canonical(pred_xy, width, height),
        marking_support.to_canonical(gt_xy, width, height),
        delta,
    )
    out = {
        "D3p_support_f1": core["support_f1"],
        "D3p_support_precision": core["support_precision"],
        "D3p_support_recall": core["support_recall"],
        "D3p_loc_err_median": core["loc_err_median"],
        "D3p_loc_err_p95": core["loc_err_p95"],
        "D3p_n_pred_points": core["n_pred_points"],
        "D3p_n_gt_points": core["n_gt_points"],
        "D3p_pred_source": pred_src,
        "D3p_gt_source": gt_src,
        "D3p_delta": float(delta),
        # D6' — unsupported-marking response = fraction of reference marking with
        # no prediction within delta = 1 - recall (folded in, not shipped apart).
        "D6p_unsupported_marking_ratio": (
            None if core["support_recall"] is None else float(1.0 - core["support_recall"])
        ),
    }
    if bands:
        band = marking_support.band_localization(pred_xy, gt_xy, width, height, delta)
        for label, scores in band.items():
            out[f"D4p_{label}_f1"] = scores["support_f1"]
            out[f"D4p_{label}_recall"] = scores["support_recall"]
    return out


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
    delta: Optional[float] = None,
    gt_lanes: Optional[List[np.ndarray]] = None,
) -> dict:
    """Build one per-image D record.

    ``delta`` opts in to the D3' centerline-localization metric
    (:func:`compute_d3_prime`). It is the frozen tolerance as a fraction of the
    image diagonal and MUST come from a designated calibration split. When
    ``delta`` is ``None`` (default) the D3' fields are omitted and the record is
    exactly the legacy Table-16 record — so no published number changes until a
    calibration split is declared. The two bases are additive, never blended.
    """
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
    if delta is not None:
        # Derive image size from whichever mask carries a shape; fall back to GT.
        shape = None
        for m in (pred_mask, gt_mask):
            arr = np.asarray(m) if m is not None else None
            if arr is not None and arr.ndim >= 2 and arr.size:
                shape = arr.shape[:2]
                break
        if shape is not None:
            h, w = int(shape[0]), int(shape[1])
            if gt_lanes is None and gt_lane_json is not None:
                gt_lanes = lanes_from_lane_json(gt_lane_json) or None
            record.update(
                compute_d3_prime(
                    delta=delta, width=w, height=h,
                    pred_mask=pred_mask, gt_mask=gt_mask,
                    pred_lanes=pred_lanes, gt_lanes=gt_lanes,
                )
            )
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


def _summarize_d3_prime(eligible: list[dict]) -> Optional[dict]:
    """Aggregate the D3' centerline-localization fields, if any record carries
    them (i.e. build_d_record was called with a frozen delta). Per-image first,
    None excluded (never coerced to 0). Returns None when D3' was not computed."""
    have = [r for r in eligible if "D3p_support_f1" in r]
    if not have:
        return None
    f1 = [r["D3p_support_f1"] for r in have if r.get("D3p_support_f1") is not None]
    prec = [r["D3p_support_precision"] for r in have if r.get("D3p_support_precision") is not None]
    rec = [r["D3p_support_recall"] for r in have if r.get("D3p_support_recall") is not None]
    err_med = [r["D3p_loc_err_median"] for r in have if r.get("D3p_loc_err_median") is not None]
    err_p95 = [r["D3p_loc_err_p95"] for r in have if r.get("D3p_loc_err_p95") is not None]
    unsup = [r["D6p_unsupported_marking_ratio"] for r in have
             if r.get("D6p_unsupported_marking_ratio") is not None]
    deltas = {float(r["D3p_delta"]) for r in have if r.get("D3p_delta") is not None}
    out = {
        "D3p_num_scored": len(have),
        "D3p_delta": (deltas.pop() if len(deltas) == 1 else sorted(deltas)),
        "D3p_support_f1_mean": safe_mean(f1),
        "D3p_support_precision_mean": safe_mean(prec),
        "D3p_support_recall_mean": safe_mean(rec),
        "D3p_loc_err_median_mean": safe_mean(err_med),
        "D3p_loc_err_p95_mean": safe_mean(err_p95),
        "D6p_unsupported_marking_ratio_mean": safe_mean(unsup),
    }
    for label in ("upper", "middle", "lower"):
        vals = [r[f"D4p_{label}_f1"] for r in have if r.get(f"D4p_{label}_f1") is not None]
        out[f"D4p_{label}_f1_mean"] = safe_mean(vals)
    return out


def summarize_d_records(records: list[dict]) -> dict:
    eligible = [r for r in records if r["gt_eligible"]]
    d3 = [r["D3_iou"] for r in eligible if r["D3_iou"] is not None]
    d4 = [r["D4_near_iou"] for r in eligible if r["D4_near_iou"] is not None]
    d2 = [r["D2_count_match"] for r in eligible if r["D2_count_match"] is not None]
    gt_px = sum(r["D6_gt_pixels"] for r in eligible)
    fn_px = sum(r["D6_fn_pixels"] for r in eligible)
    d7 = [r["D7_confidence_mean"] for r in records if r["D7_confidence_mean"] is not None]
    n_standardized = sum(1 for r in records if r.get("stroke_standardized"))
    d3_prime = _summarize_d3_prime(eligible)
    summary = {
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
            f"D4 near-field region = image rows {int(NEAR_FIELD_TOP_FRACTION*100)}%-{int(NEAR_FIELD_BOTTOM_FRACTION*100)}% down "
            f"(bottom {int(marking_support.HOOD_IGNORE_FRACTION*100)}% excluded as assumed ego-vehicle hood); not a physical distance.",
            "D5 uses human 'Observed Marking Visibility' tags, not dark-pixel thresholds.",
            "D2 counts from merged masks (connected_components source) are approximate per the draft caveat.",
            "D6 is pixel-weighted across the group (sum FN / sum GT), not a mean of per-image ratios.",
        ],
    }
    if d3_prime is not None:
        summary["D3_prime"] = d3_prime
        summary["notes"].append(
            "D3' (D3p_*/D4p_*/D6p_*) is the un-thickened centerline-localization "
            "basis at a frozen delta (fraction of image diagonal); it does NOT "
            "dilate predictions and is reported alongside, never blended with, "
            "the legacy stroke-standardized D3/D4/D6."
        )
    return summary
