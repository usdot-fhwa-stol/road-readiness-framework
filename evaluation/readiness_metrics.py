"""Lane-marking machine-readability and reference-detectability metrics.

Canonical v2 flow:
    build_metric_record(sample, pred_mask, image_rgb, ...)
    summarize_records(records)

Legacy function names remain as wrappers, but canonical I/D/C metrics are 0..1
and R1/R2 are 0..100.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr

from evaluation.lane_continuity import (
    LaneContinuityConfig,
    analyze_lane_pattern_continuity,
    extract_guidance_lanes_from_mask,
    lanes_from_lane_json,
    prepare_guidance_lanes,
)
from evaluation.lane_width_stability import (
    LaneWidthConfig,
    analyze_lane_width_stability,
)
from evaluation.lane_geometry_complexity import (
    GeometryComplexityConfig,
    analyze_lane_geometry_complexity,
)
from evaluation.lane_wear import analyze_lane_wear
from evaluation.lane_width_simple import analyze_lane_width_simple
EPS = 1e-6
# v8 repoints canonical I1/I4 and all of D1-D7 at the single draft-Table-16/
# Table-15-aligned engines (evaluation/lane_wear.py, evaluation/lane_width_simple.py,
# evaluation/d_metrics.py) so there is exactly one place each metric is computed.
# lane_continuity.py/lane_width_stability.py (the earlier "heavy" I1/I4 engines)
# are kept only as internal geometry/context plumbing that I5 still depends on --
# their own top-level scores are no longer reported as I1/I4. Operational
# I1_pred/I4_pred (scored against predicted, not GT, geometry) have been dropped;
# they were never part of the draft spec. I5/I5_pred are unaffected.
METRIC_VERSION = "readiness_metrics_v8_draft_table16_consolidated"


def safe_binary_mask(mask: Any) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {arr.shape}")
    return (arr > 0).astype(np.uint8)


def safe_mean(values: Iterable[Any]) -> Optional[float]:
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fval):
            vals.append(fval)
    return float(np.mean(vals)) if vals else None


def weighted_mean_ignore_none(metric_dict: dict, weights: dict[str, float]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for key, weight in weights.items():
        value = _finite_or_none(metric_dict.get(key))
        if value is None:
            continue
        num += value * float(weight)
        den += float(weight)
    return None if den <= 0 else float(num / den)


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    return fval if np.isfinite(fval) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        fval = float(value)
        return fval if np.isfinite(fval) else None
    return value


def _to_gray_float(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
    return cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)


def _odd(size: int, minimum: int = 3) -> int:
    size = max(int(size), minimum)
    return size if size % 2 else size + 1


def _resize_mask_if_needed(mask: Any, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = safe_binary_mask(mask)
    if mask.shape == shape_hw:
        return mask
    if mask.size == 0:
        return np.zeros(shape_hw, dtype=np.uint8)
    return cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))


def _valid_lanes(lanes: Optional[List[np.ndarray]]) -> list[np.ndarray]:
    valid = []
    for lane in lanes or []:
        arr = np.asarray(lane, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
            valid.append(arr[:, :2])
    return valid


def _flatten_meta(meta: Any) -> list[str]:
    vals = []
    if isinstance(meta, dict):
        for key, value in meta.items():
            vals.append(str(key).lower())
            vals.extend(_flatten_meta(value))
    elif isinstance(meta, (list, tuple, set)):
        for value in meta:
            vals.extend(_flatten_meta(value))
    elif meta is not None:
        vals.append(str(meta).lower())
    return vals


def _component_infos(mask: Any, min_area: Optional[int] = None) -> list[dict]:
    binary = safe_binary_mask(mask)
    if binary.size == 0 or _mask_area(binary) == 0:
        return []
    if min_area is None:
        min_area = int(max(5, min(80, _mask_area(binary) * 0.002)))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    infos = []
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        coords_yx = np.column_stack(np.where(labels == idx))
        if coords_yx.shape[0] < 2:
            continue
        coords_xy = coords_yx[:, ::-1].astype(np.float32)
        centered = coords_xy - coords_xy.mean(axis=0, keepdims=True)
        if centered.shape[0] >= 3:
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            major = max(float(np.sqrt(max(eigvals[0], 0.0)) * 4.0), float(max(w, h)))
            minor = max(float(np.sqrt(max(eigvals[-1], 0.0)) * 4.0), float(min(w, h)), 1.0)
            angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        else:
            major = float(max(w, h))
            minor = float(max(1, min(w, h)))
            angle = 0.0
        infos.append({
            "label": idx,
            "area": area,
            "bbox": (x, y, w, h),
            "centroid_x": float(centroids[idx][0]),
            "centroid_y": float(centroids[idx][1]),
            "major_length": major,
            "minor_length": minor,
            "elongation": float(major / (minor + EPS)),
            "orientation_deg": angle,
        })
    return infos


def _component_count(mask: Any) -> int:
    return len(_component_infos(mask))


def _estimate_marking_instances(mask: Any, lanes: Optional[List[np.ndarray]] = None) -> int:
    valid_lanes = _valid_lanes(lanes)
    if valid_lanes:
        return len(valid_lanes)
    binary = safe_binary_mask(mask)
    if binary.size == 0 or _mask_area(binary) == 0:
        return 0
    raw_count = _component_count(binary)
    if raw_count <= 1:
        return raw_count
    h, w = binary.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_odd(max(3, min(9, w // 160))), _odd(max(21, min(91, h // 10)), 21)),
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    grouped = _component_count(closed)
    return int(max(1, min(raw_count, grouped or raw_count)))


def _pattern_from_meta(meta: Optional[dict]) -> Optional[str]:
    text = " ".join(_flatten_meta(meta or {}))
    has_dashed = any(t in text for t in ("dash", "dashed", "broken", "dotted", "skip", "segmented"))
    has_solid = any(t in text for t in ("solid", "continuous"))
    if has_dashed and has_solid:
        return "mixed"
    if has_dashed:
        return "dashed"
    if has_solid:
        return "solid"
    return None


def _dash_geometry_stats(infos: list[dict]) -> dict:
    if len(infos) < 3:
        return {"aligned": False, "spacing_cv": None, "duty_cycle": None, "alignment_score": None, "regularity_score": None}
    centers = np.array([[i["centroid_x"], i["centroid_y"]] for i in infos], dtype=np.float32)
    centered = centers - centers.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    coords = centered @ eigvecs[:, 0]
    residual = centered @ eigvecs[:, 1]
    spread = float(np.std(coords))
    residual_ratio = float(np.std(residual) / (spread + EPS))
    alignment_score = float(np.clip(1.0 - residual_ratio / 0.55, 0.0, 1.0))
    spacings = np.diff(np.sort(coords))
    spacings = spacings[spacings > 1.0]
    if spacings.size:
        median_spacing = float(np.median(spacings))
        spacing_cv = float(np.std(spacings) / (median_spacing + EPS))
        spacing_score = float(np.clip(1.0 - spacing_cv / 0.75, 0.0, 1.0))
        duty_cycle = float(np.clip(np.median([i["major_length"] for i in infos]) / (median_spacing + EPS), 0.0, 1.0))
        if duty_cycle < 0.2:
            duty_score = float(np.clip(duty_cycle / 0.2, 0.0, 1.0) * 0.6)
        elif duty_cycle > 0.85:
            duty_score = float(np.clip((1.0 - duty_cycle) / 0.15, 0.0, 1.0) * 0.8)
        else:
            duty_score = 1.0
    else:
        spacing_cv = None
        spacing_score = 0.0
        duty_cycle = None
        duty_score = 0.0
    regularity = float(np.clip(0.40 * alignment_score + 0.35 * spacing_score + 0.25 * duty_score, 0.0, 1.0))
    return {
        "aligned": bool(alignment_score >= 0.45),
        "spacing_cv": spacing_cv,
        "duty_cycle": duty_cycle,
        "alignment_score": alignment_score,
        "regularity_score": regularity,
    }


def _thickness_values(mask: Any) -> list[float]:
    vals = []
    for info in _component_infos(mask):
        major = max(float(info["major_length"]), 1.0)
        thickness = float(info["area"] / major)
        if np.isfinite(thickness) and thickness > 0:
            vals.append(thickness)
    return vals


def _thickness_stats(mask: Any) -> dict:
    vals = _thickness_values(mask)
    if not vals:
        return {"score": None, "mean_thickness_px": None, "thickness_cv": None, "num_segments": 0}
    mean = float(np.mean(vals))
    cv = 0.0 if len(vals) == 1 else float(np.std(vals) / (mean + EPS))
    return {
        "score": float(1.0 - np.clip(cv, 0.0, 1.0)),
        "mean_thickness_px": mean,
        "thickness_cv": cv,
        "num_segments": len(vals),
    }


def compute_local_background_ring(gt_mask: np.ndarray, outer_kernel: int = 21, inner_kernel: int = 5, road_mask: Optional[np.ndarray] = None) -> np.ndarray:
    mask = safe_binary_mask(gt_mask)
    if mask.size == 0:
        return mask.astype(bool)
    outer_kernel = _odd(outer_kernel)
    inner_kernel = _odd(inner_kernel)
    outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_kernel, outer_kernel)))
    inner = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_kernel, inner_kernel)))
    ring = (outer > 0) & ~(inner > 0) & ~(mask > 0)
    if road_mask is not None:
        ring &= _resize_mask_if_needed(road_mask, mask.shape).astype(bool)
    return ring


def infer_marking_pattern(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None) -> dict:
    mask = safe_binary_mask(gt_mask)
    infos = _component_infos(mask)
    component_count = len(infos)
    instance_count = _estimate_marking_instances(mask, lanes)
    meta_pattern = _pattern_from_meta(meta)
    dash_stats = _dash_geometry_stats(infos)
    notes = []
    if meta_pattern:
        pattern_type = meta_pattern
        notes.append("pattern inferred from metadata")
    elif component_count == 0:
        pattern_type = "unknown"
        notes.append("empty marking mask")
    elif component_count >= max(4, instance_count * 3 if instance_count else 4):
        median_elong = float(np.median([i["elongation"] for i in infos])) if infos else 0.0
        duty = dash_stats.get("duty_cycle")
        spacing_cv = dash_stats.get("spacing_cv")
        if dash_stats.get("aligned") and median_elong >= 1.2 and (spacing_cv is None or spacing_cv <= 0.95) and (duty is None or duty <= 0.85):
            pattern_type = "dashed"
            notes.append("multiple aligned repeated components")
        else:
            pattern_type = "mixed"
            notes.append("many components without a clean repeated dash pattern")
    elif component_count <= max(3, instance_count * 2 if instance_count else 3):
        pattern_type = "solid"
        notes.append("one/few long components relative to estimated instances")
    else:
        pattern_type = "unknown"
        notes.append("mask geometry does not clearly distinguish solid from dashed")
    expected_segment_count = component_count if pattern_type in {"dashed", "mixed"} else max(instance_count, component_count)
    return _json_safe({
        "pattern_type": pattern_type,
        "instance_count_estimate": int(instance_count),
        "component_count": int(component_count),
        "expected_segment_count": int(expected_segment_count),
        "dash_geometry": dash_stats,
        "notes": notes,
    })


def compute_i2_local_contrast(image_rgb: np.ndarray, gt_mask: np.ndarray, road_mask: Optional[np.ndarray] = None, outer_kernel: int = 21, inner_kernel: int = 5) -> Optional[float]:
    mask = safe_binary_mask(gt_mask)
    if mask.size == 0 or _mask_area(mask) == 0:
        return None
    gray = _to_gray_float(image_rgb)
    mask = _resize_mask_if_needed(mask, gray.shape)
    marking = mask > 0
    ring = compute_local_background_ring(mask, outer_kernel, inner_kernel, road_mask=road_mask)
    if not np.any(ring):
        fallback = np.zeros_like(marking, dtype=bool)
        fallback[int(mask.shape[0] * 0.40):, :] = True
        fallback &= ~marking
        if road_mask is not None:
            fallback &= _resize_mask_if_needed(road_mask, mask.shape).astype(bool)
        ring = fallback
    if not np.any(ring) or not np.any(marking):
        return None
    return float(np.clip(abs(float(np.mean(gray[marking])) - float(np.mean(gray[ring]))) / 255.0, 0.0, 1.0))


def compute_i3_boundary_sharpness(image_rgb: np.ndarray, gt_mask: np.ndarray, road_mask: Optional[np.ndarray] = None) -> Optional[float]:
    mask = safe_binary_mask(gt_mask)
    if mask.size == 0 or _mask_area(mask) == 0:
        return None
    gray = _to_gray_float(image_rgb)
    mask = _resize_mask_if_needed(mask, gray.shape)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    if not np.any(boundary):
        return None
    boundary_band = cv2.dilate(boundary.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    ring = compute_local_background_ring(mask, outer_kernel=21, inner_kernel=7, road_mask=road_mask)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    boundary_vals = grad[boundary_band]
    if boundary_vals.size == 0:
        return None
    boundary_med = float(np.percentile(boundary_vals, 75))
    background_med = float(np.percentile(grad[ring], 75)) if np.any(ring) else float(np.percentile(grad, 50))
    return float(np.clip((boundary_med - background_med) / (boundary_med + background_med + EPS), 0.0, 1.0))


def _component_quality_scores(image_rgb: np.ndarray, gt_mask: np.ndarray, road_mask: Optional[np.ndarray] = None) -> list[float]:
    mask = safe_binary_mask(gt_mask)
    scores = []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = int(max(5, min(80, _mask_area(mask) * 0.002)))
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == idx).astype(np.uint8)
        contrast = compute_i2_local_contrast(image_rgb, comp, road_mask=road_mask, outer_kernel=17, inner_kernel=3)
        sharpness = compute_i3_boundary_sharpness(image_rgb, comp, road_mask=road_mask)
        thickness = _thickness_stats(comp)["mean_thickness_px"]
        contrast_score = 0.0 if contrast is None else float(np.clip(contrast / 0.25, 0.0, 1.0))
        sharpness_score = 0.0 if sharpness is None else float(np.clip(sharpness / 0.35, 0.0, 1.0))
        if thickness is None:
            thickness_score = 0.0
        else:
            thickness_score = float(min(np.clip(thickness / 2.0, 0.0, 1.0), np.clip(45.0 / max(thickness, 1.0), 0.0, 1.0)))
        area_score = float(np.clip(area / 20.0, 0.0, 1.0))
        scores.append(float(np.clip(0.42 * contrast_score + 0.35 * sharpness_score + 0.18 * thickness_score + 0.05 * area_score, 0.0, 1.0)))
    return scores


def _select_i1_guidance(
    image_shape: tuple[int, int],
    mask: Any,
    *,
    native_lanes: Optional[Iterable[Any]] = None,
    lane_json: Optional[dict] = None,
    config: Optional[LaneContinuityConfig] = None,
    source_prefix: str,
) -> tuple[list[Any], Optional[str]]:
    """Select the highest-priority credible geometry representation."""

    cfg = config or LaneContinuityConfig()
    native_candidates = [] if native_lanes is None else list(native_lanes)
    candidates = [
        (native_candidates, f"native_{source_prefix}_polyline"),
        (lanes_from_lane_json(lane_json), f"{source_prefix}_lane_json"),
    ]
    for candidate, source in candidates:
        if candidate and prepare_guidance_lanes(candidate, image_shape, config=cfg):
            return candidate, source
    mask_lanes = extract_guidance_lanes_from_mask(mask, config=cfg)
    if mask_lanes and prepare_guidance_lanes(mask_lanes, image_shape, config=cfg):
        return mask_lanes, f"{source_prefix}_mask_grouped_centerline"
    return [], None


def compute_i1_pattern_continuity_details(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    lanes: Optional[List[np.ndarray]] = None,
    meta: Optional[dict] = None,
    road_mask: Optional[np.ndarray] = None,
    object_mask: Optional[np.ndarray] = None,
    homography: Optional[np.ndarray] = None,
    config: Optional[LaneContinuityConfig] = None,
    debug: bool = False,
) -> dict:
    """Return detailed canonical I1 using GT only as spatial guidance."""

    image_shape = tuple(np.asarray(image_rgb).shape[:2])
    lane_json = (meta or {}).get("lane_json") if isinstance(meta, dict) else None
    guidance, detail_source = _select_i1_guidance(
        image_shape,
        gt_mask,
        native_lanes=lanes,
        lane_json=lane_json,
        config=config,
        source_prefix="gt",
    )
    result = analyze_lane_pattern_continuity(
        image_rgb,
        guidance,
        geometry_source="ground_truth",
        road_mask=road_mask,
        object_mask=object_mask,
        trusted_pattern_hints=meta,
        homography=homography,
        config=config,
        debug=debug,
    )
    result["geometry_detail_source"] = detail_source
    result["pattern_type"] = result.get("intended_pattern", "unknown")
    if detail_source is None:
        result["unavailable_reason"] = "no_credible_ground_truth_guidance_geometry"
    return _json_safe(result)


def compute_i1_pattern_continuity(image_rgb: np.ndarray, gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None, road_mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Backward-compatible scalar wrapper for canonical GT-guided I1."""

    details = compute_i1_pattern_continuity_details(
        image_rgb,
        gt_mask,
        lanes=lanes,
        meta=meta,
        road_mask=road_mask,
    )
    return _finite_or_none(details.get("continuity_score"))


def compute_i4_legacy_thickness_stability(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None) -> Optional[float]:
    """Legacy paint-stripe thickness consistency; this is not lane width."""

    _ = lanes, meta
    return _thickness_stats(gt_mask)["score"]


def compute_i4_thickness_stability(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None) -> Optional[float]:
    """Deprecated compatibility alias for the explicitly named legacy metric."""

    return compute_i4_legacy_thickness_stability(gt_mask, lanes=lanes, meta=meta)


def _polyline_curvature_complexity(lanes: List[np.ndarray], height: int, width: int) -> Optional[float]:
    scores = []
    for lane in _valid_lanes(lanes):
        pts = lane[np.isfinite(lane).all(axis=1)]
        if pts.shape[0] < 4 or np.ptp(pts[:, 1]) < 10:
            continue
        try:
            a, b, _ = np.polyfit(pts[:, 1], pts[:, 0], deg=2)
        except np.linalg.LinAlgError:
            continue
        proxy = abs(float(a)) * (height ** 2) / max(width, 1)
        slope_change = abs(float(2 * a * np.ptp(pts[:, 1]))) / max(abs(float(b)), 1.0)
        scores.append(float(np.clip(0.75 * proxy + 0.25 * slope_change, 0.0, 1.0)))
    return safe_mean(scores)


def _mask_curvature_complexity(mask: np.ndarray) -> Optional[float]:
    binary = safe_binary_mask(mask)
    if binary.size == 0 or _mask_area(binary) == 0:
        return None
    h, w = binary.shape
    scores = []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    min_area = int(max(10, min(100, _mask_area(binary) * 0.002)))
    for idx in range(1, n_labels):
        if int(stats[idx, cv2.CC_STAT_AREA]) < min_area:
            continue
        ys, xs = np.where(labels == idx)
        if ys.size < 20 or np.ptp(ys) < 10:
            continue
        bins = np.linspace(float(ys.min()), float(ys.max()), num=min(24, max(6, int(np.ptp(ys) // 8))))
        cy, cx = [], []
        for y0, y1 in zip(bins[:-1], bins[1:]):
            sel = (ys >= y0) & (ys < y1)
            if np.count_nonzero(sel) >= 3:
                cy.append(float((y0 + y1) * 0.5))
                cx.append(float(np.mean(xs[sel])))
        if len(cy) < 4:
            continue
        try:
            a, b, _ = np.polyfit(np.asarray(cy), np.asarray(cx), deg=2)
        except np.linalg.LinAlgError:
            continue
        proxy = abs(float(a)) * (h ** 2) / max(w, 1)
        slope_change = abs(float(2 * a * (max(cy) - min(cy)))) / max(abs(float(b)), 1.0)
        scores.append(float(np.clip(0.75 * proxy + 0.25 * slope_change, 0.0, 1.0)))
    if scores:
        return safe_mean(scores)
    centers = [float(np.mean(np.where(binary[y] > 0)[0])) for y in range(binary.shape[0]) if np.any(binary[y] > 0)]
    return None if len(centers) < 3 else float(np.clip(np.std(centers) / max(binary.shape[1], 1), 0.0, 1.0))


def compute_i5_legacy_quadratic_proxy(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None) -> Optional[float]:
    """Legacy raw x(y) quadratic-fit curvature proxy; this is not physical curvature.

    Preserved unchanged (not deprecated in behavior) so historical values under
    ``I5_legacy_quadratic_proxy`` remain reproducible. It is a normalized
    image-space heuristic, not a robust differential-geometry curvature
    estimate, and it is superseded by
    :func:`evaluation.lane_geometry_complexity.analyze_lane_geometry_complexity`
    for the canonical/operational ``I5`` fields.
    """

    mask = safe_binary_mask(gt_mask)
    if mask.size == 0:
        return None
    if lanes:
        val = _polyline_curvature_complexity(lanes, mask.shape[0], mask.shape[1])
        if val is not None:
            return val
    return _mask_curvature_complexity(mask)


def compute_i5_geometry_complexity(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None) -> Optional[float]:
    """Deprecated compatibility alias for the explicitly named legacy metric."""

    return compute_i5_legacy_quadratic_proxy(gt_mask, lanes=lanes)


def compute_i5_geometry_complexity_details(
    gt_mask: np.ndarray,
    lanes: Optional[List[np.ndarray]] = None,
    meta: Optional[dict] = None,
    image_shape: Optional[tuple[int, int]] = None,
    road_mask: Optional[np.ndarray] = None,
    object_mask: Optional[np.ndarray] = None,
    homography: Optional[np.ndarray] = None,
    camera_model: Optional[dict] = None,
    calibration_meta: Optional[dict] = None,
    config: Optional[GeometryComplexityConfig] = None,
    debug: bool = False,
) -> dict:
    """Detailed canonical GT-guided I5, selecting geometry like I1/I4 do."""

    mask = safe_binary_mask(gt_mask)
    shape = tuple(image_shape) if image_shape is not None else mask.shape
    lane_json = (meta or {}).get("lane_json") if isinstance(meta, dict) else None
    guidance, detail_source = _select_i1_guidance(
        shape, mask, native_lanes=lanes, lane_json=lane_json, source_prefix="gt"
    )
    result = analyze_lane_geometry_complexity(
        guidance,
        image_shape=shape,
        geometry_source="ground_truth",
        road_mask=road_mask,
        object_mask=object_mask,
        homography=homography,
        camera_model=camera_model,
        calibration_meta=calibration_meta,
        config=config,
        debug=debug,
    )
    result["geometry_detail_source"] = detail_source
    if detail_source is None:
        result["unavailable_reason"] = "no_credible_ground_truth_guidance_geometry"
    return _json_safe(result)


def compute_i6_marking_instances(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None) -> dict:
    _ = meta
    raw_count = _component_count(gt_mask)
    instance_count = _estimate_marking_instances(gt_mask, lanes)
    topology = None if raw_count == 0 else float(np.clip(raw_count / max(instance_count, 1), 1.0, 10.0) / 10.0)
    return _json_safe({"I6_marking_instance_count": int(instance_count), "I6_raw_component_count": int(raw_count), "I6_topology_complexity": topology})


# D1-D7 (Detection Performance Indicators) are computed exclusively by
# evaluation.d_metrics.build_d_record (draft Table-16-aligned); see its use in
# build_metric_record above. The former per-image D1-D8 reimplementations here
# have been removed in favor of that single engine.


def compute_r1_marking_readability_score(metric_dict: dict) -> Optional[float]:
    legacy_thickness = metric_dict.get(
        "I4_legacy_thickness_stability",
        metric_dict.get("I4_thickness_stability"),
    )
    if legacy_thickness is None and "I4_lane_width_stability" not in metric_dict:
        # Historical v2/v3 records stored the thickness proxy only as I4.
        legacy_thickness = metric_dict.get("I4")
    values = {
        "I1_pattern_continuity": metric_dict.get("I1_pattern_continuity", metric_dict.get("I1")),
        "I2_local_contrast": metric_dict.get("I2_local_contrast", metric_dict.get("I2")),
        "I3_boundary_sharpness": metric_dict.get("I3_boundary_sharpness", metric_dict.get("I3")),
        "I4_legacy_thickness_stability": legacy_thickness,
    }
    score = weighted_mean_ignore_none(values, {"I1_pattern_continuity": 0.35, "I2_local_contrast": 0.35, "I3_boundary_sharpness": 0.20, "I4_legacy_thickness_stability": 0.10})
    return None if score is None else float(score * 100.0)


def compute_r2_reference_detectability_score(metric_dict: dict) -> Optional[float]:
    d6 = _finite_or_none(metric_dict.get("D6_image_missed_ratio", metric_dict.get("D6")))
    values = {
        "D3_f1": metric_dict.get("D3_f1"),
        "D3_iou": metric_dict.get("D3_iou", metric_dict.get("D3")),
        "D4_near_iou": metric_dict.get("D4_near_iou", metric_dict.get("D4")),
        "D6_detected_ratio": None if d6 is None else float(np.clip(1.0 - d6, 0.0, 1.0)),
        "D7_confidence_mean": metric_dict.get("D7_confidence_mean", metric_dict.get("D7")),
    }
    score = weighted_mean_ignore_none(values, {"D3_f1": 0.35, "D3_iou": 0.20, "D4_near_iou": 0.20, "D6_detected_ratio": 0.20, "D7_confidence_mean": 0.05})
    return None if score is None else float(score * 100.0)


def compute_r4_machine_readability_class(r1: Optional[float], r2: Optional[float], r3: Optional[dict] = None) -> dict:
    if r1 is None and r2 is None:
        status = "LOW_MACHINE_READABILITY"
        reason = "R1 and R2 could not be computed from available image evidence"
    elif (r1 is not None and r1 >= 80.0) and (r2 is not None and r2 >= 80.0):
        status = "HIGH_MACHINE_READABILITY"
        reason = f"Marking readability ({r1:.1f}) and reference detectability ({r2:.1f}) are both high"
    elif (r1 is not None and r1 >= 60.0) or (r2 is not None and r2 >= 60.0):
        status = "MODERATE_MACHINE_READABILITY"
        reason = f"At least one score is moderate or better (R1={r1}, R2={r2})"
    else:
        status = "LOW_MACHINE_READABILITY"
        reason = f"Both available scores are below moderate thresholds (R1={r1}, R2={r2})"
    return _json_safe({"status": status, "reason": reason, "R1": r1, "R2": r2, "bottleneck": r3, "warning": "Image-based proxy only; not a field-certified road-readiness standard."})


def _lane_width_context_from_details(width_details: dict) -> Optional[dict]:
    """Derive an I5 ``lane_width_context`` from the same source's own I4 result.

    Only used for uncalibrated measurement spaces, and always built from the
    matching geometry source (canonical I4 for canonical I5, prediction I4 for
    I5_pred) so I5_pred never sees ground-truth lane width.
    """

    pairs = width_details.get("pair_diagnostics") or []
    widths = [w for w in (_finite_or_none(pair.get("median_width")) for pair in pairs) if w is not None]
    median = safe_mean(widths)
    return None if median is None else {"median_width": median}


def _merged_metadata(sample: Any) -> tuple[dict, dict, dict]:
    sample_meta = dict(getattr(sample, "meta", {}) or {})
    target = getattr(sample, "target", None)
    target_meta = dict(getattr(target, "meta", {}) or {})
    return sample_meta, target_meta, {**target_meta, **sample_meta}


def build_metric_record(
    sample: Any,
    pred_mask: np.ndarray,
    image_rgb: np.ndarray,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    model_name: str = "yolopx",
    lane_prob: Optional[np.ndarray] = None,
    drivable_mask: Optional[np.ndarray] = None,
    object_mask: Optional[np.ndarray] = None,
    pred_lanes: Optional[List[np.ndarray]] = None,
    pred_lane_json: Optional[dict] = None,
    pred_geometry_source: Optional[str] = None,
    i1_config: Optional[LaneContinuityConfig] = None,
    homography: Optional[np.ndarray] = None,
    i1_debug: bool = False,
    camera_model: Optional[dict] = None,
    calibration_meta: Optional[dict] = None,
    i4_config: Optional[LaneWidthConfig] = None,
    i4_debug: bool = False,
    i5_config: Optional[GeometryComplexityConfig] = None,
    i5_debug: bool = False,
) -> dict:
    """Build one metric record with source-separated canonical/operational I1/I4/I5."""

    target = getattr(sample, "target", None)
    gt_mask = safe_binary_mask(getattr(target, "mask", None))
    image_h, image_w = image_rgb.shape[:2]
    if gt_mask.size == 0:
        gt_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    elif gt_mask.shape != (image_h, image_w):
        gt_mask = cv2.resize(gt_mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    pred = _resize_mask_if_needed(pred_mask, gt_mask.shape)
    prob = None if lane_prob is None else np.asarray(lane_prob, dtype=np.float32)
    if prob is not None and prob.shape != gt_mask.shape:
        prob = cv2.resize(prob, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    lanes = getattr(target, "lanes", None)
    sample_meta, target_meta, metadata = _merged_metadata(sample)
    shared_homography = homography if homography is not None else sample_meta.get(
        "image_to_road_homography",
        sample_meta.get("homography"),
    )
    shared_camera_model = camera_model if camera_model is not None else sample_meta.get("camera_model")
    shared_calibration_meta = (
        calibration_meta
        if calibration_meta is not None
        else sample_meta.get("calibration_meta", sample_meta.get("calibration"))
    )
    thick = _thickness_stats(gt_mask)
    i6 = compute_i6_marking_instances(gt_mask, lanes=lanes, meta=metadata)
    canonical_meta = dict(target_meta)

    canonical_width_guidance, canonical_width_detail_source = _select_i1_guidance(
        (image_h, image_w),
        gt_mask,
        native_lanes=lanes,
        lane_json=canonical_meta.get("lane_json"),
        config=i1_config,
        source_prefix="gt",
    )

    canonical_wear_details = analyze_lane_wear(image_rgb, canonical_width_guidance)
    i1 = _finite_or_none(canonical_wear_details.get("I1"))

    canonical_width_simple_details = analyze_lane_width_simple(
        canonical_width_guidance, image_shape=(image_h, image_w)
    )
    i4 = _finite_or_none(canonical_width_simple_details.get("I4"))

    # analyze_lane_width_stability (the earlier "heavy" I4 engine) stays wired in
    # here purely as context: I5 needs its perspective-normalized width profile to
    # scale curvature, even though its own score is no longer reported as I4 --
    # evaluation/lane_width_simple.py is the single reported I4 engine now.
    canonical_width_details = analyze_lane_width_stability(
        canonical_width_guidance,
        image_shape=(image_h, image_w),
        geometry_source="ground_truth",
        image_rgb=image_rgb,
        road_mask=None,
        object_mask=None,
        homography=shared_homography,
        camera_model=shared_camera_model,
        calibration_meta=shared_calibration_meta,
        config=i4_config,
        debug=i4_debug,
    )
    canonical_width_details["geometry_detail_source"] = canonical_width_detail_source

    prediction_guidance, prediction_detail_source = _select_i1_guidance(
        (image_h, image_w),
        pred,
        native_lanes=pred_lanes,
        lane_json=pred_lane_json,
        config=i1_config,
        source_prefix="pred",
    )
    width_source_defaults = {
        "native_pred_polyline": "native_polyline",
        "pred_lane_json": "lane_json",
        "pred_mask_grouped_centerline": "mask_derived",
    }
    prediction_width_source = pred_geometry_source or width_source_defaults.get(
        prediction_detail_source,
        prediction_detail_source or "unavailable",
    )
    # Same context-only use as canonical_width_details above, this time feeding
    # I5_pred. Operational I1_pred/I4_pred (scored against predicted, not GT,
    # geometry) are not part of the draft spec and are no longer computed.
    prediction_width_details = analyze_lane_width_stability(
        prediction_guidance,
        image_shape=(image_h, image_w),
        geometry_source=prediction_width_source,
        image_rgb=image_rgb,
        road_mask=drivable_mask,
        object_mask=object_mask,
        homography=shared_homography,
        camera_model=shared_camera_model,
        calibration_meta=shared_calibration_meta,
        config=i4_config,
        debug=i4_debug,
    )
    prediction_width_details["geometry_detail_source"] = prediction_detail_source
    if prediction_detail_source is None:
        prediction_width_details["unavailable_reason"] = "no_credible_prediction_boundary_geometry"

    i2 = compute_i2_local_contrast(image_rgb, gt_mask, road_mask=drivable_mask)
    i3 = compute_i3_boundary_sharpness(image_rgb, gt_mask, road_mask=drivable_mask)
    legacy_i5 = compute_i5_legacy_quadratic_proxy(gt_mask, lanes=lanes)

    canonical_geometry_details = analyze_lane_geometry_complexity(
        canonical_width_guidance,
        image_shape=(image_h, image_w),
        geometry_source="ground_truth",
        # Canonical I5 never consumes processor-derived road/object masks.
        road_mask=None,
        object_mask=None,
        homography=shared_homography,
        camera_model=shared_camera_model,
        calibration_meta=shared_calibration_meta,
        lane_width_context=_lane_width_context_from_details(canonical_width_details),
        config=i5_config,
        debug=i5_debug,
    )
    canonical_geometry_details["geometry_detail_source"] = canonical_width_detail_source
    i5 = _finite_or_none(canonical_geometry_details.get("alignment_complexity"))

    prediction_geometry_details = analyze_lane_geometry_complexity(
        prediction_guidance,
        image_shape=(image_h, image_w),
        geometry_source=prediction_width_source,
        road_mask=drivable_mask,
        object_mask=object_mask,
        homography=shared_homography,
        camera_model=shared_camera_model,
        calibration_meta=shared_calibration_meta,
        # I5_pred must never see GT-derived lane width.
        lane_width_context=_lane_width_context_from_details(prediction_width_details),
        config=i5_config,
        debug=i5_debug,
    )
    prediction_geometry_details["geometry_detail_source"] = prediction_detail_source
    if prediction_detail_source is None:
        prediction_geometry_details["unavailable_reason"] = "no_credible_prediction_boundary_geometry"
    i5_pred = _finite_or_none(prediction_geometry_details.get("alignment_complexity"))

    from evaluation.d_metrics import build_d_record  # local import: d_metrics imports mask utils back from here

    visibility_tag = ((sample_meta.get("tags") or {}).get("summary") or {}).get("Observed Marking Visibility")
    d_record = build_d_record(
        sample_id=str(getattr(sample, "image_id", "")),
        pred_mask=pred,
        gt_mask=gt_mask,
        gt_lane_json=canonical_meta.get("lane_json"),
        pred_lanes=pred_lanes,
        lane_prob=prob,
        visibility_tag=visibility_tag,
        image_path=str(getattr(sample, "image_path", "")),
    )
    record = {
        "metric_version": METRIC_VERSION, "dataset": dataset, "split": split, "model_name": model_name,
        "image_id": getattr(sample, "image_id", None), "image_path": str(getattr(sample, "image_path", "")),
        "width": int(getattr(sample, "width", image_w) or image_w), "height": int(getattr(sample, "height", image_h) or image_h),
        "sample_meta": sample_meta, "target_meta": target_meta, "metadata": metadata,
        "gt_nonzero": int(np.count_nonzero(gt_mask > 0)), "pred_nonzero": int(np.count_nonzero(pred > 0)),
        "I1": i1,
        "I1_pattern_continuity": i1,
        "I1_percent": None if i1 is None else float(i1 * 100.0),
        "I1_condition": canonical_wear_details.get("condition"),
        "I1_worn_fraction": canonical_wear_details.get("worn_fraction"),
        "I1_faded_fraction": canonical_wear_details.get("faded_fraction"),
        "I1_geometry_source": "ground_truth",
        "I1_geometry_detail_source": canonical_width_detail_source,
        "I1_valid_lane_count": canonical_wear_details.get("valid_lane_count"),
        "I1_lane_count": canonical_wear_details.get("lane_count"),
        "I1_lane_diagnostics": canonical_wear_details.get("lanes", []),
        "I1_unavailable_reason": canonical_wear_details.get("unavailable_reason"),
        "I2": i2, "I2_local_contrast": i2,
        "I3": i3, "I3_boundary_sharpness": i3,
        "I4": i4,
        "I4_lane_width_stability": i4,
        "I4_geometry_source": "ground_truth",
        "I4_geometry_detail_source": canonical_width_detail_source,
        "I4_pair_count": len(canonical_width_simple_details.get("pairs") or []),
        "I4_pair_diagnostics": canonical_width_simple_details.get("pairs", []),
        "I4_unavailable_reason": canonical_width_simple_details.get("unavailable_reason"),
        "I4_legacy_thickness_stability": thick["score"],
        "I4_legacy_mean_thickness_px": thick["mean_thickness_px"],
        "I4_legacy_thickness_cv": thick["thickness_cv"],
        "I4_legacy_num_segments": thick["num_segments"],
        "I5": i5,
        "I5_geometry_complexity": i5,
        "I5_alignment_complexity": canonical_geometry_details.get("alignment_complexity"),
        "I5_topology_complexity": canonical_geometry_details.get("topology_complexity"),
        "I5_profile_class": canonical_geometry_details.get("profile_class"),
        "I5_complexity_confidence": canonical_geometry_details.get("complexity_confidence"),
        "I5_geometry_source": "ground_truth",
        "I5_geometry_detail_source": canonical_width_detail_source,
        "I5_measurement_space": canonical_geometry_details.get("measurement_space"),
        "I5_metric_scale_available": canonical_geometry_details.get("metric_scale_available"),
        "I5_bendiness": canonical_geometry_details.get("bendiness"),
        "I5_mean_abs_curvature": canonical_geometry_details.get("mean_abs_curvature"),
        "I5_p95_abs_curvature": canonical_geometry_details.get("p95_abs_curvature"),
        "I5_curvature_variation": canonical_geometry_details.get("curvature_variation"),
        "I5_p95_curvature_gradient": canonical_geometry_details.get("p95_curvature_gradient"),
        "I5_reversal_count": canonical_geometry_details.get("reversal_count"),
        "I5_reversal_density": canonical_geometry_details.get("reversal_density"),
        "I5_total_abs_heading_change": canonical_geometry_details.get("total_abs_heading_change"),
        "I5_tortuosity": canonical_geometry_details.get("tortuosity"),
        "I5_primitive_count": canonical_geometry_details.get("primitive_count"),
        "I5_valid_lane_count": canonical_geometry_details.get("valid_lane_count"),
        "I5_unknown_lane_count": canonical_geometry_details.get("unknown_lane_count"),
        "I5_lane_diagnostics": canonical_geometry_details.get("lane_diagnostics", []),
        "I5_transform_diagnostics": canonical_geometry_details.get("transform_diagnostics"),
        "I5_aggregation_method": canonical_geometry_details.get("aggregation_method"),
        "I5_unavailable_reason": canonical_geometry_details.get("unavailable_reason"),
        "I5_pred": i5_pred,
        "I5_pred_geometry_complexity": i5_pred,
        "I5_pred_alignment_complexity": prediction_geometry_details.get("alignment_complexity"),
        "I5_pred_topology_complexity": prediction_geometry_details.get("topology_complexity"),
        "I5_pred_profile_class": prediction_geometry_details.get("profile_class"),
        "I5_pred_complexity_confidence": prediction_geometry_details.get("complexity_confidence"),
        "I5_pred_geometry_source": prediction_geometry_details.get("geometry_source"),
        "I5_pred_geometry_detail_source": prediction_detail_source,
        "I5_pred_measurement_space": prediction_geometry_details.get("measurement_space"),
        "I5_pred_metric_scale_available": prediction_geometry_details.get("metric_scale_available"),
        "I5_pred_lane_diagnostics": prediction_geometry_details.get("lane_diagnostics", []),
        "I5_pred_unavailable_reason": prediction_geometry_details.get("unavailable_reason"),
        "I5_legacy_quadratic_proxy": legacy_i5,
        "I6": i6["I6_marking_instance_count"], **i6,
        "D1": d_record["D1_detection"], "D1_detection": d_record["D1_detection"],
        "D2": d_record["D2_count_match"], "D2_count_match": d_record["D2_count_match"],
        "D2_gt_lane_count": d_record["D2_gt_lane_count"], "D2_pred_lane_count": d_record["D2_pred_lane_count"],
        "D2_pred_count_source": d_record["D2_pred_count_source"],
        "D3": d_record["D3_iou"], "D3_iou": d_record["D3_iou"], "D3_f1": d_record["D3_f1"],
        "D3_precision": d_record["D3_precision"], "D3_recall": d_record["D3_recall"],
        "D3_tp": d_record["D3_tp"], "D3_fp": d_record["D3_fp"], "D3_fn": d_record["D3_fn"],
        "D4": d_record["D4_near_iou"], "D4_near_iou": d_record["D4_near_iou"],
        "D5_visibility_tag": d_record["visibility_tag"], "D5_visibility_group": d_record["D5_visibility_group"],
        "D6": d_record["D6_image_missed_ratio"], "D6_image_missed_ratio": d_record["D6_image_missed_ratio"],
        "D6_gt_pixels": d_record["D6_gt_pixels"], "D6_fn_pixels": d_record["D6_fn_pixels"],
        "D7": d_record["D7_confidence_mean"], "D7_confidence_mean": d_record["D7_confidence_mean"],
        "D_stroke_standardized": d_record.get("stroke_standardized"),
        "D_pred_stroke_px": d_record.get("pred_stroke_px"), "D_gt_stroke_px": d_record.get("gt_stroke_px"),
        "D_gt_eligible": d_record.get("gt_eligible"),
    }
    record["R1"] = compute_r1_marking_readability_score(record)
    record["R1_marking_readability_score"] = record["R1"]
    record["R2"] = compute_r2_reference_detectability_score(record)
    record["R2_reference_detectability_score"] = record["R2"]
    record["unavailable_reasons"] = {}
    if d_record["D7_confidence_mean"] is None:
        record["unavailable_reasons"]["D7_confidence_mean"] = "lane_probability_unavailable"
    if record["gt_nonzero"] == 0:
        record["unavailable_reasons"]["gt_metrics"] = "empty_ground_truth_mask"
    if i1 is None:
        record["unavailable_reasons"]["I1_pattern_continuity"] = canonical_wear_details.get(
            "unavailable_reason"
        ) or "insufficient_canonical_i1_evidence"
    if i4 is None:
        record["unavailable_reasons"]["I4_lane_width_stability"] = canonical_width_simple_details.get(
            "unavailable_reason"
        ) or "insufficient_canonical_lane_width_evidence"
    if i5 is None:
        record["unavailable_reasons"]["I5_alignment_complexity"] = canonical_geometry_details.get(
            "unavailable_reason"
        ) or "insufficient_canonical_geometry_evidence"
    if i5_pred is None:
        record["unavailable_reasons"]["I5_pred_alignment_complexity"] = prediction_geometry_details.get(
            "unavailable_reason"
        ) or "insufficient_prediction_geometry_evidence"
    return _json_safe(record)


def _paired_values(records: list[dict], x_key: str, y_key: str, transform_y=None) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for record in records:
        x = _finite_or_none(record.get(x_key))
        y = _finite_or_none(record.get(y_key))
        if x is None or y is None:
            continue
        if transform_y is not None:
            y = transform_y(y)
        if y is None or not np.isfinite(y):
            continue
        xs.append(x)
        ys.append(float(y))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _association_detail(records: list[dict], x_key: str, y_key: str, min_samples: int, transform_y=None) -> dict:
    x, y = _paired_values(records, x_key, y_key, transform_y=transform_y)
    detail = {"x": x_key, "y": y_key, "n": int(len(x)), "pearson": None, "spearman": None, "reason": None}
    if len(x) < min_samples:
        detail["reason"] = f"insufficient_samples_n_lt_{min_samples}"
        return detail
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        detail["reason"] = "zero_variance"
        return detail
    try:
        pearson, pearson_p = pearsonr(x, y)
        spearman, spearman_p = spearmanr(x, y)
    except Exception as exc:
        detail["reason"] = f"correlation_failed_{exc.__class__.__name__}"
        return detail
    detail.update({"pearson": _finite_or_none(pearson), "pearson_p": _finite_or_none(pearson_p), "spearman": _finite_or_none(spearman), "spearman_p": _finite_or_none(spearman_p)})
    return detail


def _bootstrap_mean_difference_ci(
    low_values: np.ndarray, high_values: np.ndarray, n_boot: int = 1000, seed: int = 13
) -> tuple[float, float]:
    """Percentile bootstrap CI for a difference of independent group means."""

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        low_sample = rng.choice(low_values, size=low_values.size, replace=True)
        high_sample = rng.choice(high_values, size=high_values.size, replace=True)
        diffs[i] = float(np.mean(low_sample) - np.mean(high_sample))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _tertile_effect_detail(records: list[dict], x_key: str, y_key: str, min_samples: int) -> dict:
    """Bottom-vs-top tertile performance difference with a bootstrap CI."""

    x, y = _paired_values(records, x_key, y_key)
    detail = {"x": x_key, "y": y_key, "n": int(len(x)), "effect": None, "reason": None}
    if len(x) < min_samples:
        detail["reason"] = f"insufficient_samples_n_lt_{min_samples}"
        return detail
    if float(np.std(x)) <= EPS:
        detail["reason"] = "zero_variance"
        return detail
    order = np.argsort(x)
    y_sorted = y[order]
    third = max(1, len(x) // 3)
    bottom, top = y_sorted[:third], y_sorted[-third:]
    if bottom.size == 0 or top.size == 0:
        detail["reason"] = "empty_tertile_group"
        return detail
    ci_low, ci_high = _bootstrap_mean_difference_ci(bottom, top)
    detail.update(
        {
            "n_bottom": int(bottom.size),
            "n_top": int(top.size),
            "bottom_tertile_mean": float(np.mean(bottom)),
            "top_tertile_mean": float(np.mean(top)),
            "effect": float(np.mean(bottom) - np.mean(top)),
            "bootstrap_ci_95": [ci_low, ci_high],
        }
    )
    return detail


def _stratified_associations(records: list[dict], group_key: str, x_key: str, y_key: str, min_samples: int) -> dict:
    """Per-stratum Spearman association, grouping by a record-level field."""

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        value = record.get(group_key)
        if value is not None:
            groups[str(value)].append(record)
    return {value: _association_detail(items, x_key, y_key, min_samples) for value, items in groups.items()}


def _c4_adjusted_regression(records: list[dict], min_samples: int) -> dict:
    """Optional OLS with dataset/model fixed effects; association only, never causal.

    Restricted to the single measurement space with the most samples so
    calibrated-metric and image-proxy I5 values are never combined in one
    regression without explicit stratification.
    """

    min_n = max(min_samples * 3, 30)
    rows = []
    for record in records:
        y = _finite_or_none(record.get("D3_f1"))
        x5 = _finite_or_none(record.get("I5_alignment_complexity"))
        if y is None or x5 is None:
            continue
        rows.append(
            {
                "y": y,
                "I5_alignment_complexity": x5,
                "I1_pattern_continuity": _finite_or_none(record.get("I1_pattern_continuity")) or 0.0,
                "I2_local_contrast": _finite_or_none(record.get("I2_local_contrast")) or 0.0,
                "I3_boundary_sharpness": _finite_or_none(record.get("I3_boundary_sharpness")) or 0.0,
                "dataset": str(record.get("dataset")),
                "model_name": str(record.get("model_name")),
                "measurement_space": str(record.get("I5_measurement_space")),
            }
        )
    if len(rows) < min_n:
        return {"available": False, "n": len(rows), "reason": f"insufficient_samples_n_lt_{min_n}"}
    spaces = [row["measurement_space"] for row in rows]
    majority_space = max(set(spaces), key=spaces.count)
    filtered = [row for row in rows if row["measurement_space"] == majority_space]
    excluded = len(rows) - len(filtered)
    if len(filtered) < min_n:
        return {
            "available": False,
            "n": len(filtered),
            "reason": "insufficient_same_measurement_space_samples",
            "measurement_space": majority_space,
            "excluded_other_measurement_space": excluded,
        }
    datasets = sorted({row["dataset"] for row in filtered})
    models = sorted({row["model_name"] for row in filtered})
    names = ["intercept", "I5_alignment_complexity", "I1_pattern_continuity", "I2_local_contrast", "I3_boundary_sharpness"]
    columns = [
        np.ones(len(filtered)),
        np.asarray([row["I5_alignment_complexity"] for row in filtered]),
        np.asarray([row["I1_pattern_continuity"] for row in filtered]),
        np.asarray([row["I2_local_contrast"] for row in filtered]),
        np.asarray([row["I3_boundary_sharpness"] for row in filtered]),
    ]
    for dataset_name in datasets[1:]:
        columns.append(np.asarray([1.0 if row["dataset"] == dataset_name else 0.0 for row in filtered]))
        names.append(f"dataset__{dataset_name}")
    for model_value in models[1:]:
        columns.append(np.asarray([1.0 if row["model_name"] == model_value else 0.0 for row in filtered]))
        names.append(f"model__{model_value}")
    design = np.column_stack(columns)
    if design.shape[0] <= design.shape[1]:
        return {"available": False, "n": len(filtered), "reason": "insufficient_degrees_of_freedom_for_fixed_effects"}
    y = np.asarray([row["y"] for row in filtered], dtype=np.float64)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = None if ss_tot <= EPS else float(1.0 - ss_res / ss_tot)
    return _json_safe(
        {
            "available": True,
            "n": len(filtered),
            "measurement_space": majority_space,
            "excluded_other_measurement_space": excluded,
            "predictors": names,
            "coefficients": {name: float(value) for name, value in zip(names, coefficients)},
            "r_squared": r_squared,
            "note": "Association only, adjusted for dataset/model fixed effects and I1-I3; not a causal effect estimate.",
        }
    )


def compute_c4_geometry_sensitivity_detail(records: list[dict], min_samples: int = 10) -> dict:
    """Canonical-I5-only geometry-sensitivity analysis for C4.

    Uses ``I5_alignment_complexity``/``I5_geometry_complexity`` (ground-truth
    geometry) exclusively; ``I5_pred`` never enters C4, so a model's own
    prediction geometry is never used as the predictor of that same model's
    detection metrics.
    """

    x_key = "I5_alignment_complexity"
    primary = _association_detail(records, x_key, "D3_f1", min_samples)
    iou = _association_detail(records, x_key, "D3_iou", min_samples)
    detected_ratio = _association_detail(
        records, x_key, "D6_image_missed_ratio", min_samples, transform_y=lambda v: 1.0 - v
    )
    tertile = _tertile_effect_detail(records, x_key, "D3_f1", min_samples)

    x_legacy, y_legacy = _paired_values(records, "I5_geometry_complexity", "D3_f1")
    legacy_median_split = {"n": int(len(x_legacy)), "detection_metric": "D3_f1", "effect": None, "reason": None}
    if len(x_legacy) < min_samples:
        legacy_median_split["reason"] = f"insufficient_samples_n_lt_{min_samples}"
    elif float(np.std(x_legacy)) <= EPS:
        legacy_median_split["reason"] = "zero_variance"
    else:
        med = float(np.median(x_legacy))
        low = y_legacy[x_legacy <= med]
        high = y_legacy[x_legacy > med]
        if len(low) and len(high):
            legacy_median_split.update(
                {
                    "n_low": int(len(low)),
                    "n_high": int(len(high)),
                    "low_complexity_mean": float(np.mean(low)),
                    "high_complexity_mean": float(np.mean(high)),
                    "effect": float(np.mean(low) - np.mean(high)),
                }
            )
        else:
            legacy_median_split["reason"] = "empty_low_or_high_complexity_group"

    per_component = {
        "bendiness": _association_detail(records, "I5_bendiness", "D3_f1", min_samples),
        "p95_curvature": _association_detail(records, "I5_p95_abs_curvature", "D3_f1", min_samples),
        "curvature_variation": _association_detail(records, "I5_curvature_variation", "D3_f1", min_samples),
        "reversal_density": _association_detail(records, "I5_reversal_density", "D3_f1", min_samples),
        "topology_complexity": _association_detail(records, "I5_topology_complexity", "D3_f1", min_samples),
    }
    strata = {
        "by_model": _stratified_associations(records, "model_name", x_key, "D3_f1", min_samples),
        "by_dataset": _stratified_associations(records, "dataset", x_key, "D3_f1", min_samples),
        "by_measurement_space": _stratified_associations(
            records, "I5_measurement_space", x_key, "D3_f1", min_samples
        ),
    }
    total = len(records)
    with_geometry = int(sum(1 for r in records if _finite_or_none(r.get(x_key)) is not None))
    return _json_safe(
        {
            "n_total_records": total,
            "n_with_canonical_geometry": with_geometry,
            "primary": primary,
            "iou": iou,
            "detected_ratio": detected_ratio,
            "tertile_effect": tertile,
            "legacy_median_split": legacy_median_split,
            "per_component": per_component,
            "strata": strata,
            "adjusted_regression": _c4_adjusted_regression(records, min_samples),
            "effect": legacy_median_split.get("effect"),
            "reason": legacy_median_split.get("reason"),
            "caveat": (
                "Associations only; do not interpret as a causal effect of geometry "
                "complexity on detector performance. Uncalibrated (image-proxy) I5 "
                "values are only comparable within a homogeneous dataset/camera group."
            ),
        }
    )


def compute_condition_degradation(records: list[dict], condition_key: str, baseline_value: Optional[str] = None, min_samples: int = 10) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        metadata = record.get("metadata") or {}
        value = record.get(condition_key, metadata.get(condition_key))
        if value is not None:
            groups[str(value)].append(record)
    groups = {k: v for k, v in groups.items() if len(v) >= min_samples}
    if len(groups) < 2:
        return []
    if baseline_value is None:
        baseline_value = next((v for v in ("clear", "daytime", "day", "normal", "low") if v in groups), None)
    if baseline_value is None or baseline_value not in groups:
        baseline_value = sorted(groups, key=lambda k: (-len(groups[k]), k))[0]
    def scores(items):
        r1 = safe_mean(r.get("R1_marking_readability_score") for r in items)
        r2 = safe_mean(r.get("R2_reference_detectability_score") for r in items)
        return (None if r1 is None else r1 / 100.0, None if r2 is None else r2 / 100.0)
    base_r, base_d = scores(groups[baseline_value])
    ranking = []
    for value, items in groups.items():
        r, d = scores(items)
        read_drop = 0.0 if base_r is None or r is None else base_r - r
        det_drop = 0.0 if base_d is None or d is None else base_d - d
        ranking.append({"condition_key": condition_key, "condition": value, "baseline": baseline_value, "n": len(items), "readability_drop": float(read_drop), "detection_drop": float(det_drop), "degradation": float(read_drop + det_drop)})
    return sorted(ranking, key=lambda item: item["degradation"], reverse=True)


def _choose_condition_key(records: list[dict]) -> Optional[str]:
    for key in ("condition", "weather", "timeofday", "scene"):
        values = {str((r.get("metadata") or {}).get(key)) for r in records if (r.get("metadata") or {}).get(key) is not None}
        if len(values) >= 2:
            return key
    return None


def compute_correlations_from_records(records: list[dict], min_samples: int = 10) -> dict:
    c1 = _association_detail(records, "I1_pattern_continuity", "D3_f1", min_samples)
    c1_miss = _association_detail(records, "I1_pattern_continuity", "D6_image_missed_ratio", min_samples, transform_y=lambda v: 1.0 - v)
    c2_iou = _association_detail(records, "I2_local_contrast", "D3_iou", min_samples)
    c2_f1 = _association_detail(records, "I2_local_contrast", "D3_f1", min_samples)
    c3_f1 = _association_detail(records, "I3_boundary_sharpness", "D3_f1", min_samples)
    c3_iou = _association_detail(records, "I3_boundary_sharpness", "D3_iou", min_samples)
    c4_detail = compute_c4_geometry_sensitivity_detail(records, min_samples=min_samples)
    condition_key = _choose_condition_key(records)
    c5 = compute_condition_degradation(records, condition_key, min_samples=min_samples) if condition_key else []
    return _json_safe({
        "C1": c1.get("spearman"), "C1_pattern_continuity_to_detectability": c1.get("spearman"), "C1_detail": {"primary": c1, "missed_marking_detected_ratio": c1_miss},
        "C2": c2_iou.get("spearman"), "C2_contrast_to_iou": c2_iou.get("spearman"), "C2_detail": {"D3_iou": c2_iou, "D3_f1": c2_f1},
        "C3": c3_f1.get("spearman"), "C3_sharpness_to_detectability": c3_f1.get("spearman"), "C3_detail": {"D3_f1": c3_f1, "D3_iou": c3_iou},
        "C4": c4_detail.get("effect"), "C4_geometry_sensitivity": c4_detail.get("effect"), "C4_detail": c4_detail,
        "C5": c5, "C5_condition_degradation_ranking": c5, "C5_detail": {"condition_key": condition_key, "ranking": c5, "reason": None if c5 else "insufficient_condition_metadata_or_samples"},
    })


def compute_r3_bottleneck(summary_or_correlations: dict) -> dict:
    correlations = summary_or_correlations.get("layer3", summary_or_correlations)
    layer1 = summary_or_correlations.get("layer1_avg", {})
    layer2 = summary_or_correlations.get("layer2", {})
    candidates = []
    def add(name, strength, evidence):
        val = _finite_or_none(strength)
        if val is not None:
            candidates.append({"name": name, "strength": float(abs(val)), "evidence": evidence})
    add("low pattern continuity", correlations.get("C1_pattern_continuity_to_detectability"), "C1 Spearman")
    add("low local contrast", correlations.get("C2_contrast_to_iou"), "C2 Spearman")
    add("low boundary sharpness", correlations.get("C3_sharpness_to_detectability"), "C3 Spearman")
    add("geometry sensitivity", correlations.get("C4_geometry_sensitivity"), "C4 effect")
    c5 = correlations.get("C5_condition_degradation_ranking") or correlations.get("C5") or []
    if c5 and isinstance(c5[0], dict):
        add(f"worst condition stratum: {c5[0].get('condition')}", c5[0].get("degradation"), "C5 degradation")
    add("low pattern continuity", 1.0 - (layer1.get("I1_pattern_continuity") or layer1.get("I1") or 1.0), "descriptive worst-score fallback")
    add("low local contrast", 1.0 - (layer1.get("I2_local_contrast") or layer1.get("I2") or 1.0), "descriptive worst-score fallback")
    add("low boundary sharpness", 1.0 - (layer1.get("I3_boundary_sharpness") or layer1.get("I3") or 1.0), "descriptive worst-score fallback")
    add("high missed marking ratio", layer2.get("D6_image_missed_ratio") or layer2.get("D6"), "descriptive worst-score fallback")
    add("occlusion/visibility robustness gap", layer2.get("D5_visibility_degraded_gap"), "descriptive worst-score fallback")
    if not candidates:
        return {"name": "unavailable", "strength": None, "evidence": "no computable correlation or descriptive metric", "reason": "insufficient_samples_or_metric_variance"}
    candidates.sort(key=lambda item: item["strength"], reverse=True)
    result = candidates[0]
    if result["evidence"] == "descriptive worst-score fallback":
        result["reason"] = "correlations unavailable or weak; using worst aggregate diagnostic"
    return _json_safe(result)


def summarize_records(records: list[dict], min_corr_samples: int = 10) -> dict:
    records = [_json_safe(r) for r in records]
    scalar_keys = [
        "I1", "I1_pattern_continuity", "I1_worn_fraction", "I1_faded_fraction",
        "I2", "I2_local_contrast", "I3", "I3_boundary_sharpness",
        "I4", "I4_lane_width_stability", "I4_pair_count",
        "I4_legacy_thickness_stability", "I4_legacy_mean_thickness_px", "I4_legacy_thickness_cv", "I4_legacy_num_segments",
        "I5", "I5_geometry_complexity", "I5_alignment_complexity", "I5_topology_complexity",
        "I5_complexity_confidence", "I5_bendiness", "I5_mean_abs_curvature", "I5_p95_abs_curvature",
        "I5_curvature_variation", "I5_p95_curvature_gradient", "I5_reversal_count", "I5_reversal_density",
        "I5_total_abs_heading_change", "I5_tortuosity", "I5_primitive_count",
        "I5_valid_lane_count", "I5_unknown_lane_count", "I5_legacy_quadratic_proxy",
        "I5_pred", "I5_pred_geometry_complexity", "I5_pred_alignment_complexity", "I5_pred_topology_complexity",
        "I5_pred_complexity_confidence",
        "I6", "I6_marking_instance_count", "I6_raw_component_count", "I6_topology_complexity",
        "D1", "D1_detection", "D2", "D2_count_match", "D3", "D3_iou", "D3_f1", "D3_precision", "D3_recall",
        "D4", "D4_near_iou", "D6", "D6_image_missed_ratio", "D7", "D7_confidence_mean",
        "R1", "R1_marking_readability_score", "R2", "R2_reference_detectability_score",
    ]
    means = {key: safe_mean(r.get(key) for r in records) for key in scalar_keys}

    # D6 (Detection Gap Ratio) must be pixel-weighted across the group -- sum(FN)
    # / sum(GT) -- not a mean of per-image ratios, per draft Table 16.
    eligible = [r for r in records if r.get("D_gt_eligible")]
    gt_px = sum(r.get("D6_gt_pixels") or 0 for r in eligible)
    fn_px = sum(r.get("D6_fn_pixels") or 0 for r in eligible)
    d6_weighted = float(fn_px / gt_px) if gt_px else None
    means["D6"] = d6_weighted
    means["D6_image_missed_ratio"] = d6_weighted

    # D5 (Occlusion/Visibility Robustness Gap) is inherently a group-level
    # comparison (clear vs. occluded/degraded visibility groups), reusing
    # evaluation.d_metrics's own grouping logic on these same per-image records.
    from evaluation.d_metrics import _gap as _d5_group_gap

    d5_occlusion_gap = _d5_group_gap(eligible, "clear", {"occluded"})
    d5_visibility_degraded_gap = _d5_group_gap(eligible, "clear", {"occluded", "degraded"})

    layer1_keys = [k for k in scalar_keys if k.startswith("I")]
    layer2_keys = [k for k in scalar_keys if k.startswith("D")]
    layer1 = {key: means.get(key) for key in layer1_keys}
    layer2 = {key: means.get(key) for key in layer2_keys}
    layer2.update({
        "D5_occlusion_robustness_gap": d5_occlusion_gap,
        "D5_visibility_degraded_gap": d5_visibility_degraded_gap,
    })
    r1 = compute_r1_marking_readability_score(layer1)
    r2 = compute_r2_reference_detectability_score(layer2)
    layer3 = compute_correlations_from_records(records, min_samples=min_corr_samples)
    r3 = compute_r3_bottleneck({"layer1_avg": layer1, "layer2": layer2, "layer3": layer3})
    r4 = compute_r4_machine_readability_class(r1, r2, r3)
    confidence_available = bool(any(r.get("D7_confidence_mean") is not None for r in records))
    return _json_safe({
        "metric_version": METRIC_VERSION, "num_records": len(records),
        "gt_nonzero_images": int(sum(1 for r in records if (r.get("gt_nonzero") or 0) > 0)),
        "pred_nonzero_images": int(sum(1 for r in records if (r.get("pred_nonzero") or 0) > 0)),
        "confidence_available": confidence_available, "d7_available": confidence_available,
        "metrics_mean": means, "layer1_avg": layer1, "layer2": layer2, "layer3": layer3,
        "layer4": {"R1": r1, "R1_marking_readability_score": r1, "R2": r2, "R2_reference_detectability_score": r2, "R3": r3, "R3_bottleneck": r3, "R4": r4, "R4_machine_readability_class": r4},
    })


# Legacy wrappers.
def compute_i1_continuity(gt_mask: np.ndarray) -> Optional[float]:
    """Return unavailable because mask-only input has no physical-paint evidence."""

    _ = gt_mask
    return None


def compute_i2_contrast(image: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    return compute_i2_local_contrast(image, gt_mask)


def compute_i3_sharpness(image: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    return compute_i3_boundary_sharpness(image, gt_mask)


def compute_i4_width_stability(
    gt_mask: np.ndarray,
    lanes: Optional[List[np.ndarray]] = None,
    homography: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Scalar canonical lane-width profile stability from GT geometry only."""

    mask = safe_binary_mask(gt_mask)
    guidance = list(lanes or []) or extract_guidance_lanes_from_mask(mask)
    return _finite_or_none(
        analyze_lane_width_stability(
            guidance,
            image_shape=mask.shape,
            geometry_source="ground_truth",
            homography=homography,
        ).get("lane_width_stability")
    )


def compute_i5_alignment_complexity(
    gt_mask: np.ndarray,
    lanes: Optional[List[np.ndarray]] = None,
    homography: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Scalar canonical alignment-complexity from GT geometry only (new engine)."""

    mask = safe_binary_mask(gt_mask)
    guidance = list(lanes or []) or extract_guidance_lanes_from_mask(mask)
    return _finite_or_none(
        analyze_lane_geometry_complexity(
            guidance,
            image_shape=mask.shape,
            geometry_source="ground_truth",
            homography=homography,
        ).get("alignment_complexity")
    )


def compute_i5_curvature(gt_mask: np.ndarray) -> Optional[float]:
    """Legacy scalar wrapper; returns the unchanged quadratic-proxy value."""

    return compute_i5_geometry_complexity(gt_mask)


def compute_i6_lane_count(gt_mask: np.ndarray) -> int:
    return int(compute_i6_marking_instances(gt_mask)["I6_marking_instance_count"])


# Group-level D1-D8 convenience wrappers removed: D-metrics are computed
# exclusively via evaluation.d_metrics.build_d_record now (see build_metric_record).


def compute_c1_continuity_correlation(infra_list: List[dict], detection_list: List[dict]) -> Optional[float]:
    return compute_correlations_from_records([{**i, **d} for i, d in zip(infra_list, detection_list)], min_samples=2)["C1"]


def compute_c2_contrast_correlation(infra_list: List[dict], detection_list: List[dict]) -> Optional[float]:
    return compute_correlations_from_records([{**i, **d} for i, d in zip(infra_list, detection_list)], min_samples=2)["C2"]


def compute_c3_sharpness_correlation(infra_list: List[dict], detection_list: List[dict]) -> Optional[float]:
    return compute_correlations_from_records([{**i, **d} for i, d in zip(infra_list, detection_list)], min_samples=2)["C3"]


def compute_c4_curvature_impact(infra_list: List[dict], detection_list: List[dict]) -> Optional[float]:
    return compute_correlations_from_records([{**i, **d} for i, d in zip(infra_list, detection_list)], min_samples=4)["C4"]


def compute_c5_degradation_ranking(infra_by_condition: dict, detection_by_condition: dict) -> list[Tuple[str, float]]:
    if "clear" not in infra_by_condition or "clear" not in detection_by_condition:
        return []
    base_r = compute_r1_marking_readability_score(infra_by_condition["clear"])
    base_d = compute_r2_reference_detectability_score(detection_by_condition["clear"])
    rankings = []
    for condition, infra in infra_by_condition.items():
        if condition == "clear" or condition not in detection_by_condition:
            continue
        r = compute_r1_marking_readability_score(infra)
        d = compute_r2_reference_detectability_score(detection_by_condition[condition])
        degradation = 0.0
        if base_r is not None and r is not None:
            degradation += (base_r - r) / 100.0
        if base_d is not None and d is not None:
            degradation += (base_d - d) / 100.0
        rankings.append((condition, float(degradation)))
    return sorted(rankings, key=lambda item: item[1], reverse=True)


def compute_r1_infrastructure_score(infra: dict, weights: Optional[dict] = None) -> Optional[float]:
    _ = weights
    return compute_r1_marking_readability_score(infra)


def compute_r2_detection_score(detection: dict, weights: Optional[dict] = None) -> Optional[float]:
    _ = weights
    return compute_r2_reference_detectability_score(detection)


def compute_r4_classification(r1: float, r2: float, r3: Any) -> dict:
    return compute_r4_machine_readability_class(r1, r2, r3 if isinstance(r3, dict) else {"legacy": r3})


def _sample_meta_value(sample: Any, key: str) -> Any:
    target = getattr(sample, "target", None)
    return (getattr(sample, "meta", {}) or {}).get(key) or (getattr(target, "meta", {}) or {}).get(key)


def filter_by_weather(samples, weather: str):
    return [s for s in samples if _sample_meta_value(s, "weather") == weather]


def filter_by_context(samples, context: str):
    def infer(sample) -> Optional[str]:
        scene = _sample_meta_value(sample, "scene") or ""
        condition = _sample_meta_value(sample, "condition")
        if condition in {"curve", "cross", "crowded", "no_marking", "shadow", "dazzle"}:
            return str(condition)
        target = getattr(sample, "target", None)
        lane_count = compute_i6_lane_count(getattr(target, "mask", None)) if target is not None else 0
        if scene == "highway" and lane_count <= 2:
            return "rural_highway"
        if scene in {"urban", "city street"} and 2 <= lane_count <= 3:
            return "urban_street"
        if scene in {"urban", "city street"} and lane_count > 3:
            return "urban_intersection"
        return condition
    return [s for s in samples if infer(s) == context]


def filter_by_time(samples, time: str):
    return [s for s in samples if _sample_meta_value(s, "timeofday") == time]


def filter_by_traffic(samples, traffic: str):
    def infer(sample) -> str:
        direct = _sample_meta_value(sample, "traffic_proxy")
        if direct is not None:
            return str(direct)
        img_path = getattr(sample, "image_path", None)
        img = cv2.imread(str(img_path)) if img_path else None
        if img is None:
            return "unknown"
        density = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2] < 100))
        if density > 0.30:
            return "high"
        if density > 0.15:
            return "medium"
        return "low"
    return [s for s in samples if infer(s) == traffic]

