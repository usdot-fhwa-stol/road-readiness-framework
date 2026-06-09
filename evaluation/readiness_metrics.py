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

EPS = 1e-6
METRIC_VERSION = "readiness_metrics_v2_pattern_aware"


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


def compute_i1_pattern_continuity(image_rgb: np.ndarray, gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None, road_mask: Optional[np.ndarray] = None) -> Optional[float]:
    mask = safe_binary_mask(gt_mask)
    if mask.size == 0 or _mask_area(mask) == 0:
        return None
    info = infer_marking_pattern(mask, lanes=lanes, meta=meta)
    quality = safe_mean(_component_quality_scores(image_rgb, mask, road_mask=road_mask))
    quality = 0.0 if quality is None else quality
    if info["pattern_type"] == "dashed":
        regularity = _finite_or_none((info.get("dash_geometry") or {}).get("regularity_score"))
        if regularity is None:
            regularity = 0.55 if info["component_count"] >= 2 else 0.0
        score = 0.58 * quality + 0.42 * regularity
    elif info["pattern_type"] == "solid":
        h, w = mask.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(max(3, min(9, w // 160))), _odd(max(31, min(101, h // 8)), 31)))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        area_ratio = float(_mask_area(mask) / (_mask_area(closed) + EPS))
        instance_count = max(int(info.get("instance_count_estimate") or 1), 1)
        component_count = max(int(info.get("component_count") or 1), 1)
        continuity = float(np.clip(0.70 * area_ratio + 0.30 * np.clip(instance_count / component_count, 0.0, 1.0), 0.0, 1.0))
        score = 0.50 * quality + 0.50 * continuity
    else:
        regularity = _finite_or_none((info.get("dash_geometry") or {}).get("regularity_score"))
        if regularity is None:
            regularity = float(np.clip(info["instance_count_estimate"] / max(info["component_count"], 1), 0.0, 1.0))
        score = 0.60 * quality + 0.40 * regularity
    return float(np.clip(score, 0.0, 1.0))


def compute_i4_thickness_stability(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None) -> Optional[float]:
    _ = lanes, meta
    return _thickness_stats(gt_mask)["score"]


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


def compute_i5_geometry_complexity(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None) -> Optional[float]:
    mask = safe_binary_mask(gt_mask)
    if mask.size == 0:
        return None
    if lanes:
        val = _polyline_curvature_complexity(lanes, mask.shape[0], mask.shape[1])
        if val is not None:
            return val
    return _mask_curvature_complexity(mask)


def compute_i6_marking_instances(gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None) -> dict:
    _ = meta
    raw_count = _component_count(gt_mask)
    instance_count = _estimate_marking_instances(gt_mask, lanes)
    topology = None if raw_count == 0 else float(np.clip(raw_count / max(instance_count, 1), 1.0, 10.0) / 10.0)
    return _json_safe({"I6_marking_instance_count": int(instance_count), "I6_raw_component_count": int(raw_count), "I6_topology_complexity": topology})


def compute_tolerant_precision_recall_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 5) -> dict:
    pred = safe_binary_mask(pred_mask)
    gt = _resize_mask_if_needed(gt_mask, pred.shape) if pred.size else safe_binary_mask(gt_mask)
    if pred.size == 0 and gt.size != 0:
        pred = np.zeros_like(gt)
    if gt.size == 0 and pred.size != 0:
        gt = np.zeros_like(pred)
    pred_bool = pred > 0
    gt_bool = gt > 0
    pred_area = int(np.count_nonzero(pred_bool))
    gt_area = int(np.count_nonzero(gt_bool))
    if pred_area == 0 and gt_area == 0:
        return {"precision": None, "recall": None, "f1": None, "pred_area": 0, "gt_area": 0, "reason": "no_predicted_or_gt_marking_pixels"}
    if pred_area == 0:
        return {"precision": None, "recall": 0.0 if gt_area > 0 else None, "f1": 0.0 if gt_area > 0 else None, "pred_area": pred_area, "gt_area": gt_area, "reason": "empty_prediction"}
    if gt_area == 0:
        return {"precision": 0.0, "recall": None, "f1": 0.0, "pred_area": pred_area, "gt_area": gt_area, "reason": "empty_ground_truth"}
    dist_to_pred = cv2.distanceTransform((~pred_bool).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_gt = cv2.distanceTransform((~gt_bool).astype(np.uint8), cv2.DIST_L2, 3)
    recall = float(np.mean(dist_to_pred[gt_bool] <= tolerance_px))
    precision = float(np.mean(dist_to_gt[pred_bool] <= tolerance_px))
    f1 = float(2.0 * precision * recall / (precision + recall + EPS))
    return {"precision": precision, "recall": recall, "f1": f1, "pred_area": pred_area, "gt_area": gt_area, "reason": None}


def compute_d1_valid_detection_single(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 5, min_recall: float = 0.30, min_precision: float = 0.30, min_pred_area: int = 20) -> int:
    t = compute_tolerant_precision_recall_f1(pred_mask, gt_mask, tolerance_px=tolerance_px)
    precision = t.get("precision")
    recall = t.get("recall")
    pred_area = int(t.get("pred_area") or 0)
    if precision is None or recall is None:
        return 0
    return int(pred_area >= min_pred_area and recall >= min_recall and precision >= min_precision)


def compute_d2_instance_agreement_single(pred_mask: np.ndarray, gt_mask: np.ndarray, gt_lanes: Optional[List[np.ndarray]] = None, pred_lanes: Optional[List[np.ndarray]] = None) -> Optional[float]:
    gt_count = len(_valid_lanes(gt_lanes)) if gt_lanes else _estimate_marking_instances(gt_mask)
    pred_count = len(_valid_lanes(pred_lanes)) if pred_lanes else _estimate_marking_instances(pred_mask)
    if gt_count == 0 and pred_count == 0:
        return None
    if gt_count == 0:
        return 0.0
    return float(1.0 - min(abs(pred_count - gt_count) / max(gt_count, 1), 1.0))


def compute_d3_iou_single(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    pred = safe_binary_mask(pred_mask)
    gt = _resize_mask_if_needed(gt_mask, pred.shape) if pred.size else safe_binary_mask(gt_mask)
    if pred.size == 0 and gt.size != 0:
        pred = np.zeros_like(gt)
    if gt.size == 0 and pred.size != 0:
        gt = np.zeros_like(pred)
    if pred.size == 0 and gt.size == 0:
        return None
    union = int(np.count_nonzero((pred > 0) | (gt > 0)))
    if union == 0:
        return None
    tp = int(np.count_nonzero((pred > 0) & (gt > 0)))
    return float(tp / (union + EPS))


def compute_d3_tolerant_f1_single(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 5) -> Optional[float]:
    return _finite_or_none(compute_tolerant_precision_recall_f1(pred_mask, gt_mask, tolerance_px)["f1"])


def _band_slice(height: int, band: str) -> slice:
    if band == "far":
        return slice(0, height // 3)
    if band == "mid":
        return slice(height // 3, 2 * height // 3)
    if band == "near":
        return slice(2 * height // 3, height)
    raise ValueError(f"Unknown band: {band}")


def compute_d4_band_metrics_single(pred_mask: np.ndarray, gt_mask: np.ndarray, band: str, tolerance_px: int = 5) -> dict:
    pred = safe_binary_mask(pred_mask)
    gt = _resize_mask_if_needed(gt_mask, pred.shape) if pred.size else safe_binary_mask(gt_mask)
    if pred.size == 0 and gt.size != 0:
        pred = np.zeros_like(gt)
    if gt.size == 0 and pred.size != 0:
        gt = np.zeros_like(pred)
    if pred.size == 0:
        return {"iou": None, "tolerant_f1": None, "reason": "empty_masks"}
    ys = _band_slice(pred.shape[0], band)
    pred_band = pred[ys, :]
    gt_band = gt[ys, :]
    if _mask_area(gt_band) == 0:
        return {"iou": None, "tolerant_f1": None, "reason": "no_gt_markings_in_band"}
    return {"iou": compute_d3_iou_single(pred_band, gt_band), "tolerant_f1": compute_d3_tolerant_f1_single(pred_band, gt_band, tolerance_px), "reason": None}


def _region_iou(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> Optional[float]:
    if int(np.count_nonzero((gt > 0) & region)) == 0:
        return None
    union = int(np.count_nonzero(((pred > 0) | (gt > 0)) & region))
    if union == 0:
        return None
    tp = int(np.count_nonzero((pred > 0) & (gt > 0) & region))
    return float(tp / (union + EPS))


def compute_d5_dark_region_iou_gap_single(pred_mask: np.ndarray, gt_mask: np.ndarray, image_rgb: np.ndarray, min_gt_pixels: int = 20) -> Optional[float]:
    pred = safe_binary_mask(pred_mask)
    gt = _resize_mask_if_needed(gt_mask, pred.shape) if pred.size else safe_binary_mask(gt_mask)
    if pred.size == 0 and gt.size != 0:
        pred = np.zeros_like(gt)
    if gt.size == 0 or _mask_area(gt) == 0:
        return None
    gray = _to_gray_float(image_rgb)
    if gray.shape != gt.shape:
        gray = cv2.resize(gray, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    dark = gray < min(100.0, float(np.percentile(gray, 35)))
    normal = ~dark
    if np.count_nonzero((gt > 0) & dark) < min_gt_pixels or np.count_nonzero((gt > 0) & normal) < min_gt_pixels:
        return None
    iou_dark = _region_iou(pred, gt, dark)
    iou_normal = _region_iou(pred, gt, normal)
    if iou_dark is None or iou_normal is None:
        return None
    return float(iou_normal - iou_dark)


def compute_d6_missed_marking_ratio_single(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    pred = safe_binary_mask(pred_mask)
    gt = _resize_mask_if_needed(gt_mask, pred.shape) if pred.size else safe_binary_mask(gt_mask)
    if pred.size == 0 and gt.size != 0:
        pred = np.zeros_like(gt)
    gt_area = int(np.count_nonzero(gt > 0))
    if gt_area == 0:
        return None
    fn = int(np.count_nonzero((pred == 0) & (gt > 0)))
    return float(fn / (gt_area + EPS))


def compute_d8_confidence_mean_single(pred_mask: np.ndarray, lane_prob: Optional[np.ndarray]) -> Optional[float]:
    if lane_prob is None:
        return None
    prob = np.asarray(lane_prob, dtype=np.float32)
    if prob.size == 0:
        return None
    pred = _resize_mask_if_needed(pred_mask, prob.shape)
    prob = np.clip(prob, 0.0, 1.0)
    pred_pixels = pred > 0
    if not np.any(pred_pixels):
        return 0.0
    return float(np.mean(prob[pred_pixels]))


def compute_r1_marking_readability_score(metric_dict: dict) -> Optional[float]:
    values = {
        "I1_pattern_continuity": metric_dict.get("I1_pattern_continuity", metric_dict.get("I1")),
        "I2_local_contrast": metric_dict.get("I2_local_contrast", metric_dict.get("I2")),
        "I3_boundary_sharpness": metric_dict.get("I3_boundary_sharpness", metric_dict.get("I3")),
        "I4_thickness_stability": metric_dict.get("I4_thickness_stability", metric_dict.get("I4")),
    }
    score = weighted_mean_ignore_none(values, {"I1_pattern_continuity": 0.35, "I2_local_contrast": 0.35, "I3_boundary_sharpness": 0.20, "I4_thickness_stability": 0.10})
    return None if score is None else float(score * 100.0)


def compute_r2_reference_detectability_score(metric_dict: dict) -> Optional[float]:
    near_key = "D4_near_f1_r5" if metric_dict.get("D4_near_f1_r5") is not None else "D4_near_iou"
    d6 = _finite_or_none(metric_dict.get("D6_missed_marking_ratio", metric_dict.get("D6")))
    values = {
        "D3_tolerant_f1_r5": metric_dict.get("D3_tolerant_f1_r5"),
        "D3_iou": metric_dict.get("D3_iou", metric_dict.get("D3")),
        near_key: metric_dict.get(near_key),
        "D6_detected_ratio": None if d6 is None else float(np.clip(1.0 - d6, 0.0, 1.0)),
        "D8_confidence_mean": metric_dict.get("D8_confidence_mean", metric_dict.get("D8")),
    }
    score = weighted_mean_ignore_none(values, {"D3_tolerant_f1_r5": 0.35, "D3_iou": 0.20, near_key: 0.20, "D6_detected_ratio": 0.20, "D8_confidence_mean": 0.05})
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


def _merged_metadata(sample: Any) -> tuple[dict, dict, dict]:
    sample_meta = dict(getattr(sample, "meta", {}) or {})
    target = getattr(sample, "target", None)
    target_meta = dict(getattr(target, "meta", {}) or {})
    return sample_meta, target_meta, {**target_meta, **sample_meta}


def build_metric_record(sample: Any, pred_mask: np.ndarray, image_rgb: np.ndarray, dataset: Optional[str] = None, split: Optional[str] = None, model_name: str = "yolopx", lane_prob: Optional[np.ndarray] = None, drivable_mask: Optional[np.ndarray] = None, object_mask: Optional[np.ndarray] = None) -> dict:
    _ = object_mask
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
    pattern_info = infer_marking_pattern(gt_mask, lanes=lanes, meta=metadata)
    thick = _thickness_stats(gt_mask)
    i6 = compute_i6_marking_instances(gt_mask, lanes=lanes, meta=metadata)
    tolerant = compute_tolerant_precision_recall_f1(pred, gt_mask, tolerance_px=5)
    d4 = {band: compute_d4_band_metrics_single(pred, gt_mask, band, tolerance_px=5) for band in ("near", "mid", "far")}
    i1 = compute_i1_pattern_continuity(image_rgb, gt_mask, lanes=lanes, meta=metadata, road_mask=drivable_mask)
    i2 = compute_i2_local_contrast(image_rgb, gt_mask, road_mask=drivable_mask)
    i3 = compute_i3_boundary_sharpness(image_rgb, gt_mask, road_mask=drivable_mask)
    i4 = thick["score"]
    i5 = compute_i5_geometry_complexity(gt_mask, lanes=lanes)
    d3 = compute_d3_iou_single(pred, gt_mask)
    d5 = compute_d5_dark_region_iou_gap_single(pred, gt_mask, image_rgb)
    d6 = compute_d6_missed_marking_ratio_single(pred, gt_mask)
    d8 = compute_d8_confidence_mean_single(pred, prob)
    record = {
        "metric_version": METRIC_VERSION, "dataset": dataset, "split": split, "model_name": model_name,
        "image_id": getattr(sample, "image_id", None), "image_path": str(getattr(sample, "image_path", "")),
        "width": int(getattr(sample, "width", image_w) or image_w), "height": int(getattr(sample, "height", image_h) or image_h),
        "sample_meta": sample_meta, "target_meta": target_meta, "metadata": metadata,
        "gt_nonzero": int(np.count_nonzero(gt_mask > 0)), "pred_nonzero": int(np.count_nonzero(pred > 0)),
        "I1": i1, "I1_pattern_continuity": i1, "I1_percent": None if i1 is None else float(i1 * 100.0), "I1_pattern_type": pattern_info.get("pattern_type"), "I1_pattern_info": pattern_info,
        "I2": i2, "I2_local_contrast": i2,
        "I3": i3, "I3_boundary_sharpness": i3,
        "I4": i4, "I4_thickness_stability": i4, "I4_mean_thickness_px": thick["mean_thickness_px"], "I4_thickness_cv": thick["thickness_cv"],
        "I5": i5, "I5_geometry_complexity": i5,
        "I6": i6["I6_marking_instance_count"], **i6,
        "D1": compute_d1_valid_detection_single(pred, gt_mask, tolerance_px=5), "D1_valid_detection": compute_d1_valid_detection_single(pred, gt_mask, tolerance_px=5),
        "D2": compute_d2_instance_agreement_single(pred, gt_mask, gt_lanes=lanes), "D2_instance_agreement": compute_d2_instance_agreement_single(pred, gt_mask, gt_lanes=lanes),
        "D3": d3, "D3_iou": d3, "D3_tolerant_precision_r5": tolerant.get("precision"), "D3_tolerant_recall_r5": tolerant.get("recall"), "D3_tolerant_f1_r5": _finite_or_none(tolerant.get("f1")),
        "D4_near_iou": d4["near"]["iou"], "D4_mid_iou": d4["mid"]["iou"], "D4_far_iou": d4["far"]["iou"],
        "D4_near_f1_r5": d4["near"]["tolerant_f1"], "D4_mid_f1_r5": d4["mid"]["tolerant_f1"], "D4_far_f1_r5": d4["far"]["tolerant_f1"],
        "D4_band_reasons": {band: d4[band]["reason"] for band in d4 if d4[band]["reason"]},
        "D5": d5, "D5_dark_region_iou_gap": d5, "D5_metric_type": "dark_region_sensitivity",
        "D6": d6, "D6_missed_marking_ratio": d6,
        "D7": None, "D7_temporal_jitter": None, "D7_reason": "unavailable_single_frame_evaluation",
        "D8": d8, "D8_confidence_mean": d8, "D8_available": d8 is not None,
    }
    record["R1"] = compute_r1_marking_readability_score(record)
    record["R1_marking_readability_score"] = record["R1"]
    record["R2"] = compute_r2_reference_detectability_score(record)
    record["R2_reference_detectability_score"] = record["R2"]
    record["unavailable_reasons"] = {"D7_temporal_jitter": "unavailable_single_frame_evaluation"}
    if d8 is None:
        record["unavailable_reasons"]["D8_confidence_mean"] = "lane_probability_unavailable"
    if record["gt_nonzero"] == 0:
        record["unavailable_reasons"]["gt_metrics"] = "empty_ground_truth_mask"
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
    c1 = _association_detail(records, "I1_pattern_continuity", "D3_tolerant_f1_r5", min_samples)
    c1_miss = _association_detail(records, "I1_pattern_continuity", "D6_missed_marking_ratio", min_samples, transform_y=lambda v: 1.0 - v)
    c2_iou = _association_detail(records, "I2_local_contrast", "D3_iou", min_samples)
    c2_f1 = _association_detail(records, "I2_local_contrast", "D3_tolerant_f1_r5", min_samples)
    c3_f1 = _association_detail(records, "I3_boundary_sharpness", "D3_tolerant_f1_r5", min_samples)
    c3_iou = _association_detail(records, "I3_boundary_sharpness", "D3_iou", min_samples)
    x, y = _paired_values(records, "I5_geometry_complexity", "D3_tolerant_f1_r5")
    c4_detail = {"n": int(len(x)), "detection_metric": "D3_tolerant_f1_r5", "effect": None, "reason": None}
    if len(x) < min_samples:
        c4_detail["reason"] = f"insufficient_samples_n_lt_{min_samples}"
    elif float(np.std(x)) <= EPS:
        c4_detail["reason"] = "zero_variance"
    else:
        med = float(np.median(x))
        low = y[x <= med]
        high = y[x > med]
        if len(low) and len(high):
            c4_detail.update({"n_low": int(len(low)), "n_high": int(len(high)), "low_complexity_mean": float(np.mean(low)), "high_complexity_mean": float(np.mean(high)), "effect": float(np.mean(low) - np.mean(high))})
        else:
            c4_detail["reason"] = "empty_low_or_high_complexity_group"
    condition_key = _choose_condition_key(records)
    c5 = compute_condition_degradation(records, condition_key, min_samples=min_samples) if condition_key else []
    return _json_safe({
        "C1": c1.get("spearman"), "C1_pattern_continuity_to_detectability": c1.get("spearman"), "C1_detail": {"primary": c1, "missed_marking_detected_ratio": c1_miss},
        "C2": c2_iou.get("spearman"), "C2_contrast_to_iou": c2_iou.get("spearman"), "C2_detail": {"D3_iou": c2_iou, "D3_tolerant_f1_r5": c2_f1},
        "C3": c3_f1.get("spearman"), "C3_sharpness_to_detectability": c3_f1.get("spearman"), "C3_detail": {"D3_tolerant_f1_r5": c3_f1, "D3_iou": c3_iou},
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
    add("high missed marking ratio", layer2.get("D6_missed_marking_ratio") or layer2.get("D6"), "descriptive worst-score fallback")
    add("dark-region degradation", layer2.get("D5_dark_region_iou_gap") or layer2.get("D5"), "descriptive worst-score fallback")
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
        "I1", "I1_pattern_continuity", "I2", "I2_local_contrast", "I3", "I3_boundary_sharpness", "I4", "I4_thickness_stability", "I4_mean_thickness_px", "I4_thickness_cv", "I5", "I5_geometry_complexity", "I6", "I6_marking_instance_count", "I6_raw_component_count", "I6_topology_complexity",
        "D1", "D1_valid_detection", "D2", "D2_instance_agreement", "D3", "D3_iou", "D3_tolerant_precision_r5", "D3_tolerant_recall_r5", "D3_tolerant_f1_r5", "D4_near_iou", "D4_mid_iou", "D4_far_iou", "D4_near_f1_r5", "D4_mid_f1_r5", "D4_far_f1_r5", "D5", "D5_dark_region_iou_gap", "D6", "D6_missed_marking_ratio", "D8", "D8_confidence_mean", "R1", "R1_marking_readability_score", "R2", "R2_reference_detectability_score",
    ]
    means = {key: safe_mean(r.get(key) for r in records) for key in scalar_keys}
    layer1_keys = [k for k in scalar_keys if k.startswith("I")]
    layer2_keys = [k for k in scalar_keys if k.startswith("D")]
    layer1 = {key: means.get(key) for key in layer1_keys}
    layer2 = {key: means.get(key) for key in layer2_keys}
    layer2.update({"D7": None, "D7_temporal_jitter": None, "D7_reason": "unavailable_single_frame_evaluation"})
    r1 = compute_r1_marking_readability_score(layer1)
    r2 = compute_r2_reference_detectability_score(layer2)
    layer3 = compute_correlations_from_records(records, min_samples=min_corr_samples)
    r3 = compute_r3_bottleneck({"layer1_avg": layer1, "layer2": layer2, "layer3": layer3})
    r4 = compute_r4_machine_readability_class(r1, r2, r3)
    return _json_safe({
        "metric_version": METRIC_VERSION, "num_records": len(records),
        "gt_nonzero_images": int(sum(1 for r in records if (r.get("gt_nonzero") or 0) > 0)),
        "pred_nonzero_images": int(sum(1 for r in records if (r.get("pred_nonzero") or 0) > 0)),
        "confidence_available": bool(any(r.get("D8_confidence_mean") is not None for r in records)), "d7_available": False,
        "metrics_mean": means, "layer1_avg": layer1, "layer2": layer2, "layer3": layer3,
        "layer4": {"R1": r1, "R1_marking_readability_score": r1, "R2": r2, "R2_reference_detectability_score": r2, "R3": r3, "R3_bottleneck": r3, "R4": r4, "R4_machine_readability_class": r4},
    })


# Legacy wrappers.
def _pseudo_image_from_mask(gt_mask: np.ndarray) -> np.ndarray:
    mask = safe_binary_mask(gt_mask)
    img = np.full((*mask.shape, 3), 50, dtype=np.uint8)
    img[mask > 0] = 220
    return img


def compute_i1_continuity(gt_mask: np.ndarray) -> Optional[float]:
    return compute_i1_pattern_continuity(_pseudo_image_from_mask(gt_mask), gt_mask)


def compute_i2_contrast(image: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    return compute_i2_local_contrast(image, gt_mask)


def compute_i3_sharpness(image: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    return compute_i3_boundary_sharpness(image, gt_mask)


def compute_i4_width_stability(gt_mask: np.ndarray) -> Optional[float]:
    return compute_i4_thickness_stability(gt_mask)


def compute_i5_curvature(gt_mask: np.ndarray) -> Optional[float]:
    return compute_i5_geometry_complexity(gt_mask)


def compute_i6_lane_count(gt_mask: np.ndarray) -> int:
    return int(compute_i6_marking_instances(gt_mask)["I6_marking_instance_count"])


def compute_d1_detection_rate(pred_masks: List[np.ndarray], gt_masks: Optional[List[np.ndarray]] = None) -> Optional[float]:
    if not pred_masks:
        return None
    if gt_masks is None:
        return safe_mean(1.0 if _mask_area(pm) > 0 else 0.0 for pm in pred_masks)
    return safe_mean(compute_d1_valid_detection_single(pm, gm) for pm, gm in zip(pred_masks, gt_masks))


def compute_d2_lane_count_accuracy(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Optional[float]:
    return safe_mean(compute_d2_instance_agreement_single(pm, gm) for pm, gm in zip(pred_masks, gt_masks))


def compute_d3_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Optional[float]:
    return safe_mean(compute_d3_iou_single(pm, gm) for pm, gm in zip(pred_masks, gt_masks))


def compute_d4_near_field_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Optional[float]:
    return safe_mean(compute_d4_band_metrics_single(pm, gm, "near")["iou"] for pm, gm in zip(pred_masks, gt_masks))


def compute_d5_occlusion_gap(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], images: List[np.ndarray]) -> Optional[float]:
    return safe_mean(compute_d5_dark_region_iou_gap_single(pm, gm, img) for pm, gm, img in zip(pred_masks, gt_masks, images))


def compute_d6_detection_gap(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Optional[float]:
    return safe_mean(compute_d6_missed_marking_ratio_single(pm, gm) for pm, gm in zip(pred_masks, gt_masks))


def compute_d8_confidence_mean(pred_masks: Optional[List[np.ndarray]] = None, lane_probs: Optional[List[np.ndarray]] = None) -> Optional[float]:
    if pred_masks is None or lane_probs is None:
        return None
    return safe_mean(compute_d8_confidence_mean_single(pm, prob) for pm, prob in zip(pred_masks, lane_probs))


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


class ReadinessMetrics:
    """V2 per-image lane-marking readability/detectability metrics."""

    def build_metric_record(self, *args, **kwargs) -> dict:
        return build_metric_record(*args, **kwargs)

    def summarize_records(self, records: list[dict], min_corr_samples: int = 10) -> dict:
        return summarize_records(records, min_corr_samples=min_corr_samples)

    def compute_infrastructure(self, image: np.ndarray, gt_mask: np.ndarray, lanes: Optional[List[np.ndarray]] = None, meta: Optional[dict] = None, road_mask: Optional[np.ndarray] = None) -> dict:
        i1 = compute_i1_pattern_continuity(image, gt_mask, lanes=lanes, meta=meta, road_mask=road_mask)
        i2 = compute_i2_local_contrast(image, gt_mask, road_mask=road_mask)
        i3 = compute_i3_boundary_sharpness(image, gt_mask, road_mask=road_mask)
        thick = _thickness_stats(gt_mask)
        i5 = compute_i5_geometry_complexity(gt_mask, lanes=lanes)
        i6 = compute_i6_marking_instances(gt_mask, lanes=lanes, meta=meta)
        return _json_safe({"I1": i1, "I1_pattern_continuity": i1, "I2": i2, "I2_local_contrast": i2, "I3": i3, "I3_boundary_sharpness": i3, "I4": thick["score"], "I4_thickness_stability": thick["score"], "I4_mean_thickness_px": thick["mean_thickness_px"], "I4_thickness_cv": thick["thickness_cv"], "I5": i5, "I5_geometry_complexity": i5, "I6": i6["I6_marking_instance_count"], **i6})

    def compute_detection(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], images: List[np.ndarray], lane_probs: Optional[List[Optional[np.ndarray]]] = None) -> dict:
        lane_probs = lane_probs or [None] * len(pred_masks)
        rows = []
        for pred, gt, img, prob in zip(pred_masks, gt_masks, images, lane_probs):
            tolerant = compute_tolerant_precision_recall_f1(pred, gt)
            d4 = {band: compute_d4_band_metrics_single(pred, gt, band) for band in ("near", "mid", "far")}
            d1 = compute_d1_valid_detection_single(pred, gt)
            d2 = compute_d2_instance_agreement_single(pred, gt)
            d3 = compute_d3_iou_single(pred, gt)
            d5 = compute_d5_dark_region_iou_gap_single(pred, gt, img)
            d6 = compute_d6_missed_marking_ratio_single(pred, gt)
            d8 = compute_d8_confidence_mean_single(pred, prob)
            rows.append({"D1": d1, "D1_valid_detection": d1, "D2": d2, "D2_instance_agreement": d2, "D3": d3, "D3_iou": d3, "D3_tolerant_precision_r5": tolerant.get("precision"), "D3_tolerant_recall_r5": tolerant.get("recall"), "D3_tolerant_f1_r5": tolerant.get("f1"), "D4_near_iou": d4["near"]["iou"], "D4_mid_iou": d4["mid"]["iou"], "D4_far_iou": d4["far"]["iou"], "D4_near_f1_r5": d4["near"]["tolerant_f1"], "D4_mid_f1_r5": d4["mid"]["tolerant_f1"], "D4_far_f1_r5": d4["far"]["tolerant_f1"], "D5": d5, "D5_dark_region_iou_gap": d5, "D6": d6, "D6_missed_marking_ratio": d6, "D7": None, "D7_temporal_jitter": None, "D8": d8, "D8_confidence_mean": d8})
        if not rows:
            return {}
        result = {key: safe_mean(row.get(key) for row in rows) for key in rows[0].keys()}
        result.update({"D7": None, "D7_temporal_jitter": None, "D7_reason": "unavailable_single_frame_evaluation"})
        return _json_safe(result)

    def compute_correlations(self, infra_list: List[dict], detection_list: List[dict], infra_by_condition: Optional[dict] = None, detection_by_condition: Optional[dict] = None) -> dict:
        result = compute_correlations_from_records([{**i, **d} for i, d in zip(infra_list, detection_list)], min_samples=10)
        if infra_by_condition or detection_by_condition:
            result["C5_legacy"] = compute_c5_degradation_ranking(infra_by_condition or {}, detection_by_condition or {})
        return result

    def compute_verdict(self, infra: dict, detection: dict, correlation: dict) -> dict:
        r1 = compute_r1_marking_readability_score(infra)
        r2 = compute_r2_reference_detectability_score(detection)
        r3 = compute_r3_bottleneck({"layer1_avg": infra, "layer2": detection, "layer3": correlation})
        return {"R1": r1, "R2": r2, "R3": r3, "R4": compute_r4_machine_readability_class(r1, r2, r3)}

    def evaluate_stratified(self, samples, pred_masks: List[np.ndarray], weather: Optional[str] = None, context: Optional[str] = None, time: Optional[str] = None, traffic: Optional[str] = None, min_samples: int = 10) -> Optional[dict]:
        pairs = list(zip(samples, pred_masks))
        if weather:
            allowed = {id(s) for s in filter_by_weather([x[0] for x in pairs], weather)}; pairs = [(s, p) for s, p in pairs if id(s) in allowed]
        if context:
            allowed = {id(s) for s in filter_by_context([x[0] for x in pairs], context)}; pairs = [(s, p) for s, p in pairs if id(s) in allowed]
        if time:
            allowed = {id(s) for s in filter_by_time([x[0] for x in pairs], time)}; pairs = [(s, p) for s, p in pairs if id(s) in allowed]
        if traffic:
            allowed = {id(s) for s in filter_by_traffic([x[0] for x in pairs], traffic)}; pairs = [(s, p) for s, p in pairs if id(s) in allowed]
        if len(pairs) < min_samples:
            return None
        records = []
        for sample, pred in pairs:
            img_bgr = cv2.imread(str(getattr(sample, "image_path", "")))
            if img_bgr is None:
                continue
            records.append(build_metric_record(sample, pred, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        if len(records) < min_samples:
            return None
        summary = summarize_records(records)
        summary["records"] = records
        return summary
