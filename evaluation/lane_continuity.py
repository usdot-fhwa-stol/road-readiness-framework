"""Classical RGB analysis of lane-marking pattern continuity.

Geometry inputs locate a marking corridor. They are never used as evidence that
paint is present. Visible-paint evidence is sampled from ``image_rgb`` and the
intended pattern is fitted separately from the condition of expected paint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


EPS = 1e-8
TRUSTED_PATTERN_FIELDS = {
    "style",
    "pattern",
    "lane_style",
    "line_style",
    "marking_type",
    "line_type",
}


@dataclass(frozen=True)
class LaneContinuityConfig:
    """Thresholds and sampling settings for lane-continuity analysis."""

    sample_spacing_px: float = 2.0
    max_longitudinal_samples: int = 700
    min_geometry_points: int = 2
    min_lane_support_px: float = 42.0
    min_valid_samples: int = 32
    min_valid_fraction: float = 0.45
    min_pattern_support_px: float = 52.0
    min_expected_weight: float = 5.0
    min_expected_samples: int = 8
    min_pattern_q: float = 0.22
    center_half_width_fraction: float = 0.005
    center_half_width_min_px: float = 1.5
    center_half_width_max_px: float = 7.0
    side_inner_factor: float = 1.8
    side_outer_factor: float = 4.5
    cross_section_samples: int = 31
    paint_delta_floor: float = 3.0
    paint_delta_scale: float = 62.0
    edge_delta_scale: float = 58.0
    white_luminance_normalizer: float = 220.0
    neutral_chroma_scale: float = 34.0
    yellow_chroma_floor: float = 2.0
    yellow_chroma_scale: float = 32.0
    yellow_hue_center: float = 27.0
    yellow_hue_scale: float = 20.0
    stripe_consistency_scale: float = 75.0
    stripe_consistency_floor: float = 0.35
    luminance_evidence_weight: float = 0.50
    paired_contrast_weight: float = 0.22
    edge_evidence_weight: float = 0.16
    color_evidence_weight: float = 0.12
    width_consistency_evidence_floor: float = 0.82
    side_reliability_intercept: float = 1.25
    occupancy_threshold_min: float = 0.30
    occupancy_threshold_max: float = 0.62
    occupancy_default_threshold: float = 0.44
    min_visible_cycles: float = 3.0
    dotted_min_visible_cycles: float = 5.0
    min_painted_runs: int = 3
    min_gaps: int = 2
    min_period_samples: float = 7.0
    max_candidate_cycles: float = 12.0
    min_duty_ratio: float = 0.16
    max_duty_ratio: float = 0.78
    dotted_max_duty_ratio: float = 0.43
    min_periodicity_score: float = 0.16
    min_dashed_fit_advantage: float = 0.08
    max_run_length_cv: float = 0.85
    max_gap_length_cv: float = 0.85
    solid_min_occupancy: float = 0.52
    solid_min_evidence_mean: float = 0.46
    # Visible-paint presence gates. A pattern can only be inferred or scored when
    # actual bright paint is visible along the corridor. Without these gates a
    # frame with no visible marking (e.g. a pitch-dark night scene) is wrongly
    # explained as a dashed marking whose paint is "all in the gaps", yielding a
    # confident near-zero continuity instead of an honest "unknown".
    min_visible_paint_evidence: float = 0.30
    visible_paint_percentile: float = 85.0
    min_dash_on_evidence: float = 0.30
    min_dash_on_off_contrast: float = 0.10
    missing_paint_threshold: float = 0.22
    faded_paint_threshold: float = 0.52
    intact_degradation_max: float = 0.12
    partial_missing_degradation_min: float = 0.43
    fragmented_missing_fraction_min: float = 0.10
    long_missing_fraction_min: float = 0.18
    glare_luminance: float = 247.0
    dark_unreliable_luminance: float = 12.0
    mask_min_component_area: int = 12
    mask_close_height_fraction: float = 0.18
    mask_close_width_fraction: float = 0.014
    max_lanes_from_mask: int = 12
    aggregation_support_cap_ratio: float = 2.0
    geometry_order_y_ratio: float = 0.35
    robust_fit_min_y_span_px: float = 8.0
    robust_outlier_min_px: float = 3.0
    robust_outlier_mad_multiplier: float = 4.5
    geometry_quality_floor: float = 0.35
    smoothing_fit_weight: float = 0.35
    mask_min_height_support_ratio: float = 0.45
    normalization_vertical_span_fraction: float = 0.18
    geometry_scale_intercept: float = 0.34
    geometry_scale_slope: float = 0.66
    geometry_scale_min: float = 0.24
    geometry_scale_max: float = 1.15
    local_window_min_samples: int = 28
    strip_scale_intercept: float = 0.62
    strip_scale_slope: float = 0.55
    strip_scale_min: float = 0.55
    strip_scale_max: float = 1.18
    object_occlusion_gain: float = 2.0
    side_difference_scale: float = 100.0
    side_reliability_floor: float = 0.25
    center_glare_fraction: float = 0.75
    preserve_local_geometry: bool = False
    side_glare_fraction: float = 0.45
    glare_reliability_multiplier: float = 0.12
    darkness_reliability_multiplier: float = 0.35
    background_luminance_difference_max: float = 95.0
    background_chroma_difference_max: float = 55.0
    background_reliability_multiplier: float = 0.35
    occupancy_min_signal_range: float = 0.06
    period_candidate_count: int = 34
    duty_candidate_count: int = 7
    duty_search_radius: float = 0.05
    phase_candidate_count: int = 24
    strong_periodicity_score: float = 0.25
    alternative_periodicity_min_cycles: float = 4.5
    alternative_periodicity_min_margin: float = 0.15
    dotted_max_run_gap_ratio: float = 0.70
    faded_expected_fraction_min: float = 0.20
    aggregation_min_confidence: float = 0.15
    aggregation_min_valid_fraction: float = 0.10
    aggregation_short_lane_cap_multiplier: float = 4.0


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    return value


def _odd(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 else value + 1


def _points_array(lane: Any) -> np.ndarray:
    if lane is None:
        return np.empty((0, 2), dtype=np.float64)
    if isinstance(lane, Mapping):
        lane = lane.get("points", lane.get("polyline", lane.get("lane", [])))
    if isinstance(lane, np.ndarray):
        array = np.asarray(lane, dtype=np.float64)
    else:
        rows = []
        for point in lane or []:
            if isinstance(point, Mapping):
                rows.append([point.get("x"), point.get("y")])
            else:
                rows.append(point)
        try:
            array = np.asarray(rows, dtype=np.float64)
        except (TypeError, ValueError):
            return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    return array[:, :2]


def lanes_from_lane_json(lane_json: Optional[Mapping[str, Any]]) -> list[np.ndarray]:
    """Convert TuSimple-style lane JSON to float ``(x, y)`` polylines."""

    if not isinstance(lane_json, Mapping):
        return []
    h_samples = np.asarray(lane_json.get("h_samples", []), dtype=np.float64)
    if h_samples.ndim != 1:
        return []
    lanes = []
    for lane_xs in lane_json.get("lanes", []) or []:
        xs = np.asarray(lane_xs, dtype=np.float64)
        if xs.ndim != 1 or xs.size != h_samples.size:
            continue
        valid = np.isfinite(xs) & np.isfinite(h_samples) & (xs >= 0)
        if np.count_nonzero(valid) >= 2:
            lanes.append(np.column_stack([xs[valid], h_samples[valid]]))
    return lanes


def parse_trusted_pattern_hint(metadata: Any) -> tuple[Optional[str], list[str]]:
    """Parse only documented pattern fields, never arbitrary metadata text."""

    values: list[str] = []

    def visit(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        for key, value in item.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in TRUSTED_PATTERN_FIELDS and value is not None:
                values.append(str(value).strip().lower().replace("-", "_").replace(" ", "_"))
            elif isinstance(value, Mapping):
                visit(value)

    visit(metadata)
    mapping = {
        "solid": "solid",
        "continuous": "solid",
        "double_solid": "solid",
        "dashed": "dashed",
        "dash": "dashed",
        "broken": "dashed",
        "dotted": "dotted",
        "solid_dashed": "mixed",
        "dashed_solid": "mixed",
        "mixed": "mixed",
    }
    normalized = [mapping[value] for value in values if value in mapping]
    if not normalized:
        return None, []
    unique = set(normalized)
    if len(unique) == 1:
        return normalized[0], values
    if unique <= {"solid", "dashed", "dotted"}:
        return "mixed", values
    return normalized[0], values


def _deduplicate_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    rounded = np.round(points, decimals=3)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(indices)]


def _order_points(points: np.ndarray, cfg: LaneContinuityConfig) -> np.ndarray:
    y_span = float(np.ptp(points[:, 1]))
    x_span = float(np.ptp(points[:, 0]))
    if y_span >= cfg.geometry_order_y_ratio * max(x_span, 1.0):
        return points[np.lexsort((points[:, 0], points[:, 1]))]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    return points[np.argsort(projection)]


def _robust_smooth(
    points: np.ndarray, cfg: LaneContinuityConfig
) -> tuple[np.ndarray, float]:
    """Reject gross x(y) outliers and lightly smooth without extrapolation."""

    if points.shape[0] < 5 or np.ptp(points[:, 1]) < cfg.robust_fit_min_y_span_px:
        return points, 1.0
    degree = min(3, points.shape[0] - 2)
    keep = np.ones(points.shape[0], dtype=bool)
    for _ in range(2):
        try:
            coeff = np.polyfit(points[keep, 1], points[keep, 0], degree)
        except (np.linalg.LinAlgError, ValueError):
            return points, 0.7
        residual = points[:, 0] - np.polyval(coeff, points[:, 1])
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        threshold = max(
            cfg.robust_outlier_min_px,
            cfg.robust_outlier_mad_multiplier * 1.4826 * mad,
        )
        candidate = np.abs(residual - median) <= threshold
        if np.count_nonzero(candidate) < max(4, points.shape[0] // 2):
            break
        keep = candidate
    retained = points[keep]
    quality = float(
        np.clip(
            np.count_nonzero(keep) / points.shape[0],
            cfg.geometry_quality_floor,
            1.0,
        )
    )
    if retained.shape[0] < 4:
        return retained, quality
    try:
        degree = min(3, retained.shape[0] - 2)
        coeff = np.polyfit(retained[:, 1], retained[:, 0], degree)
        smoothed = retained.copy()
        fitted_x = np.polyval(coeff, retained[:, 1])
        smoothed[:, 0] = (
            (1.0 - cfg.smoothing_fit_weight) * retained[:, 0]
            + cfg.smoothing_fit_weight * fitted_x
        )
        return smoothed, quality
    except (np.linalg.LinAlgError, ValueError):
        return retained, quality


def _resample_polyline(points: np.ndarray, spacing: float, max_samples: int) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    keep = np.concatenate([[True], segment_lengths > 1e-4])
    points = points[keep]
    if points.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    support = float(arc[-1])
    count = int(np.clip(np.floor(support / max(spacing, 0.5)) + 1, 2, max_samples))
    samples = np.linspace(0.0, support, count)
    return np.column_stack([np.interp(samples, arc, points[:, 0]), np.interp(samples, arc, points[:, 1])])


def prepare_guidance_lanes(
    lane_geometries: Optional[Iterable[Any]],
    image_shape: Sequence[int],
    config: Optional[LaneContinuityConfig] = None,
) -> list[dict[str, Any]]:
    """Validate, smooth, and uniformly resample lane geometry in image space."""

    cfg = config or LaneContinuityConfig()
    height, width = int(image_shape[0]), int(image_shape[1])
    prepared = []
    for original_index, lane in enumerate(lane_geometries or []):
        points = _points_array(lane)
        if points.shape[0] < cfg.min_geometry_points:
            continue
        valid = np.isfinite(points).all(axis=1)
        valid &= (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
        valid &= (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
        points = _deduplicate_points(points[valid])
        if points.shape[0] < cfg.min_geometry_points:
            continue
        points = _order_points(points, cfg)
        if cfg.preserve_local_geometry:
            geometry_quality = 1.0
        else:
            points, geometry_quality = _robust_smooth(points, cfg)
        points = _order_points(_deduplicate_points(points), cfg)
        if points.shape[0] < cfg.min_geometry_points:
            continue
        support = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if support < cfg.min_lane_support_px:
            continue
        samples = _resample_polyline(points, cfg.sample_spacing_px, cfg.max_longitudinal_samples)
        if samples.shape[0] < cfg.min_geometry_points:
            continue
        prepared.append(
            {
                "original_index": original_index,
                "points": samples,
                "geometry_quality": geometry_quality,
                "support_px": float(np.sum(np.linalg.norm(np.diff(samples, axis=0), axis=1))),
            }
        )
    return prepared


def extract_guidance_lanes_from_mask(
    mask: Any,
    *,
    config: Optional[LaneContinuityConfig] = None,
) -> list[np.ndarray]:
    """Group disconnected mask segments into guidance centerlines.

    Closing is used only on a disposable geometry copy. The returned lanes are
    spatial guides; this function's mask is never a visible-paint signal.
    """

    cfg = config or LaneContinuityConfig()
    array = np.asarray(mask) if mask is not None else np.empty((0, 0))
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2 or array.size == 0:
        return []
    binary = (array > 0).astype(np.uint8)
    if not np.any(binary):
        return []
    height, width = binary.shape
    raw_count, raw_labels, raw_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for label in range(1, raw_count):
        if int(raw_stats[label, cv2.CC_STAT_AREA]) >= cfg.mask_min_component_area:
            cleaned[raw_labels == label] = 1
    if not np.any(cleaned):
        return []
    kernel_width = _odd(max(3, int(round(width * cfg.mask_close_width_fraction))))
    kernel_height = _odd(max(9, int(round(height * cfg.mask_close_height_fraction))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
    geometry_mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(geometry_mask, 8)
    lanes: list[np.ndarray] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if (
            area < cfg.mask_min_component_area
            or component_height
            < cfg.min_lane_support_px * cfg.mask_min_height_support_ratio
        ):
            continue
        ys, xs = np.where(labels == label)
        row_points = []
        for y in range(y0, y0 + component_height, max(2, int(round(cfg.sample_spacing_px * 2)))):
            row_x = xs[ys == y]
            if row_x.size:
                row_points.append([float(np.median(row_x)), float(y)])
        if len(row_points) >= 2:
            lanes.append(np.asarray(row_points, dtype=np.float64))
    lanes.sort(key=lambda lane: float(np.mean(lane[:, 0])))
    return lanes[: cfg.max_lanes_from_mask]


def _bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    map_x = xs.astype(np.float32)
    map_y = ys.astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _nearest_sample(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(
        mask.astype(np.uint8),
        xs.astype(np.float32),
        ys.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _resize_mask(mask: Any, shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2 or array.size == 0:
        return None
    binary = (array > 0).astype(np.uint8)
    if binary.shape != shape:
        binary = cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return binary


def _longitudinal_coordinate(
    points: np.ndarray,
    image_shape: tuple[int, int],
    homography: Optional[np.ndarray],
    cfg: LaneContinuityConfig,
) -> tuple[np.ndarray, str]:
    if homography is not None:
        matrix = np.asarray(homography, dtype=np.float64)
        if matrix.shape == (3, 3) and np.isfinite(matrix).all():
            projected = cv2.perspectiveTransform(points.astype(np.float32)[None, :, :], matrix)[0]
            if np.isfinite(projected).all():
                steps = np.linalg.norm(np.diff(projected, axis=0), axis=1)
                if np.count_nonzero(steps > EPS) >= 2:
                    return np.concatenate([[0.0], np.cumsum(steps)]), "calibrated_ipm"
    height = image_shape[0]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    vertical_span = float(np.ptp(points[:, 1]))
    if vertical_span >= cfg.normalization_vertical_span_fraction * height:
        midpoint_y = 0.5 * (points[:-1, 1] + points[1:, 1])
        local_scale = np.clip(
            cfg.geometry_scale_intercept
            + cfg.geometry_scale_slope * midpoint_y / max(height - 1, 1),
            cfg.geometry_scale_min,
            cfg.geometry_scale_max,
        )
        return np.concatenate([[0.0], np.cumsum(steps / local_scale)]), "geometry_normalized"
    if points.shape[0] >= cfg.local_window_min_samples:
        return np.concatenate([[0.0], np.cumsum(steps)]), "local_window_normalized"
    return np.concatenate([[0.0], np.cumsum(steps)]), "image_arc_length_fallback"


def rectify_lane_strip(
    image_rgb: np.ndarray,
    centerline: np.ndarray,
    *,
    road_mask: Any = None,
    object_mask: Any = None,
    homography: Optional[np.ndarray] = None,
    geometry_quality: float = 1.0,
    config: Optional[LaneContinuityConfig] = None,
) -> dict[str, Any]:
    """Sample an RGB strip along a centerline using local tangent normals."""

    cfg = config or LaneContinuityConfig()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
    image = image[..., :3].astype(np.uint8, copy=False)
    height, width = image.shape[:2]
    points = np.asarray(centerline, dtype=np.float64)
    tangent = np.gradient(points, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    stable_tangent = tangent_norm > 1e-5
    tangent_norm = np.maximum(tangent_norm, 1e-5)
    tangent = tangent / tangent_norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    perspective = np.clip(
        cfg.strip_scale_intercept
        + cfg.strip_scale_slope * points[:, 1] / max(height - 1, 1),
        cfg.strip_scale_min,
        cfg.strip_scale_max,
    )
    base_half_width = np.clip(
        width * cfg.center_half_width_fraction,
        cfg.center_half_width_min_px,
        cfg.center_half_width_max_px,
    )
    half_width = np.clip(
        base_half_width * perspective,
        cfg.center_half_width_min_px,
        cfg.center_half_width_max_px,
    )
    relative_offsets = np.linspace(
        -cfg.side_outer_factor,
        cfg.side_outer_factor,
        cfg.cross_section_samples,
        dtype=np.float64,
    )
    offsets = half_width[:, None] * relative_offsets[None, :]
    xs = points[:, 0, None] + normal[:, 0, None] * offsets
    ys = points[:, 1, None] + normal[:, 1, None] * offsets
    in_frame = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
    strip_rgb = _bilinear_sample(image, xs, ys)

    center_columns = np.abs(relative_offsets) <= 1.0
    left_columns = (relative_offsets <= -cfg.side_inner_factor) & (relative_offsets >= -cfg.side_outer_factor)
    right_columns = (relative_offsets >= cfg.side_inner_factor) & (relative_offsets <= cfg.side_outer_factor)

    center_frame = np.mean(in_frame[:, center_columns], axis=1)
    left_frame = np.mean(in_frame[:, left_columns], axis=1)
    right_frame = np.mean(in_frame[:, right_columns], axis=1)
    q = np.minimum(np.minimum(center_frame, left_frame), right_frame).astype(np.float64)
    q *= float(np.clip(geometry_quality, 0.0, 1.0))
    q *= stable_tangent.astype(np.float64)

    road = _resize_mask(road_mask, (height, width))
    if road is not None:
        sampled_road = _nearest_sample(road, xs, ys) > 0
        left_road = np.mean(sampled_road[:, left_columns], axis=1)
        right_road = np.mean(sampled_road[:, right_columns], axis=1)
        center_road = np.mean(sampled_road[:, center_columns], axis=1)
        q *= np.minimum(np.minimum(left_road, right_road), center_road)

    objects = _resize_mask(object_mask, (height, width))
    if objects is not None:
        sampled_objects = _nearest_sample(objects, xs, ys) > 0
        occluded = np.mean(sampled_objects[:, np.abs(relative_offsets) <= cfg.side_inner_factor], axis=1)
        q *= np.clip(1.0 - cfg.object_occlusion_gain * occluded, 0.0, 1.0)

    s, coordinate_method = _longitudinal_coordinate(
        points, (height, width), homography, cfg
    )
    outer_left = points - normal * (half_width * cfg.side_outer_factor)[:, None]
    outer_right = points + normal * (half_width * cfg.side_outer_factor)[:, None]
    return {
        "rgb": strip_rgb,
        "q_geometry": np.clip(q, 0.0, 1.0),
        "relative_offsets": relative_offsets,
        "center_columns": center_columns,
        "left_columns": left_columns,
        "right_columns": right_columns,
        "centerline": points,
        "corridor_left": outer_left,
        "corridor_right": outer_right,
        "s": s,
        "coordinate_method": coordinate_method,
        "half_width": half_width,
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    denominator = float(np.sum(weights))
    if denominator <= EPS:
        return None
    return float(np.sum(values * weights) / denominator)


def _weighted_percentile(
    values: np.ndarray, weights: np.ndarray, percentile: float
) -> Optional[float]:
    """Weighted percentile over strictly positive-weight samples.

    Used to summarise the strongest paint evidence along a corridor so that a
    marking with visible paint somewhere is distinguished from a corridor whose
    evidence is entirely at the noise floor.
    """

    mask = weights > EPS
    if not np.any(mask):
        return None
    v = np.asarray(values, dtype=np.float64)[mask]
    w = np.asarray(weights, dtype=np.float64)[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    total = float(np.sum(w))
    if total <= EPS:
        return float(np.median(v))
    quantile = np.clip(percentile / 100.0, 0.0, 1.0)
    return float(np.interp(quantile * total, cumulative, v))


def compute_paint_evidence(
    strip: Mapping[str, Any],
    *,
    config: Optional[LaneContinuityConfig] = None,
) -> dict[str, np.ndarray]:
    """Compute continuous paint evidence ``p`` and observation validity ``q``."""

    cfg = config or LaneContinuityConfig()
    rgb = np.asarray(strip["rgb"], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    center = np.asarray(strip["center_columns"], dtype=bool)
    left = np.asarray(strip["left_columns"], dtype=bool)
    right = np.asarray(strip["right_columns"], dtype=bool)

    center_l = np.percentile(lab[:, center, 0], 75, axis=1)
    center_l_median = np.median(lab[:, center, 0], axis=1)
    left_l = np.median(lab[:, left, 0], axis=1)
    right_l = np.median(lab[:, right, 0], axis=1)
    side_l = 0.5 * (left_l + right_l)
    delta_l = center_l - side_l
    relative_luminance = np.clip(
        (delta_l - cfg.paint_delta_floor) / cfg.paint_delta_scale,
        0.0,
        1.0,
    )
    paired_contrast = np.clip(
        np.minimum(center_l - left_l, center_l - right_l) / cfg.paint_delta_scale,
        0.0,
        1.0,
    )

    center_a = np.median(lab[:, center, 1], axis=1)
    center_b = np.median(lab[:, center, 2], axis=1)
    side_a = 0.5 * (np.median(lab[:, left, 1], axis=1) + np.median(lab[:, right, 1], axis=1))
    side_b = 0.5 * (np.median(lab[:, left, 2], axis=1) + np.median(lab[:, right, 2], axis=1))
    neutral_distance = np.sqrt((center_a - 128.0) ** 2 + (center_b - 128.0) ** 2)
    white_likelihood = np.clip(
        center_l_median / cfg.white_luminance_normalizer, 0.0, 1.0
    ) * np.exp(-neutral_distance / cfg.neutral_chroma_scale)
    yellow_shift = np.clip(
        (center_b - side_b - cfg.yellow_chroma_floor) / cfg.yellow_chroma_scale,
        0.0,
        1.0,
    )
    yellow_hue = np.median(hsv[:, center, 0], axis=1)
    yellow_saturation = np.median(hsv[:, center, 1], axis=1) / 255.0
    yellow_likelihood = yellow_shift * yellow_saturation * np.exp(
        -((yellow_hue - cfg.yellow_hue_center) / cfg.yellow_hue_scale) ** 2
    )
    color_likelihood = np.maximum(white_likelihood, yellow_likelihood)

    center_std = np.std(lab[:, center, 0], axis=1)
    width_consistency = np.clip(
        1.0 - center_std / cfg.stripe_consistency_scale,
        cfg.stripe_consistency_floor,
        1.0,
    )
    edge_response = np.clip(
        np.minimum(np.abs(center_l_median - left_l), np.abs(center_l_median - right_l))
        / cfg.edge_delta_scale,
        0.0,
        1.0,
    )
    base_evidence = (
        cfg.luminance_evidence_weight * relative_luminance
        + cfg.paired_contrast_weight * paired_contrast
        + cfg.edge_evidence_weight * edge_response
        + cfg.color_evidence_weight * color_likelihood
    )
    p = base_evidence * (
        cfg.width_consistency_evidence_floor
        + (1.0 - cfg.width_consistency_evidence_floor) * width_consistency
    )
    p = np.clip(p, 0.0, 1.0)

    q = np.asarray(strip["q_geometry"], dtype=np.float64).copy()
    side_difference = np.abs(left_l - right_l)
    q *= np.clip(
        cfg.side_reliability_intercept
        - side_difference / cfg.side_difference_scale,
        cfg.side_reliability_floor,
        1.0,
    )
    center_saturation = np.mean(np.max(rgb[:, center, :], axis=2) >= 253, axis=1)
    side_saturation = 0.5 * (
        np.mean(np.max(rgb[:, left, :], axis=2) >= 253, axis=1)
        + np.mean(np.max(rgb[:, right, :], axis=2) >= 253, axis=1)
    )
    glare = (side_l >= cfg.glare_luminance) | (
        (center_saturation > cfg.center_glare_fraction)
        & (side_saturation > cfg.side_glare_fraction)
    )
    q[glare] *= cfg.glare_reliability_multiplier
    severe_darkness = np.maximum(center_l_median, side_l) <= cfg.dark_unreliable_luminance
    q[severe_darkness] *= cfg.darkness_reliability_multiplier
    chroma_disagreement = np.abs(center_a - side_a) + np.abs(center_b - side_b)
    implausible_background = (
        side_difference > cfg.background_luminance_difference_max
    ) & (chroma_disagreement > cfg.background_chroma_difference_max)
    q[implausible_background] *= cfg.background_reliability_multiplier
    return {
        "p": np.clip(p, 0.0, 1.0),
        "q": np.clip(q, 0.0, 1.0),
        "relative_luminance": relative_luminance,
        "paired_contrast": paired_contrast,
        "white_likelihood": white_likelihood,
        "yellow_likelihood": yellow_likelihood,
    }


def _adaptive_occupancy(p: np.ndarray, q: np.ndarray, cfg: LaneContinuityConfig) -> tuple[np.ndarray, float]:
    valid_values = p[q >= cfg.min_pattern_q]
    if (
        valid_values.size < 2
        or float(np.ptp(valid_values)) < cfg.occupancy_min_signal_range
    ):
        threshold = cfg.occupancy_default_threshold
    else:
        values_u8 = np.clip(np.round(valid_values * 255.0), 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(otsu / 255.0)
        threshold = float(np.clip(threshold, cfg.occupancy_threshold_min, cfg.occupancy_threshold_max))
    return p >= threshold, threshold


def _run_statistics(state: np.ndarray, valid: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    runs: list[tuple[bool, float]] = []
    active_state: Optional[bool] = None
    start = 0
    for index in range(state.size + 1):
        current = None if index == state.size or not valid[index] else bool(state[index])
        if current != active_state:
            if active_state is not None and index > start:
                end_index = min(index, state.size - 1)
                length = float(max(s[end_index] - s[start], EPS))
                runs.append((active_state, length))
            active_state = current
            start = index
    painted = np.asarray([length for is_paint, length in runs if is_paint], dtype=np.float64)
    gaps = np.asarray([length for is_paint, length in runs if not is_paint], dtype=np.float64)

    def stats(values: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        if values.size == 0:
            return None, None
        mean = float(np.mean(values))
        cv = 0.0 if values.size == 1 else float(np.std(values) / max(mean, EPS))
        return mean, cv

    mean_run, run_cv = stats(painted)
    mean_gap, gap_cv = stats(gaps)
    return {
        "painted_run_count": int(painted.size),
        "gap_count": int(gaps.size),
        "mean_run_length": mean_run,
        "median_run_length": None if painted.size == 0 else float(np.median(painted)),
        "run_length_cv": run_cv,
        "mean_gap_length": mean_gap,
        "median_gap_length": None if gaps.size == 0 else float(np.median(gaps)),
        "gap_length_cv": gap_cv,
    }


def _weighted_correlation(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum <= EPS:
        return 0.0
    mean_a = float(np.sum(a * weights) / weight_sum)
    mean_b = float(np.sum(b * weights) / weight_sum)
    centered_a = a - mean_a
    centered_b = b - mean_b
    denominator = np.sqrt(
        float(np.sum(weights * centered_a**2)) * float(np.sum(weights * centered_b**2))
    )
    if denominator <= EPS:
        return 0.0
    return float(np.clip(np.sum(weights * centered_a * centered_b) / denominator, -1.0, 1.0))


def _periodicity_at_period(p: np.ndarray, q: np.ndarray, s: np.ndarray, period: float) -> float:
    if p.size < 4 or period <= EPS:
        return 0.0
    median_step = float(np.median(np.diff(s))) if s.size > 1 else 1.0
    lag = int(round(period / max(median_step, EPS)))
    if lag < 2 or lag >= p.size - 2:
        return 0.0
    weights = q[:-lag] * q[lag:]
    corr = _weighted_correlation(p[:-lag], p[lag:], weights)
    return float(np.clip(corr, 0.0, 1.0))


def _fit_periodic_template(
    p: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    cfg: LaneContinuityConfig,
    painted_run_length: Optional[float] = None,
) -> dict[str, Any]:
    valid = q >= cfg.min_pattern_q
    if np.count_nonzero(valid) < cfg.min_valid_samples:
        return {"template": None, "score": None, "period": None, "duty_ratio": None, "phase": None, "periodicity": None, "cycles": 0.0}
    valid_s = s[valid]
    span = float(valid_s[-1] - valid_s[0])
    median_step = float(np.median(np.diff(s))) if s.size > 1 else 1.0
    min_period = max(cfg.min_period_samples * median_step, span / cfg.max_candidate_cycles)
    max_period = span / max(cfg.min_visible_cycles - 0.15, 1.0)
    if max_period <= min_period or span <= EPS:
        return {"template": None, "score": None, "period": None, "duty_ratio": None, "phase": None, "periodicity": None, "cycles": 0.0}
    periods = np.linspace(min_period, max_period, cfg.period_candidate_count)
    best: Optional[dict[str, Any]] = None
    weight_sum = float(np.sum(q))
    for period in periods:
        if painted_run_length is None:
            duties = np.linspace(
                cfg.min_duty_ratio,
                cfg.max_duty_ratio,
                cfg.duty_candidate_count,
            )
        else:
            duty_center = float(
                np.clip(
                    painted_run_length / max(period, EPS),
                    cfg.min_duty_ratio,
                    cfg.max_duty_ratio,
                )
            )
            duties = np.linspace(
                max(cfg.min_duty_ratio, duty_center - cfg.duty_search_radius),
                min(cfg.max_duty_ratio, duty_center + cfg.duty_search_radius),
                cfg.duty_candidate_count,
            )
        for duty in duties:
            for phase_fraction in np.linspace(
                0.0,
                1.0,
                cfg.phase_candidate_count,
                endpoint=False,
            ):
                phase = float(phase_fraction * period)
                template = (np.mod(s - phase, period) < duty * period).astype(np.float64)
                mae = float(np.sum(q * np.abs(p - template)) / max(weight_sum, EPS))
                score = float(np.clip(1.0 - mae, 0.0, 1.0))
                if best is None or score > best["score"]:
                    best = {
                        "template": template,
                        "score": score,
                        "period": float(period),
                        "duty_ratio": float(duty),
                        "phase": phase,
                    }
    assert best is not None
    best["periodicity"] = _periodicity_at_period(p, q, s, best["period"])
    best["cycles"] = float(span / max(best["period"], EPS))
    return best


def fit_pattern_hypotheses(
    p: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    *,
    trusted_pattern_hints: Any = None,
    config: Optional[LaneContinuityConfig] = None,
) -> dict[str, Any]:
    """Fit solid and periodic hypotheses using continuous RGB paint evidence."""

    cfg = config or LaneContinuityConfig()
    occupancy, occupancy_threshold = _adaptive_occupancy(p, q, cfg)
    valid = q >= cfg.min_pattern_q
    run_stats = _run_statistics(occupancy, valid, s)
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = float(np.mean(valid)) if valid.size else 0.0
    support = float(s[valid][-1] - s[valid][0]) if valid_count >= 2 else 0.0
    evidence_mean = _weighted_mean(p, q)
    occupancy_fraction = _weighted_mean(occupancy.astype(np.float64), q)
    solid_score = evidence_mean
    # Strongest paint evidence anywhere along the valid corridor. When even this
    # is at the noise floor, no physical marking is visible and no pattern can be
    # inferred from RGB, regardless of how "periodic" the residual noise looks.
    valid_weights = q * valid.astype(np.float64)
    paint_peak = _weighted_percentile(p, valid_weights, cfg.visible_paint_percentile)
    visible_paint_present = (
        paint_peak is not None and paint_peak >= cfg.min_visible_paint_evidence
    )
    median_run = _finite_float(run_stats.get("median_run_length"))
    periodic = _fit_periodic_template(p, q, s, cfg, painted_run_length=median_run)
    dashed_score = _finite_float(periodic.get("score"))
    # Amplitude check for the dashed hypothesis: the fitted painted ("on") phase
    # must contain genuinely brighter paint than the fitted gaps ("off" phase).
    # This blocks classifying uniformly-dark noise as dashed with tiny duty.
    dash_on_evidence: Optional[float] = None
    dash_on_off_contrast: Optional[float] = None
    template = periodic.get("template")
    if template is not None:
        template_on = np.asarray(template, dtype=np.float64) > 0.5
        on_weights = q * template_on.astype(np.float64)
        off_weights = q * (~template_on).astype(np.float64)
        on_mean = _weighted_mean(p, on_weights)
        off_mean = _weighted_mean(p, off_weights)
        dash_on_evidence = on_mean
        if on_mean is not None and off_mean is not None:
            dash_on_off_contrast = float(on_mean - off_mean)
    score_margin = None if dashed_score is None or solid_score is None else float(dashed_score - solid_score)
    visible_cycles = float(periodic.get("cycles") or 0.0)
    periodicity = float(periodic.get("periodicity") or 0.0)
    duty = _finite_float(periodic.get("duty_ratio"))
    run_cv = _finite_float(run_stats.get("run_length_cv"))
    gap_cv = _finite_float(run_stats.get("gap_length_cv"))
    enough_support = (
        valid_count >= cfg.min_valid_samples
        and valid_fraction >= cfg.min_valid_fraction
        and support >= cfg.min_pattern_support_px
    )
    repeated = (
        run_stats["painted_run_count"] >= cfg.min_painted_runs
        and run_stats["gap_count"] >= cfg.min_gaps
        and visible_cycles >= cfg.min_visible_cycles
    )
    regular_runs = run_cv is not None and run_cv <= cfg.max_run_length_cv
    regular_gaps = gap_cv is not None and gap_cv <= cfg.max_gap_length_cv
    periodic_advantage = score_margin is not None and score_margin >= cfg.min_dashed_fit_advantage
    periodicity_supported = periodicity >= cfg.min_periodicity_score and (
        periodicity >= cfg.strong_periodicity_score
        or (
            visible_cycles >= cfg.alternative_periodicity_min_cycles
            and score_margin is not None
            and score_margin >= cfg.alternative_periodicity_min_margin
        )
    )
    dash_amplitude_ok = (
        dash_on_evidence is not None
        and dash_on_evidence >= cfg.min_dash_on_evidence
        and dash_on_off_contrast is not None
        and dash_on_off_contrast >= cfg.min_dash_on_off_contrast
    )
    dashed_supported = (
        visible_paint_present
        and enough_support
        and repeated
        and regular_runs
        and regular_gaps
        and periodicity_supported
        and periodic_advantage
        and dash_amplitude_ok
        and duty is not None
        and cfg.min_duty_ratio <= duty <= cfg.max_duty_ratio
    )
    solid_supported = (
        visible_paint_present
        and enough_support
        and evidence_mean is not None
        and occupancy_fraction is not None
        and evidence_mean >= cfg.solid_min_evidence_mean
        and occupancy_fraction >= cfg.solid_min_occupancy
        and not dashed_supported
    )

    rgb_pattern = "unknown"
    if dashed_supported:
        mean_run = _finite_float(run_stats.get("mean_run_length"))
        mean_gap = _finite_float(run_stats.get("mean_gap_length"))
        if (
            duty is not None
            and duty <= cfg.dotted_max_duty_ratio
            and visible_cycles >= cfg.dotted_min_visible_cycles
            and mean_run is not None
            and mean_gap is not None
            and mean_run < cfg.dotted_max_run_gap_ratio * mean_gap
        ):
            rgb_pattern = "dotted"
        else:
            rgb_pattern = "dashed"
    elif solid_supported:
        rgb_pattern = "solid"

    trusted_pattern, trusted_values = parse_trusted_pattern_hint(trusted_pattern_hints)
    notes: list[str] = []
    if trusted_pattern is not None:
        intended_pattern = trusted_pattern
        compatible = (
            rgb_pattern == trusted_pattern
            or {rgb_pattern, trusted_pattern} <= {"dashed", "dotted"}
            or rgb_pattern == "unknown"
        )
        if compatible and rgb_pattern != "unknown":
            pattern_source = "combination/prior"
        else:
            pattern_source = "trusted_target_metadata"
        if rgb_pattern not in {"unknown", trusted_pattern} and not ({rgb_pattern, trusted_pattern} <= {"dashed", "dotted"}):
            notes.append(f"RGB inference ({rgb_pattern}) conflicts with trusted hint ({trusted_pattern})")
    else:
        intended_pattern = rgb_pattern
        pattern_source = "RGB_pattern_inference" if rgb_pattern != "unknown" else "unknown"

    if not enough_support:
        notes.append("insufficient reliable longitudinal support for RGB pattern inference")
    if not visible_paint_present and trusted_pattern is None:
        notes.append("no visible paint evidence above noise floor; pattern is unknown")
    if intended_pattern == "dotted" and rgb_pattern != "dotted" and trusted_pattern is None:
        intended_pattern = "dashed"
        notes.append("dotted evidence was insufficient; retained dashed/broken family")

    if intended_pattern in {"dashed", "dotted"} and periodic.get("template") is None:
        expected_template = None
    elif intended_pattern in {"dashed", "dotted"}:
        expected_template = periodic["template"]
    elif intended_pattern == "solid":
        expected_template = np.ones_like(p, dtype=np.float64)
    else:
        expected_template = None

    if intended_pattern == "unknown":
        confidence = 0.0
    elif trusted_pattern is not None:
        rgb_agreement = 1.0 if compatible and rgb_pattern != "unknown" else 0.55
        confidence = float(np.clip(0.72 + 0.23 * rgb_agreement * min(valid_fraction, 1.0), 0.0, 0.98))
    elif intended_pattern in {"dashed", "dotted"}:
        margin_term = 0.0 if score_margin is None else np.clip(score_margin / 0.35, 0.0, 1.0)
        confidence = float(
            np.clip(
                0.18
                + 0.30 * periodicity
                + 0.27 * margin_term
                + 0.25 * np.clip(visible_cycles / 6.0, 0.0, 1.0),
                0.0,
                0.96,
            )
        )
    else:
        separation = 0.0 if score_margin is None else np.clip(-score_margin / 0.25, 0.0, 1.0)
        confidence = float(np.clip(0.30 + 0.42 * (occupancy_fraction or 0.0) + 0.18 * separation, 0.0, 0.93))

    return {
        "intended_pattern": intended_pattern,
        "pattern_source": pattern_source,
        "pattern_confidence": confidence,
        "rgb_inferred_pattern": rgb_pattern,
        "trusted_hint_values": trusted_values,
        "expected_template": expected_template,
        "occupancy": occupancy,
        "occupancy_threshold": occupancy_threshold,
        "valid_sample_count": valid_count,
        "valid_sample_fraction": valid_fraction,
        "usable_support": support,
        "paint_evidence_mean": evidence_mean,
        "paint_evidence_peak": paint_peak,
        "visible_paint_present": bool(visible_paint_present),
        "occupancy_fraction": occupancy_fraction,
        "solid_fit_score": solid_score,
        "dashed_fit_score": dashed_score,
        "model_score_margin": score_margin,
        "fitted_period": _finite_float(periodic.get("period")),
        "duty_ratio": duty,
        "periodicity_score": periodicity if periodic.get("period") is not None else None,
        "visible_cycle_count": visible_cycles,
        **run_stats,
        "notes": notes,
    }


def _max_missing_fraction(expected: np.ndarray, missing: np.ndarray, s: np.ndarray) -> float:
    expected_total = float(np.sum(expected))
    if expected_total <= EPS:
        return 0.0
    longest = 0
    current = 0
    for is_expected, is_missing in zip(expected, missing):
        if is_expected and is_missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    _ = s
    return float(longest / expected_total)


def compute_pattern_condition(
    p: np.ndarray,
    q: np.ndarray,
    expected_template: Optional[np.ndarray],
    *,
    config: Optional[LaneContinuityConfig] = None,
) -> dict[str, Any]:
    """Score only positions where the selected intended pattern expects paint."""

    cfg = config or LaneContinuityConfig()
    if expected_template is None:
        return {
            "continuity_score": None,
            "degradation_score": None,
            "condition": "unknown_condition",
            "expected_paint_evidence_mean": None,
            "missing_expected_fraction": None,
            "longest_missing_expected_fraction": None,
            "unavailable_reason": "intended_pattern_or_expected_template_unavailable",
        }
    expected = np.asarray(expected_template, dtype=np.float64) > 0.5
    weights = q * expected.astype(np.float64)
    denominator = float(np.sum(weights))
    if denominator < cfg.min_expected_weight or np.count_nonzero(weights > 0) < cfg.min_expected_samples:
        return {
            "continuity_score": None,
            "degradation_score": None,
            "condition": "unknown_condition",
            "expected_paint_evidence_mean": None,
            "expected_paint_peak": None,
            "missing_expected_fraction": None,
            "longest_missing_expected_fraction": None,
            "unavailable_reason": "insufficient_reliable_expected_paint_support",
        }
    # Even with a selected pattern (including a trusted-metadata hint), condition
    # can only be scored where paint is actually visible. If the strongest
    # evidence along the expected-paint corridor is at the noise floor, the
    # marking cannot be seen at all and we must return None rather than a
    # fabricated near-zero continuity.
    valid_expected = expected & (q >= cfg.min_pattern_q)
    expected_peak = _weighted_percentile(
        p, q * valid_expected.astype(np.float64), cfg.visible_paint_percentile
    )
    if expected_peak is None or expected_peak < cfg.min_visible_paint_evidence:
        return {
            "continuity_score": None,
            "degradation_score": None,
            "condition": "unknown_condition",
            "expected_paint_evidence_mean": None,
            "expected_paint_peak": expected_peak,
            "missing_expected_fraction": None,
            "longest_missing_expected_fraction": None,
            "unavailable_reason": "no_visible_paint_evidence",
        }
    continuity = float(np.clip(np.sum(weights * p) / max(denominator, EPS), 0.0, 1.0))
    degradation = float(np.clip(1.0 - continuity, 0.0, 1.0))
    missing = expected & (p < cfg.missing_paint_threshold) & (q >= cfg.min_pattern_q)
    missing_fraction = float(np.sum(q * missing) / max(denominator, EPS))
    longest_missing = _max_missing_fraction(expected, missing, np.arange(p.size, dtype=np.float64))
    weak = expected & (p < cfg.faded_paint_threshold) & ~missing
    weak_fraction = float(np.sum(q * weak) / max(denominator, EPS))
    if degradation <= cfg.intact_degradation_max and missing_fraction < cfg.fragmented_missing_fraction_min * 0.5:
        condition = "intact"
    elif degradation >= cfg.partial_missing_degradation_min or longest_missing >= cfg.long_missing_fraction_min:
        condition = "partially_missing"
    elif missing_fraction >= cfg.fragmented_missing_fraction_min:
        condition = "fragmented"
    elif (
        weak_fraction >= cfg.faded_expected_fraction_min
        or degradation > cfg.intact_degradation_max
    ):
        condition = "faded"
    else:
        condition = "intact"
    return {
        "continuity_score": continuity,
        "degradation_score": degradation,
        "condition": condition,
        "expected_paint_evidence_mean": continuity,
        "expected_paint_peak": expected_peak,
        "missing_expected_fraction": missing_fraction,
        "longest_missing_expected_fraction": longest_missing,
        "unavailable_reason": None,
    }


def aggregate_lane_results(
    lane_results: Sequence[Mapping[str, Any]],
    *,
    config: Optional[LaneContinuityConfig] = None,
) -> dict[str, Any]:
    """Aggregate valid lanes with confidence and bounded support weighting."""

    cfg = config or LaneContinuityConfig()
    valid = [lane for lane in lane_results if _finite_float(lane.get("continuity_score")) is not None]
    unknown_count = sum(
        1
        for lane in lane_results
        if lane.get("intended_pattern") == "unknown" or _finite_float(lane.get("continuity_score")) is None
    )
    if not valid:
        reason = "no_valid_lane_scores" if lane_results else "no_credible_guidance_geometry"
        return {
            "continuity_score": None,
            "degradation_score": None,
            "valid_lane_count": 0,
            "unknown_lane_count": int(unknown_count),
            "aggregation_method": "confidence_x_sqrt_bounded_support",
            "worst_valid_lane_score": None,
            "median_valid_lane_score": None,
            "intended_pattern": "unknown",
            "pattern_confidence": None,
            "condition": "unknown_condition",
            "pattern_source": "unknown",
            "unavailable_reason": reason,
        }
    supports = np.asarray([max(float(lane.get("usable_support") or 0.0), EPS) for lane in valid])
    support_cap = max(
        min(
            float(np.median(supports)) * cfg.aggregation_support_cap_ratio,
            float(np.min(supports))
            * cfg.aggregation_short_lane_cap_multiplier,
        ),
        EPS,
    )
    scores = np.asarray([float(lane["continuity_score"]) for lane in valid])
    weights = []
    for lane, support in zip(valid, supports):
        confidence = max(
            float(lane.get("pattern_confidence") or 0.0),
            cfg.aggregation_min_confidence,
        )
        valid_fraction = max(
            float(lane.get("valid_sample_fraction") or 0.0),
            cfg.aggregation_min_valid_fraction,
        )
        weights.append(confidence * valid_fraction * np.sqrt(min(support, support_cap)))
    weights_array = np.asarray(weights, dtype=np.float64)
    if float(np.sum(weights_array)) <= EPS:
        weights_array = np.ones_like(scores)
    continuity = float(np.sum(weights_array * scores) / np.sum(weights_array))
    patterns = {str(lane.get("intended_pattern")) for lane in valid}
    aggregate_pattern = next(iter(patterns)) if len(patterns) == 1 else "mixed"
    conditions = {str(lane.get("condition")) for lane in valid}
    if len(conditions) == 1:
        aggregate_condition = next(iter(conditions))
    elif "partially_missing" in conditions:
        aggregate_condition = "partially_missing"
    elif "fragmented" in conditions:
        aggregate_condition = "fragmented"
    elif "faded" in conditions:
        aggregate_condition = "faded"
    else:
        aggregate_condition = "intact"
    pattern_sources = {str(lane.get("pattern_source")) for lane in valid}
    aggregate_source = next(iter(pattern_sources)) if len(pattern_sources) == 1 else "combination/prior"
    pattern_confidence = float(
        np.sum(weights_array * np.asarray([float(lane.get("pattern_confidence") or 0.0) for lane in valid]))
        / np.sum(weights_array)
    )
    return {
        "continuity_score": continuity,
        "degradation_score": float(1.0 - continuity),
        "valid_lane_count": len(valid),
        "unknown_lane_count": int(unknown_count),
        "aggregation_method": "confidence_x_sqrt_bounded_support",
        "worst_valid_lane_score": float(np.min(scores)),
        "median_valid_lane_score": float(np.median(scores)),
        "intended_pattern": aggregate_pattern,
        "pattern_confidence": pattern_confidence,
        "condition": aggregate_condition,
        "pattern_source": aggregate_source,
        "unavailable_reason": None,
    }


def analyze_lane_pattern_continuity(
    image_rgb: np.ndarray,
    lane_geometries: Optional[Iterable[Any]],
    *,
    geometry_source: str,
    road_mask: Any = None,
    object_mask: Any = None,
    trusted_pattern_hints: Any = None,
    homography: Optional[np.ndarray] = None,
    config: Optional[LaneContinuityConfig] = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Analyze intended lane pattern and observed expected-paint continuity.

    Args:
        image_rgb: Original RGB image. It is the only visible-paint evidence.
        lane_geometries: Polylines used solely to locate marking corridors.
        geometry_source: Provenance string recorded in output diagnostics.
        road_mask: Optional drivable-area mask used to validate side bands.
        object_mask: Optional object/occlusion mask used to invalidate samples.
        trusted_pattern_hints: Explicit target metadata fields for canonical I1.
        homography: Optional image-to-ground homography for longitudinal scale.
        config: Optional threshold configuration.
        debug: Include sampled signals and corridor geometry when true.

    Returns:
        JSON-safe image aggregate and compact diagnostics for every usable lane.
    """

    cfg = config or LaneContinuityConfig()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        result = aggregate_lane_results([], config=cfg)
        result.update({"geometry_source": geometry_source, "lane_diagnostics": [], "config": asdict(cfg) if debug else None})
        result["unavailable_reason"] = "invalid_or_empty_rgb_image"
        return _json_safe(result)

    prepared = prepare_guidance_lanes(lane_geometries, image.shape[:2], config=cfg)
    lane_results: list[dict[str, Any]] = []
    for lane_index, lane in enumerate(prepared):
        strip = rectify_lane_strip(
            image,
            lane["points"],
            road_mask=road_mask,
            object_mask=object_mask,
            homography=homography,
            geometry_quality=lane["geometry_quality"],
            config=cfg,
        )
        evidence = compute_paint_evidence(strip, config=cfg)
        p, q = evidence["p"], evidence["q"]
        hypotheses = fit_pattern_hypotheses(
            p,
            q,
            strip["s"],
            trusted_pattern_hints=trusted_pattern_hints,
            config=cfg,
        )
        condition = compute_pattern_condition(p, q, hypotheses["expected_template"], config=cfg)
        unavailable_reason = condition["unavailable_reason"]
        if hypotheses["intended_pattern"] == "unknown":
            unavailable_reason = "intended_pattern_unknown"
        diagnostic: dict[str, Any] = {
            "lane_index": lane_index,
            "geometry_source": geometry_source,
            "pattern_source": hypotheses["pattern_source"],
            "longitudinal_coordinate_method": strip["coordinate_method"],
            "intended_pattern": hypotheses["intended_pattern"],
            "pattern_confidence": hypotheses["pattern_confidence"],
            "condition": condition["condition"],
            "continuity_score": condition["continuity_score"] if hypotheses["intended_pattern"] != "unknown" else None,
            "degradation_score": condition["degradation_score"] if hypotheses["intended_pattern"] != "unknown" else None,
            "valid_sample_fraction": hypotheses["valid_sample_fraction"],
            "usable_support": hypotheses["usable_support"],
            "usable_support_px": lane["support_px"],
            "paint_evidence_mean": hypotheses["paint_evidence_mean"],
            "paint_evidence_peak": hypotheses["paint_evidence_peak"],
            "expected_paint_evidence_mean": condition["expected_paint_evidence_mean"],
            "painted_run_count": hypotheses["painted_run_count"],
            "gap_count": hypotheses["gap_count"],
            "visible_cycle_count": hypotheses["visible_cycle_count"],
            "mean_run_length": hypotheses["mean_run_length"],
            "run_length_cv": hypotheses["run_length_cv"],
            "mean_gap_length": hypotheses["mean_gap_length"],
            "gap_length_cv": hypotheses["gap_length_cv"],
            "fitted_period": hypotheses["fitted_period"],
            "duty_ratio": hypotheses["duty_ratio"],
            "periodicity_score": hypotheses["periodicity_score"],
            "solid_fit_score": hypotheses["solid_fit_score"],
            "dashed_fit_score": hypotheses["dashed_fit_score"],
            "model_score_margin": hypotheses["model_score_margin"],
            "occupancy_threshold": hypotheses["occupancy_threshold"],
            "missing_expected_fraction": condition["missing_expected_fraction"],
            "longest_missing_expected_fraction": condition["longest_missing_expected_fraction"],
            "unavailable_reason": unavailable_reason,
            "notes": hypotheses["notes"],
        }
        if debug:
            expected = hypotheses["expected_template"]
            missing = None if expected is None else ((expected > 0.5) & (p < cfg.missing_paint_threshold) & (q >= cfg.min_pattern_q))
            diagnostic["debug"] = {
                "centerline": strip["centerline"],
                "corridor_left": strip["corridor_left"],
                "corridor_right": strip["corridor_right"],
                "s": strip["s"],
                "paint_evidence": p,
                "validity": q,
                "occupancy": hypotheses["occupancy"].astype(np.uint8),
                "expected_template": expected,
                "missing_expected_paint": missing,
            }
        lane_results.append(diagnostic)

    aggregate = aggregate_lane_results(lane_results, config=cfg)
    aggregate.update(
        {
            "geometry_source": geometry_source,
            "lane_diagnostics": lane_results,
        }
    )
    if debug:
        aggregate["config"] = asdict(cfg)
    return _json_safe(aggregate)


__all__ = [
    "LaneContinuityConfig",
    "aggregate_lane_results",
    "analyze_lane_pattern_continuity",
    "compute_paint_evidence",
    "compute_pattern_condition",
    "extract_guidance_lanes_from_mask",
    "fit_pattern_hypotheses",
    "lanes_from_lane_json",
    "parse_trusted_pattern_hint",
    "prepare_guidance_lanes",
    "rectify_lane_strip",
]
