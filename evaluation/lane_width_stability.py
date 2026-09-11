"""Perspective-aware classical geometry analysis for lane-width stability.

The module measures the distance between *adjacent lane-boundary centerlines*.
It does not measure paint-stripe thickness.  Native polylines, lane JSON, and
mask-derived centerlines all enter through the geometry utilities shared with
the I1 lane-continuity implementation.

Physical units are reported only when the caller supplies a calibrated,
metric-scale image-to-road-plane transform.  Without calibration the same
orthogonal sampling and profile fitting are used on a conservative image-space
region, and widths are explicitly reported as fractions of image width.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
from scipy.signal import savgol_filter

from evaluation.lane_continuity import (
    LaneContinuityConfig,
    extract_guidance_lanes_from_mask,
    prepare_guidance_lanes,
)


EPS = 1e-9


@dataclass(frozen=True)
class LaneWidthConfig:
    """All sampling, geometry, inference, and scoring thresholds.

    The residual tolerances are image-analysis calibration parameters.  They
    are not roadway-design or maintenance standards.
    """

    # Boundary preparation and robust fitting.
    sample_spacing_px: float = 2.0
    max_boundary_samples: int = 700
    min_source_points: int = 2
    min_boundary_support_px: float = 15.0

    # Longitudinal-orientation plausibility (image space).  Forward-facing
    # travel-lane boundaries recede toward the vanishing point and are
    # image-vertical dominant.  A boundary whose principal image axis is more
    # horizontal than this bound is lateral clutter (crosswalk / stop-line
    # stripes, wall-texture trails, guardrail edges) rather than a travel-lane
    # boundary, and is rejected before the longitudinal frame is estimated so a
    # single horizontal detection cannot skew the shared axis.  Set the flag
    # False (or the angle to 90) for genuinely top-down / non-forward cameras.
    enforce_longitudinal_orientation: bool = True
    max_boundary_axis_angle_from_vertical_deg: float = 60.0
    # Dash-fragment grouping (extract_grouped_boundaries_from_mask): two raw
    # mask fragments are merged into one logical boundary only if they are
    # compatible when compared near their own shared gap, not by extrapolating
    # both out to a fixed distant reference row (which can spuriously agree by
    # coincidence for two unrelated fragments).
    max_group_gap_px: float = 140.0
    group_lateral_gap_fraction: float = 0.055
    group_slope_mismatch_tolerance: float = 0.45
    fit_degree: int = 3
    fit_irls_iterations: int = 8
    fit_huber_delta: float = 1.5
    fit_floor_px: float = 1.5
    fit_floor_metric: float = 0.04
    fit_inlier_sigma: float = 3.5
    max_fit_residual_px: float = 8.0
    max_fit_residual_metric: float = 0.35

    # Calibration and conservative fallback.
    max_homography_condition: float = 1.0e9
    min_projected_corner_area: float = 1.0e-8
    fallback_min_y_fraction: float = 0.36
    fallback_max_y_fraction: float = 0.985
    image_boundary_margin_px: float = 4.0
    max_resolution_ratio: float = 4.5
    min_resolution_reliability: float = 0.08
    fallback_measurement_confidence: float = 0.25
    calibrated_scale_free_measurement_confidence: float = 0.55
    calibrated_metric_measurement_confidence: float = 0.95
    calibrated_metric_scale_confidence: float = 0.95

    # Double-line identification.
    double_line_max_lane_spacing_fraction: float = 0.24
    double_line_max_separation_metric: float = 0.75
    double_line_max_separation_image_fraction: float = 0.045
    double_line_min_overlap_fraction: float = 0.62
    double_line_max_tangent_angle_deg: float = 8.0

    # Adjacent-boundary pairing.
    min_pair_overlap_fraction: float = 0.42
    min_pair_support_px: float = 48.0
    min_pair_support_metric: float = 4.0
    min_pair_width_metric: float = 0.65
    max_pair_width_metric: float = 12.0
    min_pair_width_image_fraction: float = 0.025
    max_pair_width_image_fraction: float = 0.70
    max_pair_tangent_angle_deg: float = 38.0
    min_drivable_support: float = 0.35
    pair_check_samples: int = 31
    # Travel-lane corridor plausibility.  A real lane extends farther along the
    # travel direction than it is wide, its two boundaries stay roughly
    # parallel, and its apparent width does not swing beyond a single lane plus
    # a plausible taper/merge.  These reject outline-tracing detections
    # (crosswalk / stop-line boxes) and boundaries that leave the road surface
    # (a line climbing a wall relative to the other) instead of scoring nonsense
    # geometry as a confident width.  ``max_pair_separation_shape_residual`` is
    # the robust residual of the raw boundary separation around its best smooth
    # {constant, linear, single-change-point} model, so smooth intentional
    # tapers and merges pass while erratic, non-parallel geometry is rejected.
    min_pair_corridor_aspect_ratio: float = 1.1
    max_pair_relative_separation_change: float = 1.0
    max_pair_separation_shape_residual: float = 0.18

    # Orthogonal profile sampling and evidence sufficiency.
    sample_spacing_metric: float = 0.75
    min_width_samples: int = 12
    max_width_samples: int = 160
    min_valid_sample_fraction: float = 0.36
    min_sample_reliability: float = 0.08
    support_edge_fraction: float = 0.08
    max_normal_intersection_distance_factor: float = 1.8
    max_intersection_tangent_angle_deg: float = 42.0

    # Robust expected-profile inference.
    profile_irls_iterations: int = 10
    profile_huber_delta_relative: float = 0.045
    min_model_score_improvement: float = 2.5
    min_piecewise_score_improvement: float = 3.5
    min_nonconstant_relative_change: float = 0.055
    min_transition_relative_change: float = 0.12
    min_piecewise_simple_model_p90: float = 0.012
    max_plausible_relative_change: float = 1.40
    change_point_min_fraction: float = 0.24
    change_point_max_fraction: float = 0.76
    change_point_candidates: int = 15
    min_segment_samples: int = 6
    irregular_relative_p90: float = 0.18
    irregular_stability_threshold: float = 0.50

    # Residual stability and expected-profile constancy.
    stability_lambda: float = 0.60
    tau_relative_mad: float = 0.045
    tau_relative_p90: float = 0.12
    constancy_relative_change_tolerance: float = 0.36

    # Confidence/support-aware multi-pair aggregation.
    aggregation_support_cap_ratio: float = 2.0
    aggregation_median_weight: float = 0.65


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


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_points(lane: Any) -> np.ndarray:
    """Parse common polyline containers for diagnostics before shared prep."""

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


def extract_grouped_boundaries_from_mask(
    mask: Any,
    *,
    config: Optional[LaneWidthConfig] = None,
) -> list[np.ndarray]:
    """Return grouped marking centerlines using the shared I1 mask geometry.

    The shared extractor closes a disposable geometry copy to associate aligned
    dash components, then returns row-wise centerlines.  Morphological bridging
    is never treated as evidence that paint or a boundary was continuous.
    """

    cfg = config or LaneWidthConfig()
    continuity_config = LaneContinuityConfig(
        sample_spacing_px=cfg.sample_spacing_px,
        max_longitudinal_samples=cfg.max_boundary_samples,
        min_geometry_points=cfg.min_source_points,
        min_lane_support_px=cfg.min_boundary_support_px,
        mask_close_height_fraction=0.24,
        mask_close_width_fraction=0.025,
    )
    fragments = extract_guidance_lanes_from_mask(mask, config=continuity_config)
    if len(fragments) <= 2:
        return fragments
    height, width = np.asarray(mask).shape[:2]

    items = []
    for fragment in fragments:
        degree = min(2, fragment.shape[0] - 1)
        coefficients = np.polyfit(fragment[:, 1], fragment[:, 0], degree)
        derivative = np.polyder(coefficients)
        items.append(
            {
                "fragment": fragment,
                "coeff": coefficients,
                "deriv": derivative,
                "y_min": float(fragment[:, 1].min()),
                "y_max": float(fragment[:, 1].max()),
            }
        )
    items.sort(key=lambda item: item["y_min"])

    groups: list[list[dict]] = []
    for item in items:
        selected = None
        for group in groups:
            # Compare against the group's nearest member (by y-range gap) at a
            # reference row close to the actual shared gap, so neither fit is
            # trusted far beyond its own observed support. Extrapolating both
            # fragments out to one fixed distant row (e.g. the image bottom)
            # can make two unrelated fragments agree purely by coincidence.
            nearest, gap = None, None
            for member in group:
                if item["y_min"] > member["y_max"]:
                    candidate_gap = item["y_min"] - member["y_max"]
                elif member["y_min"] > item["y_max"]:
                    candidate_gap = member["y_min"] - item["y_max"]
                else:
                    candidate_gap = 0.0
                if gap is None or candidate_gap < gap:
                    gap, nearest = candidate_gap, member
            if nearest is None or gap > cfg.max_group_gap_px:
                continue
            if item["y_min"] > nearest["y_max"]:
                reference_y = 0.5 * (nearest["y_max"] + item["y_min"])
            elif nearest["y_min"] > item["y_max"]:
                reference_y = 0.5 * (item["y_max"] + nearest["y_min"])
            else:
                reference_y = 0.5 * (
                    max(item["y_min"], nearest["y_min"]) + min(item["y_max"], nearest["y_max"])
                )
            existing_x = float(np.polyval(nearest["coeff"], reference_y))
            candidate_x = float(np.polyval(item["coeff"], reference_y))
            existing_slope = float(np.polyval(nearest["deriv"], reference_y))
            candidate_slope = float(np.polyval(item["deriv"], reference_y))
            if (
                abs(existing_x - candidate_x) <= cfg.group_lateral_gap_fraction * width
                and abs(existing_slope - candidate_slope) <= cfg.group_slope_mismatch_tolerance
            ):
                selected = group
                break
        if selected is None:
            groups.append([item])
        else:
            selected.append(item)

    grouped = []
    for group in groups:
        points = np.vstack([member["fragment"] for member in group])
        rows = np.round(points[:, 1]).astype(int)
        centerline = [
            [float(np.median(points[rows == row, 0])), float(row)]
            for row in sorted(set(rows.tolist()))
        ]
        if len(centerline) >= 2:
            grouped.append(np.asarray(centerline, dtype=np.float64))
    grouped.sort(key=lambda lane: float(np.mean(lane[:, 0])))
    return grouped


def _camera_homography(camera_model: Any) -> tuple[Optional[np.ndarray], Optional[str]]:
    if not isinstance(camera_model, Mapping):
        return None, None
    for key in ("image_to_road_homography", "image_to_ground_homography", "homography", "ipm"):
        if key in camera_model:
            try:
                matrix = np.asarray(camera_model[key], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if matrix.shape == (3, 3):
                return matrix, f"camera_model.{key}"
    try:
        intrinsics = np.asarray(camera_model.get("intrinsics"), dtype=np.float64)
        rotation = np.asarray(camera_model.get("rotation"), dtype=np.float64)
        translation = np.asarray(camera_model.get("translation"), dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None, None
    if intrinsics.shape != (3, 3) or rotation.shape != (3, 3):
        return None, None
    ground_to_image = intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    try:
        return np.linalg.inv(ground_to_image), "camera_model.intrinsics_extrinsics"
    except np.linalg.LinAlgError:
        return None, None


def determine_measurement_space(
    *,
    image_shape: Sequence[int],
    homography: Any = None,
    camera_model: Any = None,
    calibration_meta: Any = None,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Validate calibration and choose an explicit measurement space."""

    cfg = config or LaneWidthConfig()
    height, width = int(image_shape[0]), int(image_shape[1])
    source = None
    matrix = None
    if homography is not None:
        try:
            candidate = np.asarray(homography, dtype=np.float64)
        except (TypeError, ValueError):
            candidate = np.empty((0, 0))
        if candidate.shape == (3, 3):
            matrix = candidate
            source = "explicit_image_to_road_homography"
    if matrix is None:
        matrix, source = _camera_homography(camera_model)

    meta = dict(calibration_meta) if isinstance(calibration_meta, Mapping) else {}
    if isinstance(camera_model, Mapping):
        for key in ("metric_scale_available", "scale_known", "units", "road_plane_units", "homography_source"):
            if key not in meta and key in camera_model:
                meta[key] = camera_model[key]
    fallback_reason = None
    condition = None
    projected_area = None
    inverse = None
    if matrix is not None:
        if not np.isfinite(matrix).all():
            fallback_reason = "homography_contains_nonfinite_values"
            matrix = None
        else:
            scale = float(np.linalg.norm(matrix))
            normalized = matrix / max(scale, EPS)
            condition = float(np.linalg.cond(normalized))
            if not np.isfinite(condition) or condition > cfg.max_homography_condition:
                fallback_reason = "homography_ill_conditioned"
                matrix = None
            else:
                try:
                    inverse = np.linalg.inv(matrix)
                except np.linalg.LinAlgError:
                    fallback_reason = "homography_singular"
                    matrix = None
                if matrix is not None:
                    corners = np.asarray(
                        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
                        dtype=np.float64,
                    )
                    projected, valid = _project_points(corners, matrix)
                    if np.count_nonzero(valid) == 4:
                        projected_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
                    if projected_area is None or projected_area < cfg.min_projected_corner_area:
                        fallback_reason = "homography_has_negligible_valid_area"
                        matrix = None
                        inverse = None

    if matrix is None:
        if fallback_reason is None:
            fallback_reason = "calibrated_road_plane_transform_unavailable"
        return {
            "measurement_space": "projective_normalized",
            "homography": None,
            "inverse_homography": None,
            "homography_source": source,
            "homography_condition_number": condition,
            "valid_transformed_area": projected_area,
            "metric_scale_available": False,
            "metric_scale_confidence": 0.0,
            "measurement_confidence": cfg.fallback_measurement_confidence,
            "width_units": "image_width_fraction",
            "far_field_exclusion": f"image_y < {cfg.fallback_min_y_fraction:.3f} * image_height",
            "fallback_reason": fallback_reason,
        }

    scale_declared = meta.get("metric_scale_available") is True or meta.get("scale_known") is True
    units = str(meta.get("units", meta.get("road_plane_units", "m"))).strip().lower()
    metric_scale = scale_declared and units in {"m", "meter", "meters", "metre", "metres"}
    if not metric_scale:
        space = "calibrated_scale_free_bev"
        width_units = "road_plane_units"
        metric_scale_confidence = 0.0
        measurement_confidence = cfg.calibrated_scale_free_measurement_confidence
    else:
        space = "calibrated_metric_bev"
        width_units = "m"
        metric_scale_confidence = cfg.calibrated_metric_scale_confidence
        measurement_confidence = cfg.calibrated_metric_measurement_confidence
    return {
        "measurement_space": space,
        "homography": matrix,
        "inverse_homography": inverse,
        "homography_source": str(meta.get("homography_source", source)),
        "homography_condition_number": condition,
        "valid_transformed_area": projected_area,
        "metric_scale_available": bool(metric_scale),
        "metric_scale_confidence": metric_scale_confidence,
        "measurement_confidence": measurement_confidence,
        "width_units": width_units,
        "far_field_exclusion": "transform-resolution reliability gating",
        "fallback_reason": None,
    }


def _project_points(points: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points[:, :2], np.ones(points.shape[0])])
    projected_h = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    denominator = projected_h[:, 2]
    valid = np.isfinite(projected_h).all(axis=1) & (np.abs(denominator) > 1e-10)
    projected = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    projected[valid] = projected_h[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _shared_prepare(
    lane_geometries: Optional[Iterable[Any]],
    image_shape: Sequence[int],
    cfg: LaneWidthConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    continuity_config = LaneContinuityConfig(
        sample_spacing_px=cfg.sample_spacing_px,
        max_longitudinal_samples=cfg.max_boundary_samples,
        min_geometry_points=cfg.min_source_points,
        min_lane_support_px=cfg.min_boundary_support_px,
        preserve_local_geometry=True,
    )
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for boundary_id, lane in enumerate(lane_geometries or []):
        source_points = _as_points(lane)
        result = prepare_guidance_lanes([lane], image_shape, config=continuity_config)
        if not result:
            rejected.append({"boundary_id": boundary_id, "reason": "invalid_or_insufficient_boundary_geometry"})
            continue
        boundary = dict(result[0])
        boundary.update(
            {
                "boundary_id": boundary_id,
                "source_boundary_ids": [boundary_id],
                "source_point_count": int(source_points.shape[0]),
                "input_points": source_points,
                "collapsed_double_line": False,
            }
        )
        prepared.append(boundary)
    return prepared, rejected


def _local_tangent_angle_from_vertical_deg(ordered_points: np.ndarray) -> float:
    """Robust median local-tangent angle (deg) of an ordered boundary path.

    Zero means the boundary locally advances purely longitudinally (image- or
    road-plane vertical); ninety means it locally advances purely laterally
    (horizontal).  This is computed from consecutive-sample step vectors along
    the boundary's own path order, not from the aggregate point scatter: a
    boundary that curves smoothly or sweeps wide under perspective (e.g. a
    median edge or curb near the frame edge) can have a net start-to-end
    displacement, or overall bounding-box aspect, that looks "wide," even
    though every local step still advances mostly forward. Only a boundary
    whose steps are *themselves* predominantly lateral -- a crosswalk stripe,
    stop-line, or other horizontal clutter -- should score high here.
    """

    points = np.asarray(ordered_points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    keep = lengths > EPS
    if not np.any(keep):
        return 0.0
    deltas = deltas[keep]
    angles = np.degrees(np.arctan2(np.abs(deltas[:, 0]), np.abs(deltas[:, 1])))
    return float(np.median(angles))


def _separation_shape_residual(stations: np.ndarray, separation: np.ndarray) -> float:
    """Robust relative residual of a separation profile around a smooth model.

    Fits constant, linear, and single-change-point continuous piecewise-linear
    models to the raw boundary separation and returns the smallest robust MAD
    residual, normalised by the median separation.  A smooth taper, merge, or
    curved-but-parallel lane yields a small value; an outline or a boundary that
    diverges erratically from its partner yields a large one.
    """

    s = np.asarray(stations, dtype=np.float64)
    sep = np.asarray(separation, dtype=np.float64)
    if s.size < 4:
        return 0.0
    median = max(float(np.median(sep)), EPS)
    z = (s - float(np.min(s))) / max(float(np.ptp(s)), EPS)
    designs = [np.ones((s.size, 1)), np.column_stack([np.ones(s.size), z])]
    for change in np.linspace(0.3, 0.7, 5):
        if np.count_nonzero(z <= change) < 2 or np.count_nonzero(z > change) < 2:
            continue
        designs.append(np.column_stack([np.ones(s.size), z, np.maximum(z - change, 0.0)]))
    best = float("inf")
    for design in designs:
        coefficients, *_ = np.linalg.lstsq(design, sep, rcond=None)
        residual = sep - design @ coefficients
        robust = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        best = min(best, robust)
    return float(best / median)


def _mean_direction(point_sets: Sequence[np.ndarray]) -> np.ndarray:
    directions = []
    reference = None
    for points in point_sets:
        if points.shape[0] < 2:
            continue
        vector = points[-1] - points[0]
        norm = float(np.linalg.norm(vector))
        if norm <= EPS:
            continue
        vector = vector / norm
        if reference is None:
            reference = vector
        elif float(np.dot(vector, reference)) < 0:
            vector = -vector
        directions.append(vector)
    if not directions:
        return np.asarray([0.0, 1.0])
    direction = np.mean(np.asarray(directions), axis=0)
    norm = float(np.linalg.norm(direction))
    return direction / max(norm, EPS)


def _huber_polynomial_fit(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    *,
    degree: int,
    floor: float,
    cfg: LaneWidthConfig,
) -> dict[str, Any]:
    t_min, t_max = float(np.min(longitudinal)), float(np.max(longitudinal))
    center = 0.5 * (t_min + t_max)
    half_span = max(0.5 * (t_max - t_min), EPS)
    z = (longitudinal - center) / half_span
    degree = int(min(max(degree, 1), max(longitudinal.size - 2, 1)))
    design = np.column_stack([z ** power for power in range(degree + 1)])
    weights = np.ones(longitudinal.size, dtype=np.float64)
    coefficients = np.linalg.lstsq(design, lateral, rcond=None)[0]
    for _ in range(cfg.fit_irls_iterations):
        residual = lateral - design @ coefficients
        median = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - median)))
        scale = max(mad, floor)
        normalized = np.abs(residual - median) / max(cfg.fit_huber_delta * scale, EPS)
        weights = np.ones_like(normalized)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
        root = np.sqrt(weights)
        try:
            updated = np.linalg.lstsq(design * root[:, None], lateral * root, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if float(np.linalg.norm(updated - coefficients)) < 1e-9:
            coefficients = updated
            break
        coefficients = updated
    residual = lateral - design @ coefficients
    residual_median = float(np.median(residual))
    residual_mad = 1.4826 * float(np.median(np.abs(residual - residual_median)))
    inlier_threshold = max(cfg.fit_inlier_sigma * max(residual_mad, floor), floor * 2.0)
    inliers = np.abs(residual - residual_median) <= inlier_threshold
    robust_residual = 1.4826 * float(np.median(np.abs(residual[inliers] - np.median(residual[inliers])))) if np.any(inliers) else None
    return {
        "coefficients": coefficients,
        "degree": degree,
        "center": center,
        "half_span": half_span,
        "t_min": t_min,
        "t_max": t_max,
        "inliers": inliers,
        "weights": weights,
        "robust_residual": robust_residual,
    }


def _locally_smoothed_curve(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    *,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppress isolated detector spikes while retaining supported local shape."""

    order = np.argsort(longitudinal)
    t = np.asarray(longitudinal[order], dtype=np.float64)
    values = np.asarray(lateral[order], dtype=np.float64)
    unique_t, unique_indices = np.unique(np.round(t, 9), return_index=True)
    t, values = t[unique_indices], values[unique_indices]
    if t.size < 5:
        return t, values
    window = min(9, t.size if t.size % 2 else t.size - 1)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    local_median = np.asarray([np.median(padded[index : index + window]) for index in range(values.size)])
    local_residual = values - local_median
    scale = 1.4826 * float(np.median(np.abs(local_residual - np.median(local_residual))))
    spike_threshold = max(7.0 * scale, 4.0 * floor)
    cleaned = values.copy()
    spikes = np.abs(local_residual) > spike_threshold
    cleaned[spikes] = local_median[spikes]
    smooth_window = min(11, cleaned.size if cleaned.size % 2 else cleaned.size - 1)
    if smooth_window >= 5:
        cleaned = savgol_filter(cleaned, smooth_window, min(3, smooth_window - 2), mode="interp")
    return t, cleaned


def _eval_fit(boundary: Mapping[str, Any], longitudinal: np.ndarray | float) -> np.ndarray:
    t = np.asarray(longitudinal, dtype=np.float64)
    if "curve_t" in boundary:
        return np.interp(t, np.asarray(boundary["curve_t"]), np.asarray(boundary["curve_lateral"]))
    z = (t - float(boundary["fit_center"])) / float(boundary["fit_half_span"])
    coefficients = np.asarray(boundary["coefficients"], dtype=np.float64)
    return sum(coefficients[power] * z ** power for power in range(coefficients.size))


def _eval_fit_derivative(boundary: Mapping[str, Any], longitudinal: np.ndarray | float) -> np.ndarray:
    t = np.asarray(longitudinal, dtype=np.float64)
    if "curve_t" in boundary:
        curve_t = np.asarray(boundary["curve_t"], dtype=np.float64)
        curve_lateral = np.asarray(boundary["curve_lateral"], dtype=np.float64)
        derivative = np.gradient(curve_lateral, curve_t)
        return np.interp(t, curve_t, derivative)
    z = (t - float(boundary["fit_center"])) / float(boundary["fit_half_span"])
    coefficients = np.asarray(boundary["coefficients"], dtype=np.float64)
    result = np.zeros_like(t, dtype=np.float64)
    for power in range(1, coefficients.size):
        result += power * coefficients[power] * z ** (power - 1)
    return result / float(boundary["fit_half_span"])


def _curve_points(
    boundary: Mapping[str, Any],
    longitudinal: np.ndarray,
    origin: np.ndarray,
    long_axis: np.ndarray,
    lateral_axis: np.ndarray,
) -> np.ndarray:
    lateral = _eval_fit(boundary, longitudinal)
    return origin + longitudinal[:, None] * long_axis + lateral[:, None] * lateral_axis


def validate_and_fit_boundaries(
    lane_geometries: Optional[Iterable[Any]],
    *,
    image_shape: Sequence[int],
    measurement: Mapping[str, Any],
    lane_confidences: Any = None,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Use shared I1 validation, then robustly fit each observed boundary."""

    cfg = config or LaneWidthConfig()
    prepared, rejected = _shared_prepare(lane_geometries, image_shape, cfg)
    transform = measurement.get("homography")
    transformed_sets = []
    retained_prepared = []
    for boundary in prepared:
        image_points = np.asarray(boundary["points"], dtype=np.float64)
        if transform is None:
            work_points = image_points.copy()
            valid = np.ones(image_points.shape[0], dtype=bool)
        else:
            work_points, valid = _project_points(image_points, np.asarray(transform))
        work_points = work_points[valid]
        image_points = image_points[valid]
        if work_points.shape[0] < cfg.min_source_points:
            rejected.append({"boundary_id": boundary["boundary_id"], "reason": "boundary_invalid_after_road_plane_transform"})
            continue
        # Reject lateral clutter (near-horizontal detections such as crosswalk /
        # stop-line stripes or wall-texture trails) before it can be paired as a
        # travel-lane boundary or skew the shared longitudinal axis.  Orientation
        # is judged from local, per-step tangents along the boundary's own path
        # (in the measurement frame: road plane when calibrated, image space
        # otherwise), not from the aggregate point scatter -- a genuinely curved
        # or perspective-widened lane boundary can have a wide net/aggregate
        # extent while still advancing forward at every local step, and must not
        # be mistaken for lateral clutter whose steps are themselves horizontal.
        if cfg.enforce_longitudinal_orientation:
            axis_angle = _local_tangent_angle_from_vertical_deg(work_points)
            if axis_angle > cfg.max_boundary_axis_angle_from_vertical_deg:
                rejected.append(
                    {
                        "boundary_id": boundary["boundary_id"],
                        "reason": "boundary_orientation_not_longitudinal",
                        "axis_angle_from_vertical_deg": axis_angle,
                    }
                )
                continue
        boundary["work_points"] = work_points
        boundary["image_points"] = image_points
        transformed_sets.append(work_points)
        retained_prepared.append(boundary)
    if not transformed_sets:
        return {"boundaries": [], "rejected_boundaries": rejected, "origin": None, "long_axis": None, "lateral_axis": None}

    all_points = np.vstack(transformed_sets)
    origin = np.mean(all_points, axis=0)
    long_axis = _mean_direction(transformed_sets)
    lateral_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float64)
    if transform is None and long_axis[1] < 0:
        long_axis = -long_axis
        lateral_axis = -lateral_axis
    # Give diagnostics a stable image-left to image-right ordering even when a
    # calibrated road coordinate system uses the opposite lateral sign.
    image_x = np.concatenate([boundary["image_points"][:, 0] for boundary in retained_prepared])
    work_lateral = np.concatenate([(boundary["work_points"] - origin) @ lateral_axis for boundary in retained_prepared])
    if image_x.size >= 2 and float(np.std(image_x)) > EPS and float(np.std(work_lateral)) > EPS:
        if float(np.corrcoef(image_x, work_lateral)[0, 1]) < 0:
            lateral_axis = -lateral_axis

    confidence_values: dict[int, float] = {}
    if isinstance(lane_confidences, Mapping):
        confidence_values = {int(key): float(value) for key, value in lane_confidences.items() if _finite_float(value) is not None}
    elif lane_confidences is not None:
        for index, value in enumerate(lane_confidences):
            if _finite_float(value) is not None:
                confidence_values[index] = float(value)

    fitted = []
    space = str(measurement.get("measurement_space", ""))
    metric_space = space == "calibrated_metric_bev"
    scale_free_space = space == "calibrated_scale_free_bev"
    coordinate_span = max(float(np.ptp(all_points[:, 0])), float(np.ptp(all_points[:, 1])), EPS)
    if metric_space:
        floor = cfg.fit_floor_metric
        residual_limit = cfg.max_fit_residual_metric
    elif scale_free_space:
        floor = max(coordinate_span * 0.001, EPS)
        residual_limit = max(coordinate_span * 0.02, floor * 6.0)
    else:
        floor = cfg.fit_floor_px
        residual_limit = cfg.max_fit_residual_px
    for boundary in retained_prepared:
        centered = boundary["work_points"] - origin
        longitudinal = centered @ long_axis
        lateral = centered @ lateral_axis
        if float(np.ptp(longitudinal)) <= EPS:
            rejected.append({"boundary_id": boundary["boundary_id"], "reason": "boundary_has_no_longitudinal_support"})
            continue
        order = np.argsort(longitudinal)
        longitudinal, lateral = longitudinal[order], lateral[order]
        image_points = boundary["image_points"][order]
        fit = _huber_polynomial_fit(
            longitudinal,
            lateral,
            degree=cfg.fit_degree,
            floor=floor,
            cfg=cfg,
        )
        residual = fit["robust_residual"]
        if residual is None or residual > residual_limit:
            rejected.append({"boundary_id": boundary["boundary_id"], "reason": "boundary_fit_residual_too_large", "robust_residual": residual})
            continue
        inlier_fraction = float(
            min(
                np.mean(fit["inliers"]),
                float(boundary.get("geometry_quality", 1.0)),
            )
        )
        curve_t, curve_lateral = _locally_smoothed_curve(
            longitudinal,
            lateral,
            floor=floor,
        )
        residual_score = float(np.exp(-residual / max(residual_limit * 0.45, EPS)))
        input_confidence = float(np.clip(confidence_values.get(boundary["boundary_id"], 1.0), 0.0, 1.0))
        geometry_confidence = float(
            np.clip(
                0.45 * inlier_fraction
                + 0.25 * float(boundary.get("geometry_quality", 1.0))
                + 0.20 * residual_score
                + 0.10 * input_confidence,
                0.0,
                1.0,
            )
        )
        fitted.append(
            {
                "boundary_id": int(boundary["boundary_id"]),
                "source_boundary_ids": list(boundary["source_boundary_ids"]),
                "source_point_count": int(boundary["source_point_count"]),
                "inlier_count": int(round(boundary["source_point_count"] * inlier_fraction)),
                "inlier_fraction": inlier_fraction,
                "fitted_support_interval": [fit["t_min"], fit["t_max"]],
                "t_min": fit["t_min"],
                "t_max": fit["t_max"],
                "fit_center": fit["center"],
                "fit_half_span": fit["half_span"],
                "coefficients": fit["coefficients"],
                "curve_t": curve_t,
                "curve_lateral": curve_lateral,
                "fit_degree": fit["degree"],
                "robust_residual": residual,
                "extrapolated_fraction": 0.0,
                "geometry_confidence": geometry_confidence,
                "collapsed_double_line": False,
                "image_points": image_points,
                "work_points": boundary["work_points"][order],
            }
        )
    return {
        "boundaries": fitted,
        "rejected_boundaries": rejected,
        "origin": origin,
        "long_axis": long_axis,
        "lateral_axis": lateral_axis,
    }


def _common_interval(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[float, float, float]:
    start = max(float(left["t_min"]), float(right["t_min"]))
    end = min(float(left["t_max"]), float(right["t_max"]))
    overlap = max(end - start, 0.0)
    shorter = max(min(float(left["t_max"]) - float(left["t_min"]), float(right["t_max"]) - float(right["t_min"])), EPS)
    return start, end, float(overlap / shorter)


def _tangent_angle_degrees(left: Mapping[str, Any], right: Mapping[str, Any], stations: np.ndarray) -> float:
    left_slope = _eval_fit_derivative(left, stations)
    right_slope = _eval_fit_derivative(right, stations)
    return float(np.degrees(np.median(np.abs(np.arctan(left_slope) - np.arctan(right_slope)))))


def _sort_boundaries(boundaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not boundaries:
        return []
    reference = float(np.median([0.5 * (float(item["t_min"]) + float(item["t_max"])) for item in boundaries]))
    return sorted(
        boundaries,
        key=lambda item: float(_eval_fit(item, np.clip(reference, item["t_min"], item["t_max"]))),
    )


def _fit_logical_boundary(
    points: np.ndarray,
    *,
    boundary_id: str,
    source_ids: list[Any],
    geometry_confidence: float,
    residual: float,
    cfg: LaneWidthConfig,
    fit_floor: float,
) -> dict[str, Any]:
    fit = _huber_polynomial_fit(points[:, 0], points[:, 1], degree=cfg.fit_degree, floor=fit_floor, cfg=cfg)
    return {
        "boundary_id": boundary_id,
        "source_boundary_ids": source_ids,
        "source_point_count": int(points.shape[0]),
        "inlier_count": int(np.count_nonzero(fit["inliers"])),
        "inlier_fraction": float(np.mean(fit["inliers"])),
        "fitted_support_interval": [fit["t_min"], fit["t_max"]],
        "t_min": fit["t_min"],
        "t_max": fit["t_max"],
        "fit_center": fit["center"],
        "fit_half_span": fit["half_span"],
        "coefficients": fit["coefficients"],
        "fit_degree": fit["degree"],
        "robust_residual": residual,
        "extrapolated_fraction": 0.0,
        "geometry_confidence": geometry_confidence,
        "collapsed_double_line": True,
        "image_points": np.empty((0, 2), dtype=np.float64),
        "work_points": np.empty((0, 2), dtype=np.float64),
    }


def collapse_double_line_boundaries(
    boundaries: Sequence[dict[str, Any]],
    *,
    image_shape: Sequence[int],
    measurement_space: str,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Collapse close parallel marking stripes into one logical boundary."""

    cfg = config or LaneWidthConfig()
    ordered = _sort_boundaries(boundaries)
    if len(ordered) < 2:
        return {"logical_boundaries": list(ordered), "double_line_groups": []}
    reference = float(np.median([0.5 * (b["t_min"] + b["t_max"]) for b in ordered]))
    lateral_values = np.asarray(
        [float(_eval_fit(boundary, np.clip(reference, boundary["t_min"], boundary["t_max"]))) for boundary in ordered]
    )
    positive_gaps = np.diff(lateral_values)
    positive_gaps = positive_gaps[positive_gaps > EPS]
    spacing_reference = float(np.percentile(positive_gaps, 75)) if positive_gaps.size else 0.0
    if positive_gaps.size >= 3:
        spacing_reference = max(spacing_reference, float(np.median(positive_gaps)) * 1.4)
    metric_space = measurement_space == "calibrated_metric_bev"
    scale_free_space = measurement_space == "calibrated_scale_free_bev"
    if metric_space:
        absolute_limit = cfg.double_line_max_separation_metric
        fit_floor = cfg.fit_floor_metric
    elif scale_free_space:
        typical_support = float(np.median([boundary["t_max"] - boundary["t_min"] for boundary in ordered]))
        absolute_limit = max(0.04 * typical_support, EPS)
        fit_floor = max(typical_support * 0.001, EPS)
    else:
        absolute_limit = cfg.double_line_max_separation_image_fraction * float(image_shape[1])
        fit_floor = cfg.fit_floor_px
    logical: list[dict[str, Any]] = []
    groups = []
    index = 0
    while index < len(ordered):
        if index + 1 >= len(ordered):
            logical.append(ordered[index])
            break
        first, second = ordered[index], ordered[index + 1]
        start, end, overlap_fraction = _common_interval(first, second)
        if end > start:
            stations = np.linspace(start, end, 31)
            separation = float(np.median(_eval_fit(second, stations) - _eval_fit(first, stations)))
            tangent_angle = _tangent_angle_degrees(first, second, stations)
        else:
            separation = float("inf")
            tangent_angle = float("inf")
        scale_close = spacing_reference <= EPS or separation <= cfg.double_line_max_lane_spacing_fraction * spacing_reference
        is_double = (
            separation > 0
            and separation <= absolute_limit
            and scale_close
            and overlap_fraction >= cfg.double_line_min_overlap_fraction
            and tangent_angle <= cfg.double_line_max_tangent_angle_deg
        )
        if not is_double:
            logical.append(first)
            index += 1
            continue
        stations = np.linspace(start, end, max(32, min(120, int((end - start) / max(fit_floor, 0.1)))))
        center_lateral = 0.5 * (_eval_fit(first, stations) + _eval_fit(second, stations))
        source_ids = list(first["source_boundary_ids"]) + list(second["source_boundary_ids"])
        collapsed = _fit_logical_boundary(
            np.column_stack([stations, center_lateral]),
            boundary_id=f"double_{'_'.join(str(item) for item in source_ids)}",
            source_ids=source_ids,
            geometry_confidence=float(min(first["geometry_confidence"], second["geometry_confidence"]) * 0.95),
            residual=float(max(first["robust_residual"], second["robust_residual"])),
            cfg=cfg,
            fit_floor=fit_floor,
        )
        logical.append(collapsed)
        groups.append(
            {
                "logical_boundary_id": collapsed["boundary_id"],
                "source_boundary_ids": source_ids,
                "median_separation": separation if (metric_space or scale_free_space) else separation / max(float(image_shape[1]), 1.0),
                "separation_units": "road_plane_units" if (metric_space or scale_free_space) else "image_width_fraction",
                "overlap_fraction": overlap_fraction,
                "median_tangent_difference_deg": tangent_angle,
            }
        )
        index += 2
    return {"logical_boundaries": _sort_boundaries(logical), "double_line_groups": groups}


def _resize_binary(mask: Any, shape: tuple[int, int]) -> Optional[np.ndarray]:
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


def _work_to_image(points: np.ndarray, measurement: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    inverse = measurement.get("inverse_homography")
    if inverse is None:
        points = np.asarray(points, dtype=np.float64)
        return points.copy(), np.isfinite(points).all(axis=1)
    return _project_points(points, np.asarray(inverse, dtype=np.float64))


def pair_adjacent_boundaries(
    boundaries: Sequence[dict[str, Any]],
    *,
    image_shape: Sequence[int],
    measurement: Mapping[str, Any],
    origin: np.ndarray,
    long_axis: np.ndarray,
    lateral_axis: np.ndarray,
    road_mask: Any = None,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Validate only neighboring logical boundaries as travel-lane pairs."""

    cfg = config or LaneWidthConfig()
    ordered = _sort_boundaries(boundaries)
    accepted: list[dict[str, Any]] = []
    rejected = []
    height, width = int(image_shape[0]), int(image_shape[1])
    road = _resize_binary(road_mask, (height, width))
    space = str(measurement["measurement_space"])
    metric_space = space == "calibrated_metric_bev"
    scale_free_space = space == "calibrated_scale_free_bev"
    if metric_space:
        min_width = cfg.min_pair_width_metric
        max_width = cfg.max_pair_width_metric
        min_support = cfg.min_pair_support_metric
    elif scale_free_space:
        reference = float(np.median([0.5 * (item["t_min"] + item["t_max"]) for item in ordered]))
        reference_lateral = np.asarray(
            [float(_eval_fit(item, np.clip(reference, item["t_min"], item["t_max"]))) for item in ordered]
        )
        typical_gap = float(np.median(np.diff(reference_lateral))) if len(ordered) > 1 else 1.0
        min_width = max(0.12 * typical_gap, EPS)
        max_width = max(6.0 * typical_gap, min_width * 2.0)
        min_support = 0.0
    else:
        min_width = cfg.min_pair_width_image_fraction * width
        max_width = cfg.max_pair_width_image_fraction * width
        min_support = cfg.min_pair_support_px
    for index, (left, right) in enumerate(zip(ordered[:-1], ordered[1:])):
        start, end, overlap_fraction = _common_interval(left, right)
        diagnostic = {
            "candidate_index": index,
            "left_boundary_id": left["boundary_id"],
            "right_boundary_id": right["boundary_id"],
            "left_source_boundary_ids": left["source_boundary_ids"],
            "right_source_boundary_ids": right["source_boundary_ids"],
            "overlap_fraction": overlap_fraction,
        }
        if end <= start or overlap_fraction < cfg.min_pair_overlap_fraction:
            diagnostic["reason"] = "insufficient_overlapping_support"
            rejected.append(diagnostic)
            continue
        if end - start < min_support:
            diagnostic["reason"] = "overlap_support_too_short"
            rejected.append(diagnostic)
            continue
        stations = np.linspace(start, end, cfg.pair_check_samples)
        left_lateral = _eval_fit(left, stations)
        right_lateral = _eval_fit(right, stations)
        separation = right_lateral - left_lateral
        diagnostic["minimum_candidate_separation"] = float(np.min(separation))
        diagnostic["maximum_candidate_separation"] = float(np.max(separation))
        if np.any(separation <= 0):
            diagnostic["reason"] = "boundary_crossing_or_reversed_order"
            rejected.append(diagnostic)
            continue
        median_separation = float(np.median(separation))
        if median_separation < min_width:
            diagnostic["reason"] = "implausibly_narrow_lane_region"
            rejected.append(diagnostic)
            continue
        if median_separation > max_width:
            diagnostic["reason"] = "implausibly_wide_lane_region"
            rejected.append(diagnostic)
            continue
        tangent_angle = _tangent_angle_degrees(left, right, stations)
        diagnostic["median_tangent_difference_deg"] = tangent_angle
        if tangent_angle > cfg.max_pair_tangent_angle_deg:
            diagnostic["reason"] = "relative_tangent_incompatible"
            rejected.append(diagnostic)
            continue
        # Travel-lane corridor plausibility: the corridor must be longer than it
        # is wide, its apparent width must not swing beyond a single lane plus a
        # plausible transition, and the two boundaries must stay parallel enough
        # that their separation is explained by a smooth model.  This rejects
        # outline-tracing (crosswalk / stop-line box) detections and boundaries
        # that leave the road surface, rather than scoring nonsense geometry.
        corridor_aspect_ratio = float((end - start) / max(median_separation, EPS))
        diagnostic["corridor_aspect_ratio"] = corridor_aspect_ratio
        if corridor_aspect_ratio < cfg.min_pair_corridor_aspect_ratio:
            diagnostic["reason"] = "implausible_corridor_aspect_ratio"
            rejected.append(diagnostic)
            continue
        relative_separation_change = float(np.ptp(separation) / max(median_separation, EPS))
        diagnostic["relative_separation_change"] = relative_separation_change
        if relative_separation_change > cfg.max_pair_relative_separation_change:
            diagnostic["reason"] = "implausible_lane_width_variation"
            rejected.append(diagnostic)
            continue
        separation_shape_residual = _separation_shape_residual(stations, separation)
        diagnostic["separation_shape_residual"] = separation_shape_residual
        if separation_shape_residual > cfg.max_pair_separation_shape_residual:
            diagnostic["reason"] = "boundaries_not_consistently_parallel"
            rejected.append(diagnostic)
            continue
        center_lateral = 0.5 * (left_lateral + right_lateral)
        center_work = origin + stations[:, None] * long_axis + center_lateral[:, None] * lateral_axis
        center_image, image_valid = _work_to_image(center_work, measurement)
        drivable_support = None
        if road is not None:
            rounded = np.round(center_image).astype(np.int64)
            inside = image_valid & (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
            support_values = np.zeros(stations.size, dtype=np.float64)
            support_values[inside] = road[rounded[inside, 1], rounded[inside, 0]]
            drivable_support = float(np.mean(support_values[inside])) if np.any(inside) else 0.0
            if drivable_support < cfg.min_drivable_support:
                diagnostic["reason"] = "lane_region_lacks_drivable_area_support"
                diagnostic["drivable_support"] = drivable_support
                rejected.append(diagnostic)
                continue
        diagnostic.update(
            {
                "pair_index": len(accepted),
                "left": left,
                "right": right,
                "support_interval": [start, end],
                "support_length": float(end - start),
                "drivable_support": drivable_support,
            }
        )
        accepted.append(diagnostic)
    return {"accepted_pairs": accepted, "rejected_pairs": rejected}


def _uniform_centerline_stations(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    start: float,
    end: float,
    *,
    origin: np.ndarray,
    long_axis: np.ndarray,
    lateral_axis: np.ndarray,
    spacing: float,
    maximum: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_t = np.linspace(start, end, max(240, minimum * 8))
    dense_lateral = 0.5 * (_eval_fit(left, dense_t) + _eval_fit(right, dense_t))
    dense_points = origin + dense_t[:, None] * long_axis + dense_lateral[:, None] * lateral_axis
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense_points, axis=0), axis=1))])
    support = float(arc[-1])
    count = int(np.clip(np.floor(support / max(spacing, EPS)) + 1, minimum, maximum))
    target_arc = np.linspace(0.0, support, count)
    stations = np.interp(target_arc, arc, dense_t)
    lateral = 0.5 * (_eval_fit(left, stations) + _eval_fit(right, stations))
    points = origin + stations[:, None] * long_axis + lateral[:, None] * lateral_axis
    return stations, points, target_arc


def construct_lane_centerlines(
    lane_geometries: Optional[Iterable[Any]],
    *,
    image_shape: Sequence[int],
    road_mask: Any = None,
    homography: Any = None,
    camera_model: Any = None,
    calibration_meta: Any = None,
    lane_confidences: Any = None,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Build per-lane centerlines by pairing adjacent marking boundaries.

    Reuses the same measurement-space determination, robust boundary fitting,
    double-line collapsing, and adjacent-boundary pairing already used for
    lane-width stability, so callers such as the I5 geometry-complexity engine
    never re-derive homography handling, boundary grouping, or pairing logic.
    Every accepted pair contributes one centerline built from dense,
    uniformly arc-length-sampled points shared between both boundaries.
    Logical boundaries that cannot be paired (e.g. only one visible marking,
    or a single pre-merged lane-centerline mask blob) are returned individually
    so callers still receive usable geometry.

    Returned centerlines carry both the measurement-space polyline (road-plane
    units when calibrated, raw image pixels otherwise) and the corresponding
    image-space polyline, so downstream callers can re-check horizon
    proximity or occlusion without re-deriving the homography.
    """

    cfg = config or LaneWidthConfig()
    measurement = determine_measurement_space(
        image_shape=image_shape,
        homography=homography,
        camera_model=camera_model,
        calibration_meta=calibration_meta,
        config=cfg,
    )
    fitted = validate_and_fit_boundaries(
        lane_geometries,
        image_shape=image_shape,
        measurement=measurement,
        lane_confidences=lane_confidences,
        config=cfg,
    )
    boundaries = fitted["boundaries"]
    origin, long_axis, lateral_axis = fitted["origin"], fitted["long_axis"], fitted["lateral_axis"]
    if origin is None or not boundaries:
        return {
            "centerlines": [],
            "measurement": measurement,
            "rejected_boundary_reasons": fitted["rejected_boundaries"],
            "rejected_pair_reasons": [],
            "double_line_groups": [],
            "logical_boundary_diagnostics": [],
            "logical_boundary_count": 0,
            "source_boundary_count": len(boundaries),
        }
    collapsed = collapse_double_line_boundaries(
        boundaries,
        image_shape=image_shape,
        measurement_space=measurement["measurement_space"],
        config=cfg,
    )
    logical = collapsed["logical_boundaries"]
    pairing = pair_adjacent_boundaries(
        logical,
        image_shape=image_shape,
        measurement=measurement,
        origin=origin,
        long_axis=long_axis,
        lateral_axis=lateral_axis,
        road_mask=road_mask,
        config=cfg,
    )
    metric_like = str(measurement["measurement_space"]).startswith("calibrated")
    spacing = cfg.sample_spacing_metric if metric_like else max(cfg.sample_spacing_px, 1.0)
    centerlines: list[dict[str, Any]] = []
    paired_ids = set()
    for pair in pairing["accepted_pairs"]:
        left, right = pair["left"], pair["right"]
        start, end = pair["support_interval"]
        _, points, arc = _uniform_centerline_stations(
            left,
            right,
            start,
            end,
            origin=origin,
            long_axis=long_axis,
            lateral_axis=lateral_axis,
            spacing=spacing,
            maximum=cfg.max_width_samples,
            minimum=cfg.min_width_samples,
        )
        image_points, image_valid = _work_to_image(points, measurement)
        centerlines.append(
            {
                "representation_type": "paired_centerline",
                "source_boundary_ids": list(pair["left_source_boundary_ids"]) + list(pair["right_source_boundary_ids"]),
                "points_measurement": points,
                "points_image": image_points,
                "points_image_valid": image_valid,
                "geometry_confidence": float(min(left["geometry_confidence"], right["geometry_confidence"])),
                "support_length": float(pair["support_length"]),
                "drivable_support": pair.get("drivable_support"),
            }
        )
        paired_ids.add(left["boundary_id"])
        paired_ids.add(right["boundary_id"])

    for boundary in logical:
        if boundary["boundary_id"] in paired_ids:
            continue
        span = max(float(boundary["t_max"]) - float(boundary["t_min"]), EPS)
        count = int(np.clip(np.floor(span / max(spacing, EPS)) + 1, cfg.min_width_samples, cfg.max_width_samples))
        stations = np.linspace(float(boundary["t_min"]), float(boundary["t_max"]), count)
        points = _curve_points(boundary, stations, origin, long_axis, lateral_axis)
        image_points, image_valid = _work_to_image(points, measurement)
        centerlines.append(
            {
                "representation_type": "boundary",
                "source_boundary_ids": list(boundary["source_boundary_ids"]),
                "points_measurement": points,
                "points_image": image_points,
                "points_image_valid": image_valid,
                "geometry_confidence": float(boundary["geometry_confidence"]),
                "support_length": span,
                "drivable_support": None,
            }
        )

    logical_boundary_diagnostics = []
    for boundary in logical:
        ends = _curve_points(
            boundary,
            np.asarray([boundary["t_min"], boundary["t_max"]], dtype=np.float64),
            origin,
            long_axis,
            lateral_axis,
        )
        image_ends, image_ends_valid = _work_to_image(ends, measurement)
        logical_boundary_diagnostics.append(
            {
                "boundary_id": boundary["boundary_id"],
                "t_min": float(boundary["t_min"]),
                "t_max": float(boundary["t_max"]),
                "image_y_at_t_min": float(image_ends[0, 1]) if image_ends_valid[0] else None,
                "image_y_at_t_max": float(image_ends[1, 1]) if image_ends_valid[1] else None,
            }
        )

    return {
        "centerlines": centerlines,
        "measurement": measurement,
        "rejected_boundary_reasons": fitted["rejected_boundaries"],
        "rejected_pair_reasons": pairing["rejected_pairs"],
        "double_line_groups": collapsed["double_line_groups"],
        "logical_boundary_diagnostics": logical_boundary_diagnostics,
        "logical_boundary_count": len(logical),
        "source_boundary_count": len(boundaries),
    }


def _normal_intersection(
    curve: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    expected_sign: float,
) -> tuple[Optional[np.ndarray], Optional[int], bool]:
    signed_tangent = (curve - center) @ tangent
    candidates = []
    for index in range(curve.shape[0] - 1):
        first, second = float(signed_tangent[index]), float(signed_tangent[index + 1])
        if first == 0.0:
            fraction = 0.0
        elif first * second > 0.0 or abs(first - second) <= EPS:
            continue
        else:
            fraction = first / (first - second)
        point = curve[index] + fraction * (curve[index + 1] - curve[index])
        normal_distance = float(np.dot(point - center, normal))
        if normal_distance * expected_sign > 0:
            candidates.append((abs(normal_distance), point, index))
    if not candidates:
        nearest = int(np.argmin(np.abs(signed_tangent)))
        point = curve[nearest]
        normal_distance = float(np.dot(point - center, normal))
        tolerance = 0.03 * max(np.ptp(curve[:, 0]), np.ptp(curve[:, 1]), 1.0)
        if abs(float(signed_tangent[nearest])) < tolerance and normal_distance * expected_sign > 0:
            return point, nearest, False
        return None, None, False
    candidates.sort(key=lambda item: item[0])
    ambiguous = len(candidates) > 1 and candidates[1][0] <= candidates[0][0] * 1.15
    return candidates[0][1], candidates[0][2], ambiguous


def _sample_mask_at_points(mask: Optional[np.ndarray], points: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros(points.shape[0], dtype=bool)
    height, width = mask.shape
    rounded = np.round(points).astype(np.int64)
    valid = (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    result = np.ones(points.shape[0], dtype=bool)
    result[valid] = mask[rounded[valid, 1], rounded[valid, 0]] > 0
    return result


def _local_transform_resolution(image_point: np.ndarray, matrix: Optional[np.ndarray]) -> Optional[float]:
    if matrix is None:
        return 1.0
    probes = np.asarray([image_point, image_point + [1.0, 0.0], image_point + [0.0, 1.0]], dtype=np.float64)
    projected, valid = _project_points(probes, matrix)
    if not np.all(valid):
        return None
    return float(max(np.linalg.norm(projected[1] - projected[0]), np.linalg.norm(projected[2] - projected[0])))


def sample_normal_width_profile(
    pair: Mapping[str, Any],
    *,
    image_shape: Sequence[int],
    measurement: Mapping[str, Any],
    origin: np.ndarray,
    long_axis: np.ndarray,
    lateral_axis: np.ndarray,
    object_mask: Any = None,
    config: Optional[LaneWidthConfig] = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Measure width at centerline-normal intersections with reliability q(s)."""

    cfg = config or LaneWidthConfig()
    height, width = int(image_shape[0]), int(image_shape[1])
    left, right = pair["left"], pair["right"]
    start, end = map(float, pair["support_interval"])
    space = str(measurement["measurement_space"])
    calibrated_space = space.startswith("calibrated")
    metric_space = space == "calibrated_metric_bev"
    if metric_space:
        spacing = cfg.sample_spacing_metric
    elif calibrated_space:
        spacing = max((end - start) / 60.0, EPS)
    else:
        spacing = max(cfg.sample_spacing_px * 3.0, 4.0)
    stations, centers, arc = _uniform_centerline_stations(
        left,
        right,
        start,
        end,
        origin=origin,
        long_axis=long_axis,
        lateral_axis=lateral_axis,
        spacing=spacing,
        maximum=cfg.max_width_samples,
        minimum=cfg.min_width_samples,
    )
    dense_stations = np.linspace(start, end, max(500, stations.size * 5))
    left_curve = _curve_points(left, dense_stations, origin, long_axis, lateral_axis)
    right_curve = _curve_points(right, dense_stations, origin, long_axis, lateral_axis)
    mean_slope = 0.5 * (_eval_fit_derivative(left, stations) + _eval_fit_derivative(right, stations))
    tangents = long_axis + mean_slope[:, None] * lateral_axis
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), EPS)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    if float(np.median(normals @ lateral_axis)) < 0:
        normals = -normals

    object_binary = _resize_binary(object_mask, (height, width))
    raw_widths = np.full(stations.size, np.nan, dtype=np.float64)
    reliability = np.zeros(stations.size, dtype=np.float64)
    left_intersections = np.full((stations.size, 2), np.nan, dtype=np.float64)
    right_intersections = np.full((stations.size, 2), np.nan, dtype=np.float64)
    image_centers, image_center_valid = _work_to_image(centers, measurement)
    resolutions = np.full(stations.size, np.nan, dtype=np.float64)
    base_geometry_confidence = float(min(left["geometry_confidence"], right["geometry_confidence"]))
    if metric_space:
        residual_limit = cfg.max_fit_residual_metric
    elif calibrated_space:
        residual_limit = max((end - start) * 0.02, EPS)
    else:
        residual_limit = cfg.max_fit_residual_px
    residual_factor = float(
        np.exp(-(float(left["robust_residual"]) + float(right["robust_residual"])) / max(2.0 * residual_limit, EPS))
    )
    maximum_distance = cfg.max_normal_intersection_distance_factor * max(
        float(pair.get("maximum_candidate_separation") or 0.0),
        float(pair.get("minimum_candidate_separation") or 0.0),
        EPS,
    )
    for index, (station, center, tangent, normal) in enumerate(zip(stations, centers, tangents, normals)):
        left_point, left_index, left_ambiguous = _normal_intersection(left_curve, center, tangent, normal, -1.0)
        right_point, right_index, right_ambiguous = _normal_intersection(right_curve, center, tangent, normal, 1.0)
        if left_point is None or right_point is None or left_ambiguous or right_ambiguous:
            continue
        measured_width = float(np.dot(right_point - left_point, normal))
        if measured_width <= 0 or measured_width > maximum_distance:
            continue
        left_tangent = left_curve[min((left_index or 0) + 1, left_curve.shape[0] - 1)] - left_curve[max((left_index or 0) - 1, 0)]
        right_tangent = right_curve[min((right_index or 0) + 1, right_curve.shape[0] - 1)] - right_curve[max((right_index or 0) - 1, 0)]
        left_tangent /= max(float(np.linalg.norm(left_tangent)), EPS)
        right_tangent /= max(float(np.linalg.norm(right_tangent)), EPS)
        angle = max(
            np.degrees(np.arccos(np.clip(abs(float(np.dot(left_tangent, tangent))), 0.0, 1.0))),
            np.degrees(np.arccos(np.clip(abs(float(np.dot(right_tangent, tangent))), 0.0, 1.0))),
        )
        if angle > cfg.max_intersection_tangent_angle_deg:
            continue
        edge_distance = min(index, stations.size - 1 - index) / max(stations.size - 1, 1)
        edge_factor = float(np.clip(edge_distance / max(cfg.support_edge_fraction, EPS), 0.25, 1.0))
        q = base_geometry_confidence * residual_factor * edge_factor
        if not image_center_valid[index]:
            continue
        image_point = image_centers[index]
        margin = min(image_point[0], width - 1 - image_point[0], image_point[1], height - 1 - image_point[1])
        if margin < 0:
            continue
        if margin < cfg.image_boundary_margin_px:
            q *= max(margin / max(cfg.image_boundary_margin_px, EPS), 0.05)
        if not calibrated_space:
            normalized_y = image_point[1] / max(height - 1, 1)
            if normalized_y < cfg.fallback_min_y_fraction or normalized_y > cfg.fallback_max_y_fraction:
                continue
        resolution = _local_transform_resolution(image_point, measurement.get("homography"))
        if resolution is None or not np.isfinite(resolution):
            continue
        resolutions[index] = resolution
        raw_widths[index] = measured_width
        reliability[index] = q
        left_intersections[index] = left_point
        right_intersections[index] = right_point

    finite_resolution = np.isfinite(resolutions) & np.isfinite(raw_widths)
    median_resolution = float(np.median(resolutions[finite_resolution])) if np.any(finite_resolution) else None
    if median_resolution is not None:
        resolution_ratio = resolutions / max(median_resolution, EPS)
        bad_resolution = finite_resolution & (resolution_ratio > cfg.max_resolution_ratio)
        reliability[bad_resolution] = 0.0
        reliability[finite_resolution] *= np.clip(
            median_resolution / np.maximum(resolutions[finite_resolution], EPS),
            cfg.min_resolution_reliability,
            1.0,
        )

    image_left, left_image_valid = _work_to_image(left_intersections, measurement)
    image_right, right_image_valid = _work_to_image(right_intersections, measurement)
    intersection_valid = left_image_valid & right_image_valid
    for image_points in (image_left, image_right):
        margins = np.minimum.reduce(
            [
                image_points[:, 0],
                width - 1 - image_points[:, 0],
                image_points[:, 1],
                height - 1 - image_points[:, 1],
            ]
        )
        boundary_factor = np.clip(
            margins / max(cfg.image_boundary_margin_px, EPS),
            0.0,
            1.0,
        )
        reliability *= np.where(np.isfinite(boundary_factor), boundary_factor, 0.0)
    reliability[~intersection_valid] = 0.0
    if object_binary is not None:
        occluded = (
            _sample_mask_at_points(object_binary, image_centers)
            | _sample_mask_at_points(object_binary, image_left)
            | _sample_mask_at_points(object_binary, image_right)
            | ~left_image_valid
            | ~right_image_valid
        )
        reliability[occluded] = 0.0

    reliability[reliability < cfg.min_sample_reliability] = 0.0
    valid = np.isfinite(raw_widths) & (reliability >= cfg.min_sample_reliability)
    valid_fraction = float(np.count_nonzero(valid) / max(stations.size, 1))
    output_scale = 1.0 if calibrated_space else 1.0 / max(float(width), 1.0)
    unavailable_reason = None
    if np.count_nonzero(valid) < cfg.min_width_samples:
        unavailable_reason = "insufficient_reliable_normal_width_samples"
    elif valid_fraction < cfg.min_valid_sample_fraction:
        unavailable_reason = "insufficient_valid_width_sample_fraction"
    effective_geometry_confidence = (
        float(np.mean(reliability[valid])) if np.any(valid) else 0.0
    )
    result: dict[str, Any] = {
        "valid_sample_count": int(np.count_nonzero(valid)),
        "valid_sample_fraction": valid_fraction,
        "support_length": float(arc[-1] * output_scale),
        "transformed_resolution": None if median_resolution is None else float(median_resolution),
        "geometry_confidence": effective_geometry_confidence,
        "unavailable_reason": unavailable_reason,
        "_s": arc[valid],
        "_widths": raw_widths[valid] * output_scale,
        "_weights": reliability[valid],
        "_output_scale": output_scale,
    }
    if debug:
        expected_debug_shape = raw_widths.shape
        result["debug"] = {
            "centerline_stations_image": image_centers,
            "left_intersections_image": image_left,
            "right_intersections_image": image_right,
            "normals_image": image_right - image_left,
            "raw_width_samples": raw_widths * output_scale,
            "sample_reliability": reliability,
            "valid_samples": valid,
            "expected_width_profile": np.full(expected_debug_shape, np.nan),
            "residuals": np.full(expected_debug_shape, np.nan),
            "rejected_outliers": np.zeros(expected_debug_shape, dtype=bool),
        }
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> Optional[float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    return float(values[min(int(np.searchsorted(cumulative, target, side="left")), values.size - 1)])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    denominator = float(np.sum(weights))
    return None if denominator <= EPS else float(np.sum(values * weights) / denominator)


def _profile_fit(design: np.ndarray, widths: np.ndarray, weights: np.ndarray, cfg: LaneWidthConfig) -> dict[str, Any]:
    root = np.sqrt(np.maximum(weights, 0.0))
    coefficients = np.linalg.lstsq(design * root[:, None], widths * root, rcond=None)[0]
    reference_width = max(float(np.median(widths)), EPS)
    robust_weights = weights.copy()
    for _ in range(cfg.profile_irls_iterations):
        residual = widths - design @ coefficients
        normalized = np.abs(residual) / max(cfg.profile_huber_delta_relative * reference_width, EPS)
        huber = np.ones_like(normalized)
        outside = normalized > 1.0
        huber[outside] = 1.0 / normalized[outside]
        robust_weights = weights * huber
        root = np.sqrt(np.maximum(robust_weights, 0.0))
        updated = np.linalg.lstsq(design * root[:, None], widths * root, rcond=None)[0]
        if float(np.linalg.norm(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    expected = design @ coefficients
    relative = (widths - expected) / np.maximum(expected, EPS)
    delta = cfg.profile_huber_delta_relative
    absolute = np.abs(relative)
    huber_loss = np.where(absolute <= delta, 0.5 * absolute**2, delta * (absolute - 0.5 * delta))
    loss = _weighted_mean(huber_loss, weights)
    return {
        "coefficients": coefficients,
        "expected": expected,
        "robust_weights": robust_weights,
        "loss": loss,
    }


def fit_width_profile_hypotheses(
    s: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
    *,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Fit constant, linear-taper, and continuous one-change-point profiles."""

    cfg = config or LaneWidthConfig()
    s = np.asarray(s, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if widths.size < cfg.min_width_samples or np.any(widths <= 0):
        return {"selected_model": "unknown", "expected": None, "model_scores": {}, "selection_margin": None, "change_points": []}
    s_normalized = (s - float(np.min(s))) / max(float(np.ptp(s)), EPS)
    n_effective = max(float(np.sum(weights) / max(np.max(weights), EPS)), 2.0)
    models: dict[str, dict[str, Any]] = {}

    def add_model(name: str, design: np.ndarray, parameter_count: int, change_points: list[float]) -> None:
        fit = _profile_fit(design, widths, weights, cfg)
        if fit["loss"] is None or np.any(fit["expected"] <= 0):
            return
        bic = n_effective * np.log(max(2.0 * float(fit["loss"]), 1e-12)) + parameter_count * np.log(n_effective)
        fit.update({"bic": float(bic), "change_points": change_points})
        models[name] = fit

    add_model("constant", np.ones((widths.size, 1)), 1, [])
    add_model("linear_taper", np.column_stack([np.ones(widths.size), s_normalized]), 2, [])
    if widths.size >= 2 * cfg.min_segment_samples:
        best_piecewise = None
        for change in np.linspace(cfg.change_point_min_fraction, cfg.change_point_max_fraction, cfg.change_point_candidates):
            if np.count_nonzero(s_normalized <= change) < cfg.min_segment_samples or np.count_nonzero(s_normalized > change) < cfg.min_segment_samples:
                continue
            design = np.column_stack([np.ones(widths.size), s_normalized, np.maximum(s_normalized - change, 0.0)])
            fit = _profile_fit(design, widths, weights, cfg)
            if fit["loss"] is None or np.any(fit["expected"] <= 0):
                continue
            bic = n_effective * np.log(max(2.0 * float(fit["loss"]), 1e-12)) + 4 * np.log(n_effective)
            fit.update({"bic": float(bic), "change_points": [float(change)]})
            if best_piecewise is None or fit["bic"] < best_piecewise["bic"]:
                best_piecewise = fit
        if best_piecewise is not None:
            models["piecewise_transition"] = best_piecewise

    constant = models.get("constant")
    selected_name = "constant"
    if constant is None:
        return {"selected_model": "unknown", "expected": None, "model_scores": {}, "selection_margin": None, "change_points": []}
    linear = models.get("linear_taper")
    if linear is not None:
        linear_change = abs(float(linear["expected"][-1] - linear["expected"][0])) / max(float(np.median(linear["expected"])), EPS)
        if constant["bic"] - linear["bic"] >= cfg.min_model_score_improvement and linear_change >= cfg.min_nonconstant_relative_change:
            selected_name = "linear_taper"
    piecewise = models.get("piecewise_transition")
    selected = models[selected_name]
    if piecewise is not None:
        piecewise_change = float(np.ptp(piecewise["expected"])) / max(float(np.median(piecewise["expected"])), EPS)
        selected_relative = np.abs(widths - selected["expected"]) / np.maximum(selected["expected"], EPS)
        selected_p90 = _weighted_quantile(selected_relative, weights, 0.90)
        if (
            selected["bic"] - piecewise["bic"] >= cfg.min_piecewise_score_improvement
            and piecewise_change >= cfg.min_transition_relative_change
            and selected_p90 is not None
            and selected_p90 >= cfg.min_piecewise_simple_model_p90
        ):
            selected_name = "piecewise_transition"
            selected = piecewise
    selected = models[selected_name]
    alternative_scores = [
        float(model["bic"])
        for name, model in models.items()
        if name != selected_name
    ]
    margin = None if not alternative_scores else float(min(alternative_scores) - selected["bic"])
    return {
        "selected_model": selected_name,
        "expected": selected["expected"],
        "robust_weights": selected["robust_weights"],
        "model_scores": {name: float(model["bic"]) for name, model in models.items()},
        "selection_margin": margin,
        "change_points": selected["change_points"],
        "s_normalized": s_normalized,
        "coefficients": selected["coefficients"],
    }


def score_width_profile_stability(
    sampled: Mapping[str, Any],
    *,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Separate residual profile stability from expected-profile constancy."""

    cfg = config or LaneWidthConfig()
    unavailable_reason = sampled.get("unavailable_reason")
    if unavailable_reason:
        return {
            "profile_type": "unknown",
            "profile_confidence": 0.0,
            "stability_score": None,
            "width_profile_stability": None,
            "width_constancy_score": None,
            "median_width": None,
            "mean_width": None,
            "robust_relative_mad": None,
            "relative_p90_residual": None,
            "conventional_width_cv": None,
            "min_width": None,
            "max_width": None,
            "fitted_slope": None,
            "relative_width_change": None,
            "change_points": [],
            "profile_model_scores": {},
            "selected_model_margin": None,
            "unavailable_reason": unavailable_reason,
            "notes": ["Width samples were not converted to zero or included in aggregation."],
        }
    s = np.asarray(sampled["_s"], dtype=np.float64)
    widths = np.asarray(sampled["_widths"], dtype=np.float64)
    weights = np.asarray(sampled["_weights"], dtype=np.float64)
    fitted = fit_width_profile_hypotheses(s, widths, weights, config=cfg)
    expected_value = fitted.get("expected")
    if expected_value is None:
        return {
            "profile_type": "unknown",
            "profile_confidence": 0.0,
            "stability_score": None,
            "width_profile_stability": None,
            "width_constancy_score": None,
            "median_width": None,
            "mean_width": None,
            "robust_relative_mad": None,
            "relative_p90_residual": None,
            "conventional_width_cv": None,
            "min_width": None,
            "max_width": None,
            "fitted_slope": None,
            "relative_width_change": None,
            "change_points": [],
            "profile_model_scores": fitted.get("model_scores", {}),
            "selected_model_margin": fitted.get("selection_margin"),
            "unavailable_reason": "expected_width_profile_unavailable",
            "notes": [],
        }
    expected = np.asarray(expected_value, dtype=np.float64)
    relative_error = np.abs(widths - expected) / np.maximum(expected, EPS)
    median_relative_error = _weighted_quantile(relative_error, weights, 0.50)
    r_mad = None if median_relative_error is None else float(1.4826 * median_relative_error)
    r_90 = _weighted_quantile(relative_error, weights, 0.90)
    if r_mad is None or r_90 is None:
        stability = None
    else:
        instability = (
            cfg.stability_lambda * np.clip(r_mad / cfg.tau_relative_mad, 0.0, 1.0)
            + (1.0 - cfg.stability_lambda) * np.clip(r_90 / cfg.tau_relative_p90, 0.0, 1.0)
        )
        stability = float(1.0 - np.clip(instability, 0.0, 1.0))
    median_width = _weighted_quantile(widths, weights, 0.50)
    mean_width = _weighted_mean(widths, weights)
    expected_median = max(float(np.median(expected)), EPS)
    relative_change = float(np.ptp(expected) / expected_median)
    width_constancy = float(1.0 - np.clip(relative_change / cfg.constancy_relative_change_tolerance, 0.0, 1.0))
    selected_model = str(fitted["selected_model"])
    if selected_model == "piecewise_transition":
        differences = np.diff(expected)
        monotonic = bool(np.all(differences >= -1e-8) or np.all(differences <= 1e-8))
        profile_type = "merge_or_split_transition" if monotonic and relative_change >= cfg.min_transition_relative_change else "piecewise_transition"
    else:
        profile_type = selected_model
    if (
        stability is not None
        and (stability < cfg.irregular_stability_threshold or (r_90 is not None and r_90 > cfg.irregular_relative_p90))
    ):
        profile_type = "irregular_or_noisy"
    margin = _finite_float(fitted.get("selection_margin"))
    margin_confidence = 0.4 if margin is None else float(np.clip(margin / 12.0, 0.0, 1.0))
    residual_confidence = 0.0 if r_90 is None else float(np.clip(1.0 - r_90 / 0.30, 0.0, 1.0))
    profile_confidence = float(
        np.clip(
            0.35 * float(sampled.get("valid_sample_fraction") or 0.0)
            + 0.35 * margin_confidence
            + 0.30 * residual_confidence,
            0.0,
            1.0,
        )
    )
    conventional_mean = float(np.mean(widths))
    conventional_cv = float(np.std(widths) / max(conventional_mean, EPS))
    support = max(float(s[-1] - s[0]), EPS)
    fitted_slope = float((expected[-1] - expected[0]) / support)
    robust_weights = np.asarray(fitted.get("robust_weights", weights), dtype=np.float64)
    rejected_outliers = robust_weights < 0.55 * np.maximum(weights, EPS)
    notes = []
    if profile_type in {"linear_taper", "merge_or_split_transition", "piecewise_transition"}:
        notes.append("Expected width change is reported separately and is not itself an instability penalty.")
    result = {
        "profile_type": profile_type,
        "profile_confidence": profile_confidence,
        "stability_score": stability,
        "width_profile_stability": stability,
        "width_constancy_score": width_constancy,
        "median_width": median_width,
        "mean_width": mean_width,
        "robust_relative_mad": r_mad,
        "relative_p90_residual": r_90,
        "conventional_width_cv": conventional_cv,
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
        "fitted_slope": fitted_slope,
        "relative_width_change": relative_change,
        "change_points": fitted.get("change_points", []),
        "profile_model_scores": fitted.get("model_scores", {}),
        "selected_model_margin": margin,
        "unavailable_reason": None,
        "notes": notes,
        "_expected": expected,
        "_relative_error": relative_error,
        "_rejected_outliers": rejected_outliers,
    }
    return result


def _weighted_pair_median(values: np.ndarray, weights: np.ndarray) -> float:
    result = _weighted_quantile(values, weights, 0.50)
    return float(result) if result is not None else float(np.median(values))


def aggregate_pair_results(
    pair_results: Sequence[Mapping[str, Any]],
    *,
    rejected_pair_count: int = 0,
    config: Optional[LaneWidthConfig] = None,
) -> dict[str, Any]:
    """Robust, confidence- and bounded-support-aware pair aggregation."""

    cfg = config or LaneWidthConfig()
    valid = [pair for pair in pair_results if _finite_float(pair.get("stability_score")) is not None]
    unknown_count = rejected_pair_count + sum(1 for pair in pair_results if _finite_float(pair.get("stability_score")) is None)
    if not valid:
        return {
            "lane_width_stability": None,
            "width_profile_stability": None,
            "width_constancy": None,
            "profile_type": "unknown",
            "profile_confidence": 0.0,
            "geometry_confidence": 0.0,
            "valid_pair_count": 0,
            "unknown_pair_count": int(unknown_count),
            "worst_pair_score": None,
            "median_pair_score": None,
            "aggregation_method": "confidence_x_sqrt_bounded_support_robust_median_mean",
            "unavailable_reason": "no_valid_lane_pair_width_profiles",
        }
    supports = np.asarray([max(float(pair.get("support_length") or 0.0), EPS) for pair in valid], dtype=np.float64)
    support_cap = max(float(np.median(supports)) * cfg.aggregation_support_cap_ratio, EPS)
    weights = np.asarray(
        [
            max(float(pair.get("geometry_confidence") or 0.0), 0.10)
            * max(float(pair.get("profile_confidence") or 0.0), 0.10)
            * np.sqrt(min(support, support_cap))
            for pair, support in zip(valid, supports)
        ],
        dtype=np.float64,
    )
    if float(np.sum(weights)) <= EPS:
        weights = np.ones(len(valid), dtype=np.float64)
    scores = np.asarray([float(pair["stability_score"]) for pair in valid], dtype=np.float64)
    weighted_median = _weighted_pair_median(scores, weights)
    weighted_mean = float(np.sum(weights * scores) / np.sum(weights))
    aggregate_stability = float(
        cfg.aggregation_median_weight * weighted_median
        + (1.0 - cfg.aggregation_median_weight) * weighted_mean
    )
    constancies = np.asarray([float(pair["width_constancy_score"]) for pair in valid], dtype=np.float64)
    aggregate_constancy = _weighted_pair_median(constancies, weights)
    profile_confidence = float(
        np.sum(weights * np.asarray([float(pair["profile_confidence"]) for pair in valid])) / np.sum(weights)
    )
    geometry_confidence = float(
        np.sum(weights * np.asarray([float(pair["geometry_confidence"]) for pair in valid])) / np.sum(weights)
    )
    profile_types = {str(pair["profile_type"]) for pair in valid}
    if len(profile_types) == 1:
        profile_type = next(iter(profile_types))
    elif "irregular_or_noisy" in profile_types:
        profile_type = "irregular_or_noisy"
    elif "merge_or_split_transition" in profile_types:
        profile_type = "merge_or_split_transition"
    else:
        profile_type = "piecewise_transition"
    return {
        "lane_width_stability": aggregate_stability,
        "width_profile_stability": aggregate_stability,
        "width_constancy": aggregate_constancy,
        "profile_type": profile_type,
        "profile_confidence": profile_confidence,
        "geometry_confidence": geometry_confidence,
        "valid_pair_count": len(valid),
        "unknown_pair_count": int(unknown_count),
        "worst_pair_score": float(np.min(scores)),
        "median_pair_score": float(np.median(scores)),
        "aggregation_method": "confidence_x_sqrt_bounded_support_robust_median_mean",
        "unavailable_reason": None,
    }


def _boundary_diagnostic(boundary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "boundary_id": boundary["boundary_id"],
        "source_boundary_ids": boundary["source_boundary_ids"],
        "source_point_count": boundary["source_point_count"],
        "inlier_count": boundary["inlier_count"],
        "inlier_fraction": boundary["inlier_fraction"],
        "fitted_support_interval": boundary["fitted_support_interval"],
        "robust_residual": boundary["robust_residual"],
        "extrapolated_fraction": boundary["extrapolated_fraction"],
        "geometry_confidence": boundary["geometry_confidence"],
        "collapsed_double_line": boundary["collapsed_double_line"],
    }


def _measurement_diagnostic(measurement: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "measurement_space": measurement.get("measurement_space"),
        "homography_source": measurement.get("homography_source"),
        "homography_condition_number": measurement.get("homography_condition_number"),
        "valid_transformed_area": measurement.get("valid_transformed_area"),
        "metric_scale_available": measurement.get("metric_scale_available"),
        "metric_scale_confidence": measurement.get("metric_scale_confidence"),
        "measurement_confidence": measurement.get("measurement_confidence"),
        "far_field_exclusion": measurement.get("far_field_exclusion"),
        "fallback_reason": measurement.get("fallback_reason"),
    }


def _debug_boundary_polyline(
    boundary: Mapping[str, Any],
    measurement: Mapping[str, Any],
    origin: np.ndarray,
    long_axis: np.ndarray,
    lateral_axis: np.ndarray,
) -> np.ndarray:
    stations = np.linspace(float(boundary["t_min"]), float(boundary["t_max"]), 100)
    work = _curve_points(boundary, stations, origin, long_axis, lateral_axis)
    image, valid = _work_to_image(work, measurement)
    return image[valid]


def analyze_lane_width_stability(
    lane_geometries: Optional[Iterable[Any]],
    *,
    image_shape: Sequence[int],
    geometry_source: str,
    image_rgb: Any = None,
    road_mask: Any = None,
    object_mask: Any = None,
    homography: Any = None,
    camera_model: Any = None,
    calibration_meta: Any = None,
    lane_confidences: Any = None,
    config: Optional[LaneWidthConfig] = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Analyze perspective-normalized lane-width stability for one image.

    Stability is residual consistency around the selected constant, taper, or
    piecewise profile.  Width constancy measures change in that selected
    expected profile.  Neither geometry confidence nor calibration confidence
    is multiplied into the stability score.
    """

    _ = image_rgb  # Reserved for optional prediction-corridor center refinement.
    cfg = config or LaneWidthConfig()
    if len(image_shape) < 2 or int(image_shape[0]) <= 1 or int(image_shape[1]) <= 1:
        return _json_safe(
            {
                "lane_width_stability": None,
                "width_profile_stability": None,
                "width_constancy": None,
                "profile_type": "unknown",
                "profile_confidence": 0.0,
                "geometry_confidence": 0.0,
                "geometry_source": geometry_source,
                "measurement_space": "unavailable",
                "valid_pair_count": 0,
                "unknown_pair_count": 0,
                "pair_diagnostics": [],
                "rejected_pair_reasons": [],
                "unavailable_reason": "invalid_image_shape",
            }
        )
    measurement = determine_measurement_space(
        image_shape=image_shape,
        homography=homography,
        camera_model=camera_model,
        calibration_meta=calibration_meta,
        config=cfg,
    )
    fitted = validate_and_fit_boundaries(
        lane_geometries,
        image_shape=image_shape,
        measurement=measurement,
        lane_confidences=lane_confidences,
        config=cfg,
    )
    boundaries = fitted["boundaries"]
    origin = fitted["origin"]
    long_axis = fitted["long_axis"]
    lateral_axis = fitted["lateral_axis"]
    collapsed = collapse_double_line_boundaries(
        boundaries,
        image_shape=image_shape,
        measurement_space=measurement["measurement_space"],
        config=cfg,
    )
    logical_boundaries = collapsed["logical_boundaries"]
    boundary_diagnostics = [_boundary_diagnostic(boundary) for boundary in boundaries]
    logical_diagnostics = [_boundary_diagnostic(boundary) for boundary in logical_boundaries]
    transform_diagnostics = _measurement_diagnostic(measurement)

    if origin is None or len(logical_boundaries) < 2:
        reason = "no_credible_boundary_geometry" if not logical_boundaries else "only_one_logical_boundary"
        result = {
            "lane_width_stability": None,
            "width_profile_stability": None,
            "width_constancy": None,
            "profile_type": "unknown",
            "profile_confidence": 0.0,
            "geometry_confidence": 0.0,
            "geometry_source": geometry_source,
            "measurement_space": measurement["measurement_space"],
            "homography_source": measurement["homography_source"],
            "metric_scale_available": measurement["metric_scale_available"],
            "metric_scale_confidence": measurement["metric_scale_confidence"],
            "measurement_confidence": measurement["measurement_confidence"],
            "width_units": measurement["width_units"],
            "valid_pair_count": 0,
            "unknown_pair_count": 0,
            "worst_pair_score": None,
            "median_pair_score": None,
            "pair_diagnostics": [],
            "rejected_pair_reasons": [],
            "rejected_boundary_reasons": fitted["rejected_boundaries"],
            "boundary_diagnostics": boundary_diagnostics,
            "logical_boundary_diagnostics": logical_diagnostics,
            "double_line_groups": collapsed["double_line_groups"],
            "transform_diagnostics": transform_diagnostics,
            "aggregation_method": "confidence_x_sqrt_bounded_support_robust_median_mean",
            "unavailable_reason": reason,
        }
        if debug:
            result["debug"] = {"source_boundaries_image": [], "logical_boundaries_image": []}
        return _json_safe(result)

    pairing = pair_adjacent_boundaries(
        logical_boundaries,
        image_shape=image_shape,
        measurement=measurement,
        origin=origin,
        long_axis=long_axis,
        lateral_axis=lateral_axis,
        road_mask=road_mask,
        config=cfg,
    )
    pair_results: list[dict[str, Any]] = []
    for candidate in pairing["accepted_pairs"]:
        sampled = sample_normal_width_profile(
            candidate,
            image_shape=image_shape,
            measurement=measurement,
            origin=origin,
            long_axis=long_axis,
            lateral_axis=lateral_axis,
            object_mask=object_mask,
            config=cfg,
            debug=debug,
        )
        scored = score_width_profile_stability(sampled, config=cfg)
        diagnostic = {
            "pair_index": candidate["pair_index"],
            "left_boundary_id": candidate["left_boundary_id"],
            "right_boundary_id": candidate["right_boundary_id"],
            "source_boundary_ids": list(candidate["left_source_boundary_ids"]) + list(candidate["right_source_boundary_ids"]),
            "geometry_source": geometry_source,
            "measurement_space": measurement["measurement_space"],
            "homography_source": measurement["homography_source"],
            "metric_scale_available": measurement["metric_scale_available"],
            "metric_scale_confidence": measurement["metric_scale_confidence"],
            "measurement_confidence": measurement["measurement_confidence"],
            "width_units": measurement["width_units"],
            "geometry_confidence": sampled["geometry_confidence"],
            "valid_sample_count": sampled["valid_sample_count"],
            "valid_sample_fraction": sampled["valid_sample_fraction"],
            "support_length": sampled["support_length"],
            "transformed_resolution": sampled["transformed_resolution"],
            "drivable_support": candidate.get("drivable_support"),
            **{key: value for key, value in scored.items() if not key.startswith("_")},
        }
        if debug:
            debug_payload = dict(sampled.get("debug", {}))
            valid_mask = np.asarray(debug_payload.get("valid_samples", []), dtype=bool)
            expected_full = np.full(valid_mask.shape, np.nan, dtype=np.float64)
            residual_full = np.full(valid_mask.shape, np.nan, dtype=np.float64)
            outlier_full = np.zeros(valid_mask.shape, dtype=bool)
            expected = scored.get("_expected")
            residual = scored.get("_relative_error")
            outliers = scored.get("_rejected_outliers")
            if expected is not None and np.count_nonzero(valid_mask) == len(expected):
                expected_full[valid_mask] = np.asarray(expected)
                residual_full[valid_mask] = np.asarray(residual)
                outlier_full[valid_mask] = np.asarray(outliers)
            debug_payload.update(
                {
                    "expected_width_profile": expected_full,
                    "residuals": residual_full,
                    "rejected_outliers": outlier_full,
                    "change_points": scored.get("change_points", []),
                }
            )
            diagnostic["debug"] = debug_payload
        pair_results.append(diagnostic)

    aggregate = aggregate_pair_results(
        pair_results,
        rejected_pair_count=len(pairing["rejected_pairs"]),
        config=cfg,
    )
    result = {
        **aggregate,
        "geometry_source": geometry_source,
        "measurement_space": measurement["measurement_space"],
        "homography_source": measurement["homography_source"],
        "metric_scale_available": measurement["metric_scale_available"],
        "metric_scale_confidence": measurement["metric_scale_confidence"],
        "measurement_confidence": measurement["measurement_confidence"],
        "width_units": measurement["width_units"],
        "source_boundary_count": len(boundaries),
        "logical_boundary_count": len(logical_boundaries),
        "pair_diagnostics": pair_results,
        "rejected_pair_reasons": pairing["rejected_pairs"],
        "rejected_boundary_reasons": fitted["rejected_boundaries"],
        "boundary_diagnostics": boundary_diagnostics,
        "logical_boundary_diagnostics": logical_diagnostics,
        "double_line_groups": collapsed["double_line_groups"],
        "transform_diagnostics": transform_diagnostics,
    }
    if not pairing["accepted_pairs"] and result["unavailable_reason"]:
        result["unavailable_reason"] = "no_credible_adjacent_boundary_pairs"
    if debug:
        result["debug"] = {
            "source_boundaries_image": [
                _debug_boundary_polyline(boundary, measurement, origin, long_axis, lateral_axis)
                for boundary in boundaries
            ],
            "logical_boundaries_image": [
                _debug_boundary_polyline(boundary, measurement, origin, long_axis, lateral_axis)
                for boundary in logical_boundaries
            ],
            "logical_boundary_ids": [boundary["boundary_id"] for boundary in logical_boundaries],
            "config": asdict(cfg),
        }
    return _json_safe(result)


def lane_width_stability_score(*args: Any, **kwargs: Any) -> Optional[float]:
    """Scalar convenience wrapper around :func:`analyze_lane_width_stability`."""

    return _finite_float(analyze_lane_width_stability(*args, **kwargs).get("lane_width_stability"))


__all__ = [
    "LaneWidthConfig",
    "aggregate_pair_results",
    "analyze_lane_width_stability",
    "collapse_double_line_boundaries",
    "construct_lane_centerlines",
    "determine_measurement_space",
    "extract_grouped_boundaries_from_mask",
    "fit_width_profile_hypotheses",
    "lane_width_stability_score",
    "pair_adjacent_boundaries",
    "sample_normal_width_profile",
    "score_width_profile_stability",
    "validate_and_fit_boundaries",
]
