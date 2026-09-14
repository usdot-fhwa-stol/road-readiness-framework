"""Classical differential-geometry analysis of visible lane-alignment complexity.

This module implements I5 as a transparent, deterministic, CPU-only classical
computer-vision / differential-geometry pipeline. It characterizes the visible
plan-view geometric *difficulty* of lane alignment and, separately, topology
(splits/merges). It never characterizes pavement-marking condition, detector
quality, or ADS readiness, and it never trains or applies a learned complexity
classifier.

Geometry sources (never mixed within one analysis call):

- ``ground_truth``: canonical I5, GT polylines/lane_json/mask only.
- a prediction identifier (``native_polyline`` / ``lane_json`` / ``mask_derived``):
  operational I5_pred, evaluated-model geometry only. Never receives GT masks,
  GT lane points, GT lane count, or GT metadata.

Pipeline, reusing the shared I1/I4 geometry engines rather than duplicating them:

1. ``select_lane_geometry`` — the same native-polyline > lane_json >
   mask-grouped-centerline priority already used for I1, via
   :mod:`evaluation.lane_continuity`.
2. ``pair_boundaries_and_construct_centerlines`` — reuses
   :func:`evaluation.lane_width_stability.construct_lane_centerlines` (boundary
   fitting, double-line collapsing, adjacent-boundary pairing) so a lane
   centerline is built from paired left/right markings when reliable pairs
   exist, and a single boundary's own curve is used otherwise. This also means
   parallel offsets of one road curve and thick-vs-thin marking masks converge
   on the same construction path used for I4.
3. ``determine_measurement_space`` (reused from I4) picks calibrated metric BEV,
   calibrated scale-free BEV, or an uncalibrated fallback; ``transform_to_road_plane``
   then decides, for the uncalibrated case, between lane-width-normalized and a
   normalized image-space proxy, and applies that scale explicitly.
4. ``robustly_fit_parametric_curve`` cleans x(s), y(s) with a local median
   despiking pass and Savitzky-Golay smoothing (both bounded-window, hence
   local) over an approximately arc-length parameter, then fits an
   interpolating spline through the cleaned sequence purely for analytic
   tangents — never a single global quadratic or a high-degree polynomial,
   and never a global smoothing spline whose automatic knot selection could
   ring across the whole domain in response to one real, localized curvature
   transition.
5. ``calculate_curvature_profile`` computes signed curvature kappa(s) via a
   local three-point (Menger) formula on the densely resampled fitted curve,
   with reliability weights for edge tapering, horizon proximity, and object
   occlusion.
6. ``calculate_alignment_components`` / ``classify_alignment_profile`` produce
   bendiness, peak curvature, curvature variation, reversal statistics,
   tortuosity, and a profile class (straight / constant_curve /
   transition_curve / compound_curve / reverse_curve / irregular_or_noisy /
   unknown) using a BIC-penalized weighted model comparison.
7. ``normalize_complexity_components`` / component RMS combine a documented,
   versioned, non-standard saturating map into one 0-1 scalar, kept separate
   from ``complexity_confidence``.
8. ``infer_lane_topology`` is a separate, conservative diagnostic of supported
   lane-count changes across the visible extent; it is never folded into the
   alignment-complexity scalar and never claims certainty from one image.

None of the reference scales (``tau_*``), weights, or thresholds below are
ISO/SAE/ASAM/PIARC/FHWA/AASHTO standards. They are documented, versioned,
image-analysis calibration parameters for this pilot, exercised by the
sensitivity checks in ``tests/test_i5_geometry_complexity.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from evaluation.lane_continuity import (
    LaneContinuityConfig,
    extract_guidance_lanes_from_mask,
    lanes_from_lane_json,
    prepare_guidance_lanes,
)
from evaluation.lane_width_stability import (
    LaneWidthConfig,
    construct_lane_centerlines,
    determine_measurement_space,
)

EPS = 1e-9
GEOMETRY_COMPLEXITY_VERSION = "i5_geometry_complexity_v1"


@dataclass(frozen=True)
class GeometryComplexityConfig:
    """Explicit, versioned thresholds for I5 alignment/topology complexity.

    None of these values are engineering or regulatory standards. They are
    image-analysis calibration parameters for this pilot.
    """

    config_version: str = GEOMETRY_COMPLEXITY_VERSION

    # Geometry validation (shared with I1's guidance-lane preparation).
    sample_spacing_px: float = 2.0
    max_longitudinal_samples: int = 700
    min_geometry_points: int = 4
    min_lane_support_px: float = 60.0

    # Robust parametric-curve fitting.
    min_points_cubic: int = 8
    min_points_quadratic: int = 5
    min_points_linear: int = 4
    spline_residual_floor_relative: float = 0.25

    # Curvature sampling.
    curvature_oversample_factor: float = 3.0
    curvature_samples_min: int = 24
    curvature_samples_max: int = 240
    curvature_menger_window_samples: int = 40
    curvature_menger_window_max_fraction: float = 0.2

    # Reliability weighting.
    edge_taper_fraction: float = 0.06
    horizon_min_y_fraction: float = 0.34
    horizon_taper_fraction: float = 0.12
    min_sample_reliability: float = 0.08

    # Curvature deadband / reversal detection (Schmitt trigger), as a fraction
    # of the active measurement space's bendiness reference scale (tau).
    curvature_deadband_low_fraction: float = 0.20
    curvature_deadband_high_fraction: float = 0.45
    min_reversal_separation_fraction: float = 0.08

    # Alignment-profile classification.
    primitive_min_support_fraction: float = 0.18
    primitive_bic_margin_min: float = 3.0
    change_point_candidates: int = 11
    change_point_min_fraction: float = 0.22
    change_point_max_fraction: float = 0.78
    straight_bendiness_fraction: float = 0.05
    irregular_residual_fraction: float = 0.5
    min_curvature_change_fraction: float = 0.15

    # Alignment-complexity saturating map reference scales (tau), documented
    # and versioned per measurement space; never presented as a standard.
    tau_bendiness_metric: float = 0.02
    tau_p95_curvature_metric: float = 0.05
    tau_curvature_variation_metric: float = 0.01
    tau_reversal_density_metric: float = 0.05

    tau_bendiness_scale_free: float = 0.35
    tau_p95_curvature_scale_free: float = 0.9
    tau_curvature_variation_scale_free: float = 0.6
    tau_reversal_density_scale_free: float = 0.6

    tau_bendiness_image_proxy: float = 1.2
    tau_p95_curvature_image_proxy: float = 3.0
    tau_curvature_variation_image_proxy: float = 2.5
    tau_reversal_density_image_proxy: float = 2.0

    component_weight_bendiness: float = 1.0
    component_weight_p95_curvature: float = 1.0
    component_weight_curvature_variation: float = 1.0
    component_weight_reversal_density: float = 1.0

    # Multi-lane aggregation.
    aggregation_support_cap_ratio: float = 2.0
    aggregation_min_confidence: float = 0.15

    # Topology (kept separate from alignment complexity).
    topology_band_fraction: float = 0.30


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


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    denom = float(np.sum(weights))
    return None if denom <= EPS else float(np.sum(values * weights) / denom)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> Optional[float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None
    v, w = values[valid], weights[valid]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w)
    target = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), v.size - 1)
    return float(v[index])


def _weighted_lstsq(design: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.maximum(weights, 0.0))
    return np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]


def _sample_mask_at_points(mask: Any, points: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    height, width = int(image_shape[0]), int(image_shape[1])
    if array.shape != (height, width):
        array = cv2.resize((array > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    binary = array > 0
    rounded = np.round(points).astype(np.int64)
    result = np.zeros(points.shape[0], dtype=bool)
    valid = (
        np.isfinite(points).all(axis=1)
        & (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    result[valid] = binary[rounded[valid, 1], rounded[valid, 0]]
    return result


# ---------------------------------------------------------------------------
# Geometry validation, selection, and centerline construction (reuse layer).
# ---------------------------------------------------------------------------

def validate_lane_geometry(
    lane_geometries: Optional[Iterable[Any]],
    image_shape: Sequence[int],
    config: Optional[GeometryComplexityConfig] = None,
) -> list[dict[str, Any]]:
    """Sanitize, order, robustly smooth, and resample candidate polylines.

    A thin wrapper around the shared I1 guidance-lane preparation, so geometry
    validation (nonfinite/duplicate/out-of-frame rejection, consistent
    ordering, short-lane rejection) is identical across I1, I4, and I5.
    """

    cfg = config or GeometryComplexityConfig()
    continuity_cfg = LaneContinuityConfig(
        sample_spacing_px=cfg.sample_spacing_px,
        max_longitudinal_samples=cfg.max_longitudinal_samples,
        min_geometry_points=cfg.min_geometry_points,
        min_lane_support_px=cfg.min_lane_support_px,
    )
    return prepare_guidance_lanes(lane_geometries, image_shape, config=continuity_cfg)


def extract_grouped_centerlines_from_mask(
    mask: Any,
    config: Optional[GeometryComplexityConfig] = None,
) -> list[np.ndarray]:
    """Group dashed/disconnected mask components into guidance centerlines.

    Reuses the shared I1 mask-grouping extractor. A thick segmentation stroke
    does not change the recovered centerline: morphological closing operates
    on a disposable geometry copy, never on curvature-bearing pixel width.
    """

    cfg = config or GeometryComplexityConfig()
    continuity_cfg = LaneContinuityConfig(
        sample_spacing_px=cfg.sample_spacing_px,
        max_longitudinal_samples=cfg.max_longitudinal_samples,
        min_geometry_points=cfg.min_geometry_points,
        min_lane_support_px=cfg.min_lane_support_px,
    )
    return extract_guidance_lanes_from_mask(mask, config=continuity_cfg)


def select_lane_geometry(
    image_shape: Sequence[int],
    *,
    native_lanes: Optional[Iterable[Any]] = None,
    lane_json: Optional[Mapping[str, Any]] = None,
    mask: Any = None,
    config: Optional[GeometryComplexityConfig] = None,
) -> tuple[list[Any], Optional[str]]:
    """Select the highest-priority credible geometry representation.

    Priority: native float-coordinate polylines, then compatible lane_json,
    then mask-grouped centerlines, then unavailable. Identical priority order
    to canonical/operational I1, applied independently to whichever geometry
    source (ground truth or one evaluated model) the caller supplies.
    """

    cfg = config or GeometryComplexityConfig()
    native_candidates = [] if native_lanes is None else list(native_lanes)
    candidates = [
        (native_candidates, "native_polyline"),
        (lanes_from_lane_json(lane_json), "lane_json"),
    ]
    for candidate, source in candidates:
        if candidate and validate_lane_geometry(candidate, image_shape, config=cfg):
            return candidate, source
    mask_lanes = extract_grouped_centerlines_from_mask(mask, config=cfg)
    if mask_lanes and validate_lane_geometry(mask_lanes, image_shape, config=cfg):
        return mask_lanes, "mask_grouped_centerline"
    return [], None


def pair_boundaries_and_construct_centerlines(
    lane_geometries: Optional[Iterable[Any]],
    *,
    image_shape: Sequence[int],
    road_mask: Any = None,
    homography: Any = None,
    camera_model: Any = None,
    calibration_meta: Any = None,
    config: Optional[GeometryComplexityConfig] = None,
) -> dict[str, Any]:
    """Reuse the I4 boundary-fitting/double-line/pairing engine for centerlines."""

    cfg = config or GeometryComplexityConfig()
    width_cfg = LaneWidthConfig(
        sample_spacing_px=cfg.sample_spacing_px,
        max_boundary_samples=cfg.max_longitudinal_samples,
        min_source_points=cfg.min_geometry_points,
        min_boundary_support_px=cfg.min_lane_support_px,
    )
    return construct_lane_centerlines(
        lane_geometries,
        image_shape=image_shape,
        road_mask=road_mask,
        homography=homography,
        camera_model=camera_model,
        calibration_meta=calibration_meta,
        config=width_cfg,
    )


def determine_i5_measurement_space(
    raw_measurement: Mapping[str, Any],
    *,
    lane_width_context: Optional[Mapping[str, Any]] = None,
    config: Optional[GeometryComplexityConfig] = None,
) -> dict[str, Any]:
    """Layer I5's own labels/priority on top of the shared I4 measurement space.

    Priority: calibrated metric BEV, calibrated scale-free BEV,
    lane-width-normalized (only when an explicit same-source lane-width
    estimate is supplied), then a normalized image-space proxy.
    """

    _ = config
    space = raw_measurement.get("measurement_space")
    if space == "calibrated_metric_bev":
        return {
            "i5_measurement_space": "calibrated_metric_bev",
            "scale_source": "homography_metric_scale",
            "metric_scale_available": True,
        }
    if space == "calibrated_scale_free_bev":
        return {
            "i5_measurement_space": "calibrated_scale_free_bev",
            "scale_source": "homography_scale_free",
            "metric_scale_available": False,
        }
    width = None
    if isinstance(lane_width_context, Mapping):
        candidate = _finite_float(lane_width_context.get("median_width"))
        if candidate is not None and candidate > 0:
            width = candidate
    if width is not None:
        return {
            "i5_measurement_space": "lane_width_normalized",
            "scale_source": "lane_width_context",
            "metric_scale_available": False,
            "lane_width_scale": width,
        }
    return {
        "i5_measurement_space": "normalized_image_proxy",
        "scale_source": "image_diagonal",
        "metric_scale_available": False,
    }


def transform_to_road_plane(
    centerline: Mapping[str, Any],
    *,
    image_shape: Sequence[int],
    i5_space: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    """Apply the final I5 measurement-space scale to a centerline's points.

    Calibrated points (already road-plane meters or scale-free road-plane
    units from :func:`pair_boundaries_and_construct_centerlines`) pass through
    unchanged. Uncalibrated points are explicitly normalized — by an
    externally supplied lane width when available, otherwise by the image
    diagonal — so curvature is dimensionless and comparable across images
    from the same camera configuration.
    """

    points = np.asarray(centerline["points_measurement"], dtype=np.float64)
    space = i5_space["i5_measurement_space"]
    if space in ("calibrated_metric_bev", "calibrated_scale_free_bev"):
        return points.copy(), 1.0
    if space == "lane_width_normalized":
        scale = max(float(i5_space["lane_width_scale"]), EPS)
        return points / scale, scale
    diagonal = float(np.hypot(image_shape[0], image_shape[1]))
    scale = max(diagonal, EPS)
    return points / scale, scale


def _tau_reference_for_space(space: str, cfg: GeometryComplexityConfig) -> dict[str, float]:
    if space == "calibrated_metric_bev":
        return {
            "bendiness": cfg.tau_bendiness_metric,
            "p95_curvature": cfg.tau_p95_curvature_metric,
            "curvature_variation": cfg.tau_curvature_variation_metric,
            "reversal_density": cfg.tau_reversal_density_metric,
        }
    if space in ("calibrated_scale_free_bev", "lane_width_normalized"):
        return {
            "bendiness": cfg.tau_bendiness_scale_free,
            "p95_curvature": cfg.tau_p95_curvature_scale_free,
            "curvature_variation": cfg.tau_curvature_variation_scale_free,
            "reversal_density": cfg.tau_reversal_density_scale_free,
        }
    return {
        "bendiness": cfg.tau_bendiness_image_proxy,
        "p95_curvature": cfg.tau_p95_curvature_image_proxy,
        "curvature_variation": cfg.tau_curvature_variation_image_proxy,
        "reversal_density": cfg.tau_reversal_density_image_proxy,
    }


# ---------------------------------------------------------------------------
# Robust parametric-curve fitting and curvature.
# ---------------------------------------------------------------------------

def _local_median_despike_and_smooth(
    values: np.ndarray, cfg: GeometryComplexityConfig, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Replace local spikes with a windowed median, then Savitzky-Golay smooth.

    Both steps are local (bounded window), unlike a global smoothing-spline
    fit: a smoothing spline's automatic knot selection can, for a curve that
    contains one real sharp transition, remove interior knots in a way that
    forces the fitted curve to ring across the *entire* domain to partially
    accommodate that transition. A local median filter followed by a local
    polynomial (Savitzky-Golay) filter cannot ring far from a localized
    feature, since each output sample only ever depends on a small
    neighborhood of input samples. This is the same technique already used
    for I4 boundary smoothing (:func:`evaluation.lane_width_stability._locally_smoothed_curve`).

    Returns ``(cleaned, spike_mask)`` where ``spike_mask`` marks points
    replaced by the local median (hard outliers), for inlier-fraction
    reporting.
    """

    n = values.size
    if n < 5:
        return values.copy(), np.zeros(n, dtype=bool)
    window = min(9, n if n % 2 else n - 1)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    local_median = np.asarray([np.median(padded[i : i + window]) for i in range(n)])
    local_residual = values - local_median
    scale = 1.4826 * float(np.median(np.abs(local_residual - np.median(local_residual))))
    spike_threshold = max(7.0 * scale, 4.0 * floor)
    spikes = np.abs(local_residual) > spike_threshold
    cleaned = values.copy()
    cleaned[spikes] = local_median[spikes]
    smooth_window = min(11, cleaned.size if cleaned.size % 2 else cleaned.size - 1)
    if smooth_window >= 5:
        cleaned = savgol_filter(cleaned, smooth_window, min(3, smooth_window - 2), mode="interp")
    return cleaned, spikes


def robustly_fit_parametric_curve(
    points: np.ndarray,
    *,
    aux_points: Optional[np.ndarray] = None,
    config: Optional[GeometryComplexityConfig] = None,
) -> Optional[dict[str, Any]]:
    """Fit c(s) = [x(s), y(s)] as a robust, arc-length-parameterized curve.

    Sparse hard outliers are first replaced by a windowed local median, then
    the sequence is Savitzky-Golay smoothed (a local, bounded-window robust
    smoother — see :func:`_local_median_despike_and_smooth`), and finally an
    interpolating cubic (or lower-degree, for short lanes) spline is fit
    through the cleaned sequence purely to obtain analytic tangents and a
    densely resampled curve. Curvature itself is later computed from that
    dense resampling using a local three-point (Menger) formula rather than
    the spline's second derivative, so a single unconstrained global
    quadratic or a high-degree global polynomial is never the effective
    model, and derivative estimates are never dominated by point-level noise.
    Extrapolation outside the observed arc-length support is never performed.

    ``aux_points`` (e.g. the same centerline's image-space coordinates) is
    filtered identically to ``points`` and returned aligned to the same arc
    length, so callers can later look up image position at any analysis
    arc-length sample without re-deriving the geometry.
    """

    cfg = config or GeometryComplexityConfig()
    pts = np.asarray(points, dtype=np.float64)
    aux = None if aux_points is None else np.asarray(aux_points, dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    if aux is not None:
        finite &= np.isfinite(aux).all(axis=1)
    pts = pts[finite]
    aux = None if aux is None else aux[finite]
    if pts.shape[0] < cfg.min_points_linear:
        return None
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], segment_lengths > 1e-9])
    pts = pts[keep]
    aux = None if aux is None else aux[keep]
    if pts.shape[0] < cfg.min_points_linear:
        return None

    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    support = float(s[-1])
    if support <= EPS:
        return None
    n = pts.shape[0]
    if n >= cfg.min_points_cubic:
        k = 3
    elif n >= cfg.min_points_quadratic:
        k = 2
    else:
        k = 1

    # The outlier floor must scale with the *active coordinate system* (raw
    # pixels, road-plane meters, or a normalized 0-1 proxy) rather than being
    # one absolute constant, otherwise the same physical geometry would be
    # treated very differently depending on measurement space alone.
    typical_spacing = float(np.median(np.diff(s)))
    floor = max(cfg.spline_residual_floor_relative * typical_spacing, EPS)

    smoothed_x, spikes_x = _local_median_despike_and_smooth(pts[:, 0], cfg, floor)
    smoothed_y, spikes_y = _local_median_despike_and_smooth(pts[:, 1], cfg, floor)
    spikes = spikes_x | spikes_y
    inlier_fraction = float(np.mean(~spikes))

    try:
        x_spline = UnivariateSpline(s, smoothed_x, k=k, s=0.0)
        y_spline = UnivariateSpline(s, smoothed_y, k=k, s=0.0)
    except Exception:
        return None

    residual = np.hypot(pts[:, 0] - smoothed_x, pts[:, 1] - smoothed_y)
    inlier_mask = ~spikes
    if np.any(inlier_mask):
        inlier_residual = residual[inlier_mask]
        robust_residual = 1.4826 * float(np.median(np.abs(inlier_residual - np.median(inlier_residual))))
    else:
        robust_residual = float(np.median(residual))
    return {
        "x_spline": x_spline,
        "y_spline": y_spline,
        "k": k,
        "s_min": float(s[0]),
        "s_max": float(s[-1]),
        "support": support,
        "s_source": s,
        "aux_source": aux,
        "source_point_count": n,
        "inlier_fraction": inlier_fraction,
        "robust_residual": robust_residual,
    }


def calculate_curvature_profile(
    fit: Mapping[str, Any],
    *,
    image_shape: Sequence[int],
    object_mask: Any = None,
    config: Optional[GeometryComplexityConfig] = None,
) -> dict[str, Any]:
    """Sample signed curvature kappa(s) via a local three-point (Menger) estimate.

    Curvature is evaluated from the fitted spline's *positions* using a local
    circumradius formula rather than the spline's global second derivative:

        kappa(s) ~= 2 * cross(p1-p0, p2-p1) / (|p0p1| * |p1p2| * |p2p0|)

    for a small local window [p0, p1, p2] around each sample. A global
    second derivative of a smoothing spline can ring across the *entire*
    domain when the curve contains one real, localized sharp transition (few
    effective knots trying to represent a kink elsewhere), which would
    corrupt curvature estimates far from that transition. The local formula
    only ever looks at a small neighborhood, so it is immune to that
    far-field artifact while remaining exact for the reference case this
    module is built around: a true circular arc has constant Menger
    curvature equal to 1/radius, independent of window size.
    """

    cfg = config or GeometryComplexityConfig()
    s_min, s_max = fit["s_min"], fit["s_max"]
    support = s_max - s_min
    n_samples = int(
        np.clip(
            round(cfg.curvature_oversample_factor * fit["source_point_count"]),
            cfg.curvature_samples_min,
            cfg.curvature_samples_max,
        )
    )
    if support <= EPS or n_samples < 2 * max(1, int(cfg.curvature_oversample_factor)) + 1:
        return {"s": np.array([]), "kappa": np.array([]), "weight": np.array([]), "image_xy": None, "valid_fraction": 0.0}
    # The window widens the local baseline used for curvature so that residual
    # high-frequency position noise (left over after despiking/smoothing) is
    # averaged out rather than amplified by close differentiation, while
    # remaining bounded to a fraction of the sample count so short lanes still
    # get a (smaller) local, not global, curvature baseline.
    window = max(1, min(int(cfg.curvature_menger_window_samples), int(n_samples * cfg.curvature_menger_window_max_fraction)))

    s = np.linspace(s_min, s_max, n_samples)
    x_vals = fit["x_spline"](s)
    y_vals = fit["y_spline"](s)
    dx = fit["x_spline"].derivative(1)(s)
    dy = fit["y_spline"].derivative(1)(s)
    speed2 = dx * dx + dy * dy
    stable = speed2 > 1e-10

    idx = np.arange(n_samples)
    lo = np.clip(idx - window, 0, n_samples - 1)
    hi = np.clip(idx + window, 0, n_samples - 1)
    in_range = (idx - window >= 0) & (idx + window < n_samples)
    p0x, p0y = x_vals[lo], y_vals[lo]
    p1x, p1y = x_vals, y_vals
    p2x, p2y = x_vals[hi], y_vals[hi]
    d01 = np.hypot(p1x - p0x, p1y - p0y)
    d12 = np.hypot(p2x - p1x, p2y - p1y)
    d02 = np.hypot(p2x - p0x, p2y - p0y)
    denom = d01 * d12 * d02
    cross = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
    well_conditioned = denom > 1e-12
    kappa = np.where(well_conditioned, 2.0 * cross / np.maximum(denom, 1e-12), 0.0)
    stable = stable & in_range & well_conditioned

    # A small taper still slightly discounts confidence very close to the
    # observed support edge (extrapolation risk in the underlying fit), on
    # top of the hard window-based exclusion above.
    edge_distance = np.minimum(s - s_min, s_max - s) / max(support, EPS)
    edge_factor = np.clip(edge_distance / max(cfg.edge_taper_fraction, EPS), 0.15, 1.0)
    weight = edge_factor * stable.astype(np.float64)

    image_xy = None
    if fit.get("aux_source") is not None:
        image_x = np.interp(s, fit["s_source"], fit["aux_source"][:, 0])
        image_y = np.interp(s, fit["s_source"], fit["aux_source"][:, 1])
        image_xy = np.column_stack([image_x, image_y])
        height = float(image_shape[0])
        normalized_y = image_y / max(height - 1.0, 1.0)
        horizon_factor = np.clip(
            (normalized_y - cfg.horizon_min_y_fraction) / max(cfg.horizon_taper_fraction, EPS), 0.05, 1.0
        )
        weight = weight * horizon_factor
        if object_mask is not None:
            occluded = _sample_mask_at_points(object_mask, image_xy, image_shape)
            weight = weight * (~occluded).astype(np.float64)

    weight = np.clip(weight, 0.0, 1.0)
    valid_fraction = float(np.mean(weight >= cfg.min_sample_reliability)) if weight.size else 0.0
    return {
        "s": s,
        "kappa": kappa,
        "weight": weight,
        "image_xy": image_xy,
        "valid_fraction": valid_fraction,
        "menger_window": window,
    }


def _count_reversals(
    s: np.ndarray, kappa: np.ndarray, weight: np.ndarray, cfg: GeometryComplexityConfig, tau_bendiness: float
) -> tuple[int, list[float]]:
    """Schmitt-trigger sign-reversal count with a deadband and min separation.

    A reversal is only registered once |kappa| exceeds the high threshold with
    a sign opposite the last *confirmed* sign, and only after a minimum
    arc-length separation from the previous reversal. Small oscillations
    around zero, or noise below the low threshold, never flip the confirmed
    sign and never count as a reverse curve.
    """

    low = cfg.curvature_deadband_low_fraction * tau_bendiness
    high = cfg.curvature_deadband_high_fraction * tau_bendiness
    min_sep = cfg.min_reversal_separation_fraction * max(float(s[-1] - s[0]), EPS)
    confirmed_sign = 0
    last_change_s = -np.inf
    count = 0
    events: list[float] = []
    for si, ki, wi in zip(s, kappa, weight):
        if wi < cfg.min_sample_reliability:
            continue
        magnitude = abs(ki)
        if confirmed_sign == 0:
            # Not yet armed: only a confident excursion past the high
            # threshold establishes the initial confirmed sign.
            if magnitude >= high:
                confirmed_sign = 1 if ki > 0 else -1
                last_change_s = si
            continue
        if magnitude <= low:
            # Inside the deadband: too weak to confirm a reversal, but also
            # not held against the already-confirmed sign.
            continue
        if magnitude >= high:
            candidate_sign = 1 if ki > 0 else -1
            if candidate_sign != confirmed_sign and (si - last_change_s) >= min_sep:
                count += 1
                events.append(float(si))
                confirmed_sign = candidate_sign
                last_change_s = si
    return count, events


def calculate_alignment_components(
    profile: Mapping[str, Any], *, tau_bendiness: float, config: Optional[GeometryComplexityConfig] = None
) -> Optional[dict[str, Any]]:
    """Curvature-profile scalar diagnostics (bendiness, p95, variation, ...).

    Total absolute heading change and mean absolute curvature are related by
    support length, so only one (bendiness) is fed into the alignment-
    complexity scalar; heading change is retained purely as a diagnostic.
    """

    cfg = config or GeometryComplexityConfig()
    s, kappa, weight = profile["s"], profile["kappa"], profile["weight"]
    if s.size < 4:
        return None
    valid = weight >= cfg.min_sample_reliability
    if np.count_nonzero(valid) < max(4, cfg.curvature_samples_min // 3):
        return None
    s_v, k_v, w_v = s[valid], kappa[valid], weight[valid]
    order = np.argsort(s_v)
    s_v, k_v, w_v = s_v[order], k_v[order], w_v[order]
    valid_support_length = float(s_v[-1] - s_v[0]) if s_v.size >= 2 else 0.0

    abs_k = np.abs(k_v)
    bendiness = _weighted_mean(abs_k, w_v)
    p95 = _weighted_quantile(abs_k, w_v, 0.95)

    # The curvature gradient dkappa/ds is estimated with the same wide,
    # bounded local baseline used for curvature itself (rather than an
    # immediate-neighbor np.gradient), so it is not dominated by point-level
    # noise left over after despiking/smoothing (curvature is already a
    # second derivative of position; differencing it again at the finest
    # sample spacing would otherwise amplify residual noise further).
    window = int(profile.get("menger_window", 1)) if hasattr(profile, "get") else 1
    if s_v.size >= 2 * window + 1:
        index = np.arange(s_v.size)
        lo = np.clip(index - window, 0, s_v.size - 1)
        hi = np.clip(index + window, 0, s_v.size - 1)
        grad_valid = (index - window >= 0) & (index + window < s_v.size)
        ds_span = s_v[hi] - s_v[lo]
        grad_valid &= ds_span > EPS
        dkappa = np.where(grad_valid, (k_v[hi] - k_v[lo]) / np.maximum(ds_span, EPS), 0.0)
    elif s_v.size >= 3:
        dkappa = np.gradient(k_v, s_v)
        grad_valid = np.ones_like(k_v, dtype=bool)
    else:
        dkappa = np.zeros_like(k_v)
        grad_valid = np.zeros_like(k_v, dtype=bool)
    abs_dk = np.abs(dkappa)
    gradient_weight = w_v * grad_valid.astype(np.float64)
    variation = _weighted_mean(abs_dk, gradient_weight)
    g95 = _weighted_quantile(abs_dk, gradient_weight, 0.95)
    heading_change = float(np.trapz(abs_k, s_v)) if s_v.size >= 2 else float((bendiness or 0.0) * valid_support_length)
    mean_k = _weighted_mean(k_v, w_v) or 0.0
    rms = float(np.sqrt(max(_weighted_mean(k_v * k_v, w_v) or 0.0, 0.0)))
    std = float(np.sqrt(max(_weighted_mean((k_v - mean_k) ** 2, w_v) or 0.0, 0.0)))
    reversal_count, reversal_events = _count_reversals(s_v, k_v, w_v, cfg, tau_bendiness)
    reversal_density = reversal_count / max(valid_support_length, EPS)

    return {
        "bendiness": bendiness,
        "p95_curvature": p95,
        "curvature_variation": variation,
        "p95_curvature_gradient": g95,
        "heading_change": heading_change,
        "curvature_rms": rms,
        "curvature_std": std,
        "reversal_count": reversal_count,
        "reversal_density": reversal_density,
        "reversal_events": reversal_events,
        "valid_support_length": valid_support_length,
        "_s": s_v,
        "_kappa": k_v,
        "_weight": w_v,
    }


def segment_geometry_primitives(
    components: Mapping[str, Any], *, tau_bendiness: float, config: Optional[GeometryComplexityConfig] = None
) -> dict[str, Any]:
    """Fit straight / constant-curvature / linear-changing-curvature (clothoid)
    / one-change-point piecewise curvature hypotheses to kappa(s) using a
    BIC-penalized weighted-least-squares model comparison. Returns model
    scores and the selection margin; classification (including the
    reverse-curve override) happens in :func:`classify_alignment_profile`.
    """

    cfg = config or GeometryComplexityConfig()
    s_v, k_v, w_v = components["_s"], components["_kappa"], components["_weight"]
    n_eff = max(float(np.sum(w_v) / max(np.max(w_v), EPS)), 2.0)
    s_norm = (s_v - s_v[0]) / max(s_v[-1] - s_v[0], EPS)

    def bic_of(residual: np.ndarray, weights: np.ndarray, params: int) -> float:
        loss = float(np.sum(weights * residual * residual) / max(np.sum(weights), EPS))
        return n_eff * np.log(max(loss, 1e-12)) + params * np.log(n_eff)

    models: dict[str, dict[str, Any]] = {}
    models["straight"] = {"bic": bic_of(k_v, w_v, 0), "params": 0, "expected": np.zeros_like(k_v), "change_points": []}

    b = _weighted_mean(k_v, w_v) or 0.0
    models["constant_curve"] = {
        "bic": bic_of(k_v - b, w_v, 1),
        "params": 1,
        "expected": np.full_like(k_v, b),
        "change_points": [],
    }

    design = np.column_stack([np.ones_like(s_norm), s_norm])
    coeff = _weighted_lstsq(design, k_v, w_v)
    expected_lin = design @ coeff
    models["transition_curve"] = {
        "bic": bic_of(k_v - expected_lin, w_v, 2),
        "params": 2,
        "expected": expected_lin,
        "change_points": [],
    }

    min_support = max(4, int(cfg.primitive_min_support_fraction * s_v.size))
    if s_v.size >= 2 * min_support:
        best = None
        for change in np.linspace(cfg.change_point_min_fraction, cfg.change_point_max_fraction, cfg.change_point_candidates):
            left_n = int(np.count_nonzero(s_norm <= change))
            right_n = int(np.count_nonzero(s_norm > change))
            if left_n < min_support or right_n < min_support:
                continue
            design_pw = np.column_stack([np.ones_like(s_norm), s_norm, np.maximum(s_norm - change, 0.0)])
            coeff_pw = _weighted_lstsq(design_pw, k_v, w_v)
            expected_pw = design_pw @ coeff_pw
            bic_pw = bic_of(k_v - expected_pw, w_v, 4)
            if best is None or bic_pw < best["bic"]:
                best = {"bic": bic_pw, "params": 4, "expected": expected_pw, "change_points": [float(change)]}
        if best is not None:
            models["piecewise"] = best

    # A more flexible model may always reduce raw SSE somewhat, even for a
    # perfectly constant-curvature curve, purely from spline-approximation
    # noise. Only let "transition_curve"/"piecewise" win the selection when
    # the curvature change they claim is large enough, relative to the
    # active measurement space's reference scale, to be geometrically
    # meaningful rather than a fitting artifact.
    change_threshold = cfg.min_curvature_change_fraction * max(tau_bendiness, EPS)
    candidates = {"straight": models["straight"], "constant_curve": models["constant_curve"]}
    if "transition_curve" in models:
        change = float(abs(models["transition_curve"]["expected"][-1] - models["transition_curve"]["expected"][0]))
        if change >= change_threshold:
            candidates["transition_curve"] = models["transition_curve"]
    if "piecewise" in models:
        change = float(np.ptp(models["piecewise"]["expected"]))
        if change >= change_threshold:
            candidates["piecewise"] = models["piecewise"]

    ordered = sorted(candidates.items(), key=lambda item: item[1]["bic"])
    best_name, best_model = ordered[0]
    margin = None if len(ordered) < 2 else float(ordered[1][1]["bic"] - ordered[0][1]["bic"])
    weighted_resid = float(np.sqrt(max(_weighted_mean((k_v - best_model["expected"]) ** 2, w_v) or 0.0, 0.0)))
    return {
        "best_name": best_name,
        "best_model": best_model,
        "model_scores": {name: float(model["bic"]) for name, model in models.items()},
        "selection_margin": margin,
        "weighted_residual": weighted_resid,
    }


def classify_alignment_profile(
    profile: Mapping[str, Any],
    components: Mapping[str, Any],
    *,
    tau_bendiness: float,
    config: Optional[GeometryComplexityConfig] = None,
) -> dict[str, Any]:
    """Classify the visible curvature profile; describes geometry, not design intent."""

    cfg = config or GeometryComplexityConfig()
    _ = profile
    segmentation = segment_geometry_primitives(components, tau_bendiness=tau_bendiness, config=cfg)
    best_name = segmentation["best_name"]
    best_model = segmentation["best_model"]
    margin = segmentation["selection_margin"]
    weighted_resid = segmentation["weighted_residual"]
    bendiness = components.get("bendiness") or 0.0
    reversal_count = components.get("reversal_count", 0)
    valid_fraction = float(np.clip(components["_weight"].mean() if components["_weight"].size else 0.0, 0.0, 1.0))

    if bendiness <= cfg.straight_bendiness_fraction * tau_bendiness and reversal_count == 0:
        profile_class = "straight"
        primitive_count = 1
    elif reversal_count >= 1:
        profile_class = "reverse_curve"
        primitive_count = reversal_count + 1
    elif best_name == "constant_curve":
        profile_class = "constant_curve"
        primitive_count = 1
    elif best_name == "transition_curve":
        profile_class = "transition_curve"
        primitive_count = 1
    elif best_name == "piecewise":
        expected = best_model["expected"]
        same_sign = bool(np.sum(expected >= 0) >= 0.85 * expected.size) or bool(np.sum(expected <= 0) >= 0.85 * expected.size)
        profile_class = "compound_curve" if same_sign else "irregular_or_noisy"
        primitive_count = 2
    else:
        profile_class = "straight"
        primitive_count = 1

    if (
        profile_class not in ("straight", "reverse_curve")
        and margin is not None
        and margin < cfg.primitive_bic_margin_min
        and weighted_resid > cfg.irregular_residual_fraction * max(tau_bendiness, EPS)
    ):
        profile_class = "irregular_or_noisy"
        primitive_count = 0

    margin_term = 0.0 if margin is None else float(np.clip(margin / max(cfg.primitive_bic_margin_min, EPS), 0.0, 1.0))
    residual_term = float(np.clip(1.0 - weighted_resid / max(tau_bendiness, EPS), 0.0, 1.0))
    profile_confidence = float(np.clip(0.30 * valid_fraction + 0.35 * margin_term + 0.35 * residual_term, 0.0, 1.0))

    return {
        "profile_class": profile_class,
        "primitive_count": int(primitive_count),
        "model_scores": segmentation["model_scores"],
        "selection_margin": margin,
        "change_points": best_model.get("change_points", []),
        "profile_confidence": profile_confidence,
    }


def normalize_complexity_components(
    components: Mapping[str, Any], *, tau: Mapping[str, float], config: Optional[GeometryComplexityConfig] = None
) -> dict[str, Optional[float]]:
    """Versioned saturating map n_j = 1 - exp(-f_j / tau_j) per component."""

    _ = config

    def sat(value: Optional[float], reference: Optional[float]) -> Optional[float]:
        if value is None or reference is None or reference <= 0:
            return None
        return float(1.0 - np.exp(-max(value, 0.0) / reference))

    return {
        "bendiness": sat(components.get("bendiness"), tau.get("bendiness")),
        "p95_curvature": sat(components.get("p95_curvature"), tau.get("p95_curvature")),
        "curvature_variation": sat(components.get("curvature_variation"), tau.get("curvature_variation")),
        "reversal_density": sat(components.get("reversal_density"), tau.get("reversal_density")),
    }


def aggregate_component_rms(
    normalized: Mapping[str, Optional[float]], config: Optional[GeometryComplexityConfig] = None
) -> Optional[float]:
    """Combine normalized components with an equal-weighted RMS by default."""

    cfg = config or GeometryComplexityConfig()
    weights = {
        "bendiness": cfg.component_weight_bendiness,
        "p95_curvature": cfg.component_weight_p95_curvature,
        "curvature_variation": cfg.component_weight_curvature_variation,
        "reversal_density": cfg.component_weight_reversal_density,
    }
    numerator = 0.0
    denominator = 0.0
    for key, weight in weights.items():
        value = normalized.get(key)
        if value is None:
            continue
        numerator += weight * value * value
        denominator += weight
    return None if denominator <= 0 else float(np.sqrt(numerator / denominator))


# ---------------------------------------------------------------------------
# Per-lane orchestration, aggregation, and topology.
# ---------------------------------------------------------------------------

def _analyze_single_lane(
    lane_index: int,
    centerline: Mapping[str, Any],
    *,
    image_shape: Sequence[int],
    i5_space: Mapping[str, Any],
    object_mask: Any,
    tau: Mapping[str, float],
    config: GeometryComplexityConfig,
    debug: bool,
) -> dict[str, Any]:
    base = {
        "lane_index": lane_index,
        "source_boundary_ids": centerline.get("source_boundary_ids"),
        "representation_type": centerline.get("representation_type"),
        "measurement_space": i5_space["i5_measurement_space"],
        "metric_scale_available": i5_space["metric_scale_available"],
    }

    def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
        return {
            **base,
            "profile_class": "unknown",
            "profile_confidence": 0.0,
            "complexity_score": None,
            "complexity_confidence": 0.0,
            "bendiness": None,
            "mean_abs_curvature": None,
            "p95_abs_curvature": None,
            "curvature_variation": None,
            "p95_curvature_gradient": None,
            "reversal_count": None,
            "reversal_density": None,
            "total_abs_heading_change": None,
            "tortuosity": None,
            "primitive_count": 0,
            "valid_support_length": None,
            "valid_sample_fraction": 0.0,
            "extrapolated_fraction": 0.0,
            "unavailable_reason": reason,
            "notes": [reason],
            **extra,
        }

    points_analysis, _scale = transform_to_road_plane(centerline, image_shape=image_shape, i5_space=i5_space)
    image_points = np.asarray(centerline["points_image"], dtype=np.float64)
    fit = robustly_fit_parametric_curve(points_analysis, aux_points=image_points, config=config)
    if fit is None:
        return unavailable("insufficient_support_for_curve_fit")

    profile = calculate_curvature_profile(fit, image_shape=image_shape, object_mask=object_mask, config=config)
    components = calculate_alignment_components(profile, tau_bendiness=tau["bendiness"], config=config)
    if components is None:
        return unavailable(
            "insufficient_reliable_curvature_samples",
            source_point_count=fit["source_point_count"],
            inlier_fraction=fit["inlier_fraction"],
            fit_rmse=fit["robust_residual"],
        )

    classification = classify_alignment_profile(profile, components, tau_bendiness=tau["bendiness"], config=config)
    normalized = normalize_complexity_components(components, tau=tau, config=config)
    complexity_score = aggregate_component_rms(normalized, config=config)

    chord = float(
        np.hypot(
            fit["x_spline"](fit["s_max"]) - fit["x_spline"](fit["s_min"]),
            fit["y_spline"](fit["s_max"]) - fit["y_spline"](fit["s_min"]),
        )
    )
    tortuosity = None if chord <= EPS else float(fit["support"] / chord - 1.0)

    fit_confidence = float(np.clip(fit["inlier_fraction"], 0.0, 1.0))
    complexity_confidence = float(
        np.clip(
            0.35 * fit_confidence
            + 0.35 * classification["profile_confidence"]
            + 0.30 * float(np.clip(profile.get("valid_fraction", 0.0), 0.0, 1.0)),
            0.0,
            1.0,
        )
    )

    result = {
        **base,
        "profile_class": classification["profile_class"],
        "profile_confidence": classification["profile_confidence"],
        "complexity_score": complexity_score,
        "complexity_confidence": complexity_confidence,
        "bendiness": components["bendiness"],
        "mean_abs_curvature": components["bendiness"],
        "p95_abs_curvature": components["p95_curvature"],
        "curvature_variation": components["curvature_variation"],
        "p95_curvature_gradient": components["p95_curvature_gradient"],
        "reversal_count": components["reversal_count"],
        "reversal_density": components["reversal_density"],
        "total_abs_heading_change": components["heading_change"],
        "tortuosity": tortuosity,
        "curvature_rms": components["curvature_rms"],
        "curvature_std": components["curvature_std"],
        "primitive_count": classification["primitive_count"],
        "model_scores": classification["model_scores"],
        "selection_margin": classification["selection_margin"],
        "change_points": classification["change_points"],
        "source_point_count": fit["source_point_count"],
        "inlier_fraction": fit["inlier_fraction"],
        "fit_rmse": fit["robust_residual"],
        "valid_support_length": components["valid_support_length"],
        "valid_sample_fraction": profile.get("valid_fraction"),
        "extrapolated_fraction": 0.0,
        "unavailable_reason": None,
        "notes": [],
    }
    if debug:
        result["debug"] = {
            "s": profile["s"],
            "kappa": profile["kappa"],
            "abs_kappa": np.abs(profile["kappa"]),
            "weight": profile["weight"],
            "image_xy": profile.get("image_xy"),
        }
    return result


def aggregate_lane_complexity(
    lane_results: Sequence[Mapping[str, Any]], config: Optional[GeometryComplexityConfig] = None
) -> dict[str, Any]:
    """Confidence- and bounded-support-aware aggregation across usable lanes.

    Unavailable lanes are never averaged in as zero, and one very long lane
    cannot dominate the aggregate without bound.
    """

    cfg = config or GeometryComplexityConfig()
    valid = [lane for lane in lane_results if lane.get("complexity_score") is not None]
    unknown_count = len(lane_results) - len(valid)
    if not valid:
        return {
            "alignment_complexity": None,
            "complexity_confidence": 0.0,
            "median_lane_complexity": None,
            "max_lane_complexity": None,
            "valid_lane_count": 0,
            "unknown_lane_count": int(unknown_count),
            "aggregation_method": "confidence_x_sqrt_bounded_support",
            "unavailable_reason": "no_valid_lane_geometry_complexity" if lane_results else "no_credible_boundary_geometry",
        }
    supports = np.asarray([max(float(lane.get("valid_support_length") or 0.0), EPS) for lane in valid])
    cap = max(float(np.median(supports)) * cfg.aggregation_support_cap_ratio, EPS)
    weights = np.asarray(
        [
            max(float(lane.get("complexity_confidence") or 0.0), cfg.aggregation_min_confidence) * np.sqrt(min(s, cap))
            for lane, s in zip(valid, supports)
        ]
    )
    if float(np.sum(weights)) <= EPS:
        weights = np.ones(len(valid))
    scores = np.asarray([float(lane["complexity_score"]) for lane in valid])
    aggregate = float(np.sum(weights * scores) / np.sum(weights))
    confidences = np.asarray([float(lane.get("complexity_confidence") or 0.0) for lane in valid])
    aggregate_confidence = float(np.sum(weights * confidences) / np.sum(weights))
    return {
        "alignment_complexity": aggregate,
        "complexity_confidence": aggregate_confidence,
        "median_lane_complexity": float(np.median(scores)),
        "max_lane_complexity": float(np.max(scores)),
        "valid_lane_count": len(valid),
        "unknown_lane_count": int(unknown_count),
        "aggregation_method": "confidence_x_sqrt_bounded_support",
        "unavailable_reason": None,
    }


def infer_lane_topology(
    centerline_result: Mapping[str, Any], *, image_shape: Sequence[int], config: Optional[GeometryComplexityConfig] = None
) -> dict[str, Any]:
    """Conservative, separate diagnostic of supported lane-count changes.

    Compares the number of logical boundaries with image-space support near
    the bottom of the frame against the number near the top. Never folded
    into alignment complexity and never claims certainty from one image: an
    ambiguous or under-supported case returns ``topology_complexity=None``.
    """

    cfg = config or GeometryComplexityConfig()
    diagnostics = centerline_result.get("logical_boundary_diagnostics") or []
    usable = [d for d in diagnostics if d.get("image_y_at_t_min") is not None and d.get("image_y_at_t_max") is not None]
    if len(usable) < 2:
        return {
            "topology_complexity": None,
            "supported_split_events": 0,
            "supported_merge_events": 0,
            "branch_node_count": 0,
            "max_graph_degree": len(usable),
            "lane_count_near": None,
            "lane_count_far": None,
            "unavailable_reason": "insufficient_boundary_evidence_for_topology",
        }
    height = float(image_shape[0])
    near_threshold = (1.0 - cfg.topology_band_fraction) * height
    far_threshold = cfg.topology_band_fraction * height
    near_count = 0
    far_count = 0
    for d in usable:
        y_lo = min(d["image_y_at_t_min"], d["image_y_at_t_max"])
        y_hi = max(d["image_y_at_t_min"], d["image_y_at_t_max"])
        if y_hi >= near_threshold:
            near_count += 1
        if y_lo <= far_threshold:
            far_count += 1
    if near_count < 2 or far_count < 2:
        return {
            "topology_complexity": None,
            "supported_split_events": 0,
            "supported_merge_events": 0,
            "branch_node_count": 0,
            "max_graph_degree": max(near_count, far_count),
            "lane_count_near": near_count,
            "lane_count_far": far_count,
            "unavailable_reason": "insufficient_near_or_far_band_support",
        }
    lane_count_change = abs(near_count - far_count)
    topology_complexity = float(np.clip(lane_count_change / max(near_count, far_count, 1), 0.0, 1.0))
    return {
        "topology_complexity": topology_complexity,
        "supported_split_events": int(near_count > far_count),
        "supported_merge_events": int(far_count > near_count),
        "branch_node_count": int(lane_count_change),
        "max_graph_degree": int(max(near_count, far_count)),
        "lane_count_near": near_count,
        "lane_count_far": far_count,
        "unavailable_reason": None,
    }


def _aggregate_profile_class(lane_results: Sequence[Mapping[str, Any]]) -> str:
    classes = {lane["profile_class"] for lane in lane_results if lane.get("complexity_score") is not None}
    if not classes:
        return "unknown"
    if len(classes) == 1:
        return next(iter(classes))
    return "mixed"


def _field_mean(lane_results: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [lane[key] for lane in lane_results if lane.get("complexity_score") is not None and lane.get(key) is not None]
    return float(np.mean(values)) if values else None


def _field_sum(lane_results: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [lane[key] for lane in lane_results if lane.get("complexity_score") is not None and lane.get(key) is not None]
    return float(np.sum(values)) if values else None


def _transform_diagnostics(measurement: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if not measurement:
        return None
    return {
        "measurement_space": measurement.get("measurement_space"),
        "homography_source": measurement.get("homography_source"),
        "homography_condition_number": measurement.get("homography_condition_number"),
        "valid_transformed_area": measurement.get("valid_transformed_area"),
        "metric_scale_available": measurement.get("metric_scale_available"),
        "metric_scale_confidence": measurement.get("metric_scale_confidence"),
        "far_field_exclusion": measurement.get("far_field_exclusion"),
        "fallback_reason": measurement.get("fallback_reason"),
    }


def _public_lane_diagnostic(lane: Mapping[str, Any], *, debug: bool) -> dict[str, Any]:
    result = {key: value for key, value in lane.items() if not key.startswith("_") and key != "debug"}
    if debug and "debug" in lane:
        result["debug"] = lane["debug"]
    return result


def _unavailable_result(
    geometry_source: str,
    reason: str,
    *,
    measurement: Optional[Mapping[str, Any]] = None,
    i5_space: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    space = (i5_space or {}).get("i5_measurement_space", "unavailable")
    return _json_safe(
        {
            "alignment_complexity": None,
            "geometry_complexity": None,
            "topology_complexity": None,
            "profile_class": "unknown",
            "complexity_confidence": 0.0,
            "geometry_source": geometry_source,
            "measurement_space": space,
            "metric_scale_available": (i5_space or {}).get("metric_scale_available", False),
            "bendiness": None,
            "mean_abs_curvature": None,
            "p95_abs_curvature": None,
            "curvature_variation": None,
            "p95_curvature_gradient": None,
            "reversal_count": None,
            "reversal_density": None,
            "total_abs_heading_change": None,
            "tortuosity": None,
            "primitive_count": 0,
            "valid_lane_count": 0,
            "unknown_lane_count": 0,
            "median_lane_complexity": None,
            "max_lane_complexity": None,
            "aggregation_method": "confidence_x_sqrt_bounded_support",
            "lane_diagnostics": [],
            "transform_diagnostics": _transform_diagnostics(measurement),
            "unavailable_reason": reason,
        }
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def analyze_lane_geometry_complexity(
    lane_geometries: Optional[Iterable[Any]] = None,
    *,
    image_shape: Sequence[int],
    geometry_source: str,
    image_rgb: Any = None,
    road_mask: Any = None,
    object_mask: Any = None,
    homography: Any = None,
    camera_model: Any = None,
    calibration_meta: Any = None,
    lane_width_context: Optional[Mapping[str, Any]] = None,
    config: Optional[GeometryComplexityConfig] = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Analyze visible plan-view lane-alignment geometry complexity.

    ``lane_geometries`` must already be the caller's highest-priority credible
    geometry representation (native polylines, lane_json-derived polylines, or
    mask-grouped centerlines) — the same priority-selection convention used by
    canonical/operational I1 and I4 (see ``_select_i1_guidance`` in
    :mod:`evaluation.readiness_metrics`). This function does not re-select
    among geometry sources itself, so canonical and prediction-guided callers
    reuse exactly the same selected geometry as I1/I4 for a given image
    rather than re-deriving it.

    ``geometry_source`` records provenance only (e.g. ``"ground_truth"``,
    ``"native_polyline"``, ``"mask_derived"``); this function itself never
    inspects it to change behavior. Callers must pass either ground-truth
    geometry or one evaluated model's geometry — never both — to keep
    canonical and prediction-guided I5 independent.

    ``image_rgb`` is accepted for interface symmetry with the debug-
    visualization utilities and is not used for geometry (I5 is classical
    geometry, not appearance).
    """

    _ = image_rgb
    cfg = config or GeometryComplexityConfig()
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 1 or width <= 1:
        return _unavailable_result(geometry_source, "invalid_image_shape")

    selected = list(lane_geometries or [])
    if not selected:
        return _unavailable_result(geometry_source, "no_credible_geometry")

    raw_measurement = determine_measurement_space(
        image_shape=(height, width), homography=homography, camera_model=camera_model, calibration_meta=calibration_meta
    )
    i5_space = determine_i5_measurement_space(raw_measurement, lane_width_context=lane_width_context, config=cfg)

    centerline_result = pair_boundaries_and_construct_centerlines(
        selected,
        image_shape=(height, width),
        road_mask=road_mask,
        homography=homography,
        camera_model=camera_model,
        calibration_meta=calibration_meta,
        config=cfg,
    )
    centerlines = centerline_result.get("centerlines") or []
    if not centerlines:
        return _unavailable_result(
            geometry_source,
            "no_credible_boundary_geometry",
            measurement=raw_measurement,
            i5_space=i5_space,
        )

    tau = _tau_reference_for_space(i5_space["i5_measurement_space"], cfg)
    lane_results = [
        _analyze_single_lane(
            lane_index,
            centerline,
            image_shape=(height, width),
            i5_space=i5_space,
            object_mask=object_mask,
            tau=tau,
            config=cfg,
            debug=debug,
        )
        for lane_index, centerline in enumerate(centerlines)
    ]

    aggregate = aggregate_lane_complexity(lane_results, config=cfg)
    topology = infer_lane_topology(centerline_result, image_shape=(height, width), config=cfg)

    result = {
        "alignment_complexity": aggregate.get("alignment_complexity"),
        "geometry_complexity": aggregate.get("alignment_complexity"),
        "topology_complexity": topology.get("topology_complexity"),
        "profile_class": _aggregate_profile_class(lane_results),
        "complexity_confidence": aggregate.get("complexity_confidence", 0.0),
        "geometry_source": geometry_source,
        "measurement_space": i5_space["i5_measurement_space"],
        "metric_scale_available": i5_space["metric_scale_available"],
        "bendiness": _field_mean(lane_results, "bendiness"),
        "mean_abs_curvature": _field_mean(lane_results, "bendiness"),
        "p95_abs_curvature": _field_mean(lane_results, "p95_abs_curvature"),
        "curvature_variation": _field_mean(lane_results, "curvature_variation"),
        "p95_curvature_gradient": _field_mean(lane_results, "p95_curvature_gradient"),
        "reversal_count": _field_sum(lane_results, "reversal_count"),
        "reversal_density": _field_mean(lane_results, "reversal_density"),
        "total_abs_heading_change": _field_sum(lane_results, "total_abs_heading_change"),
        "tortuosity": _field_mean(lane_results, "tortuosity"),
        "primitive_count": _field_sum(lane_results, "primitive_count"),
        "valid_lane_count": aggregate.get("valid_lane_count", 0),
        "unknown_lane_count": aggregate.get("unknown_lane_count", 0),
        "median_lane_complexity": aggregate.get("median_lane_complexity"),
        "max_lane_complexity": aggregate.get("max_lane_complexity"),
        "aggregation_method": aggregate.get("aggregation_method"),
        "lane_diagnostics": [_public_lane_diagnostic(lane, debug=debug) for lane in lane_results],
        "topology_supported_split_events": topology.get("supported_split_events"),
        "topology_supported_merge_events": topology.get("supported_merge_events"),
        "topology_branch_node_count": topology.get("branch_node_count"),
        "topology_max_graph_degree": topology.get("max_graph_degree"),
        "topology_lane_count_near": topology.get("lane_count_near"),
        "topology_lane_count_far": topology.get("lane_count_far"),
        "topology_unavailable_reason": topology.get("unavailable_reason"),
        "transform_diagnostics": _transform_diagnostics(raw_measurement),
        "rejected_boundary_reasons": centerline_result.get("rejected_boundary_reasons"),
        "rejected_pair_reasons": centerline_result.get("rejected_pair_reasons"),
        "double_line_groups": centerline_result.get("double_line_groups"),
        "unavailable_reason": aggregate.get("unavailable_reason"),
    }
    if debug:
        result["config"] = asdict(cfg)
    return _json_safe(result)


__all__ = [
    "GEOMETRY_COMPLEXITY_VERSION",
    "GeometryComplexityConfig",
    "aggregate_component_rms",
    "aggregate_lane_complexity",
    "analyze_lane_geometry_complexity",
    "calculate_alignment_components",
    "calculate_curvature_profile",
    "classify_alignment_profile",
    "determine_i5_measurement_space",
    "determine_measurement_space",
    "extract_grouped_centerlines_from_mask",
    "infer_lane_topology",
    "normalize_complexity_components",
    "pair_boundaries_and_construct_centerlines",
    "robustly_fit_parametric_curve",
    "segment_geometry_primitives",
    "select_lane_geometry",
    "transform_to_road_plane",
    "validate_lane_geometry",
]
