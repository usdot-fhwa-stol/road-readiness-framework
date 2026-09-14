"""Behavioral tests for the classical I5 lane-alignment geometry-complexity engine.

Synthetic curves are generated independently (world/BEV curvature integration,
piecewise curvature segments, or a synthetic pinhole ground-plane homography)
rather than derived from one shared polyline, so tests are not circular.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from evaluation.lane_geometry_complexity import (
    GeometryComplexityConfig,
    aggregate_lane_complexity,
    analyze_lane_geometry_complexity,
    calculate_alignment_components,
    calculate_curvature_profile,
    determine_i5_measurement_space,
    extract_grouped_centerlines_from_mask,
    infer_lane_topology,
    pair_boundaries_and_construct_centerlines,
    robustly_fit_parametric_curve,
)
from evaluation.readiness_metrics import (
    build_metric_record,
    compute_correlations_from_records,
    compute_i5_geometry_complexity,
    compute_i5_geometry_complexity_details,
    compute_i5_legacy_quadratic_proxy,
    compute_r1_marking_readability_score,
)
from lane_eval.manifest import PredictionManifestReader, PredictionManifestWriter
from lane_eval.schema.lane import LaneTarget
from lane_eval.schema.sample import LaneSample


HEIGHT = 700
WIDTH = 500
Y0 = 380.0
Y1 = 690.0


# ---------------------------------------------------------------------------
# Synthetic geometry generators (independent of the analysis code).
# ---------------------------------------------------------------------------

def _piecewise_curvature_points(
    segments: list[tuple[float, float]],
    *,
    n_per_unit: float = 0.6,
    x0: float = 220.0,
    y0: float = Y0,
    heading0: float = np.pi / 2.0,
) -> np.ndarray:
    """Integrate a piecewise-constant (or, via many tiny segments, smoothly
    varying) curvature profile kappa(s) into a plan-view polyline. This
    generates geometry directly from a *curvature specification*, independent
    of any spline-fitting or projection code under test.
    """

    total_length = sum(length for _, length in segments)
    n = max(int(total_length * n_per_unit), 80)
    s = np.linspace(0.0, total_length, n)
    ds = s[1] - s[0]
    kappa = np.zeros(n)
    cursor = 0.0
    for k, length in segments:
        in_segment = (s >= cursor - 1e-9) & (s <= cursor + length + 1e-9)
        kappa[in_segment] = k
        cursor += length
    heading = heading0 + np.concatenate([[0.0], np.cumsum(0.5 * (kappa[:-1] + kappa[1:]) * ds)])
    x = x0 + np.concatenate([[0.0], np.cumsum(np.cos(heading[:-1]) * ds)])
    y = y0 + np.concatenate([[0.0], np.cumsum(np.sin(heading[:-1]) * ds)])
    return np.column_stack([x, y])


def _straight_points(span: float = 300.0, n: int = 90, x0: float = 220.0, y0: float = Y0) -> np.ndarray:
    return _piecewise_curvature_points([(0.0, span)], n_per_unit=n / span, x0=x0, y0=y0)


def _arc_points(radius: float, span: float = 300.0, n: int = 90, x0: float = 220.0, y0: float = Y0) -> np.ndarray:
    return _piecewise_curvature_points([(1.0 / radius, span)], n_per_unit=n / span, x0=x0, y0=y0)


def _clothoid_points(k0: float, k1: float, span: float = 300.0, n: int = 120, x0: float = 220.0, y0: float = Y0) -> np.ndarray:
    n = max(n, 80)
    segments = []
    steps = 40
    step_len = span / steps
    for i in range(steps):
        frac = (i + 0.5) / steps
        segments.append((k0 + (k1 - k0) * frac, step_len))
    return _piecewise_curvature_points(segments, n_per_unit=n / span, x0=x0, y0=y0)


def _s_curve_points(radius: float, span: float = 300.0, n: int = 120, x0: float = 220.0, y0: float = Y0) -> np.ndarray:
    half = span / 2.0
    return _piecewise_curvature_points([(1.0 / radius, half), (-1.0 / radius, half)], n_per_unit=n / span, x0=x0, y0=y0)


def _compound_points(r1: float, r2: float, span: float = 300.0, n: int = 120, x0: float = 220.0, y0: float = Y0) -> np.ndarray:
    half = span / 2.0
    return _piecewise_curvature_points([(1.0 / r1, half), (1.0 / r2, half)], n_per_unit=n / span, x0=x0, y0=y0)


def _rasterize(points: np.ndarray, *, height: int = HEIGHT, width: int = WIDTH, thickness: int = 5, dashed: bool = False) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    if not dashed:
        cv2.polylines(mask, [pts], False, 1, thickness, cv2.LINE_8)
        return mask
    for start in range(0, len(pts) - 6, 6):
        if (start // 6) % 2 == 0:
            cv2.polylines(mask, [pts[start:start + 4]], False, 1, thickness, cv2.LINE_8)
    return mask


def _ground_plane_homography(focal: float, cx: float, cy: float, cam_height: float, tilt_rad: float) -> np.ndarray:
    """Synthetic pinhole camera projecting a flat ground plane (X, Z) -> (u, v).

    Returns the road-to-image homography; callers invert it for the
    image-to-road homography this module's ``homography`` parameter expects.
    """

    row0 = [focal, 0.0, 0.0]
    row1 = [0.0, -focal * np.sin(tilt_rad), focal * cam_height * np.cos(tilt_rad)]
    row2 = [0.0, np.cos(tilt_rad), cam_height * np.sin(tilt_rad)]
    base = np.array([row0, row1, row2], dtype=np.float64)
    k_matrix = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return k_matrix @ base


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points, np.ones(points.shape[0])])
    out = hom @ matrix.T
    return out[:, :2] / out[:, 2:3]


def _world_arc(radius: float, z0: float = 12.0, span: float = 40.0, n: int = 90) -> np.ndarray:
    """World (lateral X meters, forward Z meters) constant-curvature arc."""

    theta = np.linspace(0.0, span / radius, n)
    x = radius * (1.0 - np.cos(theta))
    z = z0 + radius * np.sin(theta)
    return np.column_stack([x, z])


def _sample(image_shape: tuple[int, int], mask: np.ndarray, lane: np.ndarray, **meta) -> LaneSample:
    h, w = image_shape
    return LaneSample(
        image_id="synthetic",
        image_path="/tmp/i5_synthetic.png",
        width=w,
        height=h,
        target=LaneTarget(mask=mask, lanes=[lane], meta=meta),
    )


def _analyze(lanes, image_shape=(HEIGHT, WIDTH), geometry_source="ground_truth", **kwargs) -> dict:
    return analyze_lane_geometry_complexity(lanes, image_shape=image_shape, geometry_source=geometry_source, **kwargs)


# ---------------------------------------------------------------------------
# 1-3: straight vs constant-radius arcs, radius ordering.
# ---------------------------------------------------------------------------

def test_straight_line_has_low_alignment_complexity():
    result = _analyze([_straight_points()])
    assert result["profile_class"] == "straight"
    assert result["alignment_complexity"] < 0.02


def test_constant_radius_arc_has_higher_complexity_than_straight():
    straight = _analyze([_straight_points()])
    arc = _analyze([_arc_points(500.0)])
    assert arc["profile_class"] == "constant_curve"
    assert arc["alignment_complexity"] > straight["alignment_complexity"] + 0.05


def test_smaller_radius_arc_has_higher_bendiness_and_peak_curvature():
    gentle = _analyze([_arc_points(2000.0)])
    sharp = _analyze([_arc_points(300.0)])
    assert sharp["bendiness"] > gentle["bendiness"]
    assert sharp["p95_abs_curvature"] > gentle["p95_abs_curvature"]
    assert sharp["alignment_complexity"] > gentle["alignment_complexity"]


# ---------------------------------------------------------------------------
# 4: point-density invariance.
# ---------------------------------------------------------------------------

def test_point_density_does_not_change_metrics_much():
    dense = _analyze([_arc_points(600.0, n=220)])
    sparse = _analyze([_arc_points(600.0, n=25)])
    assert dense["bendiness"] == pytest.approx(sparse["bendiness"], rel=0.20)
    assert dense["profile_class"] == sparse["profile_class"] == "constant_curve"


# ---------------------------------------------------------------------------
# 5: rigid translation/rotation invariance of the underlying curvature math.
# ---------------------------------------------------------------------------

def test_rigid_translation_and_rotation_do_not_change_curvature_math():
    cfg = GeometryComplexityConfig()
    pts = _arc_points(500.0, x0=0.0, y0=0.0)
    theta = np.radians(35.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = pts @ rotation.T + np.array([5000.0, -3000.0])

    # A fixed, identical, comfortably reliable image-space aux track for both
    # fits isolates the rotation/translation invariance of the curvature math
    # from the (deliberately position-dependent) horizon-reliability model.
    aux = np.column_stack([np.linspace(100.0, 110.0, pts.shape[0]), np.linspace(500.0, 650.0, pts.shape[0])])

    fit_a = robustly_fit_parametric_curve(pts, aux_points=aux, config=cfg)
    fit_b = robustly_fit_parametric_curve(rotated, aux_points=aux, config=cfg)
    profile_a = calculate_curvature_profile(fit_a, image_shape=(1000, 1000), config=cfg)
    profile_b = calculate_curvature_profile(fit_b, image_shape=(1000, 1000), config=cfg)
    comp_a = calculate_alignment_components(profile_a, tau_bendiness=cfg.tau_bendiness_image_proxy, config=cfg)
    comp_b = calculate_alignment_components(profile_b, tau_bendiness=cfg.tau_bendiness_image_proxy, config=cfg)
    assert comp_a["bendiness"] == pytest.approx(comp_b["bendiness"], rel=0.03)
    assert comp_a["curvature_variation"] == pytest.approx(comp_b["curvature_variation"], rel=0.10)


# ---------------------------------------------------------------------------
# 6: parallel offsets of the same curve.
# ---------------------------------------------------------------------------

def test_parallel_offsets_of_same_curve_have_similar_alignment_complexity():
    left = _arc_points(600.0, x0=180.0)
    right = _arc_points(600.0, x0=260.0)
    result_left = _analyze([left])
    result_right = _analyze([right])
    assert result_left["bendiness"] == pytest.approx(result_right["bendiness"], rel=0.15)


def test_paired_boundary_centerline_matches_single_offset_curvature():
    # Two parallel constant-radius boundaries (a travel lane) should produce a
    # paired centerline whose curvature matches either boundary closely.
    left = _arc_points(600.0, x0=180.0)
    right = _arc_points(600.0, x0=260.0)
    single = _analyze([left])
    paired = _analyze([left, right])
    assert paired["valid_lane_count"] == 1
    assert paired["lane_diagnostics"][0]["representation_type"] == "paired_centerline"
    assert paired["bendiness"] == pytest.approx(single["bendiness"], rel=0.15)


# ---------------------------------------------------------------------------
# 7-9: clothoid, compound curve, reverse curve.
# ---------------------------------------------------------------------------

def test_clothoid_has_greater_curvature_variation_than_constant_arc():
    mean_k = 1.0 / 600.0
    constant = _analyze([_arc_points(600.0)])
    clothoid = _analyze([_clothoid_points(mean_k * 0.15, mean_k * 1.85)])
    assert clothoid["curvature_variation"] > constant["curvature_variation"] * 2.0
    # A continuously-varying curvature ramp is fit by either a single linear
    # trend or (if a single change point happens to score better under the
    # BIC penalty) a same-sign piecewise model; either way it must not
    # collapse back to "constant_curve".
    assert clothoid["profile_class"] in {"transition_curve", "compound_curve", "irregular_or_noisy"}
    assert clothoid["profile_class"] != "constant_curve"


def test_compound_curve_is_distinguished_from_one_constant_arc():
    compound = _analyze([_compound_points(1400.0, 260.0)])
    single = _analyze([_arc_points(700.0)])
    assert compound["profile_class"] in {"compound_curve", "transition_curve"}
    assert compound["profile_class"] != "unknown"
    assert single["profile_class"] == "constant_curve"


def test_reverse_s_curve_produces_a_supported_sign_reversal():
    result = _analyze([_s_curve_points(320.0)])
    assert result["profile_class"] == "reverse_curve"
    assert result["reversal_count"] >= 1


def test_small_oscillation_around_zero_is_not_a_false_reverse_curve():
    rng = np.random.default_rng(3)
    straight = _straight_points(n=140)
    noisy = straight.copy()
    noisy[:, 0] += rng.normal(0.0, 0.35, size=noisy.shape[0])
    result = _analyze([noisy])
    assert result["profile_class"] != "reverse_curve"
    assert (result["reversal_count"] or 0) == 0


# ---------------------------------------------------------------------------
# 11-13: robust smoothing, outlier handling, no high-degree oscillation.
# ---------------------------------------------------------------------------

def test_noisy_straight_lane_remains_low_complexity_after_robust_smoothing():
    # Sub-pixel-scale i.i.d. jitter (realistic for annotation/detector
    # position noise) must not read as meaningful curvature after smoothing.
    rng = np.random.default_rng(11)
    straight = _straight_points(n=150)
    noisy = straight.copy()
    noisy[:, 0] += rng.normal(0.0, 0.2, size=noisy.shape[0])
    result = _analyze([noisy])
    assert result["alignment_complexity"] < 0.15
    assert result["profile_class"] in {"straight", "constant_curve"}


def test_sparse_large_outliers_are_downweighted_not_dominant():
    rng = np.random.default_rng(12)
    straight = _straight_points(n=150)
    clean = _analyze([straight])
    contaminated = straight.copy()
    outlier_idx = rng.choice(contaminated.shape[0], size=4, replace=False)
    # Large enough to clearly qualify as outliers relative to the ~0 baseline,
    # but small enough to survive the shared upstream I4 boundary-fit gate
    # (very large, e.g. 60px, outliers are already rejected there, before I5
    # ever sees the geometry — see test_very_short_visible_support... for the
    # "reject the whole boundary" path).
    contaminated[outlier_idx, 0] += rng.choice([-1, 1], size=4) * 12.0
    result = _analyze([contaminated])
    assert result["alignment_complexity"] < clean["alignment_complexity"] + 0.15
    assert result["lane_diagnostics"][0]["inlier_fraction"] > 0.85


def test_fit_does_not_use_a_single_high_degree_global_polynomial():
    # A high-degree polynomial interpolant would oscillate wildly outside the
    # densely-sampled region for unevenly spaced points; the spline-based fit
    # must not reproduce that behavior even with clustered, uneven spacing.
    rng = np.random.default_rng(9)
    s_uneven = np.sort(rng.uniform(0.0, 1.0, size=60)) ** 2
    span = 300.0
    ys = Y0 + s_uneven * span
    xs = 220.0 + 4.0 * np.sin(s_uneven * 3.0)
    lane = np.column_stack([xs, ys])
    result = _analyze([lane])
    assert result["unavailable_reason"] is None
    assert result["p95_abs_curvature"] < 5.0


# ---------------------------------------------------------------------------
# 14-19: perspective, calibration, horizon, ill-conditioning.
# ---------------------------------------------------------------------------

def test_calibrated_bev_recovers_consistent_curvature_despite_perspective():
    focal, cx, cy, cam_height, tilt = 800.0, 320.0, 240.0, 1.5, np.radians(8.0)
    road_to_image = _ground_plane_homography(focal, cx, cy, cam_height, tilt)
    image_to_road = np.linalg.inv(road_to_image)

    world = _world_arc(radius=60.0, z0=10.0, span=35.0)
    image_points = _project(road_to_image, world)

    calibrated = _analyze(
        [image_points],
        image_shape=(480, 640),
        homography=image_to_road,
        calibration_meta={"units": "m", "metric_scale_available": True},
    )
    assert calibrated["measurement_space"] == "calibrated_metric_bev"
    assert calibrated["metric_scale_available"] is True
    # True world curvature is 1/60 per meter; the recovered metric bendiness
    # should be in the right physical ballpark, not an arbitrary image-space
    # number.
    assert calibrated["bendiness"] == pytest.approx(1.0 / 60.0, rel=0.35)

    uncalibrated = _analyze([image_points], image_shape=(480, 640))
    assert uncalibrated["measurement_space"] == "normalized_image_proxy"
    # Raw image-space curvature is a different (and here, much larger) number
    # than the metric curvature — the whole point of calibration.
    assert uncalibrated["bendiness"] != pytest.approx(calibrated["bendiness"], rel=0.35)


def test_different_focal_length_and_resolution_give_similar_calibrated_bendiness():
    world = _world_arc(radius=80.0, z0=12.0, span=35.0)

    def calibrated_bendiness(focal, cx, cy, shape):
        road_to_image = _ground_plane_homography(focal, cx, cy, 1.5, np.radians(8.0))
        image_to_road = np.linalg.inv(road_to_image)
        image_points = _project(road_to_image, world)
        result = _analyze([image_points], image_shape=shape, homography=image_to_road, calibration_meta={"units": "m", "metric_scale_available": True})
        return result

    result_a = calibrated_bendiness(800.0, 320.0, 240.0, (480, 640))
    result_b = calibrated_bendiness(1400.0, 640.0, 480.0, (960, 1280))
    assert result_a["measurement_space"] == result_b["measurement_space"] == "calibrated_metric_bev"
    assert result_a["bendiness"] == pytest.approx(result_b["bendiness"], rel=0.10)


def test_camera_pitch_does_not_dominate_calibrated_complexity():
    # Kept close to the camera (z0=5m, 18m span) so the projected arc stays
    # within the reliable (non-horizon) image band across both tilt angles.
    world = _world_arc(radius=90.0, z0=5.0, span=18.0)

    def calibrated_bendiness(tilt_deg):
        road_to_image = _ground_plane_homography(800.0, 320.0, 240.0, 1.5, np.radians(tilt_deg))
        image_to_road = np.linalg.inv(road_to_image)
        image_points = _project(road_to_image, world)
        return _analyze(
            [image_points], image_shape=(480, 640), homography=image_to_road, calibration_meta={"units": "m", "metric_scale_available": True}
        )

    low_tilt = calibrated_bendiness(6.0)
    high_tilt = calibrated_bendiness(10.0)
    assert low_tilt["bendiness"] is not None and high_tilt["bendiness"] is not None
    assert low_tilt["bendiness"] == pytest.approx(high_tilt["bendiness"], rel=0.20)


def test_uncalibrated_geometry_is_labeled_image_proxy_without_metric_units():
    result = _analyze([_arc_points(500.0)])
    assert result["measurement_space"] == "normalized_image_proxy"
    assert result["metric_scale_available"] is False


def test_ill_conditioned_homography_falls_back_to_uncalibrated():
    lane = _arc_points(500.0)
    degenerate_homography = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = _analyze([lane], homography=degenerate_homography)
    assert result["measurement_space"] != "calibrated_metric_bev"
    assert result["measurement_space"] != "calibrated_scale_free_bev"


def test_far_horizon_samples_are_confidence_gated():
    # A lane placed almost entirely near the top of the frame (the horizon
    # region) should have low valid-sample support even though the geometry
    # itself is a clean arc.
    lane = _arc_points(500.0, y0=10.0, span=120.0)
    near_horizon = _analyze([lane], image_shape=(HEIGHT, WIDTH))
    assert (
        near_horizon["unavailable_reason"] is not None
        or near_horizon["lane_diagnostics"][0].get("valid_sample_fraction", 1.0) < 0.5
    )


# ---------------------------------------------------------------------------
# 20-22: mask thickness, dashed grouping, native polyline preservation.
# ---------------------------------------------------------------------------

def test_thick_and_thin_masks_produce_similar_centerline_complexity():
    lane = _arc_points(550.0)
    thin_mask = _rasterize(lane, thickness=3)
    thick_mask = _rasterize(lane, thickness=27)
    thin_lanes = extract_grouped_centerlines_from_mask(thin_mask)
    thick_lanes = extract_grouped_centerlines_from_mask(thick_mask)
    thin_result = _analyze(thin_lanes)
    thick_result = _analyze(thick_lanes)
    assert thin_result["bendiness"] == pytest.approx(thick_result["bendiness"], rel=0.25)


def test_dashed_mask_components_are_grouped_into_one_logical_lane():
    lane = _arc_points(1500.0, n=200)
    dashed_mask = _rasterize(lane, thickness=5, dashed=True)
    grouped = extract_grouped_centerlines_from_mask(dashed_mask)
    assert len(grouped) == 1
    result = _analyze(grouped)
    assert result["valid_lane_count"] == 1


def test_native_polylines_are_used_without_a_mask_round_trip(tmp_path):
    lane = _arc_points(500.0)
    path = tmp_path / "pred.json"
    writer = PredictionManifestWriter(str(path), "clrernet", "synthetic", save_masks=False)
    polylines = [[{"x": float(x), "y": float(y)} for x, y in lane]]
    unrelated_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    writer.add("one", "/tmp/one.png", unrelated_mask, polylines=polylines)
    writer.write()
    prediction = PredictionManifestReader(str(path)).get("one")
    assert prediction.meta["geometry_source"] == "native_polyline"
    result = _analyze(prediction.lanes, geometry_source="native_polyline")
    assert result["profile_class"] == "constant_curve"
    assert result["unavailable_reason"] is None


# ---------------------------------------------------------------------------
# 23, 25-27: boundary/paired compatibility, extra lanes, short support.
# ---------------------------------------------------------------------------

def test_one_boundary_and_paired_centerline_are_geometrically_compatible():
    left = _arc_points(700.0, x0=200.0)
    right = _arc_points(700.0, x0=270.0)
    boundary_only = _analyze([left])
    paired = _analyze([left, right])
    assert boundary_only["lane_diagnostics"][0]["representation_type"] == "boundary"
    assert paired["lane_diagnostics"][0]["representation_type"] == "paired_centerline"
    assert boundary_only["bendiness"] == pytest.approx(paired["bendiness"], rel=0.20)


def test_extra_parallel_lanes_do_not_automatically_increase_alignment_complexity():
    two_lanes = [_arc_points(600.0, x0=150.0), _arc_points(600.0, x0=230.0)]
    four_lanes = two_lanes + [_arc_points(600.0, x0=310.0), _arc_points(600.0, x0=390.0)]
    result_two = _analyze(two_lanes)
    result_four = _analyze(four_lanes)
    assert result_four["alignment_complexity"] == pytest.approx(result_two["alignment_complexity"], rel=0.15)
    assert result_four["valid_lane_count"] > result_two["valid_lane_count"]


def test_very_short_visible_support_returns_unknown():
    short_lane = _straight_points(span=15.0, n=10)
    result = _analyze([short_lane])
    assert result["profile_class"] == "unknown"
    assert result["alignment_complexity"] is None
    assert result["unavailable_reason"] is not None


def test_nonfinite_duplicate_and_unordered_points_are_handled_safely():
    lane = _arc_points(500.0, n=60)
    corrupted = lane.copy()
    corrupted = np.vstack([corrupted, corrupted[10:12]])  # duplicates
    corrupted[5] = [np.nan, np.nan]
    corrupted[15] = [np.inf, 20.0]
    rng = np.random.default_rng(5)
    shuffled = corrupted[rng.permutation(corrupted.shape[0])]
    result = _analyze([shuffled])
    assert result["unavailable_reason"] is None
    assert result["profile_class"] == "constant_curve"


# ---------------------------------------------------------------------------
# 24, 26: topology separate from alignment.
# ---------------------------------------------------------------------------

def test_merge_and_split_topology_is_reported_separately_from_alignment():
    # Three straight, parallel boundaries spanning the whole visible range
    # (near AND far) vs. an extra boundary that only appears near the camera
    # (a supported lane-count change / merge-like topology).
    full_a = _straight_points(x0=150.0, span=300.0)
    full_b = _straight_points(x0=230.0, span=300.0)
    full_c = _straight_points(x0=310.0, span=300.0)
    extra_near = _straight_points(x0=390.0, span=140.0, y0=Y1 - 140.0)

    stable = _analyze([full_a, full_b, full_c])
    with_topology_change = _analyze([full_a, full_b, full_c, extra_near])

    assert stable["profile_class"] == "straight"
    assert with_topology_change["profile_class"] == "straight"
    # Alignment complexity is driven by curvature, not by lane count.
    assert with_topology_change["bendiness"] == pytest.approx(stable["bendiness"], abs=0.02)
    if with_topology_change["topology_complexity"] is not None and stable["topology_complexity"] is not None:
        assert with_topology_change["topology_complexity"] >= stable["topology_complexity"]


def test_topology_diagnostic_is_unavailable_with_insufficient_boundary_evidence():
    result = _analyze([_straight_points()])
    assert result["topology_complexity"] is None


def test_infer_lane_topology_is_conservative_by_construction():
    empty = infer_lane_topology({"logical_boundary_diagnostics": []}, image_shape=(HEIGHT, WIDTH))
    assert empty["topology_complexity"] is None
    assert empty["unavailable_reason"] == "insufficient_boundary_evidence_for_topology"


# ---------------------------------------------------------------------------
# I4 width variation must not leak into curvature complexity.
# ---------------------------------------------------------------------------

def test_lane_width_taper_does_not_become_curvature_complexity():
    # A straight roadway whose lane narrows (width taper) should still be
    # geometrically "straight" for I5 even though I4 would show a taper.
    n = 120
    ys = np.linspace(Y0, Y1, n)
    left_x = np.full(n, 180.0)
    right_x = 320.0 - 60.0 * (ys - Y0) / (Y1 - Y0)  # width shrinks from 140 to 80 px
    left = np.column_stack([left_x, ys])
    right = np.column_stack([right_x, ys])
    result = _analyze([left, right])
    assert result["profile_class"] == "straight"
    assert result["alignment_complexity"] < 0.05


# ---------------------------------------------------------------------------
# Legacy proxy migration and canonical/prediction independence (via build_metric_record).
# ---------------------------------------------------------------------------

def _mask_from_lane(lane: np.ndarray) -> np.ndarray:
    return _rasterize(lane, thickness=5)


def test_legacy_quadratic_proxy_is_preserved_unchanged():
    lane = _arc_points(400.0)
    mask = _mask_from_lane(lane)
    legacy_direct = compute_i5_legacy_quadratic_proxy(mask, lanes=[lane])
    legacy_via_alias = compute_i5_geometry_complexity(mask, lanes=[lane])
    assert legacy_direct == pytest.approx(legacy_via_alias)


def test_compute_i5_geometry_complexity_details_uses_new_engine():
    lane = _arc_points(400.0)
    mask = _mask_from_lane(lane)
    details = compute_i5_geometry_complexity_details(mask, lanes=[lane], image_shape=(HEIGHT, WIDTH))
    assert details["profile_class"] == "constant_curve"
    assert details["alignment_complexity"] is not None


def test_canonical_i5_and_r1_are_independent_of_prediction():
    lane = _arc_points(500.0)
    mask = _mask_from_lane(lane)
    sample = _sample((HEIGHT, WIDTH), mask, lane, style="solid")
    unrelated_lane = _arc_points(150.0, x0=100.0)
    record_a = build_metric_record(sample, mask, np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8), pred_lanes=[lane])
    record_b = build_metric_record(
        sample, _mask_from_lane(unrelated_lane), np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8), pred_lanes=[unrelated_lane]
    )
    assert record_a["I5"] == pytest.approx(record_b["I5"])
    assert record_a["I5_alignment_complexity"] == pytest.approx(record_b["I5_alignment_complexity"])
    assert record_a["R1"] == pytest.approx(record_b["R1"])
    # And prediction geometry clearly differs, proving this isn't a no-op comparison.
    assert record_a["I5_pred"] != pytest.approx(record_b["I5_pred"], rel=0.05)


def test_prediction_guided_i5_does_not_leak_gt_geometry_or_metadata():
    lane_gt_1 = _arc_points(300.0)
    lane_gt_2 = _arc_points(1800.0)
    pred_lane = _arc_points(500.0, x0=210.0)
    pred_mask = _mask_from_lane(pred_lane)
    image = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)

    sample_1 = _sample((HEIGHT, WIDTH), _mask_from_lane(lane_gt_1), lane_gt_1, style="solid")
    sample_2 = _sample((HEIGHT, WIDTH), _mask_from_lane(lane_gt_2), lane_gt_2, style="dashed")

    record_1 = build_metric_record(sample_1, pred_mask, image, pred_lanes=[pred_lane])
    record_2 = build_metric_record(sample_2, pred_mask, image, pred_lanes=[pred_lane])

    assert record_1["I5_pred"] == pytest.approx(record_2["I5_pred"])
    assert record_1["I5_pred_profile_class"] == record_2["I5_pred_profile_class"]
    # GT geometry clearly differs between the two samples.
    assert record_1["I5"] != pytest.approx(record_2["I5"], rel=0.05)


def test_i5_is_unaffected_by_marking_contrast_or_brightness():
    lane = _arc_points(450.0)
    mask = _mask_from_lane(lane)
    sample = _sample((HEIGHT, WIDTH), mask, lane, style="solid")
    bright_image = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
    dark_image = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    record_bright = build_metric_record(sample, mask, bright_image)
    record_dark = build_metric_record(sample, mask, dark_image)
    assert record_bright["I5"] == pytest.approx(record_dark["I5"])
    assert record_bright["I5_alignment_complexity"] == pytest.approx(record_dark["I5_alignment_complexity"])


def test_existing_r1_formula_is_unchanged_by_i5():
    base = {
        "I1_pattern_continuity": 0.8,
        "I2_local_contrast": 0.6,
        "I3_boundary_sharpness": 0.5,
        "I4_legacy_thickness_stability": 0.9,
    }
    straight_variant = dict(base, I5_geometry_complexity=0.02, I5_alignment_complexity=0.02)
    sharp_variant = dict(base, I5_geometry_complexity=0.95, I5_alignment_complexity=0.95)
    assert compute_r1_marking_readability_score(straight_variant) == pytest.approx(
        compute_r1_marking_readability_score(sharp_variant)
    )


def test_existing_compute_i5_geometry_complexity_signature_remains_usable():
    lane = _arc_points(400.0)
    mask = _mask_from_lane(lane)
    # Original two-positional-argument call must still work.
    value = compute_i5_geometry_complexity(mask, [lane])
    assert value is not None and 0.0 <= value <= 1.0


def test_json_serialization_succeeds_with_allow_nan_false():
    lane = _arc_points(400.0)
    mask = _mask_from_lane(lane)
    sample = _sample((HEIGHT, WIDTH), mask, lane, style="solid")
    image = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
    record = build_metric_record(sample, mask, image)
    json.dumps(record, allow_nan=False)


def test_old_prediction_manifest_without_geometry_fields_remains_usable(tmp_path):
    lane = _arc_points(400.0)
    mask = _mask_from_lane(lane)
    sample = _sample((HEIGHT, WIDTH), mask, lane, style="solid")
    image = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
    # Original three-argument call must still work and must not raise even
    # though I5 now consumes prediction-manifest fields.
    writer = PredictionManifestWriter(str(tmp_path / "pred.json"), "legacy_model", "synthetic", save_masks=False)
    writer.add("synthetic", "/tmp/i5_synthetic.png", mask)
    writer.write()
    record = build_metric_record(sample, mask, image)
    assert "I5_pred_unavailable_reason" in record


# ---------------------------------------------------------------------------
# C4 revision: synthetic monotonic recovery and skip-reason reporting.
# ---------------------------------------------------------------------------

def _synthetic_c4_records(n: int = 40, seed: int = 21) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        complexity = float(i) / (n - 1)
        # Detection quality decreases monotonically (with small noise) as
        # geometry complexity increases — a known ground-truth relationship.
        f1 = float(np.clip(0.92 - 0.55 * complexity + rng.normal(0.0, 0.03), 0.0, 1.0))
        records.append(
            {
                "I5_alignment_complexity": complexity,
                "I5_geometry_complexity": complexity,
                "I5_bendiness": complexity,
                "I5_p95_abs_curvature": complexity,
                "I5_curvature_variation": complexity * 0.5,
                "I5_reversal_density": 0.0,
                "I5_topology_complexity": None,
                "I5_measurement_space": "normalized_image_proxy",
                "D3_f1": f1,
                "D3_iou": f1 * 0.9,
                "D6_missed_marking_ratio": 1.0 - f1,
                "dataset": "synthetic",
                "model_name": "synthetic_model",
            }
        )
    return records


def test_c4_recovers_known_monotonic_geometry_performance_relationship():
    records = _synthetic_c4_records()
    correlations = compute_correlations_from_records(records, min_samples=10)
    detail = correlations["C4_detail"]
    assert detail["primary"]["spearman"] < -0.5
    assert detail["tertile_effect"]["effect"] > 0.1
    assert detail["tertile_effect"]["bootstrap_ci_95"][0] < detail["tertile_effect"]["bootstrap_ci_95"][1]
    assert correlations["C4"] is not None


def test_c4_reports_insufficient_sample_and_zero_variance_reasons():
    small = _synthetic_c4_records(n=4)
    correlations = compute_correlations_from_records(small, min_samples=10)
    assert "insufficient_samples" in correlations["C4_detail"]["primary"]["reason"]

    constant = _synthetic_c4_records(n=20)
    for record in constant:
        record["I5_alignment_complexity"] = 0.5
        record["I5_geometry_complexity"] = 0.5
    correlations_const = compute_correlations_from_records(constant, min_samples=10)
    assert correlations_const["C4_detail"]["primary"]["reason"] == "zero_variance"


def test_c4_does_not_use_prediction_guided_i5():
    # I5_pred fields must never be read by the canonical C4 computation.
    records = _synthetic_c4_records()
    for record in records:
        record["I5_pred_alignment_complexity"] = 1.0 - record["I5_alignment_complexity"]
    correlations = compute_correlations_from_records(records, min_samples=10)
    # Effect direction should still reflect the canonical (not the inverted
    # prediction-guided) relationship.
    assert correlations["C4_detail"]["primary"]["spearman"] < -0.5


# ---------------------------------------------------------------------------
# Aggregation: unavailable lanes are never averaged as zero.
# ---------------------------------------------------------------------------

def test_aggregate_lane_complexity_never_averages_unavailable_as_zero():
    lane_results = [
        {"complexity_score": 0.6, "complexity_confidence": 0.8, "valid_support_length": 100.0},
        {"complexity_score": None, "complexity_confidence": 0.0, "valid_support_length": 0.0},
    ]
    aggregate = aggregate_lane_complexity(lane_results)
    assert aggregate["valid_lane_count"] == 1
    assert aggregate["unknown_lane_count"] == 1
    assert aggregate["alignment_complexity"] == pytest.approx(0.6)


def test_measurement_space_priority_prefers_calibrated_over_lane_width_normalized():
    raw_calibrated = {"measurement_space": "calibrated_metric_bev"}
    space = determine_i5_measurement_space(raw_calibrated, lane_width_context={"median_width": 3.5})
    assert space["i5_measurement_space"] == "calibrated_metric_bev"

    raw_uncalibrated = {"measurement_space": "projective_normalized"}
    with_width = determine_i5_measurement_space(raw_uncalibrated, lane_width_context={"median_width": 40.0})
    assert with_width["i5_measurement_space"] == "lane_width_normalized"
    without_width = determine_i5_measurement_space(raw_uncalibrated, lane_width_context=None)
    assert without_width["i5_measurement_space"] == "normalized_image_proxy"


def test_pair_boundaries_and_construct_centerlines_reused_directly():
    left = _arc_points(500.0, x0=200.0)
    right = _arc_points(500.0, x0=270.0)
    result = pair_boundaries_and_construct_centerlines([left, right], image_shape=(HEIGHT, WIDTH))
    assert len(result["centerlines"]) == 1
    assert result["centerlines"][0]["representation_type"] == "paired_centerline"
