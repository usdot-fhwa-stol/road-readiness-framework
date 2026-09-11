"""Behavioral tests for perspective-normalized, geometry-based I4."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from evaluation.lane_width_stability import (
    LaneWidthConfig,
    analyze_lane_width_stability,
    extract_grouped_boundaries_from_mask,
    fit_width_profile_hypotheses,
)
from evaluation.readiness_metrics import build_metric_record
from evaluation.visualize_failures import save_i4_lane_width_debug
from lane_eval.manifest import PredictionManifestReader, PredictionManifestWriter
from lane_eval.schema.lane import LaneTarget
from lane_eval.schema.sample import LaneSample


HEIGHT = 360
WIDTH = 480
CX = 240.0
HORIZON = 72.0
FOCAL = 300.0
VERTICAL_SCALE = 1260.0
GROUND_TO_IMAGE = np.asarray(
    [[FOCAL, CX, 0.0], [0.0, HORIZON, VERTICAL_SCALE], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
IMAGE_TO_GROUND = np.linalg.inv(GROUND_TO_IMAGE)


def _project_ground(points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(points.shape[0])])
    projected = homogeneous @ GROUND_TO_IMAGE.T
    return projected[:, :2] / projected[:, 2, None]


def _road_pair(
    profile="constant",
    *,
    center_curve=0.0,
    y0=5.0,
    y1=42.0,
    count=100,
    lane_offset=0.0,
):
    longitudinal = np.linspace(y0, y1, count)
    normalized = (longitudinal - y0) / (y1 - y0)
    center_x = lane_offset + center_curve * (longitudinal - 22.0) ** 2 / 400.0
    derivative = center_curve * 2.0 * (longitudinal - 22.0) / 400.0
    if profile == "constant":
        width = np.full(count, 3.7)
    elif profile == "taper":
        width = 4.3 - 1.35 * normalized
    elif profile == "transition":
        width = 4.0 - 0.15 * normalized - 1.25 * np.maximum(normalized - 0.50, 0.0)
    elif profile == "piecewise":
        width = 3.7 + 0.7 * np.maximum(normalized - 0.35, 0.0) - 0.8 * np.maximum(normalized - 0.70, 0.0)
    elif profile == "irregular":
        width = 3.7 + 0.48 * np.sin(10.0 * np.pi * normalized) + 0.16 * np.sin(27.0 * np.pi * normalized)
    else:
        width = np.asarray(profile(longitudinal, normalized), dtype=np.float64)
    normal = np.column_stack([np.ones(count), -derivative])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    center = np.column_stack([center_x, longitudinal])
    left_ground = center - 0.5 * width[:, None] * normal
    right_ground = center + 0.5 * width[:, None] * normal
    return _project_ground(left_ground), _project_ground(right_ground), left_ground, right_ground


def _analyze(lanes, **kwargs):
    selected_homography = kwargs.pop("homography", IMAGE_TO_GROUND)
    if selected_homography is not None and "calibration_meta" not in kwargs:
        kwargs["calibration_meta"] = {"metric_scale_available": True, "units": "m"}
    return analyze_lane_width_stability(
        lanes,
        image_shape=(HEIGHT, WIDTH),
        geometry_source=kwargs.pop("geometry_source", "test_native_polyline"),
        homography=selected_homography,
        **kwargs,
    )


def _rasterize(lanes, thickness=8, dashed=False):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for lane in lanes:
        points = np.round(lane).astype(np.int32)
        if dashed:
            for start in range(0, points.shape[0] - 2, 10):
                cv2.polylines(mask, [points[start : start + 6]], False, 1, thickness, cv2.LINE_AA)
        else:
            cv2.polylines(mask, [points], False, 1, thickness, cv2.LINE_AA)
    return mask


def _sample(lanes, mask=None, target_meta=None, sample_meta=None):
    if mask is None:
        mask = _rasterize(lanes, thickness=5)
    return LaneSample(
        image_id="synthetic",
        image_path="/tmp/synthetic.png",
        width=WIDTH,
        height=HEIGHT,
        target=LaneTarget(lanes=lanes, mask=mask, meta=target_meta or {}),
        meta=sample_meta or {},
    )


def _blank_image():
    return np.full((HEIGHT, WIDTH, 3), 55, dtype=np.uint8)


def test_straight_parallel_boundaries_recover_constant_metric_width():
    left, right, _, _ = _road_pair("constant")
    result = _analyze([left, right])
    pair = result["pair_diagnostics"][0]
    assert result["measurement_space"] == "calibrated_metric_bev"
    assert result["profile_type"] == "constant"
    assert result["lane_width_stability"] > 0.95
    assert result["width_constancy"] > 0.95
    assert pair["median_width"] == pytest.approx(3.7, abs=0.08)
    assert pair["width_units"] == "m"


def test_curved_constant_width_uses_normals_not_horizontal_pixel_span():
    left, right, _, _ = _road_pair("constant", center_curve=4.0)
    raw_horizontal = np.abs(right[:, 0] - left[:, 0])
    result = _analyze([left, right])
    pair = result["pair_diagnostics"][0]
    assert np.std(raw_horizontal) / np.mean(raw_horizontal) > 0.30
    assert pair["median_width"] == pytest.approx(3.7, abs=0.12)
    assert pair["stability_score"] > 0.88


def test_perspective_shrinkage_is_removed_by_independent_ipm():
    left, right, _, _ = _road_pair("constant")
    raw_width = np.abs(right[:, 0] - left[:, 0])
    assert raw_width[-1] < raw_width[0] * 0.20
    result = _analyze([left, right])
    assert result["lane_width_stability"] > 0.95
    assert result["profile_type"] == "constant"


def test_smooth_taper_has_high_profile_stability_but_lower_constancy():
    left, right, _, _ = _road_pair("taper")
    result = _analyze([left, right])
    pair = result["pair_diagnostics"][0]
    assert pair["profile_type"] == "linear_taper"
    assert pair["stability_score"] > 0.86
    assert pair["width_constancy_score"] < 0.55
    assert pair["relative_width_change"] > 0.25


def test_supported_piecewise_merge_or_split_transition():
    left, right, _, _ = _road_pair("transition")
    result = _analyze([left, right])
    pair = result["pair_diagnostics"][0]
    assert pair["profile_type"] == "merge_or_split_transition"
    assert pair["change_points"]
    assert pair["stability_score"] > 0.78


def test_irregular_profile_scores_below_clean_taper():
    taper_left, taper_right, _, _ = _road_pair("taper")
    noisy_left, noisy_right, _, _ = _road_pair("irregular")
    taper = _analyze([taper_left, taper_right])
    noisy = _analyze([noisy_left, noisy_right])
    assert noisy["lane_width_stability"] < taper["lane_width_stability"] - 0.20
    assert noisy["profile_type"] == "irregular_or_noisy"


def test_localized_width_outlier_affects_tail_diagnostic():
    def outlier(_, normalized):
        width = np.full(normalized.size, 3.7)
        width[(normalized > 0.46) & (normalized < 0.57)] += 1.15
        return width

    left, right, _, _ = _road_pair(outlier)
    result = _analyze([left, right])
    pair = result["pair_diagnostics"][0]
    assert pair["relative_p90_residual"] > 0.08
    assert pair["stability_score"] < 0.90


def test_sparse_polyline_outliers_are_robustly_rejected_or_downweighted():
    left, right, _, _ = _road_pair("constant")
    right = right.copy()
    right[[12, 41, 78], 0] += np.asarray([35.0, -42.0, 38.0])
    result = _analyze([left, right])
    assert result["valid_pair_count"] == 1
    assert result["lane_width_stability"] > 0.80
    assert min(item["inlier_fraction"] for item in result["boundary_diagnostics"]) < 1.0


def test_near_horizon_resolution_is_reliability_gated():
    left, right, _, _ = _road_pair("constant", y1=95.0, count=180)
    result = _analyze([left, right], debug=True)
    debug = result["pair_diagnostics"][0]["debug"]
    reliability = np.asarray(debug["sample_reliability"], dtype=float)
    assert np.count_nonzero(reliability == 0.0) > 0
    assert result["pair_diagnostics"][0]["valid_sample_fraction"] < 1.0


def test_ill_conditioned_homography_never_claims_metric_width():
    left, right, _, _ = _road_pair("constant")
    bad = np.asarray([[1.0, 0.0, 0.0], [0.0, 1e-13, 0.0], [0.0, 0.0, 1.0]])
    result = _analyze([left, right], homography=bad)
    assert result["measurement_space"] == "projective_normalized"
    assert result["metric_scale_available"] is False
    assert result["width_units"] != "m"
    assert result["metric_scale_confidence"] == 0.0
    assert 0.0 < result["measurement_confidence"] <= 0.25


def test_no_calibration_uses_apparent_units_only():
    left, right, _, _ = _road_pair("constant")
    result = _analyze([left, right], homography=None)
    assert result["measurement_space"] == "projective_normalized"
    assert result["metric_scale_available"] is False
    assert result["metric_scale_confidence"] == 0.0
    assert result["width_units"] == "image_width_fraction"
    assert all(pair["width_units"] != "m" for pair in result["pair_diagnostics"])


def test_calibrated_median_width_accuracy_and_scale_free_label():
    left, right, _, _ = _road_pair("constant")
    metric = _analyze([left, right])
    scale_free = _analyze(
        [left, right],
        calibration_meta={"metric_scale_available": False, "units": "arbitrary"},
    )
    assert metric["pair_diagnostics"][0]["median_width"] == pytest.approx(3.7, abs=0.08)
    assert scale_free["measurement_space"] == "calibrated_scale_free_bev"
    assert scale_free["metric_scale_available"] is False
    assert scale_free["width_units"] == "road_plane_units"


def test_dashed_mask_components_group_into_logical_centerlines():
    left, right, _, _ = _road_pair("constant")
    mask = _rasterize([left, right], thickness=6, dashed=True)
    grouped = extract_grouped_boundaries_from_mask(mask)
    assert len(grouped) == 2
    result = _analyze(grouped, geometry_source="mask_derived")
    assert result["valid_pair_count"] == 1
    assert result["pair_diagnostics"][0]["median_width"] == pytest.approx(3.7, abs=0.30)


def test_thick_mask_uses_centerlines_and_does_not_inflate_width():
    left, right, _, _ = _road_pair("constant")
    thin = extract_grouped_boundaries_from_mask(_rasterize([left, right], thickness=4))
    thick = extract_grouped_boundaries_from_mask(_rasterize([left, right], thickness=18))
    thin_width = _analyze(thin, geometry_source="mask_derived")["pair_diagnostics"][0]["median_width"]
    thick_width = _analyze(thick, geometry_source="mask_derived")["pair_diagnostics"][0]["median_width"]
    assert thick_width == pytest.approx(thin_width, abs=0.20)


def test_continuous_float_polylines_pair_without_rasterization():
    left, right, _, _ = _road_pair("constant")
    result = _analyze([left.astype(np.float64), right.astype(np.float64)], geometry_source="native_polyline")
    assert result["geometry_source"] == "native_polyline"
    assert result["valid_pair_count"] == 1
    assert result["source_boundary_count"] == 2


def test_double_solid_is_collapsed_and_not_treated_as_tiny_lane():
    longitudinal = np.linspace(5.0, 42.0, 100)

    def line(x):
        return _project_ground(np.column_stack([np.full(longitudinal.size, x), longitudinal]))

    result = _analyze([line(-2.00), line(-1.72), line(1.85)])
    assert result["logical_boundary_count"] == 2
    assert len(result["double_line_groups"]) == 1
    assert result["valid_pair_count"] == 1
    assert result["pair_diagnostics"][0]["median_width"] > 3.0


def test_three_adjacent_boundaries_produce_two_pairs():
    longitudinal = np.linspace(5.0, 42.0, 100)
    lanes = [
        _project_ground(np.column_stack([np.full(longitudinal.size, x), longitudinal]))
        for x in (-3.7, 0.0, 3.7)
    ]
    result = _analyze(lanes)
    assert result["valid_pair_count"] == 2
    assert [pair["pair_index"] for pair in result["pair_diagnostics"]] == [0, 1]


def test_crossing_boundaries_are_rejected():
    longitudinal = np.linspace(5.0, 42.0, 100)
    normalized = (longitudinal - 5.0) / 37.0
    left_ground = np.column_stack([-1.8 + 4.0 * normalized, longitudinal])
    right_ground = np.column_stack([1.8 - 4.0 * normalized, longitudinal])
    result = _analyze([_project_ground(left_ground), _project_ground(right_ground)])
    assert result["valid_pair_count"] == 0
    assert any(reason["reason"] == "boundary_crossing_or_reversed_order" for reason in result["rejected_pair_reasons"])


def test_insufficient_overlap_and_single_boundary_are_unknown():
    left, _, _, _ = _road_pair("constant", y0=5.0, y1=20.0)
    _, right, _, _ = _road_pair("constant", y0=25.0, y1=42.0)
    no_overlap = _analyze([left, right])
    one = _analyze([left])
    assert no_overlap["lane_width_stability"] is None
    assert no_overlap["unknown_pair_count"] >= 1
    assert one["lane_width_stability"] is None
    assert one["unavailable_reason"] == "only_one_logical_boundary"


def test_nonfinite_duplicate_and_out_of_frame_points_are_safe():
    left, right, _, _ = _road_pair("constant")
    contaminated = np.vstack([left, left[20], [np.nan, 120.0], [np.inf, 180.0], [-100.0, 200.0]])
    result = _analyze([contaminated, right])
    assert result["valid_pair_count"] == 1
    assert result["lane_width_stability"] is not None


def test_geometry_near_image_boundary_reduces_confidence():
    y = np.linspace(145.0, 340.0, 100)
    edge = [np.column_stack([np.full(y.size, 2.0), y]), np.column_stack([np.full(y.size, 102.0), y])]
    center = [np.column_stack([np.full(y.size, 170.0), y]), np.column_stack([np.full(y.size, 270.0), y])]
    edge_result = _analyze(edge, homography=None)
    center_result = _analyze(center, homography=None)
    assert edge_result["geometry_confidence"] < center_result["geometry_confidence"]


def test_object_occlusion_invalidates_affected_samples():
    left, right, _, _ = _road_pair("constant")
    clear = _analyze([left, right])
    objects = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    objects[155:235, 80:400] = 1
    occluded = _analyze([left, right], object_mask=objects)
    assert occluded["pair_diagnostics"][0]["valid_sample_count"] < clear["pair_diagnostics"][0]["valid_sample_count"]


def test_multiple_pairs_with_different_profiles_aggregate_valid_pairs_only():
    longitudinal = np.linspace(5.0, 42.0, 110)
    normalized = (longitudinal - 5.0) / 37.0
    left = _project_ground(np.column_stack([np.full(longitudinal.size, -3.7), longitudinal]))
    center = _project_ground(np.column_stack([np.zeros(longitudinal.size), longitudinal]))
    right = _project_ground(np.column_stack([3.7 - 0.9 * normalized, longitudinal]))
    result = _analyze([left, center, right])
    assert result["valid_pair_count"] == 2
    assert len(result["pair_diagnostics"]) == 2
    assert result["lane_width_stability"] > 0.75


def test_unavailable_pairs_are_not_averaged_as_zero():
    good = {
        "stability_score": 0.9,
        "width_constancy_score": 1.0,
        "profile_confidence": 0.8,
        "geometry_confidence": 0.9,
        "support_length": 20.0,
        "profile_type": "constant",
    }
    unavailable = dict(good, stability_score=None, width_constancy_score=None, profile_type="unknown")
    from evaluation.lane_width_stability import aggregate_pair_results

    result = aggregate_pair_results([good, unavailable])
    assert result["lane_width_stability"] == pytest.approx(0.9)
    assert result["valid_pair_count"] == 1
    assert result["unknown_pair_count"] == 1


def test_canonical_i4_is_model_independent_and_r1_stays_legacy():
    left, right, _, _ = _road_pair("constant")
    sample = _sample([left, right])
    first = build_metric_record(
        sample,
        _rasterize([left, right], thickness=5),
        _blank_image(),
        pred_lanes=[left, right],
        homography=IMAGE_TO_GROUND,
    )
    shifted = [lane + np.asarray([70.0, 0.0]) for lane in (left, right)]
    second = build_metric_record(
        sample,
        _rasterize(shifted, thickness=14),
        _blank_image(),
        pred_lanes=shifted,
        homography=IMAGE_TO_GROUND,
    )
    assert second["I4"] == pytest.approx(first["I4"])
    assert second["I4_pair_diagnostics"] == first["I4_pair_diagnostics"]
    assert second["R1"] == pytest.approx(first["R1"])


def test_legacy_thickness_fields_are_explicit_and_still_calculated():
    left, right, _, _ = _road_pair("constant")
    record = build_metric_record(
        _sample([left, right]),
        _rasterize([left, right]),
        _blank_image(),
        pred_lanes=[left, right],
        homography=IMAGE_TO_GROUND,
    )
    assert record["I4_legacy_thickness_stability"] is not None
    assert record["I4_legacy_mean_thickness_px"] is not None
    assert record["I4_legacy_num_segments"] >= 1
    assert "I4_thickness_stability" not in record


def test_manifest_backward_compatibility_and_native_float_geometry(tmp_path):
    path = tmp_path / "pred.json"
    writer = PredictionManifestWriter(str(path), "clrernet", "synthetic", save_masks=False)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    writer.add("one", "/tmp/one.png", mask)
    writer.add("two", "/tmp/two.png", mask, polylines=[[{"x": 30.25, "y": 40.5}, {"x": 31.75, "y": 300.125}]])
    writer.write()
    reader = PredictionManifestReader(str(path))
    assert reader.get("one").meta["geometry_source"] == "mask_derived"
    native = reader.get("two")
    assert native.meta["geometry_source"] == "native_polyline"
    assert native.lanes[0][0, 0] == pytest.approx(30.25)

    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps({"samples": [{"sample_id": "old", "prediction": {"lane_json": {"h_samples": [10, 20], "lanes": [[30, 31]]}}}]}))
    assert PredictionManifestReader(str(old_path)).get("old").meta["geometry_source"] == "lane_json"


def test_json_safety_and_debug_visualization(tmp_path):
    left, right, _, _ = _road_pair("transition")
    result = _analyze([left, right], debug=True)
    json.dumps(result, allow_nan=False)
    output = tmp_path / "i4_debug.png"
    save_i4_lane_width_debug(_blank_image(), result, str(output))
    rendered = cv2.imread(str(output))
    assert rendered is not None
    assert rendered.shape[0] > HEIGHT


def test_direct_profile_fit_is_robust_to_sparse_random_outliers():
    rng = np.random.default_rng(22)
    s = np.linspace(0.0, 1.0, 60)
    widths = 4.2 - 1.0 * s + rng.normal(0.0, 0.015, s.size)
    widths[[8, 31, 53]] += np.asarray([0.8, -0.7, 0.9])
    fitted = fit_width_profile_hypotheses(s, widths, np.ones_like(s), config=LaneWidthConfig())
    assert fitted["selected_model"] == "linear_taper"
    assert np.median(np.abs(fitted["expected"] - (4.2 - s))) < 0.05


def test_camera_model_homography_is_supported_without_separate_argument():
    left, right, _, _ = _road_pair("constant")
    result = _analyze(
        [left, right],
        homography=None,
        camera_model={
            "image_to_road_homography": IMAGE_TO_GROUND,
            "metric_scale_available": True,
            "units": "m",
        },
    )
    assert result["measurement_space"] == "calibrated_metric_bev"
    assert result["homography_source"] == "camera_model.image_to_road_homography"
    assert result["pair_diagnostics"][0]["median_width"] == pytest.approx(3.7, abs=0.08)


def test_homography_without_explicit_scale_never_reports_meters():
    left, right, _, _ = _road_pair("constant")
    result = analyze_lane_width_stability(
        [left, right],
        image_shape=(HEIGHT, WIDTH),
        geometry_source="native_polyline",
        homography=IMAGE_TO_GROUND,
    )
    assert result["measurement_space"] == "calibrated_scale_free_bev"
    assert result["metric_scale_available"] is False
    assert result["width_units"] == "road_plane_units"


def test_drivable_mask_is_optional_support_and_can_reject_non_lane_region():
    left, right, _, _ = _road_pair("constant")
    no_mask = _analyze([left, right])
    non_drivable = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    rejected = _analyze([left, right], road_mask=non_drivable)
    assert no_mask["valid_pair_count"] == 1
    assert rejected["valid_pair_count"] == 0
    assert rejected["rejected_pair_reasons"][0]["reason"] == "lane_region_lacks_drivable_area_support"


def test_short_width_profile_selects_unknown_instead_of_overfitting():
    s = np.linspace(0.0, 1.0, 5)
    result = fit_width_profile_hypotheses(s, np.full(5, 3.7), np.ones(5))
    assert result["selected_model"] == "unknown"
    assert result["expected"] is None


# ---------------------------------------------------------------------------
# Implausible-geometry rejection (spec test 35).  Two "boundaries" that do not
# plausibly bound a travel lane must be rejected as a pair (unavailable with an
# explicit reason), never scored as a confident stable or unstable width.  Each
# case is rendered independently rather than derived from the metric's own
# machinery.
# ---------------------------------------------------------------------------


def _uncalibrated(lanes, **kwargs):
    return analyze_lane_width_stability(
        lanes,
        image_shape=(HEIGHT, WIDTH),
        geometry_source="mask_derived",
        homography=None,
        **kwargs,
    )


def test_outline_tracing_horizontal_stripes_are_rejected_not_scored():
    """A crosswalk / stop-line outline (near-horizontal stripes) is not a lane."""

    top = np.column_stack([np.linspace(60.0, 300.0, 30), np.linspace(232.0, 236.0, 30)])
    bottom = np.column_stack([np.linspace(40.0, 320.0, 30), np.linspace(300.0, 304.0, 30)])
    result = _uncalibrated([top, bottom])
    assert result["lane_width_stability"] is None
    assert result["valid_pair_count"] == 0
    reasons = {item.get("reason") for item in result["rejected_boundary_reasons"]}
    reasons |= {item.get("reason") for item in result["rejected_pair_reasons"]}
    assert "boundary_orientation_not_longitudinal" in reasons


def test_boundary_climbing_a_wall_relative_to_partner_is_rejected():
    """One boundary leaving the road surface (diverging up a wall) is rejected."""

    y = np.linspace(150.0, 330.0, 60)
    road = np.column_stack([np.full(y.size, 180.0), y])
    # Climbs strongly to the right as it recedes: apparent separation swings far
    # beyond any single travel lane, and it is not parallel to the road boundary.
    wall = np.column_stack([np.linspace(430.0, 232.0, y.size), y])
    result = _uncalibrated([road, wall])
    assert result["lane_width_stability"] is None
    assert result["valid_pair_count"] == 0
    reasons = {item.get("reason") for item in result["rejected_pair_reasons"]}
    assert reasons & {
        "implausible_lane_width_variation",
        "boundaries_not_consistently_parallel",
        "implausible_corridor_aspect_ratio",
        "relative_tangent_incompatible",
    }


def test_erratically_non_parallel_boundaries_are_rejected():
    """Boundaries whose separation wanders erratically do not bound a lane."""

    y = np.linspace(150.0, 340.0, 90)
    left = np.column_stack([np.full(y.size, 170.0), y])
    wobble = 40.0 * np.sin(y / 7.0) + 26.0 * np.sin(y / 3.1)
    right = np.column_stack([270.0 + wobble, y])
    result = _uncalibrated([left, right])
    # Rejected (unavailable, no valid pair) rather than scored, whether the
    # erratic geometry is caught at the boundary-fit or the parallelism stage.
    assert result["lane_width_stability"] is None
    assert result["valid_pair_count"] == 0
    reasons = {item.get("reason") for item in result["rejected_boundary_reasons"]}
    reasons |= {item.get("reason") for item in result["rejected_pair_reasons"]}
    assert reasons - {None}


def test_separation_shape_residual_flags_non_parallel_boundaries():
    """The parallelism metric is small for smooth width changes, large for wobble."""

    from evaluation.lane_width_stability import _separation_shape_residual

    stations = np.linspace(0.0, 40.0, 60)
    constant = np.full(stations.size, 3.7)
    taper = 4.3 - 0.03 * stations
    merge = 3.7 - 0.05 * np.maximum(stations - 22.0, 0.0)
    erratic = 3.7 + 0.9 * np.sin(stations / 2.0) + 0.5 * np.sin(stations / 0.7)
    assert _separation_shape_residual(stations, constant) < 0.02
    assert _separation_shape_residual(stations, taper) < 0.02
    assert _separation_shape_residual(stations, merge) < 0.05
    assert _separation_shape_residual(stations, erratic) > 0.18


def test_squat_outline_corridor_is_rejected_by_aspect_ratio():
    """A box-like corridor (as wide as it is long) is not a travel lane."""

    y = np.linspace(240.0, 340.0, 50)
    left = np.column_stack([np.full(y.size, 150.0), y])
    right = np.column_stack([np.full(y.size, 330.0), y])
    result = _uncalibrated([left, right])
    assert result["lane_width_stability"] is None
    assert any(
        item.get("reason") == "implausible_corridor_aspect_ratio"
        for item in result["rejected_pair_reasons"]
    )


def test_orientation_gate_is_load_bearing_for_horizontal_clutter():
    """Disabling the longitudinal-orientation gate lets crosswalk clutter score.

    This confirms the gate is exactly what prevents near-horizontal clutter from
    being scored as a confident width, and that it is configurable rather than
    hard-wired for hypothetical non-forward-facing cameras.
    """

    top = np.column_stack([np.linspace(60.0, 300.0, 30), np.linspace(232.0, 236.0, 30)])
    bottom = np.column_stack([np.linspace(40.0, 320.0, 30), np.linspace(300.0, 304.0, 30)])
    gated = _uncalibrated([top, bottom])
    permissive = analyze_lane_width_stability(
        [top, bottom],
        image_shape=(HEIGHT, WIDTH),
        geometry_source="mask_derived",
        homography=None,
        config=LaneWidthConfig(enforce_longitudinal_orientation=False),
    )
    assert gated["lane_width_stability"] is None
    assert permissive["lane_width_stability"] is not None


def test_legitimate_taper_and_merge_survive_the_plausibility_gates():
    """The new gates must not reject smooth intentional width changes."""

    taper_left, taper_right, _, _ = _road_pair("taper")
    transition_left, transition_right, _, _ = _road_pair("transition")
    taper = _analyze([taper_left, taper_right])
    transition = _analyze([transition_left, transition_right])
    assert taper["valid_pair_count"] == 1
    assert transition["valid_pair_count"] == 1


@pytest.mark.parametrize(
    "mask_name",
    ["b7edcfdc-f018989f", "be60191c-cade58a2"],
)
def test_real_world_implausible_predictions_are_unavailable(mask_name):
    """Regression guard for the two documented YOLOPX failure cases.

    ``b7edcfdc`` traced a tunnel side wall and a pavement noise trail and
    previously scored I4_pred=0.985; ``be60191c`` traced a crosswalk outline and
    previously scored I4_pred=0.204.  Both must now be unavailable.
    """

    import os

    mask_path = (
        "outputs/full_model_rankings/predictions/yolopx/bdd100k_lane/masks/"
        f"{mask_name}.png"
    )
    if not os.path.exists(mask_path):
        pytest.skip("real-world prediction fixture not present in this checkout")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    grouped = extract_grouped_boundaries_from_mask(mask)
    result = analyze_lane_width_stability(
        grouped,
        image_shape=mask.shape,
        geometry_source="mask_derived",
        homography=None,
    )
    assert result["lane_width_stability"] is None
    assert result["valid_pair_count"] == 0
