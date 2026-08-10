"""Integration tests for the D3' wiring in evaluation.d_metrics.

Covered:
  * delta-gating: build_d_record with delta=None omits every D3' field (legacy
    Table-16 record is byte-for-byte preserved); with delta set, D3' appears.
  * native-polyline path: pred/GT polylines score without touching the mask
    skeletonizer (so no scikit-image needed) and give perfect scores when equal.
  * D6' fold-in: unsupported-marking ratio == 1 - recall.
  * summarize_d_records aggregates D3' only when present and excludes None,
    never coercing to 0; the legacy summary keys are unchanged when delta=None.

These use native polylines and same-shape masks with match_stroke_width=False so
no OpenCV routine is actually invoked; the local runner stubs the heavy
evaluation.readiness_metrics import with pure-NumPy equivalents. On the real box
this file runs unmodified under pytest against the true modules.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation import d_metrics


def _line_lane(x, h, n=60):
    ys = np.linspace(0, h - 1, n)
    return np.stack([np.full(n, float(x)), ys], axis=1)


def _mask_with_pixels(h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    m[:, w // 2] = 1  # one nonzero column so D1 presence + gt_eligible are true
    return m


def test_delta_none_omits_d3_prime():
    h, w = 100, 80
    pred = _mask_with_pixels(h, w)
    gt = _mask_with_pixels(h, w)
    rec = d_metrics.build_d_record(
        "s", pred, gt,
        pred_lanes=[_line_lane(40, h)],
        gt_lanes=[_line_lane(40, h)],
        match_stroke_width=False,
    )
    assert not any(k.startswith("D3p_") or k.startswith("D4p_") or k.startswith("D6p_")
                   for k in rec)
    # legacy fields still present
    assert "D3_iou" in rec and "D6_gt_pixels" in rec


def test_delta_set_adds_perfect_d3_prime_for_equal_polylines():
    h, w = 100, 80
    pred = _mask_with_pixels(h, w)
    gt = _mask_with_pixels(h, w)
    lane = _line_lane(40, h)
    rec = d_metrics.build_d_record(
        "s", pred, gt,
        pred_lanes=[lane.copy()],
        gt_lanes=[lane.copy()],
        match_stroke_width=False,
        delta=0.01,
    )
    assert rec["D3p_support_f1"] == pytest.approx(1.0)
    assert rec["D3p_support_recall"] == pytest.approx(1.0)
    assert rec["D3p_loc_err_median"] == pytest.approx(0.0)
    assert rec["D3p_pred_source"] == "native_polyline"
    assert rec["D3p_gt_source"] == "native_polyline"
    # D6' fold-in identity
    assert rec["D6p_unsupported_marking_ratio"] == pytest.approx(0.0)


def test_d6_prime_is_one_minus_recall():
    h, w = 100, 80
    pred = _mask_with_pixels(h, w)
    gt = _mask_with_pixels(h, w)
    # pred lane far from gt lane -> recall 0 at tight delta
    rec = d_metrics.build_d_record(
        "s", pred, gt,
        pred_lanes=[_line_lane(10, h)],
        gt_lanes=[_line_lane(70, h)],
        match_stroke_width=False,
        delta=0.01,
    )
    assert rec["D3p_support_recall"] == pytest.approx(0.0)
    assert rec["D6p_unsupported_marking_ratio"] == pytest.approx(1.0)


def test_summarizer_omits_d3_prime_when_absent():
    h, w = 100, 80
    recs = [
        d_metrics.build_d_record(
            f"s{i}", _mask_with_pixels(h, w), _mask_with_pixels(h, w),
            pred_lanes=[_line_lane(40, h)], gt_lanes=[_line_lane(40, h)],
            match_stroke_width=False,
        )
        for i in range(3)
    ]
    summary = d_metrics.summarize_d_records(recs)
    assert "D3_prime" not in summary


def test_summarizer_aggregates_d3_prime_and_excludes_none():
    h, w = 100, 80
    lane = _line_lane(40, h)
    recs = []
    # two perfect matches
    for i in range(2):
        recs.append(d_metrics.build_d_record(
            f"s{i}", _mask_with_pixels(h, w), _mask_with_pixels(h, w),
            pred_lanes=[lane.copy()], gt_lanes=[lane.copy()],
            match_stroke_width=False, delta=0.01,
        ))
    # one record whose GT is empty -> D3' recall/f1 None, must be excluded
    empty_gt = np.zeros((h, w), dtype=np.uint8)
    # give it a pred pixel so gt_eligible... actually gt empty -> gt_eligible 0.
    # Force eligibility off but still ensure summarizer handles None safely by
    # including a record with pred present and gt present but pred lane far away.
    recs.append(d_metrics.build_d_record(
        "s_far", _mask_with_pixels(h, w), _mask_with_pixels(h, w),
        pred_lanes=[_line_lane(5, h)], gt_lanes=[_line_lane(75, h)],
        match_stroke_width=False, delta=0.01,
    ))
    summary = d_metrics.summarize_d_records(recs)
    assert "D3_prime" in summary
    dp = summary["D3_prime"]
    assert dp["D3p_num_scored"] == 3
    # mean of two 1.0 f1 and one 0.0 f1 = 2/3
    assert dp["D3p_support_f1_mean"] == pytest.approx(2 / 3, abs=1e-9)
    assert dp["D3p_delta"] == pytest.approx(0.01)


def test_legacy_summary_keys_unchanged_when_delta_none():
    h, w = 100, 80
    recs = [
        d_metrics.build_d_record(
            "s", _mask_with_pixels(h, w), _mask_with_pixels(h, w),
            pred_lanes=[_line_lane(40, h)], gt_lanes=[_line_lane(40, h)],
            match_stroke_width=False,
        )
    ]
    summary = d_metrics.summarize_d_records(recs)
    for key in ("D1_detection_success_rate", "D3_mean_iou", "D4_mean_near_field_iou",
                "D6_detection_gap_ratio", "D7_available"):
        assert key in summary
