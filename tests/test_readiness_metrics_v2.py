import json

import numpy as np
import pytest

from lane_eval.schema.lane import LaneTarget
from lane_eval.schema.sample import LaneSample
from evaluation.readiness_metrics import (
    build_metric_record,
    compute_correlations_from_records,
    compute_d1_valid_detection_single,
    compute_d3_iou_single,
    compute_d8_confidence_mean_single,
    compute_i1_pattern_continuity,
    compute_i2_local_contrast,
    compute_r2_reference_detectability_score,
    compute_tolerant_precision_recall_f1,
    compute_d6_missed_marking_ratio_single,
    summarize_records,
)


def _sample(image_id, mask, meta=None):
    h, w = mask.shape
    return LaneSample(
        image_id=image_id,
        image_path=f"/tmp/{image_id}.jpg",
        width=w,
        height=h,
        target=LaneTarget(mask=mask),
        meta=meta or {},
    )


def _dashed_mask(h=120, w=160):
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(20, 110, 20):
        mask[y:y + 10, 78:84] = 1
    return mask


def _image_for_mask(mask, road=50, lane=220):
    img = np.full((*mask.shape, 3), road, dtype=np.uint8)
    img[mask > 0] = lane
    return img


def test_local_contrast_uses_local_ring_not_distant_sky():
    h, w = 160, 180
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[90:150, 88:94] = 1
    img = np.full((h, w, 3), 45, dtype=np.uint8)
    img[:60, :, :] = 245
    img[mask > 0] = 210

    assert compute_i2_local_contrast(img, mask) > 0.55


def test_dashed_continuity_is_high_and_drops_for_missing_or_low_contrast_dashes():
    mask = _dashed_mask()
    img = _image_for_mask(mask)
    clear = compute_i1_pattern_continuity(img, mask, meta={"style": "dashed"})

    missing = mask.copy()
    missing[60:70, :] = 0
    missing_score = compute_i1_pattern_continuity(img, missing, meta={"style": "dashed"})

    low_img = _image_for_mask(mask, road=50, lane=60)
    low_score = compute_i1_pattern_continuity(low_img, mask, meta={"style": "dashed"})

    assert clear > 0.70
    assert missing_score < clear
    assert low_score < clear


def test_solid_continuity_drops_for_gap():
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[20:110, 78:84] = 1
    img = _image_for_mask(mask)
    continuous = compute_i1_pattern_continuity(img, mask, meta={"style": "solid"})

    gapped = mask.copy()
    gapped[55:75, :] = 0
    gapped_score = compute_i1_pattern_continuity(img, gapped, meta={"style": "solid"})

    assert gapped_score < continuous


def test_tolerant_f1_and_zero_prediction_detection():
    gt = np.zeros((80, 100), dtype=np.uint8)
    gt[20:70, 48:54] = 1
    assert compute_d3_iou_single(gt, gt) == pytest.approx(1.0)
    assert compute_tolerant_precision_recall_f1(gt, gt, tolerance_px=5)["f1"] == pytest.approx(1.0)

    shifted = np.roll(gt, 2, axis=1)
    assert compute_d3_iou_single(shifted, gt) < 0.6
    assert compute_tolerant_precision_recall_f1(shifted, gt, tolerance_px=5)["f1"] > 0.95

    zero = np.zeros_like(gt)
    assert compute_d6_missed_marking_ratio_single(zero, gt) == pytest.approx(1.0)
    assert compute_d1_valid_detection_single(zero, gt) == 0


def test_d8_none_and_r2_renormalization_and_real_probability():
    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[5:10, 5:10] = 1
    assert compute_d8_confidence_mean_single(pred, None) is None

    metrics = {"D3_tolerant_f1_r5": 1.0, "D3_iou": 0.5, "D4_near_f1_r5": 1.0, "D6_missed_marking_ratio": 0.0, "D8_confidence_mean": None}
    assert compute_r2_reference_detectability_score(metrics) == pytest.approx(((0.35 + 0.10 + 0.20 + 0.20) / 0.95) * 100.0)

    prob = np.full((20, 20), 0.2, dtype=np.float32)
    prob[pred > 0] = 0.8
    assert compute_d8_confidence_mean_single(pred, prob) == pytest.approx(0.8)


def test_correlations_use_per_image_values_and_report_skip_reasons():
    records = []
    for i in range(12):
        v = i / 11.0
        records.append({
            "I1_pattern_continuity": v,
            "I2_local_contrast": v,
            "I3_boundary_sharpness": v,
            "I5_geometry_complexity": v,
            "D3_tolerant_f1_r5": v,
            "D3_iou": v,
            "D6_missed_marking_ratio": 1.0 - v,
            "R1_marking_readability_score": v * 100,
            "R2_reference_detectability_score": v * 100,
            "metadata": {"weather": "clear" if i < 6 else "rainy"},
        })
    corr = compute_correlations_from_records(records, min_samples=10)
    assert corr["C1"] > 0.95
    assert corr["C2"] > 0.95

    small = compute_correlations_from_records(records[:5], min_samples=10)
    assert small["C1"] is None
    assert "insufficient_samples" in small["C1_detail"]["primary"]["reason"]

    constant = [dict(r, I2_local_contrast=0.5) for r in records]
    zero_var = compute_correlations_from_records(constant, min_samples=10)
    assert zero_var["C2"] is None
    assert zero_var["C2_detail"]["D3_iou"]["reason"] == "zero_variance"


def test_record_stratification_preserves_image_prediction_alignment():
    records = []
    for i, weather in enumerate(["clear", "rainy", "clear"]):
        gt = np.zeros((40, 40), dtype=np.uint8)
        gt[10:30, 18:22] = 1
        pred = np.zeros_like(gt)
        pred[10:10 + (i + 1) * 5, 18:22] = 1
        img = _image_for_mask(gt)
        records.append(build_metric_record(_sample(f"img_{i}", gt, {"weather": weather}), pred, img, dataset="synthetic"))

    clear = [r for r in records if r["metadata"].get("weather") == "clear"]
    assert [r["image_id"] for r in clear] == ["img_0", "img_2"]
    assert [r["pred_nonzero"] for r in clear] == [20, 60]


def test_summarize_records_is_json_serializable_without_nan():
    records = []
    for i in range(3):
        gt = np.zeros((60, 60), dtype=np.uint8)
        gt[15:50, 28:32] = 1
        img = _image_for_mask(gt)
        records.append(build_metric_record(_sample(f"json_{i}", gt), gt.copy(), img))
    summary = summarize_records(records, min_corr_samples=10)
    json.dumps(summary, allow_nan=False)
    assert summary["layer2"]["D7_temporal_jitter"] is None
