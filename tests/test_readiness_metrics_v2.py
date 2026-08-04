import json

import numpy as np
import pytest

from lane_eval.schema.lane import LaneTarget
from lane_eval.schema.sample import LaneSample
from evaluation.d_metrics import (
    compute_d1_detection,
    compute_d3_iou,
    compute_d6_image_counts,
    compute_d7_confidence,
)
from evaluation.readiness_metrics import (
    build_metric_record,
    compute_correlations_from_records,
    compute_i1_pattern_continuity,
    compute_i2_local_contrast,
    compute_r2_reference_detectability_score,
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


def _dashed_mask(h=240, w=160):
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(20, 230, 25):
        mask[y:y + 12, 78:84] = 1
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


def test_dashed_continuity_is_high_and_invisible_dashes_are_unavailable():
    """Clear dashes score high; dashes below paint detectability return None.

    The redesigned I1 uses relative center-versus-side paint evidence, so it is
    intentionally robust to moderate brightness changes. When contrast collapses
    to the noise floor (dashes at luminance 60 on a road of 50) the marking is
    effectively invisible, and the engine must report insufficient evidence
    rather than fabricate a confident low continuity -- the same failure mode as
    the pitch-dark night frame.
    """

    paint = _dashed_mask()
    guidance = np.zeros_like(paint)
    guidance[15:230, 78:84] = 1
    img = _image_for_mask(paint)
    clear = compute_i1_pattern_continuity(img, guidance, meta={"style": "dashed"})

    invisible_img = _image_for_mask(paint, road=50, lane=60)
    invisible = compute_i1_pattern_continuity(invisible_img, guidance, meta={"style": "dashed"})

    # A faded-but-still-visible dash (clear contrast reduced, delta ~40) must
    # still produce a real, credible score rather than being discarded.
    faded_img = _image_for_mask(paint, road=50, lane=95)
    faded = compute_i1_pattern_continuity(faded_img, guidance, meta={"style": "dashed"})

    assert clear > 0.60
    assert invisible is None
    assert faded is not None and faded > 0.50


def test_solid_continuity_drops_for_gap():
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[20:110, 78:84] = 1
    img = _image_for_mask(mask)
    continuous = compute_i1_pattern_continuity(img, mask, meta={"style": "solid"})

    gapped_img = img.copy()
    gapped_img[55:75, :] = 50
    gapped_score = compute_i1_pattern_continuity(gapped_img, mask, meta={"style": "solid"})

    assert gapped_score < continuous


def test_d3_iou_and_zero_prediction_detection():
    gt = np.zeros((80, 100), dtype=np.uint8)
    gt[20:70, 48:54] = 1
    assert compute_d3_iou(gt, gt)["D3_iou"] == pytest.approx(1.0)

    shifted = np.roll(gt, 2, axis=1)
    assert compute_d3_iou(shifted, gt)["D3_iou"] < 0.6

    zero = np.zeros_like(gt)
    assert compute_d6_image_counts(zero, gt)["D6_image_missed_ratio"] == pytest.approx(1.0)
    assert compute_d1_detection(zero)["D1_detection"] == 0


def test_d7_none_and_r2_renormalization_and_real_probability():
    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[5:10, 5:10] = 1
    assert compute_d7_confidence(None, pred)["D7_confidence_mean"] is None

    metrics = {"D3_f1": 1.0, "D3_iou": 0.5, "D4_near_iou": 1.0, "D6_image_missed_ratio": 0.0, "D7_confidence_mean": None}
    assert compute_r2_reference_detectability_score(metrics) == pytest.approx(((0.35 + 0.10 + 0.20 + 0.20) / 0.95) * 100.0)

    prob = np.full((20, 20), 0.2, dtype=np.float32)
    prob[pred > 0] = 0.8
    assert compute_d7_confidence(prob, pred)["D7_confidence_mean"] == pytest.approx(0.8)


def test_correlations_use_per_image_values_and_report_skip_reasons():
    records = []
    for i in range(12):
        v = i / 11.0
        records.append({
            "I1_pattern_continuity": v,
            "I2_local_contrast": v,
            "I3_boundary_sharpness": v,
            "I5_geometry_complexity": v,
            "D3_f1": v,
            "D3_iou": v,
            "D6_image_missed_ratio": 1.0 - v,
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
    assert summary["layer2"]["D7_confidence_mean"] is None
