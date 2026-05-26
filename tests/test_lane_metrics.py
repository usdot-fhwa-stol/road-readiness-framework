"""Tests for LaneSegmentationEvaluator (wraps YOLOPX SegmentationMetric)."""
import numpy as np
import pytest

YOLOPX_REPO = "/home/gauravb/Projects/road_readiness_t3/YOLOPX"


@pytest.fixture
def evaluator():
    from lane_eval.evaluators.lane_segmentation import LaneSegmentationEvaluator
    return LaneSegmentationEvaluator(yolopx_repo=YOLOPX_REPO)


def test_perfect_prediction(evaluator):
    gt = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    evaluator.update(gt, gt)
    r = evaluator.compute()
    assert r["lane_iou"] == pytest.approx(1.0)
    assert r["lane_accuracy"] == pytest.approx(1.0)


def test_all_zeros_prediction(evaluator):
    gt   = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    pred = np.zeros_like(gt)
    evaluator.update(pred, gt)
    r = evaluator.compute()
    assert r["lane_iou"] == pytest.approx(0.0)


def test_shape_mismatch_raises(evaluator):
    pred = np.zeros((100, 200), dtype=np.uint8)
    gt   = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluator.update(pred, gt)


def test_accumulates_over_multiple_images(evaluator):
    gt = np.ones((10, 10), dtype=np.uint8)
    evaluator.update(gt, gt)
    evaluator.update(gt, gt)
    r = evaluator.compute()
    assert r["num_images"] == 2
    assert r["lane_iou"] == pytest.approx(1.0)


def test_reset_clears_state(evaluator):
    gt = np.ones((10, 10), dtype=np.uint8)
    evaluator.update(gt, gt)
    evaluator.reset()
    r = evaluator.compute()
    assert r["num_images"] == 0
