"""Unit tests for the native TuSimple / CULane lane evaluators."""
import numpy as np

from lane_eval.evaluators import TuSimpleNativeEvaluator, CULaneNativeEvaluator


def _vertical_lane(x, h_samples):
    """A perfectly vertical lane at column x across all h_samples."""
    return [x for _ in h_samples]


H = list(range(10, 110, 10))  # [10..100]


def _lane_json(lanes):
    return {"h_samples": H, "lanes": lanes}


# ── TuSimple ──────────────────────────────────────────────────────────────────

def test_tusimple_perfect_match():
    gt = _lane_json([_vertical_lane(100, H), _vertical_lane(400, H)])
    ev = TuSimpleNativeEvaluator()
    ev.update(gt, gt)  # predict exactly the GT
    r = ev.compute()
    assert r["accuracy"] == 1.0
    assert r["fp"] == 0.0 and r["fn"] == 0.0
    assert r["num_images"] == 1


def test_tusimple_empty_prediction():
    gt = _lane_json([_vertical_lane(100, H), _vertical_lane(400, H)])
    ev = TuSimpleNativeEvaluator()
    ev.update(_lane_json([]), gt)
    r = ev.compute()
    assert r["accuracy"] == 0.0
    assert r["fn"] == 1.0   # all GT lanes missed


def test_tusimple_far_off_prediction_low_acc():
    gt = _lane_json([_vertical_lane(100, H)])
    pred = _lane_json([_vertical_lane(300, H)])  # 200px off, way beyond 20px thresh
    ev = TuSimpleNativeEvaluator()
    ev.update(pred, gt)
    r = ev.compute()
    assert r["accuracy"] == 0.0
    assert r["fp"] == 1.0   # the prediction matched nothing
    assert r["fn"] == 1.0


# ── CULane ──────────────────────────────────────────────────────────────────

def test_culane_perfect_match():
    gt = _lane_json([_vertical_lane(100, H), _vertical_lane(400, H)])
    ev = CULaneNativeEvaluator()
    ev.update(gt, gt, height=120, width=500)
    r = ev.compute()
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0


def test_culane_missed_lane():
    gt = _lane_json([_vertical_lane(100, H), _vertical_lane(400, H)])
    pred = _lane_json([_vertical_lane(100, H)])  # only one of two predicted
    ev = CULaneNativeEvaluator()
    ev.update(pred, gt, height=120, width=500)
    r = ev.compute()
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 1
    assert r["recall"] == 0.5 and r["precision"] == 1.0


def test_culane_false_positive():
    gt = _lane_json([_vertical_lane(100, H)])
    pred = _lane_json([_vertical_lane(100, H), _vertical_lane(450, H)])  # one spurious
    ev = CULaneNativeEvaluator()
    ev.update(pred, gt, height=120, width=500)
    r = ev.compute()
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 0


def test_culane_no_overlap_is_fp_and_fn():
    gt = _lane_json([_vertical_lane(100, H)])
    pred = _lane_json([_vertical_lane(450, H)])  # far away, IoU 0
    ev = CULaneNativeEvaluator()
    ev.update(pred, gt, height=120, width=500)
    r = ev.compute()
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1
