"""Unit tests for evaluation.delta_calibration (delta + operating-point freeze).

These exercise the pure selection math on synthetic per-image distance
distributions (the GPU box supplies the real ones):
  * F1-vs-delta curve is monotone non-decreasing in delta and respects the
    None-semantics (empty GT -> recall excluded; empty pred -> precision
    excluded);
  * the knee lands at the characteristic error scale of a bimodal distribution;
  * select_delta prefers the knee, falls back to the error percentile;
  * bootstrap stability reports tight for a large clean set and is deterministic
    under a fixed seed;
  * operating-point selection picks the tau that maximizes calibration F1.

Pure NumPy; runs in the minimal repo env (imports evaluation.marking_support
only indirectly — delta_calibration does not need cv2).
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation import delta_calibration as dc


def _rec(gt_to_pred, pred_to_gt):
    return {"gt_to_pred": np.asarray(gt_to_pred, float),
            "pred_to_gt": np.asarray(pred_to_gt, float)}


# --------------------------------------------------------------------------- #
# support_f1_curve
# --------------------------------------------------------------------------- #
def test_curve_monotone_in_delta():
    # errors around 0.01; recall/precision rise as delta grows
    d = [_rec(np.full(100, 0.01), np.full(100, 0.01)) for _ in range(5)]
    out = dc.support_f1_curve(d, delta_grid=[0.001, 0.005, 0.02, 0.05])
    f1 = [row["mean_f1"] for row in out["curve"]]
    assert f1[0] <= f1[1] <= f1[2] <= f1[3]
    assert f1[0] == pytest.approx(0.0)   # delta well below error -> no support
    assert f1[-1] == pytest.approx(1.0)  # delta well above error -> full support


def test_curve_empty_gt_excluded_from_recall():
    # one image has empty GT (recall undefined), one normal
    d = [
        {"gt_to_pred": np.zeros(0), "pred_to_gt": np.full(10, 0.001)},  # empty GT
        _rec(np.full(10, 0.001), np.full(10, 0.001)),
    ]
    out = dc.support_f1_curve(d, delta_grid=[0.01])
    row = out["curve"][0]
    # recall mean over the ONE image that defines recall -> 1.0
    assert row["mean_recall"] == pytest.approx(1.0)
    # precision defined for both images -> 1.0
    assert row["mean_precision"] == pytest.approx(1.0)
    # f1 averaged only over images where both defined -> 1 image
    assert row["n_images_f1"] == 1


def test_curve_empty_pred_excluded_from_precision():
    d = [
        {"gt_to_pred": np.full(10, np.inf), "pred_to_gt": np.zeros(0)},  # empty pred
        _rec(np.full(10, 0.001), np.full(10, 0.001)),
    ]
    out = dc.support_f1_curve(d, delta_grid=[0.01])
    row = out["curve"][0]
    # image 1: recall over inf distances = 0; image 2: recall 1 -> mean 0.5
    assert row["mean_recall"] == pytest.approx(0.5)
    # precision defined only for image 2 -> 1.0
    assert row["mean_precision"] == pytest.approx(1.0)
    assert row["n_images_f1"] == 1


def test_curve_rejects_bad_grid():
    with pytest.raises(ValueError):
        dc.support_f1_curve([], delta_grid=[])
    with pytest.raises(ValueError):
        dc.support_f1_curve([], delta_grid=[-0.1])


# --------------------------------------------------------------------------- #
# knee
# --------------------------------------------------------------------------- #
def test_knee_at_characteristic_scale():
    # all errors ~0.01; F1 curve steps up near delta=0.01, knee should be near it
    d = [_rec(np.full(200, 0.01), np.full(200, 0.01)) for _ in range(10)]
    grid = list(np.geomspace(0.001, 0.1, 21))
    res = dc.select_delta(d, delta_grid=grid)
    assert res["delta_knee"] is not None
    # knee within a factor of ~3 of the true scale (grid is coarse/log)
    assert 0.003 <= res["delta_knee"] <= 0.03


def test_select_delta_prefers_knee():
    d = [_rec(np.full(200, 0.01), np.full(200, 0.01)) for _ in range(10)]
    res = dc.select_delta(d, delta_grid=list(np.geomspace(0.001, 0.1, 21)))
    assert res["selection_rule"] == "f1_knee"
    assert res["delta_selected"] == res["delta_knee"]


def test_select_delta_fallback_to_percentile_when_flat():
    # constant-perfect curve (all zeros) -> F1 flat at 1.0 -> no knee -> percentile
    d = [_rec(np.zeros(50), np.zeros(50)) for _ in range(5)]
    res = dc.select_delta(d, delta_grid=[0.001, 0.01, 0.05])
    assert res["delta_knee"] is None
    assert res["selection_rule"].startswith("gt_error_p")
    assert res["delta_selected"] == pytest.approx(0.0)  # p90 of all-zero errors


def test_percentile_excludes_infinite():
    # half the images have empty preds (inf errors); percentile uses finite only
    d = [_rec(np.full(10, 0.02), np.full(10, 0.02)) for _ in range(5)]
    d += [{"gt_to_pred": np.full(10, np.inf), "pred_to_gt": np.zeros(0)} for _ in range(5)]
    p = dc._percentile_delta(d, 90.0)
    assert p == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# bootstrap stability
# --------------------------------------------------------------------------- #
def test_bootstrap_deterministic_and_stable():
    d = [_rec(np.full(200, 0.01), np.full(200, 0.01)) for _ in range(200)]
    grid = list(np.geomspace(0.001, 0.1, 21))
    a = dc.bootstrap_delta_stability(d, delta_grid=grid, n_boot=50, seed=3)
    b = dc.bootstrap_delta_stability(d, delta_grid=grid, n_boot=50, seed=3)
    assert a["deltas"] == b["deltas"]                    # deterministic
    assert a["stable_within_one_grid_step"] in (True, False)
    # identical images -> bootstrap delta should barely move
    assert a["stable_within_one_grid_step"] is True


def test_bootstrap_empty():
    out = dc.bootstrap_delta_stability([], n_boot=10)
    assert out["stable_within_one_grid_step"] is None


# --------------------------------------------------------------------------- #
# operating point
# --------------------------------------------------------------------------- #
def test_operating_point_picks_best_tau():
    # tau=0.5 gives tight errors (high F1); other taus give large errors (low F1)
    def dists_at(tau):
        err = 0.005 if abs(tau - 0.5) < 1e-9 else 0.2
        return [_rec(np.full(50, err), np.full(50, err)) for _ in range(5)]

    res = dc.select_operating_point([0.3, 0.5, 0.7], dists_at, reference_delta=0.01)
    assert res["tau_selected"] == pytest.approx(0.5)
    assert res["tau_f1"] == pytest.approx(1.0)


def test_operating_point_rejects_bad_delta():
    with pytest.raises(ValueError):
        dc.select_operating_point([0.5], lambda t: [], reference_delta=-1.0)
