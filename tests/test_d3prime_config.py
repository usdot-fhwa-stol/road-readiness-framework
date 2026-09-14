"""Unit tests for evaluation.d3prime_config (frozen delta loader).

The load-bearing contract: when no frozen config exists, load_frozen_delta
returns None so D3' stays dormant and the legacy pipeline is unchanged. A
present config returns its delta; a malformed one fails loudly.

Pure stdlib; runs in the minimal repo env.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from evaluation import d3prime_config as cfg


def _write(tmp, obj):
    path = os.path.join(tmp, "frozen.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def test_absent_config_returns_none():
    # a path that does not exist -> None (dormant), not an error
    assert cfg.load_frozen_delta("dataset/calibration/__does_not_exist__.json") is None
    assert cfg.load_config("dataset/calibration/__does_not_exist__.json") is None


def test_present_config_returns_delta():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"frozen_config_version": "x", "delta": 0.012})
        assert cfg.load_frozen_delta(path) == pytest.approx(0.012)


def test_malformed_config_missing_delta_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"frozen_config_version": "x"})
        with pytest.raises(ValueError):
            cfg.load_frozen_delta(path)


def test_malformed_config_negative_delta_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"delta": -0.01})
        with pytest.raises(ValueError):
            cfg.load_frozen_delta(path)


def test_operating_point_substring_match():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"delta": 0.01, "operating_points": {"yolopx": 0.5, "clrernet": 0.3}})
        assert cfg.operating_point("yolopx_culane", path) == pytest.approx(0.5)
        assert cfg.operating_point("clrernet", path) == pytest.approx(0.3)
        assert cfg.operating_point("unknown_model", path) is None


def test_build_frozen_config_shape():
    calib = {"calibration_split_version": "v1", "seed": 7, "n_calibration": 10,
             "n_total": 200, "achieved_fraction": 0.05,
             "manifest_fingerprint_sha256": "abc"}
    out = cfg.build_frozen_config(
        delta=0.011, operating_points={"yolopx": None},
        selection_rule="f1_knee", calibration_artifact=calib,
        stability={"stable_within_one_grid_step": True}, delta_grid=[0.002, 0.05],
    )
    assert out["delta"] == pytest.approx(0.011)
    assert out["delta_units"] == "fraction_of_image_diagonal"
    assert out["selection_rule"] == "f1_knee"
    assert out["calibration"]["manifest_fingerprint_sha256"] == "abc"
    assert out["bootstrap_stability"]["stable_within_one_grid_step"] is True


def test_build_frozen_config_rejects_bad_delta():
    with pytest.raises(ValueError):
        cfg.build_frozen_config(delta=-1.0, operating_points={}, selection_rule="x",
                                calibration_artifact={})
