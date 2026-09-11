"""Unit tests for evaluation.calibration_split (stratified delta-freeze split).

These exercise the correctness-critical properties of the draw:
  * stratification by dataset x coarse visibility tier,
  * determinism (same manifest+seed+fraction -> identical ids),
  * seed-independence of one stratum from another,
  * the per-stratum floor (rare tiers not dropped to zero),
  * the calibration ids are a subset of the manifest and de-dupe correctly,
  * partition() returns disjoint calibration/eval id lists covering the manifest.

Pure stdlib + NumPy; runs in the minimal repo env.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation import calibration_split as cs


def _sample(sid, dataset, vis=None):
    meta = {}
    if vis is not None:
        meta = {"predicted_tags": {"by_dimension": {"Observed Marking Visibility": vis}}}
    return {"sample_id": sid, "dataset": dataset, "meta": meta}


def _manifest(n_per=(("culane", 100), ("tusimple", 100)), vis=("Clearly Visible",)):
    samples = []
    for ds, n in n_per:
        for i in range(n):
            samples.append(_sample(f"{ds}_{i}", ds, list(vis)))
    return samples


# --------------------------------------------------------------------------- #
# coarse tier mapping
# --------------------------------------------------------------------------- #
def test_tier_occluded_dominates():
    assert cs.coarse_visibility_tier(["Clearly Visible", "Occluded"]) == "occluded"


def test_tier_degraded():
    assert cs.coarse_visibility_tier(["Low Contrast"]) == "degraded"
    assert cs.coarse_visibility_tier(["Faded / Worn / Not Visible"]) == "degraded"


def test_tier_clear_only_when_no_degraded():
    assert cs.coarse_visibility_tier(["Clearly Visible"]) == "clear"


def test_tier_unknown_when_empty():
    assert cs.coarse_visibility_tier([]) == "unknown"
    assert cs.coarse_visibility_tier(["something else"]) == "unknown"


def test_stratum_reads_manifest_layout():
    s = _sample("x", "culane", ["Occluded"])
    assert cs.stratum_of(s) == ("culane", "occluded")


# --------------------------------------------------------------------------- #
# the draw
# --------------------------------------------------------------------------- #
def test_fraction_and_size():
    art = cs.select_calibration_ids(_manifest(), fraction=0.05, min_per_stratum=1)
    # 200 images, 5% -> ~10; each stratum (2) draws round(0.05*100)=5 -> 10 total
    assert art["n_calibration"] == 10
    assert art["n_total"] == 200
    assert art["n_strata"] == 2


def test_determinism_same_seed():
    a = cs.select_calibration_ids(_manifest(), fraction=0.1, seed=7)
    b = cs.select_calibration_ids(_manifest(), fraction=0.1, seed=7)
    assert a["calibration_sample_ids"] == b["calibration_sample_ids"]


def test_different_seed_changes_draw():
    a = cs.select_calibration_ids(_manifest(), fraction=0.1, seed=1)
    b = cs.select_calibration_ids(_manifest(), fraction=0.1, seed=2)
    assert a["calibration_sample_ids"] != b["calibration_sample_ids"]


def test_ids_are_subset_and_unique():
    samples = _manifest()
    art = cs.select_calibration_ids(samples, fraction=0.1)
    ids = {s["sample_id"] for s in samples}
    chosen = art["calibration_sample_ids"]
    assert len(chosen) == len(set(chosen))          # unique
    assert set(chosen).issubset(ids)                # subset


def test_stratification_hits_every_stratum():
    # two datasets x two tiers -> 4 strata, all should be represented
    samples = []
    for ds in ("culane", "tusimple"):
        for i in range(100):
            samples.append(_sample(f"{ds}_clear_{i}", ds, ["Clearly Visible"]))
        for i in range(100):
            samples.append(_sample(f"{ds}_deg_{i}", ds, ["Low Contrast"]))
    art = cs.select_calibration_ids(samples, fraction=0.05, min_per_stratum=1)
    assert art["n_strata"] == 4
    for rep in art["per_stratum"].values():
        assert rep["n_calibration"] >= 1


def test_floor_keeps_rare_stratum():
    # one dataset has a rare occluded tier of only 3 images; floor=5 -> take all 3
    samples = _manifest(n_per=(("culane", 100),), vis=("Clearly Visible",))
    samples += [_sample(f"culane_occ_{i}", "culane", ["Occluded"]) for i in range(3)]
    art = cs.select_calibration_ids(samples, fraction=0.01, min_per_stratum=5)
    occ = art["per_stratum"]["culane/occluded"]
    assert occ["n_total"] == 3
    assert occ["n_calibration"] == 3  # capped at stratum size, not 5


def test_floor_zero_allows_empty_small_stratum_draw():
    samples = _manifest(n_per=(("culane", 100),))
    art = cs.select_calibration_ids(samples, fraction=0.05, min_per_stratum=0)
    # 5% of 100 = 5
    assert art["per_stratum"]["culane/clear"]["n_calibration"] == 5


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        cs.select_calibration_ids(_manifest(), fraction=0.0)
    with pytest.raises(ValueError):
        cs.select_calibration_ids(_manifest(), fraction=1.0)


def test_duplicate_ids_raise():
    samples = [_sample("dup", "culane", ["Clearly Visible"]),
               _sample("dup", "tusimple", ["Clearly Visible"])]
    with pytest.raises(ValueError):
        cs.select_calibration_ids(samples, fraction=0.5)


def test_fingerprint_order_independent():
    a = cs._manifest_fingerprint(["b", "a", "c"])
    b = cs._manifest_fingerprint(["c", "b", "a"])
    assert a == b


def test_partition_disjoint_and_covers(tmp_path=None):
    # build a tiny manifest file and partition it
    import json, tempfile, os
    samples = _manifest(n_per=(("culane", 20), ("tusimple", 20)))
    art = cs.select_calibration_ids(samples, fraction=0.25, min_per_stratum=1)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metadata": {}, "samples": samples}, f)
        calib, ev = cs.partition(path, art)
        all_ids = {s["sample_id"] for s in samples}
        assert set(calib).isdisjoint(set(ev))
        assert set(calib) | set(ev) == all_ids
        assert set(calib) == set(art["calibration_sample_ids"])
    finally:
        os.remove(path)
