"""Unit tests for evaluation.marking_support (D3' centerline localization).

These exercise the correctness-critical, pure-NumPy math:
  * coordinate/translation invariance within delta,
  * resolution invariance of canonical coordinates,
  * empty-case None semantics (never coerced to 0),
  * perfect-prediction gives F1=1 and zero localization error,
  * delta is an explicit parameter (not hardcoded), and changing it changes the
    score in the expected direction,
  * polyline densification is uniform and resolution-aware.

They import only numpy + the module under test, so they run in the minimal repo
env (no cv2/scipy/scikit-image). The mask->centerline extraction
(``mask_to_points``) depends on scikit-image/OpenCV and is validated on the
inference box; it is intentionally not unit-tested here.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation import marking_support as ms


# --------------------------------------------------------------------------- #
# canonicalization
# --------------------------------------------------------------------------- #
def test_diagonal():
    assert ms.diagonal(3, 4) == pytest.approx(5.0)


def test_to_canonical_scales_by_diagonal():
    pts = np.array([[3.0, 4.0]])
    can = ms.to_canonical(pts, 3, 4)
    assert can == pytest.approx(np.array([[3 / 5, 4 / 5]]))


def test_to_canonical_empty():
    can = ms.to_canonical(np.zeros((0, 2)), 100, 200)
    assert can.shape == (0, 2)


def test_to_canonical_resolution_invariance():
    """A pixel at the same fractional location maps to the same canonical pt."""
    base = np.array([[100.0, 200.0]])
    scaled = base * 2.0
    a = ms.to_canonical(base, 400, 300)
    b = ms.to_canonical(scaled, 800, 600)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# nearest-neighbour distances
# --------------------------------------------------------------------------- #
def test_min_distances_basic():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    b = np.array([[0.0, 3.0], [10.0, 4.0]])
    d = ms._min_distances(a, b)
    assert d == pytest.approx(np.array([3.0, 4.0]))


def test_min_distances_empty_b_is_inf():
    a = np.array([[0.0, 0.0]])
    d = ms._min_distances(a, np.zeros((0, 2)))
    assert np.isinf(d).all()


def test_min_distances_empty_a():
    d = ms._min_distances(np.zeros((0, 2)), np.array([[1.0, 1.0]]))
    assert d.shape == (0,)


def test_min_distances_chunking_matches_bruteforce():
    rng = np.random.RandomState(0)
    a = rng.rand(500, 2)
    b = rng.rand(300, 2)
    got = ms._min_distances(a, b)
    # brute force reference
    ref = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert got == pytest.approx(ref)


# --------------------------------------------------------------------------- #
# localization_scores
# --------------------------------------------------------------------------- #
def _line_points(n=50, x=0.5):
    ys = np.linspace(0.0, 1.0, n)
    return np.stack([np.full(n, x), ys], axis=1)


def test_perfect_prediction():
    g = _line_points()
    r = ms.localization_scores(g.copy(), g.copy(), delta=1e-6)
    assert r["support_precision"] == pytest.approx(1.0)
    assert r["support_recall"] == pytest.approx(1.0)
    assert r["support_f1"] == pytest.approx(1.0)
    assert r["loc_err_median"] == pytest.approx(0.0)
    assert r["loc_err_p95"] == pytest.approx(0.0)


def test_translation_within_delta_is_perfect_recall():
    g = _line_points()
    p = g + np.array([0.01, 0.0])  # shift by 0.01 of diagonal
    r = ms.localization_scores(p, g, delta=0.02)
    assert r["support_recall"] == pytest.approx(1.0)
    assert r["support_precision"] == pytest.approx(1.0)
    assert r["loc_err_median"] == pytest.approx(0.01, abs=1e-9)


def test_translation_beyond_delta_drops_recall():
    g = _line_points()
    p = g + np.array([0.05, 0.0])
    r = ms.localization_scores(p, g, delta=0.02)
    assert r["support_recall"] == pytest.approx(0.0)
    assert r["support_precision"] == pytest.approx(0.0)
    # error still reported honestly as the true distance
    assert r["loc_err_median"] == pytest.approx(0.05, abs=1e-9)


def test_delta_is_a_parameter_not_hardcoded():
    """Same geometry, larger delta -> recall goes up. Proves delta drives it."""
    g = _line_points()
    p = g + np.array([0.03, 0.0])
    tight = ms.localization_scores(p, g, delta=0.01)
    loose = ms.localization_scores(p, g, delta=0.10)
    assert tight["support_recall"] == pytest.approx(0.0)
    assert loose["support_recall"] == pytest.approx(1.0)


def test_empty_gt_returns_none():
    p = _line_points()
    r = ms.localization_scores(p, np.zeros((0, 2)), delta=0.02)
    assert r["support_recall"] is None
    assert r["support_f1"] is None
    assert r["loc_err_median"] is None
    assert r["loc_err_p95"] is None
    # GT empty -> every prediction is a false positive (nearest GT is at +inf),
    # so precision is a defined 0.0, not undefined and not 1.0.
    assert r["support_precision"] == pytest.approx(0.0)


def test_empty_pred_vs_nonempty_gt():
    g = _line_points()
    r = ms.localization_scores(np.zeros((0, 2)), g, delta=0.02)
    assert r["support_precision"] is None  # no predictions
    assert r["support_recall"] == pytest.approx(0.0)  # gt has no support within delta
    assert r["support_f1"] is None  # precision undefined -> f1 undefined
    # localization error is unbounded (pred empty) -> reported None, not fabricated
    assert r["loc_err_median"] is None
    assert r["loc_err_p95"] is None


def test_both_empty():
    r = ms.localization_scores(np.zeros((0, 2)), np.zeros((0, 2)), delta=0.02)
    assert r["support_precision"] is None
    assert r["support_recall"] is None
    assert r["support_f1"] is None


def test_partial_recall():
    g = _line_points(n=100)
    # shift only the top half far away
    p = g.copy()
    p[:50] += np.array([0.5, 0.0])
    r = ms.localization_scores(p, g, delta=0.02)
    assert r["support_recall"] == pytest.approx(0.5, abs=0.02)


def test_negative_or_nan_delta_raises():
    g = _line_points()
    with pytest.raises(ValueError):
        ms.localization_scores(g, g, delta=-0.1)
    with pytest.raises(ValueError):
        ms.localization_scores(g, g, delta=np.nan)


# --------------------------------------------------------------------------- #
# band_localization
# --------------------------------------------------------------------------- #
def test_band_localization_splits_by_y():
    W, H = 100, 1000
    # gt spans full height; pred only present in the near band [0.80, 0.95).
    ys = np.arange(0, 1000, 5.0)
    g_xy = np.stack([np.full(len(ys), 50.0), ys], axis=1)
    near = g_xy[(g_xy[:, 1] >= 800) & (g_xy[:, 1] < 950)]
    r = ms.band_localization(near, g_xy, W, H, delta=0.02)
    assert set(r.keys()) == {"upper", "middle", "lower"}
    # The near/lower band [0.80, 0.95) is fully covered by prediction; recall = 1.
    assert r["lower"]["support_recall"] == pytest.approx(1.0)
    # No prediction in the upper/middle bands.
    assert r["upper"]["support_recall"] == pytest.approx(0.0)
    assert r["middle"]["support_recall"] == pytest.approx(0.0)


def test_band_edges_are_40_40_15_with_hood_excluded():
    # The documented layout: upper 40% / middle 40% / lower(near) 15% / hood 5%.
    assert ms.BAND_EDGES == ((0.00, 0.40), (0.40, 0.80), (0.80, 0.95))
    assert ms.BAND_LABELS == ("upper", "middle", "lower")
    assert ms.HOOD_IGNORE_FRACTION == pytest.approx(0.05)
    assert ms.NEAR_BAND_FRACTION == pytest.approx(0.15)


def test_hood_region_is_excluded_from_all_bands():
    # Points in the bottom 5% (assumed ego-vehicle hood) must contribute to NO
    # band: with GT only in the hood, every band's recall is undefined (None),
    # never counted as detected or missed.
    W, H = 100, 1000
    hood_ys = np.arange(960, 1000, 2.0)  # y/H in [0.96, 1.0) -> hood
    g_hood = np.stack([np.full(len(hood_ys), 50.0), hood_ys], axis=1)
    r = ms.band_localization(g_hood, g_hood, W, H, delta=0.02)
    for label in ("upper", "middle", "lower"):
        # No GT points land in any band, so recall is None (empty GT), not 0/1.
        assert r[label]["support_recall"] is None


# --------------------------------------------------------------------------- #
# polyline densification
# --------------------------------------------------------------------------- #
def test_polyline_to_points_uniform_spacing():
    line = np.array([[0.0, 0.0], [0.0, 10.0]])
    pts = ms.polyline_to_points(line, step_px=1.0)
    assert pts.shape[0] >= 11
    dif = np.diff(pts[:, 1])
    assert np.allclose(dif, dif[0])  # uniform


def test_polyline_to_points_single_vertex():
    pts = ms.polyline_to_points(np.array([[5.0, 5.0]]))
    assert pts.shape == (1, 2)


def test_polyline_to_points_empty():
    pts = ms.polyline_to_points(np.zeros((0, 2)))
    assert pts.shape == (0, 2)


def test_polylines_to_points_dicts():
    lanes = [[{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 4.0}]]
    pts = ms.polylines_to_points(lanes, step_px=1.0)
    assert pts.shape[0] >= 5
    assert pts[:, 0] == pytest.approx(np.zeros(pts.shape[0]))


def test_polylines_to_points_skips_garbage():
    lanes = [[{"x": np.nan, "y": 1.0}, {"x": 2.0, "y": 3.0}, {"x": 2.0, "y": 9.0}]]
    pts = ms.polylines_to_points(lanes)
    assert np.isfinite(pts).all()


def test_polyline_densification_resolution_invariant_in_canonical():
    """A line densified at 1px in a 2x image, then canonicalized, matches the
    1x version canonicalized — placement is resolution-free."""
    line1 = np.array([[10.0, 10.0], [10.0, 50.0]])
    line2 = line1 * 2.0
    p1 = ms.to_canonical(ms.polyline_to_points(line1, 1.0), 100, 100)
    p2 = ms.to_canonical(ms.polyline_to_points(line2, 2.0), 200, 200)
    # same count (step scaled with resolution) and same canonical coords
    assert p1.shape == p2.shape
    assert p1 == pytest.approx(p2)
