import numpy as np
import pytest

from lane_eval.converters.lanes_to_mask import lanes_to_mask, ensure_binary_mask


def test_empty_lanes_returns_zero_mask():
    mask = lanes_to_mask([], height=100, width=200)
    assert mask.shape == (100, 200)
    assert mask.sum() == 0
    assert mask.dtype == np.uint8


def test_single_lane_produces_nonzero_pixels():
    lane = np.array([[10, 10], [190, 90]], dtype=np.float32)
    mask = lanes_to_mask([lane], height=100, width=200, thickness=4)
    assert mask.sum() > 0
    assert mask.dtype == np.uint8


def test_output_values_are_only_0_and_1():
    lane = np.array([[0, 0], [50, 50], [100, 99]], dtype=np.float32)
    mask = lanes_to_mask([lane], height=100, width=100, thickness=8)
    unique = set(mask.flatten().tolist())
    assert unique <= {0, 1}


def test_out_of_bounds_does_not_crash():
    # Points well outside image bounds
    lane = np.array([[-500, -500], [5000, 5000]], dtype=np.float32)
    mask = lanes_to_mask([lane], height=100, width=100, thickness=4)
    assert mask.shape == (100, 100)


def test_single_point_lane_is_skipped():
    lane = np.array([[50, 50]], dtype=np.float32)
    mask = lanes_to_mask([lane], height=100, width=100)
    assert mask.sum() == 0


def test_ensure_binary_mask():
    arr = np.array([[0, 128, 255], [1, 0, 127]], dtype=np.uint8)
    binary = ensure_binary_mask(arr)
    assert binary.dtype == np.uint8
    assert set(binary.flatten().tolist()) <= {0, 1}
    assert binary[0, 0] == 0
    assert binary[0, 1] == 1
    assert binary[0, 2] == 1
