"""CurveLanes adapter tests using fake data."""
import json
import cv2
import numpy as np
import pytest


def _make_fake_curvelanes(tmp_path, split="valid"):
    img_dir   = tmp_path / split / "images"
    label_dir = tmp_path / split / "labels"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    stem = "abcdef1234567890"
    img = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)

    label = {"Lines": [
        [{"x": "100.0", "y": "200.0"}, {"x": "200.0", "y": "400.0"}],
        [{"x": "500.0", "y": "200.0"}, {"x": "600.0", "y": "400.0"}],
    ]}
    with open(label_dir / f"{stem}.lines.json", "w") as f:
        json.dump(label, f)

    return tmp_path


def test_adapter_length(tmp_path):
    from lane_eval.datasets.curvelanes import CurveLanesAdapter
    _make_fake_curvelanes(tmp_path)
    adapter = CurveLanesAdapter(root=str(tmp_path), split="valid")
    assert len(adapter) == 1


def test_sample_has_lanes_and_mask(tmp_path):
    from lane_eval.datasets.curvelanes import CurveLanesAdapter
    _make_fake_curvelanes(tmp_path)
    adapter = CurveLanesAdapter(root=str(tmp_path), split="valid")
    s = adapter[0]
    assert s.target.lanes is not None and len(s.target.lanes) == 2
    assert s.target.mask is not None
    assert s.target.mask.sum() > 0
    assert s.target.mask.dtype == np.uint8


def test_missing_label_returns_empty_mask(tmp_path):
    from lane_eval.datasets.curvelanes import CurveLanesAdapter
    img_dir = tmp_path / "valid" / "images"
    img_dir.mkdir(parents=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "nolabel.jpg"), img)
    adapter = CurveLanesAdapter(root=str(tmp_path), split="valid")
    s = adapter[0]
    assert s.target.mask.sum() == 0
