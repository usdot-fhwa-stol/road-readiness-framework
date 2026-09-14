"""CULane adapter tests — works even when images are not extracted from MP4s."""
import numpy as np
import pytest


def _make_fake_culane(tmp_path):
    driver_dir = tmp_path / "driver_23_30frame" / "test.MP4"
    driver_dir.mkdir(parents=True)

    list_dir = tmp_path / "list"
    list_dir.mkdir()

    # annotation file (no actual image needed)
    annot = driver_dir / "00020.lines.txt"
    annot.write_text("100 200 300 400\n500 100 600 200\n")

    # list file references the jpg (which may not exist)
    val_txt = list_dir / "val.txt"
    val_txt.write_text("/driver_23_30frame/test.MP4/00020.jpg\n")

    return tmp_path


def test_adapter_length(tmp_path):
    from lane_eval.datasets.culane import CULaneAdapter
    _make_fake_culane(tmp_path)
    adapter = CULaneAdapter(root=str(tmp_path), split="val")
    assert len(adapter) == 1


def test_sample_has_lanes_even_without_image(tmp_path):
    from lane_eval.datasets.culane import CULaneAdapter
    _make_fake_culane(tmp_path)
    adapter = CULaneAdapter(root=str(tmp_path), split="val")
    s = adapter[0]
    assert s.target.lanes is not None and len(s.target.lanes) == 2
    assert s.target.mask is not None
    assert s.target.mask.sum() > 0
    # fallback dimensions
    assert s.height == 590
    assert s.width == 1640


def test_missing_list_file_raises(tmp_path):
    from lane_eval.datasets.culane import CULaneAdapter
    with pytest.raises(FileNotFoundError):
        CULaneAdapter(root=str(tmp_path), split="val")
