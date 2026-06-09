"""TuSimple adapter tests using a fake JSON-lines annotation file."""
import json
import cv2
import numpy as np
import pytest


def _make_fake_tusimple(tmp_path):
    clips_dir = tmp_path / "clips" / "0530" / "1492626047222176976_0"
    clips_dir.mkdir(parents=True)

    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imwrite(str(clips_dir / "20.jpg"), img)

    record = {
        "raw_file": "clips/0530/1492626047222176976_0/20.jpg",
        "lanes": [[  # two lanes
            -2, -2, -2, -2, 632, 625, 617, 608, 596, 584,
            569, 548, 523, 501, 475, 452, 428, 406, 381, 356
        ], [
            -2, -2, -2, -2, -2, -2, -2, 715, 693, 671,
            647, 622, 594, 567, 540, 515, 487, 460, 434, 408
        ]],
        "h_samples": [
            160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
            260, 270, 280, 290, 300, 310, 320, 330, 340, 350
        ],
    }

    annot_file = tmp_path / "val_gt.json"
    with open(annot_file, "w") as f:
        f.write(json.dumps(record) + "\n")

    return tmp_path, annot_file


def test_adapter_length(tmp_path):
    from lane_eval.datasets.tusimple import TuSimpleAdapter
    _, annot = _make_fake_tusimple(tmp_path)
    adapter = TuSimpleAdapter(root=str(tmp_path), annotation_files=[str(annot)])
    assert len(adapter) == 1


def test_sample_fields(tmp_path):
    from lane_eval.datasets.tusimple import TuSimpleAdapter
    _, annot = _make_fake_tusimple(tmp_path)
    adapter = TuSimpleAdapter(root=str(tmp_path), annotation_files=[str(annot)])
    s = adapter[0]
    assert s.width == 1280
    assert s.height == 720
    assert len(s.target.lanes) == 2
    assert s.target.mask is not None
    assert s.target.mask.sum() > 0
    assert s.target.mask.dtype == np.uint8


def test_negative_x_filtered_out(tmp_path):
    from lane_eval.datasets.tusimple import TuSimpleAdapter
    _, annot = _make_fake_tusimple(tmp_path)
    adapter = TuSimpleAdapter(root=str(tmp_path), annotation_files=[str(annot)])
    s = adapter[0]
    for lane in s.target.lanes:
        assert (lane[:, 0] >= 0).all(), "negative x coords should be filtered"
