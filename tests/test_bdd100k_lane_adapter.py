"""BDD100K adapter tests using fake data in tmp_path."""
import cv2
import numpy as np
import pytest


def _make_fake_bdd100k(tmp_path):
    img_dir  = tmp_path / "images" / "val"
    mask_dir = tmp_path / "ll_seg_annotations" / "val"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for stem in ["frame_0001", "frame_0002"]:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[300:310, :] = 255
        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

    return img_dir, mask_dir


def test_adapter_length(tmp_path):
    from lane_eval.datasets.bdd100k import BDD100KLaneAdapter
    img_dir, mask_dir = _make_fake_bdd100k(tmp_path)
    adapter = BDD100KLaneAdapter(image_root=str(img_dir), lane_mask_root=str(mask_dir))
    assert len(adapter) == 2


def test_sample_fields(tmp_path):
    from lane_eval.datasets.bdd100k import BDD100KLaneAdapter
    img_dir, mask_dir = _make_fake_bdd100k(tmp_path)
    adapter = BDD100KLaneAdapter(image_root=str(img_dir), lane_mask_root=str(mask_dir))
    s = adapter[0]
    assert s.width == 1280
    assert s.height == 720
    assert s.target.mask is not None
    assert s.target.mask.dtype == np.uint8
    assert set(np.unique(s.target.mask)) <= {0, 1}
    assert s.target.mask.sum() > 0


def test_no_pairs_raises(tmp_path):
    from lane_eval.datasets.bdd100k import BDD100KLaneAdapter
    with pytest.raises(FileNotFoundError):
        BDD100KLaneAdapter(image_root=str(tmp_path / "x"), lane_mask_root=str(tmp_path / "y"))
