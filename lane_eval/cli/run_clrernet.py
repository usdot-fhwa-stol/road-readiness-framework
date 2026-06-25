#!/usr/bin/env python3
"""Run CLRerNet (DLA-34 backbone) on a universal lane_eval manifest.

Stage 1 of the CLRerNet flow. CLRerNet is an mmdetection-based lane-LINE
detector, so it lives in its own venv (mmdet / mmengine / the repo's
``libs.api.inference``). This runner is therefore *inference only*: for every
sample in the input manifest it

  * runs CLRerNet,
  * maps the predicted lane points from the model's fixed 1640x590 letterbox
    space back to the original image,
  * rasterises them into a binary lane mask, and
  * records that mask in a universal *prediction manifest* (via the shared
    ``PredictionManifestWriter``) — masks go in a ``masks/`` dir beside it.

No metric math happens here. Scoring is stage 2
(``lane_eval.cli.eval_segmentation_manifest``), which runs in the normal repo
env and scores these masks against the GT manifest with the SAME YOLOPX
``SegmentationMetric`` used for every other model — so the comparison is fair.

Run with the CLRerNet venv interpreter, e.g. via run_clrernet.sh:
    /shared/src/CLRerNet/clrernet/bin/python -m lane_eval.cli.run_clrernet ...
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _make_overlay(image_path: str, gt_mask, pred_mask: np.ndarray, out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    overlay = img.copy()
    if gt_mask is not None:
        gt = (np.asarray(gt_mask) > 0)
        overlay[gt] = (0.55 * overlay[gt] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    pred = (np.asarray(pred_mask) > 0)
    overlay[pred] = (0.55 * overlay[pred] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _rasterize_lanes(lanes, height: int, width: int, thickness: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for lane in lanes:
        pts = []
        for point in lane:
            x = int(round(point["x"]))
            y = int(round(point["y"]))
            if 0 <= x < width and 0 <= y < height:
                pts.append([x, y])
        if len(pts) >= 2:
            cv2.polylines(mask, [np.asarray(pts, dtype=np.int32)],
                          isClosed=False, color=1, thickness=thickness)
    return mask


def _letterbox_params(orig_w: int, orig_h: int, target_w: int = 1640, target_h: int = 590):
    """CLRerNet is trained/evaluated on the CULane 1640x590 letterbox. Recover the
    scale + padding used so we can map predicted points back to the original image."""
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    return scale, pad_left, pad_top


def _clrernet_points_to_original(preds, orig_w: int, orig_h: int):
    scale, pad_left, pad_top = _letterbox_params(orig_w, orig_h)
    lanes = []
    for lane in preds:
        converted = []
        for x, y in lane:
            ox = (float(x) - pad_left) / scale
            oy = (float(y) - pad_top) / scale
            if 0 <= ox < orig_w and 0 <= oy < orig_h:
                converted.append({"x": ox, "y": oy})
        if len(converted) >= 2:
            lanes.append(converted)
    return lanes


def _lanes_to_lane_json(lanes, height: int, step: int = 10) -> dict:
    """Resample CLRerNet's predicted polylines onto fixed h_samples rows so the
    prediction manifest carries the model's OWN exact lane geometry (TuSimple
    ``{h_samples, lanes}`` format). This makes the native CULane/TuSimple F1
    metrics faithful to CLRerNet and independent of the mask rasterisation
    thickness. Rows outside a lane's y-range are marked missing (-2)."""
    from lane_eval.converters import make_h_samples, NO_POINT

    h_samples = make_h_samples(height, step)
    out_lanes = []
    for lane in lanes:
        ys = np.asarray([p["y"] for p in lane], dtype=np.float32)
        xs = np.asarray([p["x"] for p in lane], dtype=np.float32)
        order = np.argsort(ys)
        ys, xs = ys[order], xs[order]
        y_min, y_max = float(ys[0]), float(ys[-1])
        row = [int(round(float(np.interp(h, ys, xs)))) if y_min <= h <= y_max else NO_POINT
               for h in h_samples]
        if sum(1 for v in row if v != NO_POINT) >= 2:
            out_lanes.append(row)
    return {"h_samples": list(h_samples), "lanes": out_lanes}


def parse_args():
    p = argparse.ArgumentParser(description="Run CLRerNet on a universal lane_eval manifest (inference only).")
    p.add_argument("--manifest", required=True, help="Universal input manifest JSON")
    p.add_argument("--pred-manifest", required=True, help="Output prediction manifest JSON")
    p.add_argument("--clrernet-root", default="/shared/src/CLRerNet")
    p.add_argument("--config", default="/shared/src/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    p.add_argument("--checkpoint", default="/shared/src/CLRerNet/clrernet_culane_dla34_ema.pth")
    p.add_argument("--model-name", default="clrernet")
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--mask-thickness", type=int, default=4,
                   help="Stroke width (px) used to rasterise predicted lane lines into a mask")
    p.add_argument("--overlay-dir", default=None)
    p.add_argument("--overlay-sample", type=int, default=0)
    p.add_argument("--overlay-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    clrernet_root = Path(args.clrernet_root).resolve()
    sys.path.insert(0, str(clrernet_root))

    from mmengine.config import Config
    from mmdet.apis import init_detector

    from libs.api.inference import inference_one_image
    from lane_eval.manifest import ManifestDataset, PredictionManifestWriter

    dataset = ManifestDataset(args.manifest)
    dataset_name = args.dataset or getattr(dataset, "name", "manifest")
    total = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

    out_path = Path(args.pred_manifest).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CLRerNet's config tries to build a CULane test dataset during init. Give it a
    # tiny local data list so init_detector can finish without relying on the
    # shared repo-side dataset/culane/list/test.txt state.
    first_sample = dataset[0]
    dummy_list = out_path.parent / "clrernet_dummy_test.txt"
    dummy_list.write_text(str(first_sample.image_path) + "\n")

    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.dataset.data_list = str(dummy_list)

    overlay_dir = Path(args.overlay_dir).expanduser().resolve() if args.overlay_dir else None

    old_cwd = os.getcwd()
    os.chdir(str(clrernet_root))
    try:
        print("Loading CLRerNet model...")
        model = init_detector(cfg, args.checkpoint, device=args.device)

        writer = PredictionManifestWriter(str(out_path), args.model_name, dataset_name, save_masks=True)

        overlay_indices = set()
        if overlay_dir and args.overlay_sample > 0:
            rng = random.Random(args.overlay_seed)
            overlay_indices = set(rng.sample(range(total), min(args.overlay_sample, total)))

        for idx in tqdm(range(total), desc=f"Running CLRerNet on {dataset_name}"):
            sample = dataset[idx]
            _, preds = inference_one_image(model, sample.image_path)
            lanes_original = _clrernet_points_to_original(preds, sample.width, sample.height)
            pred_mask = _rasterize_lanes(lanes_original, sample.height, sample.width, args.mask_thickness)
            # Carry CLRerNet's exact lane geometry so the native CULane/TuSimple
            # F1 (the paper's metric) is faithful, not re-derived from the mask.
            lane_json = _lanes_to_lane_json(lanes_original, sample.height)
            writer.add(sample.image_id, sample.image_path, pred_mask, lane_json=lane_json)

            if idx in overlay_indices and overlay_dir is not None:
                safe_id = str(sample.image_id).replace("/", "_").replace("\\", "_")
                overlay_path = overlay_dir / dataset_name / f"{safe_id}_overlay.jpg"
                _make_overlay(sample.image_path, sample.target.mask, pred_mask, overlay_path)

        written = writer.write()
        print(f"Wrote prediction manifest: {written}")
        print(f"Processed samples: {total}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
