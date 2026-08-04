from __future__ import annotations

import argparse
import json
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
            cv2.polylines(
                mask,
                [np.asarray(pts, dtype=np.int32)],
                isClosed=False,
                color=1,
                thickness=thickness,
            )

    return mask


def _letterbox_params(orig_w: int, orig_h: int, target_w: int = 1640, target_h: int = 590):
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


def parse_args():
    p = argparse.ArgumentParser(description="Run CLRerNet on a universal lane_eval manifest.")

    p.add_argument("--manifest", required=True, help="Universal input manifest JSON")
    p.add_argument("--pred-manifest", required=True, help="Output prediction manifest JSON")
    p.add_argument("--clrernet-root", default="/shared/src/CLRerNet")
    p.add_argument("--config", default="/shared/src/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    p.add_argument("--checkpoint", default="/shared/src/CLRerNet/clrernet_culane_dla34_ema.pth")
    p.add_argument("--model-name", default="clrernet")
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--mask-thickness", type=int, default=4)
    p.add_argument("--overlay-dir", default=None)
    p.add_argument("--overlay-sample", type=int, default=0)
    p.add_argument("--overlay-seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    clrernet_root = Path(args.clrernet_root).resolve()
    sys.path.insert(0, str(clrernet_root))

    from mmdet.apis import init_detector
    from mmengine.config import Config

    from libs.api.inference import inference_one_image
    from lane_eval.manifest import ManifestDataset, PredictionManifestWriter

    dataset = ManifestDataset(args.manifest)
    dataset_name = args.dataset or getattr(dataset, "name", "manifest")
    total = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

    out_path = Path(args.pred_manifest).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CLRerNet's config tries to build a CULane test dataset during init.
    # Give it a tiny local data list so init_detector can finish without relying
    # on shared repo-side dataset/culane/list/test.txt state.
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

        writer = PredictionManifestWriter(
            str(out_path),
            args.model_name,
            dataset_name,
            save_masks=True,
        )

        overlay_indices = set()
        if overlay_dir and args.overlay_sample > 0:
            rng = random.Random(args.overlay_seed)
            overlay_indices = set(rng.sample(range(total), min(args.overlay_sample, total)))

        for idx in tqdm(range(total), desc=f"Running CLRerNet on {dataset_name}"):
            sample = dataset[idx]

            src, preds = inference_one_image(model, sample.image_path)
            lanes_original = _clrernet_points_to_original(
                preds,
                sample.width,
                sample.height,
            )
            pred_mask = _rasterize_lanes(
                lanes_original,
                sample.height,
                sample.width,
                args.mask_thickness,
            )

            writer.add(
                sample.image_id,
                sample.image_path,
                pred_mask,
                polylines=lanes_original,
                prediction_meta={"coordinate_space": "original_image"},
            )

            if idx in overlay_indices and overlay_dir is not None:
                safe_id = str(sample.image_id).replace('/', '_').replace('\\\\', '_')
                overlay_path = overlay_dir / dataset_name / f'{safe_id}_overlay.jpg'
                _make_overlay(sample.image_path, sample.target.mask, pred_mask, overlay_path)

        written = writer.write()
        print(f"Wrote prediction manifest: {written}")
        print(f"Processed samples: {total}")

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
