from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMAGENET_MEAN = np.array([0.3598, 0.3653, 0.3662], dtype=np.float32)
IMAGENET_STD = np.array([0.2573, 0.2663, 0.2756], dtype=np.float32)


def _safe(name) -> str:
    return str(name).replace("/", "_").replace("\\", "_")


def _make_overlay(image_path: str, gt_mask, pred_mask: np.ndarray, out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return

    overlay = img.copy()

    if gt_mask is not None:
        gt = np.asarray(gt_mask) > 0
        overlay[gt] = (0.55 * overlay[gt] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)

    pred = np.asarray(pred_mask) > 0
    overlay[pred] = (0.55 * overlay[pred] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _preprocess(image_path: str, width: int, height: int, device: torch.device) -> torch.Tensor:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).float().to(device)


def _load_state(model, weight_path: str, device: torch.device) -> None:
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint.get("net", checkpoint)

    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(stripped)


def _predict_mask(model, image_path: str, orig_w: int, orig_h: int, args, device: torch.device) -> np.ndarray:
    x = _preprocess(image_path, args.input_width, args.input_height, device)

    with torch.no_grad():
        seg_pred, exist_pred = model(x)[:2]

    label = torch.argmax(seg_pred, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    exist = exist_pred.squeeze(0).detach().cpu().numpy()

    for lane_idx in range(4):
        if exist[lane_idx] <= args.exist_threshold:
            label[label == (lane_idx + 1)] = 0

    pred_mask_small = (label > 0).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred_mask


def parse_args():
    p = argparse.ArgumentParser(description="Run SCNN on a universal lane_eval manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--pred-manifest", required=True)
    p.add_argument("--scnn-root", default="/shared/src/SCNN_Pytorch")
    p.add_argument(
        "--weights",
        default="/shared/src/SCNN_Pytorch/experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.pth",
    )
    p.add_argument("--model-name", default="scnn")
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--input-width", type=int, default=800)
    p.add_argument("--input-height", type=int, default=288)
    p.add_argument("--exist-threshold", type=float, default=0.5)
    p.add_argument("--overlay-dir", default=None)
    p.add_argument("--overlay-sample", type=int, default=0)
    p.add_argument("--overlay-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    scnn_root = Path(args.scnn_root).expanduser().resolve()
    sys.path.insert(0, str(scnn_root))

    from model import SCNN
    from lane_eval.manifest import ManifestDataset, PredictionManifestWriter

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    dataset = ManifestDataset(args.manifest)
    dataset_name = args.dataset or getattr(dataset, "name", "manifest")
    total = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

    out_path = Path(args.pred_manifest).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SCNN from {args.weights}")
    print(f"Using device: {device}")
    model = SCNN(input_size=(args.input_width, args.input_height), pretrained=False).to(device)
    _load_state(model, args.weights, device)
    model.eval()

    writer = PredictionManifestWriter(
        str(out_path),
        args.model_name,
        dataset_name,
        save_masks=True,
    )

    overlay_dir = Path(args.overlay_dir).expanduser().resolve() if args.overlay_dir else None
    overlay_indices = set()
    if overlay_dir and args.overlay_sample > 0:
        rng = random.Random(args.overlay_seed)
        overlay_indices = set(rng.sample(range(total), min(args.overlay_sample, total)))

    skipped = 0

    for idx in tqdm(range(total), desc=f"Running SCNN on {dataset_name}"):
        sample = dataset[idx]

        try:
            pred_mask = _predict_mask(
                model,
                sample.image_path,
                sample.width,
                sample.height,
                args,
                device,
            )
        except Exception as exc:
            skipped += 1
            print(f"SKIP {sample.image_id}: {exc}")
            continue

        writer.add(sample.image_id, sample.image_path, pred_mask)

        if idx in overlay_indices and overlay_dir is not None:
            overlay_path = overlay_dir / dataset_name / f"{_safe(sample.image_id)}_overlay.jpg"
            _make_overlay(sample.image_path, sample.target.mask, pred_mask, overlay_path)

    written = writer.write()
    print(f"Wrote prediction manifest: {written}")
    print(f"Processed samples: {total - skipped}")
    print(f"Skipped samples: {skipped}")


if __name__ == "__main__":
    main()
