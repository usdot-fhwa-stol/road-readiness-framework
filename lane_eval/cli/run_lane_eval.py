#!/usr/bin/env python3
"""Evaluate YOLOPX lane output on any supported dataset.

Uses YOLOPX's own model loading, preprocessing (letterbox_for_img), and
SegmentationMetric.  No new metric math is introduced here.

Example — BDD100K (all images available):
    python -m lane_eval.cli.run_lane_eval \
        --yolopx-repo /home/gauravb/Projects/road_readiness_t3/YOLOPX \
        --weights /path/to/epoch-195.pth \
        --dataset bdd100k_lane \
        --image-root /shared/data/bdd100k/images/val \
        --lane-mask-root /shared/data/bdd100k/ll_seg_annotations/val \
        --output outputs/results/yolopx_bdd100k_lane.json

Example — CurveLanes:
    python -m lane_eval.cli.run_lane_eval \
        --yolopx-repo /home/gauravb/Projects/road_readiness_t3/YOLOPX \
        --weights /path/to/epoch-195.pth \
        --dataset curvelanes \
        --root /shared/data/Curvelanes \
        --split valid \
        --output outputs/results/yolopx_curvelanes_lane.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ImageNet normalisation constants — same values used in YOLOPX demo.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor_and_normalise(img_rgb_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB numpy array → CHW float32 normalised tensor.

    Replaces torchvision.transforms.ToTensor + Normalize to avoid the
    numpy-2.x / torchvision-0.13 ABI mismatch (torch.from_numpy breaks there).
    """
    arr = np.ascontiguousarray(img_rgb_hwc, dtype=np.float32) / 255.0
    t = torch.tensor(arr).permute(2, 0, 1)   # HWC -> CHW, stays float32
    return (t - _MEAN) / _STD


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_yolopx(repo: str) -> None:
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _load_model(yolopx_repo: str, weights: str, requested_device: torch.device):
    """Load YOLOPX model.  Returns (model, actual_device, use_half).

    If the requested CUDA device is present but PyTorch has no compiled kernels
    for it (e.g. Blackwell sm_120 with PyTorch < 2.4), we catch the RuntimeError
    and fall back to CPU float32 automatically.
    """
    _add_yolopx(yolopx_repo)
    from lib.models import get_net  # noqa: E402

    # Load weights to CPU first so we can test device placement safely
    model = get_net(cfg=None)
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = requested_device
    use_half = False

    if requested_device.type != "cpu":
        try:
            model.to(requested_device)
            model.half()
            # Smoke-test: tiny forward pass to catch "no kernel image" errors
            dummy = torch.zeros(1, 3, 384, 640, device=requested_device, dtype=torch.float16)
            with torch.no_grad():
                model(dummy)
            use_half = True
            print(f"Using device: {requested_device} (float16)")
        except RuntimeError as exc:
            print(
                f"WARNING: {requested_device} not usable with this PyTorch build "
                f"({exc.__class__.__name__}: {str(exc)[:120]}).\n"
                "Falling back to CPU (float32). "
                "To use the GPU, upgrade PyTorch to >= 2.4 with CUDA 12.x support."
            )
            model.to("cpu").float()
            device = torch.device("cpu")
            use_half = False
    else:
        print("Using device: cpu (float32)")

    return model, device, use_half


def _preprocess(img_bgr: np.ndarray, img_size: int, yolopx_repo: str):
    """Letterbox + ToTensor + Normalize.  Returns (tensor, shapes)."""
    _add_yolopx(yolopx_repo)
    from lib.utils import letterbox_for_img  # noqa: E402

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    img_lb, ratio, pad = letterbox_for_img(img_rgb, img_size, auto=True)
    h, w = img_lb.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img_t = _to_tensor_and_normalise(img_lb)
    return img_t, shapes


def _extract_lane_mask(
    ll_seg_out: torch.Tensor,
    shapes,
    device: torch.device,
) -> np.ndarray:
    """Strip letterbox padding, interpolate to original size, argmax → binary mask."""
    _, _, height, width = ll_seg_out.shape
    (h0, w0), (_, pad) = shapes
    pad_w, pad_h = int(pad[0]), int(pad[1])

    ll_cropped = ll_seg_out[:, :, pad_h: height - pad_h, pad_w: width - pad_w]
    ll_up = F.interpolate(ll_cropped.float(), size=(h0, w0), mode="bilinear", align_corners=False)
    _, ll_mask = torch.max(ll_up, dim=1)
    return ll_mask.int().squeeze().cpu().numpy().astype(np.uint8)


# ── dataset builder from flat CLI args ──────────────────────────────────────

def _build_adapter(args):
    # Manifest mode: read the universal manifest instead of a native dataset.
    if args.manifest:
        from lane_eval.manifest import ManifestDataset  # noqa: E402
        return ManifestDataset(args.manifest)

    from lane_eval.datasets import build_dataset_adapter  # noqa: E402

    name = args.dataset
    if name == "bdd100k_lane":
        return build_dataset_adapter(
            name,
            image_root=args.image_root,
            lane_mask_root=args.lane_mask_root,
            split=args.split,
        )
    elif name in ("curvelanes", "culane", "tusimple"):
        kwargs = dict(root=args.root, split=args.split)
        if args.mask_thickness:
            kwargs["mask_thickness"] = args.mask_thickness
        if name == "tusimple" and args.annotation_file:
            kwargs["annotation_files"] = [args.annotation_file]
        return build_dataset_adapter(name, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ── DataLoader plumbing ──────────────────────────────────────────────────────

class _EvalDataset(Dataset):
    def __init__(self, adapter, total: int, img_size: int, yolopx_repo: str):
        self.adapter = adapter
        self.total = total
        self.img_size = img_size
        self.yolopx_repo = yolopx_repo
        _add_yolopx(yolopx_repo)  # ensure path set in parent before fork

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx):
        sample = self.adapter[idx]
        if not Path(sample.image_path).exists():
            return None
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            return None
        gt_mask = sample.target.mask
        if gt_mask is None:
            return None
        img_t, shapes = _preprocess(img_bgr, self.img_size, self.yolopx_repo)
        return img_t, shapes, gt_mask, sample.image_id, sample.image_path


def _collate(batch):
    """Filter None (skipped) items; stack tensors when all shapes match."""
    valid = [b for b in batch if b is not None]
    n_skipped = len(batch) - len(valid)
    if not valid:
        return None, n_skipped
    imgs, shapes, masks, ids, paths = zip(*valid)
    try:
        imgs_t = torch.stack(imgs)          # works when all letterbox sizes match
    except RuntimeError:
        imgs_t = list(imgs)                 # variable-size fallback (CurveLanes)
    return (imgs_t, list(shapes), list(masks), list(ids), list(paths)), n_skipped


# ── main eval loop ───────────────────────────────────────────────────────────

def run(args) -> dict:
    requested = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, device, use_half = _load_model(args.yolopx_repo, args.weights, requested)

    # YOLOPX SegmentationMetric — same class used in YOLOPX validate()
    _add_yolopx(args.yolopx_repo)
    from lib.core.evaluate import SegmentationMetric  # noqa: E402

    ll_metric = SegmentationMetric(2)

    adapter = _build_adapter(args)
    # In manifest mode the dataset name comes from the manifest metadata.
    if not args.dataset:
        args.dataset = getattr(adapter, "name", "manifest")
    total = min(len(adapter), args.max_samples) if args.max_samples else len(adapter)

    dataset = _EvalDataset(adapter, total, args.img_size, args.yolopx_repo)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )

    collect_per_image = bool(args.per_image or args.per_image_json or args.per_image_csv)
    per_records: list[dict] = []
    saver = None
    if args.save_pred_dir:
        from evaluation.save_predictions import PredictionSaver
        saver = PredictionSaver(args.save_pred_dir, args.model_name, args.dataset, total,
                                overlay_sample=args.overlay_sample)

    pred_writer = None
    if args.pred_manifest:
        from lane_eval.manifest import PredictionManifestWriter
        pred_writer = PredictionManifestWriter(args.pred_manifest, args.model_name, args.dataset)

    def _record(pred_mask, gt_mask, image_id, image_path):
        if saver is not None:
            saver.save(image_id, image_path, pred_mask, gt_mask)
        if pred_writer is not None:
            pred_writer.add(image_id, image_path, pred_mask)
        if not collect_per_image:
            return
        from evaluation.per_image_metrics import per_image_scores
        rec = {"model": args.model_name, "dataset": args.dataset, "split": args.split,
               "image_id": image_id, "image_path": image_path}
        rec.update(per_image_scores(pred_mask, gt_mask))
        per_records.append(rec)

    skipped = 0
    processed = 0
    for batch_data, n_skipped in tqdm(loader, desc=f"Evaluating {args.dataset}"):
        skipped += n_skipped
        if batch_data is None:
            continue

        imgs, shapes_list, gt_masks, image_ids, image_paths = batch_data

        if isinstance(imgs, torch.Tensor):
            # Batched path: all images had the same letterbox size
            imgs = imgs.to(device)
            if use_half:
                imgs = imgs.half()
            with torch.no_grad():
                _, _, ll_seg_out = model(imgs)
            for i in range(imgs.shape[0]):
                pred_mask = _extract_lane_mask(ll_seg_out[i:i+1], shapes_list[i], device)
                gt_mask = gt_masks[i]
                if pred_mask.shape != gt_mask.shape:
                    gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                ll_metric.addBatch(pred_mask.astype(np.int64), gt_mask.astype(np.int64))
                _record(pred_mask, gt_mask, image_ids[i], image_paths[i])
                processed += 1
        else:
            # Fallback: variable-size images (e.g. CurveLanes) — still uses num_workers
            for img_t, shapes, gt_mask, image_id, image_path in zip(imgs, shapes_list, gt_masks, image_ids, image_paths):
                img_t = img_t.unsqueeze(0).to(device)
                if use_half:
                    img_t = img_t.half()
                with torch.no_grad():
                    _, _, ll_seg_out = model(img_t)
                pred_mask = _extract_lane_mask(ll_seg_out, shapes, device)
                if pred_mask.shape != gt_mask.shape:
                    gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                ll_metric.addBatch(pred_mask.astype(np.int64), gt_mask.astype(np.int64))
                _record(pred_mask, gt_mask, image_id, image_path)
                processed += 1

    # All four headline metrics come from YOLOPX's own SegmentationMetric
    # confusion matrix — no new metric math is introduced:
    #   recall    = lineAccuracy()          = TP / (TP + FN)
    #   precision = classPixelAccuracy()[1] = TP / (TP + FP)
    #   F1        = harmonic mean of precision and recall
    recall    = float(ll_metric.lineAccuracy())
    precision = float(ll_metric.classPixelAccuracy()[1])
    f1        = float(2 * precision * recall / (precision + recall + 1e-12))
    results = {
        "model": args.model_name,
        "dataset": args.dataset,
        "task": "lane",
        "split": args.split,
        "num_images": processed,
        "skipped": skipped,
        "threshold": "argmax",
        "img_size": args.img_size,
        "metrics": {
            "lane_iou":       float(ll_metric.IntersectionOverUnion()),
            "lane_f1":        f1,
            "lane_precision": precision,
            "lane_recall":    recall,
            "lane_accuracy":  recall,    # alias: YOLOPX "Acc" == lane-class recall
            "lane_miou":      float(ll_metric.meanIntersectionOverUnion()),
            "pixel_accuracy": float(ll_metric.pixelAccuracy()),
        },
    }

    if collect_per_image:
        from evaluation.per_image_metrics import per_image_paths, write_per_image
        dj, dc = per_image_paths(args.output)
        pj = args.per_image_json or dj
        pc = args.per_image_csv or dc
        write_per_image(per_records, pj, pc)
        results["per_image_json"] = pj
        results["per_image_csv"] = pc
        results["per_image_count"] = len(per_records)
        print(f"  Saved per-image metrics -> {pj} (+ .csv), {len(per_records)} rows")

    if saver is not None:
        results["pred_masks_saved"] = saver.n_mask
        results["pred_overlays_saved"] = saver.n_overlay
        print(f"  Saved {saver.n_mask} pred masks + {saver.n_overlay} overlays -> {saver.mask_dir.parent}")

    if pred_writer is not None:
        out = pred_writer.write()
        results["pred_manifest"] = str(out)
        results["pred_manifest_count"] = len(pred_writer.samples)
        print(f"  Saved prediction manifest ({len(pred_writer.samples)} samples) -> {out}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOPX lane branch on a dataset")

    # model
    p.add_argument("--yolopx-repo", required=True, help="Path to YOLOPX repo root")
    p.add_argument("--weights",     required=True, help="Path to YOLOPX .pth checkpoint")
    p.add_argument("--model-name",  default="yolopx", help="Label for this model in the output JSON")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--img-size",    type=int, default=640)
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Images per forward pass (higher = more GPU utilisation)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes for parallel image loading")

    # dataset (shared) — --dataset is optional when --manifest is given, since
    # the manifest carries the dataset name in its metadata.
    p.add_argument("--dataset",  default=None,
                   choices=["bdd100k_lane", "curvelanes", "culane", "tusimple"])
    p.add_argument("--manifest", default=None,
                   help="Universal manifest JSON to use as the input source "
                        "(replaces the native --dataset/--root/--image-root args)")
    p.add_argument("--split",    default="val")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Stop after N samples (useful for quick debugging)")

    # bdd100k-specific
    p.add_argument("--image-root",     default=None, help="[bdd100k_lane] image directory")
    p.add_argument("--lane-mask-root", default=None, help="[bdd100k_lane] lane mask directory")

    # other datasets
    p.add_argument("--root", default=None, help="[curvelanes/culane/tusimple] dataset root")
    p.add_argument("--annotation-file", default=None,
                   help="[tusimple] explicit path to GT JSON (e.g. test_label.json)")
    p.add_argument("--mask-thickness", type=int, default=16)

    # output
    p.add_argument("--output", default="outputs/results/lane_eval.json")
    p.add_argument("--per-image", action="store_true",
                   help="Also save per-image IoU/F1/Precision/Recall as JSON + CSV "
                        "(in a per_image/ subdir next to --output)")
    p.add_argument("--per-image-json", default=None, help="Explicit per-image JSON path")
    p.add_argument("--per-image-csv", default=None, help="Explicit per-image CSV path")
    p.add_argument("--save-pred-dir", default=None,
                   help="If set, save predicted lane-mask PNG per image under DIR/<model>/<dataset>/masks/")
    p.add_argument("--overlay-sample", type=int, default=100,
                   help="Number of pred-vs-GT overlay JPGs to save per dataset (0 = none)")
    p.add_argument("--pred-manifest", default=None,
                   help="If set, write predictions (mask PNG + derived lane_json) to this "
                        "universal-format JSON; masks go in a masks/ dir beside it")

    args = p.parse_args()
    if not args.manifest and not args.dataset:
        p.error("either --manifest or --dataset must be provided")
    return args


if __name__ == "__main__":
    args = parse_args()
    results = run(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Lane Evaluation Results ===")
    print(f"  model        : {results['model']}")
    print(f"  dataset      : {results['dataset']} ({results['split']})")
    print(f"  images       : {results['num_images']}  (skipped: {results['skipped']})")
    m = results["metrics"]
    print(f"  lane_iou      : {m['lane_iou']:.4f}")
    print(f"  lane_f1       : {m['lane_f1']:.4f}")
    print(f"  lane_precision: {m['lane_precision']:.4f}")
    print(f"  lane_recall   : {m['lane_recall']:.4f}")
    print(f"  lane_miou     : {m['lane_miou']:.4f}")
    print(f"  pixel_acc     : {m['pixel_accuracy']:.4f}")
    print(f"\nSaved to: {out_path}")
