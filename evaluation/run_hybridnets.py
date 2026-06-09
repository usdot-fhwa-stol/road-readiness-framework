#!/usr/bin/env python3
"""Evaluate HybridNets lane segmentation on a dataset.

Fairness design: HybridNets is only used for *inference* (its own model,
weights, and preprocessing). Ground truth comes from THIS repo's `lane_eval`
adapters and scoring uses the SAME `SegmentationMetric` (from the YOLOPX repo)
and the SAME JSON schema as `lane_eval.cli.run_lane_eval`. So the model
comparison is apples-to-apples — only the inference differs, never the metric.

Must be run with the HybridNets venv interpreter (it has timm/efficientnet/etc.),
e.g. via run_hybridnets.sh:
    /shared/src/HybridNets/hybridnets/bin/python -m evaluation.run_hybridnets ...

HybridNets seg is MULTICLASS over {background=0, road=1, lane=2}; the lane mask
is `argmax(seg, 1) == 2`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# --- import paths -----------------------------------------------------------
# Our repo (lane_eval adapters), the HybridNets source (backbone/utils/hybridnets),
# and the YOLOPX repo (shared SegmentationMetric). No bare-name collisions:
# HybridNets uses top-level `utils`/`backbone`; ours is namespaced `lane_eval`;
# YOLOPX's metric is `lib.core.evaluate`.
def _setup_paths(hybridnets_repo: str, yolopx_repo: str) -> None:
    ours = str(Path(__file__).resolve().parents[1])
    for p in (ours, hybridnets_repo, yolopx_repo):
        if p and p not in sys.path:
            sys.path.insert(0, p)


# Populated by main() before the dataset workers fork, so workers inherit them.
_LETTERBOX = None
_INPUT_HW = (384, 640)   # HybridNets-D3 fixed input (H, W); 384=3*128, 640=5*128 -> BiFPN-safe
_MEAN = None
_STD = None


def _init_hybridnets_preproc(hybridnets_repo: str, project: str):
    """Load letterbox + project params (image_size, mean, std)."""
    global _LETTERBOX, _INPUT_HW, _MEAN, _STD
    import torch
    from utils.utils import letterbox, Params  # HybridNets
    _LETTERBOX = letterbox
    params = Params(str(Path(hybridnets_repo) / "projects" / f"{project}.yml"))
    size = params.model["image_size"]   # e.g. [640, 384] = [W, H]
    if isinstance(size, (list, tuple)):
        _INPUT_HW = (min(size), max(size))   # (H, W); all our datasets are landscape
    else:
        _INPUT_HW = (int(size), int(size))
    _MEAN = torch.tensor(params.mean, dtype=torch.float32).view(3, 1, 1)
    _STD = torch.tensor(params.std, dtype=torch.float32).view(3, 1, 1)
    return params


def _preprocess(img_bgr: np.ndarray):
    """HybridNets eval preprocessing: RGB -> letterbox to the fixed training
    input (384x640, divisible by 128 so the BiFPN is shape-safe for any aspect
    ratio) -> ImageNet-normalised CHW tensor. Returns (tensor, shapes) with
    shapes = ((h0, w0), (ratio, pad)).

    (The demo's resize-then-auto-letterbox only guarantees /32, which crashes the
    BiFPN on non-128-divisible sizes e.g. CurveLanes 660x1570 -> 288x640.)"""
    import torch
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    (lb_img, _), ratio, pad = _LETTERBOX((img_rgb, None), _INPUT_HW, auto=False, scaleup=False)
    arr = np.ascontiguousarray(lb_img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - _MEAN) / _STD
    shapes = ((h0, w0), (ratio, pad))
    return t, shapes


def _load_model(hybridnets_repo: str, weights: str, params, requested_device):
    import torch
    from backbone import HybridNetsBackbone
    from utils.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

    sd = torch.load(weights, map_location="cpu", weights_only=True)
    w = sd["segmentation_head.0.weight"]
    if w.size(0) == 1:
        seg_mode = BINARY_MODE
    else:
        seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE

    model = HybridNetsBackbone(
        num_classes=len(params.obj_list), compound_coef=3,
        ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
        seg_classes=len(params.seg_list), backbone_name=None, seg_mode=seg_mode,
    )
    model.load_state_dict(sd)
    model.requires_grad_(False)
    model.eval()

    # lane class index in the argmax map: background=0 then seg_list order (+1).
    lane_index = list(params.seg_list).index("lane") + 1

    device = requested_device
    use_half = False
    if requested_device.type != "cpu":
        try:
            model = model.to(requested_device).half()
            dummy = torch.zeros(1, 3, 384, 640, device=requested_device, dtype=torch.float16)
            with torch.no_grad():
                model(dummy)
            use_half = True
            print(f"Device: {requested_device} (float16)")
        except RuntimeError as exc:
            print(f"WARNING: GPU unusable ({exc.__class__.__name__}); falling back to CPU.")
            model = model.to("cpu").float()
            device = torch.device("cpu")
    else:
        print("Device: cpu (float32)")
    return model, device, use_half, seg_mode, lane_index


def _lane_mask_from_seg(seg_argmax_i: np.ndarray, shapes, lane_index: int) -> np.ndarray:
    """Undo letterbox padding, resize to original, keep the lane class."""
    (h0, w0), (_, pad) = shapes
    pad_w, pad_h = int(pad[0]), int(pad[1])
    H, W = seg_argmax_i.shape
    cropped = seg_argmax_i[pad_h:H - pad_h, pad_w:W - pad_w] if (pad_h > 0 or pad_w > 0) else seg_argmax_i
    full = cv2.resize(cropped.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (full == lane_index).astype(np.uint8)


def _build_adapter(args):
    # Manifest mode: read the universal manifest instead of a native dataset.
    if args.manifest:
        from lane_eval.manifest import ManifestDataset
        return ManifestDataset(args.manifest)

    from lane_eval.datasets import build_dataset_adapter
    name = args.dataset
    if name == "bdd100k_lane":
        return build_dataset_adapter(name, image_root=args.image_root,
                                     lane_mask_root=args.lane_mask_root, split=args.split)
    kwargs = dict(root=args.root, split=args.split)
    if args.mask_thickness:
        kwargs["mask_thickness"] = args.mask_thickness
    if name == "tusimple" and args.annotation_file:
        kwargs["annotation_files"] = [args.annotation_file]
    return build_dataset_adapter(name, **kwargs)


class _EvalDataset:
    def __init__(self, adapter, total: int):
        self.adapter = adapter
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = self.adapter[idx]
        if not Path(sample.image_path).exists():
            return None
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None or sample.target.mask is None:
            return None
        t, shapes = _preprocess(img_bgr)
        return t, shapes, sample.target.mask, sample.image_id, sample.image_path


def _collate(batch):
    import torch
    valid = [b for b in batch if b is not None]
    n_skipped = len(batch) - len(valid)
    if not valid:
        return None, n_skipped
    imgs, shapes, masks, ids, paths = zip(*valid)
    try:
        imgs_t = torch.stack(imgs)
    except RuntimeError:
        imgs_t = list(imgs)  # variable letterbox sizes -> per-image fallback
    return (imgs_t, list(shapes), list(masks), list(ids), list(paths)), n_skipped


def run(args) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from lib.core.evaluate import SegmentationMetric  # YOLOPX shared metric

    params = _init_hybridnets_preproc(args.hybridnets_repo, args.project)
    requested = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    model, device, use_half, seg_mode, lane_index = _load_model(
        args.hybridnets_repo, args.weights, params, requested)
    print(f"seg_mode={seg_mode} lane_index={lane_index}")

    ll_metric = SegmentationMetric(2)
    adapter = _build_adapter(args)
    # In manifest mode the dataset name comes from the manifest metadata.
    if not args.dataset:
        args.dataset = getattr(adapter, "name", "manifest")
    total = min(len(adapter), args.max_samples) if args.max_samples else len(adapter)
    loader = DataLoader(_EvalDataset(adapter, total), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=_collate)

    def _forward_seg(x):
        with torch.no_grad():
            seg = model(x)[-1]          # (B, 3, H, W)
        return torch.max(seg, dim=1)[1].detach().cpu().numpy().astype(np.uint8)  # (B, H, W)

    collect_per_image = bool(args.per_image or args.per_image_json or args.per_image_csv)
    per_records = []
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

    skipped = processed = 0
    for batch_data, n_skip in tqdm(loader, desc=f"HybridNets [{args.dataset}]"):
        skipped += n_skip
        if batch_data is None:
            continue
        imgs, shapes_list, gt_masks, image_ids, image_paths = batch_data
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.to(device)
            imgs = imgs.half() if use_half else imgs
            seg_argmax = _forward_seg(imgs)
        else:
            # variable letterbox sizes -> run each image individually
            seg_argmax = []
            for t in imgs:
                xi = t.unsqueeze(0).to(device)
                xi = xi.half() if use_half else xi
                seg_argmax.append(_forward_seg(xi)[0])
        for i in range(len(shapes_list)):
            argmax_i = seg_argmax[i]
            lane = _lane_mask_from_seg(argmax_i, shapes_list[i], lane_index)
            gt = gt_masks[i]
            if lane.shape != gt.shape:
                gt = cv2.resize(gt, (lane.shape[1], lane.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt > 0).astype(np.int64)
            ll_metric.addBatch(lane.astype(np.int64), gt_bin)
            _record(lane, gt_bin, image_ids[i], image_paths[i])
            processed += 1
    if skipped:
        print(f"  Skipped {skipped} images (missing file or mask)")

    recall    = float(ll_metric.lineAccuracy())
    precision = float(ll_metric.classPixelAccuracy()[1])
    f1        = float(2 * precision * recall / (precision + recall + 1e-12))
    results = {
        "model": args.model_name, "dataset": args.dataset, "task": "lane", "split": args.split,
        "num_images": processed, "skipped": skipped, "threshold": "argmax", "img_size": list(_INPUT_HW),
        "metrics": {
            "lane_iou": float(ll_metric.IntersectionOverUnion()),
            "lane_f1": f1, "lane_precision": precision, "lane_recall": recall,
            "lane_accuracy": recall,
            "lane_miou": float(ll_metric.meanIntersectionOverUnion()),
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


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HybridNets lane branch (scored by the shared harness)")
    p.add_argument("--hybridnets-repo", required=True, help="Path to the (restored) HybridNets source repo")
    p.add_argument("--yolopx-repo", required=True, help="Path to YOLOPX repo (for the shared SegmentationMetric)")
    p.add_argument("--weights", required=True)
    p.add_argument("--project", default="bdd100k", help="HybridNets project yml under projects/")
    p.add_argument("--model-name", default="hybridnets")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    # --dataset is optional when --manifest is given (name comes from the manifest).
    p.add_argument("--dataset", default=None, choices=["bdd100k_lane", "curvelanes", "culane", "tusimple"])
    p.add_argument("--manifest", default=None,
                   help="Universal manifest JSON to use as the input source "
                        "(replaces the native --dataset/--root/--image-root args)")
    p.add_argument("--split", default="val")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--image-root", default=None)
    p.add_argument("--lane-mask-root", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--annotation-file", default=None)
    p.add_argument("--mask-thickness", type=int, default=16)
    p.add_argument("--output", default="outputs/results/hybridnets_lane.json")
    p.add_argument("--per-image", action="store_true",
                   help="Also save per-image IoU/F1/Precision/Recall as JSON + CSV")
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
    _setup_paths(args.hybridnets_repo, args.yolopx_repo)
    results = run(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    m = results["metrics"]
    print("\n=== HybridNets Lane Evaluation ===")
    print(f"  model        : {results['model']}")
    print(f"  dataset      : {results['dataset']} ({results['split']})")
    print(f"  images       : {results['num_images']}  (skipped: {results['skipped']})")
    print(f"  lane_iou      : {m['lane_iou']:.4f}")
    print(f"  lane_f1       : {m['lane_f1']:.4f}")
    print(f"  lane_precision: {m['lane_precision']:.4f}")
    print(f"  lane_recall   : {m['lane_recall']:.4f}")
    print(f"\nSaved to: {out_path}")
