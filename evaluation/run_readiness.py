#!/usr/bin/env python3
"""Road readiness assessment runner.

Runs YOLOPX inference on a dataset, feeds predictions + GT masks into
ReadinessMetrics, and writes a JSON report.

Usage — BDD100K:
    python -m evaluation.run_readiness \
        --yolopx-repo /home/gauravb/Projects/road_readiness_t3/YOLOPX \
        --weights     /home/gauravb/Projects/road_readiness_t3/YOLOPX/weights/epoch-195.pth \
        --dataset     bdd100k_lane \
        --image-root      /shared/data/bdd100k/images/val \
        --lane-mask-root  /shared/data/bdd100k/ll_seg_annotations/val \
        --output      outputs/readiness/bdd100k.json

Usage — CurveLanes:
    python -m evaluation.run_readiness \
        --yolopx-repo /home/gauravb/Projects/road_readiness_t3/YOLOPX \
        --weights     /home/gauravb/Projects/road_readiness_t3/YOLOPX/weights/epoch-195.pth \
        --dataset     curvelanes \
        --root        /shared/data/Curvelanes \
        --split       valid \
        --output      outputs/readiness/curvelanes.json

Usage — quick dry-run with synthetic data (no model needed):
    python -m evaluation.run_readiness --dry-run --output outputs/readiness/dry_run.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── model helpers (same as run_lane_eval.py) ────────────────────────────────

_MEAN = None  # lazy-initialised below to avoid importing torch at top-level
_STD  = None


def _init_norm():
    import torch
    global _MEAN, _STD
    if _MEAN is None:
        _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        _STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _add_yolopx(repo: str) -> None:
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _to_tensor(img_rgb: np.ndarray):
    import torch
    _init_norm()
    arr = np.ascontiguousarray(img_rgb, dtype=np.float32) / 255.0
    t = torch.tensor(arr).permute(2, 0, 1)
    return (t - _MEAN) / _STD


def _load_model(yolopx_repo: str, weights: str, requested_device):
    import torch
    _add_yolopx(yolopx_repo)
    from lib.models import get_net  # noqa: E402 (YOLOPX internal)

    model = get_net(cfg=None)
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = requested_device
    use_half = False

    if requested_device.type != "cpu":
        try:
            model.to(requested_device).half()
            dummy = torch.zeros(1, 3, 384, 640, device=requested_device, dtype=torch.float16)
            with torch.no_grad():
                model(dummy)
            use_half = True
            print(f"Device: {requested_device} (float16)")
        except RuntimeError as exc:
            print(f"WARNING: GPU unusable ({exc.__class__.__name__}). Falling back to CPU.")
            model.to("cpu").float()
            device = torch.device("cpu")
    else:
        print("Device: cpu (float32)")

    return model, device, use_half


def _preprocess(img_bgr: np.ndarray, img_size: int, yolopx_repo: str):
    _add_yolopx(yolopx_repo)
    from lib.utils import letterbox_for_img  # noqa: E402

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    img_lb, ratio, pad = letterbox_for_img(img_rgb, img_size, auto=True)
    h, w = img_lb.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    return _to_tensor(img_lb), shapes


def _extract_lane_mask(ll_out, shapes, device) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    _, _, H, W = ll_out.shape
    (h0, w0), (_, pad) = shapes
    pw, ph = int(pad[0]), int(pad[1])
    cropped = ll_out[:, :, ph: H - ph, pw: W - pw]
    up = F.interpolate(cropped.float(), size=(h0, w0), mode="bilinear", align_corners=False)
    _, mask = torch.max(up, dim=1)
    return mask.int().squeeze().cpu().numpy().astype(np.uint8)


# ── dataset builder ──────────────────────────────────────────────────────────

def _build_adapter(args):
    from lane_eval.datasets import build_dataset_adapter

    name = args.dataset
    if name == "bdd100k_lane":
        kw = dict(image_root=args.image_root, lane_mask_root=args.lane_mask_root, split=args.split)
        if args.det_annotations_root:
            kw["det_annotations_root"] = args.det_annotations_root
        return build_dataset_adapter(name, **kw)
    kw = dict(root=args.root, split=args.split)
    if args.mask_thickness:
        kw["mask_thickness"] = args.mask_thickness
    if name == "tusimple" and args.annotation_file:
        kw["annotation_files"] = [args.annotation_file]
    return build_dataset_adapter(name, **kw)


# ── inference loop ───────────────────────────────────────────────────────────

class _InferenceDataset:
    """Thin wrapper so DataLoader can parallelise image loading + preprocessing."""

    def __init__(self, adapter, total: int, img_size: int, yolopx_repo: str):
        self.adapter = adapter
        self.total = total
        self.img_size = img_size
        self.yolopx_repo = yolopx_repo
        _add_yolopx(yolopx_repo)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx):
        sample = self.adapter[idx]
        if not Path(sample.image_path).exists():
            return None
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None or sample.target.mask is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t, shapes = _preprocess(img_bgr, self.img_size, self.yolopx_repo)
        return img_t, shapes, sample.target.mask, img_rgb, sample


def _collate(batch):
    """Drop None items; stack tensors when letterbox sizes match (same-size datasets)."""
    valid = [b for b in batch if b is not None]
    n_skip = len(batch) - len(valid)
    if not valid:
        return None, n_skip
    imgs, shapes, gt_masks, images_rgb, samples = zip(*valid)
    try:
        imgs_t = __import__("torch").stack(imgs)
    except RuntimeError:
        imgs_t = list(imgs)  # variable-size fallback (CurveLanes)
    return (imgs_t, list(shapes), list(gt_masks), list(images_rgb), list(samples)), n_skip


def _run_inference(args) -> Tuple[list, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns (samples, pred_masks, gt_masks, images_rgb).
    All lists are aligned (same index = same frame).
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    requested = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    model, device, use_half = _load_model(args.yolopx_repo, args.weights, requested)
    adapter = _build_adapter(args)
    total = min(len(adapter), args.max_samples) if args.max_samples else len(adapter)

    dataset = _InferenceDataset(adapter, total, args.img_size, args.yolopx_repo)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )

    samples, pred_masks, gt_masks, images_rgb = [], [], [], []
    skipped = 0

    for batch_data, n_skip in tqdm(loader, desc=f"Inference [{args.dataset}]"):
        skipped += n_skip
        if batch_data is None:
            continue

        imgs, shapes_list, batch_gt, batch_rgb, batch_samples = batch_data

        if isinstance(imgs, torch.Tensor):
            imgs = imgs.to(device)
            if use_half:
                imgs = imgs.half()
            with torch.no_grad():
                _, _, ll_out = model(imgs)
            for i in range(imgs.shape[0]):
                pred_mask = _extract_lane_mask(ll_out[i:i+1], shapes_list[i], device)
                gt_mask = batch_gt[i]
                if pred_mask.shape != gt_mask.shape:
                    gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                samples.append(batch_samples[i])
                pred_masks.append(pred_mask)
                gt_masks.append(gt_mask)
                images_rgb.append(batch_rgb[i])
        else:
            # Variable-size fallback: run one at a time
            for img_t, shapes, gt_mask, img_rgb, sample in zip(
                imgs, shapes_list, batch_gt, batch_rgb, batch_samples
            ):
                img_t = img_t.unsqueeze(0).to(device)
                if use_half:
                    img_t = img_t.half()
                with torch.no_grad():
                    _, _, ll_out = model(img_t)
                pred_mask = _extract_lane_mask(ll_out, shapes, device)
                if pred_mask.shape != gt_mask.shape:
                    gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                samples.append(sample)
                pred_masks.append(pred_mask)
                gt_masks.append(gt_mask)
                images_rgb.append(img_rgb)

    if skipped:
        print(f"  Skipped {skipped} images (missing file or mask)")
    return samples, pred_masks, gt_masks, images_rgb


# ── dry-run (no model) ───────────────────────────────────────────────────────

def _dry_run(n: int = 50):
    """Generate synthetic samples for testing without a model or dataset."""
    from lane_eval.schema.sample import LaneSample
    from lane_eval.schema.lane import LaneTarget
    import tempfile, os

    rng = np.random.default_rng(0)
    samples, pred_masks, gt_masks, images_rgb = [], [], [], []

    tmpdir = tempfile.mkdtemp()
    for i in range(n):
        h, w = 720, 1280
        img = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)

        gt = np.zeros((h, w), dtype=np.uint8)
        for lane_x in [350, 500, 650]:
            for y in range(200, 600):
                x = lane_x + int(rng.normal(0, 2))
                if 0 <= x < w:
                    gt[y, max(0, x - 8): x + 8] = 1

        # Add brightness to lane pixels so contrast is measurable
        img[gt == 1] = np.clip(img[gt == 1].astype(int) + 60, 0, 255).astype(np.uint8)

        pred = gt.copy()
        noise_idx = rng.integers(0, gt.size, size=gt.size // 20)
        pred.flat[noise_idx] = 1 - pred.flat[noise_idx]

        img_path = os.path.join(tmpdir, f"img_{i:04d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        sample = LaneSample(
            image_id=f"dry_{i:04d}",
            image_path=img_path,
            width=w,
            height=h,
            target=LaneTarget(mask=gt),
        )
        samples.append(sample)
        pred_masks.append(pred)
        gt_masks.append(gt)
        images_rgb.append(img)

    return samples, pred_masks, gt_masks, images_rgb


# ── metrics computation ──────────────────────────────────────────────────────

def _per_image_layer2(img: np.ndarray, gt: np.ndarray, pm: np.ndarray) -> dict:
    """Compute all Layer-2 metrics for a single image (used for C1-C3 and failure detection)."""
    from scipy.ndimage import label as ndi_label

    d1 = 100.0 if np.sum(pm) > 0 else 0.0

    tp = int(np.sum((pm > 0) & (gt > 0)))
    fp = int(np.sum((pm > 0) & (gt == 0)))
    fn = int(np.sum((pm == 0) & (gt > 0)))
    d3 = tp / (tp + fp + fn + 1e-6)

    # D2: lane count match
    _, pred_cc = ndi_label(pm)
    _, gt_cc   = ndi_label(gt)
    d2 = 100.0 if pred_cc == gt_cc else 0.0

    # D4: near-field IoU (bottom half)
    h = pm.shape[0]
    pm_n, gm_n = pm[h // 2:, :], gt[h // 2:, :]
    tp4 = int(np.sum((pm_n > 0) & (gm_n > 0)))
    fp4 = int(np.sum((pm_n > 0) & (gm_n == 0)))
    fn4 = int(np.sum((pm_n == 0) & (gm_n > 0)))
    d4 = tp4 / (tp4 + fp4 + fn4 + 1e-6)

    # D5: occlusion gap
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    occ = hsv[:, :, 2] < 100
    vis = ~occ
    def _riou(mask):
        t = int(np.sum((pm > 0) & (gt > 0) & mask))
        u = int(np.sum(((pm > 0) | (gt > 0)) & mask))
        return t / (u + 1e-6)
    iou_vis = _riou(vis) if vis.any() else 0.0
    iou_occ = _riou(occ) if occ.any() else 0.0
    d5 = iou_vis - iou_occ

    # D6: detection gap
    fn_px = int(np.sum((pm == 0) & (gt > 0)))
    gt_px = int(np.sum(gt > 0))
    d6 = fn_px / (gt_px + 1e-6)

    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6}


def _compute_all(
    samples, pred_masks, gt_masks, images_rgb, stratify: List[str],
    save_failures: bool = False, failures_dir: str = "outputs/readiness/failures",
    failures_max_side: int = 1280,
) -> dict:
    from tqdm import tqdm
    from evaluation.readiness_metrics import (
        ReadinessMetrics,
        filter_by_weather, filter_by_time, filter_by_context, filter_by_traffic,
    )

    rm = ReadinessMetrics()

    # Per-image Layer 1 + full per-image Layer 2 (for C1-C3 and failure detection)
    layer1_list = []
    layer2_per_img = []   # full per-image dicts (I1-I6 + D1-D6)
    print("Computing Layer 1 (infrastructure)…")
    for img, gt, pm in tqdm(
        zip(images_rgb, gt_masks, pred_masks), total=len(gt_masks)
    ):
        l1 = rm.compute_infrastructure(img, gt)
        l2 = _per_image_layer2(img, gt, pm)
        layer1_list.append(l1)
        layer2_per_img.append({**l1, **l2})

    # Aggregate Layer 2
    print("Computing Layer 2 (detection)…")
    layer2 = rm.compute_detection(pred_masks, gt_masks, images_rgb)

    print("Computing Layer 3 (correlation)…")
    # Pass only D1/D3 per image — the keys correlations actually use
    corr_per_img = [{"D1": m["D1"], "D3": m["D3"]} for m in layer2_per_img]
    layer3 = rm.compute_correlations(layer1_list, corr_per_img)

    # Layer 4
    avg_infra = {k: float(np.mean([m[k] for m in layer1_list])) for k in layer1_list[0]}
    layer4 = rm.compute_verdict(avg_infra, layer2, layer3)

    # Failure images
    if save_failures:
        from evaluation.visualize_failures import save_failure_images
        print(f"Saving failure images → {failures_dir}")
        counts = save_failure_images(
            samples, images_rgb, pred_masks, gt_masks,
            layer2_per_img, failures_dir, max_side=failures_max_side,
        )
        saved_total = sum(counts.values())
        for metric, n in sorted(counts.items()):
            if n:
                print(f"  {metric}: {n} failures saved")
        print(f"  Total: {saved_total} images saved across {sum(1 for n in counts.values() if n)} metrics")

    report = {
        "num_samples": len(samples),
        "overall": {
            "layer1_avg": avg_infra,
            "layer2": layer2,
            "layer3": {
                k: (v if not isinstance(v, list) else [(c, float(s)) for c, s in v])
                for k, v in layer3.items()
            },
            "layer4": {
                "R1": layer4["R1"],
                "R2": layer4["R2"],
                "R3": list(layer4["R3"]),
                "R4": layer4["R4"],
            },
        },
        "strata": {},
    }

    # Optional stratification
    FILTER_MAP = {
        "weather":  (filter_by_weather,  ["clear", "rainy", "foggy", "night"]),
        "time":     (filter_by_time,     ["day", "night"]),
        "context":  (filter_by_context,  ["rural_highway", "urban_street", "urban_intersection"]),
        "traffic":  (filter_by_traffic,  ["low", "medium", "high"]),
    }

    for dim in stratify:
        if dim not in FILTER_MAP:
            print(f"  WARNING: unknown stratification dim '{dim}', skipping")
            continue
        fn, values = FILTER_MAP[dim]
        print(f"Stratifying by {dim}…")
        report["strata"][dim] = {}
        for val in values:
            filtered = fn(samples, val)
            if len(filtered) < 10:
                print(f"  {dim}={val}: {len(filtered)} samples (< 10, skipped)")
                continue
            # Align pred_masks to filtered samples
            sample_set = set(id(s) for s in filtered)
            indices = [i for i, s in enumerate(samples) if id(s) in sample_set]
            f_pred = [pred_masks[i] for i in indices]
            f_gt   = [gt_masks[i]   for i in indices]
            f_imgs = [images_rgb[i] for i in indices]

            fl1 = []
            fl2_per_img = []
            for img, gt, pm in zip(f_imgs, f_gt, f_pred):
                fl1.append(rm.compute_infrastructure(img, gt))
                d1 = 100.0 if np.sum(pm) > 0 else 0.0
                tp = int(np.sum((pm > 0) & (gt > 0)))
                fp = int(np.sum((pm > 0) & (gt == 0)))
                fn = int(np.sum((pm == 0) & (gt > 0)))
                d3 = tp / (tp + fp + fn + 1e-6)
                fl2_per_img.append({"D1": d1, "D3": d3})
            fl2 = rm.compute_detection(f_pred, f_gt, f_imgs)
            fl3 = rm.compute_correlations(fl1, fl2_per_img)
            avg = {k: float(np.mean([m[k] for m in fl1])) for k in fl1[0]}
            fl4 = rm.compute_verdict(avg, fl2, fl3)

            report["strata"][dim][val] = {
                "num_samples": len(filtered),
                "layer1_avg": avg,
                "layer2": fl2,
                "layer4": {
                    "R1": fl4["R1"],
                    "R2": fl4["R2"],
                    "R3": list(fl4["R3"]),
                    "R4": fl4["R4"],
                },
            }
            verdict = fl4["R4"]["status"]
            print(f"  {dim}={val}: n={len(filtered)}, R1={fl4['R1']:.1f}, R2={fl4['R2']:.1f} → {verdict}")

    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Road readiness assessment with YOLOPX")

    # model
    p.add_argument("--yolopx-repo", default=None, help="Path to YOLOPX repo root")
    p.add_argument("--weights",     default=None, help="Path to YOLOPX .pth checkpoint")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--img-size",     type=int, default=640)
    p.add_argument("--batch-size",   type=int, default=8,
                   help="Images per GPU forward pass (default: 8)")
    p.add_argument("--num-workers",  type=int, default=4,
                   help="DataLoader worker processes for parallel image loading (default: 4)")

    # dataset
    p.add_argument("--dataset",  default=None,
                   choices=["bdd100k_lane", "curvelanes", "culane", "tusimple"])
    p.add_argument("--split",    default="val")
    p.add_argument("--max-samples", type=int, default=None)

    # bdd100k
    p.add_argument("--image-root",           default=None)
    p.add_argument("--lane-mask-root",       default=None)
    p.add_argument("--det-annotations-root", default=None,
                   help="[bdd100k_lane] path to det_annotations/{split}/ directory "
                        "for weather/scene/timeofday metadata (enables --stratify)")

    # other datasets
    p.add_argument("--root",            default=None)
    p.add_argument("--annotation-file", default=None)
    p.add_argument("--mask-thickness",  type=int, default=16)

    # readiness options
    p.add_argument(
        "--stratify", nargs="*", default=[],
        choices=["weather", "time", "context", "traffic"],
        help="Dimensions to stratify results by (requires metadata in samples)",
    )

    # output
    p.add_argument("--output", default="outputs/readiness/report.json")

    # failure visualisation
    p.add_argument("--save-failures", action="store_true",
                   help="Save raw + overlay images for every frame that fails a metric threshold")
    p.add_argument("--failures-dir", default="outputs/readiness/failures",
                   help="Root folder for per-metric failure images (default: outputs/readiness/failures)")
    p.add_argument("--failures-max-side", type=int, default=1280,
                   help="Max image dimension when saving failures (default: 1280)")

    # dry-run mode
    p.add_argument("--dry-run", action="store_true",
                   help="Run on synthetic data — no model or dataset required")
    p.add_argument("--dry-run-n", type=int, default=50,
                   help="Number of synthetic samples for dry-run")

    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print(f"Dry-run mode: generating {args.dry_run_n} synthetic samples…")
        samples, pred_masks, gt_masks, images_rgb = _dry_run(args.dry_run_n)
    else:
        if not args.yolopx_repo or not args.weights or not args.dataset:
            print("ERROR: --yolopx-repo, --weights, and --dataset are required unless --dry-run")
            sys.exit(1)
        samples, pred_masks, gt_masks, images_rgb = _run_inference(args)

    if not samples:
        print("ERROR: no valid samples found, nothing to evaluate")
        sys.exit(1)

    print(f"\nRunning ReadinessMetrics on {len(samples)} samples…")
    report = _compute_all(
        samples, pred_masks, gt_masks, images_rgb, args.stratify,
        save_failures=args.save_failures,
        failures_dir=args.failures_dir,
        failures_max_side=args.failures_max_side,
    )

    report["dataset"] = args.dataset or "dry_run"
    report["split"]   = args.split

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    ov = report["overall"]
    l4 = ov["layer4"]
    print("\n" + "=" * 55)
    print("ROAD READINESS REPORT")
    print("=" * 55)
    print(f"  Samples evaluated : {report['num_samples']}")
    print(f"  Infrastructure R1 : {l4['R1']:.1f} / 100")
    print(f"  Detection     R2  : {l4['R2']:.1f} / 100")
    print(f"  Bottleneck    R3  : {l4['R3'][0]} (strength={l4['R3'][1]:.3f})")
    print(f"  Verdict       R4  : {l4['R4']['status']}")
    print(f"  Reason            : {l4['R4']['reason']}")
    print("-" * 55)
    l1 = ov["layer1_avg"]
    print(f"  I1 continuity     : {l1['I1']:.1f}%")
    print(f"  I2 contrast       : {l1['I2']:.3f}")
    print(f"  I3 sharpness      : {l1['I3']:.3f}")
    print(f"  I4 width std      : {l1['I4']:.2f} px")
    print(f"  I5 curvature      : {l1['I5']:.4f}")
    print(f"  I6 lane count avg : {l1['I6']:.1f}")
    print("-" * 55)
    l2 = ov["layer2"]
    print(f"  D1 detection rate : {l2['D1']:.1f}%")
    print(f"  D2 count accuracy : {l2['D2']:.1f}%")
    print(f"  D3 IoU            : {l2['D3']:.3f}")
    print(f"  D4 near-field IoU : {l2['D4']:.3f}")
    print(f"  D5 occlusion gap  : {l2['D5']:.3f}")
    print(f"  D6 detection gap  : {l2['D6']:.3f}")
    print("=" * 55)
    print(f"Saved → {out_path}\n")


if __name__ == "__main__":
    main()
