#!/usr/bin/env python3
"""Lane-marking machine-readability runner.

Runs YOLOPX inference when model arguments are provided, or a synthetic dry-run
when --dry-run is used. The v2 output is per-image-record first and then
aggregated with evaluation.readiness_metrics.summarize_records.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

_MEAN = None
_STD = None


def _init_norm():
    import torch
    global _MEAN, _STD
    if _MEAN is None:
        _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _add_yolopx(repo: str) -> None:
    if repo and repo not in sys.path:
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
    from lib.models import get_net

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
    from lib.utils import letterbox_for_img

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    img_lb, ratio, pad = letterbox_for_img(img_rgb, img_size, auto=True)
    h, w = img_lb.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    return _to_tensor(img_lb), shapes


def extract_lane_mask_and_probability(ll_seg_out, shapes, device=None, threshold: Optional[float] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Undo letterbox, return binary lane mask and optional lane probability.

    2-channel lane logits use softmax(channel=1) and argmax unless threshold is
    given. 1-channel logits use sigmoid and threshold 0.5 by default. For other
    channel counts, the mask preserves class-1 argmax behavior and probability is
    unavailable because the lane channel is ambiguous.
    """
    import torch
    import torch.nn.functional as F

    if ll_seg_out.ndim != 4:
        raise ValueError(f"Expected BCHW ll_seg_out, got shape {tuple(ll_seg_out.shape)}")
    _, channels, height, width = ll_seg_out.shape
    (h0, w0), (_, pad) = shapes
    pad_w, pad_h = int(pad[0]), int(pad[1])
    y0, y1 = pad_h, height - pad_h if pad_h > 0 else height
    x0, x1 = pad_w, width - pad_w if pad_w > 0 else width
    if y1 <= y0 or x1 <= x0:
        y0, y1, x0, x1 = 0, height, 0, width
    cropped = ll_seg_out[:, :, y0:y1, x0:x1]
    up = F.interpolate(cropped.float(), size=(h0, w0), mode="bilinear", align_corners=False)

    if channels == 1:
        prob_t = torch.sigmoid(up[:, 0])
        mask_t = prob_t >= (0.5 if threshold is None else threshold)
        prob = prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        mask = mask_t.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return mask, prob
    if channels == 2:
        prob_all = torch.softmax(up, dim=1)
        lane_prob = prob_all[:, 1]
        if threshold is None:
            mask_t = torch.argmax(prob_all, dim=1) == 1
        else:
            mask_t = lane_prob >= threshold
        prob = lane_prob.squeeze(0).detach().cpu().numpy().astype(np.float32)
        mask = mask_t.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return mask, prob

    _, cls = torch.max(up, dim=1)
    mask = (cls == 1).int().squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return mask, None


def _extract_lane_mask(ll_out, shapes, device) -> np.ndarray:
    mask, _ = extract_lane_mask_and_probability(ll_out, shapes, device=device)
    return mask


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


class _InferenceDataset:
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
    valid = [b for b in batch if b is not None]
    n_skip = len(batch) - len(valid)
    if not valid:
        return None, n_skip
    imgs, shapes, gt_masks, images_rgb, samples = zip(*valid)
    try:
        imgs_t = __import__("torch").stack(imgs)
    except RuntimeError:
        imgs_t = list(imgs)
    return (imgs_t, list(shapes), list(gt_masks), list(images_rgb), list(samples)), n_skip


def _run_inference(args) -> Tuple[list, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]]]:
    """Return aligned samples, pred masks, GT masks, RGB images, lane prob maps."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    requested = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    model, device, use_half = _load_model(args.yolopx_repo, args.weights, requested)
    adapter = _build_adapter(args)
    total = min(len(adapter), args.max_samples) if args.max_samples else len(adapter)
    loader = DataLoader(
        _InferenceDataset(adapter, total, args.img_size, args.yolopx_repo),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )

    samples, pred_masks, gt_masks, images_rgb, lane_probs = [], [], [], [], []
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
            iterator = ((ll_out[i:i + 1], shapes_list[i], batch_gt[i], batch_rgb[i], batch_samples[i]) for i in range(imgs.shape[0]))
        else:
            single_outputs = []
            for img_t, shapes, gt_mask, img_rgb, sample in zip(imgs, shapes_list, batch_gt, batch_rgb, batch_samples):
                img_t = img_t.unsqueeze(0).to(device)
                if use_half:
                    img_t = img_t.half()
                with torch.no_grad():
                    _, _, ll_out = model(img_t)
                single_outputs.append((ll_out, shapes, gt_mask, img_rgb, sample))
            iterator = single_outputs
        for ll_slice, shapes, gt_mask, img_rgb, sample in iterator:
            pred_mask, lane_prob = extract_lane_mask_and_probability(ll_slice, shapes, device=device)
            if pred_mask.shape != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            samples.append(sample)
            pred_masks.append(pred_mask)
            gt_masks.append(gt_mask)
            images_rgb.append(img_rgb)
            lane_probs.append(lane_prob)
    if skipped:
        print(f"  Skipped {skipped} images (missing file or mask)")
    return samples, pred_masks, gt_masks, images_rgb, lane_probs


def _dry_run(n: int = 50):
    from lane_eval.schema.sample import LaneSample
    from lane_eval.schema.lane import LaneTarget
    import tempfile
    import os

    rng = np.random.default_rng(0)
    samples, pred_masks, gt_masks, images_rgb, lane_probs = [], [], [], [], []
    tmpdir = tempfile.mkdtemp()
    for i in range(n):
        h, w = 720, 1280
        img = np.full((h, w, 3), 58, dtype=np.uint8)
        img[:180, :, :] = 190
        gt = np.zeros((h, w), dtype=np.uint8)
        for lane_x in [360, 520, 690]:
            for y in range(210, 650):
                x = lane_x + int(0.00055 * (y - 430) ** 2) + int(rng.normal(0, 1))
                gt[y, max(0, x - 5): min(w, x + 6)] = 1
        img[gt > 0] = 225
        pred = gt.copy()
        if i % 3 == 0:
            pred[260:330, :] = 0
        if i % 5 == 0:
            pred = np.roll(pred, 3, axis=1)
        noise = rng.random(gt.shape) < 0.001
        pred[noise] = 1
        img_path = os.path.join(tmpdir, f"img_{i:04d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sample = LaneSample(
            image_id=f"dry_{i:04d}", image_path=img_path, width=w, height=h,
            target=LaneTarget(mask=gt, meta={"style": "solid"}),
            meta={"weather": "clear" if i % 2 == 0 else "rainy", "timeofday": "daytime", "condition": "clear" if i % 2 == 0 else "wet"},
        )
        samples.append(sample)
        pred_masks.append(pred.astype(np.uint8))
        gt_masks.append(gt)
        images_rgb.append(img)
        lane_probs.append(None)
    return samples, pred_masks, gt_masks, images_rgb, lane_probs


def _run_manifest_predictions(gt_manifest: str, pred_manifest: str, max_samples: Optional[int] = None):
    """Load aligned GT samples and prediction geometry without model inference."""

    from lane_eval.manifest import ManifestDataset, PredictionManifestReader

    dataset = ManifestDataset(gt_manifest)
    predictions = PredictionManifestReader(pred_manifest)
    total = min(len(dataset), max_samples) if max_samples else len(dataset)
    samples, pred_masks, gt_masks, images_rgb, lane_probs, prediction_inputs = [], [], [], [], [], []
    for index in range(total):
        sample = dataset[index]
        prediction = predictions.get(sample.image_id)
        if prediction is None:
            continue
        image_bgr = cv2.imread(str(sample.image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        gt_mask = sample.target.mask
        if gt_mask is None:
            gt_mask = np.zeros((height, width), dtype=np.uint8)
        pred_mask = prediction.mask
        if pred_mask is None:
            pred_mask = np.zeros((height, width), dtype=np.uint8)
        samples.append(sample)
        pred_masks.append(pred_mask)
        gt_masks.append(gt_mask)
        images_rgb.append(image_rgb)
        lane_probs.append(prediction.prob)
        prediction_inputs.append(
            {
                "lanes": prediction.lanes,
                "lane_json": prediction.meta.get("lane_json"),
                "geometry_source": prediction.meta.get("geometry_source"),
            }
        )
    return samples, pred_masks, gt_masks, images_rgb, lane_probs, prediction_inputs


def _record_stratum_value(record: dict, dim: str) -> Optional[str]:
    meta = record.get("metadata") or {}
    if dim == "time":
        return meta.get("timeofday")
    if dim == "context":
        return meta.get("condition") or meta.get("scene")
    if dim == "traffic":
        return meta.get("traffic_proxy")
    return meta.get(dim)


def _scalar_for_csv(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _flat_record(record: dict) -> dict:
    flat = {}
    for key, value in record.items():
        if _scalar_for_csv(value):
            flat[key] = value
    for key, value in (record.get("metadata") or {}).items():
        if _scalar_for_csv(value):
            flat[f"metadata.{key}"] = value
    return flat


def _write_jsonl(path: str, records: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + "\n")


def _write_csv(path: str, records: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flat_record(r) for r in records]
    fieldnames = sorted({key for row in rows for key in row})
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compute_all(
    args,
    samples,
    pred_masks,
    gt_masks,
    images_rgb,
    lane_probs,
    prediction_inputs=None,
) -> tuple[dict, list[dict]]:
    from tqdm import tqdm
    from evaluation.readiness_metrics import METRIC_VERSION, build_metric_record, summarize_records

    dataset = args.dataset or "dry_run"
    print("Building per-image metric records...")
    records = []
    if prediction_inputs is None:
        prediction_inputs = [{} for _ in samples]
    iterator = zip(samples, pred_masks, images_rgb, lane_probs, prediction_inputs)
    for sample, pred, img, prob, prediction_input in tqdm(iterator, total=len(samples)):
        records.append(
            build_metric_record(
                sample,
                pred,
                img,
                dataset=dataset,
                split=args.split,
                model_name=args.model_name,
                lane_prob=prob,
                pred_lanes=prediction_input.get("lanes"),
                pred_lane_json=prediction_input.get("lane_json"),
                pred_geometry_source=prediction_input.get("geometry_source"),
            )
        )

    overall = summarize_records(records, min_corr_samples=args.min_stratum_samples)
    strata = {}
    for dim in args.stratify:
        groups: dict[str, list[dict]] = {}
        for record in records:
            value = _record_stratum_value(record, dim)
            if value is None:
                continue
            groups.setdefault(str(value), []).append(record)
        strata[dim] = {}
        for value, group in sorted(groups.items()):
            if len(group) < args.min_stratum_samples:
                strata[dim][value] = {"num_samples": len(group), "skipped": True, "reason": f"n < {args.min_stratum_samples}"}
                continue
            strata[dim][value] = summarize_records(group, min_corr_samples=args.min_stratum_samples)
            strata[dim][value]["num_samples"] = len(group)

    if args.save_failures:
        from evaluation.visualize_failures import save_failure_images
        counts = save_failure_images(samples, images_rgb, pred_masks, gt_masks, records, args.failures_dir, max_side=args.failures_max_side)
        print(f"Saved {sum(counts.values())} failure example files across {sum(1 for n in counts.values() if n)} metrics")
    if args.save_good:
        from evaluation.visualize_failures import save_good_images
        counts = save_good_images(samples, images_rgb, pred_masks, gt_masks, records, args.good_dir, max_side=args.failures_max_side)
        print(f"Saved {sum(counts.values())} good example files across {sum(1 for n in counts.values() if n)} metrics")

    report = {
        "metric_version": METRIC_VERSION,
        "num_samples": len(records),
        "dataset": dataset,
        "split": args.split,
        "model_name": args.model_name,
        "confidence_available": bool(any(r.get("D7_confidence_mean") is not None for r in records)),
        "d7_available": bool(any(r.get("D7_confidence_mean") is not None for r in records)),
        "overall": overall,
        "strata": strata,
        "warnings_limitations": [
            "Image-based proxy only; not a field-certified road-readiness standard.",
            "D7 confidence is None unless lane probability was extracted from model logits.",
            "Video-based temporal jitter is not computed; this is a single-frame evaluation.",
            "D5 occlusion/visibility gap requires human 'Observed Marking Visibility' tags and is only reported at the group level (not per image).",
        ],
    }
    return report, records


def parse_args():
    p = argparse.ArgumentParser(description="Lane-marking machine-readability assessment with YOLOPX")
    p.add_argument("--yolopx-repo", default=None, help="Path to YOLOPX repo root")
    p.add_argument("--weights", default=None, help="Path to YOLOPX .pth checkpoint")
    p.add_argument("--manifest", default=None, help="Universal GT manifest for prediction-manifest scoring")
    p.add_argument("--pred-manifest", default=None, help="Prediction manifest carrying masks and optional native geometry")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dataset", default=None, choices=["bdd100k_lane", "curvelanes", "culane", "tusimple"])
    p.add_argument("--split", default="val")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--image-root", default=None)
    p.add_argument("--lane-mask-root", default=None)
    p.add_argument("--det-annotations-root", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--annotation-file", default=None)
    p.add_argument("--mask-thickness", type=int, default=16)
    p.add_argument("--stratify", nargs="*", default=[], choices=["weather", "time", "context", "traffic", "condition", "scene"])
    p.add_argument("--min-stratum-samples", type=int, default=10)
    p.add_argument("--model-name", default="yolopx")
    p.add_argument("--output", default="outputs/readiness/report.json")
    p.add_argument("--per-image-output", default=None, help="Optional JSONL path for one metric record per image")
    p.add_argument("--per-image-csv", default=None, help="Optional flat CSV path for scalar per-image metrics")
    p.add_argument("--output-manifest", default=None, help="Dir: write an output manifest (input manifest + per-sample core/D/I/R metrics) + CSV + overlay images")
    p.add_argument("--save-failures", action="store_true")
    p.add_argument("--failures-dir", default="outputs/readiness/failures")
    p.add_argument("--save-good", action="store_true")
    p.add_argument("--good-dir", default="outputs/readiness/good")
    p.add_argument("--failures-max-side", type=int, default=1280)
    p.add_argument("--dry-run", action="store_true", help="Run on synthetic data; no model or dataset required")
    p.add_argument("--dry-run-n", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    prediction_inputs = None
    if args.dry_run:
        print(f"Dry-run mode: generating {args.dry_run_n} synthetic samples...")
        samples, pred_masks, gt_masks, images_rgb, lane_probs = _dry_run(args.dry_run_n)
    elif args.manifest or args.pred_manifest:
        if not args.manifest or not args.pred_manifest:
            print("ERROR: --manifest and --pred-manifest must be supplied together")
            sys.exit(1)
        loaded = _run_manifest_predictions(args.manifest, args.pred_manifest, args.max_samples)
        samples, pred_masks, gt_masks, images_rgb, lane_probs, prediction_inputs = loaded
        if args.dataset is None:
            from lane_eval.manifest import ManifestDataset

            args.dataset = ManifestDataset(args.manifest).name
    else:
        if not args.yolopx_repo or not args.weights or not args.dataset:
            print("ERROR: model arguments are required unless --dry-run or manifest scoring is used")
            sys.exit(1)
        samples, pred_masks, gt_masks, images_rgb, lane_probs = _run_inference(args)

    if not samples:
        print("ERROR: no valid samples found, nothing to evaluate")
        sys.exit(1)

    print(f"\nRunning readiness metrics on {len(samples)} aligned records...")
    report, records = _compute_all(
        args,
        samples,
        pred_masks,
        gt_masks,
        images_rgb,
        lane_probs,
        prediction_inputs=prediction_inputs,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    if args.per_image_output:
        _write_jsonl(args.per_image_output, records)
    if args.per_image_csv:
        _write_csv(args.per_image_csv, records)
    if args.output_manifest:
        from evaluation.eval_output_manifest import write_output_manifest
        write_output_manifest(
            args.manifest, records, report.get("overall", {}), args.output_manifest,
            model=args.model_name, dataset=args.dataset, split=args.split,
            samples=samples, images_rgb=images_rgb, pred_masks=pred_masks, gt_masks=gt_masks,
        )

    l4 = report["overall"]["layer4"]
    r3 = l4["R3"]
    r4 = l4["R4"]
    print("\n" + "=" * 60)
    print("LANE-MARKING MACHINE-READABILITY REPORT")
    print("=" * 60)
    print(f"  Samples evaluated : {report['num_samples']}")
    print(f"  Dataset / split   : {report['dataset']} / {report['split']}")
    print(f"  Model             : {report['model_name']}")
    print(f"  R1 readability    : {l4['R1']:.1f} / 100" if l4.get("R1") is not None else "  R1 readability    : unavailable")
    print(f"  R2 detectability  : {l4['R2']:.1f} / 100" if l4.get("R2") is not None else "  R2 detectability  : unavailable")
    print(f"  R3 bottleneck     : {r3.get('name')} (strength={r3.get('strength')})")
    print(f"  R4 class          : {r4.get('status')}")
    print(f"  Confidence D7     : {'available' if report['confidence_available'] else 'unavailable'}")
    print(f"Saved summary -> {out_path}")
    if args.per_image_output:
        print(f"Saved per-image JSONL -> {args.per_image_output}")
    if args.per_image_csv:
        print(f"Saved per-image CSV -> {args.per_image_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
