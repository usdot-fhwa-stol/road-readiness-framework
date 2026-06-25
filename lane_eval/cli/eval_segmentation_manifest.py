#!/usr/bin/env python3
"""Score a prediction manifest's lane masks against the GT manifest's masks.

Stage 2 of the CLRerNet flow (also usable for any model that emits a prediction
manifest in a foreign env). CLRerNet runs inference in its own mmdet venv and
only writes predicted lane-mask PNGs; this stage runs in the normal repo env
and turns those masks into the SAME outputs the YOLOPX / HybridNets runners
produce:

  * a summary JSON (IoU / F1 / Precision / Recall + miou / pixel_acc), and
  * optional per-image JSON + CSV.

Both GT and prediction masks are scored with YOLOPX's ``SegmentationMetric`` —
the exact metric used for every other model — so the comparison stays
apples-to-apples (no new metric math here).

Pairs samples by sample_id (the prediction manifest is generated from the GT
manifest, so ids match exactly).

Example:
    python -m lane_eval.cli.eval_segmentation_manifest \
        --gt-manifest   manifests/culane/manifest_culane.json \
        --pred-manifest outputs_cs/clrernet/predictions/culane/predictions.json \
        --yolopx-repo   /home/gauravb/Projects/road_readiness_t3/YOLOPX \
        --model-name clrernet --dataset culane --split test --per-image \
        --output outputs_cs/clrernet/results/clrernet_culane.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _add_path(p: str) -> None:
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _load_binary_mask(path):
    if not path:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def _dilate(mask: np.ndarray, tau: int) -> np.ndarray:
    """Widen a lane mask by a disk of radius ``tau`` px.

    Pixel-IoU between a thin lane *line* and a thick GT seg label is dominated by
    the stroke-WIDTH mismatch, not by where the line actually is: a perfectly
    placed thin prediction over a ~9px CULane/TuSimple GT scores low recall,
    while a fat prediction over a ~2.6px BDD GT scores low precision. Dilating
    BOTH masks by the same tolerance before scoring (a "relaxed"/buffered IoU)
    normalises the widths so the metric reflects geometric agreement within
    ``tau`` px. Scoring itself is still the shared YOLOPX SegmentationMetric.
    """
    if tau <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tau + 1, 2 * tau + 1))
    return (cv2.dilate(mask, k) > 0).astype(np.uint8)


def run(args) -> dict:
    # Repo root (for evaluation.*) and YOLOPX repo (for SegmentationMetric).
    _add_path(str(Path(__file__).resolve().parents[2]))
    _add_path(args.yolopx_repo)
    from lib.core.evaluate import SegmentationMetric  # YOLOPX shared metric
    from evaluation.per_image_metrics import per_image_scores, per_image_paths, write_per_image

    gt_doc = json.loads(Path(args.gt_manifest).expanduser().read_text())
    pred_doc = json.loads(Path(args.pred_manifest).expanduser().read_text())

    gt_by_id = {s["sample_id"]: s for s in gt_doc["samples"]}
    pred_samples = pred_doc["samples"]
    if args.max_samples:
        pred_samples = pred_samples[: args.max_samples]

    dataset = (args.dataset or pred_doc.get("metadata", {}).get("dataset")
               or gt_doc.get("metadata", {}).get("dataset"))
    model = args.model_name or pred_doc.get("metadata", {}).get("model", "model")

    metric = SegmentationMetric(2)
    collect_per_image = bool(args.per_image or args.per_image_json or args.per_image_csv)
    per_records: list[dict] = []
    processed = skipped = 0

    for ps in tqdm(pred_samples, desc=f"Scoring {model} on {dataset}"):
        sid = ps["sample_id"]
        gs = gt_by_id.get(sid)
        if gs is None:
            skipped += 1
            continue
        gt_mask = _load_binary_mask(gs.get("ground_truth", {}).get("mask_path"))
        pred_mask = _load_binary_mask(ps.get("prediction", {}).get("mask_path"))
        if gt_mask is None or pred_mask is None:
            skipped += 1
            continue
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Normalise stroke width before scoring so a thin lane-line prediction is
        # compared to the thick GT seg label on geometry, not raster width.
        gt_eval = _dilate(gt_mask, args.tolerance_px)
        pred_eval = _dilate(pred_mask, args.tolerance_px)

        metric.addBatch(pred_eval.astype(np.int64), gt_eval.astype(np.int64))
        if collect_per_image:
            rec = {"model": model, "dataset": dataset, "split": args.split,
                   "image_id": sid, "image_path": ps.get("image_path") or gs.get("image_path")}
            rec.update(per_image_scores(pred_eval, gt_eval))
            per_records.append(rec)
        processed += 1

    if skipped:
        print(f"  Skipped {skipped} samples (no GT match or missing mask)")

    recall = float(metric.lineAccuracy())
    precision = float(metric.classPixelAccuracy()[1])
    f1 = float(2 * precision * recall / (precision + recall + 1e-12))
    results = {
        "model": model, "dataset": dataset, "task": "lane", "split": args.split,
        "num_images": processed, "skipped": skipped, "threshold": "mask",
        "tolerance_px": int(args.tolerance_px),
        "metrics": {
            "lane_iou": float(metric.IntersectionOverUnion()),
            "lane_f1": f1, "lane_precision": precision, "lane_recall": recall,
            "lane_accuracy": recall,
            "lane_miou": float(metric.meanIntersectionOverUnion()),
            "pixel_accuracy": float(metric.pixelAccuracy()),
        },
    }

    if collect_per_image:
        dj, dc = per_image_paths(args.output)
        pj = args.per_image_json or dj
        pc = args.per_image_csv or dc
        write_per_image(per_records, pj, pc)
        results["per_image_json"] = pj
        results["per_image_csv"] = pc
        results["per_image_count"] = len(per_records)
        print(f"  Saved per-image metrics -> {pj} (+ .csv), {len(per_records)} rows")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Score a prediction manifest's masks against GT manifest masks "
                                            "with the shared YOLOPX SegmentationMetric.")
    p.add_argument("--gt-manifest", required=True, help="GT manifest_<dataset>.json")
    p.add_argument("--pred-manifest", required=True, help="Prediction manifest (from run_clrernet)")
    p.add_argument("--yolopx-repo", required=True, help="Path to YOLOPX repo (for the shared SegmentationMetric)")
    p.add_argument("--model-name", default="clrernet")
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--tolerance-px", type=int, default=0,
                   help="Stroke-width tolerance (px). Dilates BOTH pred and GT masks by this "
                        "radius before scoring so a thin lane-LINE prediction is matched to the "
                        "thick GT seg label on geometry rather than raster width. 0 = raw IoU "
                        "(default, backward-compatible). ~8 suits the CULane/TuSimple ~16px labels.")
    p.add_argument("--output", required=True, help="Summary JSON path")
    p.add_argument("--per-image", action="store_true",
                   help="Also save per-image IoU/F1/Precision/Recall as JSON + CSV (per_image/ subdir)")
    p.add_argument("--per-image-json", default=None, help="Explicit per-image JSON path")
    p.add_argument("--per-image-csv", default=None, help="Explicit per-image CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    m = results["metrics"]
    print("\n=== CLRerNet (mask) Lane Evaluation ===")
    print(f"  model        : {results['model']}")
    print(f"  dataset      : {results['dataset']} ({results['split']})")
    print(f"  images       : {results['num_images']}  (skipped: {results['skipped']})")
    print(f"  tolerance_px : {results['tolerance_px']}  (0 = raw pixel IoU)")
    print(f"  lane_iou      : {m['lane_iou']:.4f}")
    print(f"  lane_f1       : {m['lane_f1']:.4f}")
    print(f"  lane_precision: {m['lane_precision']:.4f}")
    print(f"  lane_recall   : {m['lane_recall']:.4f}")
    print(f"\nSaved to: {out_path}")
