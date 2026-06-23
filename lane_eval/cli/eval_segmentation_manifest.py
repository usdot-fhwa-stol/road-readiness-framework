from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _load_binary_mask(path: str):
    if not path:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def _scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    lane_iou = tp / (tp + fp + fn + 1e-12)
    bg_iou = tn / (tn + fp + fn + 1e-12)
    return {
        "lane_iou": float(lane_iou),
        "lane_f1": float(f1),
        "lane_precision": float(precision),
        "lane_recall": float(recall),
        "lane_accuracy": float(recall),
        "lane_miou": float((lane_iou + bg_iou) / 2),
        "pixel_accuracy": float((tp + tn) / (tp + fp + fn + tn + 1e-12)),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Score prediction manifest masks against GT manifest masks."
    )
    p.add_argument("--gt-manifest", required=True)
    p.add_argument("--pred-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--yolopx-repo", default=None, help="Accepted for compatibility; not required")
    p.add_argument("--model-name", default="clrernet")
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--no-per-image",
        action="store_true",
        help="Do not include per-image metrics in the output JSON",
    )
    return p.parse_args()


def main():
    args = parse_args()

    gt_data = json.loads(Path(args.gt_manifest).expanduser().read_text())
    pred_data = json.loads(Path(args.pred_manifest).expanduser().read_text())

    gt_by_id = {s["sample_id"]: s for s in gt_data["samples"]}
    pred_samples = pred_data["samples"]

    dataset = (
        args.dataset
        or pred_data.get("metadata", {}).get("dataset")
        or gt_data.get("metadata", {}).get("dataset")
    )
    model = args.model_name or pred_data.get("metadata", {}).get("model", "model")

    total_tp = total_fp = total_fn = total_tn = 0
    skipped = 0
    processed = 0
    per_image = []

    for pred_sample in tqdm(pred_samples, desc=f"Scoring {model} on {dataset}"):
        sample_id = pred_sample["sample_id"]
        gt_sample = gt_by_id.get(sample_id)
        if gt_sample is None:
            skipped += 1
            continue

        gt_mask_path = gt_sample.get("ground_truth", {}).get("mask_path")
        pred_mask_path = pred_sample.get("prediction", {}).get("mask_path")

        gt_mask = _load_binary_mask(gt_mask_path)
        pred_mask = _load_binary_mask(pred_mask_path)

        if gt_mask is None or pred_mask is None:
            skipped += 1
            continue

        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(
                pred_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        pred = pred_mask > 0
        gt = gt_mask > 0

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        processed += 1

        if not args.no_per_image:
            rec = {
                "sample_id": sample_id,
                "image_path": pred_sample.get("image_path") or gt_sample.get("image_path"),
                "gt_mask_path": gt_mask_path,
                "pred_mask_path": pred_mask_path,
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "tn_pixels": tn,
            }
            rec.update(_scores(tp, fp, fn, tn))
            per_image.append(rec)

    results = {
        "model": model,
        "dataset": dataset,
        "task": "lane",
        "split": args.split,
        "num_images": processed,
        "skipped": skipped,
        "threshold": "mask",
        "metrics": _scores(total_tp, total_fp, total_fn, total_tn),
        "pixel_counts": {
            "tp_pixels": total_tp,
            "fp_pixels": total_fp,
            "fn_pixels": total_fn,
            "tn_pixels": total_tn,
        },
    }

    if not args.no_per_image:
        results["per_image_count"] = len(per_image)
        results["per_image"] = per_image

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_image"}, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
