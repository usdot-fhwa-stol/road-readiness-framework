#!/usr/bin/env python3
"""Run the draft-aligned D1-D7 detection performance indicators on a tagged
subset (GT manifest + cached prediction manifest -- no model inference).

Usage:
    python -m evaluation.run_d_metrics \
        --manifest categorized_manifest/output/universal_manifest_culane_tagged.json \
        --pred-manifest outputs/tagged_subset_eval_v1/pred/yolopx_culane_tagged_pred.json \
        --model-name yolopx --output-dir outputs/d_metrics_tagged/yolopx_culane \
        [--render-panels]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.d_metrics import build_d_record, summarize_d_records
from lane_eval.manifest import ManifestDataset, PredictionManifestReader


def _raw_mask_paths(pred_manifest: str) -> dict[str, str]:
    data = json.loads(Path(pred_manifest).read_text())
    return {s["sample_id"]: (s.get("prediction") or {}).get("mask_path")
            for s in data.get("samples", [])}


def _fallback_pred_mask(mask_path: str | None) -> np.ndarray | None:
    """The cached pred manifests store repo-relative mask paths, which the
    reader resolves against the manifest dir and misses; retry against CWD."""
    if not mask_path or not Path(mask_path).exists():
        return None
    loaded = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return (loaded > 0).astype(np.uint8) if loaded is not None else None


def load_pairs(gt_manifest: str, pred_manifest: str):
    """Yield (sample, image_rgb, gt_mask, pred_mask, pred_lanes, lane_prob)."""
    dataset = ManifestDataset(gt_manifest)
    predictions = PredictionManifestReader(pred_manifest)
    raw_mask_paths = _raw_mask_paths(pred_manifest)
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = predictions.get(sample.image_id)
        if prediction is None:
            print(f"WARN: no prediction for {sample.image_id}; counted as no-output for D1")
            prediction = None
        image_bgr = cv2.imread(str(sample.image_path))
        if image_bgr is None:
            print(f"WARN: unreadable image {sample.image_path}; skipping")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        gt_mask = sample.target.mask
        if gt_mask is None:
            gt_mask = np.zeros((height, width), dtype=np.uint8)
        if prediction is not None:
            pred_mask = prediction.mask
            if pred_mask is None:
                pred_mask = _fallback_pred_mask(raw_mask_paths.get(sample.image_id))
            if pred_mask is None:
                print(f"WARN: prediction mask unreadable for {sample.image_id}; counted as no-output for D1")
                pred_mask = np.zeros((height, width), dtype=np.uint8)
            pred_lanes, lane_prob = prediction.lanes, prediction.prob
        else:
            pred_mask, pred_lanes, lane_prob = np.zeros((height, width), dtype=np.uint8), None, None
        yield sample, image_rgb, gt_mask, pred_mask, pred_lanes, lane_prob


def _visibility_tag(sample) -> str | None:
    tags = ((sample.meta or {}).get("tags") or {}).get("summary") or {}
    return tags.get("Observed Marking Visibility")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred-manifest", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--render-panels", action="store_true",
                    help="Also write per-image RAW/PRED/D1-D6 side-by-side panels")
    ap.add_argument("--panel-max-side", type=int, default=640)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = ManifestDataset(args.manifest).name

    records = []
    render_inputs = []
    for sample, image_rgb, gt_mask, pred_mask, pred_lanes, lane_prob in load_pairs(args.manifest, args.pred_manifest):
        record = build_d_record(
            sample_id=sample.image_id,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            gt_lane_json=(sample.target.meta or {}).get("lane_json"),
            pred_lanes=pred_lanes,
            lane_prob=lane_prob,
            visibility_tag=_visibility_tag(sample),
            image_path=str(sample.image_path),
        )
        record["dataset"] = dataset_name
        record["model"] = args.model_name
        records.append(record)
        if args.render_panels:
            from evaluation.d_metrics import standardize_stroke_width
            display_pred, _ = standardize_stroke_width(pred_mask, gt_mask)
            render_inputs.append((sample, image_rgb, gt_mask, display_pred, record))

    summary = summarize_d_records(records)
    summary["dataset"] = dataset_name
    summary["model"] = args.model_name

    (args.output_dir / "d_metrics_summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.output_dir / "d_metrics_per_image.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in records for k in r})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with open(args.output_dir / "d_metrics_per_image.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if args.render_panels:
        from evaluation.render_d_metric_panels import render_sample_panels
        panel_dir = args.output_dir / "panels"
        n_files = 0
        for sample, image_rgb, gt_mask, pred_mask, record in render_inputs:
            n_files += render_sample_panels(
                panel_dir, sample.image_id, image_rgb, gt_mask, pred_mask, record,
                group_summary=summary, max_side=args.panel_max_side,
                gt_lane_json=(sample.target.meta or {}).get("lane_json"),
            )
        print(f"Wrote {n_files} panel images -> {panel_dir}")

    print(f"\n=== D metrics: {args.model_name} / {dataset_name} "
          f"({summary['num_processed_images']} images, {summary['num_annotation_eligible']} eligible) ===")
    def fmt(v):
        return "unavailable" if v is None else f"{v:.4f}"
    print(f"  D1 detection success rate : {fmt(summary['D1_detection_success_rate'])}")
    print(f"  D2 lane count accuracy    : {fmt(summary['D2_lane_count_accuracy'])} (n={summary['D2_num_eligible']})")
    print(f"  D3 mean IoU               : {fmt(summary['D3_mean_iou'])} (median {fmt(summary['D3_iou_median'])})")
    print(f"  D4 near-field mean IoU    : {fmt(summary['D4_mean_near_field_iou'])}")
    d5s, d5v = summary["D5_occlusion_robustness_gap"], summary["D5_visibility_degraded_gap"]
    print(f"  D5 occlusion gap (strict) : {fmt(d5s['gap'])} (clear n={d5s['n_reference']}, occluded n={d5s['n_comparison']})")
    print(f"  D5 visibility gap (proxy) : {fmt(d5v['gap'])} (clear n={d5v['n_reference']}, degraded n={d5v['n_comparison']})")
    print(f"  D6 detection gap ratio    : {fmt(summary['D6_detection_gap_ratio'])}")
    print(f"  D7 confidence mean        : {fmt(summary['D7_confidence_mean'])}")
    print(f"Saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
