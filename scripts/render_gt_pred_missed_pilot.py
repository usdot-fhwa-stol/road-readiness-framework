"""Pilot run of the GT/predicted/missed D+R metric panels (see
evaluation/render_gt_pred_missed_panels.py) -- a small N-images-per-dataset
sample so the color/layout choice can be reviewed before rendering the full
dataset. Reuses already-computed D-metric per-image records
(outputs/d_metrics_full/d_metrics_v2) and the R2 CSV
(pilot_output/combined_i_and_d_metrics_with_r2.csv) -- no D-metric
recomputation, only image I/O + drawing.

Usage (from repo root):
    PYTHONPATH=. python3 scripts/render_gt_pred_missed_pilot.py \
        [--model yolopx] [--n-per-dataset 5] \
        [--out-root outputs/dr_metrics_panels]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, ".")
from lane_eval.manifest import ManifestDataset, PredictionManifestReader  # noqa: E402
from evaluation.d_metrics import lanes_from_lane_json  # noqa: E402
from evaluation.render_gt_pred_missed_panels import (  # noqa: E402
    render_raw, render_pred, render_d1, render_d2, render_d3, render_d4,
    render_d5, render_d6, render_r2,
)

GT_MANIFEST = {
    "culane": "dataset/manifests/manifest_culane.json",
    "tusimple": "dataset/manifests/manifest_tusimple_with_masks.json",
    "curvelanes": "dataset/manifests/manifest_curvelanes_with_masks.json",
    "bdd100k": "dataset/manifests/manifest_bdd100k.json",
}
R2_CSV = "pilot_output/combined_i_and_d_metrics_with_r2.csv"
DATASET_DIR = Path("dataset")


def _resolve_image_path(image_path: str) -> Path:
    """GT manifests under dataset/manifests/ store image paths relative to
    dataset/ (e.g. 'bdd100k/images/foo.jpg'), not repo root."""
    p = Path(image_path)
    if p.is_absolute() or p.exists():
        return p
    return DATASET_DIR / p


def load_records(jsonl_path):
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["sample_id"]] = rec
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolopx", choices=["yolopx", "clrernet"])
    ap.add_argument("--datasets", nargs="+", default=list(GT_MANIFEST.keys()))
    ap.add_argument("--n-per-dataset", type=int, default=5)
    ap.add_argument("--out-root", default="outputs/dr_metrics_panels")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    r2_df = pd.read_csv(R2_CSV).set_index("sample_id")

    for dataset in args.datasets:
        pred_manifest_path = f"outputs/d_metrics_full/pred/{args.model}/{dataset}_pred.json"
        jsonl_path = f"outputs/d_metrics_full/d_metrics_v2/{args.model}_{dataset}/d_metrics_per_image.jsonl"
        if not Path(jsonl_path).exists():
            print(f"SKIP {dataset}: no per-image records at {jsonl_path}")
            continue

        gt_dataset = ManifestDataset(GT_MANIFEST[dataset])
        predictions = PredictionManifestReader(pred_manifest_path)
        records = load_records(jsonl_path)

        picked = 0
        written = {"RAW": 0, "PRED": 0, "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0, "D6": 0, "R2": 0}
        tag = f"{args.model}_{dataset}"
        for idx in range(len(gt_dataset)):
            if picked >= args.n_per_dataset:
                break
            sample = gt_dataset[idx]
            sid = sample.image_id
            record = records.get(sid)
            if record is None or record.get("D3_iou") is None:
                continue
            gt_mask = sample.target.mask
            prediction = predictions.get(sid)
            pred_mask = prediction.mask if prediction else None
            if gt_mask is None or pred_mask is None:
                continue
            image_bgr = cv2.imread(str(_resolve_image_path(sample.image_path)))
            if image_bgr is None:
                continue

            pred_lanes = prediction.lanes if prediction else None
            gt_lane_json = (sample.target.meta or {}).get("lane_json")
            gt_lanes = lanes_from_lane_json(gt_lane_json) if gt_lane_json else None

            render_raw(out_root / "RAW" / tag / f"{sid}.png", image_bgr); written["RAW"] += 1
            render_pred(out_root / "PRED" / tag / f"{sid}.png", image_bgr, pred_mask); written["PRED"] += 1
            render_d1(out_root / "D1" / tag / f"{sid}.png", image_bgr, pred_mask, record, dataset, sid); written["D1"] += 1
            render_d2(out_root / "D2" / tag / f"{sid}.png", image_bgr, gt_lanes, pred_lanes, record, dataset, sid,
                       gt_mask=gt_mask, pred_mask=pred_mask); written["D2"] += 1
            render_d3(out_root / "D3" / tag / f"{sid}.png", image_bgr, gt_mask, pred_mask, record, dataset, sid); written["D3"] += 1
            render_d4(out_root / "D4" / tag / f"{sid}.png", image_bgr, gt_mask, pred_mask, record, dataset, sid); written["D4"] += 1
            render_d5(out_root / "D5" / tag / f"{sid}.png", image_bgr, gt_mask, pred_mask, record, dataset, sid); written["D5"] += 1
            render_d6(out_root / "D6" / tag / f"{sid}.png", image_bgr, gt_mask, pred_mask, record, dataset, sid); written["D6"] += 1
            if sid in r2_df.index:
                render_r2(out_root / "R2" / tag / f"{sid}.png", image_bgr, gt_mask, pred_mask,
                          r2_df.loc[sid].to_dict(), args.model, dataset, sid)
                written["R2"] += 1
            picked += 1

        print(f"{dataset}: wrote {written} (picked {picked}/{args.n_per_dataset} requested) -> {out_root}/*/{tag}/")


if __name__ == "__main__":
    main()
