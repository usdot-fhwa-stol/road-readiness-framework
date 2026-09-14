"""Full run of the GT/predicted/missed D+R metric panels (see
evaluation/render_gt_pred_missed_panels.py), restricted to exactly the image
set in pilot_output/combined_i_and_d_metrics_with_r2.csv -- not the larger
full-dataset manifests used elsewhere in this repo (e.g. the 34,680-frame
CULane manifest under manifests/culane/). That CSV's per-dataset row counts
(culane 2000, tusimple 2350, curvelanes 2300, bdd100k 2648) match
dataset/manifests/manifest_<ds>*.json exactly -- the same manifests the pilot
script already used -- so this is the same 4 manifests, just with no
per-dataset sample cap.

Usage (from repo root):
    PYTHONPATH=. python3 scripts/render_gt_pred_missed_full.py \
        [--model yolopx] [--out-root outputs/dr_metrics_panels]
"""
import argparse
import json
import sys
import time
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
    ap.add_argument("--out-root", default="outputs/dr_metrics_panels")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    df = pd.read_csv(R2_CSV)
    df = df[df[f"{args.model}_iou"].notna()]
    allowed_by_dataset = {
        ds: set(df.loc[df["dataset_name"] == ds, "sample_id"])
        for ds in args.datasets
    }
    for ds, ids in allowed_by_dataset.items():
        print(f"CSV allow-list: {ds} = {len(ids)} sample_ids")

    r2_df = pd.read_csv(R2_CSV).set_index("sample_id")

    grand_total_written = 0
    t_start = time.time()
    for dataset in args.datasets:
        allowed = allowed_by_dataset.get(dataset, set())
        if not allowed:
            print(f"SKIP {dataset}: no allowed sample_ids in CSV")
            continue
        pred_manifest_path = f"outputs/d_metrics_full/pred/{args.model}/{dataset}_pred.json"
        jsonl_path = f"outputs/d_metrics_full/d_metrics_v2/{args.model}_{dataset}/d_metrics_per_image.jsonl"
        if not Path(jsonl_path).exists():
            print(f"SKIP {dataset}: no per-image records at {jsonl_path}")
            continue

        gt_dataset = ManifestDataset(GT_MANIFEST[dataset])
        predictions = PredictionManifestReader(pred_manifest_path)
        records = load_records(jsonl_path)

        tag = f"{args.model}_{dataset}"
        written = {"RAW": 0, "PRED": 0, "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0, "D6": 0, "R2": 0}
        processed = 0
        skipped_not_allowed = 0
        skipped_no_data = 0
        t0 = time.time()

        for idx in range(len(gt_dataset)):
            sample = gt_dataset[idx]
            sid = sample.image_id
            if sid not in allowed:
                skipped_not_allowed += 1
                continue
            record = records.get(sid)
            if record is None or record.get("D3_iou") is None:
                skipped_no_data += 1
                continue
            gt_mask = sample.target.mask
            prediction = predictions.get(sid)
            pred_mask = prediction.mask if prediction else None
            if gt_mask is None or pred_mask is None:
                skipped_no_data += 1
                continue
            image_bgr = cv2.imread(str(_resolve_image_path(sample.image_path)))
            if image_bgr is None:
                skipped_no_data += 1
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
            processed += 1
            if processed % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                print(f"  {dataset}: {processed}/{len(allowed)} done "
                      f"({rate:.1f} img/s, {elapsed:.0f}s elapsed)", flush=True)

        grand_total_written += sum(written.values())
        print(f"{dataset}: processed {processed}/{len(allowed)} allowed "
              f"(skipped_not_allowed={skipped_not_allowed}, skipped_no_data={skipped_no_data}) "
              f"wrote {written} -> {out_root}/*/{tag}/", flush=True)

    print(f"DONE. total files written = {grand_total_written}, "
          f"elapsed = {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
