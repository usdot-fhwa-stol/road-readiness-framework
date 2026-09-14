"""Render D2/D4/D5/D6 visualization panels (I-metrics house style, see
evaluation/render_i_style_d_panels.py) for a dataset x model combo.

Reuses already-computed GT/pred masks and the per-image D-metric JSONL
(outputs/d_metrics_full/d_metrics_v2/<model>_<ds>/d_metrics_per_image.jsonl)
instead of recomputing D-metrics -- this only does image I/O + drawing.

Run with CWD=dataset/ and PYTHONPATH=<repo root> (same convention as the
run_d_metrics.py invocations), e.g.:
    cd dataset && PYTHONPATH=$REPO_ROOT python3 \
        ../scripts/render_i_style_d_metrics.py --model yolopx --dataset culane \
        [--limit 5] [--out-root ../outputs/d_metrics_full/i_style_panels]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "..")
from lane_eval.manifest import ManifestDataset, PredictionManifestReader  # noqa: E402
from evaluation.d_metrics import lanes_from_lane_json  # noqa: E402
from evaluation.render_i_style_d_panels import render_d2, render_d4, render_d5, render_d6  # noqa: E402

GT_MANIFEST = {
    "culane": "manifests/manifest_culane.json",
    "tusimple": "manifests/manifest_tusimple_with_masks.json",
    "curvelanes": "manifests/manifest_curvelanes_with_masks.json",
    "bdd100k": "manifests/manifest_bdd100k.json",
}


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
    ap.add_argument("--model", required=True, choices=["yolopx", "clrernet"])
    ap.add_argument("--dataset", required=True, choices=list(GT_MANIFEST.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default="../outputs/d_metrics_full/i_style_panels")
    ap.add_argument("--repo-root", default="..")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    pred_manifest_path = repo_root / f"outputs/d_metrics_full/pred/{args.model}/{args.dataset}_pred.json"
    jsonl_path = repo_root / f"outputs/d_metrics_full/d_metrics_v2/{args.model}_{args.dataset}/d_metrics_per_image.jsonl"
    out_root = Path(args.out_root)

    gt_dataset = ManifestDataset(GT_MANIFEST[args.dataset])
    predictions = PredictionManifestReader(str(pred_manifest_path))
    records = load_records(jsonl_path)

    total = len(gt_dataset) if args.limit is None else min(args.limit, len(gt_dataset))
    written = {"D2": 0, "D4": 0, "D5": 0, "D6": 0}
    for idx in range(total):
        sample = gt_dataset[idx]
        sid = sample.image_id
        record = records.get(sid)
        if record is None:
            print(f"WARN: no D-metric record for {sid}; skipping")
            continue
        image_bgr = cv2.imread(str(sample.image_path))
        if image_bgr is None:
            print(f"WARN: unreadable image for {sid}; skipping")
            continue
        gt_mask = sample.target.mask
        prediction = predictions.get(sid)
        pred_mask = prediction.mask if prediction else None
        pred_lanes = prediction.lanes if prediction else None
        gt_lane_json = (sample.target.meta or {}).get("lane_json")
        gt_lanes = lanes_from_lane_json(gt_lane_json) if gt_lane_json else None

        if gt_mask is None or pred_mask is None:
            # D4/D5/D6 need pixel masks; D2 only needs lane instances.
            pass
        else:
            render_d4(out_root / "D4" / f"{args.model}_{args.dataset}" / f"{sid}.jpg",
                       image_bgr, gt_mask, pred_mask, record, args.dataset, sid)
            written["D4"] += 1
            render_d5(out_root / "D5" / f"{args.model}_{args.dataset}" / f"{sid}.jpg",
                       image_bgr, gt_mask, pred_mask, record, args.dataset, sid)
            written["D5"] += 1
            render_d6(out_root / "D6" / f"{args.model}_{args.dataset}" / f"{sid}.jpg",
                       image_bgr, gt_mask, pred_mask, record, args.dataset, sid)
            written["D6"] += 1

        render_d2(out_root / "D2" / f"{args.model}_{args.dataset}" / f"{sid}.jpg",
                   image_bgr, gt_lanes, pred_lanes, record, args.dataset, sid,
                   gt_mask=gt_mask, pred_mask=pred_mask)
        written["D2"] += 1

    print(f"{args.model}/{args.dataset}: wrote {written}")


if __name__ == "__main__":
    main()
