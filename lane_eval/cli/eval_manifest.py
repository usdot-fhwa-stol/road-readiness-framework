#!/usr/bin/env python3
"""Score a prediction manifest against a GT manifest with native lane metrics.

Pairs samples by sample_id and runs the dataset-appropriate native evaluator:
  * tusimple              -> TuSimple accuracy / FP / FN
  * culane / curvelanes   -> CULane line-IoU F1 (precision / recall / F1)

The GT lane_json comes from manifest_<dataset>.json (built by build_manifest);
the predicted lane_json comes from run_lane_eval's --pred-manifest output.

Example:
    python -m lane_eval.cli.eval_manifest \
        --gt-manifest   manifests/tusimple/manifest_tusimple.json \
        --pred-manifest outputs/pred/yolopx_tusimple_pred.json \
        --output        outputs/results/yolopx_tusimple_native.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_metric(metric: str, gt_meta: dict) -> str:
    if metric != "auto":
        return metric
    dataset = (gt_meta.get("metadata", {}).get("dataset") or "").lower()
    return "tusimple" if dataset == "tusimple" else "culane"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-manifest", required=True, help="GT manifest_<dataset>.json")
    p.add_argument("--pred-manifest", required=True, help="Prediction manifest from run_lane_eval")
    p.add_argument("--metric", default="auto", choices=["auto", "tusimple", "culane"],
                   help="Native metric to use (auto = infer from GT dataset name)")
    p.add_argument("--output", default=None, help="Optional JSON path to write results")
    args = p.parse_args()

    gt_doc = _load(args.gt_manifest)
    pred_doc = _load(args.pred_manifest)
    metric = _resolve_metric(args.metric, gt_doc)

    gt_by_id = {s["sample_id"]: s for s in gt_doc["samples"]}
    pred_by_id = {s["sample_id"]: s for s in pred_doc["samples"]}
    common = [sid for sid in pred_by_id if sid in gt_by_id]
    missing = len(pred_by_id) - len(common)

    if metric == "tusimple":
        from lane_eval.evaluators import TuSimpleNativeEvaluator
        ev = TuSimpleNativeEvaluator()
        for sid in common:
            ev.update(pred_by_id[sid]["prediction"]["lane_json"],
                      gt_by_id[sid]["ground_truth"]["lane_json"])
    else:
        from lane_eval.evaluators import CULaneNativeEvaluator
        ev = CULaneNativeEvaluator()
        for sid in common:
            g = gt_by_id[sid]
            ev.update(pred_by_id[sid]["prediction"]["lane_json"],
                      g["ground_truth"]["lane_json"],
                      height=g["height"], width=g["width"])

    results = {
        "metric": metric,
        "dataset": gt_doc.get("metadata", {}).get("dataset"),
        "model": pred_doc.get("metadata", {}).get("model"),
        "num_evaluated": len(common),
        "num_pred_without_gt": missing,
        "metrics": ev.compute(),
    }

    print(json.dumps(results, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
