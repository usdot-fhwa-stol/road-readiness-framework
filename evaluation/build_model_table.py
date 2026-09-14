#!/usr/bin/env python3
"""Aggregate per-(model, dataset) lane-eval JSONs into a model-comparison table.

Reads every *.json produced by ``lane_eval.cli.run_lane_eval`` in ``--results-dir``,
groups them by model, and emits two markdown tables:

  1. a raw per-(model, dataset) table (the backing data — up to 4 models x
     4 datasets = 16 rows), and
  2. a per-model summary (Table 12): mean IoU / F1 / Precision / Recall across
     that model's datasets, plus Cross-Dataset Consistency.

Cross-Dataset Consistency = 1 - (std / mean) of the primary metric across a
model's datasets (coefficient-of-variation based; 1.0 = identical everywhere),
clamped to [0, 1].

Output Usability and Preliminary Assessment are left as placeholders ('-') for
manual judgement after reviewing the numbers — they are not computable metrics.

All four headline metrics are computed inside this repo's harness (YOLOPX's
SegmentationMetric, applied identically to every model), never by any model's
native evaluation code, so the comparison stays fair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

METRIC_KEYS = ["lane_iou", "lane_f1", "lane_precision", "lane_recall"]
METRIC_LABELS = {
    "lane_iou": "IoU",
    "lane_f1": "F1",
    "lane_precision": "Precision",
    "lane_recall": "Recall",
}


def _load(results_dir: Path) -> list[dict]:
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue  # skip per-image arrays / non-summary JSONs
        m = d.get("metrics") or {}
        if "lane_iou" not in m:
            continue  # not a lane-eval result file
        rows.append({
            "model": d.get("model", jf.stem),
            "dataset": d.get("dataset", "?"),
            "split": d.get("split", "?"),
            "num_images": d.get("num_images", 0),
            **{k: m.get(k) for k in METRIC_KEYS},
        })
    return rows


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


def _consistency(values: list) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if len(vals) < 2:
        return None
    mu = mean(vals)
    if mu == 0:
        return None
    cv = pstdev(vals) / mu
    return max(0.0, min(1.0, 1.0 - cv))


def build_tables(rows: list[dict], primary: str = "lane_f1") -> str:
    rows = sorted(rows, key=lambda r: (r["model"], r["dataset"]))
    out: list[str] = []

    out.append("## Per-(model, dataset) results\n")
    out.append("| Model | Dataset | Split | Images | IoU | F1 | Precision | Recall |")
    out.append("|---|---|---|--:|--:|--:|--:|--:|")
    for r in rows:
        out.append(
            f"| {r['model']} | {r['dataset']} | {r['split']} | {r['num_images']} | "
            f"{_fmt(r['lane_iou'])} | {_fmt(r['lane_f1'])} | "
            f"{_fmt(r['lane_precision'])} | {_fmt(r['lane_recall'])} |"
        )

    models: dict[str, list[dict]] = {}
    for r in rows:
        models.setdefault(r["model"], []).append(r)

    out.append("\n## Table 12. Preliminary model comparison\n")
    out.append(
        "| Candidate Model | Mean IoU | Mean F1 | Mean Precision | Mean Recall | "
        "Cross-Dataset Consistency | Output Usability | Preliminary Assessment |"
    )
    out.append("|---|--:|--:|--:|--:|--:|---|---|")
    for model, mrows in sorted(models.items()):
        means = {}
        for k in METRIC_KEYS:
            vals = [r[k] for r in mrows if isinstance(r[k], (int, float))]
            means[k] = mean(vals) if vals else None
        cons = _consistency([r[primary] for r in mrows])
        out.append(
            f"| {model} | {_fmt(means['lane_iou'])} | {_fmt(means['lane_f1'])} | "
            f"{_fmt(means['lane_precision'])} | {_fmt(means['lane_recall'])} | "
            f"{_fmt(cons)} | - | - |"
        )

    out.append(
        f"\n_n_datasets per model varies; Mean = average across that model's datasets. "
        f"Cross-Dataset Consistency = 1 - (std/mean) of {METRIC_LABELS[primary]} across "
        f"those datasets (1.0 = identical across domains). Output Usability / Preliminary "
        f"Assessment left blank for manual judgement._"
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate lane-eval JSONs into a model-comparison table")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--out", default=None, help="Optional path to write the markdown table")
    ap.add_argument("--primary", default="lane_f1", choices=METRIC_KEYS,
                    help="Metric used for the Cross-Dataset Consistency column")
    args = ap.parse_args()

    rows = _load(Path(args.results_dir))
    if not rows:
        print(f"No lane-eval result JSONs found in {args.results_dir}")
        return

    table = build_tables(rows, primary=args.primary)
    print("\n" + table + "\n")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        print(f"Saved table -> {args.out}")


if __name__ == "__main__":
    main()
