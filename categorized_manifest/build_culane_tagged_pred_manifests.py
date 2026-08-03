#!/usr/bin/env python3
"""Build prediction manifests for the CULane tagged subset from EXISTING
per-image results -- no new model inference. Reuses whatever mask files the
earlier full CULane yolopx/clrernet runs already wrote to disk; only
`build_tagged_subset_manifests.py`'s GT manifest needs to exist first.

Usage:
    python3 categorized_manifest/build_culane_tagged_pred_manifests.py \\
        --full-root outputs/full_model_rankings \\
        --output-dir outputs/tagged_subset_eval/pred
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _resolve(p, cwd_root: Path):
    if not p:
        return None
    q = Path(p)
    if q.exists():
        return q
    other_machine_root = Path("/home/karthikab/Projects/road-readiness-framework")
    try:
        r = cwd_root / q.relative_to(other_machine_root)
        return r if r.exists() else None
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-root", type=Path, default=Path("outputs/full_model_rankings"),
                     help="Root containing the existing full-dataset yolopx per-image results/masks.")
    ap.add_argument("--output-dir", type=Path, required=True,
                     help="Where to write the two prediction manifests (one per model).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sample_ids = json.loads((repo_root / "categorized_manifest/output/_culane_tagged_sample_ids.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # yolopx: existing per-image json + mask files under <full_root>/predictions/yolopx/culane/masks/
    yolopx_rows = json.loads((args.full_root / "results/per_image/yolopx_culane_per_image.json").read_text())
    yolopx_by_id = {r["image_id"]: r for r in yolopx_rows}
    yolopx_samples = []
    for sid in sample_ids:
        row = yolopx_by_id.get(sid)
        mask_path = args.full_root / "predictions/yolopx/culane/masks" / f"{sid}.png"
        if row is None or not mask_path.exists():
            print(f"WARN yolopx: no cached result/mask for {sid}")
            continue
        yolopx_samples.append({
            "sample_id": sid, "image_path": row["image_path"],
            "prediction": {"mask_path": str(mask_path), "lane_json": None, "geometry_source": "mask_derived"},
        })
    (args.output_dir / "yolopx_culane_tagged_pred.json").write_text(json.dumps(
        {"metadata": {"model": "yolopx", "dataset": "culane", "num_samples": len(yolopx_samples)},
         "samples": yolopx_samples}, indent=2))
    print(f"yolopx: {len(yolopx_samples)}/{len(sample_ids)} prediction masks found")

    # clrernet: existing per-image CSV has an absolute pred_mask_path column.
    clrernet_csv = Path("outputs/results/per_image_csv/clrernet_culane_per_image_metrics.csv")
    clrernet_rows = {r["sample_id"]: r for r in csv.DictReader(clrernet_csv.open())}
    clrernet_samples = []
    for sid in sample_ids:
        row = clrernet_rows.get(sid)
        mask_path = _resolve(row.get("pred_mask_path"), repo_root) if row else None
        if row is None or mask_path is None:
            print(f"WARN clrernet: no cached result/mask for {sid}")
            continue
        clrernet_samples.append({
            "sample_id": sid, "image_path": row["image_path"],
            "prediction": {"mask_path": str(mask_path), "lane_json": None, "geometry_source": "mask_derived"},
        })
    (args.output_dir / "clrernet_culane_tagged_pred.json").write_text(json.dumps(
        {"metadata": {"model": "clrernet", "dataset": "culane", "num_samples": len(clrernet_samples)},
         "samples": clrernet_samples}, indent=2))
    print(f"clrernet: {len(clrernet_samples)}/{len(sample_ids)} prediction masks found")


if __name__ == "__main__":
    main()
