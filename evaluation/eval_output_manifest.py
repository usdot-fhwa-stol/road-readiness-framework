#!/usr/bin/env python3
"""Build an EVALUATION OUTPUT MANIFEST: take the input manifest and add each
sample's computed metric values, then emit JSON + CSV + per-sample images.

Metric families (from evaluation.readiness_metrics per-image records + the
aggregate report):
  core : primary detection scalars  -> D3_iou / D3_f1 / D3_precision / D3_recall
  D    : detection metrics          -> D1..D8   (per image)
  I    : intrinsic image metrics    -> I1..I6   (per image)
  R    : readability / detectability -> R1..R4   (per image R1/R2; R3/R4 aggregate)
  C    : correlations               -> C1..C5   (DATASET-LEVEL only; goes in metadata)

Two ways to use it:
  * Integrated  - `run_readiness.py --output-manifest DIR` calls
    `write_output_manifest(...)` after it computes records (also saves overlays).
  * Standalone  - merge an existing per-image JSONL back onto a manifest:
      python3 -m evaluation.eval_output_manifest \
          --in-manifest dataset/manifests/manifest_all.json \
          --per-image outputs/readiness/per_image.jsonl \
          --report outputs/readiness/report.json \
          --out-dir outputs/readiness/output_manifest [--copy-images]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

CORE_MAP = {"D3_iou": "iou", "D3_f1": "f1", "D3_precision": "precision", "D3_recall": "recall"}


def _num(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "item"):          # numpy scalar
        try:
            return o.item()
        except Exception:
            return o
    return o


def bucket_record(rec: dict) -> dict:
    """Split one per-image record into {core, D, I, R} by key prefix."""
    core, D, I, R = {}, {}, {}, {}
    for k, name in CORE_MAP.items():
        if k in rec and rec[k] is not None:
            core[name] = rec[k]
    for k, v in rec.items():
        head, nxt = k[:1], k[1:2]
        if not nxt.isdigit():
            continue
        if head == "I":
            I[k] = v
        elif head == "D":
            D[k] = v
        elif head == "R":
            R[k] = v
    fam = {}
    if core:
        fam["core"] = core
    if D:
        fam["D"] = D
    if I:
        fam["I"] = I
    if R:
        fam["R"] = R
    return fam


def _rec_keys(rec):
    """Candidate join keys for a record, most-specific first."""
    out = []
    for k in ("sample_id", "image_id", "image_path"):
        v = rec.get(k)
        if v:
            out.append(str(v))
            out.append(os.path.basename(str(v)))
    return out


def _sample_keys(s):
    out = []
    for k in ("sample_id", "image_id", "image_path", "source_image_path"):
        v = s.get(k)
        if v:
            out.append(str(v))
            out.append(os.path.basename(str(v)))
    return out


def build_output_manifest(in_manifest: dict, records: list, overall: dict | None,
                          model: str | None = None, dataset: str | None = None,
                          split: str | None = None, metric_version=None) -> tuple[dict, dict]:
    """Merge per-image records onto the input manifest. Returns (manifest, summary)."""
    samples = in_manifest.get("samples", [])
    # index records by every candidate key
    by_key = {}
    for r in records:
        for k in _rec_keys(r):
            by_key.setdefault(k, r)

    matched, present = 0, {"core": 0, "D": 0, "I": 0, "R": 0}
    for i, s in enumerate(samples):
        rec = None
        for k in _sample_keys(s):
            if k in by_key:
                rec = by_key[k]
                break
        if rec is None and len(records) == len(samples):
            rec = records[i]                        # positional fallback
        if rec is None:
            s.setdefault("metrics", None)
            continue
        fam = bucket_record(rec)
        s["metrics"] = _json_safe(fam)
        matched += 1
        for f in present:
            if f in fam:
                present[f] += 1

    overall = overall or {}
    families_present = [f for f in ("core", "D", "I", "R") if present.get(f)]
    aggregate = {
        "I": _json_safe(overall.get("layer1_avg")),
        "D": _json_safe(overall.get("layer2")),
        "R": _json_safe(overall.get("layer4")),
    }
    correlations_C = _json_safe(overall.get("layer3"))    # dataset-level C1..C5
    core_agg = None
    l2 = overall.get("layer2") or {}
    if l2:
        core_agg = _json_safe({v: l2.get(k) for k, v in CORE_MAP.items() if k in l2})

    meta = in_manifest.setdefault("metadata", {})
    meta["evaluation"] = _json_safe({
        "model": model, "dataset": dataset, "split": split,
        "metric_version": metric_version or overall.get("metric_version"),
        "num_samples": len(samples), "num_scored": matched,
        "families_present_per_sample": families_present,
        "families_absent_per_sample": [f for f in ("core", "D", "I", "R") if f not in families_present],
        "C_is_dataset_level_only": True,
        "aggregate": {**aggregate, "core": core_agg},
        "correlations_C": correlations_C,
        "note": "Per-sample metrics under sample.metrics.{core,D,I,R}; C1-C5 are "
                "dataset-level correlations (metadata.evaluation.correlations_C).",
    })
    summary = {
        "num_samples": len(samples), "num_scored": matched,
        "per_sample_present": {f: present[f] for f in present},
        "C_present": bool(correlations_C),
        "R_aggregate_present": bool(aggregate["R"]),
    }
    return in_manifest, summary


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def to_csv_rows(manifest: dict) -> list[dict]:
    rows = []
    for s in manifest.get("samples", []):
        row = {"sample_id": s.get("sample_id"), "dataset": s.get("dataset"),
               "split": s.get("split"), "image_path": s.get("image_path")}
        m = s.get("metrics") or {}
        for fam in ("core", "D", "I", "R"):
            for k, v in (m.get(fam) or {}).items():
                if _num(v) or v is None or isinstance(v, str):
                    row[k if fam != "core" else f"core_{k}"] = v
        rows.append(row)
    return rows


def write_csv(manifest: dict, path: str):
    rows = to_csv_rows(manifest)
    cols = ["sample_id", "dataset", "split", "image_path"]
    cols += sorted({k for r in rows for k in r if k not in cols})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def save_overlays(out_dir: str, samples: list, records: list, images_rgb, pred_masks,
                  gt_masks, max_side: int = 1280) -> int:
    """Save one pred-vs-GT overlay per sample; set sample.metrics.overlay_image."""
    import cv2
    import numpy as np
    from evaluation.visualize_failures import make_overlay, _resize_if_needed
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, s in enumerate(samples):
        if i >= len(images_rgb):
            break
        try:
            ov = make_overlay(images_rgb[i], gt_masks[i], pred_masks[i])
            ov = _resize_if_needed(ov, max_side)
            name = (s.get("sample_id") or f"img_{i}").replace("/", "_") + ".jpg"
            cv2.imwrite(str(img_dir / name), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            if isinstance(s.get("metrics"), dict):
                s["metrics"]["overlay_image"] = f"images/{name}"
            n += 1
        except Exception:
            continue
    return n


def copy_source_images(out_dir: str, samples: list, dataset_root: str | None = None) -> int:
    """Mode-A fallback: copy each sample's source image into out_dir/images/."""
    import shutil
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, s in enumerate(samples):
        src = None
        for cand in (s.get("image_path"), s.get("source_image_path")):
            if not cand:
                continue
            p = cand if os.path.isabs(cand) else os.path.join(dataset_root or ".", cand)
            if os.path.exists(p):
                src = p
                break
        if not src:
            continue
        name = (s.get("sample_id") or f"img_{i}").replace("/", "_") + os.path.splitext(src)[1]
        try:
            shutil.copy2(src, img_dir / name)
            if isinstance(s.get("metrics"), dict):
                s["metrics"]["image"] = f"images/{name}"
            n += 1
        except Exception:
            continue
    return n


def write_all(manifest: dict, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "output_manifest.json"
    with open(mpath, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2)
    cpath = write_csv(manifest, str(out / "output_metrics.csv"))
    return {"manifest": str(mpath), "csv": cpath}


# --------------------------------------------------------------------------- #
# Integrated entry (called by run_readiness with in-memory records + arrays)
# --------------------------------------------------------------------------- #
def manifest_from_records(records: list) -> dict:
    """Minimal manifest when no input manifest file is available (e.g. dry-run)."""
    return {"metadata": {"source": "synthesized from per-image records"},
            "samples": [{"sample_id": r.get("sample_id") or r.get("image_id") or f"img_{i}",
                         "image_id": r.get("image_id"), "image_path": r.get("image_path"),
                         "dataset": r.get("dataset"), "split": r.get("split")}
                        for i, r in enumerate(records)]}


def write_output_manifest(in_manifest_path, records: list, overall: dict, out_dir: str,
                          model=None, dataset=None, split=None, samples=None,
                          images_rgb=None, pred_masks=None, gt_masks=None) -> dict:
    if in_manifest_path and os.path.exists(in_manifest_path):
        with open(in_manifest_path) as f:
            man = json.load(f)
    else:
        man = manifest_from_records(records)
    man, summary = build_output_manifest(man, records, overall, model=model,
                                         dataset=dataset, split=split)
    imgs = 0
    if images_rgb is not None and pred_masks is not None and gt_masks is not None:
        imgs = save_overlays(out_dir, man.get("samples", []), records,
                             images_rgb, pred_masks, gt_masks)
    paths = write_all(man, out_dir)
    print(_report_text(summary, imgs, paths))
    return {**summary, "images_saved": imgs, **paths}


def _report_text(summary, imgs, paths):
    fp = summary["per_sample_present"]
    lines = ["", "Output manifest written:",
             f"  scored {summary['num_scored']}/{summary['num_samples']} samples",
             "  per-sample metric families:"]
    for f in ("core", "D", "I", "R"):
        got = fp.get(f, 0)
        lines.append(f"     {f:5} {'ADDED' if got else 'absent':7} ({got} samples)")
    lines.append(f"     C     {'ADDED (dataset-level, in metadata)' if summary['C_present'] else 'absent'}")
    lines.append(f"  overlay images: {imgs}")
    lines.append(f"  -> {paths['manifest']}")
    lines.append(f"  -> {paths['csv']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Standalone CLI (merge an existing per-image JSONL onto a manifest)
# --------------------------------------------------------------------------- #
def _load_records(path: str) -> list:
    recs = []
    with open(path) as f:
        if path.endswith(".csv"):
            recs = list(csv.DictReader(f))
        else:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-manifest", required=True)
    ap.add_argument("--per-image", required=True, help="per-image JSONL (or CSV) from run_readiness")
    ap.add_argument("--report", default=None, help="aggregate report.json (for C/R aggregate)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy each sample's source image into out-dir/images/")
    ap.add_argument("--dataset-root", default=None, help="root to resolve relative image_path")
    args = ap.parse_args()

    with open(args.in_manifest) as f:
        man = json.load(f)
    records = _load_records(args.per_image)
    overall = None
    if args.report:
        rep = json.load(open(args.report))
        overall = rep.get("overall", rep)
    man, summary = build_output_manifest(man, records, overall, model=args.model)
    imgs = 0
    if args.copy_images:
        root = args.dataset_root or os.path.dirname(os.path.dirname(os.path.abspath(args.in_manifest)))
        imgs = copy_source_images(args.out_dir, man.get("samples", []), root)
    paths = write_all(man, args.out_dir)
    print(_report_text(summary, imgs, paths))


if __name__ == "__main__":
    main()
