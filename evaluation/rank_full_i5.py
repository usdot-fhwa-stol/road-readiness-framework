"""Compute full-dataset canonical + prediction-guided I5 rankings.

Mirrors the structure of ``evaluation.rank_full_i1_i4``: reads a dataset's
universal manifest for ground truth and each model's existing per-image
result file for prediction mask paths, then computes I5 directly via
``analyze_lane_geometry_complexity`` (not the full ``build_metric_record``,
which would also recompute I1-I4/D-metrics we don't need here) for speed
across full datasets.

Canonical I5 uses each sample's own GT geometry (native lane_json where the
dataset provides it, e.g. CurveLanes/TuSimple, otherwise the GT mask).
Prediction-guided I5_pred uses only that model's own predicted mask -- never
GT -- exactly like I5_pred does inside build_metric_record.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from lane_eval.manifest import ManifestDataset
from evaluation.lane_continuity import (
    LaneContinuityConfig,
    extract_guidance_lanes_from_mask,
    lanes_from_lane_json,
    prepare_guidance_lanes,
)
from evaluation.lane_geometry_complexity import analyze_lane_geometry_complexity


def _resolve(p):
    if not p:
        return None
    q = Path(p)
    if q.exists():
        return q
    m = Path("/home/karthikab/Projects/road-readiness-framework")
    try:
        r = Path.cwd() / q.relative_to(m)
        return r if r.exists() else q
    except ValueError:
        return q


def _select_guidance(image_shape, mask, native_lanes=None, lane_json=None):
    cfg = LaneContinuityConfig()
    candidates = [(native_lanes or [], "native_polyline"), (lanes_from_lane_json(lane_json), "lane_json")]
    for candidate, source in candidates:
        if candidate and prepare_guidance_lanes(candidate, image_shape, config=cfg):
            return candidate, source
    mask_lanes = extract_guidance_lanes_from_mask(mask, config=cfg) if mask is not None else []
    if mask_lanes and prepare_guidance_lanes(mask_lanes, image_shape, config=cfg):
        return mask_lanes, "mask_grouped_centerline"
    return [], None


def _work(item):
    dataset, model, sid, imgp, predp, gt_mask_path, gt_lane_json = item
    im = cv2.imread(str(imgp), cv2.IMREAD_COLOR)
    if im is None:
        return {
            "dataset": dataset, "model": model, "sample_id": sid, "image_path": str(imgp),
            "I5": None, "I5_pred": None, "reason": "image_unavailable",
        }
    h, w = im.shape[:2]
    out = {"dataset": dataset, "model": model, "sample_id": sid, "image_path": str(imgp)}

    try:
        gt_mask = None
        if gt_mask_path:
            loaded = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            if loaded is not None:
                gt_mask = (loaded > 0).astype(np.uint8)
        guidance, src = _select_guidance((h, w), gt_mask, lane_json=gt_lane_json)
        if guidance:
            gt_res = analyze_lane_geometry_complexity(guidance, image_shape=(h, w), geometry_source="ground_truth")
            out.update({
                "I5": gt_res.get("alignment_complexity"),
                "I5_profile_class": gt_res.get("profile_class"),
                "I5_topology": gt_res.get("topology_complexity"),
                "I5_bendiness": gt_res.get("bendiness"),
                "I5_confidence": gt_res.get("complexity_confidence"),
                "I5_geometry_source": src,
                "I5_unavailable_reason": gt_res.get("unavailable_reason"),
            })
        else:
            out.update({"I5": None, "I5_unavailable_reason": "no_credible_gt_geometry"})
    except Exception as e:
        out.update({"I5": None, "I5_unavailable_reason": f"exception:{e.__class__.__name__}"})

    try:
        pm = cv2.imread(str(predp), cv2.IMREAD_GRAYSCALE) if predp else None
        pred_mask = (pm > 0).astype(np.uint8) if pm is not None else None
        guidance, src = _select_guidance((h, w), pred_mask)
        if guidance:
            pred_res = analyze_lane_geometry_complexity(
                guidance, image_shape=(h, w), geometry_source=src or "mask_derived"
            )
            out.update({
                "I5_pred": pred_res.get("alignment_complexity"),
                "I5_pred_profile_class": pred_res.get("profile_class"),
                "I5_pred_topology": pred_res.get("topology_complexity"),
                "I5_pred_bendiness": pred_res.get("bendiness"),
                "I5_pred_confidence": pred_res.get("complexity_confidence"),
                "I5_pred_geometry_source": src,
                "I5_pred_unavailable_reason": pred_res.get("unavailable_reason"),
            })
        else:
            out.update({"I5_pred": None, "I5_pred_unavailable_reason": "prediction_mask_or_geometry_unavailable"})
    except Exception as e:
        out.update({"I5_pred": None, "I5_pred_unavailable_reason": f"exception:{e.__class__.__name__}"})

    return out


def _load_prediction_sources(root: Path, dataset: str, models: list[str]) -> dict[str, list[dict]]:
    sources = {}
    for model in models:
        if model == "yolopx":
            json_path = root / "results/per_image" / f"yolopx_{dataset}_per_image.json"
            csv_path = root / "results/per_image" / f"yolopx_{dataset}_per_image.csv"
            if json_path.exists():
                sources[model] = json.loads(json_path.read_text())
            elif csv_path.exists():
                sources[model] = list(csv.DictReader(csv_path.open()))
        else:
            csv_path = Path("outputs/results/per_image_csv") / f"{model}_{dataset}_per_image_metrics.csv"
            if csv_path.exists():
                sources[model] = list(csv.DictReader(csv_path.open()))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/full_model_rankings"),
                     help="Where existing yolopx per-image results/predictions are READ from "
                          "(clrernet/other models are always read from outputs/results/per_image_csv/, "
                          "independent of --root).")
    ap.add_argument("--output-root", type=Path, default=None,
                     help="Where rankings_i5/ and i5_full_records.json are WRITTEN. "
                          "Defaults to --root (old behavior). Pass a fresh directory to keep "
                          "results from different runs separate without touching --root's data.")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--datasets", nargs="*", default=["bdd100k_lane", "culane", "curvelanes", "tusimple"])
    ap.add_argument("--models", nargs="*", default=["yolopx", "clrernet"])
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    output_root = args.output_root or args.root

    allout = []
    for ds in args.datasets:
        samples = {}
        for s in ManifestDataset(f"manifests/{ds}/manifest_{ds}.json").iter_samples():
            samples[str(s.image_id)] = {
                "image_path": s.image_path,
                "gt_mask_path": s.target.mask_path,
                "gt_lane_json": (s.target.meta or {}).get("lane_json"),
            }
        sources = _load_prediction_sources(args.root, ds, args.models)
        for model, rows in sources.items():
            if args.max_samples:
                rows = rows[: args.max_samples]
            items = []
            for row in rows:
                sid = str(row.get("image_id", row.get("sample_id", "")))
                s = samples.get(sid)
                if not s:
                    continue
                pp = (
                    (args.root / "predictions/yolopx" / ds / "masks" / f"{sid}.png")
                    if model == "yolopx"
                    else _resolve(row.get("pred_mask_path"))
                )
                items.append((ds, model, sid, s["image_path"], pp, s["gt_mask_path"], s["gt_lane_json"]))
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                out = list(ex.map(_work, items, chunksize=16))
            allout.extend(out)
            n_gt = sum(1 for r in out if r.get("I5") is not None)
            n_pred = sum(1 for r in out if r.get("I5_pred") is not None)
            print(f"{ds} {model}: {len(out)} images, I5(GT) valid={n_gt}, I5_pred valid={n_pred}", flush=True)

    outdir = output_root / "rankings_i5"
    outdir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for model in args.models:
            rows = [r for r in allout if r["dataset"] == ds and r["model"] == model]
            if not rows:
                continue
            for metric in ("I5", "I5_pred"):
                valid = sorted([r for r in rows if r.get(metric) is not None], key=lambda r: r[metric], reverse=True)
                d = outdir / ds / model
                d.mkdir(parents=True, exist_ok=True)
                for name, data in [("top50", valid[:50]), ("worst", list(reversed(valid[-50:])))]:
                    (d / f"{metric}_{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False))
            summary = {
                "dataset": ds, "model": model, "total": len(rows),
                "I5_valid": sum(r.get("I5") is not None for r in rows),
                "I5_pred_valid": sum(r.get("I5_pred") is not None for r in rows),
                "I5_profile_class_counts": {
                    c: sum(r.get("I5_profile_class") == c for r in rows)
                    for c in {r.get("I5_profile_class") for r in rows if r.get("I5_profile_class")}
                },
                "I5_pred_profile_class_counts": {
                    c: sum(r.get("I5_pred_profile_class") == c for r in rows)
                    for c in {r.get("I5_pred_profile_class") for r in rows if r.get("I5_pred_profile_class")}
                },
            }
            (outdir / ds / model / "summary.json").write_text(json.dumps(summary, indent=2))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "i5_full_records.json").write_text(json.dumps(allout, allow_nan=False))
    print("saved", len(allout))


if __name__ == "__main__":
    main()
