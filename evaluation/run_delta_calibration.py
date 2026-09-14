#!/usr/bin/env python3
"""Freeze the D3' tolerance delta on the calibration split (GPU-box driver).

Pipeline
--------
Inputs: the GT manifest, one or more cached prediction manifests (one per
processor, produced by run_lane_eval / run_clrernet), and the calibration-split
artifact from ``evaluation.calibration_split``. This driver:

  1. restricts to the calibration sample_ids only (never the eval set);
  2. for each (processor, calibration image) reduces prediction + GT to canonical
     centerline point sets (native polyline where available, else mask skeleton)
     exactly as ``evaluation.d_metrics.compute_d3_prime`` does, and computes the
     per-image gt->pred / pred->gt nearest-neighbour distance arrays;
  3. selects delta at the F1-vs-delta knee (cross-checked against the p90
     localization error) via ``evaluation.delta_calibration``;
  4. runs the bootstrap-stability gate (answers "was 5% enough?");
  5. writes the frozen config via ``evaluation.d3prime_config`` and a full
     diagnostics JSON (the F1-vs-delta curve, per-processor deltas, bootstrap).

It does NOT sweep detection thresholds tau: the cached prediction manifests were
produced at each processor's existing operating point, so the recorded
``operating_points`` are those the manifests already embed (documented, not
re-optimized). A true tau sweep needs raw per-pixel logits / per-curve scores at
multiple thresholds, which the cached masks do not carry; that is noted in the
output rather than faked. (Per-curve CLRerNet scores ARE preserved now — a future
extension can threshold on them; see docs/d_metrics_audit.md.)

Requires cv2 (mask skeletonization goes through marking_support.mask_to_points,
which needs scikit-image or cv2.ximgproc). Runs on the inference box, not the
minimal repo env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import calibration_split as cs
from evaluation import delta_calibration as dc
from evaluation import d3prime_config
from evaluation import marking_support
from evaluation.d_metrics import _centerline_points, lanes_from_lane_json
from lane_eval.manifest import ManifestDataset, PredictionManifestReader


def _per_image_distances(gt_manifest: str, pred_manifest: str, calib_ids: set,
                         step_px: float = 1.0) -> list[dict]:
    """Compute canonical gt->pred / pred->gt distance arrays for each calibration
    image, using the SAME centerline reduction as compute_d3_prime."""
    dataset = ManifestDataset(gt_manifest)
    preds = PredictionManifestReader(pred_manifest)
    out = []
    for i in range(len(dataset)):
        sample = dataset[i]
        sid = str(sample.image_id)
        if sid not in calib_ids:
            continue
        w, h = int(sample.width), int(sample.height)
        gt_lane_json = (sample.target.meta or {}).get("lane_json")
        gt_lanes = lanes_from_lane_json(gt_lane_json) or None
        prediction = preds.get(sid)
        pred_lanes = prediction.lanes if prediction is not None else None
        pred_mask = prediction.mask if prediction is not None else None
        gt_mask = sample.target.mask

        pred_xy, _ = _centerline_points(pred_mask, pred_lanes, step_px)
        gt_xy, _ = _centerline_points(gt_mask, gt_lanes, step_px)
        pred_c = marking_support.to_canonical(pred_xy, w, h)
        gt_c = marking_support.to_canonical(gt_xy, w, h)
        out.append({
            "sample_id": sid,
            "gt_to_pred": marking_support._min_distances(gt_c, pred_c),
            "pred_to_gt": marking_support._min_distances(pred_c, gt_c),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-manifest", required=True, help="Universal GT manifest JSON")
    ap.add_argument("--pred-manifest", action="append", required=True, metavar="MODEL=PATH",
                    help="Prediction manifest as model=path (repeatable, one per processor)")
    ap.add_argument("--calibration-split", required=True, help="Calibration-split artifact JSON")
    ap.add_argument("--out-config", default=d3prime_config.DEFAULT_CONFIG_PATH,
                    help="Where to write the frozen D3' config")
    ap.add_argument("--out-diagnostics", default="dataset/calibration/d3prime_calibration_diagnostics.json")
    ap.add_argument("--error-percentile", type=float, default=90.0)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--step-px", type=float, default=1.0)
    args = ap.parse_args(argv)

    calib_art = json.loads(Path(args.calibration_split).read_text(encoding="utf-8"))
    calib_ids = set(calib_art.get("calibration_sample_ids", []))
    if not calib_ids:
        raise SystemExit("calibration split has no sample ids")

    # Parse model=path prediction manifests.
    model_manifests = {}
    for spec in args.pred_manifest:
        if "=" not in spec:
            raise SystemExit(f"--pred-manifest must be model=path, got {spec!r}")
        model, path = spec.split("=", 1)
        model_manifests[model.strip()] = path.strip()

    grid = list(dc.DEFAULT_DELTA_GRID)

    per_model = {}
    pooled = []  # distances across all processors -> the shared delta
    for model, pred_path in model_manifests.items():
        dists = _per_image_distances(args.gt_manifest, pred_path, calib_ids, args.step_px)
        sel = dc.select_delta(dists, grid, args.error_percentile)
        per_model[model] = {
            "n_calibration_images_scored": len(dists),
            "delta_selected": sel["delta_selected"],
            "delta_knee": sel["delta_knee"],
            "selection_rule": sel["selection_rule"],
            f"delta_p{int(args.error_percentile)}": sel[f"delta_p{int(args.error_percentile)}"],
            "f1_curve": sel["f1_curve"],
        }
        pooled.extend(dists)

    shared = dc.select_delta(pooled, grid, args.error_percentile)
    stability = dc.bootstrap_delta_stability(
        pooled, grid, args.error_percentile, n_boot=args.n_boot, seed=args.seed
    )

    delta = shared["delta_selected"]
    if delta is None:
        raise SystemExit("delta selection failed (insufficient calibration evidence)")

    # Operating points: whatever produced the cached pred manifests (documented,
    # not swept). Recorded as None to signal "manifest default", with a note.
    operating_points = {model: None for model in model_manifests}

    frozen = d3prime_config.build_frozen_config(
        delta=delta,
        operating_points=operating_points,
        selection_rule=shared["selection_rule"],
        calibration_artifact=calib_art,
        stability={k: v for k, v in stability.items() if k != "deltas"},
        delta_grid=grid,
        notes=[
            "delta frozen on the held-out calibration split; do not re-tune on eval.",
            "delta is a fraction of the image diagonal (resolution-free, isotropic).",
            "operating_points=None means the cached prediction manifests' existing "
            "thresholds were used; a tau sweep needs raw logits/scores and was not "
            "performed here.",
            f"shared delta chosen by '{shared['selection_rule']}' across "
            f"{len(model_manifests)} processor(s) over {len(pooled)} image-scorings.",
        ],
    )

    Path(args.out_config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_config).write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    diagnostics = {
        "shared": {k: v for k, v in shared.items() if k != "f1_curve"} | {"f1_curve": shared["f1_curve"]},
        "bootstrap_stability": stability,
        "per_model": per_model,
        "calibration_split_fingerprint": calib_art.get("manifest_fingerprint_sha256"),
        "models": list(model_manifests),
    }
    Path(args.out_diagnostics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_diagnostics).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Frozen delta = {delta:.5f} (rule: {shared['selection_rule']}, "
          f"fraction of image diagonal)")
    print(f"Bootstrap stable within one grid step: "
          f"{stability.get('stable_within_one_grid_step')}")
    print(f"  bootstrap IQR: {stability.get('delta_bootstrap_iqr')}")
    for model, info in per_model.items():
        print(f"  {model:12s} delta={info['delta_selected']} "
              f"(knee={info['delta_knee']}, n={info['n_calibration_images_scored']})")
    print(f"Wrote frozen config -> {args.out_config}")
    print(f"Wrote diagnostics   -> {args.out_diagnostics}")
    if not stability.get("stable_within_one_grid_step"):
        print("WARNING: delta not stable at this calibration size; consider "
              "raising the calibration fraction (e.g. 10%).")


if __name__ == "__main__":
    main()
