"""Render one [raw | pred-vs-GT | I1-highlight] panel per sample.

Reuses the exact same GT/prediction pairing as evaluation.run_readiness
(``_run_manifest_predictions``) and the exact same metric computation
(``build_metric_record``) that produces the reported I1/I1_pred numbers --
this script only adds visualization on top, no new metric logic.

Usage:
    python -m evaluation.render_i1_overlay_panels \\
        --manifest categorized_manifest/output/universal_manifest_curvelane_tagged.json \\
        --pred-manifest outputs/tagged_subset_eval/pred/yolopx_curvelane_tagged_pred.json \\
        --model-name yolopx --dataset curvelane_tagged \\
        --output-dir outputs/tagged_subset_eval/i1_panels/yolopx_curvelane_tagged
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from evaluation.readiness_metrics import build_metric_record
from evaluation.run_readiness import _run_manifest_predictions
from evaluation.visualize_failures import save_i1_overlay_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="GT universal manifest JSON")
    ap.add_argument("--pred-manifest", required=True, help="Prediction manifest JSON")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=640)
    args = ap.parse_args()

    samples, pred_masks, gt_masks, images_rgb, lane_probs, prediction_inputs = _run_manifest_predictions(
        args.manifest, args.pred_manifest, max_samples=args.max_samples
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saved, n_unavailable = 0, 0
    iterator = zip(samples, pred_masks, gt_masks, images_rgb, lane_probs, prediction_inputs)
    for sample, pred, gt, img, prob, prediction_input in tqdm(iterator, total=len(samples)):
        record = build_metric_record(
            sample,
            pred,
            img,
            dataset=args.dataset,
            split=args.split,
            model_name=args.model_name,
            lane_prob=prob,
            pred_lanes=prediction_input.get("lanes"),
            pred_lane_json=prediction_input.get("lane_json"),
            pred_geometry_source=prediction_input.get("geometry_source"),
            i1_debug=True,
        )
        image_id = str(getattr(sample, "image_id", "sample")).replace("/", "_").replace("\\", "_")
        save_i1_overlay_panel(
            img, gt, pred, record, out_dir / f"{image_id}.jpg", max_side=args.max_side
        )
        n_saved += 1
        if record.get("I1") is None:
            n_unavailable += 1

    print(f"Saved {n_saved} I1 overlay panels -> {out_dir} ({n_unavailable} with I1 unavailable)")


if __name__ == "__main__":
    main()
