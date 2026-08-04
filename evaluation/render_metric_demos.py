"""Render the 5 pilot demo overlays for the minimal I1 / I4 / I5 metrics.

Each output image has the metric score burned in and the regions that drove the
score highlighted:

* ``i1_high_clear.jpg`` / ``i1_low_worn.jpg`` — new lane-wear I1
  (green = paint present, orange = faded, red = worn/broken/missing).
* ``i4_stable_width.jpg`` — minimal lane-width stability I4
  (green tick = width matches the linear perspective model, red = deviates).
* ``i5_low_straight.jpg`` / ``i5_high_curved.jpg`` — geometry-complexity I5
  (centerline coloured by local curvature: green = straight, red = tight).

Usage:
    python -m evaluation.render_metric_demos [--output-dir outputs/metric_demos]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.lane_geometry_complexity import analyze_lane_geometry_complexity
from evaluation.lane_wear import analyze_lane_wear, render_wear_overlay, select_gt_geometry
from evaluation.lane_width_simple import analyze_lane_width_simple, render_width_overlay

# Hand-picked representative frames from the human-tagged pilot subset.
DEMOS = {
    "ffdf59d5b8a53aac923f5a1498060a96": ("i1_high_clear", "i1"),
    # Tunnel with genuinely faded/worn lane lines (tag: Faded / Worn / Not
    # Visible) — paint is faintly present in the raw image and mostly red in the
    # overlay. Chosen over unmarked-street frames whose zigzag GT looks wrong.
    "ffe28390647ebe0ebd704148f1341b70": ("i1_low_worn", "i1"),
    "fffc361b3993fabb46aa03be8a522195": ("i4_stable_width", "i4"),
    "fff82417afd42556e4d14986b48fa750": ("i5_low_straight", "i5"),
    "fffe1d98bbd73bec32507b7ebeffa5a6": ("i5_high_curved", "i5"),
}
MANIFESTS = (
    "categorized_manifest/output/universal_manifest_culane_tagged.json",
    "categorized_manifest/output/universal_manifest_curvelane_tagged.json",
)


def _shrink(img: np.ndarray, max_side: int = 1050) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (25, 25, 25), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def render_i5_curvature_overlay(image_rgb: np.ndarray, analysis: dict) -> np.ndarray:
    """Centerlines coloured by local |curvature| (green low -> red high), BGR."""

    bgr = cv2.cvtColor(
        np.asarray(image_rgb)[..., :3].astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR
    ).copy()
    kappas = []
    for lane in analysis.get("lane_diagnostics", []):
        debug = lane.get("debug")
        if isinstance(debug, dict):
            k = np.abs(np.asarray(debug.get("kappa", []), dtype=np.float64))
            kappas.append(k[np.isfinite(k)])
    # Normalize colour by the image's own 95th-percentile curvature.
    reference = max(float(np.percentile(np.concatenate(kappas), 95)), 1e-9) if kappas else 1e-9
    for lane in analysis.get("lane_diagnostics", []):
        debug = lane.get("debug")
        if not isinstance(debug, dict):
            continue
        xy = np.asarray(debug.get("image_xy", []), dtype=np.float64)
        kappa = np.abs(np.asarray(debug.get("kappa", []), dtype=np.float64))
        if xy.ndim != 2 or xy.shape[0] < 2:
            continue
        for i in range(xy.shape[0] - 1):
            if not np.isfinite(xy[i:i + 2]).all():
                continue
            t = float(np.clip((kappa[i] if i < kappa.size and np.isfinite(kappa[i]) else 0.0) / reference, 0.0, 1.0))
            colour = (int(40 + 30 * t), int(200 * (1.0 - t) + 40 * t), int(70 + 160 * t))  # green->red
            cv2.line(bgr, tuple(np.round(xy[i]).astype(int)), tuple(np.round(xy[i + 1]).astype(int)),
                     colour, 4, cv2.LINE_AA)
    cv2.putText(bgr, "centerline colour: green=straight  red=high curvature",
                (10, bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="outputs/metric_demos")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = {}
    for manifest in MANIFESTS:
        for sample in json.loads(Path(manifest).read_text())["samples"]:
            if sample["sample_id"] in DEMOS:
                samples[sample["sample_id"]] = sample

    for sample_id, (name, metric) in DEMOS.items():
        sample = samples[sample_id]
        rgb = cv2.cvtColor(cv2.imread(sample["image_path"]), cv2.COLOR_BGR2RGB)
        gt = sample["ground_truth"]
        lanes, _ = select_gt_geometry(
            gt, rgb.shape[:2],
            load_mask=lambda p=gt.get("mask_path"): cv2.imread(p, 0),
        )
        if metric == "i1":
            analysis = analyze_lane_wear(rgb, lanes)
            overlay = render_wear_overlay(rgb, analysis, title="I1 continuity/wear")
            score = analysis.get("I1")
        elif metric == "i4":
            analysis = analyze_lane_width_simple(lanes, rgb.shape[:2])
            overlay = render_width_overlay(rgb, analysis)
            score = analysis.get("I4")
        else:
            analysis = analyze_lane_geometry_complexity(
                lanes, image_shape=rgb.shape[:2], geometry_source="ground_truth", debug=True
            )
            overlay = render_i5_curvature_overlay(rgb, analysis)
            score = analysis.get("alignment_complexity")
            label = "N/A" if score is None else f"{score:.3f}"
            overlay = _banner(
                overlay,
                f"I5 geometry complexity: {label}  ({analysis.get('profile_class')})",
            )
        cv2.imwrite(str(out / f"{name}.jpg"), _shrink(overlay), [cv2.IMWRITE_JPEG_QUALITY, 92])
        # Save the untouched source frame next to each overlay for comparison.
        raw = cv2.imread(sample["image_path"])
        cv2.imwrite(str(out / f"{name}_RAW.jpg"), _shrink(raw), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{name:18s} {metric.upper()} = {'N/A' if score is None else round(float(score), 3)}"
              f"   src={sample['image_path']}")
    print(f"wrote 5 demo images -> {out}/")


if __name__ == "__main__":
    main()
