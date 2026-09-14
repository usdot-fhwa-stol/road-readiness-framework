"""Run the new GT-guided lane-wear I1 over a tagged manifest.

For every sample it computes the I1 wear/continuity score from GT geometry + the
RGB image (no model), writes an overlay JPG (score burned in, worn/broken regions
in red) and a scores CSV/JSON. It also prints a validation summary that compares
the score against the human ``Observed Marking Visibility`` tag, which is the
ground truth the metric is meant to reproduce.

Usage:
    python -m evaluation.run_lane_wear \
        --manifest tagging/output/universal_manifest_culane_tagged.json \
        --output-dir outputs/lane_wear/culane_tagged
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from evaluation.lane_wear import (
    LaneWearConfig,
    analyze_lane_wear,
    render_wear_overlay,
    select_gt_geometry,
)


def _load_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_mask_factory(mask_path: Optional[str]):
    def _load():
        if not mask_path:
            return None
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return _load


def _visibility_tag(sample: dict[str, Any]) -> str:
    tags = sample.get("meta", {}).get("tags", {}).get("summary", {})
    return tags.get("Observed Marking Visibility", "?")


def _marking_tag(sample: dict[str, Any]) -> str:
    tags = sample.get("meta", {}).get("tags", {}).get("summary", {})
    return tags.get("Pavement Marking Type & Configuration", "?")


def run(manifest_path: str, output_dir: str, max_side: int = 1280) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    samples = manifest["samples"]
    out = Path(output_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    cfg = LaneWearConfig()

    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        rgb = _load_rgb(sample["image_path"])
        if rgb is None:
            rows.append({"sample_id": sample_id, "I1": None,
                         "unavailable_reason": "image_unreadable"})
            continue
        gt = sample.get("ground_truth", {})
        lanes, geom_source = select_gt_geometry(
            gt, rgb.shape[:2], load_mask=_load_mask_factory(gt.get("mask_path"))
        )
        analysis = analyze_lane_wear(rgb, lanes, config=cfg)

        title = f"{sample_id[:28]}"
        overlay = render_wear_overlay(rgb, analysis, title=title)
        overlay = _resize(overlay, max_side)
        cv2.imwrite(str(out / "overlays" / f"{sample_id}.jpg"), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        rows.append({
            "sample_id": sample_id,
            "I1": analysis.get("I1"),
            "condition": analysis.get("condition"),
            "worn_fraction": analysis.get("worn_fraction"),
            "faded_fraction": analysis.get("faded_fraction"),
            "valid_lane_count": analysis.get("valid_lane_count"),
            "geometry_source": geom_source,
            "unavailable_reason": analysis.get("unavailable_reason"),
            "visibility_tag": _visibility_tag(sample),
            "marking_tag": _marking_tag(sample),
        })

    (out / "scores.json").write_text(json.dumps(rows, indent=2))
    _write_csv(out / "scores.csv", rows)
    summary = _validation_summary(rows)
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(manifest_path, out, summary)
    return summary


def _resize(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = ["sample_id", "I1", "condition", "worn_fraction", "faded_fraction",
              "valid_lane_count", "geometry_source", "unavailable_reason",
              "visibility_tag", "marking_tag"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bucket(tag: str) -> str:
    t = tag.lower()
    if "faded" in t or "worn" in t or "not visible" in t or "missing" in t or "low contrast" in t:
        return "worn/faded"
    if "clearly visible" in t:
        return "clearly_visible"
    return "other"


def _validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        score = r.get("I1")
        if score is None:
            continue
        groups[_bucket(r.get("visibility_tag", "?"))].append(float(score))
    stats = {
        g: {
            "n": len(v),
            "mean_I1": round(float(np.mean(v)), 4),
            "median_I1": round(float(np.median(v)), 4),
            "min_I1": round(float(np.min(v)), 4),
            "max_I1": round(float(np.max(v)), 4),
        }
        for g, v in groups.items()
    }
    sep = None
    if "clearly_visible" in stats and "worn/faded" in stats:
        sep = round(stats["clearly_visible"]["mean_I1"] - stats["worn/faded"]["mean_I1"], 4)
    return {
        "n_total": len(rows),
        "n_scored": sum(1 for r in rows if r.get("I1") is not None),
        "by_visibility_tag": stats,
        "clear_minus_worn_mean_gap": sep,
    }


def _print_summary(manifest_path: str, out: Path, summary: dict[str, Any]) -> None:
    print(f"\n=== lane-wear I1  ({Path(manifest_path).name}) ===")
    print(f"scored {summary['n_scored']}/{summary['n_total']} samples -> {out}")
    for group, s in sorted(summary["by_visibility_tag"].items()):
        print(f"  {group:16s} n={s['n']:3d}  mean I1={s['mean_I1']:.3f}  "
              f"median={s['median_I1']:.3f}  [{s['min_I1']:.2f}, {s['max_I1']:.2f}]")
    if summary["clear_minus_worn_mean_gap"] is not None:
        print(f"  --> Clearly-Visible minus Worn/Faded mean gap: "
              f"{summary['clear_minus_worn_mean_gap']:+.3f}  (want positive)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()
    run(args.manifest, args.output_dir, max_side=args.max_side)


if __name__ == "__main__":
    main()
