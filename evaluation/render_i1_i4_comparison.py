"""Compute model-guided I1/I4 and render raw/prediction panels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.readiness_metrics import build_metric_record
from lane_eval.manifest import ManifestDataset


DATASETS = ("bdd100k_lane", "culane", "curvelanes", "tusimple")


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else data.get("per_image", [])
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    try:
        return None if value in (None, "", "null") else float(value)
    except (TypeError, ValueError):
        return None


def _resolve(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    marker = Path("/home/karthikab/Projects/road-readiness-framework")
    try:
        candidate = Path.cwd() / path.relative_to(marker)
    except ValueError:
        return path
    return candidate if candidate.exists() else path


def _panel(image: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 92), (20, 20, 20), -1)
    cv2.putText(output, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(output, line, (16, 57 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 2, cv2.LINE_AA)
    return output


def _prediction_overlay(image: np.ndarray, mask_path: Path | None) -> np.ndarray:
    output = image.copy()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path and mask_path.exists() else None
    if mask is None:
        cv2.putText(output, "prediction unavailable", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return output
    if mask.shape != output.shape[:2]:
        mask = cv2.resize(mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
    tint = np.zeros_like(output)
    tint[:] = (60, 210, 90)
    active = mask > 0
    output[active] = (0.55 * output[active] + 0.45 * tint[active]).astype(np.uint8)
    return output


def render_dataset(dataset: str, root: Path, manifests: Path, limit: int) -> list[dict]:
    manifest_path = manifests / dataset / f"manifest_{dataset}.json"
    samples = ManifestDataset(str(manifest_path))
    yo_rows = {str(row["image_id"]): row for row in _rows(root / "results" / "per_image" / f"yolopx_{dataset}_per_image.json")}
    clr_rows = {str(row["sample_id"]): row for row in _rows(Path("outputs/results/per_image_csv") / f"clrernet_{dataset}_per_image_metrics.csv")}
    model_rows = {
        "yolopx": yo_rows,
        "clrernet": clr_rows,
    }
    sample_by_id = {str(sample.image_id): sample for sample in samples}
    records: list[dict] = []
    for model, rows in model_rows.items():
        out_i1 = root / "I1" / dataset / model
        out_i4 = root / "I4" / dataset / model
        out_i1.mkdir(parents=True, exist_ok=True)
        out_i4.mkdir(parents=True, exist_ok=True)
        count = 0
        for sample_id, row in rows.items():
            if count >= limit or sample_id not in sample_by_id:
                continue
            sample = sample_by_id[sample_id]
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            pred_value = (
                root / "predictions" / "yolopx" / dataset / "masks" / f"{sample_id}.png"
                if model == "yolopx"
                else _resolve(row.get("pred_mask_path"))
            )
            pred = cv2.imread(str(pred_value), cv2.IMREAD_GRAYSCALE) if pred_value and pred_value.exists() else None
            if image is None or pred is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            record = build_metric_record(sample, pred, image_rgb, dataset=dataset, model_name=model)
            i1 = _float(record.get("I1_pred_pattern_continuity"))
            i4 = _float(record.get("I4_pred_lane_width_stability"))
            lane_f1 = _float(row.get("lane_f1"))
            result = {
                "dataset": dataset,
                "model": model,
                "sample_id": sample_id,
                "I1_pred_pattern_continuity": i1,
                "I1_pred_pattern_type": record.get("I1_pred_pattern_type"),
                "I1_pred_condition": record.get("I1_pred_condition"),
                "I1_pred_degradation_score": record.get("I1_pred_degradation_score"),
                "I4_pred_lane_width_stability": i4,
                "I4_pred_width_profile_type": record.get("I4_pred_width_profile_type"),
                "lane_f1": lane_f1,
            }
            records.append(result)
            stem = f"{count + 1:02d}_{sample_id}.jpg"
            raw_panel = _panel(image, "RAW RGB", [])
            pred_panel = _prediction_overlay(image, pred_value)
            i1_text = "I1 unavailable" if i1 is None else f"I1 {i1:.3f} ({result['I1_pred_pattern_type']})"
            i4_text = "I4 unavailable" if i4 is None else f"I4 {i4:.3f} ({result['I4_pred_width_profile_type']})"
            f1_text = "F1 unavailable" if lane_f1 is None else f"F1 {lane_f1:.3f}"
            cv2.imwrite(str(out_i1 / stem), cv2.hconcat([raw_panel, _panel(pred_panel, model, [i1_text, f1_text])]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out_i4 / stem), cv2.hconcat([raw_panel, _panel(pred_panel, model, [i4_text, f1_text])]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            count += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/model_comparison_20"))
    parser.add_argument("--manifests", type=Path, default=Path("manifests"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    all_records = []
    for dataset in args.datasets:
        records = render_dataset(dataset, args.root, args.manifests, args.limit)
        all_records.extend(records)
        print(f"{dataset}: {len(records)} metric records")
    (args.root / "i1_i4_metrics.json").write_text(json.dumps(all_records, indent=2, allow_nan=False))
    print(f"saved {len(all_records)} records -> {args.root / 'i1_i4_metrics.json'}")


if __name__ == "__main__":
    main()
