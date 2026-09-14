"""Render raw/CLRerNet/YOLOPX comparison sheets from per-image predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DATASETS = ("bdd100k_lane", "culane", "curvelanes", "tusimple")


def _read_rows(path: Path) -> dict[str, dict]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("per_image", [])
    else:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    return {str(row.get("image_id", row.get("sample_id"))): row for row in rows}


def _number(row: dict, key: str) -> float | None:
    value = row.get(key)
    try:
        return float(value) if value not in (None, "", "null") else None
    except (TypeError, ValueError):
        return None


def _path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    # Existing CSVs were produced from another checkout path.
    marker = Path("/home/karthikab/Projects/road-readiness-framework")
    current = Path.cwd()
    try:
        relative = path.relative_to(marker)
    except ValueError:
        return path
    candidate = current / relative
    return candidate if candidate.exists() else path


def _overlay(image: np.ndarray, mask_path: Path | None, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    if mask_path is None or not mask_path.exists():
        cv2.putText(output, "prediction unavailable", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return output
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return output
    if mask.shape != output.shape[:2]:
        mask = cv2.resize(mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = mask > 0
    tint = np.zeros_like(output)
    tint[:] = color
    output[binary] = (0.55 * output[binary] + 0.45 * tint[binary]).astype(np.uint8)
    return output


def _label(image: np.ndarray, title: str, row: dict) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), (20, 20, 20), -1)
    cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    iou = _number(row, "lane_iou")
    f1 = _number(row, "lane_f1")
    text = f"IoU {iou:.3f}  F1 {f1:.3f}" if iou is not None and f1 is not None else "score unavailable"
    cv2.putText(canvas, text, (18, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 220, 255), 2, cv2.LINE_AA)
    return canvas


def render_dataset(dataset: str, root: Path, limit: int) -> int:
    yo_rows = _read_rows(root / "results" / "per_image" / f"yolopx_{dataset}_per_image.json")
    clr_rows = _read_rows(root.parent / "results" / "per_image_csv" / f"clrernet_{dataset}_per_image_metrics.csv")
    ids = [sample_id for sample_id in yo_rows if sample_id in clr_rows][:limit]
    output_dir = root / "images" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sample_id in ids:
        yo = yo_rows[sample_id]
        clr = clr_rows[sample_id]
        image_path = _path(yo.get("image_path") or clr.get("image_path"))
        if image_path is None or not image_path.exists():
            continue
        raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            continue
        clr_mask = _path(clr.get("pred_mask_path"))
        yo_mask = root / "predictions" / "yolopx" / dataset / "masks" / f"{sample_id}.png"
        raw = _label(raw_bgr, "RAW RGB", {"lane_iou": None, "lane_f1": None})
        clr_panel = _label(_overlay(raw_bgr, clr_mask, (40, 170, 255)), "CLRerNet", clr)
        yo_panel = _label(_overlay(raw_bgr, yo_mask, (70, 220, 90)), "YOLOPX", yo)
        panels = [raw, clr_panel, yo_panel]
        target_height = min(480, max(240, raw.shape[0]))
        panels = [cv2.resize(panel, (int(panel.shape[1] * target_height / panel.shape[0]), target_height)) for panel in panels]
        composite = cv2.hconcat(panels)
        cv2.putText(composite, f"{dataset}  {sample_id}", (18, composite.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"{written + 1:02d}_{sample_id}.jpg"), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/model_comparison_20"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    for dataset in args.datasets:
        print(f"{dataset}: wrote {render_dataset(dataset, args.root, args.limit)} images")


if __name__ == "__main__":
    main()
