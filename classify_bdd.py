#!/usr/bin/env python3
"""Batch zero-shot road-geometry classification with SigLIP2.

Recursively scans an image directory and writes one row per image to:
  - road_geometry_scores.csv
  - road_geometry_scores.json

The CSV is flushed after every batch. Re-running resumes from the existing CSV;
use --overwrite to restart from the beginning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm.auto import tqdm


MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_DATASET_DIR = Path("/shared/data/bdd100k/images/test")
DEFAULT_OUTPUT_DIR = Path("/shared/data/bdd100k/road_geometry_outputs")

# The prompt text is kept identical to the working single-image version.
CLASS_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "Straight Road",
        "Straight Road Score",
        "a forward-facing dashcam image of a straight road",
    ),
    (
        "Curved Road",
        "Curved Road Score",
        "a forward-facing dashcam image of a curved road",
    ),
    (
        "Roundabout",
        "Roundabout Score",
        "a forward-facing dashcam image of a road entering or circulating around a roundabout",
    ),
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

CSV_FIELDS = [
    "#",
    "File Name",
    "Summary — road geometry prediction",
    "Relative Path",
    "Predicted Label",
    "Predicted Score",
    "Straight Road Score",
    "Curved Road Score",
    "Roundabout Score",
    "Score Margin",
    "Status",
    "Error",
]

FLOAT_FIELDS = {
    "Predicted Score",
    "Straight Road Score",
    "Curved Road Score",
    "Roundabout Score",
    "Score Margin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify road images as straight, curved, or roundabout with "
            "SigLIP2 and export CSV/JSON results."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Image directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Hugging Face model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16). Reduce it after CUDA OOM.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device: "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional image limit for a small test run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete prior outputs and restart instead of resuming.",
    )
    return parser.parse_args()


def resolve_pipeline_device(requested: str) -> int:
    value = requested.strip().lower()
    if value == "auto":
        return 0 if torch.cuda.is_available() else -1
    if value == "cpu":
        return -1
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        try:
            index = int(value.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {requested!r}") from exc
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {index} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )
        return index
    raise ValueError(
        f"Unsupported device {requested!r}; use auto, cpu, cuda, or cuda:N."
    )


def discover_images(root: Path, limit: int | None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        paths = paths[:limit]
    return paths


def load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc


def normalize_outputs(outputs: Any, expected: int) -> list[list[dict[str, Any]]]:
    """Normalize the pipeline's one-image and multi-image return structures."""
    if expected == 1 and isinstance(outputs, list) and outputs:
        if isinstance(outputs[0], dict):
            outputs = [outputs]

    if not isinstance(outputs, list) or len(outputs) != expected:
        actual = len(outputs) if isinstance(outputs, list) else type(outputs).__name__
        raise RuntimeError(
            f"Unexpected pipeline output: expected {expected} result(s), got {actual}."
        )
    if any(not isinstance(result, list) for result in outputs):
        raise RuntimeError("Each image result must be a list of label-score dictionaries.")
    return outputs


def classify_with_recovery(
    classifier: Any,
    items: Sequence[tuple[int, Path, Image.Image]],
    batch_size: int,
) -> list[tuple[int, Path, list[dict[str, Any]] | None, str | None]]:
    """Classify a batch; split it recursively if one image or batch fails."""
    if not items:
        return []

    images = [item[2] for item in items]
    candidate_labels = [spec[2] for spec in CLASS_SPECS]

    try:
        # Omitting hypothesis_template intentionally preserves the behavior of
        # the user's working single-image pipeline code.
        outputs = classifier(
            images,
            candidate_labels=candidate_labels,
            batch_size=min(batch_size, len(images)),
        )
        normalized = normalize_outputs(outputs, len(items))
        return [
            (index, path, prediction, None)
            for (index, path, _image), prediction in zip(items, normalized)
        ]
    except Exception as exc:
        if torch.cuda.is_available() and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()

        if len(items) == 1:
            index, path, _image = items[0]
            return [(index, path, None, f"{type(exc).__name__}: {exc}")]

        midpoint = len(items) // 2
        return classify_with_recovery(
            classifier, items[:midpoint], batch_size
        ) + classify_with_recovery(classifier, items[midpoint:], batch_size)


def blank_row(index: int, path: Path, root: Path) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "#": str(index),
            "File Name": path.name,
            "Relative Path": path.relative_to(root).as_posix(),
        }
    )
    return row


def make_error_row(
    index: int,
    path: Path,
    root: Path,
    error: str,
) -> dict[str, str]:
    row = blank_row(index, path, root)
    row.update({"Status": "error", "Error": error})
    return row


def make_success_row(
    index: int,
    path: Path,
    root: Path,
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    score_by_prompt = {
        str(item["label"]): float(item["score"])
        for item in predictions
    }

    missing = [prompt for _label, _column, prompt in CLASS_SPECS if prompt not in score_by_prompt]
    if missing:
        raise RuntimeError("Missing model output for: " + "; ".join(missing))

    scored_classes = [
        (label, score_column, score_by_prompt[prompt])
        for label, score_column, prompt in CLASS_SPECS
    ]
    ranked = sorted(scored_classes, key=lambda item: item[2], reverse=True)
    predicted_label, _score_column, predicted_score = ranked[0]
    margin = predicted_score - ranked[1][2]

    row = blank_row(index, path, root)
    row.update(
        {
            "Summary — road geometry prediction": predicted_label,
            "Predicted Label": predicted_label,
            "Predicted Score": f"{predicted_score:.10f}",
            "Score Margin": f"{margin:.10f}",
            "Status": "ok",
            "Error": "",
        }
    )
    for _label, score_column, score in scored_classes:
        row[score_column] = f"{score:.10f}"
    return row


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                f"Existing CSV schema differs from this script: {csv_path}. "
                "Use --overwrite or choose another output directory."
            )
        return [dict(row) for row in reader]


def sort_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(csv_path)
    rows.sort(key=lambda row: int(row["#"]))

    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return rows


def json_value(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return float(value) if value else None


def write_json(
    json_path: Path,
    rows: Sequence[dict[str, str]],
    *,
    dataset_dir: Path,
    model_id: str,
    discovered_count: int,
) -> None:
    results = []
    for row in rows:
        results.append(
            {
                "index": int(row["#"]),
                "file_name": row["File Name"],
                "relative_path": row["Relative Path"],
                "predicted_label": row["Predicted Label"] or None,
                "predicted_score": json_value(row, "Predicted Score"),
                "scores": {
                    "straight_road": json_value(row, "Straight Road Score"),
                    "curved_road": json_value(row, "Curved Road Score"),
                    "roundabout": json_value(row, "Roundabout Score"),
                },
                "score_margin": json_value(row, "Score Margin"),
                "status": row["Status"],
                "error": row["Error"] or None,
            }
        )

    successful = sum(result["status"] == "ok" for result in results)
    document = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "dataset_directory": str(dataset_dir),
            "images_discovered": discovered_count,
            "rows_written": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "candidate_labels": {
                label: prompt for label, _column, prompt in CLASS_SPECS
            },
            "score_semantics": (
                "SigLIP2 image-text matching scores. They need not sum to 1 and "
                "are not calibrated three-class probabilities. The predicted "
                "label is the class with the largest score."
            ),
        },
        "predictions": results,
    }

    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(json_path)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "road_geometry_scores.csv"
    json_path = output_dir / "road_geometry_scores.json"

    if args.overwrite:
        csv_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)

    image_paths = discover_images(dataset_dir, args.limit)
    if not image_paths:
        raise RuntimeError(f"No supported images were found under {dataset_dir}")

    indexed_paths = list(enumerate(image_paths, start=1))
    existing_rows = read_csv_rows(csv_path)
    processed = {row["Relative Path"] for row in existing_rows}
    pending = [
        (index, path)
        for index, path in indexed_paths
        if path.relative_to(dataset_dir).as_posix() not in processed
    ]

    print(f"Dataset       : {dataset_dir}")
    print(f"Images found  : {len(image_paths):,}")
    print(f"Already done  : {len(existing_rows):,}")
    print(f"Pending       : {len(pending):,}")
    print(f"CSV output    : {csv_path}")
    print(f"JSON output   : {json_path}")

    if pending:
        pipeline_device = resolve_pipeline_device(args.device)
        device_name = f"cuda:{pipeline_device}" if pipeline_device >= 0 else "cpu"
        print(f"Loading model : {args.model_id} on {device_name}")
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The transformers package is required. Install it with: "
                "python -m pip install -U transformers pillow tqdm"
            ) from exc

        classifier = pipeline(
            task="zero-shot-image-classification",
            model=args.model_id,
            device=pipeline_device,
        )

        append = csv_path.exists() and csv_path.stat().st_size > 0
        mode = "a" if append else "w"
        with csv_path.open(mode, encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if not append:
                writer.writeheader()
                handle.flush()

            total_batches = math.ceil(len(pending) / args.batch_size)
            for start in tqdm(
                range(0, len(pending), args.batch_size),
                total=total_batches,
                desc="Classifying",
                unit="batch",
            ):
                path_batch = pending[start : start + args.batch_size]
                loaded: list[tuple[int, Path, Image.Image]] = []

                for index, path in path_batch:
                    try:
                        loaded.append((index, path, load_rgb_image(path)))
                    except Exception as exc:
                        writer.writerow(
                            make_error_row(
                                index,
                                path,
                                dataset_dir,
                                f"{type(exc).__name__}: {exc}",
                            )
                        )

                try:
                    inference_results = classify_with_recovery(
                        classifier, loaded, args.batch_size
                    )
                    for index, path, predictions, error in inference_results:
                        if predictions is None:
                            row = make_error_row(
                                index,
                                path,
                                dataset_dir,
                                error or "Unknown inference error",
                            )
                        else:
                            try:
                                row = make_success_row(
                                    index, path, dataset_dir, predictions
                                )
                            except Exception as exc:
                                row = make_error_row(
                                    index,
                                    path,
                                    dataset_dir,
                                    f"{type(exc).__name__}: {exc}",
                                )
                        writer.writerow(row)
                finally:
                    for _index, _path, image in loaded:
                        image.close()

                # Checkpoint after each batch for safe resume after interruption.
                handle.flush()

    rows = sort_csv_rows(csv_path)
    write_json(
        json_path,
        rows,
        dataset_dir=dataset_dir,
        model_id=args.model_id,
        discovered_count=len(image_paths),
    )

    successful = sum(row["Status"] == "ok" for row in rows)
    print("\nCompleted")
    print(f"Rows           : {len(rows):,}")
    print(f"Successful     : {successful:,}")
    print(f"Errors         : {len(rows) - successful:,}")
    print(f"CSV            : {csv_path}")
    print(f"JSON           : {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed CSV batches were preserved; rerun to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
