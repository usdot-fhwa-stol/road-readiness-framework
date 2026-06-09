"""Convert any registered dataset into the universal manifest format.

The manifest is a single JSON file per dataset:

    {
      "metadata": {"dataset": ..., "num_samples": N, "h_sample_step": 10},
      "samples": [
        {
          "sample_id": ...,
          "image_path": ...,
          "width": W, "height": H,
          "ground_truth": {
            "mask_path": ...,            # link to a rasterised lane mask
            "natural_gt": "lanes"|"mask", # which representation is native to the dataset
            "lane_json": {"h_samples": [...], "lanes": [[x, ...], ...]}
          },
          "meta": {...}                  # passthrough dataset attributes (weather, condition, ...)
        },
        ...
      ]
    }

This module only orchestrates: it reuses the existing dataset adapters
(lane_eval.datasets) to parse each native format and the converters
(lane_eval.converters) to derive the two ground-truth representations. No
parsing or geometry is reimplemented here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import cv2

from ..schema.sample import LaneSample
from ..converters import (
    make_h_samples,
    polylines_to_lane_json,
    mask_to_lane_json,
)


def sample_to_entry(
    sample: LaneSample,
    mask_dir: Path,
    step: int = 10,
    save_masks: bool = True,
) -> dict:
    """Build one manifest entry from a LaneSample.

    Ground-truth representations:
      * Polyline-native datasets (lanes present) -> lane_json by resampling the
        polylines; natural_gt = "lanes".
      * Mask-native datasets (mask only) -> lane_json approximated from the mask;
        natural_gt = "mask".

    The mask link prefers an existing on-disk mask (target.mask_path). Otherwise,
    if save_masks, the in-memory rasterised mask is written to
    `mask_dir/<sample_id>.png` (0/255) and that path is recorded.
    """
    target = sample.target

    if target.lanes is not None:
        natural_gt = "lanes"
        lane_json = polylines_to_lane_json(target.lanes, sample.height, sample.width, step)
    else:
        natural_gt = "mask"
        if target.mask is not None:
            lane_json = mask_to_lane_json(target.mask, step)
        else:
            lane_json = {"h_samples": make_h_samples(sample.height, step), "lanes": []}

    # Resolve the mask link.
    if target.mask_path:
        mask_path: Optional[str] = target.mask_path
    elif save_masks and target.mask is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        out_mask = mask_dir / f"{sample.image_id}.png"
        cv2.imwrite(str(out_mask), (target.mask > 0).astype("uint8") * 255)
        mask_path = str(out_mask)
    else:
        mask_path = None

    entry = {
        "sample_id": sample.image_id,
        "image_path": sample.image_path,
        "width": sample.width,
        "height": sample.height,
        "ground_truth": {
            "mask_path": mask_path,
            "natural_gt": natural_gt,
            "lane_json": lane_json,
        },
    }
    if sample.meta:
        entry["meta"] = sample.meta
    return entry


def build_manifest(
    adapter: Iterable[LaneSample],
    dataset_name: str,
    out_dir: Path,
    step: int = 10,
    save_masks: bool = True,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> tuple[Path, int]:
    """Generate and write the manifest for one dataset.

    `adapter` is any object yielding LaneSample (a registered dataset adapter).
    Writes `out_dir/<dataset>/manifest_<dataset>.json` and, when masks must be
    rasterised, `out_dir/<dataset>/masks/*.png`. Returns (manifest_path, count).
    """
    out_dir = Path(out_dir)
    dataset_dir = out_dir / dataset_name
    mask_dir = dataset_dir / "masks"

    total = len(adapter) if hasattr(adapter, "__len__") else None
    if max_samples is not None and total is not None:
        total = min(total, max_samples)

    iterator = adapter.iter_samples() if hasattr(adapter, "iter_samples") else iter(adapter)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=total, desc=f"manifest:{dataset_name}")
        except ImportError:
            pass

    samples = []
    for i, sample in enumerate(iterator):
        if max_samples is not None and i >= max_samples:
            break
        samples.append(sample_to_entry(sample, mask_dir, step=step, save_masks=save_masks))

    manifest = {
        "metadata": {
            "dataset": dataset_name,
            "num_samples": len(samples),
            "h_sample_step": step,
        },
        "samples": samples,
    }

    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / f"manifest_{dataset_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path, len(samples)
