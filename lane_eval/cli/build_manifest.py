#!/usr/bin/env python3
"""Convert a dataset into the universal manifest JSON.

Reuses the existing dataset adapters (lane_eval.datasets) and converters; this
CLI only wires arguments to the right adapter constructor, mirroring
run_lane_eval's per-dataset argument handling.

Examples
--------
TuSimple (test split):
    python -m lane_eval.cli.build_manifest \
        --dataset tusimple \
        --root /shared/data/TUSimple/test_set \
        --annotation-file /shared/data/TUSimple/test_label.json \
        --split test \
        --out-dir manifests

BDD100K (mask-native GT; lane_json is approximated from the mask):
    python -m lane_eval.cli.build_manifest \
        --dataset bdd100k_lane \
        --image-root /shared/data/bdd100k/images/val \
        --lane-mask-root /shared/data/bdd100k/ll_seg_annotations/val \
        --det-annotations-root /shared/data/bdd100k/det_annotations/val \
        --out-dir manifests
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _build_adapter(args):
    """Construct the dataset adapter from CLI args (matches run_lane_eval)."""
    from lane_eval.datasets import build_dataset_adapter

    name = args.dataset
    if name == "bdd100k_lane":
        kwargs = dict(
            image_root=args.image_root,
            lane_mask_root=args.lane_mask_root,
            split=args.split,
        )
        if args.det_annotations_root:
            kwargs["det_annotations_root"] = args.det_annotations_root
        return build_dataset_adapter(name, **kwargs)

    kwargs = dict(root=args.root, split=args.split)
    if args.mask_thickness:
        kwargs["mask_thickness"] = args.mask_thickness
    if name == "tusimple" and args.annotation_file:
        kwargs["annotation_files"] = [args.annotation_file]
    return build_dataset_adapter(name, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["tusimple", "culane", "curvelanes", "bdd100k_lane"])
    p.add_argument("--split", default="val")
    p.add_argument("--out-dir", default="manifests",
                   help="Root output folder; manifest written to <out-dir>/<dataset>/")
    p.add_argument("--h-step", type=int, default=10,
                   help="Spacing (pixels) between h_samples rows")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of samples (for quick tests)")
    p.add_argument("--no-save-masks", action="store_true",
                   help="Do not rasterise/write masks for polyline datasets")

    # bdd100k_lane
    p.add_argument("--image-root", default=None, help="[bdd100k_lane] image directory")
    p.add_argument("--lane-mask-root", default=None, help="[bdd100k_lane] lane mask directory")
    p.add_argument("--det-annotations-root", default=None,
                   help="[bdd100k_lane] per-image attribute JSON directory")

    # curvelanes / culane / tusimple
    p.add_argument("--root", default=None, help="[curvelanes/culane/tusimple] dataset root")
    p.add_argument("--annotation-file", default=None, help="[tusimple] JSON-lines label file")
    p.add_argument("--mask-thickness", type=int, default=16)

    args = p.parse_args()

    from lane_eval.manifest import build_manifest

    adapter = _build_adapter(args)
    print(f"Loaded {len(adapter)} samples for '{args.dataset}'")

    manifest_path, count = build_manifest(
        adapter,
        dataset_name=args.dataset,
        out_dir=Path(args.out_dir),
        step=args.h_step,
        save_masks=not args.no_save_masks,
        max_samples=args.max_samples,
    )
    print(f"Wrote {count} samples -> {manifest_path}")


if __name__ == "__main__":
    main()
