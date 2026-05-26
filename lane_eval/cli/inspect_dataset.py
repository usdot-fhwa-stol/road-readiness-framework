#!/usr/bin/env python3
"""Quick inspection of a dataset adapter — no model required.

Example:
    python -m lane_eval.cli.inspect_dataset \
        --dataset bdd100k_lane \
        --image-root /shared/data/bdd100k/images/val \
        --lane-mask-root /shared/data/bdd100k/ll_seg_annotations/val \
        --n 5
"""
from __future__ import annotations
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a lane dataset adapter")
    p.add_argument("--dataset", required=True,
                   choices=["bdd100k_lane", "curvelanes", "culane", "tusimple"])
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=5, help="Number of samples to print")
    # bdd100k
    p.add_argument("--image-root",     default=None)
    p.add_argument("--lane-mask-root", default=None)
    # others
    p.add_argument("--root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from lane_eval.datasets import build_dataset_adapter

    name = args.dataset
    if name == "bdd100k_lane":
        adapter = build_dataset_adapter(name, image_root=args.image_root,
                                        lane_mask_root=args.lane_mask_root,
                                        split=args.split)
    else:
        adapter = build_dataset_adapter(name, root=args.root, split=args.split)

    print(f"Dataset : {name}  split={args.split}")
    print(f"Samples : {len(adapter)}")
    print()

    for i in range(min(args.n, len(adapter))):
        s = adapter[i]
        mask_nz = int(np.count_nonzero(s.target.mask)) if s.target.mask is not None else -1
        n_lanes = len(s.target.lanes) if s.target.lanes is not None else "N/A"
        print(f"  [{i:04d}] id={s.image_id}  size={s.width}x{s.height}"
              f"  lanes={n_lanes}  mask_nonzero={mask_nz}")


if __name__ == "__main__":
    main()
