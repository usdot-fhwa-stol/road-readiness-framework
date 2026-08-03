#!/usr/bin/env python3
"""Build a 20-image subset manifest per dataset where BOTH image_path and
ground_truth_path are guaranteed to exist on disk.

The categorized spreadsheet can't satisfy that for every dataset (its BDD rows
are test-split with no lane masks; its TuSimple rows are unresolvable), so here
we sample directly from each dataset's split that ships ground truth:

  tusimple   train_set/clips/<..>.jpg      + train_set/seg_label/<..>.png
  bdd100k    images/val/<id>.jpg           + ll_seg_annotations/val/<id>.png
  culane     <frame>.jpg (from val_gt.txt) + laneseg_label_w16/<..>.png
  curvelanes valid/images/<hash>.jpg       + valid/labels/<hash>.lines.json

Output: output/subset_<dataset>.json  (same sample schema as the other
manifests: sample_id, image_path, ground_truth_path, gt_type, split). These
are then fed to tag_images.py to attach predicted_tags.
"""
import json
from pathlib import Path

DATA = Path("/shared/data")
OUT = Path(__file__).resolve().parent / "output"
N = 20


def tusimple():
    base = DATA / "TUSimple" / "train_set"
    rows = []
    for jf in ("label_data_0313.json", "label_data_0531.json", "label_data_0601.json"):
        for line in (base / jf).read_text().splitlines():
            raw = json.loads(line)["raw_file"]              # clips/<date>/<ts>/20.jpg
            img = base / raw
            gt = base / "seg_label" / raw.replace("clips/", "").replace(".jpg", ".png")
            if img.exists() and gt.exists():
                rows.append((raw, img, gt))
        if len(rows) >= N:
            break
    return "tusimple", "train", "lane_mask_png", rows[:N]


def bdd100k():
    imgs = DATA / "bdd100k" / "images" / "val"
    masks = DATA / "bdd100k" / "ll_seg_annotations" / "val"
    rows = []
    for img in sorted(imgs.glob("*.jpg")):
        gt = masks / (img.stem + ".png")
        if gt.exists():
            rows.append((img.name, img, gt))
        if len(rows) >= N:
            break
    return "bdd100k", "val", "lane_mask_png", rows


def culane():
    # Only the test drivers (100/193/37) have extracted .jpg frames on disk;
    # their masks live under laneseg_label_w16_test.
    root = DATA / "culane"
    rows = []
    for line in (root / "list" / "test.txt").read_text().splitlines():
        rel = line.split()[0].lstrip("/")               # driver_.../<seg>.MP4/<f>.jpg
        img = root / rel
        gt = root / "laneseg_label_w16_test" / rel.replace(".jpg", ".png")
        if img.exists() and gt.exists():
            rows.append((rel, img, gt))
        if len(rows) >= N:
            break
    return "culane", "test", "lane_mask_png", rows


def curvelanes():
    base = DATA / "Curvelanes" / "valid"
    rows = []
    for img in sorted((base / "images").glob("*.jpg")):
        gt = base / "labels" / (img.stem + ".lines.json")
        if gt.exists():
            rows.append((img.name, img, gt))
        if len(rows) >= N:
            break
    return "curvelanes", "valid", "lines_json", rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (tusimple, bdd100k, culane, curvelanes):
        dataset, split, gt_type, rows = fn()
        samples = []
        for i, (name, img, gt) in enumerate(rows, 1):
            samples.append({
                "sample_id": f"{dataset}_{i}",
                "file_name": name,
                "image_path": str(img.resolve()),
                "ground_truth_path": str(gt.resolve()),
                "gt_type": gt_type,
                "split": split,
                "image_found": True,
                "gt_found": True,
            })
        manifest = {
            "metadata": {"dataset": dataset, "split": split,
                         "num_samples": len(samples), "gt_type": gt_type,
                         "note": "image+GT both guaranteed present"},
            "samples": samples,
        }
        p = OUT / f"subset_{dataset}.json"
        p.write_text(json.dumps(manifest, indent=2))
        print(f"{dataset:11s} {len(samples):2d} samples (image+gt) -> {p.name}")
        if len(samples) < N:
            print(f"   !! only {len(samples)} pairs found (<{N})")


if __name__ == "__main__":
    main()
