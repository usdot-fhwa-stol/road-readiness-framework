#!/usr/bin/env python3
"""Build a manifest from the new pilot tagged.xlsx (human tags) for the rows we
can resolve on /shared/data: BDD100K val/test, CurveLane, CuLane. Waymo /
Mapillary / TuSimple are skipped (no images on disk / bare numeric ids).

Each row's human labels = union of the 6 'Summary' dimension columns (2..7),
semicolon-separated. Output: output/pilot_manifest.json (same shape the taggers
expect: samples[].image_path + tags.labels), so tag_images.py + eval_tagger.py
work unchanged.

  python3 build_pilot_manifest.py
"""
import glob
import json
import os
import random

import openpyxl

XLSX = os.path.join(os.path.dirname(__file__), "..", "tagged.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "output", "pilot_manifest.json")
CULANE = "/shared/data/culane"
BDD = "/shared/data/bdd100k"
CURVE = "/shared/data/Curvelanes"
TUS = "/shared/data/TUSimple"
TUS_SAMPLE_N = 500        # TuSimple has no pilot tags -> random test/val sample
TUS_SEED = 0

SHEET_MAP = {           # sheet -> (dataset, split-or-None)
    "BDD100K val": ("bdd100k", "val"),
    "BDD100K test": ("bdd100k", "test"),
    "CurveLane": ("curvelanes", None),
    "CuLane": ("culane", None),
}
SUMMARY_COLS = range(2, 8)


def human_labels(row):
    labels = []
    for c in SUMMARY_COLS:
        if row[c]:
            for t in str(row[c]).split(";"):
                t = t.strip()
                if t and t not in labels:
                    labels.append(t)
    return labels


def resolve_bdd(fname, split):
    ip = f"{BDD}/images/{split}/{fname}"
    name = os.path.splitext(fname)[0]
    gt = f"{BDD}/ll_seg_annotations/{split}/{name}.png"
    gt = gt if os.path.exists(gt) else None
    return (ip if os.path.exists(ip) else None, None, gt, split)


def resolve_curvelanes(fname):
    name = os.path.splitext(fname)[0]
    for split in ("train", "valid", "test"):
        ip = f"{CURVE}/{split}/images/{fname}"
        if os.path.exists(ip):
            gt = f"{CURVE}/{split}/labels/{name}.lines.json"
            return ip, None, (gt if os.path.exists(gt) else None), split
    return None, None, None, None


def resolve_culane(fname):
    seg = fname.replace(".mp4", ".MP4")           # xlsx lowercases .mp4
    test_drivers = set(os.listdir(f"{CULANE}/laneseg_label_w16_test"))
    for driver in os.listdir(CULANE):
        d = f"{CULANE}/{driver}/{seg}"
        if os.path.isdir(d):
            frame = f"{d}/00000.jpg"                # representative frame to tag
            split = "test" if driver in test_drivers else "train"
            lane = "laneseg_label_w16_test" if split == "test" else "laneseg_label_w16"
            gt = f"{CULANE}/{lane}/{driver}/{seg}"   # per-segment mask dir
            gt = gt if os.path.isdir(gt) else None
            return (frame if os.path.exists(frame) else None), d, gt, split
    return None, None, None, None


def sample_tusimple(n=TUS_SAMPLE_N, seed=TUS_SEED):
    """Random distinct-scene sample: the canonical labeled frame (20.jpg) of n
    random clips from TuSimple's evaluation sets (test_set, and val_set if it
    exists on disk). No human tags -> tags.labels=[] (predictions only)."""
    eval_sets = [("test_set", "test"), ("val_set", "val")]
    clips = []
    for setname, split in eval_sets:
        base = f"{TUS}/{setname}/clips"
        if os.path.isdir(base):
            clips += [(d, split) for d in glob.glob(f"{base}/*/*") if os.path.isdir(d)]
    random.seed(seed)
    random.shuffle(clips)
    out = []
    for d, split in clips:
        ip = os.path.join(d, "20.jpg")
        if os.path.exists(ip):
            reldir = os.path.relpath(d, TUS)
            out.append({
                "sample_id": f"pilot_tusimple_{split}_{reldir.replace('/', '_')}",
                "dataset": "tusimple", "split": split,
                "file_name": os.path.relpath(ip, TUS),
                "image_path": ip, "representative_frame": None,
                "ground_truth_path": None, "gt_found": False,
                "image_found": True, "tags": {"labels": []},
            })
        if len(out) >= n:
            break
    return out


def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    samples, stats = [], {}
    for sheet, (dataset, fixed_split) in SHEET_MAP.items():
        # keep real filename rows only (drop trailing 'TOTAL (count)' footers)
        rows = [r for r in list(wb[sheet].iter_rows(values_only=True))[2:]
                if r[1] and "." in str(r[1]) and "TOTAL" not in str(r[1]).upper()]
        found = 0
        for i, r in enumerate(rows, 1):
            fname = str(r[1]).strip()
            labels = human_labels(r)
            if dataset == "bdd100k":
                ip, rep, gt, split = resolve_bdd(fname, fixed_split)
            elif dataset == "curvelanes":
                ip, rep, gt, split = resolve_curvelanes(fname)
            else:
                ip, rep, gt, split = resolve_culane(fname)
            image_found = ip is not None
            found += image_found
            samples.append({
                "sample_id": f"pilot_{dataset}_{split or '?'}_{i}",
                "dataset": dataset, "split": split, "file_name": fname,
                "image_path": ip, "representative_frame": rep,
                "ground_truth_path": gt, "gt_found": gt is not None,
                "image_found": image_found,
                "tags": {"labels": labels},
            })
        stats[sheet] = (len(rows), found)

    tus = sample_tusimple()
    samples += tus
    stats[f"TuSimple (random {TUS_SAMPLE_N})"] = (len(tus), len(tus))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"metadata": {"source": "tagged.xlsx pilot",
                                "note": "human tags = union of 6 summary columns"},
                   "samples": samples}, f, indent=2)
    print(f"wrote {OUT}: {len(samples)} rows")
    for sheet, (tot, found) in stats.items():
        print(f"  {sheet:15} rows={tot:3} image_found={found}")
    unresolved = [s["file_name"] for s in samples if not s["image_found"]]
    print(f"  unresolved images: {len(unresolved)}", unresolved[:5])


if __name__ == "__main__":
    main()
