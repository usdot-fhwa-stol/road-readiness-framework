#!/usr/bin/env python3
"""PHASE 2 of the sampled-dataset build: copy selected samples into the repo
`dataset/` folder (uniform layout) and emit the universal manifest.

Reads dataset/_selected/<ds>.selected.jsonl (from build_sampled_dataset.py),
copies each image + annotation into dataset/<ds>/{images,annotations,attributes},
reads width/height + BDD native attributes, and writes a universal manifest
entry per sample. Per-dataset so it can run in parallel (one agent per dataset);
`--merge` combines into manifest_all.json.

  python3 assemble_manifest.py --dataset bdd100k
  python3 assemble_manifest.py --merge
"""
import argparse
import glob
import json
import os
import shutil

from PIL import Image

from tag_codes import encode_labels

ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
SEL_DIR = os.path.join(ROOT, "_selected")
MAN_DIR = os.path.join(ROOT, "manifests")
TAGGER = "google/gemma-3-27b-it"

ANNO_EXT = {"bdd_ll_seg_mask": ".png", "culane_laneseg_mask": ".png",
            "curvelanes_lines_json": ".lines.json", "tusimple_lane_json": ".lane.json"}


def bdd_attributes(attr_path):
    if attr_path and os.path.exists(attr_path):
        try:
            return json.load(open(attr_path)).get("attributes")  # weather/scene/timeofday
        except Exception:
            return None
    return None


def curvelanes_lane_json(lines_path):
    try:
        d = json.load(open(lines_path))
        return {"lines": [[{"x": float(p["x"]), "y": float(p["y"])} for p in line]
                          for line in d.get("Lines", [])]}
    except Exception:
        return None


def assemble(name):
    sel = os.path.join(SEL_DIR, f"{name}.selected.jsonl")
    if not os.path.exists(sel):
        print(f"[{name}] no selection file {sel}")
        return
    img_dir = os.path.join(ROOT, name, "images")
    ann_dir = os.path.join(ROOT, name, "annotations")
    attr_dir = os.path.join(ROOT, name, "attributes")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    samples = []
    for line in open(sel):
        try:
            c = json.loads(line)
        except Exception:
            continue
        if "error" in c or not c.get("predicted_tags"):
            continue
        sid = c["sample_id"]
        natural = c.get("natural_gt")

        ext = os.path.splitext(c["source_image_path"])[1] or ".jpg"
        img_dst = os.path.join(img_dir, sid + ext)
        if not os.path.exists(img_dst):
            shutil.copy2(c["source_image_path"], img_dst)
        try:
            with Image.open(img_dst) as im:
                W, H = im.size
        except Exception:
            W = H = None

        ann_dst = os.path.join(ann_dir, sid + ANNO_EXT.get(c["gt_type"], ".gt"))
        lane_json = None
        if c["gt_type"] == "tusimple_lane_json":
            rec = c.get("lane_record") or {}
            json.dump(rec, open(ann_dst, "w"))
            lane_json = rec
        elif c["gt_type"] == "curvelanes_lines_json":
            src = c.get("source_gt_path")
            if src and os.path.exists(src):
                if not os.path.exists(ann_dst):
                    shutil.copy2(src, ann_dst)
                lane_json = curvelanes_lane_json(src)
        else:                                            # mask datasets (bdd, culane)
            src = c.get("source_gt_path")
            if src and os.path.exists(src) and not os.path.exists(ann_dst):
                shutil.copy2(src, ann_dst)

        existing = None
        if name == "bdd100k":
            existing = bdd_attributes(c.get("attr_path"))
            if existing:
                os.makedirs(attr_dir, exist_ok=True)
                json.dump(existing, open(os.path.join(attr_dir, sid + ".json"), "w"))

        samples.append({
            "sample_id": sid, "dataset": name, "split": c.get("split"),
            "image_path": os.path.relpath(img_dst, ROOT),
            "source_image_path": c["source_image_path"],
            "width": W, "height": H,
            "ground_truth": {
                "mask_path": os.path.relpath(ann_dst, ROOT) if natural == "mask" else None,
                "lines_path": os.path.relpath(ann_dst, ROOT) if natural == "lanes" else None,
                "source_gt_path": c.get("source_gt_path"),
                "natural_gt": natural, "lane_json": lane_json,
            },
            "meta": {
                "existing_tags": existing,
                "predicted_tags": c["predicted_tags"],
                "tags_encoded": encode_labels(
                    (c["predicted_tags"] or {}).get("labels", [])),
            },
        })

    os.makedirs(MAN_DIR, exist_ok=True)
    out = os.path.join(MAN_DIR, f"manifest_{name}.json")
    json.dump({"metadata": {"dataset": name, "num_samples": len(samples),
                            "tagger": TAGGER, "encoding_ref": "tag_codes.json"},
               "samples": samples}, open(out, "w"), indent=2)
    print(f"[{name}] wrote {out}: {len(samples)} samples "
          f"(images+annotations under dataset/{name}/)")


def merge():
    alls = []
    for f in sorted(glob.glob(os.path.join(MAN_DIR, "manifest_*.json"))):
        if os.path.basename(f) == "manifest_all.json":
            continue
        alls += json.load(open(f)).get("samples", [])
    out = os.path.join(MAN_DIR, "manifest_all.json")
    json.dump({"metadata": {"num_samples": len(alls),
                            "datasets": sorted({s["dataset"] for s in alls}),
                            "tagger": TAGGER}, "samples": alls},
              open(out, "w"), indent=2)
    print(f"merged {len(alls)} samples -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bdd100k", "culane", "tusimple", "curvelanes"])
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.dataset:
        assemble(a.dataset)
    if a.merge:
        merge()


if __name__ == "__main__":
    main()
