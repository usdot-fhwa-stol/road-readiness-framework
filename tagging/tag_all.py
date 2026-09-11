#!/usr/bin/env python3
"""Tag EVERY image in all 4 datasets (all splits) with Gemma-3-27B and write a
resumable JSONL per dataset (one line per image) for tag-based grouping.

Output: full_tags/<dataset>.tags.jsonl  -- append-safe + resumable (re-run to
continue where it stopped; already-tagged image_paths are skipped).

Each line: {sample_id, dataset, split, image_path, ground_truth_path, gt_found,
gt_type, labels:[...], predicted_tags:{by_dimension,labels,scores}}.

Group later with pandas:
  df = pd.read_json('full_tags/culane.tags.jsonl', lines=True)
  df.explode('labels').groupby('labels')['image_path'].apply(list)

Usage:
  python3 tag_all.py --dataset all                 # full run
  python3 tag_all.py --dataset culane --limit 15   # quick benchmark/test
"""
import argparse
import glob
import json
import os
import time

from backends import make_backend          # imports torchaudio stub first
from taxonomy import TAXONOMY

CULANE = "/shared/data/culane"
BDD = "/shared/data/bdd100k"
TUS = "/shared/data/TUSimple"
CURVE = "/shared/data/Curvelanes"
OUT_DIR = os.path.join(os.path.dirname(__file__), "full_tags")


# --------------------------------------------------------------------------- #
# Enumerators: yield one record (paths resolved, tags not yet filled) per image
# --------------------------------------------------------------------------- #
def enum_bdd100k():
    for split in ("train", "val", "test"):
        for ip in sorted(glob.glob(f"{BDD}/images/{split}/*.jpg")):
            name = os.path.splitext(os.path.basename(ip))[0]
            gt, found = None, False
            if split in ("train", "val"):
                g = f"{BDD}/ll_seg_annotations/{split}/{name}.png"
                if os.path.exists(g):
                    gt, found = g, True
            yield dict(sample_id=f"bdd100k_{split}_{name}", dataset="bdd100k",
                       split=split, image_path=ip, ground_truth_path=gt,
                       gt_found=found, gt_type="bdd_ll_seg_mask")


def enum_culane():
    test_drivers = set(os.listdir(f"{CULANE}/laneseg_label_w16_test"))
    for ip in sorted(glob.glob(f"{CULANE}/driver_*/*.MP4/*.jpg")):
        rel = os.path.relpath(ip, CULANE)            # driver/seg.MP4/frame.jpg
        driver = rel.split(os.sep)[0]
        split = "test" if driver in test_drivers else "train"
        lane = "laneseg_label_w16_test" if split == "test" else "laneseg_label_w16"
        gt = f"{CULANE}/{lane}/{rel[:-4]}.png"       # swap .jpg -> .png under laneseg
        found = os.path.exists(gt)
        if not found:
            gt = None
        yield dict(sample_id="culane_" + rel[:-4].replace("/", "_"),
                   dataset="culane", split=split, image_path=ip,
                   ground_truth_path=gt, gt_found=found,
                   gt_type="culane_laneseg_mask")


def enum_tusimple():
    for setname, split in (("train_set", "train"), ("test_set", "test")):
        base = f"{TUS}/{setname}"
        for ip in sorted(glob.glob(f"{base}/clips/*/*/*.jpg")):
            frame = os.path.splitext(os.path.basename(ip))[0]
            reldir = os.path.relpath(os.path.dirname(ip), f"{base}/clips")
            gt, found = None, False
            if setname == "train_set" and frame == "20":
                g = f"{base}/seg_label/{reldir}/20.png"
                if os.path.exists(g):
                    gt, found = g, True
            yield dict(sample_id=f"tusimple_{split}_{reldir.replace('/', '_')}_{frame}",
                       dataset="tusimple", split=split, image_path=ip,
                       ground_truth_path=gt, gt_found=found,
                       gt_type="tusimple_seg_label")


def enum_curvelanes():
    for split in ("train", "valid", "test"):
        for ip in sorted(glob.glob(f"{CURVE}/{split}/images/*.jpg")):
            name = os.path.splitext(os.path.basename(ip))[0]
            gt, found = None, False
            if split in ("train", "valid"):
                g = f"{CURVE}/{split}/labels/{name}.lines.json"
                if os.path.exists(g):
                    gt, found = g, True
            yield dict(sample_id=f"curvelanes_{split}_{name}", dataset="curvelanes",
                       split=split, image_path=ip, ground_truth_path=gt,
                       gt_found=found, gt_type="curvelanes_lines_json")


ENUM = {"bdd100k": enum_bdd100k, "culane": enum_culane,
        "tusimple": enum_tusimple, "curvelanes": enum_curvelanes}


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_path"])
                except Exception:
                    pass
    return done


def _flush_batch(backend, pending, f):
    """Tag one batch of (rec, image); write a JSONL line per rec. Returns
    (n_written, n_errors). Falls back to per-image on a batch-level failure so
    one bad image/OOM can't sink the whole batch."""
    if not pending:
        return 0, 0
    imgs = [im for _, im in pending]
    try:
        results = backend.predict_batch(imgs)
    except Exception:
        results = []
        for _, im in pending:
            try:
                results.append(backend.predict(im))
            except Exception as e2:
                results.append({"error": repr(e2)})
    errs = 0
    for (rec, _), res in zip(pending, results):
        if res.get("labels"):
            rec["predicted_tags"] = res
            rec["labels"] = res["labels"]
        else:
            rec["error"] = res.get("error", "empty_prediction")
            rec["predicted_tags"] = res
            rec["labels"] = res.get("labels", [])
            errs += 1
        f.write(json.dumps(rec) + "\n")
    f.flush()
    return len(pending), errs


def run_dataset(name, backend, limit=0, splits=None, batch_size=1):
    from PIL import Image
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{name}.tags.jsonl")
    done = load_done(out)
    print(f"[{name}] resuming: {len(done)} already tagged (bs={batch_size}) "
          f"-> {out}", flush=True)
    n, errors, t0 = 0, 0, time.time()
    pending = []
    with open(out, "a") as f:
        for rec in ENUM[name]():
            if splits and rec["split"] not in splits:
                continue
            if limit and n + len(pending) >= limit:
                break
            if rec["image_path"] in done:
                continue
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
            except Exception as e:
                rec["error"] = repr(e)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                errors += 1
                continue
            pending.append((rec, img))
            if len(pending) >= batch_size:
                w, e = _flush_batch(backend, pending, f)
                n += w
                errors += e
                pending = []
                dt = time.time() - t0
                print(f"[{name}] {n} tagged, {dt / n:.2f}s/img, {errors} errors",
                      flush=True)
        w, e = _flush_batch(backend, pending, f)   # tail
        n += w
        errors += e
    dt = time.time() - t0
    rate = dt / n if n else 0
    print(f"[{name}] DONE_DATASET tagged={n} errors={errors} "
          f"{rate:.2f}s/img total={dt:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all", "bdd100k", "culane", "tusimple", "curvelanes"])
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="images per batched forward pass (see bench: 12B fits 16+)")
    ap.add_argument("--limit", type=int, default=0,
                    help="tag only N new images (per dataset) -- for benchmarking")
    ap.add_argument("--splits", default="",
                    help="comma-separated splits to include, e.g. 'test,val' "
                         "(empty = all splits)")
    ap.add_argument("--count-only", action="store_true",
                    help="dry-run: enumerate + count per split, no model/tagging")
    args = ap.parse_args()

    splits = {s.strip() for s in args.splits.split(",") if s.strip()} or None
    datasets = list(ENUM) if args.dataset == "all" else [args.dataset]

    if args.count_only:
        from collections import Counter
        grand = 0
        for ds in datasets:
            c = Counter(r["split"] for r in ENUM[ds]())
            sel = sum(v for k, v in c.items() if not splits or k in splits)
            grand += sel
            print(f"[{ds}] per-split {dict(c)} | selected({splits or 'all'})={sel}",
                  flush=True)
        print(f"TOTAL selected images: {grand}", flush=True)
        return

    print(f"loading {args.model} (4bit) ...", flush=True)
    backend = make_backend("vlm", args.model, "cuda", load_4bit=True)
    print("model ready", flush=True)

    for ds in datasets:
        run_dataset(ds, backend, limit=args.limit, splits=splits,
                    batch_size=args.batch_size)
    print("ALL_DATASETS_DONE", flush=True)


if __name__ == "__main__":
    main()
