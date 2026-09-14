#!/usr/bin/env python3
"""PHASE 1 of the sampled-dataset build: coverage-driven sampling + tagging.

For each dataset, sample from the GT-bearing split (test if it has GT, else
val/valid), one representative frame per video scene, SHUFFLED, and tag in
batches with Gemma-4-26B-A4B (AWQ). Keep going until every taxonomy tag has
>= --min-per-tag samples AND total >= --min-total, or the pool is exhausted, or
the --cap is hit. Writes a resumable checkpoint per dataset:

    dataset/_selected/<dataset>.selected.jsonl

one line per selected image: {sample_id, dataset, split, source_image_path,
source_gt_path, gt_type, natural_gt, lane_record?, predicted_tags}. Phase 2
(assemble_manifest.py) copies files into dataset/ and builds the universal
manifest.

  python3 build_sampled_dataset.py --dataset all --count-only     # pool sizes
  python3 build_sampled_dataset.py --dataset all                  # run
"""
import argparse
import glob
import json
import os
import random
import time

from taxonomy import TAXONOMY

CULANE = "/shared/data/culane"
BDD = "/shared/data/bdd100k"
TUS = "/shared/data/TUSimple"
CURVE = "/shared/data/Curvelanes"
SEL_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "_selected")
ALL_TAGS = [t for tags in TAXONOMY.values() for t in tags]


# --------------------------------------------------------------------------- #
# Pool builders: yield candidate dicts (paths only, no tags yet), GT-bearing,
# distinct scenes, from the chosen split.
# --------------------------------------------------------------------------- #
def pool_bdd100k():
    split = "val"                                   # test has no annotations
    for ip in sorted(glob.glob(f"{BDD}/images/{split}/*.jpg")):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mask = f"{BDD}/ll_seg_annotations/{split}/{stem}.png"
        det = f"{BDD}/det_annotations/{split}/{stem}.json"
        if os.path.exists(mask):
            yield dict(sample_id=f"bdd100k_{split}_{stem}", dataset="bdd100k",
                       split=split, source_image_path=ip, source_gt_path=mask,
                       gt_type="bdd_ll_seg_mask", natural_gt="mask",
                       attr_path=det if os.path.exists(det) else None)


def pool_culane(stride=10):
    # test has only 349 distinct video segments -> too few for the 500 floor.
    # Take every `stride`-th frame per segment (~1s apart at 30fps) so the pool
    # clears 500 while staying spread across segments. Real scene diversity is
    # still ~349, so rare tags will legitimately fall short in CuLane.
    split = "test"
    test_drivers = os.listdir(f"{CULANE}/laneseg_label_w16_test")
    for driver in sorted(test_drivers):
        for seg in sorted(glob.glob(f"{CULANE}/{driver}/*.MP4")):
            frames = sorted(glob.glob(f"{seg}/*.jpg"))[::stride]
            for frame in frames:
                rel = os.path.relpath(frame, CULANE)[:-4]
                mask = f"{CULANE}/laneseg_label_w16_test/{rel}.png"
                if not os.path.exists(mask):
                    continue
                yield dict(sample_id="culane_" + rel.replace("/", "_"),
                           dataset="culane", split=split, source_image_path=frame,
                           source_gt_path=mask, gt_type="culane_laneseg_mask",
                           natural_gt="mask")


def pool_tusimple():
    split = "test"
    recs = {}
    with open(f"{TUS}/test_label.json") as f:       # test GT: lane polylines
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["raw_file"]] = r
    for raw, r in recs.items():                     # each labeled frame = a scene
        ip = f"{TUS}/test_set/{raw}"
        if os.path.exists(ip):
            sid = "tusimple_test_" + raw.replace("/", "_").replace(".jpg", "")
            yield dict(sample_id=sid, dataset="tusimple", split=split,
                       source_image_path=ip, source_gt_path=None,
                       gt_type="tusimple_lane_json", natural_gt="lanes",
                       lane_record={"lanes": r["lanes"], "h_samples": r["h_samples"]})


def pool_curvelanes():
    split = "valid"                                 # test has no labels
    for ip in sorted(glob.glob(f"{CURVE}/{split}/images/*.jpg")):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lab = f"{CURVE}/{split}/labels/{stem}.lines.json"
        if os.path.exists(lab):
            yield dict(sample_id=f"curvelanes_{split}_{stem}", dataset="curvelanes",
                       split=split, source_image_path=ip, source_gt_path=lab,
                       gt_type="curvelanes_lines_json", natural_gt="lanes")


POOLS = {"bdd100k": pool_bdd100k, "culane": pool_culane,
         "tusimple": pool_tusimple, "curvelanes": pool_curvelanes}


def load_checkpoint(path):
    done, counts, n = set(), {t: 0 for t in ALL_TAGS}, 0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                done.add(d["source_image_path"])
                n += 1
                for t in (d.get("predicted_tags") or {}).get("labels", []):
                    if t in counts:
                        counts[t] += 1
    return done, counts, n


def coverage_met(counts, n, min_per_tag, min_total):
    return n >= min_total and all(c >= min_per_tag for c in counts.values())


def stop_reason(s, args):
    if coverage_met(s["counts"], s["n"], args.min_per_tag, args.min_total):
        return "coverage"
    if s["n"] >= args.min_total and (s["n"] - s["progress_n"]) >= args.patience:
        return "stalled"
    if s["n"] >= args.cap:
        return "cap"
    if s["i"] >= len(s["pool"]):
        return "pool_exhausted"
    return None


def tag_chunk(name, backend, s, args):
    """Tag up to args.round_chunk images of this dataset (in GPU sub-batches of
    args.batch_size), updating counts + checkpoint file in place."""
    from PIL import Image, ImageOps
    chunk = s["pool"][s["i"]: s["i"] + args.round_chunk]
    s["i"] += len(chunk)
    for j in range(0, len(chunk), args.batch_size):
        sub = chunk[j:j + args.batch_size]
        imgs, keep = [], []
        for c in sub:
            try:
                imgs.append(ImageOps.exif_transpose(
                    Image.open(c["source_image_path"])).convert("RGB"))
                keep.append(c)
            except Exception as e:
                c["error"] = repr(e)
                s["f"].write(json.dumps(c) + "\n")
        if not keep:
            continue
        try:
            results = backend.predict_batch(imgs)
        except Exception:
            results = [backend.predict(im) for im in imgs]
        for c, res in zip(keep, results):
            c["predicted_tags"] = res
            helped = False
            for t in res.get("labels", []):
                if t in s["counts"]:
                    s["counts"][t] += 1
                    if s["counts"][t] == args.min_per_tag:   # tag JUST reached target
                        helped = True                         # = real coverage gain
            s["n"] += 1
            if helped:
                s["progress_n"] = s["n"]
            s["f"].write(json.dumps(c) + "\n")
        s["f"].flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all"] + list(POOLS))
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="GPU sub-batch (27B 4-bit fits ~8)")
    ap.add_argument("--round-chunk", type=int, default=50,
                    help="images to tag per dataset before rotating to the next")
    ap.add_argument("--min-per-tag", type=int, default=50)
    ap.add_argument("--min-total", type=int, default=500)
    ap.add_argument("--patience", type=int, default=600,
                    help="stop a dataset after this many new samples with no progress "
                         "on any under-target tag (>= min-total) -- don't chase rare tags")
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()

    names = list(POOLS) if args.dataset == "all" else [args.dataset]

    if args.count_only:
        for nm in names:
            print(f"[{nm}] pool size = {sum(1 for _ in POOLS[nm]())}", flush=True)
        return

    os.makedirs(SEL_DIR, exist_ok=True)
    from backends import make_backend
    print(f"loading {args.model} (4-bit) ...", flush=True)
    backend = make_backend("vlm", args.model, "cuda", load_4bit=True)
    print("model ready", flush=True)

    # per-dataset state (resumable from each dataset's checkpoint)
    states = {}
    for nm in names:
        out = os.path.join(SEL_DIR, f"{nm}.selected.jsonl")
        done, counts, n = load_checkpoint(out)
        pool = [c for c in POOLS[nm]() if c["source_image_path"] not in done]
        random.seed(args.seed)
        random.shuffle(pool)
        states[nm] = dict(out=out, counts=counts, n=n, pool=pool, i=0,
                          progress_n=n, f=open(out, "a"), finished=False)
        print(f"[{nm}] pool={len(pool)+len(done)} resume={len(done)} "
              f"target >={args.min_per_tag}/tag & >={args.min_total} total, "
              f"cap={args.cap}", flush=True)

    # ROUND-ROBIN: tag round-chunk images per dataset, rotate, repeat until all met
    rnd = 0
    while any(not s["finished"] for s in states.values()):
        rnd += 1
        for nm in names:
            s = states[nm]
            if s["finished"]:
                continue
            r = stop_reason(s, args)
            if r:
                s["finished"] = True
                low = {t: c for t, c in s["counts"].items() if c < args.min_per_tag}
                print(f"[{nm}] DONE_SELECT n={s['n']} stop={r} "
                      f"under-{args.min_per_tag}={low}", flush=True)
                continue
            tag_chunk(nm, backend, s, args)
            miss = sum(1 for v in s["counts"].values() if v < args.min_per_tag)
            print(f"[round {rnd}][{nm}] n={s['n']} "
                  f"tags<{args.min_per_tag}: {miss}/{len(ALL_TAGS)} "
                  f"stall={s['n']-s['progress_n']}", flush=True)

    for s in states.values():
        s["f"].close()
    print("ALL_SELECT_DONE", flush=True)


if __name__ == "__main__":
    main()
