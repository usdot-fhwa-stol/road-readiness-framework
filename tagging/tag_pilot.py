#!/usr/bin/env python3
"""Batched tagging of a manifest (e.g. output/pilot_manifest.json) with Gemma-12B.
Writes predicted_tags back into each sample so eval_tagger.py can score the rows
that have human tags. Rows without a resolvable image are marked skipped.

  python3 tag_pilot.py --in output/pilot_manifest.json \
      --out output/pilot_manifest.tagged.json --batch-size 16
"""
import argparse
import json
import time

from backends import make_backend
from PIL import Image, ImageOps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="output/pilot_manifest.json")
    ap.add_argument("--out", dest="out", default="output/pilot_manifest.tagged.json")
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    manifest = json.load(open(args.inp))
    samples = manifest["samples"]
    print(f"loading {args.model} (4bit) bs={args.batch_size} ...", flush=True)
    backend = make_backend("vlm", args.model, "cuda", load_4bit=True)
    print("model ready", flush=True)

    # gather taggable (sample, image); mark the rest skipped
    todo = []
    skipped = 0
    for s in samples:
        ip = s.get("image_path")
        if not ip or not s.get("image_found", ip is not None):
            s["predicted_tags"] = None
            s["predicted_note"] = "no resolvable image"
            skipped += 1
            continue
        try:
            img = ImageOps.exif_transpose(Image.open(ip)).convert("RGB")
        except Exception as e:
            s["predicted_tags"] = None
            s["predicted_note"] = f"open failed: {e}"
            skipped += 1
            continue
        todo.append((s, img))

    n, t0 = 0, time.time()
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        try:
            results = backend.predict_batch([im for _, im in chunk])
        except Exception:
            results = [backend.predict(im) for _, im in chunk]
        for (s, _), res in zip(chunk, results):
            s["predicted_tags"] = res
        n += len(chunk)
        dt = time.time() - t0
        print(f"  tagged {n}/{len(todo)}  {dt / n:.2f}s/img", flush=True)

    manifest.setdefault("metadata", {})["tagger"] = {
        "model": args.model, "batch_size": args.batch_size,
        "num_tagged": n, "num_skipped": skipped,
    }
    json.dump(manifest, open(args.out, "w"), indent=2)
    print(f"DONE tagged={n} skipped={skipped} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
