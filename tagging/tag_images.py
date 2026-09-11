#!/usr/bin/env python3
"""Auto-tag road images and write an updated manifest whose samples carry both
the resolved image / ground-truth locations and the predicted tags, so results
can be sorted/grouped by tag.

Input  : a manifest from build_categorized_manifest.py (samples have
         `image_path`, `ground_truth_path`, and optional human `tags`).
Output : the same manifest with `predicted_tags` added to every sample:
             {"by_dimension": {...}, "labels": [...], "scores": {...}}
Human `tags` (if present) are preserved untouched for evaluation.

Backends (--backend):
  siglip : google/siglip2-base-patch16-224 zero-shot (fast; weak on the fine
           semantic dimensions).
  vlm    : Qwen/Qwen2.5-VL-7B-Instruct generative, prompted with the taxonomy
           (stronger on the semantic dimensions; slower, needs GPU).

Usage:
  python3 tagging/tag_images.py --backend vlm \
      --in-manifest  tagging/output/manifest_bdd100k.json \
      --out-manifest tagging/output/manifest_bdd100k.vlm.json
  # optional: --limit N, --model <hf id>, --device cuda|cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from backends import make_backend, SIGLIP_DEFAULT, VLM_DEFAULT


def _image_for_sample(sample):
    """Resolve the file to feed the model; CuLane samples point at a segment
    directory, so use their representative frame."""
    p = sample.get("image_path")
    if not p:
        return None
    path = Path(p)
    if path.is_dir():
        rep = sample.get("representative_frame")
        return Path(rep) if rep else None
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-manifest", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--backend", choices=["siglip", "vlm", "internvl", "phi4"], default="siglip")
    ap.add_argument("--model", default=None, help="override HF model id for the backend")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-4bit", action="store_true", help="load VLM in 4-bit (bnb nf4) for big models")
    ap.add_argument("--limit", type=int, default=0, help="tag only first N samples (0 = all)")
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    manifest = json.loads(args.in_manifest.read_text())
    samples = manifest["samples"]
    if args.limit:
        samples = samples[: args.limit]

    # None lets make_backend pick each backend's own default (internvl/phi4 have
    # their own); only siglip/vlm get an explicit default here.
    _default = {"siglip": SIGLIP_DEFAULT, "vlm": VLM_DEFAULT}.get(args.backend)
    model_id = args.model or _default
    print(f"loading backend={args.backend} model={model_id} device={args.device} "
          f"4bit={args.load_4bit} ...")
    backend = make_backend(args.backend, model_id, args.device, load_4bit=args.load_4bit)

    tagged = skipped = 0
    for s in samples:
        img_path = _image_for_sample(s)
        if img_path is None or not img_path.exists():
            s["predicted_tags"] = None
            s["predicted_note"] = "no resolvable image to tag"
            skipped += 1
            continue
        try:
            image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            s["predicted_tags"] = None
            s["predicted_note"] = f"could not open image: {e}"
            skipped += 1
            continue
        s["predicted_tags"] = backend.predict(image)
        tagged += 1
        if tagged % 10 == 0:
            print(f"  tagged {tagged} ...")

    manifest["samples"] = samples
    manifest.setdefault("metadata", {})["tagger"] = {
        "backend": args.backend, "model": model_id,
        "num_tagged": tagged, "num_skipped": skipped,
    }
    args.out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"tagged={tagged} skipped={skipped} -> {args.out_manifest}")


if __name__ == "__main__":
    main()
