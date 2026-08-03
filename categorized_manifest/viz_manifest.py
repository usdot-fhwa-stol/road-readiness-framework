#!/usr/bin/env python3
"""Visualize sampled-dataset images filtered by tag.

Reads a manifest (default dataset/manifests/manifest_all.json), lets you pick
tags, and renders a grid of the matching images with their tags captioned
(matched tags marked with *). Works from the CLI or inside a notebook
(see explore_tags.ipynb for an interactive ipywidgets version).

CLI examples:
  python3 viz_manifest.py --list-tags                         # tag -> image count
  python3 viz_manifest.py --tags "Roundabouts"                # grid of roundabout images
  python3 viz_manifest.py --tags "Nighttime,Glare" --mode all # images with BOTH tags
  python3 viz_manifest.py --tags Roundabouts --dataset curvelanes --limit 12 --save rb.png
"""
import argparse
import json
import math
import os
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(REPO, "dataset")
DEFAULT_MANIFEST = os.path.join(DATASET_ROOT, "manifests", "manifest_all.json")


def load(path=DEFAULT_MANIFEST):
    """Load a manifest -> list of sample dicts."""
    with open(path) as f:
        return json.load(f)["samples"]


def _labels(s):
    return (s.get("meta", {}).get("predicted_tags") or {}).get("labels", [])


def tag_counts(samples, dataset=None):
    """Counter of tag -> number of images carrying it (optionally one dataset)."""
    c = Counter()
    for s in samples:
        if dataset and s["dataset"] != dataset:
            continue
        for t in _labels(s):
            c[t] += 1
    return c


def datasets(samples):
    return sorted({s["dataset"] for s in samples})


def filter_samples(samples, tags=None, mode="any", dataset=None):
    """Keep samples matching the tag query.
       mode='any' -> has at least one of `tags`; 'all' -> has every tag."""
    want = set(tags or [])
    out = []
    for s in samples:
        if dataset and s["dataset"] != dataset:
            continue
        have = set(_labels(s))
        if want:
            if mode == "all" and not want.issubset(have):
                continue
            if mode == "any" and not (want & have):
                continue
        out.append(s)
    return out


def resolve_image(s):
    """Repo copy first, fall back to the original /shared/data source path."""
    p = os.path.join(DATASET_ROOT, s.get("image_path", ""))
    if os.path.exists(p):
        return p
    return s.get("source_image_path")


def show(samples, highlight=None, cols=4, max_n=20, figsize=None, save=None):
    """Render a grid of images with tag captions. `highlight` tags get a *mark*."""
    import matplotlib.pyplot as plt
    from PIL import Image
    hl = set(highlight or [])
    shown = samples[:max_n]
    n = len(shown)
    if n == 0:
        print("no matching images")
        return None
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (cols * 3.3, rows * 3.2))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        s = shown[i]
        path = resolve_image(s)
        try:
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception as e:
            ax.text(0.5, 0.5, f"[image unavailable]\n{e}", ha="center", va="center",
                    fontsize=7, transform=ax.transAxes)
        labels = _labels(s)
        ordered = [t for t in labels if t in hl] + [t for t in labels if t not in hl]
        marked = [(f"*{t}*" if t in hl else t) for t in ordered]
        # wrap to a narrow column so captions never bleed into the neighbour
        import textwrap
        lines = textwrap.wrap(", ".join(marked), width=34)
        extra = ""
        if len(lines) > 4:
            lines, extra = lines[:4], " …"
        cap = f"{s['dataset']} · {s.get('split','')}\n" + "\n".join(lines) + extra
        ax.set_title(cap, fontsize=6, loc="center")
    fig.subplots_adjust(hspace=0.55, wspace=0.06)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
        print(f"saved {n} image(s) -> {save}")
        plt.close(fig)
    return fig


def view(samples, tags=None, mode="any", dataset=None, cols=4, max_n=20,
         figsize=None, save=None):
    """One-call helper: filter by tags then render (highlighting the query tags)."""
    sel = filter_samples(samples, tags=tags, mode=mode, dataset=dataset)
    print(f"{len(sel)} image(s) match tags={tags or 'any'} mode={mode} "
          f"dataset={dataset or 'all'}; showing up to {max_n}")
    return show(sel, highlight=tags, cols=cols, max_n=max_n, figsize=figsize, save=save)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--tags", default="", help="comma-separated tag names")
    ap.add_argument("--mode", choices=["any", "all"], default="any")
    ap.add_argument("--dataset", default=None,
                    choices=["bdd100k", "culane", "tusimple", "curvelanes"])
    ap.add_argument("--limit", type=int, default=20, help="max images to show")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--save", default=None, help="write grid to PNG instead of showing")
    ap.add_argument("--list-tags", action="store_true", help="print tag -> count and exit")
    args = ap.parse_args()

    samples = load(args.manifest)
    if args.list_tags:
        for t, c in tag_counts(samples, dataset=args.dataset).most_common():
            print(f"{c:6d}  {t}")
        return
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    fig = view(samples, tags=tags, mode=args.mode, dataset=args.dataset,
               cols=args.cols, max_n=args.limit, save=args.save)
    if fig is not None and not args.save:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
