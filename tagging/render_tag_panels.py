"""Side-by-side visualization: image | dataset/name/tags text panel.

Reads records from tagging/full_tags/<dataset>.tags.jsonl
(one JSON object per line, produced by tag_all.py) and renders, for each
sampled image:

    [ image ] [ dataset, image name, and predicted tags grouped by dimension ]

Outputs one PNG per sample under <out-dir>/panels/, plus a single stacked
"gallery" PNG (<out-dir>/gallery.png) for quick browsing.

Before drawing, conflicting "Observed Marking Visibility" tags are resolved
(e.g. "Clearly Visible" alongside "Faded / Worn / Not Visible") using the same
occluded > degraded > clear precedence evaluation/d_metrics.py already applies
for D5 grouping. Every resolved conflict is written to
<out-dir>/tag_conflicts.jsonl for review.

Usage:
    python3 tagging/render_tag_panels.py --dataset bdd100k --limit 12
    python3 tagging/render_tag_panels.py --dataset all --limit 20 --seed 7

    # or point straight at any tagged manifest (jsonl, or {"samples": [...]} json)
    python3 tagging/render_tag_panels.py \\
        --dataset tagging/output/pilot_manifest.tagged.json --limit 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

TAGS_DIR = Path(__file__).parent / "full_tags"

PANEL_IMG_HEIGHT = 360   # rendered height of the source image
TEXT_PANEL_WIDTH = 480
PAD = 14
LINE_GAP = 4
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)
DIM_COLOR = (90, 90, 90)
HEADER_COLOR = (10, 60, 140)


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_HEADER = _font(20)
F_LABEL = _font(15)
F_BODY = _font(14)


def _load_file(path: Path) -> list[dict]:
    """Load records from either a *.jsonl (one record/line) or a
    {"metadata": ..., "samples": [...]} manifest json (or a bare json list)."""
    if path.suffix == ".jsonl":
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unrecognized manifest format: {path}")


def load_records(dataset: str) -> list[dict]:
    direct = Path(dataset)
    if direct.is_file():
        return _load_file(direct)
    paths = [TAGS_DIR / f"{dataset}.tags.jsonl"] if dataset != "all" else sorted(TAGS_DIR.glob("*.tags.jsonl"))
    records = []
    for p in paths:
        if p.exists():
            records.extend(_load_file(p))
    return records


# Mirrors evaluation/d_metrics.py's visibility-tag precedence (occluded >
# degraded > clear), kept as local literals so this stays a standalone
# visualization script. Keep in sync with d_metrics._OCCLUDED_TAGS /
# _DEGRADED_TAGS / _CLEAR_TAGS if that taxonomy changes.
_VISIBILITY_DIM = "Observed Marking Visibility"
_OCCLUDED_TAGS = {"occluded", "partially missing"}
_DEGRADED_TAGS = {"faded / worn / not visible", "low contrast"} | _OCCLUDED_TAGS
_CLEAR_TAGS = {"clearly visible"}


def resolve_tag_conflicts(record: dict) -> tuple[dict, Optional[dict]]:
    """Drop "Clearly Visible" from a record's visibility tags when a
    degraded/occluded visibility tag is also present -- they're contradictory.

    Returns (by_dimension dict to render, conflict-log-entry or None).
    """
    by_dim = (record.get("predicted_tags") or {}).get("by_dimension") or {}
    vis_tags = by_dim.get(_VISIBILITY_DIM)
    if not vis_tags:
        return by_dim, None

    degraded_hits = [t for t in vis_tags if t.strip().lower() in _DEGRADED_TAGS]
    clear_hits = [t for t in vis_tags if t.strip().lower() in _CLEAR_TAGS]
    if not (degraded_hits and clear_hits):
        return by_dim, None

    resolved = dict(by_dim)
    resolved[_VISIBILITY_DIM] = [t for t in vis_tags if t not in clear_hits]
    conflict = {
        "sample_id": record.get("sample_id"),
        "dataset": record.get("dataset"),
        "image_path": record.get("image_path"),
        "dimension": _VISIBILITY_DIM,
        "original_tags": vis_tags,
        "dropped": clear_hits,
        "kept": resolved[_VISIBILITY_DIM],
    }
    return resolved, conflict


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def render_panel(record: dict, by_dimension: Optional[dict] = None) -> Optional[Image.Image]:
    if not record.get("image_path"):
        return None
    img_path = Path(record["image_path"])
    if not img_path.exists():
        return None
    src = Image.open(img_path).convert("RGB")
    scale = PANEL_IMG_HEIGHT / src.height
    img = src.resize((max(1, round(src.width * scale)), PANEL_IMG_HEIGHT))

    canvas = Image.new("RGB", (img.width + TEXT_PANEL_WIDTH, PANEL_IMG_HEIGHT), BG_COLOR)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    text_x = img.width + PAD
    text_w = TEXT_PANEL_WIDTH - 2 * PAD
    y = PAD

    draw.text((text_x, y), record.get("dataset", ""), font=F_HEADER, fill=HEADER_COLOR)
    y += F_HEADER.size + LINE_GAP + 2

    for line in _wrap(f"{img_path.name}  ({record.get('split', '?')})", F_LABEL, text_w, draw):
        draw.text((text_x, y), line, font=F_LABEL, fill=DIM_COLOR)
        y += F_LABEL.size + LINE_GAP
    y += 8

    by_dim = by_dimension if by_dimension is not None else (record.get("predicted_tags") or {}).get("by_dimension") or {}
    for dim, tags in by_dim.items():
        if y > PANEL_IMG_HEIGHT - 20:
            draw.text((text_x, y), "...", font=F_BODY, fill=DIM_COLOR)
            break
        for line in _wrap(dim + ":", F_LABEL, text_w, draw):
            draw.text((text_x, y), line, font=F_LABEL, fill=TEXT_COLOR)
            y += F_LABEL.size + LINE_GAP
        for line in _wrap(", ".join(tags) if tags else "-", F_BODY, text_w, draw):
            draw.text((text_x + 10, y), line, font=F_BODY, fill=DIM_COLOR)
            y += F_BODY.size + LINE_GAP
        y += 6

    draw.line([(img.width, 0), (img.width, PANEL_IMG_HEIGHT)], fill=(210, 210, 210), width=1)
    return canvas


def build_gallery(panels: list[Image.Image]) -> Image.Image:
    if not panels:
        raise ValueError("no panels to build a gallery from")
    width = max(p.width for p in panels)
    total_h = sum(p.height for p in panels) + 6 * (len(panels) - 1)
    gallery = Image.new("RGB", (width, total_h), (235, 235, 235))
    y = 0
    for p in panels:
        gallery.paste(p, (0, y))
        y += p.height + 6
    return gallery


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                     help="bdd100k | culane | tusimple | curvelanes | all, "
                          "or a direct path to a tagged manifest (.jsonl or .json)")
    ap.add_argument("--limit", type=int, default=12, help="number of images to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="tagging/output/tag_panels")
    ap.add_argument("--no-gallery", action="store_true", help="skip the stacked gallery.png")
    args = ap.parse_args()

    records = load_records(args.dataset)
    if not records:
        raise SystemExit(f"no tag records found for dataset={args.dataset!r} under {TAGS_DIR}")

    random.seed(args.seed)
    sample = random.sample(records, k=min(args.limit, len(records)))

    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    conflicts = []
    for rec in sample:
        by_dim, conflict = resolve_tag_conflicts(rec)
        if conflict:
            conflicts.append(conflict)
        panel = render_panel(rec, by_dimension=by_dim)
        if panel is None:
            print(f"skip (image missing): {rec.get('image_path')}")
            continue
        out_path = panels_dir / f"{rec['sample_id']}.png"
        panel.save(out_path)
        rendered.append(panel)
        print(f"wrote {out_path}")

    if rendered and not args.no_gallery:
        gallery_path = out_dir / "gallery.png"
        build_gallery(rendered).save(gallery_path)
        print(f"wrote {gallery_path}")

    if conflicts:
        conflicts_path = out_dir / "tag_conflicts.jsonl"
        with conflicts_path.open("w") as f:
            for c in conflicts:
                f.write(json.dumps(c) + "\n")
        print(f"wrote {conflicts_path} ({len(conflicts)} conflicting-tag records resolved)")
    else:
        print("no conflicting visibility tags found")

    print(f"{len(rendered)}/{len(sample)} panels rendered from {len(records)} available records")


if __name__ == "__main__":
    main()
