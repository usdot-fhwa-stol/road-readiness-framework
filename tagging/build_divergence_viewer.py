#!/usr/bin/env python3
"""Build a self-contained HTML viewer of the cases where two tagging runs
disagree, so a human can adjudicate the hard ones (task b).

For every image both runs tagged, it computes the per-dimension disagreement,
ranks images by total disagreement, and renders each as a card: the image
(embedded as base64 so the file is portable), and a 6-row table with each
model's tags + "why" reasoning side by side, disagreeing dimensions highlighted.

Usage:
  python3 tagging/build_divergence_viewer.py \
      --a dataset/gateway_tags        --a-name mini \
      --b dataset/gateway_tags_sonnet --b-name sonnet \
      --image-root dataset --out dataset/divergence_viewer.html --top 120
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import TAXONOMY  # noqa: E402

DATASETS = ["bdd100k", "culane", "curvelanes", "tusimple"]


def load_run(run_dir):
    out = {}
    for ds in DATASETS:
        p = Path(run_dir) / f"{ds}.tags.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            pt = d.get("predicted_tags") or {}
            bd = pt.get("by_dimension") or {}
            out[d["sample_id"]] = {
                "dataset": d.get("dataset"), "split": d.get("split"),
                "image_path": d.get("image_path"),
                "by_dim": {dim: list(bd.get(dim, [])) for dim in TAXONOMY},
                "reasoning": pt.get("reasoning") or {},
            }
    return out


def thumb_data_uri(abs_path, max_side=560):
    try:
        im = ImageOps.exif_transpose(Image.open(abs_path)).convert("RGB")
    except Exception:
        return None
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def n_disagree(a, b):
    return sum(1 for dim in TAXONOMY if set(a["by_dim"][dim]) != set(b["by_dim"][dim]))


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
header{position:sticky;top:0;background:#1f2933;color:#fff;padding:14px 22px;z-index:10;box-shadow:0 2px 6px rgba(0,0,0,.2)}
header h1{margin:0;font-size:18px}
header .sub{font-size:13px;opacity:.8;margin-top:4px}
.controls{padding:10px 22px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:56px;z-index:9}
.controls label{font-size:13px;margin-right:16px;cursor:pointer}
.card{background:#fff;margin:16px 22px;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.12);overflow:hidden}
.card .top{display:flex;gap:16px;padding:14px}
.card img{max-width:560px;width:100%;height:auto;border-radius:6px;background:#000}
.meta{flex:1;min-width:340px}
.meta .id{font-family:monospace;font-size:12px;color:#555;word-break:break-all}
.badge{display:inline-block;background:#e11d48;color:#fff;border-radius:12px;padding:1px 10px;font-size:12px;margin-left:8px}
.ds{display:inline-block;background:#3b82f6;color:#fff;border-radius:4px;padding:1px 8px;font-size:11px;text-transform:uppercase}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid #e3e3e3;padding:6px 9px;vertical-align:top;text-align:left}
th{background:#f0f2f5;font-weight:600}
tr.diff td{background:#fff7ed}
tr.diff .dim{border-left:3px solid #e11d48}
.tags{font-weight:600}
.why{color:#666;font-size:12px;margin-top:3px;font-style:italic}
.agree .dim{border-left:3px solid #16a34a}
.colA{color:#1d4ed8}.colB{color:#7c3aed}
.hide{display:none}
.legend{font-size:12px}
"""

JS = """
function applyFilter(){
  var only=document.getElementById('onlyDiff').checked;
  document.querySelectorAll('.card').forEach(function(c){
    var d=parseInt(c.dataset.ndiff||'0',10);
    c.classList.toggle('hide', only && d===0);
  });
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--image-root", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("dataset/divergence_viewer.html"))
    ap.add_argument("--top", type=int, default=120,
                    help="render the N most-divergent shared images")
    args = ap.parse_args()

    A, B = load_run(args.a), load_run(args.b)
    na, nb = args.a_name, args.b_name
    shared = sorted(set(A) & set(B))
    ranked = sorted(shared, key=lambda s: -n_disagree(A[s], B[s]))
    sel = ranked[: args.top]
    n_any = sum(1 for s in shared if n_disagree(A[s], B[s]) > 0)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{na} vs {nb} divergence</title><style>{CSS}</style></head><body>",
        "<header><h1>Tagger divergence viewer &mdash; "
        f"<span class='colA'>{html.escape(na)}</span> vs "
        f"<span class='colB'>{html.escape(nb)}</span></h1>",
        f"<div class='sub'>{len(shared)} shared images &middot; "
        f"{n_any} with any disagreement &middot; showing top {len(sel)} "
        "by # disagreeing dimensions. "
        "<span class='legend'>Rows: <span style='color:#e11d48'>&#9646;</span> disagree, "
        "<span style='color:#16a34a'>&#9646;</span> agree.</span></div></header>",
        "<div class='controls'><label><input type='checkbox' id='onlyDiff' "
        "onchange='applyFilter()' checked> show only rows... (cards always show; "
        "toggle hides fully-agreeing cards)</label></div>",
    ]

    for sid in sel:
        a, b = A[sid], B[sid]
        nd = n_disagree(a, b)
        abs_img = args.image_root / a["image_path"]
        uri = thumb_data_uri(str(abs_img))
        img_tag = (f"<img src='{uri}'>" if uri
                   else f"<div style='color:#a00'>image missing: "
                        f"{html.escape(str(abs_img))}</div>")
        rows = []
        for dim in TAXONOMY:
            sa, sb = set(a["by_dim"][dim]), set(b["by_dim"][dim])
            diff = sa != sb
            cls = "diff" if diff else "agree"
            ra = html.escape(a["reasoning"].get(dim, ""))
            rb = html.escape(b["reasoning"].get(dim, ""))
            rows.append(
                f"<tr class='{cls}'><td class='dim'>{html.escape(dim)}</td>"
                f"<td><div class='tags colA'>{html.escape(', '.join(sorted(sa)) or '—')}</div>"
                f"<div class='why'>{ra}</div></td>"
                f"<td><div class='tags colB'>{html.escape(', '.join(sorted(sb)) or '—')}</div>"
                f"<div class='why'>{rb}</div></td></tr>")
        parts.append(
            f"<div class='card' data-ndiff='{nd}'>"
            f"<div class='top'>{img_tag}"
            f"<div class='meta'><span class='ds'>{html.escape(a['dataset'] or '')}</span>"
            f"<span class='badge'>{nd} / 6 dims differ</span>"
            f"<div class='id'>{html.escape(sid)}</div>"
            "<table><tr><th>Dimension</th>"
            f"<th class='colA'>{html.escape(na)}</th>"
            f"<th class='colB'>{html.escape(nb)}</th></tr>"
            + "".join(rows) + "</table></div></div></div>")

    parts.append(f"<script>{JS}applyFilter();</script></body></html>")
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}  ({len(sel)} cards, {n_any}/{len(shared)} "
          f"images had disagreement)")


if __name__ == "__main__":
    main()
