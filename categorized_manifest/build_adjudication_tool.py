#!/usr/bin/env python3
"""Build a self-contained HTML tool for HUMAN adjudication of the tags (task 9).

Model-vs-model agreement is a weak proxy for correctness (a rewrite can raise
quality yet lower agreement, and vice-versa). The only reliable signal is a
human deciding against the image. This tool surfaces the cases that most need a
human -- where two runs disagree on a dimension -- and lets you click the
correct tag set per dimension. It EXPORTS your verdicts as JSON (download button)
which feeds back in two ways:
  1. score which model / definition set is actually right (ground truth), and
  2. become curated few-shot examples for in-context learning.

Images are embedded as base64 so the file is portable (open by double-click; no
server, no path issues on OneDrive).

Usage:
  python3 categorized_manifest/build_adjudication_tool.py \
      --a dataset/gateway_tags_v3_mini --a-name mini \
      --b dataset/gateway_tags_v2_sonnet --b-name sonnet \
      --image-root dataset --out dataset/adjudicate.html \
      --max-images 60           # cap how many images to review
      --dimensions "Pavement Marking Type & Configuration,Observed Marking Visibility"
                                # optionally focus only on the hard dimensions
"""
from __future__ import annotations

import argparse
import base64
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
                "dataset": d.get("dataset"), "image_path": d.get("image_path"),
                "by_dim": {dim: list(bd.get(dim, [])) for dim in TAXONOMY},
                "reasoning": pt.get("reasoning") or {},
            }
    return out


def thumb_data_uri(abs_path, max_side=640):
    try:
        im = ImageOps.exif_transpose(Image.open(abs_path)).convert("RGB")
    except Exception:
        return None
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#eef1f4;color:#16202b}
header{position:sticky;top:0;background:#0f2f4f;color:#fff;padding:12px 20px;z-index:20;box-shadow:0 2px 8px rgba(0,0,0,.25)}
header h1{margin:0;font-size:17px}
header .sub{font-size:12px;opacity:.85;margin-top:3px}
#bar{position:sticky;top:52px;background:#fff;border-bottom:1px solid #d5d5d5;padding:8px 20px;z-index:19;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
#bar button{background:#0f2f4f;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:13px;cursor:pointer}
#bar button.ghost{background:#e2e8f0;color:#0f2f4f}
#prog{font-size:13px;color:#333}
.card{background:#fff;margin:16px 20px;border-radius:10px;box-shadow:0 1px 5px rgba(0,0,0,.12);overflow:hidden}
.card .hd{display:flex;justify-content:space-between;padding:8px 14px;background:#f4f6f9;border-bottom:1px solid #e3e3e3;font-size:12px}
.id{font-family:monospace;color:#555;word-break:break-all}
.ds{background:#2563eb;color:#fff;border-radius:4px;padding:1px 8px;text-transform:uppercase;font-size:11px}
.body{display:flex;gap:16px;padding:14px}
.body img{max-width:640px;width:100%;border-radius:6px;background:#000}
.dims{flex:1;min-width:360px}
.dim{border:1px solid #e5e7eb;border-radius:8px;margin-bottom:12px;overflow:hidden}
.dim.settled{opacity:.55}
.dim h3{margin:0;padding:7px 11px;font-size:13px;background:#f0f3f7;border-bottom:1px solid #e5e7eb}
.dim h3 .tick{color:#16a34a;font-weight:700}
.props{display:flex;gap:8px;padding:8px 11px;font-size:12px;flex-wrap:wrap}
.prop{flex:1;min-width:150px;background:#fafbfc;border:1px solid #eee;border-radius:6px;padding:6px 8px}
.prop .who{font-weight:700;font-size:11px;text-transform:uppercase}
.who.a{color:#1d4ed8}.who.b{color:#7c3aed}
.why{color:#666;font-style:italic;margin-top:3px}
.opts{padding:6px 11px 10px;display:flex;flex-wrap:wrap;gap:6px}
.opts label{font-size:12px;background:#eef2f7;border:1px solid #dbe2ea;border-radius:14px;padding:3px 10px;cursor:pointer;user-select:none}
.opts label.picked{background:#16a34a;color:#fff;border-color:#16a34a}
.opts label.diffA{box-shadow:inset 0 0 0 2px #1d4ed8}
.opts label.diffB{box-shadow:inset 0 0 0 2px #7c3aed}
.act{padding:6px 11px 11px;display:flex;gap:8px}
.act button{font-size:12px;border:0;border-radius:6px;padding:5px 10px;cursor:pointer}
.usebtn{background:#1d4ed8;color:#fff}.usebtn.b{background:#7c3aed}
.unsure{background:#f59e0b;color:#fff}
.clearb{background:#e2e8f0;color:#333}
.legend{font-size:11px;color:#eee}
"""

JS = r"""
// STATE[sid][dim] = {tags:[...], unsure:bool}
var STATE = {};
function keyDim(el){return {sid:el.dataset.sid, dim:el.dataset.dim};}
function ensure(sid,dim){STATE[sid]=STATE[sid]||{}; STATE[sid][dim]=STATE[sid][dim]||{tags:[],unsure:false}; return STATE[sid][dim];}
function toggleTag(el){
  var sid=el.dataset.sid, dim=el.dataset.dim, tag=el.dataset.tag;
  var st=ensure(sid,dim); st.unsure=false;
  var i=st.tags.indexOf(tag);
  if(i>=0){st.tags.splice(i,1);} else {st.tags.push(tag);}
  render(sid,dim);
}
function useSet(sid,dim,which){
  var arr = JSON.parse(document.getElementById('set_'+which+'_'+cssid(sid)+'_'+cssid(dim)).textContent);
  var st=ensure(sid,dim); st.tags=arr.slice(); st.unsure=false; render(sid,dim);
}
function markUnsure(sid,dim){var st=ensure(sid,dim); st.unsure=true; st.tags=[]; render(sid,dim);}
function clearDim(sid,dim){STATE[sid]&&delete STATE[sid][dim]; render(sid,dim);}
function cssid(s){return s.replace(/[^a-zA-Z0-9_]/g,'_');}
function render(sid,dim){
  var box=document.getElementById('dim_'+cssid(sid)+'_'+cssid(dim));
  var st=(STATE[sid]||{})[dim];
  box.querySelectorAll('.opts label').forEach(function(l){
    l.classList.toggle('picked', !!(st&&!st.unsure&&st.tags.indexOf(l.dataset.tag)>=0));
  });
  var tick=box.querySelector('.tick');
  tick.textContent = st? (st.unsure? '  ⚑ ask me' : (st.tags.length? '  ✓ '+st.tags.length : '')) : '';
  box.classList.toggle('settled', !!st);
  updateProgress();
}
function updateProgress(){
  var settled=0;
  Object.keys(STATE).forEach(function(sid){Object.keys(STATE[sid]).forEach(function(){});});
  document.querySelectorAll('.dim').forEach(function(b){
    var sid=b.dataset.sid, dim=b.dataset.dim;
    if(STATE[sid]&&STATE[sid][dim]) settled++;
  });
  var total=document.querySelectorAll('.dim').length;
  document.getElementById('prog').textContent = settled+' / '+total+' dimensions adjudicated';
}
function exportJSON(){
  var out={meta:{a:META.a,b:META.b,exported:new Date().toISOString()}, verdicts:[]};
  document.querySelectorAll('.dim').forEach(function(b){
    var sid=b.dataset.sid, dim=b.dataset.dim;
    var st=(STATE[sid]||{})[dim]; if(!st) return;
    out.verdicts.push({sample_id:sid, dataset:b.dataset.ds, image_path:b.dataset.img,
       dimension:dim, truth_tags:st.unsure?null:st.tags, unsure:!!st.unsure,
       a_tags:JSON.parse(b.dataset.a), b_tags:JSON.parse(b.dataset.b)});
  });
  var blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  var a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='adjudication_verdicts.json'; a.click();
}
function saveLocal(){localStorage.setItem('adj_state', JSON.stringify(STATE)); alert('Saved to this browser.');}
function loadLocal(){var s=localStorage.getItem('adj_state'); if(s){STATE=JSON.parse(s);
  document.querySelectorAll('.dim').forEach(function(b){render(b.dataset.sid,b.dataset.dim);}); }}
window.addEventListener('load', function(){loadLocal(); updateProgress();});
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--image-root", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("dataset/adjudicate.html"))
    ap.add_argument("--max-images", type=int, default=60)
    ap.add_argument("--dimensions", default="",
                    help="comma-separated dimensions to review (default: all "
                         "dimensions that disagree)")
    ap.add_argument("--only-disagree", action="store_true", default=True)
    args = ap.parse_args()

    A, B = load_run(args.a), load_run(args.b)
    na, nb = args.a_name, args.b_name
    focus = [d.strip() for d in args.dimensions.split(",") if d.strip()] or list(TAXONOMY)
    shared = sorted(set(A) & set(B))

    # rank images by number of disagreeing (focused) dimensions, desc
    def ndis(s):
        return sum(1 for dim in focus
                   if set(A[s]["by_dim"][dim]) != set(B[s]["by_dim"][dim]))
    ranked = sorted((s for s in shared if ndis(s) > 0), key=lambda s: -ndis(s))
    sel = ranked[: args.max_images]

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Adjudicate tags</title><style>{CSS}</style></head><body>",
        "<header><h1>Human adjudication &mdash; pick the correct tags per "
        "dimension</h1>",
        f"<div class='sub'>Reviewing {len(sel)} images where "
        f"<span style='color:#8ab4ff'>{na}</span> and "
        f"<span style='color:#c4a2ff'>{nb}</span> disagree. Click chips to set "
        "truth; use a model's set with the buttons; ⚑ flags 'ask me / unsure'. "
        "Export when done.</div></header>",
        "<div id='bar'>",
        "<button onclick='exportJSON()'>⬇ Export verdicts JSON</button>",
        "<button class='ghost' onclick='saveLocal()'>Save (this browser)</button>",
        "<button class='ghost' onclick='loadLocal()'>Reload saved</button>",
        "<span id='prog'>0 / 0</span></div>",
        f"<script>var META={{a:{json.dumps(na)}, b:{json.dumps(nb)}}};</script>",
    ]

    for sid in sel:
        a, b = A[sid], B[sid]
        uri = thumb_data_uri(str(args.image_root / a["image_path"]))
        img_tag = (f"<img src='{uri}'>" if uri else
                   "<div style='color:#a00'>image missing</div>")
        dim_html = []
        for dim in focus:
            sa, sb = a["by_dim"][dim], b["by_dim"][dim]
            if args.only_disagree and set(sa) == set(sb):
                continue
            cid = f"dim_{_css(sid)}_{_css(dim)}"
            allowed = list(TAXONOMY[dim])
            opts = []
            for t in allowed:
                extra = (" diffA" if t in sa and t not in sb else
                         " diffB" if t in sb and t not in sa else "")
                opts.append(
                    f"<label class='opt{extra}' data-sid=\"{_a(sid)}\" "
                    f"data-dim=\"{_a(dim)}\" data-tag=\"{_a(t)}\" "
                    f"onclick='toggleTag(this)'>{_h(t)}</label>")
            dim_html.append(
                f"<div class='dim' id='{cid}' data-sid=\"{_a(sid)}\" "
                f"data-dim=\"{_a(dim)}\" data-ds=\"{_a(a['dataset'] or '')}\" "
                f"data-img=\"{_a(a['image_path'])}\" "
                f"data-a='{_j(sa)}' data-b='{_j(sb)}'>"
                f"<h3>{_h(dim)}<span class='tick'></span></h3>"
                "<div class='props'>"
                f"<div class='prop'><span class='who a'>{_h(na)}</span>: "
                f"{_h(', '.join(sa) or '—')}<div class='why'>{_h(a['reasoning'].get(dim,''))}</div></div>"
                f"<div class='prop'><span class='who b'>{_h(nb)}</span>: "
                f"{_h(', '.join(sb) or '—')}<div class='why'>{_h(b['reasoning'].get(dim,''))}</div></div>"
                "</div>"
                f"<div class='opts'>{''.join(opts)}</div>"
                "<div class='act'>"
                f"<button class='usebtn' onclick='useSet(\"{_a(sid)}\",\"{_a(dim)}\",\"a\")'>use {_h(na)}</button>"
                f"<button class='usebtn b' onclick='useSet(\"{_a(sid)}\",\"{_a(dim)}\",\"b\")'>use {_h(nb)}</button>"
                f"<button class='unsure' onclick='markUnsure(\"{_a(sid)}\",\"{_a(dim)}\")'>⚑ ask me</button>"
                f"<button class='clearb' onclick='clearDim(\"{_a(sid)}\",\"{_a(dim)}\")'>clear</button>"
                "</div>"
                f"<span id='set_a_{_css(sid)}_{_css(dim)}' style='display:none'>{_j(sa)}</span>"
                f"<span id='set_b_{_css(sid)}_{_css(dim)}' style='display:none'>{_j(sb)}</span>"
                "</div>")
        parts.append(
            f"<div class='card'><div class='hd'>"
            f"<span><span class='ds'>{_h(a['dataset'] or '')}</span> "
            f"<span class='id'>{_h(sid)}</span></span></div>"
            f"<div class='body'>{img_tag}<div class='dims'>{''.join(dim_html)}</div>"
            "</div></div>")

    parts.append(f"<script>{JS}</script></body></html>")
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}  ({len(sel)} images, focus dims: {focus})")


import html as _htmlmod


def _h(s):
    return _htmlmod.escape(str(s))


def _a(s):  # attribute-safe
    return _htmlmod.escape(str(s), quote=True)


def _j(obj):
    return _htmlmod.escape(json.dumps(list(obj)), quote=True)


def _css(s):
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))


if __name__ == "__main__":
    main()
