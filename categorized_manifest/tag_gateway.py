#!/usr/bin/env python3
"""Re-tag the road images through the Leidos Model Gateway (lmg.leadai.leidos.com)
instead of the local Gemma-3 VLM.

Why this exists: the Gemma-3-27B tags in dataset/manifests/*.json have poor
recall on co-occurring conditions (Shadow / Glare / Wet Pavement) and are noisy
on the fine semantic dimensions (Pavement Marking, Roadway Context). The gateway
hosts far stronger vision models (GPT-5.x, Claude 4.x) behind an OpenAI-compatible
Chat Completions API. This script drives them with a redesigned, decision-rule
prompt (see _build_prompt) that explicitly asks for ALL applicable multi-labels.

Zero third-party deps: uses stdlib urllib for HTTP (the env has no openai/requests),
PIL only to resize images before base64.

Auth / endpoint come from the same env the web_search.ps1 helper uses:
  OPENAI_BASE_URL  (default https://lmg.leadai.leidos.com)
  OPENAI_API_KEY   (falls back to ANTHROPIC_API_KEY)

Input : dataset/manifests/manifest_<ds>.json  (samples carry a dataset-relative
        image_path plus split / ground_truth).
Output: <out-dir>/<ds>.tags.jsonl  -- one line per image, append-safe + RESUMABLE
        (re-run to continue; already-tagged sample_ids are skipped).

Each line:
  {sample_id, dataset, split, image_path, model,
   predicted_tags: {by_dimension, labels, scores:{}, _unmatched?, _raw?},
   usage, latency_s, error?}

Usage:
  # smoke test: 3 images from one dataset
  python3 categorized_manifest/tag_gateway.py --dataset bdd100k --limit 3

  # full run, default model (gpt-5-4-mini-public), 8 concurrent requests
  python3 categorized_manifest/tag_gateway.py --dataset all

  # higher-quality pass on a subset
  python3 categorized_manifest/tag_gateway.py --dataset culane --limit 100 \
      --model gpt-5-4-public
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# taxonomy.py is pure-python (no torch), so importing it here is safe.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import TAXONOMY  # noqa: E402

DEFAULT_MODEL = "gpt-5-4-mini-public"
DEFAULT_BASE = "https://lmg.leadai.leidos.com"

# repo root = parent of categorized_manifest/
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_DIR = REPO_ROOT / "dataset" / "manifests"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUT_DIR = REPO_ROOT / "dataset" / "gateway_tags"

DATASETS = ["bdd100k", "culane", "curvelanes", "tusimple"]


# --------------------------------------------------------------------------- #
# Prompt: redesigned to fix the two failure modes found in the Gemma tags --
# (1) low recall on co-occurring conditions, (2) ambiguity on the semantic dims.
# --------------------------------------------------------------------------- #
def _build_prompt():
    lines = [
        "You are an expert road-infrastructure annotator labeling ONE "
        "forward-facing dashcam road image for a lane-marking readiness dataset.",
        "",
        "Assign tags across 6 independent dimensions. Each dimension is "
        "MULTI-LABEL: select EVERY tag for which there is clear visual evidence "
        "-- do NOT stop at one. Co-occurring conditions are common and expected "
        "(a daytime image can be Daylight + Shadow + Wet Pavement at once; a road "
        "can show Solid Line + Dashed Line + Edge Line at once).",
        "",
        "Hard rules:",
        "- Use ONLY the exact tag strings listed below, copied verbatim (including "
        "spaces, slashes, capitalization). Never invent, merge, abbreviate, or "
        "translate a tag.",
        "- A tag needs clear visual evidence in THIS image; do not infer from "
        "context you cannot see.",
        "- Every dimension gets at least one tag. If nothing else applies, use the "
        'catch-all: Operational Scenario -> "General Roadway Segments"; Roadway '
        'Surface Type -> "Undetermined"; Observed Marking Visibility -> "Not '
        'Applicable"; Pavement Marking Type & Configuration -> "Unmarked Pavement".',
        "",
        "Per-dimension guidance:",
        "- Operational Scenario: tag special road situations when clearly present "
        "(Intersections, Roundabouts, Ramps / Merges, At-Grade Rail Crossings, "
        "Tolling Stations, Work Zones, Managed Lanes). Use \"General Roadway "
        "Segments\" for ordinary open road.",
        "- Roadway Context & Facility Type: TWO separable axes. (a) functional "
        "class -- Freeway/Expressway (grade-separated, high speed, no driveways) | "
        "Arterial Roadway (major multi-lane through-road, signals/driveways) | "
        "Collector Roadway (moderate, links local to arterial) | Local Street / "
        "Road (residential, low speed). (b) setting -- Urban (dense buildings) | "
        "Rural (open land). Tag ONE functional class (best judgment from lane "
        "count, medians, speed cues) AND the setting.",
        "- Roadway Surface Type: asphalt (dark gray/black) vs concrete (light gray, "
        "often with joints) vs Unpaved (dirt/gravel). \"Undetermined\" only if "
        "genuinely ambiguous.",
        "- Pavement Marking Type & Configuration: list EVERY marking visible "
        "ANYWHERE in the frame -- center lines, lane dividers, AND shoulder/edge "
        "lines. \"Unmarked Pavement\" only when NO paint is visible at all.",
        "- Observed Marking Visibility: quality of the lane paint (Clearly Visible; "
        "Faded / Worn / Not Visible; Low Contrast; Partially Missing; Occluded by "
        "vehicles/objects; Not Applicable when no markings). Multiple may apply.",
        "- Lighting & Weather: tag ALL that apply -- exactly one ambient-light tag "
        "(Daylight / Nighttime / Dusk / Dawn) PLUS any of Shadow, Glare, Rain, Fog, "
        "Snow, Wet Pavement, Icy Road that are visible. Wet reflective road => Wet "
        "Pavement even in daytime.",
        "",
        "Dimensions and allowed tags (meaning in parentheses):",
    ]
    for dim, tags in TAXONOMY.items():
        lines.append(f"{dim}:")
        for tag, desc in tags.items():
            short = desc.replace("a forward-facing dashcam photo ", "").strip()
            lines.append(f'  - "{tag}" ({short})')
    lines += _RESPONSE_SPEC
    return "\n".join(lines)


# Shared JSON-response spec appended to every prompt variant.
_RESPONSE_SPEC = [
    "",
    "Respond with ONLY a JSON object. Each key is a dimension name exactly as "
    "written above. Each value is an object with two fields:",
    '  "tags": [ the chosen exact tag strings ],',
    '  "why":  a single short clause (max ~15 words) citing the specific '
    "visible evidence for those tags.",
    "No prose outside the JSON, no markdown, no code fences.",
    "",
    "Example value: "
    '{"tags": ["Daylight", "Wet Pavement"], "why": "bright overcast sky; road '
    'surface is dark and reflective"}',
]


def _build_prompt_standard():
    """Stronger prompt built from definitions.py: per-dimension decision rules
    plus, for each tag, the positive cue and the exclusion (neg). Grounded in
    MUTCD / FHWA / MSMT-729 (see definitions.py docstring)."""
    from definitions import DEFINITIONS
    lines = [
        "You are an expert road-infrastructure annotator labeling ONE "
        "forward-facing dashcam road image for a lane-marking readiness dataset. "
        "Definitions below follow US road standards (MUTCD, FHWA functional "
        "classification, Maryland MSMT-729 marking-condition scale).",
        "",
        "Assign tags across 6 independent dimensions (multi-label: emit EVERY tag "
        "clearly supported by the image). For each dimension follow its decision "
        "rule, then apply each tag only when its 'use when' cue is met and its "
        "'not when' exclusion does not apply.",
        "",
        "Global rules:",
        "- Use ONLY the exact tag strings shown, copied verbatim (spaces, slashes, "
        "capitalization). Never invent, merge, abbreviate, or translate a tag.",
        "- Require clear visual evidence in THIS frame; do not infer unseen context.",
        "- Every dimension gets at least one tag (use its catch-all if nothing else "
        "applies).",
    ]
    for dim, spec in DEFINITIONS.items():
        lines.append("")
        lines.append(f"### {dim}")
        lines.append(f"Rule: {spec['rules']}")
        lines.append("Tags:")
        for tag, d in spec["tags"].items():
            lines.append(f'  - "{tag}"')
            lines.append(f"      use when: {d['cue']}")
            lines.append(f"      not when: {d['neg']}")
    lines += _RESPONSE_SPEC
    return "\n".join(lines)


# Prompt is selected at import via TAG_DEFS env (default: standard). tag_gateway
# also exposes --defs to set it explicitly before the module builds PROMPT.
def build_prompt(defs="standard"):
    return _build_prompt_standard() if defs == "standard" else _build_prompt()


PROMPT = build_prompt(os.environ.get("TAG_DEFS", "standard"))
_VALID = {dim: set(tags) for dim, tags in TAXONOMY.items()}


# --------------------------------------------------------------------------- #
# Response parsing (tolerant): mirrors backends._labels_from_json so the output
# shape is identical to the local-VLM path, but re-implemented here to avoid
# importing backends.py (which eagerly imports torch).
# --------------------------------------------------------------------------- #
def _norm_tag(s):
    s = re.sub(r"\s*/\s*", " / ", str(s).strip())
    return re.sub(r"\s+", " ", s).lower()


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _labels_from_json(raw):
    """Parse the model JSON into the predicted_tags dict. Accepts either the new
    per-dimension {"tags": [...], "why": "..."} shape or a legacy bare list, so
    older responses still parse. Keeps only taxonomy tags (tolerant matching);
    records the per-dimension reasoning under `reasoning`."""
    parsed = _extract_json(raw)
    by_dim, flat, unmatched, reasoning = {}, [], [], {}
    for dim, valid in _VALID.items():
        norm2canon = {_norm_tag(t): t for t in valid}
        chosen = []
        val = parsed.get(dim) if isinstance(parsed, dict) else None
        if isinstance(val, dict):                 # new {tags, why} shape
            got = val.get("tags", [])
            why = val.get("why")
            if why:
                reasoning[dim] = str(why)
        else:                                     # legacy bare list / string
            got = val if val is not None else []
        if isinstance(got, str):
            got = [got]
        for t in got or []:
            canon = norm2canon.get(_norm_tag(t))
            if canon and canon not in chosen:
                chosen.append(canon)
            elif canon is None and str(t).strip():
                unmatched.append(str(t))
        by_dim[dim] = chosen
        flat.extend(chosen)
    out = {"by_dimension": by_dim, "labels": flat, "scores": {},
           "reasoning": reasoning, "_raw": raw if not flat else None}
    if unmatched:
        out["_unmatched"] = unmatched
    return out


# --------------------------------------------------------------------------- #
# Gateway client
# --------------------------------------------------------------------------- #
class Gateway:
    def __init__(self, model, base=None, api_key=None, max_side=1024,
                 timeout=120, max_retries=5, prompt=None):
        self.model = model
        self.base = (base or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise SystemExit("No API key: set OPENAI_API_KEY (or ANTHROPIC_API_KEY).")
        self.prompt = prompt or PROMPT
        self.max_side = max_side
        self.timeout = timeout
        self.max_retries = max_retries
        self.url = self.base + "/v1/chat/completions"
        # OpenAI json-object mode needs the word 'json' in the prompt (it's there)
        # and is only reliably supported on the gpt-* models here.
        self.use_json_mode = model.startswith("gpt")

    def _encode_image(self, path):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if self.max_side and max(im.size) > self.max_side:
            im.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def _body(self, b64, drop_temp=False):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": self.prompt}]}],
            "max_tokens": 1024,
        }
        if not drop_temp:
            body["temperature"] = 0
        if self.use_json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def predict(self, image_path):
        """Returns (predicted_tags_dict, usage_dict, latency_s, error_or_None)."""
        try:
            b64 = self._encode_image(image_path)
        except (UnidentifiedImageError, OSError) as e:
            return None, {}, 0.0, f"image_open_error: {e!r}"

        drop_temp = False
        last_err = None
        for attempt in range(self.max_retries):
            payload = json.dumps(self._body(b64, drop_temp)).encode()
            req = urllib.request.Request(
                self.url, data=payload,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read().decode())
                dt = time.time() - t0
                raw = d["choices"][0]["message"]["content"] or ""
                return _labels_from_json(raw), d.get("usage", {}), round(dt, 2), None
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:300]
                last_err = f"HTTP {e.code}: {detail}"
                # Some reasoning models reject temperature != 1 -> retry without it.
                if e.code == 400 and "temperature" in detail.lower() and not drop_temp:
                    drop_temp = True
                    continue
                # Rate limit / transient server errors -> backoff and retry.
                if e.code in (408, 409, 425, 429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                return None, {}, 0.0, last_err  # non-retryable
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = f"net_error: {e!r}"
                time.sleep(min(2 ** attempt, 30))
                continue
        return None, {}, 0.0, last_err or "exhausted_retries"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def load_done(path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
    return done


def _in_sample(sample_id, frac, seed):
    """Deterministic membership test: stable hash of (seed, sample_id) mapped to
    [0,1). Same seed+frac selects the SAME images every run and across models, so
    a cheap model and an expensive model can be compared on an identical subset."""
    if frac >= 1.0:
        return True
    h = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    # first 8 hex chars -> 32-bit int -> [0,1)
    return (int(h[:8], 16) / 0xFFFFFFFF) < frac


def iter_samples(manifest_path, image_root, sample_frac=1.0, seed=0):
    m = json.loads(manifest_path.read_text())
    for s in m.get("samples", []):
        ip = s.get("image_path")
        sid = s.get("sample_id")
        if not ip or not sid:
            continue
        if not _in_sample(sid, sample_frac, seed):
            continue
        yield {
            "sample_id": sid,
            "dataset": s.get("dataset"),
            "split": s.get("split"),
            "image_path": ip,
            "abs_path": str((image_root / ip)),
        }


def run_dataset(ds, gw, manifest_dir, image_root, out_dir, limit, concurrency,
                sample_frac=1.0, seed=0):
    manifest_path = manifest_dir / f"manifest_{ds}.json"
    if not manifest_path.exists():
        print(f"[{ds}] SKIP: missing {manifest_path}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds}.tags.jsonl"
    done = load_done(out_path)

    todo = []
    for rec in iter_samples(manifest_path, image_root, sample_frac, seed):
        if rec["sample_id"] in done:
            continue
        todo.append(rec)
        if limit and len(todo) >= limit:
            break

    print(f"[{ds}] {len(done)} already done; tagging {len(todo)} "
          f"(model={gw.model}, concurrency={concurrency}) -> {out_path}", flush=True)
    if not todo:
        return

    n = errs = 0
    t0 = time.time()
    # Append as results arrive; a single lock-free writer thread via the main
    # loop consuming futures keeps the JSONL append atomic per line.
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut2rec = {ex.submit(gw.predict, rec["abs_path"]): rec for rec in todo}
        for fut in as_completed(fut2rec):
            rec = fut2rec[fut]
            pred, usage, dt, err = fut.result()
            line = {
                "sample_id": rec["sample_id"], "dataset": rec["dataset"],
                "split": rec["split"], "image_path": rec["image_path"],
                "model": gw.model, "predicted_tags": pred,
                "usage": usage, "latency_s": dt,
            }
            if err:
                line["error"] = err
                errs += 1
            f.write(json.dumps(line) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"[{ds}] {n}/{len(todo)} done, {errs} errors, "
                      f"{rate:.1f} img/s", flush=True)
    dt = time.time() - t0
    print(f"[{ds}] DONE tagged={n} errors={errs} in {dt:.0f}s "
          f"({n / dt if dt else 0:.1f} img/s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    choices=["all"] + DATASETS)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"gateway model id (default {DEFAULT_MODEL})")
    ap.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT,
                    help="base dir that manifest image_path is relative to")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=0,
                    help="tag only N new images per dataset (0 = all)")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="tag only a deterministic fraction of each dataset "
                         "(e.g. 0.02 = 2%%). Same --seed selects the same images "
                         "across models, for fair comparison.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --sample-frac selection (keep identical across "
                         "models being compared)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel gateway requests")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="resize longest image side to this before upload")
    ap.add_argument("--defs", choices=["standard", "legacy"], default="standard",
                    help="'standard' = decision-rule prompt from definitions.py "
                         "(MUTCD/FHWA/MSMT-729); 'legacy' = the original prompt")
    args = ap.parse_args()

    prompt = build_prompt(args.defs)
    gw = Gateway(args.model, max_side=args.max_side, prompt=prompt)
    print(f"gateway={gw.base}  model={gw.model}  json_mode={gw.use_json_mode}  "
          f"defs={args.defs}  prompt_chars={len(prompt)}", flush=True)

    if args.sample_frac < 1.0:
        print(f"sampling {args.sample_frac:.1%} of each dataset (seed={args.seed})",
              flush=True)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, gw, args.manifest_dir, args.image_root, args.out_dir,
                    args.limit, args.concurrency, args.sample_frac, args.seed)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
