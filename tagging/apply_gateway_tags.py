#!/usr/bin/env python3
"""Replace the old (Gemma) predicted tags in dataset/manifests/*.json with the
final Sonnet gateway tags from dataset/gateway_tags_final/*.tags.jsonl.

For every sample it:
  - swaps meta.predicted_tags  -> the gateway record's predicted_tags
    (by_dimension, labels, scores, reasoning)
  - recomputes meta.tags_encoded from the new labels (tag_codes.encode_labels),
    so the per-dim codes / multi-hot stay consistent with the new tags
  - leaves meta.existing_tags (native dataset attrs) untouched
  - updates metadata.tagger to record the new source

Originals are backed up to dataset/manifests/_backup_gemma/ before writing.
Matching is by sample_id (verified 1:1). manifest_all.json is rebuilt by
concatenating the per-dataset gateway files.

Usage:
  python3 tagging/apply_gateway_tags.py            # do it
  python3 tagging/apply_gateway_tags.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_codes import encode_labels  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "dataset" / "manifests"
GATEWAY_DIR = REPO / "dataset" / "gateway_tags_final"
BACKUP_DIR = MANIFEST_DIR / "_backup_gemma"
DATASETS = ["bdd100k", "culane", "curvelanes", "tusimple"]
NEW_TAGGER = "claude-4-5-sonnet-aws-comm-il2 (gateway, standard defs v4)"


def load_gateway(ds):
    """sample_id -> predicted_tags dict from the gateway JSONL."""
    out = {}
    with open(GATEWAY_DIR / f"{ds}.tags.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["sample_id"]] = d.get("predicted_tags") or {}
    return out


def apply_to_manifest(path, tags_by_id, dry_run):
    m = json.loads(path.read_text(encoding="utf-8"))
    updated = missing = 0
    for s in m["samples"]:
        pt = tags_by_id.get(s["sample_id"])
        if not pt:
            missing += 1
            continue
        labels = pt.get("labels", [])
        s.setdefault("meta", {})
        s["meta"]["predicted_tags"] = pt
        s["meta"]["tags_encoded"] = encode_labels(labels)
        updated += 1
    m.setdefault("metadata", {})["tagger"] = NEW_TAGGER
    if not dry_run:
        path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return updated, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # load all gateway tags once (also used to rebuild manifest_all)
    all_tags = {}
    for ds in DATASETS:
        t = load_gateway(ds)
        all_tags[ds] = t
        print(f"[{ds}] gateway records: {len(t)}")
    combined = {sid: pt for t in all_tags.values() for sid, pt in t.items()}

    # backup originals
    if not args.dry_run:
        BACKUP_DIR.mkdir(exist_ok=True)
        for f in MANIFEST_DIR.glob("manifest_*.json"):
            dst = BACKUP_DIR / f.name
            if not dst.exists():                       # never clobber a backup
                shutil.copy2(f, dst)
        print(f"backed up originals -> {BACKUP_DIR}")

    # per-dataset manifests
    for ds in DATASETS:
        p = MANIFEST_DIR / f"manifest_{ds}.json"
        if p.exists():
            u, miss = apply_to_manifest(p, all_tags[ds], args.dry_run)
            print(f"[{ds}] updated={u} missing={miss} -> {p.name}"
                  + ("  (dry-run)" if args.dry_run else ""))

    # combined manifest
    p_all = MANIFEST_DIR / "manifest_all.json"
    if p_all.exists():
        u, miss = apply_to_manifest(p_all, combined, args.dry_run)
        print(f"[all] updated={u} missing={miss} -> {p_all.name}"
              + ("  (dry-run)" if args.dry_run else ""))

    print("DONE" + (" (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
