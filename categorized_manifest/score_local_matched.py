#!/usr/bin/env python3
"""Score the local bake-off models (SigLIP2 / Gemma-12B / Gemma-27B) using the
EXACT same metric definitions as score_vs_human.py (which scored the gateway
models GPT-5.4-mini / Claude-4.5-Sonnet against 108 human adjudication
verdicts), restricted to the same 3 shared dimensions.

This does NOT make the comparison image-for-image apples-to-apples -- the
local bake-off's 20-img/dataset eval subset and the gateway's 108 verdict
images are disjoint (confirmed: 0 overlap across all 4 datasets). It only
removes the dimension-count (6 vs 3) and metric-formula mismatch, using
predictions/ground truth that already exist on disk (no model rerun).

Ground truth here = the human tags baked into categorized_manifest/output/
manifest_<ds>.<model>.json (tags.labels), NOT the gateway's adjudication
verdicts file (which does not exist on this machine).

Usage:
  python3 categorized_manifest/score_local_matched.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import TAG_TO_DIM  # noqa: E402

SHARED_DIMS = [
    "Roadway Context & Facility Type",
    "Pavement Marking Type & Configuration",
    "Observed Marking Visibility",
]
DATASETS = ["bdd100k", "culane", "curvelane", "tusimple"]
MODEL_SUFFIXES = {"SigLIP2": "siglip20", "Gemma-3-12B": "gemma12", "Gemma-3-27B": "gemma27"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def load_samples(model_suffix):
    samples = []
    for ds in DATASETS:
        path = os.path.join(OUT_DIR, f"manifest_{ds}.{model_suffix}.json")
        if not os.path.exists(path):
            continue
        m = json.load(open(path))
        samples.extend(m["samples"])
    return samples


def score(model_name, model_suffix):
    samples = load_samples(model_suffix)
    per = defaultdict(lambda: {"n": 0, "exact": 0, "jac": 0.0, "tp": 0, "fp": 0, "fn": 0})
    for s in samples:
        gold_labels = s.get("tags", {}).get("labels") or []
        pred_by_dim = (s.get("predicted_tags") or {}).get("by_dimension") or {}
        for dim in SHARED_DIMS:
            truth = {t for t in gold_labels if TAG_TO_DIM.get(t) == dim}
            got = set(pred_by_dim.get(dim, []))
            p = per[dim]
            p["n"] += 1
            p["exact"] += (got == truth)
            p["jac"] += jaccard(got, truth)
            p["tp"] += len(got & truth)
            p["fp"] += len(got - truth)
            p["fn"] += len(truth - got)

    print(f"================= {model_name} ================= "
          f"({len(samples)} images, {len(SHARED_DIMS)} shared dims)")
    print(f"{'Dimension':<42}{'n':>4} {'exact':>7} {'Jacc':>6} {'P':>5} {'R':>5} {'F1':>5}")
    print("-" * 78)
    tot = {"n": 0, "exact": 0, "jac": 0.0, "tp": 0, "fp": 0, "fn": 0}
    for dim in SHARED_DIMS:
        p = per[dim]
        n = p["n"]
        P = p["tp"] / (p["tp"] + p["fp"]) if p["tp"] + p["fp"] else 0.0
        R = p["tp"] / (p["tp"] + p["fn"]) if p["tp"] + p["fn"] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        print(f"{dim:<42}{n:>4} {p['exact']/n*100:6.1f}% {p['jac']/n:6.3f} "
              f"{P:5.2f} {R:5.2f} {F:5.2f}")
        for k in tot:
            tot[k] += p[k]
    n = tot["n"]
    P = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else 0.0
    R = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    print("-" * 78)
    print(f"{'OVERALL':<42}{n:>4} {tot['exact']/n*100:6.1f}% "
          f"{tot['jac']/n:6.3f} {P:5.2f} {R:5.2f} {F:5.2f}\n")
    return {
        "model": model_name, "n_images": len(samples), "n_pairs": n,
        "exact_pct": round(tot["exact"] / n * 100, 1),
        "jaccard": round(tot["jac"] / n, 3),
        "precision": round(P, 2), "recall": round(R, 2), "f1": round(F, 2),
        "per_dim": {dim: {
            "f1": round(2 * (per[dim]["tp"] / (per[dim]["tp"] + per[dim]["fp"]) if per[dim]["tp"] + per[dim]["fp"] else 0.0)
                         * (per[dim]["tp"] / (per[dim]["tp"] + per[dim]["fn"]) if per[dim]["tp"] + per[dim]["fn"] else 0.0)
                         / max((per[dim]["tp"] / (per[dim]["tp"] + per[dim]["fp"]) if per[dim]["tp"] + per[dim]["fp"] else 0.0)
                               + (per[dim]["tp"] / (per[dim]["tp"] + per[dim]["fn"]) if per[dim]["tp"] + per[dim]["fn"] else 0.0), 1e-9), 2)
        } for dim in SHARED_DIMS},
    }


if __name__ == "__main__":
    results = [score(name, suffix) for name, suffix in MODEL_SUFFIXES.items()]
    with open(os.path.join(OUT_DIR, "local_matched_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT_DIR}/local_matched_scores.json")
