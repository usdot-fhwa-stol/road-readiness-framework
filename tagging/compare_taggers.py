#!/usr/bin/env python3
"""Compare two gateway tagging runs (e.g. gpt-5-4-mini vs claude-4-5-sonnet) on
the images they BOTH tagged, to see where a cheap and an expensive model agree.

Because there is no local human ground truth, "agreement with a stronger model"
is a proxy for confidence: dimensions where mini and sonnet agree are safe to
trust from the cheap full run; dimensions where they diverge are where the extra
spend would change labels.

Reports, per dimension:
  - exact-set agreement  : both models chose the identical tag set for the image
  - mean Jaccard         : average |A∩B| / |A∪B| over images (partial credit)
  - per-image tag deltas : which tags one model added that the other didn't

Usage:
  python3 tagging/compare_taggers.py \
      --a dataset/gateway_tags        --a-name mini \
      --b dataset/gateway_tags_sonnet --b-name sonnet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import TAG_TO_DIM, TAXONOMY  # noqa: E402

DATASETS = ["bdd100k", "culane", "curvelanes", "tusimple"]


def load_run(run_dir):
    """dir of <ds>.tags.jsonl -> {sample_id: {by_dimension:{dim:set}, ...}}"""
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
                "dataset": d.get("dataset"),
                "by_dim": {dim: set(bd.get(dim, [])) for dim in TAXONOMY},
                "labels": set(pt.get("labels") or []),
                "reasoning": pt.get("reasoning") or {},
                "error": d.get("error"),
            }
    return out


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run A dir (e.g. mini)")
    ap.add_argument("--b", required=True, help="run B dir (e.g. sonnet)")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--examples", type=int, default=6,
                    help="print this many disagreeing examples")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON path to dump the full comparison")
    args = ap.parse_args()

    A = load_run(args.a)
    B = load_run(args.b)
    shared = sorted(set(A) & set(B))
    na, nb = args.a_name, args.b_name
    print(f"{na}: {len(A)} imgs | {nb}: {len(B)} imgs | shared (both tagged): "
          f"{len(shared)}\n")
    if not shared:
        print("No overlap. Did both runs use the same --sample-frac/--seed?")
        return

    # per-dimension stats over shared images
    exact = defaultdict(int)
    jac = defaultdict(float)
    # tag-level: how often each model emits a tag the other omits
    only_a = defaultdict(lambda: defaultdict(int))
    only_b = defaultdict(lambda: defaultdict(int))
    n_err_a = sum(1 for s in shared if A[s]["error"])
    n_err_b = sum(1 for s in shared if B[s]["error"])

    for s in shared:
        for dim in TAXONOMY:
            sa, sb = A[s]["by_dim"][dim], B[s]["by_dim"][dim]
            if sa == sb:
                exact[dim] += 1
            jac[dim] += jaccard(sa, sb)
            for t in sa - sb:
                only_a[dim][t] += 1
            for t in sb - sa:
                only_b[dim][t] += 1

    n = len(shared)
    print(f"Errors on shared set: {na}={n_err_a}, {nb}={n_err_b}\n")
    print(f"{'Dimension':<40} {'exact%':>7} {'Jaccard':>8}")
    print("-" * 60)
    dim_summ = {}
    macro_ex = macro_j = 0.0
    for dim in TAXONOMY:
        ex = exact[dim] / n
        jm = jac[dim] / n
        macro_ex += ex
        macro_j += jm
        dim_summ[dim] = {"exact_frac": round(ex, 3), "mean_jaccard": round(jm, 3),
                         f"{na}_only": dict(only_a[dim]),
                         f"{nb}_only": dict(only_b[dim])}
        print(f"{dim:<40} {ex*100:6.1f}% {jm:8.3f}")
    print("-" * 60)
    print(f"{'MACRO AVG':<40} {macro_ex/len(TAXONOMY)*100:6.1f}% "
          f"{macro_j/len(TAXONOMY):8.3f}\n")

    # the tags that most drive disagreement
    print("Top tags one model adds that the other omits "
          f"(count over {n} shared images):")
    for dim in TAXONOMY:
        a_only = sorted(only_a[dim].items(), key=lambda x: -x[1])[:3]
        b_only = sorted(only_b[dim].items(), key=lambda x: -x[1])[:3]
        if a_only or b_only:
            print(f"  {dim}")
            if a_only:
                print(f"    {na}-only: " +
                      ", ".join(f"{t}×{c}" for t, c in a_only))
            if b_only:
                print(f"    {nb}-only: " +
                      ", ".join(f"{t}×{c}" for t, c in b_only))

    # concrete disagreeing examples (largest label-set symmetric difference)
    ranked = sorted(
        shared,
        key=lambda s: -len(A[s]["labels"] ^ B[s]["labels"]))
    print(f"\nMost-divergent examples (up to {args.examples}):")
    for s in ranked[:args.examples]:
        diff = A[s]["labels"] ^ B[s]["labels"]
        if not diff:
            break
        print(f"\n  [{A[s]['dataset']}] {s}")
        for dim in TAXONOMY:
            sa, sb = A[s]["by_dim"][dim], B[s]["by_dim"][dim]
            if sa != sb:
                print(f"    {dim}")
                print(f"      {na}: {sorted(sa)}  | why: {A[s]['reasoning'].get(dim,'')}")
                print(f"      {nb}: {sorted(sb)}  | why: {B[s]['reasoning'].get(dim,'')}")

    if args.out:
        payload = {
            "counts": {na: len(A), nb: len(B), "shared": len(shared)},
            "errors": {na: n_err_a, nb: n_err_b},
            "per_dimension": dim_summ,
            "macro": {"exact_frac": round(macro_ex/len(TAXONOMY), 3),
                      "mean_jaccard": round(macro_j/len(TAXONOMY), 3)},
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
