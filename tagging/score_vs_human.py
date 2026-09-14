#!/usr/bin/env python3
"""Score one or more tagging runs against the human adjudication verdicts.

The verdicts JSON (exported from adjudicate.html) is the only ground truth we
have, so this is the real accuracy signal -- unlike model-vs-model agreement.
Only the (sample_id, dimension) pairs a human settled are scored; 'unsure'/
'ask me' verdicts are skipped.

Reports, per dimension and overall, for each run:
  exact% (whole tag-set matches human), mean Jaccard, and micro tag-level P/R/F1.

Usage:
  python3 tagging/score_vs_human.py \
      --verdicts dataset/adjudication_verdicts.json \
      --run mini_v4=dataset/gateway_tags_v4_mini \
      --run sonnet_v4=dataset/gateway_tags_v4_sonnet \
      --run mini_v3=dataset/gateway_tags_v3_mini \
      --run sonnet_v2=dataset/gateway_tags_v2_sonnet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import TAXONOMY  # noqa: E402


def load_run(run_dir):
    out = {}
    for fp in glob.glob(os.path.join(run_dir, "*.tags.jsonl")):
        for line in open(fp):
            try:
                j = json.loads(line)
            except Exception:
                continue
            pt = j.get("predicted_tags") or {}
            bd = pt.get("by_dimension") or {}
            out[j["sample_id"]] = {dim: set(bd.get(dim, [])) for dim in TAXONOMY}
    return out


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--run", action="append", default=[],
                    help="name=dir (repeatable)")
    args = ap.parse_args()

    V = json.load(open(args.verdicts))["verdicts"]
    truth = [(x["sample_id"], x["dimension"], set(x["truth_tags"]))
             for x in V if not x.get("unsure") and x.get("truth_tags") is not None]
    dims_present = sorted({d for _, d, _ in truth}, key=list(TAXONOMY).index)

    runs = {}
    for spec in args.run:
        name, _, d = spec.partition("=")
        runs[name] = load_run(d)

    print(f"Scoring {len(truth)} human verdicts across "
          f"{len(dims_present)} dimensions.\n")

    # per run: per dim exact/jaccard + tag-level tp/fp/fn
    for name, run in runs.items():
        per = defaultdict(lambda: {"n": 0, "exact": 0, "jac": 0.0,
                                   "tp": 0, "fp": 0, "fn": 0})
        for sid, dim, tset in truth:
            got = run.get(sid, {}).get(dim, set())
            p = per[dim]
            p["n"] += 1
            p["exact"] += (got == tset)
            p["jac"] += jaccard(got, tset)
            p["tp"] += len(got & tset)
            p["fp"] += len(got - tset)
            p["fn"] += len(tset - got)
        print(f"================= {name} =================")
        print(f"{'Dimension':<42}{'n':>3} {'exact':>7} {'Jacc':>6} "
              f"{'P':>5} {'R':>5} {'F1':>5}")
        print("-" * 78)
        tot = {"n": 0, "exact": 0, "jac": 0.0, "tp": 0, "fp": 0, "fn": 0}
        for dim in dims_present:
            p = per[dim]
            n = p["n"]
            P = p["tp"] / (p["tp"] + p["fp"]) if p["tp"] + p["fp"] else 0.0
            R = p["tp"] / (p["tp"] + p["fn"]) if p["tp"] + p["fn"] else 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            print(f"{dim:<42}{n:>3} {p['exact']/n*100:6.1f}% {p['jac']/n:6.3f} "
                  f"{P:5.2f} {R:5.2f} {F:5.2f}")
            for k in tot:
                tot[k] += p[k]
        n = tot["n"]
        P = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else 0.0
        R = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        print("-" * 78)
        print(f"{'OVERALL':<42}{n:>3} {tot['exact']/n*100:6.1f}% "
              f"{tot['jac']/n:6.3f} {P:5.2f} {R:5.2f} {F:5.2f}\n")


if __name__ == "__main__":
    main()
