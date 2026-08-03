#!/usr/bin/env python3
"""Score predicted tags in a tagged manifest against the human tags.

Reports per-dimension micro precision/recall/F1 over samples that have both
human `tags.labels` and `predicted_tags.labels`.

Usage:
  python3 categorized_manifest/eval_tagger.py categorized_manifest/output/manifest_bdd100k.tagged.json
"""
import json
import sys
from collections import defaultdict

from taxonomy import TAG_TO_DIM, TAXONOMY


def main(path):
    manifest = json.loads(open(path).read())
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    n = 0
    for s in manifest["samples"]:
        human = s.get("tags", {}).get("labels")
        pred = s.get("predicted_tags")
        if not human or not pred:
            continue
        n += 1
        gold = set(human)
        got = set(pred["labels"])
        for dim in TAXONOMY:
            g = {t for t in gold if TAG_TO_DIM.get(t) == dim}
            p = {t for t in got if TAG_TO_DIM.get(t) == dim}
            tp[dim] += len(g & p)
            fp[dim] += len(p - g)
            fn[dim] += len(g - p)

    print(f"\nEvaluated {n} samples with both human + predicted tags: {path}\n")
    print(f"{'Dimension':<38} {'P':>6} {'R':>6} {'F1':>6}  (tp/fp/fn)")
    print("-" * 78)
    tt = ft = nt = 0
    for dim in TAXONOMY:
        t, f, m = tp[dim], fp[dim], fn[dim]
        tt += t; ft += f; nt += m
        p = t / (t + f) if t + f else 0.0
        r = t / (t + m) if t + m else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{dim:<38} {p:6.2f} {r:6.2f} {f1:6.2f}  ({t}/{f}/{m})")
    P = tt / (tt + ft) if tt + ft else 0.0
    R = tt / (tt + nt) if tt + nt else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    print("-" * 78)
    print(f"{'MICRO (all dimensions)':<38} {P:6.2f} {R:6.2f} {F1:6.2f}  ({tt}/{ft}/{nt})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "categorized_manifest/output/manifest_bdd100k.tagged.json")
