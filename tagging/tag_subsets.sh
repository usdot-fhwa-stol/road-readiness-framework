#!/usr/bin/env bash
#
# One command: build the image+GT subsets, tag them, and write a combined
# manifest. Every sample is guaranteed to have image_path AND ground_truth_path.
#
# Usage:
#   ./tag_subsets.sh [-b BACKEND] [-n N]
#     -b BACKEND   siglip | vlm   (default: vlm)
#     -n N         images/dataset  (default: 20)
#
# Examples:
#   ./tag_subsets.sh                 # vlm, 20 imgs/dataset, all 4 datasets
#   ./tag_subsets.sh -b siglip -n 10
#
# Outputs (output/):
#   subset_<dataset>.json         image+GT subset manifests
#   subset_<dataset>.<bk><N>.json tagged per dataset
#   subset_all.<bk><N>.json       combined, with a `dataset` field per sample
#
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="vlm"; N="20"
while getopts "b:n:h" opt; do
  case "$opt" in
    b) BACKEND="$OPTARG" ;;
    n) N="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "run with -h for help" >&2; exit 1 ;;
  esac
done

DATASETS="tusimple bdd100k culane curvelanes"

echo "[1/3] building image+GT subsets ..."
python3 build_eval_subset.py

echo "[2/3] tagging (backend=$BACKEND, $N imgs/dataset) ..."
for DS in $DATASETS; do
  python3 tag_images.py --backend "$BACKEND" --limit "$N" \
      --in-manifest  "output/subset_${DS}.json" \
      --out-manifest "output/subset_${DS}.${BACKEND}${N}.json"
done

echo "[3/3] combining ..."
python3 - "$BACKEND" "$N" $DATASETS <<'PY'
import json, sys
backend, n, *datasets = sys.argv[1:]
alls = []
for ds in datasets:
    for s in json.load(open(f"output/subset_{ds}.{backend}{n}.json"))["samples"]:
        s["dataset"] = ds
        alls.append(s)
out = {"metadata": {"datasets": datasets, "per_dataset": int(n),
                    "num_samples": len(alls), "backend": backend,
                    "note": "image_path + ground_truth_path guaranteed present"},
       "samples": alls}
p = f"output/subset_all.{backend}{n}.json"
json.dump(out, open(p, "w"), indent=2)
print(f"  {len(alls)} samples -> {p}")
PY

echo "Done."
