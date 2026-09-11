#!/usr/bin/env bash
#
# Auto-tag the categorized datasets and score the predictions against the
# human labels. Wraps tag_images.py + eval_tagger.py.
#
# Usage:
#   ./run_tagging.sh [-b BACKEND] [-n N] [-d "DS1 DS2 ..."]
#
#   -b BACKEND   siglip | vlm            (default: vlm)
#   -n N         images per dataset,     (default: 20; use 'all' for everything)
#   -d DATASETS  space-separated list    (default: "bdd100k curvelane culane")
#
# Examples:
#   ./run_tagging.sh                     # vlm, 20 imgs/dataset, all 3 datasets
#   ./run_tagging.sh -b siglip -n 20     # fast SigLIP backend
#   ./run_tagging.sh -b vlm -n all       # vlm on every image
#   ./run_tagging.sh -d "bdd100k" -n 5   # one dataset, quick check
#
# Notes:
#   * TuSimple is intentionally excluded -- its sheet rows are bare numeric ids
#     with no resolvable image path.
#   * The vlm backend needs Qwen/Qwen2.5-VL-7B-Instruct downloaded (~16 GB) and
#     a CUDA GPU; siglip uses google/siglip2-base (already cached).
#   * Outputs: output/manifest_<dataset>.<backend><N>.json  (paths + predicted_tags)
#
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="vlm"
N="20"
DATASETS="bdd100k curvelane culane"

while getopts "b:n:d:h" opt; do
  case "$opt" in
    b) BACKEND="$OPTARG" ;;
    n) N="$OPTARG" ;;
    d) DATASETS="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "run with -h for help" >&2; exit 1 ;;
  esac
done

# 'all' -> no limit (0 = every sample in tag_images.py)
LIMIT="$N"; TAG="$N"
if [ "$N" = "all" ]; then LIMIT=0; TAG="all"; fi

echo "backend=$BACKEND  images/dataset=$N  datasets: $DATASETS"
echo

for DS in $DATASETS; do
  IN="output/manifest_${DS}.json"
  OUT="output/manifest_${DS}.${BACKEND}${TAG}.json"
  if [ ! -f "$IN" ]; then
    echo "!! missing $IN (run build_tagging.py first) -- skipping $DS" >&2
    continue
  fi
  echo "=================== $DS ==================="
  python3 tag_images.py --backend "$BACKEND" --limit "$LIMIT" \
      --in-manifest "$IN" --out-manifest "$OUT"
  python3 eval_tagger.py "$OUT"
  echo
done

echo "Done. Tagged manifests written to output/manifest_*.${BACKEND}${TAG}.json"
