#!/usr/bin/env bash
# Generate universal-format manifests for all four lane datasets.
#
# Paths are the verified locations on this machine (see project memory):
#   - TuSimple : /shared/data/TUSimple/test_set + test_label.json (test split)
#   - CULane   : /shared/data/culane, TEST split only (val images not extracted)
#   - CurveLanes: /shared/data/Curvelanes, valid split
#   - BDD100K  : /shared/data/bdd100k val images + ll_seg lane masks
#
# Output: manifests/<dataset>/manifest_<dataset>.json (+ masks/ where rasterised)
#
# Usage:
#   bash scripts/build_manifests.sh            # all datasets, full
#   MAX=50 bash scripts/build_manifests.sh     # quick smoke test (50 samples each)
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-manifests}"
EXTRA=""
if [[ -n "${MAX:-}" ]]; then
  EXTRA="--max-samples ${MAX}"
fi

run() { echo "+ python -m lane_eval.cli.build_manifest $*"; python -m lane_eval.cli.build_manifest "$@" $EXTRA; }

run --dataset tusimple \
    --root /shared/data/TUSimple/test_set \
    --annotation-file /shared/data/TUSimple/test_label.json \
    --split test \
    --out-dir "$OUT_DIR"

run --dataset culane \
    --root /shared/data/culane \
    --split test \
    --out-dir "$OUT_DIR"

run --dataset curvelanes \
    --root /shared/data/Curvelanes \
    --split valid \
    --out-dir "$OUT_DIR"

run --dataset bdd100k_lane \
    --image-root /shared/data/bdd100k/images/val \
    --lane-mask-root /shared/data/bdd100k/ll_seg_annotations/val \
    --det-annotations-root /shared/data/bdd100k/det_annotations/val \
    --out-dir "$OUT_DIR"

echo "Done. Manifests under: $OUT_DIR/"
