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

# Dataset locations — override via env for other machines (defaults match the
# verified layout on the project host).
TUSIMPLE_ROOT="${TUSIMPLE_ROOT:-/shared/data/TUSimple/test_set}"
TUSIMPLE_ANNO="${TUSIMPLE_ANNO:-/shared/data/TUSimple/test_label.json}"
CULANE_ROOT="${CULANE_ROOT:-/shared/data/culane}"
CULANE_SPLIT="${CULANE_SPLIT:-test}"
CURVELANES_ROOT="${CURVELANES_ROOT:-/shared/data/Curvelanes}"
CURVELANES_SPLIT="${CURVELANES_SPLIT:-valid}"
BDD_IMAGES="${BDD_IMAGES:-/shared/data/bdd100k/images/val}"
BDD_MASKS="${BDD_MASKS:-/shared/data/bdd100k/ll_seg_annotations/val}"
BDD_DET="${BDD_DET:-/shared/data/bdd100k/det_annotations/val}"

EXTRA=""
if [[ -n "${MAX:-}" ]]; then
  EXTRA="--max-samples ${MAX}"
fi

run() { echo "+ python -m lane_eval.cli.build_manifest $*"; python -m lane_eval.cli.build_manifest "$@" $EXTRA; }

run --dataset tusimple \
    --root "$TUSIMPLE_ROOT" \
    --annotation-file "$TUSIMPLE_ANNO" \
    --split test \
    --out-dir "$OUT_DIR"

run --dataset culane \
    --root "$CULANE_ROOT" \
    --split "$CULANE_SPLIT" \
    --out-dir "$OUT_DIR"

run --dataset curvelanes \
    --root "$CURVELANES_ROOT" \
    --split "$CURVELANES_SPLIT" \
    --out-dir "$OUT_DIR"

run --dataset bdd100k_lane \
    --image-root "$BDD_IMAGES" \
    --lane-mask-root "$BDD_MASKS" \
    --det-annotations-root "$BDD_DET" \
    --out-dir "$OUT_DIR"

echo "Done. Manifests under: $OUT_DIR/"
