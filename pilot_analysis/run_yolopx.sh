#!/usr/bin/env bash
# Run YOLOPX lane evaluation across the available datasets and emit the 4
# standard segmentation metrics (IoU, F1, Precision, Recall) per dataset, then
# build the preliminary model-comparison table (Table 12).
#
# All metrics come from YOLOPX's own SegmentationMetric (lib/core/evaluate.py),
# computed inside this repo's harness — so when other models are added they are
# scored by the SAME metric code (fair comparison), not their native eval.
#
# Datasets: BDD100K, CurveLanes, CULane. (TuSimple is not downloaded yet.)
#
# Usage:
#   ./run_yolopx.sh                 # full run, all images
#   MAX_SAMPLES=20 ./run_yolopx.sh  # quick smoke test (20 imgs/dataset)
#   DEVICE=cpu ./run_yolopx.sh      # force CPU
# Any config var below can be overridden via the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- config (override via env) ----------------------------------------------
PY="${PYTHON:-python}"
YOLOPX_REPO="${YOLOPX_REPO:-}"
: "${YOLOPX_REPO:?YOLOPX_REPO must point to the YOLOPX repo root}"
WEIGHTS="${WEIGHTS:-$YOLOPX_REPO/weights/epoch-195.pth}"
MODEL_NAME="${MODEL_NAME:-yolopx}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"            # 0 = all images
PER_IMAGE="${PER_IMAGE:-1}"                # 1 = also save per-image JSON+CSV
SAVE_PRED="${SAVE_PRED:-1}"                # 1 = save predicted mask PNG per image (+ overlay sample)
OVERLAY_SAMPLE="${OVERLAY_SAMPLE:-100}"   # pred-vs-GT overlay JPGs per dataset
OUT_DIR="${OUT_DIR:-outputs/results}"
PRED_DIR="${PRED_DIR:-outputs/results/predictions}"

# dataset roots
BDD_IMAGES="${BDD_IMAGES:-/shared/data/bdd100k/images/val}"
BDD_MASKS="${BDD_MASKS:-/shared/data/bdd100k/ll_seg_annotations/val}"
CURVELANES_ROOT="${CURVELANES_ROOT:-/shared/data/Curvelanes}"
CURVELANES_SPLIT="${CURVELANES_SPLIT:-valid}"
CULANE_ROOT="${CULANE_ROOT:-/shared/data/culane}"
CULANE_SPLIT="${CULANE_SPLIT:-test}"   # CULane val images aren't extracted; test is the standard split
TUSIMPLE_ROOT="${TUSIMPLE_ROOT:-/shared/data/TUSimple/test_set}"
TUSIMPLE_ANNO="${TUSIMPLE_ANNO:-/shared/data/TUSimple/test_label.json}"

# -----------------------------------------------------------------------------
if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: weights not found at $WEIGHTS" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

MAXARG=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then MAXARG=(--max-samples "$MAX_SAMPLES"); fi
PERIMG=()
if [[ "$PER_IMAGE" == "1" ]]; then PERIMG=(--per-image); fi
PREDARG=()
if [[ "$SAVE_PRED" == "1" ]]; then PREDARG=(--save-pred-dir "$PRED_DIR" --overlay-sample "$OVERLAY_SAMPLE"); fi

run() {  # $1 = label for logs; rest = dataset-specific args
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo ">>> YOLOPX on ${name}"
  echo "=================================================================="
  if ! "$PY" -m lane_eval.cli.run_lane_eval \
      --yolopx-repo "$YOLOPX_REPO" \
      --weights "$WEIGHTS" \
      --model-name "$MODEL_NAME" \
      --device "$DEVICE" \
      --img-size "$IMG_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      "${MAXARG[@]}" "${PERIMG[@]}" "${PREDARG[@]}" \
      "$@"; then
    echo "WARN: ${name} run failed — continuing with remaining datasets" >&2
  fi
}

run "bdd100k_lane" \
  --dataset bdd100k_lane \
  --image-root "$BDD_IMAGES" \
  --lane-mask-root "$BDD_MASKS" \
  --split val \
  --output "$OUT_DIR/${MODEL_NAME}_bdd100k_lane.json"

run "curvelanes" \
  --dataset curvelanes \
  --root "$CURVELANES_ROOT" \
  --split "$CURVELANES_SPLIT" \
  --output "$OUT_DIR/${MODEL_NAME}_curvelanes.json"

run "culane" \
  --dataset culane \
  --root "$CULANE_ROOT" \
  --split "$CULANE_SPLIT" \
  --output "$OUT_DIR/${MODEL_NAME}_culane.json"

run "tusimple" \
  --dataset tusimple \
  --root "$TUSIMPLE_ROOT" \
  --annotation-file "$TUSIMPLE_ANNO" \
  --split test \
  --output "$OUT_DIR/${MODEL_NAME}_tusimple.json"

echo ""
echo "=================================================================="
echo ">>> Building model-comparison table (Table 12)"
echo "=================================================================="
"$PY" -m evaluation.build_model_table --results-dir "$OUT_DIR" --out "$OUT_DIR/model_comparison.md"
