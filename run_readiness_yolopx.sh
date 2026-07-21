#!/usr/bin/env bash
# Run the full I/D/C/R readiness metrics (evaluation/run_readiness.py) with
# YOLOPX across all 4 datasets, emitting per-dataset:
#   - aggregated JSON report  (I1-I6, D1-D8, C1-C5, R1-R4)
#   - per-image JSONL         (one full metric record per image)
#   - per-image CSV           (scalar metrics, easy to open in pandas/Excel)
#   - failure images          (overlays for frames that fail key thresholds)
#
# Usage:
#   ./run_readiness_yolopx.sh                  # full run, all images
#   MAX_SAMPLES=20 ./run_readiness_yolopx.sh   # quick smoke test
#   DEVICE=cpu ./run_readiness_yolopx.sh       # force CPU
# Any config var below can be overridden via the environment.
set -euo pipefail

cd "$(dirname "$0")"

# ---- config (override via env) ----------------------------------------------
PY="${PYTHON:-python}"
YOLOPX_REPO="${YOLOPX_REPO:-/home/gauravb/Projects/road_readiness_t3/YOLOPX}"
: "${YOLOPX_REPO:?YOLOPX_REPO must point to the YOLOPX repo root}"
WEIGHTS="${WEIGHTS:-$YOLOPX_REPO/weights/epoch-195.pth}"
MODEL_NAME="${MODEL_NAME:-yolopx}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"           # 0 = all images
SAVE_FAILURES="${SAVE_FAILURES:-1}"       # 1 = save overlay images for failure cases
SAVE_GOOD="${SAVE_GOOD:-0}"              # 1 = save overlay images for good cases
OUT_DIR="${OUT_DIR:-outputs/readiness}"
FAILURES_DIR="${FAILURES_DIR:-$OUT_DIR/failures}"
GOOD_DIR="${GOOD_DIR:-$OUT_DIR/good}"

# dataset roots (mirror run_yolopx.sh paths)
BDD_IMAGES="${BDD_IMAGES:-/shared/data/bdd100k/images/val}"
BDD_MASKS="${BDD_MASKS:-/shared/data/bdd100k/ll_seg_annotations/val}"
BDD_DET_ANNO="${BDD_DET_ANNO:-}"         # optional: enables C5 weather/time strata
CURVELANES_ROOT="${CURVELANES_ROOT:-/shared/data/Curvelanes}"
CURVELANES_SPLIT="${CURVELANES_SPLIT:-valid}"
CULANE_ROOT="${CULANE_ROOT:-/shared/data/culane}"
CULANE_SPLIT="${CULANE_SPLIT:-test}"
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

FAILARG=()
if [[ "$SAVE_FAILURES" == "1" ]]; then FAILARG=(--save-failures --failures-dir "$FAILURES_DIR"); fi

GOODARG=()
if [[ "$SAVE_GOOD" == "1" ]]; then GOODARG=(--save-good --good-dir "$GOOD_DIR"); fi

run() {
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo ">>> Readiness metrics — YOLOPX on ${name}"
  echo "=================================================================="
  if ! "$PY" -m evaluation.run_readiness \
      --yolopx-repo "$YOLOPX_REPO" \
      --weights "$WEIGHTS" \
      --model-name "$MODEL_NAME" \
      --device "$DEVICE" \
      --img-size "$IMG_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      "${MAXARG[@]}" "${FAILARG[@]}" "${GOODARG[@]}" \
      "$@"; then
    echo "WARN: ${name} run failed — continuing with remaining datasets" >&2
  fi
}

# BDD100K — has weather/timeofday metadata so pass --stratify for C5
BDD_DET_ARG=()
if [[ -n "$BDD_DET_ANNO" ]]; then BDD_DET_ARG=(--det-annotations-root "$BDD_DET_ANNO"); fi

run "bdd100k_lane" \
  --dataset bdd100k_lane \
  --image-root "$BDD_IMAGES" \
  --lane-mask-root "$BDD_MASKS" \
  "${BDD_DET_ARG[@]}" \
  --split val \
  --stratify weather time context \
  --output "$OUT_DIR/${MODEL_NAME}_bdd100k_lane_readiness.json" \
  --per-image-output "$OUT_DIR/${MODEL_NAME}_bdd100k_lane_per_image.jsonl" \
  --per-image-csv    "$OUT_DIR/${MODEL_NAME}_bdd100k_lane_per_image.csv"

run "curvelanes" \
  --dataset curvelanes \
  --root "$CURVELANES_ROOT" \
  --split "$CURVELANES_SPLIT" \
  --output "$OUT_DIR/${MODEL_NAME}_curvelanes_readiness.json" \
  --per-image-output "$OUT_DIR/${MODEL_NAME}_curvelanes_per_image.jsonl" \
  --per-image-csv    "$OUT_DIR/${MODEL_NAME}_curvelanes_per_image.csv"

run "culane" \
  --dataset culane \
  --root "$CULANE_ROOT" \
  --split "$CULANE_SPLIT" \
  --output "$OUT_DIR/${MODEL_NAME}_culane_readiness.json" \
  --per-image-output "$OUT_DIR/${MODEL_NAME}_culane_per_image.jsonl" \
  --per-image-csv    "$OUT_DIR/${MODEL_NAME}_culane_per_image.csv"

run "tusimple" \
  --dataset tusimple \
  --root "$TUSIMPLE_ROOT" \
  --annotation-file "$TUSIMPLE_ANNO" \
  --split test \
  --output "$OUT_DIR/${MODEL_NAME}_tusimple_readiness.json" \
  --per-image-output "$OUT_DIR/${MODEL_NAME}_tusimple_per_image.jsonl" \
  --per-image-csv    "$OUT_DIR/${MODEL_NAME}_tusimple_per_image.csv"

echo ""
echo "=================================================================="
echo ">>> Done. Outputs written to: $OUT_DIR"
echo "=================================================================="
