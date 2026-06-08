#!/usr/bin/env bash
# Run HybridNets lane evaluation across the available datasets, scored by the
# SAME shared harness as YOLOPX (our lane_eval GT adapters + YOLOPX's
# SegmentationMetric) and written into the SAME outputs/results/, so the
# comparison table (Table 12) covers both models apples-to-apples.
#
# Runs under the HybridNets venv (it has timm/efficientnet/etc.).
#
# Usage:
#   ./run_hybridnets.sh                 # full run, all images
#   MAX_SAMPLES=20 ./run_hybridnets.sh  # quick smoke test
set -euo pipefail
cd "$(dirname "$0")"

PY="${HYBRIDNETS_PY:-/shared/src/HybridNets/hybridnets/bin/python}"
HN_REPO="${HYBRIDNETS_REPO:-/home/gauravb/Projects/road_readiness_t3/HybridNets}"
YOLOPX_REPO="${YOLOPX_REPO:-/home/gauravb/Projects/road_readiness_t3/YOLOPX}"  # for shared SegmentationMetric
WEIGHTS="${WEIGHTS:-$HN_REPO/weights/hybridnets.pth}"
MODEL_NAME="${MODEL_NAME:-hybridnets}"
PROJECT="${PROJECT:-bdd100k}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PER_IMAGE="${PER_IMAGE:-1}"                # 1 = also save per-image JSON+CSV
SAVE_PRED="${SAVE_PRED:-1}"                # 1 = save predicted mask PNG per image (+ overlay sample)
OVERLAY_SAMPLE="${OVERLAY_SAMPLE:-100}"   # pred-vs-GT overlay JPGs per dataset
OUT_DIR="${OUT_DIR:-outputs/results}"
PRED_DIR="${PRED_DIR:-outputs/results/predictions}"

BDD_IMAGES="${BDD_IMAGES:-/shared/data/bdd100k/images/val}"
BDD_MASKS="${BDD_MASKS:-/shared/data/bdd100k/ll_seg_annotations/val}"
CURVELANES_ROOT="${CURVELANES_ROOT:-/shared/data/Curvelanes}"
CURVELANES_SPLIT="${CURVELANES_SPLIT:-valid}"
CULANE_ROOT="${CULANE_ROOT:-/shared/data/culane}"
CULANE_SPLIT="${CULANE_SPLIT:-test}"
TUSIMPLE_ROOT="${TUSIMPLE_ROOT:-/shared/data/TUSimple/test_set}"
TUSIMPLE_ANNO="${TUSIMPLE_ANNO:-/shared/data/TUSimple/test_label.json}"

[[ -f "$WEIGHTS" ]] || { echo "ERROR: weights not found at $WEIGHTS" >&2; exit 1; }
[[ -f "$HN_REPO/hybridnets/model.py" ]] || { echo "ERROR: HybridNets source missing at $HN_REPO" >&2; exit 1; }
mkdir -p "$OUT_DIR"

MAXARG=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then MAXARG=(--max-samples "$MAX_SAMPLES"); fi
PERIMG=()
if [[ "$PER_IMAGE" == "1" ]]; then PERIMG=(--per-image); fi
PREDARG=()
if [[ "$SAVE_PRED" == "1" ]]; then PREDARG=(--save-pred-dir "$PRED_DIR" --overlay-sample "$OVERLAY_SAMPLE"); fi

run() {
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo ">>> HybridNets on ${name}"
  echo "=================================================================="
  if ! "$PY" -m evaluation.run_hybridnets \
      --hybridnets-repo "$HN_REPO" --yolopx-repo "$YOLOPX_REPO" \
      --weights "$WEIGHTS" --project "$PROJECT" --model-name "$MODEL_NAME" \
      --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
      "${MAXARG[@]}" "${PERIMG[@]}" "${PREDARG[@]}" "$@"; then
    echo "WARN: ${name} run failed — continuing with remaining datasets" >&2
  fi
}

run "bdd100k_lane" --dataset bdd100k_lane --image-root "$BDD_IMAGES" --lane-mask-root "$BDD_MASKS" \
  --split val --output "$OUT_DIR/${MODEL_NAME}_bdd100k_lane.json"

run "curvelanes" --dataset curvelanes --root "$CURVELANES_ROOT" --split "$CURVELANES_SPLIT" \
  --output "$OUT_DIR/${MODEL_NAME}_curvelanes.json"

run "culane" --dataset culane --root "$CULANE_ROOT" --split "$CULANE_SPLIT" \
  --output "$OUT_DIR/${MODEL_NAME}_culane.json"

run "tusimple" --dataset tusimple --root "$TUSIMPLE_ROOT" --annotation-file "$TUSIMPLE_ANNO" --split test \
  --output "$OUT_DIR/${MODEL_NAME}_tusimple.json"

echo ""
echo "=================================================================="
echo ">>> Rebuilding model-comparison table (all models in $OUT_DIR)"
echo "=================================================================="
"$PY" -m evaluation.build_model_table --results-dir "$OUT_DIR" --out "$OUT_DIR/model_comparison.md"
