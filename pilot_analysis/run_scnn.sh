#!/usr/bin/env bash
# Run SCNN through the lane_eval shared manifest/prediction-writer flow.

set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-scnn}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-outputs/scnn}"
OVERLAY_DIR="${OVERLAY_DIR:-$OUT_DIR/overlays}"
OVERLAY_SAMPLE="${OVERLAY_SAMPLE:-0}"
OVERLAY_SEED="${OVERLAY_SEED:-42}"
EXIST_THRESHOLD="${EXIST_THRESHOLD:--1}"

SCNN_ROOT="${SCNN_ROOT:-/shared/src/SCNN_Pytorch}"
SCNN_WEIGHTS="${SCNN_WEIGHTS:-/shared/src/SCNN_Pytorch/experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.pth}"

TUSIMPLE_MANIFEST="${TUSIMPLE_MANIFEST:-manifests/tusimple/manifest_tusimple.json}"
CULANE_MANIFEST="${CULANE_MANIFEST:-manifests/culane/manifest_culane.json}"
BDD100K_MANIFEST="${BDD100K_MANIFEST:-manifests/bdd100k_lane/manifest_bdd100k_lane.json}"
CURVELANE_MANIFEST="${CURVELANE_MANIFEST:-manifests/curvelanes/manifest_curvelanes.json}"

mkdir -p "$OUT_DIR"

MAXARG=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
  MAXARG=(--max-samples "$MAX_SAMPLES")
fi

run() {
  local dataset="$1"
  local manifest="$2"

  echo ""
  echo "=================================================================="
  echo ">>> SCNN on ${dataset}"
  echo "=================================================================="

  "$PY" -m lane_eval.cli.run_scnn \
    --manifest "$manifest" \
    --pred-manifest "$OUT_DIR/${dataset}_predictions.json" \
    --dataset "$dataset" \
    --model-name "$MODEL_NAME" \
    --device "$DEVICE" \
    --scnn-root "$SCNN_ROOT" \
    --weights "$SCNN_WEIGHTS" \
    --exist-threshold "$EXIST_THRESHOLD" \
    --overlay-dir "$OVERLAY_DIR" \
    --overlay-sample "$OVERLAY_SAMPLE" \
    --overlay-seed "$OVERLAY_SEED" \
    "${MAXARG[@]}"
}

run "tusimple" "$TUSIMPLE_MANIFEST"
run "culane" "$CULANE_MANIFEST"
run "bdd100k_lane" "$BDD100K_MANIFEST"
run "curvelanes" "$CURVELANE_MANIFEST"

echo ""
echo "SCNN prediction manifests written to: $OUT_DIR"
