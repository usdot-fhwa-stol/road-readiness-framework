#!/usr/bin/env bash
# Run CLRerNet through the lane_eval shared manifest/prediction-writer flow.
#
# Usage:
#   ./run_clrernet.sh
#   MAX_SAMPLES=10 ./run_clrernet.sh
#   DEVICE=cpu ./run_clrernet.sh

set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-clrernet}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-outputs/clrernet}"
OVERLAY_DIR="${OVERLAY_DIR:-$OUT_DIR/overlays}"
OVERLAY_SAMPLE="${OVERLAY_SAMPLE:-0}"
OVERLAY_SEED="${OVERLAY_SEED:-42}"

CLRERNET_ROOT="${CLRERNET_ROOT:-/shared/src/CLRerNet}"
CLRERNET_CONFIG="${CLRERNET_CONFIG:-/shared/src/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py}"
CLRERNET_CHECKPOINT="${CLRERNET_CHECKPOINT:-/shared/src/CLRerNet/clrernet_culane_dla34_ema.pth}"

TUSIMPLE_MANIFEST="${TUSIMPLE_MANIFEST:-$HOME/clrernet_adapter/tusimple_manifest_500.json}"
CULANE_MANIFEST="${CULANE_MANIFEST:-$HOME/clrernet_adapter/culane_manifest_500.json}"
BDD100K_MANIFEST="${BDD100K_MANIFEST:-$HOME/clrernet_adapter/bdd100k_manifest_500.json}"
CURVELANE_MANIFEST="${CURVELANE_MANIFEST:-$HOME/clrernet_adapter/curvelane_manifest_500.json}"

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
  echo ">>> CLRerNet on ${dataset}"
  echo "=================================================================="

  "$PY" -m lane_eval.cli.run_clrernet \
    --manifest "$manifest" \
    --pred-manifest "$OUT_DIR/${dataset}_predictions.json" \
    --dataset "$dataset" \
    --model-name "$MODEL_NAME" \
    --device "$DEVICE" \
    --clrernet-root "$CLRERNET_ROOT" \
    --config "$CLRERNET_CONFIG" \
    --checkpoint "$CLRERNET_CHECKPOINT" \
    --overlay-dir "$OVERLAY_DIR" \
    --overlay-sample "$OVERLAY_SAMPLE" \
    --overlay-seed "$OVERLAY_SEED" \
    "${MAXARG[@]}"
}

run "tusimple" "$TUSIMPLE_MANIFEST"
run "culane" "$CULANE_MANIFEST"
run "bdd100k" "$BDD100K_MANIFEST"
run "curvelane" "$CURVELANE_MANIFEST"

echo ""
echo "CLRerNet prediction manifests written to: $OUT_DIR"
