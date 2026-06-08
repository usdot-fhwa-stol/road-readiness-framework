#!/usr/bin/env bash
# Run YOLOPX lane evaluation reading the UNIVERSAL MANIFESTS (not native dataset
# adapters), writing the same result JSON + per-image files as run_yolopx.sh.
# Used to confirm the manifest + adapter path reproduces the native-adapter
# core metrics.
#
# Usage:
#   ./scripts/run_yolopx_manifest.sh            # all datasets, full
#   MAX_SAMPLES=20 ./scripts/run_yolopx_manifest.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
YOLOPX_REPO="${YOLOPX_REPO:-/home/gauravb/Projects/road_readiness_t3/YOLOPX}"
WEIGHTS="${WEIGHTS:-$YOLOPX_REPO/weights/epoch-195.pth}"
MODEL_NAME="${MODEL_NAME:-yolopx}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests}"
OUT_DIR="${OUT_DIR:-outputs/results}"
mkdir -p "$OUT_DIR"

MAXARG=(); [[ "$MAX_SAMPLES" -gt 0 ]] && MAXARG=(--max-samples "$MAX_SAMPLES")

# dataset -> split label (does not affect metrics; matches the native-run labels)
DATASETS=(bdd100k_lane curvelanes culane tusimple)
declare -A SPLIT=( [bdd100k_lane]=val [curvelanes]=valid [culane]=test [tusimple]=test )

for ds in "${DATASETS[@]}"; do
  echo ""
  echo "=================================================================="
  echo ">>> YOLOPX on ${ds} (manifest)"
  echo "=================================================================="
  if ! "$PY" -m lane_eval.cli.run_lane_eval \
      --yolopx-repo "$YOLOPX_REPO" --weights "$WEIGHTS" --model-name "$MODEL_NAME" \
      --device "$DEVICE" --img-size "$IMG_SIZE" --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" --per-image \
      --manifest "$MANIFEST_DIR/$ds/manifest_$ds.json" --split "${SPLIT[$ds]}" \
      "${MAXARG[@]}" --output "$OUT_DIR/${MODEL_NAME}_${ds}.json"; then
    echo "WARN: ${ds} run failed — continuing" >&2
  fi
done

echo ""
echo ">>> Building model-comparison table"
"$PY" -m evaluation.build_model_table --results-dir "$OUT_DIR" --out "$OUT_DIR/model_comparison.md" || true
