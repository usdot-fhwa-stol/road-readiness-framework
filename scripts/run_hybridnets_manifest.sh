#!/usr/bin/env bash
# Run HybridNets lane evaluation reading the UNIVERSAL MANIFESTS, scored by the
# same shared harness, writing the same result JSON + per-image files as
# run_hybridnets.sh. Runs under the HybridNets venv.
#
# Usage:
#   ./scripts/run_hybridnets_manifest.sh
#   MAX_SAMPLES=20 ./scripts/run_hybridnets_manifest.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${HYBRIDNETS_PY:-/shared/src/HybridNets/hybridnets/bin/python}"
HN_REPO="${HYBRIDNETS_REPO:-/home/gauravb/Projects/road_readiness_t3/HybridNets}"
YOLOPX_REPO="${YOLOPX_REPO:-/home/gauravb/Projects/road_readiness_t3/YOLOPX}"
WEIGHTS="${WEIGHTS:-$HN_REPO/weights/hybridnets.pth}"
MODEL_NAME="${MODEL_NAME:-hybridnets}"
PROJECT="${PROJECT:-bdd100k}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests}"
OUT_DIR="${OUT_DIR:-outputs/results}"
mkdir -p "$OUT_DIR"

MAXARG=(); [[ "$MAX_SAMPLES" -gt 0 ]] && MAXARG=(--max-samples "$MAX_SAMPLES")

DATASETS=(bdd100k_lane curvelanes culane tusimple)
declare -A SPLIT=( [bdd100k_lane]=val [curvelanes]=valid [culane]=test [tusimple]=test )

for ds in "${DATASETS[@]}"; do
  echo ""
  echo "=================================================================="
  echo ">>> HybridNets on ${ds} (manifest)"
  echo "=================================================================="
  if ! "$PY" -m evaluation.run_hybridnets \
      --hybridnets-repo "$HN_REPO" --yolopx-repo "$YOLOPX_REPO" \
      --weights "$WEIGHTS" --project "$PROJECT" --model-name "$MODEL_NAME" \
      --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --per-image \
      --manifest "$MANIFEST_DIR/$ds/manifest_$ds.json" --split "${SPLIT[$ds]}" \
      "${MAXARG[@]}" --output "$OUT_DIR/${MODEL_NAME}_${ds}.json"; then
    echo "WARN: ${ds} run failed — continuing" >&2
  fi
done

echo ""
echo ">>> Rebuilding model-comparison table"
"$PY" -m evaluation.build_model_table --results-dir "$OUT_DIR" --out "$OUT_DIR/model_comparison.md" || true
