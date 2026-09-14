#!/usr/bin/env bash
# Run I1/I4/I5 (full readiness pipeline) on the human-tagged CurveLane +
# CULane subset (tagging/). TuSimple and BDD100K are skipped:
# TuSimple's tagged rows are unresolvable (no id->clip mapping) and
# BDD100K's are test-split with no lane GT in this dataset copy.
#
# CULane's 46 tagged frames already have cached yolopx/clrernet predictions
# from the full dataset run, so no new inference runs for it. CurveLane's 40
# tagged images are from `train` (the full run only covered `valid`), so this
# script runs fresh YOLOPX + CLRerNet inference on just those 40 images.
#
# Usage:
#   YOLOPX_REPO=/shared/src/YOLOPX ./scripts/eval_tagged_subset.sh
#
# Env vars (all optional except YOLOPX_REPO):
#   OUT_ROOT          output folder for this run (default: outputs/tagged_subset_eval)
#   FULL_ROOT         where the existing full-dataset results/predictions live
#                     (default: outputs/full_model_rankings)
#   YOLOPX_REPO       path to the YOLOPX repo root (required)
#   YOLOPX_WEIGHTS    default: $YOLOPX_REPO/weights/epoch-195.pth
#   CLRERNET_ROOT     default: /shared/src/CLRerNet
#   CLRERNET_CKPT     default: $CLRERNET_ROOT/clrernet_culane_dla34_ema.pth
#   CLRERNET_PYTHON   python interpreter with mmcv/mmdet installed for CLRerNet
#                     (default: $CLRERNET_ROOT/clrernet/bin/python -- CLRerNet
#                     needs its own venv; it is NOT installed into the main env)
#   DEVICE            default: cuda:0
#   I1_PANEL_MAX_SIDE max image side (px) for the I1 overlay panels (default: 640)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-outputs/tagged_subset_eval}"
FULL_ROOT="${FULL_ROOT:-outputs/full_model_rankings}"
YOLOPX_REPO="${YOLOPX_REPO:-}"
: "${YOLOPX_REPO:?YOLOPX_REPO must point to the YOLOPX repo root}"
YOLOPX_WEIGHTS="${YOLOPX_WEIGHTS:-$YOLOPX_REPO/weights/epoch-195.pth}"
CLRERNET_ROOT="${CLRERNET_ROOT:-/shared/src/CLRerNet}"
CLRERNET_CKPT="${CLRERNET_CKPT:-$CLRERNET_ROOT/clrernet_culane_dla34_ema.pth}"
CLRERNET_PYTHON="${CLRERNET_PYTHON:-$CLRERNET_ROOT/clrernet/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
I1_PANEL_MAX_SIDE="${I1_PANEL_MAX_SIDE:-640}"

mkdir -p "$OUT_ROOT/pred" "$OUT_ROOT/readiness" "$OUT_ROOT/overlays"

echo "=================================================================="
echo ">>> 1/6  Building tagged-subset GT manifests (CurveLane + CULane)"
echo "=================================================================="
"$PY" tagging/build_tagged_subset_manifests.py

echo ""
echo "=================================================================="
echo ">>> 2/6  Building CULane prediction manifests from EXISTING results"
echo "         (no new inference -- reuses the full-dataset run's masks)"
echo "=================================================================="
"$PY" tagging/build_culane_tagged_pred_manifests.py \
    --full-root "$FULL_ROOT" --output-dir "$OUT_ROOT/pred"

echo ""
echo "=================================================================="
echo ">>> 3/6  YOLOPX inference on the 40 CurveLane (train) tagged images"
echo "=================================================================="
"$PY" -m lane_eval.cli.run_lane_eval \
    --yolopx-repo "$YOLOPX_REPO" --weights "$YOLOPX_WEIGHTS" --model-name yolopx \
    --device "$DEVICE" \
    --manifest tagging/output/universal_manifest_curvelane_tagged.json --split train \
    --pred-manifest "$OUT_ROOT/pred/yolopx_curvelane_tagged_pred.json" \
    --output "$OUT_ROOT/readiness/yolopx_curvelane_tagged_lane_eval.json" --per-image \
    --save-pred-dir "$OUT_ROOT/overlays" --overlay-sample 40

echo ""
echo "=================================================================="
echo ">>> 4/6  CLRerNet inference on the 40 CurveLane (train) tagged images"
echo "=================================================================="
# CLRerNet needs its own venv (mmcv/mmdet) -- NOT the main env's python.
"$CLRERNET_PYTHON" -m lane_eval.cli.run_clrernet \
    --manifest tagging/output/universal_manifest_curvelane_tagged.json \
    --pred-manifest "$OUT_ROOT/pred/clrernet_curvelane_tagged_pred.json" \
    --clrernet-root "$CLRERNET_ROOT" --checkpoint "$CLRERNET_CKPT" \
    --model-name clrernet --dataset curvelane_tagged --device "$DEVICE" \
    --overlay-dir "$OUT_ROOT/overlays/clrernet" --overlay-sample 40

echo ""
echo "=================================================================="
echo ">>> 5/6  I1/I4/I5 (full readiness pipeline) for all 4 combinations"
echo "=================================================================="
declare -A GT_MANIFEST=(
  [curvelane]="tagging/output/universal_manifest_curvelane_tagged.json"
  [culane]="tagging/output/universal_manifest_culane_tagged.json"
)
for ds in curvelane culane; do
  for model in yolopx clrernet; do
    pred="$OUT_ROOT/pred/${model}_${ds}_tagged_pred.json"
    echo "--- $ds / $model ---"
    "$PY" -m evaluation.run_readiness \
        --manifest "${GT_MANIFEST[$ds]}" --pred-manifest "$pred" \
        --model-name "$model" \
        --output "$OUT_ROOT/readiness/${ds}_${model}_readiness.json" \
        --per-image-output "$OUT_ROOT/readiness/${ds}_${model}_readiness.jsonl" \
        --per-image-csv "$OUT_ROOT/readiness/${ds}_${model}_readiness.csv"
  done
done

echo ""
echo "=================================================================="
echo ">>> 6/6  I1 overlay panels (raw | pred-vs-GT | I1 highlighted region)"
echo "=================================================================="
for ds in curvelane culane; do
  for model in yolopx clrernet; do
    pred="$OUT_ROOT/pred/${model}_${ds}_tagged_pred.json"
    echo "--- $ds / $model ---"
    "$PY" -m evaluation.render_i1_overlay_panels \
        --manifest "${GT_MANIFEST[$ds]}" --pred-manifest "$pred" \
        --model-name "$model" --dataset "${ds}_tagged" \
        --output-dir "$OUT_ROOT/i1_panels/${model}_${ds}_tagged" \
        --max-side "$I1_PANEL_MAX_SIDE"
  done
done

echo ""
echo "Done. Results under: $OUT_ROOT/"
echo "  $OUT_ROOT/readiness/<dataset>_<model>_readiness.json   -- summary (R1/R2/R3/R4, I/D means, C1-C5)"
echo "  $OUT_ROOT/readiness/<dataset>_<model>_readiness.jsonl  -- one full metric record per image (I1/I4/I5 canonical + _pred)"
echo "  $OUT_ROOT/readiness/<dataset>_<model>_readiness.csv    -- flat per-image scalar metrics"
echo "  $OUT_ROOT/pred/*_masks/, pred/*.json                    -- raw binary predicted lane masks + pred manifests"
echo "  $OUT_ROOT/overlays/yolopx/<dataset>/overlays/*.jpg      -- yolopx pred-vs-GT overlay images (TP/FN/FP)"
echo "  $OUT_ROOT/overlays/clrernet/curvelane_tagged/*.jpg      -- clrernet pred-vs-GT overlay images"
echo "  $OUT_ROOT/i1_panels/<model>_<dataset>_tagged/*.jpg      -- [raw | pred-vs-GT | I1 highlight] panel per image"
