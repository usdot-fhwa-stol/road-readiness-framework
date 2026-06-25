#!/usr/bin/env bash
# Run CLRerNet (DLA-34 backbone) lane evaluation across the available datasets,
# scored by the SAME shared harness as YOLOPX / HybridNets (the universal
# manifests + YOLOPX's SegmentationMetric), and write everything under
# outputs_cs/ so the model-comparison table covers CLRerNet apples-to-apples.
#
# CLRerNet is an mmdetection-based lane-LINE detector, so it runs in its own
# venv. This is therefore a THREE-STAGE flow per dataset:
#   1. inference     (CLRerNet venv) -> prediction manifest (exact lane points +
#                    mask PNGs) + overlays
#   2. native score  (repo env) -> CULane / TuSimple lane-level F1 — THIS is the
#                    paper's metric (lanes rasterised at 30px, matched at IoU>=0.5)
#                    and the headline number to compare against the paper.
#   3. pixel score   (repo env, YOLOPX SegmentationMetric) -> pixel IoU/F1 summary
#                    + per-image JSON/CSV, for the shared cross-model seg table.
#
# IMPORTANT — why two metrics: CLRerNet outputs thin lane *lines*, not dense
# segmentation. Pixel IoU between a thin predicted line and the GT (stage 3)
# drastically understates a line detector and is NOT comparable to the paper —
# the paper uses the stage-2 lane-level F1. Use stage 2 to sanity-check against
# the paper (~0.80 F1 on CULane); stage 3 only to slot CLRerNet into the existing
# pixel-seg comparison table alongside the segmentation models.
#
# Input is the universal manifest for each dataset (same manifests YOLOPX/
# HybridNets/SCNN use). The CULane-trained DLA-34 checkpoint is run on all four
# datasets — this is the cross-dataset generalisation eval.
#
# Usage:
#   ./run_clrernet.sh                 # full run, all images
#   MAX_SAMPLES=20 ./run_clrernet.sh  # quick smoke test (20 imgs/dataset)
#   DEVICE=cpu ./run_clrernet.sh      # force CPU
# Any config var below can be overridden via the environment.
set -euo pipefail

cd "$(dirname "$0")"

# ---- config (override via env) ----------------------------------------------
# Stage-1 (inference) interpreter: the CLRerNet venv (mmdet/mmengine/libs).
CLRERNET_PY="${CLRERNET_PY:-/shared/src/CLRerNet/clrernet/bin/python}"
# Stage-2 (scoring) interpreter: the normal repo env (has YOLOPX's metric).
PY="${PYTHON:-python}"
YOLOPX_REPO="${YOLOPX_REPO:-/home/gauravb/Projects/road_readiness_t3/YOLOPX}"  # shared SegmentationMetric

MODEL_NAME="${MODEL_NAME:-clrernet}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"            # 0 = all images
PER_IMAGE="${PER_IMAGE:-1}"               # 1 = also save per-image JSON+CSV
MASK_THICKNESS="${MASK_THICKNESS:-16}"    # stroke width for rasterising lane lines -> mask.
                                          # Only affects the stage-3 pixel-seg metric (NOT the
                                          # stage-2 native F1, which uses exact points). There is
                                          # no single fair width for pixel-IoU across datasets
                                          # (GT line thickness differs); 16 ~ CULane GT width.
MASK_TOLERANCE="${MASK_TOLERANCE:-8}"     # stage-3 stroke-width tolerance (px). Raw pixel-IoU
                                          # between a thin lane LINE and the thick GT seg label is
                                          # dominated by width mismatch, not geometry (verified:
                                          # CULane IoU 0.16->0.38, BDD precision 0.07->0.42). This
                                          # dilates BOTH masks by the tolerance before the SAME
                                          # SegmentationMetric so the score reflects geometric
                                          # agreement within N px. 0 = raw IoU. ~8 ~ half a lane.
OVERLAY_SAMPLE="${OVERLAY_SAMPLE:-100}"   # pred-vs-GT overlay JPGs per dataset (0 = none)
OVERLAY_SEED="${OVERLAY_SEED:-42}"

# CLRerNet model (DLA-34 backbone, CULane-trained, EMA weights).
CLRERNET_ROOT="${CLRERNET_ROOT:-/shared/src/CLRerNet}"
CLRERNET_CONFIG="${CLRERNET_CONFIG:-$CLRERNET_ROOT/configs/clrernet/culane/clrernet_culane_dla34_ema.py}"
CLRERNET_CHECKPOINT="${CLRERNET_CHECKPOINT:-$CLRERNET_ROOT/clrernet_culane_dla34_ema.pth}"

# Output layout under outputs_cs/.
OUT_DIR="${OUT_DIR:-outputs_cs/clrernet}"
PRED_DIR="${PRED_DIR:-$OUT_DIR/predictions}"   # prediction manifests + mask PNGs
RESULTS_DIR="${RESULTS_DIR:-$OUT_DIR/results}" # pixel-seg summary JSONs (+ per_image/ subdir)
NATIVE_DIR="${NATIVE_DIR:-$OUT_DIR/results_native}"  # paper-comparable lane-level F1 JSONs
OVERLAY_DIR="${OVERLAY_DIR:-$OUT_DIR/overlays}"

# Universal input manifests (built by lane_eval.cli.build_manifest).
TUSIMPLE_MANIFEST="${TUSIMPLE_MANIFEST:-manifests/tusimple/manifest_tusimple.json}"
CULANE_MANIFEST="${CULANE_MANIFEST:-manifests/culane/manifest_culane.json}"
BDD100K_MANIFEST="${BDD100K_MANIFEST:-manifests/bdd100k_lane/manifest_bdd100k_lane.json}"
CURVELANES_MANIFEST="${CURVELANES_MANIFEST:-manifests/curvelanes/manifest_curvelanes.json}"

# -----------------------------------------------------------------------------
[[ -f "$CLRERNET_CONFIG" ]]     || { echo "ERROR: config not found at $CLRERNET_CONFIG" >&2; exit 1; }
[[ -f "$CLRERNET_CHECKPOINT" ]] || { echo "ERROR: checkpoint not found at $CLRERNET_CHECKPOINT" >&2; exit 1; }
[[ -x "$CLRERNET_PY" ]]         || { echo "ERROR: CLRerNet python not found at $CLRERNET_PY" >&2; exit 1; }
mkdir -p "$PRED_DIR" "$RESULTS_DIR" "$NATIVE_DIR"

MAXARG=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then MAXARG=(--max-samples "$MAX_SAMPLES"); fi
PERIMG=()
if [[ "$PER_IMAGE" == "1" ]]; then PERIMG=(--per-image); fi
OVLARG=()
if [[ "$OVERLAY_SAMPLE" -gt 0 ]]; then
  OVLARG=(--overlay-dir "$OVERLAY_DIR" --overlay-sample "$OVERLAY_SAMPLE" --overlay-seed "$OVERLAY_SEED")
fi

run() {  # $1 = dataset label, $2 = split, $3 = input manifest
  local ds="$1" split="$2" manifest="$3"
  local pred_manifest="$PRED_DIR/${ds}/predictions.json"

  echo ""
  echo "=================================================================="
  echo ">>> CLRerNet on ${ds}"
  echo "=================================================================="

  if [[ ! -f "$manifest" ]]; then
    echo "WARN: manifest not found ($manifest) — skipping ${ds}" >&2
    return
  fi

  # --- stage 1: inference (CLRerNet venv) -> prediction manifest + masks ---
  if ! "$CLRERNET_PY" -m lane_eval.cli.run_clrernet \
      --manifest "$manifest" \
      --pred-manifest "$pred_manifest" \
      --dataset "$ds" \
      --model-name "$MODEL_NAME" \
      --device "$DEVICE" \
      --clrernet-root "$CLRERNET_ROOT" \
      --config "$CLRERNET_CONFIG" \
      --checkpoint "$CLRERNET_CHECKPOINT" \
      --mask-thickness "$MASK_THICKNESS" \
      "${MAXARG[@]}" "${OVLARG[@]}"; then
    echo "WARN: ${ds} inference failed — continuing with remaining datasets" >&2
    return
  fi

  # --- stage 2: NATIVE lane-level F1 (the paper's metric) ---
  # CULane line-IoU F1 (30px, IoU>=0.5) for culane/curvelanes/bdd100k_lane;
  # TuSimple accuracy/FP/FN for tusimple. Uses CLRerNet's exact lane points.
  if ! "$PY" -m lane_eval.cli.eval_manifest \
      --gt-manifest "$manifest" \
      --pred-manifest "$pred_manifest" \
      --output "$NATIVE_DIR/${MODEL_NAME}_${ds}_native.json"; then
    echo "WARN: ${ds} native scoring failed — continuing" >&2
  fi

  # --- stage 3: pixel-seg scoring (shared YOLOPX SegmentationMetric) ---
  if ! "$PY" -m lane_eval.cli.eval_segmentation_manifest \
      --gt-manifest "$manifest" \
      --pred-manifest "$pred_manifest" \
      --yolopx-repo "$YOLOPX_REPO" \
      --model-name "$MODEL_NAME" \
      --dataset "$ds" \
      --split "$split" \
      --tolerance-px "$MASK_TOLERANCE" \
      --output "$RESULTS_DIR/${MODEL_NAME}_${ds}.json" \
      "${PERIMG[@]}" "${MAXARG[@]}"; then
    echo "WARN: ${ds} pixel scoring failed — continuing with remaining datasets" >&2
  fi
}

run "bdd100k_lane" "val"   "$BDD100K_MANIFEST"
run "curvelanes"   "valid" "$CURVELANES_MANIFEST"
run "culane"       "test"  "$CULANE_MANIFEST"
run "tusimple"     "test"  "$TUSIMPLE_MANIFEST"

echo ""
echo "=================================================================="
echo ">>> Building model-comparison table (all models in $RESULTS_DIR)"
echo "=================================================================="
"$PY" -m evaluation.build_model_table --results-dir "$RESULTS_DIR" --out "$RESULTS_DIR/model_comparison.md"

echo ""
echo "CLRerNet results written under: $OUT_DIR"
echo "  predictions   : $PRED_DIR/<dataset>/predictions.json (+ masks/)"
echo "  NATIVE F1     : $NATIVE_DIR/${MODEL_NAME}_<dataset>_native.json   <-- compare to the paper"
echo "  pixel summary : $RESULTS_DIR/${MODEL_NAME}_<dataset>.json"
echo "  per-image     : $RESULTS_DIR/per_image/${MODEL_NAME}_<dataset>_per_image.{json,csv}"
[[ "$OVERLAY_SAMPLE" -gt 0 ]] && echo "  overlays      : $OVERLAY_DIR/<dataset>/"
echo ""
echo "NOTE: the NATIVE F1 (stage 2) is the paper's metric — use it to compare to"
echo "the CLRerNet paper (~0.80 F1 on CULane). The pixel-seg numbers (stage 3)"
echo "understate a thin-line detector and exist only for the shared seg table."
