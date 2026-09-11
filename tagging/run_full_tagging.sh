#!/bin/bash
# Full test/val/valid tagging of all 4 datasets (BDD100k, CuLane, TuSimple,
# CurveLanes) with Gemma-3-12B, batched. RUN THIS YOURSELF when ready.
#
# It DETACHES (setsid) so it keeps running after you close the terminal, and is
# RESUMABLE: re-run this exact script any time and it skips already-tagged images
# (state lives in full_tags/<dataset>.tags.jsonl).
#
# Usage:
#   ./run_full_tagging.sh            # default batch size
#   ./run_full_tagging.sh 32         # override batch size
#
# Watch progress:   tail -f <the log path it prints>
#                   wc -l full_tags/*.tags.jsonl
# Stop:             pkill -f tag_all.py
set -euo pipefail
cd "$(dirname "$0")"

BATCH_SIZE="${1:-32}"          # 32 = 1.09s/img @ 18.2GB peak (sweet spot)
MODEL="google/gemma-3-12b-it"
SPLITS="test,val,valid"        # NOTE: CurveLanes val dir is 'valid', not 'val'
LOG="/tmp/tagall_full_$(date +%Y%m%d_%H%M%S).log"

echo "Launching FULL tagging:"
echo "  model=$MODEL  batch_size=$BATCH_SIZE  splits=$SPLITS"
echo "  log=$LOG"
echo "  outputs=full_tags/{bdd100k,culane,tusimple,curvelanes}.tags.jsonl"

setsid nohup python3 tag_all.py \
  --dataset all --splits "$SPLITS" \
  --model "$MODEL" --batch-size "$BATCH_SIZE" \
  > "$LOG" 2>&1 < /dev/null &

echo "started PID $!"
echo "watch:  tail -f $LOG   |   wc -l full_tags/*.tags.jsonl"
echo "stop:   pkill -f tag_all.py    (re-run this script to resume)"
