#!/usr/bin/env bash
# Personal eval container — only you can start it (it runs as your UID).
# Usage:
#   ./docker_eval.sh build              # build the image once
#   ./docker_eval.sh shell              # drop into an interactive shell
#   ./docker_eval.sh run <dataset> ...  # run lane eval directly
#
# Examples:
#   ./docker_eval.sh run bdd100k_lane \
#       --image-root /shared/data/bdd100k/images/val \
#       --lane-mask-root /shared/data/bdd100k/ll_seg_annotations/val \
#       --output /workspace/outputs/results/yolopx_bdd100k.json
#
#   ./docker_eval.sh run curvelanes \
#       --root /shared/data/Curvelanes \
#       --split valid \
#       --output /workspace/outputs/results/yolopx_curvelanes.json

set -euo pipefail

IMAGE="yolopx-eval:personal"
FRAMEWORK_DIR="$(cd "$(dirname "$0")" && pwd)"
YOLOPX_DIR="/home/gauravb/Projects/road_readiness_t3/YOLOPX"
SHARED_DATA="/shared/data"
WEIGHTS="${YOLOPX_WEIGHTS:-${YOLOPX_DIR}/weights/epoch-195.pth}"

DOCKER_RUN=(
    docker run --gpus all --rm -it
    --user "$(id -u):$(id -g)"
    -v "${FRAMEWORK_DIR}:/workspace"
    -v "${YOLOPX_DIR}:/yolopx"
    -v "${SHARED_DATA}:/shared/data"
    -e HOME=/tmp
    -w /workspace
    "${IMAGE}"
)

case "${1:-help}" in
  build)
    docker build -t "${IMAGE}" "${FRAMEWORK_DIR}"
    echo "Image built: ${IMAGE}"
    ;;

  shell)
    "${DOCKER_RUN[@]}" /bin/bash
    ;;

  run)
    shift  # remove 'run', rest is dataset + extra args
    DATASET="${1}"; shift
    "${DOCKER_RUN[@]}" python -m lane_eval.cli.run_lane_eval \
        --yolopx-repo /yolopx \
        --weights "${WEIGHTS}" \
        --dataset "${DATASET}" \
        "$@"
    ;;

  *)
    sed -n '2,20p' "$0"
    ;;
esac
