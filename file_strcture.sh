#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-lane-eval-adapters}"

mkdir -p "$REPO_NAME"
cd "$REPO_NAME"

# Top-level files
touch README.md
touch pyproject.toml
touch requirements.txt
touch .gitignore

# Configs
mkdir -p configs/datasets
mkdir -p configs/models
mkdir -p configs/eval

touch configs/datasets/bdd100k_lane.yaml
touch configs/datasets/tusimple.yaml
touch configs/datasets/culane.yaml
touch configs/datasets/curvelanes.yaml

touch configs/models/yolopx.yaml

touch configs/eval/yolopx_bdd100k_lane.yaml
touch configs/eval/yolopx_tusimple.yaml
touch configs/eval/yolopx_culane.yaml
touch configs/eval/yolopx_curvelanes.yaml

# Main package
mkdir -p lane_eval

touch lane_eval/__init__.py

# Schema / canonical data structures
mkdir -p lane_eval/schema
touch lane_eval/schema/__init__.py
touch lane_eval/schema/sample.py
touch lane_eval/schema/lane.py

# Dataset adapters
mkdir -p lane_eval/datasets
touch lane_eval/datasets/__init__.py
touch lane_eval/datasets/base.py
touch lane_eval/datasets/registry.py
touch lane_eval/datasets/bdd100k.py
touch lane_eval/datasets/tusimple.py
touch lane_eval/datasets/culane.py
touch lane_eval/datasets/curvelanes.py

# Model adapters
mkdir -p lane_eval/models
touch lane_eval/models/__init__.py
touch lane_eval/models/base.py
touch lane_eval/models/registry.py
touch lane_eval/models/yolopx.py

# Image/model transforms
mkdir -p lane_eval/transforms
touch lane_eval/transforms/__init__.py
touch lane_eval/transforms/letterbox.py
touch lane_eval/transforms/resize.py
touch lane_eval/transforms/normalize.py
touch lane_eval/transforms/inverse.py

# Converters
mkdir -p lane_eval/converters
touch lane_eval/converters/__init__.py
touch lane_eval/converters/lanes_to_mask.py
touch lane_eval/converters/mask_to_lanes.py
touch lane_eval/converters/tusimple_export.py
touch lane_eval/converters/culane_export.py
touch lane_eval/converters/curvelanes_export.py

# Evaluators
mkdir -p lane_eval/evaluators
touch lane_eval/evaluators/__init__.py
touch lane_eval/evaluators/lane_segmentation.py
touch lane_eval/evaluators/tusimple_native.py
touch lane_eval/evaluators/culane_native.py

# Visualization utilities
mkdir -p lane_eval/visualization
touch lane_eval/visualization/__init__.py
touch lane_eval/visualization/overlay.py
touch lane_eval/visualization/save_debug.py

# General utilities
mkdir -p lane_eval/utils
touch lane_eval/utils/__init__.py
touch lane_eval/utils/image_io.py
touch lane_eval/utils/geometry.py
touch lane_eval/utils/logging.py
touch lane_eval/utils/config.py

# CLI entry points
mkdir -p lane_eval/cli
touch lane_eval/cli/__init__.py
touch lane_eval/cli/evaluate.py
touch lane_eval/cli/inspect_dataset.py
touch lane_eval/cli/visualize_predictions.py

# Scripts
mkdir -p scripts
touch scripts/eval_yolopx_bdd100k_lane.sh
touch scripts/eval_yolopx_tusimple.sh
touch scripts/eval_yolopx_culane.sh
touch scripts/eval_yolopx_curvelanes.sh

chmod +x scripts/*.sh

# Tests
mkdir -p tests
touch tests/test_tusimple_adapter.py
touch tests/test_culane_adapter.py
touch tests/test_curvelanes_adapter.py
touch tests/test_bdd100k_lane_adapter.py
touch tests/test_lanes_to_mask.py
touch tests/test_lane_metrics.py

# Outputs
mkdir -p outputs/results
mkdir -p outputs/predictions
mkdir -p outputs/visualizations

# Docs
mkdir -p docs
touch docs/dataset_formats.md
touch docs/yolopx_adapter.md
touch docs/evaluation_protocol.md

# Basic .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.env

# Data and outputs
data/
datasets/
checkpoints/
weights/
outputs/results/*.json
outputs/predictions/*
outputs/visualizations/*

# Logs
*.log
wandb/
runs/

# IDE
.vscode/
.idea/

# OS
.DS_Store
EOF

# Minimal README
cat > README.md <<'EOF'
# Lane Eval Adapters

A task-aware evaluation framework for lane detection across multiple datasets and models.

Initial scope:

- Models: YOLOPX
- Datasets: BDD100K, TuSimple, CULane, CurveLanes
- Task: lane detection / lane segmentation evaluation

Core idea:

```text
dataset adapter -> canonical lane target
model adapter   -> canonical lane prediction
evaluator       -> metrics

EOF
