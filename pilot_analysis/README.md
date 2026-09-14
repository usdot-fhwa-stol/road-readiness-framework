# Pilot analysis (4-model comparison)

Early in the project, four reference models were run through this repo's shared
`lane_eval` harness to pick which model(s) to standardize on: **YOLOPX**,
**HybridNets**, **CLRerNet**, and **SCNN**. The comparison used the same
ground-truth adapters and the same `SegmentationMetric` scoring for every
model, so only inference differed (see `evaluation/build_model_table.py` and
the model-comparison table in the top-level `README.md`).

**YOLOPX and CLRerNet were chosen** as the framework's reference processors
going forward. Everything in this folder is retained for historical/audit
purposes only and is not part of the current pipeline.

## Contents

- `run_yolopx.sh`, `run_hybridnets.sh`, `run_scnn.sh` — the original
  native-adapter driver scripts (pre-universal-manifest), one per model.
- `run_hybridnets_manifest.sh` — HybridNets run off the universal manifest,
  used to validate the manifest path reproduces native-adapter metrics.

`run_clrernet.sh` and `scripts/run_yolopx_manifest.sh` remain at their
original locations since YOLOPX and CLRerNet are still active. The underlying
model code for HybridNets (`evaluation/run_hybridnets.py`) and SCNN
(`lane_eval/cli/run_scnn.py`) also stays in place, since both are invoked as
Python modules (`python -m evaluation.run_hybridnets` /
`python -m lane_eval.cli.run_scnn`) and moving them would break that.

Paths inside these scripts have been adjusted (`cd "$(dirname "$0")/.."`) to
still resolve correctly from this subfolder.

## R2 / I-metrics correlation analysis

A second, related pilot analysis (run from repo root, e.g.
`python pilot_analysis/build_clean_combined_i_d.py`):

- `i_metrics_summary.csv` — input fixture (per-image I-metric scores).
- `build_clean_combined_i_d.py` — joins `i_metrics_summary.csv` with per-model
  core metrics into `pilot_output/combined_i_and_d_metrics.csv`.
- `compute_r2_scores.py` — adds R2 correlation columns, writing
  `pilot_output/combined_i_and_d_metrics_with_r2.csv`.
- `render_r2_panels.py` — renders R2 visualization panels from that CSV.

`pilot_output/` is gitignored (regenerable). These scripts use plain
repo-root-relative paths (no `__file__` gymnastics), so run them from the repo
root, not from inside `pilot_analysis/`.
