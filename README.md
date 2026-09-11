# road-readiness-framework

A framework to test and analyze lane-detection networks against one another and
across multiple datasets. It converts every supported dataset into a single
**universal manifest format**, runs each model's inference off that manifest,
and scores all models with the **same** metric code — so comparisons are
apples-to-apples and only the model's inference differs, never the evaluation.

Currently supported models: **YOLOPX**, **HybridNets**, and **CLRerNet**.
Currently supported datasets: **TuSimple**, **CULane**, **CurveLanes**, **BDD100K**.

## Pipeline

```
ROAD-READINESS LANE PIPELINE  —  datasets ─ manifest ─ adapter ─ model ─ prediction ─ metrics
==================================================================================================

 NATIVE DATASETS                MANIFEST                 ADAPTER            MODEL
 (raw, per-format)         (universal format)         (input side)      (inference)
─────────────────         ──────────────────         ────────────      ─────────────

 TuSimple ──────┐          build_manifest.py
 (JSON lines)   │          + dataset adapters
                │          + converters
 CULane ────────┤   ──►    ─────────────────   ──►   ManifestDataset
 (.lines.txt /  │          manifests/<ds>/            (lane_eval/        ┌─► YOLOPX
  laneseg PNG)  │          manifest_<ds>.json          manifest/         │   (run_lane_eval
                │            • image_path               dataset.py)      │    --manifest)
 CurveLanes ────┤   ──►      • width / height   ──►    reads manifest ───┤
 (.lines.json)  │            • ground_truth:           → LaneSample      │
                │              - mask_path              (img + GT)        └─► HybridNets
 BDD100K ───────┘              - lane_json                                    (run_hybridnets
 (lane mask PNG)               - natural_gt                                    --manifest)


 MODEL              PREDICTION                       METRICS
 (inference)        (output side)                    (scoring)
 ───────────        ─────────────                    ─────────

 YOLOPX ────┐       PredictionManifestWriter         ┌─ SEGMENTATION (mask IoU/F1)
            │       (lane_eval/manifest/             │   lane_segmentation.py
            ├─ ──►   prediction_writer.py)    ──►     │   (YOLOPX SegmentationMetric)
            │         • pred mask PNG                 │
 HybridNets ┘         • pred lane_json        ──►     └─ NATIVE (lane points)
                      (mask → lanes)                     eval_manifest.py
                                                          ├─ TuSimple: accuracy / FP / FN
                                                          └─ CULane:   precision / recall / F1


PER-DATASET METRIC ROUTING
──────────────────────────
 TuSimple    ─ manifest ─ adapter ─ {YOLOPX, HybridNets} ─ prediction ─ segmentation + TuSimple-native
 CULane      ─ manifest ─ adapter ─ {YOLOPX, HybridNets} ─ prediction ─ segmentation + CULane-native
 CurveLanes  ─ manifest ─ adapter ─ {YOLOPX, HybridNets} ─ prediction ─ segmentation + CULane-native*
 BDD100K     ─ manifest ─ adapter ─ {YOLOPX, HybridNets} ─ prediction ─ segmentation (mask-only GT)

 * CurveLanes uses the CULane-style line-IoU F1 evaluator.
   Note: CULane / BDD100K manifest GT lane_json is mask-derived (natural_gt=mask),
   so their native lane-point F1 is approximate; TuSimple/CurveLanes GT is native polylines.
```

The **manifest is the convergence point**: all four native formats collapse into
one schema, so everything downstream (adapter, models, metrics) is
dataset-agnostic. One adapter (`ManifestDataset`) feeds both models identically
via the `--manifest` flag.

## The universal manifest format

`build_manifest` converts each dataset into `manifests/<dataset>/manifest_<dataset>.json`:

```json
{
  "metadata": { "dataset": "tusimple", "num_samples": 2782, "h_sample_step": 10 },
  "samples": [
    {
      "sample_id": "clips_0530_1492626760788443246_0_20",
      "image_path": "/shared/data/TUSimple/test_set/clips/0530/.../20.jpg",
      "width": 1280,
      "height": 720,
      "ground_truth": {
        "mask_path": "/abs/path/to/repo/manifests/tusimple/masks/<sample_id>.png",
        "natural_gt": "lanes",
        "lane_json": {
          "h_samples": [10, 20, 30, "...", 710],
          "lanes": [[-2, -2, "...", 632, 625, "..."], "..."]
        }
      },
      "meta": {
        "camera_model": {
          "image_to_road_homography": [["...", "...", "..."], ["...", "...", "..."], ["...", "...", "..."]]
        },
        "calibration_meta": {
          "homography_source": "surveyed_camera_calibration",
          "metric_scale_available": true,
          "units": "m"
        }
      }
    }
  ]
}
```

- `lane_json` is the TuSimple-style representation: `h_samples` are row coordinates
  (every 10px, skipping 0 and the image height); `lanes` are x-positions per row,
  with `-2` marking a missing point.
- `natural_gt` indicates which ground-truth representation is native to the dataset
  (`lanes` for polyline datasets, `mask` for mask-only datasets).
- Every sample carries **both** a `mask_path` and a `lane_json`, so any evaluator
  (mask-based or lane-point-based) can be used.
- Sample `meta.camera_model`, `meta.image_to_road_homography`, and
  `meta.calibration_meta` are optional. When present, readiness I4 **and I5**
  share the same sensor calibration for both canonical and operational
  geometry (I5's calibrated bird's-eye-view curvature is only reported when
  `metric_scale_available: true` is explicitly set alongside `units: "m"` —
  merely setting `units` is not treated as an implicit metric-scale
  declaration). Old manifests without calibration remain valid and use the
  projective-normalized fallback (I5 reports this as `normalized_image_proxy`,
  with no metric curvature units).

Prediction manifests retain the same legacy fields and may additionally carry
native geometry:

```json
{
  "prediction": {
    "mask_path": "masks/sample.png",
    "lane_json": {"h_samples": [10, 20], "lanes": [[320, 321]]},
    "polylines": [[{"x": 319.75, "y": 10.5}, {"x": 321.25, "y": 20.5}]],
    "geometry_source": "native_polyline",
    "meta": {"coordinate_space": "original_image"}
  }
}
```

`polylines` and `meta` are optional. `geometry_source` is
`native_polyline`, `lane_json`, or `mask_derived`. Old prediction
manifests without these fields remain readable, and the original
`PredictionManifestWriter.add(sample_id, image_path, pred_mask)` call remains
valid.

## Datasets

| Dataset | Native GT | `natural_gt` | Samples | Split |
|---|---|---|--:|---|
| TuSimple | polyline JSON-lines | `lanes` | 2,782 | test |
| CULane | `laneseg` PNG masks (+ `.lines.txt`) | `mask` | 34,680 | test |
| CurveLanes | `.lines.json` polylines | `lanes` | 20,000 | valid |
| BDD100K | lane-mask PNGs | `mask` | 10,000 | val |

## Quickstart

### 1. Build the manifests
```bash
bash scripts/build_manifests.sh          # all four datasets (~15 min)
# or one dataset:
python -m lane_eval.cli.build_manifest \
    --dataset tusimple \
    --root /shared/data/TUSimple/test_set \
    --annotation-file /shared/data/TUSimple/test_label.json \
    --split test
```
Output goes to `manifests/<dataset>/` (gitignored — regenerable artifacts).

### 2. Run a model off the manifest
```bash
# YOLOPX (all datasets)
bash scripts/run_yolopx_manifest.sh

# HybridNets (pilot-only, runs under its own venv — see pilot_analysis/)
bash pilot_analysis/run_hybridnets_manifest.sh

# CLRerNet (runs under the CLRerNet venv)
source /shared/src/CLRerNet/clrernet/bin/activate
bash run_clrernet.sh

# or a single run:
python -m lane_eval.cli.run_lane_eval \
    --yolopx-repo /path/to/YOLOPX \
    --weights /path/to/epoch-195.pth \
    --manifest manifests/tusimple/manifest_tusimple.json \
    --pred-manifest outputs/pred/yolopx_tusimple_pred.json \
    --output outputs/results/yolopx_tusimple.json --per-image
```
Results are written to `outputs/results/<model>_<dataset>.json`.

Prediction-guided operational I1 can be emitted directly from matching GT and
prediction manifests without rerunning inference:

```bash
python -m evaluation.run_readiness \
    --manifest manifests/tusimple/manifest_tusimple.json \
    --pred-manifest outputs/pred/clrernet_tusimple_pred.json \
    --model-name clrernet \
    --output outputs/readiness/clrernet_tusimple.json \
    --per-image-output outputs/readiness/clrernet_tusimple.jsonl
```

Per-image records contain canonical GT-guided `I1*` fields and operational
`I1_pred*` fields. Both use the original RGB image as paint evidence.
Prediction geometry only selects the operational sampling corridor, and
`I1_pred` is not included in R1. When no visible paint exists along the corridor
(for example a pitch-dark night frame), I1 returns `None` with an explicit
unavailable reason rather than a fabricated near-zero continuity.

### 3. Score with native lane-point metrics
```bash
python -m lane_eval.cli.eval_manifest \
    --gt-manifest   manifests/tusimple/manifest_tusimple.json \
    --pred-manifest outputs/pred/yolopx_tusimple_pred.json
```

### 4. Build an evaluation output manifest (per-sample metrics + CSV + images)

Turn an evaluation run into an **output manifest** — the input manifest with each
sample's computed metric values attached, plus a flat CSV and per-sample overlay
images. Metric families:

- **core** — primary detection scalars (`D3` IoU / F1 / precision / recall)
- **D** — detection metrics `D1..D8`
- **I** — intrinsic image metrics `I1..I6`
- **R** — readability / detectability `R1..R4` (R1/R2 per-sample; R3/R4 aggregate)
- **C** — `C1..C5` correlations, **dataset-level only** (→ `metadata.evaluation.correlations_C`)

Integrated into a run (also saves pred-vs-GT overlays):

```bash
python -m evaluation.run_readiness --manifest <gt> --pred-manifest <pred> \
    --output-manifest outputs/readiness/output_manifest
```

Or merge an existing per-image JSONL back onto any manifest:

```bash
python3 -m evaluation.eval_output_manifest \
    --in-manifest dataset/manifests/manifest_all.json \
    --per-image outputs/readiness/per_image.jsonl \
    --report outputs/readiness/report.json \
    --out-dir outputs/readiness/output_manifest --copy-images
```

Writes `output_manifest.json` (each sample gains `metrics.{core,D,I,R}` +
`overlay_image`), `output_metrics.csv` (one row per sample), and `images/`. Every
run prints which metric families were added vs. absent.

## Results

> **Full pilot-analysis results:** [`results/Pilot_Analysis_Results.csv`](results/Pilot_Analysis_Results.csv)
> — one row per image (9,298 images across all 4 datasets) with I1–I6, the 43
> scenario/condition tags, per-model (YOLOPX/CLRerNet) D-metrics, and the R1/R2
> composite scores. This is the table behind the pilot-analysis presentation
> (see `pilot_analysis/` for the model-comparison scripts that led to selecting
> YOLOPX + CLRerNet, and `docs/metrics_reference.md` for what each metric means).

The table below predates that full pilot run — it's the original **shared**
mask-style segmentation metrics (mask IoU / F1 / precision / recall) used to
choose which models to standardize on:

| Model | Dataset | Split | Images | IoU | F1 | Precision | Recall |
|---|---|---|--:|--:|--:|--:|--:|
| yolopx | tusimple | test | 2,782 | 0.4960 | 0.6631 | 0.6340 | 0.6950 |
| yolopx | culane | test | 34,680 | 0.2626 | 0.4159 | 0.4616 | 0.3785 |
| yolopx | curvelanes | valid | 20,000 | 0.3845 | 0.5554 | 0.5292 | 0.5844 |
| yolopx | bdd100k_lane | val | 10,000 | 0.2006 | 0.3342 | 0.2058 | 0.8893 |
| hybridnets | tusimple | test | 2,782 | 0.4721 | 0.6414 | 0.7763 | 0.5464 |
| hybridnets | culane | test | 34,680 | 0.1982 | 0.3309 | 0.5972 | 0.2288 |
| hybridnets | curvelanes | valid | 20,000 | 0.3267 | 0.4925 | 0.5950 | 0.4201 |
| hybridnets | bdd100k_lane | val | 10,000 | 0.2291 | 0.3727 | 0.2496 | 0.7355 |
| clrernet | tusimple | test | 2,782 | 0.1353 | 0.2383 | 0.8021 | 0.1399 |
| clrernet | culane | test | 34,680 | 0.1961 | 0.3279 | 0.8361 | 0.2039 |
| clrernet | curvelanes | valid | 20,000 | 0.1374 | 0.2416 | 0.8015 | 0.1422 |
| clrernet | bdd100k_lane | val | 10,000 | 0.1232 | 0.2193 | 0.2525 | 0.1938 |

### Preliminary model comparison (mean across datasets)

| Model | Mean IoU | Mean F1 | Mean Precision | Mean Recall | Cross-Dataset Consistency |
|---|--:|--:|--:|--:|--:|
| yolopx | 0.3359 | 0.4922 | 0.4576 | 0.6368 | 0.7430 |
| hybridnets | 0.3065 | 0.4594 | 0.5545 | 0.4827 | 0.7373 |
| clrernet | 0.1480 | 0.2568 | 0.6730 | 0.1700 | 0.8367 |


### CLRerNet adapter note

CLRerNet is integrated through the same universal manifest and prediction-writer
path as the other models. The adapter reads `ManifestDataset`, runs CLRerNet
inference, preserves its original-image float polylines as the primary geometry,
also rasterizes them into binary masks for segmentation evaluation, writes both
representations through `PredictionManifestWriter`, and can save random overlay
images for visual QA. Rasterized masks and compatible lane_json remain available
to existing evaluators.

The CLRerNet results above should be interpreted as mask-style segmentation
scores. CLRerNet naturally predicts sparse lane polylines, while the shared
segmentation metrics compare rasterized prediction masks against ground-truth
lane masks. This is why CLRerNet shows high precision on TuSimple, CULane, and
CurveLanes but lower recall: the predicted mask is thinner/sparser than the
ground-truth mask.

_Cross-Dataset Consistency = 1 − (std/mean) of F1 across datasets (1.0 = identical
across domains)._ The comparison table is auto-generated by
`evaluation/build_model_table.py` into `outputs/results/model_comparison.md`.

> **Manifest path is validated.** Running both models through the manifest +
> adapter path reproduces the original native-adapter metrics **exactly** (max
> metric difference `0.00e+00` across all 8 model×dataset runs, with identical
> image counts).

## Repository layout

```
lane_eval/
  datasets/      native dataset adapters (tusimple, culane, curvelanes, bdd100k) → LaneSample
  converters/    lanes↔mask, lanes↔tusimple lane_json (universal conversions)
  manifest/      build_manifest (generator), ManifestDataset (reader),
                 PredictionManifestWriter (prediction output)
  evaluators/    lane_segmentation (mask IoU/F1), tusimple_native, culane_native
  cli/           build_manifest, run_lane_eval (YOLOPX), run_clrernet, eval_manifest
  schema/        LaneSample / LaneTarget / LanePrediction
evaluation/      readiness_metrics/d_metrics (I/D/R engine), run_hybridnets,
                 per_image_metrics, build_model_table, save_predictions
tagging/         VLM-based scenario/condition tagging (43-tag taxonomy)
scripts/         build_manifests.sh, run_yolopx_manifest.sh, eval_yolopx_*.sh,
                 figure/panel renderers
configs/         dataset + model + eval configs
manifests/       generated manifests + masks (gitignored, regenerable)
outputs/         results JSON, per-image metrics, comparison table
docs/            dataset/evaluation/metrics reference docs, Algorithms/ (I/D
                 metric definitions), dev_notes/ (audit trail, not the spec)
pilot_analysis/  historical 4-model (YOLOPX/HybridNets/CLRerNet/SCNN) comparison
tests/           pytest suite (adapters, metrics, calibration)
```

## Design notes

- **One metric for all models.** Mask scoring uses YOLOPX's own
  `SegmentationMetric` for every model — no model uses its own native eval, so the
  comparison is fair.
- **Models are inference-only.** Each model contributes only its weights and
  preprocessing; ground truth, manifest, and scoring are shared.
- **HybridNets runs in its own venv** (`/shared/src/HybridNets/hybridnets/bin/python`)
  because of its timm/efficientnet dependencies; it still imports this repo's
  `lane_eval` adapters and the shared metric.
- **Caveat:** CULane and BDD100K provide ground truth as masks, so their manifest
  `lane_json` is approximated from the mask (connected components). Native
  lane-point F1 for these is therefore approximate; TuSimple and CurveLanes use
  native polyline ground truth.
