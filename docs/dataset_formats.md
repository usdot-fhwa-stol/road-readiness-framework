# Manifest & Dataset Formats

**Scope.** The on-disk JSON formats the evaluation pipeline reads and writes: the
**universal GT manifest**, the **prediction manifest**, the **tag block** the
metrics consume, and the per-dataset quirks that matter for the D metrics. This
documents the live readers/writers in
[lane_eval/manifest/](../lane_eval/manifest/); it is the field dictionary the
[evaluation protocol](evaluation_protocol.md) refers to.

All coordinates are **original-image pixels** `(x, y)`, origin top-left. No
format carries temporal, calibration, or pose data.

---

## 1. Universal GT manifest

Produced by `lane_eval.manifest.generator`, read by
[`ManifestDataset`](../lane_eval/manifest/dataset.py). One file per dataset
(`manifest_<dataset>.json`) plus a combined `manifest_all.json` (9298 samples,
used by the calibration split).

```jsonc
{
  "metadata": {
    "num_samples": 9298,
    "datasets": ["bdd100k", "culane", "curvelanes", "tusimple"],
    "tagger": "claude-4-5-sonnet-... (gateway, standard defs v4)"
  },
  "samples": [
    {
      "sample_id": "bdd100k_val_c9dc96aa-01dec7af",  // unique key; pairs GT↔pred
      "dataset": "bdd100k",
      "split": "val",
      "image_path": "bdd100k/images/....jpg",         // repo-relative
      "source_image_path": "/shared/data/....jpg",     // original absolute
      "width": 1280,
      "height": 720,
      "ground_truth": {
        "mask_path": "bdd100k/annotations/....png",    // binarized > 0 on load
        "lines_path": null,
        "source_gt_path": "/shared/data/....png",
        "natural_gt": "mask",                          // "mask" | "polyline"
        "lane_json": null                              // present for polyline GT
      },
      "meta": { /* tag block — see §3 */ }
    }
  ]
}
```

### What the reader exposes

`ManifestDataset[i]` → `LaneSample(image_id, image_path, width, height, target,
meta)` where `target = LaneTarget(mask, mask_path, meta)` and `target.meta`
carries `lane_json`, `natural_gt`, and any lane-style pattern hints (`style`,
`pattern`, `lane_style`, `line_style`, `marking_type`, `line_type`) lifted from
`ground_truth`. `sample.meta` is the raw sample-level `meta` block (§3).

- Relative `mask_path` is resolved against the manifest's own directory, so a
  manifest is readable from any working directory.
- Mask is loaded grayscale and binarized (`> 0 → 1`); a missing/unreadable mask
  yields `mask = None` (the D-runner substitutes an all-zero mask so the image
  still counts as a no-output for D1).

### `lane_json` (TuSimple-style, when present)

```jsonc
{ "h_samples": [240, 250, ...],           // shared y rows
  "lanes": [ [x0, x1, ...], ... ] }        // one x per h_sample; -2 = absent
```
`d_metrics.lanes_from_lane_json` turns this into a list of `Nx2` polylines,
keeping only points with `x >= 0` and lanes with ≥ 2 valid points.

---

## 2. Prediction manifest

Written by [`PredictionManifestWriter.add`](../lane_eval/manifest/prediction_writer.py),
read by [`PredictionManifestReader`](../lane_eval/manifest/dataset.py)
(`prediction_manifest_version: 2`).

```jsonc
{
  "metadata": { "prediction_manifest_version": 2, "model": "clrernet", ... },
  "samples": [
    {
      "sample_id": "tusimple_..._20",     // must match the GT manifest
      "prediction": {
        "mask_path": "pred_masks/....png",         // optional; binarized on load
        "lane_json": { "h_samples": [...], "lanes": [...] },
        "geometry_source": "native_polyline",       // native_polyline|lane_json|mask_derived|unavailable
        "polylines": [ [[x,y],[x,y],...], ... ],     // native float curves (CLRerNet)
        "scores":    [ 0.98, 0.71, ... ],            // per-curve conf, index-aligned to polylines
        "meta": { ... }                              // optional
      }
    }
  ]
}
```

### Contract details that matter for the metrics

- **`polylines` ↔ `scores` are index-aligned.** A lane with < 2 finite points is
  dropped, and its score is dropped in lockstep, so `scores[i]` always refers to
  `polylines[i]`. The reader exposes them as `LanePrediction.lanes` (list of
  `Nx2`) and `LanePrediction.scores` (float array, `NaN` for a missing score).
  `scores` is `None` when the manifest carries none.
- **CLRerNet per-curve confidence is preserved end to end** (Phase A of the
  audit): `run_clrernet` → writer `scores` → reader `LanePrediction.scores`.
  This unblocks a native D2 count and a real per-curve D7, instead of the old
  degraded mask-only ingest.
- **`geometry_source`** is inferred when absent: `native_polyline` if polylines
  exist, else `lane_json`, else `mask_derived`, else `unavailable`. D3′ prefers
  native polylines (densified, never thick-rasterized) and falls back to
  skeletonizing the mask.
- **`prob` (dense probability map)** is *not* part of this format. YOLOPX's dense
  softmax is discarded by the manifest writer, so D7 is generally unavailable
  from cached manifests for YOLOPX. (Audit §4, §7 risk 8.)

---

## 3. Tag block (`sample.meta`) — where the D5 grouping comes from

Tags are attached by the gateway tagger (see the [Gateway tagging setup]
memory). The **live, correct** location is:

```jsonc
"meta": {
  "existing_tags": { "weather": "...", "scene": "...", "timeofday": "..." },
  "predicted_tags": {
    "by_dimension": {
      "Observed Marking Visibility": ["Faded / Worn / Not Visible", "Low Contrast"],
      "Pavement Marking Type & Configuration": ["Solid Line", "Dashed Line", ...],
      "Operational Scenario": [...], "Roadway Context & Facility Type": [...],
      "Roadway Surface Type": [...], "Lighting & Weather": [...]
    },
    "labels": [...], "scores": {}, "reasoning": { "<dimension>": "..." }
  },
  "tags_encoded": { "per_dim_codes": { ... } }
}
```

`Observed Marking Visibility` is a **list** and drives the D5 visibility groups
(`clear` / `degraded` / `occluded`) and the calibration-split tiers.

> **⚠ Known defect.** The D-runner and `readiness_metrics.py` read the tag from
> `meta.tags.summary["Observed Marking Visibility"]` — a path that does **not
> exist** in these manifests (the real path is
> `meta.predicted_tags.by_dimension[...]`, and it is a list, not a scalar). So
> D5 grouping is currently empty in the live D-runner. The calibration-split
> tier reader
> ([calibration_split.py](../evaluation/calibration_split.py) `_sample_visibility_tags`)
> uses the correct path. Fixing the runner is tracked in
> [evaluation_protocol.md §7](evaluation_protocol.md#7-known-defects-open-tracked).

Manifest variants on disk: `*.json` (untagged), `*.vlm20.json` / `*.siglip20.json`
(sampled tag experiments), `*.tagged.json` (full tag block). Only the tag block
above is consumed by the metrics.

---

## 4. Per-dataset quirks

| Dataset | Resolution | GT geometry | `lane_json` | Notes |
|---|---|---|---|---|
| **CULane** | 1640×590 | mask (`natural_gt="mask"`) | `null` | all lane categories → 1; occlusion-extrapolated annotation conflates visible paint with inferred lane (audit §7 risk 5). D2 uses CC proxy. |
| **TuSimple** | 1280×720 | polyline | present | one frame per clip (ids end `_20`); native instance count available. |
| **BDD100K** | 1280×720 | mask | `null` | hashed ids; D2 uses CC proxy. |
| **CurveLanes** | mixed (2 resolutions) | polyline | present | canonical normalization removes the resolution difference for D3′. |

**Cross-dataset caveat.** The fixed ~16 px GT stroke is a different fraction of
image height per dataset (16/590 vs 16/720); no dataset normalizes this. The
canonical diagonal normalization (evaluation protocol §4) removes it for the
D3′ centerline basis; the legacy pixel D3/D4/D6 basis does not.

---

## 5. Calibration artifacts

| File | Written by | Contents |
|---|---|---|
| [dataset/calibration/calibration_split.json](../dataset/calibration/calibration_split.json) | `calibration_split.py` | held-out sample ids, per-stratum counts, seed, manifest fingerprint (480/9298 = 5.16%, 16 strata) |
| `dataset/calibration/d3prime_frozen.json` | `run_delta_calibration.py` | frozen `delta`, units (`fraction_of_image_diagonal`), selection rule, bootstrap-stability verdict, calibration provenance. **Absent until a freeze is run → D3′ dormant.** |
| `dataset/calibration/d3prime_calibration_diagnostics.json` | `run_delta_calibration.py` | full F1-vs-δ curve, per-processor δ, bootstrap detail |

Field-level meaning of the frozen config is in
[d3prime_config.py](../evaluation/d3prime_config.py) `build_frozen_config`.
