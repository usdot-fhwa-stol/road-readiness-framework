# D-Metric Evaluation Protocol

**Scope.** How the detection ("D") performance indicators are computed end to
end: what is detected, which processors and datasets are supported, the exact
per-image and group math as implemented in code, and the calibration protocol
that freezes the one tunable tolerance (`delta`) used by the optional
centerline-localization basis (D3′).

This document describes the **live code** in
[evaluation/d_metrics.py](../evaluation/d_metrics.py) and its calibration
harness. Where it differs from the report specification, that is recorded in
[docs/metrics_reference.md](metrics_reference.md#implementation-status-and-deviations)
and the audit in [docs/dev_notes/d_metrics_audit.md](dev_notes/d_metrics_audit.md). For the meaning
of every field the metric writes, see the metrics reference; this file is the
*procedure*, not the field dictionary.

---

## 0. What the D metrics measure (and what they do not)

The detected object is **visible longitudinal lane-line markings / marking
evidence**, as recovered by frozen reference image processors from public
imagery. It is **not** a lane, lane center, corridor, vehicle path, or road
boundary.

> D metrics measure the **machine detectability of roadway-marking evidence**
> using frozen reference image processors. They do **not** measure ADS safety,
> lane-keeping performance, or field-certified road readiness, and must not be
> reported as if they did. (Report pp. 20–21, Objectives.)

Every threshold in this protocol is an **image-analysis** tolerance, calibrated
on data, never a roadway-design or safety standard.

---

## 1. Inputs

The runner ([evaluation/run_d_metrics.py](../evaluation/run_d_metrics.py))
consumes two artifacts and performs **no model inference** — predictions are
read from a cached prediction manifest:

| Input | Produced by | Read by |
|---|---|---|
| GT manifest (`manifest_<dataset>.json`) | `lane_eval.manifest.generator` | `ManifestDataset` |
| Prediction manifest (`*_pred.json`) | `run_lane_eval` (YOLOPX) / `run_clrernet` (CLRerNet) | `PredictionManifestReader` |

Both formats are specified in
[docs/dataset_formats.md](dataset_formats.md). The runner pairs GT and
prediction by `sample_id`; a missing prediction is counted as a no-output image
for D1 (not skipped), so detection success is not silently inflated.

---

## 2. The two spatial bases (never blended)

D3/D4/D6 and D3′ are **two different spatial bases** for the same underlying
question — placement agreement of recovered marking vs reference. They are
reported side by side and **never** averaged together.

### 2a. Legacy pixel basis — D3 / D4 / D6 (always on)

Predicted and GT markings are compared as **binary pixel masks**. Before pixel
TP/FP/FN are counted, the prediction is passed through
`standardize_stroke_width`: if the predicted stroke is materially thinner than
the reference stroke (as with a polyline detector rasterized a few px wide vs
the ~16 px reference rendering), the prediction is **dilated** to the reference
width (dilate-only, never erode), applied identically to every processor. This
is the report's chapter-3 "common representation" step.

**Known bias (documented, not hidden):** dilating the thin output inflates pixel
overlap and is the mechanism behind the reported CLRerNet-over-YOLOPX ordering
(audit §2, §7 risk 1). D3′ exists precisely to measure placement *without*
thickening.

### 2b. Centerline basis — D3′ (opt-in, δ-gated)

D3′ reduces each marking to a **1-px centerline point cloud** — native
polylines for CLRerNet (densified, **not** thick-rasterized), mask skeleton for
YOLOPX — then scores nearest-neighbour precision/recall/F1 and localization
error at a frozen tolerance `delta`, in a resolution-free, isotropic canonical
frame (§4). Nothing is dilated. See
[evaluation/marking_support.py](../evaluation/marking_support.py).

**Gating.** `build_d_record(..., delta=None)` (the default) omits every D3′
field, so the record is **byte-for-byte the legacy Table-16 record**. D3′
activates only when a frozen `delta` is supplied. This is deliberate: **no
published number moves** until someone freezes `delta` on the calibration split
and points the runner at the config (§5).

---

## 3. Per-image metrics (as implemented)

All from [evaluation/d_metrics.py](../evaluation/d_metrics.py). "→ None" means
the metric is genuinely undefined for that image and is **excluded** from
aggregation, never coerced to 0.

| ID | Function | Definition | → None when |
|---|---|---|---|
| **D1** | `compute_d1_detection` | `1[ pred pixels > 0 OR valid pred lanes > 0 ]` — presence only, no accuracy gate | never |
| **D2** | `compute_d2_lane_count` | `1[ pred_count == gt_count ]`. Pred count = native polyline count if available, else connected components (`MIN_COMPONENT_AREA=30`); source recorded in `D2_pred_count_source` | GT count == 0 |
| **D3** | `compute_d3_iou` | pixel `IoU = TP/(TP+FP+FN)` on the stroke-standardized prediction; P/R/F1 reported for context | GT empty |
| **D4** | `compute_d4_near_field_iou` | D3 IoU restricted to the **near band, rows 80%–95%** (`NEAR_FIELD_TOP_FRACTION=0.80`..`NEAR_FIELD_BOTTOM_FRACTION=0.95`); bottom 5% excluded as assumed ego-vehicle hood. An **image band**, not a physical near-field (was lower 50%) | pred empty |
| **D6** | `compute_d6_image_counts` | per-image `FN/GT` pixels; the group ratio pools these (§3a) | — (ratio None if GT==0) |
| **D7** | `compute_d7_confidence` | mean `lane_prob` over predicted pixels | `lane_prob` absent |
| **D5 group** | `visibility_group` + `_gap` | per-image visibility group only; the gap is a group metric (§3a) | group `not_applicable` |

### D3′ per-image fields (only when `delta` supplied), `compute_d3_prime`

Canonical nearest-neighbour scores at frozen `delta` (fraction of image
diagonal):

- `D3p_support_precision / _recall / _f1`
- `D3p_loc_err_median / _p95` — localization error in diagonal fractions
- `D4p_{upper,middle,lower}_f1 / _recall` — per-band readability (replaces D4)
- `D6p_unsupported_marking_ratio = 1 − recall` — folded in, **not** shipped as a
  separate metric (it is exactly the recall complement)
- provenance: `D3p_pred_source / _gt_source` (`native_polyline` / `mask_skeleton`
  / `empty`), `D3p_delta`, `D3p_n_pred_points / _n_gt_points`

None-semantics (from `marking_support.localization_scores`): empty GT → recall /
F1 / error None; empty pred → precision / F1 None; empty pred but non-empty GT →
precision is a defined **0** (every prediction unsupported), and GT→pred error is
None rather than a fabricated magnitude.

---

## 3a. Group aggregation (`summarize_d_records`)

Per-image first, then aggregate; `None` excluded throughout.

- **D1 success rate** = mean D1 over **all processed** images (not just eligible).
- **D2 accuracy** = mean `D2_count_match` over images with GT count > 0;
  `D2_component_fallback_share` reports how often the CC proxy was used.
- **D3** = mean / median / std of per-image IoU over annotation-eligible images
  (`gt_eligible = D6_gt_pixels > 0`).
- **D4** = mean near-field IoU.
- **D5** = `_gap(clear, {occluded})` and a broader
  `_gap(clear, {occluded, degraded})`. Reported only when **both** groups have
  ≥ `MIN_GROUP_FOR_D5 = 3` images; otherwise `gap=None` with a reason. Uses the
  human "Observed Marking Visibility" tags, **not** dark-pixel thresholds.
- **D6** = pooled `Σ FN / Σ GT` across eligible images (pixel-weighted, **not**
  a mean of per-image ratios).
- **D7** = mean confidence where available, else None; `D7_available` flags it.
- **D3′** (`_summarize_d3_prime`, only if any record carries D3′): per-image mean
  of each D3′ field, plus per-band means. Reported under the `D3_prime` key
  alongside — never blended into — the legacy D3/D4/D6.

---

## 4. Canonical normalization (D3′)

Image `W×H`, diagonal `D = sqrt(W² + H²)`. Every centerline point is normalized
`p̄ = (x/D, y/D)`. Dividing **both** axes by the same `D` makes distances
**resolution-free and isotropic**: `delta` is the same fraction of the diagonal
on both axes, unlike `(x/W, y/H)` which stretches distance per axis on
non-square images. Point-to-set distance is Euclidean nearest-neighbour;
`delta` is a dimensionless fraction of the diagonal. See
[marking_support.py](../evaluation/marking_support.py) `to_canonical`,
`_min_distances`, `localization_scores`.

---

## 5. Calibration protocol — freezing `delta`

`delta` is the **only** free parameter in the D3′ basis and MUST be frozen on a
held-out calibration split, **never** tuned on the evaluation set (audit §7 risk
6). The harness makes sufficiency *measured*, not asserted.

### 5.1 Stratified calibration split — [evaluation/calibration_split.py](../evaluation/calibration_split.py)

- Draws ~5% of samples, stratified by **(dataset × coarse visibility tier)**,
  with a per-stratum floor (`min_per_stratum=5`) and a per-stratum RNG seed so
  strata are independent and the draw is deterministic.
- Visibility tier from `meta.predicted_tags.by_dimension["Observed Marking
  Visibility"]`: `occluded` dominates → else `degraded` → else `clear` → else
  `unknown`.
- Records an order-independent manifest fingerprint (sha256) so the freeze can
  be shown to have used calibration, not eval, data.
- Artifact: [dataset/calibration/calibration_split.json](../dataset/calibration/calibration_split.json)
  — the current draw is **480 / 9298 = 5.16%**, 16 strata.

**Finding surfaced by the draw:** the strictly-occluded tiers are tiny (CULane 2,
CurveLanes 2, BDD 8, TuSimple 10 — the floor grabbed all available). `delta`
therefore cannot be calibrated on occluded imagery; it is set by the clear +
degraded mass. This is reported, not smoothed over.

### 5.2 δ selection + stability gate — [evaluation/delta_calibration.py](../evaluation/delta_calibration.py)

On the calibration images only:

1. Build per-image gt→pred / pred→gt canonical distance arrays (same centerline
   reduction as `compute_d3_prime`).
2. Compute the mean-F1-vs-δ curve over a geometric grid
   (`DEFAULT_DELTA_GRID`, 15 points from 0.002 to 0.05).
3. Select `delta` at the **Kneedle knee** of that curve, cross-checked against
   the p90 pooled localization error; fall back to the percentile rule if the
   curve has too few finite points.
4. **Bootstrap-stability gate** (`bootstrap_delta_stability`, 200 resamples of
   the calibration images): if `delta*` is stable within one grid step, 5% was
   enough; if not, raise the fraction (e.g. 10%) and re-freeze.

### 5.3 Freeze + read-back — [evaluation/d3prime_config.py](../evaluation/d3prime_config.py)

- The GPU-box driver
  [evaluation/run_delta_calibration.py](../evaluation/run_delta_calibration.py)
  writes `dataset/calibration/d3prime_frozen.json` (delta + provenance:
  split fingerprint, seed, achieved fraction, selection rule, bootstrap verdict)
  plus a diagnostics JSON (the full F1-vs-δ curve, per-processor δ).
- At scoring time, `run_d_metrics` resolves `delta` as: explicit `--delta` wins,
  else `d3prime_config.load_frozen_delta(--d3prime-config)`, else **None →
  D3′ dormant**. A present-but-malformed config raises rather than silently
  reverting to legacy.

**Operating points (`tau`).** Not swept here: the cached prediction manifests were
produced at each processor's existing operating point, so `operating_points` is
recorded as `None` ("manifest default"). A true τ sweep needs raw per-pixel
logits / per-curve scores at multiple thresholds, which the cached masks do not
carry; this is documented in the config `notes`, not faked.

---

## 6. Processor × dataset applicability

| D metric needs… | YOLOPX (dense seg) | CLRerNet (scored polylines) |
|---|---|---|
| Presence (D1) | ✅ nonzero mask | ✅ any curve |
| Centerline placement (D3′) | ✅ skeletonize mask | ✅ native polyline |
| Instance count (D2) | ❌ no instances → CC proxy | ✅ native curves |
| Per-line confidence (D7) | ❌ dense pseudo-prob only | ✅ native score (now preserved end to end) |

| Dataset | Native GT geometry | D3′ | D2 count | Temporal |
|---|---|---|---|---|
| CULane 1640×590 | mask-only (`lane_json=null`) | ✅ (GT centerline from mask) | CC proxy | ❌ |
| TuSimple 1280×720 | polyline (`lane_json`) | ✅ | ✅ | ❌ |
| BDD100K 1280×720 | mask-only (`lane_json=null`) | ✅ | CC proxy | ❌ |
| CurveLanes mixed | polyline | ✅ | ✅ | ❌ |

No dataset carries temporal, calibration, or pose data (audit §5), so no
metric-distance or temporal-stability claim is made.

---

## 7. Known defects (open, tracked)

1. **Visibility tag mis-read → D5 empty.** Both
   [run_d_metrics.py](../evaluation/run_d_metrics.py) `_visibility_tag` and
   [readiness_metrics.py:894](../evaluation/readiness_metrics.py#L894) read
   `meta.tags.summary["Observed Marking Visibility"]`, but the manifest stores
   the tag (as a **list**) at
   `meta.predicted_tags.by_dimension["Observed Marking Visibility"]`. On the
   current manifests D5 grouping is therefore empty. The calibration-split tier
   reader uses the correct path; the D-runner does not. **Not yet fixed** —
   fixing it changes which images enter the D5 group, so it is scoped as a
   separate, reviewed change.
2. **Stroke-dilation bias in D3/D4/D6** (§2a) — the reason D3′ exists; removing
   the dilation from the metric path is Phase C (changes published numbers,
   awaits go-ahead).
3. **D2 cross-model comparability** — a native polyline count and a CC-proxy
   count are not the same quantity and must not be compared as if equal.

---

## 8. Running it

```bash
# Legacy Table-16 D-record only (D3' dormant — nothing depends on delta):
python -m evaluation.run_d_metrics \
    --manifest   dataset/manifests/manifest_culane.json \
    --pred-manifest outputs/.../yolopx_culane_pred.json \
    --model-name yolopx --output-dir outputs/d_metrics/yolopx_culane

# Freeze delta on the calibration split (GPU box; needs cv2/skimage):
python -m evaluation.run_delta_calibration \
    --gt-manifest dataset/manifests/manifest_all.json \
    --pred-manifest yolopx=outputs/.../yolopx_all_pred.json \
    --pred-manifest clrernet=outputs/.../clrernet_all_pred.json \
    --calibration-split dataset/calibration/calibration_split.json \
    --out-config dataset/calibration/d3prime_frozen.json

# Score WITH D3' active (reads the frozen config automatically if present):
python -m evaluation.run_d_metrics \
    --manifest dataset/manifests/manifest_culane.json \
    --pred-manifest outputs/.../yolopx_culane_pred.json \
    --model-name yolopx --output-dir outputs/d_metrics/yolopx_culane \
    --d3prime-config dataset/calibration/d3prime_frozen.json
```

Outputs: `d_metrics_summary.json` (group), `d_metrics_per_image.csv`,
`d_metrics_per_image.jsonl`, and optional side-by-side panels with
`--render-panels`.
