# D-Metric Evaluation Methodology (audited, construct-valid)

**What this is.** The construct-valid methodology for the D-suite (detection
performance indicators), produced by auditing the proposed methodology in
`.rr_audit_tmp/proposal.md` against five axes — (a) the Task 3 objective + binding
construct, (b) dataset semantics, (c) actual YOLOPX / CLRerNet outputs, (d) relevant
standards, (e) peer-reviewed lane-eval literature. It **supersedes the scratch
proposal** as the design target. The audit reasoning behind every verdict lives in
[d_metrics_proposal_audit.md](d_metrics_proposal_audit.md); the live-engine behaviour
it revises is documented in [metrics_reference.md](../metrics_reference.md) and
[evaluation_protocol.md](../evaluation_protocol.md).

**What this is NOT.** It does not describe numbers the pipeline publishes today. The
live engine faithfully implements the v2-report Table 16 (presence-based D1,
stroke-dilated pixel-IoU D3, pooled D6, …). This methodology is the audited successor;
each section is tagged with its activation status so the doc never misrepresents a
current output.

## Binding construct (do not weaken)

The detected object is **visible longitudinal lane-line markings / marking
evidence** — *not* lanes, lane corridors, lane centers, vehicle paths, or road
boundaries. The D-suite measures **machine detectability of roadway-marking evidence
using frozen reference image processors**. It does **not** claim ADS safety,
lane-keeping, or field-certified road readiness. Every threshold here is an
**image-analysis** tolerance in normalized image space — never a ground distance,
never a safety margin.

## Status legend

| Tag | Meaning |
|---|---|
| **LIVE** | Already computed by `evaluation/d_metrics.py` today. |
| **δ-GATED** | Computed only once a tolerance `delta` is frozen on the calibration split; dormant (`None`) until then. The number differs from the current published value → a **number move**, gated behind δ-freeze (see [evaluation_protocol.md §5](../evaluation_protocol.md)). |
| **τ_C-GATED** | Needs the CLRerNet per-curve `scores` threaded into `build_d_record` (small code change; scores are preserved end-to-end). |
| **DEFER** | Not computable on the current data/cache; blocked with a stated unblock condition. |
| **DIAGNOSTIC** | Reported for interpretation only; never a headline D metric, never entered into any composite. |

## Component verdicts at a glance

| # | Component | Verdict | Status |
|---|---|---|---|
| §1 | Construct + image as unit of analysis | KEEP | LIVE |
| §2 | Canonical coordinate system | REVISE → `(x/D, y/D)` | LIVE (code already correct) |
| §3 | Marking-support representation (V, T, Sᵞ, Sᶜ, rules) | REVISE | mixed (Sᶜ τ_C-GATED; Sᵞ τ_Y DEFER) |
| §4 | **D1 — Visible Marking Support Coverage (recall)** | **REPLACE presence** *(user decision)* | **δ-GATED** |
| §5 | D2 — Unsupported Marking Response (1−precision) | REVISE | δ-GATED |
| §6 | D3 — Marking-Support Localization Error | REVISE (drop "cap at diagonal") | δ-GATED |
| §7 | D4 — Image-band readability profile | REVISE + RENAME | δ-GATED |
| §8 | D5 — Marking-Condition Sensitivity | REVISE (fix tag path first) | DEFER (tag-path defect) |
| §9 | D6 — Reference-processor concurrence | REJECT as a D metric | DIAGNOSTIC |
| §10 | D7 — confidence / temporal | DEFER (temporal); relocate confidence-mean | DEFER / DIAGNOSTIC |
| §11 | D8 — Operating-point robustness | DEFER | DEFER |
| §12 | Calibration / freeze protocol | REVISE | mixed |
| §13 | Aggregation & uncertainty | REVISE | mixed |
| §14 | Statistical & sensitivity analysis | REVISE | mixed |
| §15 | Validation tests | REVISE | mixed |
| §16 | Literature & standards traceability | REVISE | this doc |

## How each D metric differs from the "core" metric it resembles

The chapter-3 **core metrics** (IoU, Precision, Recall, F1) are generic pixel-mask
statistics. Several D metrics carry similar names but are **different quantities** —
different unit, matching rule, and coordinate space. This table is the guard against
reading a D metric as if it were its core namesake.

| D metric | Core metric it resembles | Same or different | What actually differs |
|---|---|---|---|
| **D1** = coverage recall (centerline) | core **Recall** = `TP/(TP+FN)` on mask pixels | **Different** | Unit = GT *centerline point*, not pixel. Match = nearest prediction **within δ**, not exact pixel overlap. Prediction **not thickened** (1 px skeleton / polyline) vs. core recall's dilate-to-GT-width. Scored in isotropic `(x/D,y/D)` space, not raw pixels. |
| **D2** = 1 − precision (centerline) | core **Precision** = `TP/(TP+FP)` on mask pixels | **Complement + different basis** | Value is the complement (unsupported response), and the underlying precision is the same centerline/δ/no-thickening basis as D1 — not pixel precision. |
| **D3** = localization error (median, p95) | core **IoU** = `TP/(TP+FP+FN)` | **Different quantity entirely** | IoU is a dimensionless overlap *ratio*; D3 is a *distance* (fraction of image diagonal) between GT points and their nearest prediction. Not comparable to IoU; smaller is better vs. IoU's larger is better. |
| **D4** = per-band recall / loc-error profile | core **IoU restricted to a region** (live D4 near-field) | **Different basis + different bands** | Same centerline/δ basis as D1 (not pixel IoU). Bands are **upper 40% / middle 40% / near 15%**, with the **bottom 5% excluded as the assumed ego-vehicle hood** (was: lower 50% near-field / equal thirds). Both the near-field D4 and the profile now share this layout. See §7. |
| **D5** = ΔD1 / ΔD3_95 across condition groups | *(no core metric)* | **Not a core metric** | A stratified *difference* of D1/D3 between clear and degraded groups. Descriptive, not causal. |
| **D6** = concurrence diagnostic; missed = `1 − D1` | core **Recall complement** (`FN/GT`) | **Retired as a metric** | The live pooled `ΣFN/ΣGT` pools pixels across images (violates image-as-unit); its per-image form `1 − recall` **equals `1 − D1`**, so it is not shipped as a separate number. Concurrence (model agreement) is a diagnostic, not detectability. |
| **D7** = CLRerNet confidence mean | *(no core metric)* | **Not a core metric** | Model-reported per-curve confidence (CLRerNet only). YOLOPX per-pixel prob is discarded at write; the two confidences are **not comparable**. |

Only **D1/D2** share a name-and-family resemblance with a core metric while measuring a
genuinely different thing (the recall/precision *of centerline placement within δ*,
not of pixel overlap). **D3** reuses no core value at all. **D5/D7** have no core
counterpart. Full per-axis contrast for D1 is in the discussion notes; the general rule:
a D metric on the **centerline basis** is never the same number as its pixel-mask
namesake.

---

## 1. Evaluation construct and unit of analysis — **KEEP · LIVE**

Construct: *the extent to which reference-annotated longitudinal pavement-marking
evidence is detectably and spatially reliably represented by frozen reference image
processors under different roadway and image conditions.* NOT lane existence, lane
localization, path-planning, or ADS safety. **Unit of analysis = the image.** Always
aggregate at the image level; never pool pixels across images (the one live violation
of this — pooled D6 = ΣFN/ΣGT — is retired in §9/§13).

**Provenance caveat (required).** The reference "marking support" is
*annotation-derived*, and annotation semantics vary by dataset. In particular **CULane
GT is occlusion-extrapolated** — it annotates paint *behind* occluders, so
"reference-annotated" ≠ "visibly painted" there. This biases every GT-side metric:
it *deflates* recall/D1 (penalizes not detecting inferred paint) and *inflates*
precision/D2 (predictions over occluded-but-annotated paint count as supported). A
true fix needs a per-pixel visibility mask that does not exist → **DEFER**; the caveat
is not deferred and must accompany any CULane D-number. (See
[dataset_formats.md](../dataset_formats.md) and audit cross-cutting finding 1.)

## 2. Canonical coordinate system — **REVISE → `(x/D, y/D)` · LIVE**

Normalize every point `p̄ = (x/D, y/D)`, `D = √(W² + H²)`. **Reject the proposal's
per-axis `(x/W, y/H)`.** A single frozen tolerance δ must be *isotropic* to be one
meaningful distance; per-axis normalization with one δ yields an **elliptical** pixel
tolerance (`dx = δ·W`, `dy = δ·H`) whose anisotropy is 2.78× on CULane 1640×590,
1.78× on 1280×720, and *varies within* CurveLanes (two resolutions). Because lane
lines are near-vertical, the quality-relevant error is cross-track (horizontal) —
exactly the axis per-axis normalization makes most lenient, and inconsistently so.
The diagonal form gives a resolution-free **circle**, invariant under uniform scaling,
and matches all standard benchmark tolerances. `delta` is a **dimensionless fraction
of the diagonal**, taken as an explicit argument, never chosen internally.
Implemented in [marking_support.py](../../evaluation/marking_support.py)
(`to_canonical`, `diagonal`); this section documents what the code already does.

## 3. Common marking-support representation — **REVISE · mixed**

- **Vᵢ** = reference-annotated GT centerline support (set of GT centerline points for
  image *i*). Drop the word "visible" — it is annotation-derived, not a visibility
  proof; carry the CULane provenance caveat from §1.
- **Tᵢ** = permissible marking tube for evaluating unsupported predictions.
  Define **Tᵢ ≡ the δ-tube of Vᵢ** — no corridor geometry exists in the data, so any
  broader "permissible corridor" is undefined here and is **DEFER**red.
- **Sᵢᵞ(τ_Y)** = YOLOPX support = **skeletonized** binary mask to 1 px. Skeleton at the
  *fixed* stored operating point works now; a **tunable τ_Y is DEFERred** — the
  per-pixel probability is discarded at manifest write (`pred_mask > 0`; verified in
  [prediction_writer.py](../../lane_eval/manifest/prediction_writer.py)), so no τ_Y sweep is
  possible from cache.
- **Sᵢᶜ(τ_C)** = CLRerNet support = union of native polylines with per-curve
  confidence ≥ τ_C. The `scores` are preserved end-to-end → **τ_C-GATED** (thread
  `scores` + a `tau_c` filter into `build_d_record`; small change).
- **Rules — KEEP.** Skeletonize YOLOPX to 1 px; do **not** thicken CLRerNet curves;
  do **not** equate YOLOPX per-pixel probability with CLRerNet per-curve confidence.
  These are the core construct-validity fix and are literature-aligned (§16). Freeze
  the point-sampling / resampling density on the calibration split and pin it in a
  test (§15).

## 4. D1 — Visible Marking Support Coverage — **REPLACE presence · δ-GATED**

**D1 = recall of GT centerline points within δ of some prediction point.** Undefined
(`None`, never 0) when `|Vᵢ| = 0`. **This replaces presence-only "Detection Success
Rate."** *(Explicit user decision, overriding the audit's add-don't-replace
recommendation.)*

Consequences, accepted:
- **Number move / δ-gated.** Presence was computable on all 9298 images with no GT and
  no δ; recall requires GT and a frozen δ. Until δ is frozen (§12), **D1 is `None`** —
  there is no interim presence fallback. The all-images / no-GT output-availability
  signal is **dropped**.
- **Redundancy resolved by the replacement.** Recall already exists in code as
  `D3p_support_recall`; the former D6 "missed-marking" quantity is `1 − recall`, which
  now equals **`1 − D1` exactly**. A separate `1 − recall` D6 would therefore be a
  literal duplicate of D1 → the D6 detectability role **collapses into D1** (see §9);
  "missed marking" is reported as the reading `1 − D1`, not as its own metric.

## 5. D2 — Unsupported Marking Response — **REVISE · δ-GATED**

**D2 = proportion of prediction support points farther than δ from Tᵢ = `1 − precision`**
(already `1 − D3p_precision` in code). Empty-prediction and empty-GT policies must be
explicit and match the live None-semantics (empty prediction → excluded + counted in an
empty-rate, not scored 0). **Native lane-count agreement is demoted to a DIAGNOSTIC**
(reported only where genuine native instances exist — CLRerNet polylines, TuSimple
polyline GT — never from connected-component counts on merged masks); it is not the D2
headline. Note the CULane precision-inflation caveat from §1.

## 6. D3 — Marking-Support Localization Error — **REVISE · δ-GATED**

**D3_50 (median) and D3_95 (p95)** of residuals `eᵢ(p̄) = d(p̄, S)` on the un-thickened
centerline basis, in the isotropic canonical frame, at frozen δ. **Drop the proposal's
"cap at diagonal" censoring** — capping missing predictions at the diagonal is a biased
imputation that violates the no-coercion rule (§13). Instead **exclude** images with no
prediction and **report the empty/undefined rate alongside**, matching the live
None-semantics. Localization error is a nonnegative **fraction of the image diagonal**,
never a physical distance. (Live basis: `d_metrics.compute_d3_prime`, reported as
`D3p_loc_err_median / _p95`.)

## 7. D4 — Image-band readability profile — **REVISE + RENAME · δ-GATED**

Recall/localization computed **by image band**, reported as `D4p_{upper,middle,lower}`.
**Rename away from "longitudinal" / "near-field"** — without camera calibration these
are *image bands*, not physical distances or a vehicle-relative near field. **Forbid
any flat-ground homography** to fabricate distance. Apply a **minimum-n gate** per band.
Caveat the CULane far/upper band (its occlusion-extrapolated GT concentrates where paint
is least visible). No physical-distance claim unless real calibration exists (it does not).

**Band layout (fractions of image height from the top).** The bands are **not** equal
thirds. A fixed bottom slice is excluded as the assumed ego-vehicle hood, so the "near"
band is a road region rather than sheet metal:

| band | `y/H` range | share |
|---|---|---|
| upper | `[0.00, 0.40)` | 40% |
| middle | `[0.40, 0.80)` | 40% |
| lower / near | `[0.80, 0.95)` | 15% |
| **hood — excluded** | `[0.95, 1.00]` | 5% (covered by no band) |

Defined once in [marking_support.py](../../evaluation/marking_support.py) (`BAND_EDGES`,
`HOOD_IGNORE_FRACTION`, `NEAR_BAND_FRACTION`); the legacy near-field D4
(`compute_d4_near_field_iou`) derives the **same** near band `[0.80, 0.95)` from those
constants, so both bases agree. **Hood caveat (required):** the 5% hood exclusion is a
**fixed, uncalibrated image-region assumption applied identically to every dataset** —
the hood is not truly a constant fraction across cameras, and some images have no hood
at all (a small amount of genuine near-field road is then also dropped). It is a
uniform image choice, **not** a dataset-specific exception, and carries no
physical-distance meaning.

## 8. D5 — Marking-Condition Sensitivity — **REVISE · DEFER (blocked)**

**ΔD1 and ΔD3_95** between a condition group *g* and a clear baseline *r*, with sample
counts, descriptive only (not causal). **Blocked by the tag-path defect:** the runner
reads `meta.tags.summary[...]` (nonexistent); the correct path is
`meta.predicted_tags.by_dimension["Observed Marking Visibility"]` (a list) — already
used by `calibration_split.py`. **Fix the tag path first, then compute.** Lead with
**clear-vs-degraded** (thousands per stratum). **No bootstrap CI on occluded strata** —
they are degenerate (dataset-wide 22/9298; after the calibration draw the eval set has
0 occluded for CULane and CurveLanes); occluded is **descriptive-only**.

## 9. D6 — **REJECT as a D metric · DIAGNOSTIC**

Reference-processor **concurrence** (per GT point: Both / YOLOPX-only / CLRerNet-only /
Neither) measures **model agreement, not detectability**, and is redundant with the
D1/D3 margins. **Demoted to a DIAGNOSTIC** ("concurrence ≠ correctness"). The former
live D6 (pooled `ΣFN/ΣGT`) is retired for two reasons: it pools pixels across images
(violates §1) **and**, under the §4 replacement, its per-image form `1 − recall`
equals `1 − D1`. **There is no standalone D6 detectability number** in this
methodology; missed-marking is the reading `1 − D1`.

## 10. D7 — confidence / temporal — **DEFER / DIAGNOSTIC**

- **Temporal flip rate: DEFER.** No temporal/sequence data exists (single frames; e.g.
  TuSimple installs one frame per clip). Do **not** substitute prediction-only temporal
  consistency — a stable hallucination would look "consistent." Unblock: multi-frame
  sequences with GT.
- **Confidence mean: relocate to a CLRerNet-only DIAGNOSTIC.** Only CLRerNet carries a
  meaningful per-curve confidence; YOLOPX per-pixel probability is discarded at write
  (§3). The two are different quantities and **must not be compared or pooled across
  models** (see §16). Report as unavailable where absent; never substitute a placeholder.

## 11. D8 — Operating-point robustness — **DEFER**

Max D1 subject to D2 ≤ β, with the full D1-vs-D2 curve. **DEFERred:** presupposes §4/§5,
and a threshold sweep is not computable from the single-operating-point cache
(YOLOPX prob discarded; only CLRerNet `scores` survive → at most a CLRerNet-only curve
once τ_C is wired). Never compare τ_Y vs τ_C numerically. Unblock: re-inference that
persists YOLOPX per-pixel probability.

## 12. Calibration protocol — **REVISE · mixed**

Distinct calibration / final partitions;
[calibration_split.py](../../evaluation/calibration_split.py) →
`dataset/calibration/calibration_split.json`. Freeze on the **calibration partition
only**, record in output metadata:
- **δ — freezable now** ([delta_calibration.py](../../evaluation/delta_calibration.py) +
  `run_delta_calibration.py` → `dataset/calibration/d3prime_frozen.json`; knee
  selection + bootstrap stability gate).
- **τ_C — freezable once wired** (τ_C-GATED).
- **τ_Y, β, band boundaries** — **not all frozen** on this data; scope honestly rather
  than claim a freeze. τ_Y is DEFERred (§3).
- **"Sequence-aware split" is vacuous** here (single frames, no adjacent frames). The
  real, documented risk is that the stratified draw **depletes the occluded eval
  strata**; the split must protect condition strata, not sequences.

## 13. Aggregation and uncertainty — **REVISE · mixed**

Per-image first → per-processor / per-dataset → condition-stratified; macro-average
only if justified. **Bootstrap 95% CI by image** (temporal-by-sequence is vacuous
here); wire per-image CI into `summarize_d_records` (pattern already in repo). **No
pooling of pixels** → the pooled D6 is retired (§9). `None` + reason is **excluded,
never coerced to zero**. **The P/R/F1 triple is reported once:** D1 (recall),
D2 (1−precision), and the F1/localization summary are the *same* triple — never
triple-counted in any composite. **No composite D score** until construct validity,
non-redundancy, stability, uncertainty, weighting, sensitivity, and anti-gaming are all
demonstrated; on single-frame pilot data the honest outcome is **"no D composite for
this pilot,"** not an open TODO.

## 14. Statistical and sensitivity analysis — **REVISE · mixed**

Report sample sizes and CIs. **δ-sensitivity** of every conclusion is a real, built
strength (report metrics across a δ grid, not just at the frozen point). **Band-choice
sensitivity** is cheap to add. **τ_Y sensitivity is not computable** (prob discarded);
**τ_C sensitivity** becomes available once wired.

## 15. Validation tests — **REVISE · mixed**

Split explicitly, and **do not list gates for unbuilt metrics**:
- **DONE** — ≈35 existing numpy-only tests (canonical normalization, perfect
  prediction → recall=1 / (1−precision)=0 / loc_err=0, uniform offset, partial
  coverage, unsupported extra, empty pred, empty GT, None exclusion, threshold
  monotonicity, resolution scaling).
- **ADD NOW** — dashed-marking coverage; coordinate restoration after crop/pad/resize
  round-trip; point-sampling-density freeze; empty-rate reporting for excluded
  predictions.
- **DEFER-until-built** — duplicate-CLRerNet-curve handling (needs τ_C wiring);
  temporal/unavailable cases (no temporal data); D8 monotonicity (metric deferred).

## 16. Literature and standards traceability — **REVISE · this doc**

Four-way taxonomy, each metric assigned to exactly one bucket:
1. **Native model-qualifying metrics** — IoU / P / R / F1 (generic ML statistics;
   Jaccard 1912, Powers 2011).
2. **New D metrics** — D1–D7 as revised above.
3. **Diagnostics** — native lane count (§5), reference-processor concurrence (§9),
   CLRerNet confidence mean (§10).
4. **Deferred** — τ_Y-tunable support, D7-temporal, D8.

Required traceability statements:
- **Standards ≠ image tolerances.** No roadway standard (MUTCD / FHWA / MSMT-729 /
  ASAM / ASTM / NCHRP) prescribes any D metric; δ, bands, and localization error are
  **image-space** tolerances, not standard-derived physical thresholds. The MUTCD /
  MSMT-729 grounding in project memory applies to the *tag taxonomy*, not to the D
  metrics. See [d_metrics_audit.md](d_metrics_audit.md).
- **Lane-eval literature anchors.** The centerline / distance-within-δ basis is
  **closer to standard practice** than raw merged-mask pixel IoU: TuSimple uses point
  accuracy (`|Δx| < 20/cosθ`, GT lane matched at ≥0.85 of points, FP/FN lane-level);
  CULane uses lane-level F1 (each lane rasterized 30 px wide, lane IoU, TP if IoU>0.5);
  CLRerNet/LaneIoU uses an IoU matrix with one-to-one linear-sum assignment (TP if
  matched IoU>0.5/0.75). **All use tolerant / distance / instance matching, not raw
  merged-mask pixel IoU** — the live stroke-dilated D3 is the documented mechanism
  behind the spurious CLRerNet-over-YOLOPX ordering.
- **Two confidences are not comparable.** YOLOPX per-pixel probability and CLRerNet
  per-curve confidence are different quantities on different supports; never equate,
  compare, or pool them (§10).

---

## Explicitly NOT doing (standing constraint)

No composite D score. No arbitrary weights. No dataset-specific exceptions. No
physical-distance, ground-truth-distance, or ADS/road-readiness claim. No coercion of
`None` to zero. No claim that "reference-annotated" means "visibly painted" (CULane).
