# Critical Audit — Proposed D-Metric Evaluation Methodology (v2)

**Status.** Findings only. **No code changed.** This audits the user's 16-section
"Proposed D-Metric Evaluation Methodology" against five axes — (a) the Task 3
objective + binding construct, (b) dataset semantics, (c) actual YOLOPX / CLRerNet
outputs, (d) relevant standards, (e) peer-reviewed lane-eval literature — grounded
in the **v2 report** (`task3draft1_report_2.pdf`, Table 16) and the live code.
Every verdict was produced by an independent auditor and then subjected to an
adversarial refutation pass (14 agents, 0 errors). Supersedes the v1-report audit
in [d_metrics_audit.md](d_metrics_audit.md) for the D-suite.

Proposal text audited: `.rr_audit_tmp/proposal.md`. Report extract: `.rr_audit_tmp/report2.txt`.

**Successor.** The construct-valid methodology these verdicts produced is written up
as a standalone target in [d_metrics_methodology.md](d_metrics_methodology.md), with
each component tagged LIVE / δ-GATED / τ_C-GATED / DEFER / DIAGNOSTIC. Note one
decision recorded there overrides §2.2 below: **D1 replaces presence with recall**
(user decision), so the "add, don't replace" recommendation in §2.2 is superseded —
concurrence is demoted (§9) and missed-marking is read as `1 − D1`.

---

## 0. The core framing the audit establishes

The **live engine already faithfully implements the v2 report's Table 16** (D1
presence-only, D2 lane-count, D3 pixel-IoU on stroke-dilated masks, D4 lower-half
IoU, D5 occlusion gap, D6 pooled ΣFN/ΣGT, D7 confidence mean, R2 composite). The
report is therefore **not** the construct-valid target — its own text disclaims the
R-composites as "analytical choices ... not validated safety or risk weights."

So the proposal is a **construct-validity upgrade that departs from Table 16**.
The decisive external anchor: **every standard lane benchmark uses tolerant /
distance / instance matching, not raw merged-mask pixel IoU** — TuSimple point
accuracy (|Δx| < 20/cosθ, GT lane matched at ≥0.85 of points), CULane lane-level F1
(30px line rasterization, IoU>0.5), CLRerNet/LaneIoU one-to-one linear-sum
assignment. The proposal's distance-within-δ recall/localization is **closer to
standard practice** than the live D3. The live D3's dilate-to-16px-then-pixel-IoU
is the documented mechanism behind the spurious CLRerNet>YOLOPX ordering.

**Most of the proposal's machinery already exists in code**, dormant: the
centerline basis (`marking_support.py`), the isotropic canonical normalization,
recall/precision/F1 at δ, per-band recall, the δ-freeze harness (calibration split
480/9298, knee selection, bootstrap stability), and 35 passing numpy tests. The
work is mostly **activation + relabeling + honest scoping**, not new invention.

---

## 1. Verdict summary (post-verification)

| # | Proposal component | Verdict | One-line |
|---|---|---|---|
| §1 | Construct + image as unit | **KEEP** | Matches binding construct verbatim; add CULane "occlusion-extrapolated GT ≠ visible paint" caveat. |
| §2 | Normalization `p̄=(x/W,y/H)` | **REVISE→ use `(x/D,y/D)`** | Per-axis δ is anisotropic & non-comparable across datasets; the built diagonal normalization is correct. |
| §3 V_i | Visible GT centerline support | **REVISE** | Drop the word "visible"; tag CULane/mask-set provenance. |
| §3 T_i | Permissible marking corridor | **REVISE** | Define T_i ≡ δ-tube of V_i (no corridor geometry exists); defer any broader variant. |
| §3 S_i^Y(τ_Y) | Skeletonized YOLOPX @ tunable τ_Y | **DEFER (τ_Y)** | Skeleton at fixed op-point works now; tunable τ_Y impossible — YOLOPX prob discarded at write. |
| §3 S_i^C(τ_C) | CLRerNet curves ≥ τ_C | **REVISE (wire it)** | τ_C available end-to-end but not yet threaded into the metric; small change. |
| §3 rules | Skeletonize/don't-thicken/don't-equate | **KEEP** | The core fix; literature-aligned. Freeze point-sampling density (doc/test item). |
| §4 D1 | Coverage-recall, **replaces** presence | **REVISE (ADD, don't replace)** | Recall is the valid detectability measure — but ADD it (already = D3p_recall); KEEP presence as "Output Availability." |
| §5 D2 | Unsupported response, **replaces** count | **REVISE** | Adopt 1−precision (=1−D3p_precision); specify empty policies; demote count to native-instance diagnostic. |
| §6 D3 | Localization error D3_50 / D3_95 | **REVISE** | Keep median+p95 on centerline basis; **drop "cap at diagonal"** (biased; violates §13) — exclude+report empty rate. |
| §7 D4 | Longitudinal readability profile | **REVISE** | Image bands are honest; rename (not "longitudinal"), forbid flat-ground homography, min-n gate, caveat CULane far band. |
| §8 D5 | Condition sensitivity ΔD1/ΔD3_95 + CI | **REVISE** | Fix tag-path bug first; **no bootstrap CI on tiny occluded strata**; lead with clear-vs-degraded. |
| §9 D6 | Reference-processor concurrence | **REJECT (as a D metric)** | Measures model agreement, not detectability; redundant with D1/D3 margins. Demote to diagnostic; keep 1−recall as the D6 quantity. |
| §10 D7 | Temporal flip rate | **DEFER** | No temporal/sequence data; caution vs prediction-only consistency is sound. Relocate confidence-mean to a CLRerNet-only diagnostic. |
| §11 D8 | Operating-point robustness curve | **DEFER** | τ sweep not computable from single-op-point cache; presupposes §4/§5; strengthen cross-model guard. |
| §12 | Calibration protocol / freeze | **REVISE** | δ freezable now; τ_Y/τ_C/β/bands are NOT all frozen — scope honestly. "Sequence-aware split" is vacuous (single frames); real risk = split depletes occluded strata. |
| §13 | Aggregation & uncertainty | **REVISE** | Principles correct; wire per-image bootstrap CI; replace pooled D6 with per-image 1−recall; state the D6 supersession explicitly. |
| §14 | Statistical & sensitivity analysis | **REVISE** | δ-sensitivity is a real built strength; τ_Y sensitivity not computable; band sensitivity cheap to add. |
| §15 | Validation tests | **REVISE** | Split into DONE (≈35 existing tests) / ADD-NOW / DEFER-until-metric-built; don't list gates for unbuilt metrics. |
| §16 | Literature & standards traceability | **REVISE** | 4-way taxonomy is sound; must add explicit "MUTCD/FHWA ≠ image tolerances" + lane-eval citations + D7-two-confidences note. |

**Tally:** KEEP 2 · REVISE 12 · DEFER 3 (τ_Y, D7-temporal, D8) · REJECT 1 (D6-concurrence).
No component was adopted verbatim except the construct statement (§1) and the
representation rules (§3-rules). No component is worthless — the intent survives
everywhere except D6-as-a-detection-metric.

---

## 2. The two hard adjudications

### 2.1 Normalization: reject `(x/W, y/H)`, keep the built `(x/D, y/D)`

A single frozen tolerance δ must be **isotropic** to be one meaningful distance.
Per-axis `(x/W, y/H)` with one δ yields an **elliptical** pixel tolerance
`dx=δ·W, dy=δ·H`:

- CULane 1640×590 → horizontal tolerance **2.78×** the vertical.
- TuSimple/BDD 1280×720 → **1.78×**. CurveLanes carries **two** resolutions → the
  anisotropy varies **within** one dataset.

So one δ means different pixel distances per axis **and** different anisotropy per
dataset — δ is not cross-dataset comparable. Because lane-lines are near-vertical,
the quality-relevant error is **cross-track (horizontal)** — exactly the axis
per-axis normalization makes most lenient, and inconsistently so. The diagonal
`(x/D, y/D), D=√(W²+H²)` gives a resolution-free **circle** (≈17.4px CULane, ≈14.7px
TuSimple at δ=0.01), invariant under uniform scaling — which is what
`marking_support.to_canonical` already implements and whose docstring already
rejects `(x/W,y/H)` for this exact reason. All standard benchmarks use isotropic
pixel tolerances. **Recommendation: change proposal §2 to `(x/D,y/D)`; the code is
already correct.** (Caveat retained: δ is an image-space tolerance, not a ground
distance — no calibration exists.)

### 2.2 D1: the proposal's one clear over-reach — "replace" should be "add"

Presence-only D1 (report/live) and coverage-recall measure **different properties**.
Presence = output availability / non-abstention: computable on all 9298 images with
no GT and no δ, and it is the only R2 term not collinear with spatial overlap.
Coverage-recall = the construct-valid, literature-aligned detectability measure —
**but it already exists** as `D3p_support_recall`, and its complement is `D6p`.
Redefining D1 as recall would **triple-count recall** across D1/D3′/D6, which the
proposal's own §13 non-redundancy rule forbids. **Recommendation: ADD recall
(report it as the recall arm of the D3′ P/R/F1 triple), KEEP presence renamed to
"Output Availability" so it stops reading as detection accuracy.** Same logic
applies to §5 D2: adopt 1−precision, but keep native-instance count as a diagnostic
rather than deleting it.

---

## 3. Cross-cutting findings (recur across components)

1. **CULane occlusion-extrapolated GT** breaks the word "visible" and biases every
   GT-side metric: it *deflates* recall/D1 (penalizes not detecting inferred paint
   behind occluders) and *inflates* precision/D2 (predictions on occluded-but-
   annotated paint count as supported). Must be caveated wherever V_i is used; a
   true fix needs a per-pixel visibility mask that does not exist → that correction
   is DEFERRED, the caveat is not.
2. **YOLOPX per-pixel probability is discarded at manifest-write** (writer stores
   `pred_mask > 0`; verified in `prediction_writer.py`). This single fact forces
   DEFER on τ_Y (§3), the τ_Y freeze (§12), τ_Y sensitivity (§14), and the YOLOPX
   arm of D8 (§11). CLRerNet per-curve `scores` **are** preserved → τ_C work is
   feasible now.
3. **The P/R/F1 triple must be reported once.** Proposal D1(recall), D2(1−precision),
   D3′(F1) are the same triple; never triple-count them in any composite.
4. **The tag-path defect blocks D5 entirely.** The runner reads
   `meta.tags.summary[...]` (nonexistent); the real path is
   `meta.predicted_tags.by_dimension["Observed Marking Visibility"]` (a list).
   `calibration_split.py` already uses the correct path.
5. **Tiny occluded strata** (dataset-wide 22 of 9298; after the calibration draw the
   eval set has 0 occluded for CULane and CurveLanes) make any occluded-stratum
   bootstrap CI degenerate/empty. Lead condition-sensitivity with clear-vs-degraded
   (thousands per stratum); keep occluded descriptive-only.
6. **No composite is defensible on this pilot.** Both the report R2 and the live R2
   blend a presence flag, an IoU, and two non-comparable confidences with self-
   admittedly unvalidated weights. The proposal's "no composite until 7 criteria
   met" is correct; on single-frame pilot data the honest outcome is "no D
   composite for this pilot," not an open TODO.

---

## 4. Recommended implementation order (evidence-supported subset only)

Nothing here is code yet — this is the change-set the evidence supports, smallest
and safest first. **Awaiting go-ahead.**

**Group A — documentation / relabeling (no published number moves):**
- §2 normalization: state `(x/D,y/D)` in the proposal (code already matches).
- §1/§3: drop "visible" → "reference-annotated centerline"; add CULane caveat.
- §4/§5: reframe as ADD-recall / adopt-1−precision + rename presence to "Output
  Availability"; keep count as native-instance diagnostic.
- §9: demote concurrence to a diagnostic; keep 1−recall as the D6 detectability quantity.
- §10/§11/§16: mark D7-temporal, τ_Y, D8 DEFER with explicit unblocks; add the
  MUTCD/FHWA-≠-image-tolerance + literature-anchor statements.

**Group B — small, well-scoped code (pure-numpy testable):**
- §6: drop "cap at diagonal"; exclude empty predictions + report empty rate (matches existing None-semantics).
- §13/§14: wire per-image bootstrap CI into `summarize_d_records` (pattern already in repo).
- §3/§5: thread CLRerNet `scores` into `build_d_record`; add τ_C filter param.
- §15: add the writable-now tests (dashed, crop/pad round-trip); cite the ≈35 existing.

**Group C — needs the GPU box / re-inference (blocked, sequenced):**
- Freeze δ (and, once wired, τ_C) on the calibration split; fix the split so it
  doesn't deplete occluded eval strata.
- Fix the D5 tag path, then compute clear-vs-degraded sensitivity.
- τ_Y / D8 remain DEFERRED until inference is re-run persisting YOLOPX per-pixel prob.

**Explicitly NOT doing** (per standing constraint): no composite D score, no
arbitrary weights, no dataset-specific exceptions, no physical-distance/ADS claim.
