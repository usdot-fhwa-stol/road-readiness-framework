# D-Metric Audit — Road Readiness Framework (Task 3)

**Scope:** Research-methodology + software-engineering audit of the detection ("D") metrics on branch `feature/D_metrics`, governed by `task3draft1_report.pdf`. Neither the current code nor the proposed redesign is assumed correct.

**Method:** Report extraction (pdftotext) + independent file reads + a 6-evidence / 3-adversarial-verify agent sweep. Every load-bearing claim below is cited to `file:line` or report page and was confirmed by ≥2 independent passes. Where two sources disagreed, the disagreement is reported as a finding, not smoothed over.

**Status:** AUDIT ONLY. No code has been modified. Section 8 (revisions) and Section 9 (tests) are *proposed* and await sign-off.

---

## 0. The governing constraint (from the report, verbatim intent)

Report p20–21 (Objectives): the pilot *"is not intended to replicate proprietary ADS/ADAS perception functions or directly assess how a specific ADS-equipped vehicle would process imagery,"* and *"uses reference image processing methods to evaluate what pavement marking information can be derived"* from public imagery.

**Therefore the detected object is: visible longitudinal lane-line *markings / marking evidence*, as recovered by frozen reference processors.** It is NOT a lane, lane center, corridor, vehicle path, or road boundary. D metrics measure *machine detectability of roadway-marking evidence*; they must not claim to measure ADS safety, lane-keeping, or field-certified readiness. This constraint is the yardstick for every verdict below.

---

## 1. Deliverable 1 — Task-3-goal → metric traceability

| Report anchor | What it asks D to measure | Current metric that claims to serve it | Does it actually? |
|---|---|---|---|
| p20–21 objective: derive *marking information* via reference processors | Presence of recoverable marking evidence | D1 Detection Success (presence) | **Yes** — presence-only, representation-agnostic. Cleanest fit. |
| §3.3 (p45–46) core metrics IoU/P/R/F1 on a "common representation" | Placement agreement of recovered marking vs reference | D3 Segmentation IoU (+P/R/F1) | **Partly** — measures placement, but on a *thickened* prediction (see §2), and the "common representation" is under-specified (three conflicting definitions). |
| Table 16 (p74–75) "Near-Field IoU" | Readability where markings matter most | D4 Near-Field IoU (lower 50% rows) | **Weakly** — it is D3 on the lower image half; explicitly *not* a metric-distance claim (no calibration exists). Rename it a "band," not "near-field." |
| Table 16 "Occlusion Robustness Gap" | Sensitivity of detectability to degraded conditions | D5 (clear − degraded mean D3) | **Conditionally** — depends entirely on VLM-predicted visibility tags; strictly-occluded tags are extremely sparse (CULane ~2, BDD ~4 images). |
| Table 16 "Detection Gap Ratio" | Fraction of reference marking missed | D6 ΣFN/ΣGT | **Redundant** — mathematically = 1 − pixel recall, already produced natively. |
| Table 16 "Confidence Mean" | Processor's own confidence in its output | D7 Confidence Mean | **Broken for cross-model use** — see §4. |
| Table 16 "Lane Count Accuracy" | Count agreement of marking instances | D2 exact count match | **Cross-model hazard** — YOLOPX has no instances; falls back to connected-component proxy. |
| (implied) temporal stability | Frame-to-frame detection stability | D7-temporal (reserved) | **Correctly None** — no temporal data exists (§5). |

**Key traceability finding:** the report's §3.3 core metrics (IoU/P/R/F1, cited to Jaccard 1912 + Powers 2011) are generic ML statistics. D3 and D6 are those same quantities re-numbered as D-indicators; they are **not** new road-readiness measurements, and **no** roadway standard (MUTCD/FHWA/ASAM/ASTM/NCHRP) prescribes them. The MUTCD/MSMT-729 grounding in project memory applies to the *tag taxonomy*, not to any D metric.

---

## 2. Deliverable 2 — Per-metric verdict (current suite)

Verdicts use the report objective (§0) as the test. "Redundant" = duplicates a quantity the harness already produces natively.

| Metric | Construct (as coded) | Math / impl problem | Task-3 fit | **Verdict** |
|---|---|---|---|---|
| **D1** `compute_d1_detection` | `int(pixel_count>0 or lane_count>0)` | None material. Docs claim recall/precision gating (metrics_reference.md:944) — **that code does not exist**; live D1 is pure presence. | Good — presence of marking evidence. | **KEEP** (fix stale doc). |
| **D2** `compute_d2_lane_count` | Exact `int(pred_count==gt_count)`; native polylines else connected components (MIN_AREA=30) | YOLOPX has no instances → counts CC of a dense mask (flagged approximate). CLRerNet native curves dropped to mask in cached path → also CC. Exact-match, no tolerance. Compares native count vs proxy count across models. | Weak & non-comparable across processors. | **REVISE** — report as marking-*fragment* count per representation; never compare a native count to a CC-proxy count as if equal. Or **DEFER** until CLRerNet curves are preserved. |
| **D3** `compute_d3_iou` | Pixel IoU TP/(TP+FP+FN) **on stroke-standardized (dilated) prediction** | Runs on a prediction dilated toward GT width (dilate-never-erode). Inflates overlap for thin outputs; the reported CLRerNet>YOLOPX ordering is driven by this stroke handling (report p~51–58). Three conflicting "common representation" definitions exist (see §7). | Placement agreement — the right idea, wrong basis. | **REVISE** — compute placement on an un-thickened representation (skeleton/centerline distance), not dilated pixel overlap. |
| **D4** `compute_d4_near_field_iou` | D3 on lower 50% rows (NEAR_FIELD_FRACTION=0.5) | Same dilation problem as D3. "Near-field" implies distance; no calibration exists → it is an *image band*, not a physical near field. | Legitimate as a band profile. | **REVISE + RENAME** — "lower-band readability"; inherits D3's basis fix. |
| **D5** `_gap` (group) | mean(D3\|clear) − mean(D3\|degraded), MIN_GROUP=3 | Depends on VLM-predicted visibility tags (not annotation truth). Strictly-occluded tags nearly absent. Inherits D3 basis. Report as a raw Δ, which is correct. | Conditionally useful as ΔD3 with CIs. | **REVISE** — report Δ with bootstrap CI + group counts; state tag-provenance caveat; no single "condition score." |
| **D6** `compute_d6_image_counts` + pooled ΣFN/ΣGT | Missed-pixel ratio; group = ΣFN/ΣGT | = 1 − pixel recall. Carries no information beyond recall the harness already computes. | Redundant. | **REPLACE/REMOVE** — either drop, or redefine as an *unsupported-marking* response (fraction of reference marking with no nearby prediction) on the un-thickened basis, distinct from raw recall. |
| **D7** `compute_d7_confidence` | mean(prob) over predicted pixels | YOLOPX prob = dense softmax pseudo-prob; CLRerNet native = per-curve score — **incompatible quantities**. In practice **always None** from manifests (prob never serialized/read; CLRerNet score dropped at ingest). Spec itself warns not to compare across models. | Not a validated cross-model metric. | **REMOVE** as a cross-model D metric — keep only as a per-processor diagnostic *if* the confidence path is actually wired end-to-end. |
| **D7-temporal** (reserved) | hardcoded None | No temporal/calibration data exists anywhere (§5). | N/A single-frame. | **KEEP as None**, exclude from aggregation. |
| **R2** (composite) | weighted mean D3_f1/D3_iou/D4/D6/D7 ×100 | Blends stroke-standardized IoU + F1 (correlated, both weighted), a redundant D6, and an always-None D7. Composite of unvalidated + redundant terms. | Composite not justified. | **HOLD** — do not ship a composite until component validity is shown (report/spec agree). |
| **R4** labels | HIGH/MODERATE/LOW on R1/R2 thresholds | The categorical readiness labels the charter explicitly prohibits. | Prohibited. | **REMOVE** the fixed labels. |

---

## 3. Deliverable 3 — Math specs for the metrics that survive

Notation: image $W\times H$; canonical normalized coord $\bar p=(x/D,\,y/D)$ with $D=\sqrt{W^2+H^2}$ (the image diagonal). Dividing **both** axes by the same $D$ makes distances resolution-free **and isotropic** — a tolerance $\delta$ is the same fraction of the diagonal on both axes, unlike $(x/W,y/H)$ which stretches distance differently per axis on non-square images. Reference-processor marking-support set $A$ = the recovered marking geometry in canonical coords (YOLOPX: threshold logits at the frozen operating point → skeletonize to 1-px centerlines → normalize; CLRerNet: restore native polylines to original coords → normalize; **do not** rasterize curves with a thick stroke, **do not** equate YOLOPX pixel-prob with CLRerNet curve-confidence). Reference GT marking set $G$ likewise as normalized centerline points. Point-to-set distance $d(\bar p,A)=\min_{\bar q\in A}\lVert\bar p-\bar q\rVert_2$. Tolerance $\delta$ **frozen on a dev/calibration split**, never the test set.

- **D1 — Visible Marking Support Presence** (per image): $D1=\mathbb 1[\,|A|>0\,]$. Range {0,1}; aggregate = success rate over all processed images. Undefined never.
- **D3′ — Marking Support Localization** (per image, replaces dilated IoU): with $\delta$ frozen,
  $\text{recall}= \frac{|\{\bar g\in G: d(\bar g,A)\le\delta\}|}{|G|}$, $\text{precision}=\frac{|\{\bar a\in A: d(\bar a,G)\le\delta\}|}{|A|}$, $F1=\frac{2PR}{P+R}$. Also report localization error $e=\{d(\bar g,A):\bar g\in G\}$ as **median + p95** (units: fraction of image diagonal). Undefined (→None) if $G=\varnothing$ (recall/F1/error) or if $A=\varnothing$ (precision/F1); when $A=\varnothing$ the GT→pred distances are unbounded, so error is reported as None rather than a fabricated magnitude. Note that when $G=\varnothing$ but $A\neq\varnothing$, precision is a defined **0** (every prediction is unsupported), not None.
- **D4′ — Lower-band readability** (rename): D3′ restricted to $y/H\ge 0.5$. Report as a 3-band profile (upper/middle/lower). No metric-distance claim unless calibration is present.
- **D5′ — Condition sensitivity**: $\Delta = \overline{D3'}\big|_{\text{clear}} - \overline{D3'}\big|_{\text{condition}}$ with bootstrap 95% CI; report $n$ per group; None if either group $<$ MIN_GROUP. State that condition labels are VLM-inferred.
- **D6′ — Unsupported-marking response** (only if kept distinct from recall): fraction of reference marking with **no** support within $\delta$ = $1-\text{recall}$ on the centerline basis — *equivalent to D3′ recall complement*, so recommend **folding into D3′** rather than shipping separately.
- **Aggregation (all):** per-image first, then bootstrap mean + 95% CI. `None` is excluded, never coerced to 0. No composite, no fixed labels, until validity is demonstrated on a held-out split.

*(Full D2/D7 specs deferred pending the representation/confidence-preservation fixes they depend on.)*

---

## 4. Deliverable 4 — Compatibility matrices

**Model × metric** (native capability, not what the code currently forces):

| Metric needs… | YOLOPX (dense seg logits) | CLRerNet (scored polylines) |
|---|---|---|
| Presence (D1) | ✅ nonzero mask | ✅ any curve |
| Centerline placement (D3′) | ✅ skeletonize mask | ✅ native polyline (no thick raster) |
| Instance count (D2) | ❌ no instances → CC proxy only | ✅ native curves **(currently dropped to mask)** |
| Per-line confidence (D7) | ❌ only dense pseudo-prob | ✅ native score **(currently dropped at ingest)** |
| Dense prob map | ✅ softmax/sigmoid **(discarded by manifest writer)** | ❌ none |

**Dataset × metric** (from installed manifests):

| | CULane 1640×590 | TuSimple 1280×720 | BDD100K 1280×720 | CurveLanes mixed |
|---|---|---|---|---|
| Native GT geometry | mask-only (w16, lane_json=None) | polyline | mask-only (lane_json=None) | polyline |
| D3′/D4′ | ✅ (GT centerline from mask) | ✅ | ✅ | ✅ (2 resolutions → normalize) |
| D2 instance count | CC proxy | ✅ | CC proxy | ✅ |
| D5 condition | ✅ scenario + VLM tags | VLM tags | VLM tags | VLM tags |
| Temporal | ❌ | ❌ | ❌ | ❌ |
| Calibration/near-field-in-meters | ❌ | ❌ | ❌ | ❌ |

**Cross-dataset caveat:** the fixed 16 px GT stroke is a different fraction of image height per dataset (16/590 vs 16/1440); no field normalizes this. Canonical normalization (§3) removes it for centerline metrics.

---

## 5. Deliverable 5 — Temporal-data audit

Exhaustive scan of all adapters, schema, converters, generators, and the four installed manifests (2000/2350/2648/2300 samples): **zero** `sequence_id`, `frame_index`, `timestamp`, neighbor-link, calibration, intrinsic, extrinsic, or pose fields. Schema `LaneSample` carries only `image_id, image_path, width, height, target, meta`.

- **CULane** sample_ids embed `.MP4_<frame>`, but installed frames within a clip are sparse/non-consecutive (deltas 300–1980, ≥10 s @30 fps). No neighbors on disk.
- **TuSimple** installs exactly one frame per clip (all ids end `_20`).
- **BDD100K / CurveLanes** use hashed ids, no temporal structure.

**Conclusion:** D7-temporal correctly stays `None` (`reason=unavailable_single_frame_evaluation`) and must be excluded from aggregation. Prediction-only frame consistency is not computable and, even if it were, would be a diagnostic — not a readiness D metric.

---

## 6. Deliverable 6 — Migration map (current → recommended)

| Current field / name | Recommended | Rationale |
|---|---|---|
| `D1_detection` | keep `D1_marking_support_presence` | presence, unchanged |
| `standardize_stroke_width` → D3/D4/D6 dilation | **remove from the metric path**; skeletonize instead | dilated overlap is the forbidden thick-stroke basis |
| `D3_iou` (dilated pixel IoU) | `D3_support_localization_f1` + `D3_loc_err_median/_p95` on centerline @ frozen δ | placement not raster width; resolution-free |
| `D4_near_iou` | `D4_band_readability` (upper/mid/lower) | no distance claim without calibration |
| `D5_*_gap` | `D5_condition_delta` + bootstrap CI + group n | honest uncertainty; VLM-tag caveat |
| `D6_detection_gap_ratio` (ΣFN/ΣGT) | fold into D3′ recall, or `D6_unsupported_marking_ratio` on centerline | remove redundancy with native recall |
| `D7_confidence_mean` | per-processor diagnostic only; drop from cross-model D | incompatible quantities; always None in manifests |
| CLRerNet ingest `for x,y in lane` | preserve per-curve score → `scores` in writer/reader | unblocks true D2/D7 for CLRerNet |
| `PredictionManifestWriter.add(...)` | add `scores`/`prob` params | end-to-end confidence + curve fidelity |
| R2 composite / R4 labels | gate behind validity demonstration; remove fixed labels | charter prohibits arbitrary composite/labels |
| `metrics_reference.md` D1-D8 tables | rewrite to match live code | docs describe removed code |

---

## 7. Deliverable 7 — Scientific-validity risks & unresolved decisions

1. **Thick-stroke basis biases model ranking.** D3/D4/D6 dilate the prediction; the report itself attributes CLRerNet's win to stroke width/tolerance handling. The headline ordering is not reproducible against a single, stated D3 definition.
2. **Three conflicting D3 definitions.** Live code = plain IoU on prediction-only dilation; `metrics_reference.md:617` = fixed 16 px + symmetric 8 px; `Algorithms/D3_*.md` + `metrics_reference.md:946` = distance-transform tolerant-F1 @ 5 px "primary." **Only the first is implemented.** Which produced the report's published tables is undeterminable from the repo.
3. **Docs describe removed code.** `metrics_reference.md`/`Algorithms/*` reference `compute_d*_single`, recall/precision-gated D1, dark-pixel D5 proxy — none exist in `.py`. An auditor trusting the docs would attribute nonexistent tolerances (5 px, 0.30 gates) to live metrics.
4. **CLRerNet is evaluated as a degraded proxy** of itself: 4 px raster → dilated → CC-counted → no confidence. Its native scored-polyline strengths are discarded before the metric code.
5. **CULane GT conflates visible paint with occlusion-extrapolated annotation** (all categories → 1, no visibility flag), directly against the visible-marking constraint.
6. **No calibration split.** Every threshold (δ, near-field fraction, stroke tolerance, argmax) is a hardcoded constant chosen off-data; there is no dev/test partition to freeze them on.
7. **Redundancy.** D3 ≈ YOLOPX-native IoU; D6 = 1 − recall. The D-suite does not capture CLRerNet's native instance-F1@50.
8. **D7 confidence is structurally unavailable** in the manifest path and semantically incomparable across models even when present.

**Unresolved decisions requiring the user:** (a) frozen operating points τ_Y (YOLOPX) and τ_C (CLRerNet) and δ — need a declared dev split; (b) whether to preserve CLRerNet curves+scores now (code change) or defer D2/D7; (c) whether D6 is dropped or redefined; (d) whether any composite (R2) or labels (R4) survive at all.

---

## 8. Deliverable 8 — Proposed scoped revisions (NOT YET APPLIED)

*Presented for approval per the process constraint. Each item traces to a finding above.*

- **d_metrics.py:** replace `standardize_stroke_width`-based D3/D4/D6 with a centerline (skeleton for YOLOPX, native polyline for CLRerNet) + frozen-δ localization F1 and median/p95 error (fixes risks 1–2, §2 verdicts). Fold D6 into D3′ recall.
- **CLRerNet path (`run_clrernet.py`, `prediction_writer.py`, `dataset.py`, `schema/lane.py`):** capture and serialize per-curve `scores`; stop dropping confidence and stop the cached mask-only ingest — unblocks true D2/D7 (fixes risk 4). *This is the largest change; may be deferred.*
- **Naming:** D4 → band-readability; D5 → condition-delta + CI; remove R4 fixed labels; gate R2 behind validity (fixes risks 7, charter prohibitions).
- **Docs:** rewrite `metrics_reference.md` D-tables + fill empty `evaluation_protocol.md`/`dataset_formats.md` to match live code and declare the single D3 definition + the dev/calibration split (fixes risks 2–3, 6).
- **Config:** expose δ, operating points, near-field fraction, MIN_GROUP as config keys sourced from a declared calibration partition (fixes risk 6).

## 9. Deliverable 9 — Validation tests to add (NOT YET APPLIED)

1. **Coordinate/translation invariance** — same marking at two pixel offsets within δ yields ~equal D3′.
2. **Resolution invariance** — GT+pred scaled 1×/2× yield identical canonical D3′ (currently untested; all existing masks share GT dims).
3. **Empty cases** — empty GT → D3′ None (not 0); empty pred vs non-empty GT → recall 0, precision None.
4. **Perfect prediction** — D1=1, D3′ F1=1, loc error=0.
5. **Threshold handling** — δ/operating points read from calibration split, not hardcoded; changing test data does not change δ.
6. **None-exclusion from aggregation** — D7/temporal None excluded, never coerced to 0; verify at the group summarizer (`summarize_d_records`, currently untested).
7. **Thick-stroke regression guard** — assert the metric path does not dilate predictions.
