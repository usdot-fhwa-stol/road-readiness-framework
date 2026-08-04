# Pavement-Marking Assessment Metrics Reference

This document defines the metric framework used in the pavement marking pilot analysis. It follows the specification in Chapter 4 of the project report (`draft.docx`, "Metrics for Pavement-Marking Assessment and Analytical Measures", tables 14–18). **The report is the specification; where the current code differs, the differences are recorded in [Implementation Status and Deviations](#implementation-status-and-deviations).**

The framework is an image-based proxy for pavement-marking machine-readability and reference-processor detectability. It is not a field-certified road-readiness standard. It uses public imagery, compatible reference lane-line annotations where available, dataset metadata and screening tags, and reference processors (YOLOPX as primary, CLRerNet as supplemental).

The framework defines **22 metrics across 4 groups**:

| Metric Group | Metric IDs | Primary Inputs | Intended Use |
|---|---|---|---|
| Image- and annotation-based pavement marking indicators | I1–I6 | Roadway images and compatible reference lane-line annotations | Characterize visible marking features: continuity, contrast, edge clarity, apparent lane-width variation, curvature complexity, reference annotation availability |
| Detection performance indicators | D1–D7 | Standardized processor outputs and compatible reference annotations; model confidence where available | Characterize processor detection quality: output availability, lane-boundary count agreement, spatial overlap, missed detections, condition-specific differences, confidence |
| Cross-factor and condition-based analysis measures | C1–C5 | Per-image I and D results plus scenario, roadway context, surface type, and visual-condition tags | Explore whether marking characteristics, image geometry, or conditions are associated with changes in processor output quality |
| Exploratory integrated assessment outputs | R1–R4 | Aggregated I, D, and C results plus evidence-sufficiency information | Synthesize findings, identify potential bottlenecks, support exploratory comparison across image sets or conditions |

Unless otherwise noted, these indicators apply primarily to **lane lines**, the pavement marking features represented in the compatible reference annotations and processor outputs used for the pilot.

The metric groups support infrastructure-focused interpretation of pavement marking detectability in public roadway imagery. **They do not represent validated measures of ADS deployment readiness, roadway safety, or the performance of any specific ADS- or ADAS-equipped vehicle.** The exploratory integrated assessment outputs summarize pilot findings and identify areas for further investigation; they are not intended to support deployment decisions or roadway certification.

A measure is applied only where the needed imagery, compatible reference annotations, processor outputs, metadata, and screening information are available in a compatible format. Not every measure applies to every image or dataset. **Missing or incompatible annotations are documented as unavailable, not interpreted as a zero-value result.**

The canonical implementation is per-image first. Each image produces one JSON-serializable metric record with dataset, split, model name, image identity, dimensions, sample and target metadata, mask occupancy, I metrics, D metrics, diagnostics, and per-image R1/R2 scores. Aggregation then computes overall summaries, strata summaries, correlations C1–C5, and R1–R4 outputs.

---

## Group 1 — Image- and Annotation-Based Pavement Marking Indicators

This group characterizes visible lane-line features and roadway-image characteristics that may affect the visibility and interpretation of pavement markings. Indicators are derived from roadway-facing images and, where available, compatible reference lane-line or pavement marking annotations. Reference annotations may be provided as polylines, lane boundaries, lane instances, or binary masks and are converted to a common representation where necessary.

These indicators describe image-based characteristics only. **They do not measure retroreflectivity, material type, installation age, compliance with design standards, or verified maintenance condition.**

### I1 — Lane Continuity Score

I1 is a pattern-aware, image-based estimate of whether the portions of a lane
marking that were intended to be painted remain visually present and trackable.
It separates:

- **intended pattern:** solid, dashed, dotted when supported, mixed when
  explicitly justified, or unknown;
- **observed condition:** intact, faded, fragmented, partially_missing, or
  unknown_condition.

A worn solid line remains intended_pattern = solid and receives a degraded
condition label. Fragmentation is not an intended-pattern class.

**Primary inputs:** the original RGB road image plus geometry that locates each
lane-marking corridor. Geometry may be a float-coordinate polyline, compatible
lane_json, or grouped centerlines fitted from a marking mask. Geometry is a
spatial guide only. A mask pixel or rasterized polyline is never treated as
proof that physical paint exists at that location.

#### Canonical and operational uses

Canonical benchmark I1 uses GT geometry in this priority order: native GT
polylines, GT lane_json, grouped GT-mask centerlines, then unavailable. It emits
the backward-compatible I1, I1_pattern_continuity, I1_percent, I1_pattern_type,
and I1_pattern_info fields and records I1_geometry_source = ground_truth.
Canonical I1 is independent of the evaluated model and remains the only I1
value used by R1.

Prediction-guided operational I1 uses native predicted float polylines,
prediction-manifest lane_json, grouped prediction-mask centerlines, then
unavailable. Its fields use the I1_pred_* prefix. It never receives GT style
metadata, GT lane points, or GT masks. It is not included in R1 by default.

This distinction matters for polyline detectors: a continuous CLRerNet
centerline can cross a physically dashed marking. The centerline locates the
sampling corridor but cannot establish whether RGB paint is solid or dashed.

#### Lane-aligned RGB evidence

For each sanitized, smoothed, and resampled centerline, I1 constructs a
lane-aligned strip. Local tangents define cross-lane normals; an adaptive center
band samples the likely marking and left/right side bands sample nearby
pavement. Bilinear interpolation is used for RGB and nearest-neighbor sampling
for optional road and object masks. RGB conversions use RGB-aware OpenCV
constants.

At longitudinal sample i, the engine computes:

- p_i in [0,1]: continuous visible-paint evidence from center-versus-side
  luminance, white/yellow color likelihood, paired boundary response, and
  consistency across the stripe width;
- q_i in [0,1]: observation reliability reduced for boundary clipping,
  unavailable side pavement, explicit object occlusion, glare/saturation,
  non-road background, and unstable geometry.

These features estimate local paint evidence only; I2, I3, and I4 are not added
as weighted terms inside I1. A broad shadow affecting paint and adjacent
pavement similarly can preserve relative evidence, while genuinely unreliable
samples receive lower q_i.

#### Intended-pattern inference

Canonical I1 may use only explicit target fields (style, pattern, lane_style,
line_style, marking_type, or line_type) as trusted pattern hints. Arbitrary
metadata text is not searched. Otherwise, the RGB signal is compared against a
solid hypothesis and bounded periodic dashed templates:

    m_s(i) = 1
    m_d(s; T, r, phi) = 1 when mod(s - phi, T) < rT, else 0

The periodic fit estimates period T, painted duty ratio r, and phase phi from
continuous p_i, weighted by q_i. Dashed classification requires repeated
alternating runs and gaps, normally at least three visible cycles, consistent
normalized run/gap lengths, a meaningful periodicity peak, a plausible duty
ratio, and a fit advantage over solid. Dotted is retained separately only for
sufficiently many repeated short-duty cycles. Weak, contradictory, occluded,
or short-range evidence returns unknown rather than forcing solid or dashed.

**Visible-paint presence gate.** A pattern is inferred only when actual bright
paint is visible along the corridor. The engine measures the strongest paint
evidence (a high percentile of p_i over reliable samples); when even this is at
the noise floor, no marking can be seen and the pattern is unknown regardless of
how periodic the residual signal looks. The dashed hypothesis additionally
requires the fitted painted phase to be genuinely brighter than the fitted gaps.
Without this gate, a frame with no visible marking — for example a pitch-dark
night scene — is wrongly explained as a dashed marking whose paint happens to
lie "in the gaps", producing a confident near-zero continuity instead of an
honest unknown. This gate is what prevents that fabrication.

Perspective priority is calibrated homography/IPM, geometry-derived scale
normalization, local-window normalization, then image arc length with reduced
confidence. No camera calibration is mandatory and no jurisdiction-specific
dash length is assumed.

#### Continuity, degradation, and condition

After intended pattern selection, only expected-paint locations are scored:

    D = 1 - sum_i(q_i m_i p_i) / (sum_i(q_i m_i) + epsilon)
    C = 1 - D

m_i is one where the selected pattern expects paint and zero in an intentional
gap; p_i is RGB paint evidence; q_i is reliability; and epsilon prevents
division by zero. Thus intentional dashed gaps do not increase degradation,
while a missing expected dash does. Continuity C and degradation D are both in
[0,1]; higher continuity is better and higher degradation means more expected
paint appears missing or severely faded. Pattern confidence is epistemic
confidence and is never multiplied into the physical-condition score.
Insufficient expected-paint support returns None. Condition is likewise scored
only where paint is actually visible: if the strongest evidence along the
expected-paint corridor is at the noise floor, continuity and degradation return
None with reason `no_visible_paint_evidence` — even when a trusted metadata hint
supplies the intended pattern. A marking that is essentially gone (or unseeable
in the dark) is reported as unavailable, not as a fabricated near-zero score. A
faded-but-still-visible marking retains a real, reduced score because the metric
uses relative center-versus-side evidence and is therefore robust to moderate
brightness loss.

Condition labels combine continuous degradation with the spatial organization
of low evidence. They are diagnostics, not field-certified pavement-condition
classes.

#### Multiple lanes and diagnostics

Each usable lane emits geometry and pattern sources, coordinate method, support,
valid fraction, pattern and confidence, condition, continuity/degradation,
run/gap statistics, fitted period/duty ratio, periodicity, competing-hypothesis
scores, and an explicit unavailable reason. Image aggregation uses confidence
x valid_fraction x sqrt(bounded_support) weights, so unavailable lanes are
excluded rather than averaged as zero and one extremely long lane cannot
dominate without bound. The aggregate also reports valid/unknown lane counts,
worst valid score, median valid score, and the aggregation method.

Canonical records add `I1_pattern_confidence`, `I1_condition`,
`I1_degradation_score`, `I1_pattern_source`, `I1_valid_lane_count`,
`I1_unknown_lane_count`, `I1_lane_diagnostics`, and `I1_unavailable_reason`.
Operational records use the corresponding `I1_pred_*` names, including
`I1_pred_geometry_source`. Each lane diagnostic contains `lane_index`,
`geometry_source`, `pattern_source`, `longitudinal_coordinate_method`,
`intended_pattern`, `pattern_confidence`, `condition`, `continuity_score`,
`degradation_score`, `valid_sample_fraction`, `usable_support`, evidence means,
run/gap counts and length statistics, cycle count, period, duty ratio,
periodicity and fit scores, score margin, notes, and `unavailable_reason`.

Normal JSON records store compact diagnostics only. Debug mode can additionally
render corridor boundaries, valid/invalid samples, p_i, q_i, occupancy, the
fitted template, expected-paint positions, and likely missing expected paint.

**Expected range:** 0–1 or unavailable (None).

**Interpretation and limitations:** I1 is an image-based proxy, not a
field-certified marking assessment. Shadows, glare, wet roads, construction or
temporary markings, repairs, occlusion, inaccurate geometry, poor calibration,
and insufficient view distance can lower confidence or make the result
unavailable.

### I2 — Boundary Contrast Ratio

**Primary inputs:** roadway image and compatible lane-line or pavement marking annotation.

**Calculation:** the absolute difference between the mean grayscale luminance of annotated marking pixels and adjacent pavement pixels, normalized by the image luminance range.

**Expected range:** 0–1.

**Interpretation and limitations:** a lower value indicates lower apparent contrast between the marking and surrounding pavement. The measure may be affected by lighting, shadow, glare, camera exposure, wet pavement, and image compression. **It is not a measurement of retroreflectivity or paint quality.**

### I3 — Boundary Sharpness Score

**Primary inputs:** roadway image and compatible binary lane-line or pavement marking mask.

**Calculation:** a boundary band is created by dilating the edge of the compatible reference mask using a documented buffer size. The score is the proportion of pixels within that band identified as image edges using a documented edge-detection method.

**Expected range:** 0–1.

**Interpretation and limitations:** a lower value indicates that the marking boundary is less visually distinct in the image. Results may be influenced by image resolution, motion blur, focus, compression, lighting, and pavement texture, in addition to marking condition.

### I4 — Lane Width Stability

I4 is a classical-computer-vision and geometry proxy for the stability of the
distance between adjacent lane-boundary centerlines. It does not use a learned
width model.

#### Marking width is not lane width

Marking width is the thickness of one painted stripe. Lane width is the
road-plane lateral distance between two logical adjacent boundary centerlines.
The historical component-area divided by major-axis calculation estimates
marking thickness in pixels and is retained only under the `I4_legacy_*`
names. It is never presented as lane width.

The implementation never measures the leftmost and rightmost marking pixels in
an entire image row, does not treat individual dash components as separate
boundaries, and derives mask centerlines rather than using an outer edge of a
thick segmentation stroke.

#### Canonical and prediction-guided modes

Canonical I4 uses GT geometry only: native GT boundary polylines, GT
`lane_json`, grouped GT-mask centerlines, then unavailable. Its source is
always `ground_truth`. Changing a model prediction while GT, image, and
shared camera calibration stay fixed cannot change canonical I4.

Operational `I4_pred` uses prediction geometry only: native floating-point
prediction polylines, prediction-manifest `lane_json`, grouped
prediction-mask centerlines, then unavailable. It receives no GT geometry,
pairing, lane count, masks, or target metadata. Native CLRerNet coordinates are
therefore analyzed without a mask round trip.

#### Boundary grouping and adjacent pairing

Polylines are filtered for finite, in-frame, unique points and retained only on
observed support. Shared I1 validation is followed by a robust Huber polynomial
diagnostic and a locally smoothed piecewise curve. This suppresses isolated
detector spikes without erasing supported curvature or width transitions.
Diagnostics record source and inlier counts, support, robust residual,
extrapolated fraction, and geometry confidence.

For mask-only outputs, a disposable morphology copy and fitted continuation
associate longitudinally ordered, laterally compatible dash fragments. Row
centroids form the marking centerline. Morphological bridging is geometry
grouping only; it is not evidence that paint or a boundary was physically
continuous.

Close, overlapping, parallel stripes are tested as a possible double-line
marking using scale-aware separation, candidate lane spacing, support overlap,
and tangent compatibility. A supported double line becomes one logical
centerline for pairing while source IDs remain in diagnostics. Its two stripes
cannot become an implausibly narrow travel lane.

Only logical adjacent boundaries are paired. Candidates require overlapping
support, consistent left/right order, no crossing, plausible separation,
compatible tangents, and adequate confidence. An optional drivable-area mask
may support the region between boundaries but is not mandatory. Every rejected
candidate retains an explicit reason.

**Travel-lane plausibility gating.** Two boundaries are scored as a lane only
when they plausibly bound a driveable travel lane. Three classical geometric
gates enforce this, so implausible geometry is reported as *unavailable* with an
explicit reason rather than scored as a confident (stable or unstable) width:

1. *Longitudinal orientation* (`boundary_orientation_not_longitudinal`). A
   forward-facing travel-lane boundary recedes toward the vanishing point and is
   longitudinal-dominant. Orientation is judged in the measurement frame — the
   road plane when calibrated (so a genuinely curved lane that merely looks
   diagonal in the raw image is not penalized), and image space otherwise. A
   boundary whose principal axis is more lateral than
   `max_boundary_axis_angle_from_vertical_deg` (default 60°) is lateral clutter
   (crosswalk / stop-line stripes, wall-texture trails, guardrail edges) and is
   rejected *before* the shared longitudinal axis is estimated, so a single
   horizontal detection cannot skew the frame. Configurable via
   `enforce_longitudinal_orientation` for non-forward-facing cameras.
2. *Corridor aspect ratio* (`implausible_corridor_aspect_ratio`). A travel lane
   is longer along the travel direction than it is wide. Pairs whose overlapping
   longitudinal support divided by median separation is below
   `min_pair_corridor_aspect_ratio` (default 1.1) are squat outline/box
   detections, not lanes.
3. *Boundary parallelism* (`boundaries_not_consistently_parallel`,
   `implausible_lane_width_variation`). The apparent width must not swing beyond
   a single lane plus a plausible transition
   (`max_pair_relative_separation_change`), and the raw separation must be
   explained by a smooth `{constant, linear, single-change-point}` model — the
   robust residual around that best smooth model, relative to median separation,
   must stay under `max_pair_separation_shape_residual` (default 0.18). Smooth
   intentional tapers, merges, and curved-but-parallel lanes pass; boundaries
   that diverge erratically or trace an outline are rejected.

These gates target two documented real-world failures of a naïve width proxy: a
prediction that traced a tunnel side wall and a pavement noise trail (previously
a near-perfect stable width) and one that traced a crosswalk outline (previously
a confident low width). Both are now correctly unavailable. An erratic or
noisy — but genuinely lane-like — width profile is still *scored* (low), not
rejected; only geometry that cannot plausibly bound a travel lane is rejected.

#### Perspective and measurement spaces

Measurement-space priority is:

1. calibrated image-to-road homography/IPM;
2. road-plane homography from supplied camera intrinsics and extrinsics;
3. another caller-supplied, independently validated road calibration;
4. conservative `projective_normalized` image-space fallback;
5. unavailable.

Transforms are checked for finite values, invertibility, condition number, and
valid transformed area. Local road-units-per-pixel resolution reduces
reliability; ill-resolved near-horizon samples are excluded. A transform is
never estimated by forcing the measured pair to have constant width.

`calibrated_metric_bev` reports meters only when supplied calibration has
known metric scale. `calibrated_scale_free_bev` reports generic road-plane
units. Without calibration, `projective_normalized` limits analysis to a
conservative mid/near image region, reports `image_width_fraction`, fits
expected smooth apparent-width trend as part of the profile, and carries lower
measurement confidence. Metric-scale confidence is zero in both fallback and
scale-free spaces. A monocular image by itself never produces meters.

#### Orthogonal sampling and reliability

At approximately uniform centerline arc-length station `s_i`, the local
tangent defines unit normal `n_i`. That normal line is explicitly intersected
with both fitted boundaries:

```
w_i = (r_i - l_i) dot n_i
```

`l_i` and `r_i` are left and right normal intersections. Failed, reversed,
ambiguous, crossing, or unsupported intersections are missing, not zero.
Reliability `q_i` combines boundary residuals, distance from observed
support, transformed resolution, horizon/boundary proximity, object-mask
occlusion, tangent compatibility, and intersection quality.

#### Expected profile and scoring

Robust weighted fits compete among constant, linear-taper, and continuous
one-change-point piecewise-linear profiles. Selection uses Huber loss, BIC,
minimum segment support, minimum score and practical residual improvement,
positive width, and plausible change. Output types are `constant`,
`linear_taper`, `merge_or_split_transition`, `piecewise_transition`,
`irregular_or_noisy`, and `unknown`. A smooth taper or supported
merge/split may be stable; nonconstant does not mean degraded.

For selected expected widths `m_i`:

```
e_i = abs(w_i - m_i) / (m_i + epsilon)
r_mad = 1.4826 * weighted_median(e_i, q_i)
r_90 = weighted_quantile(e_i, q_i, 0.90)

instability =
    lambda * clip(r_mad / tau_mad, 0, 1)
    + (1 - lambda) * clip(r_90 / tau_90, 0, 1)

width_profile_stability = 1 - clip(instability, 0, 1)
```

`w_i` is measured width, `m_i` is selected expected width, `q_i` is
sample reliability, `epsilon` prevents division by zero, `lambda` mixes
central and tail residuals, and `tau_mad` and `tau_90` are configurable
image-analysis tolerances. The tolerances must be calibrated on synthetic and
reviewed imagery; they are not roadway-design or maintenance standards.

`width_profile_stability` is residual consistency around the expected
profile. `width_constancy` separately measures how little that expected
profile changes. A clean constant lane should be high on both. A clean taper
may have high profile stability and lower constancy. Low constancy alone is not
an infrastructure penalty. Conventional width CV is diagnostic only because
it mixes intended change with residual noise.

#### Pair and image outputs

Each accepted pair reports boundary/source IDs, geometry and measurement
source, scale availability, profile type/confidence, geometry confidence,
stability, constancy, median/mean/min/max width and units, robust relative MAD,
relative 90th-percentile residual, conventional CV, slope, relative change,
change points, sample counts/coverage, support, transformed resolution, model
scores/margin, notes, and unavailable reason.

Image aggregation includes valid pairs only. Weights use geometry confidence,
profile confidence, and square-root support capped relative to median support.
A weighted median is blended with a weighted mean, preventing one long pair
from dominating without bound. Valid and unknown counts, median and worst valid
pair scores, rejection reasons, and aggregation method remain separate.
Unavailable pairs are never zero. Confidence is not multiplied into an
individual stability score.

Canonical fields use `I4_*`; operational fields use `I4_pred_*`. They
include profile stability/constancy/type/confidence, geometry
source/confidence, measurement space/confidence, metric-scale
availability/confidence, units,
valid/unknown pair counts, median/worst scores, pair and transform diagnostics,
aggregation method, and unavailable reason.

**Interpretation and limitations:** I4 is an image/geometry-based proxy. It is
not a roadway-design-compliance measurement and not a field-certified
lane-width survey. Physical width in meters is reported only with valid scale.

### I5 — Lane-Alignment Geometry Complexity

I5 is a classical computer-vision and differential-geometry measure of visible
**plan-view lane-alignment geometry complexity** — how geometrically demanding
the visible alignment is (tighter curves, more curvature change, more sign
reversals) — separate from a conservative **topology** diagnostic (supported
lane-count changes). It does not use a learned complexity classifier and does
not characterize road quality, pavement-marking condition, detector quality,
or ADS readiness. A higher value means the alignment is geometrically more
demanding, not that the road is defective, unsafe, non-compliant, or that an
ADS will fail. Curved roads, ramps, roundabouts, and reverse curves must not
automatically reduce R1 — see [R1](#r1--exploratory-infrastructure-indicator-score)
below.

There is no established universal scalar for "road geometric complexity." I5
is aligned with, but does not claim to *be*, PIARC-style bendiness (total
absolute heading change / section length), road-design Curvature Change Rate,
and ASAM OpenDRIVE's geometry primitives (line, constant-curvature arc,
clothoid, parametric cubic) and its separation of continuous alignment from
junction/linkage topology.

#### Canonical and prediction-guided modes

Canonical I5 uses GT geometry only (`I5_geometry_source = "ground_truth"`) and
is the only I5 value used by C4; it is independent of whichever model is being
evaluated. Operational `I5_pred` uses one evaluated model's geometry only —
never GT masks, GT lane points, GT lane count, or GT metadata — and must not be
used as the primary predictor of that same model's own D metrics (that would
confound detector failure with estimated road complexity). Both share the same
geometry-source priority as I1/I4: native float polylines, then compatible
`lane_json`, then mask-grouped centerlines, then unavailable.

#### Geometry and centerlines

I5 reuses the I1 geometry-validation/mask-grouping utilities and the I4
boundary-fitting, double-line-collapsing, and adjacent-pairing engine
(`evaluation.lane_width_stability.construct_lane_centerlines`) rather than
reimplementing them. When two adjacent boundaries pair plausibly, I5 analyzes
the **paired centerline**; otherwise it analyzes the **boundary's own curve**
and labels the representation accordingly. This is why parallel offsets of one
road curve and thick-vs-thin YOLOPX masks converge on comparable curvature,
and why dashed components are grouped into one logical lane rather than each
being scored separately.

#### Perspective and measurement space

Priority: `calibrated_metric_bev` (explicit homography with an
affirmatively-declared metric scale — curvature in 1/m) →
`calibrated_scale_free_bev` (valid homography, no declared metric scale) →
`lane_width_normalized` (no calibration, but a same-source lane-width estimate
is available to scale by) → `normalized_image_proxy` (coordinates divided by
the image diagonal; no metric units; **not comparable across camera
configurations**). Curvature is never estimated directly from raw, unscaled
x(y) image coordinates, and calibration is never estimated by forcing the
analyzed lane to be straight or constant curvature.

#### Robust curve fitting and curvature

Each centerline is fit as `c(s) = [x(s), y(s)]` over an approximately uniform
arc-length parameter using a local (bounded-window) despiking and smoothing
pass, then an interpolating spline purely for analytic tangents — never a
single unconstrained global quadratic and never a high-degree global
polynomial. Signed curvature is evaluated via a local three-point (Menger)
formula on the densely resampled curve rather than a global second derivative:
a global smoothing spline can ring across the *entire* domain when the curve
contains one real sharp transition (a compound curve), which a bounded local
formula cannot.

```
kappa(s) = (x'(s)*y''(s) - y'(s)*x''(s)) / ((x'(s)^2 + y'(s)^2)^(3/2) + eps)
```

Reported components: **bendiness** (weighted mean |kappa|), **p95 curvature**,
**curvature variation** (weighted mean |dkappa/ds|, estimated with the same
wide local baseline as curvature itself so it is not dominated by point-level
noise), **p95 curvature gradient**, **total absolute heading change**
(diagnostic only — related to bendiness by support length, so not
independently weighted), **reversal count/density** (Schmitt-trigger deadband
with a minimum arc-length separation, so noise around zero cannot create false
reverse curves), and **tortuosity** (`arc length / chord length - 1`). Sample
reliability weights combine local-window validity, edge proximity, horizon
proximity, and object-mask occlusion; excluded samples are never converted to
zero curvature.

#### Alignment profile classification

Weighted-least-squares hypotheses (straight, constant curvature,
linear/transition curvature, and an optional single-change-point piecewise
model) are compared by BIC. A more flexible model may always reduce raw
residual somewhat even for a perfectly constant curve, so a documented
magnitude gate (relative to the reference curvature scale) is required before
accepting a non-constant model, in addition to the BIC margin. `profile_class`
is one of `straight`, `constant_curve`, `transition_curve`, `compound_curve`
(piecewise, predominantly one sign), `reverse_curve` (a supported sign
reversal), `irregular_or_noisy`, or `unknown`. This describes visible geometry
only; it is never presented as the road's official design element.

#### Alignment complexity scalar and topology

Bendiness, p95 curvature, curvature variation, and reversal density are each
mapped through a versioned saturating curve `n_j = 1 - exp(-f_j/tau_j)` (tau
values documented per measurement space in `GeometryComplexityConfig`; **not**
engineering standards) and combined by an equal-weighted RMS into
`I5_alignment_complexity` (aliased as `I5`/`I5_geometry_complexity` for
compatibility). Dataset min/max normalization is deliberately not used, so
historical scores do not shift when images are added to or removed from a
dataset.

Topology (`I5_topology_complexity`) is a **separate**, conservative 0–1
diagnostic of supported lane-count changes between the near and far bands of
the visible extent, reported only with sufficient boundary evidence and never
folded into alignment complexity. It is not the same thing as I6: I6 is the
amount of identifiable marking-*instance* information; I5's topology
diagnostic is about supported connectivity changes (splits/merges) among
logical lanes. Single-image topology inference is inherently uncertain and
frequently returns unavailable.

#### Multi-lane aggregation

One diagnostic is reported per usable lane/centerline (profile class,
confidence, complexity, curvature components, fit quality, unavailable
reason). Aggregation uses only valid lanes with confidence- and
bounded-support-aware weights, so unavailable lanes are never averaged in as
zero and one very long lane cannot dominate without bound. Extra parallel
lanes at the same alignment therefore do not automatically increase the
aggregate.

#### Legacy quadratic-proxy migration

The previous algorithm (a single unconstrained quadratic fit `x(y)=ay^2+by+c`
in raw image pixels) is preserved unchanged under `I5_legacy_quadratic_proxy`
(and the unchanged function `compute_i5_legacy_quadratic_proxy`/
`compute_i5_geometry_complexity`) for historical comparison. It is **not**
directly comparable to the new `I5`/`I5_alignment_complexity` — do not treat
them as the same quantity in downstream trend charts.

**Expected range:** `I5`/`I5_alignment_complexity` and `I5_topology_complexity`
are 0–1, or `None` with an explicit `I5_unavailable_reason` when geometry
cannot be estimated reliably (never a fabricated low-complexity score).
`I5_complexity_confidence` is a separate 0–1 epistemic confidence, never
multiplied into the score.

**Interpretation and limitations:** requires an affirmatively-declared metric
scale for physical curvature units; otherwise results are an image-space or
lane-width-relative proxy, not comparable across camera configurations. Road
nonplanarity (grade, crest/sag vertical curves, superelevation), short visible
distance, the image horizon, occlusion, work zones, and detector errors
(directly affecting `I5_pred`) all limit usable support and confidence. This is
not a roadway design-compliance metric, not a safety score, and does not
establish causal ADS performance effects. See
[`Algorithms/I5_geometry_complexity.md`](../Algorithms/I5_geometry_complexity.md)
for the full algorithm.

### I6 — Lane Count (GT)

**Primary inputs:** compatible reference lane-boundary annotations and image-level metadata.

**Calculation:** counts the distinct annotated lane-boundary instances represented in the ground-truth annotation for each image. For this metric, "lane count" refers to **annotated lane-boundary instances, not the number of travel lanes**. Connected-component counts from a merged binary mask are not used as a substitute.

**Expected range:** integer ≥ 0.

**Interpretation and limitations:** provides descriptive context on the amount of lane-boundary reference information available for image-level evaluation. Calculated only when distinct lane-boundary instances can be identified reliably from the source annotation. **When annotations are missing, incompatible, or represented only as a merged mask without separable instances, the value is reported as unavailable rather than zero.**

> **Parameter documentation.** Implementation parameters for these indicators — edge-detection settings, boundary-buffer size, minimum segment length, and image-row sampling rules — must be documented with the pilot analysis results. Sensitivity checks should be used where appropriate to determine whether the selected parameters materially affect the observed patterns.

---

## Group 2 — Detection Performance Indicators

This group evaluates the quality and usability of the standardized processor outputs. **D1 is calculated across all applicable processed images; D2 through D6 are calculated only where compatible reference annotations and processor outputs are available.** Images without compatible annotations may be retained for visual review and qualitative reporting but are excluded from annotation-based quantitative measures.

The core measures IoU, F1 Score, Precision, and Recall are defined in the chapter 3 model evaluation approach. **Segmentation IoU is the primary spatial-agreement measure for D3 and the related cross-factor analyses.** F1 Score, Precision, and Recall may be reported alongside IoU for additional context on overall detection quality, false detections, and missed detections.

### D1 — Detection Success Rate

**Primary inputs:** standardized processor outputs and image-level records.

**Calculation:** the percentage of applicable processed images for which the standardized processor output contains at least one predicted lane-related pixel or lane-boundary instance.

**Expected range:** 0–1.

**Interpretation and limitations:** a lower value indicates that the processor generated no lane-related output for a greater share of processed images. **Detection success does not indicate whether the output is accurate, complete, or correctly aligned with a reference annotation.**

*YOLOPX support:* uses the binary lane mask extracted from `ll_seg_out`.

### D2 — Lane Count Accuracy

**Primary inputs:** predicted lane-boundary instances and compatible ground-truth lane-boundary instances.

**Calculation:** the percentage of eligible images for which the number of predicted lane-boundary instances **equals** the number of distinct ground-truth lane-boundary instances.

**Expected range:** 0–1.

**Interpretation and limitations:** a lower value may indicate missed boundaries, merged boundaries, or false boundaries. Lane count refers to identifiable lane-boundary instances, not travel lanes. **Connected-component counts from merged masks should not be used as a substitute when lane-boundary instances cannot be distinguished reliably.**

*YOLOPX support:* predicted lane mask components grouped by the documented instance heuristic.

### D3 — Segmentation Intersection over Union (IoU)

**Primary inputs:** standardized processor outputs and compatible reference annotations.

**Calculation:** for each eligible image, `IoU = TP / (TP + FP + FN)` using the common evaluation representation described in chapter 3. Image-level IoU values are then summarized across the applicable image group.

**Expected range:** 0–1.

**Interpretation and limitations:** higher values indicate stronger spatial overlap between predicted and reference lane-related outputs. Results depend on the compatibility and quality of the reference annotations and the selected evaluation representation.

*Evaluation representation:* lane outputs are rendered to a binary lane mask at a fixed stroke width of 16 px, with a stroke-width tolerance of 8 px applied symmetrically to both predicted and reference masks before pixel-level TP/FP/FN are counted. The same width and tolerance are applied identically to all processors, so thin-line and dense-mask outputs are compared on geometric agreement rather than raster width.

### D4 — Near-Field IoU

**Primary inputs:** standardized processor outputs and compatible reference annotations.

**Calculation:** applies the D3 IoU calculation only within a documented lower image region, such as the lower half of the image.

**Expected range:** 0–1.

**Interpretation and limitations:** provides a focused view of agreement in the roadway area closest to the camera within the image. **Without camera calibration, the lower image region should not be interpreted as a fixed physical distance from the vehicle.**

*YOLOPX support:* uses the predicted lane mask after undoing letterbox padding.

### D5 — Occlusion Robustness Gap

**Primary inputs:** D3 image-level IoU results and observed marking visibility tags.

**Calculation:** the difference between mean IoU for images with nonoccluded markings and mean IoU for images tagged as occluded:

```
D5 = mean(IoU_nonoccluded) - mean(IoU_occluded)
```

**Expected range:** −1 to +1.

**Interpretation and limitations:** a positive value indicates lower spatial agreement for images with occluded markings. Reported only where both groups contain a sufficient number of eligible images. **Occlusion is identified using documented image-level screening tags or review criteria; dark-pixel thresholds alone should not be used to define occlusion.**

See the [deviations section](#implementation-status-and-deviations) — the current code implements a dark-region intensity proxy, which this specification explicitly rules out. Both are documented pending occlusion screening tags in the manifest.

### D6 — Detection Gap Ratio

**Primary inputs:** standardized processor outputs and compatible reference masks or equivalent representations.

**Calculation:** the proportion of available reference marking pixels missed by the processor — the sum of false-negative pixels divided by the sum of ground-truth marking pixels **across the applicable image group** (pooled, not per-image averaged):

```
D6 = sum(FN) / sum(GT)
```

**Expected range:** 0–1.

**Interpretation and limitations:** a higher value indicates that a larger share of available reference marking information was missed. Applicable only when compatible mask-based or equivalent reference representations are available.

### D7 — Confidence Mean

**Primary inputs:** model-generated confidence information, where available.

**Calculation:** the mean confidence associated with the processor's lane-related outputs across the applicable images or image group.

**Expected range:** 0–1.

**Interpretation and limitations:** lower confidence may indicate greater model uncertainty. **Confidence definitions, scales, and calibration differ across processors; confidence values should not be compared directly across models unless comparability has been established.** Where meaningful lane-output confidence is not available, the metric is reported as unavailable. No placeholder confidence value is substituted.

*YOLOPX support:* for two-channel `ll_seg_out`, softmax over channels is applied and channel 1 is the lane probability; for one-channel output, sigmoid is used. If the channel layout is ambiguous, probability is unavailable and D7 is reported as unavailable.

> **Complementarity.** These indicators provide complementary information. A processor may produce lane-related outputs for most images (high D1) but still show low spatial agreement (D3) or a high detection gap ratio (D6). Similarly, a processor may show strong overall IoU but a substantial occlusion robustness gap, indicating lower agreement for images with occluded markings.

---

## Group 3 — Cross-Factor and Condition-Based Analysis Measures

This group examines potential associations among visible pavement marking characteristics, roadway-image geometry, image conditions, and processor output quality, combining the I-series indicators with the D-series indicators.

**The arrows in the measure names identify the relationship examined. They do not indicate that a marking characteristic, roadway condition, or image condition *causes* a change in processor performance.**

These analyses are exploratory. Observed patterns may reflect differences in dataset composition, camera configuration, image quality, annotation practices, processor training data, or other factors not represented in the pilot image set. Where sufficient eligible images are available, analyses are conducted separately by processor and, where appropriate, by source dataset or other relevant image group. **Each reported comparison identifies the sample size, grouping approach, selected performance measure, and any exclusions applied.**

### C1 — Continuity → Detection

**Primary inputs:** image-level I1 results and image-level D1 outcomes.

**Calculation:** the **Pearson point-biserial** correlation between the Lane Continuity Score and the binary Detection Success Rate across eligible images. For this analysis, D1 equals 1 when the processor produces a nonempty lane-related output and 0 otherwise.

**Expected range:** −1 to +1.

**Interpretation and limitations:** a positive value indicates that higher lane continuity scores are associated with greater detection success. **The result does not establish that improving lane continuity would directly improve processor performance.**

### C2 — Contrast → IoU

**Primary inputs:** image-level I2 and D3 results.

**Calculation:** the **Pearson** correlation between the Boundary Contrast Ratio and Segmentation IoU across eligible images.

**Expected range:** −1 to +1.

**Interpretation and limitations:** a positive value indicates that higher apparent marking-to-pavement contrast is associated with stronger spatial agreement. The relationship may also reflect lighting, camera exposure, shadow, wet pavement, image quality, and dataset-specific characteristics.

### C3 — Sharpness → IoU

**Primary inputs:** image-level I3 and D3 results.

**Calculation:** the **Pearson** correlation between the Boundary Sharpness Score and Segmentation IoU across eligible images.

**Expected range:** −1 to +1.

**Interpretation and limitations:** a positive value indicates that greater boundary sharpness is associated with stronger spatial agreement. **Boundary sharpness is an image-based measure and should not be interpreted as a direct measure of pavement marking material condition.**

### C4 — Geometry Complexity → Performance Gap

**Primary inputs:** image-level canonical I5 (`I5_alignment_complexity`,
GT-guided only — `I5_pred` never enters C4) and D3 results.

**Calculation** (`compute_c4_geometry_sensitivity_detail` in
`evaluation/readiness_metrics.py`):

1. Spearman correlation between `I5_alignment_complexity` and
   `D3_tolerant_f1_r5` (primary), `D3_iou`, and the detected ratio
   (`1 - D6_missed_marking_ratio`).
2. A bottom-vs-top tertile performance difference with a percentile bootstrap
   95% confidence interval.
3. The legacy median-split effect, retained for backward compatibility:

   ```
   C4 = mean(D3_tolerant_f1_r5 | I5 <= median) - mean(D3_tolerant_f1_r5 | I5 > median)
   ```

4. Per-component associations (bendiness, p95 curvature, curvature variation,
   reversal density, topology complexity) with `D3_tolerant_f1_r5`.
5. Per-model, per-dataset, and per-measurement-space stratified associations.
6. An optional dataset/model fixed-effects-adjusted regression of
   `D3_tolerant_f1_r5` on `I5_alignment_complexity`, `I1`, `I2`, and `I3`,
   restricted to samples sharing one measurement space (calibrated and
   image-proxy I5 values are never combined in one regression without
   stratification), gated on a minimum sample count.

Sample counts and skip reasons (`insufficient_samples_n_lt_N`,
`zero_variance`, `insufficient_same_measurement_space_samples`, …) are
reported at every level.

**Expected range:** the primary Spearman correlation and legacy median-split
`C4` scalar are −1 to +1 (`C4` stays backward compatible with the historical
median-split scalar).

**Interpretation and limitations:** a positive median-split effect or negative
Spearman correlation indicates the higher-complexity group had lower mean
spatial agreement. All of this is **association, not causation** — it does
not establish that geometric complexity caused the observed difference, and
may reflect dataset composition, camera configuration, or processor training
domain. Uncalibrated (`normalized_image_proxy`) I5 values are only comparable
within a homogeneous dataset/camera group, which is why measurement-space
stratification and the regression's single-measurement-space restriction
exist.

### C5 — Condition Degradation Ranking

**Primary inputs:** I1, I2, I3, D1, D3, D4, and applicable screening group tags.

**Calculation:** for each selected condition group, the mean value of each component metric is calculated and compared with a documented reference condition (such as clear, daylight, dry, or nonoccluded). Before combining, **all component differences are converted to a common 0–1 scale and oriented so that higher values indicate more favorable results.** The mean difference across the selected components is the condition degradation score. Condition groups are ranked from largest to smallest score.

**Expected output:** ranked list of condition groups with a standardized degradation score from −1 to +1.

**Interpretation and limitations:** a larger positive score indicates less favorable observed results relative to the reference condition. The ranking is exploratory and depends on the selected reference condition, component measures, normalization method, and sample size. **It is not an infrastructure investment priority ranking or a readiness score.** Results are not reported for groups with insufficient eligible images.

---

## Group 4 — Exploratory Integrated Assessment Outputs

This group integrates selected results from the preceding analyses to support structured reporting and interpretation of the pilot findings, at an image-group, dataset, or condition-group level as appropriate.

**R1 and R2 are exploratory composite scores.** They are not validated measures of roadway readiness for ADS deployment, do not represent the performance of any specific ADS- or ADAS-equipped vehicle, and should not be used for roadway certification, maintenance prioritization, or deployment decisions. **The component weights are analytical choices for this pilot and do not represent validated safety or risk weights.**

Where an integrated score or summary is prepared, the applicable component measures, normalization approach, weighting approach, sample size, exclusions, and data limitations are documented with the results. Individual indicators remain available for review and are reported alongside any integrated result.

### R1 — Exploratory Infrastructure Indicator Score

**Primary inputs:** I1–I5, with I6 reported separately as descriptive context.

**Calculation:**

```
R1 = 100/0.95 × [ 0.30(I1/100) + 0.30(I2) + 0.20(I3) + 0.10 N(I4) + 0.05 N(I5) ]
```

N(I4) and N(I5) are normalized 0–1 terms, **oriented so that higher values represent more favorable apparent lane-width stability and lower apparent curvature complexity.** The normalization bounds are established using documented reference bounds set before final analysis, selected based on the pilot data structure, and tested through sensitivity analysis to determine whether alternative bounds materially affect the resulting score. Bounds and sensitivity checks are documented with the pilot results.

**Current compatibility treatment:** metric version
`readiness_metrics_v5_geometry_complexity` does not silently substitute the
new geometry indicator into the historical R1 implementation. R1 continues to
use `I4_legacy_thickness_stability` at its prior 10% weight.
`I4_lane_width_stability` is reported separately, and every `I4_pred_*`
field remains outside R1. Historical and canonical lane-width I4 values must
not be treated as the same quantity. Intentional taper, merge, or split
geometry is not automatically converted to an infrastructure-readability
penalty.

**I5 is not included in canonical R1.** `compute_r1_marking_readability_score`
uses only `I1`, `I2`, `I3`, and `I4_legacy_thickness_stability` — neither the
new alignment-complexity scalar nor any prior N(I5) term is summed in.
Geometric complexity (curvature, tortuosity, reversal density, topology) is
context, not infrastructure degradation: a curved road, ramp, roundabout, or
reverse curve must not automatically lower R1. `I5_pred` never affects
canonical R1 either. This is a deliberate historical-behavior preservation,
not an oversight — see [C4](#c4--geometry-complexity--performance-gap) for
where geometric complexity is actually analyzed (as an association with
detection performance, stratified by measurement space, never as a quality
penalty).

I6 is reported separately because it describes ground-truth lane-boundary availability rather than an image-based pavement marking characteristic.

**Expected range:** 0–100, where higher values indicate more favorable results under the documented exploratory scoring approach.

**Interpretation and limitations:** provides a concise summary of selected visible pavement marking and roadway-image characteristics. **It does not measure physical marking condition, retroreflectivity, material quality, compliance with design standards, or verified maintenance need.** Calculated only when the needed component measures are available.

### R2 — Exploratory Detection Performance Score

**Primary inputs:** D1–D6, and D7 where meaningful model-confidence information is available.

**Calculation:**

```
R2 = 100 × [ 0.25(D1/100) + 0.20(D2/100) + 0.20(D3) + 0.15(D4)
           + 0.08 R(D5) + 0.08(1 − D6) + 0.04(D7) ]

where R(D5) = 1 − |D5|
```

R(D5) is an occlusion-robustness term; values closer to 1 indicate a smaller observed difference between occluded and nonoccluded image groups. **D7 is included only where meaningful confidence information is available; otherwise the remaining component weights are re-normalized and the treatment is documented.**

**Expected range:** 0–100, where higher values indicate more favorable results under the documented exploratory scoring approach.

**Interpretation and limitations:** provides a concise summary of processor output quality for the selected pilot configuration. **The score does not represent the performance of a complete ADS or ADAS perception system.** Results should be interpreted with the underlying D-series measures, particularly when component measures show conflicting patterns.

### R3 — Bottleneck Identification

**Primary inputs:** C1–C5, supported by relevant I- and D-series results.

**Calculation:** identifies the pavement marking characteristic, roadway-image factor, or condition associated with the largest **adequately supported** adverse association, performance difference, or condition-related degradation result. The determination considers effect magnitude, direction, sample size, consistency across applicable groups, and data-quality limitations.

**Expected output:** identified factor or condition, supporting measure or measures, direction of the observed pattern, and evidence-sufficiency information.

**Interpretation and limitations:** identifies **a potential area for further investigation rather than a confirmed cause** of processor behavior. A reported bottleneck may reflect confounding factors, dataset composition, limited sample size, or other characteristics not isolated in the pilot analysis.

### R4 — Exploratory Pilot Findings Profile

**Primary inputs:** R1, R2, R3, and evidence-sufficiency information.

**Calculation:** organizes the integrated pilot findings into a descriptive profile using documented criteria. The profile may distinguish among:

- more favorable observed results
- mixed or variable results
- less favorable observed results
- findings with limited or insufficient evidence

**R1 and R2 support the profile but do not determine it alone.** R3 is a diagnostic output that helps identify areas for further examination; it does not directly determine the R4 profile.

**Expected output:** descriptive findings profile.

**Interpretation and limitations:** supports communication of pilot findings, uncertainty, and evidence limitations. **It is not a roadway readiness classification and must not use labels such as "ready," "marginal," or "not ready."**

---

## Interdependency Diagram

```
Per-image inputs
  image + compatible reference annotation           standardized processor output
  (+ metadata, screening tags, road mask)           (+ model confidence, where available)
       |                                                       |
       v                                                       v
I1 lane continuity score                          D1 detection success rate
I2 boundary contrast ratio                        D2 lane count accuracy
I3 boundary sharpness score                       D3 segmentation IoU
I4 lane width stability                           D4 near-field IoU
I5 alignment complexity + topology (context)      D5 occlusion robustness gap
I6 lane count (GT) - descriptive context          D6 detection gap ratio
                                                  D7 confidence mean (where available)
       |                                                       |
       +--------------------- per-image record ----------------+
                                  |
                                  v
              C1-C5 cross-factor / condition analyses
                                  |
        +-------------------------+-------------------------+
        v                                                   v
R1 infrastructure indicator score              R3 bottleneck identification
R2 detection performance score                          (from C1-C5)
        |                                                   |
        +--------------------+------------------------------+
                             v
              R4 exploratory pilot findings profile
                  (+ evidence-sufficiency information)
```

I1–I4 support R1; I5 is context used only by C4 (geometry-sensitivity analysis), never summed into R1; I6 provides descriptive context on available ground-truth lane-boundary information. The D-series supports R2. The C-series supports R3. R1, R2, R3, and evidence-sufficiency information together support the descriptive profile R4.

---

## Dataset Metadata Coverage

| Field / Use | BDD100K | CULane | CurveLanes | TuSimple |
|-------------|:-------:|:------:|:----------:|:--------:|
| Per-image GT mask | yes | yes | yes, rasterized from polylines | yes, rasterized from polylines |
| GT polylines | usually no in YOLOPX-prepared masks | yes when `.lines.txt` is used | yes | yes |
| Weather | yes with detection annotations | no; scenario stored as `condition` | no | no |
| Time of day | yes with detection annotations | night scenario can map to `timeofday=night` | no | no |
| Scene/context | yes with detection annotations | scenario condition lists | no | no |
| Lane style metadata (for I1 dashed handling) | dataset-dependent, often absent | absent | absent | absent |
| Occlusion / visibility tags (for D5) | partial, via detection annotations | scenario tags only | no | no |
| Lane confidence for D7 | model-output dependent | model-output dependent | model-output dependent | model-output dependent |

---

## Implementation Status and Deviations

**Where this specification and the current code disagree, the specification above is authoritative and the code below is the current state.** These are open items, not alternative definitions.

The canonical implementation is `evaluation/readiness_metrics.py` (`METRIC_VERSION = "readiness_metrics_v7_i4_lane_width_plausibility"`). I1 follows the RGB-based pattern/condition definition above, I4 uses the lane-width engine in `evaluation/lane_width_stability.py`, and I5 uses the classical differential-geometry engine in `evaluation/lane_geometry_complexity.py`; the remaining historical deviations are listed below.

### Naming and numbering

The code emits **two keys per metric** — a short alias (`I1`) and a long name (`I1_pattern_continuity`). Long names encode the older algorithm, not the specification's indicator names. Any rename must move both keys together.

The code also retains a **legacy wrapper layer** (`readiness_metrics.py` lines 936–1048) whose function names already track this specification closely: `compute_i4_width_stability`, `compute_i6_lane_count`, `compute_d1_detection_rate`, `compute_d2_lane_count_accuracy`, `compute_d5_occlusion_gap`, `compute_d6_detection_gap`, `compute_r1_infrastructure_score`. These wrappers currently alias the newer, divergent implementations rather than implementing the specification.

**ID shift:** the code numbers Confidence Mean as **D8** (`compute_d8_confidence_mean_single`, keys `D8`, `D8_confidence_mean`, `D8_available`) and reserves **D7** for a temporal-jitter metric that has no compute function — it is hardcoded to `None` with reason `unavailable_single_frame_evaluation`. **This specification has no temporal metric; Confidence Mean is D7.** The code's total is therefore 23 IDs against the specification's 22.

### Algorithmic deviations

| ID | Specification | Current code (`evaluation/readiness_metrics.py`) |
|---|---|---|
| I1 | Pattern-aware expected-paint continuity defined above | `evaluation/lane_continuity.py` samples RGB along sanitized guidance geometry, fits competing solid/dashed/dotted hypotheses, excludes intentional gaps, and reports condition separately. Canonical GT-guided I1 remains in R1; operational prediction-guided I1 is diagnostic only |
| I2 | Normalized by image luminance range | `compute_i2_local_contrast` (:343) — marking mean vs a local background ring, divided by 255 |
| I3 | Proportion of dilated-band pixels detected as edges | `compute_i3_boundary_sharpness` (:363) — Sobel median ratio, boundary band vs background ring |
| I4 | Perspective-normalized adjacent-boundary width-profile stability | `analyze_lane_width_stability` performs source-separated geometry selection, double-line collapse, adjacent pairing, orthogonal samples, and robust constant/taper/piecewise selection. The prior paint-thickness proxy remains only in `I4_legacy_*` and historical R1 |
| I5 | Std of row-wise lane-center x ÷ image width | `analyze_lane_geometry_complexity` (`evaluation/lane_geometry_complexity.py`) performs source-separated geometry selection (reusing the I4 boundary/pairing engine), calibrated/scale-free/proxy measurement-space selection, robust local despiking + Savitzky-Golay smoothing, local (Menger) signed-curvature estimation, BIC-penalized alignment-profile classification, and a separate topology diagnostic. The prior raw quadratic proxy remains only in `I5_legacy_quadratic_proxy` and is not summed into R1 or the new I5 |
| I6 | Missing/merged → unavailable | `compute_i6_marking_instances` (:506) — empty GT returns **0**, not unavailable |
| D1 | Nonempty output | `compute_d1_valid_detection_single` (:539) — thresholded: tolerant recall ≥ 0.30, precision ≥ 0.30, predicted area ≥ 20 px, 5 px tolerance |
| D2 | Exact-match rate | `compute_d2_instance_agreement_single` (:549) — graded `1 − min(\|N_pred − N_gt\| / max(N_gt,1), 1)` |
| D3 | Plain IoU primary | Both exist (`compute_d3_iou_single` :559, `compute_d3_tolerant_f1_single` :575), but **tolerant F1 @ 5 px is treated as primary** downstream |
| D4 | Near-field (lower region) only | `compute_d4_band_metrics_single` (:589) — three bands: near/mid/far vertical thirds |
| D5 | Tag-based occlusion gap; dark-pixel thresholds explicitly ruled out | `compute_d5_dark_region_iou_gap_single` (:616) — **dark-region intensity proxy**, the approach this specification rules out. Blocked on occlusion screening tags reaching the manifest; both are documented per project decision |
| D6 | Pooled `ΣFN / ΣGT` across group | `compute_d6_missed_marking_ratio_single` (:637) — per-image `1 − recall`, then averaged |
| C1 | Pearson point-biserial, I1 ↔ binary D1 | `compute_correlations_from_records` (:846) — Spearman, I1 ↔ D3 tolerant F1 |
| C2, C3 | Pearson | Spearman primary (both reported where possible) |
| C4 | Median I5 split on **IoU** | `compute_c4_geometry_sensitivity_detail` — canonical-I5-only Spearman (primary: tolerant F1; also IoU and detected ratio), tertile effect with bootstrap CI, per-component and per-stratum associations, and an optional fixed-effects regression; the legacy median-split-on-tolerant-F1 scalar is retained as `C4`/`C4_detail.legacy_median_split` for backward compatibility |
| C5 | Mean of 0–1-oriented I1, I2, I3, D1, D3, D4 differences | `compute_condition_degradation` (:810) — uses normalized R1/R2 components |
| R1 | `100/0.95 × [0.30 I1 + 0.30 I2 + 0.20 I3 + 0.10 N(I4) + 0.05 N(I5)]` | `compute_r1_marking_readability_score` (:670) — `0.35 I1 + 0.35 I2 + 0.20 I3 + 0.10 I4`; **I5 not included**; no N() normalization |
| R2 | `0.25 D1 + 0.20 D2 + 0.20 D3 + 0.15 D4 + 0.08 R(D5) + 0.08(1−D6) + 0.04 D7` | `compute_r2_reference_detectability_score` (:684) — `0.35 D3_tolerant_f1_r5 + 0.20 D3_iou + 0.20 D4_near + 0.20 (1−D6) + 0.05 D8`; **D1, D2, and R(D5) absent** |
| R4 | Descriptive findings profile; fixed labels prohibited | `compute_r4_machine_readability_class` (:692) — returns `HIGH_/MODERATE_/LOW_MACHINE_READABILITY` on `R1 ≥ 80 and R2 ≥ 80` / `R1 ≥ 60 or R2 ≥ 60` / else. **These are exactly the fixed classification labels the specification prohibits** |

### Open specification questions

Two internal inconsistencies in the source tables should be resolved before the code is aligned:

1. **I1 scaling.** I1's range is given as 0–1, but R1 uses the term `0.30(I1/100)`, implying a 0–100 input. The same applies to `D1/100` and `D2/100` in R2 where D1 and D2 are both specified as 0–1. Either the ranges or the divisors need correcting.
2. **D6 aggregation.** D6 is defined as a pooled group-level ratio, but R2 consumes `(1 − D6)` alongside per-image terms. The level at which R2 is evaluated (per-image vs group) should be stated explicitly.

### Downstream code affected by any rename

- `evaluation/readiness_metrics.py` — record dicts (:740–772), `scalar_keys` list (:910–913), R1/R2 weight dicts, `compute_r3_bottleneck` alias fallbacks (:887–898)
- `evaluation/visualize_failures.py` — `DEFAULT_FAILURE_THRESHOLDS` (:20–31) and `DEFAULT_GOOD_THRESHOLDS` (:36–46) hardcode long-form keys, and metric keys become **on-disk directory names** (:143). Unmatched keys are dropped silently (:140), so a partial rename fails without error
- `evaluation/run_readiness.py` — `D8_confidence_mean` (:373); CSV column headers derive from record keys
- `tests/test_readiness_metrics_v2.py` — imports compute functions by name; pins R2 weights numerically (:109) and asserts `layer2.D7_temporal_jitter` (:171)
- `Algorithms/*.md` — 14 files whose **filenames encode metric IDs** (`D7_temporal_jitter.md`, `D8_confidence_mean.md`, `I4_thickness_stability.md`, `D5_dark_region_iou_gap.md`, …), plus the link table in `Algorithms/README.md`
- `evaluation/lane_geometry_complexity.py` — the I5 engine itself, and its reuse of `evaluation.lane_width_stability.construct_lane_centerlines`/`determine_measurement_space`
- Existing `outputs/readiness/*_readiness.json` and sibling `.jsonl`/`.csv` artifacts become stale on any rename

---

## Limitations

- Public imagery cannot directly measure retroreflectivity, bead loss, wet-night visibility, or field luminance.
- Mask annotations do not always preserve lane-marking type. I1 uses masks only to recover guidance geometry and infers unhinted pattern from RGB; inaccurate geometry, shadows, glare, wet roads, construction markings, occlusion, repairs, poor calibration, and insufficient view distance can reduce confidence or produce an unknown result.
- D5 requires documented occlusion screening tags. Dark-pixel thresholds alone are not a valid substitute, and the current implementation's dark-region proxy should be read as provisional.
- D7 is unavailable unless meaningful lane confidence is extracted from model outputs. No placeholder confidence is used, and confidence is not comparable across processors without established comparability.
- Results are reference-processor dependent. A different lane detector can produce different D-series, C-series, and R2 values on the same imagery. YOLOPX is the primary processor and CLRerNet the supplemental one; comparison between them tests whether findings are robust to processor choice.
- D3 and D4 results depend on the rasterization stroke width and tolerance defined in the evaluation methodology. Thin-line detectors score lower on pixel IoU than dense-mask models even after standardization, which is why precision and recall are also reported.
- The C-series measures are associations, not causal relationships. Observed patterns may reflect dataset composition, camera configuration, annotation practice, or processor training domain.
- R1 and R2 component weights are analytical choices for this pilot, not validated safety or risk weights.
- I4 depends strongly on camera calibration quality. Road nonplanarity, crest
  and sag vertical curves, unmodeled pitch/roll, camera-height changes, and
  ill-conditioned far-field transforms can bias road-plane widths.
- Low image resolution and horizon proximity limit usable support. These
  samples are reliability-gated, so short or distant lanes may be unavailable
  rather than assigned a score.
- Merges, splits, smooth tapers, work zones, and double lines require enough
  observed support to distinguish intended transition from incorrect pairing
  or noise. The reported profile type is a geometric hypothesis, not a
  roadway-design classification.
- Occlusion can leave too few reliable normal intersections. Object masks are
  optional and can themselves contain errors.
- Detector and annotation jitter, missing boundary instances, merged masks,
  inaccurate native polylines, and mask rasterization can affect boundary
  grouping and pair selection.
- Single-frame I4 does not invent temporal evidence. Camera or geometry
  filtering across ordered frames should be applied only when timestamps,
  stable calibration, and ego-motion are actually available.
- I5 metric curvature (1/m) is reported only with an affirmatively-declared
  road-plane metric scale; otherwise results are a `lane_width_normalized` or
  `normalized_image_proxy` proxy and are not comparable across camera
  configurations. I5 assumes a locally planar road — grade, crest/sag vertical
  curves, and superelevation are not modeled and can bias a calibrated result.
- I5 is not a roadway design-compliance metric, not a safety score, and does
  not establish causal ADS performance effects; C4 associations may reflect
  dataset composition, camera configuration, or processor training domain
  rather than a true geometry-performance relationship.
- I5's topology diagnostic is conservative by design and frequently
  unavailable from a single image; it should not be treated as a reliable
  merge/split detector on its own.
- `I5_pred` reflects that model's own detected geometry and is directly
  affected by its detection errors — it must not be used as the primary
  predictor of that same model's D metrics, and canonical C4 uses GT-guided
  I5 only.
