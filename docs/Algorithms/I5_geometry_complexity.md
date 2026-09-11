# I5 — Lane-Alignment Geometry Complexity

**Module:** `evaluation/lane_geometry_complexity.py`
**Public functions:** `analyze_lane_geometry_complexity`, and the readiness-metrics
wrappers `compute_i5_geometry_complexity_details`, `compute_i5_alignment_complexity`,
`compute_i5_legacy_quadratic_proxy`, `compute_i5_geometry_complexity` (legacy alias).
**Output:** `alignment_complexity` (aliased as `I5`/`I5_geometry_complexity`) in
`[0, 1]`, higher = geometrically more demanding visible plan-view alignment;
`None` with an explicit `unavailable_reason` when geometry cannot be estimated
reliably. `complexity_confidence` is a separate 0–1 epistemic-confidence value,
never multiplied into the score.
**Uses the image?** No — classical projective geometry and differential
curvature analysis on lane geometry only (RGB pixels are not inspected).

## Meaning

I5 characterizes **geometric context and difficulty**, not road quality,
pavement-marking condition, detector quality, or ADS readiness. A high I5 value
means the visible alignment is geometrically more demanding (tighter curves,
more curvature change, more sign reversals). It does not mean the road is
defective, unsafe, non-compliant, or that an ADS will fail. Curved roads,
ramps, roundabouts, and reverse curves must not automatically look "bad."

There is no established universal scalar for "road geometric complexity." I5 is
aligned with, but does not claim to *be*, several established interpretable
concepts:

- **PIARC-style bendiness** — total absolute heading change divided by section
  length.
- **Road-design Curvature Change Rate (CCR)** — sum of absolute deflection
  angles divided by segment length over a homogeneous segment.
- **ASAM OpenDRIVE geometry primitives** — line, constant-curvature arc,
  linearly-changing-curvature (clothoid/spiral), and parametric-cubic
  fallback — mirrored by I5's `profile_class`.
- **ASAM/OpenDRIVE's separation of continuous alignment from lane linkage /
  junction topology** — mirrored by I5's split between alignment complexity
  and topology complexity.

None of the tau reference scales, weights, or thresholds below are ISO, SAE,
ASAM, FHWA, AASHTO, or PIARC standards. They are documented, versioned,
image-analysis calibration parameters for this pilot (`GeometryComplexityConfig`),
exercised by the sensitivity/behavioral checks in
`tests/test_i5_geometry_complexity.py`.

## Canonical vs. prediction-guided

- **Canonical I5** (`I5*`, no `_pred` suffix) uses **ground-truth geometry
  only** (`I5_geometry_source = "ground_truth"`). It is the only I5 value used
  by C4 and is independent of whichever model is being evaluated: changing
  the predicted mask/lanes while GT and image stay fixed does not change it.
- **Prediction-guided I5** (`I5_pred*`) uses **one evaluated model's geometry
  only**. It never reads GT masks, GT lane points, GT lane count, or GT
  metadata. It is diagnostic only — it must not be used as the primary
  predictor of that same model's own D metrics (that would confound detector
  failure with estimated road complexity).

Both share the same underlying engine and the same geometry-source priority
already used for I1/I4: native float polylines, then compatible `lane_json`,
then mask-grouped centerlines, then unavailable
(`evaluation.readiness_metrics._select_i1_guidance`).

## Geometry extraction and centerline construction

`analyze_lane_geometry_complexity` reuses the shared I1/I4 geometry engines
rather than re-implementing them:

1. **Validation** (`validate_lane_geometry`, wraps `prepare_guidance_lanes`):
   rejects nonfinite/duplicate/out-of-frame points, orders points consistently,
   rejects too-short lanes, and lightly smooths — identical to I1.
2. **Mask grouping** (`extract_grouped_centerlines_from_mask`, wraps
   `extract_guidance_lanes_from_mask`): groups dashed/disconnected components
   into one centerline per logical marking. Morphological closing is used only
   for grouping, never as evidence of continuous paint, and a thick
   segmentation stroke does not change the recovered centerline.
3. **Boundary fitting, double-line collapsing, and pairing**
   (`pair_boundaries_and_construct_centerlines`, wraps
   `evaluation.lane_width_stability.construct_lane_centerlines`, itself built
   from `validate_and_fit_boundaries` / `collapse_double_line_boundaries` /
   `pair_adjacent_boundaries`): when two adjacent logical boundaries pair
   plausibly, I5 uses the **paired centerline**; otherwise it falls back to
   the **boundary's own curve**. This is the same reason parallel offsets of
   one road curve and thick-vs-thin YOLOPX masks converge on comparable
   curvature — they are built from the same shared corridor/boundary engine
   used for I4.

## Perspective and measurement space

Priority, reusing `evaluation.lane_width_stability.determine_measurement_space`
then layering I5's own labels via `determine_i5_measurement_space`:

1. `calibrated_metric_bev` — explicit image-to-road homography with
   metric units affirmatively declared (`calibration_meta["metric_scale_available"] = True`
   and `units in {"m", ...}`). Curvature is reported in 1/m.
2. `calibrated_scale_free_bev` — a valid homography without a declared metric
   scale. Curvature is in 1/road-plane-unit.
3. `lane_width_normalized` — no calibration, but a same-source lane-width
   estimate (`lane_width_context`, e.g. from that source's own I4 result) is
   available. Coordinates are divided by that width.
4. `normalized_image_proxy` — no calibration and no lane width. Coordinates
   are divided by the image diagonal. Curvature has no metric units and is
   **not comparable across camera configurations**; C4 comparisons should be
   restricted to a homogeneous dataset/camera group when this mode is active.

`transform_to_road_plane` applies the resulting scale; curvature is never
estimated directly from raw, unscaled x(y) image coordinates. Calibration is
never estimated by forcing the analyzed lane to be straight or constant
curvature. Transform diagnostics (measurement space, homography source,
condition number, valid transformed area, fallback reason) are always
recorded (`I5_transform_diagnostics`).

## Robust curve fitting

`robustly_fit_parametric_curve` fits `c(s) = [x(s), y(s)]` over an
approximately uniform arc-length parameter `s`:

1. A windowed local-median filter replaces sparse hard outliers (points far
   from their local neighborhood median), analogous to a RANSAC-style
   initialization.
2. A Savitzky-Golay filter (the same technique already used for I4 boundary
   smoothing, `evaluation.lane_width_stability._locally_smoothed_curve`)
   smooths the cleaned sequence. Both steps are **local** (bounded window),
   which matters: a global smoothing spline's automatic knot selection can,
   for a curve containing one real sharp transition (a compound curve), remove
   interior knots in a way that forces the fit to ring across the *entire*
   domain — this was directly observed during development and is why I5 does
   not rely on a single global smoothing-spline fit with automatically chosen
   smoothing.
3. An interpolating cubic (or quadratic/linear for very short lanes) spline is
   fit through the cleaned sequence purely to obtain analytic tangents and a
   densely resampled curve. This is never a single unconstrained global
   quadratic and never a high-degree global polynomial.

Extrapolation outside the observed arc-length support is never performed.
Inlier fraction and a robust residual are recorded per lane.

## Curvature profile

Curvature is evaluated via a **local three-point (Menger) formula** on the
fitted, densely resampled curve, with a wide bounded local window rather than
adjacent samples:

```
kappa(s) ~= 2 * cross(p1-p0, p2-p1) / (|p0p1| * |p1p2| * |p2p0|)
```

This is exact for a true circular arc (constant curvature = 1/radius,
independent of window size) and — unlike a global spline second derivative —
cannot ring across the whole domain in response to one localized sharp
feature elsewhere in the curve, since it only ever looks at a bounded local
neighborhood. The curvature *gradient* `dkappa/ds` is estimated with that same
wide local baseline rather than immediate-neighbor differencing, so it is not
dominated by point-level noise (differencing an already-differentiated
quantity again at the finest sample spacing would otherwise amplify residual
noise sharply).

Reported components:

- **Bendiness** `B`: weighted mean of `|kappa(s)|`.
- **P95 curvature** `K95`: weighted 95th-percentile of `|kappa(s)|`.
- **Curvature variation** `V`: weighted mean of `|dkappa/ds|`.
- **P95 curvature gradient** `G95`: weighted 95th-percentile of `|dkappa/ds|`.
- **Total absolute heading change** `H`: trapezoidal integral of `|kappa(s)|`
  over `s` (a diagnostic; not independently weighted in the scalar, since it
  is mathematically related to bendiness by support length).
- **Reversal count / density**: see below.
- **Tortuosity**: `L / chord_length - 1`, from the fitted curve's endpoints.
- Conventional curvature RMS and standard deviation (diagnostics only).

Sample reliability weights combine: local curvature-window validity, edge
proximity to the observed support, horizon proximity (image-space `y` below a
configured fraction of image height), and explicit object-mask occlusion.
Invalid or excluded samples are never converted to zero curvature — they are
excluded from all weighted statistics.

**Reversal detection** uses a Schmitt-trigger deadband: a sign reversal is
only registered once `|kappa|` exceeds a "high" threshold (a documented
fraction of the space's `tau_bendiness` reference scale) with a sign opposite
the last *confirmed* sign, and only after a minimum arc-length separation from
the previous reversal. Small oscillations around zero never flip the
confirmed sign and never count as a reverse curve.

## Alignment profile classification

`segment_geometry_primitives` fits weighted-least-squares hypotheses to
`kappa(s)` — straight (0), constant (1 parameter), linear/transition-curve (2
parameters), and an optional single-change-point piecewise model (4
parameters, candidate change points searched over a bounded grid) — and
selects among them by BIC. Critically, a more flexible model is only allowed
to win when the curvature change it claims exceeds a documented fraction of
`tau_bendiness`: with real (if small) numerical fitting noise, a more flexible
model can always reduce raw SSE somewhat even for a perfectly constant curve,
so a magnitude gate (not just a BIC margin) is required before accepting
`transition_curve` or `compound_curve` over `constant_curve`.

`classify_alignment_profile` turns this into `profile_class`:

- `straight` — bendiness negligible relative to `tau_bendiness` and no
  reversal.
- `reverse_curve` — at least one supported sign reversal (checked before the
  BIC winner, since a reversal is not representable by any of the
  single-sign hypotheses above).
- `constant_curve` / `transition_curve` — the corresponding BIC winner.
- `compound_curve` — the piecewise model wins and both segments are
  predominantly the same sign.
- `irregular_or_noisy` — piecewise but mixed-sign (not a coherent reverse
  curve), or the winning model's residual is still large relative to
  `tau_bendiness` with a weak BIC margin.
- `unknown` — insufficient reliable curvature samples.

The classification describes visible geometry only; it is never presented as
the road's official design element.

## Alignment complexity scalar

Each component `f_j` (bendiness, p95 curvature, curvature variation, reversal
density) is mapped through a versioned saturating curve
`n_j = 1 - exp(-f_j / tau_j)`, where `tau_j` is a configured reference scale
in the active measurement space's units (separate `tau` sets for
`calibrated_metric_bev`, `calibrated_scale_free_bev`/`lane_width_normalized`,
and `normalized_image_proxy`). Components are combined by an equal-weighted
RMS:

```
alignment_complexity = sqrt(sum(w_j * n_j^2) / sum(w_j))
```

Total heading change and mean absolute curvature are mathematically related by
support length, so only bendiness (not heading change) is a weighted
component. Dataset min/max normalization is deliberately not used — it would
make historical scores shift whenever images are added or removed from a
dataset.

## Topology (separate from alignment)

`infer_lane_topology` is a conservative, separate diagnostic
(`I5_topology_complexity`) comparing the number of logical boundaries with
image-space support near the bottom of the frame ("near") against the number
near the top ("far"). A lane-count difference is reported as a topology
complexity in `[0, 1]` together with supported-split/merge counts; anything
with fewer than two usable boundaries in either band returns `None` with an
explicit reason — single-image topology inference is inherently uncertain, and
this module never claims certainty from one image. Topology is **not** folded
into `alignment_complexity`, and it is not the same thing as I6: I6 is the
amount of identifiable marking-*instance* information; I5's topology
diagnostic is about supported *connectivity changes* among logical lanes
(splits/merges), not instance counting. Raw connected-component count is never
treated as topology.

## Multi-lane aggregation

Each usable lane/lane-centerline gets its own diagnostic entry (profile class,
confidence, complexity, bendiness, p95, variation, reversal stats, tortuosity,
fit quality, etc. — see `I5_lane_diagnostics`). `aggregate_lane_complexity`
combines only the valid lanes with confidence- and bounded-support-aware
weights (`confidence * sqrt(min(support, support_cap))`), so unavailable lanes
are never averaged in as zero and one very long lane cannot dominate without
bound. Extra parallel lanes at the same alignment therefore do not
automatically increase the aggregate (they average toward the same value, not
sum); a real lane-count change instead shows up in the separate topology
diagnostic.

## Legacy quadratic-proxy migration

The previous algorithm (`compute_i5_legacy_quadratic_proxy`, still available
under that name and under `compute_i5_geometry_complexity` as an unchanged
compatibility alias) fit a single unconstrained quadratic `x(y) = ay^2+by+c` in
raw image pixels — exactly the kind of raw x(y) proxy this redesign replaces.
Its value is preserved unchanged under `I5_legacy_quadratic_proxy` in every
record for historical comparison, but it is **not directly comparable** to the
new `I5`/`I5_alignment_complexity` — do not treat them as the same quantity in
downstream analysis or trend charts.

## R1 and C4

I5 is **not** included in `compute_r1_marking_readability_score`. Higher
curvature or topology complexity is context, not infrastructure degradation,
and `I5_pred` never affects canonical R1. C4 (`compute_c4_geometry_sensitivity_detail`
in `evaluation/readiness_metrics.py`) uses **canonical I5 only** and reports
Spearman correlations (overall, IoU, detected-ratio), a bottom/top-tertile
effect with a bootstrap CI, per-component associations (bendiness, p95,
variation, reversal density, topology), per-model/per-dataset/per-measurement-space
strata, sample counts and skip reasons, and an optional dataset/model
fixed-effects-adjusted regression (restricted to one measurement space at a
time, explicitly non-causal). The legacy median-split effect is retained as
`C4_detail.legacy_median_split` and the top-level `C4`/`C4_geometry_sensitivity`
scalar stays backward compatible with it.

## Debug visualization

`evaluation.visualize_failures.render_i5_lane_debug` / `save_i5_lane_debug`
(requires `analyze_lane_geometry_complexity(..., debug=True)`) render each
analyzed lane's centerline colored by profile class, low-reliability samples,
and a kappa(s) chart with change points. Normal (non-debug) JSON records never
carry the full curvature arrays.

## Limitations

- Requires an explicit, affirmatively-declared metric scale for physical
  curvature units; otherwise results are an image-space or lane-width-relative
  proxy, not comparable across camera configurations.
- Assumes a locally planar road; grade, crest/sag vertical curves,
  superelevation, and general road nonplanarity are not modeled and can bias a
  calibrated result.
- Short visible distance, the image horizon, and occlusion reduce usable
  support and confidence, or make a lane unavailable outright.
- Work zones, temporary markings, and incomplete or merged annotations can
  affect geometry-source selection identically to I1/I4.
- Detector errors (missed or spurious lane segments) directly affect
  `I5_pred`; this is expected and is exactly why `I5_pred` must not be used to
  explain that same model's own detection metrics.
- Single-image topology inference is inherently uncertain; the topology
  diagnostic is conservative and frequently `None`.
- This is not a roadway design-compliance metric, not a safety score, and does
  not establish causal ADS performance effects.
