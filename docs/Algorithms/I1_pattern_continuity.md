# I1 - RGB Pattern Continuity

**Public functions:** compute_i1_pattern_continuity,
compute_i1_pattern_continuity_details, and
analyze_lane_pattern_continuity

**Output:** continuity and degradation in [0, 1], or None when evidence is
insufficient. Higher continuity is better; higher degradation means more
expected paint appears absent or severely faded.

## Meaning

I1 estimates expected-paint continuity. It first identifies the intended
marking pattern (solid, dashed, sufficiently supported dotted, mixed, or
unknown), then evaluates only positions at which that pattern expects paint.
Observed condition is reported separately as intact, faded, fragmented,
partially_missing, or unknown_condition.

A worn solid marking remains an intended solid. An intentional dashed gap is
not wear.

## Inputs and geometry priority

The original RGB image is the only physical-paint evidence. Ground-truth and
prediction representations locate sampling corridors:

1. native float polylines;
2. compatible lane_json;
3. grouped centerlines recovered from a mask;
4. unavailable.

Canonical I1 uses only GT geometry and optional explicit target pattern fields.
Operational I1_pred uses only prediction geometry and receives no GT hints.
Morphological closing is permitted on a disposable mask copy for geometry
grouping, never on the RGB paint-evidence signal.

## Algorithm

1. Reject nonfinite, out-of-frame, duplicate, short, and unstable geometry.
   Robustly smooth and resample each supported centerline.
2. Estimate local tangents and normals. Bilinearly sample an adaptive center
   marking band and left/right pavement side bands from RGB.
3. Compute continuous paint evidence p_i from relative luminance, white/yellow
   likelihood, paired stripe edges, and cross-stripe consistency.
4. Compute validity q_i from corridor support, road-mask compatibility,
   explicit object occlusion, glare/saturation, and geometry stability.
5. Create adaptive occupancy only for run/gap diagnostics. Fit the continuous
   p_i signal against solid and bounded periodic templates.
6. Require repeated alternating intervals, normally at least three cycles,
   consistent normalized runs/gaps, periodicity, plausible duty ratio, and a
   fit margin before selecting dashed. Otherwise select a supported solid
   hypothesis or return unknown.
6b. Visible-paint presence gate. Infer a pattern only when the strongest paint
   evidence along the corridor (a high percentile of p_i over reliable samples)
   exceeds a visible-paint floor, and require the dashed painted phase to be
   brighter than the fitted gaps. Otherwise return unknown. This blocks
   explaining uniformly dark, paint-free frames (e.g. a pitch-dark night scene)
   as a dashed marking with a confident near-zero continuity. The same floor is
   applied to the expected-paint corridor before scoring condition, so an unseen
   marking returns None (reason `no_visible_paint_evidence`) rather than a
   fabricated score, even under a trusted metadata hint.
7. Score only expected-paint positions:

       D = 1 - sum(q_i m_i p_i) / (sum(q_i m_i) + epsilon)
       C = 1 - D

   Here m_i is one where the selected pattern expects paint and zero in an
   intentional gap.
8. Assign a diagnostic condition from degradation and the spatial structure of
   missing/weak expected paint.
9. Aggregate valid lanes with confidence x valid fraction x square root of
   bounded support. Unavailable lanes are never averaged as zero.

Perspective uses a supplied homography first, then geometry-derived scale
normalization, local-window normalization, and finally image arc length. No
fixed real-world dash length is assumed.

## Compatibility and R1

compute_i1_pattern_continuity remains the scalar compatibility wrapper.
Canonical I1 continues to populate I1, I1_pattern_continuity, I1_percent,
I1_pattern_type, and I1_pattern_info and is the only I1 used in R1.
Prediction-guided results use I1_pred_* fields and do not affect R1.

Pattern confidence is not multiplied into continuity. I2, I3, and I4 are not
added as I1 score terms, avoiding double counting in R1.

## Debugging and limitations

Debug mode retains corridor geometry, p_i, q_i, occupancy, the fitted template,
and likely missing expected paint for developer visualization. Normal JSON
records omit these arrays.

This is an image-based proxy, not a field-certified pavement-marking
assessment. Shadows, glare, wet pavement, temporary markings, repairs,
occlusion, inaccurate geometry, poor calibration, and insufficient view
distance may lower confidence or make the result unavailable.
