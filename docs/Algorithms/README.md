# Metric Algorithms

Algorithm definitions for the Layer 1 image-readability metrics (**I1–I6**) and the
Layer 2 reference-processor detectability metrics (**D1–D8**), as implemented in
[`evaluation/readiness_metrics.py`](../../evaluation/readiness_metrics.py).

Each file states, for one metric: the inputs it consumes, the exact step-by-step
computation, the output range, edge-case behaviour, and any known caveats about
what the number does and does not mean.

## Conventions used across all metrics

- **`gt_mask`** — binary ground-truth marking mask, `1` = marking pixel. For the
  polyline datasets (CULane, TuSimple, CurveLanes) this mask is *rasterised* from
  polylines at a fixed stroke width (`lane_eval/converters/lanes_to_mask.py`,
  default 16 px), so it is a fixed-width ribbon centred on the paint, **not** a
  pixel-accurate paint footprint. This caveat is repeated where it materially
  affects a metric.
- **`pred_mask`** — binary predicted marking mask from the reference model.
- **`image_rgb`** — the source image, `HxWx3`. Converted to grayscale float for all
  intensity/gradient work.
- **local background ring** — `dilate(M, outer) AND NOT dilate(M, inner) AND NOT M`,
  optionally intersected with a road/drivable mask. This is the strip of road
  immediately surrounding the marking; see `compute_local_background_ring`.
- **`None`** — every metric returns `None` (not `0`) when it cannot be computed
  (empty GT, no usable background, degenerate geometry). `None` values are ignored
  and weights renormalised when aggregated into R1/R2.
- **Tolerance** — pixel-distance tolerance for "tolerant" precision/recall is 5 px
  by default, applied via a distance transform.

## Layer 1 — Image readability (I metrics)

| ID | File | One-line definition | Uses image? |
|----|------|---------------------|:-----------:|
| I1 | [I1_pattern_continuity.md](I1_pattern_continuity.md) | Pattern-aware continuity of the markings | yes |
| I2 | [I2_local_contrast.md](I2_local_contrast.md) | Marking-vs-adjacent-road intensity contrast | yes |
| I3 | [I3_boundary_sharpness.md](I3_boundary_sharpness.md) | Edge crispness of marking boundaries | yes |
| I4 | [I4_thickness_stability.md](I4_thickness_stability.md) | Consistency of marking width across segments | no |
| I5 | [I5_geometry_complexity.md](I5_geometry_complexity.md) | Classical differential-geometry alignment complexity + topology (context) | no |
| I6 | [I6_marking_instances.md](I6_marking_instances.md) | Number of marking instances + topology (context) | no |

## Layer 2 — Reference detectability (D metrics)

| ID | File | One-line definition |
|----|------|---------------------|
| D1 | [D1_valid_detection.md](D1_valid_detection.md) | Did the model validly detect the marking? (0/1) |
| D2 | [D2_instance_agreement.md](D2_instance_agreement.md) | Agreement of predicted vs GT instance count |
| D3 | [D3_iou_and_tolerant_f1.md](D3_iou_and_tolerant_f1.md) | IoU and tolerant F1 overlap |
| D4 | [D4_band_metrics.md](D4_band_metrics.md) | Near / mid / far vertical-band overlap |
| D5 | [D5_dark_region_iou_gap.md](D5_dark_region_iou_gap.md) | Detection drop in dark regions |
| D6 | [D6_missed_marking_ratio.md](D6_missed_marking_ratio.md) | Fraction of GT marking pixels missed |
| D7 | [D7_temporal_jitter.md](D7_temporal_jitter.md) | Reserved — temporal jitter (not computed single-frame) |
| D8 | [D8_confidence_mean.md](D8_confidence_mean.md) | Mean model confidence over predicted pixels |

## How they combine

```
R1 = 100 * weighted_mean(I1:0.35, I2:0.35, I3:0.20, I4:0.10)     # readability
R2 = 100 * weighted_mean(D3_f1:0.35, D3_iou:0.20, D4_near:0.20,   # detectability
                         (1-D6):0.20, D8:0.05)
```

I5 and I6 are **context variables** — they are not summed into R1. D1, D2, D5, D7
are diagnostics and are not over-weighted in R2.
