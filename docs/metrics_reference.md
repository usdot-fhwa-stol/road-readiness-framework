# Road Readiness / Lane-Marking Machine-Readability Metrics Reference

This framework is an image-based proxy for pavement-marking machine-readability and reference-processor detectability. It is not a field-certified road-readiness standard. It uses public imagery, ground-truth lane masks or polylines when available, dataset metadata, and reference processors such as YOLOPX/YOLOPv2.

The framework has 23 metric IDs across 4 layers: 22 computed metrics plus one reserved temporal metric, D7. D7 is reserved for temporal jitter and is not computed for single-frame datasets or single-frame evaluation runs.

The canonical implementation is per-image first. Each image produces one JSON-serializable metric record with dataset, split, model name, image identity, dimensions, sample and target metadata, mask occupancy, I metrics, D metrics, optional diagnostics, and per-image R1/R2 scores. Aggregation then computes overall summaries, strata summaries, correlations C1-C5, and R1-R4 summary outputs.

## Layer 1 - Infrastructure / Marking Image-Readability

Layer 1 answers: how visually readable are the lane or pavement markings in the image? These metrics use the image and the GT marking mask. They do not treat raw connected-component count as infrastructure quality.

### I1 - `I1_pattern_continuity`

Old problem corrected: the previous Canny-connected-component score penalized dashed lines because dashed markings are discontinuous by design, and it mostly measured annotation geometry.

Calculation: `infer_marking_pattern(gt_mask, lanes, meta)` first returns `solid`, `dashed`, `mixed`, or `unknown`. Metadata is used when it contains lane style/type fields; otherwise the mask geometry is analyzed using component size, elongation, alignment, spacing, and duty cycle. Dashed markings are scored by dash component image quality and dash regularity, without penalizing expected gaps. Solid markings are scored by image quality plus continuity after morphology-based gap bridging.

Output: 0 to 1, higher is better. `I1_percent` is also included for compatibility. Empty GT returns `None` with an unavailable reason in the record.

Project role: measures whether expected marking patterns are visually trackable by a machine vision stack.

### I2 - `I2_local_contrast`

Old problem corrected: the previous contrast compared marking pixels against all non-marking pixels, including sky, sidewalks, vehicles, and buildings.

Calculation: `I2 = abs(mean(gray[M]) - mean(gray[B])) / 255`, where `M` is the GT marking mask and `B` is a local background ring around the marking: `dilate(M, outer) AND NOT dilate(M, inner)`. If a road/drivable mask is available, the ring is restricted to it. If the ring is empty, the fallback is a lower-image road-like ROI rather than the whole image.

Output: 0 to 1, higher is better. Empty GT or no usable local background returns `None`.

Project role: measures local marking-to-road contrast, the contrast a lane detector actually sees near the marking.

### I3 - `I3_boundary_sharpness`

Old problem corrected: fixed Canny edge density around the GT boundary could score asphalt texture, shadows, or image noise as sharp lane boundaries.

Calculation: Sobel gradient magnitude is computed on grayscale. A GT boundary band is compared against the local background ring:

`I3 = clip((median(G[boundary_band]) - median(G[background_ring])) / (median(G[boundary_band]) + median(G[background_ring]) + eps), 0, 1)`

If the background ring is unavailable, the fallback compares boundary gradients to a robust image-gradient percentile.

Output: 0 to 1, higher is better. Empty GT or missing boundary returns `None`.

Project role: measures whether marking edges are locally crisp rather than merely present.

### I4 - `I4_thickness_stability`

Old problem corrected: the previous row-wise left/right span measured distance across multiple lane markings, not pavement-marking width.

Calculation: connected marking components are measured as segments. Segment thickness is approximated as `component_area / major_axis_length`. Stability is:

`I4 = 1 - clip(std(thickness_values) / (mean(thickness_values) + eps), 0, 1)`

Diagnostics include `I4_mean_thickness_px` and `I4_thickness_cv`.

Output: 0 to 1, higher is more stable. Empty GT returns `None`.

Project role: captures inconsistent paint width, worn markings, and abnormal segments without measuring lane width.

### I5 - `I5_geometry_complexity`

Old problem corrected: curvature was treated like bad infrastructure. Curves and complex lane geometry are context/difficulty, not degradation.

Calculation: when polylines exist, each lane is fit with `x(y) = ay^2 + by + c`, and a normalized curvature proxy is computed from the quadratic term and slope change. If only a mask exists, centerlines are estimated from components and the same proxy is attempted; otherwise a conservative mask-center fallback is used.

Output: 0 to 1, higher means more geometric complexity. Unavailable fitting returns `None`.

Project role: context variable for stratification and C4 geometry sensitivity. It is not a direct penalty in R1.

### I6 - `I6_marking_instance_count`

Old problem corrected: raw connected components are not lane/marking instances; a dashed lane can have many components.

Calculation: if `sample.target.lanes` exists, `len(lanes)` is used. Otherwise, components are grouped with a lane-direction closing heuristic so dashed segments on the same marking count as one likely instance. The record also reports `I6_raw_component_count` and `I6_topology_complexity`.

Output: integer instance count plus optional topology diagnostics. Empty GT gives 0.

Project role: context/gating variable. More or fewer markings are not inherently better infrastructure.

## Layer 2 - Reference-Processor Detectability

Layer 2 answers: how well did the reference processor detect markings that are present? Inputs are predicted mask, GT mask, image, and optional lane probability.

### D1 - `D1_valid_detection`

Calculation: per-image D1 is 1 only when tolerant recall, tolerant precision, and predicted area pass minimum thresholds:

`recall >= 0.30`, `precision >= 0.30`, and `pred_area >= 20` by default, with a 5 px tolerance.

Output: 0 or 1 per image; aggregate is the mean. Higher is better.

YOLOPX support: uses the binary lane mask extracted from `ll_seg_out`.

### D2 - `D2_instance_agreement`

Calculation: grouped marking instance count agreement:

`D2 = 1 - min(abs(N_pred - N_gt) / max(N_gt, 1), 1)`

GT lanes are used when available; otherwise grouped mask instances are used.

Output: 0 to 1, higher is better. If both GT and prediction have no markings, the value is unavailable rather than treated as success.

YOLOPX support: uses predicted lane mask components grouped by the same instance heuristic.

### D3 - `D3_iou` and `D3_tolerant_f1_r5`

Calculation: standard IoU is `TP / (TP + FP + FN + eps)`. Tolerant F1 uses distance transforms: tolerant recall is the fraction of GT pixels within 5 px of prediction, tolerant precision is the fraction of predicted pixels within 5 px of GT, and F1 is the harmonic mean.

Output: 0 to 1, higher is better. If both masks are empty, overlap is unavailable.

YOLOPX support: uses the binary lane segmentation mask. Tolerant F1 is preferred for thin lane masks where small lateral shifts can destroy IoU.

### D4 - `D4_near_iou`, `D4_mid_iou`, `D4_far_iou`

Calculation: the image is split into lower, middle, and upper vertical thirds: near, mid, and far. IoU is computed in each band. Optional tolerant F1 fields are also emitted: `D4_near_f1_r5`, `D4_mid_f1_r5`, `D4_far_f1_r5`.

Output: 0 to 1, higher is better. If a band has no GT markings, that band returns `None`.

YOLOPX support: uses the predicted lane mask after undoing letterbox padding.

### D5 - `D5_dark_region_iou_gap`

Old name corrected: this is not called occlusion robustness unless a true object/occlusion mask is used.

Calculation: without object masks, dark-region sensitivity is computed as:

`D5 = IoU_normal_region - IoU_dark_region`

Dark regions are inferred from image intensity. The metric returns `None` if either dark or normal regions have too few GT marking pixels.

Output: typically -1 to 1. Lower absolute degradation is better; positive values mean detection is worse in dark regions.

YOLOPX support: uses predicted lane mask and image intensity. It does not use object-detection outputs in the current single-frame path.

### D6 - `D6_missed_marking_ratio`

Calculation: pixel missed ratio:

`D6 = FN / (GT + eps) = 1 - recall`

Output: 0 to 1, lower is better. Empty GT returns `None`.

YOLOPX support: compares predicted lane mask to GT marking mask.

### D7 - `D7_temporal_jitter`

Calculation: reserved for temporally aligned frame sequences. It is not computed for current single-frame dataset evaluation.

Output: `None` with reason `unavailable_single_frame_evaluation`.

YOLOPX support: unavailable unless the evaluation pipeline provides frame sequences and temporal alignment.

### D8 - `D8_confidence_mean`

Old problem corrected: the placeholder value 0.5 has been removed.

Calculation: if a real lane probability map is available, D8 is `mean(lane_prob[pred_mask > 0])`. If the predicted mask is empty while a probability map exists, the implementation returns 0.0 for confidence over detected lane pixels. If no probability map is available, D8 is `None`.

Output: 0 to 1, higher means higher confidence. `None` is ignored and weights are renormalized in R2.

YOLOPX support: for two-channel `ll_seg_out`, softmax over channels is used and channel 1 is the lane probability. For one-channel output, sigmoid is used. If the channel layout is ambiguous, probability is unavailable and D8 is `None`.

## Layer 3 - Correlation and Bottleneck Metrics

C metrics are computed only from paired per-image records. They are not computed from repeated aggregate detection dictionaries.

Associations report both Pearson and Spearman where possible. The primary scalar is Spearman for C1-C3 because relationships can be monotonic but nonlinear. If `n < 10` by default, or if either variable has zero variance, the scalar is `None` and the detail dictionary includes a reason.

### C1 - `C1_pattern_continuity_to_detectability`

Primary association: Spearman between `I1_pattern_continuity` and `D3_tolerant_f1_r5`. Details also include the association to detected ratio derived from `1 - D6_missed_marking_ratio`.

Output: -1 to 1 or `None`. Positive means more trackable continuity tends to improve detectability.

### C2 - `C2_contrast_to_iou`

Primary association: Spearman between `I2_local_contrast` and `D3_iou`. Details also include tolerant F1.

Output: -1 to 1 or `None`. Positive means better local contrast tends to improve detection overlap.

### C3 - `C3_sharpness_to_detectability`

Primary association: Spearman between `I3_boundary_sharpness` and `D3_tolerant_f1_r5`. Details also include IoU.

Output: -1 to 1 or `None`.

### C4 - `C4_geometry_sensitivity`

Calculation: detection drop from low-complexity to high-complexity images:

`C4 = mean(D in low I5 half) - mean(D in high I5 half)`

`D3_tolerant_f1_r5` is used when available.

Output: effect size or `None`. Positive means detection is worse on high-complexity geometry. Details include group sizes and means.

### C5 - `C5_condition_degradation_ranking`

Calculation: groups records by a metadata condition key such as `condition`, `weather`, `timeofday`, or `scene`. For each condition with enough samples:

`degradation = readability_drop + detection_drop`

Drops are relative to a baseline such as `clear`, `daytime`, `normal`, or the largest available baseline group. Readability and detection use normalized R1/R2 components.

Output: sorted list of condition degradation dictionaries, or empty list with a reason when metadata or samples are insufficient.

## Layer 4 - Summary / Readiness Metrics

Layer 4 avoids hard deployment labels. It reports image-based machine-readability and reference-processor detectability.

### R1 - `R1_marking_readability_score`

Calculation: weighted average of visual readability metrics only:

`R1 = 100 * weighted_mean(I1, I2, I3, I4)`

Default weights: I1 0.35, I2 0.35, I3 0.20, I4 0.10. `None` values are ignored and weights are renormalized.

Output: 0 to 100. I5 and I6 are kept as context, not direct penalties or bonuses.

### R2 - `R2_reference_detectability_score`

Calculation: weighted average of reference detection metrics:

- `D3_tolerant_f1_r5`, weight 0.35
- `D3_iou`, weight 0.20
- `D4_near_f1_r5` or `D4_near_iou`, weight 0.20
- `1 - D6_missed_marking_ratio`, weight 0.20
- `D8_confidence_mean`, weight 0.05 only when real probability exists

`None` values are ignored and weights are renormalized. D1 and D2 are diagnostic and not over-weighted.

Output: 0 to 100.

### R3 - `R3_bottleneck`

Calculation: chooses the most supported bottleneck from valid C metrics and summary diagnostics. Candidate bottlenecks include low pattern continuity, low local contrast, low boundary sharpness, high missed marking ratio, dark-region degradation, geometry sensitivity, and worst condition stratum.

Output: dictionary with name, strength, evidence, and reason when a descriptive fallback is used.

### R4 - `R4_machine_readability_class`

Calculation:

```
if R1 >= 80 and R2 >= 80:
    HIGH_MACHINE_READABILITY
elif R1 >= 60 or R2 >= 60:
    MODERATE_MACHINE_READABILITY
else:
    LOW_MACHINE_READABILITY
```

Output: status, reason, R1, R2, bottleneck, and a warning that this is an image-based proxy, not field-certified road readiness.

## Interdependency Diagram

```
Per-image inputs
  image_rgb + gt_mask + optional lanes/meta/road_mask
       |                         pred_mask + optional lane_prob
       v                                      v
I1 pattern continuity                  D1 valid detection
I2 local contrast                      D2 instance agreement
I3 boundary sharpness                  D3 IoU + tolerant F1
I4 thickness stability                 D4 near/mid/far metrics
I5 geometry complexity                 D5 dark-region sensitivity
I6 instance/topology context           D6 missed marking ratio
                                       D7 reserved temporal jitter
                                       D8 optional confidence
       |                                      |
       +------------- per-image record -------+
                         |
                         v
          C1-C5 paired correlations / degradation
                         |
                         v
          R1 readability, R2 detectability
                         |
                         v
          R3 bottleneck, R4 machine-readability class
```

## Dataset Metadata Coverage

| Field / Use | BDD100K | CULane | CurveLanes | TuSimple |
|-------------|:-------:|:------:|:----------:|:--------:|
| Per-image GT mask | yes | yes | yes, rasterized from polylines | yes, rasterized from polylines |
| GT polylines | usually no in YOLOPX-prepared masks | yes when `.lines.txt` is used | yes | yes |
| Weather | yes with detection annotations | no; scenario is stored as `condition` | no | no |
| Time of day | yes with detection annotations | night scenario can map to `timeofday=night` | no | no |
| Scene/context | yes with detection annotations | scenario condition lists | no | no |
| Lane style metadata | dataset-dependent and often absent | absent | absent | absent |
| Lane probability for D8 | model-output dependent | model-output dependent | model-output dependent | model-output dependent |

## Limitations

- Public imagery cannot directly measure retroreflectivity, bead loss, wet-night visibility, or field luminance.
- Mask annotations do not always preserve lane-marking type. Solid/dashed inference from masks is a proxy when metadata lacks marking type.
- D5 is dark-region sensitivity, not true occlusion robustness, unless aligned object/occlusion masks are supplied.
- D8 is unavailable unless lane probability is extracted from model logits; no placeholder confidence is used.
- Results are reference-processor dependent. A different lane detector can produce different R2/C metrics on the same imagery.
- Single-frame datasets cannot support temporal jitter D7 without sequence alignment.
