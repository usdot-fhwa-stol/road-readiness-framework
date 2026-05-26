# Road Readiness Metrics Reference

17 metrics across 4 layers. Layers 1–2 run per image; Layers 3–4 aggregate over the full evaluation set (or stratum).

---

## Layer 1 — Infrastructure Readiness
*Input: ground-truth mask only. Answers: "how good is the road markings themselves?"*

| ID | Name | Equation / Calculation | Output Range | Road Readiness Relevance |
|----|------|------------------------|:------------:|--------------------------|
| **I1** | Lane Continuity Score | `boundary = Canny(gt_mask × 255, 50, 150)` → connected-component analysis → `(pixels in components ≥ 50px / total boundary pixels) × 100` | 0 – 100 | Broken or dashed markings reduce detector confidence and increase false-negative rate. A low I1 predicts high D6 (missed lanes). |
| **I2** | Boundary Contrast Ratio | `gray = 0.299R + 0.587G + 0.114B` → `\|mean(gray[lane]) − mean(gray[road])\| / 255` | 0 – 1 | Faded or weathered paint lowers the luminance gap between marking and asphalt. Low contrast is the primary cause of detection failure in overcast and night conditions. |
| **I3** | Boundary Sharpness Score | `boundary_region = dilate(Canny(gt_mask), 5×5)` → `sum(Canny(image) & boundary_region) / sum(boundary_region)` | 0 – 1 | Blurry or worn-edge markings produce soft boundaries that CNN feature extractors struggle to localise. Correlates with D3 (IoU) via C3. |
| **I4** | Lane Width Stability | Per-row: `left = first lane pixel`, `right = last lane pixel`, `width = right − left` → `std(widths)` across all rows with lane pixels | pixels (lower = better) | High variance in apparent lane width indicates worn paint, road damage, or overlapping markings. Unstable width causes inconsistent mask predictions across the frame. |
| **I5** | Road Curvature Complexity | Per-row: `x_center = mean(x positions of lane pixels)` → `std(x_centers) / image_width` | 0 – 1 (lower = straighter) | Tight curves shift lane position rapidly across rows. Models trained predominantly on highway data underperform on curved roads; I5 quantifies this geometric difficulty. |
| **I6** | Lane Count (GT) | `scipy.ndimage.label(gt_mask)` → number of connected components | integer ≥ 0 | Verifies that the dataset sample actually contains lane annotations. Zero means no evaluable infrastructure; used to gate R1 scoring. |

---

## Layer 2 — Detection Performance
*Input: predicted mask vs. ground-truth mask. Answers: "how well does the model detect what's there?"*

| ID | Name | Equation / Calculation | Output Range | Road Readiness Relevance |
|----|------|------------------------|:------------:|--------------------------|
| **D1** | Detection Success Rate | `(images where sum(pred_mask) > 0) / total_images × 100` | 0 – 100 % | A frame with zero detections is a complete failure — the AV has no lane reference. This is the hardest safety floor: even partial detection is better than none. |
| **D2** | Lane Count Accuracy | `(images where connected_components(pred) == connected_components(gt)) / total × 100` | 0 – 100 % | Wrong lane count means the model is either hallucinating lanes or merging adjacent ones. Either error causes incorrect lateral position estimation. |
| **D3** | Segmentation IoU | Per image: `TP / (TP + FP + FN + ε)`, then `mean` over all images. `TP = (pred > 0) & (gt > 0)` etc. | 0 – 1 | The standard pixel-level accuracy metric. Directly measures how much of the true lane area is correctly captured. Used as the primary detection quality signal in Layer 3 correlations. |
| **D4** | Near-Field IoU | Same IoU formula applied only to `pred[H/2:H, :]` and `gt[H/2:H, :]` (bottom half of image) | 0 – 1 | Lanes in the bottom half correspond to 0–30 m from the vehicle — the immediate control horizon. Near-field accuracy is more safety-critical than distant lane detection; a model can fail on distant lanes without immediate danger. |
| **D5** | Occlusion Robustness Gap | `occlusion_mask = HSV_V < 100 (dark pixels)` → `IoU_visible − IoU_occluded` computed in each region separately | −1 – 1 (lower = more robust) | Vehicles, shadows, and tunnels occlude lane markings. A large gap means the model degrades sharply when markings are partially hidden — a frequent real-world scenario. |
| **D6** | Detection Gap Ratio | `sum(FN pixels) / sum(GT pixels + ε)` across all images | 0 – 1 (lower = better) | Measures the fraction of true lane pixels the model completely missed. High D6 means the vehicle is operating with an incomplete lane map, which is particularly dangerous at lane-change decision points. |
| **D8** | Confidence Mean | `mean(model confidence scores)` *(placeholder: 0.5 until YOLOPv2 confidence is extracted from `ll_seg_out`)* | 0 – 1 | Low mean confidence is a signal that the model is uncertain about its own predictions, warranting lower trust in downstream planning modules. Currently a stub pending model integration. |

---

## Layer 3 — Correlation & Bottleneck Analysis
*Input: per-image I1–I6 and D1/D3 arrays. Answers: "which infrastructure property most limits detection?"*

| ID | Name | Equation / Calculation | Output Range | Road Readiness Relevance |
|----|------|------------------------|:------------:|--------------------------|
| **C1** | Continuity → Detection | `Pearson(I1_per_image, D1_per_image)` | −1 – +1 | Quantifies whether frames with fragmented markings (low I1) systematically produce detection failures (D1 = 0). A strong positive correlation means improving marking continuity would directly raise the detection rate. |
| **C2** | Contrast → IoU | `Pearson(I2_per_image, D3_per_image)` | −1 – +1 | Tests whether low-contrast markings cause lower segmentation accuracy. Guides investment: if C2 is high, repainting faded markings yields the largest detection improvement per cost. |
| **C3** | Sharpness → IoU | `Pearson(I3_per_image, D3_per_image)` | −1 – +1 | Tests whether blurry/worn boundaries degrade IoU. Distinguishes worn-paint failure (C3 high) from other failure modes. |
| **C4** | Curvature → Performance Gap | `mean(D3 \| I5 < median(I5)) − mean(D3 \| I5 ≥ median(I5))` | −1 – +1 (positive = curves hurt) | Splits frames into straight vs. curved halves and measures the IoU drop on curves. A high C4 indicates the model was not trained on sufficient curved-road data and would fail on winding roads. |
| **C5** | Condition Degradation Ranking | For each condition `c`: `Δ = (mean(I1,I2,I3)\|clear − mean(I1,I2,I3)\|c) + (mean(D1,D3,D4)\|clear − mean(D1,D3,D4)\|c)` → sorted descending | List of (condition, Δ) | Combines infrastructure and detection drops into a single degradation score relative to clear-day baseline. Directly answers: "in what weather/scene should we invest first?" Requires weather/scene metadata (BDD100K or CULane scenario list files). |

---

## Layer 4 — Readiness Verdict
*Input: aggregated L1–L3 results. Answers: "is this road/sensor combination ready for deployment?"*

| ID | Name | Equation / Calculation | Output | Road Readiness Relevance |
|----|------|------------------------|--------|--------------------------|
| **R1** | Infrastructure Readiness Score | Weighted sum: `0.30×(I1/100) + 0.30×I2 + 0.20×I3 + 0.10×max(1−I4/50,0) + 0.05×max(1−I5/0.5,0) + 0.05×(I6>0)`, scaled to 0–100 | 0 – 100 | A single infrastructure quality number. Weights I1 and I2 highest (0.30 each) because continuity and contrast are the two dominant causes of detection failure observed in lane-marking research. |
| **R2** | Detection Readiness Score | Weighted sum: `0.25×(D1/100) + 0.20×(D2/100) + 0.20×D3 + 0.15×D4 + 0.08×(1−D5) + 0.08×(1−D6) + 0.04×D8`, scaled to 0–100 | 0 – 100 | Summarises model performance. D1 (0.25) and D3 (0.20) carry the most weight because a missed detection and a low-IoU detection are respectively the most dangerous failure modes for AV control. |
| **R3** | Bottleneck Identification | `argmax(\|C1\|, \|C2\|, \|C3\|, \|C4\|, max degradation from C5)` → `(name, strength)` | (str, float) | Pinpoints *which* property (continuity, contrast, sharpness, curvature, or a specific condition) is most limiting overall readiness. Drives actionable infrastructure maintenance priorities. |
| **R4** | Readiness Classification | `R1 > 80 AND R2 > 85` → **READY**; `R1 > 60 OR R2 > 70` → **MARGINAL**; else → **NOT_READY** | READY / MARGINAL / NOT_READY | Binary-ish deployment gate. Thresholds are conservative by design: both infrastructure and detection must be strong to reach READY, reflecting the redundancy requirement of safety-critical systems. |

---

## Metric Interdependencies

```
I1 (continuity) ──────┐
I2 (contrast)   ──────┤──► C1–C4 (correlations) ──► R3 (bottleneck)
I3 (sharpness)  ──────┘                              │
I4 (width std)  ────────────────────────────────┐    │
I5 (curvature)  ────────────────────────────────┤──► R1 ──► R4 (verdict)
I6 (lane count) ────────────────────────────────┘    │
                                                      │
D1 (detection rate) ──┐                              │
D2 (count accuracy) ──┤                              │
D3 (IoU)         ─────┤──────────────────────────► R2 ──► R4
D4 (near-field)  ─────┤
D5 (occlusion)   ─────┤
D6 (gap ratio)   ─────┘

C5 (condition ranking) uses all of {I1,I2,I3} and {D1,D3,D4} stratified by weather/scene
```

## Dataset Metadata Coverage

| Metric / Filter | BDD100K | CULane (test_split) | CurveLanes | TuSimple |
|-----------------|:-------:|:-------------------:|:----------:|:--------:|
| C5 weather stratification | ✅ via `det_annotations_root` | ✅ via scenario list files (`test0_normal`, `test8_night`, …) | ❌ | ❌ |
| C5 time-of-day | ✅ `timeofday` field | ❌ | ❌ | ❌ |
| C5 scene type | ✅ `scene` field (highway / city street / …) | ❌ | ❌ | ❌ |
| Per-image GT mask | ✅ | ✅ (laneseg_label_w16 preferred) | ✅ (rasterised from polylines) | ✅ (rasterised from polylines) |
