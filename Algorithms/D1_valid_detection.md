# D1 — Valid Detection

**Function:** `compute_d1_valid_detection_single(pred_mask, gt_mask, tolerance_px=5, min_recall=0.30, min_precision=0.30, min_pred_area=20)`
**Output:** `0` or `1` per image; aggregate = mean over images. Higher = better.
**Question answered:** *Did the model produce a valid detection of the marking at all (a coarse pass/fail gate)?*

## Inputs

- `pred_mask` — predicted binary marking mask.
- `gt_mask` — GT marking mask.

## Algorithm

```
1. t = compute_tolerant_precision_recall_f1(pred_mask, gt_mask, tolerance=5)
       (distance-transform based; see D3)
2. If tolerant precision or recall is None -> return 0.
3. Return 1 iff ALL hold, else 0:
        pred_area >= 20
        recall    >= 0.30
        precision >= 0.30
```

## Rationale

A binary "is this detection usable" gate. The thresholds are permissive because the
intent is to separate genuine detections from near-empty / spurious ones, not to
grade quality (that is D3's job).

## Edge cases

- No predicted or GT pixels → precision/recall `None` → D1 = 0.
- Very small predictions (< 20 px) are treated as non-detections regardless of overlap.

## Caveats

- **Threshold-dependent.** The 0.30 / 0.30 / 20 px cutoffs are fixed defaults; the
  aggregate mean is sensitive to them. Report them alongside the number.
- Diagnostic in R2 — intentionally not weighted so a permissive gate cannot inflate
  the detectability score.
