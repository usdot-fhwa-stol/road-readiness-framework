# D3 — IoU and Tolerant F1

**Functions:** `compute_d3_iou_single`, `compute_tolerant_precision_recall_f1`, `compute_d3_tolerant_f1_single`
**Outputs:** `D3_iou`, `D3_tolerant_precision_r5`, `D3_tolerant_recall_r5`, `D3_tolerant_f1_r5` — all `[0, 1]`, higher = better; `None` if both masks empty.
**Question answered:** *How well does the predicted marking mask overlap the GT — strictly (IoU) and with a small tolerance (F1)?*

## Inputs

- `pred_mask`, `gt_mask` — resized to a common shape before comparison.

## Algorithm — strict IoU

```
1. Align shapes (resize GT to pred, nearest).
2. If union == 0 -> return None.
3. IoU = TP / (TP + FP + FN + eps)
       = |pred AND gt| / |pred OR gt|
```

## Algorithm — tolerant precision / recall / F1 (5 px)

```
1. Align shapes. Handle empty cases:
      pred empty & gt empty -> all None (reason: no pixels)
      pred empty & gt nonempty -> precision None, recall 0, f1 0
      gt empty & pred nonempty -> precision 0, recall None, f1 0
2. dist_to_pred = distance transform of NOT pred    # px distance to nearest pred pixel
   dist_to_gt   = distance transform of NOT gt
3. recall    = fraction of GT pixels with dist_to_pred <= 5
   precision = fraction of pred pixels with dist_to_gt  <= 5
4. f1 = 2*precision*recall / (precision + recall + eps)
```

## Rationale

Lane masks are thin; a lateral shift of a few pixels can destroy IoU while the
detection is operationally fine. Tolerant F1 credits predictions that land within
5 px of GT, so it is the **preferred** overlap metric for lanes (and carries the
largest R2 weight, 0.35). Strict IoU is retained (weight 0.20) as a stricter
reference.

## Edge cases

- Both empty → `None` (excluded from aggregation, not counted as 0).
- One-sided empties → precision/recall set per the table above.

## Caveats

- Tolerance is fixed at 5 px; on very high-resolution images 5 px is a smaller
  physical tolerance than on low-res — report the value.
- IoU compares against the **rasterised GT ribbon**, whose width is an annotation
  choice; the pred-vs-GT stroke-width mismatch caps achievable IoU. Tolerant F1 is
  much less affected by this and is the metric to trust.
