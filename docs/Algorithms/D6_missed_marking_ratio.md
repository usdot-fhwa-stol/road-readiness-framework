# D6 — Missed Marking Ratio

**Function:** `compute_d6_missed_marking_ratio_single(pred_mask, gt_mask)`
**Output:** float in `[0, 1]`, **lower = better**; `None` if GT empty.
**Question answered:** *What fraction of the GT marking pixels did the model miss?*

## Inputs

- `pred_mask`, `gt_mask` — aligned to a common shape.

## Algorithm

```
1. Align shapes. If GT area == 0 -> return None.
2. FN = pixels where gt == 1 AND pred == 0     # false negatives
3. D6 = FN / (GT_area + eps)  =  1 - pixel_recall
```

## Rationale

A direct, interpretable miss rate — the complement of pixel recall. In R2 it is
folded in as a *detected* ratio, `1 - D6`, with weight 0.20, so that "more missed"
lowers detectability.

## Edge cases

- Empty GT → `None`.
- Empty prediction, non-empty GT → D6 = 1.0 (everything missed).

## Caveats

- **Strict pixel overlap** (no tolerance): a prediction shifted a few px off the
  rasterised ribbon counts those GT pixels as missed even if operationally detected.
  D6 therefore reads pessimistically against thin/rasterised GT; cross-check with the
  tolerant recall in D3.
- Pixel-level, not instance-level: partially detecting every lane can give the same
  D6 as fully missing one lane.
