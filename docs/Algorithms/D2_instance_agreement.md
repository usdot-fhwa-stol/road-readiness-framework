# D2 — Instance Agreement

**Function:** `compute_d2_instance_agreement_single(pred_mask, gt_mask, gt_lanes, pred_lanes)`
**Output:** float in `[0, 1]`, higher = better; `None` if both sides have no markings.
**Question answered:** *Did the model recover the right number of marking instances?*

## Inputs

- `pred_mask`, `gt_mask` — predicted / GT masks (component grouping fallback).
- `gt_lanes`, `pred_lanes` — optional polylines (preferred count).

## Algorithm

```
1. N_gt   = len(valid gt_lanes)   if gt_lanes   else grouped instance count of gt_mask
2. N_pred = len(valid pred_lanes) if pred_lanes else grouped instance count of pred_mask
3. If N_gt == 0 and N_pred == 0 -> return None      # nothing to agree on
4. If N_gt == 0                 -> return 0.0        # predicted spurious instances
5. D2 = 1 - min( |N_pred - N_gt| / max(N_gt, 1), 1 )
```

## Rationale

Counts instances (grouped markings), not raw components, using the same
dashed-merge heuristic as I6 so a dashed lane is not counted as many instances.
A normalised absolute-difference score: exact match = 1, off-by-`N_gt` or worse = 0.

## Edge cases

- Both empty → `None` (not treated as a success).
- GT empty but prediction non-empty → 0.0 (false instances penalised).

## Caveats

- **Count-only.** D2 says nothing about *where* the instances are — a prediction can
  match the count while being spatially wrong (that is captured by D3/D4).
- Sensitive to the instance-grouping heuristic when polylines are absent.
- Diagnostic in R2 — not weighted into the detectability score.
