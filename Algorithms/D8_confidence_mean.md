# D8 — Confidence Mean

**Function:** `compute_d8_confidence_mean_single(pred_mask, lane_prob)`
**Output:** float in `[0, 1]`, higher = more confident; `None` if no probability map.
**Question answered:** *How confident is the model, on average, over the pixels it predicted as marking?*

## Inputs

- `pred_mask` — predicted binary mask.
- `lane_prob` — per-pixel lane probability map from the model (optional).

## Algorithm

```
1. If lane_prob is None or empty -> return None.
2. Resize pred_mask to prob shape; clip prob to [0, 1].
3. If no predicted pixels -> return 0.0.
4. D8 = mean( lane_prob[ pred_mask > 0 ] )
```

### How `lane_prob` is derived (YOLOPX)

```
two-channel ll_seg_out : softmax over channels, channel 1 = lane probability
one-channel ll_seg_out : sigmoid
ambiguous layout       : probability unavailable -> D8 = None
```

## Rationale

Reports the model's own confidence where it committed to a detection. The earlier
placeholder value of 0.5 was removed — D8 is now either a real measurement or `None`.

## Edge cases

- No probability map → `None` (ignored in R2, weights renormalised).
- Probability map present but empty prediction → 0.0.

## Caveats

- **Availability is model-dependent.** Only present when the reference model exposes
  usable lane logits; otherwise `None`. Do not compare D8 across models with
  different confidence calibration.
- Smallest R2 weight (0.05) by design — confidence is weakly indicative of
  correctness and must not dominate the detectability score.
