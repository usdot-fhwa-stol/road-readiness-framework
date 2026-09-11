# D4 — Near / Mid / Far Band Metrics

**Function:** `compute_d4_band_metrics_single(pred_mask, gt_mask, band, tolerance_px=5)`
**Outputs:** `D4_near_iou`, `D4_mid_iou`, `D4_far_iou`, plus `D4_near_f1_r5`, `D4_mid_f1_r5`, `D4_far_f1_r5` — each `[0, 1]`, higher = better; `None` if that band has no GT.
**Question answered:** *Does detection quality hold up with distance — near the vehicle vs at the horizon?*

## Inputs

- `pred_mask`, `gt_mask`.
- `band` ∈ {`near`, `mid`, `far`}.

## Algorithm

```
1. Split the image into vertical thirds:
        far  = top third      (slice 0            .. H/3)
        mid  = middle third   (slice H/3          .. 2H/3)
        near = bottom third   (slice 2H/3         .. H)
2. Crop pred and gt to the band rows.
3. If the band has no GT markings -> return {iou: None, f1: None, reason}.
4. iou = D3 strict IoU on the band crop.
   f1  = D3 tolerant F1 (5 px) on the band crop.
```

## Rationale

Perspective makes far markings tiny and low-contrast; aggregate IoU hides distance
effects. Splitting into near/mid/far exposes where a detector fails. `D4_near_f1_r5`
(or `D4_near_iou` when F1 is unavailable) feeds R2 with weight 0.20 — near-field
detection is the most safety-relevant band.

## Edge cases

- Band with no GT markings → `None` for that band (with a reason string), excluded
  from aggregation.

## Caveats

- Bands are fixed geometric thirds of the image, **not** true ground distance; the
  horizon is not detected, so "far" is an image-row proxy for distance.
- Far-band GT is the thinnest part of the rasterised ribbon, so far-band IoU is the
  most affected by stroke-width mismatch — prefer the tolerant-F1 variant there.
