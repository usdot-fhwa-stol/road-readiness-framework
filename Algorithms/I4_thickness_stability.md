# I4 — Thickness Stability

**Function:** `compute_i4_thickness_stability(gt_mask, lanes, meta)`
**Output:** float in `[0, 1]`, higher = more stable width; `None` if GT empty.
**Question answered:** *Is the paint width consistent, or does it vary (worn / patchy / abnormal segments)?*
**Uses the image?** No — GT geometry only.

## Inputs

- `gt_mask` — binary marking mask. (`lanes`, `meta` accepted but unused.)

## Algorithm

```
1. For each connected component (above a min-area threshold):
        major = major-axis length (from PCA of the component pixels)
        thickness_i = area_i / major_i          # mean width along the segment
   Keep only finite, positive thickness values.
2. If no components -> return None.
3. mean = mean(thickness values)
   cv   = 0 if a single component else std(thickness) / (mean + eps)
4. I4 = 1 - clip(cv, 0, 1)
```

`I4_mean_thickness_px` and `I4_thickness_cv` are emitted as diagnostics.

## Rationale

The previous approach measured row-wise left/right span, which is *lane width*
(distance between two markings), not *paint width*. Approximating each segment's
width as `area / length` gives a per-segment thickness; a low coefficient of
variation across segments indicates uniform, well-maintained paint.

## Edge cases

- No qualifying components → `None`.
- Single component → CV defined as 0 → I4 = 1.0.

## Caveats

- **On rasterised polyline GT this is nearly a constant.** The mask stroke width is
  fixed by the rasteriser (16 px), so thickness variation reflects only anti-aliasing
  and endpoint effects. Observed CULane I4 = 0.965 ± 0.003 across every image — the
  metric is reading back the stroke width that was passed in, carrying almost no
  information about the actual road.
- **No perspective normalisation.** Real paint narrows with distance under
  perspective, so even a perfectly maintained road produces high thickness CV (and a
  lower I4) if measured in raw image pixels. A faithful version should normalise each
  segment's thickness by its image row (distance) before taking the CV.
- Only meaningful on datasets with true segmentation masks that preserve real paint
  width (e.g. BDD100K), and even then only after perspective normalisation.
