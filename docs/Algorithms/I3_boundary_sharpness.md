# I3 — Boundary Sharpness

**Function:** `compute_i3_boundary_sharpness(image_rgb, gt_mask, road_mask)`
**Output:** float in `[0, 1]`, higher = better; `None` if GT empty or no boundary.
**Question answered:** *Are the marking edges locally crisp (a strong intensity step) rather than merely present or blurred into the road?*

## Inputs

- `image_rgb` → grayscale float.
- `gt_mask` — binary marking mask.
- `road_mask` — optional, restricts the background ring.

## Algorithm

```
1. If gt_mask empty or zero area -> return None.
2. gray = to_gray_float(image_rgb); resize gt_mask to gray shape.
3. boundary       = morphological gradient of gt_mask (the 1-px mask outline)
4. boundary_band  = dilate(boundary, ellipse 5x5)      # a few-px band along the edge
5. ring           = local background ring (outer=21, inner=7, road_mask)
6. G  = Sobel gradient magnitude of gray (sqrt(gx^2 + gy^2))
7. boundary_med   = 75th percentile of G over boundary_band
8. background_med = 75th percentile of G over ring
                    (fallback: 50th percentile of G over whole image if ring empty)
9. I3 = clip( (boundary_med - background_med)
              / (boundary_med + background_med + eps) , 0, 1)
10. return I3
```

## Rationale

A raw edge-density (e.g. Canny count) around the GT boundary can be triggered by
asphalt texture, shadows, or noise. I3 instead asks whether the gradient *at the
marking edge* exceeds the gradient of *nearby road* — a contrast-normalised ratio,
so uniformly noisy or uniformly smooth images do not inflate the score.

## Edge cases

- Empty GT or no boundary pixels → `None`.
- Empty ring → falls back to a global gradient percentile.

## Caveats

- **Saturating ratio.** Because it is normalised as `(a-b)/(a+b)`, I3 compresses at
  the top end: a razor-sharp edge and a moderately sharp edge can both approach the
  same value, so I3 discriminates "edge present vs absent" better than "sharp vs
  slightly blurred."
- Inherits the ribbon caveat: the GT boundary is the rasterised ribbon edge, which
  sits on asphalt a few px away from the true paint edge, so the measured gradient is
  a lower bound on the real edge sharpness.
