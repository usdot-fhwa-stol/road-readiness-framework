# I2 — Local Contrast

**Function:** `compute_i2_local_contrast(image_rgb, gt_mask, road_mask, outer_kernel=21, inner_kernel=5)`
**Output:** float in `[0, 1]`, higher = better; `None` if GT empty or no usable background.
**Question answered:** *How much brighter/darker is the paint than the road immediately next to it — the contrast a detector actually sees?*

## Inputs

- `image_rgb` → grayscale float.
- `gt_mask` — binary marking mask (defines pixel set `M`).
- `road_mask` — optional drivable mask, restricts the background ring to road.

## Algorithm

```
1. If gt_mask empty or has zero area -> return None.
2. gray = to_gray_float(image_rgb)
3. Resize gt_mask to gray shape (nearest).
4. M   = marking pixels (gt_mask > 0)
5. ring = local background ring around M:
        ring = dilate(M, outer=21) AND NOT dilate(M, inner=5) AND NOT M
        if road_mask: ring = ring AND road_mask
6. Fallback if ring empty:
        ring = lower 60% of the image, minus M, (minus non-road if road_mask given)
7. If ring still empty or M empty -> return None.
8. I2 = clip( | mean(gray[M]) - mean(gray[ring]) | / 255 , 0, 1)
9. return I2
```

## Rationale

The earlier version compared markings against *all* non-marking pixels (sky,
vehicles, buildings), which is not the contrast a lane detector operates on. The
local ring measures paint-vs-adjacent-asphalt only — a physically meaningful
readability signal.

## Edge cases

- Empty GT → `None`.
- No ring and no road-restricted fallback → `None`.

## Caveats

- **Dominated by the rasterisation artifact.** On the polyline datasets the "marking
  mask" is a 16 px ribbon that includes substantial asphalt on either side of the
  actual paint. `mean(gray[M])` therefore averages paint *with* road, driving I2
  toward 0 (observed CULane mean ≈ 0.008 — a ~2 gray-level difference, versus 100+
  for real white paint on asphalt). To make I2 measure the road rather than the
  annotator's stroke width, the paint footprint must first be segmented *inside* the
  ribbon (e.g. intensity threshold within the band) and I2 computed on that.
- Because I2 has weight 0.35 in R1 but structurally caps low, R1 is compressed and
  the R4 `HIGH` threshold (R1 ≥ 80) is effectively unreachable on these datasets.
