# I1 — Pattern Continuity

**Function:** `compute_i1_pattern_continuity(image_rgb, gt_mask, lanes, meta, road_mask)`
**Output:** float in `[0, 1]`, higher = better; `None` if GT is empty.
**Question answered:** *Are the markings — solid or dashed — continuous and trackable the way a lane-following stack would expect?*

## Inputs

- `image_rgb` — source image (used for per-component visual quality).
- `gt_mask` — binary marking mask.
- `lanes` — optional GT polylines (used to estimate instance count).
- `meta` — optional metadata (used to read a declared solid/dashed style).
- `road_mask` — optional drivable-area mask (restricts the background ring).

## Algorithm

```
1. If gt_mask is empty -> return None.

2. pattern = infer_marking_pattern(gt_mask, lanes, meta)   # solid | dashed | mixed | unknown
      a. If meta declares a lane style -> use it.
      b. Else classify from mask geometry:
           - component_count, per-component elongation
           - dash geometry: PCA alignment, spacing coefficient-of-variation,
             duty cycle (median segment length / median spacing)
           - many aligned regular components  -> dashed
           - one/few long components          -> solid
           - many irregular components        -> mixed
           - otherwise                        -> unknown

3. quality = mean over components of _component_quality_score, where each
   component's quality is:
        0.42*contrast + 0.35*sharpness + 0.18*thickness + 0.05*area   (each 0..1)
   (contrast and sharpness here reuse the I2 / I3 formulas at component scale.)

4. Combine quality with a pattern-appropriate continuity term:

   DASHED:  regularity = dash regularity score
                       = 0.40*alignment + 0.35*spacing_regularity + 0.25*duty_cycle_score
            I1 = 0.58*quality + 0.42*regularity

   SOLID:   close gaps with a tall vertical kernel, then
            area_ratio   = area(mask) / area(closed_mask)          # 1.0 = no gaps bridged
            instance_term = clip(instance_count / component_count, 0, 1)
            continuity   = clip(0.70*area_ratio + 0.30*instance_term, 0, 1)
            I1 = 0.50*quality + 0.50*continuity

   MIXED / UNKNOWN:
            regularity = dash regularity if available,
                         else clip(instance_count / component_count, 0, 1)
            I1 = 0.60*quality + 0.40*regularity

5. return clip(I1, 0, 1)
```

## Rationale

The v2 design deliberately **does not** penalise dashed lanes for having gaps — a
dashed marking is discontinuous by design. The pattern is inferred first, then the
continuity term is chosen so that "expected" gaps (dashed) are scored on *regularity*
while unexpected gaps (solid) are scored on *how much morphology had to bridge*.

## Edge cases

- Empty GT → `None`.
- Dashed pattern with < 2 components and no regularity score → regularity defaults to 0.
- Solid closing kernel scales with image height/width.

## Caveats

- **Pattern branch rarely fires on rasterised datasets.** Because polyline GT is a
  continuous fixed-width ribbon, `infer_marking_pattern` returns `solid` ~100 % of
  the time on CULane / TuSimple / CurveLanes; the dashed logic only exercises on
  datasets with true segmentation masks (e.g. BDD100K).
- **Double-counting in R1.** The `quality` term already contains contrast and
  sharpness (the I2/I3 signals). Since I1, I2, I3 all feed R1, contrast's *effective*
  weight in R1 is higher than its nominal 0.35, and the C1 correlation (I1 vs
  detectability) partly restates C2/C3.
