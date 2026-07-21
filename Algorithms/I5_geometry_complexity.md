# I5 — Geometry Complexity

**Function:** `compute_i5_geometry_complexity(gt_mask, lanes)`
**Output:** float in `[0, 1]`, higher = more complex geometry; `None` if unfittable.
**Question answered:** *How curved / geometrically difficult is the lane layout?* This is a **context / difficulty** variable, not a quality penalty.
**Uses the image?** No — GT geometry only.

## Inputs

- `gt_mask` — binary marking mask (fallback path).
- `lanes` — GT polylines (preferred path).

## Algorithm

```
Preferred (polylines available):
  For each lane polyline with >= 4 points and vertical span >= 10 px:
      Fit x(y) = a*y^2 + b*y + c          (quadratic, x as function of y)
      proxy        = |a| * H^2 / W        # normalised curvature magnitude
      slope_change = |2a * span_y| / max(|b|, 1)
      score_i = clip(0.75*proxy + 0.25*slope_change, 0, 1)
  I5 = mean(score_i)

Fallback (mask only):
  Estimate a centerline per component by binning rows and averaging x;
  fit the same quadratic and compute the same proxy.
  Last-resort fallback: std of per-row mask centers / width.
```

## Rationale

Curvature is *difficulty*, not degradation — a sharp curve is not "bad
infrastructure." I5 is therefore kept out of the R1 readability score and used only
for stratification and the C4 geometry-sensitivity analysis (does detection degrade
on high-complexity geometry?).

## Edge cases

- Fewer than 4 usable points / near-vertical-only lanes → that lane skipped.
- No fittable lane and no usable centerline → `None`.

## Caveats

- **Not a readiness signal by itself** — never sum I5 into a quality score.
- On mostly-straight datasets (e.g. CULane) the quadratic term is tiny and I5 has
  near-zero variance (observed CULane I5 = 0.027 ± 0.004), so C4 has little signal to
  work with there; it is more useful on CurveLanes / BDD.
- The proxy is a normalised heuristic, not a physical radius of curvature; treat it
  as an ordinal difficulty ranking, not a metric curvature.
