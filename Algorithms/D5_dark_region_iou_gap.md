# D5 — Dark-Region IoU Gap

**Function:** `compute_d5_dark_region_iou_gap_single(pred_mask, gt_mask, image_rgb, min_gt_pixels=20)`
**Output:** float, typically `[-1, 1]`; `None` if a region lacks enough GT. Lower = better (0 = no gap).
**Question answered:** *Does detection degrade in dark regions (shadow / night) compared with normally-lit regions?*

## Inputs

- `pred_mask`, `gt_mask`, `image_rgb`.

## Algorithm

```
1. Align shapes; if GT empty -> return None.
2. gray = to_gray_float(image_rgb), resized to GT shape.
3. dark   = gray < min(100, 35th percentile of gray)     # adaptive dark threshold
   normal = NOT dark
4. Require >= 20 GT marking pixels in BOTH dark and normal regions, else None.
5. iou_dark   = region-restricted IoU over dark pixels
   iou_normal = region-restricted IoU over normal pixels
6. If either is None -> return None.
7. D5 = iou_normal - iou_dark
        (> 0 means detection is WORSE in dark regions)
```

## Rationale

A sensitivity/robustness diagnostic: rather than an absolute score, it measures the
*gap* between well-lit and dark performance on the same image, isolating the lighting
effect from the image's overall difficulty.

## Edge cases

- Fewer than 20 GT pixels in either region → `None` (can't compare fairly).
- Either region IoU undefined → `None`.

## Caveats

- **This is dark-region sensitivity, NOT true occlusion robustness.** It uses image
  intensity, not an object/occlusion mask. Naming it "occlusion robustness" would be
  incorrect unless real occlusion masks are supplied.
- The dark threshold is adaptive per image, so D5 values are only loosely comparable
  across images with very different global brightness.
- Diagnostic in R2 — not weighted into the detectability score.
