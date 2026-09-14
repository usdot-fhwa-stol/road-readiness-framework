# D7 — Temporal Jitter (Reserved)

**Status:** reserved — **not computed** in single-frame evaluation.
**Output:** `None`, with reason `unavailable_single_frame_evaluation`.
**Question it would answer:** *How stable is the detection frame-to-frame — does the predicted lane jitter under a static/near-static scene?*

## Why it is not computed

Every dataset in the current pipeline (CULane, TuSimple, CurveLanes, BDD100K lane)
is evaluated as **independent single frames**. Temporal jitter requires temporally
aligned consecutive frames of the same scene, which the pipeline does not provide.
The record therefore always carries:

```
"D7": None,
"D7_temporal_jitter": None,
"D7_reason": "unavailable_single_frame_evaluation"
```

## Intended algorithm (if sequences were available)

```
1. Take a sequence of consecutive frames of the same scene.
2. For each frame, extract the predicted lane centerline(s).
3. Register frames (compensate ego-motion) so a static lane maps to itself.
4. D7 = temporal variance / jitter of the registered centerline positions
        (higher = more unstable prediction).
```

## Caveats

- Not part of R2 today; it is excluded from aggregation, not counted as 0.
- Enabling it requires video/sequence datasets plus ego-motion compensation; it is a
  future extension, documented here for completeness of the I/D metric set.
