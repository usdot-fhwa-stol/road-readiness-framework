# I6 — Marking Instances

**Function:** `compute_i6_marking_instances(gt_mask, lanes, meta)`
**Output:** dict — `I6_marking_instance_count` (int), `I6_raw_component_count` (int), `I6_topology_complexity` (float or None).
**Question answered:** *How many distinct marking instances are present, and how fragmented is the topology?* **Context / gating** variable, not a quality score.
**Uses the image?** No — GT geometry only.

## Inputs

- `gt_mask` — binary marking mask.
- `lanes` — GT polylines (preferred count). (`meta` accepted but unused.)

## Algorithm

```
1. raw_count = number of connected components above the min-area threshold.
2. instance_count = _estimate_marking_instances(gt_mask, lanes):
        if lanes present -> len(valid lanes)
        else:
            if raw_count <= 1 -> raw_count
            else: close with a tall, thin vertical kernel so dashed segments on the
                  same marking merge, then count grouped components.
                  instance_count = clamp(1 .. raw_count)
3. topology_complexity =
        None if raw_count == 0
        else clip(raw_count / max(instance_count,1), 1, 10) / 10
   (higher = one marking is fragmented into many components, i.e. dashed / worn)
4. return {instance_count, raw_count, topology_complexity}
```

## Rationale

Raw connected-component count is *not* a count of lane markings — a single dashed
lane is many components. I6 separates the two ideas: `instance_count` (how many
markings) and `topology_complexity` (how fragmented each marking is). More or fewer
markings is not inherently better infrastructure, so I6 never enters R1.

## Edge cases

- Empty GT → instance_count 0, topology `None`.
- Vertical closing kernel is what makes dashed segments count as one instance in the
  absence of polylines; its size scales with image dimensions.

## Caveats

- **Only as good as the instance grouping.** Without polylines, the dashed-merge
  heuristic can under- or over-merge depending on dash spacing and lane angle.
- On polyline datasets `instance_count = len(lanes)` exactly and is reliable; the
  topology signal is then trivial (raw ≈ instances after rasterisation).
- Use for stratification / gating (e.g. skip images with 0 instances), not scoring.
