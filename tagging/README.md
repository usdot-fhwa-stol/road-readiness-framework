# Categorized-dataset manifests

Rebuilds full image + ground-truth locations (and carries the categorization
tags) from `TO_25-203_All_Datasets_Categorized.xlsx`, whose "File Name" column
only stores a bare name — the full dataset-relative path was never saved.

Scope: **TuSimple, BDD100K, CurveLane, CuLane** (the Waymo / Mapillary sheets
are ignored).

## Run

```bash
python3 tagging/build_categorized_manifest.py   # needs openpyxl
```

## Outputs (`output/`)

- `manifest_<dataset>.json` — one per dataset. Each sample has `sample_id`,
  `file_name` (raw from sheet), resolved `image_path` + `ground_truth_path`,
  `split`, `image_found`/`gt_found`, and `tags` (both the 6 summary-dimension
  strings and the flat list of active one-hot labels).
- `resolution_status.csv` — every row across all 4 sheets with found/missing
  flags, resolved paths, label count, and a note. This is the "what's found vs
  not" report.

## How each dataset's name was resolved

| Dataset   | Sheet name form            | Image location                         | Ground truth                                             |
|-----------|----------------------------|----------------------------------------|---------------------------------------------------------|
| BDD100K   | `<id>.jpg`                 | `bdd100k/images/{val,train,test}/`     | `bdd100k/ll_seg_annotations/{val,train}/<id>.png`       |
| CuLane    | `<seg>.mp4`                | `culane/driver_*/<seg>.MP4/` (a dir)   | per-frame `<seg>.MP4/*.lines.txt` + `laneseg_label_w16{,_test}/…` |
| CurveLane | `<hash>.jpg`               | `Curvelanes/{valid,train}/images/`     | `Curvelanes/{valid,train}/labels/<hash>.lines.json`     |
| TuSimple  | bare number (e.g. `64500`) | — (unresolvable)                       | —                                                       |

## Auto-tagging (predict the tags with a model)

`tag_images.py` predicts the 43 tags for each image with a zero-shot **SigLIP2**
model (`google/siglip2-base-patch16-224`) and writes an *updated* manifest: every sample keeps its `image_path` /
`ground_truth_path` and gains a `predicted_tags` block, so you can sort/group by
tag. Human `tags` (when present) are preserved for evaluation.

```bash
# taxonomy.py holds the 6 dimensions -> 43 tags -> prompts + per-dimension config
python3 tag_images.py --in-manifest  output/manifest_bdd100k.json \
                      --out-manifest output/manifest_bdd100k.tagged.json
python3 eval_tagger.py output/manifest_bdd100k.tagged.json   # scores vs human tags
```

`predicted_tags` shape (mirrors the human `tags`, so grouping code is identical):
```json
"predicted_tags": {
  "by_dimension": {"Lighting & Weather": ["Daylight","Glare"], ...},
  "labels": ["General Roadway Segments", "Daylight", "Glare", ...],
  "scores": {"Daylight": 0.14, ...}   // within-dimension softmax confidence
}
```

### Accuracy (micro F1 vs the human labels, ~145 images)

| Dimension | BDD | CurveLane | CuLane |
|---|---|---|---|
| Lighting & Weather | 0.47 | 0.49 | **0.59** |
| Roadway Surface Type | 0.33 | 0.12 | **0.74** |
| Roadway Context & Facility | 0.33 | 0.27 | 0.41 |
| Operational Scenario | 0.15 | 0.36 | 0.15 |
| Observed Marking Visibility | 0.12 | 0.16 | 0.13 |
| Pavement Marking Config | 0.11 | 0.09 | 0.03 |

**Takeaway:** SigLIP2 zero-shot is reliable on *visual* dimensions
(Lighting & Weather, and Surface/Context) but weak on the *fine semantic* ones
(Operational Scenario, Pavement Marking, Marking Visibility) — its
within-dimension scores there are nearly flat (~uniform), so top-1 is close to
noise. Those dimensions need a generative VLM (e.g. Qwen2.5-VL / InternVL, or a
vision API), which the tagger is structured to swap in per dimension. Thresholds
in `taxonomy.DIMENSION_CONFIG` trade precision vs recall (lower = more tags).

## Findings (what could NOT be found)

- **TuSimple — 14/14 unresolved.** The sheet stores bare numeric ids (`985`,
  `64500`, …). TuSimple images live at `clips/<date>/<timestamp>/<frame>.jpg`;
  there is no numeric id in that layout and no id→clip mapping was saved. The
  numbers also don't match any file on disk. **Needs the original mapping from
  whoever filled the sheet.**
- **BDD100K — 50 images found, 0 ground truth.** Every sampled image is from the
  **test** split, and `ll_seg_annotations/` only ships `train/` and `val/`
  lane masks — BDD test has no public lane GT. Images are located; masks don't
  exist in this dataset copy.
- **CuLane — 46/48 found.** Two segments are absent from this CuLane copy
  entirely (not under any `driver_*` dir, not referenced in `list/`):
  `05252322_0559.mp4`, `05252246_0542.mp4`. Note: CuLane rows name a whole
  ~360-frame video segment, not a single frame — `image_path` is the segment
  directory and GT is per-frame.
- **CurveLane — 49/49 found** (images fall in `train/` and `valid/`; `test/`
  has no labels but none of the sheet's rows landed there).
