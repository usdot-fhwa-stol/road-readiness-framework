#!/usr/bin/env python3
"""Build universal-manifest-format GT manifests for the human-tagged subset.

Only CurveLane and CULane are usable (see categorized_manifest/README.md):
TuSimple's tagged rows are unresolvable (bare numeric ids, no clip mapping)
and BDD100K's tagged rows are test-split with no lane GT in this dataset
copy.

CurveLane: the 40 unique tagged images are from the *train* split, which the
existing full model-comparison run never covered (that run only processed
`valid`). No cached predictions exist for these images, so this script
builds a small GT manifest for them via the existing CurveLanesAdapter (no
new GT-parsing logic) -- inference still needs to be run separately.

CULane: the 46 tagged rows each name a whole ~180-frame video segment; only
the single human-tagged `representative_frame` per segment is evaluated
here. Those frames are already part of the existing full CULane `test`
manifest/prediction run, so this script filters the *existing* standard
manifest down to just those 46 samples -- no new GT parsing, no new
inference required.

Each output sample's `meta` carries the human `tags` (and `predicted_tags`
when present) so they can be used for stratification later.

Usage:
    python3 categorized_manifest/build_tagged_subset_manifests.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"

CURVELANES_ROOT = "/shared/data/Curvelanes"
STANDARD_CULANE_MANIFEST = REPO_ROOT / "manifests" / "culane" / "manifest_culane.json"


def _dedup_by_file_name(samples: list[dict]) -> list[dict]:
    seen = {}
    for s in samples:
        seen.setdefault(s["file_name"], s)  # first occurrence wins
    return list(seen.values())


def build_curvelane_subset_manifest() -> Path:
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lane_eval.datasets.curvelanes import CurveLanesAdapter
    from lane_eval.manifest.generator import sample_to_entry

    tagged = json.loads((OUT / "manifest_curvelane.tagged.json").read_text())
    unique = _dedup_by_file_name(tagged["samples"])
    stems = [Path(s["file_name"]).stem for s in unique]
    tags_by_stem = {Path(s["file_name"]).stem: s for s in unique}

    stems_file = OUT / "_curvelane_tagged_stems.txt"
    stems_file.write_text("\n".join(stems) + "\n")

    adapter = CurveLanesAdapter(root=CURVELANES_ROOT, split="train", list_file=str(stems_file))
    mask_dir = OUT / "masks" / "curvelane_tagged"
    entries = []
    for sample in adapter.iter_samples():
        entry = sample_to_entry(sample, mask_dir, step=10, save_masks=True)
        tag_row = tags_by_stem.get(sample.image_id, {})
        entry["meta"] = {
            "sample_id": tag_row.get("sample_id"),
            "tags": tag_row.get("tags"),
            "predicted_tags": tag_row.get("predicted_tags"),
        }
        entries.append(entry)

    manifest = {
        "metadata": {"dataset": "curvelanes", "split": "train", "num_samples": len(entries),
                     "note": "human-tagged subset (categorized_manifest); train split, not in the "
                             "existing full valid-split model-comparison run"},
        "samples": entries,
    }
    out_path = OUT / "universal_manifest_curvelane_tagged.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"curvelane tagged subset: {len(entries)}/{len(unique)} resolved -> {out_path}")
    return out_path


def build_culane_subset_manifest() -> Path:
    tagged = json.loads((OUT / "manifest_culane.tagged.json").read_text())
    standard = json.loads(STANDARD_CULANE_MANIFEST.read_text())
    by_sample_id = {s["sample_id"]: s for s in standard["samples"]}

    entries = []
    missing = []
    for row in tagged["samples"]:
        if not row.get("gt_found") or not row.get("representative_frame"):
            continue
        driver = row["driver"]
        segment = Path(row["image_path"]).name  # "<seg>.MP4"
        frame_stem = Path(row["representative_frame"]).stem  # "00000"
        sample_id = f"{driver}_{segment}_{frame_stem}"
        std_entry = by_sample_id.get(sample_id)
        if std_entry is None:
            missing.append(sample_id)
            continue
        entry = dict(std_entry)  # shallow copy; reuses the exact existing GT (mask_path/lane_json)
        entry["meta"] = {
            "sample_id": row.get("sample_id"),
            "tags": row.get("tags"),
            "predicted_tags": row.get("predicted_tags"),
        }
        entries.append(entry)

    manifest = {
        "metadata": {"dataset": "culane", "split": "test", "num_samples": len(entries),
                     "note": "human-tagged subset (categorized_manifest); one representative frame "
                             "per tagged segment, reusing the existing standard test-split GT"},
        "samples": entries,
    }
    out_path = OUT / "universal_manifest_culane_tagged.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"culane tagged subset: {len(entries)} resolved, {len(missing)} missing -> {out_path}")
    if missing:
        print("  missing sample_ids:", missing)
    return out_path, [e["sample_id"] for e in entries]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_curvelane_subset_manifest()
    _, culane_sample_ids = build_culane_subset_manifest()
    (OUT / "_culane_tagged_sample_ids.json").write_text(json.dumps(culane_sample_ids, indent=2))


if __name__ == "__main__":
    main()
