"""
Combine tag data from dataset/_selected/*.jsonl with I-metric results from
pilot_output/*.csv into a single flat CSV, joined on image filename.

Join key: sample_id (from jsonl) == image_name/filename (from pilot csv, sans
the fact pilot filenames are exactly f"{sample_id}.jpg").
"""
import json
import glob
from collections import OrderedDict
import pandas as pd

SELECTED_DIR = "dataset/_selected"
PILOT_DIR = "pilot_output"
OUT_PATH = "pilot_output/combined_tags_metrics.csv"

# D-metrics are per-model (detection performance), unlike the GT-only I-metrics
# above, so both models' per-image CSVs are merged in, prefixed by model name.
D_METRICS_DIR = "outputs/d_metrics_full/d_metrics_v2"
D_METRICS_DATASETS = ["culane", "tusimple", "curvelanes", "bdd100k"]
D_METRICS_MODELS = ["yolopx", "clrernet"]
# Dropped: sample_id/dataset are already carried by the tags frame; image_path
# and model are redundant once the column prefix encodes the model.
D_METRICS_DROP_COLS = ["sample_id", "dataset", "image_path", "model"]

TAG_DIMENSIONS = [
    "Operational Scenario",
    "Roadway Context & Facility Type",
    "Roadway Surface Type",
    "Pavement Marking Type & Configuration",
    "Observed Marking Visibility",
    "Lighting & Weather",
]

# Fixed order for the per-tag binary columns: grouped by dimension, then the
# order each tag first appears across the _selected jsonl files.
TAG_DIMENSION_ORDER = [
    "Operational Scenario",
    "Roadway Context & Facility Type",
    "Roadway Surface Type",
    "Pavement Marking Type & Configuration",
    "Observed Marking Visibility",
    "Lighting & Weather",
]


def load_tags():
    rows = []
    tag_vocab = OrderedDict()  # dim -> OrderedDict(tag -> None), preserves first-seen order
    for fp in sorted(glob.glob(f"{SELECTED_DIR}/*.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sample_id = d["sample_id"]
                row = {
                    "sample_id": sample_id,
                    "image_name": f"{sample_id}.jpg",
                    "dataset": d.get("dataset"),
                    "split": d.get("split"),
                    "source_image_path": d.get("source_image_path"),
                    "source_gt_path": d.get("source_gt_path"),
                    "gt_type": d.get("gt_type"),
                    "natural_gt": d.get("natural_gt"),
                }
                by_dim = d.get("predicted_tags", {}).get("by_dimension", {})
                for dim in TAG_DIMENSIONS:
                    col = "tag_" + dim.lower().replace(" & ", "_").replace(
                        " ", "_"
                    ).replace("/", "_")
                    vals = by_dim.get(dim, [])
                    row[col] = "; ".join(vals)
                labels = d.get("predicted_tags", {}).get("labels", [])
                row["tag_labels_all"] = "; ".join(labels)
                row["_label_set"] = set(labels)
                for dim in TAG_DIMENSIONS:
                    bucket = tag_vocab.setdefault(dim, OrderedDict())
                    for v in by_dim.get(dim, []):
                        bucket.setdefault(v, None)
                rows.append(row)
    df = pd.DataFrame(rows)
    dupes = df["image_name"].duplicated().sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate image_name keys across _selected jsonl files")

    # One binary column per individual tag (0/1), grouped by dimension in
    # first-seen order, named exactly as the tag label itself.
    for dim in TAG_DIMENSION_ORDER:
        for tag in tag_vocab.get(dim, {}):
            if tag in df.columns:
                raise ValueError(f"Tag name collides with an existing column: {tag!r}")
            df[tag] = df["_label_set"].apply(lambda s, t=tag: int(t in s)).astype(int)

    df = df.drop(columns=["_label_set"])
    return df


def load_pilot_metric(csv_path, prefix):
    df = pd.read_csv(csv_path)
    # De-duplicate on image_name (keep first) just in case.
    df = df.drop_duplicates(subset="image_name", keep="first")
    rename = {}
    for col in df.columns:
        if col in ("dataset_name", "filename", "image_name"):
            continue
        rename[col] = col if col.startswith(prefix) else f"{prefix}_{col}"
    df = df.rename(columns=rename)
    df = df.drop(columns=["dataset_name", "filename"])
    return df


def load_d_metrics(model):
    """Concatenate one model's per-dataset d_metrics_per_image.csv files
    (evaluation/run_d_metrics.py output) into a single frame keyed by
    sample_id, with every metric column prefixed d_{model}_."""
    frames = []
    for ds in D_METRICS_DATASETS:
        path = f"{D_METRICS_DIR}/{model}_{ds}/d_metrics_per_image.csv"
        df = pd.read_csv(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    dupes = df["sample_id"].duplicated().sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate sample_id keys in {model} D-metrics")
    sample_ids = df["sample_id"]
    df = df.drop(columns=[c for c in D_METRICS_DROP_COLS if c in df.columns])
    df = df.rename(columns={c: f"d_{model}_{c}" for c in df.columns})
    df.insert(0, "sample_id", sample_ids)
    return df


def main():
    tags_df = load_tags()
    print(f"Loaded {len(tags_df)} tagged samples from {SELECTED_DIR}")

    combined = tags_df
    metric_files = {
        "i1": f"{PILOT_DIR}/i1_selected_results.csv",
        "i2": f"{PILOT_DIR}/i2_selected_results.csv",
        "i3": f"{PILOT_DIR}/i3_selected_results.csv",
        "i4": f"{PILOT_DIR}/i4_selected_results.csv",
        "i6": f"{PILOT_DIR}/i6_selected_results.csv",
    }
    for prefix, path in metric_files.items():
        metric_df = load_pilot_metric(path, prefix)
        before = len(combined)
        combined = combined.merge(metric_df, on="image_name", how="left")
        matched = combined[f"{prefix}_status"].notna().sum() if f"{prefix}_status" in combined else None
        print(f"Merged {path}: {len(metric_df)} rows -> matched {matched}/{before}")

    for model in D_METRICS_MODELS:
        d_df = load_d_metrics(model)
        before = len(combined)
        combined = combined.merge(d_df, on="sample_id", how="left")
        matched = combined[f"d_{model}_D1_detection"].notna().sum()
        print(f"Merged D-metrics ({model}): {len(d_df)} rows -> matched {matched}/{before}")

    combined.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows x {len(combined.columns)} cols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
