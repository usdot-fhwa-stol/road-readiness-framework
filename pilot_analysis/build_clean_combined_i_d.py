"""
Build a clean, minimal combined tags+I+D metrics CSV: one row per image
(dataset_name, image_name, sample_id), all one-hot tag columns, I1-I6 from
i_metrics_summary.csv, and per-model core metrics (iou/f1/precision/recall) +
D2/D4/D5/D6 from the evaluation/run_d_metrics.py per-image outputs
(outputs/d_metrics_full/d_metrics_v2).

Every non-identifier column is either binary (0/1) or a real number:
  - tags: already one-hot 0/1 (from build_combined_tags_metrics_csv.load_tags)
  - D5 (categorical visibility group) is one-hot encoded here into
    <model>_D5_<group> columns instead of carried as a string.
  - everything else is already numeric.

Join key: sample_id == i_metrics_summary.csv's image_name minus ".jpg"
         == d_metrics per-image CSV's sample_id (both already share this id)
         == build_combined_tags_metrics_csv.load_tags()'s sample_id.

Column naming: <model>_<metric>, e.g. yolopx_iou, yolopx_f1, yolopx_D2,
yolopx_D4_upper, yolopx_D5_clear, yolopx_D6. D1/D3(raw)/D7 and all D3'/stroke
bookkeeping fields are dropped -- core iou/f1/precision/recall (sourced from
D3_*) already cover the "D3 as core metric" role.

Also adds <model>_panel_dir: the directory holding that image's rendered
D-metric visualization panels (evaluation/render_d_metric_panels.py output,
via `run_d_metrics.py --render-panels`). Each directory holds 8 files per
sample_id: <sample_id>__RAW.jpg, __PRED.jpg, __D1.jpg .. __D6.jpg. These are
path strings, not metrics -- excluded from the binary/numeric check below,
same as the other identifier columns.
"""
import sys

import pandas as pd

sys.path.insert(0, "scripts")
from build_combined_tags_metrics_csv import load_tags  # noqa: E402

I_METRICS_PATH = "pilot_analysis/i_metrics_summary.csv"
D_METRICS_DIR = "outputs/d_metrics_full/d_metrics_v2"
DATASETS = ["culane", "tusimple", "curvelanes", "bdd100k"]
MODELS = ["yolopx", "clrernet"]
OUT_PATH = "pilot_output/combined_i_and_d_metrics.csv"

# Non-tag metadata columns load_tags() also produces (paths, free-text tag
# strings, etc.) -- excluded here since they're neither binary nor numeric.
TAGS_NON_BINARY_COLS = {
    "sample_id", "image_name", "dataset", "split", "source_image_path",
    "source_gt_path", "gt_type", "natural_gt",
    "tag_operational_scenario", "tag_roadway_context_facility_type",
    "tag_roadway_surface_type", "tag_pavement_marking_type_configuration",
    "tag_observed_marking_visibility", "tag_lighting_weather",
    "tag_labels_all",
}

# core (D3-derived) + D2/D4/D6, renamed <model>_<name>. D5 is handled
# separately (one-hot encoded) since it's categorical, not numeric.
D_COLUMN_MAP = {
    "D3_iou": "iou",
    "D3_f1": "f1",
    "D3_precision": "precision",
    "D3_recall": "recall",
    "D2_count_match": "D2",
    "D4p_upper_f1": "D4_upper",
    "D4p_middle_f1": "D4_middle",
    "D4p_lower_f1": "D4_lower",
    "D6_image_missed_ratio": "D6",
}
D5_GROUPS = ["clear", "degraded", "occluded", "not_applicable"]
# Panel filename pattern within <model>_panel_dir: "<sample_id>__<TYPE>.jpg"
# for TYPE in RAW, PRED, D1, D2, D3, D4, D5, D6.
PANEL_TYPES = ["RAW", "PRED", "D1", "D2", "D3", "D4", "D5", "D6"]


def load_binary_tags():
    df = load_tags()
    tag_cols = [c for c in df.columns if c not in TAGS_NON_BINARY_COLS]
    return df[["sample_id"] + tag_cols]


def one_hot_d5(df, model):
    """Replace the categorical <model>_D5 column with one binary column per
    visibility group: <model>_D5_clear, <model>_D5_degraded, etc."""
    col = f"{model}_D5"
    norm = df[col].fillna("not_applicable").str.lower()
    for group in D5_GROUPS:
        df[f"{model}_D5_{group}"] = (norm == group).astype(int)
    return df.drop(columns=[col])


def load_i_metrics():
    df = pd.read_csv(I_METRICS_PATH)
    df["sample_id"] = df["image_name"].str.replace(r"\.jpg$", "", regex=True)
    dupes = df["sample_id"].duplicated().sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate sample_id keys in {I_METRICS_PATH}")
    return df[["sample_id", "dataset_name", "image_name", "I1", "I2", "I3", "I4", "I5", "I6"]]


def load_d_metrics(model):
    frames = []
    for ds in DATASETS:
        path = f"{D_METRICS_DIR}/{model}_{ds}/d_metrics_per_image.csv"
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    dupes = df["sample_id"].duplicated().sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate sample_id keys in {model} D-metrics")
    keep = ["sample_id", "dataset", "D5_visibility_group"] + list(D_COLUMN_MAP.keys())
    rename = {k: f"{model}_{v}" for k, v in D_COLUMN_MAP.items()}
    rename["D5_visibility_group"] = f"{model}_D5"
    df = df[keep].rename(columns=rename)
    df[f"{model}_panel_dir"] = df["dataset"].apply(
        lambda ds: f"{D_METRICS_DIR}/{model}_{ds}/panels"
    )
    return df.drop(columns=["dataset"])


def main():
    combined = load_i_metrics()
    print(f"Loaded {len(combined)} rows from {I_METRICS_PATH}")

    tags_df = load_binary_tags()
    before = len(combined)
    combined = combined.merge(tags_df, on="sample_id", how="left")
    print(f"Merged tags: {len(tags_df)} rows, {len(tags_df.columns) - 1} tag columns -> matched "
          f"{combined[tags_df.columns[1]].notna().sum()}/{before}")

    for model in MODELS:
        d_df = load_d_metrics(model)
        before = len(combined)
        combined = combined.merge(d_df, on="sample_id", how="left")
        matched = combined[f"{model}_iou"].notna().sum()
        print(f"Merged D-metrics ({model}): {len(d_df)} rows -> matched {matched}/{before}")
        combined = one_hot_d5(combined, model)

    id_cols = ["dataset_name", "image_name", "sample_id"]
    path_cols = [f"{m}_panel_dir" for m in MODELS]
    ordered = (
        id_cols
        + ["I1", "I2", "I3", "I4", "I5", "I6"]
        + [c for c in tags_df.columns if c != "sample_id"]
        + [f"{m}_{v}" for m in MODELS for v in D_COLUMN_MAP.values()]
        + [f"{m}_D5_{g}" for m in MODELS for g in D5_GROUPS]
        + path_cols
    )
    combined = combined[ordered]

    non_numeric = [
        c for c in combined.columns
        if c not in id_cols and c not in path_cols and not pd.api.types.is_numeric_dtype(combined[c])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric metric/tag columns found: {non_numeric}")

    combined.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows x {len(combined.columns)} cols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
