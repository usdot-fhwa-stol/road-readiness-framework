"""Compute per-model R2 (reference-detectability) scores on the combined I+D CSV.

R2 is the detection-theoretic half of the readiness score: it summarizes how
reliably a model detects/localizes lane markings, as opposed to R1 (marking
readability), which is estimation-theoretic (how well the marking's own
geometry can be measured). Reuses the single canonical R2 definition in
evaluation/readiness_metrics.py (compute_r2_reference_detectability_score) --
no reimplementation.

Column mapping from pilot_output/combined_i_and_d_metrics.csv (built by
scripts/build_clean_combined_i_d.py) to the keys that function expects:
  D3_f1            <- {model}_f1
  D3_iou           <- {model}_iou
  D4_near_iou      <- {model}_D4_lower   (lower band = nearest to the vehicle,
                                           matching the legacy D4 near-field
                                           row band; see d_metrics.py NEAR_FIELD_*)
  D6_detected_ratio <- 1 - {model}_D6    (D6 is a *missed*-pixel ratio, i.e. a
                                           miss-probability estimate; R2 wants
                                           the detection-rate complement)
  D7_confidence_mean -> not present in this CSV (dropped upstream); the
                         weighted mean renormalizes over the remaining terms.
"""
import sys

import pandas as pd

sys.path.insert(0, ".")
from evaluation.readiness_metrics import compute_r2_reference_detectability_score  # noqa: E402

IN_PATH = "pilot_output/combined_i_and_d_metrics.csv"
OUT_PATH = "pilot_output/combined_i_and_d_metrics_with_r2.csv"
MODELS = ["yolopx", "clrernet"]


def r2_for_row(row, model):
    d6 = row.get(f"{model}_D6")
    metric_dict = {
        "D3_f1": row.get(f"{model}_f1"),
        "D3_iou": row.get(f"{model}_iou"),
        "D4_near_iou": row.get(f"{model}_D4_lower"),
        "D6_image_missed_ratio": d6,
    }
    return compute_r2_reference_detectability_score(metric_dict)


def main():
    df = pd.read_csv(IN_PATH)
    for model in MODELS:
        df[f"{model}_R2"] = df.apply(lambda row: r2_for_row(row, model), axis=1)
        n = df[f"{model}_R2"].notna().sum()
        print(f"{model}_R2: {n}/{len(df)} rows, mean={df[f'{model}_R2'].mean():.2f}")
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
