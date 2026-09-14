"""R2 (reference-detectability) overlay panels.

Rather than re-deriving pixel masks, this reuses the D3 TP/FN/FP panel that
evaluation/render_d_metric_panels.py already rendered for every sample (the
same TP/FN/FP coding underlies D4/D6, which feed R2 alongside D3 itself) and
prepends a header with the R2 formula breakdown and final score. Writes
<panel_dir>/<sid>__R2.jpg alongside the existing D1-D6 panels.

Formula (evaluation/readiness_metrics.py:compute_r2_reference_detectability_score,
weights renormalized because D7 confidence isn't available for cached
predictions):
    R2 = 100 * (0.35*f1 + 0.20*iou + 0.20*D4_near_iou + 0.20*(1-D6)) / 0.95
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluation.render_d_metric_panels import _fmt, _save

CSV_PATH = "pilot_output/combined_i_and_d_metrics_with_r2.csv"
MODELS = ["yolopx", "clrernet"]
_HEADER_COLOR = (255, 220, 120)  # BGR


def _val(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(xf) else xf


def _add_header(img: np.ndarray, lines: list[str]) -> np.ndarray:
    line_h = 26
    header_h = 14 + line_h * len(lines)
    header = np.zeros((header_h, img.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        color = _HEADER_COLOR if i == 0 else (200, 200, 200)
        cv2.putText(header, line, (10, 28 + line_h * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 1, cv2.LINE_AA)
    return np.vstack([header, img])


def render_r2_panel(panel_dir: Path, sample_id: str, model: str, row) -> bool:
    d3_path = panel_dir / f"{sample_id}__D3.jpg"
    img = cv2.imread(str(d3_path))
    if img is None:
        return False

    f1 = _val(row.get(f"{model}_f1"))
    iou = _val(row.get(f"{model}_iou"))
    d4 = _val(row.get(f"{model}_D4_lower"))
    d6 = _val(row.get(f"{model}_D6"))
    detected = None if d6 is None else 1.0 - d6
    r2 = _val(row.get(f"{model}_R2"))

    lines = [
        f"R2 Reference-Detectability Score ({model}) = {_fmt(r2)}",
        f"0.35*f1({_fmt(f1)}) + 0.20*iou({_fmt(iou)}) + 0.20*D4_near({_fmt(d4)}) "
        f"+ 0.20*(1-D6)({_fmt(detected)})  [renorm /0.95, D7 confidence unavailable]",
    ]
    out = _add_header(img, lines)
    _save(panel_dir / f"{sample_id}__R2.jpg", out)
    return True


def main():
    df = pd.read_csv(CSV_PATH)
    for model in MODELS:
        sub = df[df[f"{model}_R2"].notna()]
        written = 0
        for i, (_, row) in enumerate(sub.iterrows()):
            panel_dir = Path(row[f"{model}_panel_dir"])
            if render_r2_panel(panel_dir, row["sample_id"], model, row):
                written += 1
            if (i + 1) % 2000 == 0:
                print(f"{model}: {i + 1}/{len(sub)} processed")
        print(f"{model}: wrote {written}/{len(sub)} R2 panels")


if __name__ == "__main__":
    main()
