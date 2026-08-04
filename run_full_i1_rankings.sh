#!/usr/bin/env bash
set -euo pipefail

# Re-run prediction-guided I1 rankings after changing the continuity algorithm.
# Existing model predictions are reused; no inference is performed here.
PYTHON="${PYTHON:-/home/gauravb/miniconda3/envs/rr_2/bin/python}"
ROOT="${ROOT:-outputs/full_model_rankings}"
WORKERS="${WORKERS:-16}"

cd "$(dirname "${BASH_SOURCE[0]}")"

"${PYTHON}" -m evaluation.rank_full_i1_i4 \
  --root "${ROOT}" \
  --workers "${WORKERS}" \
  --datasets bdd100k_lane culane curvelanes tusimple

# Render only the I1 top-50 and worst-score panels.  I4 panels are not touched.
"${PYTHON}" - <<'PY'
from pathlib import Path
import json, cv2, numpy as np
root = Path("outputs/full_model_rankings")
for ranking in root.glob("rankings/*/*/I1_*.json"):
    dataset, model = ranking.parts[-3], ranking.parts[-2]
    kind = ranking.stem.removeprefix("I1_")
    rows = json.loads(ranking.read_text())
    out = root / "images" / "I1" / dataset / model / kind
    out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, 1):
        raw = cv2.imread(row["image_path"])
        if raw is None:
            continue
        pred = cv2.imread(row.get("pred_path", ""), cv2.IMREAD_GRAYSCALE)
        panel = raw.copy()
        if pred is not None:
            if pred.shape != panel.shape[:2]:
                pred = cv2.resize(pred, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
            active = pred > 0
            panel[active] = (0.55 * panel[active] + 0.45 * np.array([60, 210, 90])).astype(np.uint8)
        score = row.get("I1")
        label = "I1 unavailable" if score is None else f"I1: {float(score):.3f}"
        for image, title in ((raw, "RAW"), (panel, f"{model.upper()} PRED  {label}")):
            cv2.rectangle(image, (0, 0), (image.shape[1], 52), (20, 20, 20), -1)
            cv2.putText(image, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{i:02d}_{row['sample_id']}.jpg"), cv2.hconcat([raw, panel]), [cv2.IMWRITE_JPEG_QUALITY, 92])
PY

echo "I1 rankings and images written under ${ROOT}/rankings and ${ROOT}/images/I1"
