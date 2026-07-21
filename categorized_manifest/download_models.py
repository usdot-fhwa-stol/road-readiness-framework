#!/usr/bin/env python3
"""Sequentially download the VLM bake-off models with auto-retry/resume.

The HF link on this box is slow/flaky; parallel downloads starve each other, so
this pulls one model at a time and retries on stalls. Resumable — re-run any
time and it continues from the partial cache in ~/.cache/huggingface.

  python3 download_models.py
"""
import time
from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen3-VL-8B-Instruct", "QWEN8B"),
    ("OpenGVLab/InternVL3-8B-hf", "INTERNVLHF"),   # -hf = native transformers class
    ("QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ", "QWEN30B"),  # 4-bit, fits 24GB
]

for repo, tag in MODELS:
    for attempt in range(60):
        try:
            snapshot_download(repo, max_workers=4)
            print(f"DONE {tag}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"RETRY {tag} {attempt}: {str(e)[:80]}", flush=True)
            time.sleep(8)
    else:
        print(f"GAVEUP {tag}", flush=True)
print("ALL_DONE", flush=True)
