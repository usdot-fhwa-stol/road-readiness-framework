#!/usr/bin/env python3
"""Sequentially download the VLM bake-off models with auto-retry/resume.

The HF link on this box is slow/flaky; parallel downloads starve each other, so
this pulls one model at a time and retries on stalls. Resumable — re-run any
time and it continues from the partial cache in ~/.cache/huggingface.

  python3 download_models.py
"""
import os
# Google's Gemma 3 repos are Xet-backed. huggingface_hub 1.23.0 without hf_xet
# throws "Unable to parse string as hex hash value" on them -> force the classic
# HTTPS/LFS download path instead of Xet. Must be set before hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import time
from huggingface_hub import snapshot_download

# US-built VLMs only (export-compliance: no models built outside the USA).
# All use the native transformers image-text-to-text interface, so the existing
# VlmBackend loads them unchanged (pass --load-4bit for the two Gemma models).
#   Phi-4-multimodal : Microsoft, UNGATED, ~11GB -> runs bf16 on 24GB, no login.
#   gemma-3-12b-it   : Google, GATED, ~24GB -> load 4-bit on 24GB.
#   gemma-3-27b-it   : Google, GATED, ~54GB -> load 4-bit; best quality that fits.
# GATED repos need `huggingface-cli login` + license accepted on the model page,
# else snapshot_download 401s. Ungated model is first so results land fastest.
MODELS = [
    ("microsoft/Phi-4-multimodal-instruct", "PHI4MM"),  # Microsoft, ungated, bf16
    ("google/gemma-3-12b-it", "GEMMA3_12B"),            # Google, GATED, 4-bit
    ("google/gemma-3-27b-it", "GEMMA3_27B"),            # Google, GATED, 4-bit
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
