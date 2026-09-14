"""Load the frozen D3' calibration config (delta + operating points).

This is the single source of truth for the frozen tolerance ``delta`` and the
per-processor operating points ``tau``. It is written ONCE by the calibration
driver (``evaluation.run_delta_calibration``) after selecting values on the
held-out calibration split, and read by the evaluation CLIs at scoring time.

Contract
--------
  * If no frozen config exists, ``load_frozen_delta`` returns ``None`` and the
    D3' metric stays dormant (``build_d_record`` default ``delta=None``), so the
    live pipeline is unchanged until a freeze is performed. This is deliberate:
    nothing about the published numbers moves until someone freezes delta on the
    calibration split and points the CLI at the config.
  * The config records provenance (calibration split fingerprint, selection
    rule, bootstrap-stability verdict) so a reader can verify the delta was
    frozen on calibration data, not tuned on the eval set.

Pure stdlib; safe to import in the minimal env.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Default location of the frozen config artifact. Kept alongside the calibration
# split so the two travel together.
DEFAULT_CONFIG_PATH = "dataset/calibration/d3prime_frozen.json"

FROZEN_CONFIG_VERSION = "d3prime_frozen_v1"


def load_config(path: Optional[str] = None) -> Optional[dict]:
    """Load the frozen D3' config dict, or None if it does not exist."""
    p = Path(path or DEFAULT_CONFIG_PATH)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_frozen_delta(path: Optional[str] = None) -> Optional[float]:
    """Return the frozen delta, or None when no config is present.

    ``None`` keeps D3' dormant. A present-but-malformed config raises, because a
    corrupt freeze should fail loudly rather than silently reverting to legacy.
    """
    cfg = load_config(path)
    if cfg is None:
        return None
    delta = cfg.get("delta")
    if delta is None:
        raise ValueError(f"frozen config {path or DEFAULT_CONFIG_PATH} has no 'delta'")
    delta = float(delta)
    if delta < 0 or delta != delta:  # negative or NaN
        raise ValueError(f"frozen delta must be finite non-negative, got {delta!r}")
    return delta


def operating_point(model_name: str, path: Optional[str] = None) -> Optional[float]:
    """Return the frozen operating-point tau for ``model_name`` (e.g. 'yolopx',
    'clrernet'), or None if unset. Matching is case-insensitive and substring so
    'yolopx_culane' matches the 'yolopx' entry."""
    cfg = load_config(path)
    if cfg is None:
        return None
    taus = cfg.get("operating_points") or {}
    key = str(model_name).lower()
    for name, tau in taus.items():
        if str(name).lower() in key or key in str(name).lower():
            return None if tau is None else float(tau)
    return None


def build_frozen_config(
    delta: float,
    operating_points: dict,
    selection_rule: str,
    calibration_artifact: dict,
    stability: Optional[dict] = None,
    delta_grid=None,
    notes: Optional[list] = None,
) -> dict:
    """Assemble the frozen-config dict from calibration outputs (no I/O)."""
    if delta is None or float(delta) < 0 or float(delta) != float(delta):
        raise ValueError(f"delta must be finite non-negative, got {delta!r}")
    return {
        "frozen_config_version": FROZEN_CONFIG_VERSION,
        "delta": float(delta),
        "delta_units": "fraction_of_image_diagonal",
        "selection_rule": selection_rule,
        "operating_points": {k: (None if v is None else float(v))
                             for k, v in (operating_points or {}).items()},
        "delta_grid": [float(x) for x in (delta_grid or [])],
        "calibration": {
            "calibration_split_version": calibration_artifact.get("calibration_split_version"),
            "seed": calibration_artifact.get("seed"),
            "n_calibration": calibration_artifact.get("n_calibration"),
            "n_total": calibration_artifact.get("n_total"),
            "achieved_fraction": calibration_artifact.get("achieved_fraction"),
            "manifest_fingerprint_sha256": calibration_artifact.get("manifest_fingerprint_sha256"),
        },
        "bootstrap_stability": stability,
        "notes": notes or [
            "delta is frozen on the held-out calibration split and must NOT be "
            "re-tuned on the evaluation set.",
            "delta is a fraction of the image diagonal (resolution-free, isotropic).",
            "Operating points tau are per-processor detection thresholds selected "
            "on calibration data at the reference delta.",
        ],
    }
