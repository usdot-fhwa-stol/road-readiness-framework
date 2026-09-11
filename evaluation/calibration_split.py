"""Stratified calibration split for freezing the D3' tolerance delta.

Why this exists
---------------
D3' (``evaluation.marking_support`` / ``evaluation.d_metrics.compute_d3_prime``)
has exactly one free parameter: the localization tolerance ``delta`` (a fraction
of the image diagonal). Per the audit (docs/d_metrics_audit.md, risk 6), delta
MUST be frozen on a held-out calibration split and never tuned on the evaluation
set. This module carves that split out of the universal manifest.

The draw is:
  * **stratified** across ``dataset x coarse-visibility-tier`` so delta is not
    frozen on one resolution/dataset or only on clearly-visible markings — it
    sees the same mixture (including the sparse degraded/occluded tiers) the
    eval set contains;
  * **seeded + deterministic** — the same manifest + seed + fraction always
    yields the same calibration sample_ids, so the freeze is reproducible;
  * **floored** — every non-empty stratum contributes at least ``min_per_stratum``
    images (when it has that many), so a rare tier is not dropped to zero.

The result is written as a JSON artifact recording the selected sample_ids and
full provenance (seed, fraction, per-stratum counts, manifest fingerprint). The
calibration ids are then EXCLUDED from the reported evaluation set.

This module is pure stdlib + NumPy (no cv2/torch/scikit-image) so it runs in the
minimal repo environment and is unit-testable there.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Coarse "Observed Marking Visibility" tiers. These mirror the tag sets in
# evaluation.d_metrics (kept as literals here so this module needs no cv2 import
# from that package); if the d_metrics sets change, change these together.
_OCCLUDED_TAGS = {"occluded", "partially missing"}
_DEGRADED_TAGS = {"faded / worn / not visible", "low contrast"} | _OCCLUDED_TAGS
_CLEAR_TAGS = {"clearly visible"}

CALIBRATION_SPLIT_VERSION = "calib_split_v1_stratified_diag_delta"


def coarse_visibility_tier(tags: Iterable[str]) -> str:
    """Map raw 'Observed Marking Visibility' tags to {clear,degraded,occluded,unknown}.

    Occlusion dominates (it is the most severe and the D5-relevant tier); a
    non-occluded but faded/low-contrast image is ``degraded``; ``clear`` only
    when a clearly-visible tag is present without any degraded tag; otherwise
    ``unknown`` (no usable visibility signal).
    """
    norm = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
    if norm & _OCCLUDED_TAGS:
        return "occluded"
    if norm & _DEGRADED_TAGS:
        return "degraded"
    if norm & _CLEAR_TAGS:
        return "clear"
    return "unknown"


def _sample_visibility_tags(sample: dict) -> list:
    """Pull the 'Observed Marking Visibility' tag list from a manifest sample.

    Looks in ``meta.predicted_tags.by_dimension`` (the tagged universal manifest
    layout) and, as a fallback, a top-level ``predicted_tags`` (the *.selected
    jsonl layout). Returns [] when absent so the sample lands in ``unknown``.
    """
    meta = sample.get("meta") or {}
    for container in (meta.get("predicted_tags"), sample.get("predicted_tags")):
        if isinstance(container, dict):
            by_dim = container.get("by_dimension") or {}
            vis = by_dim.get("Observed Marking Visibility")
            if isinstance(vis, list):
                return vis
    return []


def stratum_of(sample: dict) -> tuple[str, str]:
    """(dataset, coarse-visibility-tier) stratum key for one manifest sample."""
    dataset = str(sample.get("dataset", "unknown"))
    tier = coarse_visibility_tier(_sample_visibility_tags(sample))
    return dataset, tier


def _manifest_fingerprint(sample_ids: list[str]) -> str:
    """Order-independent fingerprint of the sample-id set, so the artifact can
    later be checked against the manifest it was drawn from."""
    h = hashlib.sha256()
    for sid in sorted(sample_ids):
        h.update(sid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def select_calibration_ids(
    samples: list[dict],
    fraction: float = 0.05,
    seed: int = 20260810,
    min_per_stratum: int = 5,
) -> dict:
    """Draw a stratified calibration subset of ``samples``.

    Parameters
    ----------
    samples : list of manifest sample dicts (each must have ``sample_id`` and
        ``dataset``; visibility tags are read where present).
    fraction : target overall share to hold out (0 < fraction < 1).
    seed : RNG seed; fixed for reproducibility.
    min_per_stratum : floor on the number drawn from each non-empty stratum
        (capped at the stratum size). Guarantees rare tiers are represented.

    Returns a dict artifact with the calibration ids and full provenance. The
    per-stratum target is ``max(min_per_stratum, round(fraction * n_stratum))``,
    capped at the stratum size; draws are seeded per stratum over sorted ids so
    the result is deterministic and independent of input ordering.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0,1), got {fraction}")
    if min_per_stratum < 0:
        raise ValueError("min_per_stratum must be >= 0")

    # Group sample_ids by stratum (sorted within stratum for determinism).
    strata: dict[tuple[str, str], list[str]] = {}
    all_ids: list[str] = []
    for s in samples:
        sid = str(s["sample_id"])
        all_ids.append(sid)
        strata.setdefault(stratum_of(s), []).append(sid)

    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate sample_id in manifest; ids must be unique")

    calib_ids: list[str] = []
    stratum_report: dict[str, dict] = {}
    for key in sorted(strata):
        ids = sorted(strata[key])
        n = len(ids)
        target = min(n, max(min_per_stratum, int(round(fraction * n))))
        # Seed per stratum so adding/removing an unrelated stratum does not shift
        # another stratum's draw. Seed mixes the global seed with the stratum key.
        key_seed = (seed ^ int(hashlib.sha256("|".join(key).encode()).hexdigest()[:8], 16)) & 0x7FFFFFFF
        rng = np.random.RandomState(key_seed)
        chosen_idx = rng.choice(n, size=target, replace=False) if target else np.array([], dtype=int)
        chosen = sorted(ids[i] for i in chosen_idx)
        calib_ids.extend(chosen)
        stratum_report["/".join(key)] = {
            "n_total": n,
            "n_calibration": len(chosen),
            "fraction_of_stratum": round(len(chosen) / n, 4) if n else None,
        }

    calib_ids = sorted(calib_ids)
    n_total = len(all_ids)
    return {
        "calibration_split_version": CALIBRATION_SPLIT_VERSION,
        "seed": seed,
        "target_fraction": fraction,
        "min_per_stratum": min_per_stratum,
        "n_total": n_total,
        "n_calibration": len(calib_ids),
        "achieved_fraction": round(len(calib_ids) / n_total, 4) if n_total else None,
        "n_strata": len(strata),
        "manifest_fingerprint_sha256": _manifest_fingerprint(all_ids),
        "per_stratum": stratum_report,
        "calibration_sample_ids": calib_ids,
    }


def load_manifest_samples(manifest_path: str) -> list[dict]:
    """Load the ``samples`` list from a universal manifest JSON.

    Accepts both the ``{"metadata":..., "samples":[...]}`` layout and a bare
    list of samples.
    """
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("samples", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"unrecognized manifest structure in {manifest_path}")


def partition(manifest_path: str, calibration_artifact: dict) -> tuple[list[str], list[str]]:
    """Split a manifest's sample_ids into (calibration, evaluation) id lists
    using a previously written calibration artifact."""
    calib = set(calibration_artifact.get("calibration_sample_ids", []))
    ids = [str(s["sample_id"]) for s in load_manifest_samples(manifest_path)]
    calib_present = [i for i in ids if i in calib]
    eval_ids = [i for i in ids if i not in calib]
    return calib_present, eval_ids


def _parse_args(argv: Optional[list[str]] = None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Universal manifest JSON (metadata+samples)")
    ap.add_argument("--out", required=True, help="Output calibration-split JSON artifact")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--min-per-stratum", type=int, default=5)
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    samples = load_manifest_samples(args.manifest)
    artifact = select_calibration_ids(
        samples,
        fraction=args.fraction,
        seed=args.seed,
        min_per_stratum=args.min_per_stratum,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"Calibration split: {artifact['n_calibration']}/{artifact['n_total']} "
        f"images ({artifact['achieved_fraction']:.2%}) across {artifact['n_strata']} strata "
        f"-> {args.out}"
    )
    # Compact per-stratum echo so the draw is visible without opening the file.
    for name, rep in sorted(artifact["per_stratum"].items()):
        print(f"  {name:32s} {rep['n_calibration']:4d}/{rep['n_total']:<5d}")


if __name__ == "__main__":
    main()
