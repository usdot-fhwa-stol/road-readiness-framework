"""Simple, GT-guided lane-marking wear / continuity metric (classical CV).

This is the *new* I1. It is deliberately small and specialised for the human-
tagged pilot subset, not a generalised detector. Ground-truth geometry (lane
polylines or a rasterised lane mask) is used ONLY to locate where each lane
marking is expected to be. No detector / AI vision model is involved.

For each expected lane path we sample a **transverse intensity profile** across
the marking. A real, healthy marking is a bright stripe flanked by darker
pavement, so per-sample paint evidence ``p`` is the (illumination-robust)
contrast between the marking centre and the pavement immediately beside it. This
is the classical pavement-marking-condition signal; worn markings show up as a
low "present-paint percentage" along the path (see Vision-Based Pavement Marking
Detection and Condition Assessment, Applied Sciences 11(7):3152, 2021).

Each point along a lane is then labelled:

* ``good``  - crisp paint present               (full credit, drawn green)
* ``faded`` - paint present but weak             (partial credit, drawn orange)
* ``gap``   - inside a short, regular gap        (intentional dash, credited, dim)
* ``worn``  - expected paint missing, not a dash (no credit, drawn RED)

Intentional dash gaps are recognised with a single 1-D morphological closing
sized to the lane's own dash period, so a correctly dashed line scores as high
as a solid line while a missing dash (a double-length gap) still counts as a
discontinuity. The image score is the length-weighted mean of per-sample credit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import cv2
import numpy as np

# Reuse the *validated* geometry utilities only (no scoring logic is shared).
from evaluation.lane_continuity import (
    extract_guidance_lanes_from_mask,
    lanes_from_lane_json,
    prepare_guidance_lanes,
)

EPS = 1e-8

# Per-sample category codes (also the overlay colour keys).
GOOD, FADED, GAP, WORN, INVALID = "good", "faded", "gap", "worn", "invalid"

# BGR colours for the overlay (OpenCV order).
_CATEGORY_BGR = {
    GOOD: (70, 200, 70),      # green
    FADED: (40, 170, 240),    # orange
    GAP: (150, 150, 150),     # dim grey - intentional dash gap
    WORN: (40, 40, 230),      # red - worn / broken / missing
    INVALID: (90, 90, 90),    # unreliable sample (off-frame etc.)
}


@dataclass(frozen=True)
class LaneWearConfig:
    """A handful of tunable knobs. Kept intentionally small."""

    sample_spacing_px: float = 4.0      # along-lane sampling step
    max_samples: int = 600
    half_profile_px: float = 16.0       # transverse half-length of the profile
    center_band_px: float = 5.0         # +/- band treated as the marking itself
    flank_gap_px: float = 3.0           # skip this margin before the pavement flank
    profile_points: int = 41            # samples across one transverse profile
    # Contrast calibration fitted on the tagged pilot subset (see sweep in
    # scratch): real markings there have transverse contrast ~15-35 grey levels.
    contrast_floor: float = 3.0         # luminance contrast below this is noise
    contrast_scale: float = 20.0        # luminance contrast mapped to p=1
    flank_noise_k: float = 0.3          # marking must beat this * pavement MAD
    present_abs_floor: float = 0.35     # absolute p to ever count as "present"
    present_rel_frac: float = 0.45      # or this fraction of the lane's own peak
    faint_low: float = 0.18             # p below this = missing paint
    dash_close_factor: float = 1.7      # close gaps up to factor * median dash run
    min_valid_samples: int = 12         # else the lane is unavailable
    min_lane_support_px: float = 40.0
    # Only judge paint where it is optically resolvable. Above this image-height
    # fraction the marking is a couple of pixels wide and transverse contrast is
    # meaningless, so those samples are excluded rather than scored as "worn".
    near_field_top_fraction: float = 0.45


def _to_gray_and_yellow(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale (white-marking) and LAB-b* (yellow-marking) evidence planes."""

    rgb = np.asarray(image_rgb)
    if rgb.ndim == 2:
        gray = rgb.astype(np.float32)
        return gray, np.zeros_like(gray)
    rgb = rgb[..., :3].astype(np.uint8, copy=False)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    yellow = lab[..., 2]  # b*: high for yellow, ~128 neutral
    return gray, yellow


def _bilinear(plane: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(
        plane,
        xs.astype(np.float32),
        ys.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _paint_evidence_along_lane(
    points: np.ndarray,
    gray: np.ndarray,
    yellow: np.ndarray,
    cfg: LaneWearConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample paint evidence ``p`` in [0,1] and validity ``q`` in {0,1}."""

    height, width = gray.shape[:2]
    tangent = np.gradient(points, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    tangent = tangent / norm
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    # Perspective: markings get wider toward the camera, so the whole transverse
    # profile is scaled with image row (≈0.6x near the horizon gate, ≈2.6x at the
    # bottom). Without this the pavement flank lands *inside* a wide near-field
    # stripe and healthy paint reads as zero-contrast "worn".
    persp = 0.6 + 2.0 * points[:, 1] / max(height - 1, 1)
    offsets = np.linspace(
        -cfg.half_profile_px, cfg.half_profile_px, cfg.profile_points
    )
    scaled = persp[:, None] * offsets[None, :]
    xs = points[:, 0, None] + normal[:, 0, None] * scaled
    ys = points[:, 1, None] + normal[:, 1, None] * scaled

    in_frame = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
    valid = in_frame.mean(axis=1) >= 0.85
    # Near-field gate: exclude samples too close to the horizon to resolve paint.
    valid &= points[:, 1] >= cfg.near_field_top_fraction * height

    gray_prof = _bilinear(gray, xs, ys)
    yellow_prof = _bilinear(yellow, xs, ys)

    center = np.abs(offsets) <= cfg.center_band_px
    flank = np.abs(offsets) >= (cfg.center_band_px + cfg.flank_gap_px)

    def contrast(profile: np.ndarray, signed: bool) -> np.ndarray:
        c = profile[:, center]
        f = profile[:, flank]
        # Robust marking value (bright for white; |b*-neutral| for yellow) vs the
        # pavement median right beside it. Comparing to the *adjacent* pavement is
        # what makes this robust to shadows and global exposure changes.
        if signed:
            center_val = np.abs(np.percentile(c, 75, axis=1) - 128.0)
            flank_val = np.abs(np.median(f, axis=1) - 128.0)
        else:
            center_val = np.percentile(c, 75, axis=1)
            flank_val = np.median(f, axis=1)
        flank_mad = np.median(np.abs(f - np.median(f, axis=1, keepdims=True)), axis=1)
        raw = center_val - flank_val - cfg.flank_noise_k * flank_mad
        return np.clip((raw - cfg.contrast_floor) / cfg.contrast_scale, 0.0, 1.0)

    p = np.maximum(contrast(gray_prof, signed=False), contrast(yellow_prof, signed=True))
    q = valid.astype(np.float64)
    return p * q, q


def _label_samples(
    p: np.ndarray, q: np.ndarray, cfg: LaneWearConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Label each sample (good/faded/gap/worn/invalid) and give it 0-1 credit."""

    n = p.size
    labels = np.full(n, INVALID, dtype=object)
    credit = np.zeros(n, dtype=np.float64)
    valid = q > 0.5
    if not np.any(valid):
        return labels, credit

    peak = float(np.percentile(p[valid], 85)) if np.any(valid) else 0.0
    present_thr = max(cfg.present_abs_floor, cfg.present_rel_frac * peak)

    present = valid & (p >= present_thr)
    missing = valid & (p < cfg.faint_low)
    faded = valid & ~present & ~missing  # weak but not gone

    # Recognise intentional dash gaps: close short runs of "not-present" whose
    # length is comparable to the surrounding painted dashes. A missing dash makes
    # a much longer gap and is NOT closed, so it stays "worn".
    present_seq = present.astype(np.uint8)
    dash_len = _median_true_run(present_seq)
    close_k = max(1, int(round(cfg.dash_close_factor * dash_len)))
    closed = _binary_close_1d(present_seq, close_k).astype(bool) & valid

    for i in range(n):
        if not valid[i]:
            labels[i] = INVALID
            continue
        if present[i]:
            labels[i], credit[i] = GOOD, 1.0
        elif faded[i]:
            labels[i] = FADED
            credit[i] = float(np.clip((p[i] - cfg.faint_low) / max(present_thr - cfg.faint_low, EPS), 0.0, 1.0))
        elif closed[i]:
            labels[i], credit[i] = GAP, 1.0  # intentional dash gap
        else:
            labels[i], credit[i] = WORN, 0.0
    return labels, credit


def _median_true_run(seq: np.ndarray) -> float:
    runs = [len(r) for r in _runs(seq) if r[0] == 1]
    return float(np.median(runs)) if runs else 1.0


def _runs(seq: np.ndarray) -> list[tuple[int, int]]:
    """Return [(value, length), ...] run-length encoding of a 0/1 sequence."""

    out: list[tuple[int, int]] = []
    if seq.size == 0:
        return out
    start = 0
    for i in range(1, seq.size + 1):
        if i == seq.size or seq[i] != seq[start]:
            out.append((int(seq[start]), i - start))
            start = i
    return out


def _binary_close_1d(seq: np.ndarray, k: int) -> np.ndarray:
    """1-D morphological closing (dilate then erode) with a length-k element."""

    if k <= 1 or seq.size == 0:
        return seq.copy()
    kernel = np.ones((1, k), np.uint8)
    col = seq.reshape(1, -1).astype(np.uint8)
    closed = cv2.morphologyEx(col, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT)
    return closed.reshape(-1)


def _condition_label(worn_fraction: float, faded_fraction: float) -> str:
    if worn_fraction >= 0.30:
        return "worn_or_broken"
    if worn_fraction >= 0.10 or faded_fraction >= 0.30:
        return "partially_worn"
    if faded_fraction >= 0.12:
        return "faded"
    return "intact"


def analyze_lane_wear(
    image_rgb: np.ndarray,
    lane_geometries: Optional[Iterable[Any]],
    *,
    config: Optional[LaneWearConfig] = None,
) -> dict[str, Any]:
    """Compute the wear/continuity I1 for one image from GT geometry + RGB.

    Returns an image aggregate plus per-lane diagnostics (including per-sample
    points, labels and credit) suitable for :func:`render_wear_overlay`.
    """

    cfg = config or LaneWearConfig()
    gray, yellow = _to_gray_and_yellow(image_rgb)
    prepared = prepare_guidance_lanes(lane_geometries, gray.shape[:2])

    lanes_out: list[dict[str, Any]] = []
    for idx, lane in enumerate(prepared):
        points = np.asarray(lane["points"], dtype=np.float64)
        if lane["support_px"] < cfg.min_lane_support_px:
            continue
        p, q = _paint_evidence_along_lane(points, gray, yellow, cfg)
        labels, credit = _label_samples(p, q, cfg)
        valid = q > 0.5
        n_valid = int(np.count_nonzero(valid))
        if n_valid < cfg.min_valid_samples:
            lanes_out.append({
                "lane_index": idx, "points": points, "labels": labels,
                "credit": credit, "score": None, "support_px": lane["support_px"],
                "unavailable_reason": "insufficient_valid_samples",
            })
            continue
        score = float(np.mean(credit[valid]))
        worn_frac = float(np.mean(labels[valid] == WORN))
        faded_frac = float(np.mean(labels[valid] == FADED))
        gap_frac = float(np.mean(labels[valid] == GAP))
        good_frac = float(np.mean(labels[valid] == GOOD))
        pattern = "dashed" if gap_frac >= 0.12 else "solid"
        lanes_out.append({
            "lane_index": idx,
            "points": points,
            "labels": labels,
            "credit": credit,
            "score": score,
            "support_px": lane["support_px"],
            "n_valid": n_valid,
            "worn_fraction": worn_frac,
            "faded_fraction": faded_frac,
            "gap_fraction": gap_frac,
            "good_fraction": good_frac,
            "pattern": pattern,
            "condition": _condition_label(worn_frac, faded_frac),
            "unavailable_reason": None,
        })

    scored = [l for l in lanes_out if l.get("score") is not None]
    if not scored:
        return {
            "I1": None, "score": None, "condition": "unknown",
            "unavailable_reason": "no_scored_lanes" if lanes_out else "no_credible_geometry",
            "valid_lane_count": 0, "lane_count": len(lanes_out), "lanes": lanes_out,
        }
    weights = np.asarray([l["support_px"] for l in scored], dtype=np.float64)
    weights = weights / max(float(weights.sum()), EPS)
    image_score = float(np.sum(weights * np.asarray([l["score"] for l in scored])))
    worn_frac = float(np.sum(weights * np.asarray([l["worn_fraction"] for l in scored])))
    faded_frac = float(np.sum(weights * np.asarray([l["faded_fraction"] for l in scored])))
    return {
        "I1": image_score,
        "score": image_score,
        "worn_fraction": worn_frac,
        "faded_fraction": faded_frac,
        "condition": _condition_label(worn_frac, faded_frac),
        "valid_lane_count": len(scored),
        "lane_count": len(lanes_out),
        "unavailable_reason": None,
        "lanes": lanes_out,
    }


def select_gt_geometry(
    ground_truth: dict[str, Any],
    image_shape: tuple[int, int],
    load_mask=None,
) -> tuple[list[np.ndarray], str]:
    """Pick GT polylines (preferred) or mask centrelines as the expected paths.

    ``load_mask`` is an optional callable returning the binary GT mask; it is
    only invoked when polylines are unavailable or the GT is mask-natural.
    """

    natural = str(ground_truth.get("natural_gt", "")).lower()
    lane_json = ground_truth.get("lane_json")
    if natural in {"lanes", "polyline", "polylines", "lane_json"} and lane_json:
        lanes = lanes_from_lane_json(lane_json)
        if lanes and prepare_guidance_lanes(lanes, image_shape):
            return lanes, "gt_lane_json"
    mask = load_mask() if load_mask is not None else None
    if mask is not None:
        lanes = extract_guidance_lanes_from_mask(mask)
        if lanes and prepare_guidance_lanes(lanes, image_shape):
            return lanes, "gt_mask_grouped_centerline"
    if lane_json:
        lanes = lanes_from_lane_json(lane_json)
        if lanes and prepare_guidance_lanes(lanes, image_shape):
            return lanes, "gt_lane_json"
    return [], "none"


def render_wear_overlay(
    image_rgb: np.ndarray,
    analysis: dict[str, Any],
    *,
    title: Optional[str] = None,
) -> np.ndarray:
    """Draw the scored lanes (green/orange/dim/red) with the score burned in.

    Returns a BGR image (ready for ``cv2.imwrite``). Red segments mark where the
    expected paint is worn / broken / missing (not an intentional dash gap).
    """

    rgb = np.asarray(image_rgb)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    bgr = cv2.cvtColor(rgb[..., :3].astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR).copy()

    for lane in analysis.get("lanes", []):
        pts = np.asarray(lane["points"], dtype=np.int32)
        labels = lane["labels"]
        for i in range(len(pts) - 1):
            colour = _CATEGORY_BGR.get(labels[i], _CATEGORY_BGR[INVALID])
            thickness = 6 if labels[i] == WORN else 4
            cv2.line(bgr, tuple(pts[i]), tuple(pts[i + 1]), colour, thickness, cv2.LINE_AA)

    score = analysis.get("I1")
    condition = analysis.get("condition", "unknown")
    label = "I1: N/A" if score is None else f"I1: {score:.3f}"
    header = f"{label}   ({condition})"
    if title:
        header = f"{title}   {header}"
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 40), (25, 25, 25), -1)
    cv2.putText(bgr, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Legend.
    legend = [("good", GOOD), ("faded", FADED), ("dash gap", GAP), ("worn/broken", WORN)]
    x = 10
    y = bgr.shape[0] - 12
    for name, key in legend:
        cv2.rectangle(bgr, (x, y - 12), (x + 16, y), _CATEGORY_BGR[key], -1)
        cv2.putText(bgr, name, (x + 20, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x += 30 + 9 * len(name)
    return bgr


__all__ = [
    "LaneWearConfig",
    "analyze_lane_wear",
    "render_wear_overlay",
    "select_gt_geometry",
]
