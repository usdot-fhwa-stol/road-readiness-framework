"""Save raw + overlay images for frames that fail or excel on each readiness metric."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Thresholds — (operator, value)
#   For failures  : 'lt' → value < thresh is BAD
#                   'gt' → value > thresh is BAD
#                   'eq' → value == thresh is BAD
#   For good cases: 'gt' → value > thresh is GOOD
#                   'lt' → value < thresh is GOOD
#                   'gte'→ value >= thresh is GOOD
# ---------------------------------------------------------------------------

DEFAULT_FAILURE_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    "D3_f1": ("lt", 0.45),
    "D3_iou": ("lt", 0.25),
    "D4_near_iou": ("lt", 0.30),
    "D6_image_missed_ratio": ("gt", 0.50),
    "I2_local_contrast": ("lt", 0.12),
    "I1_pattern_continuity": ("lt", 0.45),
    "I3_boundary_sharpness": ("lt", 0.20),
    "R1_marking_readability_score": ("lt", 60.0),
    "R2_reference_detectability_score": ("lt", 60.0),
}

# Keep the old name as an alias so existing call sites don't break.
DEFAULT_THRESHOLDS = DEFAULT_FAILURE_THRESHOLDS

DEFAULT_GOOD_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    "D3_f1": ("gt", 0.80),
    "D3_iou": ("gt", 0.60),
    "D4_near_iou": ("gt", 0.65),
    "D6_image_missed_ratio": ("lt", 0.15),
    "I2_local_contrast": ("gt", 0.30),
    "I1_pattern_continuity": ("gt", 0.75),
    "I3_boundary_sharpness": ("gt", 0.45),
    "R1_marking_readability_score": ("gt", 80.0),
    "R2_reference_detectability_score": ("gt", 80.0),
}


# ---------------------------------------------------------------------------
# Threshold checks
# ---------------------------------------------------------------------------

def _fails(value: float, threshold: Tuple[str, float]) -> bool:
    op, thresh = threshold
    if op == "lt":  return value < thresh
    if op == "gt":  return value > thresh
    if op == "eq":  return value == thresh
    return False


def _is_good(value: float, threshold: Tuple[str, float]) -> bool:
    op, thresh = threshold
    if op == "gt":  return value > thresh
    if op == "lt":  return value < thresh
    if op == "gte": return value >= thresh
    return False


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def make_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Blend GT and prediction masks onto the image.

    Colour coding (alpha = 0.55):
      Yellow — TP  correctly detected lane pixel
      Green  — FN  GT lane pixel the model missed
      Red    — FP  predicted lane pixel with no GT
    """
    overlay = image_rgb.astype(np.float32)
    alpha = 0.55

    tp = (gt_mask > 0) & (pred_mask > 0)
    fn = (gt_mask > 0) & (pred_mask == 0)
    fp = (gt_mask == 0) & (pred_mask > 0)

    for c, v in zip([0, 1, 2], [255, 255, 0]):   # Yellow TP
        overlay[tp, c] = overlay[tp, c] * (1 - alpha) + v * alpha
    for c, v in zip([0, 1, 2], [0, 200, 0]):     # Green FN
        overlay[fn, c] = overlay[fn, c] * (1 - alpha) + v * alpha
    for c, v in zip([0, 1, 2], [220, 0, 0]):     # Red FP
        overlay[fp, c] = overlay[fp, c] * (1 - alpha) + v * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_if_needed(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str, good: bool = False) -> np.ndarray:
    """Burn metric + value into the top-left corner. Green background for good, black for fail."""
    img = img.copy()
    bg = (0, 80, 0) if good else (0, 0, 0)
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 6, 26), bg, -1)
    cv2.putText(img, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_i1_lane_debug(image_rgb: np.ndarray, lane_diagnostic: dict) -> np.ndarray:
    """Render one lane's corridor, validity, evidence, and fitted template."""

    debug = lane_diagnostic.get("debug")
    if not isinstance(debug, dict):
        raise ValueError("lane diagnostic has no debug payload; analyze with debug=True")
    image = np.asarray(image_rgb)[..., :3].astype(np.uint8).copy()
    centerline = np.asarray(debug.get("centerline", []), dtype=np.float64)
    left = np.asarray(debug.get("corridor_left", []), dtype=np.float64)
    right = np.asarray(debug.get("corridor_right", []), dtype=np.float64)
    p = np.asarray(debug.get("paint_evidence", []), dtype=np.float64)
    q = np.asarray(debug.get("validity", []), dtype=np.float64)
    occupancy = np.asarray(debug.get("occupancy", []), dtype=np.float64)
    expected_value = debug.get("expected_template")
    missing_value = debug.get("missing_expected_paint")
    expected = None if expected_value is None else np.asarray(expected_value, dtype=np.float64)
    missing = None if missing_value is None else np.asarray(missing_value, dtype=bool)

    def polyline(points: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
        if points.ndim == 2 and points.shape[0] >= 2:
            cv2.polylines(
                image,
                [np.round(points).astype(np.int32)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )

    polyline(left, (0, 210, 255))
    polyline(right, (0, 210, 255))
    polyline(centerline, (40, 255, 40), 2)
    if centerline.shape[0] == q.size:
        for point in centerline[q < 0.22]:
            cv2.circle(image, tuple(np.round(point).astype(int)), 2, (255, 40, 40), -1)

    height, width = image.shape[:2]
    chart_height = max(220, min(320, height // 2))
    chart = np.full((chart_height, width, 3), 22, dtype=np.uint8)
    top, bottom = 46, chart_height - 30
    cv2.line(chart, (34, bottom), (width - 12, bottom), (100, 100, 100), 1)
    cv2.line(chart, (34, top), (34, bottom), (100, 100, 100), 1)

    def draw_signal(values: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
        if values.size < 2:
            return
        xs = np.linspace(34, width - 12, values.size)
        ys = bottom - np.clip(values, 0.0, 1.0) * (bottom - top)
        points = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(chart, [points], False, color, thickness, cv2.LINE_AA)

    draw_signal(p, (255, 255, 255), 2)
    draw_signal(q, (0, 220, 255), 1)
    draw_signal(occupancy * 0.92, (170, 170, 170), 1)
    if expected is not None:
        draw_signal(expected * 0.82, (255, 210, 0), 2)
    if missing is not None and missing.size:
        xs = np.linspace(34, width - 12, missing.size)
        for x in xs[missing]:
            cv2.line(chart, (int(x), bottom), (int(x), bottom - 18), (255, 35, 35), 1)

    title = (
        f"{lane_diagnostic.get('intended_pattern')}  "
        f"confidence={lane_diagnostic.get('pattern_confidence')}  "
        f"continuity={lane_diagnostic.get('continuity_score')}  "
        f"degradation={lane_diagnostic.get('degradation_score')}"
    )
    cv2.putText(chart, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(
        chart,
        "p: white  q: cyan  occupancy: gray  expected: yellow  missing: red",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([image, chart])


def save_i1_lane_debug(image_rgb: np.ndarray, lane_diagnostic: dict, output_path: str) -> Path:
    """Save a developer-facing I1 lane diagnostic image."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_i1_lane_debug(image_rgb, lane_diagnostic)
    if not cv2.imwrite(str(output), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write I1 debug visualization: {output}")
    return output


def render_i1_highlight_overlay(image_rgb: np.ndarray, lane_diagnostics: list) -> np.ndarray:
    """Draw every lane's paint-continuity diagnostic directly on the image.

    Most images have several candidate lane corridors but only some of them
    get a confident enough dashed/solid pattern fit to actually count toward
    I1 (the rest are ``intended_pattern == "unknown"`` and contribute
    nothing). Those are drawn thin and muted so they don't compete visually
    with the lanes that actually drove the score.

    For a scored lane, I1's continuity score is the q*expected-weighted mean
    of the paint evidence signal ``p`` (see compute_pattern_condition). This
    renders that same integrand directly on the image, so the shading *is*
    the score:
      Green->Red — a position where the fitted pattern expects paint, shaded
                   by its paint evidence ``p`` (green = fully present, red =
                   absent). Red segments are exactly what pulled I1 down;
                   a lane that's mostly yellow/orange here has broadly faded
                   (not sharply missing) paint, which also lowers I1.
      Gray       — excluded from scoring (occlusion/glare/low geometry
                   reliability); doesn't affect the score either way.
      Amber      — a gap the fitted pattern does NOT expect paint in (e.g. a
                   normal dash gap); this does not affect the score.
      Thin muted — a detected lane corridor that was NOT scored at all
                   (unknown pattern); shown for context only.

    Requires each lane_diagnostic to carry a ``debug`` payload, i.e. the
    metric was computed with ``debug=True``.
    """

    image = np.asarray(image_rgb)[..., :3].astype(np.uint8).copy()
    for lane_diagnostic in lane_diagnostics or []:
        debug = lane_diagnostic.get("debug")
        if not isinstance(debug, dict):
            continue
        centerline = np.asarray(debug.get("centerline", []), dtype=np.float64)
        n = centerline.shape[0]
        if n < 2:
            continue
        scored = lane_diagnostic.get("intended_pattern") not in (None, "unknown") and debug.get(
            "expected_template"
        ) is not None
        if not scored:
            cv2.polylines(
                image, [np.round(centerline).astype(np.int32)], False, (110, 110, 150), 1, cv2.LINE_AA
            )
            continue
        q = np.asarray(debug.get("validity", []), dtype=np.float64)
        p = np.asarray(debug.get("paint_evidence", []), dtype=np.float64)
        expected = np.asarray(debug.get("expected_template"), dtype=np.float64) > 0.5
        for i in range(n - 1):
            excluded = i < q.size and q[i] < 0.22
            is_expected = i < expected.size and expected[i]
            if excluded:
                color = (150, 150, 150)
            elif is_expected:
                value = float(np.clip(p[i], 0.0, 1.0)) if i < p.size else 0.0
                color = (int(round(230 * (1 - value))), int(round(200 * value)), 30)
            else:
                color = (255, 180, 0)
            p0 = tuple(np.round(centerline[i]).astype(int))
            p1 = tuple(np.round(centerline[i + 1]).astype(int))
            cv2.line(image, p0, p1, color, 4, cv2.LINE_AA)
    return image


def _header_bar(width: int, text: str, *, good: Optional[bool], height: int = 30) -> np.ndarray:
    bg = (0, 60, 0) if good else ((60, 0, 0) if good is False else (25, 25, 25))
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 22), (min(w, len(text) * 8 + 8), h), (0, 0, 0), -1)
    cv2.putText(img, text, (4, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_i1_overlay_panel(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    record: dict,
    output_path: str,
    *,
    max_side: int = 640,
) -> Path:
    """Save a 3-panel [raw | pred-vs-GT | I1 highlight] image for one sample.

    ``record`` is a full ``build_metric_record(...)`` output: reads canonical
    ``I1``/``I1_condition``/``I1_lane_diagnostics`` (the metric actually
    reported) for the value and the highlighted region.
    """

    raw = _resize_if_needed(np.asarray(image_rgb)[..., :3].astype(np.uint8), max_side)
    pred_overlay = _resize_if_needed(make_overlay(image_rgb, gt_mask, pred_mask), max_side)
    highlight = _resize_if_needed(
        render_i1_highlight_overlay(image_rgb, record.get("I1_lane_diagnostics", [])), max_side
    )

    raw = _caption(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), "raw")
    pred_overlay = _caption(cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR), "pred vs GT (yellow=TP green=FN red=FP)")
    highlight = _caption(
        cv2.cvtColor(highlight, cv2.COLOR_RGB2BGR),
        "I1 region (green->red=paint evidence where expected, gray=excluded)",
    )

    heights = [raw.shape[0], pred_overlay.shape[0], highlight.shape[0]]
    target_h = max(heights)

    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        canvas = np.zeros((target_h, img.shape[1], 3), dtype=np.uint8)
        canvas[: img.shape[0]] = img
        return canvas

    panel = np.hstack([pad(raw), pad(pred_overlay), pad(highlight)])

    i1 = record.get("I1")
    condition = record.get("I1_condition")
    if i1 is None:
        header_text = f"I1=unavailable ({record.get('I1_unavailable_reason')})"
        good = None
    else:
        header_text = f"I1={i1:.3f} ({condition})"
        good = _is_good(float(i1), DEFAULT_GOOD_THRESHOLDS["I1_pattern_continuity"])
    header = _header_bar(panel.shape[1], header_text, good=good)

    full = np.vstack([header, panel])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), full):
        raise OSError(f"Failed to write I1 overlay panel: {output}")
    return output


def render_i4_lane_width_debug(image_rgb: np.ndarray, analysis: dict) -> np.ndarray:
    """Render I4 source/logical geometry, normal samples, and width profile."""

    top_debug = analysis.get("debug")
    if not isinstance(top_debug, dict):
        raise ValueError("I4 analysis has no debug payload; analyze with debug=True")
    image = np.asarray(image_rgb)[..., :3].astype(np.uint8).copy()

    def polyline(points, color, thickness=1):
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 2 and array.shape[0] >= 2:
            finite = array[np.isfinite(array).all(axis=1)]
            if finite.shape[0] >= 2:
                cv2.polylines(image, [np.round(finite).astype(np.int32)], False, color, thickness, cv2.LINE_AA)

    for boundary in top_debug.get("source_boundaries_image", []):
        polyline(boundary, (130, 130, 130), 1)
    logical_images = top_debug.get("logical_boundaries_image", [])
    logical_ids = top_debug.get("logical_boundary_ids", [])
    for boundary in logical_images:
        polyline(boundary, (0, 230, 255), 2)
    logical_by_id = {
        str(boundary_id): boundary
        for boundary_id, boundary in zip(logical_ids, logical_images)
    }
    for rejected in analysis.get("rejected_pair_reasons", []):
        for key in ("left_boundary_id", "right_boundary_id"):
            boundary = logical_by_id.get(str(rejected.get(key)))
            if boundary is not None:
                polyline(boundary, (255, 45, 180), 2)

    pairs = analysis.get("pair_diagnostics", [])
    for pair in pairs:
        debug = pair.get("debug")
        if not isinstance(debug, dict):
            continue
        centers = np.asarray(debug.get("centerline_stations_image", []), dtype=np.float64)
        left = np.asarray(debug.get("left_intersections_image", []), dtype=np.float64)
        right = np.asarray(debug.get("right_intersections_image", []), dtype=np.float64)
        reliability = np.asarray(debug.get("sample_reliability", []), dtype=np.float64)
        valid = np.asarray(debug.get("valid_samples", []), dtype=bool)
        for index in range(min(centers.shape[0], left.shape[0], right.shape[0])):
            if not np.isfinite(np.concatenate([left[index], right[index]])).all():
                continue
            q = reliability[index] if index < reliability.size else 0.0
            color = (40, int(80 + 175 * np.clip(q, 0.0, 1.0)), 255) if index < valid.size and valid[index] else (255, 55, 55)
            cv2.line(
                image,
                tuple(np.round(left[index]).astype(int)),
                tuple(np.round(right[index]).astype(int)),
                color,
                1,
                cv2.LINE_AA,
            )
            if index < centers.shape[0] and np.isfinite(centers[index]).all():
                cv2.circle(image, tuple(np.round(centers[index]).astype(int)), 1, color, -1)

    height, width = image.shape[:2]
    chart_height = max(250, min(360, height // 2 + 80))
    chart = np.full((chart_height, width, 3), 22, dtype=np.uint8)
    title = (
        f"{analysis.get('profile_type')}  stability={analysis.get('lane_width_stability')}  "
        f"constancy={analysis.get('width_constancy')}  {analysis.get('measurement_space')}"
    )
    cv2.putText(chart, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (238, 238, 238), 1, cv2.LINE_AA)
    cv2.putText(
        chart,
        f"valid={analysis.get('valid_pair_count')} unknown={analysis.get('unknown_pair_count')} "
        f"rejected={len(analysis.get('rejected_pair_reasons', []))} units={analysis.get('width_units')}",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    plot_top, plot_bottom = 58, chart_height - 26
    cv2.rectangle(chart, (35, plot_top), (width - 12, plot_bottom), (90, 90, 90), 1)
    if pairs and isinstance(pairs[0].get("debug"), dict):
        debug = pairs[0]["debug"]
        raw = np.asarray(debug.get("raw_width_samples", []), dtype=np.float64)
        expected = np.asarray(debug.get("expected_width_profile", []), dtype=np.float64)
        reliability = np.asarray(debug.get("sample_reliability", []), dtype=np.float64)
        residuals = np.asarray(debug.get("residuals", []), dtype=np.float64)
        outliers = np.asarray(debug.get("rejected_outliers", []), dtype=bool)
        finite_width = np.isfinite(raw) | np.isfinite(expected)
        if np.any(finite_width):
            values = np.concatenate([raw[np.isfinite(raw)], expected[np.isfinite(expected)]])
            low, high = float(np.min(values)), float(np.max(values))
            padding = max((high - low) * 0.12, max(abs(high), 1.0) * 0.02)
            low, high = low - padding, high + padding

            def draw_width(values_array, color, thickness):
                if values_array.size < 2:
                    return
                xs = np.linspace(35, width - 12, values_array.size)
                ys = plot_bottom - (values_array - low) / max(high - low, 1e-9) * (plot_bottom - plot_top)
                valid_values = np.isfinite(values_array)
                for start in np.flatnonzero(valid_values):
                    if start == 0 or not valid_values[start - 1]:
                        end = start
                        while end + 1 < values_array.size and valid_values[end + 1]:
                            end += 1
                        if end > start:
                            points = np.column_stack([xs[start:end + 1], ys[start:end + 1]]).astype(np.int32)
                            cv2.polylines(chart, [points], False, color, thickness, cv2.LINE_AA)

            draw_width(raw, (245, 245, 245), 1)
            draw_width(expected, (0, 220, 255), 2)
            xs = np.linspace(35, width - 12, max(reliability.size, 1))
            for index, q in enumerate(reliability):
                if q < 0.25:
                    cv2.circle(chart, (int(xs[index]), plot_bottom - 3), 2, (255, 50, 50), -1)
            if outliers.size == raw.size:
                for x, flag in zip(np.linspace(35, width - 12, raw.size), outliers):
                    if flag:
                        cv2.line(chart, (int(x), plot_top), (int(x), plot_top + 10), (255, 70, 255), 1)
        if residuals.size:
            residual_text = np.nanpercentile(residuals, 90) if np.any(np.isfinite(residuals)) else None
            cv2.putText(
                chart,
                f"raw:white expected:yellow low-q:red robust-outlier:magenta residual-p90={residual_text}",
                (10, chart_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (205, 205, 205),
                1,
                cv2.LINE_AA,
            )
    return np.vstack([image, chart])


def save_i4_lane_width_debug(image_rgb: np.ndarray, analysis: dict, output_path: str) -> Path:
    """Save the developer-facing I4 lane-width diagnostic image."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_i4_lane_width_debug(image_rgb, analysis)
    if not cv2.imwrite(str(output), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write I4 debug visualization: {output}")
    return output


_PROFILE_COLORS = {
    "straight": (120, 220, 120),
    "constant_curve": (0, 210, 255),
    "transition_curve": (255, 180, 0),
    "compound_curve": (255, 120, 255),
    "reverse_curve": (60, 60, 255),
    "irregular_or_noisy": (140, 140, 140),
    "unknown": (90, 90, 90),
}


def render_i5_lane_debug(image_rgb: np.ndarray, analysis: dict) -> np.ndarray:
    """Render I5 analyzed centerlines, curvature profile kappa(s), and topology.

    Requires ``analyze_lane_geometry_complexity(..., debug=True)`` so each
    lane diagnostic carries its analyzed arc-length samples, curvature, and
    sample reliability weight.
    """

    lanes = analysis.get("lane_diagnostics", [])
    if not any(isinstance(lane.get("debug"), dict) for lane in lanes):
        raise ValueError("I5 analysis has no debug payload; analyze with debug=True")
    image = np.asarray(image_rgb)[..., :3].astype(np.uint8).copy()

    def polyline(points, color, thickness=1):
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 2 and array.shape[0] >= 2:
            finite = array[np.isfinite(array).all(axis=1)]
            if finite.shape[0] >= 2:
                cv2.polylines(image, [np.round(finite).astype(np.int32)], False, color, thickness, cv2.LINE_AA)

    for lane in lanes:
        debug = lane.get("debug")
        if not isinstance(debug, dict):
            continue
        image_xy = np.asarray(debug.get("image_xy", []), dtype=np.float64)
        weight = np.asarray(debug.get("weight", []), dtype=np.float64)
        color = _PROFILE_COLORS.get(str(lane.get("profile_class")), (200, 200, 200))
        polyline(image_xy, color, 2)
        if image_xy.shape[0] == weight.shape[0]:
            for point, w in zip(image_xy, weight):
                if w < 0.08 and np.isfinite(point).all():
                    cv2.circle(image, tuple(np.round(point).astype(int)), 2, (255, 40, 40), -1)

    height, width = image.shape[:2]
    chart_height = max(240, min(340, height // 2 + 40))
    chart = np.full((chart_height, width, 3), 22, dtype=np.uint8)
    title = (
        f"{analysis.get('profile_class')}  complexity={analysis.get('alignment_complexity')}  "
        f"confidence={analysis.get('complexity_confidence')}  {analysis.get('measurement_space')}"
    )
    cv2.putText(chart, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (238, 238, 238), 1, cv2.LINE_AA)
    cv2.putText(
        chart,
        f"valid={analysis.get('valid_lane_count')} unknown={analysis.get('unknown_lane_count')} "
        f"topology={analysis.get('topology_complexity')} bendiness={analysis.get('bendiness')}",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    plot_top, plot_bottom = 58, chart_height - 26
    cv2.rectangle(chart, (35, plot_top), (width - 12, plot_bottom), (90, 90, 90), 1)
    debug_lanes = [lane for lane in lanes if isinstance(lane.get("debug"), dict)]
    if debug_lanes:
        debug = debug_lanes[0]["debug"]
        kappa = np.asarray(debug.get("kappa", []), dtype=np.float64)
        weight = np.asarray(debug.get("weight", []), dtype=np.float64)
        if kappa.size >= 2:
            finite = kappa[np.isfinite(kappa)]
            magnitude = float(np.max(np.abs(finite))) if finite.size else 1.0
            magnitude = max(magnitude, 1e-9)
            xs = np.linspace(35, width - 12, kappa.size)
            zero_y = 0.5 * (plot_top + plot_bottom)
            cv2.line(chart, (35, int(zero_y)), (width - 12, int(zero_y)), (70, 70, 70), 1)
            ys = zero_y - np.clip(kappa / magnitude, -1.0, 1.0) * (plot_bottom - plot_top) * 0.45
            points = np.column_stack([xs, ys]).astype(np.int32)
            cv2.polylines(chart, [points], False, (255, 255, 255), 2, cv2.LINE_AA)
            for index, w in enumerate(weight):
                if w < 0.08:
                    cv2.circle(chart, (int(xs[index]), plot_bottom - 3), 2, (255, 50, 50), -1)
            for lane in debug_lanes:
                for change in lane.get("change_points") or []:
                    x = 35 + float(change) * (width - 12 - 35)
                    cv2.line(chart, (int(x), plot_top), (int(x), plot_bottom), (255, 210, 0), 1)
        cv2.putText(
            chart,
            "kappa(s): white  low-reliability: red  change-points: yellow",
            (10, chart_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([image, chart])


def save_i5_lane_debug(image_rgb: np.ndarray, analysis: dict, output_path: str) -> Path:
    """Save the developer-facing I5 geometry-complexity diagnostic image."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_i5_lane_debug(image_rgb, analysis)
    if not cv2.imwrite(str(output), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write I5 debug visualization: {output}")
    return output


# ---------------------------------------------------------------------------
# Core save function
# ---------------------------------------------------------------------------

def _save_examples(
    samples: list,
    images_rgb: List[np.ndarray],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    per_image_metrics: List[dict],
    output_dir: str,
    thresholds: Dict[str, Tuple[str, float]],
    check_fn: Callable[[float, Tuple[str, float]], bool],
    good: bool,
    max_side: int,
) -> Dict[str, int]:
    out = Path(output_dir)
    counts: Dict[str, int] = {m: 0 for m in thresholds}

    metrics_in_data = set(per_image_metrics[0].keys()) if per_image_metrics else set()
    active = {m: t for m, t in thresholds.items() if m in metrics_in_data}

    for metric in active:
        (out / metric).mkdir(parents=True, exist_ok=True)

    for idx, (sample, img_rgb, pred, gt, mets) in enumerate(
        zip(samples, images_rgb, pred_masks, gt_masks, per_image_metrics)
    ):
        image_id = str(getattr(sample, "image_id", f"frame_{idx:05d}"))
        image_id = image_id.replace("/", "_").replace("\\", "_")

        overlay = None  # built once, reused across metrics for the same image

        for metric, threshold in active.items():
            val = mets.get(metric)
            if val is None or not check_fn(float(val), threshold):
                continue

            if overlay is None:
                overlay = make_overlay(img_rgb, gt, pred)

            raw_small = _resize_if_needed(img_rgb, max_side)
            ovl_small = _resize_if_needed(overlay, max_side)

            suffix = " [good]" if good else ""
            label_text = f"{metric}={val:.2f}{suffix}" if isinstance(val, float) else f"{metric}={val}{suffix}"
            ovl_labelled = _put_label(cv2.cvtColor(ovl_small, cv2.COLOR_RGB2BGR), label_text, good=good)
            raw_bgr = cv2.cvtColor(raw_small, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(out / metric / f"{image_id}_raw.jpg"),     raw_bgr)
            cv2.imwrite(str(out / metric / f"{image_id}_overlay.jpg"), ovl_labelled)
            counts[metric] += 1

    return counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_failure_images(
    samples: list,
    images_rgb: List[np.ndarray],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    per_image_metrics: List[dict],
    output_dir: str,
    thresholds: Optional[Dict[str, Tuple[str, float]]] = None,
    max_side: int = 1280,
) -> Dict[str, int]:
    """Save raw + overlay for every frame that fails a metric threshold."""
    return _save_examples(
        samples, images_rgb, pred_masks, gt_masks, per_image_metrics,
        output_dir,
        thresholds=thresholds or DEFAULT_FAILURE_THRESHOLDS,
        check_fn=_fails,
        good=False,
        max_side=max_side,
    )


def save_good_images(
    samples: list,
    images_rgb: List[np.ndarray],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    per_image_metrics: List[dict],
    output_dir: str,
    thresholds: Optional[Dict[str, Tuple[str, float]]] = None,
    max_side: int = 1280,
) -> Dict[str, int]:
    """Save raw + overlay for every frame that excels on a metric (green label)."""
    return _save_examples(
        samples, images_rgb, pred_masks, gt_masks, per_image_metrics,
        output_dir,
        thresholds=thresholds or DEFAULT_GOOD_THRESHOLDS,
        check_fn=_is_good,
        good=True,
        max_side=max_side,
    )
