"""Road readiness assessment: 4 layers, 17 metrics, mask-based."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import label
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# LAYER 1: Infrastructure (ground truth only)
# ---------------------------------------------------------------------------

def compute_i1_continuity(gt_mask: np.ndarray) -> float:
    """Lane continuity: % of boundary pixels that belong to segments >= 50px."""
    boundary = cv2.Canny((gt_mask * 255).astype(np.uint8), 50, 150)
    total_boundary = int(np.sum(boundary > 0))
    if total_boundary == 0:
        return 0.0

    labeled, _ = label(boundary)
    components = [int(np.sum(labeled == i)) for i in np.unique(labeled) if i > 0]
    continuous_pixels = sum(c for c in components if c >= 50)
    return continuous_pixels / total_boundary * 100.0


def compute_i2_contrast(image: np.ndarray, gt_mask: np.ndarray) -> float:
    """Boundary contrast: |mean_marking - mean_road| / 255."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    marking_pixels = gray[gt_mask == 1]
    road_pixels = gray[gt_mask == 0]

    if len(marking_pixels) == 0 or len(road_pixels) == 0:
        return 0.0

    return float(abs(np.mean(marking_pixels) - np.mean(road_pixels)) / 255.0)


def compute_i3_sharpness(image: np.ndarray, gt_mask: np.ndarray) -> float:
    """Boundary sharpness: image edge density within ±5px of GT boundary."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    boundary = cv2.Canny((gt_mask * 255).astype(np.uint8), 50, 150)
    if int(np.sum(boundary)) == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary_region = cv2.dilate(boundary, kernel)

    edges = cv2.Canny(gray, 50, 150)
    edge_in_boundary = int(np.sum((edges > 0) & (boundary_region > 0)))
    total_boundary = int(np.sum(boundary_region > 0))

    return (edge_in_boundary / total_boundary) if total_boundary > 0 else 0.0


def compute_i4_width_stability(gt_mask: np.ndarray) -> float:
    """Lane width stability: std of per-row lane width (pixels)."""
    widths = []
    for y in range(gt_mask.shape[0]):
        row = gt_mask[y, :]
        if np.sum(row) == 0:
            continue
        left = int(np.argmax(row > 0))
        right = int(len(row) - np.argmax(row[::-1] > 0))
        widths.append(right - left)

    return float(np.std(widths)) if len(widths) >= 2 else 0.0


def compute_i5_curvature(gt_mask: np.ndarray) -> float:
    """Road curvature: std of per-row horizontal center, normalized by width."""
    x_centers = []
    for y in range(gt_mask.shape[0]):
        row = gt_mask[y, :]
        if np.sum(row) > 0:
            x_center = float(np.mean(np.where(row > 0)[0]))
            x_centers.append(x_center)

    if len(x_centers) < 2:
        return 0.0

    return float(np.std(x_centers) / gt_mask.shape[1])


def compute_i6_lane_count(gt_mask: np.ndarray) -> int:
    """Ground truth lane count: number of connected components."""
    _, num_components = label(gt_mask)
    return int(num_components)


# ---------------------------------------------------------------------------
# LAYER 2: Detection performance (predictions vs GT)
# ---------------------------------------------------------------------------

def compute_d1_detection_rate(pred_masks: List[np.ndarray]) -> float:
    """Detection success rate: % of images with at least one predicted pixel."""
    if not pred_masks:
        return 0.0
    detected = sum(1 for pm in pred_masks if np.sum(pm) > 0)
    return detected / len(pred_masks) * 100.0


def compute_d2_lane_count_accuracy(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]
) -> float:
    """Lane count accuracy: % of images where predicted count equals GT count."""
    if not pred_masks:
        return 0.0
    correct = 0
    for pm, gm in zip(pred_masks, gt_masks):
        _, pred_count = label(pm)
        _, gt_count = label(gm)
        if pred_count == gt_count:
            correct += 1
    return correct / len(pred_masks) * 100.0


def compute_d3_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """Segmentation IoU: mean(TP / (TP + FP + FN)) across images."""
    ious = []
    for pm, gm in zip(pred_masks, gt_masks):
        tp = int(np.sum((pm > 0) & (gm > 0)))
        fp = int(np.sum((pm > 0) & (gm == 0)))
        fn = int(np.sum((pm == 0) & (gm > 0)))
        ious.append(tp / (tp + fp + fn + 1e-6))
    return float(np.mean(ious)) if ious else 0.0


def compute_d4_near_field_iou(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]
) -> float:
    """Near-field IoU: bottom half of image only."""
    ious = []
    for pm, gm in zip(pred_masks, gt_masks):
        h = pm.shape[0]
        pm_near = pm[h // 2 :, :]
        gm_near = gm[h // 2 :, :]
        tp = int(np.sum((pm_near > 0) & (gm_near > 0)))
        fp = int(np.sum((pm_near > 0) & (gm_near == 0)))
        fn = int(np.sum((pm_near == 0) & (gm_near > 0)))
        ious.append(tp / (tp + fp + fn + 1e-6))
    return float(np.mean(ious)) if ious else 0.0


def compute_d5_occlusion_gap(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    images: List[np.ndarray],
) -> float:
    """Occlusion robustness: mean (IoU_visible - IoU_occluded)."""
    gaps = []
    for pm, gm, img in zip(pred_masks, gt_masks, images):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        occluded = hsv[:, :, 2] < 100
        visible = ~occluded

        def _region_iou(mask: np.ndarray) -> float:
            tp = int(np.sum((pm > 0) & (gm > 0) & mask))
            union = int(np.sum(((pm > 0) | (gm > 0)) & mask))
            return tp / (union + 1e-6)

        iou_vis = _region_iou(visible) if np.sum(visible) > 0 else 0.0
        iou_occ = _region_iou(occluded) if np.sum(occluded) > 0 else 0.0
        gaps.append(iou_vis - iou_occ)

    return float(np.mean(gaps)) if gaps else 0.0


def compute_d6_detection_gap(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]
) -> float:
    """Detection gap: FN pixels / total GT pixels."""
    total_fn = 0
    total_gt = 0
    for pm, gm in zip(pred_masks, gt_masks):
        total_fn += int(np.sum((pm == 0) & (gm > 0)))
        total_gt += int(np.sum(gm > 0))
    return float(total_fn / (total_gt + 1e-6)) if total_gt > 0 else 0.0


def compute_d8_confidence_mean(pred_masks: Optional[List[np.ndarray]] = None) -> float:
    """Confidence mean: placeholder 0.5 until model confidence is integrated."""
    # TODO: extract per-pixel confidence from YOLOPv2 internals
    return 0.5


# ---------------------------------------------------------------------------
# LAYER 3: Correlation & bottleneck analysis
# ---------------------------------------------------------------------------

def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr, _ = pearsonr(x, y)
    return float(corr)


def compute_c1_continuity_correlation(
    infra_list: List[dict], detection_list: List[dict]
) -> float:
    """Pearson(I1, D1): continuity → detection success."""
    return _safe_pearsonr(
        np.array([m["I1"] for m in infra_list]),
        np.array([m["D1"] for m in detection_list]),
    )


def compute_c2_contrast_correlation(
    infra_list: List[dict], detection_list: List[dict]
) -> float:
    """Pearson(I2, D3): contrast → segmentation IoU."""
    return _safe_pearsonr(
        np.array([m["I2"] for m in infra_list]),
        np.array([m["D3"] for m in detection_list]),
    )


def compute_c3_sharpness_correlation(
    infra_list: List[dict], detection_list: List[dict]
) -> float:
    """Pearson(I3, D3): sharpness → segmentation IoU."""
    return _safe_pearsonr(
        np.array([m["I3"] for m in infra_list]),
        np.array([m["D3"] for m in detection_list]),
    )


def compute_c4_curvature_impact(
    infra_list: List[dict], detection_list: List[dict]
) -> float:
    """Curvature impact: mean D3 on straight roads minus mean D3 on curved roads."""
    i5 = np.array([m["I5"] for m in infra_list])
    d3 = np.array([m["D3"] for m in detection_list])
    if len(i5) < 4:
        return 0.0

    median_curv = np.median(i5)
    straight = i5 < median_curv
    curved = i5 >= median_curv

    perf_straight = float(np.mean(d3[straight])) if straight.any() else 0.0
    perf_curved = float(np.mean(d3[curved])) if curved.any() else 0.0
    return perf_straight - perf_curved


def compute_c5_degradation_ranking(
    infra_by_condition: dict, detection_by_condition: dict
) -> List[Tuple[str, float]]:
    """
    Rank conditions by combined infrastructure + performance drop vs. 'clear'.

    Args:
        infra_by_condition:     {condition: {I1, I2, I3, ...}}
        detection_by_condition: {condition: {D1, D3, D4, ...}}

    Returns:
        List[(condition, degradation_score)] sorted worst-first.
    """
    if "clear" not in infra_by_condition or "clear" not in detection_by_condition:
        return []

    def _mean_keys(d: dict, keys: List[str]) -> float:
        vals = [d.get(k, 0.0) for k in keys]
        return float(np.mean(vals)) if vals else 0.0

    clear_infra = _mean_keys(infra_by_condition["clear"], ["I1", "I2", "I3"])
    clear_perf = _mean_keys(detection_by_condition["clear"], ["D1", "D3", "D4"])

    degradation: dict[str, float] = {"clear": 0.0}
    for cond in infra_by_condition:
        if cond == "clear":
            continue
        delta_infra = clear_infra - _mean_keys(infra_by_condition[cond], ["I1", "I2", "I3"])
        delta_perf = clear_perf - _mean_keys(detection_by_condition[cond], ["D1", "D3", "D4"])
        degradation[cond] = delta_infra + delta_perf

    return sorted(degradation.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# LAYER 4: Readiness verdict
# ---------------------------------------------------------------------------

def compute_r1_infrastructure_score(
    infra: dict,
    weights: Optional[dict] = None,
) -> float:
    """Infrastructure readiness: weighted sum of normalised I1-I6 → 0-100."""
    if weights is None:
        weights = {"I1": 0.30, "I2": 0.30, "I3": 0.20, "I4": 0.10, "I5": 0.05, "I6": 0.05}

    i1 = infra["I1"] / 100.0
    i2 = infra["I2"]
    i3 = infra["I3"]
    i4 = max(1.0 - infra["I4"] / 50.0, 0.0)
    i5 = max(1.0 - infra["I5"] / 0.5, 0.0)
    i6 = 1.0 if infra["I6"] > 0 else 0.0

    total_w = sum(weights.values())
    r1 = (
        weights["I1"] * i1
        + weights["I2"] * i2
        + weights["I3"] * i3
        + weights["I4"] * i4
        + weights["I5"] * i5
        + weights["I6"] * i6
    ) / total_w

    return float(r1 * 100.0)


def compute_r2_detection_score(
    detection: dict,
    weights: Optional[dict] = None,
) -> float:
    """Detection readiness: weighted sum of normalised D1-D8 → 0-100."""
    if weights is None:
        weights = {
            "D1": 0.25, "D2": 0.20, "D3": 0.20, "D4": 0.15,
            "D5": 0.08, "D6": 0.08, "D8": 0.04,
        }

    d1 = detection["D1"] / 100.0
    d2 = detection["D2"] / 100.0
    d3 = detection["D3"]
    d4 = detection["D4"]
    d5 = 1.0 - float(np.clip(detection["D5"], 0.0, 1.0))
    d6 = 1.0 - float(np.clip(detection["D6"], 0.0, 1.0))
    d8 = detection["D8"]

    total_w = sum(weights.values())
    r2 = (
        weights["D1"] * d1
        + weights["D2"] * d2
        + weights["D3"] * d3
        + weights["D4"] * d4
        + weights["D5"] * d5
        + weights["D6"] * d6
        + weights["D8"] * d8
    ) / total_w

    return float(r2 * 100.0)


def compute_r3_bottleneck(correlation: dict) -> Tuple[str, float]:
    """Identify the metric with the largest limiting impact on readiness."""
    candidates = {
        "Continuity": abs(correlation.get("C1", 0.0)),
        "Contrast": abs(correlation.get("C2", 0.0)),
        "Sharpness": abs(correlation.get("C3", 0.0)),
        "Curvature": abs(correlation.get("C4", 0.0)),
    }

    c5 = correlation.get("C5", [])
    if c5:
        worst_condition, worst_score = c5[0]
        candidates[f"Condition ({worst_condition})"] = float(abs(worst_score))

    bottleneck = max(candidates, key=candidates.get)
    return (bottleneck, candidates[bottleneck])


def compute_r4_classification(
    r1: float, r2: float, r3: Tuple[str, float]
) -> dict:
    """Final readiness verdict: READY / MARGINAL / NOT_READY."""
    if r1 > 80 and r2 > 85:
        status = "READY"
        reason = f"Infrastructure ({r1:.1f}) and detection ({r2:.1f}) both strong"
    elif r1 > 60 or r2 > 70:
        status = "MARGINAL"
        if r1 < r2:
            reason = f"Infrastructure ({r1:.1f}) is bottleneck: {r3[0]} (strength={r3[1]:.2f})"
        else:
            reason = f"Detection ({r2:.1f}) is bottleneck: {r3[0]} (strength={r3[1]:.2f})"
    else:
        status = "NOT_READY"
        reason = f"Infrastructure ({r1:.1f}) and detection ({r2:.1f}) both insufficient"

    return {"status": status, "reason": reason}


# ---------------------------------------------------------------------------
# LAYER 5: Stratification filters
# ---------------------------------------------------------------------------

def filter_by_weather(samples, weather: str):
    """Filter LaneSamples by weather condition (clear/rainy/night/foggy)."""
    return [
        s for s in samples
        if s.target.meta.get("weather") == weather
        or s.meta.get("weather") == weather
    ]


def filter_by_context(samples, context: str):
    """Filter LaneSamples by inferred roadway context."""
    def _infer(sample) -> Optional[str]:
        scene = sample.target.meta.get("scene") or sample.meta.get("scene", "")
        if sample.target.mask is not None:
            _, lane_count = label(sample.target.mask)
        else:
            lane_count = 0

        if scene == "highway" and lane_count <= 2:
            return "rural_highway"
        if scene == "urban" and 2 <= lane_count <= 3:
            return "urban_street"
        if scene == "urban" and lane_count > 3:
            return "urban_intersection"
        return None

    return [s for s in samples if _infer(s) == context]


def filter_by_time(samples, time: str):
    """Filter LaneSamples by time of day (day/night)."""
    return [
        s for s in samples
        if s.target.meta.get("timeofday") == time
        or s.meta.get("timeofday") == time
    ]


def filter_by_traffic(samples, traffic: str):
    """Filter LaneSamples by inferred traffic density (low/medium/high)."""
    def _infer(sample) -> str:
        img_path = sample.image_path
        img = cv2.imread(img_path)
        if img is None:
            return "unknown"
        # imread returns BGR; convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        density = float(np.mean(hsv[:, :, 2] < 100))
        if density > 0.30:
            return "high"
        if density > 0.15:
            return "medium"
        return "low"

    return [s for s in samples if _infer(s) == traffic]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReadinessMetrics:
    """Road readiness assessment: 4 layers, 17 metrics."""

    # ------------------------------------------------------------------
    # Layer 1
    # ------------------------------------------------------------------

    def compute_infrastructure(self, image: np.ndarray, gt_mask: np.ndarray) -> dict:
        """Compute I1-I6 from image (H,W,3 uint8) and gt_mask (H,W uint8)."""
        return {
            "I1": compute_i1_continuity(gt_mask),
            "I2": compute_i2_contrast(image, gt_mask),
            "I3": compute_i3_sharpness(image, gt_mask),
            "I4": compute_i4_width_stability(gt_mask),
            "I5": compute_i5_curvature(gt_mask),
            "I6": compute_i6_lane_count(gt_mask),
        }

    # ------------------------------------------------------------------
    # Layer 2
    # ------------------------------------------------------------------

    def compute_detection(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
        images: List[np.ndarray],
    ) -> dict:
        """Compute D1-D8 from lists of predicted/GT masks and images."""
        return {
            "D1": compute_d1_detection_rate(pred_masks),
            "D2": compute_d2_lane_count_accuracy(pred_masks, gt_masks),
            "D3": compute_d3_iou(pred_masks, gt_masks),
            "D4": compute_d4_near_field_iou(pred_masks, gt_masks),
            "D5": compute_d5_occlusion_gap(pred_masks, gt_masks, images),
            "D6": compute_d6_detection_gap(pred_masks, gt_masks),
            "D8": compute_d8_confidence_mean(),
        }

    # ------------------------------------------------------------------
    # Layer 3
    # ------------------------------------------------------------------

    def compute_correlations(
        self,
        infra_list: List[dict],
        detection_list: List[dict],
        infra_by_condition: Optional[dict] = None,
        detection_by_condition: Optional[dict] = None,
    ) -> dict:
        """
        Compute C1-C5 from per-image infra and detection dicts.

        infra_list / detection_list: one dict per image (for C1-C4).
        infra_by_condition / detection_by_condition: {condition: avg_metrics} for C5.
        """
        return {
            "C1": compute_c1_continuity_correlation(infra_list, detection_list),
            "C2": compute_c2_contrast_correlation(infra_list, detection_list),
            "C3": compute_c3_sharpness_correlation(infra_list, detection_list),
            "C4": compute_c4_curvature_impact(infra_list, detection_list),
            "C5": compute_c5_degradation_ranking(
                infra_by_condition or {}, detection_by_condition or {}
            ),
        }

    # ------------------------------------------------------------------
    # Layer 4
    # ------------------------------------------------------------------

    def compute_verdict(self, infra: dict, detection: dict, correlation: dict) -> dict:
        """Compute R1-R4 from layer 1-3 aggregate results."""
        r1 = compute_r1_infrastructure_score(infra)
        r2 = compute_r2_detection_score(detection)
        r3 = compute_r3_bottleneck(correlation)
        r4 = compute_r4_classification(r1, r2, r3)
        return {"R1": r1, "R2": r2, "R3": r3, "R4": r4}

    # ------------------------------------------------------------------
    # Stratified evaluation
    # ------------------------------------------------------------------

    def evaluate_stratified(
        self,
        samples,
        pred_masks: List[np.ndarray],
        weather: Optional[str] = None,
        context: Optional[str] = None,
        time: Optional[str] = None,
        traffic: Optional[str] = None,
        min_samples: int = 10,
    ) -> Optional[dict]:
        """
        Compute all 4 layers for a specific stratum of samples.

        Applies each active filter in order; indices of pred_masks are kept in
        sync with filtered samples via an index list.

        Returns None when the stratum has fewer than min_samples valid images.
        """
        # Keep (sample, pred_mask) pairs together through filtering
        pairs = list(zip(samples, pred_masks))

        if weather:
            filtered_samples = filter_by_weather([p[0] for p in pairs], weather)
            pairs = [(s, p) for s, p in pairs if s in set(filtered_samples)]
        if context:
            filtered_samples = filter_by_context([p[0] for p in pairs], context)
            pairs = [(s, p) for s, p in pairs if s in set(filtered_samples)]
        if time:
            filtered_samples = filter_by_time([p[0] for p in pairs], time)
            pairs = [(s, p) for s, p in pairs if s in set(filtered_samples)]
        if traffic:
            filtered_samples = filter_by_traffic([p[0] for p in pairs], traffic)
            pairs = [(s, p) for s, p in pairs if s in set(filtered_samples)]

        if len(pairs) < min_samples:
            return None

        layer1_list: List[dict] = []
        used_pred_masks: List[np.ndarray] = []
        gt_masks: List[np.ndarray] = []
        images: List[np.ndarray] = []

        for sample, pred_mask in pairs:
            if sample.target.mask is None:
                continue

            img = cv2.imread(sample.image_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            infra = self.compute_infrastructure(img_rgb, sample.target.mask)
            layer1_list.append(infra)
            used_pred_masks.append(pred_mask)
            gt_masks.append(sample.target.mask)
            images.append(img_rgb)

        if len(layer1_list) < min_samples:
            return None

        layer2 = self.compute_detection(used_pred_masks, gt_masks, images)

        # Per-image detection dict for correlation: replicate aggregate per image
        layer2_per_image = [layer2] * len(layer1_list)

        layer3 = self.compute_correlations(layer1_list, layer2_per_image)

        avg_infra = {k: float(np.mean([m[k] for m in layer1_list])) for k in layer1_list[0]}
        layer4 = self.compute_verdict(avg_infra, layer2, layer3)

        parts = []
        if weather:
            parts.append(f"weather={weather}")
        if context:
            parts.append(f"context={context}")
        if time:
            parts.append(f"time={time}")
        if traffic:
            parts.append(f"traffic={traffic}")
        stratum_label = "_".join(parts) if parts else "all"

        return {
            "layer1": layer1_list,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
            "stratum_label": stratum_label,
            "num_samples": len(layer1_list),
        }
