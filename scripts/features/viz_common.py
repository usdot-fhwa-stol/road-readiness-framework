"""Shared helpers for the CLRerNet / YOLOPX intermediate-feature slide figures.

Dependency-light on purpose: this module is imported from two different Python
environments (the repo env for YOLOPX, the CLRerNet venv for CLRerNet), so it
sticks to numpy / cv2 / matplotlib only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Slide palette (BGR for cv2, hex for matplotlib).
INK = (28, 28, 30)
PAPER = (250, 250, 249)
ACCENT = (60, 120, 240)  # BGR -> orange-ish accent
FONT = cv2.FONT_HERSHEY_SIMPLEX


def normalize(x: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Percentile-normalise an activation map to [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    if hi - lo < 1e-8:
        hi = float(x.max())
        lo = float(x.min())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def channel_energy_map(feat: np.ndarray) -> np.ndarray:
    """Mean absolute activation across channels: (C, H, W) -> (H, W)."""
    return np.abs(np.asarray(feat, dtype=np.float32)).mean(axis=0)


def heat(x01: np.ndarray, size_wh, cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """[0,1] map -> BGR colormap image resized to (W, H)."""
    u8 = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.resize(u8, size_wh, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(u8, cmap)


def blend(img_bgr: np.ndarray, x01: np.ndarray, alpha: float = 0.55,
          cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Overlay a [0,1] activation map on an image."""
    h, w = img_bgr.shape[:2]
    hm = heat(x01, (w, h), cmap)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, hm, alpha, 0.0)


def label_tile(tile: np.ndarray, title: str, subtitle: str = "",
               bar_h: int = 34) -> np.ndarray:
    """Add a caption bar under a tile."""
    h, w = tile.shape[:2]
    extra = bar_h if not subtitle else bar_h + 22
    out = np.full((h + extra, w, 3), 255, np.uint8)
    out[:h] = tile
    cv2.putText(out, title, (8, h + 23), FONT, 0.62, INK, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (8, h + 45), FONT, 0.48, (110, 110, 110), 1,
                    cv2.LINE_AA)
    return out


def hstack_pad(tiles, gap: int = 12, bg: int = 255) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    parts = []
    for i, t in enumerate(tiles):
        if t.shape[0] != h:
            pad = np.full((h - t.shape[0], t.shape[1], 3), bg, np.uint8)
            t = np.vstack([t, pad])
        parts.append(t)
        if i != len(tiles) - 1:
            parts.append(np.full((h, gap, 3), bg, np.uint8))
    return np.hstack(parts)


def vstack_pad(rows, gap: int = 12, bg: int = 255) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    parts = []
    for i, r in enumerate(rows):
        if r.shape[1] != w:
            pad = np.full((r.shape[0], w - r.shape[1], 3), bg, np.uint8)
            r = np.hstack([r, pad])
        parts.append(r)
        if i != len(rows) - 1:
            parts.append(np.full((gap, w, 3), bg, np.uint8))
    return np.vstack(parts)


def grid(tiles, ncols: int, gap: int = 8) -> np.ndarray:
    rows = [hstack_pad(tiles[i:i + ncols], gap) for i in range(0, len(tiles), ncols)]
    return vstack_pad(rows, gap)


def banner(text: str, width: int, height: int = 46, sub: str = "") -> np.ndarray:
    """Title bar that shrinks its font until the text fits `width`."""
    out = np.full((height, width, 3), 255, np.uint8)
    scale, sub_scale = 0.85, 0.55
    while scale > 0.35:
        tw = cv2.getTextSize(text, FONT, scale, 2)[0][0]
        sw = cv2.getTextSize(sub, FONT, sub_scale, 1)[0][0] + 10 if sub else 0
        if 8 + tw + sw + 8 <= width:
            break
        scale -= 0.05
        sub_scale = max(0.4, scale * 0.65)
    cv2.putText(out, text, (8, 30), FONT, scale, INK, 2, cv2.LINE_AA)
    if sub:
        tw = cv2.getTextSize(text, FONT, scale, 2)[0][0]
        cv2.putText(out, sub, (18 + tw, 30), FONT, sub_scale, (120, 120, 120), 1,
                    cv2.LINE_AA)
    return out


def arrow(height: int, width: int = 40) -> np.ndarray:
    out = np.full((height, width, 3), 255, np.uint8)
    y = height // 2
    cv2.arrowedLine(out, (6, y), (width - 6, y), (150, 150, 150), 3,
                    tipLength=0.45)
    return out


def save(path, img) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


def draw_lanes(img_bgr: np.ndarray, lanes, color=(0, 0, 255), thickness: int = 3,
               points: bool = False) -> np.ndarray:
    """Draw a list of polylines given as [(x, y), ...] in image coordinates."""
    out = img_bgr.copy()
    for lane in lanes:
        pts = np.asarray([[int(round(x)), int(round(y))] for x, y in lane],
                         dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
        if points:
            for x, y in pts:
                cv2.circle(out, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    return out


def colorwheel(n: int):
    """n visually distinct BGR colours."""
    hsv = np.zeros((1, n, 3), np.uint8)
    hsv[0, :, 0] = np.linspace(0, 170, n, dtype=np.uint8)
    hsv[0, :, 1] = 235
    hsv[0, :, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    return [tuple(int(c) for c in bgr[i]) for i in range(n)]
