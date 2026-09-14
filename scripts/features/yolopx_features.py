"""Extract and render YOLOPX intermediate features on TuSimple images.

Produces the figure set used by the "how the model sees lane lines" slide deck:
per-stage activation maps along the encoder and the lane-line decoder, single
channel grids, Grad-CAM at several depths, occlusion sensitivity, and a
channel-selectivity ranking against the ground-truth lane mask.

Model loading / letterboxing / normalisation mirror evaluation/run_readiness.py
so the features shown are exactly the ones the evaluation pipeline consumes.

Usage:
    python -m scripts.features.yolopx_features --out outputs/feature_slides
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.features import viz_common as V  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Block indices in lib/models/YOLOP.py (MCnet.model), verified by forward hook.
ENCODER_STAGES = [
    (0, 0, "Backbone C2", "1/4 scale | 256ch"),
    (0, 1, "Backbone C3", "1/8 scale | 512ch"),
    (0, 2, "Backbone C4", "1/16 scale | 1024ch"),
    (0, 3, "Backbone C5", "1/32 scale | 1024ch"),
]
NECK_STAGES = [
    (17, None, "Neck C2", "1/4 scale | 256ch"),
    (3, None, "Neck P3", "1/8 scale | 256ch"),
    (4, None, "Neck P4", "1/16 scale | 512ch"),
]
LANE_HEAD_STAGES = [
    (18, None, "LL conv 1/4", "128ch"),
    (21, None, "LL merge C2+P3", "128ch"),
    (22, None, "LL ELAN block", "64ch"),
    (23, None, "LL PSA attention", "64ch"),
    (27, None, "LL ELAN (1/2)", "8ch"),
    (28, None, "LL PSA (1/2)", "8ch"),
    (30, None, "LL logits", "2ch, full res"),
]
LANE_LOGIT_IDX = 30      # pre-activation 2-channel lane logits
LANE_OUT_IDX = 31        # sigmoid'd lane output
SELECTIVITY_IDX = 22     # 64-channel lane-decoder block used for channel ranking


def load_model(repo: str, weights: str):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from lib.models import get_net

    model = get_net(cfg=None)
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def preprocess(img_bgr: np.ndarray, repo: str, img_size: int = 640):
    from lib.utils import letterbox_for_img

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb, ratio, pad = letterbox_for_img(rgb, img_size, auto=True)
    t = torch.tensor(np.ascontiguousarray(lb, dtype=np.float32) / 255.0).permute(2, 0, 1)
    t = ((t - MEAN) / STD)[None]
    lb_bgr = cv2.cvtColor(lb, cv2.COLOR_RGB2BGR)
    return t, lb_bgr, ratio, pad


class Tap:
    """Forward hooks over every MCnet block."""

    def __init__(self, model):
        self.acts = {}
        self.handles = [
            blk.register_forward_hook(
                lambda mod, inp, out, i=i: self.acts.__setitem__(i, out)
            )
            for i, blk in enumerate(model.model)
        ]

    def get(self, idx, sub=None):
        o = self.acts[idx]
        if sub is not None:
            o = o[sub]
        return o

    def close(self):
        for h in self.handles:
            h.remove()


def feat_np(t: torch.Tensor) -> np.ndarray:
    return t.detach()[0].float().cpu().numpy()


def stage_tile(img_lb, feat, title, sub, size=(360, 216), alpha=0.6):
    energy = V.normalize(V.channel_energy_map(feat))
    tile = V.blend(cv2.resize(img_lb, size), energy, alpha=alpha)
    return V.label_tile(tile, title, sub)


def lane_prob_from(out31: torch.Tensor) -> np.ndarray:
    """Lane-class probability map in letterbox space (matches run_readiness)."""
    p = torch.softmax(out31.float(), dim=1)[0, 1]
    return p.detach().cpu().numpy()


# ---------------------------------------------------------------- figures ----
def fig_pipeline(img_lb, tap, out_dir, stem):
    enc = [stage_tile(img_lb, feat_np(tap.get(i, s)), t, sb)
           for i, s, t, sb in ENCODER_STAGES]
    neck = [stage_tile(img_lb, feat_np(tap.get(i, s)), t, sb)
            for i, s, t, sb in NECK_STAGES]
    head = [stage_tile(img_lb, feat_np(tap.get(i, s)), t, sb)
            for i, s, t, sb in LANE_HEAD_STAGES]

    inp = V.label_tile(cv2.resize(img_lb, (360, 216)), "Input (letterboxed)",
                       "384 x 640")
    prob = lane_prob_from(tap.get(LANE_OUT_IDX))
    final = V.label_tile(prob_overlay(cv2.resize(img_lb, (360, 216)), prob),
                         "Lane-line probability", "argmax -> binary mask")

    rows = [
        V.banner("1. Shared encoder: ELANNet backbone -> PaFPN neck", 2400),
        V.hstack_pad([inp] + enc + [V.arrow(inp.shape[0])] + neck),
        V.banner("2. Lane-line decoder branch: progressive upsampling + C2 skip + PSA attention", 2400),
        V.hstack_pad(head + [V.arrow(head[0].shape[0]), final]),
    ]
    return V.save(out_dir / f"{stem}_yolopx_pipeline.png", V.vstack_pad(rows, 16))


def fig_channels(img_lb, tap, out_dir, stem, idx, n=16, title=""):
    feat = feat_np(tap.get(idx))
    order = np.argsort(-feat.std(axis=(1, 2)))[:n]
    tiles = []
    for c in order:
        m = V.normalize(feat[c])
        tiles.append(V.label_tile(V.heat(m, (200, 120), cv2.COLORMAP_INFERNO),
                                  f"ch {int(c)}", bar_h=24))
    body = V.grid(tiles, 8)
    head = V.banner(title, body.shape[1])
    return V.save(out_dir / f"{stem}_yolopx_channels_{idx}.png",
                  V.vstack_pad([head, body], 8))


def grad_cam(model, x, target_indices, lane_mask_lb):
    """Grad-CAM at several block outputs w.r.t. the lane-line logit."""
    feats, grads = {}, {}
    handles = []
    for i in target_indices:
        def fwd(mod, inp, out, i=i):
            feats[i] = out
            out.register_hook(lambda g, i=i: grads.__setitem__(i, g))
        handles.append(model.model[i].register_forward_hook(fwd))

    x = x.clone().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    out = model(x)
    logits = feats[LANE_LOGIT_IDX] if LANE_LOGIT_IDX in feats else None
    if logits is None:
        raise RuntimeError("lane logit block not tapped")
    sel = torch.tensor(lane_mask_lb[None, None], dtype=torch.float32)
    score = ((logits[:, 1:2] - logits[:, 0:1]) * sel).sum()
    score.backward()

    cams = {}
    for i in target_indices:
        a = feats[i].detach()[0]
        g = grads[i].detach()[0]
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * a).sum(0)).cpu().numpy()
        cams[i] = V.normalize(cam)
    for h in handles:
        h.remove()
    return cams


def fig_gradcam(model, x, img_lb, lane_mask_lb, out_dir, stem):
    targets = [(0, "Backbone C3-C5 (block 0)"), (4, "Neck P4 (1/16)"),
               (17, "Neck C2 (1/4)"), (22, "Lane decoder ELAN"),
               (28, "Lane decoder PSA (1/2)"), (LANE_LOGIT_IDX, "logits")]
    idxs = [t[0] for t in targets if t[0] != 0] + [LANE_LOGIT_IDX]
    idxs = sorted(set(idxs + [3, 4, 17, 22, 28]))
    cams = grad_cam(model, x, idxs, lane_mask_lb)

    show = [(3, "Neck P3  (1/8)"), (4, "Neck P4  (1/16)"), (17, "Neck C2  (1/4)"),
            (22, "Lane decoder ELAN  (1/4)"), (28, "Lane decoder PSA  (1/2)")]
    tiles = [V.label_tile(cv2.resize(img_lb, (360, 216)), "Input", "reference")]
    for i, name in show:
        tiles.append(V.label_tile(
            V.blend(cv2.resize(img_lb, (360, 216)), cams[i], 0.6, cv2.COLORMAP_TURBO),
            name, "Grad-CAM w.r.t. lane logit"))
    body = V.grid(tiles, 3)
    head = V.banner("Grad-CAM: which spatial evidence drives the lane-line logit",
                    body.shape[1])
    return V.save(out_dir / f"{stem}_yolopx_gradcam.png",
                  V.vstack_pad([head, body], 8))


def prob_overlay(img_bgr, prob, thresh=0.15, cmap=cv2.COLORMAP_TURBO):
    """Overlay a probability map only where it is actually above `thresh`."""
    h, w = img_bgr.shape[:2]
    p = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    hm = cv2.applyColorMap((np.clip(p, 0, 1) * 255).astype(np.uint8), cmap)
    a = np.clip((p - thresh) / max(1e-6, 1.0 - thresh), 0, 1)[..., None] * 0.85
    return (img_bgr * (1 - a) + hm * a).astype(np.uint8)


def fig_occlusion(model, x, img_lb, out_dir, stem, patch=40, stride=16):
    """Slide a grey patch over the input and measure what the lane head loses."""
    with torch.no_grad():
        base = lane_prob_from(model(x)[2])
    base_t = torch.tensor(base)
    base_mass = float(base_t.sum())
    _, _, H, W = x.shape
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    drop = np.zeros((len(ys), len(xs)), np.float32)
    change = np.zeros((len(ys), len(xs)), np.float32)
    grey = ((torch.zeros(3, patch, patch) + 0.5) - MEAN) / STD
    for iy, y in enumerate(ys):
        batch = []
        for xx in xs:
            xo = x.clone()
            xo[0, :, y:y + patch, xx:xx + patch] = grey
            batch.append(xo)
        with torch.no_grad():
            probs = torch.softmax(model(torch.cat(batch))[2].float(), dim=1)[:, 1]
        drop[iy] = base_mass - probs.sum(dim=(1, 2)).numpy()
        change[iy] = (probs - base_t[None]).abs().sum(dim=(1, 2)).numpy()

    size = (480, 288)
    tiles = [
        V.label_tile(cv2.resize(img_lb, size), "Input", "TuSimple frame"),
        V.label_tile(prob_overlay(cv2.resize(img_lb, size), base),
                     "Lane probability", "unoccluded reference"),
        V.label_tile(V.blend(cv2.resize(img_lb, size),
                             V.normalize(np.maximum(drop, 0)), 0.6,
                             cv2.COLORMAP_TURBO),
                     "Lane evidence lost", "hot = hiding it removes lane pixels"),
        V.label_tile(V.blend(cv2.resize(img_lb, size), V.normalize(change), 0.6,
                             cv2.COLORMAP_TURBO),
                     "Total prediction change", "hot = hiding it also creates errors"),
    ]
    body = V.grid(tiles, 2)
    head = V.banner("Occlusion test: hiding which pixels changes the lane prediction",
                    body.shape[1],
                    sub=f"{patch}px grey patch, stride {stride}, "
                        f"{len(ys) * len(xs)} forward passes")
    return V.save(out_dir / f"{stem}_yolopx_occlusion.png",
                  V.vstack_pad([head, body], 8))


def lane_prob_from_batch(out31: torch.Tensor) -> np.ndarray:
    p = torch.softmax(out31.float(), dim=1)[:, 1]
    return p.sum(dim=(1, 2)).cpu().numpy()


def fig_selectivity(img_lb, tap, gt_mask_lb, out_dir, stem, idx=SELECTIVITY_IDX):
    """Rank channels of a lane-decoder block by agreement with the GT lane mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat = feat_np(tap.get(idx))
    C, h, w = feat.shape
    gt = cv2.resize(gt_mask_lb.astype(np.float32), (w, h), cv2.INTER_AREA)
    gt = (gt > 0.1).astype(np.float32)
    gt_c = gt - gt.mean()
    corr = np.zeros(C, np.float32)
    for c in range(C):
        f = feat[c] - feat[c].mean()
        d = (np.linalg.norm(f) * np.linalg.norm(gt_c))
        corr[c] = float((f * gt_c).sum() / d) if d > 1e-8 else 0.0
    order = np.argsort(-corr)

    fig, ax = plt.subplots(figsize=(11, 2.9), dpi=150)
    ax.bar(np.arange(C), corr[order], color=["#d1603d" if v > 0 else "#5b8db8"
                                             for v in corr[order]], width=0.9)
    ax.set_xlabel(f"channels of block {idx} (sorted)", fontsize=9)
    ax.set_ylabel("corr. with GT lane mask", fontsize=9)
    ax.set_title(f"Lane selectivity of the {C} channels in the YOLOPX lane decoder "
                 f"(block {idx})", fontsize=10)
    ax.axhline(0, color="#333", lw=0.8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    tmp = out_dir / f"{stem}_yolopx_selectivity_bar.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tmp)
    plt.close(fig)
    bar = cv2.imread(str(tmp))

    def row(chans):
        tiles = []
        for c in chans:
            m = V.normalize(feat[c])
            tiles.append(V.label_tile(
                V.blend(cv2.resize(img_lb, (240, 144)), m, 0.62),
                f"ch {int(c)}   r = {corr[c]:+.2f}", bar_h=28))
        return V.hstack_pad(tiles)

    body = V.vstack_pad([
        V.banner("Most lane-selective channels", bar.shape[1],
                 sub="fire exactly on the painted markings"),
        row(order[:6]),
        V.banner("Least lane-selective channels", bar.shape[1],
                 sub="encode road surface / background, i.e. the inverse of the lanes"),
        row(order[-6:]),
    ], 10)
    out = V.vstack_pad([bar, body], 12)
    return V.save(out_dir / f"{stem}_yolopx_selectivity.png", out)


def rasterize_lanes_lb(lanes, h_samples, ratio, pad, shape_lb, thickness=6):
    """TuSimple GT polylines -> binary mask in letterbox space."""
    H, W = shape_lb[:2]
    mask = np.zeros((H, W), np.uint8)
    rw, rh = ratio if isinstance(ratio, (tuple, list)) else (ratio, ratio)
    pw, ph = pad
    for lane in lanes:
        pts = [[int(round(x * rw + pw)), int(round(y * rh + ph))]
               for x, y in zip(lane, h_samples) if x >= 0]
        if len(pts) >= 2:
            cv2.polylines(mask, [np.asarray(pts, np.int32)], False, 1, thickness)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/gauravb/Projects/road_readiness_t3/YOLOPX")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tusimple-root", default="/shared/data/TUSimple/test_set")
    ap.add_argument("--tusimple-label", default="/shared/data/TUSimple/test_label.json")
    ap.add_argument("--indices", default="0,580,1160")
    ap.add_argument("--out", default="outputs/feature_slides")
    ap.add_argument("--skip-occlusion", action="store_true")
    args = ap.parse_args()

    weights = args.weights or str(Path(args.repo) / "weights" / "epoch-195.pth")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(args.tusimple_label)]
    model = load_model(args.repo, weights)

    written = {}
    for si in [int(s) for s in args.indices.split(",")]:
        rec = recs[si]
        stem = f"ts{si:04d}"
        img_path = Path(args.tusimple_root) / rec["raw_file"]
        img = cv2.imread(str(img_path))
        x, img_lb, ratio, pad = preprocess(img, args.repo)
        gt_lb = rasterize_lanes_lb(rec["lanes"], rec["h_samples"], ratio, pad,
                                   img_lb.shape)

        tap = Tap(model)
        with torch.no_grad():
            model(x)
        prob = lane_prob_from(tap.get(LANE_OUT_IDX))
        pred_mask = (prob > 0.5).astype(np.float32)

        files = [
            fig_pipeline(img_lb, tap, out_dir, stem),
            fig_channels(img_lb, tap, out_dir, stem, 3,
                         title="Neck P3 (1/8) individual channels: generic edges, texture, "
                               "road/sky split - nothing lane-specific yet"),
            fig_channels(img_lb, tap, out_dir, stem, SELECTIVITY_IDX,
                         title="Lane decoder block 22 (1/4) individual channels: lane-shaped "
                               "responses have emerged"),
            fig_selectivity(img_lb, tap, gt_lb, out_dir, stem),
        ]
        tap.close()
        files.append(fig_gradcam(model, x, img_lb, pred_mask, out_dir, stem))
        if not args.skip_occlusion:
            files.append(fig_occlusion(model, x, img_lb, out_dir, stem))

        # reference overlay: GT vs prediction, letterbox space
        ov = img_lb.copy()
        ov[gt_lb > 0] = (0.35 * ov[gt_lb > 0] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
        ov[pred_mask > 0] = (0.35 * ov[pred_mask > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        files.append(V.save(out_dir / f"{stem}_yolopx_result.png",
                            V.label_tile(cv2.resize(ov, (720, 432)),
                                         "YOLOPX lane segmentation",
                                         "green = TuSimple GT, red = prediction")))
        written[stem] = files
        print(f"[{stem}] {img_path}")
        for f in files:
            print("   ", f)

    (out_dir / "yolopx_manifest.json").write_text(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
