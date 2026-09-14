"""Extract and render CLRerNet intermediate features on TuSimple images.

Companion to scripts/features/yolopx_features.py. CLRerNet is a row-anchor lane
*detector* (not a segmenter), so the interesting internals are different: the
DLA-34 pyramid, the 64-channel FPN levels the head samples from, the learned
priors, ROI pooling along each prior, the ROIGather cross-attention, and the
three coarse-to-fine refinement stages.

Must be run with the CLRerNet venv, e.g.

    SP=/tmp/scratch /shared/src/CLRerNet/clrernet/bin/python \
        scripts/features/clrernet_features.py --out outputs/feature_slides

Note: CLRerNet's lane NMS extension is CUDA-only. On CPU the script sets the
model's own ``test_cfg.use_nms = False`` and applies a clearly-labelled
display-only dedup so the figures still show a clean final lane set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.features import viz_common as V  # noqa: E402

CROP_SIZE = (800, 320)          # model input (W, H)
LETTERBOX = (1640, 590)         # CLRerNet canonical CULane frame
CUT_HEIGHT = 270                # rows cropped off the top before resize


# ------------------------------------------------------------------ model ----
def build_model(clrernet_root, config, checkpoint, device, scratch):
    sys.path.insert(0, str(clrernet_root))
    os.chdir(str(clrernet_root))
    from mmdet.apis import init_detector
    from mmengine.config import Config

    dummy = Path(scratch) / "clrernet_dummy_list.txt"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.write_text("dummy.jpg\n")
    cfg = Config.fromfile(str(config))
    cfg.test_dataloader.dataset.data_list = str(dummy)
    model = init_detector(cfg, str(checkpoint), device=device)
    model.bbox_head.test_cfg.as_lanes = False
    if device == "cpu":
        # the lane-NMS CUDA extension cannot run here
        model.bbox_head.test_cfg.use_nms = False
    return model


class Tap:
    """Capture every tensor the slides need out of one forward pass."""

    def __init__(self, model):
        self.d = {}
        self.handles = []
        h = self.handles.append
        h(model.data_preprocessor.register_forward_hook(
            lambda m, i, o: self.d.__setitem__("input", o["inputs"])))
        h(model.backbone.register_forward_hook(
            lambda m, i, o: self.d.__setitem__("backbone", o)))
        h(model.neck.register_forward_hook(
            lambda m, i, o: self.d.__setitem__("neck", o)))
        h(model.bbox_head.register_forward_hook(
            lambda m, i, o: self.d.__setitem__("stages", o)))

        # ROI pooling: record the prior x positions sampled at every stage.
        head = model.bbox_head
        self.pooled, self.prior_xs = [], []
        orig_pool = head.pool_prior_features

        def pool(batch_features, prior_xs, _o=orig_pool):
            self.prior_xs.append(prior_xs.detach().clone())
            out = _o(batch_features, prior_xs)
            self.pooled.append(out.detach().clone())
            return out

        head.pool_prior_features = pool
        self._restore_pool = (head, orig_pool)

        # ROIGather cross-attention: the sim_map is internal, so mirror the
        # module's own forward and stash it.
        attn = head.attention.attention  # AnchorVecFeatureMapAttention
        self.sim_maps = []
        orig_attn = attn.forward

        def attn_fwd(roi, fmap, _a=attn):
            import torch.nn.functional as F
            query = _a.f_query(roi)
            key = _a.resize(_a.f_key(fmap))
            value = _a.resize(_a.f_value(fmap)).permute(0, 2, 1)
            sim_map = F.softmax((_a.dim ** -0.5) * torch.matmul(query, key), dim=-1)
            self.sim_maps.append(sim_map.detach().clone())
            return _a.W(torch.matmul(sim_map, value))

        attn.forward = attn_fwd
        self._restore_attn = (attn, orig_attn)

    def close(self):
        for h in self.handles:
            h.remove()
        self._restore_pool[0].pool_prior_features = self._restore_pool[1]
        self._restore_attn[0].forward = self._restore_attn[1]


def run_inference(model, img_path, scratch):
    """Mirror libs.api.inference.inference_one_image, keeping scores."""
    from libs.api.inference import letterbox_to_size, get_prediction
    from libs.datasets.pipelines import Compose

    img = cv2.imread(str(img_path))
    lb, scale, pad = letterbox_to_size(img)
    ori_shape = lb.shape
    data = dict(filename=str(img_path), sub_img_name=None, img=lb, gt_points=[],
                id_classes=[], id_instances=[], img_shape=ori_shape,
                ori_shape=ori_shape)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    data = pipeline(data)
    with torch.no_grad():
        results = model.test_step(dict(inputs=[data["inputs"]],
                                       data_samples=[data["data_samples"]]))
    lanes = get_prediction(results[0]["lanes"], ori_shape[0], ori_shape[1])
    scores = [float(s) for s in results[0]["scores"]]
    return img, lb, scale, pad, lanes, scores


# ------------------------------------------------------------- geometry ------
def crop_image(lb_bgr):
    """The exact 800x320 tensor the network sees, as a viewable BGR image."""
    return cv2.resize(lb_bgr[CUT_HEIGHT:LETTERBOX[1]], CROP_SIZE,
                      interpolation=cv2.INTER_LINEAR)


def lane_polyline_crop(xs, start, length, n_strips=71):
    """Head output (normalised xs + start row + length) -> crop-space polyline."""
    W, H = CROP_SIZE
    ys = np.linspace(1.0, 0.0, len(xs))
    end = min(start + length - 1, len(xs) - 1)
    if end <= start:
        return None
    seg_x, seg_y = xs[start:end + 1], ys[start:end + 1]
    keep = (seg_x >= 0) & (seg_x <= 1)
    if keep.sum() < 2:
        return None
    return list(zip(seg_x[keep] * W, seg_y[keep] * H))


def stage_lanes(stage_dict, conf=0.0, n_strips=71):
    """Decode one refinement stage into (polyline, score) pairs in crop space."""
    logits = stage_dict["cls_logits"][0]
    scores = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    xs = stage_dict["xs"][0].detach().cpu().numpy()
    params = stage_dict["anchor_params"][0].detach().cpu().numpy()
    lengths = np.round(stage_dict["lengths"][0, :, 0].detach().cpu().numpy() * n_strips)
    out = []
    for i in range(xs.shape[0]):
        if scores[i] < conf:
            continue
        start = int(np.clip(round((1 - params[i, 0]) * n_strips), 0, n_strips))
        poly = lane_polyline_crop(xs[i], start, int(lengths[i]), n_strips)
        if poly is not None:
            out.append((poly, float(scores[i]), i))
    return out


def display_nms(cands, x_thresh=0.06):
    """Greedy dedup for display only (the real CUDA lane-NMS needs a GPU)."""
    kept = []
    for poly, score, idx in sorted(cands, key=lambda c: -c[1]):
        p = np.asarray(poly)
        ok = True
        for kpoly, _, _ in kept:
            k = np.asarray(kpoly)
            lo, hi = max(p[:, 1].min(), k[:, 1].min()), min(p[:, 1].max(), k[:, 1].max())
            if hi - lo < 20:
                continue
            grid = np.linspace(lo, hi, 20)
            dx = np.abs(np.interp(grid, p[::-1, 1], p[::-1, 0])
                        - np.interp(grid, k[::-1, 1], k[::-1, 0]))
            if dx.mean() < x_thresh * CROP_SIZE[0]:
                ok = False
                break
        if ok:
            kept.append((poly, score, idx))
    return kept


def draw_scored(base, cands, thickness=2, cmap=cv2.COLORMAP_TURBO):
    out = base.copy()
    for poly, score, _ in sorted(cands, key=lambda c: c[1]):
        c = cv2.applyColorMap(np.uint8([[score * 255]]), cmap)[0, 0]
        pts = np.asarray([[int(round(x)), int(round(y))] for x, y in poly], np.int32)
        cv2.polylines(out, [pts], False, tuple(int(v) for v in c), thickness,
                      cv2.LINE_AA)
    return out


# -------------------------------------------------------------- figures ------
def feat_np(t):
    return t.detach()[0].float().cpu().numpy()


def stage_tile(base, feat, title, sub, size=(400, 160), alpha=0.6):
    e = V.normalize(V.channel_energy_map(feat))
    return V.label_tile(V.blend(cv2.resize(base, size), e, alpha), title, sub)


def fig_pipeline(crop, tap, cands_final, out_dir, stem):
    bb = tap.d["backbone"]
    nk = tap.d["neck"]
    enc = [
        stage_tile(crop, feat_np(bb[0]), "DLA-34 level 3", "1/8 scale | 128ch"),
        stage_tile(crop, feat_np(bb[1]), "DLA-34 level 4", "1/16 scale | 256ch"),
        stage_tile(crop, feat_np(bb[2]), "DLA-34 level 5", "1/32 scale | 512ch"),
    ]
    neck = [
        stage_tile(crop, feat_np(nk[0]), "FPN P0  (40x100)", "64ch - refine stage 3"),
        stage_tile(crop, feat_np(nk[1]), "FPN P1  (20x50)", "64ch - refine stage 2"),
        stage_tile(crop, feat_np(nk[2]), "FPN P2  (10x25)", "64ch - refine stage 1"),
    ]
    inp = V.label_tile(cv2.resize(crop, (400, 160)), "Model input",
                       "800x320 crop of the letterboxed frame")
    fin = V.label_tile(draw_scored(cv2.resize(crop, (400, 160)),
                                   [(np.asarray(p) * [400 / 800, 160 / 320], s, i)
                                    for p, s, i in cands_final], 2),
                       "Detected lanes", "colour = confidence")
    rows = [
        V.banner("1. DLA-34 backbone: aggregated multi-scale features", 2200),
        V.hstack_pad([inp] + enc),
        V.banner("2. FPN neck: three 64-channel maps, one per refinement stage", 2200),
        V.hstack_pad(neck + [V.arrow(neck[0].shape[0]), fin]),
    ]
    return V.save(out_dir / f"{stem}_clrernet_pipeline.png", V.vstack_pad(rows, 16))


def fig_refinement(crop, model, tap, out_dir, stem, conf):
    stages = tap.d["stages"]
    size = (520, 208)
    sx, sy = size[0] / CROP_SIZE[0], size[1] / CROP_SIZE[1]

    def scale(cands):
        return [([(x * sx, y * sy) for x, y in p], s, i) for p, s, i in cands]

    # the learned priors, before any refinement
    head = model.bbox_head
    with torch.no_grad():
        anchor_xs, _ = head.anchor_generator.generate_anchors(
            head.anchor_generator.prior_embeddings.weight, head.prior_ys,
            head.sample_x_indices, head.img_w, head.img_h)
    prior_polys = []
    axs = anchor_xs.cpu().numpy()
    for i in range(axs.shape[0]):
        p = lane_polyline_crop(axs[i], 0, len(axs[i]))
        if p is not None:
            prior_polys.append(p)
    base = cv2.resize(crop, size)
    prior_tile = base.copy()
    for p in prior_polys:
        pts = np.asarray([[int(x * sx), int(y * sy)] for x, y in p], np.int32)
        cv2.polylines(prior_tile, [pts], False, (170, 170, 170), 1, cv2.LINE_AA)

    tiles = [V.label_tile(prior_tile, "192 learned priors",
                          "start point + angle, learned, image-independent")]
    subs = ["from FPN P2 (10x25) - coarse",
            "from FPN P1 (20x50)",
            "from FPN P0 (40x100) - fine"]
    for st in range(len(stages)):
        cands = stage_lanes(stages[st], conf=0.05)
        tiles.append(V.label_tile(draw_scored(base, scale(cands), 2),
                                  f"Refinement stage {st + 1}", subs[st]))

    final = stage_lanes(stages[-1], conf=conf)
    tiles.append(V.label_tile(draw_scored(base, scale(final), 3),
                              f"Confidence >= {conf}",
                              f"{len(final)} candidates survive"))
    kept = display_nms(final)
    tiles.append(V.label_tile(draw_scored(base, scale(kept), 3),
                              "After lane NMS", f"{len(kept)} lanes reported"))
    body = V.grid(tiles, 3)
    head_b = V.banner("Coarse-to-fine: priors are refined three times, each stage "
                      "reading a finer FPN level", body.shape[1])
    return V.save(out_dir / f"{stem}_clrernet_refinement.png",
                  V.vstack_pad([head_b, body], 8)), kept


def fig_roi_pooling(crop, model, tap, best_idx, out_dir, stem):
    """Show the 36 sample points a prior reads, and the strip it pools."""
    head = model.bbox_head
    feat_ys = head.prior_feat_ys.cpu().numpy()      # normalised, top->bottom order
    fmaps = [tap.d["neck"][2], tap.d["neck"][1], tap.d["neck"][0]]  # stage order
    names = ["stage 1 - FPN P2 (10x25)", "stage 2 - FPN P1 (20x50)",
             "stage 3 - FPN P0 (40x100)"]
    tiles = []
    for st in range(3):
        f = feat_np(fmaps[st])
        C, h, w = f.shape
        energy = V.normalize(V.channel_energy_map(f))
        tile = V.heat(energy, (520, 208))
        xs = tap.prior_xs[st][0, best_idx].cpu().numpy()
        for k in range(len(xs)):
            px = int(round(xs[k] * 520))
            py = int(round(feat_ys[k] * 208))
            if 0 <= px < 520:
                cv2.circle(tile, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(tile, (px, py), 3, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(V.label_tile(tile, names[st],
                                  f"{len(xs)} sample points along the prior"))

        pooled = tap.pooled[st][best_idx, :, :, 0].cpu().numpy()   # (64, 36)
        strip = V.heat(V.normalize(pooled.T), (260, 208), cv2.COLORMAP_VIRIDIS)
        tiles.append(V.label_tile(strip, "Pooled ROI feature",
                                  "36 rows x 64 channels"))
    body = V.grid(tiles, 2)
    head_b = V.banner("ROI pooling: a prior samples the map along its own curve",
                      body.shape[1],
                      sub=f"prior #{best_idx}, highest-confidence lane")
    return V.save(out_dir / f"{stem}_clrernet_roi_pooling.png",
                  V.vstack_pad([head_b, body], 8))


def fig_attention(crop, tap, top_idxs, out_dir, stem):
    """ROIGather cross-attention: where each prior looks in the whole image."""
    sim = tap.sim_maps[-1][0]          # [Np, 250] at the last refine stage
    tiles = [V.label_tile(cv2.resize(crop, (400, 160)), "Model input", "reference")]
    ranked = sorted(top_idxs[:5],
                    key=lambda i: -(sim[i].max() / sim[i].mean()).item())
    for idx in ranked:
        a = sim[idx].reshape(10, 25).cpu().numpy()
        peak = a.max() / max(a.mean(), 1e-9)
        tiles.append(V.label_tile(
            V.blend(cv2.resize(crop, (400, 160)),
                    V.normalize(a, lo_pct=0, hi_pct=100), 0.62,
                    cv2.COLORMAP_TURBO),
            f"Prior #{int(idx)} attention", f"peak / mean = {peak:.1f}x"))
    ncols = 3 if len(tiles) > 4 else 2
    body = V.grid(tiles, ncols)
    head_b = V.banner("ROIGather cross-attention: each lane query reads global context",
                      body.shape[1],
                      sub="softmax over a 10x25 grid covering the whole frame")
    return V.save(out_dir / f"{stem}_clrernet_attention.png",
                  V.vstack_pad([head_b, body], 8))


def fig_gradcam(crop, model, data, best_idx, out_dir, stem):
    """Grad-CAM of the top lane's confidence w.r.t. each FPN level."""
    feats, grads = {}, {}
    handles = []
    for lvl in range(3):
        def fwd(mod, inp, out, lvl=lvl):
            t = out[lvl]
            feats[lvl] = t
            t.register_hook(lambda g, lvl=lvl: grads.__setitem__(lvl, g))
        handles.append(model.neck.register_forward_hook(fwd))

    model.zero_grad(set_to_none=True)
    out = model.data_preprocessor(data, False)
    x = model.extract_feat(out["inputs"])
    stages = model.bbox_head(x)
    score = stages[-1]["cls_logits"][0, best_idx, 1]
    score.backward()

    names = ["FPN P0 (40x100)", "FPN P1 (20x50)", "FPN P2 (10x25)"]
    tiles = [V.label_tile(cv2.resize(crop, (400, 160)), "Model input", "reference")]
    for lvl in range(3):
        a, g = feats[lvl].detach()[0], grads[lvl].detach()[0]
        cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(0)).cpu().numpy()
        tiles.append(V.label_tile(
            V.blend(cv2.resize(crop, (400, 160)), V.normalize(cam), 0.62,
                    cv2.COLORMAP_TURBO), names[lvl],
            "Grad-CAM w.r.t. top lane confidence"))
    for h in handles:
        h.remove()
    body = V.grid(tiles, 2)
    head_b = V.banner("Which pyramid evidence supports the top lane", body.shape[1])
    return V.save(out_dir / f"{stem}_clrernet_gradcam.png",
                  V.vstack_pad([head_b, body], 8))


def fig_channels(crop, tap, out_dir, stem, level=0, n=16):
    f = feat_np(tap.d["neck"][level])
    order = np.argsort(-f.std(axis=(1, 2)))[:n]
    tiles = [V.label_tile(V.heat(V.normalize(f[c]), (220, 88)), f"ch {int(c)}",
                          bar_h=24) for c in order]
    body = V.grid(tiles, 8)
    head_b = V.banner(f"FPN P{level} individual channels: lane-aligned ridges, road "
                      f"edges, and horizon context", body.shape[1])
    return V.save(out_dir / f"{stem}_clrernet_channels.png",
                  V.vstack_pad([head_b, body], 8))


def crop_to_letterbox(poly):
    """Crop-space (800x320) polyline -> letterboxed 1640x590 coordinates."""
    W, H = CROP_SIZE
    return [((x / W) * LETTERBOX[0],
             (y / H) * (LETTERBOX[1] - CUT_HEIGHT) + CUT_HEIGHT) for x, y in poly]


def gt_mask_crop(gt_lanes, h_samples, scale, pad, thickness=6):
    """TuSimple GT polylines -> binary mask in the model's 800x320 crop space."""
    W, H = CROP_SIZE
    left, top = pad
    mask = np.zeros((H, W), np.uint8)
    sx = W / LETTERBOX[0]
    for lane in gt_lanes:
        pts = []
        for x, y in zip(lane, h_samples):
            if x < 0:
                continue
            xl, yl = x * scale + left, y * scale + top
            yc = yl - CUT_HEIGHT
            if 0 <= yc < H:
                pts.append([int(round(xl * sx)), int(round(yc))])
        if len(pts) >= 2:
            cv2.polylines(mask, [np.asarray(pts, np.int32)], False, 1, thickness)
    return mask


def fig_selectivity(crop, tap, gt_crop, out_dir, stem, level=0):
    """Rank the 64 FPN channels by agreement with the GT lane mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = feat_np(tap.d["neck"][level])
    C, h, w = f.shape
    gt = (cv2.resize(gt_crop.astype(np.float32), (w, h),
                     interpolation=cv2.INTER_AREA) > 0.1).astype(np.float32)
    gtc = gt - gt.mean()
    corr = np.zeros(C, np.float32)
    for c in range(C):
        fc = f[c] - f[c].mean()
        d = np.linalg.norm(fc) * np.linalg.norm(gtc)
        corr[c] = float((fc * gtc).sum() / d) if d > 1e-8 else 0.0
    order = np.argsort(-corr)

    fig, ax = plt.subplots(figsize=(11, 2.9), dpi=150)
    ax.bar(np.arange(C), corr[order],
           color=["#d1603d" if v > 0 else "#5b8db8" for v in corr[order]], width=0.9)
    ax.set_xlabel(f"channels of FPN P{level} (sorted)", fontsize=9)
    ax.set_ylabel("corr. with GT lane mask", fontsize=9)
    ax.set_title(f"Lane selectivity of the {C} channels CLRerNet's head samples "
                 f"from (FPN P{level})", fontsize=10)
    ax.axhline(0, color="#333", lw=0.8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    tmp = out_dir / f"{stem}_clrernet_selectivity_bar.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tmp)
    plt.close(fig)
    bar = cv2.imread(str(tmp))

    def row(chans):
        return V.hstack_pad([
            V.label_tile(V.blend(cv2.resize(crop, (240, 96)), V.normalize(f[c]), 0.62),
                         f"ch {int(c)}   r = {corr[c]:+.2f}", bar_h=28)
            for c in chans])

    body = V.vstack_pad([
        V.banner("Most lane-selective channels", bar.shape[1],
                 sub="ridge-like responses on the painted markings"),
        row(order[:6]),
        V.banner("Least lane-selective channels", bar.shape[1],
                 sub="road surface / sky context, the inverse of the lanes"),
        row(order[-6:]),
    ], 10)
    return V.save(out_dir / f"{stem}_clrernet_selectivity.png",
                  V.vstack_pad([bar, body], 12))


def fig_result(img_bgr, lb, kept, gt_lanes, h_samples, scale, pad, out_dir, stem):
    """Reported lanes on the letterboxed model frame and on the original image."""
    left, top = pad
    lb_show = lb.copy()
    for lane in gt_lanes:
        pts = [[int(round(x * scale + left)), int(round(y * scale + top))]
               for x, y in zip(lane, h_samples) if x >= 0]
        if len(pts) >= 2:
            cv2.polylines(lb_show, [np.asarray(pts, np.int32)], False, (0, 255, 0),
                          6, cv2.LINE_AA)
    cv2.rectangle(lb_show, (0, CUT_HEIGHT), (LETTERBOX[0] - 1, LETTERBOX[1] - 1),
                  (0, 255, 255), 4)

    H0, W0 = img_bgr.shape[:2]
    orig_show = img_bgr.copy()
    for lane in gt_lanes:
        pts = [[int(round(x)), int(round(y))]
               for x, y in zip(lane, h_samples) if x >= 0]
        if len(pts) >= 2:
            cv2.polylines(orig_show, [np.asarray(pts, np.int32)], False,
                          (0, 255, 0), 5, cv2.LINE_AA)

    for poly, score, _ in kept:
        lb_pts = crop_to_letterbox(poly)
        pts = np.asarray([[int(round(x)), int(round(y))] for x, y in lb_pts], np.int32)
        if len(pts) >= 2:
            cv2.polylines(lb_show, [pts], False, (0, 0, 255), 4, cv2.LINE_AA)
        # un-letterbox back to the original TuSimple resolution
        op = [((x - left) / scale, (y - top) / scale) for x, y in lb_pts]
        op = [(x, y) for x, y in op if 0 <= x < W0 and 0 <= y < H0]
        if len(op) >= 2:
            cv2.polylines(orig_show,
                          [np.asarray([[int(round(x)), int(round(y))]
                                       for x, y in op], np.int32)],
                          False, (0, 0, 255), 4, cv2.LINE_AA)

    tiles = [
        V.label_tile(cv2.resize(lb_show, (760, 273)), "Letterboxed model frame",
                     "1640x590; yellow box = the 800x320 crop the net actually sees"),
        V.label_tile(cv2.resize(orig_show, (760, 428)), "Original TuSimple frame",
                     "predictions un-letterboxed back to 1280x720"),
    ]
    body = V.vstack_pad(tiles)
    head_b = V.banner("CLRerNet result", body.shape[1],
                      sub="green = TuSimple GT, red = predicted polylines")
    return V.save(out_dir / f"{stem}_clrernet_result.png",
                  V.vstack_pad([head_b, body], 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clrernet-root", default="/shared/src/CLRerNet")
    ap.add_argument("--config",
                    default="/shared/src/CLRerNet/configs/clrernet/culane/"
                            "clrernet_culane_dla34_ema.py")
    ap.add_argument("--checkpoint",
                    default="/shared/src/CLRerNet/clrernet_culane_dla34_ema.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tusimple-root", default="/shared/data/TUSimple/test_set")
    ap.add_argument("--tusimple-label", default="/shared/data/TUSimple/test_label.json")
    ap.add_argument("--indices", default="0,580,1160")
    ap.add_argument("--out", default=str(REPO / "outputs" / "feature_slides"))
    ap.add_argument("--scratch", default="/tmp")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_root = Path(args.tusimple_root)
    recs = [json.loads(l) for l in open(args.tusimple_label)]

    model = build_model(args.clrernet_root, args.config, args.checkpoint,
                        args.device, args.scratch)
    conf = float(model.bbox_head.test_cfg.conf_threshold)

    written = {}
    for si in [int(s) for s in args.indices.split(",")]:
        rec = recs[si]
        stem = f"ts{si:04d}"
        img_path = ts_root / rec["raw_file"]

        tap = Tap(model)
        img, lb, scale, pad, lanes, scores = run_inference(model, img_path,
                                                           args.scratch)
        crop = crop_image(lb)
        stages = tap.d["stages"]
        final = stage_lanes(stages[-1], conf=conf)
        kept = display_nms(final)
        best_idx = kept[0][2] if kept else int(
            torch.softmax(stages[-1]["cls_logits"][0], -1)[:, 1].argmax())
        top_idxs = [i for _, _, i in kept[:5]] or [best_idx]

        files = [
            fig_pipeline(crop, tap, kept, out_dir, stem),
            fig_refinement(crop, model, tap, out_dir, stem, conf)[0],
            fig_roi_pooling(crop, model, tap, best_idx, out_dir, stem),
            fig_attention(crop, tap, top_idxs, out_dir, stem),
            fig_channels(crop, tap, out_dir, stem),
            fig_selectivity(crop, tap,
                            gt_mask_crop(rec["lanes"], rec["h_samples"], scale, pad),
                            out_dir, stem),
            fig_result(img, lb, kept, rec["lanes"], rec["h_samples"],
                       scale, pad, out_dir, stem),
        ]
        tap.close()

        # Grad-CAM needs its own graph-enabled pass.
        from libs.datasets.pipelines import Compose
        from libs.api.inference import letterbox_to_size
        img2 = cv2.imread(str(img_path))
        lb2, _, _ = letterbox_to_size(img2)
        d = dict(filename=str(img_path), sub_img_name=None, img=lb2, gt_points=[],
                 id_classes=[], id_instances=[], img_shape=lb2.shape,
                 ori_shape=lb2.shape)
        d = Compose(model.cfg.test_dataloader.dataset.pipeline)(d)
        files.append(fig_gradcam(crop, model,
                                 dict(inputs=[d["inputs"]],
                                      data_samples=[d["data_samples"]]),
                                 best_idx, out_dir, stem))

        written[stem] = files
        print(f"[{stem}] {img_path}  ({len(kept)} lanes)")
        for f in files:
            print("   ", f)

    (out_dir / "clrernet_manifest.json").write_text(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
