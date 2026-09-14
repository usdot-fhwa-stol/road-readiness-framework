"""Pack the CLRerNet / YOLOPX feature figures into a self-contained slide deck.

Reads the PNGs written by yolopx_features.py and clrernet_features.py, re-encodes
them as web-sized JPEGs, and writes a single HTML file with every image inlined
as a data URI (Artifact pages may not fetch external assets).

    python -m scripts.features.build_deck --figures outputs/feature_slides \
        --out outputs/feature_slides/deck.html
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2

SCENES = [
    ("ts0000", "Straight highway", "clips/0530/1492626760788443246_0 - solid yellow "
                                   "edge line, dashed interior, empty road"),
    ("ts0580", "Right curve", "clips/0530/1492629924572036071_0 - continuous curvature, "
                              "guardrail, strong shadow band"),
    ("ts1160", "Dense traffic", "clips/0530/1492638407037549920_0 - vehicles occluding "
                                "the markings, worn paint"),
]


def encode(path: Path, max_w: int, quality: int) -> str:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(round(h * max_w / w))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_slides():
    """Deck outline. Each figure slide names the suffix it pulls per scene."""
    return [
        dict(kind="title"),
        dict(kind="text", section="Setup", n="00", title="Two ways to find a lane",
             lede="Both models see the same TuSimple frame. What they compute from it "
                  "could hardly be more different.",
             body="compare"),

        dict(kind="fig", model="yolopx", section="YOLOPX", n="01",
             title="One encoder, three heads",
             lede="Lane lines are a side-effect of a detection backbone: the ELANNet "
                  "features are shared, and a dedicated decoder upsamples them back to "
                  "full resolution.",
             suffix="yolopx_pipeline",
             notes=["Backbone C2-C5 respond to every high-contrast structure - hillside, "
                    "horizon, guardrail - not to lanes specifically.",
                    "The lane branch pulls the 1/4-scale C2 skip back in; that is where "
                    "the thin markings survive.",
                    "By the PSA block at 1/2 resolution the map is already the answer: "
                    "everything except the paint has been suppressed."]),
        dict(kind="fig", model="yolopx", section="YOLOPX", n="02",
             title="What a mid-level channel actually encodes",
             lede="At the neck, nothing is lane-specific yet. Individual P3 channels are "
                  "oriented edges, road texture, and a road/sky split.",
             suffix="yolopx_channels_3",
             notes=["Several channels fire on the road edge and the guardrail as strongly "
                    "as on the paint.",
                    "Others encode the vanishing-point geometry - useful context, but not "
                    "a lane detector.",
                    "The lane-specific representation is built by the decoder, not "
                    "inherited from the backbone."]),
        dict(kind="fig", model="yolopx", section="YOLOPX", n="03",
             title="Where the lane representation appears",
             lede="Six blocks later, in the lane decoder, most channels have collapsed "
                  "onto the markings themselves.",
             suffix="yolopx_channels_22",
             notes=["Individual channels now trace complete lane instances end to end.",
                    "A minority stay inverted - they encode drivable surface, the "
                    "complement of the lanes.",
                    "This is the layer we rank on the next slide."]),
        dict(kind="fig", model="yolopx", section="YOLOPX", n="04",
             title="Which channels carry the lane",
             lede="Correlating each of the 64 lane-decoder channels with the TuSimple "
                  "ground-truth mask splits the block cleanly in two.",
             suffix="yolopx_selectivity",
             notes=["Roughly a quarter of the channels are strongly lane-positive "
                    "(r up to +0.7).",
                    "A comparable group is strongly lane-negative - road surface "
                    "detectors, equally informative to the final 2-class head.",
                    "The middle of the distribution is near zero: much of the block's "
                    "capacity is not spent on lanes at all."]),
        dict(kind="fig", model="yolopx", section="YOLOPX", n="05",
             title="Grad-CAM through the depth of the network",
             lede="Gradient of the lane logit back onto each feature map, showing which "
                  "spatial evidence the decision actually rests on.",
             suffix="yolopx_gradcam",
             notes=["Deep, low-resolution levels (P4) attribute to the horizon and scene "
                    "layout rather than to the paint.",
                    "The 1/8 and 1/4 neck levels attribute to the markings, but also to "
                    "the road boundary beside them.",
                    "Only in the lane decoder does attribution become the lane itself."]),
        dict(kind="fig", model="yolopx", section="YOLOPX", n="06",
             title="Occlusion: what the prediction cannot survive without",
             lede="A grey patch is slid across the input; each cell reports how much lane "
                  "probability is lost, and how much the prediction changes overall.",
             suffix="yolopx_occlusion",
             notes=["Hiding a marking removes it locally - the loss map traces the lanes.",
                    "Hiding the vanishing region costs far more than its area: it "
                    "destroys lanes far from the patch.",
                    "The change map is wider than the loss map - occlusion also creates "
                    "false positives on the road surface."]),
        dict(kind="fig", model="yolopx", section="YOLOPX", n="07",
             title="The output it was scored on",
             lede="Dense per-pixel lane-line segmentation, un-letterboxed to the original "
                  "TuSimple resolution.",
             suffix="yolopx_result",
             notes=["Every marking in view is labelled, including ones TuSimple's "
                    "four-lane annotation does not include.",
                    "Predicted strokes are wider than the annotation - the pixel-IoU "
                    "penalty this creates is a metric artefact, not a localisation error.",
                    "There are no lane instances here, only pixels."]),

        dict(kind="fig", model="clrernet", section="CLRerNet", n="08",
             title="A pyramid built to be sampled",
             lede="DLA-34 aggregates across scales, then the FPN emits three 64-channel "
                  "maps - one per refinement stage, not one per output.",
             suffix="clrernet_pipeline",
             notes=["The network only ever sees the bottom 320 rows of the frame; the "
                    "sky is cropped away before the first convolution.",
                    "All three FPN levels share a channel count so the same head can pool "
                    "from any of them.",
                    "Lane structure is already legible at P0 - the head's job is to place "
                    "curves on it, not to find it."]),
        dict(kind="fig", model="clrernet", section="CLRerNet", n="09",
             title="Coarse to fine, three times",
             lede="192 learned priors - fixed start point and angle, identical for every "
                  "image - are refined three times, each stage reading a finer level.",
             suffix="clrernet_refinement",
             notes=["Stage 1 works from the 10x25 map and already sorts plausible from "
                    "implausible directions.",
                    "Each stage adds a delta to (y0, x0, theta) plus 72 per-row offsets; "
                    "confidence sharpens as resolution rises.",
                    "Thresholding leaves near-duplicates on every real lane; NMS is what "
                    "turns them into instances."]),
        dict(kind="fig", model="clrernet", section="CLRerNet", n="10",
             title="Each prior reads its own curve",
             lede="ROI pooling samples the feature map at 36 points along the prior, "
                  "producing a 36x64 strip that is the entire evidence for that lane.",
             suffix="clrernet_roi_pooling",
             notes=["The sample points follow the current estimate, so a better estimate "
                    "reads better features - that is the refinement loop.",
                    "At 10x25 the samples are coarse and blurred together; at 40x100 they "
                    "sit on individual dashes.",
                    "The strip is flattened to a 64-d vector - classification and "
                    "regression see nothing else, locally."]),
        dict(kind="fig", model="clrernet", section="CLRerNet", n="11",
             title="ROIGather: the global look",
             lede="Because a strip alone cannot resolve an occluded lane, each query also "
                  "cross-attends to a 10x25 grid covering the whole frame.",
             suffix="clrernet_attention",
             notes=["Peak/mean tells you how much a given query relies on global context.",
                    "Some queries attend sharply to distant road structure; others stay "
                    "essentially uniform and lean on their ROI strip alone.",
                    "This module is what lets a dashed or vehicle-occluded lane still be "
                    "completed."]),
        dict(kind="fig", model="clrernet", section="CLRerNet", n="12",
             title="Which FPN channels carry the lane",
             lede="The same ground-truth correlation as YOLOPX, run on the 64 channels "
                  "the head is allowed to pool from.",
             suffix="clrernet_selectivity",
             notes=["The split is the same shape - a lane-positive group, a road-surface "
                    "group, and a near-zero middle.",
                    "Peak correlation is slightly lower than YOLOPX's decoder, which is "
                    "expected: this is a shared neck, not a lane-specific head.",
                    "Lane evidence is therefore distributed across channels, and the ROI "
                    "strip samples all 64 of them at once."]),
        dict(kind="fig", model="clrernet", section="CLRerNet", n="13",
             title="Grad-CAM on the winning lane",
             lede="Gradient of one lane's confidence back onto each pyramid level.",
             suffix="clrernet_gradcam",
             notes=["Attribution is concentrated on painted markings across all three "
                    "levels - markedly cleaner than YOLOPX's.",
                    "It is not confined to the lane being predicted: neighbouring lanes "
                    "support it too, via ROIGather.",
                    "At 10x25 attribution spreads into scene layout, the same coarse-level "
                    "behaviour both models show."]),
        dict(kind="fig", model="clrernet", section="CLRerNet", n="14",
             title="The output it was scored on",
             lede="A short list of parametric curves with confidences - no mask, and no "
                  "pixels to threshold.",
             suffix="clrernet_result",
             notes=["Curves extend past the image content into the letterbox padding: "
                    "the model was trained on 1640x590 CULane frames.",
                    "Zero-shot on TuSimple it reports a subset of the annotated lanes - "
                    "a real recall gap, not a rasterisation artefact.",
                    "What it does report is geometrically tight against the annotation."]),

        dict(kind="text", section="Wrap-up", n="15", title="What the features say",
             lede="Same frame, same evidence in the image, two different notions of what "
                  "a lane is.",
             body="takeaways"),
    ]


COMPARE_ROWS = [
    ("Task formulation", "Per-pixel binary segmentation", "Row-anchor curve detection"),
    ("Backbone / neck", "ELANNet + PaFPN, shared with detection and drivable area",
     "DLA-34 + FPN, 3 x 64ch, lane-only"),
    ("Input the net sees", "384 x 640 letterbox, full frame",
     "800 x 320, bottom 320 rows of a 1640 x 590 letterbox"),
    ("Where lanes become explicit", "In the lane decoder (block 22 onward)",
     "In the head, via priors - the neck stays generic"),
    ("Evidence per decision", "A pixel's receptive field",
     "36 samples along the curve + a 10 x 25 global attention map"),
    ("Output", "Probability map, argmax to a mask",
     "<= 4 curves with confidences, after lane NMS"),
    ("Failure mode seen here", "Wide strokes, no lane identity",
     "Misses lanes it was never trained on (zero-shot)"),
]

TAKEAWAYS = [
    ("Lane selectivity is a small part of a big block",
     "In both networks only a quarter or so of the channels in the decisive block "
     "correlate positively with the ground-truth lane mask, and a comparable group "
     "correlates negatively - encoding drivable surface. The lane signal lives in a "
     "contrast between two channel groups, not in a single lane feature."),
    ("The paint is not the only evidence",
     "Grad-CAM and the occlusion test both show the vanishing region and road boundary "
     "carrying real weight. Occluding a 40 px patch near the horizon costs more lane "
     "probability than occluding a patch of the marking itself."),
    ("Coarse levels do scene layout, fine levels do paint",
     "At 1/16 and 1/32 scale, attribution in both models drifts to horizon and road "
     "geometry. That is what CLRerNet's first refinement stage is for - it reads the "
     "coarse level to pick a direction, then re-reads finer levels to place the curve."),
    ("The architectures differ most in what they commit to",
     "YOLOPX commits per pixel and never forms a lane object; CLRerNet commits to 192 "
     "hypotheses up front and spends its whole head refining them. Every downstream "
     "readiness metric inherits that choice."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default="outputs/feature_slides")
    ap.add_argument("--out", default="outputs/feature_slides/deck.html")
    ap.add_argument("--max-width", type=int, default=1500)
    ap.add_argument("--quality", type=int, default=78)
    args = ap.parse_args()

    figdir = Path(args.figures)
    slides = build_slides()

    images, missing = {}, []
    for s in slides:
        if s["kind"] != "fig":
            continue
        for scene, _, _ in SCENES:
            name = f"{scene}_{s['suffix']}.png"
            p = figdir / name
            if not p.exists():
                missing.append(name)
                continue
            if name not in images:
                images[name] = encode(p, args.max_width, args.quality)
    if missing:
        print("WARNING: missing figures ->")
        for m in missing:
            print("   ", m)

    payload = json.dumps(dict(slides=slides, scenes=SCENES, images=images,
                              compare=COMPARE_ROWS, takeaways=TAKEAWAYS))
    template = (Path(__file__).with_name("deck_template.html")).read_text()
    html = template.replace("/*__DATA__*/", payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({mb:.1f} MB, {len(images)} images)")


if __name__ == "__main__":
    main()
