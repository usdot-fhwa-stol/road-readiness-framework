"""Tagging backends. Each exposes .predict(pil_image) -> predicted_tags dict
in the shape {by_dimension, labels, scores}, so tag_images.py is backend-agnostic.

  siglip : google/siglip2-base-patch16-224 zero-shot (fast, weak on the fine
           semantic dimensions).
  vlm    : Qwen2.5-VL generative model prompted with the taxonomy, returns the
           chosen tags as JSON (much stronger on the semantic dimensions).
"""
from __future__ import annotations

import json
import re
import sys
from unittest.mock import MagicMock

# transformers 5.x eagerly imports a broken torchaudio in this env; stub it.
if "torchaudio" not in sys.modules:
    import importlib.machinery
    _ta = MagicMock()
    _ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = _ta

import torch
import torch.nn.functional as F

from taxonomy import ALL_TAGS, DIMENSION_CONFIG, TAXONOMY

SIGLIP_DEFAULT = "google/siglip2-base-patch16-224"
VLM_DEFAULT = "Qwen/Qwen2.5-VL-7B-Instruct"


def _as_tensor(out):
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "last_hidden_state", "logits"):
        if getattr(out, attr, None) is not None:
            return getattr(out, attr)
    raise TypeError(f"cannot extract feature tensor from {type(out)}")


# --------------------------------------------------------------------------- #
# SigLIP zero-shot backend
# --------------------------------------------------------------------------- #
class SiglipBackend:
    def __init__(self, model_id=SIGLIP_DEFAULT, device="cuda"):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        prompts = [p for _, _, p in ALL_TAGS]
        with torch.no_grad():
            inp = self.processor(text=prompts, padding="max_length",
                                 return_tensors="pt").to(device)
            self.text_feats = F.normalize(_as_tensor(self.model.get_text_features(**inp)), dim=-1)

    @torch.no_grad()
    def predict(self, image):
        inp = self.processor(images=image, return_tensors="pt").to(self.device)
        img = F.normalize(_as_tensor(self.model.get_image_features(**inp)), dim=-1)
        logits = (img @ self.text_feats.T) * self.model.logit_scale.exp() + self.model.logit_bias
        sig = torch.sigmoid(logits).squeeze(0).float().cpu()
        scores = {tag: float(sig[i]) for i, (_, tag, _) in enumerate(ALL_TAGS)}
        return _select_from_scores(scores)


def _select_from_scores(sigmoid_scores):
    """Per-dimension: emit top-1 plus any tag above the dimension's rel_thresh
    (within-dimension softmax). Shared by the SigLIP backend."""
    by_dim, flat, out_scores = {}, [], {}
    for dim, tags in TAXONOMY.items():
        cfg = DIMENSION_CONFIG[dim]
        names = list(tags)
        raw = torch.tensor([sigmoid_scores[t] for t in names])
        soft = F.softmax(raw, dim=0)
        order = torch.argsort(soft, descending=True)
        chosen = [names[int(order[0])]]
        if cfg["multi_label"]:
            for j in order[1:]:
                if float(soft[j]) >= cfg["rel_thresh"]:
                    chosen.append(names[int(j)])
        by_dim[dim] = chosen
        flat.extend(chosen)
        for i, t in enumerate(names):
            out_scores[t] = round(float(soft[i]), 4)
    return {"by_dimension": by_dim, "labels": flat, "scores": out_scores}


# --------------------------------------------------------------------------- #
# Qwen2.5-VL generative backend
# --------------------------------------------------------------------------- #
def _build_vlm_prompt():
    lines = [
        "You are labeling a single forward-facing dashcam road image for a "
        "road-readiness dataset.",
        "For EACH of the 6 dimensions below, select ALL tags that apply "
        "(multi-label). Select at least one tag per dimension, using ONLY the "
        "allowed values listed - never invent tags.",
        "",
        "Dimensions and allowed tags:",
    ]
    for dim, tags in TAXONOMY.items():
        lines.append(f"- {dim}: {', '.join(tags)}")
    lines += [
        "",
        "Respond with ONLY a JSON object mapping each dimension name (exactly as "
        "written above) to a list of chosen tags. No prose, no code fences.",
    ]
    return "\n".join(lines)


class VlmBackend:
    """Generic image-text-to-text backend. Works for any model that implements
    the transformers image-text-to-text interface: Qwen2.5-VL, Qwen3-VL,
    Qwen3-VL-MoE, InternVL3, etc. Pass load_4bit=True for big models (bnb nf4)."""

    def __init__(self, model_id=VLM_DEFAULT, device="cuda",
                 max_pixels=1024 * 1024, load_4bit=False):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = device
        self.model_id = model_id
        kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
            kw["device_map"] = "auto"
        else:
            kw["device_map"] = device
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        # some processors (InternVL) don't accept max_pixels
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, max_pixels=max_pixels, trust_remote_code=True)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.prompt = _build_vlm_prompt()
        self._valid = {dim: set(tags) for dim, tags in TAXONOMY.items()}

    @torch.no_grad()
    def _generate(self, image):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt}]}]
        # unified path: tokenize + embed images straight from the messages
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def predict(self, image):
        raw = self._generate(image)
        return _labels_from_json(raw, self._valid)


def _labels_from_json(raw, valid_by_dim):
    """Parse a model's JSON response into the predicted_tags dict, keeping only
    tags in the taxonomy. Shared by all generative backends."""
    parsed = _extract_json(raw)
    by_dim, flat = {}, []
    for dim, valid in valid_by_dim.items():
        chosen = []
        got = parsed.get(dim, []) if isinstance(parsed, dict) else []
        if isinstance(got, str):
            got = [got]
        for t in got or []:
            if t in valid and t not in chosen:
                chosen.append(t)
        by_dim[dim] = chosen
        flat.extend(chosen)
    return {"by_dimension": by_dim, "labels": flat, "scores": {},
            "_raw": raw if not flat else None}


# --------------------------------------------------------------------------- #
# InternVL native backend (for OpenGVLab/InternVL3-* repos that ship remote
# code + a .chat() API and dynamic image tiling, not the unified interface).
# --------------------------------------------------------------------------- #
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _internvl_tiles(image, min_num=1, max_num=12, size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    best, best_diff = (1, 1), float("inf")
    for r in ratios:
        diff = abs(ar - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
    tw, th = size * best[0], size * best[1]
    cols = tw // size
    resized = image.resize((tw, th))
    tiles = [resized.crop(((i % cols) * size, (i // cols) * size,
                           (i % cols + 1) * size, (i // cols + 1) * size))
             for i in range(best[0] * best[1])]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((size, size)))
    return tiles


class InternVLNativeBackend:
    def __init__(self, model_id, device="cuda", max_num=12):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.model_id = model_id
        self.max_num = max_num
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True).eval().to(device)
        self.tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False)
        self.transform = _internvl_transform(448)
        self.prompt = _build_vlm_prompt()
        self._valid = {dim: set(tags) for dim, tags in TAXONOMY.items()}

    @torch.no_grad()
    def predict(self, image):
        tiles = _internvl_tiles(image, max_num=self.max_num)
        pv = torch.stack([self.transform(t) for t in tiles]).to(torch.bfloat16).to(self.device)
        cfg = dict(max_new_tokens=512, do_sample=False)
        raw = self.model.chat(self.tok, pv, "<image>\n" + self.prompt, cfg)
        return _labels_from_json(raw, self._valid)


def _extract_json(text):
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def make_backend(name, model_id=None, device="cuda", load_4bit=False):
    if name == "siglip":
        return SiglipBackend(model_id or SIGLIP_DEFAULT, device)
    if name == "vlm":
        return VlmBackend(model_id or VLM_DEFAULT, device, load_4bit=load_4bit)
    if name == "internvl":
        return InternVLNativeBackend(model_id or "OpenGVLab/InternVL3-8B", device)
    raise ValueError(f"unknown backend: {name}")
