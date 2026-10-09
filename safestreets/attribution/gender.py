"""Zero-shot CLIP gender attribution over person crops. No PA-100K training
(cut, see docs/SPRINT_PLAN.md §2) — this is a frozen, pretrained CLIP scored
against two text prompts, nothing here is fit to labelled data. Its accuracy
is therefore unknown until measured and is expected to be mediocre; report it
as such in docs/ETHICS.md, never as a validated classifier.

`open_clip` is imported lazily, inside `_load`, for the same reason as
`safestreets.attribution.person`: importability and a clean `available()`
must not require the `ml` extra to be installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Report these two labels only; the ethics tradeoffs of a non-binary label
# set are real but out of scope for a zero-shot CLIP head, see docs/ETHICS.md.
PROMPTS = {"woman": "a photo of a woman", "man": "a photo of a man"}


@dataclass(frozen=True)
class GenderClassifierConfig:
    # The "-quickgelu" variant matches OpenAI's original activation; the
    # plain "ViT-B-32" config mismatches it (open_clip warns "QuickGELU
    # mismatch"), which would silently degrade zero-shot accuracy.
    model_name: str = "ViT-B-32-quickgelu"
    pretrained: str = "openai"


class ZeroShotGenderClassifier:
    def __init__(self, cfg: GenderClassifierConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._preprocess = None
        self._text_features = None
        self._labels = list(PROMPTS.keys())

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.cfg.model_name, pretrained=self.cfg.pretrained
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(self.cfg.model_name)
        with torch.no_grad():
            text = tokenizer([PROMPTS[label] for label in self._labels])
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._model = model
        self._preprocess = preprocess
        self._text_features = text_features

    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            logger.warning("gender classifier %r unavailable", self.cfg.model_name)
            return False

    def classify(self, crops: list[np.ndarray]) -> list[dict]:
        """`crops`: HxWx3 uint8 RGB arrays (one person crop each). Returns one
        `{"label", "score", "probs"}` per crop, in order.
        """
        if not crops:
            return []
        self._load()
        import torch
        from PIL import Image

        images = torch.stack([self._preprocess(Image.fromarray(c)) for c in crops])
        with torch.no_grad():
            image_features = self._model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self._text_features.T
            probs = logits.softmax(dim=-1).numpy()

        out = []
        for row in probs:
            idx = int(np.argmax(row))
            out.append(
                {
                    "label": self._labels[idx],
                    "score": float(row[idx]),
                    "probs": {label: float(p) for label, p in zip(self._labels, row)},
                }
            )
        return out
