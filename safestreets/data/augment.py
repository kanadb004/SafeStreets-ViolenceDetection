"""Clip-consistent Albumentations augmentation + normalisation. See BUILD_PLAN Phase 3.

Augmenting a clip frame-by-frame draws a fresh random transform per frame and
destroys the temporal coherence the LSTM is supposed to model (flips alternate
mid-sequence, brightness flickers). BUILD_PLAN Phase 3 offers two fixes:
`A.ReplayCompose` (apply to frame 0, replay the same draw on the rest), or pass
every frame as an `additional_targets` image in one call so Albumentations
draws the shared parameters once internally.

This module uses the `additional_targets` form: profiling showed
`ReplayCompose.replay()` reconstructs its entire transform tree (via
`inspect.signature` introspection) on every single replay call, which made a
16-frame clip ~15x slower than necessary. Passing all 16 frames as additional
targets of one `A.Compose` call applies the identical draw to every frame in a
single pass with no replay step, at roughly 10x the throughput in local
benchmarking. Either form satisfies the "one parameter draw per clip" rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import yaml

_NORMALIZE_MODES = ("unit_range", "symmetric")


@dataclass(frozen=True)
class AugmentConfig:
    horizontal_flip_p: float = 0.5
    brightness_contrast_p: float = 0.5
    brightness_contrast_limit: float = 0.2
    random_resized_crop_p: float = 0.5
    random_resized_crop_scale_min: float = 0.8
    random_resized_crop_scale_max: float = 1.0
    gauss_noise_p: float = 0.3
    gauss_noise_std_max: float = 0.08
    normalize: str = "unit_range"

    def __post_init__(self) -> None:
        if self.normalize not in _NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {_NORMALIZE_MODES}, got {self.normalize!r}")


def load_augment_config(path: Path | str) -> AugmentConfig:
    """Load the `augment:` + `pipeline.normalize` blocks of configs/data.yaml."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    block = dict(raw.get("augment", {}))
    block["normalize"] = raw.get("pipeline", {}).get("normalize", "unit_range")
    return AugmentConfig(**block)


def _additional_targets(n_frames: int) -> dict[str, str]:
    return {f"frame{i}": "image" for i in range(1, n_frames)}


def build_train_transform(cfg: AugmentConfig, height: int, width: int, n_frames: int) -> A.Compose:
    """Stochastic transform. Apply via `apply_clip_transform`, never frame-by-frame."""
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.horizontal_flip_p),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_contrast_limit,
                contrast_limit=cfg.brightness_contrast_limit,
                p=cfg.brightness_contrast_p,
            ),
            A.RandomResizedCrop(
                size=(height, width),
                scale=(cfg.random_resized_crop_scale_min, cfg.random_resized_crop_scale_max),
                p=cfg.random_resized_crop_p,
            ),
            # noise_scale_factor < 1 generates noise at a coarser resolution and
            # upsamples it; visually indistinguishable at this clip size but
            # several times cheaper per call (measured), and this runs once per
            # clip now rather than once per frame, but still worth keeping cheap.
            A.GaussNoise(
                std_range=(0.0, cfg.gauss_noise_std_max),
                noise_scale_factor=0.25,
                p=cfg.gauss_noise_p,
            ),
        ],
        additional_targets=_additional_targets(n_frames),
    )


def build_eval_transform(height: int, width: int, n_frames: int) -> A.Compose:
    """Deterministic: resize only, nothing stochastic."""
    return A.Compose(
        [A.Resize(height=height, width=width, p=1.0)],
        additional_targets=_additional_targets(n_frames),
    )


def apply_clip_transform(transform: A.Compose, clip: np.ndarray) -> np.ndarray:
    """Apply `transform` to every frame of `clip` with one shared parameter draw.

    `clip` is (T, H, W, 3) uint8. Returns the same shape, uint8.
    """
    n_frames = clip.shape[0]
    kwargs = {"image": clip[0]}
    for i in range(1, n_frames):
        kwargs[f"frame{i}"] = clip[i]
    out = transform(**kwargs)
    frames = [out["image"]] + [out[f"frame{i}"] for i in range(1, n_frames)]
    return np.stack(frames, axis=0).astype(np.uint8)


def normalize_clip(clip: np.ndarray, mode: str) -> np.ndarray:
    """uint8 [0, 255] -> float32, per `mode`.

    "unit_range" -> [0, 1]; "symmetric" -> [-1, 1] (what MobileNetV2-style
    ImageNet backbones expect). Config-driven per BUILD_PLAN Phase 3 so Phase 4
    can switch backbones without touching this module.
    """
    out = clip.astype(np.float32) / 255.0
    if mode == "symmetric":
        out = out * 2.0 - 1.0
    elif mode != "unit_range":
        raise ValueError(f"normalize must be one of {_NORMALIZE_MODES}, got {mode!r}")
    return out
