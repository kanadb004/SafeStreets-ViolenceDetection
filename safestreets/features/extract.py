"""Frozen MobileNetV2 feature extraction over the Phase 2 clip cache.

See docs/SPRINT_PLAN.md Sprint 1 and docs/decisions/ADR-003-architecture.md. Reads
each `<dataset>_<split>.h5` cache file written by Phase 2, runs every frame through
a frozen (`trainable=False`), ImageNet-weights MobileNetV2 with `pooling='avg'`,
and writes `data/features/<dataset>_<split>.h5` holding per-frame feature vectors
`(N, n_frames, 1280)` so the Sprint 1 LSTM head trains on precomputed features
instead of raw pixels.

Deliberately **no augmentation**: extraction uses eval-mode resize only and
`trainable=False`, for a deterministic one-shot feature store (ADR-003 option b).
MobileNetV2 expects inputs in [-1, 1] (`preprocess_input`), not the `unit_range`
[0, 1] that `configs/data.yaml`'s `pipeline.normalize` sets for the live tf.data
path; this module always applies `preprocess_input` itself rather than relying on
that config, so a future change to `pipeline.normalize` cannot silently affect the
feature store.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import tf_keras
from tf_keras.applications.mobilenet_v2 import preprocess_input

# isort: on


@dataclass(frozen=True)
class FeatureConfig:
    backbone: str = "mobilenetv2"
    weights: str = "imagenet"
    pooling: str = "avg"
    n_frames: int = 16
    height: int = 112
    width: int = 112

    def sha(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_backbone(cfg: FeatureConfig) -> tf_keras.Model:
    """Frozen ImageNet MobileNetV2, GAP-pooled per frame. `trainable=False`."""
    if cfg.backbone != "mobilenetv2":
        raise ValueError(f"only 'mobilenetv2' is supported, got {cfg.backbone!r}")
    backbone = tf_keras.applications.MobileNetV2(
        input_shape=(cfg.height, cfg.width, 3),
        include_top=False,
        weights=cfg.weights,
        pooling=cfg.pooling,
    )
    backbone.trainable = False
    return backbone


def extract_clip_features(
    clips: np.ndarray, backbone: tf_keras.Model, batch_frames: int = 512
) -> np.ndarray:
    """`clips`: (N, T, H, W, 3) uint8, RGB. Returns (N, T, feature_dim) float32.

    Flattens the clip and time axes into one batch of frames (equivalent to
    `TimeDistributed(backbone)` for a frozen, stateless backbone, but lets one
    large matmul-friendly batch run through MobileNetV2 instead of looping
    per clip) and reshapes back.
    """
    n, t, h, w, c = clips.shape
    flat = clips.reshape(n * t, h, w, c)
    outputs = []
    for start in range(0, flat.shape[0], batch_frames):
        chunk = preprocess_input(flat[start : start + batch_frames].astype(np.float32))
        outputs.append(backbone(chunk, training=False).numpy())
    features = np.concatenate(outputs, axis=0)
    return features.reshape(n, t, -1).astype(np.float32)


def features_path(features_dir: Path | str, dataset: str, split: str) -> Path:
    return Path(features_dir) / f"{dataset}_{split}.h5"


def existing_features_valid(path: Path, cfg: FeatureConfig, n_expected: int) -> bool:
    """Mirrors `safestreets.data.preprocess.existing_cache_is_valid`'s idempotency check."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return (
                f.attrs.get("config_sha") == cfg.sha() and f.attrs.get("n_manifest") == n_expected
            )
    except OSError:
        return False


def build_split_features(
    cache_path: Path,
    out_path: Path,
    cfg: FeatureConfig,
    backbone: tf_keras.Model,
    batch_clips: int = 32,
) -> dict:
    """Extract features for every clip in one Phase 2 cache file."""
    start = time.time()
    with h5py.File(cache_path, "r") as src:
        n = src["clips"].shape[0]
        clip_ids = src["clip_ids"][:]
        labels = src["labels"][:]

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp_path, "w") as dst:
            feat_ds = dst.create_dataset(
                "features",
                shape=(n, cfg.n_frames, 1280),
                dtype="float32",
            )
            dst.create_dataset("labels", data=labels)
            dst.create_dataset("clip_ids", data=clip_ids)

            batch_size = max(1, batch_clips)
            for i in range(0, n, batch_size):
                clips = src["clips"][i : i + batch_size]
                feat_ds[i : i + batch_size] = extract_clip_features(clips, backbone)

            dst.attrs["backbone"] = cfg.backbone
            dst.attrs["weights"] = cfg.weights
            dst.attrs["pooling"] = cfg.pooling
            dst.attrs["n_frames"] = cfg.n_frames
            dst.attrs["config_sha"] = cfg.sha()
            dst.attrs["n_manifest"] = n
        tmp_path.replace(out_path)

    return {"n": n, "bytes": out_path.stat().st_size, "wall_s": time.time() - start}
