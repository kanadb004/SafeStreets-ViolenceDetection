"""tf.data pipeline over the Phase 2 clip cache, with clip-consistent augmentation.

See docs/BUILD_PLAN.md Phase 3. Reads every `<dataset>_<split>.h5` cache file lazily
by index (never loads a whole split into RAM), applies the Phase 3 augmentation via
`safestreets.data.augment`, and assembles `shuffle -> map -> batch -> prefetch`.
The HDF5 cache files written by Phase 2 already are the "raw read" cache; no
further `tf.data.Dataset.cache()` of pre-augmentation data is needed on top of it.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import tensorflow as tf
import yaml

from safestreets.data.augment import (
    AugmentConfig,
    apply_clip_transform,
    build_eval_transform,
    build_train_transform,
    normalize_clip,
)

# Albumentations' OpenCV backend defaults to an 8-thread pool per call. tf.data's
# own AUTOTUNE parallel map already runs many clips concurrently, so without this
# every call oversubscribes the machine's cores against itself and throughput
# drops *below* single-threaded. Let tf.data do the parallelism instead.
cv2.setNumThreads(1)


@dataclass(frozen=True)
class PipelineConfig:
    n_frames: int = 16
    height: int = 112
    width: int = 112
    normalize: str = "unit_range"
    shuffle_buffer: int = 512
    batch_size: int = 8


def load_pipeline_config(path: Path | str) -> PipelineConfig:
    """Load the `pipeline:` block of configs/data.yaml."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    block = dict(raw.get("pipeline", {}))
    block.pop("normalize", None)
    normalize = raw.get("pipeline", {}).get("normalize", "unit_range")
    return PipelineConfig(**block, normalize=normalize)


class StaleCacheError(ValueError):
    """Raised when a cache file's shape disagrees with configs/data.yaml's pipeline block."""


class CacheIndex:
    """Flat, lazily-opened index over every `*_<split>.h5` cache file for one split.

    `__getitem__` opens files on first use and keeps the handles around; it never
    reads more than one clip into memory at a time.
    """

    def __init__(self, cache_dir: Path | str, split: str, pipeline_cfg: PipelineConfig):
        self.paths = sorted(Path(cache_dir).glob(f"*_{split}.h5"))
        if not self.paths:
            raise FileNotFoundError(f"no cache files found for split={split!r} under {cache_dir}")

        self._cumulative: list[int] = []
        total = 0
        for p in self.paths:
            with h5py.File(p, "r") as f:
                shape = (int(f.attrs["n_frames"]), int(f.attrs["height"]), int(f.attrs["width"]))
                expected = (pipeline_cfg.n_frames, pipeline_cfg.height, pipeline_cfg.width)
                if shape != expected:
                    raise StaleCacheError(
                        f"{p} was built with (n_frames, height, width)={shape} but "
                        f"configs/data.yaml's pipeline block now expects {expected}. "
                        "Rebuild the Phase 2 cache (scripts/build_cache.py --force) or "
                        "revert the config change."
                    )
                total += f["clips"].shape[0]
            self._cumulative.append(total)
        self.n = total
        self._open_files: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return self.n

    def _file(self, path: Path) -> h5py.File:
        f = self._open_files.get(path)
        if f is None:
            f = h5py.File(path, "r")
            self._open_files[path] = f
        return f

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        if i < 0 or i >= self.n:
            raise IndexError(i)
        file_idx = bisect_right(self._cumulative, i)
        start = self._cumulative[file_idx - 1] if file_idx > 0 else 0
        f = self._file(self.paths[file_idx])
        local = i - start
        return f["clips"][local], int(f["labels"][local])


def make_dataset(
    split: str,
    pipeline_cfg: PipelineConfig,
    augment_cfg: AugmentConfig,
    cache_dir: Path | str,
    *,
    training: bool | None = None,
    seed: int = 1265,
) -> tf.data.Dataset:
    """Build the tf.data.Dataset for one split: float32 clips (B,T,H,W,3), labels (B,1)."""
    training = (split == "train") if training is None else training
    index = CacheIndex(cache_dir, split, pipeline_cfg)
    height, width, n_frames = pipeline_cfg.height, pipeline_cfg.width, pipeline_cfg.n_frames
    transform = (
        build_train_transform(augment_cfg, height, width, n_frames)
        if training
        else build_eval_transform(height, width, n_frames)
    )

    def generator():
        for i in range(len(index)):
            clip, label = index[i]
            yield clip, label

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(pipeline_cfg.n_frames, pipeline_cfg.height, pipeline_cfg.width, 3),
                dtype=tf.uint8,
            ),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    if training:
        ds = ds.shuffle(pipeline_cfg.shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    def _augment(clip: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        def _apply(clip_tensor: tf.Tensor) -> np.ndarray:
            augmented = apply_clip_transform(transform, clip_tensor.numpy())
            return normalize_clip(augmented, augment_cfg.normalize)

        clip_f = tf.py_function(_apply, [clip], tf.float32)
        clip_f.set_shape((pipeline_cfg.n_frames, pipeline_cfg.height, pipeline_cfg.width, 3))
        label_f = tf.reshape(tf.cast(label, tf.float32), (1,))
        return clip_f, label_f

    ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(pipeline_cfg.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
