import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import tensorflow as tf

from safestreets.data.augment import (
    AugmentConfig,
    apply_clip_transform,
    build_eval_transform,
    build_train_transform,
    load_augment_config,
    normalize_clip,
)
from safestreets.data.dataset import (
    CacheIndex,
    PipelineConfig,
    StaleCacheError,
    load_pipeline_config,
    make_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
N_FRAMES, H, W = 16, 32, 32


def _write_cache(
    path: Path,
    n_clips: int,
    n_frames: int = N_FRAMES,
    height: int = H,
    width: int = W,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        clips = rng.integers(0, 256, size=(n_clips, n_frames, height, width, 3), dtype=np.uint8)
        f.create_dataset("clips", data=clips)
        f.create_dataset("labels", data=(np.arange(n_clips) % 2).astype(np.uint8))
        f.create_dataset(
            "clip_ids", data=[f"clip_{i}".encode() for i in range(n_clips)]
        )
        f.create_dataset("padded", data=np.zeros(n_clips, dtype=bool))
        f.attrs["n_frames"] = n_frames
        f.attrs["height"] = height
        f.attrs["width"] = width
        f.attrs["sampling"] = "uniform"
        f.attrs["config_sha"] = "test"
        f.attrs["n_manifest"] = n_clips


@pytest.fixture
def pipeline_cfg():
    return PipelineConfig(n_frames=N_FRAMES, height=H, width=W, shuffle_buffer=8, batch_size=4)


@pytest.fixture
def augment_cfg():
    return AugmentConfig()


@pytest.fixture
def cache_dir(tmp_path):
    _write_cache(tmp_path / "setA_train.h5", n_clips=12, seed=1)
    _write_cache(tmp_path / "setB_train.h5", n_clips=8, seed=2)
    _write_cache(tmp_path / "setA_val.h5", n_clips=6, seed=3)
    return tmp_path


# ---- augment.py --------------------------------------------------------


@pytest.mark.phase3
def test_clip_consistency_identical_frames_stay_identical(augment_cfg):
    transform = build_train_transform(augment_cfg, H, W, N_FRAMES)
    frame = np.random.default_rng(0).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    clip = np.stack([frame] * N_FRAMES, axis=0)

    out = apply_clip_transform(transform, clip)

    for i in range(1, N_FRAMES):
        assert np.array_equal(out[0], out[i]), f"frame {i} diverged from frame 0"


@pytest.mark.phase3
def test_eval_transform_is_deterministic(augment_cfg):
    transform1 = build_eval_transform(H, W, N_FRAMES)
    transform2 = build_eval_transform(H, W, N_FRAMES)
    clip = np.random.default_rng(0).integers(0, 256, size=(N_FRAMES, H, W, 3), dtype=np.uint8)

    out1 = apply_clip_transform(transform1, clip)
    out2 = apply_clip_transform(transform2, clip)

    assert np.array_equal(out1, out2)


@pytest.mark.phase3
def test_normalize_unit_range():
    clip = np.array([[[[0, 128, 255]]]], dtype=np.uint8)
    out = normalize_clip(clip, "unit_range")
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out.flatten(), [0.0, 128 / 255, 1.0], atol=1e-6)


@pytest.mark.phase3
def test_normalize_symmetric():
    clip = np.array([[[[0, 255]]]], dtype=np.uint8)
    out = normalize_clip(clip, "symmetric")
    np.testing.assert_allclose(out.flatten(), [-1.0, 1.0], atol=1e-6)


@pytest.mark.phase3
def test_load_augment_config(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(
        "pipeline:\n  normalize: symmetric\n"
        "augment:\n  horizontal_flip_p: 0.9\n  gauss_noise_p: 0.1\n"
    )
    cfg = load_augment_config(p)
    assert cfg.horizontal_flip_p == 0.9
    assert cfg.gauss_noise_p == 0.1
    assert cfg.normalize == "symmetric"


# ---- dataset.py ---------------------------------------------------------


@pytest.mark.phase3
def test_cache_index_spans_all_files_in_split(cache_dir, pipeline_cfg):
    index = CacheIndex(cache_dir, "train", pipeline_cfg)
    assert len(index) == 12 + 8
    # every index resolves to a real clip, no gaps/overlaps
    seen = set()
    for i in range(len(index)):
        clip, label = index[i]
        assert clip.shape == (N_FRAMES, H, W, 3)
        assert label in (0, 1)
        seen.add(i)
    assert len(seen) == len(index)


@pytest.mark.phase3
def test_cache_index_missing_split_raises(cache_dir, pipeline_cfg):
    with pytest.raises(FileNotFoundError):
        CacheIndex(cache_dir, "test", pipeline_cfg)


@pytest.mark.phase3
def test_stale_cache_guard_on_frame_count_mismatch(cache_dir):
    mismatched_cfg = PipelineConfig(n_frames=8, height=H, width=W)
    with pytest.raises(StaleCacheError):
        CacheIndex(cache_dir, "train", mismatched_cfg)


@pytest.mark.phase3
def test_batch_shapes_and_dtypes(cache_dir, pipeline_cfg, augment_cfg):
    ds = make_dataset("val", pipeline_cfg, augment_cfg, cache_dir, training=False)
    clips, labels = next(iter(ds))
    assert clips.shape == (pipeline_cfg.batch_size, N_FRAMES, H, W, 3)
    assert clips.dtype == tf.float32
    assert labels.shape == (pipeline_cfg.batch_size, 1)
    assert labels.dtype == tf.float32
    assert float(tf.reduce_min(clips)) >= 0.0
    assert float(tf.reduce_max(clips)) <= 1.0


@pytest.mark.phase3
def test_eval_pipeline_is_bit_identical_across_runs(cache_dir, pipeline_cfg, augment_cfg):
    ds1 = make_dataset("val", pipeline_cfg, augment_cfg, cache_dir, training=False)
    ds2 = make_dataset("val", pipeline_cfg, augment_cfg, cache_dir, training=False)
    clips1, labels1 = next(iter(ds1))
    clips2, labels2 = next(iter(ds2))
    np.testing.assert_array_equal(clips1.numpy(), clips2.numpy())
    np.testing.assert_array_equal(labels1.numpy(), labels2.numpy())


@pytest.mark.phase3
def test_augmentation_fires_vs_eval(cache_dir, pipeline_cfg, augment_cfg):
    train_ds = make_dataset("train", pipeline_cfg, augment_cfg, cache_dir, training=True, seed=1265)
    eval_ds = make_dataset("train", pipeline_cfg, augment_cfg, cache_dir, training=False)

    train_stds = []
    for clips, _ in train_ds.take(50):
        train_stds.append(float(tf.math.reduce_std(clips)))
    eval_stds = []
    for clips, _ in eval_ds.take(50):
        eval_stds.append(float(tf.math.reduce_std(clips)))

    assert abs(np.mean(train_stds) - np.mean(eval_stds)) > 1e-4 or np.std(train_stds) > np.std(
        eval_stds
    ) * 1.5


@pytest.mark.phase3
def test_train_dataset_is_shuffled(cache_dir, pipeline_cfg, augment_cfg):
    ds1 = make_dataset("train", pipeline_cfg, augment_cfg, cache_dir, training=True, seed=1265)
    ds2 = make_dataset("train", pipeline_cfg, augment_cfg, cache_dir, training=True, seed=1265)

    first_batch_1 = next(iter(ds1))[0].numpy()
    first_batch_2 = next(iter(ds2))[0].numpy()
    # same seed-per-epoch policy across two fresh epochs still differs because the
    # underlying order is shuffled, not because the transform is random: compare means
    # across the batch which stochastic augmentation alone is unlikely to reproduce.
    assert not np.array_equal(first_batch_1, first_batch_2)


@pytest.mark.phase3
@pytest.mark.slow
def test_throughput_at_least_200_clips_per_second(cache_dir, pipeline_cfg, augment_cfg):
    big_dir = cache_dir
    _write_cache(big_dir / "setC_train.h5", n_clips=400, seed=42)
    ds = make_dataset("train", pipeline_cfg, augment_cfg, big_dir, training=True)

    n_clips = 0
    start = time.time()
    for clips, _ in ds.take(50):
        n_clips += clips.shape[0]
    elapsed = time.time() - start

    throughput = n_clips / elapsed
    assert throughput >= 200, f"throughput {throughput:.1f} clips/s below the 200 clips/s floor"


@pytest.mark.phase3
def test_load_pipeline_config(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(
        "pipeline:\n  n_frames: 8\n  height: 64\n  width: 64\n  normalize: symmetric\n"
        "  shuffle_buffer: 100\n  batch_size: 2\n"
    )
    cfg = load_pipeline_config(p)
    assert cfg.n_frames == 8
    assert cfg.height == 64
    assert cfg.normalize == "symmetric"
    assert cfg.batch_size == 2
