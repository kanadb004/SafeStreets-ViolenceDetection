from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from safestreets.features.extract import (
    FeatureConfig,
    build_backbone,
    build_split_features,
    existing_features_valid,
    extract_clip_features,
)
from safestreets.models.factory import build_model
from safestreets.training.losses import compute_pos_weight, weighted_bce
from safestreets.training.train import load_feature_split

REPO_ROOT = Path(__file__).resolve().parent.parent
N_FRAMES, H, W = 16, 112, 112


def _model_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "model.yaml") as f:
        return yaml.safe_load(f)


def _write_feature_file(
    path: Path, n: int, labels: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, N_FRAMES, 1280)).astype(np.float32)
    if labels is None:
        labels = (np.arange(n) % 2).astype(np.uint8)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("labels", data=labels)
        f.create_dataset("clip_ids", data=[f"clip_{i}".encode() for i in range(n)])
        f.attrs["backbone"] = "mobilenetv2"
        f.attrs["weights"] = "imagenet"
        f.attrs["pooling"] = "avg"
        f.attrs["n_frames"] = N_FRAMES
        f.attrs["config_sha"] = "test"
        f.attrs["n_manifest"] = n
    return features


@pytest.mark.sprint1
def test_build_model_compiles_both_archs(tmp_path: Path) -> None:
    raw = _model_cfg()
    for arch in ("lstm_head", "scratch"):
        raw["arch"] = arch
        model = build_model(raw, pos_weight=1.5)
        assert model.optimizer is not None
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        out = tmp_path / f"model_summary_{arch}.txt"
        out.write_text("\n".join(summary_lines))
        assert out.exists() and out.stat().st_size > 0


@pytest.mark.sprint1
def test_pos_weight_from_train_labels_only() -> None:
    labels = np.array([0, 0, 0, 1])
    assert compute_pos_weight(labels) == pytest.approx(3.0)


@pytest.mark.sprint1
def test_weighted_bce_penalizes_missed_positive_more() -> None:
    loss_fn = weighted_bce(pos_weight=5.0)
    import tensorflow as tf

    # A missed positive (y_true=1, y_pred small) costs more than a missed
    # negative (y_true=0, y_pred large) under pos_weight > 1.
    missed_positive = loss_fn(tf.constant([[1.0]]), tf.constant([[0.1]]))
    missed_negative = loss_fn(tf.constant([[0.0]]), tf.constant([[0.9]]))
    assert float(missed_positive) > float(missed_negative)


@pytest.mark.sprint1
def test_overfit_32_clip_subset() -> None:
    """The head must reach >=0.95 train accuracy on 32 clips within 50 epochs,
    per docs/SPRINT_PLAN.md Sprint 1 DoD. A failure here means broken label
    wiring or a degenerate architecture, not a data problem."""
    rng = np.random.default_rng(1265)
    n = 32
    labels = (np.arange(n) % 2).astype(np.float32)
    # Features correlated with the label so a real signal exists to fit.
    features = rng.normal(size=(n, N_FRAMES, 1280)).astype(np.float32)
    features[labels == 1] += 2.0

    raw = _model_cfg()
    raw["arch"] = "lstm_head"
    model = build_model(raw, pos_weight=1.0)
    history = model.fit(features, labels, epochs=50, batch_size=8, verbose=0)
    assert max(history.history["accuracy"]) >= 0.95


@pytest.mark.sprint1
def test_label_alignment_in_feature_split(tmp_path: Path) -> None:
    """A batch's labels must match the clip_ids they were loaded under."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    n = 10
    labels = (np.arange(n) % 2).astype(np.uint8)
    expected_by_id = {f"clip_{i}": int(labels[i]) for i in range(n)}
    _write_feature_file(features_dir / "rwf2000_train.h5", n, labels=labels)

    x, y, clip_ids = load_feature_split(features_dir, ["rwf2000"], "train")
    assert x.shape == (n, N_FRAMES, 1280)
    for i, clip_id in enumerate(clip_ids):
        assert int(y[i]) == expected_by_id[clip_id]


@pytest.mark.sprint1
def test_normalization_matches_preprocess_input() -> None:
    """Features must come from preprocess_input-scaled input ([-1, 1]), not raw
    [0, 255] or [0, 1]. Verified against a hand-computed single-clip forward
    pass, per docs/SPRINT_PLAN.md Sprint 1 DoD."""
    from tf_keras.applications.mobilenet_v2 import preprocess_input

    rng = np.random.default_rng(0)
    clip = rng.integers(0, 256, size=(1, N_FRAMES, H, W, 3), dtype=np.uint8)

    cfg = FeatureConfig(n_frames=N_FRAMES, height=H, width=W)
    backbone = build_backbone(cfg)
    got = extract_clip_features(clip, backbone)

    hand_frame0 = preprocess_input(clip[0, 0:1].astype(np.float32))
    expected_frame0 = backbone(hand_frame0, training=False).numpy()[0]

    np.testing.assert_allclose(got[0, 0], expected_frame0, atol=1e-5)

    # Sanity: the raw [0, 255] input would NOT match (proves preprocess_input
    # is actually doing something, not a vacuously-true comparison).
    raw_out = backbone(clip[0, 0:1].astype(np.float32), training=False).numpy()[0]
    assert not np.allclose(got[0, 0], raw_out, atol=1e-3)


@pytest.mark.sprint1
@pytest.mark.needs_data
def test_extract_split_features_matches_cache_count() -> None:
    """N_features == N_cached for a real cache file (Sprint 1 DoD)."""
    cache_path = REPO_ROOT / "data" / "cache" / "ucfcrime_test.h5"
    if not cache_path.exists():
        pytest.skip("no real cache available")
    with h5py.File(cache_path, "r") as f:
        n_cached = f["clips"].shape[0]

    cfg = FeatureConfig(n_frames=N_FRAMES, height=H, width=W)
    backbone = build_backbone(cfg)
    out_path = Path("data") / "features" / "_test_ucfcrime_test.h5"
    try:
        result = build_split_features(cache_path, out_path, cfg, backbone)
        assert result["n"] == n_cached
        with h5py.File(out_path, "r") as f:
            assert f["features"].shape == (n_cached, N_FRAMES, 1280)
            assert list(f["labels"][:]) == list(h5py.File(cache_path, "r")["labels"][:])
    finally:
        out_path.unlink(missing_ok=True)


@pytest.mark.sprint1
@pytest.mark.needs_data
def test_extraction_idempotency_skips_fast() -> None:
    """A second extraction run for an unchanged (dataset, split, config) must
    skip in well under 5s, per the Sprint 1 DoD."""
    import time

    cache_path = REPO_ROOT / "data" / "cache" / "ucfcrime_test.h5"
    if not cache_path.exists():
        pytest.skip("no real cache available")
    with h5py.File(cache_path, "r") as f:
        n_cached = f["clips"].shape[0]

    cfg = FeatureConfig(n_frames=N_FRAMES, height=H, width=W)
    out_path = Path("data") / "features" / "_test_idempotent.h5"
    try:
        backbone = build_backbone(cfg)
        build_split_features(cache_path, out_path, cfg, backbone)

        start = time.time()
        assert existing_features_valid(out_path, cfg, n_cached)
        elapsed = time.time() - start
        assert elapsed < 5.0
    finally:
        out_path.unlink(missing_ok=True)
