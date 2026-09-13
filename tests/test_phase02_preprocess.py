from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from safestreets.data.preprocess import (
    PreprocessConfig,
    build_split_cache,
    existing_cache_is_valid,
    letterbox_resize,
    read_clip,
    sample_frame_indices,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_video(path: Path, frames: list, fps: int = 10) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.mark.phase2
def test_sample_frame_indices_spans_full_duration():
    idx = sample_frame_indices(100, 16)
    assert idx.min() == 0
    assert idx.max() == 99
    assert len(idx) == 16


@pytest.mark.phase2
def test_sample_frame_indices_not_first_sixteen():
    idx = sample_frame_indices(100, 16)
    assert idx.max() > 15


@pytest.mark.phase2
def test_sample_frame_indices_loop_pads_short_clips():
    idx = sample_frame_indices(5, 16)
    assert len(idx) == 16
    assert idx.max() == 4
    assert set(idx) == {0, 1, 2, 3, 4}


@pytest.mark.phase2
def test_letterbox_preserves_aspect_ratio():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = letterbox_resize(frame, 112, 112)
    assert out.shape == (112, 112, 3)


@pytest.mark.phase2
def test_letterbox_pads_not_squashes():
    # a tall frame should letterbox with padding on left/right, not stretch.
    frame = np.full((200, 100, 3), 255, dtype=np.uint8)
    out = letterbox_resize(frame, 112, 112)
    assert out[:, 0, :].sum() == 0  # left padding column is black
    assert out[:, -1, :].sum() == 0  # right padding column is black
    assert out[:, 56, :].sum() > 0  # center column has content


@pytest.mark.phase2
def test_uniform_sampling_burned_in_index(tmp_path):
    n_source_frames = 100
    frames = []
    for i in range(n_source_frames):
        val = round(i / (n_source_frames - 1) * 255)
        frames.append(np.full((64, 64, 3), val, dtype=np.uint8))
    video_path = tmp_path / "synthetic.mp4"
    _write_video(video_path, frames)

    cfg = PreprocessConfig(n_frames=16, height=32, width=32)
    result = read_clip(str(video_path), cfg)
    assert result.frames is not None

    decoded = [round(f.mean() / 255 * (n_source_frames - 1)) for f in result.frames]
    assert min(decoded) <= 10
    assert max(decoded) >= n_source_frames - 1 - 10


@pytest.mark.phase2
def test_rgb_order_on_pure_red_video(tmp_path):
    # OpenCV frames are BGR: high blue-channel (index 0), zero elsewhere == pure blue.
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR channel 2 = red
    video_path = tmp_path / "red.mp4"
    _write_video(video_path, [frame] * 20)

    cfg = PreprocessConfig(n_frames=16, height=32, width=32)
    result = read_clip(str(video_path), cfg)
    assert result.frames is not None
    # after BGR->RGB, channel 0 (red) should dominate channel 2 (blue).
    assert result.frames[..., 0].mean() > result.frames[..., 2].mean()


@pytest.mark.phase2
def test_read_clip_skips_unreadable_file(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"not a real video file")
    cfg = PreprocessConfig()
    result = read_clip(str(bogus), cfg)
    assert result.frames is None
    assert result.skip_reason is not None


@pytest.mark.phase2
def test_short_clip_flagged_padded(tmp_path):
    frames = [np.full((64, 64, 3), i * 20, dtype=np.uint8) for i in range(5)]
    video_path = tmp_path / "short.mp4"
    _write_video(video_path, frames)

    cfg = PreprocessConfig(n_frames=16, height=32, width=32)
    result = read_clip(str(video_path), cfg)
    assert result.frames is not None
    assert result.padded is True


@pytest.mark.phase2
def test_build_split_cache_round_trip(tmp_path):
    n_clips = 20
    rows = []
    for i in range(n_clips):
        frames = [np.full((64, 64, 3), (i * 7 + f * 3) % 256, dtype=np.uint8) for f in range(20)]
        video_path = tmp_path / f"clip_{i}.mp4"
        _write_video(video_path, frames)
        rows.append(
            {"clip_id": f"clip_{i}", "path": str(video_path), "label": i % 2}
        )
    df = pd.DataFrame(rows)

    cfg = PreprocessConfig(n_frames=16, height=32, width=32)
    out_path = tmp_path / "cache.h5"
    report = build_split_cache(df, out_path, cfg, workers=1)

    assert report["n_written"] + report["n_skipped"] == report["n_manifest"] == n_clips
    assert report["n_written"] == n_clips

    label_by_clip_id = {r["clip_id"]: r["label"] for r in rows}
    with h5py.File(out_path, "r") as f:
        assert f["clips"].shape == (n_clips, 16, 32, 32, 3)
        assert f["clips"].dtype == np.uint8
        assert f["labels"].shape == (n_clips,)
        assert f.attrs["config_sha"] == cfg.sha()
        for i in range(n_clips):
            assert f["clips"][i].std() > 0
            clip_id = f["clip_ids"][i].decode()
            assert f["labels"][i] == label_by_clip_id[clip_id]


@pytest.mark.phase2
def test_cache_is_idempotent(tmp_path):
    frames = [np.full((64, 64, 3), f * 10, dtype=np.uint8) for f in range(20)]
    video_path = tmp_path / "clip_0.mp4"
    _write_video(video_path, frames)
    df = pd.DataFrame([{"clip_id": "clip_0", "path": str(video_path), "label": 1}])

    cfg = PreprocessConfig(n_frames=16, height=32, width=32)
    out_path = tmp_path / "cache.h5"
    build_split_cache(df, out_path, cfg, workers=1)

    assert existing_cache_is_valid(out_path, cfg, n_expected=1)
    assert not existing_cache_is_valid(out_path, cfg, n_expected=2)

    other_cfg = PreprocessConfig(n_frames=8, height=32, width=32)
    assert not existing_cache_is_valid(out_path, other_cfg, n_expected=1)


@pytest.mark.phase2
def test_skipped_clips_are_logged_not_faked(tmp_path):
    good_frames = [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(20)]
    good_path = tmp_path / "good.mp4"
    _write_video(good_path, good_frames)
    bad_path = tmp_path / "bad.mp4"
    bad_path.write_bytes(b"garbage")

    df = pd.DataFrame(
        [
            {"clip_id": "good", "path": str(good_path), "label": 1},
            {"clip_id": "bad", "path": str(bad_path), "label": 0},
        ]
    )
    cfg = PreprocessConfig(n_frames=16, height=32, width=32)
    out_path = tmp_path / "cache.h5"
    report = build_split_cache(df, out_path, cfg, workers=1)

    assert report["n_written"] == 1
    assert report["n_skipped"] == 1
    assert report["skipped"][0]["clip_id"] == "bad"
    with h5py.File(out_path, "r") as f:
        assert f["clips"].shape[0] == 1


@pytest.mark.phase2
@pytest.mark.needs_data
def test_cache_report_matches_manifest_counts():
    cache_dir = REPO_ROOT / "data" / "cache"
    report_path = cache_dir / "cache_report.json"
    manifest_path = REPO_ROOT / "data" / "manifests" / "clips.parquet"
    if not report_path.exists() or not manifest_path.exists():
        pytest.skip("cache not built yet; run scripts/build_cache.py first")

    import json

    report = json.loads(report_path.read_text())
    df = pd.read_parquet(manifest_path)

    for dataset, splits in report.items():
        for split, entry in splits.items():
            n_manifest = len(df[(df["dataset"] == dataset) & (df["split"] == split)])
            assert entry["n_written"] + entry["n_skipped"] == n_manifest == entry["n_manifest"]


@pytest.mark.phase2
@pytest.mark.needs_data
def test_cache_files_have_expected_shape_and_dtype():
    cache_dir = REPO_ROOT / "data" / "cache"
    h5_files = sorted(cache_dir.glob("*.h5"))
    if not h5_files:
        pytest.skip("cache not built yet; run scripts/build_cache.py first")

    for path in h5_files:
        with h5py.File(path, "r") as f:
            assert f["clips"].dtype == np.uint8
            assert f["clips"].shape[1:] == (16, 112, 112, 3)


@pytest.mark.phase2
@pytest.mark.needs_data
def test_cache_total_size_within_budget():
    cache_dir = REPO_ROOT / "data" / "cache"
    h5_files = list(cache_dir.glob("*.h5"))
    if not h5_files:
        pytest.skip("cache not built yet; run scripts/build_cache.py first")
    total_bytes = sum(p.stat().st_size for p in h5_files)
    assert total_bytes < 4 * 1024**3
