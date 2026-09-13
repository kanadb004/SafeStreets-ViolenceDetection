"""Video -> fixed-shape uint8 clip cache. See docs/BUILD_PLAN.md Phase 2 spec.

Frames are sampled uniformly across a clip's full duration (never just the first
`n_frames`), resized with aspect-preserving letterboxing (naive squashing distorts
human figures and would hurt the Phase 8 gender module), and converted BGR->RGB at
read time since OpenCV decodes BGR but every downstream consumer (Albumentations,
ImageNet backbones, ONNX) expects RGB. Pixels are stored as uint8 0-255;
normalisation happens in the Phase 3 graph, never on disk.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class PreprocessConfig:
    n_frames: int = 16
    height: int = 112
    width: int = 112
    sampling: str = "uniform"

    def sha(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_preprocess_config(path: Path | str) -> PreprocessConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PreprocessConfig(**raw)


def sample_frame_indices(n_frames: int, t: int) -> np.ndarray:
    """Uniformly sample `t` indices spanning [0, n_frames).

    Clips with fewer than `t` real frames are loop-padded (the sequence is
    repeated) rather than zero-padded, per BUILD_PLAN Phase 2 spec.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if n_frames >= t:
        return np.linspace(0, n_frames - 1, num=t).round().astype(int)
    reps = int(np.ceil(t / n_frames))
    return np.tile(np.arange(n_frames), reps)[:t]


def letterbox_resize(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize preserving aspect ratio, zero-padding to exactly (target_h, target_w)."""
    h, w = frame.shape[:2]
    scale = min(target_h / h, target_w / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


@dataclass
class ClipResult:
    frames: np.ndarray | None  # (T, H, W, 3) uint8, RGB; None if skipped
    padded: bool
    skip_reason: str | None


def read_clip(path: str, cfg: PreprocessConfig) -> ClipResult:
    """Read one clip from disk into a (T, H, W, 3) uint8 RGB tensor.

    Returns a ClipResult with frames=None and a skip_reason for unreadable or
    zero-frame files; never fabricates a zeros clip for a failed read.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            return ClipResult(None, False, "zero_frames_reported")

        indices = sample_frame_indices(n_frames, cfg.n_frames)
        padded = n_frames < cfg.n_frames
        frames = []
        last_pos = -1
        for idx in indices:
            idx = int(idx)
            if idx != last_pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            last_pos = idx
            if not ok or frame is None:
                return ClipResult(None, padded, "unreadable_frame")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(letterbox_resize(rgb, cfg.height, cfg.width))

        arr = np.stack(frames, axis=0).astype(np.uint8)
        return ClipResult(arr, padded, None)
    finally:
        cap.release()


def cache_path(cache_dir: Path, dataset: str, split: str) -> Path:
    return Path(cache_dir) / f"{dataset}_{split}.h5"


def existing_cache_is_valid(path: Path, cfg: PreprocessConfig, n_expected: int) -> bool:
    """True if `path` already holds a complete cache for `cfg` and the current manifest.

    Backs the idempotency requirement: a second `build_cache.py` run for an
    unchanged (dataset, split, config) does no recompute.
    """
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return (
                f.attrs.get("config_sha") == cfg.sha()
                and f.attrs.get("n_manifest") == n_expected
            )
    except OSError:
        return False


def build_split_cache(
    rows: pd.DataFrame,
    out_path: Path,
    cfg: PreprocessConfig,
    workers: int = 1,
) -> dict:
    """Build one `.h5` cache file for a single (dataset, split)'s manifest rows.

    Uses `ProcessPoolExecutor` (OpenCV's `VideoCapture` is not thread-safe) with
    a single writer process collecting results before touching HDF5. Returns a
    JSON-serialisable report dict; the caller is responsible for merging it into
    `cache_report.json` and for any `--purge-raw` deletion (which needs a
    post-write re-read, done by the caller after this returns).
    """
    rows = rows.reset_index(drop=True)
    n = len(rows)
    start = time.time()

    results: list[ClipResult | None] = [None] * n
    if n > 0 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(read_clip, rows["path"][i], cfg): i for i in range(n)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
    else:
        for i in range(n):
            results[i] = read_clip(rows["path"][i], cfg)

    written_idx = [i for i, r in enumerate(results) if r is not None and r.frames is not None]
    skipped = [
        {"clip_id": rows["clip_id"][i], "path": rows["path"][i], "reason": results[i].skip_reason}
        for i in range(n)
        if results[i] is None or results[i].frames is None
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as f:
        n_written = len(written_idx)
        clips_ds = f.create_dataset(
            "clips", shape=(n_written, cfg.n_frames, cfg.height, cfg.width, 3), dtype="uint8"
        )
        labels_ds = f.create_dataset("labels", shape=(n_written,), dtype="uint8")
        ids_ds = f.create_dataset("clip_ids", shape=(n_written,), dtype=h5py.string_dtype())
        padded_ds = f.create_dataset("padded", shape=(n_written,), dtype="bool")
        for j, i in enumerate(written_idx):
            r = results[i]
            clips_ds[j] = r.frames
            labels_ds[j] = int(rows["label"][i])
            ids_ds[j] = str(rows["clip_id"][i])
            padded_ds[j] = r.padded
        f.attrs["n_frames"] = cfg.n_frames
        f.attrs["height"] = cfg.height
        f.attrs["width"] = cfg.width
        f.attrs["sampling"] = cfg.sampling
        f.attrs["created_utc"] = datetime.now(UTC).isoformat()
        f.attrs["config_sha"] = cfg.sha()
        f.attrs["n_manifest"] = n
    tmp_path.replace(out_path)

    written = [
        {"clip_id": rows["clip_id"][i], "path": rows["path"][i]} for i in written_idx
    ]
    return {
        "n_manifest": n,
        "n_written": len(written_idx),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "written": written,
        "bytes": out_path.stat().st_size,
        "wall_s": time.time() - start,
    }
