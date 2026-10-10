"""ONNX export of the shipped model, bundled end to end, plus the parity check.

See docs/SPRINT_PLAN.md Sprint 5 and docs/decisions/ADR-006-deployment-target.md.

The shipped checkpoint (ADR-005) is an LSTM head over precomputed MobileNetV2
features. Exporting the head alone would force every serving process to also
load TensorFlow just to compute those features, which defeats the point of
ONNX. So the exporter rebuilds the full inference path as one tf_keras graph,

    frames (N, T, H, W, 3) float32 RGB in [0, 255]
      -> Rescaling(1/127.5, offset=-1)        (MobileNetV2 preprocess_input)
      -> TimeDistributed(frozen MobileNetV2, pooling='avg')   (N, T, 1280)
      -> the fine-tuned LSTM head                              (N, 1)

and converts that with ADR-001's recipe: tf_keras `model.export(saved_model)`
then `python -m tf2onnx.convert --saved-model`. The backbone is built by
`safestreets.features.extract.build_backbone`, the same function that wrote
the feature store, so the bundled graph is the feature store plus the head,
not a reimplementation of it.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import tf_keras
from tf_keras import layers

from safestreets.data.preprocess import load_preprocess_config
from safestreets.evaluation.evaluate import load_checkpoint
from safestreets.features.extract import FeatureConfig, build_backbone
from safestreets.inference.spec import CONFIGS_DIR, NORMALISATION, active_spec

# isort: on

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=REPO_ROOT
        )
        return bool(out.strip())
    except Exception:
        return None


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_config_from_configs(configs_dir: Path | str = CONFIGS_DIR) -> FeatureConfig:
    pre = load_preprocess_config(Path(configs_dir) / "preprocess.yaml")
    cfg = FeatureConfig(n_frames=pre.n_frames, height=pre.height, width=pre.width)
    if cfg.backbone != NORMALISATION["backbone"] or cfg.weights != NORMALISATION["weights"]:
        raise ValueError(
            f"FeatureConfig backbone {cfg.backbone}/{cfg.weights} does not match the "
            f"normalisation contract in safestreets.inference.spec"
        )
    return cfg


def build_bundled_model(head: tf_keras.Model, feat_cfg: FeatureConfig) -> tf_keras.Model:
    """Raw letterboxed RGB frames in [0, 255] -> violence probability."""
    head_t, head_dim = head.input_shape[1], head.input_shape[2]
    if head_t != feat_cfg.n_frames:
        raise ValueError(f"head expects {head_t} frames, feature config has {feat_cfg.n_frames}")
    backbone = build_backbone(feat_cfg)
    if backbone.output_shape[-1] != head_dim:
        raise ValueError(f"backbone emits {backbone.output_shape[-1]}, head expects {head_dim}")

    frames = layers.Input(
        shape=(feat_cfg.n_frames, feat_cfg.height, feat_cfg.width, 3),
        dtype="float32",
        name="frames",
    )
    x = layers.Rescaling(NORMALISATION["scale"], offset=NORMALISATION["offset"])(frames)
    x = layers.TimeDistributed(backbone, name="mobilenetv2_features")(x)
    score = head(x)
    model = tf_keras.Model(frames, score, name="safestreets_bundled")
    model.trainable = False
    return model


def convert_to_onnx(model: tf_keras.Model, onnx_path: Path, opset: int) -> str:
    """ADR-001 recipe: SavedModel export, then the tf2onnx CLI. Returns its log."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        model.export(str(saved_model_dir))
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                str(saved_model_dir),
                "--output",
                str(onnx_path),
                "--opset",
                str(opset),
            ],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"tf2onnx failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return proc.stderr


def build_sidecar(
    onnx_path: Path,
    checkpoint: Path,
    tag: str,
    opset: int,
    feat_cfg: FeatureConfig,
    configs_dir: Path | str = CONFIGS_DIR,
) -> dict:
    """Everything the engine needs to serve, and to refuse to serve, this artefact."""
    import onnxruntime as ort

    spec = active_spec(configs_dir)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (inp,) = sess.get_inputs()
    (out,) = sess.get_outputs()
    checkpoint_sha = file_sha256(checkpoint)
    return {
        "format_version": 1,
        "tag": tag,
        "model_version": f"{tag}-{checkpoint_sha[:12]}",
        "onnx_file": onnx_path.name,
        "onnx_sha256": file_sha256(onnx_path),
        "opset": opset,
        "source_checkpoint": str(Path(checkpoint).resolve().relative_to(REPO_ROOT)),
        "source_checkpoint_sha256": checkpoint_sha,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "exported_utc": datetime.now(UTC).isoformat(),
        "onnx_input_name": inp.name,
        "onnx_output_name": out.name,
        "input": spec["input"],
        "normalisation": spec["normalisation"],
        "feature_config": {
            "backbone": feat_cfg.backbone,
            "weights": feat_cfg.weights,
            "pooling": feat_cfg.pooling,
            "sha": feat_cfg.sha(),
        },
        "head": spec["head"],
        "output": {"name": out.name, "meaning": "P(violent) per clip, sigmoid", "range": [0, 1]},
        "threshold": spec["threshold"],
        "threshold_source": "configs/infer.yaml",
    }


def export(
    checkpoint: Path | str,
    out_dir: Path | str,
    tag: str,
    opset: int = 13,
    configs_dir: Path | str = CONFIGS_DIR,
) -> tuple[Path, Path, tf_keras.Model]:
    """Returns (onnx_path, sidecar_path, bundled tf_keras model for parity)."""
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    feat_cfg = feature_config_from_configs(configs_dir)
    head = load_checkpoint(checkpoint)
    bundled = build_bundled_model(head, feat_cfg)

    onnx_path = out_dir / f"safestreets_{tag}.onnx"
    convert_to_onnx(bundled, onnx_path, opset)
    sidecar = build_sidecar(onnx_path, checkpoint, tag, opset, feat_cfg, configs_dir)
    sidecar_path = onnx_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return onnx_path, sidecar_path, bundled


def load_cached_clips(
    cache_dir: Path | str, datasets: list[str], split: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Real uint8 clips from the Phase 2 cache, (N, T, H, W, 3), RGB, letterboxed."""
    clips, labels, ids = [], [], []
    for dataset in datasets:
        path = Path(cache_dir) / f"{dataset}_{split}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            clips.append(f["clips"][:])
            labels.append(f["labels"][:])
            ids.extend(c.decode() if isinstance(c, bytes) else c for c in f["clip_ids"][:])
    if not clips:
        raise FileNotFoundError(f"no clip cache files for {datasets} split={split!r}")
    return np.concatenate(clips), np.concatenate(labels).astype(int), ids


def keras_scores(model: tf_keras.Model, clips: np.ndarray, batch_size: int = 16) -> np.ndarray:
    return (
        model.predict(clips.astype(np.float32), batch_size=batch_size, verbose=0)
        .reshape(-1)
        .astype(np.float64)
    )
