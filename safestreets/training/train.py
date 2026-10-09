"""Train either the `lstm_head` (on precomputed features) or `scratch` (on the
live augmented pipeline) architecture. See docs/SPRINT_PLAN.md Sprint 1.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import os

# MLflow 3.x's file store is in maintenance mode and refuses to run without
# this; BUILD_PLAN's "./mlruns file store" intent is kept as-is rather than
# migrating to a sqlite backend under this deadline.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import h5py
import mlflow
import numpy as np
import tf_keras
import yaml

from safestreets.data.augment import load_augment_config
from safestreets.data.dataset import load_pipeline_config, make_dataset
from safestreets.models.factory import build_model, load_model_yaml
from safestreets.training.callbacks import build_callbacks
from safestreets.training.losses import compute_pos_weight

# isort: on


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    features_dir: str
    checkpoints_dir: str
    mlruns_dir: str
    tensorboard_dir: str
    train_datasets: list[str]
    val_datasets: list[str]
    batch_size: int
    epochs: int
    early_stop_patience: int
    early_stop_monitor: str
    histogram_freq: int
    scratch_epochs: int
    scratch_batch_size: int


def load_train_config(path: Path | str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_feature_split(
    features_dir: Path | str, datasets: list[str], split: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenate `{dataset}_{split}.h5` feature files for the given datasets."""
    features, labels, clip_ids = [], [], []
    for dataset in datasets:
        path = Path(features_dir) / f"{dataset}_{split}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            features.append(f["features"][:])
            labels.append(f["labels"][:])
            ids = f["clip_ids"][:]
            clip_ids.extend(c.decode() if isinstance(c, bytes) else c for c in ids)
    if not features:
        raise FileNotFoundError(f"no feature files found for {datasets} split={split!r}")
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), clip_ids


def train_lstm_head(
    model_cfg_path: Path | str, train_cfg: TrainConfig, run_tag: str = "lstm_head_baseline"
) -> dict:
    """Production run: head trained on precomputed train features, early-stopped on val_auc."""
    raw_model_cfg = load_model_yaml(model_cfg_path)
    raw_model_cfg["arch"] = "lstm_head"

    x_train, y_train, _ = load_feature_split(
        train_cfg.features_dir, train_cfg.train_datasets, "train"
    )
    x_val, y_val, _ = load_feature_split(train_cfg.features_dir, train_cfg.val_datasets, "val")

    pos_weight = compute_pos_weight(y_train)
    model = build_model(raw_model_cfg, pos_weight=pos_weight)
    callbacks = build_callbacks(
        run_tag,
        train_cfg.checkpoints_dir,
        train_cfg.tensorboard_dir,
        monitor=train_cfg.early_stop_monitor,
        patience=train_cfg.early_stop_patience,
        histogram_freq=train_cfg.histogram_freq,
    )

    mlflow.set_tracking_uri(f"file:{train_cfg.mlruns_dir}")
    mlflow.set_experiment("safestreets-sprint1")
    with mlflow.start_run(run_name=run_tag):
        mlflow.log_params(
            {
                "arch": "lstm_head",
                "pos_weight": pos_weight,
                "git_sha": _git_sha(),
                "seed": train_cfg.seed,
                **raw_model_cfg["lstm_head"],
            }
        )
        history = model.fit(
            x_train,
            y_train.astype(np.float32),
            validation_data=(x_val, y_val.astype(np.float32)),
            batch_size=train_cfg.batch_size,
            epochs=train_cfg.epochs,
            callbacks=callbacks,
            verbose=2,
        )
        best_val_auc = float(max(history.history["val_auc"]))
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.keras.log_model(model, artifact_path="model")

    return {
        "run_tag": run_tag,
        "pos_weight": pos_weight,
        "best_val_auc": best_val_auc,
        "checkpoint": str(Path(train_cfg.checkpoints_dir) / f"{run_tag}.keras"),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }


def train_scratch(
    model_cfg_path: Path | str,
    train_cfg: TrainConfig,
    data_yaml_path: Path | str,
    cache_dir: Path | str,
    run_tag: str = "scratch_comparison",
) -> dict:
    """Comparison arm: scratch CNN-LSTM, live augmented pipeline, `scratch_epochs` epochs."""
    raw_model_cfg = load_model_yaml(model_cfg_path)
    raw_model_cfg["arch"] = "scratch"

    pipeline_cfg = load_pipeline_config(data_yaml_path)
    augment_cfg = load_augment_config(data_yaml_path)

    with h5py.File(Path(cache_dir) / "rwf2000_train.h5", "r") as f:
        pos_weight_train_labels = f["labels"][:]

    pos_weight = compute_pos_weight(pos_weight_train_labels)
    model = build_model(raw_model_cfg, pos_weight=pos_weight)

    train_ds = make_dataset(
        "train", pipeline_cfg, augment_cfg, cache_dir, training=True, seed=train_cfg.seed
    )
    val_ds = make_dataset(
        "val", pipeline_cfg, augment_cfg, cache_dir, training=False, seed=train_cfg.seed
    )

    callbacks = build_callbacks(
        run_tag,
        train_cfg.checkpoints_dir,
        train_cfg.tensorboard_dir,
        monitor=train_cfg.early_stop_monitor,
        patience=train_cfg.early_stop_patience,
        histogram_freq=train_cfg.histogram_freq,
    )

    mlflow.set_tracking_uri(f"file:{train_cfg.mlruns_dir}")
    mlflow.set_experiment("safestreets-sprint1")
    with mlflow.start_run(run_name=run_tag):
        mlflow.log_params(
            {
                "arch": "scratch",
                "pos_weight": pos_weight,
                "git_sha": _git_sha(),
                "seed": train_cfg.seed,
                **raw_model_cfg["scratch"],
            }
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg.scratch_epochs,
            callbacks=callbacks,
            verbose=2,
        )
        best_val_auc = float(max(history.history["val_auc"]))
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.keras.log_model(model, artifact_path="model")

    return {
        "run_tag": run_tag,
        "pos_weight": pos_weight,
        "best_val_auc": best_val_auc,
        "checkpoint": str(Path(train_cfg.checkpoints_dir) / f"{run_tag}.keras"),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }


def save_model_summary(model: tf_keras.Model, path: Path | str) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    Path(path).write_text("\n".join(lines) + "\n")


def save_json(obj: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
