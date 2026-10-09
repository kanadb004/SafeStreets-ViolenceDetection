"""Dispatch on `cfg.model.arch in {lstm_head, scratch}`. See docs/SPRINT_PLAN.md Sprint 1."""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

from pathlib import Path

import tf_keras
import yaml

from safestreets.models.cnn_lstm import ScratchConfig, build_scratch_cnn_lstm
from safestreets.models.heads import LSTMHeadConfig, build_lstm_head
from safestreets.training.losses import weighted_bce

# isort: on

_ARCHS = ("lstm_head", "scratch")


def load_model_yaml(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(raw_cfg: dict, pos_weight: float = 1.0) -> tf_keras.Model:
    """`raw_cfg` is the parsed `configs/model.yaml` dict. Returns a compiled model."""
    arch = raw_cfg["arch"]
    if arch not in _ARCHS:
        raise ValueError(f"arch must be one of {_ARCHS}, got {arch!r}")

    if arch == "lstm_head":
        cfg = LSTMHeadConfig(**raw_cfg["lstm_head"])
        model = build_lstm_head(cfg)
        learning_rate = cfg.learning_rate
    else:
        scratch_raw = dict(raw_cfg["scratch"])
        scratch_raw["conv_filters"] = tuple(scratch_raw["conv_filters"])
        cfg = ScratchConfig(**scratch_raw)
        model = build_scratch_cnn_lstm(cfg)
        learning_rate = cfg.learning_rate

    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate),
        loss=weighted_bce(pos_weight),
        metrics=[
            tf_keras.metrics.BinaryAccuracy(name="accuracy"),
            tf_keras.metrics.AUC(name="auc"),
            tf_keras.metrics.Precision(name="precision"),
            tf_keras.metrics.Recall(name="recall"),
        ],
    )
    return model
