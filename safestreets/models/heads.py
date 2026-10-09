"""The Sprint 1 production model: an LSTM head over precomputed MobileNetV2 features.

See docs/SPRINT_PLAN.md Sprint 1 and docs/decisions/ADR-003-architecture.md.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

from dataclasses import dataclass

import tf_keras
from tf_keras import layers

# isort: on


@dataclass(frozen=True)
class LSTMHeadConfig:
    input_dim: int = 1280
    n_frames: int = 16
    units: int = 128
    dropout: float = 0.4
    learning_rate: float = 0.001


def build_lstm_head(cfg: LSTMHeadConfig) -> tf_keras.Model:
    """`LSTM(units, return_sequences) -> LSTM(units//2) -> Dropout -> Dense(1, sigmoid)`."""
    inputs = layers.Input(shape=(cfg.n_frames, cfg.input_dim), name="features")
    x = layers.LSTM(cfg.units, return_sequences=True)(inputs)
    x = layers.LSTM(cfg.units // 2)(x)
    x = layers.Dropout(cfg.dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf_keras.Model(inputs, outputs, name="lstm_head")
