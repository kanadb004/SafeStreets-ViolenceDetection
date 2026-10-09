"""The scratch TimeDistributed CNN-LSTM, report §4.3's literal architecture.

Comparison arm only (see docs/SPRINT_PLAN.md Sprint 1): trained end to end
through the live Phase 3 augmented `tf.data` pipeline on raw clips, unlike the
`lstm_head` production model which trains on frozen, precomputed features.

`GlobalAveragePooling2D`, never `Flatten`, after the last conv block: with
`Flatten`, 112x112 input after three pool stages is ~14x14x128 = 25,088
features/frame, which blows up the LSTM's parameter count and overfits 4,000
clips instantly (see `app/ml_model/model.py`'s known defect, CLAUDE.md).
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

from dataclasses import dataclass

import tf_keras
from tf_keras import layers

# isort: on


@dataclass(frozen=True)
class ScratchConfig:
    n_frames: int = 16
    height: int = 112
    width: int = 112
    conv_filters: tuple[int, ...] = (32, 64, 128)
    bottleneck_units: int = 256
    lstm_units: int = 64
    dropout: float = 0.4
    learning_rate: float = 0.001


def build_scratch_cnn_lstm(cfg: ScratchConfig) -> tf_keras.Model:
    inputs = layers.Input(shape=(cfg.n_frames, cfg.height, cfg.width, 3), name="clips")
    x = inputs
    for filters in cfg.conv_filters[:-1]:
        x = layers.TimeDistributed(layers.Conv2D(filters, 3, activation="relu", padding="same"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(
        layers.Conv2D(cfg.conv_filters[-1], 3, activation="relu", padding="same")
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.TimeDistributed(layers.Dense(cfg.bottleneck_units, activation="relu"))(x)
    x = layers.LSTM(cfg.lstm_units, return_sequences=True)(x)
    x = layers.LSTM(cfg.lstm_units // 2)(x)
    x = layers.Dropout(cfg.dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf_keras.Model(inputs, outputs, name="scratch_cnn_lstm")
