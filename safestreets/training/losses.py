"""Class-weighted binary cross-entropy. Report §4.3. See docs/SPRINT_PLAN.md Sprint 1."""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import tensorflow as tf

# isort: on


def compute_pos_weight(labels) -> float:
    """`pos_weight = n_neg / n_pos`, computed from the *train* manifest only."""
    import numpy as np

    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        raise ValueError("no positive labels found; cannot compute pos_weight")
    return n_neg / n_pos


def weighted_bce(pos_weight: float):
    """Binary cross-entropy with the positive class scaled by `pos_weight`."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(
            pos_weight * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(bce)

    loss_fn.__name__ = "weighted_bce"
    return loss_fn
