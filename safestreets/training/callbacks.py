"""MLflow, TensorBoard, checkpointing, early stopping. See docs/SPRINT_PLAN.md Sprint 1."""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

from pathlib import Path

import mlflow
import tf_keras

# isort: on


class MLflowMetricsCallback(tf_keras.callbacks.Callback):
    """Logs every epoch's metrics to the active MLflow run."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs:
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


def build_callbacks(
    run_tag: str,
    checkpoints_dir: Path | str,
    tensorboard_dir: Path | str,
    monitor: str = "val_auc",
    patience: int = 8,
    histogram_freq: int = 1,
) -> list[tf_keras.callbacks.Callback]:
    """`monitor` is maximised (AUC/accuracy), matching `mode="max"` below."""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tensorboard_dir) / run_tag

    return [
        tf_keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoints_dir / f"{run_tag}.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
        ),
        tf_keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=patience, restore_best_weights=True
        ),
        tf_keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=histogram_freq),
        MLflowMetricsCallback(),
    ]
