"""AIRTLab fine-tune of the Sprint 1 `lstm_head` baseline, plus a catastrophic
forgetting check against RWF-2000. See docs/SPRINT_PLAN.md Sprint 4 and
docs/decisions/ADR-005-finetune-and-attribution.md.

AIRTLab is small (175 recorded events, 350 clips across both camera views,
ADR-002), so a single train/val/test fit on it would be a fragile one-shot
number. Instead this module pools all three AIRTLab splits and runs 5-fold
grouped cross-validation (`group_id` from ADR-002, so both camera views of
one event always land in the same fold), reporting mean +/- std rather than
a point estimate.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tf_keras
from sklearn.model_selection import GroupKFold

from safestreets.evaluation.evaluate import load_checkpoint, predict_scores
from safestreets.evaluation.metrics import compute_metrics
from safestreets.training.losses import compute_pos_weight, weighted_bce
from safestreets.training.train import TrainConfig, load_feature_split

# isort: on

N_FOLDS = 5
FINETUNE_EPOCHS = 15
FINETUNE_PATIENCE = 5
FINETUNE_LEARNING_RATE = 1e-4
FINETUNE_BATCH_SIZE = 16


def load_airtlab_pool(
    features_dir: Path | str, manifest_path: Path | str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Concatenates AIRTLab train+val+test features into one pool and looks up
    each clip's `group_id` (ADR-002: `{label}_{clip_number}`, shared by both
    camera views of one event) from the Phase 1 manifest, so `GroupKFold`
    below can never split one event's two camera views across folds.
    """
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ids: list[str] = []
    for split in ("train", "val", "test"):
        x, y, clip_ids = load_feature_split(features_dir, ["airtlab"], split)
        x_parts.append(x)
        y_parts.append(y)
        ids.extend(clip_ids)

    manifest = pd.read_parquet(manifest_path)
    group_lookup = manifest.set_index("clip_id")["group_id"].to_dict()
    missing = [c for c in ids if c not in group_lookup]
    if missing:
        raise KeyError(f"{len(missing)} AIRTLab clip_ids missing from manifest, e.g. {missing[0]}")
    groups = np.array([group_lookup[c] for c in ids])

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0), groups, ids


def _fit_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    baseline_checkpoint: Path | str,
    seed: int,
    epochs: int = FINETUNE_EPOCHS,
) -> tf_keras.Model:
    """Starts from the Sprint 1 baseline's learned weights (not a fresh init)
    and continues training at a low learning rate: fine-tuning, not retraining.
    """
    tf_keras.utils.set_random_seed(seed)
    model = load_checkpoint(baseline_checkpoint)
    pos_weight = compute_pos_weight(y_train)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=FINETUNE_LEARNING_RATE),
        loss=weighted_bce(pos_weight),
        metrics=[tf_keras.metrics.AUC(name="auc")],
    )
    model.fit(
        x_train,
        y_train.astype(np.float32),
        batch_size=FINETUNE_BATCH_SIZE,
        epochs=epochs,
        callbacks=[
            tf_keras.callbacks.EarlyStopping(
                monitor="loss", patience=FINETUNE_PATIENCE, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    return model


def run_grouped_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    baseline_checkpoint: Path | str,
    seed: int,
    n_folds: int = N_FOLDS,
    epochs: int = FINETUNE_EPOCHS,
) -> list[dict]:
    """One fine-tune + held-out-fold evaluation per fold. Each fold starts
    fresh from `baseline_checkpoint`'s weights (never from a previous fold's
    fine-tuned weights), so folds are independent estimates of the same
    procedure, not five steps of one continued fit.
    """
    gkf = GroupKFold(n_splits=n_folds)
    fold_reports = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups)):
        model = _fit_fold(x[train_idx], y[train_idx], baseline_checkpoint, seed + fold_i, epochs)
        scores = predict_scores(model, x[test_idx])
        metrics = compute_metrics(y[test_idx], scores, threshold=0.5, n_resamples=200, seed=seed)
        fold_reports.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": metrics["accuracy"]["value"],
                "f1": metrics["f1"]["value"],
                "roc_auc": metrics["roc_auc"]["value"],
            }
        )
    return fold_reports


def summarize_cv(fold_reports: list[dict]) -> dict:
    """Mean +/- std across folds for each metric, per the Sprint 4 DoD."""
    out = {}
    for key in ("accuracy", "f1", "roc_auc"):
        values = np.array([f[key] for f in fold_reports])
        out[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


def fit_final_airtlab_model(
    x: np.ndarray,
    y: np.ndarray,
    baseline_checkpoint: Path | str,
    seed: int,
    checkpoints_dir: Path | str,
    epochs: int = FINETUNE_EPOCHS,
    run_tag: str = "finetuned_airtlab_lstm_head",
) -> Path:
    """The checkpoint actually shipped (if any, per ADR-005): one fine-tune on
    the full AIRTLab pool, no held-out fold, since the per-fold numbers above
    already measured what this procedure does on unseen AIRTLab data.
    """
    model = _fit_fold(x, y, baseline_checkpoint, seed, epochs)
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{run_tag}.keras"
    model.save(checkpoint_path)
    return checkpoint_path


def forgetting_check(
    baseline_checkpoint: Path | str, finetuned_checkpoint: Path | str, train_cfg: TrainConfig
) -> dict:
    """RWF-2000 val ROC-AUC before vs. after the AIRTLab fine-tune. This is
    the one number that decides whether the fine-tuned checkpoint is safe to
    ship (ADR-005): a fine-tune that raises AIRTLab accuracy by destroying
    RWF-2000 performance is a worse production model, not a better one.
    """
    x_val, y_val, _ = load_feature_split(train_cfg.features_dir, ["rwf2000"], "val")

    before_model = load_checkpoint(baseline_checkpoint)
    after_model = load_checkpoint(finetuned_checkpoint)

    before_scores = predict_scores(before_model, x_val)
    after_scores = predict_scores(after_model, x_val)

    before_metrics = compute_metrics(y_val, before_scores, threshold=0.5, n_resamples=200)
    after_metrics = compute_metrics(y_val, after_scores, threshold=0.5, n_resamples=200)

    return {
        "dataset": "rwf2000",
        "split": "val",
        "n": int(len(y_val)),
        "roc_auc_before": before_metrics["roc_auc"]["value"],
        "roc_auc_after": after_metrics["roc_auc"]["value"],
        "roc_auc_drop": before_metrics["roc_auc"]["value"] - after_metrics["roc_auc"]["value"],
    }


def save_json(obj: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
