"""Sprint 2 evaluation harness: load the Sprint 1 checkpoint, score feature-store
splits, select an operating threshold on validation, and render the standard
figure set. See docs/SPRINT_PLAN.md Sprint 2.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tf_keras
import yaml

from safestreets.evaluation.metrics import (
    compute_metrics,
    pr_points,
    roc_points,
    select_f1_optimal,
    select_recall_floor,
    threshold_sweep,
)
from safestreets.training.train import load_feature_split

# isort: on

RECALL_FLOOR = 0.90


def load_checkpoint(path: Path | str) -> tf_keras.Model:
    """Loaded `compile=False`: evaluation only ever calls `.predict`, never
    `.fit`/`.evaluate`, so the training-time custom loss never needs to be
    reconstructed here.
    """
    return tf_keras.models.load_model(str(path), compile=False)


def predict_scores(model: tf_keras.Model, features: np.ndarray, batch_size: int = 64) -> np.ndarray:
    return model.predict(features, batch_size=batch_size, verbose=0).reshape(-1).astype(float)


def score_split(
    model: tf_keras.Model, features_dir: Path | str, datasets: list[str], split: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features, labels, clip_ids = load_feature_split(features_dir, datasets, split)
    scores = predict_scores(model, features)
    return labels.astype(int), scores, clip_ids


def select_threshold(y_val: np.ndarray, s_val: np.ndarray) -> dict:
    """Both operating points are chosen on validation only, never on test."""
    sweep = threshold_sweep(y_val, s_val)
    f1_threshold = select_f1_optimal(sweep)
    recall_threshold = select_recall_floor(sweep, RECALL_FLOOR)
    return {
        "sweep": sweep,
        "f1_optimal": {
            "threshold": f1_threshold,
            "metrics": compute_metrics(y_val, s_val, f1_threshold, n_resamples=200),
        },
        "recall_floor": {
            "floor": RECALL_FLOOR,
            "threshold": recall_threshold,
            "metrics": compute_metrics(y_val, s_val, recall_threshold, n_resamples=200),
        },
    }


def write_infer_config(selection: dict, path: Path | str, git_sha: str) -> None:
    payload = {
        "selected_on": "val",
        "git_sha": git_sha,
        "threshold": selection["f1_optimal"]["threshold"],
        "threshold_strategy": "f1_optimal",
        "threshold_recall_floor_0_90": selection["recall_floor"]["threshold"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def render_roc(y_true, y_score, out_path: Path | str) -> None:
    points = roc_points(y_true, y_score)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(points["fpr"], points["tpr"], label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_pr(y_true, y_score, out_path: Path | str) -> None:
    points = pr_points(y_true, y_score)
    from sklearn.metrics import average_precision_score

    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(points["recall"], points["precision"], label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_confusion(y_true, y_score, threshold: float, out_path: Path | str) -> None:
    from safestreets.evaluation.metrics import confusion_counts

    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    ax.set_title(f"Confusion matrix (threshold={threshold:.3f})")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_threshold_sweep(sweep: dict, out_path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["thresholds"], sweep["f1"], label="F1")
    ax.plot(sweep["thresholds"], sweep["recall"], label="Recall")
    ax.plot(sweep["thresholds"], sweep["precision"], label="Precision")
    ax.axhline(RECALL_FLOOR, linestyle=":", color="gray", label=f"recall={RECALL_FLOOR}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Validation threshold sweep")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_json(obj: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
