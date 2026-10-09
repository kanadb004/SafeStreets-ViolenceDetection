"""Point metrics plus bootstrap 95% CIs. See docs/SPRINT_PLAN.md Sprint 2.

Point estimates are thin wrappers over `sklearn.metrics` (already a project
dependency), not reimplementations, so the "agrees with sklearn to 1e-9" DoD
item is a tautology by construction rather than a coincidence of a parallel
implementation drifting back into agreement.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Metrics that depend on a chosen threshold take (y_true, y_score, threshold);
# threshold-free metrics (the AUCs) ignore the third argument.
_POINT_METRICS: dict[str, callable] = {
    "accuracy": lambda y, s, t: accuracy_score(y, s >= t),
    "precision": lambda y, s, t: precision_score(y, s >= t, zero_division=0),
    "recall": lambda y, s, t: recall_score(y, s >= t, zero_division=0),
    "f1": lambda y, s, t: f1_score(y, s >= t, zero_division=0),
    "roc_auc": lambda y, s, t: roc_auc_score(y, s),
    "pr_auc": lambda y, s, t: average_precision_score(y, s),
}


def confusion_counts(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 1265,
) -> tuple[float, float]:
    """95% percentile bootstrap CI. Resamples lacking both classes are
    skipped (degenerate for roc_auc/pr_auc); if fewer than 2 resamples
    survive (possible on a tiny split like the 5-clip UCF-Crime test set)
    the CI is reported as `[nan, nan]` rather than a false-precision number.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        values.append(metric_fn(yt, ys, threshold))
    if len(values) < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    seed: int = 1265,
) -> dict:
    """Every metric in `_POINT_METRICS`, each with a value and a bootstrap 95% CI."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    out: dict = {"n": int(len(y_true)), "threshold": float(threshold)}
    for name, fn in _POINT_METRICS.items():
        value = float(fn(y_true, y_score, threshold))
        lo, hi = bootstrap_ci(y_true, y_score, fn, threshold, n_resamples, seed)
        out[name] = {"value": value, "ci95": [lo, hi]}
    out["confusion_matrix"] = confusion_counts(y_true, y_pred)
    return out


def roc_points(y_true, y_score) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def pr_points(y_true, y_score) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def threshold_sweep(y_true, y_score, thresholds: np.ndarray | None = None) -> dict:
    """F1 and recall at every candidate threshold, for the sweep figure and for
    picking the F1-optimal and recall>=0.90 operating points on validation.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    f1s, recalls, precisions = [], [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
    return {
        "thresholds": thresholds.tolist(),
        "f1": f1s,
        "recall": recalls,
        "precision": precisions,
    }


def select_f1_optimal(sweep: dict) -> float:
    """Highest-F1 threshold; ties broken by the higher threshold (more conservative)."""
    f1s = np.asarray(sweep["f1"])
    thresholds = np.asarray(sweep["thresholds"])
    best = np.flatnonzero(f1s == f1s.max())
    return float(thresholds[best[-1]])


def select_recall_floor(sweep: dict, floor: float = 0.90) -> float:
    """Highest threshold whose recall is still >= `floor` (maximises precision
    subject to the safety-system recall floor). Falls back to the lowest
    threshold swept if no threshold reaches the floor.
    """
    recalls = np.asarray(sweep["recall"])
    thresholds = np.asarray(sweep["thresholds"])
    ok = np.flatnonzero(recalls >= floor)
    if len(ok) == 0:
        return float(thresholds[0])
    return float(thresholds[ok].max())
