"""Cross-dataset evaluation: the lstm_head is trained on RWF-2000 + RLVS
(`configs/train.yaml`). Report the AUC on the held-out slice of that same
training domain (RLVS test) beside AUC on the two datasets the model never
saw during training (AIRTLab, UCF-Crime), to surface the generalisation drop.

**No drop at all means suspecting leakage, not celebrating** (docs/SPRINT_PLAN.md
Sprint 2). RWF-2000 contributes no test rows at all (ADR-002), so it cannot
appear as a test dataset here.
"""

from __future__ import annotations

from safestreets.evaluation.evaluate import score_split

TRAIN_DATASETS = ("rwf2000", "rlvs")
# in_domain=True: part of the training distribution (RLVS), even though the
# test *rows* were held out of training. in_domain=False: zero-shot transfer.
TEST_DATASETS = (
    ("rlvs", True),
    ("airtlab", False),
    ("ucfcrime", False),
)


def build_matrix(model, features_dir, metrics_fn) -> dict:
    """`metrics_fn(y_true, y_score) -> dict` is `compute_metrics` partially
    applied by the caller so this module stays free of a direct metrics
    dependency choice (threshold, n_resamples) and just reports the matrix.
    """
    rows = []
    in_domain_auc = None
    for dataset, in_domain in TEST_DATASETS:
        y, s, _ = score_split(model, features_dir, [dataset], "test")
        metrics = metrics_fn(y, s)
        auc = metrics["roc_auc"]["value"]
        if in_domain:
            in_domain_auc = auc
        rows.append(
            {
                "test_dataset": dataset,
                "split": "test",
                "in_domain": in_domain,
                "n": metrics["n"],
                "roc_auc": auc,
                "roc_auc_ci95": metrics["roc_auc"]["ci95"],
                "auc_drop_vs_in_domain": None,
            }
        )
    for row in rows:
        if not row["in_domain"] and in_domain_auc is not None:
            row["auc_drop_vs_in_domain"] = round(in_domain_auc - row["roc_auc"], 6)
    return {"train_datasets": list(TRAIN_DATASETS), "rows": rows}
