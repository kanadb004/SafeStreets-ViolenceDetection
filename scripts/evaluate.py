#!/usr/bin/env python
"""Sprint 2 evaluation harness.

    python scripts/evaluate.py --checkpoint artifacts/checkpoints/lstm_head_baseline.keras

Selects an operating threshold on validation, scores the combined test split
and the per-dataset cross-dataset matrix, renders the figure set, and
regenerates `docs/RESULTS.md`. See docs/SPRINT_PLAN.md Sprint 2.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safestreets.evaluation.cross_dataset import TEST_DATASETS, build_matrix
from safestreets.evaluation.evaluate import (
    load_checkpoint,
    render_confusion,
    render_pr,
    render_roc,
    render_threshold_sweep,
    save_json,
    score_split,
    select_threshold,
    write_infer_config,
)
from safestreets.evaluation.metrics import compute_metrics
from safestreets.evaluation.report import generate as generate_results_md
from safestreets.training.train import load_train_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
FIGURES_DIR = REPO_ROOT / "artifacts" / "figures"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    default_checkpoint = REPO_ROOT / "artifacts" / "checkpoints" / "lstm_head_baseline.keras"
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--train-config", default=str(CONFIGS_DIR / "train.yaml"))
    parser.add_argument("--features-dir", default=str(REPO_ROOT / "data" / "features"))
    args = parser.parse_args()

    train_cfg = load_train_config(args.train_config)
    model = load_checkpoint(args.checkpoint)

    # 1. Threshold selection on validation only.
    y_val, s_val, _ = score_split(model, args.features_dir, train_cfg.val_datasets, "val")
    selection = select_threshold(y_val, s_val)
    f1_threshold = selection["f1_optimal"]["threshold"]
    write_infer_config(selection, CONFIGS_DIR / "infer.yaml", _git_sha())
    save_json(
        {
            "split": "val",
            "datasets": train_cfg.val_datasets,
            "n": int(len(y_val)),
            "selection": selection,
        },
        REPORTS_DIR / "eval_val.json",
    )
    render_threshold_sweep(selection["sweep"], FIGURES_DIR / "threshold_sweep.png")

    # 2. Per-dataset test scores (feeds both the combined headline split and
    #    the cross-dataset matrix, scored once).
    per_dataset: dict[str, dict] = {}
    for dataset, in_domain in TEST_DATASETS:
        y, s, _ = score_split(model, args.features_dir, [dataset], "test")
        per_dataset[dataset] = {"y": y, "s": s, "in_domain": in_domain}
        metrics = compute_metrics(y, s, f1_threshold)
        save_json(
            {"split": "test", "dataset": dataset, **metrics},
            REPORTS_DIR / f"eval_{dataset}_test.json",
        )

    # 3. Combined in-domain-task test split: RLVS + AIRTLab + UCF-Crime test.
    import numpy as np

    y_test = np.concatenate([v["y"] for v in per_dataset.values()])
    s_test = np.concatenate([v["s"] for v in per_dataset.values()])
    eval_test = compute_metrics(y_test, s_test, f1_threshold)
    save_json(
        {
            "split": "test",
            "datasets": [d for d, _ in TEST_DATASETS],
            **eval_test,
        },
        REPORTS_DIR / "eval_test.json",
    )
    render_roc(y_test, s_test, FIGURES_DIR / "roc.png")
    render_pr(y_test, s_test, FIGURES_DIR / "pr.png")
    render_confusion(y_test, s_test, f1_threshold, FIGURES_DIR / "confusion.png")

    # 4. Cross-dataset matrix, reusing the per-dataset metrics_fn so it shares
    #    the same threshold/resample settings as the combined split.
    cross_dataset = build_matrix(
        model, args.features_dir, lambda y, s: compute_metrics(y, s, f1_threshold)
    )
    save_json(cross_dataset, REPORTS_DIR / "cross_dataset.json")

    # 5. Regenerate docs/RESULTS.md from what was just written.
    generate_results_md(
        reports_dir=REPORTS_DIR,
        infer_yaml_path=CONFIGS_DIR / "infer.yaml",
        out_path=REPO_ROOT / "docs" / "RESULTS.md",
        train_datasets=list(train_cfg.train_datasets),
    )

    print(f"val n={len(y_val)} f1_optimal_threshold={f1_threshold:.4f}")
    print(f"test (combined) n={len(y_test)} roc_auc={eval_test['roc_auc']['value']:.4f}")
    for row in cross_dataset["rows"]:
        tag = f"{row['test_dataset']:10s} in_domain={row['in_domain']!s:5s}"
        print(f"  {tag} roc_auc={row['roc_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
