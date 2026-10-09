#!/usr/bin/env python
"""Sprint 3 Optuna HPO over the lstm_head.

    python scripts/tune.py --trials 30 --timeout 2400
    python scripts/tune.py --trials 30 --timeout 2400 --resume

Runs (or resumes) the TPE study, writes `configs/model.best.yaml` and the two
Optuna figures, retrains the winner on train+val, scores it on test once
through Sprint 2's harness, and regenerates `docs/RESULTS.md` with the tuned
row beside the baseline. See docs/SPRINT_PLAN.md Sprint 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from safestreets.evaluation.report import generate as generate_results_md
from safestreets.training.train import load_train_config
from safestreets.training.tune import (
    best_winner_config,
    evaluate_tuned_winner,
    render_optuna_figures,
    retrain_winner_on_train_plus_val,
    run_study,
    write_best_model_yaml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIGS_DIR / "train.yaml"))
    parser.add_argument("--model-config", default=str(CONFIGS_DIR / "model.yaml"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=2400, help="seconds, 0 disables")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    storage_path = ARTIFACTS_DIR / "optuna" / "study.db"

    study = run_study(
        train_cfg,
        storage_path,
        n_trials=args.trials,
        timeout=args.timeout or None,
        seed=train_cfg.seed,
        resume=args.resume,
    )
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    print(
        f"study has {len(study.trials)} trials "
        f"({n_complete} complete, {n_pruned} pruned); best_val_auc={study.best_value:.4f}"
    )

    render_optuna_figures(study, ARTIFACTS_DIR / "figures")

    winner = best_winner_config(study)
    best_cfg = write_best_model_yaml(winner, args.model_config, CONFIGS_DIR / "model.best.yaml")
    print(f"wrote configs/model.best.yaml: {best_cfg['lstm_head']}")

    retrain_result = retrain_winner_on_train_plus_val(
        winner, train_cfg, train_cfg.checkpoints_dir
    )
    print(f"retrained winner: {retrain_result['checkpoint']}")

    with open(CONFIGS_DIR / "infer.yaml") as f:
        infer_cfg = yaml.safe_load(f)
    baseline_threshold = infer_cfg["threshold"]

    tuned = evaluate_tuned_winner(
        retrain_result["checkpoint"],
        train_cfg,
        baseline_threshold,
        ARTIFACTS_DIR / "reports",
        tag="tuned",
    )
    tuned_auc = tuned["eval_test"]["roc_auc"]["value"]

    with open(ARTIFACTS_DIR / "reports" / "eval_test.json") as f:
        import json

        baseline_auc = json.load(f)["roc_auc"]["value"]

    if tuned_auc > baseline_auc:
        print(f"tuned beats baseline: {tuned_auc:.4f} > {baseline_auc:.4f}; tuned ships")
    else:
        print(
            f"tuned does NOT beat baseline: {tuned_auc:.4f} <= {baseline_auc:.4f}; "
            "baseline ships, reporting both numbers as-is"
        )

    generate_results_md(
        reports_dir=ARTIFACTS_DIR / "reports",
        infer_yaml_path=CONFIGS_DIR / "infer.yaml",
        out_path=REPO_ROOT / "docs" / "RESULTS.md",
        train_datasets=list(train_cfg.train_datasets),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
