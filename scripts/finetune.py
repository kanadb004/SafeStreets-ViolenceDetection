#!/usr/bin/env python
"""Sprint 4 AIRTLab fine-tune of the Sprint 1 lstm_head baseline.

    python scripts/finetune.py

Runs 5-fold grouped CV on AIRTLab, fits the checkpoint that would ship on the
full AIRTLab pool, and measures catastrophic forgetting against RWF-2000 val.
See docs/SPRINT_PLAN.md Sprint 4 and docs/decisions/ADR-005-finetune-and-attribution.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safestreets.config import load_config
from safestreets.training.finetune import (
    fit_final_airtlab_model,
    forgetting_check,
    load_airtlab_pool,
    run_grouped_cv,
    save_json,
    summarize_cv,
)
from safestreets.training.train import load_train_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIGS_DIR / "train.yaml"))
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    data_cfg = load_config()
    baseline_checkpoint = Path(train_cfg.checkpoints_dir) / "lstm_head_baseline.keras"
    manifest_path = Path(data_cfg.manifests_dir) / "clips.parquet"

    x, y, groups, _ = load_airtlab_pool(train_cfg.features_dir, manifest_path)
    print(f"AIRTLab pool: {len(y)} clips, {len(set(groups))} groups")

    fold_reports = run_grouped_cv(x, y, groups, baseline_checkpoint, seed=train_cfg.seed)
    cv_summary = summarize_cv(fold_reports)
    for key, stats in cv_summary.items():
        print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    final_checkpoint = fit_final_airtlab_model(
        x, y, baseline_checkpoint, train_cfg.seed, train_cfg.checkpoints_dir
    )
    print(f"final fine-tuned checkpoint: {final_checkpoint}")

    forgetting = forgetting_check(baseline_checkpoint, final_checkpoint, train_cfg)
    print(
        f"RWF-2000 val ROC-AUC: before={forgetting['roc_auc_before']:.4f} "
        f"after={forgetting['roc_auc_after']:.4f} drop={forgetting['roc_auc_drop']:+.4f}"
    )

    report = {
        "fold_reports": fold_reports,
        "cv_summary": cv_summary,
        "final_checkpoint": str(final_checkpoint),
        "forgetting_check": forgetting,
    }
    save_json(report, ARTIFACTS_DIR / "reports" / "eval_finetuned.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
