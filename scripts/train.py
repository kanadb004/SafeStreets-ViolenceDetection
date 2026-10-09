#!/usr/bin/env python
"""Train the Sprint 1 models.

    python scripts/train.py --config configs/train.yaml --arch lstm_head
    python scripts/train.py --config configs/train.yaml --arch scratch

See docs/SPRINT_PLAN.md Sprint 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safestreets.config import load_config
from safestreets.training.train import load_train_config, save_json, train_lstm_head, train_scratch

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIGS_DIR / "train.yaml"))
    parser.add_argument("--model-config", default=str(CONFIGS_DIR / "model.yaml"))
    parser.add_argument("--arch", choices=["lstm_head", "scratch"], default="lstm_head")
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    data_cfg = load_config()

    if args.arch == "lstm_head":
        result = train_lstm_head(args.model_config, train_cfg)
    else:
        result = train_scratch(
            args.model_config, train_cfg, CONFIGS_DIR / "data.yaml", data_cfg.cache_dir
        )

    print(f"{result['run_tag']}: best_val_auc={result['best_val_auc']:.4f}")
    save_json(result, f"artifacts/reports/train_{result['run_tag']}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
