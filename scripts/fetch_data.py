#!/usr/bin/env python
"""CLI wrapper around safestreets.data.download's fetch_<dataset> functions.

    python scripts/fetch_data.py --dataset rwf2000
    python scripts/fetch_data.py --dataset all --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safestreets.config import load_config
from safestreets.data.download import (
    DatasetBlockedError,
    fetch_airtlab,
    fetch_rlvs,
    fetch_rwf2000,
    fetch_ucfcrime,
    fetch_xdviolence,
)

FETCHERS = {
    "rwf2000": fetch_rwf2000,
    "rlvs": fetch_rlvs,
    "airtlab": fetch_airtlab,
    "ucfcrime": fetch_ucfcrime,
    "xdviolence": fetch_xdviolence,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=[*FETCHERS, "all"])
    parser.add_argument("--force", action="store_true", help="re-download even if complete")
    parser.add_argument("--dry-run", action="store_true", help="print, do no I/O")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = Path(cfg.raw_dir)
    names = list(FETCHERS) if args.dataset == "all" else [args.dataset]

    exit_code = 0
    for name in names:
        if args.dry_run:
            print(f"[dry-run] would fetch {name} into {raw_dir / name}")
            continue
        try:
            dest = FETCHERS[name](raw_dir, force=args.force)
            print(f"{name}: OK ({dest})")
        except DatasetBlockedError as e:
            print(f"{name}: BLOCKED\n  {e}", file=sys.stderr)
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
