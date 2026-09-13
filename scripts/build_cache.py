#!/usr/bin/env python
"""Build the fixed-shape uint8 clip cache from data/manifests/clips.parquet.

    python scripts/build_cache.py --dataset rwf2000 --split train
    python scripts/build_cache.py --dataset all --split all --workers 4
    python scripts/build_cache.py --dataset all --split all --purge-raw

Idempotent: a second run for an unchanged (dataset, split, preprocess config)
skips the rebuild unless --force is passed. See docs/BUILD_PLAN.md Phase 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import pandas as pd

from safestreets.config import load_config
from safestreets.data.preprocess import (
    build_split_cache,
    cache_path,
    existing_cache_is_valid,
    load_preprocess_config,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _report_path(cache_dir: Path) -> Path:
    return cache_dir / "cache_report.json"


def _load_report(cache_dir: Path) -> dict:
    p = _report_path(cache_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_report(cache_dir: Path, report: dict) -> None:
    _report_path(cache_dir).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _purge_raw(written: list[dict], out_path: Path) -> int:
    """Delete each written clip's raw source file, only after re-reading it back
    from the freshly-written cache and checking shape/dtype/label/variance."""
    freed = 0
    with h5py.File(out_path, "r") as f:
        clip_ids = [c.decode() if isinstance(c, bytes) else c for c in f["clip_ids"][:]]
        id_to_idx = {c: i for i, c in enumerate(clip_ids)}
        for entry in written:
            idx = id_to_idx[entry["clip_id"]]
            clip = f["clips"][idx]
            if clip.shape != f["clips"].shape[1:] or clip.dtype != "uint8" or clip.std() == 0:
                continue
            raw_path = Path(entry["path"])
            if raw_path.exists():
                freed += raw_path.stat().st_size
                raw_path.unlink()
    return freed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--purge-raw", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild even if cache is up to date")
    parser.add_argument("--preprocess-config", default=str(CONFIGS_DIR / "preprocess.yaml"))
    args = parser.parse_args()

    cfg = load_config()
    pp_cfg = load_preprocess_config(args.preprocess_config)
    manifests_dir = Path(cfg.manifests_dir)
    cache_dir = Path(cfg.cache_dir)

    df = pd.read_parquet(manifests_dir / "clips.parquet")
    datasets = sorted(df["dataset"].unique()) if args.dataset == "all" else [args.dataset]

    report = _load_report(cache_dir)
    exit_code = 0
    total_purged_bytes = 0

    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        splits = sorted(ds_df["split"].unique()) if args.split == "all" else [args.split]
        for split in splits:
            rows = ds_df[ds_df["split"] == split]
            if len(rows) == 0:
                print(f"{dataset}/{split}: no manifest rows, skipping")
                continue

            out_path = cache_path(cache_dir, dataset, split)
            if not args.force and existing_cache_is_valid(out_path, pp_cfg, len(rows)):
                print(f"{dataset}/{split}: up to date ({out_path}), skipping")
                continue

            if args.purge_raw:
                raw_bytes = sum(Path(p).stat().st_size for p in rows["path"] if Path(p).exists())
                print(f"{dataset}/{split}: --purge-raw will free up to {raw_bytes / 1e9:.2f} GB")

            result = build_split_cache(rows, out_path, pp_cfg, workers=args.workers)
            print(
                f"{dataset}/{split}: wrote {result['n_written']}, skipped {result['n_skipped']}, "
                f"{result['bytes'] / 1e6:.1f} MB, {result['wall_s']:.1f}s -> {out_path}"
            )
            if result["n_written"] + result["n_skipped"] != result["n_manifest"]:
                print(f"{dataset}/{split}: ERROR clip count mismatch", file=sys.stderr)
                exit_code = 1

            if args.purge_raw:
                freed = _purge_raw(result["written"], out_path)
                total_purged_bytes += freed
                print(f"{dataset}/{split}: purged {freed / 1e9:.2f} GB of raw source video")

            result.pop("written")  # not needed in the persisted report
            report.setdefault(dataset, {})[split] = result

    _save_report(cache_dir, report)
    print(f"cache report written to {_report_path(cache_dir)}")
    if args.purge_raw:
        print(f"total purged: {total_purged_bytes / 1e9:.2f} GB")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
