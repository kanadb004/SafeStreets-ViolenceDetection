#!/usr/bin/env python
"""Extract frozen MobileNetV2 features from the Phase 2 clip cache.

    python scripts/extract_features.py --dataset all --split all

Idempotent: a second run for an unchanged (dataset, split, feature config)
skips the rebuild unless --force is passed. See docs/SPRINT_PLAN.md Sprint 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safestreets.config import load_config
from safestreets.features.extract import (
    FeatureConfig,
    build_backbone,
    build_split_features,
    existing_features_valid,
    features_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--force", action="store_true", help="re-extract even if up to date")
    parser.add_argument("--features-dir", default=None)
    args = parser.parse_args()

    cfg = load_config()
    cache_dir = Path(cfg.cache_dir)
    features_dir = Path(args.features_dir) if args.features_dir else Path("data/features")

    cache_paths = sorted(cache_dir.glob("*.h5"))
    if args.dataset != "all":
        cache_paths = [p for p in cache_paths if p.name.startswith(f"{args.dataset}_")]
    if args.split != "all":
        cache_paths = [p for p in cache_paths if p.name.endswith(f"_{args.split}.h5")]

    feat_cfg = FeatureConfig()
    backbone = None  # built lazily, only if any file actually needs extraction
    exit_code = 0

    for cache_path in cache_paths:
        stem = cache_path.stem  # "<dataset>_<split>"
        dataset, split = stem.rsplit("_", 1)
        out_path = features_path(features_dir, dataset, split)

        import h5py

        with h5py.File(cache_path, "r") as f:
            n_expected = f["clips"].shape[0]

        if not args.force and existing_features_valid(out_path, feat_cfg, n_expected):
            print(f"{dataset}/{split}: up to date ({out_path}), skipping")
            continue

        if backbone is None:
            backbone = build_backbone(feat_cfg)

        result = build_split_features(cache_path, out_path, feat_cfg, backbone)
        print(
            f"{dataset}/{split}: wrote {result['n']} clips, "
            f"{result['bytes'] / 1e6:.1f} MB, {result['wall_s']:.1f}s -> {out_path}"
        )
        if result["n"] != n_expected:
            print(f"{dataset}/{split}: ERROR clip count mismatch", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
