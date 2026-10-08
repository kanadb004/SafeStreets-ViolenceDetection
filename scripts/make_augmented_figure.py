#!/usr/bin/env python
"""Render a before/after grid of the Phase 3 train augmentation, one row per clip.

Picks a few clips from the cache, shows their raw frames (top row of each pair)
against the augmented frames (bottom row), so a human can eyeball that flips and
crops stay consistent within a clip rather than flickering frame to frame.

    python scripts/make_augmented_figure.py --out artifacts/figures/augmented_clips.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safestreets.config import load_config
from safestreets.data.augment import (
    apply_clip_transform,
    build_train_transform,
    load_augment_config,
)
from safestreets.data.dataset import load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/figures/augmented_clips.png")
    parser.add_argument("--n-clips", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1265)
    args = parser.parse_args()

    cfg = load_config()
    pipeline_cfg = load_pipeline_config("configs/data.yaml")
    augment_cfg = load_augment_config("configs/data.yaml")

    cache_dir = Path(cfg.cache_dir)
    h5_files = sorted(cache_dir.glob("*_train.h5"))
    if not h5_files:
        print("no train cache files found under", cache_dir, file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    picks = []
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            n = f["clips"].shape[0]
            if n > 0:
                picks.append((h5_path, int(rng.integers(0, n))))
        if len(picks) >= args.n_clips:
            break

    transform = build_train_transform(
        augment_cfg, pipeline_cfg.height, pipeline_cfg.width, pipeline_cfg.n_frames
    )
    t = pipeline_cfg.n_frames

    n_clips = len(picks)
    fig, axes = plt.subplots(n_clips * 2, t, figsize=(t * 1.2, n_clips * 2.4))
    for row, (h5_path, idx) in enumerate(picks):
        with h5py.File(h5_path, "r") as f:
            clip = f["clips"][idx]
            label = int(f["labels"][idx])
        augmented = apply_clip_transform(transform, clip)

        for col in range(t):
            ax_raw = axes[row * 2, col]
            ax_aug = axes[row * 2 + 1, col]
            ax_raw.imshow(clip[col])
            ax_aug.imshow(augmented[col])
            for ax in (ax_raw, ax_aug):
                ax.set_xticks([])
                ax.set_yticks([])
            if col == 0:
                ax_raw.set_ylabel(f"{h5_path.stem}\nlabel={label}\nraw", fontsize=6)
                ax_aug.set_ylabel("augmented", fontsize=6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
