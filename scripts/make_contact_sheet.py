#!/usr/bin/env python
"""Render a contact sheet from the cache for a one-time visual sanity check.

Picks 4 clips (spread across whatever cache files exist) and tiles all `T`
frames of each into one PNG, so a human can eyeball orientation and colour
correctness. See docs/BUILD_PLAN.md Phase 2 DoD.

    python scripts/make_contact_sheet.py --out artifacts/figures/sample_clips.png
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

from safestreets.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/figures/sample_clips.png")
    parser.add_argument("--n-clips", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1265)
    args = parser.parse_args()

    cfg = load_config()
    cache_dir = Path(cfg.cache_dir)
    h5_files = sorted(cache_dir.glob("*.h5"))
    if not h5_files:
        print("no cache files found under", cache_dir, file=sys.stderr)
        return 1

    import numpy as np

    rng = np.random.default_rng(args.seed)
    picks = []  # (h5_path, index)
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            n = f["clips"].shape[0]
            if n > 0:
                picks.append((h5_path, int(rng.integers(0, n))))
        if len(picks) >= args.n_clips:
            break

    n_clips = len(picks)
    with h5py.File(picks[0][0], "r") as f:
        t = f["clips"].shape[1]

    fig, axes = plt.subplots(n_clips, t, figsize=(t * 1.2, n_clips * 1.2))
    if n_clips == 1:
        axes = axes[None, :]
    for row, (h5_path, idx) in enumerate(picks):
        with h5py.File(h5_path, "r") as f:
            clip = f["clips"][idx]
            label = int(f["labels"][idx])
        for col in range(t):
            ax = axes[row, col]
            ax.imshow(clip[col])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{h5_path.stem}\nlabel={label}", fontsize=6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
