"""Build data/manifests/clips.parquet: one row per clip, group-aware split assigned.

See docs/decisions/ADR-002-dataset-splits.md for the grouping key chosen per
dataset and why. The one invariant every dataset must satisfy: splits are
assigned by `group_id`, never by `clip_id` (CLAUDE.md "Splits are group-aware").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd

MANIFEST_COLUMNS = [
    "clip_id",
    "dataset",
    "path",
    "label",
    "split",
    "group_id",
    "n_frames",
    "fps",
    "width",
    "height",
    "duration_s",
    "source_video",
    "category",
]

# Train/val/test bucket boundaries for hash(dataset:group_id) % 100.
_TRAIN_MAX = 70
_VAL_MAX = 85

# Fixed salt mixed into the split hash. AIRTLab (175 groups) and UCF-Crime (35
# groups) have too few groups for hash(group_id) % 100 to land near 70/15/15
# by chance; salts 0-199 were swept once (offline) against the Phase 1 DoD's
# +/-3pp tolerance and 18 was the best fit (max deviation 2.1pp, vs >3.5pp at
# salt 0). Fixed here, not re-swept per run: still fully deterministic.
_SPLIT_SALT = 18


def _probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    duration_s = n_frames / fps if fps > 0 else 0.0
    return {"n_frames": n_frames, "fps": fps, "width": width, "height": height, "duration_s": duration_s}


def _average_hash(path: Path) -> str | None:
    """8x8 average hash of the middle frame, used as an approximate near-duplicate key."""
    cap = cv2.VideoCapture(str(path))
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(n_frames // 2, 0))
        ok, frame = cap.read()
        if not ok:
            ok, frame = cap.read()
        if not ok:
            return None
    finally:
        cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    return "".join("1" if b else "0" for b in bits)


def _bucket(dataset: str, group_id: str) -> int:
    digest = hashlib.md5(f"{_SPLIT_SALT}:{dataset}:{group_id}".encode()).hexdigest()
    return int(digest, 16) % 100


def _split_for_bucket(bucket: int) -> str:
    if bucket < _TRAIN_MAX:
        return "train"
    if bucket < _VAL_MAX:
        return "val"
    return "test"


def _scan_rwf2000(raw_dir: Path) -> list[dict]:
    root = raw_dir / "rwf2000"
    rows = []
    for official_split in ("train", "val"):
        for label_name, label in (("Fight", 1), ("NonFight", 0)):
            for f in sorted((root / official_split / label_name).glob("*.avi")):
                stem = f.stem
                # The same source video id can contribute both a Fight- and a
                # NonFight-labelled segment sharing the same index, so the
                # label must be part of clip_id to keep it unique.
                clip_id = f"rwf2000_{label_name}_{stem}"
                probe = _probe_video(f)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "dataset": "rwf2000",
                        "path": str(f),
                        "label": label,
                        "group_id": stem,
                        "source_video": stem,
                        "category": label_name,
                        "_official_split": official_split,
                        **probe,
                    }
                )
    return rows


def _scan_rlvs(raw_dir: Path) -> list[dict]:
    root = raw_dir / "rlvs" / "Real Life Violence Dataset"
    rows = []
    hash_to_cluster: dict[tuple[int, str], str] = {}
    for label_name, label in (("Violence", 1), ("NonViolence", 0)):
        for f in sorted((root / label_name).glob("*.mp4")):
            stem = f.stem
            clip_id = f"rlvs_{stem}"
            probe = _probe_video(f)
            ahash = _average_hash(f)
            duration_bucket = round(probe["duration_s"])
            key = (label, ahash if ahash else clip_id, duration_bucket)
            cluster_key = hash_to_cluster.setdefault(key, f"rlvs_cluster_{len(hash_to_cluster)}")
            rows.append(
                {
                    "clip_id": clip_id,
                    "dataset": "rlvs",
                    "path": str(f),
                    "label": label,
                    "group_id": cluster_key,
                    "source_video": cluster_key,
                    "category": label_name,
                    **probe,
                }
            )
    return rows


def _scan_airtlab(raw_dir: Path) -> list[dict]:
    root = raw_dir / "airtlab" / "violence-detection-dataset"
    rows = []
    for label_name, label in (("violent", 1), ("non-violent", 0)):
        for cam_dir in sorted((root / label_name).glob("cam*")):
            for f in sorted(cam_dir.glob("*.mp4")):
                clip_number = f.stem
                # Same clip_number under cam1/ and cam2/ is the same event shot from
                # two cameras (see readme.md); group them so both views stay together.
                group_id = f"{label_name}_{clip_number}"
                clip_id = f"airtlab_{label_name}_{cam_dir.name}_{clip_number}"
                probe = _probe_video(f)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "dataset": "airtlab",
                        "path": str(f),
                        "label": label,
                        "group_id": group_id,
                        "source_video": group_id,
                        "category": label_name,
                        **probe,
                    }
                )
    return rows


def _scan_ucfcrime(raw_dir: Path) -> list[dict]:
    root = raw_dir / "ucfcrime"
    rows = []
    for category in sorted(d.name for d in root.iterdir() if d.is_dir()):
        label = 0 if category == "Normal" else 1
        for f in sorted((root / category).glob("*.mp4")):
            stem = f.stem
            clip_id = f"ucfcrime_{stem}"
            probe = _probe_video(f)
            rows.append(
                {
                    "clip_id": clip_id,
                    "dataset": "ucfcrime",
                    "path": str(f),
                    "label": label,
                    "group_id": stem,
                    "source_video": stem,
                    "category": category,
                    **probe,
                }
            )
    return rows


def build_manifest(raw_dir: Path) -> pd.DataFrame:
    rows = []
    rows += _scan_rwf2000(raw_dir)
    rows += _scan_rlvs(raw_dir)
    rows += _scan_airtlab(raw_dir)
    rows += _scan_ucfcrime(raw_dir)

    splits = []
    for r in rows:
        official = r.pop("_official_split", None)
        if official is not None:
            # RWF-2000: honour the authors' train/val split verbatim (ADR-002).
            # It contributes no rows to `test`.
            splits.append(official)
        else:
            splits.append(_split_for_bucket(_bucket(r["dataset"], r["group_id"])))

    df = pd.DataFrame(rows)
    df["split"] = splits
    df["path"] = df["path"].apply(lambda p: Path(p).as_posix())
    return df[MANIFEST_COLUMNS]


def write_splits_json(df: pd.DataFrame, path: Path) -> None:
    counts = (
        df.groupby(["dataset", "split", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "split", "label"])
    )
    payload: dict[str, dict[str, dict[str, int]]] = {}
    for row in counts.itertuples(index=False):
        payload.setdefault(row.dataset, {}).setdefault(row.split, {})[str(row.label)] = row.count
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    from safestreets.config import load_config

    cfg = load_config()
    raw_dir = Path(cfg.raw_dir)
    manifests_dir = Path(cfg.manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    df = build_manifest(raw_dir)
    df.to_parquet(manifests_dir / "clips.parquet", index=False)
    write_splits_json(df, manifests_dir / "splits.json")
    print(f"wrote {len(df)} rows to {manifests_dir / 'clips.parquet'}")


if __name__ == "__main__":
    main()
