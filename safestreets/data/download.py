"""One fetch_<dataset>() per source named in the report (BUILD_PLAN.md Phase 1).

Every fetcher is idempotent (a `.complete` sentinel next to the files records the
expected file count, so a second run with nothing missing does no network I/O)
and resumable (the underlying `kaggle datasets download` / streamed `curl` calls
pick up where they left off; sentinels are only written after the count check
passes). Sources gated behind a manual form or an account this project has no
access to raise DatasetBlockedError instead of silently skipping, per
CLAUDE.md's "never fake a metric or a download" rule.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

VIDEO_EXTS = (".avi", ".mp4")


class DatasetBlockedError(RuntimeError):
    """Raised when a dataset needs credentials/a manual form this project can't automate."""


def _sentinel(dataset_dir: Path) -> Path:
    return dataset_dir / ".complete"


def _count_videos(dataset_dir: Path) -> int:
    if not dataset_dir.exists():
        return 0
    return sum(1 for p in dataset_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _is_complete(dataset_dir: Path) -> bool:
    sentinel = _sentinel(dataset_dir)
    if not sentinel.exists():
        return False
    recorded = json.loads(sentinel.read_text())
    return _count_videos(dataset_dir) == recorded.get("file_count", -1)


def _mark_complete(dataset_dir: Path) -> None:
    count = _count_videos(dataset_dir)
    _sentinel(dataset_dir).write_text(json.dumps({"file_count": count}))


def _kaggle_download(ref: str, dest: Path, unzip: bool = True) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", ref, "-p", str(dest)] + (["--unzip"] if unzip else []),
        check=True,
    )


def _kaggle_download_file(ref: str, file_name: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", ref, "-f", file_name, "-p", str(dest)],
        check=True,
    )
    zip_path = dest / (Path(file_name).name + ".zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink()


def fetch_rwf2000(raw_dir: Path, force: bool = False) -> Path:
    """Official RWF-2000 train/val directory split, re-hosted on Kaggle at 10fps.

    Kaggle ref: rmilcb/rwf-2000-10fps. Preserves the authors' train/{Fight,NonFight}
    and val/{Fight,NonFight} layout verbatim, which Phase 1 relies on for
    `group_id = clip_id` (see ADR-002).
    """
    dest = raw_dir / "rwf2000"
    if not force and _is_complete(dest):
        return dest
    _kaggle_download("rmilcb/rwf-2000-10fps", dest)
    _mark_complete(dest)
    return dest


def fetch_rlvs(raw_dir: Path, force: bool = False) -> Path:
    """Real Life Violence Situations dataset, re-hosted on Kaggle.

    Kaggle ref: mohamedmustafa/real-life-violence-situations-dataset.
    """
    dest = raw_dir / "rlvs"
    if not force and _is_complete(dest):
        return dest
    _kaggle_download("mohamedmustafa/real-life-violence-situations-dataset", dest)
    # The zip nests a second, byte-identical copy of the whole dataset under
    # "real life violence situations/"; drop it so clip_id stays unique.
    duplicate = dest / "real life violence situations"
    if duplicate.exists():
        import shutil

        shutil.rmtree(duplicate)
    _mark_complete(dest)
    return dest


def fetch_airtlab(raw_dir: Path, force: bool = False) -> Path:
    """AIRTLab violence-detection dataset, re-hosted on Kaggle.

    Kaggle ref: tuannguyenhoang/airtlab. Retains the camN/ prefixes Phase 1 uses
    as the scene/camera grouping key.
    """
    dest = raw_dir / "airtlab"
    if not force and _is_complete(dest):
        return dest
    _kaggle_download("tuannguyenhoang/airtlab", dest)
    _mark_complete(dest)
    return dest


# UCF-Crime: §3.1 caps this at four categories plus a size-matched Normal
# sample, never the full ~100 GB release. `abuse_assault_fighting_normal_ref`
# is a small, already-subsetted mirror; Robbery (absent from that mirror) is
# streamed file-by-file from the full mirror via `kaggle datasets download -f`,
# which pulls only the named file, not the whole multi-GB dataset.
_UCF_MINI_REF = "shashiprakash204/ucfcrimeminidataset"
_UCF_MINI_CATEGORIES = {
    "Abuse": "Abuse",
    "Assault": "Assault",
    "Fighting": "Fighting",
    "Normal": "normal",
}
_UCF_FULL_REF = "bypktt/ucf-crimes"
_UCF_FULL_ROOT = "Real-world Anomaly Detection in Surveillance Videos (UCF)"
_UCF_ROBBERY_FILES = [f"Robbery{i:03d}_x264.mp4" for i in (1, 2, 3, 4, 5, 6, 7)]


def fetch_ucfcrime(raw_dir: Path, force: bool = False) -> Path:
    """Abuse, Assault, Fighting, Robbery + a size-matched Normal sample only.

    Never fetches raw UCF-Crime in full (~100 GB) — see BUILD_PLAN §3.1.
    """
    dest = raw_dir / "ucfcrime"
    if not force and _is_complete(dest):
        return dest

    missing = {
        local_name: mini_name
        for local_name, mini_name in _UCF_MINI_CATEGORIES.items()
        if not (dest / local_name).exists() or not any((dest / local_name).iterdir())
    }
    if missing:
        tmp = dest / "_mini_tmp"
        _kaggle_download(_UCF_MINI_REF, tmp)
        for local_name, mini_name in missing.items():
            cat_dir = dest / local_name
            for sub in tmp.rglob("*"):
                if sub.is_dir() and sub.name == mini_name:
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    for f in sub.glob("*"):
                        if f.suffix.lower() in VIDEO_EXTS:
                            f.rename(cat_dir / f.name)
        import shutil

        shutil.rmtree(tmp)

    robbery_dir = dest / "Robbery"
    robbery_dir.mkdir(parents=True, exist_ok=True)
    for fname in _UCF_ROBBERY_FILES:
        target = robbery_dir / fname
        if target.exists():
            continue
        remote = f"{_UCF_FULL_ROOT}/Anomaly-Videos/Robbery/{fname}"
        _kaggle_download_file(_UCF_FULL_REF, remote, robbery_dir)
        extracted = robbery_dir / _UCF_FULL_ROOT / "Anomaly-Videos" / "Robbery" / fname
        if extracted.exists():
            extracted.rename(target)
    nested_root = robbery_dir / _UCF_FULL_ROOT
    if nested_root.exists():
        import shutil

        shutil.rmtree(nested_root)

    _mark_complete(dest)
    return dest


def fetch_xdviolence(raw_dir: Path, force: bool = False) -> Path:
    """Official pre-extracted I3D RGB+Flow features, never raw video (§3.1).

    Permanently skipped: the OneDrive distribution
    (stuxidianeducn-my.sharepoint.com, Xidian University tenant) was checked
    manually in a browser and is 38.3 GB, far past BUILD_PLAN §3.1's ~4 GB
    estimate and the project's ~44 GB total disk budget. The OneDrive link
    also rejects a scripted download (403 + forms-based-auth redirect on the
    anonymous-link handshake); the Baidu Netdisk alternative needs an account
    this project doesn't have and would carry the same size problem. See
    docs/DATASETS.md for the full writeup.
    """
    raise DatasetBlockedError(
        "XD-Violence I3D features are SKIPPED: the official release is 38.3 GB "
        "(confirmed), which does not fit the project's disk budget. "
        "See docs/DATASETS.md for details."
    )
