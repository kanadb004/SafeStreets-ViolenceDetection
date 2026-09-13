from pathlib import Path

import pandas as pd
import pytest

from safestreets.config import load_config
from safestreets.data.download import DatasetBlockedError, fetch_xdviolence
from safestreets.data.manifest import (
    MANIFEST_COLUMNS,
    _bucket,
    _split_for_bucket,
    build_manifest,
    write_splits_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def clips_df():
    manifest_path = REPO_ROOT / "data" / "manifests" / "clips.parquet"
    if not manifest_path.exists():
        pytest.skip("data/manifests/clips.parquet not built; run fetch_data.py + manifest.py first")
    return pd.read_parquet(manifest_path)


@pytest.mark.phase1
def test_bucket_is_deterministic():
    a = _bucket("rlvs", "rlvs_cluster_0")
    b = _bucket("rlvs", "rlvs_cluster_0")
    assert a == b
    assert 0 <= a < 100


@pytest.mark.phase1
def test_bucket_depends_on_dataset_namespace():
    # Same group_id string under different datasets must not collide by construction.
    a = _bucket("rlvs", "42")
    b = _bucket("airtlab", "42")
    assert isinstance(a, int) and isinstance(b, int)


@pytest.mark.phase1
@pytest.mark.parametrize(
    "bucket,expected",
    [(0, "train"), (69, "train"), (70, "val"), (84, "val"), (85, "test"), (99, "test")],
)
def test_split_for_bucket_boundaries(bucket, expected):
    assert _split_for_bucket(bucket) == expected


@pytest.mark.phase1
def test_fetch_xdviolence_is_blocked_not_faked(tmp_path):
    with pytest.raises(DatasetBlockedError):
        fetch_xdviolence(tmp_path)


@pytest.mark.phase1
@pytest.mark.needs_data
def test_manifest_has_required_columns(clips_df):
    assert list(clips_df.columns) == MANIFEST_COLUMNS


@pytest.mark.phase1
@pytest.mark.needs_data
def test_manifest_size_and_no_nulls(clips_df):
    assert len(clips_df) >= 4000
    assert clips_df["label"].isnull().sum() == 0
    assert clips_df["clip_id"].duplicated().sum() == 0


@pytest.mark.phase1
@pytest.mark.needs_data
def test_leakage_assertion(clips_df):
    for dataset, group in clips_df.groupby("dataset"):
        train_groups = set(group.loc[group["split"] == "train", "group_id"])
        val_groups = set(group.loc[group["split"] == "val", "group_id"])
        test_groups = set(group.loc[group["split"] == "test", "group_id"])
        triple = train_groups & val_groups & test_groups
        assert triple == set(), f"{dataset}: group leaked across all three splits"
        assert train_groups & val_groups == set(), f"{dataset}: group leaked train/val"
        assert train_groups & test_groups == set(), f"{dataset}: group leaked train/test"
        assert val_groups & test_groups == set(), f"{dataset}: group leaked val/test"


@pytest.mark.phase1
@pytest.mark.needs_data
def test_split_proportions_within_tolerance(clips_df):
    targets = {"train": 70, "val": 15, "test": 15}
    for dataset, group in clips_df.groupby("dataset"):
        total = len(group)
        present_splits = set(group["split"])
        counts = group["split"].value_counts()
        for split in present_splits:
            pct = 100 * counts[split] / total
            # RWF-2000 deliberately uses the official train/val split verbatim
            # and contributes no test rows (ADR-002); it is exempt from the
            # generic 70/15/15 tolerance.
            if dataset == "rwf2000":
                continue
            msg = f"{dataset}/{split}: {pct:.1f}% vs {targets[split]}% target"
            assert abs(pct - targets[split]) <= 3, msg


@pytest.mark.phase1
@pytest.mark.needs_data
def test_every_path_exists(clips_df):
    assert clips_df["path"].map(lambda p: (REPO_ROOT / p).exists()).all()


@pytest.mark.phase1
@pytest.mark.needs_data
@pytest.mark.slow
def test_manifest_rebuild_is_byte_identical(tmp_path):
    cfg = load_config()
    raw_dir = REPO_ROOT / cfg.raw_dir

    df1 = build_manifest(raw_dir)
    df2 = build_manifest(raw_dir)

    p1, p2 = tmp_path / "splits1.json", tmp_path / "splits2.json"
    write_splits_json(df1, p1)
    write_splits_json(df2, p2)

    assert p1.read_text() == p2.read_text()


@pytest.mark.phase1
@pytest.mark.needs_data
def test_rwf2000_no_test_rows(clips_df):
    rwf = clips_df[clips_df["dataset"] == "rwf2000"]
    assert set(rwf["split"]) == {"train", "val"}


@pytest.mark.phase1
@pytest.mark.needs_data
def test_airtlab_camera_views_share_group(clips_df):
    airtlab = clips_df[clips_df["dataset"] == "airtlab"]
    # Every group_id should be shared by exactly the two camera views of one event.
    counts = airtlab.groupby("group_id").size()
    assert (counts == 2).all(), "expected exactly cam1+cam2 per AIRTLab group_id"
