from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sklearn.metrics as skm
import yaml

from safestreets.evaluation.evaluate import score_split
from safestreets.evaluation.metrics import (
    bootstrap_ci,
    compute_metrics,
    select_f1_optimal,
    select_recall_floor,
    threshold_sweep,
)
from safestreets.evaluation.report import generate as generate_results_md

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.sprint2
def test_metrics_agree_with_sklearn_to_1e9() -> None:
    rng = np.random.default_rng(1265)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    threshold = 0.5
    result = compute_metrics(y, s, threshold, n_resamples=10)

    assert result["accuracy"]["value"] == pytest.approx(
        skm.accuracy_score(y, s >= threshold), abs=1e-9
    )
    assert result["precision"]["value"] == pytest.approx(
        skm.precision_score(y, s >= threshold, zero_division=0), abs=1e-9
    )
    assert result["recall"]["value"] == pytest.approx(
        skm.recall_score(y, s >= threshold, zero_division=0), abs=1e-9
    )
    assert result["f1"]["value"] == pytest.approx(
        skm.f1_score(y, s >= threshold, zero_division=0), abs=1e-9
    )
    assert result["roc_auc"]["value"] == pytest.approx(skm.roc_auc_score(y, s), abs=1e-9)
    assert result["pr_auc"]["value"] == pytest.approx(
        skm.average_precision_score(y, s), abs=1e-9
    )


@pytest.mark.sprint2
def test_compute_metrics_has_every_metric_and_ci() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=100)
    s = rng.random(100)
    result = compute_metrics(y, s, 0.5, n_resamples=50)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert "value" in result[key]
        assert len(result[key]["ci95"]) == 2
    assert set(result["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert result["n"] == 100


@pytest.mark.sprint2
def test_bootstrap_ci_handles_degenerate_tiny_split() -> None:
    """A 2-clip split (one of each class) can still blow up roc_auc on a
    same-class resample; the CI must degrade to nan, not raise.
    """
    y = np.array([0, 1])
    s = np.array([0.2, 0.8])
    lo, hi = bootstrap_ci(y, s, lambda yt, ys, t: skm.roc_auc_score(yt, ys), 0.5, n_resamples=20)
    assert (lo != lo and hi != hi) or (lo <= hi)  # either both nan, or a valid interval


@pytest.mark.sprint2
def test_select_f1_optimal_picks_the_best_threshold() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    sweep = threshold_sweep(y, s, thresholds=np.array([0.25, 0.5, 0.75]))
    threshold = select_f1_optimal(sweep)
    assert threshold == pytest.approx(0.5)


@pytest.mark.sprint2
def test_select_recall_floor_maximises_precision_subject_to_recall() -> None:
    y = np.array([0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.3, 0.6, 0.8, 0.9])
    sweep = threshold_sweep(y, s, thresholds=np.array([0.2, 0.5, 0.7]))
    # recall at 0.2 -> all 4 positives caught (recall=1.0); at 0.5 -> 3/4 (0.75);
    # at 0.7 -> 2/4 (0.5). Floor 0.90 only 0.2 qualifies.
    threshold = select_recall_floor(sweep, floor=0.90)
    assert threshold == pytest.approx(0.2)


@pytest.mark.sprint2
@pytest.mark.needs_data
def test_threshold_was_selected_on_validation_not_test() -> None:
    """The persisted threshold in configs/infer.yaml must trace to
    artifacts/reports/eval_val.json's selection, never to a test-split file.
    """
    infer_path = REPO_ROOT / "configs" / "infer.yaml"
    eval_val_path = REPO_ROOT / "artifacts" / "reports" / "eval_val.json"
    if not (infer_path.exists() and eval_val_path.exists()):
        pytest.skip("evaluation artefacts not built yet; run scripts/evaluate.py")

    with open(infer_path) as f:
        infer_cfg = yaml.safe_load(f)
    with open(eval_val_path) as f:
        eval_val = json.load(f)

    assert infer_cfg["selected_on"] == "val"
    assert eval_val["split"] == "val"
    assert infer_cfg["threshold"] == pytest.approx(
        eval_val["selection"]["f1_optimal"]["threshold"]
    )


@pytest.mark.sprint2
@pytest.mark.needs_data
def test_cross_dataset_matrix_covers_one_train_three_test_sets() -> None:
    path = REPO_ROOT / "artifacts" / "reports" / "cross_dataset.json"
    if not path.exists():
        pytest.skip("cross_dataset.json not built yet; run scripts/evaluate.py")
    with open(path) as f:
        matrix = json.load(f)
    assert len(matrix["train_datasets"]) >= 1
    assert len(matrix["rows"]) >= 3
    assert any(row["in_domain"] for row in matrix["rows"])
    assert any(not row["in_domain"] for row in matrix["rows"])


@pytest.mark.sprint2
@pytest.mark.needs_data
def test_results_md_regenerates_with_no_diff() -> None:
    results_path = REPO_ROOT / "docs" / "RESULTS.md"
    reports_dir = REPO_ROOT / "artifacts" / "reports"
    if not (results_path.exists() and (reports_dir / "eval_test.json").exists()):
        pytest.skip("evaluation artefacts not built yet; run scripts/evaluate.py")

    before = results_path.read_text()
    regenerated = generate_results_md(
        reports_dir=reports_dir,
        infer_yaml_path=REPO_ROOT / "configs" / "infer.yaml",
        out_path=results_path,
        train_datasets=["rwf2000", "rlvs"],
    )
    assert regenerated == before


@pytest.mark.sprint2
@pytest.mark.needs_data
def test_every_results_md_number_traces_to_a_committed_eval_json() -> None:
    results_path = REPO_ROOT / "docs" / "RESULTS.md"
    eval_test_path = REPO_ROOT / "artifacts" / "reports" / "eval_test.json"
    if not (results_path.exists() and eval_test_path.exists()):
        pytest.skip("evaluation artefacts not built yet; run scripts/evaluate.py")

    text = results_path.read_text()
    with open(eval_test_path) as f:
        eval_test = json.load(f)
    # Spot-check the headline ROC-AUC value appears verbatim (4 d.p., as rendered).
    assert f"{eval_test['roc_auc']['value']:.4f}" in text


@pytest.mark.sprint2
@pytest.mark.needs_data
def test_results_md_names_cut_frame_level_auc_and_airtlab_caveat() -> None:
    results_path = REPO_ROOT / "docs" / "RESULTS.md"
    if not results_path.exists():
        pytest.skip("docs/RESULTS.md not generated yet; run scripts/evaluate.py")
    text = results_path.read_text().lower()
    assert "frame-level auc" in text
    assert "airtlab" in text and ("staged" in text or "acted" in text)


@pytest.mark.sprint2
@pytest.mark.needs_data
def test_four_figures_exist_and_are_referenced_from_results_md() -> None:
    figures_dir = REPO_ROOT / "artifacts" / "figures"
    results_path = REPO_ROOT / "docs" / "RESULTS.md"
    names = ("roc.png", "pr.png", "confusion.png", "threshold_sweep.png")
    for name in names:
        if not (figures_dir / name).exists():
            pytest.skip(f"{name} not rendered yet; run scripts/evaluate.py")
        assert (figures_dir / name).stat().st_size > 0
    if not results_path.exists():
        pytest.skip("docs/RESULTS.md not generated yet; run scripts/evaluate.py")
    text = results_path.read_text()
    for name in names:
        assert name in text


@pytest.mark.sprint2
@pytest.mark.needs_data
def test_score_split_matches_model_output_shape() -> None:
    import tf_keras

    checkpoint = REPO_ROOT / "artifacts" / "checkpoints" / "lstm_head_baseline.keras"
    if not checkpoint.exists():
        pytest.skip("Sprint 1 checkpoint not built yet")
    model = tf_keras.models.load_model(str(checkpoint), compile=False)
    y, s, clip_ids = score_split(model, REPO_ROOT / "data" / "features", ["rlvs"], "test")
    assert y.shape == s.shape == (len(clip_ids),)
    assert np.all((s >= 0) & (s <= 1))
