from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from safestreets.inference.engine import InferenceEngine
from safestreets.inference.spec import (
    NORMALISATION,
    SpecMismatchError,
    active_spec,
    assert_spec_matches,
    load_threshold,
)
from safestreets.inference.stream import (
    Hysteresis,
    RingBuffer,
    StreamConfig,
    StreamScorer,
    frame_step,
    hysteresis_thresholds,
    load_stream_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
SAMPLE_VIDEO = REPO_ROOT / "test" / "sample_video.avi"


def _shipped_paths() -> tuple[Path, Path]:
    from safestreets.inference.engine import default_onnx_path

    onnx_path = default_onnx_path()
    return onnx_path, onnx_path.with_suffix(".json")


def _naive_rising_edges(scores, threshold: float) -> int:
    above = [s >= threshold for s in scores]
    return sum(1 for prev, cur in zip([False] + above[:-1], above, strict=True) if cur and not prev)


# --- stream: ring buffer, EMA, hysteresis -------------------------------------


@pytest.mark.sprint5
def test_ring_buffer_emits_full_window_then_every_stride() -> None:
    buf = RingBuffer(window=16, stride=8)
    emitted = {}
    for i in range(40):
        out = buf.push(np.full((2, 2, 3), i, dtype=np.uint8))
        if out is not None:
            emitted[i] = out
    assert sorted(emitted) == [15, 23, 31, 39]
    last = emitted[39]
    assert last.shape == (16, 2, 2, 3)
    assert [int(f[0, 0, 0]) for f in last] == list(range(24, 40))


@pytest.mark.sprint5
def test_stream_config_window_and_stride_come_from_configs() -> None:
    cfg = load_stream_config()
    assert cfg.window == active_spec()["input"]["shape"][1]
    assert cfg.stride == 8
    assert 0 < cfg.ema_alpha <= 1


@pytest.mark.sprint5
def test_hysteresis_band_derives_from_the_single_infer_threshold() -> None:
    threshold = load_threshold()
    cfg = load_stream_config()
    scorer = StreamScorer(lambda w: np.zeros(1), threshold, cfg)
    assert scorer.enter_threshold == pytest.approx(threshold + cfg.hysteresis_margin)
    assert scorer.leave_threshold == pytest.approx(max(threshold - cfg.hysteresis_margin, 0.0))
    assert hysteresis_thresholds(0.98, 0.05) == (1.0, pytest.approx(0.93))


@pytest.mark.sprint5
def test_oscillating_raw_scores_produce_one_alert_not_many() -> None:
    """The Sprint 5 DoD: a series oscillating around the threshold alerts once.
    EMA disabled (alpha=1) to isolate the hysteresis latch itself."""
    threshold = load_threshold()
    series = [threshold - 0.15, threshold + 0.10] + [
        threshold + d for d in (-0.03, 0.03, -0.04, 0.04, -0.02, 0.02, -0.03, 0.03) * 3
    ]
    assert _naive_rising_edges(series, threshold) > 5  # it really does oscillate

    scorer = StreamScorer(
        lambda w: np.zeros(1), threshold, StreamConfig(ema_alpha=1.0, hysteresis_margin=0.05)
    )
    results = [scorer.push_score(s) for s in series]
    assert scorer.alert_count == 1
    assert sum(r.alert_started for r in results) == 1
    assert results[-1].alert_active


@pytest.mark.sprint5
def test_oscillation_with_shipped_stream_config_alerts_once() -> None:
    """Same DoD scenario with the configured EMA and margin, not test-only values."""
    threshold = load_threshold()
    series = [threshold - 0.1, threshold + 0.25] * 15
    assert _naive_rising_edges(series, threshold) == 15

    scorer = StreamScorer(lambda w: np.zeros(1), threshold, load_stream_config())
    for s in series:
        scorer.push_score(s)
    assert scorer.alert_count == 1


@pytest.mark.sprint5
def test_alert_rearms_after_score_falls_below_leave_threshold() -> None:
    latch = Hysteresis(enter=0.25, leave=0.15)
    edges = [latch.update(s) for s in (0.3, 0.2, 0.1, 0.2, 0.3)]
    assert edges == [True, False, False, False, True]


@pytest.mark.sprint5
def test_stream_scorer_scores_each_emitted_window_with_source_indices() -> None:
    calls = []

    def fake_score(batch: np.ndarray) -> np.ndarray:
        calls.append(batch.shape)
        return np.array([batch.mean() / 255.0])

    scorer = StreamScorer(fake_score, 0.2, StreamConfig(window=4, stride=2))
    results = []
    for i in range(8):
        r = scorer.push(np.full((2, 2, 3), 10 * i, dtype=np.uint8), source_index=i * 3)
        if r is not None:
            results.append(r)
    assert calls == [(1, 4, 2, 2, 3)] * 3
    assert [(r.start_frame, r.end_frame) for r in results] == [(0, 9), (6, 15), (12, 21)]
    assert results[0].raw_score == pytest.approx(15 / 255.0)


@pytest.mark.sprint5
def test_frame_step_matches_target_fps() -> None:
    assert frame_step(30.0, 3.2) == 9
    assert frame_step(10.0, 3.2) == 3
    assert frame_step(2.0, 3.2) == 1
    assert frame_step(0.0, 3.2) == 1


# --- spec / sidecar contract ---------------------------------------------------


@pytest.mark.sprint5
def test_normalisation_contract_matches_tf_keras_preprocess_input() -> None:
    """Pins the in-graph Rescaling constants to MobileNetV2's real preprocess_input,
    the function the Sprint 1 feature store was built with."""
    # The exact symbol safestreets/features/extract.py applied to the feature store.
    from safestreets.features.extract import preprocess_input

    x = np.arange(256, dtype=np.float32).reshape(1, 16, 16, 1).repeat(3, axis=-1)
    expected = preprocess_input(x.copy())
    ours = x * np.float32(NORMALISATION["scale"]) + np.float32(NORMALISATION["offset"])
    assert np.max(np.abs(ours - expected)) < 1e-6


@pytest.mark.sprint5
def test_active_spec_matches_itself() -> None:
    assert_spec_matches(active_spec(), active_spec())


def _write_sidecar(tmp_path: Path, sidecar: dict) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(sidecar))
    return path


@pytest.mark.sprint5
@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["normalisation"].update(scale=1.0 / 255.0, offset=0.0),  # unit_range
        lambda s: s["input"].update(shape=[None, 8, 112, 112, 3]),
        lambda s: s["input"].update(channel_order="BGR"),
        lambda s: s["head"].update(input_dim=2048),
        lambda s: s.update(threshold=0.5),
        lambda s: s.pop("normalisation"),
    ],
    ids=["normalisation", "n_frames", "channel_order", "input_dim", "threshold", "missing"],
)
def test_engine_refuses_sidecar_that_disagrees_with_active_config(tmp_path, mutate) -> None:
    sidecar = json.loads(json.dumps(active_spec()))
    sidecar.update(model_version="x", git_sha="y")
    mutate(sidecar)
    sidecar_path = _write_sidecar(tmp_path, sidecar)
    # The ONNX path does not even exist: the refusal must happen before the
    # engine touches the model file.
    with pytest.raises(SpecMismatchError):
        InferenceEngine(tmp_path / "absent.onnx", sidecar_path)


@pytest.mark.sprint5
def test_engine_refuses_to_run_without_a_sidecar(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="sidecar"):
        InferenceEngine(tmp_path / "absent.onnx")


# --- real artefacts --------------------------------------------------------------


@pytest.mark.sprint5
@pytest.mark.needs_data
def test_sidecar_is_complete_and_matches_the_onnx_file() -> None:
    onnx_path, sidecar_path = _shipped_paths()
    if not sidecar_path.exists():
        pytest.skip("ONNX artefact not built; run scripts/export_onnx.py")
    from safestreets.inference.export_onnx import file_sha256

    sidecar = json.loads(sidecar_path.read_text())
    for key in (
        "model_version",
        "git_sha",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "onnx_sha256",
        "input",
        "normalisation",
        "head",
        "threshold",
        "opset",
    ):
        assert key in sidecar, key
    assert sidecar["onnx_sha256"] == file_sha256(onnx_path)
    assert (REPO_ROOT / sidecar["source_checkpoint"]).exists()
    assert sidecar["source_checkpoint"].endswith("finetuned_airtlab_lstm_head.keras")  # ADR-005
    assert sidecar["threshold"] == pytest.approx(load_threshold())
    assert_spec_matches(sidecar, active_spec())


@pytest.mark.sprint5
@pytest.mark.needs_data
def test_onnx_matches_keras_on_real_clips() -> None:
    """Live re-check of the parity report on a slice of real cached clips."""
    onnx_path, sidecar_path = _shipped_paths()
    if not sidecar_path.exists():
        pytest.skip("ONNX artefact not built; run scripts/export_onnx.py")
    from safestreets.evaluation.evaluate import load_checkpoint
    from safestreets.inference.export_onnx import (
        build_bundled_model,
        feature_config_from_configs,
        keras_scores,
        load_cached_clips,
    )

    sidecar = json.loads(sidecar_path.read_text())
    try:
        clips, _, _ = load_cached_clips(REPO_ROOT / "data" / "cache", ["airtlab"], "test")
    except FileNotFoundError:
        pytest.skip("clip cache not built")
    clips = clips[:12]
    bundled = build_bundled_model(
        load_checkpoint(REPO_ROOT / sidecar["source_checkpoint"]), feature_config_from_configs()
    )
    engine = InferenceEngine(onnx_path, sidecar_path)
    onnx_scores = engine.predict_windows(clips)
    assert np.all((onnx_scores >= 0) & (onnx_scores <= 1))
    assert np.max(np.abs(onnx_scores - keras_scores(bundled, clips))) < 1e-4


@pytest.mark.sprint5
@pytest.mark.needs_data
def test_parity_report_meets_the_dod() -> None:
    path = REPORTS_DIR / "onnx_parity.json"
    if not path.exists():
        pytest.skip("onnx_parity.json not built; run scripts/export_onnx.py")
    report = json.loads(path.read_text())
    assert report["n_clips"] >= 50
    assert report["max_abs_diff"] < 1e-4
    assert report["roc_auc_abs_diff"] <= 0.005
    assert report["parity_passed"] and report["auc_passed"]


@pytest.mark.sprint5
@pytest.mark.needs_data
def test_latency_report_has_percentiles_and_implied_fps() -> None:
    path = REPORTS_DIR / "latency.json"
    if not path.exists():
        pytest.skip("latency.json not built; run scripts/export_onnx.py")
    report = json.loads(path.read_text())
    assert 0 < report["p50_ms"] <= report["p95_ms"] <= report["p99_ms"]
    assert report["sustainable_fps"] == pytest.approx(report["stride"] * 1000 / report["p95_ms"])
    assert report["n_iterations"] >= 100


@pytest.mark.sprint5
@pytest.mark.needs_data
def test_engine_analyses_the_sample_video() -> None:
    onnx_path, sidecar_path = _shipped_paths()
    if not (sidecar_path.exists() and SAMPLE_VIDEO.exists()):
        pytest.skip("ONNX artefact or test/sample_video.avi missing")
    engine = InferenceEngine(onnx_path, sidecar_path)
    result = engine.analyse_video(SAMPLE_VIDEO)
    assert 0.0 <= result["clip_score"] <= 1.0
    assert result["verdict"] == engine.verdict(result["clip_score"])
    assert len(result["windows"]) >= 1
    for w in result["windows"]:
        assert 0.0 <= w["raw_score"] <= 1.0
        assert 0.0 <= w["smoothed_score"] <= 1.0
