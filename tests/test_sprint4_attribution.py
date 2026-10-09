from __future__ import annotations

import pytest

from safestreets.attribution.gender import GenderClassifierConfig, ZeroShotGenderClassifier
from safestreets.attribution.person import PersonDetector, PersonDetectorConfig, summarize_tracks
from safestreets.attribution.pipeline import AttributionPipelineConfig, build_alert, run_pipeline
from safestreets.training.finetune import summarize_cv


class _FakeDetector:
    def __init__(self, available: bool, detections: list[dict] | None = None, raises=False):
        self._available = available
        self._detections = detections or []
        self._raises = raises

    def available(self) -> bool:
        return self._available

    def track_video(self, video_path):
        if self._raises:
            raise RuntimeError("boom")
        return self._detections


class _FakeGenderClassifier:
    def __init__(self, available: bool, results: list[dict] | None = None, raises=False):
        self._available = available
        self._results = results or []
        self._raises = raises

    def available(self) -> bool:
        return self._available

    def classify(self, crops):
        if self._raises:
            raise RuntimeError("boom")
        return self._results


@pytest.mark.sprint4
def test_failsafe_alert_survives_gender_override() -> None:
    """The Sprint 4 DoD's exact scenario: p_violence=0.99 with the gender
    module forced to report no women detected must still raise an alert."""
    gender_results_no_women = [{"label": "man", "score": 0.9, "probs": {"woman": 0.1, "man": 0.9}}]
    alert = build_alert(
        p_violence=0.99,
        threshold=0.2,
        priority_gender="woman",
        person_detections=[{"frame": 0, "track_id": 1, "bbox": [0, 0, 1, 1], "conf": 0.9}],
        gender_results=gender_results_no_women,
    )
    assert alert["alert"] is True
    assert alert["priority"] == "standard"


@pytest.mark.sprint4
def test_failsafe_alert_survives_missing_attribution() -> None:
    """Alert existence must not depend on attribution being available at all."""
    alert = build_alert(p_violence=0.99, threshold=0.2, person_detections=None, gender_results=None)
    assert alert["alert"] is True
    assert alert["attribution_available"] is False


@pytest.mark.sprint4
def test_below_threshold_never_alerts_even_with_women_detected() -> None:
    gender_results = [{"label": "woman", "score": 0.95, "probs": {"woman": 0.95, "man": 0.05}}]
    alert = build_alert(
        p_violence=0.05,
        threshold=0.2,
        person_detections=[{"frame": 0, "track_id": 1, "bbox": [0, 0, 1, 1], "conf": 0.9}],
        gender_results=gender_results,
    )
    assert alert["alert"] is False


@pytest.mark.sprint4
def test_priority_elevated_only_when_alert_and_priority_gender_present() -> None:
    woman_detected = [{"label": "woman", "score": 0.9, "probs": {"woman": 0.9, "man": 0.1}}]

    elevated = build_alert(0.9, 0.2, "woman", [{"track_id": 1}], woman_detected)
    assert elevated["priority"] == "elevated"

    no_alert_despite_woman = build_alert(0.05, 0.2, "woman", [{"track_id": 1}], woman_detected)
    assert no_alert_despite_woman["priority"] == "standard"


@pytest.mark.sprint4
def test_run_pipeline_falls_back_when_person_detector_raises() -> None:
    """Graceful degradation: a failing (not just missing) detector must not
    crash the pipeline, and the alert must still reflect the violence score."""
    alert = run_pipeline(
        p_violence=0.9,
        video_path="unused.mp4",
        cfg=AttributionPipelineConfig(violence_threshold=0.2),
        person_detector=_FakeDetector(available=True, raises=True),
        gender_classifier=_FakeGenderClassifier(available=False),
    )
    assert alert["alert"] is True
    assert alert["attribution_available"] is False


@pytest.mark.sprint4
def test_run_pipeline_falls_back_when_weights_missing() -> None:
    """A detector/classifier reporting `available() == False` (missing
    weights) must not be queried further, and must not crash the pipeline."""
    alert = run_pipeline(
        p_violence=0.9,
        video_path="unused.mp4",
        cfg=AttributionPipelineConfig(violence_threshold=0.2),
        person_detector=_FakeDetector(available=False),
        gender_classifier=_FakeGenderClassifier(available=False),
    )
    assert alert["alert"] is True
    assert alert["attribution_available"] is False


@pytest.mark.sprint4
def test_run_pipeline_uses_attribution_when_available() -> None:
    detections = [{"frame": 0, "track_id": 1, "bbox": [0, 0, 1, 1], "conf": 0.9}]
    gender_results = [{"label": "woman", "score": 0.9, "probs": {"woman": 0.9, "man": 0.1}}]
    alert = run_pipeline(
        p_violence=0.9,
        video_path="unused.mp4",
        cfg=AttributionPipelineConfig(violence_threshold=0.2),
        person_detector=_FakeDetector(available=True, detections=detections),
        gender_classifier=_FakeGenderClassifier(available=True, results=gender_results),
        gender_crops=["dummy"],
    )
    assert alert["alert"] is True
    assert alert["attribution_available"] is True
    assert alert["priority"] == "elevated"


@pytest.mark.sprint4
def test_person_detector_available_is_false_on_bad_weights() -> None:
    """Unknown weights path must report unavailable, not raise, per the
    Sprint 4 DoD's graceful-degradation item."""
    detector = PersonDetector(PersonDetectorConfig(weights="does-not-exist-anywhere.pt"))
    assert detector.available() is False


@pytest.mark.sprint4
def test_gender_classifier_available_is_false_on_bad_model_name() -> None:
    classifier = ZeroShotGenderClassifier(GenderClassifierConfig(model_name="not-a-real-model"))
    assert classifier.available() is False


@pytest.mark.sprint4
def test_summarize_tracks_sane_on_single_walker() -> None:
    """A single-person clip should yield one dominant track, not sixteen."""
    detections = [{"frame": i, "track_id": 1, "bbox": [0, 0, 1, 1], "conf": 0.9} for i in range(16)]
    summary = summarize_tracks(detections)
    assert summary["n_tracks"] == 1
    assert summary["frames_per_track"] == {1: 16}


@pytest.mark.sprint4
def test_summarize_cv_reports_mean_and_std() -> None:
    fold_reports = [
        {"fold": i, "n_train": 10, "n_test": 2, "accuracy": a, "f1": a, "roc_auc": a}
        for i, a in enumerate([0.8, 0.9, 0.85, 0.95, 0.9])
    ]
    summary = summarize_cv(fold_reports)
    assert summary["accuracy"]["mean"] == pytest.approx(0.88, abs=1e-9)
    assert summary["accuracy"]["std"] > 0
