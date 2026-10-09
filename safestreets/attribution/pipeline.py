"""Fuses a violence score with person/gender attribution into one alert. See
docs/SPRINT_PLAN.md Sprint 4.

The non-negotiable property (SPRINT_PLAN.md Sprint 4 notes): a high violence
score raises an alert regardless of the gender module's output, or its
availability. Attribution changes an alert's priority and label, never its
existence. `build_alert` enforces this by construction: `alert` is computed
from `p_violence` and `threshold` alone, before `person_detections` or
`gender_results` are even inspected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttributionPipelineConfig:
    violence_threshold: float
    priority_gender: str = "woman"


def build_alert(
    p_violence: float,
    threshold: float,
    priority_gender: str = "woman",
    person_detections: list[dict] | None = None,
    gender_results: list[dict] | None = None,
) -> dict:
    """`person_detections`/`gender_results` may be `None` (attribution
    unavailable or skipped) or an empty list (no people found); either way
    `alert` depends only on `p_violence >= threshold`.
    """
    alert = bool(p_violence >= threshold)
    attribution_available = person_detections is not None and gender_results is not None
    n_people = len({d["track_id"] for d in (person_detections or [])})
    genders_detected = [g["label"] for g in (gender_results or [])]
    elevated = alert and priority_gender in genders_detected

    return {
        "alert": alert,
        "p_violence": float(p_violence),
        "threshold": float(threshold),
        "attribution_available": attribution_available,
        "n_people": n_people,
        "genders_detected": genders_detected,
        "priority": "elevated" if elevated else "standard",
    }


def run_pipeline(
    p_violence: float,
    video_path: Path | str,
    cfg: AttributionPipelineConfig,
    person_detector=None,
    gender_classifier=None,
    gender_crops: list | None = None,
) -> dict:
    """End-to-end: violence score is already computed by the caller (the
    lstm_head); this only adds attribution on top, and never lets a missing
    or failing YOLO/CLIP weight stop the violence-only alert from being
    raised (Sprint 4 DoD: graceful degradation).
    """
    person_detections = None
    if person_detector is not None:
        try:
            if person_detector.available():
                person_detections = person_detector.track_video(video_path)
        except Exception:
            logger.warning(
                "person detection failed for %s; falling back to violence-only alert",
                video_path,
            )

    gender_results = None
    if gender_classifier is not None and gender_crops:
        try:
            if gender_classifier.available():
                gender_results = gender_classifier.classify(gender_crops)
        except Exception:
            logger.warning(
                "gender classification failed for %s; falling back to violence-only alert",
                video_path,
            )

    return build_alert(
        p_violence,
        cfg.violence_threshold,
        cfg.priority_gender,
        person_detections,
        gender_results,
    )
