"""Person detection and tracking over a clip's source video. See
docs/SPRINT_PLAN.md Sprint 4.

Ultralytics (`YOLO`) is imported lazily, inside `_load`, not at module level:
`safestreets.attribution` must still be importable, and `available()` must
still return a clean `False`, on a machine where the `ml` extra (and its
YOLO weights) was never installed. That is the "missing weight logs a
warning and falls back" DoD item.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonDetectorConfig:
    weights: str = "yolov8n.pt"
    person_class: int = 0
    every_n_frames: int = 4
    conf: float = 0.4
    tracker: str = "bytetrack.yaml"


class PersonDetector:
    def __init__(self, cfg: PersonDetectorConfig) -> None:
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.cfg.weights)
        return self._model

    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            logger.warning("person detector weights %r unavailable", self.cfg.weights)
            return False

    def track_video(self, video_path: Path | str) -> list[dict]:
        """Every `cfg.every_n_frames`-th frame, person-class boxes with a
        stable ByteTrack id. One row per (frame, track) detection.
        """
        model = self._load()
        results = model.track(
            source=str(video_path),
            classes=[self.cfg.person_class],
            tracker=self.cfg.tracker,
            conf=self.cfg.conf,
            vid_stride=self.cfg.every_n_frames,
            persist=False,
            verbose=False,
        )
        detections = []
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for box, track_id, conf in zip(xyxy, track_ids, confs):
                detections.append(
                    {
                        "frame": frame_idx,
                        "track_id": int(track_id),
                        "bbox": [float(v) for v in box],
                        "conf": float(conf),
                    }
                )
        return detections


def summarize_tracks(detections: list[dict]) -> dict:
    """Per-track frame counts, so a sanity check can tell "one dominant
    track, not 16" (Sprint 4 DoD) apart from a tracker that is reassigning
    ids every frame.
    """
    counts = Counter(d["track_id"] for d in detections)
    return {
        "n_detections": len(detections),
        "n_tracks": len(counts),
        "frames_per_track": dict(counts),
    }
