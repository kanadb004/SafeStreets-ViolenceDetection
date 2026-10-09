#!/usr/bin/env python
"""Sprint 4 attribution evaluation: runs person detection + tracking +
zero-shot gender attribution over the AIRTLab test clips, fuses each result
with that clip's lstm_head violence score, and writes
artifacts/reports/attribution_eval.json.

AIRTLab test (48 clips) is used rather than the full corpus: it is the one
dataset in this pipeline that is explicitly women's-safety relevant (acted
violence, visible people throughout), and keeps this CPU-only run under a
few minutes. The first 20 clips are also written to attribution_sample.json
for the Sprint 4 DoD's "hand-checked 20 clip sample".

    python scripts/attribution_eval.py

See docs/SPRINT_PLAN.md Sprint 4.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
import yaml

from safestreets.attribution.gender import GenderClassifierConfig, ZeroShotGenderClassifier
from safestreets.attribution.person import PersonDetector, PersonDetectorConfig, summarize_tracks
from safestreets.attribution.pipeline import AttributionPipelineConfig, build_alert
from safestreets.config import load_config
from safestreets.evaluation.evaluate import load_checkpoint, score_split
from safestreets.training.train import TrainConfig, load_train_config

# isort: on

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
SAMPLE_SIZE = 20


def _read_frame(path: Path | str, frame_idx: int):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _best_crop_per_track(path: Path | str, detections: list[dict], every_n_frames: int):
    """Highest-confidence detection per track id, cropped from the source
    video (not the 112x112 training cache, which is too small to classify).
    """
    best: dict[int, dict] = {}
    for det in detections:
        current = best.get(det["track_id"])
        if current is None or det["conf"] > current["conf"]:
            best[det["track_id"]] = det

    crops = []
    for det in best.values():
        frame = _read_frame(path, det["frame"] * every_n_frames)
        if frame is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(frame[y1:y2, x1:x2])
    return crops


def evaluate_clip(
    clip_id: str,
    path: str,
    p_violence: float,
    person_detector: PersonDetector,
    gender_classifier: ZeroShotGenderClassifier,
    pipeline_cfg: AttributionPipelineConfig,
    person_cfg: PersonDetectorConfig,
) -> dict:
    detections = None
    try:
        if person_detector.available():
            detections = person_detector.track_video(path)
    except Exception:
        detections = None

    gender_results = None
    track_summary = None
    if detections is not None:
        track_summary = summarize_tracks(detections)
        crops = _best_crop_per_track(path, detections, person_cfg.every_n_frames)
        try:
            if gender_classifier.available():
                gender_results = gender_classifier.classify(crops)
        except Exception:
            gender_results = None

    alert = build_alert(
        p_violence,
        pipeline_cfg.violence_threshold,
        pipeline_cfg.priority_gender,
        detections,
        gender_results,
    )
    return {
        "clip_id": clip_id,
        "track_summary": track_summary,
        "gender_results": gender_results,
        "alert": alert,
    }


def main() -> int:
    with open(CONFIGS_DIR / "attribution.yaml") as f:
        attr_cfg_raw = yaml.safe_load(f)
    person_cfg = PersonDetectorConfig(**attr_cfg_raw["person"])
    gender_cfg = GenderClassifierConfig(**attr_cfg_raw["gender"])
    pipeline_cfg = AttributionPipelineConfig(
        violence_threshold=attr_cfg_raw["violence_threshold"],
        priority_gender=attr_cfg_raw["priority_gender"],
    )

    data_cfg = load_config()
    train_cfg: TrainConfig = load_train_config(CONFIGS_DIR / "train.yaml")
    checkpoint = Path(train_cfg.checkpoints_dir) / "lstm_head_baseline.keras"
    model = load_checkpoint(checkpoint)
    y, scores, clip_ids = score_split(model, train_cfg.features_dir, ["airtlab"], "test")
    score_lookup = dict(zip(clip_ids, scores))

    manifest = pd.read_parquet(Path(data_cfg.manifests_dir) / "clips.parquet")
    path_lookup = manifest.set_index("clip_id")["path"].to_dict()

    person_detector = PersonDetector(person_cfg)
    gender_classifier = ZeroShotGenderClassifier(gender_cfg)

    results = []
    for clip_id in clip_ids:
        result = evaluate_clip(
            clip_id,
            path_lookup[clip_id],
            float(score_lookup[clip_id]),
            person_detector,
            gender_classifier,
            pipeline_cfg,
            person_cfg,
        )
        results.append(result)
        print(
            f"{clip_id}: tracks={result['track_summary']}, "
            f"genders={result['gender_results']}, alert={result['alert']['alert']}"
        )

    report = {
        "dataset": "airtlab",
        "split": "test",
        "n_clips": len(results),
        "person_detector_available": person_detector.available(),
        "gender_classifier_available": gender_classifier.available(),
        "results": results,
    }
    out_path = ARTIFACTS_DIR / "reports" / "attribution_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    sample_path = ARTIFACTS_DIR / "reports" / "attribution_sample.json"
    sample_path.write_text(json.dumps(results[:SAMPLE_SIZE], indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path} and {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
