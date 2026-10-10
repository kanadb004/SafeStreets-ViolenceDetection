"""JSON API. `POST /api/analyse` with multipart field `video`.

Response body (200):
    clip_score        float in [0, 1], P(violent) for the whole clip
    verdict           "violent" | "non_violent" (clip_score >= threshold)
    threshold         operating threshold, configs/infer.yaml
    enter_threshold   hysteresis band, configs/serve.yaml stream.hysteresis_margin
    leave_threshold
    alert_events      rising edges of the smoothed per-window alert latch
    windows           per-window timeline: index, start/end frame and seconds,
                      raw_score, smoothed_score, alert_active, alert_started
    model_version, n_frames_decoded, fps, frame_step, padded, short_video

Errors are `{"error": "<message>"}` with 400 (no file), 413 (too large),
415 (wrong extension or not decodable).
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from safestreets.web import get_engine
from safestreets.web.uploads import UploadError, analyse_upload

api = Blueprint("api", __name__, url_prefix="/api")


@api.post("/analyse")
def analyse():
    try:
        result = analyse_upload(
            request.files.get("video"),
            get_engine(),
            Path(current_app.config["UPLOAD_DIR"]),
            current_app.config["ALLOWED_EXTENSIONS"],
        )
    except UploadError as exc:
        return jsonify({"error": exc.message}), exc.status
    return jsonify(result), 200
