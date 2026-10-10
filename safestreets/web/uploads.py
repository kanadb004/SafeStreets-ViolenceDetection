"""Upload validation and the analyse-then-delete lifecycle shared by the HTML
route and the JSON API. Fixes legacy defect 3 (docs/SPRINT_PLAN.md Sprint 5):
the old upload handler had no size cap, no content check and never cleaned up.

- Size: Flask's MAX_CONTENT_LENGTH (configs/serve.yaml `web.max_content_mb`)
  rejects oversized bodies with 413 before this code runs.
- Extension: checked against `web.allowed_extensions`.
- Content: the saved file must open and decode a frame in OpenCV; an `.mp4`
  that is not really a video is rejected with 415.
- Path: a fresh UUID file name in the upload dir, never the client's name.
- Cleanup: the file is unlinked in a `finally`, on success and on any failure.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage

from safestreets.inference.engine import UndecodableVideoError, probe_video


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def extension_of(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def analyse_upload(
    file: FileStorage | None, engine, upload_dir: Path, allowed_extensions: set[str]
) -> dict:
    """Validate, save under a UUID name, analyse with `engine`, always delete."""
    if file is None or not file.filename:
        raise UploadError(400, "no video file provided (form field 'video')")
    ext = extension_of(file.filename)
    if ext not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise UploadError(415, f"unsupported file extension {ext!r}; allowed: {allowed}")

    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}.{ext}"
    try:
        file.save(path)
        if probe_video(path) is None:
            raise UploadError(415, "file could not be decoded as a video")
        try:
            return engine.analyse_video(path)
        except UndecodableVideoError as exc:
            raise UploadError(415, f"file could not be decoded as a video ({exc})") from exc
    finally:
        path.unlink(missing_ok=True)
