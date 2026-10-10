"""HTML pages and the health check. Fixes legacy defects 1 and 2
(docs/SPRINT_PLAN.md Sprint 5): the old routes imported a `predict_video` that
never existed, and the old page threw the prediction away. Here the upload form
posts to `/analyse`, which renders the verdict, the score and the per-window
timeline server side, so the result is visible with JavaScript disabled.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from safestreets.web import get_engine
from safestreets.web.uploads import UploadError, analyse_upload

main = Blueprint("main", __name__)


@main.get("/")
def index():
    return render_template("index.html")


@main.get("/about")
def about():
    return render_template("about.html")


@main.get("/healthz")
def healthz():
    try:
        engine = get_engine()
    except Exception as exc:  # report, do not crash: this is the probe endpoint
        return jsonify({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), 503
    return jsonify(
        {
            "status": "ok",
            "model_version": engine.model_version,
            "git_sha": engine.git_sha,
            "threshold": engine.threshold,
        }
    )


@main.post("/analyse")
def analyse_page():
    try:
        result = analyse_upload(
            request.files.get("video"),
            get_engine(),
            Path(current_app.config["UPLOAD_DIR"]),
            current_app.config["ALLOWED_EXTENSIONS"],
        )
    except UploadError as exc:
        return render_template("index.html", error=exc.message), exc.status
    return render_template("result.html", result=result)
