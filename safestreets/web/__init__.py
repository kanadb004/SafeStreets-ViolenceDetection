"""Flask app factory for the SafeStreets demo. Replaces the legacy `app/` package.

See docs/SPRINT_PLAN.md Sprint 5 and docs/decisions/ADR-006-deployment-target.md.

    from safestreets.web import create_app
    app = create_app()

The `InferenceEngine` (onnxruntime only, no TensorFlow) is built lazily on the
first request that needs it, so `GET /` renders even before the ONNX artefact
exists. Tests inject a fake engine via `create_app(engine=...)`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, current_app, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from safestreets.inference.spec import CONFIGS_DIR, load_yaml

ENGINE_KEY = "safestreets_engine"


def get_engine():
    engine = current_app.extensions.get(ENGINE_KEY)
    if engine is None:
        from safestreets.inference.engine import InferenceEngine

        engine = InferenceEngine(configs_dir=current_app.config["SAFESTREETS_CONFIGS_DIR"])
        current_app.extensions[ENGINE_KEY] = engine
    return engine


def create_app(
    engine=None, config_overrides: dict | None = None, configs_dir: Path | str = CONFIGS_DIR
) -> Flask:
    web_cfg = load_yaml(Path(configs_dir) / "serve.yaml")["web"]
    upload_dir = web_cfg.get("upload_dir") or str(
        Path(tempfile.gettempdir()) / "safestreets_uploads"
    )

    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=int(web_cfg["max_content_mb"]) * 1024 * 1024,
        UPLOAD_DIR=upload_dir,
        ALLOWED_EXTENSIONS=set(web_cfg["allowed_extensions"]),
        SAFESTREETS_CONFIGS_DIR=str(configs_dir),
    )
    if config_overrides:
        app.config.update(config_overrides)
    if engine is not None:
        app.extensions[ENGINE_KEY] = engine

    from safestreets.web.api import api
    from safestreets.web.routes import main

    app.register_blueprint(main)
    app.register_blueprint(api)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        message = f"upload exceeds the {limit_mb:g} MB limit"
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), 413
        return render_template("index.html", error=message), 413

    return app
