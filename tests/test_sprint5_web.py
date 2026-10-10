from __future__ import annotations

import io
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from safestreets.web import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = REPO_ROOT / "test" / "sample_video.avi"
WEB_DIR = REPO_ROOT / "safestreets" / "web"

RESULT_KEYS = {
    "clip_score": float,
    "verdict": str,
    "threshold": float,
    "enter_threshold": float,
    "leave_threshold": float,
    "alert_events": int,
    "n_frames_decoded": int,
    "fps": float,
    "frame_step": int,
    "padded": bool,
    "short_video": bool,
    "windows": list,
    "model_version": str,
}
WINDOW_KEYS = {
    "index": int,
    "start_frame": int,
    "end_frame": int,
    "raw_score": float,
    "smoothed_score": float,
    "alert_active": bool,
    "alert_started": bool,
}


def assert_valid_result(body: dict) -> None:
    """Schema check for the POST /api/analyse body (see safestreets/web/api.py)."""
    for key, typ in RESULT_KEYS.items():
        assert key in body, key
        if typ is float:
            assert isinstance(body[key], int | float) and not isinstance(body[key], bool), key
        else:
            assert isinstance(body[key], typ), key
    assert 0.0 <= body["clip_score"] <= 1.0
    assert body["verdict"] in {"violent", "non_violent"}
    expected = "violent" if body["clip_score"] >= body["threshold"] else "non_violent"
    assert body["verdict"] == expected
    assert len(body["windows"]) >= 1
    for w in body["windows"]:
        for key, typ in WINDOW_KEYS.items():
            assert key in w, key
            if typ is float:
                assert isinstance(w[key], int | float), key
            else:
                assert isinstance(w[key], typ), key
        assert 0.0 <= w["raw_score"] <= 1.0


class _FakeEngine:
    """Stands in for InferenceEngine: records what it was asked to analyse and
    whether the upload existed on disk at that moment."""

    model_version = "fake-model-0001"
    git_sha = "0123456789abcdef"
    threshold = 0.2

    def __init__(self, clip_score: float = 0.81, raises: Exception | None = None):
        self.clip_score = clip_score
        self.raises = raises
        self.seen: list[tuple[Path, bool]] = []

    def analyse_video(self, path):
        path = Path(path)
        self.seen.append((path, path.exists()))
        if self.raises:
            raise self.raises
        windows = [
            {
                "index": i,
                "start_frame": 8 * i,
                "end_frame": 8 * i + 15,
                "start_s": 0.3 * i,
                "end_s": 0.3 * i + 0.5,
                "raw_score": s,
                "smoothed_score": s,
                "alert_active": s >= 0.25,
                "alert_started": i == 1,
            }
            for i, s in enumerate((0.1, 0.9, 0.7))
        ]
        return {
            "clip_score": self.clip_score,
            "verdict": "violent" if self.clip_score >= self.threshold else "non_violent",
            "threshold": self.threshold,
            "enter_threshold": 0.25,
            "leave_threshold": 0.15,
            "alert_events": 1,
            "n_frames_decoded": 40,
            "fps": 10.0,
            "frame_step": 3,
            "padded": False,
            "short_video": False,
            "windows": windows,
            "model_version": self.model_version,
        }


@pytest.fixture
def tiny_video(tmp_path) -> Path:
    """A real, decodable 20-frame MJPG .avi, generated so the test needs no data."""
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    rng = np.random.default_rng(1265)
    for _ in range(20):
        writer.write(rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8))
    writer.release()
    assert path.stat().st_size > 0
    return path


@pytest.fixture
def upload_dir(tmp_path) -> Path:
    return tmp_path / "uploads"


def _make_client(engine, upload_dir: Path, **overrides):
    app = create_app(engine=engine, config_overrides={"UPLOAD_DIR": str(upload_dir), **overrides})
    app.testing = True
    return app.test_client()


def _post(client, url: str, path: Path | None = None, name: str | None = None, data=None):
    form = {}
    if path is not None or data is not None:
        payload = data if data is not None else path.read_bytes()
        form["video"] = (io.BytesIO(payload), name or path.name)
    return client.post(url, data=form, content_type="multipart/form-data")


def _uploads_left(upload_dir: Path) -> list[Path]:
    return list(upload_dir.iterdir()) if upload_dir.exists() else []


# --- pages and health ------------------------------------------------------------


@pytest.mark.sprint5
def test_index_renders_upload_form(upload_dir) -> None:
    resp = _make_client(_FakeEngine(), upload_dir).get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'name="video"' in html and 'enctype="multipart/form-data"' in html


@pytest.mark.sprint5
def test_about_renders(upload_dir) -> None:
    assert _make_client(_FakeEngine(), upload_dir).get("/about").status_code == 200


@pytest.mark.sprint5
def test_healthz_returns_model_version_and_git_sha(upload_dir) -> None:
    resp = _make_client(_FakeEngine(), upload_dir).get("/healthz")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model_version"] == "fake-model-0001"
    assert body["git_sha"] == "0123456789abcdef"


@pytest.mark.sprint5
def test_healthz_reports_503_when_engine_cannot_load(upload_dir, monkeypatch) -> None:
    import safestreets.inference.engine as engine_mod

    def broken(*args, **kwargs):
        raise engine_mod.SpecMismatchError("sidecar disagrees")

    monkeypatch.setattr(engine_mod, "InferenceEngine", broken)
    resp = _make_client(None, upload_dir).get("/healthz")
    assert resp.status_code == 503
    assert "SpecMismatchError" in resp.get_json()["error"]


# --- POST /api/analyse: happy path and the four 4xx cases -----------------------


@pytest.mark.sprint5
def test_api_analyse_returns_schema_valid_body(tiny_video, upload_dir) -> None:
    resp = _post(_make_client(_FakeEngine(), upload_dir), "/api/analyse", tiny_video)
    assert resp.status_code == 200
    assert_valid_result(resp.get_json())


@pytest.mark.sprint5
def test_api_rejects_missing_file(upload_dir) -> None:
    resp = _post(_make_client(_FakeEngine(), upload_dir), "/api/analyse")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


@pytest.mark.sprint5
def test_api_rejects_wrong_extension(tiny_video, upload_dir) -> None:
    engine = _FakeEngine()
    resp = _post(_make_client(engine, upload_dir), "/api/analyse", tiny_video, name="clip.txt")
    assert resp.status_code == 415
    assert "extension" in resp.get_json()["error"]
    assert engine.seen == []
    assert _uploads_left(upload_dir) == []


@pytest.mark.sprint5
def test_api_rejects_oversized_upload(upload_dir) -> None:
    engine = _FakeEngine()
    client = _make_client(engine, upload_dir, MAX_CONTENT_LENGTH=1024)
    resp = _post(client, "/api/analyse", data=b"\0" * 8192, name="big.mp4")
    assert resp.status_code == 413
    assert "limit" in resp.get_json()["error"]
    assert engine.seen == []
    assert _uploads_left(upload_dir) == []


@pytest.mark.sprint5
def test_api_rejects_non_decodable_video(upload_dir) -> None:
    engine = _FakeEngine()
    junk = np.random.default_rng(1265).integers(0, 255, 4096, dtype=np.uint8).tobytes()
    resp = _post(_make_client(engine, upload_dir), "/api/analyse", data=junk, name="fake.mp4")
    assert resp.status_code == 415
    assert "decoded" in resp.get_json()["error"]
    assert engine.seen == []  # rejected before the engine ever ran
    assert _uploads_left(upload_dir) == []


@pytest.mark.sprint5
def test_configured_upload_cap_is_set(upload_dir) -> None:
    app = create_app(engine=_FakeEngine(), config_overrides={"UPLOAD_DIR": str(upload_dir)})
    assert app.config["MAX_CONTENT_LENGTH"] == 50 * 1024 * 1024  # configs/serve.yaml


# --- cleanup ----------------------------------------------------------------------


@pytest.mark.sprint5
def test_upload_is_removed_after_analysis(tiny_video, upload_dir) -> None:
    engine = _FakeEngine()
    resp = _post(_make_client(engine, upload_dir), "/api/analyse", tiny_video)
    assert resp.status_code == 200
    (saved_path, existed_during_analysis) = engine.seen[0]
    assert existed_during_analysis
    assert saved_path.parent == upload_dir
    assert re.fullmatch(r"[0-9a-f]{32}\.avi", saved_path.name)  # UUID name, not the client's
    assert not saved_path.exists()
    assert _uploads_left(upload_dir) == []


@pytest.mark.sprint5
def test_upload_is_removed_even_when_analysis_fails(tiny_video, upload_dir) -> None:
    engine = _FakeEngine(raises=RuntimeError("engine blew up"))
    client = _make_client(engine, upload_dir)
    with pytest.raises(RuntimeError, match="engine blew up"):
        _post(client, "/api/analyse", tiny_video)
    assert engine.seen and engine.seen[0][1]
    assert _uploads_left(upload_dir) == []


# --- result page --------------------------------------------------------------------


@pytest.mark.sprint5
def test_result_page_renders_verdict_score_and_timeline(tiny_video, upload_dir) -> None:
    resp = _post(_make_client(_FakeEngine(clip_score=0.81), upload_dir), "/analyse", tiny_video)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert re.search(r'id="verdict"[^>]*>.*?Violent', html, re.S)
    assert re.search(r'id="clip-score">0\.8100<', html)
    assert html.count('class="score-track"') == 3  # one row per timeline window
    assert _uploads_left(upload_dir) == []


@pytest.mark.sprint5
def test_result_page_renders_non_violent_verdict(tiny_video, upload_dir) -> None:
    resp = _post(_make_client(_FakeEngine(clip_score=0.05), upload_dir), "/analyse", tiny_video)
    html = resp.get_data(as_text=True)
    assert re.search(r'id="verdict"[^>]*>.*?Non-violent', html, re.S)
    assert 'id="clip-score">0.0500<' in html


@pytest.mark.sprint5
def test_html_upload_errors_render_on_the_form(upload_dir) -> None:
    resp = _post(_make_client(_FakeEngine(), upload_dir), "/analyse", data=b"x", name="a.txt")
    assert resp.status_code == 415
    assert 'id="error"' in resp.get_data(as_text=True)


# --- legacy defects ------------------------------------------------------------------


@pytest.mark.sprint5
def test_legacy_app_package_and_predict_test_are_gone() -> None:
    assert not (REPO_ROOT / "app").exists()
    assert not (REPO_ROOT / "test" / "test_predict.py").exists()
    needle = "from " + "app"  # split so this file does not match itself
    paths = [REPO_ROOT / "run.py"]
    for root in ("safestreets", "scripts", "tests"):
        paths.extend((REPO_ROOT / root).rglob("*.py"))
    for path in paths:
        assert not re.search(rf"^\s*{needle}\b", path.read_text(), re.M), path


@pytest.mark.sprint5
def test_rendering_needs_no_external_cdn(upload_dir) -> None:
    for template in (WEB_DIR / "templates").glob("*.html"):
        text = template.read_text()
        assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", text), template
    client = _make_client(_FakeEngine(), upload_dir)
    html = client.get("/").get_data(as_text=True)
    assets = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert any("tailwind" in a for a in assets)
    for asset in assets:
        assert client.get(asset).status_code == 200, asset


# --- the real engine on the DoD fixture ---------------------------------------------


@pytest.mark.sprint5
@pytest.mark.needs_data
def test_api_analyse_sample_video_with_real_engine(upload_dir) -> None:
    from safestreets.inference.engine import default_onnx_path

    if not (default_onnx_path().with_suffix(".json").exists() and SAMPLE_VIDEO.exists()):
        pytest.skip("ONNX artefact or test/sample_video.avi missing")
    client = _make_client(None, upload_dir)
    health = client.get("/healthz").get_json()
    assert health["status"] == "ok" and health["model_version"] and health["git_sha"]

    resp = _post(client, "/api/analyse", SAMPLE_VIDEO)
    assert resp.status_code == 200
    body = resp.get_json()
    assert_valid_result(body)
    assert body["model_version"] == health["model_version"]
    assert _uploads_left(upload_dir) == []
