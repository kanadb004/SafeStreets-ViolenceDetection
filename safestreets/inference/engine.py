"""`InferenceEngine`: the shipped model served from ONNX Runtime, no TensorFlow.

See docs/SPRINT_PLAN.md Sprint 5 and docs/decisions/ADR-006-deployment-target.md.
The `.onnx` artefact (safestreets/inference/export_onnx.py) bundles MobileNetV2
feature extraction and the fine-tuned LSTM head into one graph, so this module
needs only onnxruntime, numpy and OpenCV. Clip decoding reuses Phase 2's
`read_clip` / `letterbox_resize` / `sample_frame_indices` unchanged, so a video
scored here is sampled and resized exactly as the training clips were.

On construction the engine loads the sidecar JSON, rebuilds the contract the
active configs imply (`safestreets.inference.spec.active_spec`), and raises
`SpecMismatchError` before it ever opens the ONNX session if they disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from safestreets.data.preprocess import letterbox_resize, load_preprocess_config, read_clip
from safestreets.inference.spec import (
    CONFIGS_DIR,
    SpecMismatchError,
    active_spec,
    assert_spec_matches,
    load_yaml,
)
from safestreets.inference.stream import StreamScorer, frame_step, load_stream_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class UndecodableVideoError(ValueError):
    """OpenCV could not decode the file into a clip."""


def probe_video(path: Path | str) -> dict | None:
    """Cheap decodability check: opens with OpenCV and decodes one frame.
    Returns basic stream info, or None if the file is not a readable video."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or frame is None or n_frames <= 0:
            return None
        return {
            "n_frames": n_frames,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }
    finally:
        cap.release()


def default_onnx_path(configs_dir: Path | str = CONFIGS_DIR) -> Path:
    onnx_cfg = load_yaml(Path(configs_dir) / "serve.yaml")["onnx"]
    out_dir = Path(onnx_cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    return out_dir / f"safestreets_{onnx_cfg['tag']}.onnx"


class InferenceEngine:
    def __init__(
        self,
        onnx_path: Path | str | None = None,
        sidecar_path: Path | str | None = None,
        configs_dir: Path | str = CONFIGS_DIR,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ):
        self.onnx_path = Path(onnx_path) if onnx_path else default_onnx_path(configs_dir)
        sidecar_path = Path(sidecar_path) if sidecar_path else self.onnx_path.with_suffix(".json")
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"sidecar {sidecar_path} missing; the engine will not serve an ONNX file "
                "whose input contract it cannot verify (run scripts/export_onnx.py)"
            )
        self.sidecar = json.loads(sidecar_path.read_text())
        # Refuse before touching the model file: a mismatched contract is fatal.
        assert_spec_matches(self.sidecar, active_spec(configs_dir))

        import onnxruntime as ort

        self.session = ort.InferenceSession(str(self.onnx_path), providers=list(providers))
        (inp,) = self.session.get_inputs()
        expected = self.sidecar["input"]["shape"][1:]
        if list(inp.shape[1:]) != list(expected):
            raise SpecMismatchError(
                f"ONNX graph input {inp.shape} does not match sidecar shape {expected}"
            )
        self._input_name = inp.name
        self.threshold = float(self.sidecar["threshold"])
        self.preprocess_cfg = load_preprocess_config(Path(configs_dir) / "preprocess.yaml")
        self.stream_cfg = load_stream_config(configs_dir)

    @property
    def model_version(self) -> str:
        return self.sidecar["model_version"]

    @property
    def git_sha(self) -> str:
        return self.sidecar["git_sha"]

    def predict_windows(self, windows: np.ndarray) -> np.ndarray:
        """(N, T, H, W, 3) uint8 RGB letterboxed windows -> (N,) P(violent)."""
        windows = np.asarray(windows)
        expected = tuple(self.sidecar["input"]["shape"][1:])
        if windows.ndim != 5 or windows.shape[1:] != expected:
            raise ValueError(f"expected (N, {expected}), got {windows.shape}")
        out = self.session.run(None, {self._input_name: windows.astype(np.float32)})[0]
        return out.reshape(-1).astype(np.float64)

    def score_clip(self, path: Path | str) -> float:
        """Whole-video score, sampled exactly like a training clip (Phase 2 read_clip)."""
        clip = read_clip(str(path), self.preprocess_cfg)
        if clip.frames is None:
            raise UndecodableVideoError(clip.skip_reason or "unreadable")
        return float(self.predict_windows(clip.frames[None])[0])

    def verdict(self, score: float) -> str:
        return "violent" if score >= self.threshold else "non_violent"

    def analyse_video(self, path: Path | str) -> dict:
        """Clip-level score plus the per-window streaming timeline for one file."""
        clip = read_clip(str(path), self.preprocess_cfg)
        if clip.frames is None:
            raise UndecodableVideoError(clip.skip_reason or "unreadable")
        clip_score = float(self.predict_windows(clip.frames[None])[0])

        scorer = StreamScorer(self.predict_windows, self.threshold, self.stream_cfg)
        cfg = self.preprocess_cfg
        cap = cv2.VideoCapture(str(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            step = frame_step(fps, self.stream_cfg.target_fps)
            windows = []
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if index % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = scorer.push(letterbox_resize(rgb, cfg.height, cfg.width), index)
                    if result is not None:
                        windows.append(result)
                index += 1
        finally:
            cap.release()
        n_decoded = index

        short_video = not windows
        if short_video:
            # Fewer sampled frames than one window: the only honest window is
            # the whole (loop-padded) clip, which is what clip_score already is.
            windows.append(scorer.push_score(clip_score, 0, max(n_decoded - 1, 0)))

        def seconds(frame_idx: int) -> float | None:
            return round(frame_idx / fps, 3) if fps > 0 else None

        timeline = []
        for w in windows:
            row = w.to_dict()
            row["start_s"] = seconds(w.start_frame)
            row["end_s"] = seconds(w.end_frame)
            timeline.append(row)

        return {
            "clip_score": clip_score,
            "verdict": self.verdict(clip_score),
            "threshold": self.threshold,
            "enter_threshold": scorer.enter_threshold,
            "leave_threshold": scorer.leave_threshold,
            "alert_events": scorer.alert_count,
            "n_frames_decoded": n_decoded,
            "fps": fps,
            "frame_step": step,
            "padded": bool(clip.padded),
            "short_video": short_video,
            "windows": timeline,
            "model_version": self.model_version,
        }
