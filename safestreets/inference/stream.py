"""Sliding-window streaming scorer: ring buffer, EMA smoothing, hysteresis alerts.

See docs/SPRINT_PLAN.md Sprint 5. Frames arrive one at a time. A ring buffer
holds the newest `window` frames (configs/preprocess.yaml's n_frames, so it
always matches what the model was trained on) and emits a window every `stride`
frames once full, i.e. 50% overlap at the default stride 8. Each window's raw
score is smoothed with an exponential moving average, and alerts use two
thresholds derived from the single operating threshold in configs/infer.yaml:

    enter = threshold + margin      leave = threshold - margin

so a smoothed score that wanders back and forth across `threshold` raises one
alert, not one per crossing. Pure numpy: no model, no TensorFlow, so the logic
is unit-testable with a fake scoring function.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from safestreets.inference.spec import CONFIGS_DIR, load_yaml


@dataclass(frozen=True)
class StreamConfig:
    window: int = 16
    stride: int = 8
    ema_alpha: float = 0.3
    hysteresis_margin: float = 0.05
    target_fps: float = 3.2

    def __post_init__(self) -> None:
        if self.window <= 0 or self.stride <= 0:
            raise ValueError("window and stride must be positive")
        if self.stride > self.window:
            raise ValueError("stride larger than window would skip frames entirely")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.hysteresis_margin < 0:
            raise ValueError("hysteresis_margin must be non-negative")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")


def load_stream_config(configs_dir: Path | str = CONFIGS_DIR) -> StreamConfig:
    """`window` comes from configs/preprocess.yaml; the rest from configs/serve.yaml."""
    configs_dir = Path(configs_dir)
    n_frames = int(load_yaml(configs_dir / "preprocess.yaml")["n_frames"])
    raw = load_yaml(configs_dir / "serve.yaml")["stream"]
    return StreamConfig(window=n_frames, **raw)


def hysteresis_thresholds(threshold: float, margin: float) -> tuple[float, float]:
    """(enter, leave), clamped into (0, 1) so neither band edge is unreachable."""
    enter = min(threshold + margin, 1.0)
    leave = max(threshold - margin, 0.0)
    if leave > enter:
        raise ValueError("leave threshold above enter threshold")
    return enter, leave


def frame_step(video_fps: float, target_fps: float) -> int:
    """Keep every k-th decoded frame so the stream runs at ~target_fps."""
    if not video_fps or video_fps <= 0 or not np.isfinite(video_fps):
        return 1
    return max(1, int(round(video_fps / target_fps)))


class RingBuffer:
    """Fixed-capacity frame buffer emitting a full window every `stride` pushes."""

    def __init__(self, window: int, stride: int):
        self.window = window
        self.stride = stride
        self._frames: deque = deque(maxlen=window)
        self._since_emit = 0
        self._emitted_any = False

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        self._frames.append(frame)
        self._since_emit += 1
        if len(self._frames) < self.window:
            return None
        if self._emitted_any and self._since_emit < self.stride:
            return None
        self._emitted_any = True
        self._since_emit = 0
        return np.stack(self._frames, axis=0)


class Hysteresis:
    """Two-threshold alert latch. `update` returns True only on a rising edge."""

    def __init__(self, enter: float, leave: float):
        self.enter = enter
        self.leave = leave
        self.active = False

    def update(self, score: float) -> bool:
        if not self.active and score >= self.enter:
            self.active = True
            return True
        if self.active and score < self.leave:
            self.active = False
        return False


class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


@dataclass
class WindowResult:
    index: int
    start_frame: int  # index into the *source* video, inclusive
    end_frame: int  # inclusive
    raw_score: float
    smoothed_score: float
    alert_active: bool
    alert_started: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "raw_score": self.raw_score,
            "smoothed_score": self.smoothed_score,
            "alert_active": self.alert_active,
            "alert_started": self.alert_started,
        }


@dataclass
class StreamScorer:
    """Feed frames in, get one WindowResult per emitted window.

    `score_fn` maps a (1, window, H, W, 3) uint8 batch to a length-1 score array;
    in production that is `InferenceEngine.predict_windows`.
    """

    score_fn: Callable[[np.ndarray], np.ndarray]
    threshold: float
    cfg: StreamConfig = field(default_factory=StreamConfig)

    def __post_init__(self) -> None:
        enter, leave = hysteresis_thresholds(self.threshold, self.cfg.hysteresis_margin)
        self._buffer = RingBuffer(self.cfg.window, self.cfg.stride)
        self._source_index: deque = deque(maxlen=self.cfg.window)
        self._ema = EMA(self.cfg.ema_alpha)
        self._latch = Hysteresis(enter, leave)
        self._n_windows = 0
        self.alert_count = 0

    @property
    def enter_threshold(self) -> float:
        return self._latch.enter

    @property
    def leave_threshold(self) -> float:
        return self._latch.leave

    def push(self, frame: np.ndarray, source_index: int | None = None) -> WindowResult | None:
        self._source_index.append(source_index if source_index is not None else -1)
        window = self._buffer.push(frame)
        if window is None:
            return None
        raw = float(np.asarray(self.score_fn(window[None, ...])).reshape(-1)[0])
        return self.push_score(raw, self._source_index[0], self._source_index[-1])

    def push_score(self, raw: float, start_frame: int = -1, end_frame: int = -1) -> WindowResult:
        """Smoothing + hysteresis for an already-scored window (also used by tests)."""
        smoothed = self._ema.update(raw)
        started = self._latch.update(smoothed)
        self.alert_count += int(started)
        result = WindowResult(
            index=self._n_windows,
            start_frame=start_frame,
            end_frame=end_frame,
            raw_score=raw,
            smoothed_score=smoothed,
            alert_active=self._latch.active,
            alert_started=started,
        )
        self._n_windows += 1
        return result
