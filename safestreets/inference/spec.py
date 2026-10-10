"""The input/normalisation contract shared by the ONNX exporter and the engine.

See docs/SPRINT_PLAN.md Sprint 5. The exporter writes this contract into the
sidecar JSON next to every `.onnx` file; `InferenceEngine` rebuilds it from the
*active* configs at load time and refuses to run if the two disagree. That is
the guard against serving an artefact exported under one preprocessing contract
while the rest of the code base has moved to another.

Deliberately free of TensorFlow imports: the engine (and so the Flask app) runs
on onnxruntime alone.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from safestreets.data.preprocess import load_preprocess_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

# MobileNetV2's own `preprocess_input` ("tf" mode): x / 127.5 - 1, mapping uint8
# [0, 255] RGB to [-1, 1]. It is baked into the exported graph as a Rescaling
# layer, so callers feed raw letterboxed RGB pixels. This is deliberately NOT
# configs/data.yaml's `pipeline.normalize` (unit_range), which only ever applied
# to the Phase 3 live tf.data path; see safestreets/features/extract.py.
# tests/test_sprint5_inference.py pins these numbers to tf_keras's real
# `preprocess_input`, so they cannot silently drift from the feature store.
NORMALISATION = {
    "backbone": "mobilenetv2",
    "weights": "imagenet",
    "mode": "tf",
    "scale": 1.0 / 127.5,
    "offset": -1.0,
    "input_range": [0, 255],
    "in_graph": True,
}


class SpecMismatchError(RuntimeError):
    """The artefact's recorded contract disagrees with the active config."""


def load_yaml(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_threshold(configs_dir: Path | str = CONFIGS_DIR) -> float:
    return float(load_yaml(Path(configs_dir) / "infer.yaml")["threshold"])


def active_spec(configs_dir: Path | str = CONFIGS_DIR) -> dict:
    """The contract the current configs imply, in the same shape as the sidecar."""
    configs_dir = Path(configs_dir)
    pre = load_preprocess_config(configs_dir / "preprocess.yaml")
    head = load_yaml(configs_dir / "model.best.yaml")["lstm_head"]
    return {
        "input": {
            "name": "frames",
            "shape": [None, pre.n_frames, pre.height, pre.width, 3],
            "dtype": "float32",
            "layout": "NTHWC",
            "channel_order": "RGB",
            "resize": "letterbox",
            "sampling": pre.sampling,
        },
        "normalisation": dict(NORMALISATION),
        "head": {"input_dim": int(head["input_dim"]), "n_frames": int(head["n_frames"])},
        "threshold": load_threshold(configs_dir),
    }


def _diff(recorded, active, path: str) -> list[str]:
    if isinstance(recorded, dict) and isinstance(active, dict):
        out = []
        for key in sorted(set(recorded) | set(active)):
            if key not in recorded or key not in active:
                out.append(f"{path}.{key}: present in only one of sidecar/active config")
            else:
                out.extend(_diff(recorded[key], active[key], f"{path}.{key}"))
        return out
    if isinstance(recorded, float) or isinstance(active, float):
        try:
            if abs(float(recorded) - float(active)) <= 1e-9:
                return []
        except (TypeError, ValueError):
            pass
        return [f"{path}: sidecar={recorded!r} active={active!r}"]
    if isinstance(recorded, list | tuple) and isinstance(active, list | tuple):
        if len(recorded) != len(active):
            return [f"{path}: sidecar={recorded!r} active={active!r}"]
        out = []
        for i, (r, a) in enumerate(zip(recorded, active, strict=True)):
            out.extend(_diff(r, a, f"{path}[{i}]"))
        return out
    return [] if recorded == active else [f"{path}: sidecar={recorded!r} active={active!r}"]


def assert_spec_matches(sidecar: dict, active: dict) -> None:
    """Raise SpecMismatchError listing every field where the artefact's
    recorded contract disagrees with the active config. Raises, never warns."""
    problems = []
    for section in ("input", "normalisation", "head", "threshold"):
        if section not in sidecar:
            problems.append(f"{section}: missing from sidecar")
            continue
        problems.extend(_diff(sidecar[section], active[section], section))
    if problems:
        raise SpecMismatchError(
            "ONNX sidecar disagrees with the active config; re-export with "
            "scripts/export_onnx.py or fix the config:\n  " + "\n  ".join(problems)
        )
