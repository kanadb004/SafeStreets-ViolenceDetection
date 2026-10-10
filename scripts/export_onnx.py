#!/usr/bin/env python
"""Sprint 5 ONNX export, Keras-vs-ONNX parity, and per-window latency.

    python scripts/export_onnx.py

1. Bundles frozen MobileNetV2 + the shipped fine-tuned LSTM head (ADR-005) into
   one graph, exports it with ADR-001's recipe to
   artifacts/onnx/safestreets_{tag}.onnx, and writes the sidecar .json.
2. Scores every real clip of the combined test split (Phase 2 clip cache) with
   both the tf_keras graph and onnxruntime; writes max abs diff and both
   ROC-AUCs to artifacts/reports/onnx_parity.json.
3. Times `InferenceEngine.predict_windows` on one 16 frame window at a time
   and writes p50/p95/p99 plus the implied sustainable FPS to
   artifacts/reports/latency.json.

See docs/SPRINT_PLAN.md Sprint 5 and docs/decisions/ADR-006-deployment-target.md.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from safestreets.config import load_config
from safestreets.evaluation.evaluate import load_checkpoint, save_json
from safestreets.inference.engine import InferenceEngine
from safestreets.inference.export_onnx import export, keras_scores, load_cached_clips
from safestreets.inference.spec import load_yaml
from safestreets.inference.stream import load_stream_config
from safestreets.training.train import load_feature_split

# isort: on

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"

PARITY_TOLERANCE = 1e-4  # Sprint 5 DoD: max abs diff under 1e-4
AUC_TOLERANCE = 0.005  # Sprint 5 DoD: ONNX ROC-AUC within 0.005 of Keras
MIN_PARITY_CLIPS = 50  # Sprint 5 DoD: 50 or more real clips


def run_parity(bundled, engine: InferenceEngine, onnx_cfg: dict, cache_dir: str) -> dict:
    datasets = list(onnx_cfg["parity_datasets"])
    split = onnx_cfg["parity_split"]
    clips, labels, clip_ids = load_cached_clips(cache_dir, datasets, split)
    if len(clips) < MIN_PARITY_CLIPS:
        raise RuntimeError(f"only {len(clips)} clips available, DoD needs {MIN_PARITY_CLIPS}")

    s_keras = keras_scores(bundled, clips)
    s_onnx = np.concatenate(
        [engine.predict_windows(clips[i : i + 16]) for i in range(0, len(clips), 16)]
    )
    diff = np.abs(s_keras - s_onnx)

    # Cross-check that the bundled graph is the Sprint 1 feature store + head,
    # not a drifted reimplementation: score the stored features with the head.
    head = load_checkpoint(REPO_ROOT / onnx_cfg["checkpoint"])
    features, f_labels, f_ids = load_feature_split(REPO_ROOT / "data" / "features", datasets, split)
    if f_ids != clip_ids:
        raise RuntimeError("feature store and clip cache disagree on clip order")
    s_head = head.predict(features, batch_size=64, verbose=0).reshape(-1).astype(np.float64)

    auc_keras = float(roc_auc_score(labels, s_keras))
    auc_onnx = float(roc_auc_score(labels, s_onnx))
    return {
        "datasets": datasets,
        "split": split,
        "n_clips": int(len(clips)),
        "n_positive": int(labels.sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p99_abs_diff": float(np.percentile(diff, 99)),
        "tolerance": PARITY_TOLERANCE,
        "parity_passed": bool(diff.max() < PARITY_TOLERANCE),
        "roc_auc_keras": auc_keras,
        "roc_auc_onnx": auc_onnx,
        "roc_auc_abs_diff": abs(auc_keras - auc_onnx),
        "auc_tolerance": AUC_TOLERANCE,
        "auc_passed": bool(abs(auc_keras - auc_onnx) <= AUC_TOLERANCE),
        "feature_store_head_max_abs_diff": float(np.abs(s_keras - s_head).max()),
        # tf_keras places predict() on Metal's /GPU:0 when present; recorded so
        # the reference side of the comparison is not ambiguous.
        "keras_devices": [d.name for d in tf.config.list_logical_devices()],
        "note": (
            "Runtime parity check (same model, tf_keras vs onnxruntime), not a "
            "generalisation estimate: the AIRTLab test clips were part of the ADR-005 "
            "fine-tune pool, so these AUCs are optimistic for AIRTLab."
        ),
    }


def run_latency(engine: InferenceEngine, cache_dir: str, n_iter: int, warmup: int, seed: int):
    clips, _, _ = load_cached_clips(cache_dir, ["rlvs", "airtlab"], "test")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))
    load_before = os.getloadavg()
    for i in range(warmup):
        engine.predict_windows(clips[order[i % len(order)]][None])
    times_ms = []
    for i in range(n_iter):
        window = clips[order[i % len(order)]][None]
        t0 = time.perf_counter()
        engine.predict_windows(window)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    load_after = os.getloadavg()

    times = np.asarray(times_ms)
    p50, p95, p99 = (float(np.percentile(times, q)) for q in (50, 95, 99))
    stream_cfg = load_stream_config(CONFIGS_DIR)
    # One window is scored every `stride` sampled frames, so at the p95 cost the
    # model keeps up with `stride * 1000 / p95_ms` sampled frames per second.
    # The stream samples source video down to `target_fps` (configs/serve.yaml),
    # so headroom = sustainable_fps / target_fps; above 1 means real time.
    sustainable_fps = stream_cfg.stride * 1000.0 / p95
    import onnxruntime as ort

    return {
        "measured_utc": datetime.now(UTC).isoformat(),
        "what": "InferenceEngine.predict_windows on one window: MobileNetV2 x16 frames + "
        "LSTM head, onnxruntime CPUExecutionProvider. Excludes video decode and letterbox.",
        "window_shape": list(clips.shape[1:]),
        "batch_size": 1,
        "n_iterations": n_iter,
        "warmup": warmup,
        "seed": seed,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": float(times.mean()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "stride": stream_cfg.stride,
        "target_fps": stream_cfg.target_fps,
        "sustainable_fps_formula": "stride * 1000 / p95_ms (sampled frames per second)",
        "sustainable_fps": sustainable_fps,
        "windows_per_s_at_p95": 1000.0 / p95,
        "realtime_headroom": sustainable_fps / stream_cfg.target_fps,
        "onnxruntime_version": ort.__version__,
        "providers": engine.session.get_providers(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "loadavg_before": list(load_before),
        "loadavg_after": list(load_after),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1265)
    parser.add_argument("--checkpoint", default=None, help="defaults to serve.yaml onnx.checkpoint")
    parser.add_argument("--latency-iters", type=int, default=300)
    parser.add_argument("--latency-warmup", type=int, default=20)
    args = parser.parse_args()

    onnx_cfg = load_yaml(CONFIGS_DIR / "serve.yaml")["onnx"]
    checkpoint = Path(args.checkpoint or REPO_ROOT / onnx_cfg["checkpoint"])
    data_cfg = load_config()
    cache_dir = str(REPO_ROOT / data_cfg.cache_dir)

    t0 = time.time()
    onnx_path, sidecar_path, bundled = export(
        checkpoint, REPO_ROOT / onnx_cfg["out_dir"], onnx_cfg["tag"], int(onnx_cfg["opset"])
    )
    print(f"exported {onnx_path} ({onnx_path.stat().st_size} bytes) in {time.time() - t0:.1f}s")
    print(f"sidecar  {sidecar_path}")

    engine = InferenceEngine(onnx_path, sidecar_path, configs_dir=CONFIGS_DIR)
    print(f"engine loaded: model_version={engine.model_version} git_sha={engine.git_sha}")

    parity = run_parity(bundled, engine, onnx_cfg, cache_dir)
    parity.update(
        {
            "onnx_file": str(onnx_path.relative_to(REPO_ROOT)),
            "model_version": engine.model_version,
            "git_sha": engine.git_sha,
        }
    )
    save_json(parity, REPORTS_DIR / "onnx_parity.json")
    print(
        f"parity: n={parity['n_clips']} max_abs_diff={parity['max_abs_diff']:.3e} "
        f"(tol {PARITY_TOLERANCE:g}) passed={parity['parity_passed']}"
    )
    print(
        f"roc_auc keras={parity['roc_auc_keras']:.6f} onnx={parity['roc_auc_onnx']:.6f} "
        f"diff={parity['roc_auc_abs_diff']:.2e} passed={parity['auc_passed']}"
    )
    print(f"bundled keras vs feature store + head: {parity['feature_store_head_max_abs_diff']:.3e}")

    latency = run_latency(engine, cache_dir, args.latency_iters, args.latency_warmup, args.seed)
    latency["model_version"] = engine.model_version
    save_json(latency, REPORTS_DIR / "latency.json")
    print(
        f"latency per window: p50={latency['p50_ms']:.2f}ms p95={latency['p95_ms']:.2f}ms "
        f"p99={latency['p99_ms']:.2f}ms  sustainable_fps={latency['sustainable_fps']:.1f} "
        f"(stride {latency['stride']}), headroom x{latency['realtime_headroom']:.1f} "
        f"over target_fps {latency['target_fps']}"
    )
    print(json.dumps({"loadavg_before": latency["loadavg_before"]}))
    ok = parity["parity_passed"] and parity["auc_passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
