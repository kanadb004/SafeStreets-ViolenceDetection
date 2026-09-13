# ADR-001: Runtime stack: TF / Keras / tf2onnx / ORT versions

- **Status:** accepted
- **Date:** 2026-09-13
- **Phase:** 0

## Context

The `tf_env` conda environment (cloned into `safestreets`, per §3.2 of `docs/BUILD_PLAN.md`) ships
TensorFlow 2.16.2 with Keras 3.10.0. `tf2onnx` does not support the Keras 3 model format, so
exporting a trained model to ONNX for Phase 9's streaming inference engine would fail without a
resolution. This blocks Phase 4 (training must produce an exportable model) and Phase 9 (ONNX
export) unless settled before any model code is written.

A second, unplanned problem surfaced during the spike: installing `onnx`, `onnxruntime`, and
`opencv-python-headless` at their latest versions pulls in `numpy>=2` and `ml_dtypes>=0.5.4`,
both of which conflict with TensorFlow 2.16.2's pins (`numpy<2.0`, `ml_dtypes~=0.3.1`). A first
`pip install tf-keras` attempt (unpinned) also silently upgraded the environment's `tensorflow`
to a generic 2.21.0 wheel and broke the Metal plugin (`dlopen` failure loading
`libmetal_plugin.dylib`). Both problems needed pinned versions, not just the Keras 3 workaround
named in the build plan.

## Options considered

1. **`tf-keras` + `TF_USE_LEGACY_KERAS=1` + `tf2onnx` from a `SavedModel` export** — the path the
   build plan expected to win. Requires pinning `tf-keras` to the `2.16.*` line to avoid it
   dragging in an incompatible `tensorflow` wheel, and pinning `onnx`/`onnxruntime` to versions
   that still accept `ml_dtypes~=0.3.1`.
2. Keras 3 native `model.export(...)` → `tf2onnx.convert --saved-model` — not attempted; option 1
   worked on the first properly-pinned try, so there was no reason to test the untested Keras-3-native
   path against `tf2onnx`, which does not officially support it.
3. Downgrade to TF 2.15 in a separate export-only env — not attempted, same reason: option 1 is
   sufficient and keeps a single environment for training and export.

## Decision

Use **tf-keras (legacy Keras) with `TF_USE_LEGACY_KERAS=1`**, export via `model.export(saved_model_dir)`,
convert with `tf2onnx.convert --saved-model`, and run in `onnxruntime`. Pin the full chain, not
just `tf-keras`, since the unpinned install order breaks the environment:

| Package | Version |
|---|---|
| `tensorflow-macos` | `2.16.2` |
| `tensorflow-metal` | `1.2.0` |
| `tf-keras` | `2.16.0` (installed as `tf-keras==2.16.*`) |
| `keras` | `3.15.1` (present as a transitive dep; legacy path bypasses it via `TF_USE_LEGACY_KERAS=1`) |
| `numpy` | `1.26.4` (must stay `<2.0` for TensorFlow 2.16.2) |
| `ml_dtypes` | `0.3.2` (must stay `~=0.3.1` for TensorFlow 2.16.2) |
| `onnx` | `1.16.1` (later versions require `ml_dtypes>=0.5.4`, incompatible with the pin above) |
| `onnxruntime` | `1.18.1` |
| `tf2onnx` | `1.17.0` |
| `opencv-python-headless` | `4.9.0.80` (versions `>=5` require `numpy>=2`) |

Install order matters: install `tensorflow-macos`, `tensorflow-metal`, and `tf-keras` together in
one pinned command so pip's resolver does not upgrade `tensorflow` as a side effect of satisfying
`tf-keras`. Install `onnx`/`onnxruntime`/`opencv-python-headless` at the pinned versions above
after, and re-pin `numpy`/`ml_dtypes` last if any later install nudges them.

`pyproject.toml` records the exact pins under the `dev`/`ml` extras so a fresh env clone gets the
identical working set.

## Consequences

- Model code (Phase 4 onward) must build and train with `tf_keras`, not `tensorflow.keras` or bare
  `keras`, and must set `TF_USE_LEGACY_KERAS=1` before importing TensorFlow. This should live in
  `safestreets/config.py` or an env bootstrap, not be repeated ad hoc per script.
- `onnx`/`onnxruntime`/`opencv-python-headless` are pinned below their latest releases. Revisit this
  ADR if a future phase needs an `onnx` feature only available in `>=1.17`, or if TensorFlow is
  upgraded past 2.16.x, since that would lift the `numpy<2`/`ml_dtypes~=0.3.1` constraint driving
  the rest of the pins.
- `tensorflow-metal` is confirmed working and gives a real speedup on this machine, so local
  training is viable; heavy phases (4, 6) can still fall back to Colab per `CLAUDE.md` without any
  code change, since entry points stay config-driven.

## Evidence

Environment: macOS, Apple M2, 16 GB unified memory, conda env `safestreets` cloned from `tf_env`.

Installed versions (`python -c "import <mod>; print(<mod>.__version__)"` for each):

```
tensorflow 2.16.2
tf_keras 2.16.0
keras 3.15.1
tf2onnx 1.17.0
onnx 1.16.1
onnxruntime 1.18.1
numpy 1.26.4
ml_dtypes 0.3.2
cv2 4.9.0
```

Parity spike (`TimeDistributed(Conv2D)+LSTM` model, export to ONNX, compare Keras vs ONNX Runtime
outputs on the same random input, seed 1265):

```
tf 2.16.2 tf_keras 2.16.0 tf2onnx 1.17.0 onnxruntime 1.18.1
...
Successfully converted TensorFlow model /tmp/spike_saved_model to ONNX
keras_out [0.4972795  0.49701098]
onnx_out [0.49727955 0.49701098]
max abs diff 5.9604645e-08
PARITY OK
```

`5.96e-08` is well within the `1e-4` tolerance required by Phase 0's Definition of Done.

Metal acceleration benchmark (10x 2048^3 matmul, `tf.matmul` under `tf.device`):

```
/GPU:0   1458 GFLOP/s     10x 2048^3 matmul
/CPU:0   322 GFLOP/s     10x 2048^3 matmul
speedup: 4.5x
```

Metal accelerates and does not crash. The GFLOP/s and speedup figures differ from the informal
note in `CLAUDE.md` (2080 GFLOP/s, 10x) taken on the same date; this run used a smaller, single-size
benchmark and a freshly cloned environment, and machine thermal/background load varies between
runs. Both runs agree on the qualitative conclusion: Metal is a net win over CPU on this machine.
