# Reproducing this project

Exact commands, in order, from a clean clone on macOS with the `conda` environments described in
`CLAUDE.md`. Docker was cut for this deadline (`docs/SPRINT_PLAN.md` §2); the `make all` smoke
path below is the covered reproduction surface instead. Every command is path-agnostic and
config-driven (reads `configs/*.yaml`), so none of it depends on this machine's absolute paths.

## 0. Environment

```bash
conda create -n safestreets --clone tf_env
conda activate safestreets
pip install -e ".[dev]"
pip install pandas pyarrow pyyaml flask pytest ruff opencv-python-headless \
  albumentations optuna mlflow onnx onnxruntime tf2onnx ultralytics open_clip_torch
```

Do not build the environment from scratch; `tf_env` already carries TensorFlow 2.16.2,
tensorflow-metal, and Keras pinned per `docs/decisions/ADR-001-runtime-stack.md`. Leave `tf_env`
itself untouched, other projects use it.

Verify the seam ADR-001 documents (`tf2onnx` cannot convert Keras 3, so every model/training
module sets `TF_USE_LEGACY_KERAS=1` before importing TensorFlow):

```bash
python -c "import safestreets.config; import tensorflow as tf; print(tf.__version__)"
```

## 1. Data: fetch, manifest, cache

```bash
python scripts/fetch_data.py --dataset all          # RWF-2000, RLVS, AIRTLab, UCF-Crime subset
                                                      # XD-Violence is permanently skipped, see docs/DATASETS.md
python -m safestreets.data.manifest                  # writes data/manifests/clips.parquet, splits.json
python scripts/build_cache.py --dataset all --split all --workers 4
                                                      # writes data/cache/<dataset>_<split>.h5
```

## 2. Sanity figures (Phases 1-3)

```bash
python scripts/make_contact_sheet.py --out artifacts/figures/sample_clips.png
python scripts/make_augmented_figure.py --out artifacts/figures/augmented_clips.png
```

## 3. Features, training (Sprint 1)

```bash
python scripts/extract_features.py --dataset all --split all   # writes data/features/<dataset>_<split>.h5
python scripts/train.py --arch lstm_head                       # production head, ~2 min
python scripts/train.py --arch scratch                         # comparison arm, 12 epochs, ~10 min
```

## 4. Evaluation (Sprint 2)

```bash
python scripts/evaluate.py --checkpoint artifacts/checkpoints/lstm_head_baseline.keras
# selects the threshold on val, writes configs/infer.yaml, scores the combined test split and
# the cross-dataset matrix, renders roc.png/pr.png/confusion.png/threshold_sweep.png, and
# regenerates docs/RESULTS.md
```

## 5. HPO (Sprint 3)

```bash
python scripts/tune.py --trials 30 --timeout 2400
# 30-trial Optuna TPE search, ~30 min; writes artifacts/optuna/study.db, configs/model.best.yaml,
# artifacts/reports/eval_test_tuned.json, artifacts/figures/optuna_{history,importances}.png
# Interrupted? Resume against the same study:
python scripts/tune.py --trials 30 --timeout 2400 --resume
```

## 6. AIRTLab fine-tune and attribution (Sprint 4)

```bash
python scripts/finetune.py
# 5-fold grouped CV + full-pool fine-tune + RWF-2000 forgetting check, ~5 min;
# writes artifacts/checkpoints/finetuned_airtlab_lstm_head.keras, eval_finetuned.json

python scripts/attribution_eval.py
# YOLOv8n+ByteTrack person tracking + zero-shot CLIP gender over AIRTLab test, ~10 min;
# writes attribution_eval.json, attribution_sample.json (first downloads yolov8n.pt and the
# CLIP weights on first run; both cache locally afterward)
```

## 7. ONNX export and serving (Sprint 5)

```bash
python scripts/export_onnx.py
# rebuilds backbone+head as one tf_keras graph, converts to ONNX (opset 13), runs the parity
# check against Keras and a per-window latency benchmark; writes artifacts/onnx/*.onnx + .json,
# artifacts/reports/onnx_parity.json, artifacts/reports/latency.json

python -c "from safestreets.web import create_app; create_app().run(port=5000)"
# serves the HTML upload form at http://127.0.0.1:5000/ and POST /api/analyse
```

## 8. Notebooks and this submission (Sprint 6)

```bash
make notebooks
# executes every notebooks/*.ipynb top to bottom, outputs saved in place
```

## One-shot smoke path

```bash
make verify-all     # ruff + full pytest suite
make notebooks       # all six notebooks, outputs saved
```

## Seeding

Every script accepts `--seed` (default 1265, `configs/data.yaml`'s `seed` field). Determinism is
exact for the manifest's split assignment (pure hash function, ADR-002) and the Optuna sampler
(`TPESampler(seed=1265)`); Keras/TensorFlow training itself is seeded but not bit-exact across
runs on this hardware (GPU/Metal non-determinism), so re-running Sprint 1/3/4 training will not
reproduce the exact float values in `docs/RESULTS.md` to the last digit, only to the same
ballpark. Nothing in this repository's committed docs is regenerated automatically from a re-run;
every number was pasted from one specific, logged run, as required by `CLAUDE.md`.

## What this does not cover

- A Docker image / fresh-clone CI reproduction was cut for this deadline (`docs/SPRINT_PLAN.md`
  §2); this document's `make verify-all && make notebooks` path is the covered substitute.
- Re-fetching XD-Violence: permanently skipped, see `docs/DATASETS.md`.
