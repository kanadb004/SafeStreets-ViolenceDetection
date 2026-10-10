# SafeStreets

A real-time violence-detection pipeline (CNN-LSTM) built for a women's-safety CCTV context: it
scores short video clips for violence, raises an alert, and attaches person/gender attribution as
an alert *priority* signal, never as a gate on whether the alert fires. Specified by
`DA1_Report_23BCE1265.pdf`; built per `docs/BUILD_PLAN.md` (Phases 0-3) and
`docs/SPRINT_PLAN.md` (Sprints 1-6, which supersede Phases 4-11 under a 1 to 1.5 day deadline).

![Architecture](artifacts/figures/architecture.png)

## Results

Combined RLVS + AIRTLab + UCF-Crime test split (n=325), Sprint 1/2 baseline checkpoint, threshold
0.2000 (F1-optimal, selected on validation). Full tables, cross-dataset numbers, the HPO
comparison, and the AIRTLab fine-tune in `docs/RESULTS.md`; this table is generated from the same
committed artefacts and must match it exactly.

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.9450 | [0.9204, 0.9663] |
| PR-AUC | 0.9471 | [0.9201, 0.9704] |
| Accuracy | 0.8769 | [0.8400, 0.9138] |
| Precision | 0.8392 | [0.7905, 0.8923] |
| Recall | 0.9543 | [0.9191, 0.9828] |
| F1 | 0.8930 | [0.8595, 0.9260] |

The **shipped** checkpoint is fine-tuned further on AIRTLab (`docs/decisions/ADR-005-finetune-and-attribution.md`):
AIRTLab 5-fold CV ROC-AUC 0.8976 +/- 0.0429 (up from 0.5259 zero-shot), at a measured cost of
0.0582 ROC-AUC on RWF-2000. ONNX serving: parity max abs diff 5.24e-05 against Keras, p95 latency
134.41 ms/window, 59.5 sustainable fps on CPU only (no GPU/CUDA), 18.6x real-time headroom.

## Quickstart

```bash
conda create -n safestreets --clone tf_env   # do not build the env from scratch, see CLAUDE.md
conda activate safestreets
pip install -e ".[dev]"
pip install pandas pyarrow pyyaml flask pytest ruff opencv-python-headless \
  albumentations optuna mlflow onnx onnxruntime tf2onnx ultralytics open_clip_torch

make verify-all      # ruff + full pytest suite
make notebooks        # executes all six notebooks/*.ipynb in place

python -c "from safestreets.web import create_app; create_app().run(port=5000)"
# -> http://127.0.0.1:5000  (upload a short clip, get a verdict and a per-window timeline)
```

Full command sequence from a clean clone, every sprint in order: `docs/REPRODUCE.md`.

## The six notebooks

Each is thin: it imports from `safestreets/`, calls it, displays the result; no copy-pasted
implementation in a cell. Run individually or via `make notebooks`.

| Notebook | Covers |
|---|---|
| `notebooks/01_data_and_preprocessing.ipynb` | the five candidate sources (one skipped), group-aware splits, the leakage assertion, the clip cache |
| `notebooks/02_augmentation.ipynb` | clip-consistent Albumentations, a live identical-frames proof, before/after grid, throughput |
| `notebooks/03_model_and_training.ipynb` | both architectures, the frozen-feature decision with its measurements, training curves, TensorBoard, MLflow |
| `notebooks/04_evaluation.ipynb` | metrics with bootstrap CIs, ROC/PR/confusion, threshold sweep, cross-dataset table |
| `notebooks/05_hpo_and_finetune.ipynb` | the Optuna study, why the tuned model doesn't ship, the AIRTLab 5-fold fine-tune, forgetting check |
| `notebooks/06_inference_and_demo.ipynb` | attribution with the fail-safe invariant proven live, ONNX parity/latency, a real sliding-window score, the Flask app |

A suggested walkthrough order and talking points for presenting these live:
`docs/PRESENTATION_GUIDE.md`.

## Architecture

Frozen MobileNetV2 (ImageNet) feature extraction, precomputed once, feeding a two-layer LSTM head
(the shipped production model); the literal report §4.3 TimeDistributed CNN-LSTM is also built and
trained as a trained-from-scratch comparison arm. See
`docs/decisions/ADR-003-architecture.md` for the measured reasoning (8.7 min/epoch end to end vs.
under 2 hours total frozen-feature, which is what makes a six-sprint, 1 to 1.5 day deadline
possible at all).

Serving bundles the frozen backbone and the fine-tuned head into one ONNX graph, run on CPU-only
ONNX Runtime behind Flask, no TensorFlow import at serve time (`docs/decisions/ADR-006-deployment-target.md`).

## Datasets

RWF-2000, RLVS, AIRTLab (women's-safety-specific fine-tuning), and a UCF-Crime subset, all
group-aware split so sibling clips from one source video never cross train/val/test
(`docs/decisions/ADR-002-dataset-splits.md`). XD-Violence was evaluated and permanently skipped:
its only accessible distribution is 38.3 GB, far over this project's ~44 GB disk budget. Full
provenance, licences, and citations: `docs/DATASETS.md`.

## Ethics

Person detection and gender attribution exist only to label an already-fired alert's *priority*;
they can never suppress or create an alert (`safestreets/attribution/pipeline.py:build_alert`,
enforced by construction). The gender module is zero-shot CLIP, not fit to any labelled data, and
its accuracy is explicitly unmeasured, not measured-and-good: see `docs/ETHICS.md` for the full
discussion, including why this is reported as a qualitative check rather than an accuracy claim.

## Model card

`docs/MODEL_CARD.md` — intended use, out-of-scope use, training data and licences, metrics,
attribution modules, limitations.

## Known limitations of this build

- Frame-level AUC on untrimmed video is out of scope (XD-Violence skipped; the UCF-Crime subset
  is 35 clips, too small on its own).
- The scratch CNN-LSTM comparison arm trained for 12 epochs only (scaled down, not a tuned run).
- The Optuna-tuned head did not beat the Sprint 1 baseline on test; the baseline ships
  (`docs/decisions/ADR-004-hpo.md`).
- The AIRTLab fine-tune that ships costs 0.0582 ROC-AUC on RWF-2000 (`docs/decisions/ADR-005-finetune-and-attribution.md`).
- A Docker image and fresh-clone CI reproduction were cut for this deadline; `make verify-all &&
  make notebooks` is the covered substitute. See `docs/SPRINT_PLAN.md` §2 for the full cut list.

## Repository layout

- `safestreets/` — the package every notebook and script calls into (data, features, models,
  training, evaluation, attribution, inference, web)
- `scripts/` — one-shot CLI entry points (`docs/REPRODUCE.md` lists them in run order)
- `configs/*.yaml` — every tunable lives here, never as a Python literal
- `notebooks/` — the six submission notebooks (`notebooks/legacy/` is the superseded Colab
  reference notebook, kept for history only)
- `docs/` — `BUILD_PLAN.md` (full original scope), `SPRINT_PLAN.md` (the plan actually executed),
  `decisions/` (ADRs), `RESULTS.md`, `MODEL_CARD.md`, `ETHICS.md`, `DATASETS.md`,
  `REPRODUCE.md`, `PRESENTATION_GUIDE.md`, `PROGRESS.md` (sprint-by-sprint evidence log)
- `tests/` — `pytest -m phaseN` / `pytest -m sprintN` per `Makefile`'s `verify` target

## Known legacy defects (fixed in Sprint 5)

The original `app/` package could not serve a request (`predict_video` did not exist) and the
original template discarded the model's prediction. Both are fixed; `app/` was deleted and
replaced by `safestreets/web`. Full list in `CLAUDE.md`.
