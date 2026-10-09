# SafeStreets — Deadline Sprint Plan

**This document supersedes Phases 4 to 11 of `docs/BUILD_PLAN.md`.** Phases 0 to 3 are `DONE` and
unchanged. `BUILD_PLAN.md` stays as the record of the full research scope and of what was cut.

**Deadline:** roughly 1 to 1.5 days from 2026-10-10.
**Deliverable:** a presentable academic submission. Six executed notebooks covering every aspect of
the project, backed by the real `safestreets/` package, with honest measured results.

---

## 1. The decision that makes this fit

The original plan trained the CNN and LSTM end to end. Measured on this machine that is
**8.7 min/epoch** (4.4 h for a 30 epoch run) and an Optuna study was **24 to 27 h**. Eight phases of
that does not fit a day.

Measured alternative, on an idle machine, 2026-10-10:

| | |
|---|---|
| MobileNetV2 (frozen, ImageNet) feature extraction | **20.2 ms/clip**, so **1.5 min for all 4332 cached clips** |
| LSTM head trained on those features | **1.96 s/epoch** (3002 clips, bs=32), 50 epochs in 1.6 min |
| 30 trial Optuna study, 25 epochs each | **24.5 min** |
| Feature store size | 16x1280 float32 = 80 KB/clip, **0.35 GB** total |

So: **freeze the CNN, precompute features once, train only the recurrent head.** Remaining compute
drops from about 56 h to under 2 h. Optuna becomes a genuine 30 trial Bayesian search instead of a
token one.

This is not a shortcut away from the report. Report §2.2 states transfer learning from a pretrained
CNN into a recurrent head is the dominant strategy in the literature, citing Mumtaz et al.
(VGG-19 + LSTM), Traore and Akhloufi (VGG16 + BiGRU), and Imah et al. (ResNet50v2 + GRU). The
TimeDistributed CNN-LSTM of §4.3 is still built and still reported, as the trained-from-scratch
comparison arm.

---

## 2. What is cut, and what is kept

Be straight about this in the report. Cut scope that is written up as if delivered is the one thing
that turns a good submission into a dishonest one.

### Kept in full
Multi-source datasets, leakage-safe grouped splits, clip-consistent Albumentations augmentation,
TimeDistributed CNN-LSTM architecture, class-weighted BCE, MLflow, TensorBoard with activation
histograms, Optuna Bayesian HPO, accuracy / precision / recall / F1 / ROC-AUC with bootstrap CIs,
cross-dataset evaluation, AIRTLab women-specific fine-tuning, ONNX export with parity check,
sliding-window real-time inference, person detection plus gender attribution, the Flask app.

### Cut, with the reason
| Cut | Why | Where to say so |
|---|---|---|
| End-to-end CNN training as the production model | 4.4 h per run against 2 min for the frozen-feature head | §2 of the report's methodology, and `docs/RESULTS.md` |
| Frame-level AUC on untrimmed video | XD-Violence is gone (38.3 GB, see ADR-002) and the UCF-Crime subset holds only 35 clips total, too few for a meaningful frame-level AUC | `docs/RESULTS.md` limitations |
| PA-100K gender classifier training | 100k images is hours. Replaced by a zero-shot CLIP gender head, which is weaker and must be labelled as such | `docs/ETHICS.md` and the model card |
| Docker image and fresh-clone reproduction | Not needed for a submission; `make all` smoke path covers it | `docs/REPRODUCE.md` |
| Knowledge distillation, audio, pose | Never in scope, listed as future work | report §5 |

### Scaled down
- Scratch CNN-LSTM: **one 12 epoch run** for the comparison arm, not a tuned 30 epoch run.
- Optuna: 30 trials on the head (cheap and real), not a search over backbone freeze depth.
- AIRTLab fine-tune: 5-fold grouped CV on the head only.

---

## 3. Sprints

Six sprints, each one Sonnet session. Same git conventions as before (`CLAUDE.md`), with
`sprint(N):` in place of `phase(N):`. **If you fall behind, drop the issue and PR ceremony and
commit straight to `main` with the same message rules, and record that you did.**

| # | Sprint | Agent | Compute | Replaces |
|---|---|---|---|---|
| 1 | Feature store, models, training | 2.5 h | ~20 min | Phase 4 |
| 2 | Evaluation harness | 2 h | ~10 min | Phase 5 |
| 3 | Optuna HPO | 1.5 h | ~30 min | Phase 6 |
| 4 | AIRTLab fine-tune + attribution | 2.5 h | ~15 min | Phases 7, 8 |
| 5 | ONNX, real-time, Flask app | 3 h | ~10 min | Phases 9, 10 |
| 6 | Notebooks, results, model card | 3.5 h | ~20 min | Phase 11 |
| | **Total** | **~15 h** | **~1.75 h** | |

---

## Sprint 1 — Feature store, models, training

**Preconditions:** Phase 3 `DONE`; `data/cache/*.h5` present; `pytest -m "phase0 or phase1 or phase2 or phase3"` green.

**Deliverables**
- `safestreets/features/extract.py` + `scripts/extract_features.py --dataset all --split all`
- `data/features/{dataset}_{split}.h5` holding `features (N,16,1280) float32`, `labels`, `clip_ids`, and root attrs `{backbone, weights, pooling, n_frames, config_sha}`
- `safestreets/models/heads.py` — `build_lstm_head(cfg)`: `LSTM(units, return_sequences) -> LSTM(units//2) -> Dropout -> Dense(1, sigmoid)`
- `safestreets/models/cnn_lstm.py` — `build_scratch_cnn_lstm(cfg)`, the §4.3 architecture, for the comparison arm
- `safestreets/models/factory.py` — dispatch on `cfg.model.arch` in `{lstm_head, scratch}`
- `safestreets/training/losses.py` — `weighted_bce(pos_weight)` from the train manifest
- `safestreets/training/callbacks.py` — MLflow, TensorBoard (`histogram_freq=1`), checkpoint, early stop on `val_auc`
- `safestreets/training/train.py` + `scripts/train.py --config configs/train.yaml`
- `configs/model.yaml`, `configs/train.yaml`
- `artifacts/checkpoints/`, `mlruns/`, `logs/fit/`
- `docs/decisions/ADR-003-architecture.md` — records the frozen-feature decision with the measured numbers from §1
- `tests/test_sprint1_model.py` (marker `sprint1`)

**Notes**
- Augmentation cannot be applied after feature extraction. Two honest options, pick one and record
  it in ADR-003: (a) extract features from **augmented** clips for N augmented epochs' worth and
  treat it as offline augmentation, or (b) train the head on clean features and report that
  augmentation was validated in Phase 3 but not used for the production head. **(b) is the
  recommended choice under this deadline** because it is simpler and defensible; the scratch arm
  does use the live augmented pipeline, so the augmentation code is still exercised end to end.
- Extract with `trainable=False` and **no** augmentation, eval transform only, for determinism.
- Feature extraction must be idempotent, same `.complete` sentinel pattern as Phase 2.
- `normalize: symmetric` is required for MobileNetV2. `configs/data.yaml` currently says
  `unit_range`. Either switch it for extraction or apply `preprocess_input` inside the extractor.
  **Assert the choice in a test** so it cannot silently drift.

**Definition of Done**
- [ ] Feature store built for all 11 `(dataset, split)` pairs; `N_features == N_cached` per file.
- [ ] `data/features` total size under 0.6 GB; `df -h ~` still shows 20 GB or more free.
- [ ] Extraction is idempotent; a second run skips completed files in under 5 s.
- [ ] Normalisation assertion test passes: features come from `preprocess_input`-scaled input, verified by comparing against a hand-computed single-clip forward pass.
- [ ] `build_model(cfg)` compiles for both `lstm_head` and `scratch`; summaries saved to `artifacts/model_summary_{arch}.txt`.
- [ ] Overfit test: the head reaches 0.95 or better train accuracy on a 32 clip subset within 50 epochs.
- [ ] Label alignment test: a batch's labels match `clips.parquet` for those exact `clip_id`s.
- [ ] Production run: head trained on RWF+RLVS train features, early stopped on `val_auc`, **`val_auc > 0.80`**. If unmet, record the real number and stop; do not proceed to Sprint 3.
- [ ] Comparison run: scratch CNN-LSTM, 12 epochs, through the live augmented Phase 3 pipeline. Its `val_auc` is recorded for the report whatever it is.
- [ ] MLflow has both runs with params, per-epoch metrics, and model artefacts.
- [ ] TensorBoard shows loss, accuracy, and weight histograms.
- [ ] `pytest -m "phase0 or phase1 or phase2 or phase3 or sprint1"` green.

**Exit Gate:** `make verify SPRINT=1`

---

## Sprint 2 — Evaluation harness

**Preconditions:** Sprint 1 `DONE`, a checkpoint meeting `val_auc > 0.80`.

**Deliverables**
- `safestreets/evaluation/metrics.py` — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, bootstrap 95% CI (1000 resamples)
- `safestreets/evaluation/evaluate.py`, `safestreets/evaluation/cross_dataset.py`
- `scripts/evaluate.py --checkpoint ... --splits ...`
- `artifacts/reports/eval_{tag}.json`, `artifacts/figures/{roc,pr,confusion,threshold_sweep}.png`
- `docs/RESULTS.md` — **generated**, never hand-typed
- `tests/test_sprint2_eval.py`

**Notes**
- Cross-dataset matrix: train on RWF+RLVS, test zero-shot on AIRTLab and the UCF-Crime subset.
  Report the drop. **No drop at all means suspecting leakage, not celebrating.**
- RWF-2000 contributes no `test` rows (ADR-002). The in-domain test split is RLVS test plus AIRTLab
  test plus UCF test. State that explicitly in `RESULTS.md` so the numbers are not misread.
- Threshold chosen on **validation**, written to `configs/infer.yaml`. Report both the F1-optimal
  point and the recall >= 0.90 point, the latter being the deployment-relevant one for a safety
  system.
- Frame-level AUC is **out of scope**, see §2. Write that in the limitations section rather than
  omitting it silently.

**Definition of Done**
- [ ] Metrics agree with `sklearn` to 1e-9 on synthetic data.
- [ ] `eval_*.json` carries every metric plus bootstrap CIs for every split evaluated.
- [ ] Cross-dataset matrix covers 1 train set x 3 test sets minimum, rendered as a table.
- [ ] Threshold sweep figure exists; selected threshold persisted to `configs/infer.yaml`; a test asserts it was selected on validation, not test.
- [ ] Four figures render and are referenced from `docs/RESULTS.md`.
- [ ] `docs/RESULTS.md` regenerates with an empty `git diff`.
- [ ] Every number in `docs/RESULTS.md` traces to a key in a committed `eval_*.json`.
- [ ] Limitations section names the dropped frame-level AUC and the AIRTLab synthetic-data caveat.
- [ ] `pytest -m "... or sprint2"` green.

**Exit Gate:** `make verify SPRINT=2`

---

## Sprint 3 — Optuna HPO

**Preconditions:** Sprint 2 `DONE`.

**Deliverables**
- `safestreets/training/tune.py` — `TPESampler`, `MedianPruner`, SQLite storage
- `scripts/tune.py --trials 30 --timeout 2400 [--resume]`
- `artifacts/optuna/study.db`, `artifacts/figures/optuna_{history,importances}.png`
- `configs/model.best.yaml`, `artifacts/checkpoints/best_*.keras`
- `docs/decisions/ADR-004-hpo.md`
- `tests/test_sprint3_tune.py`

**Search space** (report §4.4 names these four): `learning_rate` log-uniform 1e-5 to 1e-2;
`lstm_units` in {64,128,256,512}; `dropout` 0.2 to 0.6; `batch_size` in {16,32,64}.
Objective: maximise `val_auc`, one objective only.

**Definition of Done**
- [ ] 30 trials complete; measured wall time recorded in ADR-004.
- [ ] At least one trial state is `PRUNED`.
- [ ] `study.db` resumes: interrupt, re-run with `--resume`, trial count continues.
- [ ] `configs/model.best.yaml` written programmatically; loading it reproduces the best trial's architecture.
- [ ] Both Optuna figures render; importance plot interpreted in one paragraph in ADR-004.
- [ ] Winner retrained on train+val, test metrics regenerated through Sprint 2's harness, **test touched once**.
- [ ] If tuning did not beat the baseline, the baseline ships and that is reported. No cherry-picking.
- [ ] `docs/RESULTS.md` regenerated with the tuned row beside the baseline.
- [ ] `pytest -m "... or sprint3"` green.

**Exit Gate:** `make verify SPRINT=3`

---

## Sprint 4 — AIRTLab fine-tune and attribution

**Preconditions:** Sprint 3 `DONE`.

**Deliverables**
- `safestreets/training/finetune.py`, `scripts/finetune.py`
- `artifacts/checkpoints/finetuned_airtlab_*.keras`, `artifacts/reports/eval_finetuned.json`
- `safestreets/attribution/person.py` — ultralytics YOLO, person class, every 4th frame, ByteTrack
- `safestreets/attribution/gender.py` — **zero-shot CLIP** head over person crops, no training
- `safestreets/attribution/pipeline.py` — fuses violence score and attribution into an alert
- `configs/attribution.yaml`, `artifacts/reports/attribution_eval.json`
- `docs/ETHICS.md`, `docs/decisions/ADR-005-finetune-and-attribution.md`
- `tests/test_sprint4_attribution.py`

**Notes**
- AIRTLab has 242 train / 60 val / 48 test clips, grouped by (label, clip_number) per ADR-002. It is
  small and acted. 5-fold grouped CV gives an error bar instead of one fragile number.
- **Measure catastrophic forgetting**: re-evaluate on RWF-2000 val after fine-tuning and report both
  numbers side by side.
- Gender is zero-shot CLIP, so its accuracy is unknown until measured and will be mediocre. Measure
  it on AIRTLab person crops if any labels can be derived; if not, report it as **unmeasured** and
  say so plainly. Do not invent a number.
- **The ethical constraint is a correctness property, not a doc paragraph:** a high violence score
  raises an alert regardless of the gender module's output. Attribution changes an alert's label and
  priority, never its existence.

**Definition of Done**
- [ ] 5-fold grouped CV on AIRTLab reports mean +/- std for accuracy, F1, ROC-AUC.
- [ ] Forgetting check: RWF-2000 val AUC before and after fine-tuning, both in `RESULTS.md`.
- [ ] A recorded decision in ADR-005 about which checkpoint ships.
- [ ] YOLO person detection runs on cached clips; detections/frame sane on a hand-checked 20 clip sample.
- [ ] Tracking gives stable IDs: a single-walker clip yields one dominant track, not 16.
- [ ] **Fail-safe test passes:** `p_violence = 0.99` with the gender module forced to "no women detected" still raises an alert.
- [ ] Graceful degradation: a missing YOLO or CLIP weight logs a warning and falls back to violence-only alerts without crashing.
- [ ] `docs/ETHICS.md` written, naming the zero-shot limitation and the measured or explicitly unmeasured accuracy.
- [ ] `pytest -m "... or sprint4"` green.

**Exit Gate:** `make verify SPRINT=4`

---

## Sprint 5 — ONNX, real-time, Flask app

**Preconditions:** Sprint 4 `DONE`. ADR-001's ONNX recipe still valid (parity was already proven at 5.96e-08).

**Deliverables**
- `safestreets/inference/export_onnx.py`, `scripts/export_onnx.py`
- `artifacts/onnx/safestreets_{tag}.onnx` plus a sidecar `.json` (input spec, normalisation, threshold, git SHA, source checkpoint)
- `safestreets/inference/engine.py` — `InferenceEngine` over `onnxruntime`, bundling MobileNetV2 features plus the head
- `safestreets/inference/stream.py` — 16 frame ring buffer, stride 8, EMA smoothing, two-threshold hysteresis
- `artifacts/reports/{latency,onnx_parity}.json`
- `safestreets/web/` — app factory, routes, api, templates, static; `app/` and `test/test_predict.py` **deleted**
- `tests/test_sprint5_inference.py`, `tests/test_sprint5_web.py`

**Fix these named legacy defects**
1. `app/routes.py` imports `predict_video`, which does not exist. Replace with `InferenceEngine`.
2. `templates/index.html` discards `data.prediction`. Render score, verdict, and the per-window timeline.
3. Uploads have no size cap, no content-type check, no cleanup. Add `MAX_CONTENT_LENGTH`, verify decodability with OpenCV, use a UUID temp path, delete after.
4. `test/test_predict.py` normalises by `/2300.0`. Delete the file.
5. Tailwind loads from `unpkg.com`. Vendor it into `static/`.

**Definition of Done**
- [ ] ONNX export succeeds; loads in `onnxruntime`; max abs diff against Keras under 1e-4 over 50 or more real clips, written to `onnx_parity.json`.
- [ ] ONNX-path ROC-AUC matches the Keras path within 0.005.
- [ ] Latency p50/p95/p99 per window recorded in `latency.json` with the implied sustainable FPS.
- [ ] Hysteresis test: a score series oscillating around the threshold produces **one** alert, not many.
- [ ] Sidecar JSON complete; the engine refuses to run if its normalisation spec disagrees with the active config.
- [ ] `GET /` 200s; `GET /healthz` returns model version and git SHA.
- [ ] `POST /api/analyse` with `test/sample_video.avi` returns a schema-valid body with `clip_score` in [0,1].
- [ ] One 4xx test each for: no file, wrong extension, oversized, non-decodable.
- [ ] Uploaded files removed from disk after analysis, asserted in a test.
- [ ] Result page renders verdict and score, asserted by matching response HTML.
- [ ] `grep -r "from app" .` returns nothing; no external CDN needed to render.
- [ ] `pytest -m "... or sprint5"` green.

**Exit Gate:** `make verify SPRINT=5`

---

## Sprint 6 — Notebooks, results, model card

**This is the sprint that produces what gets submitted.** Budget for it; do not let it be the one
that gets rushed.

**Preconditions:** Sprint 5 `DONE`.

**Deliverables — six executed notebooks in `notebooks/`**

Every notebook is **thin**: it imports from `safestreets/`, calls it, and displays the result. No
copy-pasted implementation in cells. That is both faster and far more presentable, because it shows
a real package rather than a scratchpad.

| Notebook | Covers | Pulls from |
|---|---|---|
| `01_data_and_preprocessing.ipynb` | the five sources, licences, the manifest, **why splits are grouped**, leakage assertion, the clip cache, contact sheet | Phases 1, 2 |
| `02_augmentation.ipynb` | clip-consistent Albumentations, the identical-frames proof, before/after grid, throughput | Phase 3 |
| `03_model_and_training.ipynb` | both architectures with summaries, the frozen-feature decision **with its measured justification**, class weighting, training curves, TensorBoard histograms, MLflow | Sprint 1 |
| `04_evaluation.ipynb` | all metrics with CIs, ROC/PR/confusion, threshold sweep, cross-dataset table and the drop | Sprint 2 |
| `05_hpo_and_finetune.ipynb` | Optuna history and importances, best params, AIRTLab 5-fold CV, forgetting check | Sprints 3, 4 |
| `06_inference_and_demo.ipynb` | attribution pipeline with the fail-safe, ONNX parity and latency, sliding-window scoring on a real video, Flask screenshots | Sprints 4, 5 |

Each notebook opens with a markdown cell stating what it demonstrates and which report section it
maps to, and closes with the honest limitations of what it just showed.

**Also**
- `docs/MODEL_CARD.md` — intended use, out-of-scope use, training data and licences, metrics, limitations, ethics
- `docs/RESULTS.md` — final regeneration
- `docs/REPRODUCE.md` — exact commands in order
- `README.md` — full rewrite with the results table, quickstart, architecture figure
- `artifacts/figures/architecture.png`
- `docs/PRESENTATION_GUIDE.md` — a walkthrough order for presenting the six notebooks, what to say per notebook, the three strongest results, and the three limitations to volunteer before being asked

**Definition of Done**
- [ ] All six notebooks execute top to bottom from a clean kernel with **outputs saved**, via `jupyter nbconvert --execute --inplace`. A notebook with stale or missing outputs is not presentable.
- [ ] No notebook contains a copy-pasted reimplementation of package code.
- [ ] No notebook takes more than 5 minutes to execute, so it can be re-run live.
- [ ] Every number displayed in a notebook comes from a committed artefact or a live call, never a literal.
- [ ] `README.md`'s results table matches `docs/RESULTS.md` exactly; both generated.
- [ ] Model card covers the AIRTLab synthetic-data limitation and the zero-shot gender module's weakness.
- [ ] Every row of `BUILD_PLAN.md` §2's requirements table is marked delivered, scaled down, or cut, with a pointer to where it is covered or why it is not. Recorded in `PROGRESS.md`.
- [ ] `docs/PRESENTATION_GUIDE.md` written.
- [ ] `pytest` all markers green.

**Exit Gate:** `make verify SPRINT=6 && make verify-all`

---

## 4. Risks on this timeline

| Risk | Mitigation |
|---|---|
| `val_auc` below 0.80 in Sprint 1 | The frozen-feature head is the highest-signal cheap option already. If it underperforms, try unfreezing the top MobileNetV2 block (still cheap) before anything expensive. Budget 45 min, then ship the real number and say so. |
| Sprint 6 gets squeezed | It is the submission. If time runs short, cut Sprint 4's attribution to detection-only and Sprint 5's app to the upload path, never the notebooks. |
| Notebook execution breaks late | Run `nbconvert --execute` at the **end of every sprint** from Sprint 1 on, not only in Sprint 6. |
| CLIP or YOLO weights fail to download | Both degrade gracefully by design. Report the module as unmeasured rather than faking it. |
| Fabricated numbers under time pressure | The one unbreakable rule. `NOT RUN` is always an acceptable answer; an invented metric never is. |
