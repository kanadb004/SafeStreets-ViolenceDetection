# SafeStreets — Master Build Plan

**Target:** the system described in `DA1_Report_23BCE1265.pdf` — *"SafeStreets: A CNN-LSTM Deep
Learning Pipeline for Real-Time Violence Detection Against Women Trained on Multi-Source
Benchmark Datasets"*.

**Audience:** a Claude Sonnet session picking up the project cold, one phase per session.

---

## 0. How to use this document

**Every session, in this order:**

1. Read `docs/PROGRESS.md` → find the first phase whose status is not `DONE`. That is your phase.
2. Read **only** that phase's section below, plus `docs/decisions/` (ADRs — binding decisions).
3. Re-read the "Preconditions" block. If a precondition fails, **stop and fix the earlier phase**;
   do not work around it.
4. Build. Stay inside the phase's declared deliverables — do not start the next phase's work.
5. Run the phase's **Exit Gate** command. It must exit 0.
6. Update `docs/PROGRESS.md`: set status, fill the evidence block, append notes/surprises.
7. Commit with `phase(N): <summary>`.

**Rules that apply to every phase:**

- **No fabricated numbers, ever.** Every metric written into a doc, README, or report must be
  copy-pasted from a file produced by a command you actually ran. If a run did not happen, write
  `NOT RUN`. This is the single most important rule in this document.
- **No silent scope changes.** If the plan says something impossible, record it in
  `docs/decisions/ADR-XXX-*.md`, update this plan, and say so in PROGRESS.md.
- **Config over constants.** Anything a reviewer might want to change (frame count, resolution,
  paths, epochs) lives in `configs/*.yaml`, not in a `.py` literal.
- **Determinism.** Every script takes `--seed` (default 1265) and seeds `random`, `numpy`, and
  the framework RNG. Splits are derived from stable hashes of clip IDs, never from shuffle order.
- **Every phase ends green.** `pytest -m "phase0 or ... or phaseN"` (all phases up to yours) must
  pass, not just your own phase's tests. No phase may break an earlier phase's gate.

---

## 1. Where the repo is today (verified 2026-09-13, commit `5ce07c4`)

| Area | State | Verdict |
|---|---|---|
| `app/` Flask app | 3 static pages + `/upload` endpoint | Keep the shape, rewrite |
| `app/routes.py` | imports `predict_video` from `app.ml_model.model` | **Broken — that function does not exist.** The app cannot serve a request. |
| `app/ml_model/model.py` | 1 conv block → LSTM(64) → Dropout → Dense(1) | Below spec (§4.3 wants 2–3 blocks + stacked LSTM) |
| Training data pipeline | `pad_sequences` over **whole videos**, `batch_size=1` | Unusable — pads every clip to the longest video in the set |
| `test/test_predict.py` | a script, not a test; normalises by `/2300.0` | Bug, and not collectible by pytest |
| `train/train_model.ipynb` | Colab notebook, hardcoded `/content/drive` paths | Reference only; superseded by `safestreets/training/` |
| `templates/index.html` | uploads, then **discards `data.prediction`** | UI never shows a result |
| `requirements.txt` | `tensorflow==2.12.0`, `Flask==2.3.2` | Will not install on Python 3.12 / macOS ARM |
| `Dockerfile` | `CMD ["python3", "violence_detection_test.py"]` | File does not exist |
| `SafeStreets/`, `footages/` | empty | `SafeStreets/` to be deleted |
| Datasets, trained weights, ONNX, MLflow, Optuna, metrics, CI | **absent** | All net-new |

**Honest summary:** roughly 5% of the report's system exists. The Flask skeleton and the *idea* of
a TimeDistributed CNN-LSTM are the only salvage. Everything else is built from zero.

---

## 2. Target system (extracted from the report — this is the contract)

| Report § | Requirement | Phase |
|---|---|---|
| §4.1 | RWF-2000 + RLVS as primary binary sources | 1, 2 |
| §4.1 | AIRTLab for women-specific fine-tuning **and** evaluation | 1, 7 |
| §4.1 | UCF-Crime + XD-Violence for crime-category breadth + frame-level eval | 1, 5 |
| §4.2 | OpenCV frame extraction, resize, normalise, temporal padding/sampling | 2 |
| §4.2 | Albumentations, applied **consistently across frames within a clip** | 3 |
| §4.3 | TimeDistributed CNN-LSTM; 2–3 conv blocks + pooling; stacked LSTM; dropout after LSTM | 4 |
| §4.3 | Class-weighted binary cross-entropy | 4 |
| §4.4 | Optuna Bayesian search over lr, LSTM units, dropout, batch size | 6 |
| §4.4 | MLflow: metrics, params, artefacts | 4 |
| §4.4 | TensorBoard: loss curves, per-epoch accuracy, **activation histograms** | 4 |
| §2.10 | accuracy, precision, recall, F1, ROC-AUC on held-out **and cross-dataset** splits | 5 |
| §1 | ONNX export for efficient inference | 9 |
| §2.7, §5 | SUSAN-style: general violence detector **+** person detection **+** gender module | 8 |
| §1, §2.8 | Real-time CCTV input → binary classification → alert | 9, 10 |

---

## 3. Two constraints that shape everything

### 3.1 Disk: 44 GB free

Raw UCF-Crime is ~128 h of video (**>100 GB**); raw XD-Violence is comparable. **They do not fit.**
This is not negotiable and the plan is designed around it:

| Source | Strategy | On-disk |
|---|---|---|
| RWF-2000 | full download | ~3 GB raw → ~1.2 GB cached |
| RLVS | full download | ~2 GB raw → ~1.2 GB cached |
| AIRTLab | full download | ~3 GB raw → ~0.2 GB cached |
| UCF-Crime | **subset**: `Abuse`, `Assault`, `Fighting`, `Robbery` + matched `Normal`; stream-and-cache, delete raw per file | ~8 GB peak → ~0.8 GB cached |
| XD-Violence | **official pre-extracted I3D features only** (no raw video) for frame-level AUC | ~4 GB |

Total steady-state ≈ 12 GB. Phase 2 caches clips as `uint8` tensors and **deletes raw video as it
goes**, which is what makes this fit. If you ever find yourself downloading raw UCF-Crime or raw
XD-Violence in full, you have taken a wrong turn.

### 3.2 Compute: Apple M2, 16 GB unified memory, no CUDA

- Baseline training (Phase 4) is expected to be **hours, not minutes**, locally. That is fine.
- Full Optuna studies (Phase 6) and any heavy fine-tune are the phases most likely to need
  Google Colab. Every training entry point therefore **must** run unmodified from a Colab
  notebook: no absolute paths, all paths from config, `--data-root` overridable by env var.
- Phase 0 records which accelerator path actually works (`tensorflow-metal` or CPU) in an ADR.
  Do not assume Metal works; **measure it**.

---

## 4. Target repository layout

Build toward this. Create directories only in the phase that needs them.

```
configs/                         # YAML — all tunables
  data.yaml  model.yaml  train.yaml  tune.yaml  infer.yaml
safestreets/                     # installable package (replaces app/)
  config.py                      # typed config loader
  data/  download.py  manifest.py  preprocess.py  augment.py  dataset.py
  models/  cnn_lstm.py  backbones.py  factory.py
  training/  train.py  losses.py  callbacks.py  tune.py
  evaluation/  metrics.py  evaluate.py  cross_dataset.py  temporal.py
  inference/  export_onnx.py  engine.py  stream.py
  attribution/  person.py  gender.py  pipeline.py
  web/  __init__.py  routes.py  api.py  templates/  static/
scripts/                         # thin CLI wrappers, argparse only
tests/                           # pytest, one file per phase, markers phase0..phase11
data/                            # gitignored
  raw/  cache/  features/  manifests/
artifacts/                       # gitignored — checkpoints, onnx, reports, figures
mlruns/  logs/                   # gitignored
docs/  BUILD_PLAN.md  PROGRESS.md  decisions/  RESULTS.md  MODEL_CARD.md  ETHICS.md
```

---

## 5. Phase index

| # | Phase | Session size | Blocks |
|---|---|---|---|
| 0 | Foundation: env, packaging, config, CI, test harness | 1 | everything |
| 1 | Dataset acquisition + manifests + leakage-safe splits | 1–2 | 2 |
| 2 | Preprocessing → compact clip cache | 1 | 3, 4 |
| 3 | Augmentation + `tf.data` input pipeline | 1 | 4 |
| 4 | Model + training loop + MLflow + TensorBoard (baseline) | 1–2 | 5, 6 |
| 5 | Evaluation harness: multi-metric, cross-dataset, frame-level | 1 | 6, 7 |
| 6 | Optuna HPO + final model selection | 1–2 | 7, 9 |
| 7 | AIRTLab women-specific fine-tuning | 1 | 8 |
| 8 | Person detection + gender attribution (SUSAN-style) | 1–2 | 10 |
| 9 | ONNX export + real-time streaming inference engine | 1 | 10 |
| 10 | Flask application rebuild (upload, live, alerts, results UI) | 1–2 | 11 |
| 11 | Packaging, Docker, reproducibility, results & model card | 1 | — |

---

# Phase 0 — Foundation

**Goal:** a reproducible environment, an installable package, typed config, a working test harness,
and CI — so that every later phase has somewhere to land and something to verify against.

### Preconditions
None. This is the entry point.

### Deliverables
- `pyproject.toml` — package `safestreets`, deps, pytest config with `phase0..phase11` markers.
- `.python-version` → `3.12` (TF has no macOS ARM wheel for 3.13/3.14; system default is 3.14.5).
- `configs/*.yaml` + `safestreets/config.py` (typed loader; env-var override `SS_DATA_ROOT` etc.).
- `safestreets/__init__.py`, `safestreets/utils/seed.py` (`set_global_seed(seed)`).
- `tests/conftest.py`, `tests/test_phase00_foundation.py`.
- `.github/workflows/ci.yml` — lint + `pytest -m "not slow and not needs_data"`.
- `Makefile` with `setup`, `test`, `lint`, `verify` targets.
- `CLAUDE.md` at repo root (session bootstrap — see template in §6 of this plan).
- `docs/decisions/ADR-001-runtime-stack.md` — **the spike below, recorded**.
- Deleted: empty `SafeStreets/` dir. Moved: `train/train_model.ipynb` → `notebooks/legacy/`.

### The spike (do this first — it constrains Phases 4 and 9)
TensorFlow/Keras 3 and `tf2onnx` have a known compatibility seam. Resolve it **now**, before any
model code is written, by empirically testing in this order and recording the winner in ADR-001:

1. Current TF + `tf2onnx` from a `SavedModel` export.
2. TF 2.16/2.17 with `TF_USE_LEGACY_KERAS=1` + the `tf_keras` package + `tf2onnx`.
3. Keras 3 native `model.export()` → `onnx` path.

Test with a 30-second throwaway script: build a tiny TimeDistributed(Conv2D)+LSTM model, export to
ONNX, load in `onnxruntime`, assert outputs match Keras within `1e-4`. **Whichever works, pin those
exact versions in `pyproject.toml` and write them into ADR-001.** Also record in ADR-001 whether
`tensorflow-metal` actually accelerates or crashes — measure, don't assume.

### Implementation notes
- Keep `app/` in place and untouched this phase — Phase 10 replaces it. Do not half-migrate.
- `pytest` markers must be declared in `pyproject.toml` (`--strict-markers`) so a typo in a marker
  fails loudly instead of silently skipping a phase's tests.
- Custom markers to declare: `phase0`–`phase11`, plus `slow` and `needs_data` so CI can skip
  anything requiring downloaded datasets or a GPU.

### Definition of Done
- [ ] `python -c "import safestreets; print(safestreets.__version__)"` succeeds in a clean venv created from `pyproject.toml` alone.
- [ ] `safestreets.config.load_config()` returns a typed object; changing `configs/data.yaml` changes the returned value; `SS_DATA_ROOT=/tmp/x` overrides it.
- [ ] `set_global_seed(1265)` makes two successive `numpy` + framework RNG draws identical across processes.
- [ ] `pytest -m phase0` passes with ≥ 4 real assertions (not `assert True`).
- [ ] `pytest --strict-markers -m phase7` runs and reports "no tests ran" — not an unknown-marker error.
- [ ] ADR-001 exists and names **exact pinned versions** for `tensorflow`, `keras`/`tf_keras`, `tf2onnx`, `onnxruntime`, with a pasted transcript of the parity check.
- [ ] `make lint` (ruff) exits 0 across `safestreets/` and `tests/`.
- [ ] CI workflow green on a pushed branch.
- [ ] `SafeStreets/` directory is gone; `git status` is clean.

### Exit Gate
```bash
make verify PHASE=0
# equivalent to: ruff check . && pytest -m phase0 -q && test -f docs/decisions/ADR-001-runtime-stack.md
```

---

# Phase 1 — Datasets, manifests, leakage-safe splits

**Goal:** every clip the project will ever use is downloaded (or feature-extracted), inventoried in
one manifest, and assigned to a split that **cannot leak**.

### Preconditions
- Phase 0 `DONE`; `make verify PHASE=0` passes.
- ≥ 25 GB free disk (`df -h ~`). If not, stop and tell the user.

### Deliverables
- `safestreets/data/download.py` — one `fetch_<dataset>()` per source, each **idempotent**
  (checks a `.complete` sentinel + file count before re-downloading) and **resumable**.
- `scripts/fetch_data.py --dataset {rwf2000,rlvs,airtlab,ucfcrime,xdviolence} [--dry-run]`
- `safestreets/data/manifest.py` — builds `data/manifests/clips.parquet`.
- `data/manifests/clips.parquet` with **exactly** these columns:
  `clip_id, dataset, path, label, split, group_id, n_frames, fps, width, height, duration_s, source_video, category`
- `data/manifests/splits.json` — counts per (dataset × split × label).
- `docs/decisions/ADR-002-dataset-splits.md` — the grouping key chosen per dataset, and why.
- `tests/test_phase01_manifest.py`
- `docs/DATASETS.md` — provenance, licence, citation, and access method per dataset.

### The thing most likely to be done wrong: **split leakage**
RLVS and AIRTLab clips are cut from a smaller number of source videos/scenes/actors. A random
per-clip split puts sibling clips in both train and test and **inflates accuracy by 10–20 points**.
Every reported number in this project would then be worthless.

**Mandatory rule:** splits are assigned by `group_id`, never by `clip_id`.

| Dataset | `group_id` derivation |
|---|---|
| RWF-2000 | use the **official** train/val directory split verbatim; `group_id = clip_id` |
| RLVS | parse the source-video stem from the filename; cluster near-duplicate stems |
| AIRTLab | scene/camera/actor prefix in the filename |
| UCF-Crime | the untrimmed source video id |
| XD-Violence | the untrimmed source video id |

Assignment is `hash(group_id) % 100` bucketed into 70/15/15 — deterministic, order-independent, and
re-runnable without reshuffling. **Never** `sklearn.train_test_split` on clip rows.

### Implementation notes
- RWF-2000 requires an author access request; RLVS/AIRTLab are on Kaggle. If a download needs
  credentials or a manual form, **do not fake it and do not silently skip**: write the exact steps
  into `docs/DATASETS.md`, mark that dataset `BLOCKED` in PROGRESS.md, and continue with the rest.
- UCF-Crime: download only the four categories named in §3.1 plus a size-matched `Normal` sample.
  Stream each file, cache it (Phase 2 runs after, so for now just fetch), and record it.
- XD-Violence: fetch the **released I3D features**, not video. These feed Phase 5's frame-level
  AUC only — they never enter the CNN-LSTM training loop.
- Record real `n_frames`/`fps` by opening each file with OpenCV. Do not trust filenames.
- Add `data/`, `artifacts/`, `mlruns/`, `logs/` to `.gitignore`.

### Definition of Done
- [ ] `python scripts/fetch_data.py --dataset rwf2000` run twice: the second run re-downloads **nothing** and exits 0 in < 5 s.
- [ ] `clips.parquet` exists; `len(df) >= 4000`; zero null `label`; zero duplicate `clip_id`.
- [ ] **Leakage assertion passes:** `set(train.group_id) & set(val.group_id) & set(test.group_id) == ∅` — asserted in `tests/test_phase01_manifest.py`, not just checked by eye.
- [ ] Split proportions are 70/15/15 ± 3 pp per dataset.
- [ ] Label balance per dataset recorded in `splits.json` and reproduced in `docs/DATASETS.md`.
- [ ] Re-running manifest construction produces a **byte-identical** `splits.json`.
- [ ] Every row's `path` exists on disk (assert `df.path.map(os.path.exists).all()`).
- [ ] `docs/DATASETS.md` names licence + citation for all five sources; any blocked source is explicitly marked `BLOCKED` with the manual steps needed.
- [ ] `pytest -m "phase0 or phase1"` passes.

### Exit Gate
```bash
make verify PHASE=1
```

---

# Phase 2 — Preprocessing → compact clip cache

**Goal:** turn thousands of video files into a fixed-shape `uint8` tensor cache that trains fast and
fits on disk. This phase is what makes the whole project viable on this machine.

### Preconditions
- Phase 1 `DONE`, `clips.parquet` populated and leakage-free.

### Deliverables
- `safestreets/data/preprocess.py`
- `scripts/build_cache.py --dataset X --split Y [--workers N] [--purge-raw]`
- `data/cache/{dataset}_{split}.h5` — datasets `clips (N,T,H,W,3) uint8`, `labels (N,) uint8`,
  `clip_ids (N,) str`, and root attrs `{n_frames, height, width, sampling, created_utc, config_sha}`
- `data/cache/cache_report.json` — per-dataset: clips written, clips skipped + reason, bytes, wall time.
- `tests/test_phase02_preprocess.py`
- `artifacts/figures/sample_clips.png` — a 4×T contact sheet, eyeballed once for sanity.

### Spec (locked; change only via ADR)
- `T = 16` frames, **uniformly sampled** across the clip's full duration (not the first 16 frames —
  violence is rarely in the first half-second).
- Resize to `112 × 112`, **letterboxed** (preserve aspect ratio, pad) — naive squashing distorts
  human figures and hurts the gender module in Phase 8.
- Colour: BGR → **RGB** at read time. OpenCV gives BGR; every downstream consumer (Albumentations,
  ImageNet-pretrained backbones, ONNX) expects RGB. Getting this wrong silently costs accuracy.
- Store `uint8` 0–255. Normalisation happens in the graph (Phase 3), never on disk — this is a 4×
  disk saving over `float32` and keeps augmentation exact.
- Clips with `< T` real frames: loop-pad (repeat the sequence) rather than zero-pad, and flag them
  in the manifest as `padded=True`.
- Unreadable/zero-frame files: **skip and log**, never write a zeros clip.

### Implementation notes
- Per-clip cost ≈ `16 × 112 × 112 × 3` = 602 KB. ~4 000 clips ≈ 2.4 GB. Confirm you land near this;
  if you are an order of magnitude off, your shape or dtype is wrong.
- Parallelise with `concurrent.futures.ProcessPoolExecutor`; OpenCV `VideoCapture` is not
  thread-safe, so use **processes**, not threads.
- HDF5 writes must be from a single writer process collecting results from the pool.
- `--purge-raw` deletes a source file only **after** its clip is committed and re-read successfully.
  Default it to **off**; require the operator to pass it. Print total bytes to be freed first.
- Stamp `config_sha` (hash of the preprocessing config) into the H5 attrs. Phase 3 asserts it
  matches the active config, so a stale cache can never silently train a model.

### Definition of Done
- [ ] Every `(dataset, split)` in the manifest has a `.h5`, or an explicit skip reason in `cache_report.json`.
- [ ] `h5['clips'].shape == (N, 16, 112, 112, 3)` and `dtype == uint8` for every file.
- [ ] `N_cached + N_skipped == N_manifest` for every dataset — no clips vanish unaccounted.
- [ ] Round-trip test: cache 20 known clips, re-read, assert shape/dtype/label and that pixel variance > 0 (catches all-black clips).
- [ ] **RGB order test:** a synthetic pure-red video caches to a clip whose channel-0 mean ≫ channel-2 mean.
- [ ] **Uniform-sampling test:** a synthetic 100-frame video with frame index burned into pixel values yields sampled indices spanning ≈ 0–99, not 0–15.
- [ ] Total `data/cache` size < 4 GB; `df -h ~` still shows ≥ 15 GB free.
- [ ] Contact sheet rendered and visually confirmed to show recognisable, correctly-oriented, correctly-coloured frames.
- [ ] Cache build is idempotent — a second run skips completed files.
- [ ] `pytest -m "phase0 or phase1 or phase2"` passes.

### Exit Gate
```bash
make verify PHASE=2
```

---

# Phase 3 — Augmentation + `tf.data` pipeline

**Goal:** a fast, correct input pipeline with **clip-consistent** Albumentations augmentation, per
report §4.2.

### Preconditions
- Phase 2 `DONE`; caches present with matching `config_sha`.

### Deliverables
- `safestreets/data/augment.py` — `build_train_transform(cfg)` / `build_eval_transform(cfg)`
- `safestreets/data/dataset.py` — `make_dataset(split, cfg) -> tf.data.Dataset`
- `configs/data.yaml` augmentation block (probabilities, magnitudes)
- `tests/test_phase03_pipeline.py`
- `artifacts/figures/augmented_clips.png` — before/after grid

### The thing most likely to be done wrong: **per-frame augmentation**
Calling an Albumentations transform separately on each of the 16 frames randomises the parameters
per frame — the clip flickers, horizontal flips alternate mid-sequence, and you have destroyed
exactly the temporal coherence the LSTM exists to model. Report §4.2 calls this out explicitly.

**Mandatory:** use Albumentations' `ReplayCompose` — apply to frame 0, capture the replay dict,
then `A.ReplayCompose.replay(...)` that same dict on frames 1..15. Alternatively pass the clip as
`additional_targets={'image1': 'image', ...}`. Either way, **one parameter draw per clip.**

### Augmentation set (report §4.2)
`HorizontalFlip(p=0.5)`, `RandomBrightnessContrast(p=0.5)`, `RandomResizedCrop(scale=(0.8,1.0), p=0.5)`,
`GaussNoise(p=0.3)`. Eval transform = resize + normalise only, **nothing stochastic**.

### Implementation notes
- Normalisation lives here: `uint8 → float32 / 255.0`, then backbone-specific preprocessing if
  Phase 4 selects a pretrained backbone (MobileNetV2 wants `[-1,1]`, not `[0,1]`). Make this a
  config-driven function, not a hardcoded divide.
- Wrap the Albumentations call in `tf.numpy_function` / `tf.py_function` and **re-assert the static
  shape afterwards** with `set_shape` — otherwise the shape becomes `<unknown>` and the model
  refuses to build.
- Order: `.shuffle(buffer) → .map(augment, num_parallel_calls=AUTOTUNE) → .batch() → .prefetch(AUTOTUNE)`.
  Shuffle before batch. Cache the raw read, not the augmented output.
- Read from HDF5 lazily by index. Do not load a whole split into RAM — 16 GB is shared with the GPU.

### Definition of Done
- [ ] **Clip-consistency test (the important one):** feed a clip of 16 *identical* frames through the train transform; assert all 16 output frames are identical (`np.allclose`). A per-frame implementation fails this.
- [ ] Eval transform is deterministic: same input twice → bit-identical output.
- [ ] Batch shapes/dtypes: `(B,16,112,112,3) float32`, labels `(B,1) float32`; values within the configured normalisation range.
- [ ] Augmentation actually fires: over 200 train samples, mean pixel std across samples differs measurably from the eval pipeline's.
- [ ] Train dataset is shuffled — first batch differs across two epochs with the same seed-per-epoch policy.
- [ ] Throughput ≥ 200 clips/s on cached data (measure and record; if far below, the bottleneck is the pipeline, not the model).
- [ ] Stale-cache guard: mutating `configs/data.yaml` frame count raises a clear error rather than training on a mismatched cache.
- [ ] Before/after figure rendered; flips/crops visibly consistent within each clip row.
- [ ] `pytest -m "phase0 or ... or phase3"` passes.

### Exit Gate
```bash
make verify PHASE=3
```

---

# Phase 4 — Model, training loop, MLflow, TensorBoard

**Goal:** the report's §4.3 architecture, trained end to end, with every run tracked. This phase
produces the project's first real number.

### Preconditions
- Phase 3 `DONE`; pipeline throughput measured.
- ADR-001's pinned stack still installed.

### Deliverables
- `safestreets/models/cnn_lstm.py` — `build_scratch_cnn_lstm(cfg)` and `build_backbone_cnn_lstm(cfg)`
- `safestreets/models/backbones.py` — MobileNetV2 / ResNet50 / VGG16 factory, ImageNet weights, configurable freeze depth
- `safestreets/models/factory.py` — `build_model(cfg)` dispatching on `cfg.model.arch`
- `safestreets/training/losses.py` — `weighted_bce(pos_weight)`
- `safestreets/training/callbacks.py` — TensorBoard (with histograms), MLflow, checkpointing, early stopping, LR schedule
- `safestreets/training/train.py`, `scripts/train.py --config configs/train.yaml`
- `artifacts/checkpoints/baseline_*.keras`, `mlruns/`, `logs/fit/`
- `docs/decisions/ADR-003-architecture.md`
- `tests/test_phase04_model.py`

### Architecture (report §4.3, made concrete)

Two variants, both selectable by config. **Build both** — the report's §4.3 describes the scratch
CNN, while §2.2 and §2.8 justify transfer learning and lightweight backbones. Having both is what
lets Phase 5 report an honest comparison.

**A. `scratch` — literal §4.3 reading**
```
Input (T=16, 112, 112, 3)
TimeDistributed[ Conv2D(32,3,relu) → BN → MaxPool2 ]
TimeDistributed[ Conv2D(64,3,relu) → BN → MaxPool2 ]
TimeDistributed[ Conv2D(128,3,relu) → BN → GlobalAveragePooling2D ]
TimeDistributed[ Dense(256, relu) ]           # bottleneck — see note
LSTM(units, return_sequences=True)            # stacked, per §4.3
LSTM(units // 2)
Dropout(rate)                                 # after the LSTM, per §4.3
Dense(1, sigmoid)
```

**B. `mobilenetv2` — production variant**
Same skeleton, conv blocks replaced by `TimeDistributed(MobileNetV2(include_top=False, pooling='avg'))`,
ImageNet weights, bottom `N` layers frozen (`cfg.model.freeze_until`).

**Why the bottleneck matters:** with `Flatten` instead of `GlobalAveragePooling2D`, a 112×112 input
after three pool stages yields ~14×14×128 = 25 088 features per frame. An LSTM over that has tens of
millions of parameters, will not fit comfortably in 16 GB, and overfits 4 000 clips instantly. The
existing `app/ml_model/model.py` makes exactly this mistake at 64×64. **Do not use `Flatten` here.**

### Implementation notes
- **Class weighting (§4.3):** compute `pos_weight = n_neg / n_pos` from the *train* manifest only.
  Log the computed value to MLflow. If the combined set is near-balanced, the weight will be ≈1 —
  say so in the ADR rather than pretending it did heavy lifting.
- **TensorBoard activation histograms are explicitly required by §4.4** —
  `TensorBoard(log_dir=..., histogram_freq=1)`. This is slow; keep it on for the baseline run and
  make it config-toggleable for Optuna trials.
- **MLflow:** file store at `./mlruns`. Log params (full config as a flat dict), per-epoch metrics,
  the final model artefact, the TensorBoard log dir, and the git SHA. One experiment per phase
  (`safestreets-baseline`, later `safestreets-hpo`, `safestreets-finetune`).
- Metrics compiled into the model: `accuracy`, `AUC`, `Precision`, `Recall`. Monitor **`val_auc`**
  for early stopping and checkpointing — not `val_accuracy`, which is a poor guide under any class
  imbalance and insensitive to threshold-free ranking quality.
- Guard against the classic failure: if `val_auc ≈ 0.5` and train loss is falling, you are
  overfitting or your labels are shuffled relative to your clips. Add an explicit label-alignment
  assertion in the test file.
- Mixed precision on Metal is a known source of NaNs; leave it **off** unless ADR-001 says otherwise.

### Definition of Done
- [ ] `build_model(cfg)` returns a compiled model for **both** `arch: scratch` and `arch: mobilenetv2`; `model.summary()` saved to `artifacts/model_summary_{arch}.txt`.
- [ ] Parameter count for `scratch` is < 15 M (proves the bottleneck is in place, not `Flatten`).
- [ ] **Overfit test:** the model reaches ≥ 0.95 train accuracy on a deliberately tiny 32-clip subset within 30 epochs. If it cannot overfit 32 clips, the architecture or label wiring is broken — fix that before any full run.
- [ ] **Label-alignment test:** for a batch drawn from the pipeline, the labels match the manifest labels for those exact `clip_id`s.
- [ ] A full baseline run completes on RWF-2000 + RLVS train, ≥ 30 epochs or early-stopped.
- [ ] `val_auc > 0.80` on the held-out split. *(If not met: record the actual number, investigate, and do not proceed to Phase 6 — tuning a broken model wastes a whole session. `0.80` is the go/no-go bar, well below the 0.87–0.93 the literature reports for RWF-2000.)*
- [ ] MLflow shows ≥ 1 run with params, per-epoch metrics, and a logged model artefact; `mlflow ui` renders it.
- [ ] TensorBoard shows loss curves, per-epoch accuracy, **and weight/activation histograms** (§4.4).
- [ ] Re-running with the same seed and config reproduces `val_auc` within ±0.02.
- [ ] Best checkpoint saved with the config and git SHA embedded alongside it.
- [ ] `docs/decisions/ADR-003-architecture.md` records the chosen defaults and the measured baseline.
- [ ] `pytest -m "phase0 or ... or phase4"` passes (heavy runs marked `slow`).

### Exit Gate
```bash
make verify PHASE=4
```

---

# Phase 5 — Evaluation harness

**Goal:** the multi-metric, cross-dataset, frame-level evaluation the report commits to in §2.10 —
the section that criticises other papers for exactly the sloppiness this phase must avoid.

### Preconditions
- Phase 4 `DONE`; a checkpoint meeting the `val_auc > 0.80` bar exists.

### Deliverables
- `safestreets/evaluation/metrics.py` — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, **95% CI via bootstrap**
- `safestreets/evaluation/evaluate.py` — single-split evaluation
- `safestreets/evaluation/cross_dataset.py` — train-on-A / test-on-B matrix
- `safestreets/evaluation/temporal.py` — sliding-window frame-level scoring + frame-level AUC for UCF-Crime / XD-Violence
- `scripts/evaluate.py --checkpoint ... --splits ...`
- `artifacts/reports/eval_{tag}.json` + `docs/RESULTS.md` (generated, never hand-typed)
- `artifacts/figures/` — ROC, PR, confusion matrix, threshold sweep
- `tests/test_phase05_eval.py`

### What must be measured
1. **In-domain:** held-out test split of the training datasets.
2. **Cross-dataset matrix (§2.10):** train on RWF+RLVS → test on AIRTLab and on the UCF-Crime
   subset, zero-shot. Report the drop explicitly. The report's §5 predicts this drop is large; if
   your numbers show no drop at all, **suspect leakage in Phase 1** before celebrating.
3. **Frame-level AUC:** slide a `T=16` window with stride 8 over untrimmed UCF-Crime test videos and
   XD-Violence I3D features; score each frame by its windows' mean; compute AUC against the
   frame-level annotations. This is the metric §2.10 says the field should have converged on.
4. **Operating point:** sweep the decision threshold; report the threshold that maximises F1 and the
   one that holds recall ≥ 0.90 (the deployment-relevant point for a safety system, where a missed
   assault costs more than a false alarm). Persist the chosen threshold to `configs/infer.yaml` —
   Phase 9 consumes it.

### Implementation notes
- The threshold is **not** 0.5 by default. Derive it from the validation split, never from test.
- Bootstrap CIs (1 000 resamples) on every headline metric. With ~600 test clips, the 95% CI on
  accuracy is roughly ±3 pp; a report that claims a 1-point improvement without CIs is claiming
  noise. This is the difference between the project and the papers §2.10 criticises.
- `docs/RESULTS.md` is **generated** by `scripts/evaluate.py`. Regenerating it after a re-run must
  change the numbers. Never edit it by hand.
- Include the untuned `scratch` vs `mobilenetv2` comparison — it is a genuine finding either way.

### Definition of Done
- [ ] Metrics implementation validated against `sklearn` on synthetic data (exact agreement to 1e-9).
- [ ] `eval_{tag}.json` contains accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and bootstrap 95% CIs — for **every** split evaluated.
- [ ] Cross-dataset matrix produced for ≥ 2 train-sets × ≥ 3 test-sets, rendered as a table in `docs/RESULTS.md`.
- [ ] Frame-level AUC computed on ≥ 1 untrimmed benchmark, with the window/stride stated.
- [ ] Threshold sweep figure exists; the selected threshold is written to `configs/infer.yaml` and was chosen on **validation**, with that fact asserted in a test.
- [ ] All four figures render and are referenced from `docs/RESULTS.md`.
- [ ] `docs/RESULTS.md` regenerates deterministically — run twice, `git diff` is empty.
- [ ] Every number in `docs/RESULTS.md` traces to a key in a committed `eval_*.json`. No exceptions.
- [ ] `pytest -m "phase0 or ... or phase5"` passes.

### Exit Gate
```bash
make verify PHASE=5
```

---

# Phase 6 — Optuna hyperparameter optimisation

**Goal:** report §4.4's systematic Bayesian search, fully tracked — replacing manual trial-and-error.

### Preconditions
- Phase 5 `DONE`. A trustworthy `val_auc` exists to optimise against.

### Deliverables
- `safestreets/training/tune.py` — Optuna study, `TPESampler`, `MedianPruner`
- `scripts/tune.py --config configs/tune.yaml --trials N --timeout S`
- `artifacts/optuna/study.db` (SQLite — resumable across sessions)
- `artifacts/figures/optuna_{history,importances,parallel}.png`
- `configs/model.best.yaml` — the winning config, written out programmatically
- `artifacts/checkpoints/best_*.keras` — retrained on train+val at the best config
- `docs/decisions/ADR-004-hpo.md`
- `tests/test_phase06_tune.py`

### Search space (report §4.4 names these four; the rest are optional additions)
| Param | Range | Sampler |
|---|---|---|
| `learning_rate` | 1e-5 – 1e-2 | log-uniform |
| `lstm_units` | {64, 128, 256, 512} | categorical |
| `dropout` | 0.2 – 0.6 | uniform |
| `batch_size` | {8, 16, 32} | categorical |
| *(optional)* `freeze_until` | {0, 50, 100, 155} | categorical |
| *(optional)* `optimizer` | {adam, rmsprop} | categorical |

Objective: **maximise `val_auc`**. One objective only — do not silently switch to accuracy.

### Implementation notes
- **Budget honestly.** On an M2, one 30-epoch trial is ~20–60 min. A 50-trial study is 1–2 days of
  wall clock. Options, in order of preference: (a) `MedianPruner` + a 15-epoch cap + a 50% train
  subsample for the search, then retrain the winner fully; (b) run the study on Colab; (c) reduce
  to 20 trials and **say so** in the ADR. Do not quietly run 5 trials and call it a Bayesian search.
- SQLite storage is what makes this survive session boundaries: a later session runs
  `scripts/tune.py --resume` and continues the same study. Design for that from the start.
- Every trial logs to MLflow as a nested run under one parent.
- Prune on `val_auc` at each epoch via `TFKerasPruningCallback`.
- Retrain the winner on `train + val` with the best epoch count from the search, then evaluate on
  test **once**. Touching test during the search invalidates every number in Phase 5.

### Definition of Done
- [ ] Study completes ≥ 20 trials (or the ADR records a justified smaller number with the reason).
- [ ] `study.db` resumes correctly: kill mid-study, re-run with `--resume`, trial count continues rather than restarting.
- [ ] Pruning demonstrably active — ≥ 1 trial has state `PRUNED`.
- [ ] `best_params` written to `configs/model.best.yaml`; loading it reproduces the best trial's architecture exactly.
- [ ] Three Optuna figures rendered; parameter-importance plot interpreted in one paragraph in the ADR.
- [ ] Best config retrained on train+val; **test** metrics regenerated through Phase 5's harness.
- [ ] Tuned test ROC-AUC ≥ baseline test ROC-AUC. *(If tuning made it worse, that is a legitimate result — report it, and keep the baseline as the production model. Do not cherry-pick.)*
- [ ] MLflow shows the parent run with all trials nested under it.
- [ ] `docs/RESULTS.md` regenerated to include the tuned row alongside the baseline.
- [ ] `pytest -m "phase0 or ... or phase6"` passes.

### Exit Gate
```bash
make verify PHASE=6
```

---

# Phase 7 — AIRTLab women-specific fine-tuning

**Goal:** report §4.1's commitment — "incorporating AIRTLab for women-specific fine-tuning and
evaluation."

### Preconditions
- Phase 6 `DONE`; a tuned production checkpoint exists.
- AIRTLab cached (Phase 2) with a **scene/actor-grouped** split (Phase 1).

### Deliverables
- `safestreets/training/finetune.py`, `scripts/finetune.py`
- `artifacts/checkpoints/finetuned_airtlab_*.keras`
- `artifacts/reports/eval_finetuned.json`, updated `docs/RESULTS.md`
- `docs/decisions/ADR-005-finetuning.md`
- `tests/test_phase07_finetune.py`

### Implementation notes
- AIRTLab is **small** (~350 clips) and synthetic/acted. Fine-tuning a full network on it will
  overfit within a few epochs. Strategy: freeze the CNN, fine-tune only the LSTM + head, low LR
  (≈ 1/10 of the base), heavy early stopping, and 5-fold grouped CV to get an error bar rather than
  a single fragile number.
- **Measure catastrophic forgetting.** After fine-tuning, re-evaluate on the RWF-2000 held-out set.
  If in-domain AUC collapses, the model has traded general competence for a small synthetic set.
  Report both numbers and pick deliberately; consider keeping both checkpoints and selecting at
  inference time.
- Be candid in the ADR about what AIRTLab can and cannot support. It is acted, staged footage with
  few scenes and actors. A high number on it is **not** evidence of real-world performance on
  violence against women, and the report's own §5 says as much. Write that limitation down; it
  belongs in `docs/MODEL_CARD.md` in Phase 11.

### Definition of Done
- [ ] Fine-tuned checkpoint produced from the Phase 6 production model (provenance recorded: parent checkpoint SHA).
- [ ] 5-fold grouped CV on AIRTLab reports mean ± std for accuracy, F1, ROC-AUC.
- [ ] **Forgetting check:** RWF-2000 held-out AUC re-measured post-fine-tune and reported next to the pre-fine-tune value.
- [ ] A documented selection decision: which checkpoint ships, and why, in ADR-005.
- [ ] `docs/RESULTS.md` gains a fine-tuning section with pre/post numbers for both domains.
- [ ] Limitations paragraph drafted (feeds the Phase 11 model card).
- [ ] `pytest -m "phase0 or ... or phase7"` passes.

### Exit Gate
```bash
make verify PHASE=7
```

---

# Phase 8 — Person detection + gender attribution (SUSAN-style)

**Goal:** the module that makes this *"violence detection **against women**"* rather than generic
violence detection — report §2.7 and §5, following Pereira et al.'s SUSAN three-module design.

### Preconditions
- Phase 7 `DONE`.

### Deliverables
- `safestreets/attribution/person.py` — YOLO person detection + lightweight tracking
- `safestreets/attribution/gender.py` — gender classifier over person crops
- `safestreets/attribution/pipeline.py` — fuses violence score + attribution into an alert decision
- `configs/attribution.yaml`
- `artifacts/reports/attribution_eval.json`
- `docs/ETHICS.md` — **required deliverable, not optional**
- `tests/test_phase08_attribution.py`

### Design
```
clip ──┬─► violence model ──────────────► p_violence
       └─► YOLO(person) ─► crops ─► gender ─► {n_people, p_female per track}
                                                        │
                          alert = p_violence > τ  AND  any(p_female > τ_g)
                          severity ← p_violence, n_people, track persistence
```

- **Detector:** `ultralytics` YOLO, person class only, run on a **subsample** of frames (every 4th)
  — running it on all 16 frames of every clip is the obvious latency trap.
- **Tracking:** ByteTrack (built into ultralytics) so gender is decided per *person*, aggregated
  across frames by median, not re-decided per frame. Per-frame decisions flicker badly.
- **Gender classifier:** train on PA-100K's gender attribute (SUSAN's choice) over person crops.
  Keep a zero-shot CLIP fallback behind a config flag if PA-100K access is blocked — and mark it
  clearly as a fallback in the results, since its accuracy profile is different.

### Ethics — this is engineering scope, not a footnote
Automated gender inference from appearance is **error-prone and consequential**. It fails
disproportionately on gender-nonconforming and transgender people, on children, in low resolution,
at distance, in poor lighting, and under occlusion — which describes most real CCTV. `docs/ETHICS.md`
must state plainly:

- The gender signal is used to **prioritise and route** alerts, never to gate a human review path
  in a way that could suppress a genuine violence alert.
- Therefore: **when `p_violence` is high, the alert fires regardless of the gender module's output.**
  The attribution changes the alert's *label and priority*, not its *existence*. Encode this in
  `pipeline.py` and assert it in a test — it is a correctness property, not a preference.
- Measured accuracy of the gender module, and its failure modes, are published in the model card.
- No storage of raw person crops beyond the inference call; alerts persist scores and bounding
  boxes, not identifiable imagery.

### Definition of Done
- [ ] Person detection runs on cached clips; mean detections/frame sane on a hand-checked 20-clip sample.
- [ ] Tracking assigns stable IDs — a test video with one walking person yields one dominant track, not 16.
- [ ] Gender classifier evaluated on a held-out split; accuracy, per-class recall, and confusion matrix in `attribution_eval.json`.
- [ ] **Fail-safe test (mandatory):** with `p_violence = 0.99` and the gender module forced to return "no women detected", the pipeline still raises an alert. This test must exist and pass.
- [ ] End-to-end latency for the full attribution path measured and recorded per clip.
- [ ] Graceful degradation: if the detector or gender model is missing/fails, the pipeline logs a warning and falls back to violence-only alerts rather than crashing.
- [ ] `docs/ETHICS.md` written, covering the bullets above plus measured failure modes.
- [ ] `pytest -m "phase0 or ... or phase8"` passes.

### Exit Gate
```bash
make verify PHASE=8
```

---

# Phase 9 — ONNX export + real-time streaming inference

**Goal:** report §1's ONNX export and §2.8's real-time, edge-capable inference.

### Preconditions
- Phase 8 `DONE`. ADR-001's ONNX path still valid (re-verify — the stack may have drifted).

### Deliverables
- `safestreets/inference/export_onnx.py`, `scripts/export_onnx.py`
- `artifacts/onnx/safestreets_{arch}_{tag}.onnx` + a sidecar `.json` (input spec, normalisation, threshold, git SHA, source checkpoint SHA)
- `safestreets/inference/engine.py` — `InferenceEngine` over `onnxruntime`
- `safestreets/inference/stream.py` — sliding-window scoring over a live/file stream
- `artifacts/reports/latency.json`, `artifacts/reports/onnx_parity.json`
- `tests/test_phase09_inference.py`

### Streaming design
- Ring buffer of the last `T=16` preprocessed frames; score every `stride=8` frames (50% overlap).
- **EMA smoothing** of the score (`α ≈ 0.3`) — raw per-window scores are jumpy and produce alert storms.
- **Hysteresis:** fire on `score > τ_high` sustained for `k` consecutive windows; clear only on
  `score < τ_low`. A single threshold flickers on and off at the boundary; this is the single most
  common real-time-demo failure. Report ref [16] uses an adaptive-threshold sliding window for the
  same reason.
- Frame drop policy: if the source outruns inference, drop frames to stay live rather than queueing
  into unbounded latency. Record the drop rate.

### Definition of Done
- [ ] ONNX export succeeds for the production checkpoint; the file loads in `onnxruntime`.
- [ ] **Parity:** max absolute difference between Keras and ONNX outputs < `1e-4` over ≥ 50 real clips; written to `onnx_parity.json`.
- [ ] Metric parity: ONNX-path ROC-AUC on the test split matches the Keras-path value within 0.005.
- [ ] Latency benchmarked on this machine: p50/p95/p99 ms per 16-frame window, batch=1, and the implied sustainable FPS — in `latency.json`.
- [ ] **Real-time bar:** end-to-end sustainable throughput ≥ 15 FPS on CPU for the violence model alone. *(If unmet: reduce input resolution or switch to the lightweight backbone, and record the trade in an ADR. Do not simply claim real-time.)*
- [ ] Sidecar JSON present and complete; `InferenceEngine` refuses to run if the sidecar's normalisation spec disagrees with the active config.
- [ ] Streaming over `test/sample_video.avi` produces a timestamped score series and at least one correctly-shaped alert event object.
- [ ] Hysteresis test: a synthetic score series oscillating around τ produces **one** alert, not many.
- [ ] `pytest -m "phase0 or ... or phase9"` passes.

### Exit Gate
```bash
make verify PHASE=9
```

---

# Phase 10 — Flask application rebuild

**Goal:** replace the broken `app/` with a real application: upload analysis, live-stream analysis,
alerts, and a results UI that actually shows the prediction.

### Preconditions
- Phase 9 `DONE`; ONNX model + engine available.

### Deliverables
- `safestreets/web/__init__.py` (app factory), `routes.py`, `api.py`
- `safestreets/web/templates/` — `index.html`, `result.html`, `live.html`, `about.html`, `contact.html`
- `safestreets/web/static/` — migrated CSS, no CDN dependency
- `run.py` updated; `app/` **deleted**
- `tests/test_phase10_web.py` (Flask test client)

### API
| Method | Route | Behaviour |
|---|---|---|
| `GET` | `/` | upload form |
| `POST` | `/api/analyse` | multipart video → `{clip_score, verdict, threshold, timeline[], attribution{}, latency_ms}` |
| `GET` | `/api/jobs/<id>` | status/result for a long analysis |
| `GET` | `/live` | live-stream page |
| `GET` | `/api/stream/<source>` | SSE stream of scores/alerts |
| `GET` | `/healthz` | model loaded, version, git SHA |

### Fix these known defects explicitly
1. `predict_video` — currently imported but nonexistent. The whole app 500s. Replace with the
   Phase 9 `InferenceEngine`.
2. `templates/index.html` fetches `/upload`, receives `data.prediction`, and **throws it away**,
   printing only "Video uploaded successfully!". The user never sees a result. Render score,
   verdict, confidence, and the per-window timeline.
3. Uploads are saved by `secure_filename` into `footages/` with **no size limit, no content-type
   validation, and no cleanup** — an unbounded disk-fill vector. Add `MAX_CONTENT_LENGTH`, verify
   the file is decodable video via OpenCV before analysis, use a UUID temp path, and delete after.
4. `test/test_predict.py` normalises by `/2300.0` instead of `/255.0`. Delete the file; its role is
   taken by `tests/test_phase09_inference.py`.
5. Tailwind is loaded from `unpkg.com` — pin a vendored copy in `static/` so the UI works offline.

### Implementation notes
- Load the ONNX model **once** at app startup, not per request.
- Long videos must not block the request thread — background thread/queue with a job id, or stream
  progress over SSE. State the choice in the docstring.
- The alert object shape must match Phase 8's `pipeline.py` output exactly. Import it; do not
  redefine it in the web layer.

### Definition of Done
- [ ] `GET /` 200s; `GET /healthz` returns model version + git SHA.
- [ ] `POST /api/analyse` with `test/sample_video.avi` returns 200 and a schema-valid body with a numeric `clip_score` in [0,1].
- [ ] Rejections return 4xx with a JSON error: no file, wrong extension, oversized file, non-decodable file. One test per case.
- [ ] Uploaded files are removed from disk after analysis (asserted in a test).
- [ ] The result page **renders the verdict and score** — asserted by string-matching the response HTML, closing defect #2.
- [ ] `/live` renders and the SSE endpoint emits ≥ 3 score events for a file source.
- [ ] All routes covered by Flask test-client tests; no test requires a browser.
- [ ] `app/` and `test/test_predict.py` deleted; nothing imports them (`grep -r "from app" .` returns nothing).
- [ ] No external CDN required to render the UI.
- [ ] `pytest -m "phase0 or ... or phase10"` passes.

### Exit Gate
```bash
make verify PHASE=10
```

---

# Phase 11 — Packaging, reproducibility, and reporting

**Goal:** someone with the repo and no context can reproduce the results and run the system.

### Preconditions
- Phase 10 `DONE`.

### Deliverables
- `Dockerfile` (rewritten — the current one runs a nonexistent file), `docker-compose.yml`, `.dockerignore`
- `Makefile` — `setup, data, cache, train, tune, eval, export, serve, test, verify, all`
- `README.md` — full rewrite: what it is, results table, quickstart, architecture diagram, licence
- `docs/MODEL_CARD.md` — intended use, training data, metrics, **limitations**, ethical considerations
- `docs/RESULTS.md` — final, regenerated
- `docs/REPRODUCE.md` — exact commands, in order, from clone to results
- `artifacts/figures/architecture.png`
- `CITATION.cff`; final `docs/decisions/ADR-006-deployment.md`

### Definition of Done
- [ ] `docker build` succeeds and `docker run` serves the app; `/healthz` responds from inside the container.
- [ ] The image does **not** contain datasets or checkpoints (`.dockerignore` correct); size recorded.
- [ ] `make all` runs the full pipeline end to end on a small `--smoke` subset without manual steps.
- [ ] `docs/REPRODUCE.md` followed literally from a fresh clone on a clean venv reproduces the headline metric within the stated CI. **Actually do this**, on a fresh clone in a temp directory, and paste the transcript into PROGRESS.md.
- [ ] `README.md` results table matches `docs/RESULTS.md` exactly — both generated, neither hand-typed.
- [ ] Model card covers: intended use, out-of-scope use, training data with licences, disaggregated metrics where measurable, AIRTLab's synthetic-data limitation (Phase 7), and the gender module's failure modes (Phase 8).
- [ ] Every requirement row in this plan's §2 table maps to a shipped artefact — verified line by line and recorded as a checklist in PROGRESS.md.
- [ ] `pytest` (all phases, including `slow`) passes.
- [ ] Fresh-clone CI run green.

### Exit Gate
```bash
make verify PHASE=11 && make verify-all
```

---

# 6. `CLAUDE.md` bootstrap (create in Phase 0)

The root `CLAUDE.md` should tell a cold session exactly this:

> Read `docs/PROGRESS.md`, find the first phase that is not `DONE`, then read that phase's section
> in `docs/BUILD_PLAN.md` and only that one. Honour every ADR in `docs/decisions/`. Never write a
> metric you did not produce by running something. Finish the phase, run its Exit Gate, update
> PROGRESS.md, commit as `phase(N): ...`.

# 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Dataset access blocked (RWF-2000 form, Kaggle auth) | High | High | Phase 1 marks `BLOCKED` and proceeds with available sources; plan tolerates a subset |
| R2 | Disk exhaustion (44 GB free) | High | High | §3.1 budget; cache-then-purge; never fetch raw UCF-Crime/XD-Violence in full |
| R3 | `tf2onnx` × Keras 3 incompatibility | Medium | High | ADR-001 spike in Phase 0, **before** model code exists |
| R4 | M2 too slow for a real Optuna study | High | Medium | Pruning + subsampling + SQLite resume; Colab escape hatch; all entry points path-agnostic |
| R5 | Split leakage inflating every metric | Medium | **Critical** | Group-aware splits (Phase 1) + an explicit disjointness assertion in tests |
| R6 | `val_auc` stuck near 0.5 | Medium | High | Phase 4's 32-clip overfit test and label-alignment test catch this in minutes, not days |
| R7 | Real-time target unreachable on CPU | Medium | Medium | Lightweight backbone, lower resolution, larger stride; record the trade honestly |
| R8 | Gender module accuracy poor / ethically fraught | High | High | Fail-safe design (alert never suppressed), `docs/ETHICS.md`, measured failure modes in the model card |
| R9 | Reported numbers drift from committed artefacts | Medium | High | `RESULTS.md` is generated; the "no fabricated numbers" rule; CI regenerates and diffs |

# 8. What "the whole project is done" means

All eleven Exit Gates pass, `make verify-all` is green, `docs/RESULTS.md` is generated from
committed evaluation artefacts, and every row of §2's requirements table points at a real file.
