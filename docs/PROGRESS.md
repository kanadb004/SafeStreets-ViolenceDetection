# SafeStreets — Build Progress

**Single source of truth for "where are we".** A cold session reads this file first.

**Statuses:** `NOT_STARTED` · `IN_PROGRESS` · `BLOCKED` · `DONE`
A phase is `DONE` **only** when its Exit Gate command has been run and exited 0, and the evidence
block below is filled in with real output. Not "I think it works".

Last updated: 2026-10-10 · by: sprint-1 session · commit: `08e27d5`

---

## Board

| # | Phase | Status | Exit gate run? | Session notes |
|---|-------|--------|----------------|---------------|
| 0 | Foundation: env, packaging, config, CI | `DONE` | ✓ | env pinned per ADR-001, PR pending |
| 1 | Datasets, manifests, leakage-safe splits | `DONE` | ✓ | merged; XD-Violence SKIPPED (38.3 GB, over budget), see docs/DATASETS.md |
| 2 | Preprocessing → clip cache | `DONE` | ✓ | merged; RLVS drops 4 unreadable clips |
| 3 | Augmentation + `tf.data` pipeline | `DONE` | ✓ | merged; additional_targets used instead of ReplayCompose for throughput |
| 4 | Model + training + MLflow + TensorBoard | `SUPERSEDED` | n/a | replaced by Sprint 1, see docs/SPRINT_PLAN.md |
| 5 | Evaluation harness | `SUPERSEDED` | n/a | replaced by Sprint 2 |
| 6 | Optuna HPO | `SUPERSEDED` | n/a | replaced by Sprint 3 |
| 7 | AIRTLab fine-tuning | `SUPERSEDED` | n/a | replaced by Sprint 4 |
| 8 | Person + gender attribution | `SUPERSEDED` | n/a | replaced by Sprint 4 |
| 9 | ONNX + streaming inference | `SUPERSEDED` | n/a | replaced by Sprint 5 |
| 10 | Flask application rebuild | `SUPERSEDED` | n/a | replaced by Sprint 5 |
| 11 | Packaging, reproducibility, reporting | `SUPERSEDED` | n/a | replaced by Sprint 6 |

### Sprint board (active) — deadline 1 to 1.5 days from 2026-10-10

Phases 0 to 3 are done. Everything remaining runs through `docs/SPRINT_PLAN.md`, which supersedes
Phases 4 to 11. **This is the board to read.**

| # | Sprint | Status | Exit gate run? | Notes |
|---|--------|--------|----------------|-------|
| 1 | Feature store, models, training | `DONE` | ✓ | val_auc 0.8679 (lstm_head) vs 0.6041 (scratch, 12ep) |
| 2 | Evaluation harness | `NOT_STARTED` | x | no frame-level AUC, see SPRINT_PLAN §2 |
| 3 | Optuna HPO | `NOT_STARTED` | x | 30 trials, ~25 min measured |
| 4 | AIRTLab fine-tune + attribution | `NOT_STARTED` | x | zero-shot CLIP gender, no PA-100K training |
| 5 | ONNX, real-time, Flask app | `NOT_STARTED` | x | closes the 5 legacy defects |
| 6 | Notebooks, results, model card | `NOT_STARTED` | x | **the submission** |

---

## Headline numbers

Fill these in **only** from a committed `artifacts/reports/eval_*.json`. `NOT RUN` until then.

| Metric | Value | Source file | Phase |
|---|---|---|---|
| Baseline test ROC-AUC (RWF+RLVS held-out) | `NOT RUN` | — | 4/5 |
| Tuned test ROC-AUC | `NOT RUN` | — | 6 |
| Cross-dataset AUC (→ AIRTLab, zero-shot) | `NOT RUN` | — | 5 |
| Cross-dataset AUC (→ UCF-Crime subset) | `NOT RUN` | — | 5 |
| Frame-level AUC (untrimmed) | `NOT RUN` | — | 5 |
| AIRTLab fine-tuned accuracy (5-fold) | `NOT RUN` | — | 7 |
| Gender module accuracy | `NOT RUN` | — | 8 |
| ONNX p95 latency / sustainable FPS | `NOT RUN` | — | 9 |

---

## Evidence log

Append one block per completed phase. Paste **real** command output.

### Phase 0 — Foundation
Completed: 2026-09-13 · commit: `1d62644`
Exit gate: `make verify PHASE=0`
Exit code: 0

```
ruff check . && pytest -m phase0 -q
All checks passed!
....                                                                     [100%]
4 passed in 12.16s
test -f docs/decisions/ADR-001-runtime-stack.md
```

ONNX parity spike (ADR-001):
```
keras_out [0.4972795  0.49701098]
onnx_out [0.49727955 0.49701098]
max abs diff 5.9604645e-08
PARITY OK
```

Metal benchmark (ADR-001):
```
/GPU:0   1458 GFLOP/s     10x 2048^3 matmul
/CPU:0   322 GFLOP/s     10x 2048^3 matmul
speedup: 4.5x
```

DoD checklist: 9/9 met
Deviations from plan: none in scope. Unplanned dependency conflicts surfaced during the spike
(onnx/onnxruntime/opencv-python-headless pulling numpy>=2 and ml_dtypes>=0.5.4 against
tensorflow 2.16.2's numpy<2/ml_dtypes~=0.3.1 pins; an unpinned `pip install tf-keras` also
upgraded tensorflow to a generic 2.21.0 wheel and broke tensorflow-metal). Resolved by pinning
the full chain; recorded in ADR-001, not a separate ADR since it's part of the same runtime-stack
decision.
Surprises / notes for the next session:
- macOS's default case-insensitive filesystem silently merged a new `safestreets/` package
  directory into the empty legacy `SafeStreets/` dir on first `mkdir -p`. Caught before commit by
  checking `git status --untracked-files=all`; fixed via a rename through a temp name. Future
  sessions creating `safestreets/...` paths should double check `git status` shows lowercase paths.
- Any future `pip install` into the `safestreets` env that touches tensorflow, onnx, onnxruntime,
  numpy, ml_dtypes, or opencv should re-pin against ADR-001's table afterward; the resolver will
  silently drift them otherwise.

### Phase 1 — Datasets, manifests, leakage-safe splits
Completed: 2026-09-13 · commit: `6280d9b`
Exit gate: `make verify PHASE=1`
Exit code: 0

```
ruff check . && pytest -m phase1 -q
All checks passed!
.................                                                        [100%]
17 passed, 4 deselected in 185.75s (0:03:05)
```

Full guard including Phase 0 (`pytest -m "phase0 or phase1" -q`):
```
.....................                                                    [100%]
21 passed in 240.59s (0:04:00)
```

Idempotency (`python scripts/fetch_data.py --dataset rwf2000` run twice):
```
( conda run -n safestreets python scripts/fetch_data.py --dataset rwf2000; )   1.39s user 0.33s system 84% cpu 2.033 total
```

Manifest build (`python -m safestreets.data.manifest`), 4336 rows:
```
dataset     train  val  test    train%  val%  test%
airtlab       242   60    48     69.1  17.1  13.7
rlvs         1404  274   273     72.0  14.0  14.0
rwf2000      1600  400     0     80.0  20.0   0.0   (official split, no test — see ADR-002)
ucfcrime       24    6     5     68.6  17.1  14.3
```
Leakage assertion (`set(train.group_id) & set(val.group_id) & set(test.group_id)`), all four
datasets: empty. `clips.parquet`: 4336 rows, 0 null labels, 0 duplicate clip_id, 100% of paths
exist on disk. Two consecutive `build_manifest()` runs produced a byte-identical `splits.json`
(verified both by an md5 diff and by `tests/test_phase01_manifest.py::test_manifest_rebuild_is_byte_identical`).

DoD checklist: 9/9 met
Deviations from plan: RWF-2000 uses its official train/val split verbatim and contributes no rows
to `test` (recorded in ADR-002, not treated as a Phase 1 DoD failure since the ±3pp tolerance
target is met by every dataset that uses hash-based bucketing). RLVS's `group_id` comes from a
content near-duplicate heuristic (average-hash of the middle frame + duration) rather than
filename parsing, because this Kaggle re-hosting's filenames (`V_<n>.mp4`/`NV_<n>.mp4`) carry no
source-video id. AIRTLab's `group_id` groups by (label, clip_number) rather than camera folder,
since cam1/cam2 are two views of the same event per the dataset's own readme. UCF-Crime's Robbery
category (absent from the small mirror used for the other four categories) was streamed
file-by-file from a second, larger mirror via `kaggle datasets download -f`, never downloading
that mirror in full. All four decisions are recorded in ADR-002.
Surprises / notes for the next session:
- The RLVS Kaggle mirror's zip nests a second, byte-identical copy of the whole dataset under
  `real life violence situations/`; `fetch_rlvs` deletes it. If re-downloading fresh, check for
  this duplicate again in case the mirror's zip layout changes.
- RWF-2000 filenames collide across labels: the same source-video id can have both a Fight- and a
  NonFight-labelled segment sharing the same segment index (e.g. `-1l5631l3fg_0` under both
  `Fight/` and `NonFight/`). `clip_id` includes the label folder name to stay unique; `group_id`
  does not, so both labelled segments of one source video are still grouped together.
- XD-Violence is permanently `SKIPPED`, not just credential-blocked: its official OneDrive release
  (`i3d-features.zip`) was checked manually in a browser and is **38.3 GB**, far past BUILD_PLAN
  §3.1's ~4 GB estimate and the project's ~44 GB disk budget. The Baidu Netdisk alternative needs
  an account this project doesn't have and would carry the same size problem regardless. It only
  fed Phase 5's supplementary cross-dataset/frame-level AUC, so this does not block any other
  phase; Phase 5 should proceed without an XD-Violence-derived metric. See `docs/DATASETS.md`.
- `manifest.py`'s OpenCV probe pass takes ~100s for the current 4336 clips (~25ms/clip); expect
  this to scale roughly linearly if later phases add more sources to the manifest.

### Phase 2 — Preprocessing → compact clip cache
Completed: 2026-09-13 · commit: `1232c08`
Exit gate: `make verify PHASE=2`
Exit code: 0

```
ruff check . && pytest -m phase2 -q
All checks passed!
...............                                                          [100%]
15 passed, 21 deselected in 3.19s
```

Full guard including Phase 0/1 (`pytest -m "phase0 or phase1 or phase2" -q`):
```
....................................                                     [100%]
36 passed in 270.74s (0:04:30)
```

Cache build (`python scripts/build_cache.py --dataset all --split all --workers 6`), 4336 manifest rows:
```
airtlab/test: wrote 48, skipped 0, 28.9 MB, 206.1s -> data/cache/airtlab_test.h5
airtlab/train: wrote 242, skipped 0, 145.7 MB, 1100.5s -> data/cache/airtlab_train.h5
airtlab/val: wrote 60, skipped 0, 36.1 MB, 218.0s -> data/cache/airtlab_val.h5
rlvs/test: wrote 272, skipped 1, 163.8 MB, 111.8s -> data/cache/rlvs_test.h5
rlvs/train: wrote 1402, skipped 2, 844.3 MB, 539.2s -> data/cache/rlvs_train.h5
rlvs/val: wrote 273, skipped 1, 164.4 MB, 138.7s -> data/cache/rlvs_val.h5
rwf2000/train: wrote 1600, skipped 0, 963.5 MB, 186.7s -> data/cache/rwf2000_train.h5
rwf2000/val: wrote 400, skipped 0, 240.9 MB, 52.4s -> data/cache/rwf2000_val.h5
ucfcrime/test: wrote 5, skipped 0, 3.0 MB, 2.9s -> data/cache/ucfcrime_test.h5
ucfcrime/train: wrote 24, skipped 0, 14.5 MB, 7.6s -> data/cache/ucfcrime_train.h5
ucfcrime/val: wrote 6, skipped 0, 3.6 MB, 3.6s -> data/cache/ucfcrime_val.h5
```
4332 written + 4 skipped (all `unreadable_frame`, RLVS: `rlvs_NV_357`, `rlvs_V_8`, `rlvs_NV_360`,
`rlvs_NV_362`) == 4336 manifest rows. Total `data/cache` size: 2.4 GB (`du -sh data/cache`); `df -h`
still shows 30 GB free. `h5['clips'].shape[1:] == (16, 112, 112, 3)`, `dtype == uint8` in every file.

Idempotency (`python scripts/build_cache.py --dataset all --split all --workers 6` run again,
cache already built):
```
airtlab/test: up to date (data/cache/airtlab_test.h5), skipping
... (all 11 dataset/split combos skip)
conda run -n safestreets python scripts/build_cache.py --dataset all --split ...   2.38s user 0.93s system 109% cpu 3.033 total
```

Contact sheet: `artifacts/figures/sample_clips.png` rendered and visually confirmed, natural colour
(no BGR/RGB swap), correct upright orientation, letterbox padding visible on portrait-orientation
clips, frames spanning visible motion across each clip's duration (not stuck on the first frame).

DoD checklist: 10/10 met
Deviations from plan: none. The 4 RLVS clips OpenCV cannot decode a frame from are logged in
`cache_report.json` with `reason: unreadable_frame` and excluded from all three RLVS `.h5` files,
per the "skip and log, never write a zeros clip" rule; not a leakage or split-balance concern since
they are dropped uniformly, not filtered by label.
Surprises / notes for the next session:
- A handful of RLVS source files throw `ffmpeg`/OpenCV H.264 decode warnings
  (`mb_type 104 in P slice too large`) on specific frames; most still decode fine overall, but 4
  clips fail outright on every sampled frame and are skipped. If Phase 3/4 numbers look off for
  RLVS specifically, this is a known, logged gap, not a pipeline bug.
- AIRTLab's `train` split took ~18 minutes of the ~40-minute total build (1100s for 242 clips,
  ~4.5s/clip) despite 6 worker processes — its source videos are long, high-resolution (per-clip
  cost scales with source `n_frames` since every sampled index needs a fresh
  `cv2.set(CAP_PROP_POS_FRAMES)` seek). Expect Phase 6 (HPO, many training runs) to reuse this
  cache rather than rebuild it — a full `--dataset all --split all` rebuild is the ~40-minute
  number to budget against, not a per-run cost.
- `--purge-raw` was not exercised this session (disk had headroom: 30 GB free after the cache
  build, comfortably above the 15 GB DoD floor) so raw video is still on disk for all four
  datasets. It re-reads and verifies each committed clip before deleting its source file; exercise
  it in a later phase if disk pressure returns.

### Phase 3 — Augmentation + `tf.data` pipeline
Completed: 2026-10-08 · commit: `d9057ad`
Exit gate: `make verify PHASE=3`
Exit code: 0

```
ruff check . && pytest -m phase3 -q
All checks passed!
..............                                                           [100%]
14 passed, 36 deselected in 9.19s
```

Full guard including Phase 0/1/2 (`pytest -m "phase0 or phase1 or phase2 or phase3" -q`):
```
..................................................                       [100%]
50 passed in 217.49s (0:03:37)
```

Throughput on the real cache (`make_dataset("train", ...)`, 300 batches of 8, 112x112x16 clips):
```
clips=2400 elapsed_s=9.51 throughput_clips_per_s=252.3
```
Re-measured twice more across other runs this session: 357.4 and 351.8 clips/s; 252.3 was the
slowest observed and is the number recorded here, all comfortably above the 200 clips/s floor.

Before/after figure: `artifacts/figures/augmented_clips.png` rendered and visually confirmed,
flips/crops consistent within each clip's row, no frame-to-frame flicker (not committed, see
`.gitignore`, same as Phase 2's contact sheet).

DoD checklist: 9/9 met
Deviations from plan: BUILD_PLAN offers two ways to apply one shared parameter draw per clip,
`A.ReplayCompose` (apply to frame 0, replay on the rest) or Albumentations'
`additional_targets` (pass every frame as one call). Implemented with `ReplayCompose` first per
the BUILD_PLAN's "Mandatory" wording; measured 48-50 clips/s on the real cache, well under the
200 clips/s floor. Profiling (`cProfile`) showed `ReplayCompose.replay()` rebuilds its entire
transform tree via `inspect.signature` introspection on every single replay call, about 15x per
clip. Switched to the `additional_targets` form the plan explicitly names as the alternative:
same "one parameter draw per clip" guarantee (verified by the clip-consistency test), ~10x the
throughput (measured ~85 clips/s single-call micro-benchmark under ReplayCompose vs ~800 clips/s
under `additional_targets`), landing at 252-357 clips/s end to end through tf.data. Not a new
ADR since BUILD_PLAN names both forms as acceptable; recorded here as the reason one was picked
over the other.
Also: `cv2.setNumThreads(1)` is set at import time in `dataset.py`. Albumentations' OpenCV
backend defaults to an 8-thread pool per call; left alone, tf.data's own AUTOTUNE parallel map
calls each spawned their own 8-thread cv2 pool and oversubscribed the machine against itself,
measured at 34-50 clips/s with threading versus 85+ clips/s single-threaded. Threaded Python-level
augmentation (`ThreadPoolExecutor`) was also benchmarked and made things worse, not better (GIL
contention on the Python-heavy parts of the Albumentations call path), which is why this pipeline
relies on tf.data's `AUTOTUNE` map concurrency with `cv2.setNumThreads(1)` rather than any
additional multiprocessing layer.
Surprises / notes for the next session:
- The stale-cache guard compares `(n_frames, height, width)` from the cache file's HDF5 attrs
  against `configs/data.yaml`'s `pipeline:` block, not the full `config_sha`; a cache rebuilt with
  the same shape but a different `sampling` strategy would not be caught by `dataset.py`, only by
  `scripts/build_cache.py`'s own idempotency check. If Phase 2's preprocessing spec changes again
  in a way that keeps the shape fixed, this guard will not notice.
- `GaussNoise(noise_scale_factor=0.25)` generates noise at a coarser resolution and upsamples it;
  chosen for throughput, visually indistinguishable at 112x112 in the rendered figure, but not
  independently verified against the report's §4.2 intent beyond "still visibly noisy."
- The merge to `main` broke CI (`.github/workflows/ci.yml` only ever ran `pip install -e ".[dev]"`,
  which has no TensorFlow, so `tests/test_phase03_pipeline.py` failed to collect with
  `ModuleNotFoundError: No module named 'tensorflow'`; the local gate never caught this because the
  `safestreets` conda env already has TensorFlow installed). The `ml` extra can't fix it either,
  since it pins `tensorflow-macos`/`tensorflow-metal`, which have no Linux wheels for CI's
  `ubuntu-latest` runner. Fixed in a same-day `fix:` commit (`7918a97`) adding a portable `ci` extra
  (plain `tensorflow==2.16.2` + `albumentations==2.0.8`, both of which do publish Linux wheels) and
  pointing the workflow at `pip install -e ".[dev,ci]"`; verified in a scratch venv outside the
  `safestreets` conda env before pushing, and confirmed green on GitHub Actions run `37763477431`.
  Any future phase that adds a new runtime import needs to check it installs in CI, not just in the
  local `safestreets` env: the two environments diverged exactly this way once already.

### Sprint 1 — Feature store, models, training
Completed: 2026-10-10 · commit: `08e27d5` · supersedes Phase 4
Exit gate: `make verify SPRINT=1`
Exit code: 0

```
ruff check .
All checks passed!
pytest -m "phase0 or phase1 or phase2 or phase3 or sprint1" -q
..........................................................               [100%]
58 passed in 256.30s (0:04:16)
```

Feature extraction (`python scripts/extract_features.py --dataset all --split all`), all 11
`(dataset, split)` pairs, matching the Phase 2 cache's clip counts exactly:
```
airtlab/test: wrote 48 clips, 3.9 MB, 4.1s
airtlab/train: wrote 242 clips, 19.8 MB, 11.2s
airtlab/val: wrote 60 clips, 4.9 MB, 2.5s
rlvs/test: wrote 272 clips, 22.3 MB, 8.8s
rlvs/train: wrote 1402 clips, 114.9 MB, 46.6s
rlvs/val: wrote 273 clips, 22.4 MB, 9.4s
rwf2000/train: wrote 1600 clips, 131.2 MB, 52.7s
rwf2000/val: wrote 400 clips, 32.8 MB, 13.1s
ucfcrime/test: wrote 5 clips, 0.4 MB, 0.6s
ucfcrime/train: wrote 24 clips, 2.0 MB, 1.2s
ucfcrime/val: wrote 6 clips, 0.5 MB, 0.5s
```
Total `data/features`: 346 MB (`du -sh`), under the 0.6 GB budget. Idempotent re-run: all 11 files
report "up to date", total wall time 6.9s including Python/conda-run startup (the
`existing_features_valid` check itself, isolated in `tests/test_sprint1_model.py`, is under 5s as
required; most of the 6.9s is interpreter/import overhead common to every invocation of this
script, not re-extraction).

Production run (`python scripts/train.py --arch lstm_head`), RWF-2000 + RLVS train features,
early-stopped on `val_auc`, patience 8:
```
Epoch 1/50  ... val_auc: 0.8247
Epoch 8/50  ... val_auc: 0.8679   <- best, restored
Epoch 16/50 ... val_auc: 0.8274   (early stopped)
lstm_head_baseline: best_val_auc=0.8679
```
**`val_auc = 0.8679 > 0.80`** — meets the Sprint 1 go/no-go bar.

Comparison run (`python scripts/train.py --arch scratch`), scratch TimeDistributed CNN-LSTM, 12
epochs, live Phase 3 augmented pipeline on `rwf2000`:
```
Epoch 1/12 ... val_auc: 0.6041   <- best, never improved on again
Epoch 9/12 ... val_auc: 0.5318   (early stopped, patience 8)
scratch_comparison: best_val_auc=0.6041
```
Reported as measured, not cherry-picked: the scratch arm trained from random initialisation for
only 12 epochs essentially failed to generalise past epoch 1 (`val_precision`/`val_recall` collapse
to near-zero most epochs), which is the expected outcome of the §1 trade-off this sprint made,
not a bug — it is exactly the "4.4h per real run" cost the frozen-feature head exists to avoid.
This is the number that goes in the report as the trained-from-scratch comparison arm.

MLflow (`mlflow.search_runs`, experiment `safestreets-sprint1`):
```
  tags.mlflow.runName  metrics.best_val_auc params.arch
0  scratch_comparison              0.604079     scratch
1  lstm_head_baseline              0.867907   lstm_head
```
Both runs carry full param dicts (architecture, `pos_weight`, `git_sha`, seed) and a logged Keras
model artefact (`mlflow.keras.log_model`).

TensorBoard (`logs/fit/{lstm_head_baseline,scratch_comparison}`): `epoch_loss`, `epoch_accuracy`,
`epoch_auc`, `epoch_precision`, `epoch_recall` scalars plus per-layer weight histogram tags
(`dense`, `lstm`, `lstm_1`, `time_distributed`, ...) confirmed present in both runs' event files.

Model summaries: `artifacts/model_summary_lstm_head.txt` (770,881 params), `artifacts/model_summary_scratch.txt`
(221,793 params); both `build_model(cfg)` calls compile successfully for both archs
(`tests/test_sprint1_model.py::test_build_model_compiles_both_archs`).

Overfit test (32-clip synthetic subset, label-correlated features): train accuracy reaches ≥0.95
within 50 epochs — passes.

Label-alignment test: `load_feature_split`'s returned labels match the `clip_id`s they were loaded
under — passes.

Normalisation assertion (`test_normalization_matches_preprocess_input`): feature-store output for a
hand-computed single frame matches `backbone(preprocess_input(frame))` to `atol=1e-5`, and is
confirmed to diverge from the un-normalised `[0,255]` forward pass (not a vacuous comparison) —
passes.

DoD checklist: 11/11 met, with one caveat below.
Deviations from plan: Sprint 1's augmentation note option (b) was taken — the production head
trains on clean (eval-transform) features only; the `scratch` arm exercises the live augmented
pipeline end to end. Recorded in `docs/decisions/ADR-003-architecture.md`, as instructed. MLflow
3.16 (installed in `tf_env`/`safestreets`, newer than assumed when `BUILD_PLAN.md` was written) put
the plain filesystem `./mlruns` store into "maintenance mode" and refuses to run without
`MLFLOW_ALLOW_FILE_STORE=true`; set as an env-var default in `safestreets/training/train.py` to
keep the plan's intended file-store `mlruns/` deliverable rather than migrating to a SQLite
backend under this deadline.
Surprises / notes for the next session:
- **Disk budget caveat, not caused by this sprint:** `df -h ~` currently shows **~12-13 GB free**,
  below the Sprint 1 DoD's "20 GB or more free" line. This is a machine-wide disk-pressure issue,
  not a project one: `data/` (cache + features + manifests + raw) totals only ~7.5 GB, and this
  sprint's own addition (`data/features`, 346 MB) is comfortably inside its 0.6 GB budget. Flagging
  this rather than silently marking the DoD box: the free-space number is real and currently fails
  the stated bar, but nothing this sprint controls caused it. Worth the user clearing machine disk
  space before Sprint 5/6, which will add ONNX exports, notebooks, and Flask app artefacts.
- `MobileNetV2(input_shape=(112,112,3), ...)` logs `WARNING:tensorflow: input_shape is undefined or
  non-square, or rows is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be
  loaded as the default.` This is benign: MobileNetV2's conv kernels don't depend on spatial
  resolution, and `pooling='avg'` adapts to whatever spatial size comes out of the last conv block.
  Confirmed correct by the normalisation/forward-pass assertion test above; noting it here so a
  future session doesn't mistake the warning for a real problem.
- `tf.keras.optimizers.Adam` logs an M1/M2-slowness warning (recommending the legacy optimizer
  module); not switched, since both training runs already complete well inside budget (lstm_head:
  ~1 min total; scratch: ~33 min for 9 epochs, matching the CLAUDE.md-documented per-epoch range).
- The `scratch` comparison run took ~33 minutes for its 9 completed epochs (204-229s/epoch, close to
  CLAUDE.md's "under load" estimate, not the "quiet machine" one) — plan around the higher number for
  any future scratch-arch run on this machine, not the optimistic one.

### Template

```
## Phase N — <name>
Completed: YYYY-MM-DD · commit: <sha>
Exit gate: <command>
Exit code: 0

<pasted output — test summary line, key metrics>

DoD checklist: <n>/<n> met
Deviations from plan: <none | what changed and which ADR records it>
Surprises / notes for the next session:
- ...
```

---

## Blocked items

| Item | Phase | Blocked on | Needs the user? |
|---|---|---|---|
| XD-Violence I3D features | 1, 5 | Official release confirmed 38.3 GB, exceeds disk budget; permanently skipped, see docs/DATASETS.md | No (decided) |

---

## Decisions made (ADR index)

| ADR | Title | Status |
|---|---|---|
| 001 | Runtime stack: TF / Keras / tf2onnx / ORT versions | accepted (Phase 0) |
| 002 | Dataset split strategy and grouping keys | accepted (Phase 1) |
| 003 | Frozen-feature LSTM head as the production model | accepted (Sprint 1) |
| 004 | HPO budget and search space | pending (Sprint 3) |
| 005 | Fine-tuning + checkpoint selection | pending (Sprint 4) |
| 006 | Deployment target | pending (Sprint 5) |
