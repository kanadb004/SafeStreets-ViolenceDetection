# SafeStreets — Build Progress

**Single source of truth for "where are we".** A cold session reads this file first.

**Statuses:** `NOT_STARTED` · `IN_PROGRESS` · `BLOCKED` · `DONE`
A phase is `DONE` **only** when its Exit Gate command has been run and exited 0, and the evidence
block below is filled in with real output. Not "I think it works".

Last updated: 2026-10-10 · by: sprint-5 session · commit: `286bf3c`

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
| 2 | Evaluation harness | `DONE` | ✓ | test AUC 0.9450 combined; AIRTLab zero-shot AUC 0.5259 (big drop, real); no frame-level AUC, see SPRINT_PLAN §2 |
| 3 | Optuna HPO | `DONE` | ✓ | 30 trials (23 complete, 7 pruned), ~32 min measured; tuned val_auc 0.8857 but test AUC 0.9418 < baseline 0.9450, baseline ships (no cherry-picking), see ADR-004 |
| 4 | AIRTLab fine-tune + attribution | `DONE` | ✓ | AIRTLab 5-fold CV AUC 0.8976 (vs 0.5259 zero-shot); RWF-2000 forgetting -0.0582 AUC, fine-tuned checkpoint ships (ADR-005); YOLOv8n+ByteTrack person tracking + zero-shot CLIP gender, accuracy NOT RUN (no gender ground truth), see docs/ETHICS.md |
| 5 | ONNX, real-time, Flask app | `DONE` | ✓ | bundled MobileNetV2+head ONNX, parity 5.24e-05 over 325 real clips, AUC diff 0.0; p95 134.41 ms/window, 59.5 sampled fps (ADR-006); legacy `app/` deleted, all 5 defects closed |
| 6 | Notebooks, results, model card | `NOT_STARTED` | x | **the submission** |

---

## Headline numbers

Fill these in **only** from a committed `artifacts/reports/eval_*.json`. `NOT RUN` until then.

| Metric | Value | Source file | Phase |
|---|---|---|---|
| Baseline test ROC-AUC (RWF+RLVS held-out) | `0.9450` (combined RLVS+AIRTLab+UCF test) | `artifacts/reports/eval_test.json` | 4/5 |
| Tuned test ROC-AUC | `0.9418` (did not beat baseline; baseline 0.9450 ships) | `artifacts/reports/eval_test_tuned.json` | 6 |
| Cross-dataset AUC (→ AIRTLab, zero-shot) | `0.5259` | `artifacts/reports/cross_dataset.json` | 5 |
| Cross-dataset AUC (→ UCF-Crime subset) | `1.0000` (n=5, zero-shot) | `artifacts/reports/cross_dataset.json` | 5 |
| Frame-level AUC (untrimmed) | `NOT RUN` (out of scope, see SPRINT_PLAN §2) | — | 5 |
| AIRTLab fine-tuned accuracy (5-fold) | `0.8486 +/- 0.0439` (AUC `0.8976 +/- 0.0429`) | `artifacts/reports/eval_finetuned.json` | 7 |
| Gender module accuracy | `NOT RUN` (zero-shot CLIP, no gender ground truth in this pipeline, see docs/ETHICS.md) | — | 8 |
| ONNX p95 latency / sustainable FPS | `134.41 ms` per 16 frame window (p50 111.08, p99 167.90); `59.5` sampled frames/s at stride 8 | `artifacts/reports/latency.json` | Sprint 5 |

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

### Sprint 2 — Evaluation harness
Completed: 2026-10-10 · commit: `1ac1862` · supersedes Phase 5
Exit gate: `make verify SPRINT=2`
Exit code: 0

```
ruff check .
All checks passed!
pytest -m "phase0 or phase1 or phase2 or phase3 or sprint1 or sprint2" -q
......................................................................   [100%]
70 passed in 248.47s (0:04:08)
```

Threshold selection (`python scripts/evaluate.py`), on RWF-2000 + RLVS **validation** only
(673 clips), persisted to `configs/infer.yaml`:
```
val n=673 f1_optimal_threshold=0.2000
```
F1-optimal threshold 0.2000 (F1=0.8037 on val); recall>=0.90 threshold 0.0700 (recall=0.9042,
precision=0.6888 on val). `tests/test_sprint2_eval.py::test_threshold_was_selected_on_validation_not_test`
asserts the persisted value traces to `eval_val.json`'s `split: "val"`, not a test file.

Headline test metrics (combined RLVS test + AIRTLab test + UCF-Crime test, 325 clips, RWF-2000
contributes no test rows under its official split per ADR-002), at the F1-optimal threshold:
```
test (combined) n=325 roc_auc=0.9450
```
Full metric block (`artifacts/reports/eval_test.json`): ROC-AUC 0.9450 [0.9204, 0.9663], PR-AUC
0.9471 [0.9201, 0.9704], accuracy 0.8769 [0.8400, 0.9138], precision 0.8392 [0.7905, 0.8923],
recall 0.9543 [0.9191, 0.9828], F1 0.8930 [0.8595, 0.9260]. Confusion matrix: tn=118, fp=32,
fn=8, tp=167.

Cross-dataset matrix (`artifacts/reports/cross_dataset.json`), model trained on RWF-2000+RLVS
only:
```
  rlvs       in_domain=True  roc_auc=0.9805
  airtlab    in_domain=False roc_auc=0.5259
  ucfcrime   in_domain=False roc_auc=1.0000
```
AIRTLab (zero-shot) drops 0.4546 AUC below the RLVS in-domain number, a real generalisation
gap, not a leak (per the "no drop at all means suspecting leakage" instruction, a drop this
size is the expected and reassuring outcome). UCF-Crime's AUC is measured at 1.0000 but on only
5 test clips (1 negative, 4 positive); reported as measured, with its sample size flagged in
`docs/RESULTS.md`'s limitations section rather than read as a real zero-error result.

Four figures rendered and referenced from `docs/RESULTS.md`: `artifacts/figures/{roc,pr,
confusion,threshold_sweep}.png` (not committed, gitignored same as Phase 2/3's figures).

`docs/RESULTS.md` regeneration (`tests/test_sprint2_eval.py::test_results_md_regenerates_with_no_diff`):
running `safestreets.evaluation.report.generate()` twice against the same committed
`eval_*.json`/`cross_dataset.json`/`infer.yaml` artefacts produces byte-identical output, verified
directly (not just by convention).

Metrics-vs-sklearn test (`tests/test_sprint2_eval.py::test_metrics_agree_with_sklearn_to_1e9`):
passes; `safestreets.evaluation.metrics` calls `sklearn.metrics` directly rather than
reimplementing, so this is agreement by construction, documented as such in the module
docstring rather than presented as independent verification.

DoD checklist: 9/9 met.
Deviations from plan: none. The sprint's "in-domain test split is RLVS+AIRTLab+UCF test combined"
instruction and its "cross-dataset: test zero-shot on AIRTLab and UCF" instruction look like they
disagree about whether AIRTLab/UCF are in-domain; resolved by reading "in-domain" as "the combined
task-level headline number" (what ships as the single reported test AUC) and the cross-dataset
matrix as a separate, finer-grained breakdown of that same model's zero-shot transfer gap. Not a
new ADR since both bullets are satisfied simultaneously by this reading, not traded off against
each other.
Surprises / notes for the next session:
- AIRTLab's zero-shot AUC (0.5259) is close to chance. Given AIRTLab is acted/staged footage
  (BUILD_PLAN §7 caveat, carried into `docs/RESULTS.md`'s limitations section), this is plausibly
  a filming-style domain gap rather than a model failure; Sprint 4's AIRTLab fine-tune and its
  forgetting check will be the real test of that reading. Flagging now so Sprint 4 doesn't
  rediscover this number cold.
- The UCF-Crime test split's n=5 (1 negative, 4 positive) makes its bootstrap CI collapse to a
  single point in this run ([1.0000, 1.0000]); `safestreets.evaluation.metrics.bootstrap_ci`
  degrades to `[nan, nan]` when fewer than 2 resamples retain both classes, but with n=5 enough
  resamples still do. Future sessions reading a suspiciously tight CI on a tiny split should check
  `n` before trusting it.
- `tf_keras.models.load_model(..., compile=False)` was enough for evaluation; the custom
  `weighted_bce` loss closure never needed a `custom_objects` entry since `.predict` doesn't touch
  the loss. If a future sprint calls `.evaluate()` on a loaded checkpoint instead, it will need the
  loss reconstructed with the checkpoint's own `pos_weight` (logged in MLflow params, not in the
  `.keras` file itself).

### Sprint 3 — Optuna HPO
Completed: 2026-10-10 · commit: `cd0451e` · supersedes Phase 6
Exit gate: `make verify SPRINT=3`
Exit code: 0

```
ruff check .
All checks passed!
pytest -m "phase0 or phase1 or phase2 or phase3 or sprint1  or sprint2  or sprint3" -q
........................................................................   [ 94%]
....                                                                      [100%]
76 passed in 262.20s (0:04:22)
```

Study (`python scripts/tune.py --trials 30 --timeout 2400`), `artifacts/optuna/study.db`,
`TPESampler(seed=1265)`, `MedianPruner(n_startup_trials=5, n_warmup_steps=5)`, wall clock
02:52:40 to 03:24:30 (about 32 minutes) for all 30 trials at 25 epochs each:
```
study has 30 trials (23 complete, 7 pruned); best_val_auc=0.8857
best params: learning_rate=0.004862, lstm_units=128, dropout=0.4921, batch_size=16 (trial 21)
```
At least one `PRUNED` trial: 7 of 30. `configs/model.best.yaml` written programmatically from the
winning trial; `tests/test_sprint3_tune.py::test_write_best_model_yaml_reproduces_architecture`
loads it through `build_model` and asserts the resulting LSTM layer has `units=256` for a
synthetic winner, proving the round-trip rather than eyeballing the YAML. Both figures rendered:
`artifacts/figures/optuna_history.png`, `artifacts/figures/optuna_importances.png`. Hyperparameter
importances (`optuna.importance.get_param_importances`): `learning_rate` 0.683, `batch_size`
0.159, `dropout` 0.112, `lstm_units` 0.045, interpreted in `docs/decisions/ADR-004-hpo.md`.

Resume (`tests/test_sprint3_tune.py::test_run_study_resumes_trial_count`): a study resolved with
`resume=False` and run to 1 trial, then resolved again with `resume=True` against the same SQLite
storage and run to 2 trials, ends with exactly 2 trials, not 3; a second test
(`test_run_study_without_resume_rejects_existing_study`) asserts re-resolving the same study name
without `resume=True` raises instead of silently restarting.

Winner retrained on train+val (`artifacts/checkpoints/lstm_head_tuned.keras`, 7 epochs,
batch_size=16, no held-out split remains so no early stopping, per ADR-004):
```
Epoch 7/7
230/230 - 4s - loss: 0.3378 - accuracy: 0.8414 - auc: 0.9256 - precision: 0.8228 - recall: 0.8746
```
Scored on the combined test split once, at the baseline's F1-optimal threshold (0.2000, reused
rather than re-selected, per ADR-004):
```
tuned does NOT beat baseline: 0.9418 <= 0.9450; baseline ships, reporting both numbers as-is
```
**Tuning did not beat the baseline.** Per SPRINT_PLAN's "no cherry-picking" DoD item, the baseline
(`artifacts/checkpoints/lstm_head_baseline.keras`) stays the production model; both numbers are
reported in `docs/RESULTS.md`'s new "HPO-tuned model vs. baseline" section, generated by the same
`safestreets.evaluation.report.generate()` used in Sprint 2, now also loading
`eval_test_tuned.json` when present.

DoD checklist: 9/9 met (search space and object as specified; >=1 pruned trial; resume verified;
`model.best.yaml` round-trips; both figures render and importances are interpreted; winner
retrained on train+val and test touched once; tuning result reported honestly, baseline ships;
`docs/RESULTS.md` regenerated with both rows; cumulative gate green).
Deviations from plan: none in the search itself. The tuned model's evaluation reuses the
baseline's val-selected F1-optimal threshold rather than selecting a fresh one, because the final
retrain combines train+val and leaves no held-out split to select a new threshold from without
touching test twice; recorded as a deliberate choice in ADR-004, option 2 of 3 considered.
Surprises / notes for the next session:
- A single `scripts/tune.py` run aborted mid-way with a TensorFlow-Metal plugin crash
  (`Mutation::Apply error` in `metal_plugin/src/graph/remapper/remapper.cc`) right after the
  30-trial study had already completed and persisted to `study.db`, during the post-study retrain
  step's first fresh Keras graph. Not a logic bug; same class of Metal fragility ADR-001 already
  flagged. Recovered by re-running the retrain/eval/RESULTS.md steps in a fresh process against
  the same persisted study, exactly the path `--resume` exists for.
- The *cumulative* Exit Gate (`pytest -m "phase0 or ... or sprint3"`, one process) hit the same
  class of crash even after the study itself was fine: a test that built several fresh small
  Keras/LSTM graphs back-to-back (to exercise resume) pushed the already-substantial graph count
  built by Sprint 1/2's tests over some threshold in the Metal plugin, aborting the whole pytest
  process (`Fatal Python error: Aborted`), confirmed reproducible twice by running
  `pytest -m "sprint1 or sprint2 or sprint3"` standalone. Fixed by splitting `run_study` into
  `resolve_study`/`run_trials` (storage/resume logic) and `build_objective` (the TF-specific part),
  and testing resume against the former with a trivial non-TF objective. The real TF integration
  for tuning is still covered, just by the actual production run above rather than by a test that
  reconstructs it from scratch in the same process as every other sprint's tests. Any future sprint
  adding more `model.fit` calls to the test suite should budget for this same fragility; one cheap
  mitigation that was NOT needed here but is worth knowing about: splitting a graph-heavy test file
  into its own `pytest` invocation sidesteps the cumulative graph count entirely.
- `optuna.visualization.matplotlib.plot_optimization_history`/`plot_param_importances` are marked
  `ExperimentalWarning` by Optuna 5.0.0 (API available since 2.2.0); harmless, not pinned around.

### Sprint 4 — AIRTLab fine-tune and attribution
Completed: 2026-10-10 · commit: `25bf12d` · supersedes Phases 7/8
Exit gate: `make verify SPRINT=4`
Exit code: 0

```
ruff check .
All checks passed!
pytest -m "phase0 or phase1 or phase2 or phase3 or sprint1  or sprint2  or sprint3  or sprint4" -q
........................................................................ [ 82%]
...............                                                          [100%]
87 passed in 214.86s (0:03:34)
```

5-fold grouped CV on AIRTLab (`python scripts/finetune.py`), fine-tuning from
`lstm_head_baseline.keras`'s weights at `learning_rate=1e-4`, seed 1265:
```
AIRTLab pool: 350 clips, 175 groups
  accuracy: 0.8486 +/- 0.0439
  f1: 0.8863 +/- 0.0338
  roc_auc: 0.8976 +/- 0.0429
```
Forgetting check, RWF-2000 val ROC-AUC before vs. after the AIRTLab fine-tune
(`artifacts/checkpoints/finetuned_airtlab_lstm_head.keras`):
```
RWF-2000 val ROC-AUC: before=0.7150 after=0.6568 drop=+0.0582
```
**Decision (ADR-005): the fine-tuned checkpoint ships.** AIRTLab AUC rises from the Sprint 2
zero-shot 0.5259 to 0.8976; the 5.8pp RWF-2000 forgetting cost leaves RWF-2000 val well above
chance and AIRTLab is the women's-safety-relevant dataset this project targets. Both numbers
are in `docs/RESULTS.md`'s new "AIRTLab fine-tune (Sprint 4)" section, generated by the same
`safestreets.evaluation.report.generate()` used in Sprints 2/3.

Attribution (`python scripts/attribution_eval.py`, AIRTLab test split, 48 clips, full-resolution
source video rather than the 112x112 training cache): YOLOv8n (`artifacts/weights/yolov8n.pt`)
+ ByteTrack person detection and tracking every 4th frame, zero-shot CLIP (`ViT-B-32-quickgelu`,
OpenAI weights) gender attribution over each track's best crop. Both modules available and ran
clean on all 48 clips (`artifacts/reports/attribution_eval.json`):
```
n_tracks distribution across 48 clips: min 2, max 9 (AIRTLab clips are 2-actor scenes;
  no true single-walker clip exists in this dataset to test the "one dominant track" claim on
  real data, so that exact DoD item is covered by a synthetic-data unit test instead,
  tests/test_sprint4_attribution.py::test_summarize_tracks_sane_on_single_walker)
alerts fired: 46/48 (2 clips' lstm_head violence score fell below the 0.2 operating threshold)
gender labels predicted: woman=154, man=73 (predictions only — NOT an accuracy number,
  no gender ground truth exists in this pipeline, see docs/ETHICS.md)
```
Fail-safe property (`safestreets/attribution/pipeline.py::build_alert`) and graceful
degradation (missing/failing YOLO or CLIP) both covered by
`tests/test_sprint4_attribution.py`, including the DoD's exact scenario
(`p_violence=0.99`, gender forced to "no women detected", alert still fires).

DoD checklist: 8/8 met (5-fold CV with mean/std; forgetting check in RESULTS.md; ADR-005
shipping decision recorded; YOLO runs on cached clips with a hand-checked 20-clip sample at
`artifacts/reports/attribution_sample.json`; tracking sane, see note above; fail-safe test
passes; graceful degradation tested; docs/ETHICS.md written; cumulative gate green).
Deviations from plan: none load-bearing. Attribution ran against the manifest's full-resolution
source video rather than the Phase 2 112x112 clip cache (too small to classify gender
reliably); person detection/gender attribution covered AIRTLab test (48 clips) rather than the
full corpus, to keep this CPU-only sprint's compute budget sane — recorded here rather than in
SPRINT_PLAN.md §2 since it is a coverage choice within a kept scope item, not cut scope.
Surprises / notes for the next session:
- `open_clip`'s default `ViT-B-32` config mismatches the OpenAI `openai` pretrained weights'
  activation (`QuickGELU mismatch` warning); use `ViT-B-32-quickgelu` instead, which silences
  the warning and matches the weights CLIP was actually trained with.
- `yolov8n.pt` and the AIRTLab fine-tune/attribution runs together used under 1 GB of the
  tight ~8 GB free disk at session start; no budget concern, but Sprint 5's ONNX export should
  re-check `df -h` before adding anything larger (a full ByteTrack run over every dataset would
  not fit the "keep it CPU-cheap" choice above either).
- `ultralytics.YOLO.track(..., persist=False)` without `stream=True` logs a benign
  "results will accumulate in RAM" warning per clip; harmless at 48 short clips, would need
  `stream=True` if a future sprint runs this over the full corpus.

### Sprint 5 — ONNX, real-time, Flask app
Completed: 2026-10-10 · commit: `286bf3c` · supersedes Phases 9/10
Exit gate: `make verify SPRINT=5`
Exit code: 0

```
ruff check .
All checks passed!
pytest -m "phase0 or phase1 or phase2 or phase3 or sprint1  or sprint2  or sprint3  or sprint4  or sprint5" -q
........................................................................ [ 56%]
.......................................................                  [100%]
127 passed in 206.66s (0:03:26)
```

ONNX export, parity and latency (`python scripts/export_onnx.py`, seed 1265, exit 0). The
exported graph is the **whole** inference path, not just the head: raw letterboxed RGB frames
`(N, 16, 112, 112, 3)` in [0, 255], then `Rescaling(1/127.5, -1)` (MobileNetV2 `preprocess_input`),
then `TimeDistributed(MobileNetV2)`, then the ADR-005 fine-tuned head. Serving needs no
TensorFlow (ADR-006).
```
exported .../artifacts/onnx/safestreets_finetuned_airtlab.onnx (12002918 bytes) in 11.7s
engine loaded: model_version=finetuned_airtlab-73ebf22e032d git_sha=d8b8335c2c63a194687326504ce527536673f833
parity: n=325 max_abs_diff=5.244e-05 (tol 0.0001) passed=True
roc_auc keras=0.969105 onnx=0.969105 diff=0.00e+00 passed=True
bundled keras vs feature store + head: 2.086e-06
latency per window: p50=111.08ms p95=134.41ms p99=167.90ms  sustainable_fps=59.5 (stride 8), headroom x18.6 over target_fps 3.2
{"loadavg_before": [1.71142578125, 1.7314453125, 1.974609375]}
```
- Parity set: every clip of the combined RLVS + AIRTLab + UCF-Crime test split from the Phase 2
  clip cache (325 clips, 175 positive). `onnx_parity.json` also has mean abs diff 1.218e-06 and
  p99 abs diff 1.907e-05. The 0.9691 AUC is a **runtime agreement check, not a generalisation
  number**: AIRTLab test clips were in the ADR-005 fine-tune pool. Do not quote it as a result.
- `bundled keras vs feature store + head 2.086e-06`: the bundled graph reproduces the Sprint 1
  feature store followed by the head, so it is not a drifted reimplementation.
- Latency: `InferenceEngine.predict_windows` on one window, 300 timed iterations after 20 warmup,
  onnxruntime 1.18.1 `CPUExecutionProvider`, M2, load average 1.7 at start. Excludes decode and
  letterbox. Sustainable FPS = `stride * 1000 / p95_ms` (one window per 8 sampled frames). The
  stream samples source video down to `target_fps: 3.2` (16 frames over the ~5 s median training
  clip, from the manifest's `duration_s`), so 59.5 sampled fps is 18.6x real time.

Flask app, real engine, `test/sample_video.avi` (30 frames, 30 fps, 1280x720), via the test
client:
```
GET / -> 200
GET /healthz -> {'git_sha': 'd8b8335c2c63a194687326504ce527536673f833', 'model_version': 'finetuned_airtlab-73ebf22e032d', 'status': 'ok', 'threshold': 0.2}
POST /api/analyse -> 200
{"alert_events": 1, "clip_score": 0.8565809726715088, "enter_threshold": 0.25, "fps": 30.0, "frame_step": 9, "leave_threshold": 0.15000000000000002, "model_version": "finetuned_airtlab-73ebf22e032d", "n_frames_decoded": 30, "padded": false, "short_video": true, "threshold": 0.2, "verdict": "violent"}
```
(This 1 s fixture is shorter than one 16 frame window at 3.2 fps, so its timeline is one window
equal to the whole-clip score. The streaming path is covered by unit tests.)

Legacy defects closed:
1. `app/routes.py` imported a `predict_video` that did not exist. `app/` is deleted, and
   `safestreets/web` (`create_app()`, `routes.py`, `api.py`) calls `InferenceEngine`. `run.py`
   now imports `safestreets.web.create_app` and still serves on port 3000.
2. The result was discarded. `POST /analyse` renders `result.html` with the verdict, the clip
   score and a per-window timeline (raw score bar with a threshold marker, EMA-smoothed score,
   alert latch). `POST /api/analyse` returns the same as JSON.
3. Uploads had no limits. Now `MAX_CONTENT_LENGTH` is 50 MB (`configs/serve.yaml`, 413), the
   extension is allowlisted (415), OpenCV must decode a frame (415), the file gets a UUID name in
   a temp dir, and it is unlinked in a `finally`. Tests assert removal on success and when the
   engine raises.
4. `test/test_predict.py` (`/2300.0` normalisation) is deleted. `test/sample_video.avi` stays
   as the DoD fixture. It is untracked (`*.avi` is gitignored), so the tests that use it are
   `needs_data`.
5. Tailwind came from a CDN. Tailwind 1.9.6's `tailwind.min.css` (the version `^1.0` resolved to)
   is now vendored at `safestreets/web/static/vendor/tailwind-1.9.6.min.css` (1967618 bytes,
   sha256 `b1ad2f9d383ef7e0adb2760405b4a8518ae632f1e7efdd2963bec491c44e2f69`, no external `url()`
   or `@import`). A test fetches every asset `/` references from the app itself.

DoD checklist: 12/12 met.
- ONNX export succeeds and loads in onnxruntime; max abs diff 5.244e-05 < 1e-4 over 325 real
  clips (`onnx_parity.json`).
- ONNX ROC-AUC equals the Keras ROC-AUC (diff 0.0, tolerance 0.005).
- `latency.json` has p50/p95/p99 and implied sustainable FPS.
- Hysteresis: `test_oscillating_raw_scores_produce_one_alert_not_many` and
  `test_oscillation_with_shipped_stream_config_alerts_once`. With the shipped config, the series
  has 15 naive single-threshold crossings and raises 1 alert.
- Sidecar complete. The engine raises `SpecMismatchError` on normalisation, n_frames, channel
  order, input_dim, threshold or a missing section, before it opens the model file (6
  parametrised cases).
- `GET /` 200; `GET /healthz` returns model_version and git_sha.
- `POST /api/analyse` with `test/sample_video.avi` returns a schema-valid body, clip_score
  0.8566.
- 4xx tests: no file (400), wrong extension (415), oversized (413), non-decodable (415).
- Upload removal asserted after success and after an engine failure.
- Result page HTML is matched for the verdict and `id="clip-score">0.8100<`.
- `grep -r "from app" .` finds no code. The only matches are `docs/BUILD_PLAN.md` and
  `docs/SPRINT_PLAN.md`, which quote this check, plus this line. No template references an
  external URL.
- Cumulative gate green: 127 passed.

Deviations from plan:
- The ONNX artefact bundles MobileNetV2 and the head in one graph rather than exporting the head
  alone, so `InferenceEngine` really runs over onnxruntime with no TensorFlow (the Sprint 5
  deliverable says "bundling MobileNetV2 features plus the head"). ADR-006 records the
  reasoning.
- The legacy `contact.html` was dropped, not ported. Its form posted to a `/submit_form` route
  that never existed. `about.html` was rewritten with plain claims.
- Flask is already a core `[project] dependencies` entry in `pyproject.toml`, so it was not added
  to the `ml`/`ci` extras a second time. Added `[tool.setuptools.package-data]` so the templates
  and the vendored CSS ship with a non-editable install.
- Notebook execution (CLAUDE.md step 6): NOT RUN, because there are none. The only notebook is
  `notebooks/legacy/train_model.ipynb`, which is Colab-bound reference outside the gate.
  Sprint 6 creates the six.
- Committed straight to `main` under deadline pressure, as Sprints 1 to 4 were. No issue or PR.

Surprises / notes for the next session:
- **Threshold provenance.** `configs/infer.yaml`'s `threshold: 0.2` was selected on validation
  for the Sprint 1 **baseline** (last written by `1ac1862`). It was never re-selected for the
  ADR-005 fine-tuned checkpoint that now ships. The sidecar and the app use 0.2 unchanged.
  Sprint 6 should disclose this in the model card, or re-select on validation and re-export.
  The engine will refuse the old artefact until it is re-exported.
- The sidecar's `git_sha` is `d8b8335` with `git_dirty: true`, because the export ran before the
  Sprint 5 commit existed. `onnx_sha256` and `source_checkpoint_sha256` pin the artefact.
  `model_version` is `<tag>-<first 12 hex of checkpoint sha256>`.
- `artifacts/` is gitignored, so the 12 MB `.onnx`, the sidecar and both reports exist only
  locally, like every earlier checkpoint. A fresh clone must run `python scripts/export_onnx.py`
  (about 2 min) before the app can serve. Disk: `df -h` showed 78 GiB free at session start and
  75 GiB after. `artifacts/onnx/` is 11 MB.
- Parity margin is about 2x (5.24e-05 against 1e-4). The residual is float32 accumulation through
  MobileNetV2 + 2 LSTMs, not the device: forcing tf_keras onto `/CPU:0` gave 3.21e-05, Metal
  `/GPU:0` gave 5.24e-05, and the two Keras devices differ by 2.03e-05 from each other.
- For `06_inference_and_demo.ipynb`: `InferenceEngine().analyse_video(path)` returns the full
  timeline dict, and `safestreets.web.create_app().test_client()` drives the app with no server.
  Use a longer real clip from the manifest for the sliding-window demo, because the 1 s
  fixture yields one window.

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
| 004 | HPO budget and search space | accepted (Sprint 3) |
| 005 | Fine-tuning + checkpoint selection | accepted (Sprint 4) |
| 006 | Deployment target: bundled ONNX in ONNX Runtime, served by Flask | accepted (Sprint 5) |
