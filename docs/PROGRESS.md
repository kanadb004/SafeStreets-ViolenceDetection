# SafeStreets — Build Progress

**Single source of truth for "where are we".** A cold session reads this file first.

**Statuses:** `NOT_STARTED` · `IN_PROGRESS` · `BLOCKED` · `DONE`
A phase is `DONE` **only** when its Exit Gate command has been run and exited 0, and the evidence
block below is filled in with real output. Not "I think it works".

Last updated: 2026-09-13 · by: phase-1 session · commit: pending PR (branch `phase/01-datasets-manifests-splits`)

---

## Board

| # | Phase | Status | Exit gate run? | Session notes |
|---|-------|--------|----------------|---------------|
| 0 | Foundation: env, packaging, config, CI | `DONE` | ✓ | env pinned per ADR-001, PR pending |
| 1 | Datasets, manifests, leakage-safe splits | `DONE` | ✓ | XD-Violence BLOCKED, see docs/DATASETS.md; PR pending |
| 2 | Preprocessing → clip cache | `NOT_STARTED` | ✗ | — |
| 3 | Augmentation + `tf.data` pipeline | `NOT_STARTED` | ✗ | — |
| 4 | Model + training + MLflow + TensorBoard | `NOT_STARTED` | ✗ | — |
| 5 | Evaluation harness | `NOT_STARTED` | ✗ | — |
| 6 | Optuna HPO | `NOT_STARTED` | ✗ | — |
| 7 | AIRTLab women-specific fine-tuning | `NOT_STARTED` | ✗ | — |
| 8 | Person + gender attribution | `NOT_STARTED` | ✗ | — |
| 9 | ONNX export + streaming inference | `NOT_STARTED` | ✗ | — |
| 10 | Flask application rebuild | `NOT_STARTED` | ✗ | — |
| 11 | Packaging, reproducibility, reporting | `NOT_STARTED` | ✗ | — |

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
Completed: 2026-09-13 · commit: pending PR (branch `phase/01-datasets-manifests-splits`)
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
- XD-Violence is `BLOCKED`: the only official distributions (a Xidian University OneDrive share
  and a Baidu Netdisk share) both need an interactive browser session or an account this project
  doesn't have. Manual steps are in `docs/DATASETS.md`. Phase 5's frame-level AUC work will need
  this fetched by hand before it can run.
- `manifest.py`'s OpenCV probe pass takes ~100s for the current 4336 clips (~25ms/clip); expect
  this to scale roughly linearly if later phases add more sources to the manifest.

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
| XD-Violence I3D features | 1, 5 | Manual browser download from OneDrive/Baidu, see docs/DATASETS.md | Yes |

---

## Decisions made (ADR index)

| ADR | Title | Status |
|---|---|---|
| 001 | Runtime stack: TF / Keras / tf2onnx / ORT versions | accepted (Phase 0) |
| 002 | Dataset split strategy and grouping keys | accepted (Phase 1) |
| 003 | Architecture defaults | pending (Phase 4) |
| 004 | HPO budget and search space | pending (Phase 6) |
| 005 | Fine-tuning + checkpoint selection | pending (Phase 7) |
| 006 | Deployment target | pending (Phase 11) |
