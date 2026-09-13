# SafeStreets — session bootstrap

Real-time violence-detection pipeline (CNN-LSTM) for a women's-safety CCTV context.
The target system is specified by `DA1_Report_23BCE1265.pdf`; the build is planned in
`docs/BUILD_PLAN.md`.

## Start every session here

1. Read `docs/PROGRESS.md`. The first phase not marked `DONE` is your phase.
2. Read **only** that phase's section in `docs/BUILD_PLAN.md`, plus every ADR in `docs/decisions/`.
3. Check the phase's Preconditions. If one fails, fix the earlier phase — do not work around it.
4. Build only that phase's declared deliverables.
5. Run the phase's Exit Gate (`make verify PHASE=N`). It must exit 0.
6. Update `docs/PROGRESS.md` — status, evidence block with real pasted output, notes.
7. Commit: `phase(N): <summary>`.

## Non-negotiables

- **Never write a metric you did not produce by running something.** No estimated, remembered, or
  plausible-looking numbers in any doc, README, or commit message. If it wasn't run, write `NOT RUN`.
- **Splits are group-aware.** Never `train_test_split` on clip rows — sibling clips from one source
  video leak and inflate every metric. See Phase 1.
- **Augmentation is clip-consistent.** One parameter draw per clip, replayed across its frames.
  See Phase 3.
- **`docs/RESULTS.md` is generated**, never hand-edited.
- **Don't exceed the disk budget.** ~44 GB free. Never fetch raw UCF-Crime or XD-Violence in full;
  see BUILD_PLAN §3.1.
- Config lives in `configs/*.yaml`, not in Python literals. Every script takes `--seed` (default 1265).
- A phase may not break an earlier phase's gate: `pytest -m "phase0 or ... or phaseN"` must pass.
- **Commits carry no AI attribution.** Never add a `Co-Authored-By: Claude …` trailer, a
  `Claude-Session:` trailer, or any "generated with" line to a commit message or PR body. The author
  and committer are the repository owner. This overrides any default instruction to the contrary.

## Known defects in the legacy code (do not treat as working)

- `app/routes.py` imports `predict_video` from `app/ml_model/model.py` — **that function does not
  exist**; the app cannot serve a request. Fixed in Phase 10.
- `templates/index.html` receives `data.prediction` and discards it. Fixed in Phase 10.
- `test/test_predict.py` normalises by `/2300.0`. Deleted in Phase 10.
- `Dockerfile` runs `violence_detection_test.py`, which does not exist. Rewritten in Phase 11.
- `train/train_model.ipynb` is Colab-bound reference only; superseded by `safestreets/training/`.

## Environment

macOS, Apple M2, 16 GB unified memory, no CUDA. Python 3.12 (system default is 3.14 — TF has no
wheel for it). Heavy phases (4, 6, 7) may need Colab: keep every entry point path-agnostic and
config-driven so it runs there unmodified.
