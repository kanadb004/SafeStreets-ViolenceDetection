# SafeStreets — Build Progress

**Single source of truth for "where are we".** A cold session reads this file first.

**Statuses:** `NOT_STARTED` · `IN_PROGRESS` · `BLOCKED` · `DONE`
A phase is `DONE` **only** when its Exit Gate command has been run and exited 0, and the evidence
block below is filled in with real output. Not "I think it works".

Last updated: 2026-09-13 · by: planning session · commit: `5ce07c4`

---

## Board

| # | Phase | Status | Exit gate run? | Session notes |
|---|-------|--------|----------------|---------------|
| 0 | Foundation: env, packaging, config, CI | `NOT_STARTED` | ✗ | — |
| 1 | Datasets, manifests, leakage-safe splits | `NOT_STARTED` | ✗ | — |
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
| *(none yet)* | | | |

---

## Decisions made (ADR index)

| ADR | Title | Status |
|---|---|---|
| 001 | Runtime stack: TF / Keras / tf2onnx / ORT versions | pending (Phase 0) |
| 002 | Dataset split strategy and grouping keys | pending (Phase 1) |
| 003 | Architecture defaults | pending (Phase 4) |
| 004 | HPO budget and search space | pending (Phase 6) |
| 005 | Fine-tuning + checkpoint selection | pending (Phase 7) |
| 006 | Deployment target | pending (Phase 11) |
