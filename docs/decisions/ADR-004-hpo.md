# ADR-004: Optuna TPE search over the lstm_head, and why the baseline still ships

- **Status:** accepted
- **Date:** 2026-10-10
- **Phase:** Sprint 3

## Context

`docs/SPRINT_PLAN.md` Sprint 3 calls for a 30-trial Optuna Bayesian search over four
hyperparameters named in report §4.4: `learning_rate` (log-uniform, 1e-5 to 1e-2), `lstm_units`
(categorical, {64,128,256,512}), `dropout` (uniform, 0.2 to 0.6), `batch_size` (categorical,
{16,32,64}). Single objective: maximise `val_auc`. This is cheap because Sprint 1 froze the CNN
and trains only the LSTM head on precomputed `(N, 16, 1280)` features (ADR-003); ADR-003's own
measurement put a 30-trial, 25-epoch study at 24.5 minutes.

A second question this ADR settles: once the winning hyperparameters are chosen on train/val, how
is the final model fit and compared to the Sprint 1/2 baseline, given the deadline's "test touched
once" rule.

## Options considered

1. **Retrain the winner on train+val, evaluate at a freshly-selected threshold.** More
   standard, but selecting a new threshold needs a held-out split, and none remains once
   train+val are combined for the final fit; selecting one from a subset of training data would
   mean touching data twice, inconsistent with Sprint 2's val-only threshold-selection rule.
2. **Retrain the winner on train+val, evaluate at the baseline's already-selected F1-optimal
   threshold** (`configs/infer.yaml`, selected on val in Sprint 2 and never touched again here).
   Keeps the two rows in `docs/RESULTS.md` directly comparable (same threshold, same test set,
   scored once each) at the cost of not re-optimising the threshold for the tuned model
   specifically.
3. **Skip retraining on train+val; keep the winning trial's own checkpoint** (trained on train
   only, early-stopped on val). Avoids the threshold question entirely, but then the final model
   never trains on the val clips at all, wasting signal the baseline also didn't have, so it is
   not a fair upgrade over the baseline either.

## Decision

**Option 2.** `safestreets.training.tune.retrain_winner_on_train_plus_val` retrains the winning
architecture on train+val combined, for exactly `best_epoch` epochs (the epoch at which the
winning trial peaked during tuning, read from the trial's `best_epoch` user attribute), since no
validation signal survives to early-stop against. `safestreets.training.tune.evaluate_tuned_winner`
then scores it on the combined test split, and on the per-dataset cross-dataset matrix, at the
baseline's F1-optimal threshold, once. `docs/RESULTS.md`'s "HPO-tuned model vs. baseline" section
is generated from both rows.

Search mechanics: `optuna.samplers.TPESampler(seed=1265)`, `optuna.pruners.MedianPruner
(n_startup_trials=5, n_warmup_steps=5)`, SQLite storage at `artifacts/optuna/study.db`
(`study_name="safestreets-sprint3-lstm-head"`). `scripts/tune.py --resume` reruns against the same
storage with `load_if_exists=True` and runs only `n_trials - len(existing trials)` additional
trials, so an interrupted study continues instead of restarting.

## Consequences

**Tuning did not beat the baseline, so the baseline ships.** Measured, 2026-10-10:

| | ROC-AUC (combined test) | 95% CI |
|---|---|---|
| Baseline (`lstm_head_baseline`, Sprint 1 defaults: units=128, dropout=0.4, lr=0.001) | 0.9450 | [0.9204, 0.9663] |
| Tuned (`lstm_head_tuned`, winner: units=128, dropout=0.492, lr=0.00486, batch_size=16) | 0.9418 | [0.9150, 0.9653] |

The winning trial's own held-out `val_auc` (0.8857) did beat every other trial's `val_auc`
(search is working), but that improvement did not carry through to the retrain-on-train+val, score-
on-test step. Plausible reason: the winning trial's higher learning rate (0.00486 vs the baseline's
0.001) converges faster per epoch, which is exactly what made it win a 25-epoch budget during the
search, but the retrain step reuses that trial's `best_epoch` (7) as a fixed epoch count rather than
re-tuning epoch count for the larger train+val dataset, so the final fit may be mildly undertrained
relative to its own optimum. Recorded here rather than re-run under deadline, per SPRINT_PLAN's "no
cherry-picking" rule; `docs/RESULTS.md` reports both numbers and the baseline checkpoint
(`artifacts/checkpoints/lstm_head_baseline.keras`) stays the production model referenced by Sprint
4 onward.

- `configs/model.best.yaml` records the winning architecture regardless of whether it shipped, so
  the search result is reproducible and inspectable even though it isn't the production config.
- Hyperparameter importances (`artifacts/figures/optuna_importances.png`), computed via
  `optuna.importance.get_param_importances`: `learning_rate` dominates at 0.683, then `batch_size`
  (0.159), `dropout` (0.112), `lstm_units` least important (0.045). This matches the intuition that,
  on a frozen-feature head with saturating 25-epoch budgets, how fast the model gets there (learning
  rate, and secondarily batch size through the effective step count) matters more than the head's
  capacity (`lstm_units`) or its regularisation strength (`dropout`), within the ranges searched.
- A future session revisiting this should re-tune `best_epoch` for the train+val-sized retrain
  (e.g. a short early-stopping run against a small carved-out slice of train+val) rather than
  reusing the tuning-time epoch count verbatim, since that is the most likely source of the gap
  above.

## Evidence

Study, 2026-10-10 (`artifacts/optuna/study.db`, wall clock 02:52:40 to 03:24:30, about 32 minutes
for all 30 trials at 25 epochs each):

```
30 trials: 23 complete, 7 pruned
best_val_auc=0.8857206106185913
best params: {'learning_rate': 0.004862170040863293, 'lstm_units': 128,
              'dropout': 0.4920988433143741, 'batch_size': 16}
best_epoch (trial 21): 7
param importances: {'learning_rate': 0.683, 'batch_size': 0.159,
                     'dropout': 0.112, 'lstm_units': 0.045}
```

Retrain on train+val (7 epochs, batch_size=16), final epoch:
```
Epoch 7/7
230/230 - 4s - loss: 0.3378 - accuracy: 0.8414 - auc: 0.9256 - precision: 0.8228 - recall: 0.8746
```

Test evaluation (`scripts/tune.py`'s evaluation step, combined RLVS+AIRTLab+UCF-Crime test,
threshold 0.2000 from `configs/infer.yaml`):
```
tuned does NOT beat baseline: 0.9418 <= 0.9450; baseline ships, reporting both numbers as-is
```

Resume behaviour verified by `tests/test_sprint3_tune.py::test_run_study_resumes_trial_count`
(2 trials, then resumed to 4) and
`tests/test_sprint3_tune.py::test_run_study_without_resume_rejects_existing_study` (rerunning
without `--resume` against an existing study name raises instead of silently restarting).

A TensorFlow-Metal plugin crash (`Mutation::Apply error: fanout 'Adam/AssignSubVariableOp_7' exist
for missing node 'Adam/sub_28'`, in `metal_plugin/src/graph/remapper/remapper.cc`) aborted the
single `scripts/tune.py` process partway through the post-study retrain step, after the study
itself had already completed and persisted all 30 trials to `study.db`. This looks like the same
class of Metal graph-remapper fragility ADR-001 already flagged for this machine, triggered here by
building a second fresh Keras graph late in a long-running process with many prior graphs still
live. The retrain, evaluation, and `docs/RESULTS.md` regeneration steps were re-run to completion
in a fresh process against the same persisted study, which is exactly the resume path `--resume`
exists for; no logic in `safestreets.training.tune` changed to work around it. A future long tuning
run on this machine should expect this and budget for a restart, same as ADR-001's guidance for
TensorFlow-Metal generally.
