# Presentation guide

A walkthrough order for presenting the six notebooks live, re-running cells as you go. Each
section names what to say, not just what to show. Total runtime if re-executed live: well under
5 minutes per notebook (DoD), under 3 minutes total if you only re-run the cheap cells and let the
slower ones (attribution, ~45 s) play while you talk.

## Order and talking points

**1. `01_data_and_preprocessing.ipynb`** — Open with the scale problem: four real datasets, one
explicitly skipped (XD-Violence, 38.3 GB, over the disk budget) and say so before anyone asks.
Run the leakage assertion cell live — it is the actual check the project's exit gate runs, not a
canned pass. Point at the contact sheet image and say what it proves (orientation/colour
correctness), not more.

**2. `02_augmentation.ipynb`** — Run the identical-frames cell live: a constant clip stays
pixel-identical after one shared augmentation draw. This is the one-sentence version of "clip
augmentation applies one transform per clip, not one per frame," and the live assertion is more
convincing than the sentence. Show the before/after grid, then the throughput number, then say
plainly that the production model does not actually train on this pipeline (next notebook
explains why).

**3. `03_model_and_training.ipynb`** — This is where the frozen-feature decision earns its
credibility: read the measured table (8.7 min/epoch end-to-end vs. 2 min total frozen-feature)
before showing the architecture summaries, so the audience has the number before the design
choice. Show both `model.summary()`s, the training curves, and the live-decoded TensorBoard
histogram (not a screenshot). Name the literature citations in the markdown cell if asked why
freezing the CNN is defensible.

**4. `04_evaluation.ipynb`** — The headline result. Walk the audience through val-only threshold
selection before the test numbers, so it's clear test was touched once. The live re-score cell
(assert live == committed) is the strongest trust signal in the whole deck: use it. Then the
cross-dataset table, and name the AIRTLab drop (0.53 AUC) as the motivating problem for notebook 5,
not as a hidden flaw.

**5. `05_hpo_and_finetune.ipynb`** — Two separate stories in one notebook, keep them separate when
presenting. First: the Optuna search worked (trial 21's own val_auc beat everyone), but the tuned
model still lost on test, so the baseline ships — this is a good story about discipline
("no cherry-picking"), tell it as one. Second: the AIRTLab fine-tune, where the tradeoff (AIRTLab
AUC 0.53 -> 0.90, RWF-2000 AUC 0.72 -> 0.66) is the actual shipping decision (ADR-005) — present
both numbers together, never just the favourable one.

**6. `06_inference_and_demo.ipynb`** — Close with the fail-safe invariant cell: force the gender
module to report no women detected on a high-violence clip, show the alert still fires. This is
the single most important correctness property in the attribution design and the easiest to make
concrete live. Then the ONNX parity/latency numbers, the live sliding-window score on a real raw
video, and the Flask app's rendered HTML response as the closing "this actually runs end to end"
beat.

## Three strongest results

1. **The live re-score assertion in notebook 04**: the notebook recomputes test metrics from the
   checkpoint and feature store, in front of the audience, and asserts they match the committed
   report to two decimal places. This is not a number pasted into a slide; it's provably live.
2. **The ONNX parity and latency numbers** (notebook 06): 5.24e-05 max absolute difference between
   TensorFlow and ONNX Runtime on 325 real clips, and 59.5 sustainable fps (18.6x real-time
   headroom) on CPU only, no GPU, no CUDA. The deployment story is real, not aspirational.
3. **The fail-safe invariant, demonstrated, not asserted**: `build_alert` computing `alert` before
   it even reads attribution output is a property of the code, and the notebook proves it live by
   forcing the adversarial case (confident violence score, gender module reporting no women) and
   showing the alert survives.

## Three limitations to volunteer before being asked

1. **The AIRTLab fine-tune costs RWF-2000 performance** (ROC-AUC 0.715 -> 0.657, a 5.8-point
   drop). The project ships that tradeoff deliberately (ADR-005) because AIRTLab is the
   women's-safety-relevant dataset, but it is a real cost, not a free upgrade, and should be
   named before a reviewer finds it in the table.
2. **The gender module's accuracy is unmeasured**, not "measured and good." There is no
   person-level gender ground truth anywhere in this pipeline's datasets, so
   `attribution_eval.json`'s label counts are a qualitative check, never an accuracy number. Say
   this plainly; `docs/ETHICS.md` exists specifically so this is never glossed over.
3. **Frame-level AUC on untrimmed video was never attempted.** XD-Violence (the dataset that would
   have supported it) was never fetched over the disk budget, and the UCF-Crime subset (35 clips)
   is too small on its own. This is scope that was cut, not scope that was tried and failed; say
   so in those terms.

## If asked "why not just train the full CNN-LSTM end to end"

Point at notebook 03's measured table: 8.7 min/epoch, 4.4 h for one 30-epoch run, 24 to 27 h for a
30-trial Optuna study over it. Against a 1 to 1.5 day deadline covering six sprints, that one
decision is what makes sprints 3 through 6 possible at all. The `scratch` CNN-LSTM is still built
and trained (12 epochs, notebook 03) as the literal report §4.3 architecture and the comparison
arm, not dropped, just not the production path.
