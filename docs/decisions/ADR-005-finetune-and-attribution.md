# ADR-005: AIRTLab fine-tune shipping decision, and attribution design

- **Status:** accepted
- **Date:** 2026-10-10
- **Phase:** Sprint 4 (docs/SPRINT_PLAN.md)

## Context

Sprint 2's zero-shot evaluation found the `lstm_head` baseline (trained on RWF-2000 + RLVS
only, ADR-003) generalises poorly to AIRTLab: ROC-AUC 0.5259 on AIRTLab test, barely above
chance (`artifacts/reports/eval_airtlab_test.json`). AIRTLab is the one dataset in this
pipeline shot specifically as women's-safety relevant staged violence (SPRINT_PLAN.md §2), so
this is the gap Sprint 4's fine-tune exists to close, per the Sprint 4 Definition of Done:
5-fold grouped CV on AIRTLab, a forgetting check against RWF-2000, and a recorded decision
about which checkpoint ships.

AIRTLab has only 175 recorded events (350 clips across two camera views, ADR-002), too few
for a single train/val/test fit to produce a stable number, hence the 5-fold `GroupKFold`
over `group_id` (never splitting one event's two camera views across folds) rather than one
held-out split.

## Measurements

5-fold grouped CV, fine-tuning from the `lstm_head_baseline` checkpoint's weights at
`learning_rate=1e-4`, up to 15 epochs per fold with early stopping on training loss
(`safestreets/training/finetune.py`, `scripts/finetune.py`, seed 1265):

```
accuracy: 0.8486 +/- 0.0439
f1:       0.8863 +/- 0.0338
roc_auc:  0.8976 +/- 0.0429
```

Forgetting check, RWF-2000 val ROC-AUC, baseline checkpoint vs. the checkpoint fine-tuned on
the full AIRTLab pool:

```
roc_auc_before: 0.7150
roc_auc_after:  0.6568
roc_auc_drop:   +0.0582
```

(Full numbers in `artifacts/reports/eval_finetuned.json`.)

## Options considered

1. **Ship the baseline everywhere, use the fine-tuned checkpoint only as a measured AIRTLab
   result.** Keeps RWF-2000 performance untouched, but leaves the model that would actually
   run against AIRTLab-style footage at near-chance AUC (0.53), which defeats the point of
   fine-tuning on the dataset this project cares most about for the stated use case.
2. **Ship the AIRTLab-fine-tuned checkpoint everywhere.** AIRTLab AUC rises from 0.53 to
   0.90 (CV mean); RWF-2000 val AUC drops from 0.715 to 0.657, a 5.8 percentage point
   forgetting cost. The drop is real but RWF-2000 performance stays well above chance, and
   RWF-2000 is not itself the target deployment distribution (it is one of two training
   sources; AIRTLab is the women's-safety-specific one).
3. **Multi-head or dataset-conditioned routing** (serve the baseline for general CCTV,
   the fine-tuned head for AIRTLab-like footage). Real option, but adds a routing decision
   this sprint has no signal to make reliably, and the deadline (docs/SPRINT_PLAN.md) does
   not fit building and validating a router.

## Decision

**Ship the AIRTLab-fine-tuned checkpoint** (`artifacts/checkpoints/finetuned_airtlab_lstm_head.keras`)
as the production model going into Sprint 5's ONNX export and Flask app. The AIRTLab AUC gain
(+0.37) is large and this is the dataset that matches the stated women's-safety use case; the
RWF-2000 forgetting cost (-0.058 AUC, still 0.657, well above chance) is a real but acceptable
tradeoff given that use case, not a catastrophic failure. Both numbers are reported side by
side in `docs/RESULTS.md`, not just the favourable one.

## Consequences

- Sprint 5's ONNX export and Flask app load `finetuned_airtlab_lstm_head.keras`, not
  `lstm_head_baseline.keras`.
- `docs/RESULTS.md`'s headline test metrics (Sprint 2, trained on RWF-2000 + RLVS) describe the
  **baseline**, not this checkpoint; the fine-tune section reports the AIRTLab CV numbers and
  the forgetting check as its own, clearly separated rows, so a reader cannot conflate "the
  model evaluated in Sprint 2" with "the model shipped after Sprint 4."
- If a future sprint adds a fourth training-ish dataset or a general-CCTV deployment target,
  option 3 (routing) should be revisited rather than assumed away a second time.

### Attribution design

Person detection is Ultralytics YOLOv8n (`artifacts/weights/yolov8n.pt`, COCO class 0,
`person`) with ByteTrack, sampled every 4th frame (`configs/attribution.yaml`). Gender
attribution is zero-shot CLIP (`ViT-B-32-quickgelu`, OpenAI weights, via `open_clip`), scoring
each tracked person's highest-confidence crop against the two prompts `"a photo of a woman"` /
`"a photo of a man"`. No PA-100K training (cut, SPRINT_PLAN.md §2): this is a frozen,
pretrained model scored zero-shot, not a classifier fit to labelled data, so its accuracy is
unknown until measured and is expected to be mediocre. See `docs/ETHICS.md` for the measured
(or explicitly unmeasured) number and the reasoning for why a wrong gender label must never be
allowed to suppress an alert.

`safestreets/attribution/pipeline.py`'s `build_alert` computes `alert` from `p_violence` and
`threshold` alone, before `person_detections`/`gender_results` are even read: attribution can
only change an alert's `priority` field, never whether `alert` fires. This is enforced by
construction (the alert boolean is set in the first line of the function body) and covered by
`tests/test_sprint4_attribution.py::test_failsafe_alert_survives_gender_override`.
