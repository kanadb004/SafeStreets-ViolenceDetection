# ADR-003: Frozen-feature LSTM head as the production model

- **Status:** accepted
- **Date:** 2026-10-10
- **Phase:** Sprint 1 (supersedes Phase 4)

## Context

`docs/SPRINT_PLAN.md` §1 measured, on this machine, 2026-10-10:

| | |
|---|---|
| End-to-end `scratch` CNN-LSTM, 30 epochs | 8.7 min/epoch, 4.4 h for one run |
| MobileNetV2 (frozen, ImageNet) feature extraction, all 4332 cached clips | 20.2 ms/clip, 1.5 min total |
| LSTM head on precomputed features, 50 epochs, bs=32 | 1.96 s/epoch, 1.6 min total |
| 30-trial Optuna study, 25 epochs/trial | 24.5 min |
| Feature store size | 16x1280 float32 = 80 KB/clip, 0.35 GB total |

Eight phases of end-to-end training does not fit the 1 to 1.5 day deadline. Report §2.2 names
transfer learning from a pretrained CNN into a recurrent head as the dominant literature
strategy (Mumtaz et al., VGG-19+LSTM; Traore and Akhloufi, VGG16+BiGRU; Imah et al.,
ResNet50v2+GRU), so freezing the CNN is not a deviation from the report's own framing.

A second question this ADR must settle: augmentation (Phase 3) cannot be applied after features
are extracted, since the backbone's forward pass is baked into the stored feature, not the raw
pixels. Two options:

1. Extract features from augmented clips, replayed for several "augmented epochs," and treat it
   as offline augmentation.
2. Train the head on clean (eval-transform, no randomness) features only, and report that
   augmentation was validated in Phase 3 but is not exercised by the production head's training
   data. The `scratch` comparison arm still trains through the live augmented `tf.data` pipeline,
   so the augmentation code path stays exercised end to end by the project as a whole.

## Decision

**Freeze MobileNetV2 (ImageNet weights, `pooling='avg'`), precompute `(N, 16, 1280)` per-frame
features once, and train only a two-layer LSTM head on top.** The `scratch` TimeDistributed
CNN-LSTM (report §4.3's literal architecture) is still built and still trained, as the
trained-from-scratch comparison arm, through the live Phase 3 augmented pipeline.

For augmentation: **option 2**, clean features only for the production head. Simpler, fully
deterministic, and defensible under this deadline; recorded here rather than silently dropped.

Extraction uses `trainable=False` and the eval transform (resize only, no randomness), for a
one-shot deterministic feature store. MobileNetV2 expects `[-1, 1]`-scaled input
(`tf_keras.applications.mobilenet_v2.preprocess_input`), not the `unit_range` `[0, 1]` that
`configs/data.yaml`'s `pipeline.normalize` sets for the live `tf.data` path used by `scratch`;
`safestreets/features/extract.py` applies `preprocess_input` directly inside the extractor rather
than depending on that config value, so a future change to `pipeline.normalize` (made for the
`scratch` arm) cannot silently change what the feature store contains. Asserted by
`tests/test_sprint1_model.py::test_normalization_matches_preprocess_input`.

## Consequences

- The production model (`lstm_head`) never sees augmented input. If Sprint 2's evaluation shows
  overfitting that augmentation would plausibly have helped, that is a known, written-down gap,
  not a silent one.
- Re-extracting features (e.g. after a backbone or normalisation change) invalidates every
  downstream checkpoint; `safestreets/features/extract.py` stamps a `config_sha` attr per file so
  a stale feature store is detectable the same way Phase 2's cache is (`StaleCacheError`-style
  check at idempotency-test time, not a runtime guard inside `train.py` for Sprint 1).
- Sprint 3's Optuna search tunes the head's hyperparameters only (`learning_rate`, `lstm_units`,
  `dropout`, `batch_size`), not backbone freeze depth, since there is no backbone to unfreeze in
  this architecture. If `val_auc` undershoots the 0.80 bar, the first thing to try per
  `SPRINT_PLAN.md` §4 is unfreezing MobileNetV2's top block, before anything more expensive.

## Evidence

Measured numbers above are pasted verbatim from `docs/SPRINT_PLAN.md` §1 (recorded there
2026-10-10 on this machine, idle). Production run's actual `val_auc` and the `scratch` arm's
`val_auc` are recorded in `docs/PROGRESS.md`'s Sprint 1 evidence block, not duplicated here.
