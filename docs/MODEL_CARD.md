# Model card: SafeStreets violence detector

**Shipped artefact:** `artifacts/onnx/safestreets_finetuned_airtlab.onnx`, bundling frozen
MobileNetV2 feature extraction and the fine-tuned `lstm_head` into one ONNX graph (ADR-006).
Keras source checkpoint: `artifacts/checkpoints/finetuned_airtlab_lstm_head.keras` (ADR-005).
Generated from the real artefacts under `artifacts/reports/` and `docs/decisions/`; every number
below traces to one of those files. See `docs/RESULTS.md` for the full metrics tables this card
summarises.

## Intended use

A binary violence/no-violence classifier over short (~5 s) video clips, intended as a CCTV
screening aid for a women's-safety context: it scores a clip, raises an alert above a fixed
threshold, and optionally attaches person/gender attribution as an alert **priority** signal.
Designed for AIRTLab-style staged violence and RWF-2000/RLVS-style organic violence clips at
16 frames, 112x112, sampled at ~3.2 fps. Served via ONNX Runtime on CPU (Apple M2, no CUDA) through
the Flask app (`safestreets/web`) or `safestreets.inference.engine.InferenceEngine` directly.

## Out-of-scope use

- **Not a face recognition or re-identification system.** Person detection (YOLOv8n + ByteTrack)
  tracks box positions only; nothing in this codebase identifies a specific individual.
- **Not a validated gender classifier.** The gender module is zero-shot CLIP scored against two
  text prompts, never fit to labelled data; see "Gender attribution" below and `docs/ETHICS.md`.
- **Not evaluated on frame-level/untrimmed video.** Only clip-level, pre-trimmed (~5 s) input was
  evaluated; frame-level AUC on untrimmed footage is explicitly out of scope (`docs/SPRINT_PLAN.md`
  §2, `docs/RESULTS.md` limitations).
- **Not production-hardened for many concurrent camera streams.** Single-process Flask with CPU
  ONNX Runtime suits a demo, not a many-camera deployment (ADR-006 consequences).
- **Not validated on real-world unscripted violence.** AIRTLab, the dataset this project's
  women's-safety framing leans on most, is staged, acted footage; see "Training data" below.

## Training data

| Dataset | Role | Licence | Clips | Citation |
|---|---|---|---|---|
| RWF-2000 | primary training | research/education (Kaggle mirror, no redistribution terms stated) | 2000 | Cheng, Cai, Li, ICPR 2020 |
| RLVS | primary training | CC0 / public domain | 1951 | Soliman et al., ICICIS 2019 |
| AIRTLab | women's-safety fine-tune + eval | "freely released for research and educational purposes" | 350 | Bianculli et al., Data in Brief 33 (2020) |
| UCF-Crime (subset) | cross-dataset / zero-shot eval only | research use, original UCF CRCV release | 35 | Sultani, Chen, Shah, CVPR 2018 |
| XD-Violence | **skipped** (38.3 GB, over disk budget) | n/a | 0 | Wu et al., ECCV 2020 |

Splits are group-aware (ADR-002): sibling clips from one source video, or one AIRTLab event's two
camera views, never cross a train/val/test boundary. Full provenance in `docs/DATASETS.md`.

## Headline metrics

In-domain test split (RLVS + AIRTLab + UCF-Crime test, n=325), the Sprint 1/2 baseline checkpoint,
F1-optimal threshold 0.2000, from `artifacts/reports/eval_test.json`:

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.9450 | [0.9204, 0.9663] |
| PR-AUC | 0.9471 | [0.9201, 0.9704] |
| Accuracy | 0.8769 | [0.8400, 0.9138] |
| Precision | 0.8392 | [0.7905, 0.8923] |
| Recall | 0.9543 | [0.9191, 0.9828] |
| F1 | 0.8930 | [0.8595, 0.9260] |

**This table describes the Sprint 1/2 baseline checkpoint, not the shipped checkpoint.** The
shipped checkpoint (`finetuned_airtlab_lstm_head.keras`, ADR-005) is fine-tuned further on
AIRTLab; its own numbers are the fine-tune row below, not this table, which is kept separate so
the two are never conflated (see `docs/decisions/ADR-005-finetune-and-attribution.md`).

Cross-dataset generalisation, baseline checkpoint, zero-shot (`artifacts/reports/cross_dataset.json`):

| Test dataset | In-domain? | n | ROC-AUC | AUC drop vs. in-domain |
|---|---|---|---|---|
| RLVS | yes | 272 | 0.9805 | - |
| AIRTLab | no | 48 | 0.5259 | +0.4546 |
| UCF-Crime | no | 5 | 1.0000 | -0.0195 |

AIRTLab fine-tune, 5-fold grouped CV (`artifacts/reports/eval_finetuned.json`):

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.8486 | 0.0439 |
| F1 | 0.8863 | 0.0338 |
| ROC-AUC | 0.8976 | 0.0429 |

RWF-2000 forgetting check, same artefact: ROC-AUC 0.7150 before fine-tune, 0.6568 after (drop
+0.0582). ADR-005 ships the fine-tuned checkpoint anyway: the AIRTLab gain (0.53 -> 0.90 AUC) is
large and AIRTLab is the women's-safety-relevant dataset, while RWF-2000 stays well above chance.

ONNX serving (`artifacts/reports/onnx_parity.json`, `artifacts/reports/latency.json`): parity max
abs diff 5.24e-05 (n=325, tolerance 1e-4), ROC-AUC keras 0.969105 vs. onnx 0.969105 (diff 0.0);
p95 latency 134.41 ms/window, 59.5 sustainable fps at stride 8, 18.6x headroom over the 3.2 fps
real-time target.

## Attribution modules

- **Person detection:** YOLOv8n (COCO class 0) + ByteTrack, sampled every 4th frame
  (`configs/attribution.yaml`). Accuracy was not separately measured; it is used only to produce
  tracked crops for the gender module, and track-level summaries are reported qualitatively in
  `artifacts/reports/attribution_eval.json`.
- **Gender attribution:** zero-shot CLIP (`ViT-B-32-quickgelu`, OpenAI weights), scored against
  two prompts ("a photo of a woman" / "a photo of a man"). **Accuracy is NOT RUN as a measured
  number** — no dataset in this pipeline carries person-level gender ground truth, so there is no
  labelled accuracy to report. See `docs/ETHICS.md` for the full limitations list (binary by
  construction, visual-presentation bias, no calibration, degraded by low-res/motion-blurred
  crops).
- **Fail-safe invariant:** attribution can only change an alert's `priority` field; `alert` itself
  is computed from `p_violence >= threshold` alone, before attribution output is even read
  (`safestreets/attribution/pipeline.py:build_alert`, enforced by construction and covered by
  `tests/test_sprint4_attribution.py::test_failsafe_alert_survives_gender_override`).

## Limitations

- Frame-level AUC on untrimmed video is out of scope (XD-Violence skipped over disk budget; the
  UCF-Crime subset is too small, 35 clips total).
- AIRTLab is staged, acted violence; its cross-dataset and fine-tune numbers reflect
  generalisation across filming style for the same staged task, not to unscripted real-world
  footage.
- The scratch (trained-from-scratch) CNN-LSTM comparison arm was only trained for 12 epochs
  (scaled down from a full tuned run, `docs/SPRINT_PLAN.md` §2) and reaches 0.60 val AUC; it is a
  comparison point for the frozen-feature decision, not a competing production candidate.
- The operating threshold (0.2000) was selected on validation for the Sprint 1 baseline, not
  re-selected for the Sprint 4 fine-tuned checkpoint that actually ships (ADR-006 consequences).
- Latency/parity numbers are single measurements on one Apple M2 laptop under its own load at
  measurement time, not a guarantee across hardware or under concurrent load.

## Ethics

See `docs/ETHICS.md` for the full discussion of the fail-safe invariant and the gender module's
known weaknesses. In summary: this system is built so that a wrong or missing demographic
classification can change an alert's priority label, but can never suppress or create an alert;
the gender module is a coarse, unvalidated, binary signal, not an identity classifier, and is
never used for re-identification.
