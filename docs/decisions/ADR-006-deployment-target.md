# ADR-006: Deployment target: one bundled ONNX graph in ONNX Runtime, served by Flask

- **Status:** accepted
- **Date:** 2026-10-10
- **Phase:** Sprint 5 (docs/SPRINT_PLAN.md)

## Context

Sprint 5 has to turn the shipped model (ADR-005: `finetuned_airtlab_lstm_head.keras`) into
something a demo app and a streaming loop can call, and replace the legacy `app/` package, which
could not serve a single request (CLAUDE.md "Known defects"). Constraints already fixed by earlier
ADRs and measurements:

- The shipped checkpoint is only the LSTM head. It consumes MobileNetV2 features `(16, 1280)`,
  not pixels (ADR-003), so serving needs the frozen backbone too.
- Target hardware is an Apple M2 laptop with no CUDA (CLAUDE.md). Metal helps training, but a demo
  server should not depend on it.
- ADR-001 already proved the export chain: `tf_keras` `model.export()` to a SavedModel, then
  `tf2onnx.convert --saved-model`, run in `onnxruntime` 1.18.1, with a parity spike at 5.96e-08.
- 1 to 1.5 day deadline (SPRINT_PLAN.md). Whatever ships has to be built, tested and documented
  inside one sprint.

## Options considered

1. **Export the head only to ONNX, compute features with tf_keras at serve time.** Smallest
   graph, but every serving process still imports TensorFlow (slow start-up, large memory, and
   the TF/Keras 3 seam from ADR-001 still applies at serve time). ONNX would buy almost nothing.
2. **Bundle backbone and head into one ONNX graph, serve with ONNX Runtime CPU inside Flask.**
   Raw letterboxed RGB frames go in and P(violent) comes out. MobileNetV2's `preprocess_input`
   is baked in as a `Rescaling(1/127.5, offset=-1)` layer. The server needs onnxruntime, numpy
   and OpenCV, with no TensorFlow.
3. **TF Serving or a separate model microservice behind the Flask app.** A second process,
   gRPC/REST plumbing and container work for a single-user demo. The deadline does not fit it,
   and SPRINT_PLAN.md §2 already cuts Docker.
4. **Serve the Keras model directly from Flask.** No export risk, but no parity-checked portable
   artefact, and the same TensorFlow dependency as option 1.

## Decision

**Option 2.** `scripts/export_onnx.py` rebuilds the inference path as one `tf_keras` graph:
`Rescaling`, then `TimeDistributed(MobileNetV2)` built by the same
`safestreets.features.extract.build_backbone` that wrote the feature store, then the fine-tuned
head. It converts that graph with ADR-001's recipe (opset 13) to
`artifacts/onnx/safestreets_finetuned_airtlab.onnx`. A sidecar `.json` records the input
contract (shape, dtype, layout, RGB, letterbox, uniform sampling), the normalisation constants,
the head's `input_dim`/`n_frames`, the `configs/infer.yaml` threshold, the git SHA, and the
SHA-256 of both the checkpoint and the `.onnx` file.

`safestreets.inference.engine.InferenceEngine` loads that pair and rebuilds the same contract from
the active configs (`configs/preprocess.yaml`, `configs/model.best.yaml`, `configs/infer.yaml`).
If any field differs it raises `SpecMismatchError` before it opens the ONNX session. The Flask
app (`safestreets/web`, `create_app()`) builds one engine lazily and uses it for both the HTML
upload flow and `POST /api/analyse`.

## Consequences

- The serving path has no TensorFlow import. The exporter and the parity check still need the
  full `ml` environment.
- Changing frame count, resolution, sampling, backbone normalisation or the operating threshold
  makes the engine refuse the old artefact until `scripts/export_onnx.py` runs again. That is
  intended. A stale artefact fails loudly instead of scoring quietly under the wrong contract.
- The `.onnx` file is about 12 MB and is not committed, because `artifacts/` is gitignored, as
  with every earlier checkpoint. The sidecar's `onnx_sha256` is what ties a served file to an
  export run.
- Single-process Flask with CPU onnxruntime suits a demo, not production CCTV fan-in. Revisit
  this ADR if the target becomes many concurrent camera streams. That would mean a batching
  inference service and option 3.
- `configs/infer.yaml`'s threshold of 0.2 was selected on validation for the Sprint 1 baseline
  (Sprint 2), not re-selected for the ADR-005 fine-tuned checkpoint. The artefact ships that
  threshold unchanged. Re-selecting it is a separate, recorded decision and is not done silently
  here.

## Evidence

`python scripts/export_onnx.py`, 2026-10-10, full numbers in `artifacts/reports/onnx_parity.json`
and `artifacts/reports/latency.json`:

```
exported .../artifacts/onnx/safestreets_finetuned_airtlab.onnx (12002918 bytes) in 11.7s
engine loaded: model_version=finetuned_airtlab-73ebf22e032d git_sha=d8b8335c2c63a194687326504ce527536673f833
parity: n=325 max_abs_diff=5.244e-05 (tol 0.0001) passed=True
roc_auc keras=0.969105 onnx=0.969105 diff=0.00e+00 passed=True
bundled keras vs feature store + head: 2.086e-06
latency per window: p50=111.08ms p95=134.41ms p99=167.90ms  sustainable_fps=59.5 (stride 8), headroom x18.6 over target_fps 3.2
```

The 325 clips are the combined RLVS + AIRTLab + UCF-Crime test split from the Phase 2 clip cache.
The AIRTLab test clips were in the ADR-005 fine-tune pool, so this AUC checks that the two
runtimes agree. It is not a generalisation result.
