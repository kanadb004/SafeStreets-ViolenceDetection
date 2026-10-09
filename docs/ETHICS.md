# SafeStreets — Ethics notes: attribution and the fail-safe invariant

This document covers Sprint 4's two attribution modules
(`safestreets/attribution/person.py`, `safestreets/attribution/gender.py`) and the one
correctness property that governs how they may be used
(`safestreets/attribution/pipeline.py`). See `docs/SPRINT_PLAN.md` Sprint 4 and
`docs/decisions/ADR-005-finetune-and-attribution.md`.

## The fail-safe invariant

**A high violence score raises an alert regardless of what the gender module reports, or
whether the gender or person-detection modules are even available.** Attribution is allowed to
change an alert's `priority` field; it is never allowed to change whether `alert` fires.

This is not a policy statement layered on top of the code; it is a property of
`build_alert`'s implementation (`safestreets/attribution/pipeline.py`): `alert` is computed
from `p_violence` and `threshold` alone, in the first line of the function, before
`person_detections` or `gender_results` are read at all. `tests/test_sprint4_attribution.py`
checks this directly: `p_violence=0.99` with the gender module forced to report "no women
detected" (an empty or all-"man" `gender_results`) still produces `alert["alert"] is True`.
The reasoning: a safety system whose alert can be suppressed by a wrong demographic
classification is worse than one with no demographic classification at all, and the zero-shot
gender head below is specifically the kind of component that will sometimes be wrong.

## Why gender attribution is zero-shot, and what that means

PA-100K gender-classifier training was cut from this sprint (`docs/SPRINT_PLAN.md` §2):
training a dedicated classifier on 100k labelled images was not a 1 to 1.5 day fit. Instead,
`safestreets/attribution/gender.py` scores each tracked person's best crop against two CLIP
text prompts, `"a photo of a woman"` and `"a photo of a man"`, using a frozen, pretrained
`ViT-B-32` CLIP (OpenAI weights, via `open_clip`). **Nothing in this module is fit to labelled
data.** It is a general-purpose vision-language model's zero-shot response to two prompts, not
a validated classifier, and its accuracy on this project's footage was unknown until measured.

Measured accuracy: **NOT RUN as a labelled-accuracy number** — AIRTLab and the other datasets
used here carry violence labels, not person-level gender labels, so there is no ground truth
to score the classifier against in this pipeline. `artifacts/reports/attribution_eval.json`
reports what the classifier actually predicted on the AIRTLab test clips (label distribution,
confidence) as a qualitative check, which is a different thing from a measured accuracy
number, and should not be read as one. Per CLAUDE.md's non-negotiable, no accuracy figure is
invented here in its place.

Known limitations of a two-prompt, binary CLIP gender head, beyond the missing accuracy
number:

- **Binary by construction.** Real gender identity is not binary; this head cannot represent
  that, and forcing every person into "woman" or "man" is itself a harm when it is wrong.
- **Visual-presentation bias.** CLIP's zero-shot judgment is driven by visual presentation
  (clothing, hair, build) learned from its training distribution, not identity, and will
  misclassify people whose presentation does not match that distribution's stereotypes.
- **No calibration.** The softmax score between two prompts is a relative confidence between
  those two options, not a calibrated probability that either prompt is correct.
- **Low-resolution, motion-blurred, or partially-occluded crops** (exactly the conditions a
  violence clip's bounding boxes tend to produce) are known to degrade CLIP zero-shot accuracy
  further; this pipeline does not filter crops by size or blur before classifying them.

## What attribution is, and is not, used for

Attribution's only sanctioned use in this pipeline is `priority` labelling on an alert that
has already fired for an independent reason (the violence score). It must never be used to:

- **Suppress** an alert (see the fail-safe invariant above).
- **Identify** a specific individual; this pipeline detects and tracks person boxes and
  classifies a coarse, binary, unvalidated gender label, nothing more. There is no face
  recognition or re-identification across clips anywhere in this codebase.
- **Stand in for a measured accuracy claim** in any report, model card, or commit message
  (CLAUDE.md's non-negotiable). Every mention of this module in `docs/RESULTS.md` or the
  model card must carry this document's limitations alongside it, not just the number.
