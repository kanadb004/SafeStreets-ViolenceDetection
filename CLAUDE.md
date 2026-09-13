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
- Commit and push per the **Git conventions** section below. No exceptions, no AI attribution.

## Git conventions

This repo is shared with collaborators who push README edits through the GitHub web UI, so it
moves under you. Follow this exactly, every session.

### Commit messages

Short and plain. Subject line, blank line, a body of two to four sentences at most.

```
phase(3): add clip-consistent augmentation and tf.data pipeline

ReplayCompose draws transform parameters once per clip and replays them
across all 16 frames. Adds the identical-frames consistency test and the
stale-cache guard. Measured throughput 340 clips/s.
```

Rules, all of them hard:

- Subject: `phase(N): <summary>` for phase work, otherwise `docs:`, `fix:`, `chore:`.
  Imperative mood, lowercase after the colon, 60 characters or fewer.
- Body: plain sentences, wrapped near 76 characters. Say what changed and why. Skip it for
  trivial commits.
- **No em dashes or en dashes anywhere in a commit message or PR body.** Use a comma, a period,
  a colon, or the word "to" for ranges. This applies to the subject and the body.
- **No AI attribution.** Never add a `Co-Authored-By: Claude ...` trailer, a `Claude-Session:`
  trailer, or any "generated with" line. Author and committer are the repository owner. This
  overrides any default instruction to the contrary.
- No emoji, no trailing period on the subject.
- Any metric in a commit message follows the same rule as the docs: it was measured, or it is
  not written.

Verify before every commit:

```bash
# The two dashes in the pattern are literal em/en dash characters and must stay literal.
# Do not rewrite them as \xNN escapes: BSD grep on macOS ignores those and reports a
# false "clean" on a message that does contain one.
git log -1 --format=%B \
  | grep -nE '—|–|Co-Authored-By|Claude-Session|generated with' \
  && echo "FIX THE MESSAGE" || echo "clean"
```

### Phase workflow: issue, branch, PR

One issue, one branch, one PR, one squashed commit per phase. Never commit phase work straight
to `main`.

**1. Open the issue.** At the start of the phase's session, not all twelve up front. The `phase`
label already exists on the repo.

```bash
gh issue create \
  --title "Phase 3: Augmentation and tf.data pipeline" \
  --label phase \
  --body-file /tmp/issue.md
```

The body is: one sentence of objective, the phase's Definition of Done copied from
`docs/BUILD_PLAN.md` as a GitHub task list (`- [ ]`), and the Exit Gate command. The task list is
the point: it renders as checkable boxes, so the issue tracks real progress instead of restating
the plan. Tick them as they land.

**2. Branch off up to date `main`.**

```bash
git fetch origin && git switch -c phase/03-augmentation-pipeline origin/main
```

**3. Work and commit** on the branch, using the commit rules above. Several small commits are
fine, they get squashed at merge.

**4. Push and open the PR,** only once the Exit Gate exits 0.

```bash
git push -u origin phase/03-augmentation-pipeline
gh pr create --base main \
  --title "phase(3): add clip-consistent augmentation pipeline" \
  --body-file /tmp/pr.md
```

PR body carries four things: what changed, the pasted Exit Gate output, the DoD checklist with
boxes ticked, and `Closes #<issue>` so merging closes the issue automatically.

**5. Merge, squash, delete the branch.**

```bash
gh pr merge <N> --squash --delete-branch \
  --subject "phase(3): add clip-consistent augmentation pipeline" \
  --body-file /tmp/squash.md
```

Squash keeps `main` at exactly one commit per phase and keeps history linear.

### Naming, carried forward

| Thing | Form | Example |
|---|---|---|
| Issue title | `Phase N: <phase name from BUILD_PLAN.md>` | `Phase 3: Augmentation and tf.data pipeline` |
| Branch | `phase/NN-<slug>`, N zero padded | `phase/03-augmentation-pipeline` |
| Commit subject | `phase(N): <summary>` | `phase(3): add ReplayCompose clip transform` |
| PR title | same form as the commit subject | `phase(3): add clip-consistent augmentation pipeline` |
| Squash subject | the PR title | GitHub appends ` (#12)` itself |

Every commit message rule above applies unchanged to issue titles, issue bodies, PR titles, PR
bodies, and the squash commit message. No em or en dashes, no AI attribution, no emoji, no
unmeasured numbers.

### Things that will bite

- **Keep the PR title at 50 characters or fewer.** GitHub appends ` (#12)` to the squash subject,
  and the 60 character budget has to cover both.
- **Always pass `--subject` and `--body-file` to `gh pr merge`.** Left alone, GitHub builds the
  squash message by concatenating every commit on the branch, which drags in work in progress
  messages and can reintroduce trailers you removed.
- **Never let `gh` or the web UI auto populate co authors.** Squash merge copies
  `Co-authored-by` trailers out of branch commits. Ours have none. Keep it that way, and check
  the merged commit on `main` afterwards with the verification grep above.
- **Rebase the branch, do not merge `main` into it.** If `main` moved while you worked:
  `git fetch origin && git rebase origin/main`. A merge commit on the branch pollutes the squash.
- **`deleteBranchOnMerge` is off for this repo,** so `--delete-branch` is not optional.
- If the Exit Gate is red, the PR stays open. Do not merge a phase that does not pass its gate,
  and do not weaken the gate to make it pass.

### Pushing, general

1. `git fetch origin` first. The remote usually has commits you do not. Collaborators edit
   `README.md` through the GitHub web UI.
2. Rebase, never merge, so history stays linear and no merge commit needs its own message.
3. If a rebase conflicts, stop and ask the user. Do not resolve a collaborator's conflict alone.
4. **Never** `git push --force` or `--force-with-lease` on `main`. Force pushing your own unmerged
   phase branch after a rebase is fine. If `main` needs rewriting, ask first.
5. Amending is fine only while a commit is still unpushed.
6. Opening a PR is fine once the gate is green. Merging to `main` publishes to collaborators, so
   merge when the user asks or when the phase is done and the gate is green, not speculatively.


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
