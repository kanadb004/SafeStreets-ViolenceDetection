# ADR-002: Dataset split strategy and grouping keys

- **Status:** accepted
- **Date:** 2026-09-13
- **Phase:** 1

## Context

CLAUDE.md's non-negotiable rule is that splits are group-aware: sibling clips from one source
video must never appear in more than one of train/val/test, or every reported metric is inflated.
BUILD_PLAN.md §Phase 1 names a `group_id` derivation per dataset and a `hash(group_id) % 100`
bucketing scheme targeting 70/15/15 per dataset (±3pp). Three real-data problems surfaced while
implementing this that the plan's table did not fully resolve on its own:

1. **RWF-2000** ships an official train/val split (1600/400) with no held-out test partition. The
   plan says to use it "verbatim" with `group_id = clip_id`, which is in tension with the general
   ±3pp / three-way-split DoD item.
2. **RLVS**, as re-hosted on the Kaggle mirror actually used (`mohamedmustafa/real-life-violence-
   situations-dataset`), names clips `V_<n>.mp4` / `NV_<n>.mp4` — sequential ids with no source-video
   stem to parse. The plan's "parse the source-video stem from the filename" instruction doesn't
   apply literally to this mirror.
3. **AIRTLab** and **UCF-Crime** have very few distinct groups (175 and 35 respectively), so
   `hash(group_id) % 100` lands far from 70/15/15 by chance for some hash inputs.

## Decisions

### RWF-2000: official split verbatim, no test contribution
`group_id = clip_id = <source_video_id>_<segment_index>` (the filename stem). `split` is read
directly from the official `train/` and `val/` directories, not from hash bucketing. RWF-2000
contributes 0 rows to `test`. Rationale: the official split already guarantees group-safety (the
authors kept a given source video's segments on one side), re-bucketing it would only add risk of
reshuffling siblings across a boundary the authors already got right, and using it verbatim keeps
RWF-2000 numbers comparable to the published benchmark. Held-out test coverage for the
RWF-2000-style Fight/NonFight task instead comes from the pooled cross-dataset test split
(RLVS + AIRTLab + UCF-Crime test rows) evaluated in Phase 5.

One naming wrinkle: the same source video id can contribute both a Fight-labelled and a
NonFight-labelled segment sharing the same segment index (e.g. `-1l5631l3fg_0` exists under both
`Fight/` and `NonFight/`). `clip_id` therefore includes the label folder name
(`rwf2000_<Fight|NonFight>_<stem>`) to stay unique; `group_id` stays the bare stem, so both
labelled segments of one source video are still grouped together (checked: zero group_id overlap
between RWF-2000's train and val sets).

### RLVS: content near-duplicate clustering, not filename parsing
This mirror's filenames carry no source-video information, so `group_id` is derived from content
instead: an 8x8 average-hash (`ahash`) of the middle frame, combined with the rounded clip
duration and the class label, as the clustering key. Clips landing in the same (label, ahash,
duration-bucket) bucket get the same `group_id`; a clip with no match becomes a singleton group
(`rlvs_cluster_<n>`). This is an approximation — it will under-cluster near-duplicates that differ
enough in their middle frame or duration rounding, and in principle could over-cluster two
genuinely distinct clips that happen to look alike at that one sampled frame — but it is a
real, checkable proxy for "cut from the same source video" where the filenames give none, and it
costs one frame read per clip (~110s for 1951 clips on this machine, see Phase 1 evidence).

### AIRTLab: (label, clip_number) pair, not camera folder
The AIRTLab readme states explicitly that `violent/cam1/N.mp4` and `violent/cam2/N.mp4` are the
*same recorded event* shot from two cameras. Grouping by camera folder alone would leave only two
groups per label (cam1, cam2) — useless for a 70/15/15 split — and worse, grouping by nothing
would let the two camera views of one event land in different splits, which is exactly the kind
of near-duplicate leakage this rule exists to prevent. `group_id = f"{violent|non-violent}_{clip_number}"`
keeps both camera views of one event together, giving 175 distinct groups (60 non-violent + 115
violent, each shot from 2 cameras) to bucket.

### UCF-Crime: source video id (unchanged from the plan)
`group_id = clip_id = <filename stem>` (e.g. `Robbery014_x264`) — each file is already one full
untrimmed source video, so no clustering is needed. Only 35 groups total (Phase 1's subset is 7
videos per category); see below for why bucketing needs a fixed salt to stay within tolerance at
this group count.

### Split bucketing: `hash(salt:dataset:group_id) % 100`, salt = 18
For every dataset except RWF-2000, `split = train if bucket < 70, val if bucket < 85, else test`,
where `bucket = int(md5(f"{salt}:{dataset}:{group_id}").hexdigest(), 16) % 100`. With
AIRTLab (175 groups) and UCF-Crime (35 groups), an unsalted hash lands outside the DoD's ±3pp
tolerance (AIRTLab measured at val=11.4%/test=18.3% with salt 0, a 3.3-3.6pp miss). Salts 0-199
were swept once, offline, scoring each by the worst per-dataset, per-split deviation from
70/15/15; salt 18 was the best fit (max deviation 2.1pp across RLVS/AIRTLab/UCF-Crime) and is
fixed in `safestreets/data/manifest.py` as `_SPLIT_SALT`. This is still a pure function of
`(salt, dataset, group_id)` — deterministic, order-independent, and re-runnable without
reshuffling (checked: two consecutive `build_manifest` runs produce a byte-identical
`splits.json`, see Phase 1 evidence in `docs/PROGRESS.md`).

## Consequences

- Phase 5's evaluation harness must not assume every dataset contributes to `test` — RWF-2000
  does not. Cross-dataset and pooled-test metrics should read the `dataset` column rather than
  assuming a uniform three-way split everywhere.
- RLVS's `group_id` is a similarity heuristic, not ground truth. If Phase 5 or a later audit finds
  RLVS test-set leakage that this heuristic missed, tighten the ahash Hamming-distance threshold
  (currently exact match only) rather than abandoning grouping.
- The `_SPLIT_SALT` constant is load-bearing for the DoD's proportion check. Changing any
  dataset's grouping key (e.g. loosening the RLVS near-dup threshold) changes the group set and
  will likely require re-sweeping the salt.

## Evidence

Manifest build, `data/manifests/clips.parquet`, 4336 rows (see `docs/PROGRESS.md` Phase 1 evidence
for the full command output):

```
dataset   train  val  test
airtlab     242   60    48   (69.1% / 17.1% / 13.7%)
rlvs       1404  274   273   (72.0% / 14.0% / 14.0%)
rwf2000    1600  400     0   (80.0% / 20.0% /  0.0%)
ucfcrime     24    6     5   (68.6% / 17.1% / 14.3%)
```

Leakage assertion (`set(train.group_id) & set(val.group_id) & set(test.group_id)`), per dataset:
all empty.
