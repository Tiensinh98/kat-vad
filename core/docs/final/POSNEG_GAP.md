# POSNEG_GAP — does K5's feature already separate the accident from the normal part of the same video?

**Rule written 2026-11-11, before any number is read.** Inference only, no training.

## Question

A video-only hard negative (no caption; the idea behind `L_neg`'s n3 mining, see
`KAT_VAD_PROPOSAL.md` §4.4) pulls the pooled feature of the abnormal part of a video away from the
pooled feature of its own normal part. It has work to do only if those two features are not
already far apart.

## Definitions (per positive video of `nexar_val`, K5 s2024, whole video, protocol N-B rows)

* `pos = softmax(+logit / tau)`-pool of `vis_feats` over all frames (the same pooling `L_neg` uses,
  `tau = CONTRASTIVE_TEMP = 0.02`).
* `neg` variants, all `softmax(-logit / tau)`-pools over a frame set:
  * **N-label** (reference only; D-N1 ground-truth frame labels, which no training loss may use):
    the GT-normal frames.
  * **N-thr**: frames with `sigmoid(logit) < 0.5`.
  * **N-soft**: all frames.
* `gap = 1 - cos(pos, neg)` (cosine distance).
* `floor` = the same operators applied to the GT-normal frames only: a random half A pooled with
  `+logit`, the other half B pooled with `-logit`, `1 - cos`. It is how far apart `pos` and `neg`
  land by chance when both come from the normal population. Mean over `POSNEG_FLOOR_DRAWS`
  random splits (fixed seed), then over videos.

## Decision (gated, N-label only)

**STOP the idea iff `mean(gap_label) < 1.5 x mean(floor)`** over the videos with a defined gap and
floor. Otherwise CONTINUE (next: T2 windows, then `grad_probe` before any training).
`1.5` is `constants.POSNEG_GAP_RATIO`; it is the user's threshold, fixed on 2026-11-11.

Printed, never gated: N-thr and N-soft gaps, the share of N-thr frames that are GT-normal
(precision) and of GT-normal frames inside N-thr (recall), the softmax mass of N-soft on GT-normal
frames, the mass of `pos` on GT-abnormal frames, videos skipped and why.

## Caveats written in advance

* n = 1 checkpoint, one seed; a decision from it reads "this checkpoint", not "K5".
* Ground-truth frame labels are used only to **measure**; a loss on Nexar training data has only
  video-level labels (`labels_train.json`) and, with `dvs.theta = 1.0`, no `pseudo_frame_label`.
* `vis_feats` here is K5's feature after KIM; a different host has a different feature.

## Read-out 2026-11-11 (K5 s2024, step 1749, `nexar_val`, 112 positive videos, n = 1)

Raw: `outputs/v_final/kim/REPORTS/pos_neg_gap/pos_neg_gap.md` (Drive `Thesis-Final`). 112 scored, 0
skipped, N-thr empty on 0. The step is that of `outputs/v_final/kim/K5/s2024` (`verified.json` 1749);
`kim_whole` / `kim_t2` hold a different K5 checkpoint (1740), which this read did not use.

| Quantity | mean [95 % bootstrap over videos] |
|---|---|
| gap, N-label (gated) | 0.9690 [0.9180, 1.0223] (median 0.9194) |
| floor (gated) | 0.7410 [0.6975, 0.7894] (median 0.7167) |
| gap, N-thr / N-soft (printed) | 0.9690 [0.9180, 1.0223] each, identical to N-label |
| N-thr precision / recall | 0.9582 / 0.9638 |
| N-soft attention mass on GT-normal | 1.0000 |
| pos attention mass on GT-abnormal | 0.7993 (median 1.0000) |

**Decision by the rule above: STOP.** `mean(gap_label) / mean(floor)` = **1.308** < 1.5. The rule is
not changed after the number; the two findings below are recorded as bounds on what it means.

### What the number does not support

1. **The rule's ratio is capped, so STOP was close to forced.** `gap` is a cosine distance, about
   [0, 1] in practice (above 1 only when the cosine is negative). With `floor` = 0.741 the bar is
   1.5 x 0.741 = 1.11, which needs a mean cosine below -0.11. The measured gap 0.969 (cosine near 0)
   is close to the most this ruler allows, so the ratio cannot realistically exceed about 1.35. A
   floor this high makes the gate read STOP whatever the data. The rule was written without
   estimating its range. Read the result as "the rule says stop", not as "the features are already
   separated" and not as "they are not".
2. **Floor 0.741 says two GT-normal frames of one video are already near-orthogonal** (cosine about
   0.26): K5's `vis_feats` vary strongly from frame to frame, so a cosine between two pooled frames
   is a weak ruler in this space.

### Finding that holds regardless of the rule

**At `tau = 0.02` the three negatives are one vector.** N-label, N-thr and N-soft agree to four
digits, and N-soft puts mass 1.0000 on GT-normal frames. `softmax(-logit / 0.02)` is, to numerical
precision, an argmin: it picks the single lowest-logit frame, which is GT-normal and inside every
candidate set, so the choice of set (label, threshold, all frames) does not change the negative.
Consequences, each from this read only:

* n3 mining at this `tau` yields one frame per video, not "the normal portion of the video".
* `pos` is also close to one frame; its mass on GT-abnormal frames averages 0.80 (median 1.0), so in
  roughly a fifth of the videos (an estimate from the mean, not a count) the highest-logit frame is
  GT-normal. The per-video file `pos_neg_gap_per_video.json` was not read for this note.

### Not licensed

* "K5's pos and neg features are already well separated" or "are not": the ruler is capped (1) and
  noisy (2).
* "The video-only hard negative is useless": the gate used cosine distance between two pooled
  single frames. A different measure was not run.

### Reopening

Needs a numbered Amendment with a ruler that has no cap, written before it is run. One candidate,
**not run and not authorized**: within each video, the AUC of a linear probe (or of a frame score
on `vis_feats`) separating GT-abnormal from GT-normal frames, printed beside the same probe on
shuffled labels.
