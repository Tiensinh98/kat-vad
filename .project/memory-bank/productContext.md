# Product Context — why KAT-VAD exists

**Created:** 2026-07-31 (re-init from `b9978ff`)
**Last reviewed:** 2026-09-24 (full reconcile — the repair for the negative T2 A/B is
authorized and built; the fair test is queued, not run). Previously 2026-09-21 (the
paired T2 KIP A/B is negative and now diagnosed; see "What has actually been shown"). Earlier: 2026-09-15, the
corpus survey (every corpus with negative bags is degenerate, every clean one has
none) and TAD as a second independent negative result.

> **Branch note.** This is the `main` = **KAT-VAD v1** memory bank. The *product*
> argument below is the project's and does not change with the branch. What does
> change: on `main` the "kinematic gate" is **only** the frozen v1 MLP, so the
> caveat below is not a caveat here — it is the whole gate.

## The problem

Video anomaly detection for traffic has to answer "is this frame anomalous, and
which kind of anomaly is it?" under **weak supervision** — only video-level
labels exist, never frame-level ones at training time.

LaGoVAD's contribution is conditioning detection on a *language definition* of
what counts as anomalous, so the same trained model can be re-targeted to a new
benchmark by swapping the definition text instead of retraining. That
generalizes well semantically — and fails on motion. Its published DoTA score is
its lowest, and DoTA is the benchmark where the anomaly *is* the kinematics
(collisions, ego-motion, sudden deceleration).

## What KIP adds

Optical flow carries the motion evidence, but running RAFT at inference is too
expensive for a deployable detector. KIP resolves that trade:

- **At train time** — a PMG head regresses cached RAFT flow embeddings `e_O` from
  the temporal features `v^t`, so motion structure is *distilled into* the RGB
  representation (`L_KIP_rec`, `L_KIP_align`).
- **A kinematic gate + adaptive temporal shift** mixes channels across time
  proportionally to predicted motion magnitude, producing `v^k`. **Caveat that
  changes the story — and on `main` it is the only gate there is:** the gate's 321-parameter MLP is *never trained* — a hard
  `floor()` on the shift count kills the gradient (`core/kip/gate_shift.py:112`,
  verified: all 6 tensors get `grad is None`). The gate is input-adaptive but
  frozen at random init, so "motion-gated" describes the *input*, not a learned
  mapping.
- **At inference** — RGB only. No flow, no RAFT, no extra latency.

The result should be a detector that keeps LaGoVAD's definition-conditioned
generalization while gaining the motion sensitivity it lacks.

## What has actually been shown (latest reading 2026-09-21)

*(Reviewed 2026-09-24, second pass: no new measurement. The only change is a
planned feasibility EDA for D2City as an imported normal pool — `progress.md`.)*

**New, and the strongest evidence yet, because it is the first *paired* test.**
Every earlier KIP verdict compared arms across corpora or across branches. On T2
(DADA-2000 original, W=20) the two arms differ in exactly one config line, on the
same three seeds: **KIP-on costs 0.0129 T2 micro**, t95 [−0.0249, −0.0008], 3/3
seeds agreeing. And for the first time there is a **mechanism**, not just a sign:
`L_KIP_rec` regresses an **unnormalized** target (a constant predictor scores MSE
**31.64**; `mag_max` in raw pixel units carries **83 %** of its energy), so at
`lambda_rec = 1.0` it enters an otherwise-O(1) objective ~32× oversized and
**captures the shared temporal encoder** — gradient ratio **3.1**, and
**orthogonal** to the task (`cos` = −0.001), i.e. it spends trunk capacity rather
than fighting for it. Every task loss ends 24–32 % higher with KIP on.

**What this does and does not license.** It does *not* rehabilitate KIP: the
number is measured and negative. It does say the arms were **weighted wrong**, so
the fair test has not been run. The repair was **authorized 2026-09-23 and built**
(z-score the 23 raw flow stats into `cache/flow/v2_zscore/`, same projection;
`lambda_rec` **derived** as `1/V` of the *new* cache, ≈ 1, never swept) and **has not
been run** — until it is, the honest product statement is "KIP-v1 as weighted costs
0.013 on T2; the correctly weighted test is pending". One thing survives the repair either way: **`R²_item` = 0.283** —
the PMG head does fit 28 % of the *within-clip* flow variance from `v^t`, so
frozen CLIP carries some dynamics and KIP's premise is not refuted on T2.

### The 2026-09-06 reading, unchanged

**The product claim above did not survive its own ablations.** The measurements
are solid; the story attached to them is not.

- **The +0.09 is real but it is not motion.** MSAD-trained, zero-shot DoTA:
  **a fixed 50 % channel shift with no flow, no PMG head, no KIP losses and no
  stage-1 warm-up** (arm A2) reproduces the entire gain and is statistically
  indistinguishable from full v1 KIP (Δ = +0.0109, t95 [−0.0588, +0.0805], n=3;
  A2 − A0 = +0.1025 ± 0.0350). **KIP's measured contribution is temporal
  smoothing** — `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md`, 2026-09-01.
- **Adding the actual motion machinery makes it worse, not better.** The v3
  `rank` gate (A1 − A2 = −0.0683) and real flow with all three auxiliary losses
  (A2b − A2 = −0.0105) both *cost* AUC. (Those arms were run on branch `v3`;
  `main` cannot reproduce them — it has no `gate_type`.) **No component of KIP has been attributed
  a positive contribution.**
- **The effect's sign depends on the training corpus.** On DADA-2000 it inverts:
  A2 − A0 = **−0.0918**, A1 − A2 = **+0.0300** (`v3/RESULTS_DADA.md`, 2026-09-06).
  The cause is mechanical — DADA's median clip is 9 stride-8 frames under a
  kernel-9 score head, so the shift flattens the curve to a per-clip constant.
  A component whose sign flips with clip length is a **smoothing
  hyperparameter**, not a mechanism.
- **The proposal's ego-kinematics story stays refuted** — the gain is larger on
  third-party motion than on ego motion, now 10/10 arm-seeds. Warm-start transfer
  and under-convergence were eliminated earlier; neither was the answer either.
- **MSAD in-domain remains a bounded null** — "any in-domain effect is smaller
  than ≈1 AUC point at n=3", never "costs nothing": at SD 0.0031 the study cannot
  see a real ±0.005.
- **The inference claim still holds** on its own terms: +1.65 % parameters at
  inference (321 on the score path), RGB-only, no flow at test time. It is a
  claim about cost, not about benefit.
- **The flow target carries no spatial information** — 23 frame-global scalars
  (magnitude mean/std/max, u/v mean/std, 16-bin angle histogram) lifted to 256-d
  by a fixed seeded projection. Any "localized/peripheral motion" story is not
  expressible in the target; restate it as a *global flow-statistic residual*.

- **A second, independent negative result landed 2026-09-15 — and it is about
  the supervision, not about KIP.** On TAD, in-domain weakly-supervised training
  turns the detector into a **clip classifier**: clip-level AUC **0.9975**, micro
  AUC 0.9237 sitting on the corpus's own constant-per-clip oracle of **0.9226**,
  while frame-level `auc_macro` **falls** from the zero-shot trunk's 0.7578 to
  0.6174 and DoTA transfer falls 0.6158 → 0.5496. Three objective-side repairs
  (`dvs_anchor_mode=ignore`, `bottomk_weight`, both, plus a `bottomk_topk_pct=8`
  dose) **all failed**, and `bottomk` failed *while demonstrably working*
  — within-clip range +37 %, gap +78 %, macro down. **It created variance without
  direction.** Warm-starting from the PreVAD trunk recovers only a quarter of the
  gap: **the model is handed macro 0.7578 and TAD training destroys 0.104 of it.**

- **The corpora themselves are the third finding (2026-09-15).** Surveyed across
  all five: **every corpus with negative bags is degenerate, and every clean
  corpus has none.** MSAD has negatives and is clean but is CCTV, not traffic;
  TAD has negatives and collapses; the DADA archive has negatives and leaks its
  label through clip length; **DoTA** is clean — constant-per-clip oracle
  **0.5017**, 1,392/1,397 clips mixed — but ships **3** normal clips and
  `train_clips: 0`; the **DADA-2000 original** release is clean and has **zero**.
  That is why the live plan manufactures its negatives *inside* the abnormal
  videos rather than importing a second pool.

  A corollary worth stating: DoTA's supervised frame linear probe reaches
  `auc_macro` **0.6708** on the same frozen CLIP features where the DADA archive
  reaches **0.5228**. **The "frozen CLIP cannot localize" verdict is about that
  corpus, not about the representation** — which is exactly what the next gate
  tests.

**What the project can honestly claim today:** two rigorous negative results. (1)
A motion-induction pathway was built, and a one-line temporal smoother matches
it; the smoother itself does not generalize across training corpora. (2) On a
corpus whose bag-level task is trivially separable, weakly-supervised MIL
optimizes clip classification and *actively destroys* frame-level localization —
including localization it was given for free. Neither is a win for KIP, and both
are worth more than a +0.09 nobody can explain.

## Who it is for

A research thesis deliverable: the headline claim is that PreVAD-trained KAT-VAD
beats LaGoVAD's DoTA zero-shot number. MSAD is the current working benchmark
because the user has it on disk and it runs on one Colab A100 — and the DoTA
result above is already the MSAD-trained version of that claim.

**External-validity caveat the audit raises:** the campaign ran at **1.5 % of
LaGoVAD's training scale** (MSAD's 480 weak labels vs PreVAD's 32,673), and train
`mil` → 0.0012 is memorization of those labels. That is the single largest threat
to the result, and it is exactly what the PreVAD trunk-transfer track addresses.

**Method-class caveat, added 2026-09-06.** Comparing this project's DoTA numbers
to a fully-supervised video-transformer detector is not like-for-like.
[SimpleTAD (ICCVW 2025)](https://arxiv.org/abs/2507.09338) reports DADA-2000 →
DoTA **80.3** with a fully fine-tuned VideoMAE at 10 FPS trained on **per-frame**
labels; this project is a frozen-CLIP, video-level-label WS-VAD method at
3.75 FPS whose best DoTA number ever is **0.6423** (LaGoVAD's own released
checkpoint reaches 0.6142). **0.161 of that 0.196 gap is the method class, not
the run.** Reaching 0.80 means abandoning the baseline this thesis is built on.

## How it should work operationally

- Every entry point is an `argparse` CLI — download, preprocess, extract
  features, extract flow, build KNN cache, train, infer, evaluate, visualize.
- Everything long-running is **resumable** and Drive-persisted: Colab sessions
  die, and a dead session must never cost more than the last checkpoint interval.
- All code and tests run **data-free on CPU** so development never blocks on a
  download or a GPU.

See [[systemPatterns]] for how this is realized and [[techContext]] for the stack.
