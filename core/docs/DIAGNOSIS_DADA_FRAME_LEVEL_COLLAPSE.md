# Diagnosis — why the DADA-2000 arms have a high micro AUC, a chance-level
# `auc_macro`, and a broken zero-shot transfer to DoTA

**Date:** 2026-09-08 · **Branch:** `main` (KAT-VAD v1) · **Runs analysed:** the
seven DADA-trained arms on branch `v3` (`outputs/v3/DADA2000/*`), plus the
`core/eda/` reports in `outputs/EDA/`.

> **Status of the numbers.** Everything below is *measured*, either by
> `core.tools.eda` (data properties) or by re-reading the 383 + 1,397 per-clip
> `.npz` score curves already on disk (model properties). No new training was
> run. The reproduction script is in §9.

---

## 0. Verdict in five lines

1. The model **is not a frame-level detector on DADA-2000. It is a clip
   classifier.** Removing the per-clip mean from every score curve collapses the
   micro AUC from **0.8617 → 0.4943**. 100 % of the headline number is
   between-clip ranking.
2. A detector that reads **only the number of frames in the clip** and outputs a
   constant score scores **micro AUC 0.8654** on this test split. That is within
   0.009 of the best trained arm (0.8739) and beats four of the seven arms.
   **The reconstructed DADA-2000 corpus leaks its label through clip length.**
3. Inside an abnormal clip, normal frames score **exactly as high** as anomalous
   frames (0.4078 vs 0.4076). This is not a training failure — it is what the
   objective asks for: **no term in the loss ever pushes any frame of an
   abnormal clip down.**
4. The three causes are independent and all are protocol/data defects, **not**
   model-capacity or backbone defects: a **clip-length label leak**, a
   **receptive field that spans the whole clip**, and a **DVS pseudo-label that
   marks the entire anchor clip positive** when 65 % of it is normal.
5. The DoTA collapse is the *consequence*: a clip classifier transfers to a
   benchmark with 1,394/1,397 abnormal clips as nothing. The arm that fits DADA
   worst (A0, clip-AUC 0.767) transfers best (DoTA macro **0.6254**); the arm
   that fits DADA best (A2 s2025, clip-AUC 0.934) transfers at chance (0.5086).

---

## 1. The symptom, quantified

Per-arm decomposition of the in-domain DADA micro AUC. `clip-AUC` = AUC of the
per-clip **mean** score against the clip label (pure video classification).
`micro (clip-mean removed)` = the same micro AUC after subtracting each clip's
own mean, i.e. the part of the ranking that localization could have earned.

| Arm | micro | macro | **clip-AUC** | **micro, clip-mean removed** |
|---|---:|---:|---:|---:|
| A0 `kipoff_s2024` | 0.7050 | 0.5190 | 0.7665 | 0.4912 |
| A2 `constant_s2024` | 0.8617 | 0.5292 | 0.9260 | 0.4943 |
| A2 `constant_s2025` | **0.8739** | 0.5716 | **0.9335** | 0.5419 |
| A2b `constant_pmg_s2024` | 0.8474 | 0.4399 | 0.8815 | 0.4078 |
| A1 `rank_s2024` | 0.7136 | 0.5181 | 0.7566 | 0.4785 |
| A3 `mlp_frozen_s2024` | 0.8457 | 0.4927 | 0.8868 | 0.4536 |
| A4 `mlp_ste_s2024` † | 0.8024 | 0.4849 | 0.8338 | 0.4671 |
| — clip-level oracle (perfect clip ranking, **zero** localization) | 0.9086 | 0.5000 | 1.0000 | 0.5000 |
| — **length-only detector** (score = −clip length, constant per clip) | **0.8654** | 0.5000 | 0.8105 | 0.5000 |

† A4 trained through NaNs (`RESULTS_DADA.md` §8.1); its row is not trustworthy.

**Read the last column.** Every arm sits between 0.41 and 0.54 once the clip
offset is removed. Six of the seven are **at or below chance**.

### 1.1 Where the score mass actually goes

Mean sigmoid score, split three ways:

| Arm | positives (in abnormal clips) | **negatives in abnormal clips** | frames in all-normal clips | within-clip gap | clip-level gap |
|---|---:|---:|---:|---:|---:|
| A0 `kipoff` | 0.1160 | 0.1076 | 0.0464 | **+0.0084** | +0.0612 |
| A2 `constant_s2024` | 0.4076 | **0.4078** | 0.0493 | **−0.0002** | **+0.3585** |
| A2 `constant_s2025` | 0.4568 | 0.4389 | 0.0424 | **+0.0178** | +0.3965 |
| A2b `constant_pmg` | 0.2945 | 0.3041 | 0.0617 | −0.0096 | +0.2424 |
| A1 `rank` | 0.1110 | 0.1105 | 0.0538 | +0.0005 | +0.0567 |
| A3 `mlp_frozen` | 0.3458 | 0.3498 | 0.0639 | −0.0040 | +0.2860 |

This is the user-reported symptom, in one row: **A2 raises the entire abnormal
clip by +0.359 and separates inside it by −0.0002.** The model has learned
"this clip is an accident clip", and it applies that verdict uniformly to every
frame, including the seconds of ordinary driving before the crash.

### 1.2 Localization is at or below chance

Fraction of abnormal clips whose **argmax frame** is a true positive
(chance = the mean positive fraction, 35.1 %):

| A0 | A2 s2024 | A2 s2025 | A2b | A1 | A3 | A4 |
|---:|---:|---:|---:|---:|---:|---:|
| 27.7 % | 28.3 % | **43.5 %** | 17.3 % | 23.6 % | 29.8 % | 26.2 % |

Six of seven arms pick the anomalous frame **less often than a coin weighted by
the base rate**. Only A2 s2025 is above chance, and it is a single seed.

---

## 2. Root cause R1 — the reconstructed corpus leaks its label through clip length

**Severity: CRITICAL. New finding — not covered by `RESULTS_DADA.md`.**

Measured on the 383-clip DADA test split (from the scored `.npz` lengths):

| | abnormal clips | normal clips |
|---|---:|---:|
| n | 191 | 192 |
| median T (stride-8 sampled frames) | **7** | **19** |
| p95 / p5 | 11 | 3 |
| max / min | **17** | 1 |

* **No abnormal clip in the test split exceeds 17 sampled frames.**
* **107 of 383 test clips have T ≥ 18. Every single one is normal.**
* A ranker whose only input is `−T` reaches **clip-level AUC 0.8105** and, once
  broadcast to frames, **micro AUC 0.8654 / AP 0.2630**.

It is a property of the corpus, not of the split: `outputs/EDA/DADA2000/eda_report.md`
§1.4 gives 1,364 test frames over 195 `CarAccident` clips (**7.0** sampled
frames/clip) against 3,880 test frames over 188 `0_Normal_Driving` clips
(**20.6**). The train split is built by the same code from the same folders.

**Mechanism.** The reconstructed DADA-2000 the project is training on — not the
original >100 GB release — appears to ship accident videos **trimmed around the
accident** while normal-driving videos are kept at full length. Median raw
length is 68 frames overall (`eda_report.md` §1.1) with a max of 579; the
bimodality is entirely along the label axis.

**Why it matters more than the oracle framing.** `RESULTS_DADA.md` §4 already
establishes that micro AUC on DADA is clip classification. R1 goes further: the
clip classification itself is **partly free**, obtainable with a ruler. The
model *can* see clip length — the temporal encoder's band mask and RoPE
positions are length-dependent (`core/models/temporal_encoder.py:238`), and
`core/inference.py:95` scores each clip at its true length with no padding. So
the shortcut is available at train time and at test time.

> **Consequence for the thesis.** Any DADA-2000 in-domain number produced on
> this corpus build must be printed beside **both** the 0.9086 clip oracle
> **and** the 0.8654 length-only baseline. Currently the best arm beats the
> ruler by +0.0085.

---

## 3. Root cause R2 — every module's receptive field spans the whole clip

**Severity: CRITICAL. Extends `RESULTS_DADA.md` §5 and lesson C27.**

`RESULTS_DADA.md` §5 blames the score head. That is correct but incomplete —
there are **three** stacked whole-clip mixers on a median DADA clip (T = 9):

| Stage | Setting | Span at T = 9 |
|---|---|---|
| Temporal encoder band mask | `TEMPORAL_WINDOW = 25` (`core/constants.py:126`) → `half_window = 12` (`core/models/temporal_encoder.py:238`) | **whole clip, at layer 1 of 2** |
| KIP `KinematicShift` | 50 % channel shift by ±1 step (`core/kip/gate_shift.py:62`) | ±1 |
| `ConvScoreHead` | 1 layer, `kernel_size = 9` (`core/models/heads.py:21`, `core/constants.py:143`) | **whole clip** |

By the time the first transformer layer has run, **every token is already a
function of every frame in the clip**. The conv head then re-pools. On 55.4 % of
test clips (24.3 % of frames) the network is architecturally incapable of
assigning two frames different scores for anything but positional reasons.

`L_MIL` is at the same floor: `k = max(1, n // 16)` (`core/losses/mil.py:22`)
gives **k = 1 on 90.3 % of clips** — a plain max over 9 frames.

Combining R2 with the MIL objective gives the mechanism directly:

> A whole-clip receptive field means "raise the top-1 frame" and "raise the
> whole curve" are the **same gradient**. `L_MIL` on an abnormal clip therefore
> cannot do anything except lift the entire clip.

### 3.1 What a finer stride buys (derived from the on-disk clip lengths)

| stride | median T | p25 T | clips with T ≤ 9 | total test frames | MIL k=1 rate @ `topk_pct=16` |
|---:|---:|---:|---:|---:|---:|
| **8 (current)** | **9** | 6 | **55.4 %** | 5,244 | **90.3 %** |
| 4 | 18 | 12 | 13.6 % | 10,488 | — |
| **2** | **36** | 24 | **2.3 %** | 20,976 | **39.9 %** |
| 1 | 72 | 48 | 0.8 % | 41,952 | — |

At stride 2 with `mil_topk_pct = 8`, the k = 1 rate falls to **6.0 %** (median
k = 4). Stride 2 alone moves DADA from "unsupportable" to "comparable to
MSAD-shaped" geometry.

---

## 4. Root cause R3 — DVS labels the entire anchor clip as anomalous

**Severity: HIGH. New finding — not covered by `RESULTS_DADA.md`.**

This is the direct cause of the user's specific complaint ("normal frames still
get a high score"), and it is **independent of R1 and R2**.

`core/data/synthesis.py:83`:

```python
pseudo = torch.zeros(len(features), dtype=torch.float32)
if anchor_is_abnormal and fillers:
    pseudo[start : start + len(anchor_features)] = 1.0   # ← the WHOLE anchor clip
```

That vector is then consumed by a **dense, per-frame BCE** —
`supervised_loss` (`core/losses/dvs.py:16`), wired at `core/train.py:302` with
`pseudo_sup_weight = 1.0`.

On DADA-2000 the anchor is a full accident video, not a trimmed anomaly segment:

* mean true positive fraction of an abnormal clip = **0.351** (median 0.333);
* therefore `supervised_loss` labels **64.9 % of the anchor's frames as
  anomalous when they are annotated normal**;
* and it does so with the *densest* gradient in the objective — every frame
  counts, unlike `L_MIL`, which touches one.

The run converges on exactly that: `dvs_sup` falls **0.681 → 0.027** over 500
steps (`outputs/v3/DADA2000/constant_s2024/stage2_kip_on/metrics.jsonl`). The
model has learned the pseudo-label almost perfectly — and the pseudo-label says
*"every frame of an accident video is an accident"*.

### 4.1 The objective contains no downward pressure inside an abnormal clip

Reading all four loss terms as wired in `core/train.py:285-336`:

| Term | Abnormal clip | Normal clip |
|---|---|---|
| `mil_loss` (`core/losses/mil.py:25`) | top-k (k=1) → **1** | top-k → **0** (this pushes the whole clip down, since max ≤ 0 ⟹ all ≤ 0) |
| `supervised_loss` (`core/losses/dvs.py:16`) | *all anchor frames* → **1** (synthesized rows only) | all frames → **0** |
| `pseudo_sup_mil_loss` (`core/losses/dvs.py:31`) | top-k **inside the anchor span** → 1 | top-k → 0 |
| `multi_class_mil_loss` (`core/losses/mil.py:47`) | top-k → `CarAccident` | top-k → `Normal` |

**Not one term pushes any frame of an abnormal clip toward 0.** The loss is
perfectly satisfied by a per-clip constant. Frame-level discrimination is not
under-trained here; it is **unrequested**.

(The wiring itself is correct — `dvs_rows` at `core/train.py:295` properly
excludes un-synthesized abnormal rows, whose `y^p` is legitimately all-zero.
The defect is the *semantics of the span*, not the masking.)

At `theta_ego = 0.85` and `delta_m_ego = 2`, ~15 % of abnormal draws are
synthesized, each as 1 filler + 1 anchor. Over 20 epochs × 25 steps × batch 64
that is ≈ 2,400 rows teaching "whole accident clip = 1", against ≈ 16,000 normal
rows teaching "whole clip = 0". That is a dense, frame-level, **clip
classifier**.

---

## 5. What is *not* the cause

* **Not KIP.** KIP-off (A0) shows the same flat curves (within-clip gap
  +0.0084, macro 0.5190). Turning KIP on raises the clip offset and *lowers*
  transfer. KIP is a smoothing hyperparameter here, consistent with
  `.project/memory-bank/activeContext.md` (2026-09-01 / 2026-09-06) and lesson
  C24 — it neither causes nor fixes this.
* **Not model capacity or convergence.** `mil` falls 0.748 → 0.149 and
  `dvs_sup` 0.681 → 0.027. The model fits its objective. The objective is wrong
  for the corpus.
* **Not (yet demonstrably) the frozen-CLIP backbone.** See §7 — the evidence
  currently available does not support that conclusion, and one of the two
  probe numbers that would support it is measured under R1–R3.

---

## 6. Solution — four phases, cheapest and most decisive first

Ordering principle, inherited from the project audit: **attribution before
architecture**, and lesson **C14** — an ablation run under a known-open
precondition defect measures the defect. Every DADA A/B is currently blocked by
R1–R3. Do not run another gate arm until Phase 2 lands.

### Phase 0 — reporting (zero GPU) — ✅ **COMPLETE 2026-09-08**

| # | Action | Status |
|---|---|---|
| 0.1 | Make `auc_macro` **and** the clip-mean-removed micro the DADA headline; print the clip oracle and the length-only baseline in every DADA table. | **done** — `RESULTS_DADA.md` §3 rebuilt with a `clip-mean removed` column and both baseline rows; §4.1/§4.2 added; `DADA_V3_SETUP.md` pitfalls updated |
| 0.2 | Exclude the 4 vanished-window clips at scoring time. **Not** with `--strict`. | **done, computed** — `RESULTS_DADA.md` §3.1a, `DADA_SETUP.md` §5.1. Offline from the saved `.npz`; no retraining |
| 0.3 | Add the length-only baseline to `core.tools.eda` so no future corpus ships without it. | **done** — `protocol.clip_length_leak`, a CRITICAL verdict above clip-AUC 0.65, report §3.3, a cross-corpus row, 5 new tests (**423 collected, all pass**) |
| 0.4 | Record that no DADA number is comparable to a published frame-level AUC. | **done** — `RESULTS_DADA.md` §9, lessons **C28**/**C29** |

**Two errors were found in `RESULTS_DADA.md` while doing this**, both corrected
project-wide:

1. **The clip oracle is 0.9086, not 0.9069.** The published figure used the
   3,880-frame `0_Normal_Driving` *subgroup* count instead of the
   all-normal-*clip* count (3,896). The 16-frame difference is **exactly** the
   four vanished-window clips of 0.2 — abnormal in `meta.json`, all-zero after
   stride-8 rounding, therefore all-normal clips for every metric. So 0.1 and
   0.2 were the same defect surfacing twice. Corrected in 9 files and pinned by
   `core/tests/test_eda.py`.
2. **The length-only baseline was missing entirely**, and it changes the verdict:
   the best arm beats a ruler by **+0.0075** (379-clip corrected split).

**Corrected DADA numbers after 0.2** (379 clips; `auc_macro` unchanged on every
arm because the four clips are single-class and were already skipped):

| | 383 clips | **379 clips** |
|---|---:|---:|
| best arm (A2 s2025) micro | 0.8739 | **0.8756** |
| clip oracle | 0.9086 | **0.9082** |
| **length-only baseline** | 0.8654 | **0.8681** |
| best-arm margin over the ruler | +0.0085 | **+0.0075** |

**The leak is DADA-specific.** The same check on DoTA returns clip-level AUC
**0.5280**, micro **0.4993**, and **zero** clips separable by length — same
function, same `.npz` files.

### Phase 1 — code fixes that need no re-extraction (small, local)

| # | Fix | Detail | Risk |
|---|---|---|---|
| 1.1 | **Stop DVS from labeling the anchor interior positive.** Give `supervised_loss` a 3-valued target `{0, 1, ignore}`: filler frames stay hard negatives, anchor frames become `ignore`, and the positive pressure comes from `pseudo_sup_mil_loss`, which already restricts its top-k to the anchor span. | `core/data/synthesis.py:83`, `core/losses/dvs.py:16`, `core/train.py:302` | Changes the loss for **every** corpus. Must be an arm (`loss.dvs_anchor_mode = span\|ignore`), defaulting to today's behavior, so MSAD/PreVAD numbers stay reproducible. `trace_call_path` on `supervised_loss` before touching it. |
| 1.2 | **Add downward pressure inside abnormal clips.** A bottom-k MIL term: the lowest-k frames of an abnormal clip are pushed toward 0. ~10 LOC; `multi_class_mil_loss_v2` (`core/losses/mil.py:67`) already has the bottom-k idiom. | `core/losses/mil.py` | New loss term → new hyperparameter. Ship weight 0 by default; enable only as an arm. |
| 1.3 | **Length-controlled evaluation.** Re-score the test split with every clip truncated/padded to a common T, to measure how much of the clip-AUC is the R1 leak. | `core/evaluate.py` (eval-only flag) | None to training. Answers "does the model actually use length?" with a number. |
| 1.4 | Lower `mil_topk_pct` so k > 1 — only meaningful together with Phase 2. | config | — |

### Phase 2 — rebuild the DADA corpus and cache (fires lesson **C2**)

This invalidates **every** DADA cache and **every number in `RESULTS_DADA.md`**.
Do it deliberately, once, and re-measure the baseline before comparing anything.

| # | Change | From → To | Rationale |
|---|---|---|---|
| 2.1 | **Kill the length leak: fixed-length windows.** Sample fixed-T windows from *both* classes (accident windows placed at a random offset inside the window, normals sampled anywhere). Target ≈ 32–64 sampled frames. | variable T (7 vs 19 by class) → constant T | R1. Without this, every other fix is measured through a shortcut. |
| 2.2 | `data.frame_stride` | 8 → **2** | R2: median T 9 → 36; k=1 rate 90 % → 40 % (6 % at `topk_pct=8`). |
| 2.3 | `model.score_head_kernel` | 9 → **3** | R2: head span 100 % → 8 % of the median clip. |
| 2.4 | `model.temporal_window` | 25 → **9** | R2: the encoder must stop being global. **This is the one nobody has changed yet, and it is the largest span of the three.** |
| 2.5 | `loss.mil_topk_pct` | 16 → **8** | R2. |
| 2.6 | Re-extract CLIP + RAFT for DADA into a **new** cache path | `cache/clip/DADA2000` → `cache/clip/DADA2000_s2` | Lesson C2/C13: one cache dir per (transform, stride). Never overwrite. |

**Cost:** 4× the frames of the current cache (20,976 test + ~84 k train sampled
frames). One re-extraction; training cost is dominated by the sequence length.

### Phase 3 — re-run the decisive probe, then decide on the backbone

Re-run `core.tools.eda` with `--probe` on the **stride-2** DADA cache and on
DoTA. The DADA frame-probe number in `outputs/EDA/` (macro **0.5228**) was
measured under R1+R2 and cannot indict a backbone. See §7.

### Pre-registered readings (write these down **before** running — lesson 14)

| Prediction | If true | If false |
|---|---|---|
| Length-only baseline after 2.1 falls to ≈ 0.50 micro | R1 closed | the leak has a second channel; find it before training |
| `auc_macro` after Phase 1+2 rises **> 0.60** | there is real frame-level signal and the protocol was hiding it | the ceiling is representational or the labels are too coarse → Phase 3 decides |
| micro (clip-mean removed) rises **> 0.60** | genuine localization | still a clip classifier |
| DADA frame probe at stride 2 rises **> 0.60** | frozen CLIP is adequate; keep it | backbone becomes a live hypothesis (§7) |

**Do not** report raw DADA micro AUC as the success criterion for any of this.
It is the metric all three defects inflate.

---

## 7. "Should we use a video-aware backbone instead of image CLIP?"

**Short answer: not yet, and not as a swap. It is not the binding constraint on
the evidence we have, and adopting it now would confound every measurement.
Buy the answer with a probe first — it costs ~1–2 GPU-hours and replaces the
opinion with a number.**

### 7.1 What the evidence actually says

**On DoTA, the representation is not the bottleneck.** The supervised linear
probe on the *same* frozen-CLIP features the arms saw reaches **macro AUC
0.6708** (`outputs/EDA/DoTA/eda_report.md` §4.2). The best DADA-trained arm
reaches **0.6254**; the best number ever measured in this project on DoTA is
0.6423 (MSAD-trained). The features carry more frame-level signal than any head
we have trained has extracted. The EDA report says this in its own verdict:
*"The deficit is SUPERVISION, not representation. A better head or denser
labels can close it; a different backbone is not required."*

**On DADA, the low probe number is not admissible evidence.** The DADA frame
probe is 0.5228 — but it was computed on stride-8 features, on clips with a
median of 9 frames and 2 positives, against labels produced by rounding a
normalized window into 9 bins (4 of which vanish entirely). The feature
statistics themselves are *indistinguishable* from DoTA's:

| | DADA | DoTA |
|---|---:|---:|
| CLIP cosine at lag 1 | 0.963 | 0.965 |
| between/within feature variance | 2.72 | 3.19 |
| dims holding 90 % of variance | 419/512 | 419/512 |
| **frame probe macro AUC** | **0.5228** | **0.6708** |

Same backbone, same encoder statistics, +0.15 macro apart. What differs is the
label geometry and the sampling rate — i.e. R1 and R2. Blaming the backbone for
that gap is exactly the C14 error: attributing a defect's effect to a component.

**The project's own motion evidence points the same way.** KIP exists precisely
to inject motion into a frozen-CLIP trunk, and its measured contribution is
temporal smoothing whose *sign flips with training clip length*
(`activeContext.md`, 2026-09-06). That is a weak test of "CLIP lacks motion" —
KIP's flow target is 23 frame-global scalars with no spatial content
(`core/flow/raft_extract.py:52-81`) — but it is not evidence *for* the backbone
hypothesis either.

### 7.2 The real argument in favour, stated honestly

SimpleTAD (ICCVW 2025) reports **DADA→DoTA = 80.3** with VideoMAE-B against our
best-ever 0.6423. `RESULTS_DADA.md` §6 decomposes that as a **0.161
method-class gap**. But that gap bundles four changes at once: a video backbone,
**full fine-tuning**, **per-frame supervision**, 10 FPS sampling, and ≈2.5 M
training examples against our 500 optimizer steps. The backbone is one of five
terms, and the two supervision terms are the ones our EDA independently
identifies as our deficit.

### 7.3 Three reasons a swap *now* would be a mistake

1. **It breaks the claim.** KAT-VAD's thesis is "LaGoVAD + KIP". LaGoVAD is
   frozen CLIP ViT-B/16. Swap the backbone and you are no longer improving
   LaGoVAD, you are comparing method classes — and you lose the released-
   checkpoint gate (lesson C8b) that keeps the protocol honest.
2. **It breaks the text conditioning.** The whole point of the model is
   *definition-conditioned* detection: `H_mul` is a cosine head between video
   tokens and **CLIP text** embeddings (`core/models/heads.py:128`), and
   `L_MIL-align` / `L_neg` live in that shared space. **VideoMAE has no text
   tower and no CLIP-aligned space** — dropping it in silently voids `H_mul`,
   `mul_mil` and the caption contrastive loss. If a video backbone is adopted, it
   must be a **CLIP-aligned video model** (ViCLIP, X-CLIP, InternVideo2's
   CLIP-aligned variant), not VideoMAE.
3. **It costs every number we have.** Lessons C2/C13: a new backbone is a new
   transform, so every cached feature, every KNN cache, every flow cache, and
   every metric ever measured on them is invalidated — across MSAD, DoTA,
   PreVAD, TAD and DADA. That is the single most expensive change available, and
   right now it would be measured *through* R1, R2 and R3.

### 7.4 The decisive experiment — do this instead

**Backbone probe A/B.** Extract windowed features from a CLIP-aligned video
encoder (ViCLIP or InternVideo2-CLIP; VideoMAE only as a non-conditioned
reference) for the **DoTA test split only** (1,397 clips) and the **DADA test
split** (383 clips), then run the *identical* linear probe already implemented in
`core/eda/features.py`. Compare frame-level macro AUC against 0.6708 (DoTA) and
against the re-measured stride-2 DADA number.

Decision rule, pre-registered:

| Reading | Conclusion | Action |
|---|---|---|
| Video probe − CLIP probe **≥ +0.10 macro on DoTA** | The backbone *is* a real ceiling | Adopt as a **second stream** concatenated into the temporal encoder, keeping frozen CLIP for the text-aligned path. Budget a full re-extract. |
| Difference **within ±0.03** | The backbone is not the bottleneck | Keep frozen CLIP. Spend the effort on supervision (Phase 1) and resolution (Phase 2). |
| Between the two | Inconclusive at this cost | Re-run with a second corpus before spending anything. |

Cost: one feature extraction over 1,780 short clips + a 5-fold ridge probe.
No training, no architecture change, and the answer is a number rather than an
argument.

### 7.5 The free temporal upgrade to try first

Before any new backbone, there is a **zero-extraction** test of the same
hypothesis: feed the temporal encoder an explicit temporal-contrast stream —
`Δf_t = f_t − f_{t−1}` (and optionally `f_t − mean(f)`) computed from the
**existing** CLIP cache, concatenated or projected into `D`. That is what KIP
was supposed to buy via flow, but computed from the appearance features
themselves, with a gradient path and no frozen 321-parameter gate (lesson C24).
If explicit temporal contrast helps, the "we need temporal awareness" hypothesis
is supported and the expensive backbone becomes worth buying. If it does nothing
**at the corrected stride**, a video backbone is unlikely to rescue a
supervision problem.

> **Bottom line.** Yes, a video-aware backbone is a legitimate and probably
> eventually correct direction for a *motion-centric* traffic VAD task — that is
> the project's own thesis. But it is currently ranked **below** three measured
> defects that are cheaper to fix, and adopting it now would (a) be measured
> through those defects, (b) void the definition-conditioning the model is built
> on, unless it is CLIP-aligned, and (c) invalidate every cached artifact in the
> project. Fix R1–R3, re-run the probe, and let §7.4's number decide.

---

## 8. Limitations of this diagnosis

* Single seed for every arm but A2 (`constant_s2024` / `constant_s2025`). The
  per-arm rankings in §1 are **point estimates**, not intervals.
* The length leak (§2) is measured on the **test** split. The train split's
  per-class frame/clip ratios in `eda_report.md` §1.4 (7.0 vs 20.6) show the same
  construction, but the train leak has not been measured directly — train frame
  labels are deliberately withheld by `core/data/dada.py`.
* The stride projections in §3.1 are **derived** from on-disk stride-8 clip
  lengths (`raw ≈ 8·T`), not from a re-extraction. Treat them as ±1 frame.
* A4 (`mlp_ste`) trained through NaNs; its rows are shown for completeness only.
* §7's SimpleTAD figures are quoted from `RESULTS_DADA.md` §6, not independently
  verified here.
* **Nothing in this document may be used to tune KIP** (lesson 14). Every
  proposed change repairs a *measured* defect in the protocol or the corpus; none
  is selected against a delta.

---

## 9. Reproduction

All inputs are already on disk. No GPU, no data dir needed.

```bash
source .venv/bin/activate
python - <<'PY'
import glob, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ARM = "outputs/v3/DADA2000/constant_s2024/eval_dada/scores"
S, G = [], []
for f in sorted(glob.glob(f"{ARM}/*.npz")):
    z = np.load(f, allow_pickle=True)
    S.append(z["score"].astype(float)); G.append(z["gt"].astype(float))
Ga = np.concatenate(G)

print("micro                    ", roc_auc_score(Ga, np.concatenate(S)))
print("micro, clip-mean removed ", roc_auc_score(Ga, np.concatenate([s - s.mean() for s in S])))
print("clip-level AUC           ", roc_auc_score([g.max() > 0 for g in G], [s.mean() for s in S]))

lengths = np.concatenate([np.full(len(g), -float(len(g))) for g in G])
print("LENGTH-ONLY micro / AP   ", roc_auc_score(Ga, lengths), average_precision_score(Ga, lengths))

T = np.array([len(g) for g in G]); lab = np.array([g.max() > 0 for g in G])
print("abnormal max T =", T[lab].max(), " normal clips with T>=18 =", int((T[~lab] >= 18).sum()),
      " abnormal clips with T>=18 =", int((T[lab] >= 18).sum()))

ab = [(s, g) for s, g in zip(S, G) if g.max() > 0]
print("mean score pos / neg-in-abnormal:",
      np.concatenate([s[g > 0] for s, g in ab]).mean(),
      np.concatenate([s[g == 0] for s, g in ab]).mean())
print("mean score in all-normal clips  :",
      np.concatenate([s for s, g in zip(S, G) if g.max() == 0]).mean())
PY
```

### Sources

* Score curves: `outputs/v3/DADA2000/{kipoff,rank,constant,constant_pmg,mlp_frozen,mlp_ste}_s202{4,5}/eval_{dada,dota}/scores/*.npz`
* Training curves / configs: `outputs/v3/DADA2000/*/stage2*/{metrics.jsonl,config.yaml}`
* EDA: `outputs/EDA/{DADA2000,DoTA,DADA2000_A2_s2024}/eda_report.{md,json}`, `outputs/EDA/compare_dada_dota.md`
* Code: `core/data/synthesis.py:83`, `core/losses/dvs.py:16`, `core/losses/mil.py:22`,
  `core/train.py:285-336`, `core/models/heads.py:21`, `core/models/temporal_encoder.py:238`,
  `core/constants.py:{88,126,143,157}`, `core/data/dada.py:403`
* Prior art in-tree: `core/docs/v3/RESULTS_DADA.md` (§4 oracle, §5 kernel, §6 SimpleTAD, §10 next steps),
  `core/docs/EDA.md`, lessons **C2, C12, C13, C14, C24, C27**

### Closes

* `RESULTS_DADA.md` §10-**A** (reporting fix) — specified in §6 Phase 0, not yet applied.
* `RESULTS_DADA.md` §10-**B** (frame-level linear probe) — **run**; the answer is
  0.5228 macro on DADA vs 0.6708 on DoTA, and §7.1 explains why only the second
  number is admissible.
