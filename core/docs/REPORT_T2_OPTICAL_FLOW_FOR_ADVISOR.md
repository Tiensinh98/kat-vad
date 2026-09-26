# KAT-VAD: dataset EDA, DADA-2000 T2 (W=20) training results, and the case that the optical-flow pathway does not help

**Prepared for:** advisor review (a request for guidance on how to modify the model).
**Date:** 2026-09-26. **Code:** branch `main` (KAT-VAD v1). **Author:** sinhpham.
**Scope:**
- EDA over every corpus considered;
- training and evaluation on **one** training corpus only, **DADA-2000 original, T2 windows
  (W = 20, hop 8)**, evaluated in-domain on T2 and zero-shot on DoTA;
- the measured evidence on the optical-flow component (KIP).

Every number below was measured in this project. Intervals are t95 over seeds (n = 3,
t = 4.303), with deltas paired by seed. Durable records:
`core/docs/RESULTS_DADA_ORIG_T2.md`, `core/docs/{D2CITY,BDDA}_EDA.md`,
`.project/plans/katvad-{flow-zscore-option-a,kip-loss-scale-diagnosis,t2-learning-curve}.md`.

---

## 0. Summary

1. **Only one training corpus is usable.** Of eight corpora profiled, only the
   DADA-2000 original release, cut into fixed T2 windows, is clean (no length leak) *and*
   has negatives. Every other one either leaks its label through clip length, has no
   normals, cannot be trained on, or is separable from DADA by source alone.
2. **The trunk trains correctly on T2.** KIP-off reaches T2 `auc_macro` **0.6248**, above
   the frozen-CLIP frame linear probe (0.5983). Micro stays **below** the clip oracle
   (0.7037), so the model is localizing rather than classifying windows.
3. **The optical-flow pathway (KIP) gives no measurable benefit on T2.**
   - With the flow target as originally built, it **costs** 0.013 micro AUC (t95 excludes 0).
   - After repairing its loss scale it is **neutral**: +0.0076, t95 [−0.0002, +0.0154].
     Zero-shot DoTA is unchanged within noise.
4. **The mechanism is diagnosed.**
   - The flow target is 23 frame-global statistics.
   - Once they are equally weighted, the flow head predicts them **worse than a predictor
     that knows only which clip it is in**, and it cannot predict flow *direction* at all
     (R² 0.07).
   - The gate that is supposed to use flow never receives a gradient, so every KIP-on run
     applies a fixed ≈ 50 % temporal channel shift.
5. **More data of the same kind does not help DoTA either.** Training on 25 % → 100 % of T2
   moves zero-shot DoTA by +0.008, inside the noise.

**Request:** guidance on how to replace or redesign the motion pathway. §6 lists the options
and what each would need to show.

---

## 1. Model in one diagram (the part under discussion is KIP)

```
video ─► frozen CLIP ViT-B/16 (per sampled frame, stride 8) ─► F (L×512)
        ─► temporal encoder (2-layer Transformer, RoPE) ─► v^t (L×512)
        ─► KIP  ┌ PMGFlowHead:     v^t → ê_O (L×256)      trained to regress RAFT target e_O
                ├ KinematicShift:  gate(‖ê_O‖) → shift s_t channels of v^t across time
                └ MotionScoreHead: ê_O → ŷ_O               L_kin = MIL-BCE vs the video label
        ─► v^k ─► co-attention with CLIP text definitions ─► H_bin (anomaly curve), H_mul
```

- KIP is **train-time supervised by optical flow**: RAFT runs offline, and inference is
  RGB-only.
- KIP-off removes the whole block; `v^t` then goes straight to fusion.

**What "the flow target" actually is** (`core/flow/raft_extract.py`). For each frame, RAFT's
dense field (about 10⁶ vectors) is collapsed to **23 frame-global scalars**:
- magnitude mean / std / max;
- mean and std of *u* and *v*;
- a 16-bin angle histogram.

A fixed random matrix then lifts them to 256-d. The target has **no spatial content**: it
cannot say *where* in the frame motion happens.

---

## 2. EDA over the candidate corpora

### 2.1 Why a corpus is rejected: the two leak measurements

For weakly supervised VAD, micro AUC can be won without localizing anything:
- **Clip oracle:** give every frame its clip's label. The micro AUC this scores is how much
  of the metric is *clip classification*.
- **Length leak:** score each clip by its frame count alone. If this is high, clip length
  predicts the label.

A usable training corpus needs both near 0.5, **and** it needs negative (normal) examples.

### 2.2 Corpus table (frozen CLIP ViT-B/16, stride 8, `no_center_crop`)

| | DoTA | TAD | DADA archive (trimmed) | **DADA original T2 W=20** |
|---|---:|---:|---:|---:|
| Role | zero-shot benchmark (held out) | CCTV, tried in-domain | tried in-domain | **training corpus** |
| Train items (abn / normal) | **0** | 410 (200 / 210) | 1,530 (780 / 750) | **4,401 (3,242 / 1,159)** |
| Test items (abn / normal) | 1,397 (1,394 / **3**) | 100 (60 / 40) | 383 (191 / 192) | **1,106 (805 / 301)** |
| Test frames / positive share | 18,369 / 33.1 % | 11,045 / 7.2 % | 5,244 / 9.1 % | **22,120 / 33.2 %** |
| Median sampled length | 13 | 43 | **9** | 20 (fixed) |
| Score-head kernel coverage of median clip | 0.69 | 0.21 | **1.00** | **0.15** (kernel 3) |
| Frames in all-normal clips | 0.2 % | 78.4 % | 74.3 % | 27.2 % |
| **Clip oracle (micro)** | **0.5017** | 0.9226 | 0.9086 | **0.7037** |
| **Length leak (micro)** | 0.4993 | 0.8968 | 0.8654 | **0.5000** |
| Two-class test items (macro denominator) | 1,392 | 60 | 190 | **798** |
| Frame linear probe, `auc_macro` | **0.6708** | — | 0.5228 | 0.5983 |
| CLIP cosine, lag 1 | 0.965 | — | 0.963 | 0.968 |
| Between/within-clip feature variance | 3.19 | — | 2.72 | 2.77 |
| Verdict | clean, **cannot train** | micro = clip classification; collapses when trained | length leak + kernel spans the clip | **usable** |

Other corpora:
- **MSAD:** CCTV, clean, with negatives, but not ego-view traffic. It was the original
  training set.
- **PreVAD:** ships CLIP features only, with no pixels, so optical flow cannot be computed on
  it.

### 2.3 How T2 was built

The DADA-2000 **original** release has **no normal videos**. T2 therefore cuts **negatives
from inside the accident videos**, the frames before and after the annotated abnormal span,
into fixed windows of W = 20 sampled frames with hop 8. The geometry was gated before any
training:

| Gate W criterion | bar | W=16 | **W=20** |
|---|---|---:|---:|
| Length leak | ≤ 0.55 | 0.500 | **0.500** |
| Clip oracle | ≤ 0.75 | 0.7529 ✗ | **0.7037** |
| Abnormal sources retained | ≥ 0.90 | 0.983 | **0.957** |
| Kernel coverage (k=3) | ≤ 0.35 | 0.19 | **0.15** |
| Two-class test windows | ≥ 300 | 787 | **798** |

T2 hyper-parameters adapted to short windows: `score_head_kernel` 9 → 3, and
`mil_topk_pct` 16 → 5 (k = 4 of 20).

### 2.4 Imported normal pools: all rejected

The alternative to in-video negatives is borrowing normal clips from another dashcam
corpus. Each candidate was tested with a frozen-CLIP transfer probe (5 grouped folds). The
question is whether foreign negatives teach the accident or the *source*.

| | D2City (China, 700 clips) | BDD-A calm (US, 508 clips) | bar |
|---|---:|---:|---|
| Length leak | 0.5009 | 0.5000 | [0.45, 0.55] |
| R0: in-video negatives (reference) | 0.6763 | 0.6762 | — |
| **X: foreign negatives only**, `auc_macro` | **0.5864** | **0.5492** | ≥ 0.60 |
| Δ(X − R0), t95 | −0.090 [−0.101, −0.079] | −0.127 [−0.153, −0.101] | fail < −0.06 |
| Δ(M − R0): foreign *added* | −0.012 | −0.003 | ≥ −0.01 |
| **Source-shortcut AUC** | **1.000** | **1.000** | red flag > 0.90 |

**Every foreign dashcam corpus is perfectly separable from DADA on frozen CLIP** (DADA vs
DoTA: 0.9999 as well). A probe given foreign negatives learns the source, not the accident.

CCD (Car Crash Dataset) was also profiled:
- all clips are 5 s;
- its normals are a separate BDD100K pool, which is the same failure mode;
- crash-only windows carry a strong position shortcut (frame index alone gives AUC 0.965).

It was parked.

---

## 3. Training and evaluation on DADA-2000 T2 (W=20)

**Setup:**
- stage 2 (+ stage 1 KIP warm-up for KIP-on), 2,040 optimizer steps, batch 64, lr 5e-5;
- seeds 2024 / 2025 / 2026.
- The KIP-on and KIP-off configs differ only in `kip.enabled`, plus `lambda_rec` for v2.
- The T2 test set is scored raw. DoTA uses per-clip min-max, the protocol that reproduces
  LaGoVAD's released checkpoint.

**Arms:**
- **KIP-off:** the LaGoVAD trunk.
- **KIP-on (flow v1):** KIP with the flow target as originally built (raw pixel units).
- **KIP-on (flow v2):** the same, after z-scoring the 23 statistics on the T2 train split and
  setting `lambda_rec = 1/V` (§4.2).

### 3.1 Per-arm results (mean of 3 seeds)

| Arm | T2 micro AUC | T2 AP | **T2 macro AUC** | DoTA micro | DoTA AP | **DoTA macro** |
|---|---:|---:|---:|---:|---:|---:|
| KIP-off | 0.6182 | 0.4681 | **0.6248** | 0.5856 | 0.4023 | **0.6113** |
| KIP-on, flow v1 (raw) | 0.6054 | 0.4311 | 0.6046 | 0.5914 | 0.3820 | 0.6071 |
| KIP-on, flow v2 (z-scored) | 0.6258 | 0.4817 | 0.6305 | 0.6077 | 0.4105 | 0.6376 |
| *reference lines* | clip oracle 0.7037 | | frame probe 0.5983 | MSAD KIP-on 0.6408 | | MSAD KIP-on 0.6529 |

Per seed, T2 micro / macro:

| seed | KIP-off | KIP-on v1 | KIP-on v2 |
|---|---|---|---|
| 2024 | 0.6230 / 0.6334 | 0.6095 / 0.5983 | 0.6295 / 0.6288 |
| 2025 | 0.6164 / 0.6352 | 0.5990 / 0.6074 | 0.6275 / 0.6308 |
| 2026 | 0.6152 / 0.6057 | 0.6075 / 0.6083 | 0.6204 / 0.6320 |

**Sanity:** macro ≥ micro and micro < oracle on every arm, so none of them collapsed into a
window classifier. On TAD the same trunk did collapse: micro 0.924 ≈ oracle 0.923 while
macro fell.

### 3.2 Paired KIP A/B (Δ = KIP-on − KIP-off, n = 3)

| Pair | Metric | Δ | t95 | seed signs | reading |
|---|---|---:|---|:-:|---|
| **flow v1 − off** | **T2 micro** | **−0.0129** | **[−0.0249, −0.0008]** | − − − | **significant loss** |
| flow v1 − off | T2 AP | −0.0370 | [−0.0652, −0.0088] | − − − | significant loss |
| flow v1 − off | T2 macro | −0.0201 | [−0.0698, +0.0296] | − − + | n.s. |
| flow v1 − off | DoTA micro / macro | +0.006 / −0.004 | ±0.08 / ±0.10 | | n.s. |
| **flow v2 − off** | **T2 micro (pre-registered primary)** | **+0.0076** | **[−0.0002, +0.0154]** | + + + | **n.s.: bounded null** |
| flow v2 − off | T2 AP | +0.0136 | [−0.0032, +0.0303] | + + + | n.s. |
| flow v2 − off | T2 macro | +0.0058 | [−0.0384, +0.0499] | − − + | n.s. |
| flow v2 − off | DoTA micro / macro | +0.022 / +0.026 | [−0.011, +0.055] / [−0.019, +0.071] | + + + | n.s. |
| flow v2 − flow v1 | T2 micro / macro | +0.020 / +0.026 | [+0.001, +0.040] / [+0.016, +0.036] | + + + | the repair recovers the v1 loss |

The decision rule was written before the v2 runs: *capture removed × interval includes 0 →
"KIP-v1 neutral; write a bounded null; stop spending seeds"*. No fourth seed was added to
push the +0.0076 across zero, because that would be optional stopping.

### 3.3 Learning curve (KIP-off, fixed 2,040 steps, nested subsets of train source videos)

| Train sources | T2 micro | T2 macro | DoTA micro | DoTA macro |
|---|---:|---:|---:|---:|
| 25 % | 0.5947 | 0.6085 | 0.5773 | 0.6029 |
| 50 % | 0.6083 | 0.6180 | 0.5797 | 0.6081 |
| 100 % | 0.6182 | 0.6248 | 0.5856 | 0.6113 |

- In-domain gains ≈ +0.01 per doubling of data, and the gain is shrinking.
  - 25 → 50 %: +0.0137 [+0.002, +0.026].
  - 50 → 100 %: +0.0099 [−0.027, +0.046].
- **Zero-shot DoTA is flat over 4× data** (+0.008, inside the noise).
- Adding more T2-like data is therefore not the lever for the benchmark either.

---

## 4. The evidence that optical flow, as used here, does not help

The argument has three layers: the outcome, the mechanism, and the architecture.

### 4.1 Outcome: no benefit on T2 or DoTA

- **Flow v1 hurts:** −0.0129 T2 micro and −0.037 AP. These are the only intervals in the
  study that exclude zero, and all 3 seeds agree.
- **Flow v2 is neutral:** every interval includes zero, in-domain and zero-shot.
- **The only significant positive number (v2 − v1, +0.020) is a repair, not a gain.** It
  measures how much the broken loss was costing. It does not measure anything flow adds
  over KIP-off.

### 4.2 Why v1 hurt: the flow loss captured the shared trunk

**The target was unnormalized.** `L_KIP_rec` is an MSE on 23 raw pixel-unit statistics.

| Quantity (T2 train, 4,401 windows / 88,020 frames) | flow v1 |
|---|---:|
| MSE of an all-zeros predictor | 71.15 |
| **MSE of a constant (global-mean) predictor, V** | **31.64** |
| MSE of a per-clip-mean oracle, W | 16.21 |
| Measured `kip_rec` | 11.62 |
| Share of E[s²] carried by `mag_max` alone | **83.0 %** |
| `kip_rec` share of the total loss (stage 2) | **92.6–93.0 %** |

**It dominated the gradient.** At `lambda_rec = 1.0`, a loss ~32× larger than the O(1)
classification terms reaches the shared temporal encoder. We probed this with
`core/tools/grad_probe.py` (8 batches × 3 seeds):
- **ρ = ‖g_KIP‖ / ‖g_task‖ = 11.6 at the end of stage 1, and 3.1 at the end of stage 2**, so
  KIP moved the trunk 3× further per step than the anomaly objective did;
- **cos(g_KIP, g_task) = −0.001**, i.e. orthogonal: the flow loss did not fight the task, it
  **consumed trunk capacity**.

Corroboration:
- 44 % of the reconstruction gain came from rewriting `v^t`;
- every task loss ended **24–32 % higher** with KIP on.

**The fix, and what it shows.** Z-scoring the 23 statistics gives V = 1.005, and the weight
was set to `lambda_rec = 1/V = 0.995`. The weight was derived from the target, not swept.
Then:
- ρ fell to **0.18** (0.152 / 0.187 / 0.199);
- the flow loss share fell to ~48 %;
- the cost disappeared.

What remained is §4.1's null. **Once flow stops damaging the trunk, it adds nothing.**

### 4.3 Why v2 adds nothing: the flow head cannot learn in-clip motion from this target

On the z-scored target, all 23 statistics carry equal weight:

| Quantity (stage-2 end, 3 seeds) | value | reading |
|---|---:|---|
| R² vs constant predictor, 1 − K/V | 0.280 (0.290 / 0.271 / 0.280) | fits *something* |
| **R² of the per-clip-mean oracle** (knows only *which clip*) | **0.349** | the bar for "fits in-clip dynamics" |
| **K (0.723) vs W (0.654)** | **K > W** | **the flow head is worse than a clip-identity predictor** |
| R², magnitude block (7 stats) | 0.357 | learns *how much* a clip moves |
| **R², direction block (16-bin histogram, 69.6 % of the target)** | **0.071** | **learns essentially no motion direction** |
| Between-clip share of target variance | 34.9 % | a third of the target is clip identity |

Three consequences:
1. **The phase-4 claim that "the flow head fits 28 % of within-clip flow variance" does not
   survive normalization.** It was carried by `mag_max`, one statistic worth 83 % of the raw
   target's energy.
2. From frozen CLIP features (lag-1 cosine **0.968**: consecutive sampled frames are nearly
   identical), the model can recover **how much a clip moves**, not **how motion evolves
   inside it**.
3. **For a dashcam, frame-global flow is dominated by ego-motion.** An accident is usually a
   *local* event: one vehicle or pedestrian in a corner of the frame. Global statistics
   register the car turning and nearly miss the car cutting in. The target is measuring the
   wrong thing for this task.

### 4.4 Architecture: flow never reaches the part that is supposed to use it

- **The gate is not trained.** `KinematicShift` computes the shift as
  `(ratio * max_shift).floor().long()`, and that integer cast cuts the gradient: all 6 gate
  tensors get `grad = None`.
- **It is also near-constant in value.** Sweeping its whole reachable input range moves the
  shift by 0–4 of 128 channels across 8 seeds.
- **So every KIP-on arm on this branch applies a fixed ≈ 50 % temporal channel shift**,
  whatever the flow says. "Motion-gated" does not describe what runs.
- **The one KIP term aligned with the task is aligned by construction.**
  `cos(g_kin, g_task)` = +0.61 / +0.65 / +0.72, but `L_kin` is a MIL-BCE on the motion curve
  against **the same video label** as the main MIL loss. It is a second MIL head, not motion
  evidence. The cosine is ≈ 0 at stage 1, where `L_kin` is not optimized.

### 4.5 Corroboration from outside T2 (for context only)

These runs were not on T2, and some used the `v3` branch's code:
- **MSAD → DoTA.** KIP-on beat KIP-off by +0.09 on DoTA. An ablation that keeps **only the
  fixed 50 % shift (no flow, no flow head, no KIP losses)** is statistically
  indistinguishable from full KIP (Δ = +0.011, t95 [−0.059, +0.081]), and alone beats KIP-off
  by +0.10 ± 0.035. **All of KIP's measured gain was temporal smoothing, which requires no
  optical flow.**
- **A trainable (rank) gate was the worst KIP arm there** (−0.068 vs the fixed shift).
- **DADA archive.** The same fixed shift **reversed sign** (−0.092 on DoTA), because the
  training clips were as short as the kernel. A component whose sign flips with clip length
  is a smoothing hyper-parameter, not a motion mechanism.

### 4.6 Cost side

Flow needs a RAFT pass over every training frame. For DADA original that is 94 GiB of
frames, extracted in shards on Colab. It cannot be computed on feature-only corpora
(PreVAD). It buys nothing measurable in §4.1.

### 4.7 What this evidence does *not* show

- It does **not** show that motion is useless for traffic anomaly detection. It shows that
  **this** motion pathway is not useful:
  - frame-global flow statistics as the target;
  - an auxiliary regression from frozen per-frame CLIP;
  - a gate that does not train.
- n = 3 seeds. The v2 null is bounded: an in-domain effect larger than ≈ 1.5 AUC points is
  excluded. A smaller one is not.
- One training corpus. DoTA zero-shot intervals are wide (±0.03–0.07).

---

## 5. Where the remaining headroom is (data-driven)

| Lever | Evidence | Status |
|---|---|---|
| KIP-v1 / frame-global flow | §4: harmful → neutral | exhausted |
| More seeds | pre-registered stop | exhausted |
| More T2-like data | §3.3: DoTA flat over 4× | exhausted |
| Imported normal pools | §2.4: source shortcut 1.000 on two corpora | exhausted |
| **Supervision / head** | DoTA frame linear probe **0.6708** > best trained DoTA macro 0.6529 (0.6376 from T2) | open |
| **Representation (video-aware backbone)** | CLIP lag-1 cosine 0.968; the flow head cannot recover in-clip motion from CLIP (K > W); SimpleTAD with VideoMAE reports DADA→DoTA 80.3 (confounded with fine-tuning and frame-level labels) | open, **untested** |

---

## 6. Questions for the advisor: how should the motion pathway change?

Options we see, each with the measurement that would justify it:

1. **Keep flow supervision, change the target.** Replace the 23 frame-global statistics with
   a **spatial** target: flow statistics pooled on a grid (e.g. 4×4 regions), or
   **ego-motion-compensated residual flow**, i.e. the motion of objects rather than of the
   camera. *Justifying measurement:* the flow head must beat the per-clip oracle (K < W) on
   the new target before any A/B.
2. **Drop flow; get motion from the representation.** Add a **CLIP-aligned video encoder**
   (ViCLIP / InternVideo2-CLIP) as a second stream into the temporal encoder, and keep frozen
   CLIP for the text-conditioned path. This keeps the definition-conditioned heads intact.
   The current KIP (flow head, shift, motion head) would be retired.
   - *Justifying measurement (pre-registered, not yet run):* a frame linear probe on
     video-encoder features for the DoTA and T2 test sets. Adopt if it exceeds the CLIP probe
     by **≥ +0.10 macro on DoTA**; drop the idea if the gap is within ±0.03.
   - Cost: test-set extraction only, no training.
3. **Keep CLIP, fix the supervision.** The frozen features already carry more frame-level
   signal on DoTA (probe 0.6708) than any trained head extracts. This is independent of the
   motion question.
4. **A trainable gate** (exists on branch `v3`). This is *not* recommended by our data: it was
   the worst arm on MSAD, and it would still be driven by a flow target that carries clip
   identity rather than in-clip motion.

Specifically, we would value the advisor's view on:
- (a) whether option 1 or option 2 is the more defensible thesis contribution;
- (b) whether dropping KIP in favour of a video stream is acceptable given the thesis framing
  (*"LaGoVAD + a kinematics pathway"*);
- (c) whether a well-diagnosed negative result on flow supervision (§4) is publishable as it
  stands.
