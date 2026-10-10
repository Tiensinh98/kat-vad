# Proposed Road-Traffic Video Anomaly Detection Architecture — **KAT-VAD v2.1**
### The adopted v2 (LaGoVAD + video stream + Clip-Referenced Normalization) + **SG-NM**, training only

**Supersedes:** `KAT_VAD_PROPOSAL_v2.md` and the earlier v2.1 draft. v2.1 is the adopted v2 model, unchanged at inference, plus one training-only module (SG-NM) and the two steps that test it (E4a, E4). Before them comes one free arm, **A3-ign** (step E4-0, §11.4). It fixes the DVS anchor label that SG-NM is trained on, and so decides the base `B` that SG-NM is added to and compared against.

**Evidence base:**
- the T2 study (`REPORT_T2_OPTICAL_FLOW_FOR_ADVISOR.md`, 2026-09-26), quoted as "T2 study §x";
- the v2 results (`REPORT_V2_RESULTS.md`, 2026-10-07, with the read-outs of 2026-10-09), quoted as "results §x";
- numbers marked *derived* are computed from those, with the arithmetic shown.

**Companion:** `KAT_VAD_v2.1_ARCHITECTURE.md` (every component with its tensor shapes).
**Date:** 2026-10-11.

> **Design rule (unchanged from v2).** A component enters only if (i) it targets a failure that was **measured**, (ii) a project paper supports the mechanism, and (iii) it can be checked cheaply **before** it is trained.
> - v2 met the rule with two input changes and a protocol fix. v2.1 adds one training-only module. Its target is now measured (results §6.3), not derived.
> - **The deployed network does not change.** If SG-NM fails its gate or its rule, v2.1 is v2.
> - DoTA-CAP-dev suggested SG-NM, and DoTA-CAP-eval has been opened. Every rule below is therefore fixed now, before any SG-NM number exists, and the final read is labelled a **second look at an opened set**.

---

## S. The model in one view

```
frame ─► CLIP ViT-B/16 (frozen) ─► x ─► CRN (R2) ─► x̃ ──────────────────────────────────────────────┐
causal 16-frame clip, 1.5 s ─► VideoMAE V2-S (frozen) ─► u ∈ ℝ^384 ─► CRN (R2) ─► c·(⊘σ_u) ─► W_u (0-init) ─(+)─► h
h ─► LaGoVAD temporal encoder ─► V^t ─┬─► CoAttn(V^t, z) ─► H_bin → y^bin, H_mul → y^mul       (DoTA: stride 3, whole clip)
                                      └─► sg ─► SG-NM (training only) ─► S̃ ─► L_consist on y^bin
```

| Part | Status | Evidence |
|---|---|---|
| LaGoVAD trunk, heads, CLIP frame stream | kept | the baseline |
| VideoMAE V2-S **video stream**, zero-init residual | adopted in v2 | A3 − A1 **+0.098 [+0.090, +0.107]** on DoTA-CAP-eval, 5/5 seeds (results §4.2) |
| CRN, per-clip median (R2) | adopted in v2, inside A3 | removes the corpus shortcut on `V^t` (≈ 1.000 → 0.55–0.57); A3's gain survives position control, A2's does not (results §4.3) |
| Protocol B (DoTA at stride 3, whole clip) | adopted in v2 | +0.033 [+0.019, +0.048] on the same checkpoints (E1) |
| **SG-NM** | **new in v2.1, training only** | gated by E4a, then tested by E4 |

---

## What changed (v2 → v2.1)

| # | v2 | v2.1 | Measured gap | Support | Check before training |
|---|---|---|---|---|---|
| **N1** | SG-NM deferred | **SG-NM, training only.** A branch on the detached temporal features `sg(V^t)` learns 2 normal prototypes per sequence, scores every step by its reconstruction error `S̃`, and `L_consist` pulls `σ(y^bin)` toward `S̃`. Removed at inference. | **Gap S** (§2.2): the score is locked to the moment of impact. Its argmax lies inside the span on only 61 % of clips, and the pre-anomaly score is high (0.55 vs 0.86 inside). MIL top-k trains 4 of 20 steps, and `L_neg` is off. | DSANet: SG-NM +0.94 AP; its score curves track the ground-truth intervals instead of peaks | **E4a:** train the detached branch on A3's frozen E3 checkpoints (no trunk training); continue only if `S̃` adds ranking information to `y^bin` |
| **N2** | E3 (20 runs) closed | **E4:** A3 + SG-NM, 1 pilot + 5 seeds; A3's E3 runs are the paired control | — | — | the auxiliary rule (§11.3) |
| **N3** | "Motion stream"; "kinematics-aware" | **"Video stream"**; no motion or kinematics claim | D6: shuffling the 16 frames changes the result by ±0.0004 | — | — (wording) |
| **N4** | Δ decided on raw macro | Every Δ is printed with the **position-controlled** Δ (`f_2`) and `p_T2` beside it, and the rule has a position condition | results §4.3, §6.1–§6.2 | — | — |
| **N5** | DoTA-eval and T2-test sealed | DoTA-CAP-eval and T2-test **opened** at the v2 Final; the v2.1 final read is labelled a second look | results §8 | — | — |
| **N6** | DVS anchor label `span`: the whole anchor window is `y^p` = 1 (LaGoVAD default, never chosen for v2) | **E4-0: A3-ign**, A3 with `dvs_anchor_mode = ignore`, 5 seeds. A pre-registered rule (§11.4) picks the base `B` ∈ {A3, A3-ign} for E4a, the pilot and E4 | ≈ 54 % of an abnormal T2 window's steps are normal but trained to 1 on spliced sequences (C29, §2.2). A competing cause of Gap S, and in direct conflict with `L_consist` | C29; DADA Phase 1 and TAD measured `ignore` only in collapsed regimes, n = 1 (§11.4) | — (a flag; the code exists) |

### Changes from the earlier v2.1 draft

| Earlier draft | This version | Why |
|---|---|---|
| Built on the pre-results v2: E0–E3 still to run, sliding-window inference, 2,040 steps | Built on the adopted v2 and its results: A3, whole-clip protocol B, DoTA-CAP, 1,740 steps | v2 is closed |
| SG-NM's gap derived from T2 numbers | Measured on DoTA-CAP-dev (results §6.3), with the constraint from the failed post-hoc hold (§2.2) | the evidence now exists |
| E2(f) motion-content checks; a wording rule for "kinematics-aware" | Removed | D6 already answered the question: the stream is appearance, and "motion" may not be claimed |
| I3D as an encoder candidate | Removed | the encoder was chosen (V2-S) |
| "DVS steps with `y^p` = 1 are never candidates" | Candidates are chosen **per window segment** | Under the old rule an unspliced abnormal window, which has `y^p` = 1 on every step, had no candidates |
| E4 decided on DoTA-dev raw macro | DoTA-CAP-dev, raw **and** position-controlled | Every arm has the video stream, so DoTA-CAP is the only endpoint, and position is the central confound |
| Deferred items H1–H9 | **DF1–DF8** | The results report uses H1–H6 for its own directions |
| File paths, line numbers, setting names | Removed | The proposal describes the design, not its implementation |

---

## 0. Design summary

**Baseline.** LaGoVAD (ICLR 2026) remains the trunk: the only venue-eligible weakly supervised baseline evaluated on real traffic data. It trains correctly on T2 (T2 study §3.1).

**v2 as adopted (A3).**
- Two input changes on that trunk: a frozen VideoMAE V2-S video stream added through a zero-initialized residual, and Clip-Referenced Normalization (per-clip median) on both streams.
- DoTA read at the training step rate, with the whole clip in one pass.
- No new loss.

**What v2 achieved (DoTA-CAP-eval, 560 clips, opened once):**
- A3 macro **0.7785** vs **0.6979** for the LaGoVAD trunk on raw CLIP (A0): **+0.090 [+0.065, +0.115]**, positive on 5/5 seeds.
- Dev − eval optimism is negative for every arm, so dev selection did not inflate the numbers.

**What v2 did not achieve:**
- A cubic prior on relative position in the clip (`p_T2`, fitted on T2, no pixels) scores **0.8256**, above A3.
- After controlling for it, A3's gain is **+0.021 [+0.009, +0.034]** (results §4.3).

**What v2 leaves, from the per-clip error analysis on DoTA-CAP-dev (results §6):**
- **Gap P — the benchmark's position convention.** DoTA cuts clips so accidents sit at 0.55–0.70 of the clip. A3 is flat across position (0.70–0.81); `p_T2` ranges from 0.43 to 0.98. This is a property of the benchmark, not a model gap. It is handled by reporting rules and by the report's position-shift test, not by the model.
- **Gap S — scores locked to the moment of impact.** The score peaks 1–2 s after onset; its argmax lies inside the span on only 61 % of clips; the pre-anomaly score is high. **v2.1's change targets this gap.**
- **Gap N — non-ego accidents.** 0.697 vs 0.804 for ego accidents. Not addressed in v2.1: CLIP tiles failed their probe; detector crops are a v3 direction.

**The v2.1 change: SG-NM, training only.**
- DSANet's Self-Guided Normality Modeling, adapted to 20-step windows.
- A branch on the detached temporal features learns what *this* sequence's normal steps look like, rebuilds every step from those normal prototypes alone, and treats the reconstruction error as a dense, content-based anomaly target for every step.
- One loss pulls the detector toward that target. The branch is dropped at inference.

**Name.** D6 rules out "motion" and "kinematics-aware" for the video stream, and SG-NM is not a motion module. The acronym can stay as the project's name. The thesis text does not describe the model as kinematics-aware unless the report's direction H5 (an order-sensitive stream, DF1) passes its gate.

---

## 1. Where v2 stands

### 1.1 The numbers v2.1 starts from

| Measure | Value | Source |
|---|---|---|
| A3 macro, DoTA-CAP-eval / -dev | 0.7785 / 0.7576 | results §4.1 |
| A0 macro, DoTA-CAP-eval / -dev | 0.6979 / 0.6768 | results §4.1 |
| `p_T2` (position only), DoTA-CAP-eval / -dev | 0.8256 / 0.8220 | results §4.1 |
| A3 − A0, A3 − A1 (eval, paired t95) | +0.090 [+0.065, +0.115]; +0.098 [+0.090, +0.107] | results §4.2 |
| A3 − A0 after position control (`f_2`), eval / dev | +0.021 [+0.009, +0.034] / +0.023 [+0.013, +0.033] | results §4.3 |
| A1 − A0 (CRN alone), DoTA-eval | −0.007 [−0.034, +0.019] | results §4.2 |
| T2-test macro: A0 / A2 / A3 | 0.6307 / 0.7080 / 0.6950 | results §4.4 |
| Frame-order control (D6): ordered − shuffled | −0.0004 / −0.0003 | results §3 |
| Per-clip correlation of A3's AUC with `p_T2`'s | 0.135 | results §6.1 |
| A3 argmax inside the span | 61 % of clips | results §6.3 |
| A3 mean score before / inside the span | 0.55 / 0.86 | results §6.3 |
| A3 ego / non-ego | 0.804 / 0.697 | results §6.4 |

### 1.2 What may and may not be claimed

Inherited from the results report §5, plus the SG-NM lines (§11.3):

| ✅ May say | ❌ May not say |
|---|---|
| "v2 as a whole (CRN + V2-S) improves DoTA-CAP (n/1397) over A0" (+0.090 on eval) | "motion" or "kinematics-aware" for the video stream |
| "the video stream (V2-S) improves DoTA-CAP on top of CRN" (+0.098) | a DoTA-CAP number beside LaGoVAD's 62.60 or a full-DoTA number |
| the gain is ≈ +0.02 after controlling for the T2 position prior, on both dev and eval | "+0.09" without `p_T2` beside it |
| no selection optimism (dev ≤ eval for every arm) | "CRN improves transfer" (A1 − A0 contains 0) |
| CRN removes the T2 ↔ DoTA corpus shortcut in `V^t` | that A2/A3 "pass O1"; the required wording is "O1 FAIL as registered; not collapsed under O1′" |
| *v2.1, only if E4 passes:* "SG-NM improves DoTA-CAP-dev detection", with the second-look label on any eval number | *v2.1:* "SG-NM improves detection beyond the position prior" unless the `f_2` interval excludes 0 |
| *v2.1, only if the base rule adopts it:* "dropping the whole-anchor DVS label improves DoTA-CAP-dev detection" (§11.4) | *v2.1:* "the whole-anchor label causes Gap S" unless the §11.4 mechanism conditions hold |

---

## 2. Gap analysis

### 2.1 The T2-study gaps and how v2 closed them

| Gap (T2 study) | v2 fix | Outcome (results) |
|---|---|---|
| **G1.** Frame CLIP carries no in-clip motion: lag-1 cosine 0.968, flow head K 0.723 > W 0.654, direction R² 0.071 | Video stream (VideoMAE V2-S) | A large gain (+0.098 over CRN), but D6 shows frame order does not matter. The gain is appearance from a second, video-trained encoder, not motion. **"Motion" stays an open question** (DF1). |
| **G2.** About 75 % of the CLIP variance is clip identity; the source is separable at AUC ≈ 1.000; DoTA is flat over 4× data | CRN (R2) | The shortcut on `V^t` falls to 0.55–0.57. CRN alone does not transfer (A1 − A0 contains 0) and costs ≈ 0.013 macro in-domain (A3 vs A2 on T2-test). Inside A3, it is what makes the gain separable from position. |
| **G3.** Training read at 0.27 s per step, DoTA at 0.80 s | Protocol B (stride 3) | +0.033 [+0.019, +0.048] on the existing checkpoints |
| **G4.** An auxiliary loss captured the trunk: ρ = 3.1, every task loss up 24–32 % | No new loss in v2; a zero-init residual | No capture. v2.1's single new loss is admitted only under G4 controls (§5.3). |
| **G5.** n = 3; DoTA-macro t95 half-width 0.045 | n = 5, paired t95 | *Derived from the reported intervals:* paired half-widths on DoTA-CAP-dev were 0.012 (A3 − A1) to 0.029 (A3 − A0) |

### 2.2 The gaps v2 leaves

**Gap S — scores locked to the moment of impact** (measured; results §6.3). **This is the v2.1 target.**
- A3's within-clip z-score peaks about 1–2 s after onset (+0.64 at +20 frames) and drops to about a quarter of that by +4 s.
- The argmax lies inside the anomalous span on **only 61 %** of clips.
- The mean score before the anomaly is **0.55**, against 0.86 inside. The two are separable (inside > before on 88 % of clips), but the baseline level is high.
- The results report reads this as expected behaviour for MIL top-k: k = 4 of a 20-step window rewards only the few most salient steps. DSANet makes the same diagnosis of MIL ("fixates on the most salient segments") and proposes SG-NM as the remedy.
- **What the A3 objective actually supervises (code, checked 2026-10-11).** `L_MIL` is not the only term on `y^bin`:
  - `L_MIL` uses k = max(1, ⌊L/5⌋) on **every** sequence: 4 on an unspliced 20-step window, up to 20 on a 100-step DVS sequence.
  - `L_dvs` adds a **dense** per-step BCE on every normal sequence (all steps → 0) and on every DVS-spliced sequence. A3 trains with `dvs_anchor_mode = span`, so in a spliced abnormal sequence **every step of the anchor window is labelled 1**, although ≈ 54 % of an abnormal T2 window's steps are normal (the whole-anchor label, lesson C29). A top-k term inside the anchor (k = ⌊n_pos/4⌋) comes on top.
  - So the pre-anomaly steps of an abnormal window are pushed **up** densely on spliced sequences (≈ 30 % of anomaly anchors, θ = 0.7). That is a candidate cause of the high pre-anomaly level, alongside the MIL reading. The code already has the switch that removes it (`dvs_anchor_mode = ignore`), which has never been run on T2.
- **Two constraints from the read-outs of 2026-10-09:**
  1. **A post-hoc fix does not work.** A causal max-hold on A3's score (window = the T2 median span) lost macro overall (Δ −0.071) and on `f_2` (Δ −0.022). Its gain on long spans (+0.037) has an interval that contains 0. On the 277 of 569 clips with ≥ 3 s of normal driving after the span, it cost **−0.125**: part of the decline after the peak is the model correctly tracking the span's end. A hold extends the score in time regardless of what the frames show.
  2. **The share-bin slope is mostly position.** That A3 loses more to `p_T2` as the anomaly's share of the clip grows is largely a position effect, so it is **not** used as evidence here.
- **What a remedy must therefore do:** (i) act in **training**, and (ii) be **content-based**: keep a step high while its content stays abnormal and let it fall when the content returns to normal.
- **What else is missing from the T2 objective:** `L_neg`, LaGoVAD's within-video contrast between a video's anomalous and normal parts, is off. The T2 configuration provides no caption features (results §1). Nothing in the objective compares a window's steps with that window's own normal steps.

**Gap P — position is a benchmark convention** (results §6.1–§6.2). **Not a model target.**
- A3 is almost flat across accident position (0.70–0.81), while `p_T2` ranges from 0.43 to 0.98.
- Away from DoTA's usual cut, A3 wins by a wide margin: +0.266 when the accident centre is before 0.40 of the clip, +0.093 when it is at or after 0.70.
- `p_T2` needs the clip's end point and is unusable in streaming. Feeding it to the model would be benchmark fitting (results §8, "not proposed").
- **Handled by reporting:** every Δ is printed with `f_2` and `p_T2` beside it (§11.1). The position-shift stress test (report direction H1) is the experiment that settles how the thesis tells this story. It is a protocol, not a model change, and is listed as DF8.

**Gap N — non-ego accidents** (results §6.4). **Not addressed in v2.1.**
- Non-ego accidents score 0.697 against 0.804 for ego accidents.
- Both encoders squash the whole frame into one global vector, which dilutes small, distant events.
- CLIP tiles and text-guided windows failed their probe (report direction H3, killed). Detector crops are a v3 direction.
- SG-NM works on the same global vectors, so it is **not expected** to help here. Ego and non-ego are reported separately.

---

## 3. Baseline: unchanged, LaGoVAD

| Criterion | Pi-VAD (arXiv) | DSANet (AAAI 26) | RefineVAD (AAAI 26) | **LaGoVAD (ICLR 26)** |
|---|---|---|---|---|
| WS · venue · official code | ✅ · ❌ · ✅ | ✅ · ✅ · ✅ | ✅ · ✅ · ✅ | ✅ · ✅ · ✅ |
| Real traffic evaluation | ❌ | ❌ | ❌ | ✅ DoTA / TAD |
| Measured in this project | — | — | its trainable-gate idea was the worst KIP arm on MSAD | trains correctly on T2; the trunk of the adopted A3 |

DSANet is not the baseline. Its SG-NM is the source of v2.1's training-only module.

---

## 4. Architecture

### 4.1 Data flow

```
                  ┌─ frame @ step t ──────────► CLIP ViT-B/16 image (frozen) ─► x_t ─► CRN (R2) ─► x̃_t ─────────────────────────┐
video @ rate r ───┤                                                                                                               (+)─► h_t ∈ ℝ^512
                  └─ causal clip 16 f @ 10 fps ─► VideoMAE V2-S (frozen) ─► u_t ∈ ℝ^384 ─► CRN (R2) ─► c·(⊘σ_u) ─► W_u (zero-init) ┘

h ─► LaGoVAD temporal encoder (2 layers, RoPE, outer residual) ─► V^t ─┬─► H_bin, language-agnostic path ───────────────────┐
                                                                       ├─► CoAttn(V^t, z) ─► V^u ─► H_bin, language-guided path ─┴─► y^bin
                                                                       │                        └─► H_mul ─► y^mul
                                                                       └─► stop-grad ─► SG-NM (training only) ─► S̃ ─► L_consist vs σ(y^bin)
definition Z ─► CLIP text (frozen) + soft prompts ─► z
off the critical path:  ATS(y^bin) ─► MLLM ─► incident report      (designed only; not implemented in core/)
```

### 4.2 Components

| | Component | Status |
|---|---|---|
| (a) | Frame stream: frozen CLIP ViT-B/16 | unchanged |
| (b) | Video stream: frozen VideoMAE V2-S on the causal 1.5 s clip; `h_t = x̃_t + W_u·(c·ũ_t ⊘ σ_u)`, `W_u` zero-init | adopted in v2 (§5.1) |
| (c) | CRN with the per-clip median (R2), both streams, parameter-free | adopted in v2 (§5.2) |
| (d)–(g) | Temporal encoder, co-attention, `H_bin` (two paths and a learned blend), `H_mul` | LaGoVAD, unchanged |
| (h) | ATS reasoning layer | off the critical path; designed, **not implemented** in `core/` |
| **(i)** | **SG-NM on `sg(V^t)`: a per-step normality score `S̃` and one loss on `y^bin`** | **new in v2.1, training only (§5.3)** |

---

## 5. The components

### 5.1 Video stream (as adopted in v2)

**What it is.**
- A frozen VideoMAE V2-S (distilled, K710 post-trained) reads the 16 frames at 10 fps that end at each step: causal, 1.5 s.
- Each clip is squashed to 224², like the CLIP frame.
- Its token-mean feature `u_t ∈ ℝ^384` goes through CRN, a fixed per-channel scale `σ_u` and a fixed scalar `c`, then a zero-initialized linear map `W_u` (≈ 0.20 M parameters) adds it to the CLIP input.

**Why each choice:**
- **A video encoder.** SimpleTAD found video masked-modelling encoders to be the representation that wins on DoTA/DADA. The K-probe confirmed it here: `u` 0.760 vs `x` 0.615, and +0.043 beyond position.
- **V2-S over V2-B.** By the tie rule: transfer Δ 0.151 vs 0.133, gap < 0.02. V2-B was ahead beyond position (+0.037 vs +0.018), so it remains a candidate (DF4).
- **Frozen.** The weakly labelled data is small, and the source is separable at AUC 1.000: fine-tuning would learn DADA.
- **Causal 1.5 s.** SimpleTAD's window; it spans the approach-to-contact phase and needs no look-ahead.
- **Zero-init residual.** Training starts exactly at the CRN-only function, and the CLIP and language paths are untouched. Under AdamW zero-init fixes the **start**, not proximity, so the video-stream share `ρ_u` is logged as the evidence that the stream is used (every training step, as `motion_share` beside `‖W_u‖_F` in `metrics.jsonl`).
- **Fixed `σ_u`, no LayerNorm.** A per-token norm would erase the deviation size that CRN creates. With `σ_u` the size reaches `V^t` through the encoder's outer residual, at ≈ 0.44× the re-normalized branch, on `H_bin`'s language-agnostic path (architecture §5).
- **Squash.** Measured DADA frames are 1584 × 660 and DoTA frames 1280 × 720. Squash leaves a 1.24× horizontal and 1.09× vertical scale gap; letterbox gives 1.24× both ways and only ≈ 93 px of DADA content.

**What D6 changed.** Only the name and the claims: it is the **video stream**, a second appearance encoder trained on video. The design is unchanged.

### 5.2 Clip-Referenced Normalization (as adopted in v2)

**Definition.**

```
x̃_t = s · ( x_t − median_τ x_τ )     CLIP stream  (s: one scalar fixed on T2-train so that E‖x̃‖ = E‖x‖)
ũ_t =       u_t − median_τ u_τ        video stream (then scaled by c and σ_u)
```

The median is per dimension, over the **source video** in training and over the **clip** at test. Both CRN and the video-stream scaling are baked offline into a v2 input cache with a manifest; training and evaluation refuse a cache whose manifest does not match the run's `v2.crn` / `v2.motion`.

**What it removes.** The clip-level fixed effect: scene, camera, dataset signature. It is the within (fixed-effects) estimator, and both DoTA metrics already discard any per-clip constant.

**How R2 was chosen.** Over four references by a pre-registered check: share histograms, a position-residualized deviation, and a transfer-probe veto (v2 proposal §4.2). R2 tied with the mean on the decision metric (0.7092 vs 0.7081). Its transfer gain was +0.037 [+0.027, +0.047].

**What it does and does not do (results §7):**
- It removes the corpus shortcut.
- It is what lets A3's gain survive position control: A3 +0.021 vs A2 +0.004 ∋ 0.
- It does **not** transfer on its own.
- It has a small in-domain cost (≈ 0.013 T2-test macro).
- No stand-alone claim is made for it.

**Streaming.** R2 needs the whole clip, so the adopted pipeline is not streamable. A streaming variant would need the past-only reference.

### 5.3 SG-NM: Self-Guided Normality Modeling, training only (new in v2.1)

**What it is.** DSANet's SG-NM, sized for 20-step windows and made safe for this trunk. For each training sequence it:
1. picks steps that are probably normal;
2. distils them into 2 "normal" prototypes;
3. tries to rebuild every step from those prototypes alone;
4. treats the reconstruction error as a per-step score `S̃`, and pulls the detector's scores toward it.

It is used only in training. The deployed network is A3's.

**Why it is in v2.1:**
1. **It targets Gap S directly.**
   - MIL top-k gives 4 of 20 steps a target.
   - `S̃` gives **every** step a target, and that target compares each step with the sequence's **own** normal steps: the within-video contrast that is missing while `L_neg` is off.
2. **It is content-based and acts in training**, the two properties the failed hold lacked (§2.2).
   - `S̃` is high where a step cannot be rebuilt from the normal prototypes and low where it can.
   - A step in the aftermath that still looks abnormal (stopped vehicles, debris) keeps a high target. A step where normal driving has resumed gets a low one.
   - The hold extended scores by time alone and lost −0.125 on exactly the clips where normal driving follows the span.
3. **Paper support (DSANet, AAAI 2026):**
   - SG-NM added +0.94 AP on XD-Violence (85.00 → 85.94, Table 5).
   - Its score curves track the ground-truth intervals, where the MIL baseline fires on salient peaks (Fig. 7).
   - Normal frames sit at cosine distance 0.35 from their prototypes, anomalous ones at 0.69 (Fig. 5).
4. **It is visual.** It does not depend on the DADA category texts that make `L_neg` delicate (DF3).
5. **It cannot learn the clip-position prior directly.**
   - Training sequences are 20-step slices (5.3 s) of source videos, and neither the branch nor the trunk is told where a slice sits in its video. RoPE encodes only offsets within a sequence.
   - Position information can still enter indirectly: the R2 reference spans the whole source video, so scene drift can make the deviation larger at the ends.
   - So the rule checks the position-controlled Δ, and the position probe on `V^t` is reported (§11).

**Placement: a training-only branch on the temporal encoder's output `V^t`.** This is where DSANet attaches it: the output of its shared temporal module.
- **Not `V^u`.** The co-attention mixes in the definition text and is post-LN without an outer skip, so `V^u` is re-normalized. Normality should be visual and independent of the definition.
- **Not the input `H`.** There is no temporal context before the encoder, and CRN already supplies the clip-level reference there.
- **`V^t` keeps deviation size**, through the encoder's outer residual.

**Design** (all branch inputs detached; shapes in architecture §9):

```
inputs   V̄ = sg(V^t) ∈ B×L×512,  s̄ = sg(σ(y^bin)) ∈ B×L,  segment labels,  valid-step mask

1. candidates m_n, per window segment:  normal segment   → every valid step
                                         abnormal segment → its ⌈0.4·20⌉ = 8 valid steps with the lowest s̄
2. prototypes   P = LN( MHA(query = Q, key = value = V̄, key mask = m_n) )         Q ∈ ℝ^{2×512} learned;  P ∈ B×2×512
3. L_compact = mean over candidates of  min_k ( 1 − cos(V̄_t, P_k) )
4. decoder      G  = MLP(V̄)                                   512 → 512 → 512, GELU
                R1 = LN( MHA(G, P, P) )                        NO residual from G
                R1 = LN( R1 + FFN(R1) )                        FFN 512 → 1024 → 512
                R2 = LN( R1 + MHA(R1, P, P) )
                R  = LN( R2 + FFN(R2) )                        R ∈ B×L×512
5. e = 1 − cos(R, V̄) ∈ [0, 2]^{B×L};      L_rec = mean over candidates of e
6. S̃ = clip( (e − μ_e) / (3·σ_e), 0, 1 )                      μ_e, σ_e: EMA (0.99) of the mean and std of e over normal-segment steps, detached
7. L_consist = mean over valid steps of ( σ(y^bin) − sg(S̃) )²

branch losses   L_compact + L_rec        reach only the branch's own parameters (its input is detached)
detector loss   λ_n · L_consist          the only SG-NM term that reaches the detector, through y^bin
```

**Candidates and DVS.** DVS is on in A3: LaGoVAD's synthesis splices an anchor window with normal filler windows, and the first stage is defined per **window segment**.
- In a spliced abnormal sequence, the filler windows are labelled normal, so all their steps are candidates. Only the anchor window contributes its bottom 8.
- An unspliced window is one segment.
- This also fixes the earlier draft's rule ("`y^p` = 1 steps are never candidates"), which left an unspliced abnormal window with **no** candidates.

**Three departures from DSANet, and why** (they fix the bugs found in the earlier SG-KN draft):
1. **A detached branch with one-way consistency.** DSANet trains both sides of its consistency loss. Here the branch reads `sg(V^t)` and the target is `sg(S̃)`, so:
   - the trunk cannot collapse its features to make reconstruction easy;
   - the branch cannot take over the trunk: the only gradient reaching the detector is `L_consist`, through `y^bin` (G4);
   - `S̃` does not become a copy of the detector. Trained to rebuild normal steps, it stays an independent normality score.
   - *Cost:* the trunk's features are not pulled into a compact normal cluster, DSANet's representation side benefit. That is the price of the G4 protection.
2. **Dataset-level normalization, never per window.**
   - DSANet normalizes the reconstruction score to [0, 1]. Done per window, that forces every window, all-normal ones included, to contain a peak, which fights `L_MIL`'s normal term.
   - Here `S̃` is standardized against running statistics of normal-segment steps and clipped, so an all-normal window stays low.
   - This is also what lets SG-NM pull the high pre-anomaly baseline (0.55) down.
3. **Window-scale sizes.** DSANet uses K = 16 prototypes, 80 % of the frames as candidates and an 8-layer decoder, sized for long surveillance videos. T2 windows have 20 steps, and *derived from the T2 study:* ≈ 46 % of the steps in an abnormal window are positive (0.332 / 0.728). So:
   - **Normal segments:** all 20 steps are candidates. This is the MIL axiom: a window labelled normal has only normal steps.
   - **Abnormal segments:** the 8 lowest-scored steps (40 %). This is the largest set that stays mostly normal while the detector ranks above chance (A3 T2-test macro 0.695).
   - **K = 2 prototypes, 2 decoder layers.**

**Why the reconstruction error stays informative.**
- The first decoder layer has no residual, so its output depends on a step only through its attention weights over the 2 prototypes: one number per head, 8 in total.
- Everything after it is a per-step function of those numbers.
- Every reconstruction in a sequence therefore lies on an at-most-8-dimensional family inside a 512-dimensional space. The decoder cannot rebuild an arbitrary step, so steps far from the normal prototypes keep a high error.
- This makes DSANet's "no residual, no anomaly leakage" argument concrete.

**The known risk: contaminated prototypes.**
- An abnormal segment's bottom 8 steps can still include anomalous ones. With K = 2, one prototype could absorb them, and the anomaly would then be rebuilt well, giving it a **low** `S̃`.
- Normal segments and DVS fillers add clean candidates, but an unspliced abnormal window has only its own 8.
- The 40 % share is fixed here and **not swept** (no tuning on a Δ).
- The risk is checked three ways:
  - **E4a** (below) tests whether `S̃` ranks anything before the trunk is trained;
  - prototype usage is logged;
  - the macro AUC of `S̃` alone is reported.

**Training schedule (fixed now):**
- **The branch losses run from step 0.** They reach only the branch's own parameters.
- **`L_consist` starts at step 174**, the end of epoch 2 of 20. The branch and its running statistics must exist before the detector chases its target, the same reasoning as Pi-VAD's warm-up.
- **`λ_n` is fixed once, in a pilot on seed 2024.**
  - Run to step 300 with `λ_n` = 1.
  - Measure `ρ = ‖g_consist‖ / ‖g_task‖` on the detector parameters `L_consist` reaches, at steps 200, 250 and 300.
  - Set `λ_n = min(1, 0.2 / median ρ)`.
  - The same `λ_n` is used for all 5 seeds, and the pilot run is discarded.
- **ρ is logged every 50 steps.** A run with ρ > 0.3 at any logged step cannot be adopted (G4 put capture at ρ = 3.1).

**How it relates to CRN, `L_dvs` and `L_neg`:**
- **CRN** subtracts one median vector per clip at the input, with no parameters. **SG-NM** learns which within-sequence variations are normal (cruising versus braking, say) in the trunk's features, after CRN, and gives every step a target. They are complementary: K = 1 is not CRN, since the prototype is attention-weighted in `V^t` and the score is a decoder's reconstruction error.
- **`L_dvs`'s dense term** pushes every step of a synthetic abnormal anchor up. `L_consist` can pull down anchor steps that look normal. The interaction is bounded by the ρ ceiling, and the `L_dvs` trajectory is reported against A3's.
- **`L_neg`** contrasts a video's anomalous and normal parts through the class text, using about one step each (η = 0.02). **SG-NM** does it visually, over every step. It does not replace `L_neg`'s vision–text alignment; that remains DF3.

**Expected effect, stated before the run.**
- DSANet's gain was +0.94 AP on a different baseline. Our prototypes come from 8–20 steps per window segment.
- v2's same-input contrasts on DoTA-CAP-dev had paired half-widths of 0.012–0.029 (§2.1). The macro effect may well be at or below that.
- **A non-significant E4 is reported as "not detectable at this size", never as "no effect".**
- The mechanism measures (§7) are closer to what SG-NM changes, and are pre-registered with a predicted direction.

**What E4a tests, and why the gate comes first.** The branch is detached by design, so it can be trained on A3's frozen checkpoints exactly as it would be in E4, with no trunk training.
- If `S̃` then adds no ranking information to `y^bin`, `L_consist` has nothing to teach, and SG-NM is dropped before any trunk is trained.
- The gate's Δ is also the gain of a test-time ensemble. It is reported, but v2.1 does not deploy that ensemble: the module stays training-only, as in DSANet, and the deployed network stays A3's. The gate and E4 differ only in where the information ends up.

---

## 6. What v2.1 deliberately does not add

| Idea | Why it is not in v2.1 | Status |
|---|---|---|
| A post-hoc hold or EMA on `y^bin` | Run and killed (report direction H2(a)): −0.071 macro, −0.125 on tail clips | closed |
| MIL top-k as a fraction of the window | k already scales with the window (4 of 20), and the motivating slope is mostly position (report direction H2(b)) | closed |
| Feeding `p_T2` or clip position to the model | Benchmark fitting: needs the clip's end, fails in streaming, collapses off DoTA's usual cut (results §8) | closed |
| CLIP tiles or text-guided windows for non-ego accidents | Probe failed beyond `[x;u;p]`: text-guided windows −0.036 [−0.046, −0.026], tile mean-pool −0.012 (report direction H3) | closed; detector crops are v3 |
| More seeds or more T2-like data | The T2 learning curve is flat (+0.008 on DoTA over 4× the data) | closed |
| SG-NM deployed at inference (the E4a ensemble) | Changes the deployed network; it would be selected on the same dev set that suggested it | reported only |
| SG-NM with DSANet's sizes (K = 16, 80 % candidates, 8 layers) | Sized for long videos; with 20-step windows it would draw anomalous steps into the candidates | — |
| `L_neg` activation | A separate change with its own false-negative risk | DF3 |
| VideoMAE V2-B | An encoder change, not a training change; it would confound E4 | DF4 (report direction H4) |

---

## 7. Falsifiable predictions (SG-NM)

v2's predictions were resolved by E1–E3 and the Final (results §3–§4).

| Prediction | Measured in | Drop or qualify if… |
|---|---|---|
| `S̃` adds ranking information: the per-clip rank average of `y^bin` and `S̃` beats `y^bin` on DoTA-CAP-dev | E4a | the paired interval does not lie above 0 → SG-NM dropped before training |
| `S̃` is not a position proxy: the per-clip correlation of `S̃`'s AUC with `p_T2`'s stays near A3's 0.135 | E4a | it rises above 0.4 → any gain is read as position, and the `f_2` condition decides |
| DoTA-CAP-dev macro rises, and the position-controlled Δ is not negative | E4 | the auxiliary rule fails (§11.3) |
| **More of the span is covered:** A3's 61 % argmax-in-span share rises | E4 | it does not rise → no "coverage" claim, whatever the macro |
| **The pre-anomaly baseline falls:** A3's 0.55 mean before the span falls, and the 0.86 inside does not | E4 | the inside mean falls with it → the change is a global rescale, not a contrast |
| **No hold failure:** the macro on tail clips (≥ 3 s of normal after the span; 277 of 569) does not fall | E4 | it falls → the gain repeats the hold's failure mode, and the thesis says so |
| No new peaks in normal windows: the T2-val normal-window top-k score rises by ≤ 0.02 | E4 | guardrail fails |
| No capture: ρ ≤ 0.3 throughout; the corpus shortcut on `V^t` stays near A3's | E4 | not adoptable |
| Non-ego accidents: no prediction | E4 | reported only |

**A3-ign (E4-0), stated before the run.** Every row compares A3-ign with A3's E3 runs on DoTA-CAP-dev, seed-averaged, and is replicated on T2-val.

| Prediction | Measured in | Drop or qualify if… |
|---|---|---|
| **The pre-anomaly level falls:** A3's 0.55 mean before the span falls, and the 0.86 inside does not | E4-0 | the inside mean falls with it → a global rescale, not a contrast; no Gap S claim |
| **More of the span is covered:** A3's 61 % argmax-in-span share rises | E4-0 | it does not rise → the whole-anchor label is not what shapes the peak |
| **No hold failure:** the tail-clip macro (277 of 569) does not fall | E4-0 | it falls → the change repeats the hold's failure mode |
| **In-domain positive pressure survives:** T2-val macro no more than 0.01 below A3's | E4-0 | guardrail fails: `L_dvs-mil` and `L_MIL` alone are too weak a positive signal |
| DoTA-CAP-dev macro: **no directional prediction** | E4-0 | — (the base rule, §11.4, decides) |

---

## 8. Data and protocol

### 8.1 Sets and their status

| Set | Size | Status after v2 | Role in v2.1 |
|---|---|---|---|
| T2-train − T2-val | 1,053 sources | training | training, unchanged |
| T2-val | 219 sources / 645 windows | guardrail, pilot | guardrail, `λ_n` pilot, E4a in-domain read |
| T2-test | 1,106 windows | opened once at the v2 Final | second look only |
| **DoTA-CAP-dev** | 569 clips | the E3 decision set; the exploratory error analysis (results §6) ran on it | **the E4a and E4 decision set.** SG-NM's hypothesis came from it, so a dev decision is not an independent confirmation. |
| DoTA-CAP-eval | 560 clips | opened once at the v2 Final | second look only, labelled |
| DoTA-dev / -eval | 702 / 700 clips | endpoints for the CRN-only arms | not used: every v2.1 arm has the video stream |

**DoTA-CAP:**
- The DoTA clips whose pixels were recovered from CAP-DATA: 1,129 of 1,397 (80.8 %).
- It is representative of DoTA. CLIP-only probe 0.638 on dev, 0.636 on CAP and 0.649 on the dropped clips; A0 0.6628 / 0.6591 / 0.6787 on the same three sets (results §2).
- Its numbers are never placed beside full DoTA or LaGoVAD's 62.60.

### 8.2 Protocol (unchanged from v2)

- **DoTA:** stride 3 (0.30 s per step), the whole clip in one pass, scores interpolated to native frames, per-clip min-max for micro.
- **DADA (T2):** stride 8 (0.27 s per step), windows W = 20, hop 8.
- **Training negatives:** cut from inside the accident videos.
- **Fixed statistics, all from T2-train:** CRN references per source video; the scalar `s`; `σ_u`; the scalar `c`.

---

## 9. Training

**Loss, A3 (the E4 control):**

```
L = L_MIL + L_MIL-align + L_dvs           (LaGoVAD's losses minus L_neg, which is off; every weight 1)
L_dvs = L_dvs-sup + L_dvs-mil              (dense per-step BCE vs y^p  +  top-k MIL inside the y^p span)
```

What each term does in the code A3 was trained with:
- **`L_MIL`**: BCE on the mean of the top-k logits of `y^bin`, k = max(1, ⌊L/5⌋) per sequence (4 for a 20-step window).
- **`L_MIL-align`**: per class, the mean of the top-k similarities over time (k = max(1, ⌊L/16⌋), so 1 for a 20-step window), then cross-entropy against the class index (Normal = 0).
- **`L_dvs-sup`**: BCE per step against the pseudo label `y^p`, on every normal sequence and every DVS-spliced sequence. `dvs_anchor_mode = span`: in a spliced abnormal sequence **the whole anchor window is `y^p` = 1**, the fillers 0 (lesson C29).
- **`L_dvs-mil`**: top-k MIL with k = ⌊n_pos/4⌋ restricted to the `y^p` = 1 steps (anomaly) or over all steps (normal).
- **DVS sampling** (LaGoVAD defaults): an epoch is 2 × (number of abnormal windows) items, half abnormal anchors and half random normal anchors. With probability θ = 0.7 the anchor is used alone. Otherwise 1–4 normal filler windows (δ_m = 5 windows in total, anchor included) are spliced around it: 50 % drawn from the anchor's 10 nearest normals by central-frame CLIP feature, 50 % at random. Sequences are truncated at 512 steps.
- The ego-specific settings (`is_egocentric` = false) and the motion-aware neighbour key of the v1 spec are not used.

**Loss, E4 (A3 + SG-NM):**

```
L_total = L_MIL + L_MIL-align + L_dvs                         ← unchanged
        + λ_n · 1[step ≥ 174] · L_consist                     ← reaches the detector through y^bin
        + L_compact + L_rec                                   ← detached branch; reaches only its own parameters
λ_n = min(1, 0.2 / ρ_pilot), from one pilot on seed 2024; ceiling ρ ≤ 0.3 in every run
```

**Hyper-parameters (unchanged from v2):**
- AdamW, lr 5e-5, weight decay 0.01; linear warm-up over 20 steps, then cosine to 0; AMP on CUDA.
- Batch 64 drawn from a per-epoch permutation of the balanced item list, so it is ≈ 32 + 32 on average, not fixed; 87 steps per epoch, 20 epochs = 1,740 steps.
- Temporal encoder 2 layers × 4 heads with a local attention band of 25 steps; co-attention 2 layers × 8 heads; score kernel 3 (one Conv1d per `H_bin` path); MIL k = ⌊L/5⌋ (4 on a window); hidden 512; 32 soft prompts.
- The SG-NM branch uses the same optimizer and learning rate.

**Training data:** T2-train minus T2-val, identical for every arm.

**What trains:**
- **Frozen:** CLIP (both towers), VideoMAE V2-S.
- **Trained:** soft prompts, `W_u`, temporal encoder, co-attention, heads.
- **E4 only:** the SG-NM branch.
- `σ_u`, `c` and `s` are fixed statistics.

| Arm | CRN | Video stream | SG-NM | Input to the temporal encoder |
|---|:-:|:-:|:-:|---|
| A0 (reference) | – | – | – | `x_t` |
| **A3 (v2; E4-0 control; base if `B` = A3)** | ✓ | ✓ | – | `x̃_t + W_u·(c·ũ_t ⊘ σ_u)` |
| **A3-ign** (E4-0; base if `B` = A3-ign) | ✓ | ✓ | – | as A3; trained with `dvs_anchor_mode = ignore` |
| **`B` + SG-NM** (E4) | ✓ | ✓ | ✓ | as A3; `dvs_anchor_mode` as in `B` |

**Control integrity.**
- A3's E3 runs are the paired control for E4-0. If anything in the training pipeline that affects A3 has changed since E3, A3 is re-run on the same 5 seeds before E4-0 is read.
- A3-ign's `config.yaml` must differ from the E3 A3 run of the same seed in **exactly one line**, `loss.dvs_anchor_mode`. The run uses the same baked input cache, the same KNN cache and the same seeds (2024–2028). The notebook asserts the diff before training.
- E4's control is `B`'s own 5 runs. Every E4 run carries `B`'s `dvs_anchor_mode`.

---

## 10. Inference (unchanged from v2)

1. **Steps:** stride 8 on 30 fps sources, stride 3 on 10 fps.
2. **Encode:** CLIP on the frame at each step; VideoMAE V2-S on the causal 1.5 s clip ending there. The class texts are encoded per forward pass: LaGoVAD's verbalizer samples one definition per class (Normal, CarAccident) for each pass.
3. **CRN:** the per-dimension median over the clip (baked into the DoTA input cache at stride 3).
4. **Trunk:** the whole clip in one pass (up to 512 steps; DoTA clips are shorter) → `y^bin`, `y^mul`. The temporal encoder's attention is a local band of 25 steps, so "whole clip" means one pass, not a whole-clip receptive field.
5. **Score:** `σ(y^bin)` is interpolated linearly to native frames (the benchmark read then averages the 5 seeds per frame); macro AUC over two-class clips with a source-video cluster bootstrap; per-clip min-max for micro (benchmark) or a threshold (deployment).
6. **Optional:** ATS → MLLM report, off the critical path. Designed, not implemented in `core/`.

**SG-NM is not run at inference.**

**Cost:** CLIP ≈ 17.5 + VideoMAE V2-S ≈ 57 ≈ 75 GFLOPs per step, ≈ 250–280 GFLOPs/s: real-time on one GPU. The trunk and heads cost a negligible amount, and SG-NM adds nothing.

---

## 11. Evaluation and plan

### 11.1 Endpoints and statistics (fixed now, before any E4-0 or SG-NM run)

- **Seeds:** n = 5 (2024–2028), paired by seed with the comparison's control: A3's E3 runs for E4-0, and `B`'s runs for E4. No seeds are added after a result is seen.
- **Decision interval:** the paired t95 over the 5 seeds of the per-seed DoTA-CAP-dev macro Δ (t = 2.776). It is the only interval an adoption rule uses on raw macro.
- **Position control (required beside every Δ):**
  - `f_2 = z(ȳ) + 2·z(p_T2)`, the score v2's Final used.
  - Its Δ is computed on seed-averaged scores, with v2's clip-level interval: a cluster bootstrap over source videos, B = 10,000.
  - `p_T2` is printed beside every macro.
- **Primary endpoint:** DoTA-CAP-dev macro AUC (raw), with its `f_2` Δ.
- **Secondary, mechanism** (§7; seed-averaged, DoTA-CAP-dev, and replicated on T2-val):
  - argmax-in-span share;
  - mean score before and inside the span;
  - tail-clip macro;
  - Δ per accident-position bin (< 0.40, 0.40–0.55, 0.55–0.70, ≥ 0.70);
  - ego / non-ego Δ.
- **Diagnostics:**
  - the ρ trajectory;
  - the macro AUC of `S̃` alone;
  - the per-clip correlation of `S̃`'s AUC with `p_T2`'s;
  - prototype usage (> 95 % on one prototype means collapse);
  - the `L_dvs` trajectory;
  - the corpus-shortcut AUC on `V^t`;
  - the position probe on `V^t` (linear R² for `t / T`).
- **Guardrails** (T2-val, against the comparison's control: A3 for E4-0, `B` for E4):
  - T2-val macro no more than 0.01 below the control's;
  - the normal-window peak (mean top-k score on T2-val normal windows) rises by no more than 0.02;
  - O1′, v2's amended T2 guardrail: micro above the clip oracle is not read as collapse when window AUC and macro rise with it.
- **Also reported:** DoTA-CAP micro and AP, T2 micro, and efficiency.

### 11.2 Plan

| Step | What | Trains? | Decision |
|---|---|:-:|---|
| **E4-0** | A3-ign × 5 seeds, 1,740 steps each, paired with A3's E3 runs (§11.4) | yes | the base rule (§11.4) fixes `B` ∈ {A3, A3-ign} |
| **E4a** | SG-NM gate on `B`'s five checkpoints (below) | branch only | E4 runs only if the gate passes |
| **Pilot** | `B` + SG-NM, seed 2024, `λ_n` = 1, to step 300 | yes (discarded) | fixes `λ_n` (always on `B`; never reused across anchor modes) |
| **E4** | `B` + SG-NM × 5 seeds, 1,740 steps each | yes | the auxiliary rule (§11.3) |
| **Second look** | `B` + SG-NM on DoTA-CAP-eval and T2-test, read once if E4 adopts it; A3-ign likewise if the base rule adopts it | no | reported, labelled "second look at an opened set" |

Wherever §5.3 and the architecture document say "A3" for the E4a checkpoints or the E4 control, read **`B`**.

**E4a, the gate** (no trunk training):
1. For each of `B`'s five checkpoints (A3: the E3 runs; A3-ign: the E4-0 runs), compute the frozen `V^t` and `y^bin` on the same training stream (T2-train minus T2-val, DVS included, with `B`'s anchor mode).
2. Train only the branch (`L_compact + L_rec`) on the E4 schedule: 1,740 steps.
3. Score whole DoTA-CAP-dev clips (protocol B):
   - candidates chosen without labels: the 40 % of each clip's steps with the lowest `y^bin`;
   - `S̃` uses the running statistics frozen at the end of branch training.
4. Per seed, Δ = the macro AUC of the per-clip rank average of `y^bin` and `S̃`, minus the macro AUC of `y^bin` alone. The same Δ is computed on `f_2`.
5. **Gate:** the paired t95 of Δ over the 5 seeds lies **above 0**, and the seed-averaged `f_2` Δ is **≥ 0**. Otherwise SG-NM is dropped and v2.1 is v2.
6. **Reported:**
   - the macro AUC of `S̃` alone;
   - its per-clip correlation with `p_T2` and with `y^bin`;
   - the same Δ on T2-val;
   - `B`'s span profile on T2-val (argmax-in-span share, mean score before and inside the span), an in-domain replicate of Gap S.

**Budget:** E4-0 is 5 runs of 1,740 steps. E4a trains a 5.8 M-parameter branch on frozen features, 5 times. E4 is 1 pilot plus 5 runs of 1,740 steps. In total, 10 full runs plus a discarded pilot: half of E3's 20 runs.

**Order:** E4-0 → base rule → E4a → pilot → E4 → second look. E4a must not start before `B` is fixed, because its checkpoints and candidate ranking depend on `B`. DF3 (`L_neg`), if pursued, comes after E4 as its own step. The report's position-shift test (DF8) can run at any time, since it trains nothing. If it exists before E4 is read, E4 is also reported on it, descriptively.

### 11.3 Adoption rule and claims (fixed now)

**Adopt `B` + SG-NM** if all of the following hold:
1. the paired t95 of the (`B` + SG-NM) − `B` DoTA-CAP-dev macro Δ lies **above 0**;
2. the `f_2` Δ point estimate is **≥ 0**;
3. ρ ≤ 0.3 at every logged step of every run;
4. the guardrails hold (§11.1), against `B`.

Otherwise the final model is `B`. SG-NM is training-only, and A3-ign differs from A3 only in a training label. So the deployed network has A3's architecture in every outcome; the rules change only how it is trained.

**What the thesis may claim:**
- **"SG-NM improves DoTA-CAP detection"** requires rule 1. The eval number is labelled a second look.
- **"… beyond the position prior"** requires the `f_2` Δ **interval** to exclude 0, not only its point estimate.
- **"SG-NM makes the score cover the anomalous span"** requires both of:
  - the argmax-in-span share to rise;
  - the tail-clip macro not to fall.
  Otherwise the thesis says what the mechanism measures show, without the coverage claim.
- **A non-significant E4** is "not detectable at this size", not "no effect". The E4a result is reported either way: a passed gate with a null E4 means the information exists, but the loss did not transfer it.

### 11.4 E4-0: A3-ign and the base rule (fixed now, before any A3-ign number)

**Why this step exists.**
- A3 was trained with `dvs_anchor_mode = span`, LaGoVAD's default, which v2 never chose on purpose (§2.2).
- On a spliced abnormal sequence this labels **every** step of the anchor window `y^p` = 1, although ≈ 54 % of an abnormal T2 window's steps are normal (C29). It is a candidate cause of Gap S that needs no new module.
- It also interacts with SG-NM in three ways:
  - `L_dvs-sup` pushes the anchor's normal steps up while `L_consist` pulls them down;
  - a detector trained to raise the whole anchor ranks the anchor's steps less informatively, so the bottom-8 candidates (§5.3) are more likely to include anomalous steps;
  - an SG-NM gain over a `span` control could be the partial undoing of C29 rather than normality modelling.
- Adding `ignore` to the SG-NM arm alone would change two factors at once. So the anchor mode is decided **first**, on its own, and SG-NM is then compared against a control with the same anchor mode.

**What A3-ign changes, and what it does not.**
- **Only** spliced abnormal sequences are affected: about 30 % of abnormal anchors (θ = 0.7), i.e. ≈ 15 % of batch rows.
- On those, the anchor's steps leave the dense `L_dvs-sup` BCE; the fillers stay hard negatives (0).
- Positive pressure on the anchor still comes from `L_dvs-mil` (top-k inside the anchor, k = ⌊n_pos/4⌋) and from `L_MIL`.
- Unspliced windows and every normal sequence are trained exactly as in A3.
- No parameter, no code change: the flag exists, with tests, since 2026-09-09.

**Prior evidence, and why it does not decide.**
- `ignore` was measured twice, both times with v1 code, KIP off, n = 1 seed, and in regimes where the model could not localize:
  - DADA-2000 archive Phase 1: macro 0.5190 (`span`) vs 0.5134 (`ignore`); the score head spanned the whole clip (C27/C28).
  - TAD T-ladder: macro 0.6174 vs 0.6209 (+0.0035, below the metric's resolution); the model had collapsed into a clip classifier (C14).
- T2 is the first corpus here where the model localizes (kernel 3, macro ≥ micro), so the anchor label can matter for the first time. **No directional prediction is made on macro.**

**The run.** A3-ign × 5 seeds (2024–2028), 1,740 steps each, config-diffed against the E3 A3 run of the same seed (exactly one line differs; §9). Read with the same harness, endpoints and diagnostics as E4 (§11.1), plus the `L_dvs-sup` / `L_dvs-mil` trajectories against A3's.

**Base rule.** `B` = **A3-ign** if all of the following hold; otherwise `B` = **A3** (`span`):
1. the paired t95 of the (A3-ign − A3) DoTA-CAP-dev macro Δ lies **above 0**;
2. the `f_2` Δ point estimate is **≥ 0**;
3. the guardrails (§11.1) hold against A3.

- No tie clause, margin or tie-break is added after the number is seen (lesson 14).
- If `B` = A3-ign, A3-ign is also the new best adoptable free arm `F`, and every later costly arm must beat it.
- If `B` = A3, E4 runs on `span`, and the thesis lists the C29 interaction as a limitation of any SG-NM result.

**What the thesis may claim from E4-0:**
- **"Dropping the whole-anchor DVS label improves DoTA-CAP detection"** requires rule 1. The eval number is labelled a second look.
- **"… beyond the position prior"** requires the `f_2` Δ **interval** to exclude 0.
- **"The whole-anchor label causes (part of) Gap S"** requires all three §7 mechanism rows for A3-ign to come out as predicted: the pre-anomaly mean falls while the inside mean does not, the argmax-in-span share rises, and the tail-clip macro does not fall.
- **A non-significant E4-0** is "not detectable at this size". It is not evidence that C29 is harmless on T2.

---

## 12. Justification ledger

| Decision | Rationale | Source |
|---|---|---|
| Keep LaGoVAD | Only venue-eligible WS baseline with traffic evaluation; trains correctly on T2 | LaGoVAD; T2 study §3.1 |
| KIP and RAFT removed (v2) | Harmful, then neutral; K > W; untrained gate; 94 GiB | T2 study §4 |
| Video stream: VideoMAE V2-S, frozen, causal 1.5 s, zero-init residual (v2) | Video encoders win TAD; +0.098 over CRN on eval; frozen avoids learning DADA; V2-S by the tie rule | SimpleTAD; results §3–§4 |
| Called a video stream, not a motion stream | Frame order changes the result by ±0.0004 | D6 |
| CRN with R2 (v2) | Removes the corpus shortcut; makes the gain survive position control; no stand-alone claim | T2 study §2; results §4.3, §7 |
| Fixed `σ_u` and `c`, no LayerNorm; video share `ρ_u` logged | A per-token norm erases the deviation size; zero-init fixes only the start | v2 review rounds 2–3 |
| Protocol B | +0.033 on the same checkpoints | E1 |
| **SG-NM, training only** | Gap S is measured; a content-based dense target is what the failed hold lacked; the deployed network is unchanged | results §6.3, H2(a); DSANet |
| **On `sg(V^t)`, before the co-attention** | DSANet's placement; `V^u` is text-conditioned and re-normalized; `V^t` keeps deviation size | DSANet; architecture §5 |
| **Detached branch, one-way consistency** | No feature collapse; no capture of the trunk (G4); no circular copying of the detector | SG-KN review; T2 study §4.2 |
| **Dataset-level normalization of `S̃`** | Per-window min-max forces a peak in all-normal windows | SG-KN review |
| **Candidates per window segment; all of normal segments, bottom 8 of abnormal; K = 2; 2 decoder layers** | MIL axiom; ≈ 46 % positives in abnormal windows; DVS fillers are labelled normal; a ≤ 8-dimensional reconstruction family keeps the error informative | derived from T2 study §2.2; DSANet |
| **`L_consist` from epoch 2; `λ_n` from a ρ pilot; ρ ≤ 0.3** | The target must exist before the detector chases it; the weight is derived, not swept | Pi-VAD (warm-up); T2 study §4.2 |
| **E4a gate on frozen checkpoints** | A detached branch can be tested without training the trunk | — |
| **E4-0 (A3-ign) before SG-NM; base `B` by a pre-registered rule** | The whole-anchor label is a free, competing cause of Gap S and conflicts with `L_consist`; deciding it first keeps E4 a one-factor contrast | C29; §2.2; §11.4 |
| **Raw and position-controlled conditions; second-look label** | Position is the central confound; DoTA-CAP-dev suggested SG-NM and DoTA-CAP-eval is opened | results §4.3, §8 |
| `L_neg` stays off | Needs class masking; a separate change | DF3 |
| No context token | It would bring the source signal back into `y^bin` | v2 review round 1 |
| ATS, off-path (not implemented) | Zero detection latency | Holmes-VAU |

---

## 13. Deferred (not in v2.1; each needs its own experiment)

- **DF1 — An order-sensitive signal** (report direction H5). Only if the thesis wants "motion" or "kinematics" back.
  - **Candidates:** difference features `u_t − u_{t−k}`; VideoMAE on reversed clips, or on one frame repeated 16 times as a motion-energy control; or the v1 RAFT statistics, clip-referenced, as an auxiliary target on the video term `W_u(c·ũ⊘σ_u)` only, at ρ ≈ 0.2.
  - **Gate:** a probe first, with D6's shuffle control. If the shuffled control still contains 0, the motion claim is dropped for good.
- **DF2 — Overlap-negative MIL.** Report with and without, since it is equivalent to span labels at hop resolution.
- **DF3 — Activate `L_neg` with class names.**
  - **Captions:** an abnormal window's caption is the definition row of its class, `z[c]`, from the same text tower and soft prompts. Normal windows have no caption.
  - **Class mask:**
    - in text → video, drop other same-class foregrounds from the denominator; backgrounds, including the window's own, stay negatives;
    - in video → text, score each foreground against the **unique** class texts in the batch.
    - Without the mask, ≥ 12 same-class pairs per batch become false negatives (83 classes, the top one 12.2 %).
  - **Class source.** The code's class list for T2 is only {Normal, CarAccident}: every DADA window maps to `CarAccident`, sharing DoTA's traffic definition set. The 83 classes above are DADA's `texts` strings, which no code path reads (gap G4). DF3 therefore needs a new class source first.
  - **Checks before trusting it:**
    - a test that same-class foregrounds receive no repelling gradient;
    - the masked-pair count per batch;
    - **background purity** on T2-val (the share of background picks that are truly normal; at η = 0.02 the background is about one step);
    - ρ ≤ 0.3.
  - **Run:** its own step after E4, 5 seeds.
- **DF4 — VideoMAE V2-B** (report direction H4).
  - Evidence: ahead beyond position in transfer, +0.037 vs +0.018.
  - Run: arm A3-B, 5 seeds. V2-B must first be extracted for full T2.
  - Kill: A3-B − A3-S on `f_2` contains 0 → keep S.
- **DF5 — A clip-context token** for concept drift. Only with a stop-gradient in the `H_bin` path, or confined to `H_mul`, with the corpus-shortcut AUC re-measured on `y^bin`.
- **DF6 — DAPT of the video encoder** on BDD100K or DADA-train frames (SimpleTAD). Never on DoTA.
- **DF7 — Early-onset anomalies** (report direction H6). A3 scores 0.675 on the 54 clips whose anomaly starts in the first 20 %. Compare R2 with the past-only reference on this group: a probe on the cached features.
- **DF8 — Position-shift stress test** (report direction H1). A protocol, not a model change.
  - Re-cut the DoTA-CAP clips so the accident centre is spread evenly over [0.1, 0.9].
  - Add a same-length control crop at the original position, and keep `p_T2` frozen at its T2 fit.
  - No training: CRN is recomputed per crop, and the checkpoints are re-run on cached features.
  - It decides how the thesis tells the position story. If `p_T2` is still ≥ A3 on the shifted set, the "position is an artefact" argument is wrong.

---

## 14. Review history (condensed; the full tables are in the v2 proposal)

| Round | Main points | Outcome |
|---|---|---|
| 1 (earlier v2 draft) | `L_neg` bundled into a ladder rung; a context token leaking the source into `y^bin`; the CRN mean skewed by long accidents; SG-KN's bugs (per-window min-max, no stop-gradient, "K = 1 ≈ CRN", the "kinematic" name); too many components | `L_neg` and the token deferred; the CRN reference chosen by a reversal check; SG-KN removed, now corrected as v2.1's SG-NM (§5.3); v2 cut to two input changes |
| 2 | A per-token LayerNorm erases CRN's magnitude; the past-only reference's position artefact; A3 must beat A1; one decision interval; squash geometry | Fixed `σ_u`; position-residualized choice rule with a position ruler; costly arms must beat the free arm; paired t95 only |
| 3 (advisor) | The position fit must use normal steps only; zero-init under AdamW fixes only the start; squash over letterbox (DADA 1584 × 660); the adoption rule; the MDE degrees of freedom; magnitude after the trunk's LayerNorm; the artefact is not specific to the past-only reference | All accepted except the magnitude point, which the encoder's outer residual answers; scalar `c` and the share `ρ_u` added |
| 4 (advisor sign-off) | Where the magnitude survives (one `H_bin` path, ≈ 0.44×); costly arms vs the best adoptable free arm `F`; per-clip normalization in the stratified check; run order | Adopted; v2 ran E0–E3 and the Final on those rules |

---

## 15. Answers to the T2 study's advisor questions, updated with v2's results

**(a) Option 1 (a new flow target) or option 2 (a video representation)?**
Option 2, and it worked: +0.098 over CRN on DoTA-CAP-eval. But it worked as a representation, not as motion: D6 shows frame order does not matter. Option 1 would make sense only as DF1, behind a shuffle-controlled probe.

**(b) Is dropping KIP compatible with "LaGoVAD + a kinematics pathway"?**
Dropping KIP was right: v2 beats the KIP-off trunk by +0.090. The "kinematics pathway" wording is not compatible with D6. The defensible description is "LaGoVAD with a second, video-trained encoder and clip-referenced normalization", plus SG-NM's training-time normality modelling if E4 passes.

**(c) Is the negative result publishable?**
As a thesis chapter, yes. Together with v2, the chapter is stronger:
1. Induction cannot recover information absent from the input (KIP).
2. A video encoder's gain over frame CLIP can be appearance, not motion, and a frame-shuffle control is cheap enough to always run (D6).
3. On DoTA, a pixel-free position prior beats every model. Gains must be reported beside it, and the benchmark's clip-cutting convention deserves its own stress test (DF8).
