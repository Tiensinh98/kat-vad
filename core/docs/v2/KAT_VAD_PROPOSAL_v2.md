# Proposed Road-Traffic Video Anomaly Detection Architecture — **KAT-VAD v2**
### LaGoVAD baseline + **Motion Stream** + **Clip-Referenced Normalization**

**Supersedes:** `KAT_VAD_PROPOSAL.md` (v1, Kinematic Induction Pathway) and the earlier, larger v2 draft.
**Evidence base:** the project papers, plus `REPORT_T2_OPTICAL_FLOW_FOR_ADVISOR.md` (2026-09-26; DADA-2000 T2, W = 20).
- Numbers quoted as "report §x" were measured in that study.
- Numbers marked *derived* are computed from report numbers, with the arithmetic shown.
**Companion:** `KAT_VAD_v2_ARCHITECTURE.md` (every component with its tensor shapes).
**Date:** 2026-09-27.

> **Design rule for this version.** A component is in v2 only if (i) it fixes a failure the T2 study **measured**, (ii) a project paper supports the mechanism, and (iii) it can be checked cheaply **before** it is trained.
> - That leaves **two architectural changes**, one **protocol fix**, and **no new loss terms**.
> - Everything else from the earlier draft is either removed or listed as deferred, with the reason (§4.3, §12).
> - No change can be proven to help before training. What the design guarantees is that each change targets a measured failure, is checked first, and is kept only if a paired comparison supports it.
> - The floor is the KIP-off trunk: T2 macro 0.6248, DoTA macro 0.6113.

---

## What changed (v1 → v2)

| # | v1 | v2 | Measured failure it fixes | Support | Check before training |
|---|---|---|---|---|---|
| **1** | KIP: flow head, integer-cast gate, fixed 50 % shift, motion head, `L_KIP-rec/align`, `L_kin`, RAFT | **Removed** | Flow v1 cost 0.013 micro and flow v2 was neutral. The flow head is worse than a clip-identity predictor (K > W). Direction R² is 0.071. The gate receives no gradient (report §4). | Pi-VAD: PMG filters a motion-bearing I3D input and cannot create motion that frame CLIP lacks. | none: removing KIP returns to the KIP-off floor |
| **2** | Motion "induced" from per-frame CLIP | **Motion Stream:** a frozen VideoMAE V2 encoder on causal 1.5 s clips, added to the CLIP input through a **zero-initialized residual** | Frame CLIP carries no in-clip motion: lag-1 cosine 0.968, K 0.723 > W 0.654, direction R² 0.071 (report §4.3). | SimpleTAD: video (MVM) encoders are the representation that wins on DoTA/DADA. | E2: frozen-feature probe (the report's own pre-registered rule) |
| **3** | Raw features go into the trunk | **Clip-Referenced Normalization (CRN):** subtract a robust per-clip reference from each feature, i.e. remove the clip-level fixed effect | 73.5 % of CLIP variance is constant within a clip. The source is separable at AUC ≈ 1.000. DoTA is flat over 4× data (report §2.2, §2.4, §3.3). | Fixed-effects logic; DSANet SG-NM (normality taken from the video itself) | E2: accident-share histograms, reference choice without reversal, transfer-probe veto |
| **P** | DoTA read at 0.8 s/step vs 0.27 s/step in training *(derived)* | **Rate-matched evaluation:** DoTA at stride 3, sliding W = 20, scored at native frames | A 3× time-scale gap. Step-counted components flip sign with clip length (report §4.5). | SimpleTAD: windows matched in seconds | E1: re-score the existing checkpoints (**no training**) |
| **S** | n = 3; DoTA-macro t95 half-width 0.045 | n = 5, paired; **one** pre-registered decision interval (paired t95 over seeds); a 2 × 2 factorial | *Derived:* v1 could not detect effects below 0.045 on DoTA | — | E0b: the minimum detectable effect **on DoTA-dev**, from existing checkpoints (**no training**) |

**Losses are unchanged.** v2 trains with exactly the KIP-off configuration's loss set. `L_neg` stays **off**, as it is in that config (`captions_from_definitions: false`). The report showed that an auxiliary loss can capture the shared trunk (ρ = 3.1, §4.2), so v2 adds none.

---

## Review response (earlier v2 draft → this version)

| Review point | Verdict | Resolution |
|---|---|---|
| **S2 would switch on `L_neg` rather than "add rows".** It is off in the KIP-off T2 config. Its texts are DADA category labels (83 strings; the top one is 12.2 %), so duplicate classes in a batch become false negatives. The E4 rung therefore bundled three changes. | Correct | **Removed from v2.** Enabling `L_neg` with class-masked negatives (and same-source rows) is deferred as its own experiment (hook H3). ONM (overlap-negative MIL, the proposed E4a rung) is also deferred (hook H2). Keeping v2 to two architectural changes keeps attribution clean. |
| **[CTX] leaks the source signal back into `y^bin`.** `H_bin` blends the language-guided path, which saw [CTX]. | Correct | **[CTX] removed.** Both detection paths now see only CRN-normalized features. The source-shortcut AUC is also measured on the trunk output `V^t` and on `y^bin` (§10), not only on the input features. |
| **CRN's mean is skewed by the anomaly itself.** With a 70–80 % accident share, the reference sits in the accident and the ranking reverses. 33.2 % vs 33.1 % is only an average. "Ego-motion compensation" overclaims. | Correct | **The CRN reference is now chosen by a pre-registered check** (§4.2, E2). Per-clip accident-share histograms come first. Four references are compared, including robust and past-only ones, and any reference that reverses the ranking in the > 50 % bin is rejected. DoTA results are reported **per share bin**. The wording is now "clip-level fixed effect"; there is no ego-motion claim. |
| **SG-KN has two bugs.** Per-window min-max forces a peak even in all-normal windows. The target has no stop-gradient, so the trunk can collapse. "K = 1 ≈ CRN" is false. The effect size is below the minimum detectable effect, and "kinematic" has no basis. | Correct on all counts | **SG-KN removed.** A corrected variant (no per-window min-max, stop-gradient target, ablation-only) is recorded as hook H4. It is not in the model or the ladder. |
| **Too many components.** | Agreed | v2 is now: remove KIP, add a motion stream, add CRN, fix the evaluation protocol. The only trainable addition is one zero-initialized projection: ≈ 0.4 M parameters (ViT-B) or ≈ 0.2 M (ViT-S). |

### Review response, round 2 (2026-09-27)

| Review point | Verdict | Resolution |
|---|---|---|
| **A per-token LayerNorm on `ũ_t` erases CRN's deviation magnitude.** LN rescales every step to the same norm, so after CRN a step that barely deviates and one that deviates strongly look alike; only the direction survives. | Correct | **LayerNorm removed.** The motion stream is divided by a **fixed per-channel scale `σ_u`**, frozen on T2-train (§4.1). It is dataset-level, not per-token, so `‖ũ_t‖` keeps its meaning. |
| **R4 (past-only) carries a position shortcut.** At t = 0 the deviation is exactly 0, and its variance grows with t. Accidents sit late in the clip, so the deviation norm scores well from position alone, and the choice rule would reward that. | Correct | R4 now uses **strictly past** steps with an `N_w`-step warm-up. The choice rule scores the deviation **after regressing out position** and prints a **position-only ruler** beside every reference (§4.2). §10 adds a position probe on `V^t` for every arm. |
| **"A3 if A3 − A0 excludes 0" does not show that the motion stream contributes.** If A3 ≈ A1, a 180-GFLOP encoder would ship with no evidence for it. | Correct | A3 now needs **both** A3 − A0 **and** A3 − A1 to exclude 0. Any motion-stream claim in the thesis needs the matching contrast to exclude 0. The interaction is descriptive only (§10.3). |
| **Two intervals and no rule for which decides; the 0.022 detectable effect is a full-DoTA number, while decisions use DoTA-dev.** | Correct | **One decision interval:** paired t95 over seeds. The clip bootstrap is reported and never decides, except in E1 (n = 3). ~~New step E0b measures the detectable effect on DoTA-dev, with a pre-registered response if it is too large.~~ *Revised in round 3:* DoTA-dev is fixed at 50 %, and E0b is descriptive (§10.1). |
| **Squashing the full frame to 224² distorts motion differently per corpus.** DADA is ≈ 2.40:1 and DoTA 16:9, so horizontal motion is compressed ≈ 2.4× vs ≈ 1.78× — a ≈ 1.35× motion-geometry gap inside the motion stream. | Partly (anisotropy is real; the scale argument was incomplete) | ~~E2 probes squash and letterbox; letterbox wins ties.~~ *Reversed in round 3:* squash only, no geometry probe (§4.1). |

### Review response, round 3 (advisor, 2026-09-27)

| Review point | Verdict | Resolution |
|---|---|---|
| **1. The position fit removes real signal.** `f(t/T)` was fitted on all T2-val steps, accidents included, so it partly learns "accident steps deviate more" and subtracts it. | Correct | `f` is fitted on **normal steps only** (drift under normal driving), held constant outside the position range those steps cover, and cross-checked by a **position-stratified AUC** (§4.2). |
| **2. Zero-init does not keep the model near the baseline under AdamW.** Early Adam updates are ≈ lr·sign(g), so the motion term's growth rate is set by its input scale, which `σ_u` fixed at 1. | Correct mechanism; the size needed our numbers | Our CLIP cache is **not** unit-norm: mean L2 norm **9.87** on T2 (`outputs/EDA/DADA2000_orig_T2_w20s8/eda_report.md`), i.e. ≈ **0.44** per channel, not 0.044. At the bound the motion term reaches the CLIP channel size in ≈ 15 steps with scale 1, and ≈ 30 steps with the proposed scalar. The scalar `c` is adopted for scale consistency, but it does not change the conclusion, so the claim is corrected instead: zero-init guarantees the **start**, not proximity. The motion share `ρ_u` is logged as the evidence (§4.1). |
| **3. Letterbox does not remove the cross-corpus gap; squash should win.** | Conclusion correct; one number differs | Measured DADA frames are **1584 × 660 (2.40 : 1)** (`core/docs/DADA_ORIGIN_PHASE0.md` §3.3, 5 clips × 8 frames); the paper's 1456 × 660 does not match our files. Under squash the scale gap is 1.24× horizontal and 1.09× vertical; letterbox makes it 1.24× both ways and leaves DADA ≈ 93 px tall. **Squash only; the geometry probe is dropped** (it halves E2's extraction). The pre-training-matched variant (short-side resize + 3 crops) becomes hook H8. |
| **4. The adoption rule can ship a costlier, worse model** (A3 ≈ A1 = +0.04, A2 = +0.03 significant vs A0 → ships A2). | Correct | A2 now needs A2 − A0 **and A2 − A1** to exclude 0: the costly arm must beat the best free arm (§10.3). |
| **5. E0b's MDE rests on 2 degrees of freedom.** | Correct | **DoTA-dev fixed at 50 % now**; the conditional rule is gone. E0b is kept as a descriptive MDE, pooled over all existing seed-matched contrasts (≤ 8 df; they share the KIP-off seeds) (§10.1). |
| **6. Deviation size survives only up to the trunk's first LayerNorm.** | **Not for this code** | The layers are post-LN (`core/models/temporal_encoder.py:118`, `:161`), but `TemporalEncoder` wraps the whole stack in an outer residual, `V^t = H + Enc(H)` (`temporal_encoder.py:287`). `H`, with its magnitude, reaches `V^t` and the heads un-normalized; only the `Enc(H)` branch is re-normalized. The wording now says exactly that (architecture §5). |
| **7. The position artefact is not specific to R4.** | Correct | With scene drift, R1–R3 give a U-shaped deviation that is also high late in the clip. §4.2 now frames the artefact for all four references; the controls already applied to all four. |

### Review response, round 4 (advisor sign-off, 2026-09-27)

The review accepted round 3, including the pushback on point 6, and confirmed the numbers (CLIP ≈ 0.44 per channel; parity in ≈ 14 steps at scale 1 and ≈ 32 with `c`; 1.24× / 1.09× under squash). Remaining items:

| Review point | Verdict | Resolution |
|---|---|---|
| **On (6): how much magnitude survives, and where.** `Enc(H)` ends in a LayerNorm, so its per-token norm is ≈ √512 ≈ 22.6 at gain 1, against ‖H‖ ≈ 9.87: the un-normalized part is ≈ 0.44× the other. And CoAttn may re-normalize `V^u`. | Correct; **verified in code** | CoAttn is post-LN with no outer skip (`core/models/fusion.py:56–58`; the wrapper at `:84–86` just stacks layers), so `V^u` is re-normalized. `H_bin` reads `V^t` on its language-agnostic path (`core/models/kat_vad.py:154`). Magnitude therefore reaches **one** of the two blended `H_bin` paths, at ≈ 0.44× the normalized branch at init. Architecture §5 now says so. |
| **The A2 rule can reject a good arm.** If A1 fails its own rule, "A2 must beat A1" can send the ladder to A0 even when A2 significantly beats A0. | Correct | Costly arms (A2, A3) must beat the **best adoptable free arm** `F`: A1 if A1 passes its rule, A0 otherwise (§10.3). |
| **The stratified AUC pools across clips**, so clip-level drift scale leaks back in, and it measures something different from the per-clip `r_t` it cross-checks. | Correct | `d_t` is **min-max normalized per clip** before pooling, the same normalization as the DoTA micro protocol (§4.2). |
| **"≈ 8 df" is optimistic**: all four contrasts share the KIP-off seeds. | Correct | "≤ 8 df", with the reason (§10.1). E0b only reports. |
| **`c` missing** from the §8 arms table, the §3.1 "What is new" line and both diagrams. | Correct | Added everywhere. |
| **The round-3 table's "(advisor)" label was queried.** | Confirmed by the author | The reviewer is the advisor; the label stands. |
| **H8 cuts one way**: it can improve a working stream, never rescue a failed one. | Correct | Listed as a limitation (§12, H8). |
| **Run order for E0–E2.** | Adopted | New §10.2 run order: freeze and commit splits and rules first; DoTA-eval never loaded; E0 first; E0b + E1 on one harness; VideoMAE extraction in parallel; E2(a) before (b). |

---

## 0. Design summary

**Baseline.** KAT-VAD v2 keeps **LaGoVAD (ICLR 2026)**, the only venue-eligible WS baseline evaluated on real traffic data. The T2 study showed that it trains correctly: T2 macro 0.6248 beats the frozen-CLIP probe (0.5983), and micro stays below the clip oracle (0.7037).

**Two measured failures limit it on DoTA:**
1. Per-frame CLIP carries **no in-clip motion**. The flow head could not beat a clip-identity predictor.
2. About **three quarters of the CLIP signal is clip identity**. The source is separable at AUC ≈ 1.000, and DoTA did not move over 4× more data.

**The two architectural changes, one per failure:**
1. A **Motion Stream**: a frozen VideoMAE V2 encoder reads a causal 1.5 s clip at each step. It enters the network as a zero-initialized residual on the CLIP input, so training starts exactly at the baseline.
2. **Clip-Referenced Normalization**: each feature has a robust per-clip reference subtracted. This removes the clip-level fixed effect (scene, camera, dataset signature) that the DoTA metrics ignore and that fails to transfer. The reference is chosen by a check that rejects any reference that reverses the ranking in long-accident clips.

**Everything else.** KIP and RAFT are removed. The LaGoVAD trunk, heads and losses are unchanged. At evaluation, DoTA is read at the training step rate (0.3 s instead of 0.8 s per step) with window-matched inference, if a no-training check on the existing checkpoints supports it. The LLM stays off the critical path (Holmes-VAU ATS).

**Name.** *Kinematics-aware* in KAT-VAD means the network receives motion-bearing video features. v2 does not model explicit trajectories or velocities.

---

## 1. Gap analysis

### 1.1 Papers v2 draws on (the full per-paper table is in v1 §1)

| Paper | Traffic gap | What v2 uses |
|---|---|---|
| **LaGoVAD** (ICLR 2026) | CLIP-frame trunk with no motion; DoTA is its weakest score (62.60) | **The baseline**, unchanged: temporal encoder, definition co-attention, two-path `H_bin`, `H_mul`, and its losses |
| **SimpleTAD** (CVPR 2025 per the project summary; the official repo says ICCVW 2025, so check before citing) | Fully supervised, so not a WS baseline | Video MVM encoders (VideoMAE family) are the winning TAD representation; windows matched in seconds (1.5 s); DoTA/DADA as benchmarks; MCC/AUCMCC |
| **Pi-VAD** (arXiv 2025) | No traffic data | The lesson: induction works from a motion-bearing input (I3D), so KIP's input could not support it. Its motion gain was +0.95 AUC. |
| **DSANet** (AAAI 2026) | Motion-blind | Precedent for referencing each video to *its own* normality (SG-NM). v2 uses the idea in its simplest form (CRN), not the module. |
| **Holmes-VAU** (CVPR 2025) | No traffic benchmark | ATS: score-guided MLLM reporting off the critical path (unchanged) |

### 1.2 Measured gaps from the T2 study

**G1 — The input carries no motion.**
- Consecutive sampled CLIP frames have cosine 0.968.
- On the equally weighted target, the flow head is worse than a clip-identity predictor: K 0.723 > W 0.654.
- Direction R² is 0.071 (report §4.3).
- → **Motion Stream.**

**G2 — Clip identity and source dominate.**
- The between/within-clip variance ratio is 2.77 on T2 and 3.19 on DoTA. *Derived:* 73.5 % / 76.1 % of feature variance is constant within a clip.
- Foreign corpora are separable from DADA at AUC 1.000, and DoTA at 0.9999.
- DoTA gains +0.008 over 4× data.
- Both DoTA metrics use per-clip min-max, and macro AUC is per-clip, so any clip-constant score component is discarded.
- → **CRN.**

**G3 — Temporal-scale mismatch** *(derived; confirmed or refuted in E0).*
- DADA is 30 fps and DoTA is 10 fps, and both are read at stride 8. One step is therefore 0.27 s in training and 0.80 s on DoTA.
- Step-counted components flip sign with clip length (report §4.5).
- → **Rate-matched evaluation.**

**G4 — Auxiliary losses can capture the trunk.**
- ρ = 3.1 at the end of stage 2 (11.6 in stage 1). The shared representation was rewritten, and every task loss rose 24–32 % (report §4.2).
- → **No new loss terms; the zero-init residual fusion.**

**G5 — The evaluation was underpowered.**
- *Derived:* the DoTA-macro paired seed SD is ≈ 0.018. The t95 half-width is 0.045 at n = 3 and ≈ 0.022 at n = 5.
- → **n = 5, factorial design.**

---

## 2. Baseline selection (compared, not assumed): unchanged, LaGoVAD

| Criterion | Pi-VAD (arXiv) | DSANet (AAAI 26) | RefineVAD (AAAI 26) | **LaGoVAD (ICLR 26)** |
|---|---|---|---|---|
| WS · venue · official code | ✅ · ❌ · ✅ | ✅ · ✅ · ✅ | ✅ · ✅ · ✅ | ✅ · ✅ · ✅ |
| Real traffic evaluation | ❌ | ❌ | ❌ | ✅ DoTA / TAD |
| Motion | induced from I3D | ❌ | gated shift (variance proxy) | ❌ → **v2 adds the Motion Stream** |
| Measured in this project | — | — | its trainable-gate idea was the worst KIP arm on MSAD (−0.068 vs the fixed shift) | trains correctly on T2 (0.6248 > 0.5983; micro < oracle) |

**Pick: LaGoVAD**, now backed by in-project evidence.

---

## 3. Architecture

### 3.1 Data flow

```
                  ┌─ frame @ step t ──────────────► CLIP ViT-B/16 image (frozen) ─► x_t ∈ ℝ^512 ─► CRN ─► x̃_t ────────────────┐
video @ rate r ───┤                                                                                                          (+)─► h_t ∈ ℝ^512
                  └─ causal clip 16f @ 10 fps ─────► VideoMAE V2 (frozen) ────────► u_t ∈ ℝ^768 ─► CRN ─► c·(⊘σ_u) ─► W_u (0-init)┘
                                                                                                                               │
                                        h ∈ ℝ^{L×512} ─► LaGoVAD temporal encoder (2 layers, RoPE) ─► V^t ∈ ℝ^{L×512}      │
definition Z ─► CLIP text (frozen) + soft prompts ─► z ∈ ℝ^{C×512} ──► co-attention CoAttn(V^t, z) ─► V^u, Z^u
                                        H_bin(V^t, V^u) ─► y^bin ∈ ℝ^L        H_mul(V^u, Z^u) ─► y^mul ∈ ℝ^{L×C}
off the critical path:  ATS(y^bin) ─► MLLM ─► incident report
```

**What is unchanged from KIP-off:** CLIP (both towers), soft prompts, the temporal encoder, the co-attention, `H_bin`, `H_mul`, the losses, and the optimizer settings.

**What is new:** CRN (parameter-free) and the Motion Stream (a frozen encoder, a fixed per-channel scale `σ_u` and a fixed scalar `c`, both frozen on T2-train, and one zero-initialized linear map).

### 3.2 Components

**(a) Frame stream: frozen CLIP ViT-B/16 (unchanged).** It is LaGoVAD's own encoder and shares its space with the definition text tower.

**(b) Motion Stream (new; §4.1).**
- A frozen VideoMAE V2 encoder on the causal 1.5 s clip ending at each step gives `u_t`.
- It is added to the CLIP input as `h_t = x̃_t + W_u · (c · ũ_t ⊘ σ_u)`, with **`W_u` initialized to zero**, `σ_u` a fixed per-channel scale and `c` one fixed scalar that puts the motion input on the CLIP input's per-channel scale.

**(c) CRN (new, parameter-free; §4.2).** It is applied to both streams before fusion.

**(d)–(g) Temporal encoder, co-attention, `H_bin` (two paths plus a learned blend), `H_mul`:** LaGoVAD, unchanged.

**(h) ATS reasoning layer:** unchanged and off the critical path.

---

## 4. The two components

### 4.0 Why KIP had to go, and why the fix is an input rather than a loss

**Pi-VAD induced flow from a motion-bearing input.**
- Pi-VAD's PMG works on **I3D snippet features**: 3D-convolutional, 16 frames, Kinetics-trained, so motion is already in the input.
- It *filters* those features toward each modality.

**KIP had no motion to filter.**
- KIP applied the same idea to per-frame CLIP, where the motion is absent (G1).
- A better flow target cannot fix this. A spatial or residual target is harder, not easier, to regress from a CLS vector.
- Even with a good input, Pi-VAD's motion gain was +0.95 AUC. That is a small effect to buy with 94 GiB of RAFT.

**The motion must enter as an input.** SimpleTAD shows the representation is where the large differences are: encoder-only video MVM models beat specialized fusion architectures on DoTA/DADA. Its numbers are fine-tuned and fully supervised, so they are evidence about representations, not a target for WS zero-shot.

### 4.1 Motion Stream

```
for each step t (time τ_t):
  C_t = 16 frames at 10 fps ending at τ_t        DADA: every 3rd native frame · DoTA: native frames
        full frame resized to 224 × 224 (squash, the CLIP stream's transform); no centre crop
  u_t = MeanPool_tokens( VideoMAE-V2(C_t) )       ∈ ℝ^{d_v}, d_v = 768 (ViT-B) or 384 (ViT-S); frozen
  ũ_t = u_t − μ_t^ref   (A3, with CRN)            or   u_t − m_u   (A2, no CRN; m_u = T2-train channel mean)
  h_t = x̃_t + W_u · (c · ũ_t ⊘ σ_u)               σ_u ∈ ℝ^{d_v}: per-channel std of ũ over T2-train steps, frozen
                                                  c: one scalar = the CLIP input's per-channel RMS on T2-train
                                                     (E‖x‖ / √512 ≈ 9.87 / 22.6 ≈ 0.44; the same for A2 and A3, since s preserves E‖x‖)
                                                  W_u ∈ ℝ^{512×d_v} (+ bias), initialised to 0
```

**Why VideoMAE V2 and why frozen.**
- The distilled VideoMAE V2 checkpoints (`vit_b_k710_dl_from_giant`, `vit_s_k710_dl_from_giant`; public) combine MVM pre-training, the family SimpleTAD found best for TAD, with K710 post-training. That gives them semantic frozen features, whereas plain MAE features are weak under a frozen read-out. This is why E2 must check them before any training.
- They stay frozen because there are only 4.4 k weakly labelled windows and the source is separable at 1.000. Fine-tuning would most likely learn DADA.
- A third candidate is SimpleTAD's DAPT encoder (BDD100K), **only if the released weight is DAPT-only**. Checkpoints fine-tuned on DoTA or DADA carry benchmark frame labels and are **forbidden**.

**Why causal 1.5 s at 10 fps.** It is SimpleTAD's default window (`X_t` ends at `t`), it spans the approach-to-contact phase of a collision, and it streams with no look-ahead.

**Why fusion is a zero-initialized residual:**
1. **Training starts exactly at the baseline function, but zero-init does not keep it there.** Under Adam the first updates are ≈ lr · sign(g), whatever the gradient's size, so one step can grow a motion output channel by up to lr · Σ_j |c · ũ_j / σ_j| ≈ 5e-5 × 768 × 0.8 × 0.44 ≈ 0.014. At that bound the motion term reaches the CLIP channel size (≈ 0.44) in ≈ 30 of 2,040 steps. Zero-init therefore guarantees the **start**, not proximity to the baseline. The evidence that the stream is used is measured instead: the **motion share** `ρ_u = ‖W_u (c · ũ ⊘ σ_u)‖ / ‖x̃‖`, logged every 50 steps and reported for A2 and A3. Unlike KIP (G4), no auxiliary loss acts on the trunk; the motion term is shaped only by the task losses.
2. **The CLIP path, the heads and the language path are untouched**, so "+ Motion Stream" differs from KIP-off by one additive term. That gives clean attribution.
3. **A fixed, dataset-level scale, never a per-token one.** `σ_u` is computed once on T2-train and frozen. A per-token LayerNorm would divide each step by its own norm; after CRN that norm *is* the deviation from the clip reference, so LayerNorm would erase exactly the signal CRN creates. With `σ_u`, a step that deviates twice as far still enters twice as large, and the magnitude reaches `V^t`: the temporal encoder is post-LN inside, but wraps its layers in an outer residual, `V^t = H + Enc(H)` (`core/models/temporal_encoder.py:287`). It survives there at ≈ 0.44× the re-normalized branch at init (‖H‖ ≈ 9.87 vs ≈ √512 ≈ 22.6), and only on `H_bin`'s language-agnostic path: the co-attention is post-LN without an outer skip, so `V^u` is re-normalized (architecture §5). The scalar `c` only sets the motion input to the CLIP input's per-channel scale, the same data-derived logic as the CLIP scalar `s`.

**Why squash, and why there is no geometry probe.** DADA frames measure **1584 × 660** (2.40 : 1; `DADA_ORIGIN_PHASE0.md` §3.3, 5 clips × 8 frames; the paper's 1456 × 660 does not match our files) and DoTA frames 1280 × 720.

| Geometry | Horizontal scale, DADA vs DoTA | Vertical scale, DADA vs DoTA | DADA content height |
|---|---|---|---|
| **squash** (kept) | 224/1584 vs 224/1280 → **1.24×** gap | 224/660 vs 224/720 → **1.09×** gap | 224 px (14 patch rows) |
| letterbox | 1.24× | 1.24× | ≈ 93 px (≈ 6 patch rows) |

- Letterbox removes only the difference in anisotropy (2.40 vs 1.78). It makes the vertical gap worse and discards most of the frame, where distant vehicles disappear.
- No resize can put absolute velocities on one scale anyway: pixels per metre depend on each camera's focal length and mounting.
- Squash is therefore kept for the motion stream, as for the CLIP stream (whose caches and metrics are bound to it, lessons C2/C13), and E2 does not probe geometry.
- The variant that matches VideoMAE's pre-training, a short-side resize with 3 horizontal crops at 3× the cost, is hook H8. It runs only if E3 shows the motion stream contributes.

### 4.2 Clip-Referenced Normalization (CRN)

**Definition:**

```
x̃_t = s · ( x_t − μ_t^ref )          for the CLIP stream  (s: one scalar, fixed on T2-train so that E‖x̃‖ = E‖x‖)
ũ_t =        u_t − μ_t^ref            for the motion stream (then divided by the fixed σ_u, §4.1)
```

`μ^ref` is computed over the **source video** during training and over the **clip** at test time. That is the same unit (a full video containing an accident), not the training window.

**What CRN removes.** It removes the **clip-level fixed effect**: whatever is constant across the clip, such as scene, camera, dataset signature, and any constant component of the motion features. It does not compensate ego-motion in any geometric sense.

**Why it is justified:**
1. **The metric already ignores what CRN removes.** DoTA macro AUC is a per-clip ranking, and DoTA micro is computed after per-clip min-max. Both are invariant to any per-clip constant. About three quarters of the input variance is such a constant (G2), and it demonstrably fails to transfer: the source is separable at 1.000, and DoTA is flat over data.
2. **It is the fixed-effects (within) estimator.** Scene, camera and dataset are clip-level confounders. Demeaning within the clip estimates the event effect from within-clip variation only.
3. **Paper precedent.** DSANet shows that referencing a video to its own normal patterns sharpens localization (SG-NM). CRN is the parameter-free version at the input; it does not reuse DSANet's module.
4. **The scale is preserved.** The scalar `s` keeps the CLIP input at the norm the KIP-off hyper-parameters were tuned for.

**The known risk: the reference can sit inside the accident.**
- A plain clip mean is close to normal only when the accident is a small part of the clip.
- If an accident runs to the end and covers 70–80 % of the clip, the mean sits in the accident. Normal steps then look like the deviation, and per-clip min-max cannot undo that reversal.
- Robust statistics help only up to their **50 % breakdown point**. Beyond that, no order-free statistic can find the normal part. Only a temporal assumption can, namely that clips begin normal.

**The reference is therefore chosen from four candidates by the E2 check, not assumed:**

| Reference | Definition | Fails when |
|---|---|---|
| R1 mean | mean over all steps | the accident share is large |
| R2 median | per-dimension median | the accident share is > 50 % |
| R3 robust mean | mean of the 50 % of steps closest (ℓ2) to R2 | the accident share is > 50 % |
| R4 past-only | mean of the **strictly past** steps τ < t, for t ≥ `N_w`; for t < `N_w`, the mean of the first `N_w` steps (warm-up, `N_w` = 8 steps ≈ 2.1–2.4 s) | the clip starts inside the accident; slow scene drift; a position artefact (below) |

R4 is also the deployment form: streaming emits its first score after the `N_w`-step warm-up.

**The position artefact, and why the rule controls for it in every reference.** Any reference can make the deviation depend on *where* a step sits, not only on what happens there:
- **R4:** with the naive form (steps ≤ t) the deviation is exactly 0 at t = 0, and its variance grows with t even when nothing happens. The warm-up and the strict past remove the t = 0 degeneracy, but not the trend.
- **R1–R3:** under slow scene drift a whole-clip reference sits near the middle of the drift, so the deviation is U-shaped: high at both ends, including late in the clip.

Accidents in DoTA and DADA sit late in the clip, so a deviation that merely rises late already ranks accident steps high. CRN can therefore add a clip-position signal for any reference. The rule scores every reference's deviation **after removing its normal-driving position trend**, prints a position-only ruler beside it, and §10 probes `V^t` for position in every arm.

**The choice rule** (fixed now; E2):
1. Measure the per-clip accident-share histogram on DoTA-dev and on the DADA source videos, plus the share of clips that start normal.
2. **Position ruler.** Score every step by `p_t = t / T` alone and compute its macro AUC per share bin (< 30, 30–50, 50–70, > 70 %) on DoTA-dev and T2-val. It is printed beside every reference below.
3. For each reference, compute the parameter-free deviation `d_t = ‖x_t − μ_t^ref‖` (CLIP features; repeated on `[x ; u]` once the motion features exist). **Residualize it on the normal-driving position trend:** `r_t = d_t − f(t / T)`.
   - `f` is a cubic polynomial fitted by least squares on the **normal steps only** of T2-val (label 0: pre-accident, post-accident and normal-window steps). It therefore models how the deviation drifts with position when nothing happens, and does not absorb the accidents that sit late in the clip.
   - Outside the 5th–95th percentile of `t / T` among those normal steps, `f` is held at its boundary value; the cubic is never extrapolated.
   - `f` is fitted once on T2-val and applied unchanged to DoTA-dev.
   - Compute the macro AUC of `r_t`, per share bin, on DoTA-dev and T2-val. The raw `d_t` AUC is reported too, not decided on.
   - **Cross-check, model-free:** a position-stratified AUC. First min-max normalize `d_t` **within each clip** (the DoTA micro protocol's normalization), so a clip that drifts more does not score higher everywhere. Then split `t / T` into 5 equal bins, compute the pooled normal-vs-accident AUC within each bin, and average the bins weighted by their number of normal × accident pairs.
   - **Coverage fallback:** if the last fifth of `t / T` holds < 5 % of the normal steps (too little post-accident driving to fit `f` there), the stratified AUC replaces `r_t` as the decision metric in steps 4–5.
4. Pick the reference with the best overall DoTA-dev macro **of `r_t`**, among those whose `r_t` AUC stays ≥ 0.5 in both > 50 % bins (no reversal) and whose stratified AUC is also ≥ 0.5.
5. If none qualifies, **CRN is dropped**.
6. Veto: the fixed-effects transfer probe (T2-train → DoTA-dev) with the chosen reference must not fall below the raw-feature probe (interval entirely below 0 → CRN dropped).

**A second, smaller effect.** CRN also changes what the definition-conditioned path sees: centred features instead of raw CLIP features. The co-attention and heads are learned, so this does not break the path, but any benefit it drew from raw CLIP–text alignment is reduced. The factorial experiment (§10) measures the net effect.

### 4.3 What v2 deliberately does not add

| Idea | Why it is not in v2 | Where it went |
|---|---|---|
| New auxiliary losses (flow reconstruction, SG-NM-style consistency) | G4: an auxiliary loss captured the trunk once; its effect sizes are below the detectable effect; the draft version had bugs | hooks H1, H4 |
| Enabling `L_neg` (+ same-source rows) | Off in the KIP-off T2 config; category-label texts create false negatives without a class mask; would be a separate change | hook H3 |
| Overlap-negative MIL (ONM) | A supervision change, not architecture. It is equivalent to span labels at hop resolution (a WS-purity question). | hook H2 |
| A clip-context token | Reintroduces the source signal into `y^bin` through the blended head | hook H5 (only with a stop-gradient, or confined to `H_mul`) |
| DAPT of the video encoder | Useful but costly; first see whether the frozen stream helps | hook H6 |

### 4.4 Falsifiable predictions

| Part | If the mechanism is right… | Measured in | Dropped if… |
|---|---|---|---|
| Rate-matched evaluation | DoTA macro rises on the **existing** checkpoints | E1 | the paired interval does not exclude 0 (keep LaGoVAD's protocol) |
| CRN | the position-residualized deviation keeps the order in every share bin; the transfer probe does not drop; DoTA macro rises; the source-shortcut AUC on `V^t` falls; the position probe on `V^t` does not rise | E2, E3 | reversal in the > 50 % bins, the probe veto, or the E3 rule |
| Motion Stream | `[x ; u]` probe beats CLIP-only (report rule); A3 − A1 (or A2 − A0) excludes 0 on DoTA macro; `‖W_u‖` grows from 0 | E2, E3 | no candidate is eligible, or the E3 rule fails |

---

## 5. Modalities

| Stream | Status | Why |
|---|---|---|
| RGB frames → CLIP | inference + training | Appearance and semantics; the anchor of the definition path |
| **RGB clips → VideoMAE V2** | **inference + training (new)** | The motion signal (G1). Still RGB-only at inference. |
| Language (`Z`) | inference + training | LaGoVAD's definition conditioning (concept drift) |
| Optical flow (RAFT) | **removed** | Harmful as built, neutral once repaired, cannot be regressed from frames, costs 94 GiB (report §4) |
| Depth, pose, panoptic, audio | not used | Depth would face the same induction problem as flow; the others are human- or scene-centric, or absent in dashcams |

---

## 6. LLM integration: unchanged

Holmes-VAU's **ATS** samples frames from `y^bin` for an MLLM incident report. It adds zero detection latency and can be detached.

---

## 7. Data and preprocessing

### 7.1 Datasets (as established by the report's EDA)

| Role | Corpus |
|---|---|
| WS training | DADA-2000 original, **T2** (W = 20, hop 8): 4,401 windows (3,242 abnormal / 1,159 normal) |
| In-domain test | T2 test: 1,106 windows, scored raw |
| Zero-shot benchmark | **DoTA**: 1,397 clips, per-clip min-max (LaGoVAD protocol) |
| Not used | TAD, the DADA archive, D2City / BDD-A normals and CCD (leaks or source shortcut); PreVAD (no pixels, so no motion stream) |

### 7.2 Splits (the benchmark is protected from the checks)

- **T2-val:** 15 % of T2-train source videos, grouped.
  - All arms train on T2-train minus T2-val.
  - T2-val is the in-domain decision set.
  - The learning curve implies this costs ≈ 0.002, and it costs every arm the same.
- **DoTA-dev / DoTA-eval:** **50 % / 50 %** of clips (≈ 700 each), grouped by video. Fixed now, before any v2 run.
  - All checks and adoption decisions use DoTA-dev.
- **T2-test and DoTA-eval are opened once, for the final report.** Full DoTA is also reported for LaGoVAD comparability.

### 7.3 Preprocessing

**Rate harmonization** (subject to E1):

| Counted in steps | DADA T2 (train) | DoTA, stride 8 (current) | DoTA, stride 3 (v2) |
|---|---|---|---|
| 1 step | 0.27 s | 0.80 s | 0.30 s |
| Score kernel 3 | 0.8 s | 2.4 s | 0.9 s |
| MIL k = 4 | 1.1 s | 3.2 s | 1.2 s |
| Sequence | W = 20 → 5.3 s | whole clip, median ≈ 10.4 s | sliding W = 20 → 6.0 s |

- Scores are interpolated to native frames, and the baseline is re-scored under the same evaluator.
- Both protocols are reported.

**Motion clips:** 16 frames at 10 fps, causal, one per step, squashed to 224² like the CLIP stream (§4.1). They are extracted once and cached: every candidate on a T2-train subsample, T2-val and DoTA-dev for E2, then the full T2 set for the chosen encoder only. This replaces the RAFT pass.

**CRN:** one reference per source video (or clip) and stream, in the form chosen in E2, plus one scalar `s` (CLIP stream), one per-channel `σ_u` and one scalar `c` (motion stream), all from T2-train.

---

## 8. Training

- **Loss:** the KIP-off configuration's loss set, **unchanged**. `L_neg` stays off, as in that config, and no term is added.
- **Hyper-parameters (unchanged):** AdamW, lr 5e-5, batch 64 (32 + 32), 2,040 steps, score kernel 3, k = 4, hidden 512.
- **Frozen:** CLIP (both towers) and VideoMAE V2.
- **Trained:** the soft prompts, temporal encoder, co-attention and heads (all as before), plus the Motion Stream's `W_u` (new). `σ_u` is a fixed statistic, not trained.
- **Stage 1 (KIP warm-up):** removed. v2 trains in a single stage.
- **The four arms of the 2 × 2 factorial:**

| Arm | CRN | Motion Stream | Input to the temporal encoder |
|---|:-:|:-:|---|
| A0 (KIP-off) | – | – | `x_t` |
| A1 | ✓ | – | `x̃_t` |
| A2 | – | ✓ | `x_t + W_u·(c·(u_t − m_u) ⊘ σ_u)` |
| A3 (full v2) | ✓ | ✓ | `x̃_t + W_u·(c·ũ_t ⊘ σ_u)` |

---

## 9. Inference

1. **Steps** at the harmonized rate: stride 8 on 30 fps sources, stride 3 on 10 fps.
2. **Encode:** CLIP on the frame at each step; VideoMAE V2 on the causal 1.5 s clip ending there. The definition `Z` is encoded once and cached.
3. **CRN:** the chosen reference over the clip. If R4 was chosen, the strictly-past mean makes the whole pipeline streamable after the `N_w`-step warm-up.
4. **Trunk:** sliding windows (W = 20, hop 4) → LaGoVAD trunk → `y^bin`, `y^mul`; overlapping steps are averaged.
5. **Score:** interpolate to native frames; apply per-clip min-max (benchmark) or a threshold (deployment).
6. **Optional:** ATS → MLLM report (off-path).

**Cost per step:**

| Encoder | GFLOPs |
|---|---:|
| CLIP ViT-B/16 image | ≈ 17.5 |
| VideoMAE V2-S | ≈ 57 |
| VideoMAE V2-B | ≈ 180 |

- The trunk and heads cost a negligible amount.
- At 3.75 steps/s the total is ≈ 280–740 GFLOPs/s: real-time on one GPU. SimpleTAD's VideoMAE-S runs at 95 windows/s.
- RAFT is gone from training.

---

## 10. Evaluation and plan

### 10.1 Endpoints and statistics (fixed before any v2 run)

- **Seeds:** n = 5 (2024–2028), paired by seed. No seeds are added after a result is seen.
- **Decision interval (the only one any adoption rule uses):** the paired t95 over the 5 seeds of the per-seed DoTA-dev macro Δ (t = 2.776). It measures training variance, which is what v1's intervals were dominated by. Clip-sampling variance is what DoTA-eval checks.
  - **Reported, never decided on:** a clip-level paired bootstrap (10,000 resamples of DoTA-dev clips, seed-averaged scores).
  - **Exception, E1 only:** E1 re-scores the 3 existing checkpoints, where t = 4.303 makes the seed interval uninformative. E1 decides on the clip-level paired bootstrap instead.
- **Minimum detectable effect (MDE) on the decision set (E0b, descriptive; it decides nothing):**
  - *Derived for full DoTA:* ≈ 0.022. Decisions use DoTA-dev (50 %), so the relevant number is larger.
  - DoTA-dev is **fixed at 50 %** now. An MDE estimated from one 3-seed contrast has 2 degrees of freedom and could be off by ≈ 0.5× to 6×, too unreliable to switch the split on.
  - E0b re-scores every existing seed-matched contrast on DoTA-dev only: phase-4 KIP-on (v1) − off, phase-5 KIP-on (v2) − off, and the two learning-curve arms (25 % − 100 %, 50 % − 100 %). It pools their paired-Δ variances (**≤ 8 df**: the four contrasts share the same KIP-off seeds, since the learning curve's 100 % arm is KIP-off, so they are correlated) and reports `MDE_dev = 2.776 · SD_pooled / √5`. No training.
  - `MDE_dev` is printed beside every contrast in the E3 read-out. A non-significant costly or full contrast is reported as **"not detectable at MDE_dev"**, never as "no effect".
- **Primary endpoint:** DoTA **macro** AUC, decided on DoTA-dev and reported on DoTA-eval and on full DoTA.
- **Key secondary:** T2 macro, decided on T2-val and reported on T2-test.
- **Required breakdowns** (the CRN risk):
  - DoTA macro **per accident-share bin**, for every arm;
  - the **source-shortcut AUC** measured on the trunk output `V^t` and on `y^bin`, for every arm;
  - the **position-only ruler** (`t / T`) macro AUC printed beside every DoTA and T2 macro, and a **position probe** on `V^t` (linear, predicting `t / T`, R² reported) for every arm. An arm whose position R² rises above A0's has gained a position channel, and its macro gain is read with that caveat.
- **Also reported:** DoTA micro (LaGoVAD-comparable), AP, MCC / AUCMCC (SimpleTAD), T2 micro, and efficiency.
- **Guardrails** (on T2-val):
  - T2 micro < the clip oracle, and macro ≥ micro;
  - T2 micro no more than 0.01 below A0.

### 10.2 Plan

| Step | What | Trains? | Decision |
|---|---|:-:|---|
| **E0** | Rate audit from the feature caches: fps, stride, seconds per step, clip length in seconds | no | confirms or refutes G3 |
| **E0b** | Pooled `MDE_dev` from all existing seed-matched contrasts re-scored on DoTA-dev (§10.1) | no | descriptive only: printed beside every contrast; the dev share is already fixed at 50 % |
| **E1** | Rate-matched evaluation on the **3 existing KIP-off checkpoints**: A = stride 8 whole clip; B = stride 3 whole clip; C = stride 3 sliding W = 20 | no | adopt B or C if the clip-level paired bootstrap interval on DoTA-dev excludes 0 (the E1 exception, §10.1); otherwise keep LaGoVAD's protocol |
| **E2** | (a) accident-share histograms, DoTA and DADA sources. (b) Position ruler, then CRN reference choice on the position-residualized deviation (§4.2). (c) CRN transfer-probe veto. (d) Encoder choice: probe `[x ; u]` for VideoMAE V2-B, V2-S (and SimpleTAD DAPT if DAPT-only), all squashed; probe the representation each arm feeds the trunk (raw `u − m_u` for A2, CRN'd `ũ` for A3, both ⊘ `σ_u`); eligible if the in-domain DoTA-dev probe is ≥ +0.10 over CLIP-only (the report's rule) or the transfer probe is ≥ +0.03. Pick the best transfer probe; within 0.02, the cheaper encoder wins | no | fixes the reference and the encoder, or drops CRN or the Motion Stream before any training |
| **E3** | **2 × 2 factorial** (A0–A3) × 5 seeds = **20 runs** | yes | the rules below |
| **Final** | Open T2-test and DoTA-eval once | no | reported numbers |

**Run order** (E0b and E1 already decide on DoTA-dev, so the splits must exist first):
1. **Freeze and commit** the T2-val source list, the DoTA-dev / DoTA-eval clip lists and this proposal's rules. Record the commit hash in every read-out.
2. **Keep DoTA-eval hidden.** The re-scoring harness loads only DoTA-dev clip IDs, so no DoTA-eval number is ever printed before the final report.
3. **E0 first** (minutes). If the rate gap is not ≈ 3×, E1 is skipped.
4. **E0b and E1 together**, on one harness: both re-score existing checkpoints on DoTA-dev.
5. **Start the VideoMAE extraction for E2(d) in parallel** with steps 3–4; it takes longest.
6. **Within E2, (a) before (b):** the share bins and the coverage fallback depend on the histograms.

### 10.3 Adoption rules (fixed now)

- **Contrasts** (paired by seed, DoTA-dev macro):
  - CRN effect: A1 − A0;
  - Motion Stream effect: A2 − A0;
  - full effect: A3 − A0;
  - Motion Stream on top of CRN: A3 − A1;
  - interaction: (A3 − A2) − (A1 − A0). **Descriptive only.** Its variance is about twice that of a main contrast, so at n = 5 it decides nothing.
- **Final model** (every interval is the decision interval of §10.1). Let `F` be the **best adoptable free arm**: A1 if A1 passes the free rule below, A0 otherwise. A costly arm must beat `F`, not merely A0, and a failed A1 must not block a good costly arm.
  - **A3** if the A3 − A0 interval **and** the A3 − `F` interval exclude 0, and the guardrails hold.
  - Otherwise **A2** if the A2 − A0 interval **and** the A2 − `F` interval exclude 0, and the guardrails hold. (Beating `F` stops A3 ≈ A1 = +0.04 with a significant A2 = +0.03 from shipping A2, costlier and worse. Using `F` instead of A1 stops a failed A1 from sending a significant A2 to A0.)
  - Otherwise **A1** if its point estimate is ≥ 0 on both the primary endpoint and T2-val macro, with no reversal in the share bins. CRN is free, so it needs only a non-negative result.
  - Otherwise **A0**, and v2 is reported as a bounded null.
- **What the thesis may claim about motion.** "The motion stream improves DoTA" requires the contrast that isolates it to exclude 0: A3 − A1 if A3 is adopted (even when `F` = A0), A2 − A0 if A2 is. An A3 − A0 gain alone supports only "v2 as a whole improves DoTA".
- The DoTA-eval numbers measure any optimism from selecting on DoTA-dev.

---

## 11. Justification ledger

| Decision | Rationale | Source |
|---|---|---|
| Keep LaGoVAD | Only venue-eligible WS baseline with traffic evaluation; trains correctly on T2 | LaGoVAD; report §3.1 |
| Remove KIP and RAFT | Harmful, then neutral; K > W; untrained gate; 94 GiB | report §4 |
| Motion enters as an input, not a loss | Induction needs a motion-bearing input; frame CLIP has none | Pi-VAD (PMG from I3D); report §4.3 |
| VideoMAE V2 (distilled, frozen) | MVM video encoders win TAD; distilled checkpoints have usable frozen features; frozen avoids learning DADA | SimpleTAD; report §2.4 |
| Encoder chosen by probe | Frozen performance is not what SimpleTAD measured (fine-tuned) | SimpleTAD; the report's pre-registered rule |
| Forbid DoTA/DADA-fine-tuned weights | Label leak; not WS | SimpleTAD training protocol |
| Causal 1.5 s at 10 fps | SimpleTAD's window; streamable | SimpleTAD |
| Zero-init residual fusion | Starts at the baseline; one additive term, so clean attribution; no capture | report §4.2 (lesson) |
| CRN | ≈ 75 % clip-constant variance; source 1.000; the metric discards clip level; fixed-effects estimator | report §2.2, §2.4, §3.3; DSANet (precedent) |
| Source-video / clip as the reference unit | The same composition at train and test; a window reference manufactures label noise | report §2.2 |
| Reference chosen with a reversal check | The mean breaks with long accidents; robust statistics break at 50 %; only a past-only reference uses temporal order | review point 4 |
| Position ruler, deviation residualized on a normal-steps-only trend, stratified cross-check, strictly-past R4 with warm-up | Every reference can make the deviation rise late (R4 by construction, R1–R3 under drift), and accidents sit late; fitting the trend on all steps would subtract accident signal | review rounds 2–3 |
| Fixed per-channel `σ_u`, no LayerNorm | A per-token norm erases the deviation magnitude that CRN creates | review round 2 |
| Squash for the motion stream; no geometry probe | Measured DADA 1584 × 660 vs DoTA 1280 × 720: squash leaves a 1.24× horizontal and 1.09× vertical scale gap; letterbox gives 1.24× both ways and ≈ 93 px of DADA content | review round 3 |
| Scalar `c` on the motion term; motion share `ρ_u` logged | Under Adam, zero-init sets only the start; `c` puts both streams on one per-channel scale (CLIP RMS ≈ 0.44), and `ρ_u` is the evidence the stream is used | review round 3 |
| Costly arms must beat the best adoptable free arm `F` (A1 if it passes, else A0) | A costly arm must beat the best free arm; a failed A1 must not block it | review rounds 3–4 |
| Scalar `s` on the CLIP stream | Keeps the input norm that the KIP-off hyper-parameters were tuned for | — |
| No new losses; `L_neg` stays off | G4; `L_neg` needs class masking and is a separate change | report §4.2; review point 2 |
| No context token | It would bring the source signal back into `y^bin` | review point 3 |
| Rate-matched evaluation | 3× time-scale gap; tested on existing checkpoints | SimpleTAD; report §2.2, §4.5 |
| 2 × 2 factorial, n = 5, one decision interval (paired t95), DoTA-dev fixed at 50 %, pooled MDE reported | Main effects in 20 runs; A3 must beat A1 too; a 2-df MDE is too noisy to switch the split on | derived from report §3.2; review rounds 2–3 |
| DoTA-dev / eval split, T2-val | Decisions never touch the reported test sets | protocol hygiene |
| Per-share-bin and source-shortcut reporting | Makes CRN's known risk and its intended effect visible | review points 3–4 |
| ATS, off-path | Zero detection latency | Holmes-VAU |

---

## 12. Deferred (not in v2; each needs its own experiment)

- **H1 — Flow as an auxiliary on the motion stream.** Only if the flow head beats the per-clip oracle (K < W) on the new input; weight set from ρ.
- **H2 — Overlap-negative MIL (the E4a rung).** Report it with and without, because it is equivalent to span labels at hop resolution.
- **H3 — `L_neg` enabled (the E4b rung).** Needs a **class mask** for duplicate category labels (83 strings; the top one 12.2 %), plus same-source normal windows as hard-negative rows.
- **H4 — SG-NM-style consistency, ablation only.** Fix the draft's bugs first: no per-window min-max, a stop-gradient on the reconstruction target, and powering as an E7-style ablation, not a ladder rung.
- **H5 — A clip-context token for concept drift.** Only with a stop-gradient in the `H_bin` path, or confined to `H_mul`, and with the source-shortcut AUC re-measured on `y^bin`.
- **H6 — DAPT of VideoMAE** on BDD100K / DADA-train frames (SimpleTAD). Never on DoTA.
- **H7 — Train at DoTA's coarser rate**, if E1 shows DoTA rewards it.
- **H8 — Pre-training-matched clip geometry** for the motion stream: short-side resize to 224 with 3 horizontal crops (≈ 3× the encoder cost), averaged or concatenated. Only if E3 shows the motion stream contributes.
  - **Limitation:** H8 can improve a stream that works, but it can never rescue one that fails. If E2 makes the stream eligible only through the transfer rule (between +0.03 and +0.10) and E3 then fails, clip geometry remains an **untested cause** of that failure, and the thesis says so.

---

## 13. Answers to the report's advisor questions

**(a) Option 1 (a new flow target) or option 2 (a video representation)?**
Option 2.
- Option 1 changes the *target* of a regression whose *input* lacks the information (§4.0).
- v2 implements option 2 as a frozen VideoMAE V2 stream feeding the temporal encoder, with frozen CLIP kept for the text path. That is the report's own option-2 structure.
- Option 1 is meaningful only on top of it (H1).

**(b) Is dropping KIP compatible with "LaGoVAD + a kinematics pathway"?**
Yes, with honest wording.
- The pathway is a motion-bearing video stream plus a clip-level fixed-effect normalization.
- "Kinematics-aware" means motion-aware features, not explicit trajectory modeling.
- The KIP diagnosis is the chapter that motivates the change.

**(c) Is the negative result publishable?**
As a thesis chapter, yes. As a standalone paper, it is workshop-level, and strongest as the motivation for v2. Its general lessons are:
1. Induction cannot recover information absent from the input.
2. An unnormalized auxiliary target can capture a shared trunk while staying orthogonal to the task.
3. An integer-cast "gate" is a fixed hyper-parameter.
