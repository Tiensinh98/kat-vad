# KAT-VAD V4: Steering Expert Heads with the Clip's Own Reference

*Revision 3, 2026-10-15. Revision 2 answered the review of revision 1 (§2); revision 3 answers the feasibility review of revision 2 (§2.1). The binding decision rules now live in [`core/docs/v4/PREREG.md`](PREREG.md); where this proposal and the PREREG differ, the PREREG wins. Citations are given as [document, section]. Facts marked **[review]** come from the reviews; every file they quote is in the repository (`core/docs/v2/RESULTS_E4_0.md`, `core/docs/v2.2/RESULTS_DF8.md`, `core/docs/v2/PREREG_ADDENDUM.md` §25, `core/docs/v2/RESULTS_E2D.md`, and the code paths named). Numbers marked "derived" are arithmetic on reported numbers. Every target is a pre-registered hypothesis, not an expected result.*

## 1. Executive summary

KAT-VAD V4 is V2, trained with the A3-ign recipe unchanged, plus one input module, **Reference-Steered Expert Heads (ReSH)**. It adds **no new loss**. ReSH does four things:

1. **Tap.** It reads 5 of the 72 attention heads (12 blocks × 6 heads × 64-d) inside the frozen VideoMAE V2-S. It does this in the same forward pass that already produces V2's video feature.
2. **Select (offline, T2-train only).** Heads are ranked with HeadHunt-VAD's four separability scores and its robustness penalty, computed across folds of source videos. The scores are computed after each source video's slow trend in time has been removed. Heads that only encode "late in the video" or "aftermath" are therefore not chosen.
3. **Normalise.** The heads pass through V2's own Clip-Referenced Normalization (CRN).
4. **Steer.** A SteerVAD-style gate and direction rescale the heads once per clip. Both are computed from a 32-d projection of the CRN reference, which V2 already computes and then discards.

The rescaling is multiplicative, has no bias, is non-negative and is constant within a clip. Each channel therefore keeps a zero per-clip median. Per channel, and before its projection, ReSH adds no clip-constant offset. That offset is the corpus shortcut that CRN removed in V2 (0.9998 → 0.57).

ReSH adds **167,361 parameters (≈ 0.17 M)**, of which the steering controller is 3,009. They enter through a zero-initialised projection, so V4 starts exactly at V2.

**The claim.** Accident evidence in a frozen video encoder is concentrated in a few heads. Those heads can be selected robustly from weak labels once the time-in-video confound is removed. Adding them to V2, and rescaling them by the clip's own reference, improves zero-shot detection on DoTA-CAP without increasing V2's position-shift cost. V4 makes no motion claim.

**How it is decided.** A fixed-sequence test on DoTA-CAP-dev runs three contrasts in order:

1. A4-H − A3-ign
2. A4 − A4-H
3. A4 − A4-rand

Every step uses the frozen SC-2 bar. The two steps that decide adoption also need a non-inferiority guardrail on the DF8 position-shift set. Two probe gates run before any training, and at most 15 training runs follow (20 if the A3-ign base has to be retrained). The transfer gate is a low-probability bet (§7.3), and failing it costs no training.

**Labels.** KAT_VAD_PROPOSAL_v2.md §12 (hooks H1–H8) and REPORT_V2_RESULTS.md §8 (directions H1–H6) reuse the same labels for different things. This document always writes **proposal-Hn** or **report-Hn**.

## 2. Review response, round 1 (2026-10-15)

| Point | Verdict | Resolution |
|---|---|---|
| **A1.** A3-ign's CAP-dev macro is printed | Correct | It is 0.7912 [0.7740, 0.8077] [review: `RESULTS_E4_0.md` §1]. The two derivations are removed. The gap to `p_T2` (0.8220) is 0.0308 (§3, §8.5). |
| **A2.** The identical optimism is not a copied cell | Correct | The caveat is removed. On the rounded numbers, A3-ign's dev − eval is −0.0208; the printed −0.0209 comes from the unrounded values. |
| **A3.** V2-S has 12 blocks | Correct | The candidate pool is 72 heads [review: `core/constants.py`, `VIDEOMAE_ARCH`]. The "verify in code" item is removed. |
| **A4.** The position-shift set exists and was read | Correct | A3-ign fails X7 on 491 CAP-dev clips: early − control is −0.125 [−0.141, −0.110], and early − late is −0.199 [review: `RESULTS_DF8.md`]. DF8 is now a decision input (§8.2), no longer the proposed "fresh set" (§8.5). |
| **A5.** `f_2` is not position-free | Correct | Every `f_2` is now written "`f_2` (DoTA-CAP placement)", and it no longer licenses "beyond position" in general. The primary metric is restated in §8.1. |
| **A6.** The cache does not store the CRN reference | Correct | `build_v2_inputs` is re-baked with per-source and per-clip reference columns (CPU only). The A4-S fallback needs this re-bake too: it needs no extraction, but it is not free (§5.4). |
| **B1.** A4-static is absorbed by `W_e` | Agree; one refinement | `W_e·diag(1 + tanh v)` is a reparameterisation of `W_e`, so **A4-static is dropped**. A4 − A4-H changes the parameterisation as well as adding context, and Adam is not invariant to reparameterisation. It therefore licenses "the steering module helps", not "context helps". Context is isolated by a free test-time control instead: the trained A4 is scored with references permuted across clips (§8.3). The SteerVAD summary lists its scorer as trained together with the controller (§3). It does not say whether its "static scaling vector" row keeps the global gate, so that row cannot settle the question. |
| **B2.** Window labels will select position or aftermath heads | Correct | Selection now scores features detrended within each source by a cubic in t/T. The raw selection is also run, their overlap is reported, and each selected head's beyond-`p` probe gain is printed (§7.1). Detrending is preferred to ranking by beyond-`p` gain because `p` is one function of t/T for every source. It removes the position effect across sources, but not each source's own aftermath plateau. |
| **B3.** The gate reads the channel DF8 suspects | Correct; risk accepted and guarded | X7 non-inferiority is part of adoption: the lower bounds of ΔX7 (arm − A3-ign) must be > −0.01 (§8.2). The gate's drift across crops is logged. The permutation control shows whether the gate uses the reference at all. A4-H, the fallback adoption, has no gate. |
| **B4.** The fallback gives two shots; the bar is below SC-2 | Correct | Fixed-sequence gatekeeping in the order the review proposed, with the SC-2 bar at every step (§8.1, §8.3). Accepted cost: if the heads help only when steered, V4 reports a null. |
| **B5.** G1 has no full-layer control | Correct | G1a and G1b now include `ẽ_full`: all 6 heads (384-d) of the block with the highest mean RSS (§7.2). |
| **B6.** State G1b's prior odds honestly | Correct | Stated in §7.3: G1b is less likely than not to pass. |
| **C1.** BatchNorm sees padded and duplicated steps | Correct | BatchNorm is removed. `s` and `g` are computed once per segment, so the §5.5 property is now exact in training as well as at test. |
| **C2.** The gate can overfit | Correct | `r` is standardised and reduced by PCA to 32 dims (whitened, fitted on the 1,272 T2-train sources). The gate's hidden width is 32. The controller shrinks from 134,401 to 3,009 parameters. |
| **C3.** Stacked zero-inits | Correct | The gradient path is documented (§5.3): `W_e` trains from step 0, the up-maps from step 1, and the down-maps and gate from step 2. |
| **C4.** Seed pairing can break | Correct | ReSH is built inside `fork_rng` with its own seed. The code asserts that the trunk's step-0 `state_dict` and the first batches' indices match E4-0 (§5.3). |
| **C5.** Hooking is simpler than described | Correct | A `forward_pre_hook` on each block's `attn.proj`; its input is B × N × 384, reshaped to (6, 64) (§11). |
| **C6.** The D6 re-extraction can be cut | Correct | Cut. |
| **D.** Cost and staging | Accepted, with one change | Stages A–D as proposed. DF8 for A4-H and A4 moves into Stage C, because X7 is now an adoption guardrail; it is forward passes on cached features. Runs drop to 15 (from 20–25). The full path still costs more than SC-2's 5 runs (§7.5). |

### 2.1 Review response, round 2 (feasibility, 2026-10-15)

| Point | Verdict | Resolution |
|---|---|---|
| **F1.** X7 is a bar V2 itself never cleared. Paired ΔX7 from DF8's per-seed macros: A3 − A0 **−0.041 [−0.067, −0.014]**, A3-ign − A3 **−0.041 [−0.058, −0.025]**, A3-ign − A0 −0.082 [−0.093, −0.071]. Every adopted V2 step raised the position cost by ≈ 0.04, so "ΔX7 lower bound > −0.01" asks A4 to break that pattern | Correct | The guardrail stays (V4's claim is "without increasing V2's position-shift cost", so the claim needs it). The calibration is printed beside the rule (§8.2), and a probe-level early warning, **G1c**, runs in Stage B before any training (§7.2). |
| **F2.** DF8 for ReSH is not "cached features" | Correct | DF8 re-extracts the V2-S windows whose causal 1.5 s reaches before a crop's first frame (`position_shift heads`). ReSH needs those windows hooked too. The 7,524 crop windows move into **Stage B's** GPU pass, where G1c needs them anyway. |
| **F3.** "Cached `u` reproduced to 1e-5" will fail | Correct | DF8's re-extraction reached min cosine 0.99935 (X10(c)). The gate is now **per-row cosine ≥ 0.999** against the existing V2-S cache, as X10(c). `proj(tapped) == attention output` stays exact and is a unit test. |
| **F4.** Step labels are ambiguous (a step sits in 2–3 overlapping windows) | Correct | Selection uses **weak window labels** (the claim says "from weak labels"): a step takes a label only if every T2-train window covering it agrees; conflicting and uncovered steps are dropped from scoring. The probes (G1a–c) are measurements of feature content, not the model, and keep the project's probe convention: frame labels from the annotation span at s8 (K, K-pos, E2(d)). |
| **F5.** Detrending can absorb real signal | Correct | Pre-registered stability floor: the mean pairwise Jaccard of the five folds' top-5 sets must be **≥ 0.25** (two of five heads shared between two folds on average; chance for two random 5-of-72 sets ≈ 0.04). Below the floor the selection is noise and G1a fails. |
| **F6.** Simpler wiring | Correct | `r̂` (32-d) is baked as columns of every step of a source; DVS slices carry it without a segment index. Because `r̂` is constant over a source and the gate has no BatchNorm, the per-step gate equals the per-segment one. Built in Stage C, not Stage A. |
| **F7.** Commit the rules before the numbers | Correct | The rules are in `core/docs/v4/PREREG.md`, committed before Stage A runs. `head_selection.json` is committed before G1a reads any probe; the G1a cell refuses to run until its sha256 is pasted. |
| **F8.** Stage A should build only what the gate needs | Agree | The reference-column re-bake, `σ_e`, `m_r`, `s_r` and PCA-32 move to Stage C preparation (CPU). Stage A = one hooked V2-S pass over T2, selection, G1a. |

## 3. What V2's results say

The adopted V2 has these parts ([REPORT_V2_RESULTS.md, §0–§1]; [KAT_VAD_v2_ARCHITECTURE.md, §3–§5]):

- LaGoVAD's trunk.
- A frozen CLIP ViT-B/16 frame stream (`x ∈ ℝ^512`).
- A frozen VideoMAE V2-S stream (`u ∈ ℝ^384`) on a causal 16-frame window.
- CRN (R2, the per-dimension median over the source video in training and the clip at test).
- One zero-initialised `W_u` (≈ 0.20 M).
- LaGoVAD's losses minus `L_neg`, trained with `dvs_anchor_mode=ignore`.

It trains on DADA-2000 T2 minus T2-val (1,491 − 219 = 1,272 sources) and is read zero-shot on DoTA-CAP.

**Table 1. Key arms** (macro AUC, 5-seed mean) [REPORT_V2_RESULTS.md, §4.1, §4.4, §4.5].

| Arm | CAP-eval (560) | CAP-dev (569) | CAP-eval micro / AP | T2-test macro / micro |
|---|---|---|---|---|
| A0, raw CLIP | 0.6979 | 0.6768 | 0.651 / 0.447 | 0.6307 / 0.6228 |
| A3, CRN + video stream | 0.7785 | 0.7576 | 0.730 / 0.524 | 0.6950 / 0.7095 |
| **A3-ign (V2)** | **0.8120** (second look) | **0.7912** [0.7740, 0.8077] [review] | 0.782 / 0.596 | 0.7202 / 0.7334 |
| `p_T2`, position only, no pixels | 0.8256 | 0.8220 | 0.802 / 0.607 | — |

How the splits are used:

- **CAP-dev** is the decision set.
- **CAP-eval** has been opened twice (the Final and A3-ign's second look).
- **T2-test** is a guardrail and never adopts anything.

**Position-controlled contrasts**, all "`f_2` (DoTA-CAP placement)" [REPORT_V2_RESULTS.md, §4.3, §4.5]:

| Contrast | Split | Raw Δ | `f_2` Δ |
|---|---|---|---|
| A3 − A0 | CAP-eval / CAP-dev | +0.081 / +0.081 | +0.021 [+0.009, +0.034] / +0.023 [+0.013, +0.033] |
| A2 − A0 (no CRN) | CAP-eval | +0.067 | +0.004 [−0.010, +0.018] |
| A3-ign − A3 (E4-0) | CAP-dev | +0.0398 [+0.0347, +0.0450] | +0.0102 [+0.0048, +0.0154] |
| A3-ign − A3 (second look) | CAP-eval | +0.0376 [+0.0326, +0.0426] | +0.0157 [+0.0104, +0.0207] |

**DF8, the position-shift stress test** (491 CAP-dev clips, read 2026-10-15) [review: `RESULTS_DF8.md`]. Each clip is cut three ways at the same length: an early crop, a late crop, and a control crop with the accident at its original position. The accident centre moves between 0.30 and 0.70 [REPORT_V2_RESULTS.md, §8 report-H1].

| Quantity | Value |
|---|---|
| A3-ign position cost, early − control (X7) | **−0.125 [−0.141, −0.110]**: fails X7 |
| A3-ign, early − late | −0.199 |
| `f_2` on the early crops | 0.45–0.48 for every model |
| What the early crops' negatives are | mostly aftermath |
| Where the extra sensitivity enters (same-frame analysis) | through the crop's CRN median |

**Table 2. Ranked bottlenecks.**

| Rank | Bottleneck | Evidence | ReSH's relation |
|---|---|---|---|
| 1 | **Little evidence beyond position, and what exists is position-sensitive** | A3-ign is below `p_T2` on both splits (0.8120 < 0.8256; 0.7912 < 0.8220). `f_2` gains are +0.010 to +0.023. `u` adds only +0.043 beyond position in the K-pos probe ([REPORT_V2_RESULTS.md, §3]). DF8 position cost is −0.125, driven by aftermath negatives. | primary target (evidence); guarded (position cost) |
| 2 | Non-ego accidents | A3: 0.697 (n = 248) vs 0.804 (ego) | read-out only |
| 3 | Span coverage | argmax inside the span on 61 % of clips; A3-ign's inside-span mean falls by 0.155 vs A3 | not targeted |
| 4 | Early onset (< 20 % of clip) | A3: 0.675 (n = 54) | read-out only |

Rows 2–4 are A3 on CAP-dev and exploratory [REPORT_V2_RESULTS.md, §6]. Cheap fixes for the other bottlenecks have already failed:

- the causal hold, report-H2(a): −0.071;
- SG-NM: −0.012;
- CLIP tiles: −0.012;
- text-guided windows: −0.036.

More T2-like data is flat (+0.008 over 4×) [REPORT_V2_RESULTS.md, §7–§8].

**The video stream is not motion.** Shuffling its 16 frames changes the `[x;u;p]` probe by −0.0004 / −0.0003 (D6). The report forbids "motion" and "kinematics-aware" [REPORT_V2_RESULTS.md, §3, §5]. V4 makes no motion claim and runs no D6 analysis.

## 4. Why ReSH

**(i) V2 pools away head-level evidence.**

V2's video feature averages the final block's tokens and passes them through `fc_norm` [KAT_VAD_v2_ARCHITECTURE.md, §3.2]. HeadHunt-VAD argues that a layer's output `Concat(h_1..h_Nh) W^O` dilutes a few anomaly-sensitive heads among many background heads [headhunt-vad-summary.md, §2.4]. Its ablation [headhunt-vad-summary.md, §6.4]:

| HeadHunt variant | UCF AUC / XD AP |
|---|---|
| Selected heads | 87.03 / 82.63 |
| Full-layer features (final layer) | 80.15 / 72.10 (−6.88 / −10.53) |
| Random-K heads | 66.65 / 45.33 |

SteerVAD's selection ablation shows the same pattern: selected heads 87.15, mid-to-last layers 84.68, random heads 69.57 [steervad_paper_summary.md, §5.2].

HeadHunt's full-layer control is the final layer. V4's G1 uses a stricter control: the full layer of the best intermediate block (B5). If the selected heads only beat the final pooled `ũ`, the gain could simply be "intermediate layers are better". V2's probe fits the dilution story without proving it: `u` adds +0.132 over CLIP, but only +0.043 of that survives position [REPORT_V2_RESULTS.md, §3 P1·K, K-pos].

**(ii) V2 computes the scene reference and only subtracts it.**

CRN's reference carries the scene, camera and corpus signature [KAT_VAD_PROPOSAL_v2.md, §1.2 G2]. V2 rejected a context token [CTX] because it "reintroduces the source signal into `y^bin`" [KAT_VAD_PROPOSAL_v2.md, Review response; §4.3]. SteerVAD shows how to use context without adding it as an input [steervad_paper_summary.md, §5.2]:

| SteerVAD variant (UCF AUC; full model 87.15) | Δ |
|---|---|
| No global gate | −1.21 |
| Additive instead of multiplicative steering | −2.13 |
| Static scaling vector instead of the direction module (whether the gate is kept is not stated) | −2.94 |
| Linear probe on the raw expert heads | −5.82 |
| Context `[c, h_i]` instead of global context only | −0.80 |

ADSM independently supports conditioning on the scene: its scene label adds +2.2 macro on ShanghaiTech (84.1 → 86.3) [ADSM_paper_summary.md, §7.5].

**The caveat the review raised (B3).** On T2 the reference is the median over a whole source video (~322 frames, with a small accident share) [review], so it describes the scene. On DoTA it is the median over an accident-centred clip. DF8's same-frame analysis infers that A3-ign's position sensitivity enters through exactly this median [review]. On DoTA, then, the reference partly describes the accident and moves with the crop. ReSH feeds it to the gate. This is the main risk of the steering half, and §8.2 turns it into a guardrail.

**(iii) The paper numbers are upper bounds, not forecasts.**

- Every SteerVAD and HeadHunt delta comes from a frozen InternVL3 read by a linear or logistic scorer on UCF-Crime or XD-Violence. None comes from traffic data.
- The two papers also disagree: a plain probe of the selected heads scores 81.33 in SteerVAD and 87.03 in HeadHunt [steervad_paper_summary.md, §5.2; headhunt-vad-summary.md, §6.4].
- V2's trained temporal encoder can already reweight its input channels.

So V4 uses these deltas for direction only.

## 5. V4 architecture: one zero-initialised block between CRN and the trunk

### 5.1 Diagram

```
frame ─► CLIP ViT-B/16 (frozen) ─► x ─► CRN ─► x̃ ──────────────────────────────────────────────────┐
                                   └─ μ^x ∈ ℝ^512 ──┐                                               │
causal 16-frame clip ─► VideoMAE V2-S (frozen, one pass)                                          │
   ├─ final block → token mean → fc_norm ─► u ─► CRN ─► c·ũ⊘σ_u ─► W_u (0-init) ──────────────────(+)
   │                               └─ μ^u ∈ ℝ^384 ──┤                                               │
   └─ pre-hook on attn.proj: 5 selected heads → token mean ─► e ∈ ℝ^320                            │
╔═══════════════ ReSH: the only new block ═════════╪═══════════════════════════════════════════╗   │
║ e ─► CRN ─► h̃ = c·ẽ⊘σ_e ─► ⊙ a ─► h' ─► W_e (0-init, 320→512) ─────────────────────────────╫─(+)─► H
║                            ▲                                                                ║
║ r = [μ^x; μ^u] ∈ ℝ^896 ─► z-score ─► PCA-32 (T2-train) ─► r̂ ─┬─► gate 32→32→1 ─► s ∈ [0,1]    ║
║ (one per segment / clip)                                   └─► 5 rank-4 maps ─► g ∈ [−1,1]^320║
║ a = 1 + s·g ∈ [0,2]^320, one vector per segment                                             ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝
H ─► temporal encoder (2 layers, RoPE) ─► V^t ─► CoAttn ─► H_bin → y^bin, H_mul → y^mul     (A3-ign, unchanged)
```

### 5.2 Equations and parameters

**Tap.** For each of the 5 selected heads, take its pre-`proj` output (64-d per token). Mean-pool it over the 1,568 tokens, the same token mean that V2 uses for `u`. Concatenating the five gives `e_t ∈ ℝ^320`.

**CRN.** `ẽ_t = e_t − μ^e`, where `μ^e` is the R2 median over the source video (training) or the clip (test). Then `h̃_t = c · ẽ_t ⊘ σ_e`. Here `σ_e` is the per-channel std on T2-train and `c ≈ 0.44` is V2's existing scalar [KAT_VAD_v2_ARCHITECTURE.md, §4–§5].

**Context.**
- `r = [μ^x; μ^u] ∈ ℝ^896`: the two CRN references V2 already computes, one per segment.
- `r̂ = P((r − m_r) ⊘ s_r) ∈ ℝ^32`.
- `m_r` and `s_r` are T2-train source statistics. `P` is a whitened PCA fitted on the 1,272 T2-train source references.

**Gate.** `s = σ(w_2ᵀ ReLU(W_1 r̂ + b_1) + b_2) ∈ [0,1]`, with `W_1 ∈ ℝ^{32×32}`. SteerVAD's gate without BatchNorm.

**Directions.** For each head i = 1..5, `g_i = tanh(B_i A_i r̂) ∈ [−1,1]^64`, with `A_i ∈ ℝ^{4×32}` and `B_i ∈ ℝ^{64×4}`, no bias. This is SteerVAD's rank-4 module.

**Steering and fusion.**
- `a = 1 + s·g ∈ [0,2]^320`, one vector per segment.
- `h'_t = h̃_t ⊙ a`.
- `H_t = x̃_t + W_u(c·ũ_t ⊘ σ_u) + W_e h'_t + b_e`.

**Parameters** (derived):

| Part | Count |
|---|---|
| Gate: 32·32 + 32 + 32 + 1 | 1,089 |
| Directions: 5 × (32·4 + 4·64) | 1,920 |
| **Controller subtotal** (revision 1: 134,401) | **3,009** |
| `W_e` + `b_e`: 320·512 + 512 | 164,352 |
| **ReSH total** | **167,361 ≈ 0.17 M** |
| For comparison, V2's `W_u`: 384·512 + 512 | 197,120 ≈ 0.20 M |

### 5.3 Initialisation, gradient path and seed pairing

**Zero-init.** `W_e`, `b_e` and every `B_i` start at zero. At step 0, `H = x̃ + W_u(·)` exactly, so V4 computes V2's function.

**Gradient path (documented, not a bug).**

| Step | What receives gradient | Why the rest gets none |
|---|---|---|
| 0 | `W_e`, `b_e` only | `∂L/∂h' = W_eᵀ ∂L/∂H = 0` |
| 1 | `W_e`, `b_e`, `B_i` (now `∂L/∂h' ≠ 0` and `A_i r̂ ≠ 0`) | `A_i`'s gradient is ∝ `B_iᵀ = 0`; the gate's is ∝ `h̃ ⊙ g = 0` |
| 2 onward | everything | — |

Under AdamW a zero gradient gives no Adam step. In steps 0–1, only decoupled weight decay touches `A_i` and the gate. A side effect helps the comparison: A4 and A4-H compute the same function through step 1, so their paired difference starts at step 2 of 1,740.

**Logging.** As V2 logs `ρ_u`, log `ρ_e = ‖W_e h'‖/‖x̃‖` and the distributions of `s` and `‖g‖` every 50 steps.

**Seed pairing.**
1. Build the trunk exactly as E4-0 does.
2. Build ReSH inside `torch.random.fork_rng(devices=[])` with its own fixed seed. The global stream used for the trunk's initialisation, the data order and any dropout is then untouched.
3. Assert, for every seed:
   - the trunk's step-0 `state_dict` is bitwise equal to E4-0's;
   - the first batches' item indices are equal;
   - with `resh.enabled=false`, CAP-dev scores reproduce E4-0 within GPU nondeterminism.

If the third assert holds, the E4-0 runs serve as the A3-ign base. Otherwise A3-ign is retrained (+5 runs).

### 5.4 Frozen vs trained, cost and data work

**Frozen:** both encoders, CRN, `c`, `σ_e`, `m_r`, `s_r`, `P`, and the head list.

**Trained:** the gate, the directions and `W_e`, plus everything V2 already trains [KAT_VAD_v2_ARCHITECTURE.md, §13].

**Cost and data work:**
- **Encoder FLOPs are unchanged** (V2-S ≈ 57 GFLOPs/step [KAT_VAD_PROPOSAL_v2.md, §9]). The hook copies a tensor that `proj` consumes anyway.
- **Hooked passes.** One hooked V2-S pass over T2 (Stage A) and one over CAP-dev plus DF8's 7,524 crop-boundary windows (Stage B, F2). CAP-eval is needed only for the final read.
- **Storage.** All 72 heads are stored for T2 and CAP-dev, 4,608 float32 values per step (≈ 0.8 GB for T2 at s8).
- **Re-bake (A6, Stage C preparation).** `build_v2_inputs` currently bakes CRN and discards μ [review]. Before Stage C it is re-run to store `r̂` (the 32-d PCA-whitened `[μ^x; μ^u]`) as columns of every step, per T2 source and per DoTA clip (F6). The DF8 tool must emit them per crop, since it already re-bakes CRN on every crop [REPORT_V2_RESULTS.md, §8 report-H1].
- **Where ReSH runs.** Per segment, before DVS splicing, where CRN runs [KAT_VAD_v2_ARCHITECTURE.md, §9]. Each spliced segment is steered by its own source's reference.
- **Not streamable**, the same as V2 (R2 needs the whole clip).

### 5.5 Why multiplicative, bias-free steering: the clip-referencing property

**The argument.** Fix one reference unit: a source video in training, or a clip at test. Fix one channel j, with factor `a_j = 1 + s·g_j ∈ [0, 2]`.

1. Without BatchNorm, `s` and `g` are deterministic functions of the unit's reference. All windows of a source share that reference, so `a_j` is constant over the unit, exactly, in training and at test.
2. CRN makes `median_t(h̃_{t,j}) = 0`.
3. Because `x ↦ a_j x` with `a_j ≥ 0` is monotone, `median_t(a_j h̃_{t,j}) = a_j · median_t(h̃_{t,j}) = 0`.
4. Nothing is added within the unit. So per channel, before `W_e`, the steered heads stay clip-referenced, exactly like V2's video stream before `W_u`.

**Additive steering would break this.** `h̃ + s·g` would have per-clip median `s·g(r)`: an offset computed from the reference that identifies the corpus, which is the [CTX] leak again.

**What the property does not cover:**
- **Scale.** A corpus-dependent *scale* is still possible, because DoTA references lie outside T2's range: the corpus is separable from the reference, source AUC ≈ 1.000 [KAT_VAD_PROPOSAL_v2.md, §1.2 G2]. The source-shortcut AUC on `V^t` is therefore measured, not assumed.
- **Position.** It does not make `a` position-invariant. On DF8 crops the reference changes with the crop, so `a` does too (B3). X7 measures this (§8.2).

### 5.6 Why SteerVAD's gate penalty is dropped

SteerVAD adds `0.1·mean(s²)` over normal samples [steervad_paper_summary.md, §3]. V4 drops it for two reasons:

- **It would switch the gate off everywhere.** T2's normal windows are cut from inside the accident videos [REPORT_V2_RESULTS.md, §1]. A normal window and an abnormal window from the same source therefore share one reference and one `s`, and the penalty would push `s` to zero for almost every source.
- **V2's rule forbids new losses.** In v1, an auxiliary loss took over the trunk (ρ = 3.1) [KAT_VAD_PROPOSAL_v2.md, §1.2 G4].

## 6. The V4 claim and how it may be worded

**Claim.** *Accident evidence in a frozen video encoder is concentrated in a few attention heads, which can be selected robustly from weak labels once each video's slow time trend is removed. Adding those heads to V2, and rescaling them by the clip's own normality reference, improves zero-shot detection on DoTA-CAP (n/1397) without increasing V2's position-shift cost.*

Each sentence is licensed by its own pre-registered test. A sentence whose test fails is reported as "not detectable", never as "no effect" [KAT_VAD_PROPOSAL_v2.md, §10.1].

| Sentence | Requires |
|---|---|
| "Expert heads selected on T2 add detection evidence to V2 on DoTA-CAP (n/1397)" | step 1 passes (§8.3) |
| "… beyond the T2 position prior (DoTA-CAP placement)" | step 1 passes, and the step-1 `f_2` paired t95 lower bound > 0 |
| "Clip-referenced steering adds evidence beyond the same heads unsteered" | step 2 passes |
| "The steering uses the clip's own reference" | step 2 passes, and the permutation-control lower bounds are > 0 (§8.3) |
| "Trend-robust selection beats random heads under the same steering" | step 3 passes |
| "V4 does not raise V2's position-shift cost" | X7 non-inferiority (required for adoption anyway) |
| "V4 lowers V2's position-shift cost" | ΔX7 lower bounds > 0 (§8.2) |
| "Steering keeps each channel clip-referenced before `W_e`" | the proof in §5.5, and the shortcut AUC on `V^t` not above A3-ign's |

**May not say:**
- "motion" or "kinematics-aware" (D6);
- "CRN improves transfer";
- any DoTA-CAP number beside LaGoVAD's 62.60 or full DoTA (D14);
- a raw gain without `p_T2` beside it;
- "beyond position" without "(DoTA-CAP placement)";
- "context matters" from A4 − A4-H alone;
- "V4 beats the position prior", unless both raw macro and AP exceed `p_T2` on the set read;
- a CAP-eval number without its look label;
- "position-invariant";
- "fixes non-ego dilution" (ReSH still pools over tokens);
- "tuning-free".

## 7. Experiment plan

### 7.1 Head selection: offline, T2-train only, detrended

1. **Data.** The 1,272 T2-train sources (T2-val's 219 are held out), all steps of each source, split into 5 disjoint folds of sources.
2. **Detrend.** For each source v and each head channel, fit a cubic in τ = t/T_v by least squares over all steps of v. Keep the residuals. The intercept also removes the clip-constant part, so for selection this includes what CRN does.
3. **Labels (weak, F4).** A window is abnormal iff `positive_frames > 0` (the T2-val freeze's rule). A step takes a label only if every T2-train window covering it carries the same label; steps covered by windows of both labels, and steps outside every window, are used in the detrend fit but not scored. Classes are balanced by subsampling, the same steps for all 72 heads.
4. **Score.** For each head and fold, compute HeadHunt's four measures on the residuals [headhunt-vad-summary.md, §3.2]: LDA separability, symmetrised Gaussian KL, MMD, and the NMI of K-means (K = 2) against the labels. Normalise each across the 72 heads within the fold, then average them to get `S(k, m)`.
5. **Rank.** `RSS(k) = mean_m S − 0.5·std_m S` (HeadHunt's rule, with "prompts" replaced by folds). Keep the top 5. Commit `head_ids`, and A4-rand's draw of 5 heads disjoint from them, before any probe is read.
6. **Diagnostics** (reported; they decide nothing):
   - the same ranking without detrending (the raw selection);
   - the raw and detrended top-5 sets: shared heads and Jaccard index;
   - Spearman ρ of RSS over the 72 heads;
   - each selected head's beyond-`p` probe gain on T2 (grouped CV);
   - top-5 overlap between folds.
7. **Stability floor (F5, gating).** The mean pairwise Jaccard of the five folds' top-5 sets must be ≥ 0.25. The chance level (two random 5-of-72 sets) is printed beside it.

**Why detrend.** A no-pixel position ruler already scores 0.730 on whole DADA sources (K-pos) [REPORT_V2_RESULTS.md, §3]. T2's abnormal windows sit late and include post-crash frames. CRN removes only the clip-constant part. A cubic over a whole source (~322 frames, small accident share [review]) fits the slow trend, including a long aftermath plateau, but fits little of a short accident bump. Detrending is for selection only. The model input stays CRN'd, as in V2, so A4-H and A3-ign compare like with like. Detrending the model input as well would be a second component (see §10).

### 7.2 Probe gates

Both gates reuse V2's probe harness with `p` printed beside every number [REPORT_V2_RESULTS.md, §3].

**Feature sets used by both gates:**
- `ẽ`: the selected heads (320-d).
- `ẽ_full`: all 6 heads of the block with the highest mean RSS (384-d).
- `ẽ_rand`: 20 random 5-head draws, read at their median.
- `ũ`: V2's pooled video feature.
- `p`: the position ruler.

**Labels and features for every probe (F4).** Frame labels from the annotation span at s8, whole source videos (K's convention); features CRN'd per source or clip, as the model sees them; `p` = the cubic position ruler of K-pos. The probe is the standardised, class-balanced logistic probe of `core.eda`.

**G1a (Stage A, T2 only, CPU).** Logistic probes trained on T2-train and read on T2-val macro (per-source AUC over two-class sources): `[ẽ; p]`, `[ẽ_full; p]`, `[ẽ_rand; p]`, `[ũ; p]`, and `p` alone.
- **Pass:** the stability floor holds (§7.1), and `[ẽ; p]` is above both `[ẽ_full; p]` and the median random draw (point estimates). Cluster-bootstrap intervals are printed. A margin smaller than the paired interval's half-width is flagged in the read-out.
- **Printed only:** the raw selection's `[ẽ_raw; p]`; the in-domain increment `[x̃; ũ; ẽ; p]` − `[x̃; ũ; p]`; each selected head's `[ẽ_k; p]` − `p`.

**G1b (Stage B, CAP-dev).** Probes trained on T2-train and read on CAP-dev macro.
- **Δ** = `[x̃; ũ; ẽ; p]` − `[x̃; ũ; p]`.
- **Printed beside it:** the same Δ for `ẽ_full` and for the median `ẽ_rand`, and `p` alone.
- **Pass:** Δ ≥ +0.01 with a cluster-bootstrap lower bound > 0. This is report-H3's kill threshold.

G1b is one go/no-go on the decision set, as report-H3's probe was. Nothing is selected on CAP-dev, so its optimism is limited to that one decision.

**G1c (Stage B, DF8 crops; early warning for X7, F1).** The two G1b probes **without `p`**, `[x̃; ũ; ẽ]` and `[x̃; ũ]` (fitted on T2-train), score DF8's 491 CAP-dev clips in the `control` and `early` crops, CRN taken over each crop's own rows as DF8 does. Probe X7 = macro(early) − macro(control); ΔX7_probe = X7(with heads) − X7(without), paired per clip, cluster bootstrap over DF8's 86 source videos.
- **Fail:** the cluster-bootstrap upper bound of ΔX7_probe is < 0 (adding the heads makes the probe's position cost worse, with the interval excluding 0). Stage C does not start.
- A probe is not the trained model, so a pass does not predict X7; it only removes the cheapest way to lose 10 runs.

**If G1 fails:** stop, and report the bounded negative. The optional alternative is the fallback **A4-S**: the same controller with one 384-d direction, steering V2's existing `c·ũ⊘σ_u` before `W_u`. It has 2,753 parameters, needs the re-bake but no extraction, and is tested as one contrast against A3-ign under §8.1. Whether those 5 runs go to A4-S or to SC-2 is the advisor's call.

### 7.3 Prior odds for G1b: a low-probability bet

- **`u` itself barely adds beyond position.** E2(d) found that `u` adds ≈ 0 beyond `p` in-domain on CAP-dev [review]. In transfer it adds only +0.018 (V2-S) and +0.037 (V2-B) beyond position [REPORT_V2_RESULTS.md, §8 report-H4].
- **Both earlier probes at this bar failed.** Text-guided windows scored −0.036; tile mean-pooling scored −0.012 [REPORT_V2_RESULTS.md, §8 report-H3].
- **The heads come from the encoder already in the baseline.** Its pooled output is already in `[x̃; ũ; p]`. The heads can only add what pooling and the final block dilute.

**We expect G1b to fail more often than not.** A failure costs two hooked passes and no training. It is still a reportable thesis result: HeadHunt's dilution finding does not transfer to a 6-head video encoder in this setting.

### 7.4 Stages and arms

| Stage | Work | GPU | Proceed only if |
|---|---|---|---|
| **A** | Hooked pass over T2 (all 72 heads, cosine gate F3); detrended and raw selection, with overlap and the stability floor; commit `head_selection.json`; G1a | one V2-S pass over T2 | G1a passes |
| **B** | Hooked pass over CAP-dev (all 72 heads) and DF8's 7,524 crop-boundary windows (F2); G1b; G1c | one pass over CAP-dev + DF8 windows | G1b and G1c pass; otherwise stop (or A4-S) |
| **C** | Preparation (CPU): re-bake with `r̂` columns; fit `σ_e`, `m_r`, `s_r`, PCA-32. Train A4-H and A4 (10 runs). Score CAP-dev and T2-val. DF8 for A4-H and A4. Permutation control for A4. Steps 1–2 | 10 runs + forward passes | step 2 passes (needed for Stage D only) |
| **D** | Train A4-rand (5 runs); step 3. Final read (§8.5) | 5 runs + one pass over CAP-eval | — |

| Arm | Added to `x̃ + W_u(·)` | New parameters | Stage | Role |
|---|---|---|---|---|
| A3-ign (V2) | — | 0 | reuse E4-0 (§5.3) | base |
| A4-H | `W_e h̃` (heads, no steering) | 164,352 | C | step 1 |
| **A4 (V4)** | `W_e (h̃ ⊙ a(r̂))` | 167,361 | C | step 2 |
| A4-rand | as A4, with the committed random heads | 167,361 | D | step 3 |
| A4-S (fallback only) | `W_u((c·ũ⊘σ_u) ⊙ a_u(r̂))` | 2,753 | instead of C–D | one contrast vs A3-ign |

All arms use the A3-ign recipe: AdamW, lr 5e-5, batch 64, 1,740 steps, seeds 2024–2028 paired, and no seeds added after a result is seen [KAT_VAD_PROPOSAL_v2.md, §8, §10.1].

### 7.5 Cost against SC-2

The full path is three hooked V2-S passes (T2, CAP-dev, and CAP-eval at the end) plus 15 runs. That is three times the runs of SC-2, which is built and was deferred for GPU cost [review]. The staging puts the cheap, likely-failing gates first:

- Stages A–B train nothing.
- Stage C's 10 runs come only after both gates pass.
- Stage D's 5 runs come only after step 2 passes.

Training runs by outcome:

| Outcome | Runs |
|---|---|
| G1 fails | 0 (or 5 for A4-S) |
| Step 1 or 2 fails | 10 |
| Everything passes | 15 |
| E4-0's base can't be reused (§5.3) | +5 in any case that trains |

## 8. Decision rules (pre-registered)

### 8.1 The bar and the primary metric

**Primary metric:** DoTA-CAP (n/1397) dev macro. `f_2` is always printed and always labelled "`f_2` (DoTA-CAP placement)". DF8 shows `f_2` falls to 0.45–0.48 on the early crops for every model [review], so it controls the T2 prior only under DoTA-CAP's own placement.

**The SC-2 bar** [review: `PREREG_ADDENDUM.md` §25], frozen and applied unchanged. A contrast passes only if all of these hold:

- (a) the paired t95 over the 5 seeds has a lower bound > 0;
- (b) the clip-level cluster-bootstrap lower bound is > 0;
- (c) the paired mean is ≥ +0.01;
- (d) the `f_2` point estimate is ≥ 0;
- (e) the candidate's T2-val macro is ≥ A3-ign − 0.01;
- (f) O1′ holds against A0.

### 8.2 X7 non-inferiority on DF8

**Definition.** ΔX7 = [macro(early) − macro(control)] of the arm, minus the same quantity for A3-ign. It is computed on DF8's 491 CAP-dev clips with the existing tool (`core/tools/position_shift.py`, 7,524 crop windows [review]).

**Adoption requires** the paired-t95 and cluster-bootstrap lower bounds of ΔX7 to be **> −0.01**. In words: V4 may not make V2's position cost (−0.125) worse by more than 0.01.

**Calibration (F1, measured on DF8's per-seed macros before any V4 number).** No adopted V2 step would have passed this bar:

| Contrast | ΔX7, paired t95 over 5 seeds |
|---|---|
| A3 − A0 | −0.041 [−0.067, −0.014] |
| A3-ign − A3 | −0.041 [−0.058, −0.025] |
| A3-ign − A0 | −0.082 [−0.093, −0.071] |

With paired half-widths of 0.012–0.026, A4 needs a ΔX7 point estimate of roughly ≥ +0.005 to pass. The bar is kept because the claim it licenses ("without increasing V2's position-shift cost") is the claim V4 makes. G1c (§7.2) is the cheap early warning; the most likely expensive failure of V4 is "steps 1–2 pass on CAP-dev, X7 fails".

**Printed beside it:**
- early − late;
- each crop set's macro, and `p_T2` on each crop set;
- for steered arms, the gate's drift across crops (mean `|s_early − s_control|` and mean `‖a_early − a_control‖`).

**Optional sentence.** If both lower bounds of ΔX7 are above 0, V4 may also say "V4 lowers V2's position-shift cost" (§6).

### 8.3 Fixed-sequence gatekeeping

**The three steps:**
1. **A4-H − A3-ign.** Bar (a)–(f), plus X7 non-inferiority for A4-H. If it fails, stop: V4 is a bounded null, and A4's numbers are reported as untested.
2. **A4 − A4-H.** Bar (a)–(d) on this contrast, plus (e), (f) and X7 non-inferiority for A4 against A3-ign. If it passes, **V4 = A4**. If it fails, **V4 = A4-H**.
3. **A4 − A4-rand** (Stage D). Bar (a)–(d). It licenses the selection sentence only and does not change adoption.

Each step is tested at the 5 % level only if every earlier step passed. The family-wise error therefore stays at 5 %, and there is no other route to adoption.

**Free control for context (replaces A4-static).** Score the trained A4 on CAP-dev with each clip's reference `r̂` replaced by another clip's: 10 random permutations per seed, averaged. Report own − permuted (paired t95 and cluster bootstrap), and A4 with the mean reference (`r̂ = 0`) as a descriptive static version. This decides nothing. It licenses "the steering uses the clip's own reference" (§6), which A4 − A4-H alone cannot.

### 8.4 Secondary read-outs and claim checks

These are descriptive, on CAP-dev, comparing A3-ign with the adopted arm on one harness.

**Secondary read-outs:**
- non-ego Δ (n = 248);
- share-bin macro, argmax-in-span rate and inside-span mean;
- early-onset Δ (n = 54).

**Claim checks:**
- the source-shortcut AUC on `V^t` and the position-probe R² on `V^t`, reported for every arm as in V2 and compared with A3-ign's; a rise in either is reported as a caveat on the gain [KAT_VAD_PROPOSAL_v2.md, §10.1];
- the distributions of `s` and `‖g‖` on T2 vs DoTA-CAP;
- `ρ_e`.

**Reporting.** T2-test is read once at the end and reported with V2's micro floor: A0 − 0.01 = 0.6128 [REPORT_V2_RESULTS.md, §4.4].

### 8.5 The final number and the stretch target

**The final read.** DF8 on CAP-dev is now part of the decision, and CAP-eval has been opened twice [REPORT_V2_RESULTS.md, §8]. The final V4 number is therefore one pre-registered read of the adopted arm, paired with A3-ign, on two sets:

- **CAP-eval**, labelled a **third look**;
- **DF8's eval counterpart**, labelled "a new cut of opened footage". This applies only if its crop list is built with the same tool and frozen before any V4 run, which report-H1's plan allows ("dev, then eval").

The hooked pass over CAP-eval is needed only for the adopted arm.

**Stretch target.** Raw macro and AP above `p_T2` without using `p`. The gaps to close, all derived:

| Split | Macro gap | AP gap |
|---|---|---|
| CAP-dev | 0.8220 − 0.7912 = 0.0308 | — |
| CAP-eval (third look) | 0.8256 − 0.8120 = 0.0136 | 0.607 − 0.596 = 0.011 |

`p_T2`'s lead comes from DoTA-CAP's placement convention [REPORT_V2_RESULTS.md, §6.2; review, DF8]. Crossing it is a benchmark milestone, not evidence of detection that ignores position. The position-robust milestone is "V4 lowers V2's position-shift cost" (ΔX7 > 0).

## 9. Risks and mitigations

| Risk | Why it is real | Mitigation |
|---|---|---|
| G1b fails | `u` adds ≈ 0–0.04 beyond position; two earlier probes failed this bar (§7.3) | staged; a failure costs no training and is reportable |
| Selection still picks time or aftermath heads | labels sit late in the source video; K-pos 0.730 | detrended selection; raw-vs-detrended overlap; X7 guardrail |
| The gate reads a position-dependent reference on DoTA (B3) | DF8 same-frame analysis [review]; reference OOD (source AUC ≈ 1.000) | X7 non-inferiority; gate drift logged; permutation control; A4-H fallback has no gate |
| Controller overfits | 1,272 distinct training contexts | PCA-32 and 3,009 parameters; factor bounded in [0, 2]; no BatchNorm |
| A ViT-S has few heads | dilution evidence comes from 784 LLM heads | G1a full-layer and random controls; G1b |
| Paper deltas are upper bounds | linear scorers, surveillance data | targets are thresholds, not forecasts |
| Seed pairing breaks | ReSH construction consumes RNG | `fork_rng`, plus three asserts (§5.3) |

## 10. Alternatives considered

| Candidate (source) | Why not V4 |
|---|---|
| ADSM port: feature-space score matching or scene-conditioned AdaLN (fit 1/5) | **Missing pieces:** the PSNR term, its biggest gain, has no feature-space equivalent; the latent version loses 5.7 / 4.9 macro [ADSM_paper_summary.md, §7.3]; DADA and DoTA have no camera IDs; dashcam motion breaks the fixed-scene assumption [ADSM_paper_summary.md, §8]. **Already tried:** a feature port is the normality branch V2 closed at −0.012. |
| Span-completion pseudo-labels (LAS-VAD ACC) | It is a new loss (the ρ = 3.1 lesson). The share-bin slope is "largely position" [REPORT_V2_RESULTS.md, §6.3]. Chaining can "swallow a 1–2 s crash" [LAS-VAD_summary.md, §10]. |
| Reviving `L_neg` with same-source hard negatives (proposal-H3) | It is a new loss. The text margin is at chance (0.504) [REPORT_V2_RESULTS.md, §8 report-H3]. |
| Non-local score refinement (LAVAD, VERA) | It is post-hoc and close to the killed causal hold (−0.071); the thesis claim would be weak. |
| Reverse-attention pooling (LAVIDA) | The evidence is weak, and spatial re-pooling already failed (tiles −0.012). |
| Order-sensitive difference features (report-H5) | It would be a motion claim, which V4 does not need. |
| Detrending the model input, not just the selection | It would change V2's input definition and add a second component. It is left as a possible follow-up if X7 shows position cost dominating. |

## 11. Implementation checklist

1. **Hook** (`core/models/videomae_heads.py`). `Attention.forward` uses fused SDPA, so `attn @ v` is never materialised [review: `core/models/videomae_v2.py:91`]. Register a `forward_pre_hook` on each block's `attn.proj`; its input is B × N × 384, channel = head·64 + d (`transpose(1, 2).reshape(B, N, C)`). Unit tests: `proj(tapped) ==` the attention output, per-head slices == a per-head SDPA, the pooled output is unchanged by the hooks.
2. **Stage A extraction** (`core/tools/extract_head_features.py`). All 72 heads, token-mean, every step of every T2 source, float32. In the same pass the pooled `u` is compared with the existing V2-S cache row by row; per-row cosine ≥ 0.999 (F3) is checked before selection.
3. **`select_heads.py`:** weak step labels (F4), detrend, 5 grouped folds, four measures, RSS, top-5, stability floor (F5). Also the raw selection and the overlap diagnostics, the best block, the A4-rand draw and G1a's 20 draws. **Commit `head_selection.json`.**
4. **Gate A** (`core/tools/v4_gate.py g1a`). Refuses to run unless the selection's sha256 is given. If G1a passes, do the Stage B extraction (CAP-dev, all 72 heads, DF8's crop-boundary windows) and run G1b and G1c.
5. **Stage C preparation.** Fit and freeze on T2-train: `σ_e`, `m_r`, `s_r`, PCA-32 (log its explained variance). Re-bake with `r̂` columns per step (F6) for T2 sources, DoTA clips and DF8 crops.
6. **Training code.** ReSH built in `fork_rng`. `r̂` arrives as per-step input columns (F6), so DVS splicing needs no change. Add the three seed asserts. Log `ρ_e`, `s` and `‖g‖`.
7. **Stage C:** train, score, run DF8 and the permutation control, then steps 1–2.
8. **Stage D:** train A4-rand, run step 3, then the final read.

```python
# Stages A/B: tap per-head outputs (fused SDPA never materialises attn @ v)
taps = {}
def tap(i):
    def pre_hook(module, args):                         # args[0]: B × N × 384, heads already merged
        taps[i] = args[0].unflatten(-1, (6, 64)).mean(1)    # B × 6 × 64, token mean as for u
    return pre_hook
handles = [blk.attn.proj.register_forward_pre_hook(tap(i))
           for i, blk in enumerate(videomae.blocks)]    # 12 blocks × 6 heads = 72 candidates


class ReSH(nn.Module):
    """V4's only new block: no BatchNorm, no new loss, one steering vector per segment."""
    def __init__(self, stats, K=5, d_h=64, ctx=32, rank=4, hid=32, D=512, steer=True):
        super().__init__()
        self.steer = steer
        for k in ("c", "sigma_e"):                      # frozen T2-train statistics
            self.register_buffer(k, stats[k])
        with torch.random.fork_rng(devices=[]):         # leave the trunk's RNG stream untouched
            torch.manual_seed(stats["resh_seed"])
            if steer:                                   # A4, A4-rand
                self.gate = nn.Sequential(nn.Linear(ctx, hid), nn.ReLU(), nn.Linear(hid, 1))
                self.down = nn.ModuleList(nn.Linear(ctx, rank, bias=False) for _ in range(K))
                self.up = nn.ModuleList(nn.Linear(rank, d_h, bias=False) for _ in range(K))
                for m in self.up:
                    nn.init.zeros_(m.weight)            # g = 0; up trains from step 1, down/gate from step 2
            self.W_e = nn.Linear(K * d_h, D)            # steer=False -> A4-H
            nn.init.zeros_(self.W_e.weight); nn.init.zeros_(self.W_e.bias)   # V4 == V2 at step 0

    def forward(self, e_crn, r_hat, seg):
        # e_crn: B×L×320 CRN'd heads; r_hat: S×32 PCA-whitened references, one row per segment;
        # seg: B×L index into r_hat (padded steps may point anywhere: the trunk masks them)
        h = self.c * e_crn / self.sigma_e
        if self.steer:
            s = torch.sigmoid(self.gate(r_hat))                                            # S×1
            g = torch.cat([torch.tanh(u(d(r_hat))) for d, u in zip(self.down, self.up)], -1)  # S×320
            h = h * (1 + s * g)[seg]                    # a in [0, 2], constant within a segment
        return self.W_e(h)          # H = x_crn + W_u(c*u_crn/sigma_u) + resh(e_crn, r_hat, seg)
```

Config flags (there is no new loss key and no gate penalty):

```
model.resh.enabled=true   model.resh.head_ids=[…committed…]   model.resh.steer=true|false
model.resh.ctx=crn_ref_xu_pca32   model.resh.rank=4   model.resh.gate_hidden=32
eval.resh.ctx_permutations=10     loss.dvs_anchor_mode=ignore
```

## Sources

**Project documents:**
- REPORT_V2_RESULTS.md (§0–§9)
- KAT_VAD_v2_ARCHITECTURE.md (§3–§5, §9–§13)
- KAT_VAD_PROPOSAL_v2.md (Review responses, §1.2, §4.3, §8–§10)
- steervad_paper_summary.md (§2–§5)
- headhunt-vad-summary.md (§2–§6)
- ADSM_paper_summary.md (§7–§8)
- LAS-VAD_summary.md (§10)

**From the 2026-10-15 review [review] (not in the project):**
- `RESULTS_E4_0.md` §1: A3-ign CAP-dev macro.
- `RESULTS_DF8.md`: X7, `f_2` on crops, aftermath negatives, the same-frame analysis.
- `PREREG_ADDENDUM.md` §25: the SC-2 bar.
- `RESULTS_E2D.md`: in-domain `u` beyond `p`.
- `core/constants.py` (`VIDEOMAE_ARCH`).
- `core/models/videomae_v2.py:91`.
- `build_v2_inputs`.
- `core/tools/position_shift.py`.
