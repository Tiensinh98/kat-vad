# KIM — build and run rules (LaGoVAD-KIM)

> Branch `v_final_las_vad`. Proposal: `core/docs/final/Thesis proposal … (LaGoVAD-KIM.md`.
> Plan: `.project/plans/katvad-kim.md`. **Status: RULE, written 2026-11-06 before any KIM number.
> Nothing is built; nothing has been run.** Every later change to this file is a dated note
> below the text it changes, never an edit of a rule after its number is seen.

## 0. Scope: what KIM may touch

**Unchanged (the VideoMAE path, as in the motion-stream model W0):** `core/models/videomae_v2.py`,
`core/models/motion_residual.py` (`W_u`, zero-init), `core/tools/build_v2_inputs.py`,
`core/data/v2_inputs.py`, the `v2.*` and `model.motion_*` config keys, and every cache. The motion
input stays the baked `c·ũ/σ_u` with Nexar statistics, no CRN. **No feature is extracted for KIM**:
it reads `vt` (the RoFormer output) at run time.

**New:** `core/kim/` (module + losses), wiring in `core/models/kat_vad.py`, `core/train.py`,
`core/config.py` (`KIMConfig`, `validate_kim`), `core/constants.py` (`KIM_*`), a pilot tool and a
read-out tool under `core/tools/`, tests `core/tests/test_kim.py`, one Colab notebook under
`colab/final/`. With `kim.enabled=false` (default) every output, logged key and state-dict key equals
today's code.

## 1. Notation: proposal symbol → code name

The proposal's symbols stay in the proposal. **No proposal symbol becomes a code identifier when the
code already uses that letter for something else.**

| Proposal | Means | Clashes with (code / records) | Code name |
|---|---|---|---|
| L\_kin, λ\_k | top-k MIL on KIM's q\_t | `losses["kin"]` = KIP v1 `kinematic_loss` (MIL on ŷ\_O + β·consistency), `loss.gamma_kin`, `kip.use_lkin` | `losses["kim_mil"]`, `kim.mil_weight` |
| L\_nkc, λ\_c, τ\_c = 0.1 | normal-anchored InfoNCE | `lambda_rec`/`lambda_align`/`tau_align` (KIP), `CONTRASTIVE_TEMP` (L\_neg); τ\_c is also the proposal's text embedding | `losses["kim_nkc"]`, `kim.nkc_weight`, `kim.nkc_temp` |
| k (count), k\_t (feature) | top-k count; 384-d KIM feature | k = top-k count | count = host `k`; feature `kin_feat` |
| p\_t ("position") | learned 128-d state | `p` cubic position probe, `p_T2` position prior | `state` |
| W\_p | 128×512 state projection | SV4 `w_p` / `stp.sv2_proj` / `sv2_wp_norm` | `state_proj`, log `kim_w_state_norm` |
| u\_t | velocity | `u` = VideoMAE vector (`ũ`, `σ_u`, `motion_share`) | `vel` |
| a\_t | acceleration | head blend `a` | `acc` |
| c\_t | CLIP CLS | `c` = motion scale in `c·ũ/σ_u` | host input |
| ρ\_V | per-video median change | `rho` (C37), `ρ_u` (WI), Spearman ρ | `kim_scale` |
| q\_t, r\_t | KIM score, logits | SA²-v2 top-3 `q`; CRN `r = d − f` | `kim_score`, `kim_logits` |
| β = 0.1, α = 0.5 | prototype momentum, anchor threshold | `loss.beta_cons`, β₁…β₄; `adaptive_fuse_alpha0` | `kim.proto_momentum`, `kim.anchor_thresh` |
| M = 16 | NKC negatives | v5 `M` | `kim.nkc_negatives` |
| m\_t | valid-difference mask | `mask` / `padding_mask` (1 = valid, per step) | `kim_step_mask` (per pair) |
| W\_o | 512×384 return projection | — | `w_out`, log `kim_w_out_norm` |
| h\_t, h̃\_t | RoFormer out, after return | `vt`, `vk` (KIP slot, `kat_vad.py:351`) | same slot |
| R0…R15, R-cap | proposal rows | STP arms R1/R4/R5/R6, v5 R1/R3/R5/R5d/R7 | arms **K0…K15, Kcap** (K*n* = proposal R*n*) |
| B0, B1, B2 | baseline rows | `B` = A3-ign; protocol B; E1 arm B | **KB0, KB1**; B2 = **W0** |
| Step 0, K1–K4 | pilot checks | v5 Stage 0; v2 kill-switch K / K-pos | **KP1–KP4** |

"KIM" is never shortened to "kinematic module": KIP's ego-kinematics claim was refuted and its +0.09 is
smoothing. No KIP number is KIM evidence.

## 2. Deviations from the proposal and the host (decided before any number)

| # | Item | Rule |
|---|---|---|
| D1 | Host | W0's config (v2 Nexar A2 recipe: stage 2, 1,749 steps, `score_head_kernel` 3, `mil_topk_pct` 5, `span`, no CRN, KIP off) |
| D2 | DVS | `dvs.theta = dvs.theta_ego = 1.0` in every K row: no splice, no filler, no seam. `dvs_sup`/`dvs_sup_mil` still run on **normal bags** (all-zero labels, exact; `train.py` `dvs_rows`), weights unchanged |
| D3 | Control | **K0 = W0 + D2**, retrained s2024. W0 printed beside, never the control |
| D4 | L\_neg | not run (Nexar has no captions), as every v_final row |
| D5 | Mask | `kim_step_mask[t] = valid[t]·valid[t−1]`, first step 0. No shot cuts (Nexar continuous; TAD test only) |
| D6 | Scale | `kim_scale` per bag over valid pairs, detached, + ε = `KIM_EPS` |
| D7 | kim\_mil | mean of the top-k **probabilities** `kim_score` over valid steps, k = `max(1, n_valid // loss.mil_topk_pct)`, then BCE (user U1; the host averages logits) |
| D8 | Prototypes | buffers; μ0 ← mean `kin_feat` over valid steps of label-0 bags; μ1 ← mean over 𝒜; no grad; init = first post-warm-up batch means |
| D9 | 𝒜 | per label-1 bag: top-k valid steps by `kim_score` with `kim_score > anchor_thresh` |
| D10 | NKC negatives | per anchor: top-M by cosine among valid steps of label-0 bags in the batch, + μ0; positive μ1 in the denominator. Empty 𝒜 → `kim_nkc = 0`, logged |
| D11 | Warm-up | `kim_nkc` and prototype updates off for steps < 219 (= round(5/40 · 1,749)); `kim_mil` from step 0 |
| D12 | Values | λ\_k 1, λ\_c 0.1, M 16, α 0.5, β 0.1, τ 0.1, gate kernel 3, width 128 — fixed, no search in this pass |
| D13 | Init | `w_out` = 0 (bias-free); `state_proj`, scorer, convs Xavier; gate biases 0; base LR, one AdamW group; **module built under a private RNG** so K0 and K-rows share the host init and data order (SV4 D13) |
| D14 | Test score | host score only; `kim_score`, ‖`acc`‖ saved per step as diagnostics |
| D15 | Exclusivity | `kim.enabled` with `kip.enabled`, `stp.sa2`, `stp.sa2v2` or `v5` grid readout → `ValueError` |

## 3. Arms

| Arm | = proposal | Config delta vs W0 | First pass |
|---|---|---|---|
| **K0** | R0 | D2 only | yes |
| **K2** | R2 | D2 + `kim.enabled`, no prototypes, `kim_nkc` off | yes |
| **K5** | R5 | D2 + full KIM (prototypes + NKC) | yes |
| Kcap | R-cap | K5 with `kin_feat` = MLP(LN(vt)), same width and losses | if K5 not a cost |
| K4 | R4 | K5 with LAS-VAD contrast (argmax classes, least-similar positive, other-class negatives) + score averaging | if K5 not a cost |
| K6, K7, K8 | R6, R7, R8 | order ladder; input `vt`/`x`/`clip`; no mask | if K5 not a cost |
| K1, K3, K9–K15 | R1, R3, R9–R15 | as proposal §4.5 | by budget |

Seed s2024 first; seeds 2025/2026 for K0/K2/K5/Kcap only after §5's continue verdict.

## 4. Inputs (existing; nothing re-extracted)

- Training: W0's Nexar `window` dataset dir and baked A2 cache (Nexar statistics), as `WU_ARMS.md`.
- Test (as `ZS_TAD_DADA.md` §7, `ZS_COMPARISON.md`): DoTA-CAP-dev 569, protocol B
  (`dota_cap_dev_s3_from_s1_nexar`); DADA-2000 370 T2-test sources, s8 whole video, raw
  (`A2_S/DADA2000_orig_test_sources_nexar`); TAD 100 (macro over 60), s8, raw (`A2_S/TAD_s8_nexar`).
  Nexar val (225) for diagnostics.
- Pilot checkpoint: W0 s2024 (`Thesis-Final/outputs/wu/W0/s2024`).
- Gates before any read: J10 (ckpt `global_step` == last `metrics.jsonl` step == 1,749); config diff
  K0 vs W0 = `dvs.theta`, `dvs.theta_ego` only; K-row vs K0 = the `kim` section only; input-cache identity
  checks as the zero-shot read-outs.

## 5. Pilot KP1–KP4 (inference only, W0 s2024)

**Disclosure:** these sets are KIM's endpoints. The pilot fits nothing; every KIM read-out carries
"the KP pilot read DoTA-CAP-dev / DADA / TAD with W0 before KIM was trained".

Candidates per step (no training): ‖Δz‖₁/ρ and ‖Δ²z‖₁/ρ for z ∈ {`vt`, `x` (RoFormer input), `clip`,
motion columns}, masked as D5, ρ as D6; each read with its set's own post-processing. Printed beside:
W0's score, the cubic position probe `p` and `t/N`.

| Check | Read | Pass | Consequence (fixed now) |
|---|---|---|---|
| KP1 | macro AUC of each candidate, per set, DoTA-CAP-dev also by ego flag | ‖Δ²`vt`‖ ≥ 0.60 on DoTA-CAP-dev ego-involved | Default input stays `vt` unless another z beats ‖Δ²`vt`‖ by ≥ 0.02 on **both** DoTA-CAP-dev and DADA, then that z is the K-rows' input (recorded as selected on endpoints) |
| KP2 | rank-average of W0's score with ‖Δ²`vt`‖ vs W0, paired cluster bootstrap | low > 0 on DoTA-CAP-dev | wording of H1–H2 only |
| KP3 | share of top-5 % ‖Δ`vt`‖ steps on the first or last valid step, vs chance 2/n | ≤ 2× chance on every set | fail → D5 also zeroes the last valid pair (`KIM_EDGE_MASK`), before training |
| KP4 | median \|peak(‖Δ²`vt`‖) − onset\| (s), DoTA-CAP-dev and DADA | ≤ 0.5 s on ≥ 50 % of clips | fail → H6 dropped from the claims |

## 6. Training read-out rule (first pass)

- **Endpoints:** DoTA-CAP-dev protocol-B macro and DADA-2000 macro. TAD macro secondary (printed, never
  decides). Micro, AP, TAD clip AUC, ego / category splits printed.
- **Contrasts:** K5 − K0, K2 − K0, K5 − K2; per-video paired, 95 % cluster bootstrap (as `w0d_readout`).
  Label: **gain** iff low > 0 and Δ ≥ +0.01; **cost** iff high < 0; else **no difference shown**. Always
  "PRELIMINARY (n = 1)".
- **Stop rule:** K5 − K0 **cost on both endpoints** → KIM line closed, no variants, no LR / λ sweeps.
  Otherwise continue to §3's second group. A gain at n = 1 adopts nothing until 3 seeds.
- **Mechanics (printed, never gate):** ‖`w_out`‖ per epoch from step 0, gate means (vel/acc), |𝒜| share,
  cos(μ0, μ1), `kim_mil`, `kim_nkc`, KIM's own video AUC on Nexar val, train `mil` last epoch vs K0's
  (overfit tell), W0 and K0 side by side (cost of D2 alone), `p` / `t/N` beside every macro.
- **Forbidden phrasing:** "KIM sees kinematics" without Kcap; "motion-gated"; any KIM number next to
  LaGoVAD's 62.60.

**Prediction (written before any number):** K0 ≤ W0 by a few points (D2 removes the host's frame
pseudo-labels); K5 − K0 and K2 − K0 "no difference shown" on both endpoints, as MQR / LSP / SA²; if
anything moves it is DADA (ego-heavy) before DoTA-CAP-dev.

## 7. Build checklist (K-1, CPU, data-free)

Tests in `core/tests/test_kim.py`: `kim.enabled=false` ⇒ outputs, loss keys and state-dict keys
identical to today; `w_out = 0` ⇒ K-row scores equal K0's at step 0 (same seed); mask zeroes pairs that
touch padding; `kim_scale` detached and ignores masked pairs; `kim_mil` hand-computed; `kim_nkc`
hand-computed on a 2-bag toy (positive in denominator); prototypes carry no grad and skip warm-up;
empty 𝒜 ⇒ 0; D15 exclusivity; checkpoint round-trip; `dvs.theta = 1` ⇒ no synthesized sample and
`dvs_sup` only on label-0 rows. Quality gate CLAUDE.md §11 on the touched files.
