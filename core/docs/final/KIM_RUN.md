# KIM — build and run rules (LaGoVAD-KIM)

> Branch `v_final_las_vad`. Proposal: `core/docs/final/Thesis proposal … (LaGoVAD-KIM.md`.
> Plan: `.project/plans/katvad-kim.md`. **Status: RULE, written 2026-11-06 before any KIM number.
> K-1 BUILT 2026-11-06 (`core/kim/`, wiring, `core/tests/test_kim.py`); pilot `core/tools/kim_pilot.py`,
> read-out `core/tools/kim_readout.py`, runbook `colab/final/kim_runs.ipynb` built the same day.** Every later change to this file is a dated note
> below the text it changes, never an edit of a rule after its number is seen.
>
> **Run state (2026-11-07):** K0, K0k9, pilot and K5 read at s2024 (§9). K0 and K5 read at s2025 (§10): **§6.2 reads "reproduced",
> n = 2, nothing adopted.** Kcap read at s2025 (§11): **§6.3 fails, K5's gain is not attributed to the differences.** Kcap read at s2024 (§11.6):
> Kcap − W0 shows no difference on any set, so §11.5 reads "anything else". The user chose K5d, KIM on the DVS-on host (§12). **K5d − W0 on DADA +0.0098, below the +0.01 margin: THE KIM LINE CLOSED 2026-11-07 (§12.1, §12.2).** K2 and s2026 not run. **The user chose K5 as the main model in place of W0 (§12.3): a choice, not a rule's outcome.**
>
> **Dated note 2026-11-07:** form probe §14 run on K0 only: **no form passes, the form is closed (§14.1).** The line stays closed.

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
| **K0k9** | — (host check, §3.1) | D2 + `model.score_head_kernel=9` | yes, **before** K2 / K5 |
| **K2** | R2 | D2 + `kim.enabled`, no prototypes, `kim_nkc` off | yes |
| **K5** | R5 | D2 + full KIM (prototypes + NKC) | yes |
| Kcap | R-cap | K5 with `kin_feat` = MLP(LN(vt)), same width and losses | if K5 not a cost |
| K4 | R4 | K5 with LAS-VAD contrast (argmax classes, least-similar positive, other-class negatives) + score averaging | if K5 not a cost |
| K6, K7, K8 | R6, R7, R8 | order ladder; input `vt`/`x`/`clip`; no mask | if K5 not a cost |
| K1, K3, K9–K15 | R1, R3, R9–R15 | as proposal §4.5 | by budget |

Seed s2024 first; seeds 2025/2026 for K0/K2/K5/Kcap only after §6's continue verdict.

### 3.1 K0k9 — host head kernel (added 2026-11-06, before any K number; user)

Question: does LaGoVAD's kernel 9 beat the inherited kernel 3 on the KIM host? Kernel 3 was derived for
20-step bags (`constants.DADA_ORIGIN_SCORE_HEAD_KERNEL`, C27: kernel 9 spans 45 % of a 20-step bag); no
9-vs-3 run exists on Nexar. Indirect evidence for 3: E1 (same kernel 3, DoTA footprint 2.4 s at s8 vs
0.9 s at s3 → s3 +0.033 macro); kernel 9 at s3 = 2.7 s. RoFormer (window 25 = ±12 per layer, 2 layers)
already sees the whole 20-step bag, so the kernel sets the curve's smoothing, not its context.

- Arm: K0 + `model.score_head_kernel=9`, s2024, same data, same endpoints, read with §6's labels.
- **Decision (fixed now):** K0k9 − K0 **gain on both endpoints** (DoTA-CAP-dev and DADA) → the host of
  every K row becomes kernel 9 (K0k9 is then the control; K2 / K5 run on it). Any other outcome → kernel 3
  stays. One read, no kernel sweep (5 / 7 not run).
- **Prediction:** no difference or cost on DoTA-CAP-dev (smoother curve, short accidents), no
  difference on DADA.
- Printed: train `mil` last epoch, `p` / `t/N`, TAD macro.

## 4. Inputs (existing; nothing re-extracted)

- Training: W0's Nexar `window` dataset dir and baked A2 cache (Nexar statistics), as `WU_ARMS.md`.
- Test (as `ZS_TAD_DADA.md` §7, `ZS_COMPARISON.md`): DoTA-CAP-dev 569, protocol B
  (`dota_cap_dev_s3_from_s1_nexar`); DADA-2000 370 T2-test sources, s8 whole video, raw
  (`A2_S/DADA2000_orig_test_sources_nexar`); TAD 100 (macro over 60), s8, raw (`A2_S/TAD_s8_nexar`).
  Nexar val (225) for diagnostics.
- Pilot checkpoint: W0 s2024 (`Thesis-Final/outputs/wu/W0/s2024`). *Dated note 2026-11-06 (before any number):*
  if that checkpoint is gone from Drive, v2 Nexar A2 s2024 (`Thesis-V2/outputs/v2_nexar_window/A2/s2024`) is
  used instead — the same function (`ZS_TAD_DADA.md` §2: equal to the last digit). A W0 read missing from
  Drive is re-read from that checkpoint with the same protocol into `outputs/kim/eval/<set>/W0`.
- Gates before any read: J10 (ckpt `global_step` == last `metrics.jsonl` step == 1,749); config diff
  K0 vs W0 = `dvs.theta`, `dvs.theta_ego` only; K-row vs K0 = the `kim` section only; input-cache identity
  checks as the zero-shot read-outs.

*Dated note 2026-11-06, later (before any KIM number; user): both checkpoints above were deleted from Drive
and are gone, so the note above no longer applies.* W0 survives as its **saved reads** of the three sets and
its **`config.yaml`** (repo copies under `outputs/v_final/`). They keep their roles: W0 stays a printed row
(D3) and the config diff stays against W0's config. A W0 read missing from Drive can no longer be re-read; it
is uploaded from the repo. J10 on W0 is not re-checked here; its reads passed it when they were made
(`WU_ARMS.md`, `ZS_TAD_DADA.md`). W0 cannot be read on any new set without retraining s2024. No contrast that
decides anything uses W0.

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

### 5.1 Pilot on the decided control (dated note 2026-11-06, before any KIM number; user)

W0's checkpoint is gone (§4), and every KP check needs `vt`. Options weighed: retrain W0 for the pilot alone
(a GPU run for a row that is only printed), drop the pilot (KP3 could still change the K2 / K5 build), or read
it on the host that is trained anyway. **Chosen: the third.** The text of §5 above stands, with these changes:

- **Host:** the control decided by §3.1 (K0, or K0k9 if it wins), s2024, wherever §5 says W0. Its `vt` comes from
  the DVS-off host that K2 / K5 actually sit on.
- **Order:** K0, K0k9 → their reads → read-out 1 (kernel) → **pilot** → K2, K5. KP1 and KP3 only change the
  K2 / K5 build, so no finished arm can be invalidated.
- **KP2** compares the rank average with the control's score, not W0's.
- **Disclosure** (`KIM_PILOT_DISCLOSURE`): the pilot reads the endpoints with the control, after read-out 1 and
  before K2 / K5 train. It adds no new look at the control: read-out 1 has already printed its macros.
- **Limit:** the numbers describe the DVS-off host. Never quote them as W0's or as "the motion-stream host".
- All bars and consequences in §5's table are unchanged. Not built yet: the KP1 input switch and
  `KIM_EDGE_MASK`; if either fires, the notebook stops before K2 / K5 and the build comes first.

### 5.2 After KP3 failed (dated note 2026-11-07; user). Written after the first pilot, before any K2 / K5 number

KP3 failed on the first pilot (K0). Its share pools the first and the last difference, while §5's mask
reaches only the last, so the mask could miss the excess. Nothing in §5's table changes.

- **Mask:** §5's `KIM_EDGE_MASK` is the config flag `kim.edge_mask`, set in K2 and K5 (never K0 / K0k9, which
  have no KIM). It zeroes each bag's last valid pair, in training and at test.
- **Split (printed):** the same pilot on K0 again, into `REPORTS/pilot_edges/`, prints the first and last shares
  separately. Its verdicts must equal the first pilot's.
- **Stop, fixed now:** if the first pair's share exceeds the last's on DoTA-CAP-dev or DADA (`last_mask_fits`
  false; TAD never decides), the last-pair mask does not fit the failure: K2 / K5 are not trained and the user
  decides. No other edge mask is built without a new dated note.
- **Position (printed, never decides):** the pilot now prints the middle ruler, computed per set, and the two
  recorded DoTA-CAP-dev cubic priors (`p_T2` 0.8220, `p` 0.852). No cubic `p` is fitted on DADA / TAD.
- **Training-free baseline (printed, never decides):** `kim_readout --pilot` prints the host + `vt2` rank average
  beside the arms. A K2 / K5 gain of that size is not evidence for the learned module. §6's labels and stop rule
  do not change.

### 5.3 The mask moves to the first pair (dated note 2026-11-07, before any K2 / K5 number; user)

**Why, and what was seen:** §5.2's split stopped the notebook. Share of the top-5 % `vt1` steps, first / last pair
(chance per edge): DoTA-CAP-dev 0.119 / 0.011 (0.033), DADA 0.141 / 0.016 (0.027), TAD 0.161 / 0.026 (0.036). The
excess is on the first pair; the last pair is below chance. **This changes §5's consequence after KP3's number was
seen.** No KIM score had been seen. The cause is not measured (`vt` only).

- `kim.edge_mask` now zeroes each bag's **first** valid pair, in training and at test, replacing §5's last-pair mask
  and §5.2's stop. K5 trains with it. The last pair stays.
- If a re-read no longer has the first pair above the last on both endpoints (`first_mask_fits`), step 9 is refused
  and the user decides.
- §6.1's stop rule and prediction-free reading are unchanged. KP4's H6 drop stands.

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

### 6.1 K5 only, stricter stop (dated note 2026-11-07, before any K2 / K5 number; user)

**Prior evidence this rule should have cited:** `core/docs/v2.2/RESULTS_V22_ORDER.md` (2026-10-13). S3(a) on
explicit velocity / acceleration features: **−0.0007 [−0.0063, +0.0047]**, so TDP (IAM on the CLIP vector) was
not built. S1a already showed the change score (`m5` 0.750 on DoTA-CAP-dev, ego 0.785, non-ego 0.706), and S1c
bias at a clip's first steps. KIM differs (end-to-end, on `vt`, Nexar bag labels), so this lowers the prior and
does not refute it. "TDP" stays forbidden as a contribution.

- **First pass = K5 only.** K2 runs only if K5 passes below. §3's K2 row and the K2 contrasts wait until then.
- **To continue (K2, more seeds, ladder), K5 needs both:** a **gain** against the control on DoTA-CAP-dev **and**
  DADA, and a macro above the training-free rank average (`kim_readout --pilot`, §5.2) on both.
- **Anything else closes the KIM line**: no variants, no LR / λ sweep. This is stricter than §6's stop rule. The
  read-out's `stop_rule` field still prints §6's verdict, so this rule is read from its numbers.
- A first-pair stop from §5.2 means no K5; the user then decides between a dated note and closing.

**Prediction (written before any number):** K0 ≤ W0 by a few points (D2 removes the host's frame
pseudo-labels); K5 − K0 and K2 − K0 "no difference shown" on both endpoints, as MQR / LSP / SA²; if
anything moves it is DADA (ego-heavy) before DoTA-CAP-dev.

### 6.2 Seed 2025 (dated note 2026-11-07, before any s2025 number; user)

K5 passed §6.1 at s2024 (n = 1), and a gain at n = 1 adopts nothing (§6).

- **Runs:** K0 and K5, s2025, code and recipe of s2024: host kernel 3, K5 with `kim.edge_mask=true`. Runbook
  `colab/final/kim_seed.ipynb`, reads under `outputs/kim/eval_s2025/`. No pilot on this seed.
- **Read:** K5 − K0 per seed, §6's labels, paired within the seed. No pooling across seeds in this note.
- **Decision, fixed now:** K5 − K0 **gain on DADA** at s2025 → K5 counts as reproduced at n = 2 and the line goes on
  (Kcap, K2, s2026). Anything else → "not reproduced": no Kcap or K2 run, no variants. DoTA-CAP-dev is read beside it;
  a DoTA-only gain does not count, because DoTA sits below its own position ruler (middle ruler 0.8516).
- **K5 − W0 (printed, never decides):** paired per video by `kim_readout`, s2024 only, since W0 has no other seed. Its
  label is a description, not a rule. W0 keeps DVS on, so the contrast mixes DVS off with KIM.
- **Prediction (written before the number):** positive on DADA, smaller than s2024's +0.0484.

### 6.3 Kcap (dated note 2026-11-07, before any Kcap number; user)

Kcap asks whether K5's gain needs the differences, or only a branch of this capacity with these losses (H2).

- **Arm:** `kim.capacity_control=true`, `kim.prototypes=true`, `kim.edge_mask=false`. `kin_feat = MLP(LN(vt))`, per
  step: `Linear(D, 183)`, ReLU, `Linear(183, 3w)`, 164,535 parameters against 164,224 in `W_state` + both gates. Same
  scorer, `W_out`, prototypes, `kim_mil`, `kim_nkc`, warm-up. The mask has nothing to act on, so the config refuses it.
- **What differs from K5, beyond the differences:** K5's gates see 3 steps; Kcap sees one. Its `vt` still carries
  RoFormer's temporal context, so Kcap tests **explicit** differences, not temporal information.
- **When:** built now; it runs only after §6.2 reads "reproduced". Same seeds as K5 (s2024, s2025), runbook
  `kim_seed.ipynb` with `ARMS = (K0, K5, KCAP)`, read per seed.
- **Decision, fixed now:** K5 − Kcap **gain on DADA at both seeds** → the differences contribute, and "the gain needs
  the explicit differences" may be written with its n. Anything else → the gain is not attributed to the differences;
  §6's forbidden phrasing stays and any K5 gain is reported as branch + losses. DoTA-CAP-dev is read beside it, as in §6.2.
- **Printed, never decides:** Kcap − K0. Kcap never enters `stop_rule`.
- **Prediction (written before the number):** K5 − Kcap "no difference shown", as v2.2 S3.

## 7. Build checklist (K-1, CPU, data-free)

Tests in `core/tests/test_kim.py`: `kim.enabled=false` ⇒ outputs, loss keys and state-dict keys
identical to today; `w_out = 0` ⇒ K-row scores equal K0's at step 0 (same seed); mask zeroes pairs that
touch padding; `kim_scale` detached and ignores masked pairs; `kim_mil` hand-computed; `kim_nkc`
hand-computed on a 2-bag toy (positive in denominator); prototypes carry no grad and skip warm-up;
empty 𝒜 ⇒ 0; D15 exclusivity; checkpoint round-trip; `dvs.theta = 1` ⇒ no synthesized sample and
`dvs_sup` only on label-0 rows. Quality gate CLAUDE.md §11 on the touched files.

## 8. Tools and runbook (built 2026-11-06, before any number)

- **Pilot** `python -m core.tools.kim_pilot`: loads W0 (J10), reads labels and W0's scores from W0's own
  reads of each set (`clip_scores.npz` / `scores/`), feature rows from the bakes those reads used; writes
  `kim_pilot.{json,md}` with §5's decisions (`input`, KP1–KP4). Candidates: `vt`, `x`, `clip`, `motion` ×
  order 1 / 2; KP2 = rank-average of W0's score with `vt2`; KP3 on `vt1` step curves; KP4 on `vt2`
  (DoTA 0.1 s per native frame, DADA 8/30 s per row; TAD has no known step, so no KP4).
- **Read-out** `python -m core.tools.kim_readout`: rows by `--dota/--dada/--tad NAME DIR`; contrasts
  control − W0 (printed), K0k9 / K2 / K5 − control, K5 − K2; decisions `kernel` (§3.1, only with control
  K0) and `stop_rule` (§6). TAD never decides.
- **Runbook** `colab/final/kim_runs.ipynb`: pilot → K0, K0k9 → read-out 1 (kernel) → K2, K5 on the
  decided host (refused if the pilot changed the build) → read-out 2 (stop rule). Config diff checked
  against W0's `config.yaml`; outputs under `Thesis-Final/outputs/kim/`.

*Dated note 2026-11-06, later (§4, §5.1):* the pilot now takes the required `--name <host>`, records it as
`host` in `kim_pilot.json`, and labels its row `host`. Runbook order: W0's saved reads checked → K0, K0k9 → reads
→ read-out 1 (kernel) → pilot on the control → K2, K5 → read-out 2. A pilot of another host already on Drive
stops the notebook. Nothing in the runbook loads W0's weights.

*Dated note 2026-11-07, after §9, before any s2025 number:* §6.2's text is unchanged, the tool now applies it. At the
replicate seed `kim_readout` prints `replication` and no `stop_rule`, because §6's verdict could read "continue" beside
"not reproduced". It also prints K5 − K0k9 (never decides). `kim_seed.ipynb` can now train K2, with the mask (§5.2).

## 9. Read-out, seed 2024 — PRELIMINARY (n = 1), nothing adopted (2026-11-07)

Raw: `outputs/v_final/kim/REPORTS/`. `kim_readout.md` is the Colab read. `kim_readout_s2024.md` is the same inputs re-read
locally with the current tool: apart from the rule citation every shared line is identical, and it adds K5 − W0 and K5 − K0k9.

### 9.1 Macros and contrasts

| Set | W0 | K0 | K0k9 | K5 | Rank avg, no training | Middle ruler |
|---|---|---|---|---|---|---|
| DoTA-CAP-dev (protocol B) | 0.8110 | 0.8002 | 0.8187 | **0.8195** | 0.8139 | 0.8516 |
| DADA-2000 (370 sources) | 0.7615 | 0.7256 | 0.7233 | **0.7740** | 0.7428 | 0.7342 |
| TAD (printed, never decides) | 0.6983 | 0.6954 | 0.7163 | 0.7002 | 0.7244 | 0.6831 |

| Contrast (paired per video, 95 % CI) | DoTA-CAP-dev | DADA | TAD | Role |
|---|---|---|---|---|
| K5 − K0 | **+0.0193 [+0.0091, +0.0292]** gain | **+0.0484 [+0.0367, +0.0599]** gain | +0.0047 [−0.0247, +0.0340] | decides (§6, §6.1) |
| K0k9 − K0 | +0.0185 [+0.0108, +0.0268] gain | −0.0022 [−0.0144, +0.0100] | +0.0208 [−0.0094, +0.0538] | decides (§3.1) |
| K0 − W0 | −0.0108 [−0.0200, −0.0016] cost | −0.0359 [−0.0490, −0.0232] cost | −0.0029 [−0.0388, +0.0331] | printed |
| K5 − W0 | +0.0085 [−0.0023, +0.0190] | +0.0125 [−0.0006, +0.0253] | +0.0019 [−0.0244, +0.0286] | printed (§6.2) |
| K5 − K0k9 | +0.0008 [−0.0116, +0.0130] | +0.0507 [+0.0335, +0.0678] | −0.0161 [−0.0555, +0.0229] | post-hoc, printed |

DoTA-CAP-dev by ego flag: K5 − K0 ego +0.0176 [+0.0071, +0.0275], non-ego +0.0215 [+0.0045, +0.0384]; K5 − W0 ego −0.0012
[−0.0126, +0.0096], non-ego +0.0211 [+0.0040, +0.0380]. Recorded priors on CAP-dev: `p_T2` 0.8220, cubic `p` 0.8520.

### 9.2 Decisions

- **§3.1:** kernel 3 stays, control = K0. DoTA gains, DADA does not, so the both-endpoints rule fails. The prediction was half
  wrong: DoTA gained where none was predicted; DADA matched.
- **§5 (pilot on K0):** KP1 pass (`vt2` ego 0.798), input stays `vt`. KP2 pass (+0.0137 DoTA, +0.0172 DADA). KP3 fail, mask moved
  to the first pair (§5.3). KP4 fail, H6 dropped.
- **§6.1:** K5 **passes**: gain on both endpoints, and above the rank average on both (DoTA by +0.0056, DADA by +0.0312; these
  margins are unpaired, and DoTA's CIs overlap). The prediction "no difference shown" was wrong.
- **Next by rule:** §6.2, s2025.

### 9.3 What these numbers do not license

1. **"KIM beats the motion-stream host": no.** K5 − W0 shows no difference on any set. K5 − K0 is measured against a control that
   D2 lowered: K5 − K0 = (K5 − W0) − (K0 − W0) by construction (DADA +0.0125 + 0.0359 = +0.0484). How much of K5's gain only
   refills what DVS off removed is not separable here: no row has DVS on with KIM.
2. **DoTA-CAP-dev says little about the module.** Host kernel 9 alone gives the same macro as K5 (+0.0008), and K5 stays under its
   middle ruler and `p_T2`. On DADA the two differ by +0.0507. This contrast was chosen after K5's number, so it is description.
3. **"Sees kinematics": still forbidden (§6).** Kcap has not run. K5's `kin_feat` also holds the undifferenced `state`.
4. **H1's ego-larger prediction has no support:** ego +0.0176 against non-ego +0.0215.
5. **Regularization is not excluded:** `mil` last-epoch mean K5 0.1637, K0 0.1288, K0k9 0.1163.

### 9.4 Checks and mechanics

- **Runs:** each `metrics.jsonl` has 1,749 rows. `verified.json` records step 1,749; the checkpoints themselves stay on Drive.
  Config diffs: K0k9 − K0 = kernel only; K5 − K0 = `kim.enabled`, `kim.prototypes`, `kim.edge_mask`; K0 − W0 = `dvs.theta*` plus
  keys W0's config predates. First-batch `mil` K0 = K5 = 0.6878 (same host init, D13).
- **Which pair was masked:** the config says `edge_mask: true`, not which pair. First pair rests on the upload: `pair_mask(drop_first)`
  and `first_mask_fits` were written together, and `kim_runs.ipynb` imports the latter. `pilot_edges/kim_pilot.*` still prints "mask the
  last valid pair" because it was written by the older tool.
- **Replication uses K5's code:** `KIM` before and after the Kcap refactor (`9e81b77` vs `c69f867`) gave equal keys, init, `kin_feat` and
  score under one seed, for K5, K2 and the unmasked form.
- **K5 diagnostics, first → last logged batch (one batch, not an average):** `kim_mil` 0.8593 → 0.1262, `kim_nkc` 2.1806 → 0.2771,
  ‖`w_out`‖ 0 → 0.9335, velocity gate 0.5044 → 0.3638, acceleration gate 0.5034 → 0.4993, anchor share 0.0076 → 0.9038, cos(μ0, μ1)
  0 → −0.7319. The mean acceleration gate barely left its init; a mean cannot say whether single channels moved.

### 9.5 Required by this rule, not built

Printed-only items, so no decision above depends on them: D14's per-step `kim_score` and ‖`acc`‖, §6's KIM video AUC on Nexar val,
and §5's cubic `p` fitted on DADA / TAD (§5.2 already waived that one). The first two can be read later from the checkpoints on Drive.

## 10. Read-out, seed 2025 — §6.2 "reproduced", n = 2, nothing adopted (2026-11-07)

Raw: `outputs/v_final/kim/REPORTS/kim_readout_s2025.md`, the only s2025 file local so far. Its json, per-clip reads and run folders
stay on Drive (`outputs/kim/eval_s2025/`, `outputs/kim/{K0,K5}/s2025/`). Per-seed, no pooling (§6.2).

### 10.1 Decision

**K5 − K0 on DADA at s2025 = +0.0383 [+0.0293, +0.0473], gain → "reproduced at n = 2".** The line goes on: Kcap, K2, s2026. The
prediction (positive, smaller than s2024's +0.0484) held; the two intervals overlap, so "smaller" is not shown. A gain adopts nothing
until 3 seeds (§6).

### 10.2 Numbers by seed

| Set | Row or contrast | s2024 | s2025 |
|---|---|---|---|
| DADA (decides) | K0 macro | 0.7256 | 0.7248 |
| | K5 macro | 0.7740 | 0.7631 |
| | K5 − K0 | **+0.0484 [+0.0367, +0.0599]** gain | **+0.0383 [+0.0293, +0.0473]** gain |
| DoTA-CAP-dev (read beside) | K0 macro | 0.8002 | 0.8002 |
| | K5 macro | 0.8195 | 0.8166 |
| | K5 − K0 | +0.0193 [+0.0091, +0.0292] gain | +0.0163 [+0.0102, +0.0223] gain |
| | ego / non-ego | +0.0176 / +0.0215 | +0.0159 [+0.0087, +0.0232] / +0.0169 [+0.0069, +0.0272] |
| TAD (never decides) | K5 − K0 | +0.0047 [−0.0247, +0.0340] | −0.0082 [−0.0393, +0.0193] |

Rulers, same at both seeds: DADA middle 0.7342; DoTA-CAP-dev middle 0.8516 and `p_T2` 0.8220.

### 10.3 What this still does not license

1. **"KIM beats the DVS-on host": still not shown.** K5 − W0 exists at s2024 only and shows no difference (§9.3).
2. **"The gain needs the differences" or "sees kinematics": forbidden until §6.3 reads.** Reproduced means K5 > K0 twice, not why.
3. **Regularization is not excluded.** `mil` last-epoch mean K5 > K0 at both seeds (s2025: 0.1454 vs 0.1181).
4. **H1's ego-larger prediction has no support at either seed.**
5. **DoTA-CAP-dev:** K5 stays below its middle ruler and `p_T2` at both seeds. No K0k9 at s2025, so §9.3's point 2 is not re-read.
6. **§6.1's rank-average condition was not re-read:** no pilot on this seed (§6.2), and §6.2 does not ask for it.

### 10.4 Checks and mechanics

- **K0's DoTA macro matches s2024 to four digits (0.8002), so a stale read was checked:** `protocol_b_eval` uses one fixed bootstrap seed
  and `kim_seed.ipynb` does not override it. K0's s2024 read prints [0.7848, 0.8155] and s2025 prints [0.7847, 0.8157], so the scores differ.
  Eval dirs and run dirs carry the seed. Exact proof needs `eval_s2025/dota/K0/protocol_b_readout.json` (`runs`, `checkpoints`). DADA, TAD and the
  training curves differ as well.
- **Same host init:** first-batch `mil` K0 = K5 = 0.7057 (s2024: 0.6878). Both runs reach step 1,749.
- **K5 diagnostics, first → last logged batch (one batch, not an average):** `kim_mil` 0.7584 → 0.0719, `kim_nkc` 2.0888 → 0.2179, ‖`w_out`‖ 0 →
  0.9082, velocity gate 0.5078 → 0.3442, acceleration gate 0.5044 → 0.5015, anchor share 0.2016 → 0.9667, cos(μ0, μ1) 0 → −0.7829. As at s2024, the mean
  acceleration gate barely left its init.

*Dated note 2026-11-07, after this read-out. §6.2's text stays; this corrects its premise.* §6.2 says W0 has no other seed. **The W0 recipe does
have one: L0b** (`outputs/v_final/local/L0b/s2025/`, `LOCAL_STEP1B.md`). Its config differs from W0's only in `train.seed` and `stp.branch: host`. Its
DoTA-CAP-dev macro is 0.8058, so K5 s2025 − L0b is +0.0108, **unpaired and printed only**; a paired read needs K5's s2025 `clip_aucs.json`. L0b has no DADA
or TAD read locally. Whether its checkpoint is still on Drive is unverified. It was trained on `v_final-local`. No decision changes.

## 11. Read-out, Kcap at seed 2025 — §6.3 fails, n = 1 for Kcap (2026-11-07)

Raw: `outputs/v_final/kim/REPORTS/kim_readout_s2025.md`, re-written with the Kcap row; K0 and K5's lines match §10's. Only the `.md` is local.

### 11.1 Decision

**K5 − Kcap on DADA at s2025 = +0.0012 [−0.0056, +0.0080], no difference shown.** §6.3 needs a gain at both seeds, so s2024 cannot reverse this:
**K5's gain is not attributed to the differences. It is reported as branch + losses, and §6's forbidden phrasing stays.** The prediction "no difference
shown" held on DADA; on DoTA-CAP-dev, which is read beside it, the differences show a cost.

### 11.2 Numbers

| Macro, s2025 | K0 | K5 | Kcap |
|---|---|---|---|
| DADA (decides) | 0.7248 | 0.7631 | 0.7620 |
| DoTA-CAP-dev (read beside) | 0.8002 | 0.8166 | 0.8229 |
| TAD (never decides) | 0.6918 | 0.6836 | 0.6894 |

| Contrast, s2025 | DADA | DoTA-CAP-dev | TAD |
|---|---|---|---|
| K5 − Kcap (§6.3) | **+0.0012 [−0.0056, +0.0080]** none | −0.0063 [−0.0108, −0.0020] cost | −0.0058 [−0.0289, +0.0198] none |
| Kcap − K0 (printed, never decides) | +0.0371 [+0.0277, +0.0466] gain | +0.0226 [+0.0167, +0.0287] gain | −0.0024 [−0.0285, +0.0216] none |
| K5 − K0 (§10) | +0.0383 [+0.0293, +0.0473] gain | +0.0163 [+0.0102, +0.0223] gain | −0.0082 [−0.0393, +0.0193] none |

DoTA-CAP-dev by ego flag: K5 − Kcap ego −0.0038 [−0.0076, −0.0000], non-ego −0.0096 [−0.0179, −0.0020]; Kcap − K0 ego +0.0197 [+0.0133, +0.0261], non-ego
+0.0265 [+0.0155, +0.0382]. So a branch without differences holds K5's whole gain on DADA and more on DoTA-CAP-dev, at this seed.

### 11.3 What this does and does not license

1. **Licensed, n = 1:** at s2025, explicit differences add nothing measurable beyond a per-step branch of equal capacity with the same losses, on DADA.
2. **Not licensed: "motion does not matter."** Kcap's `vt` still carries RoFormer's temporal context and the VideoMAE motion stream (§6.3). This tests **explicit** differences only.
3. **Not licensed: "Kcap beats K5."** Kcap's DoTA margin is one seed, printed; it never decides.
4. **Not licensed: "the branch beats the DVS-on host."** Kcap − K0 is against the control DVS off lowered (§9.3). No paired Kcap − W0 exists. Unpaired and printed: Kcap s2025 − L0b on
   DoTA-CAP-dev is +0.0171; L0b has no DADA read locally. Kcap stays below DoTA-CAP-dev's middle ruler 0.8516, and at `p_T2` 0.8220 inside its CI.
5. **Why the branch helps is not measured.** Candidates: a second head on `vt`, its own `kim_mil`, `kim_nkc`, prototypes. Train `mil` last-epoch mean Kcap 0.0891, K0 0.1181, K5 0.1454: Kcap gains while
   fitting harder, so higher train `mil` does not explain both arms. One seed; this does not exclude other regularization.

### 11.4 Checks and mechanics

- **Same host init:** first-batch `mil` K0 = K5 = Kcap = 0.7057. Kcap reaches step 1,749. Parts add up: K5 − K0 = (K5 − Kcap) + (Kcap − K0) on DADA and DoTA-CAP-dev.
- **Kcap diagnostics, first → last logged batch (one batch):** `kim_mil` 1.1566 → 0.0413, `kim_nkc` 1.9995 → 0.1263, ‖`w_out`‖ 0 → 1.4680, anchor share 0 → 0.9750, cos(μ0, μ1) 0 → −0.5103. Gates print 0: Kcap has none.
- **Tool gap:** the read-out prints the K5 − Kcap row but no §6.3 decision line, and has no Kcap − W0 row. §6.3 is read here from the numbers.

### 11.5 Kcap s2024 and Kcap − W0 (dated note 2026-11-07, before any Kcap s2024 number; user)

§11.1 stands and this run cannot reverse it. The run asks what §11.3 point 4 leaves open: does branch + losses beat the DVS-on host?

- **Run:** Kcap, s2024, §6.3's arm and recipe, runbook `kim_seed.ipynb` (`SEED = 2024`, `ARMS = (K0, K5, KCAP)`). K0 and K5 are not retrained or re-read.
- **Read:** `kim_readout` now prints **Kcap − W0**, paired per video, at s2024 only because W0's reads exist at that seed. It enters no `decisions` field. W0 keeps DVS on, so the contrast mixes DVS off with branch + losses, as K5 − W0 does.
- **What follows, fixed now, on DADA as in §6.2 and §6.3, with DoTA-CAP-dev read beside it:**
  Kcap − W0 **gain on DADA** → the user may open a new dated note on branch + losses, not on kinematics. **Anything else** → Claude recommends closing the KIM line and writing up the attribution; K2 and s2026 stay unrun unless the user decides otherwise. The user makes that call. n = 1 either way, nothing adopted.
- **Prediction (written before the number):** DADA "no difference shown", as K5 − W0 at s2024; DoTA-CAP-dev positive, label not predicted. K5 − Kcap on DADA at s2024 "no difference shown".

### 11.6 Read-out, Kcap at seed 2024 — §11.5 reads "anything else" (2026-11-07)

Raw: `outputs/v_final/kim/REPORTS/kim_readout_s2024.{md,json}`, downloaded 2026-11-07, and `outputs/v_final/kim/Kcap/s2024/`; every row without Kcap matches §9. Kcap's per-clip reads and checkpoint stay on Drive.

**Kcap − W0 on DADA = +0.0009 [−0.0107, +0.0127], no difference shown.** By §11.5 Claude recommends closing the KIM line; the user decides. §6.3 still fails: K5 − Kcap on DADA has no gain at s2025.

| Macro, s2024 | W0 | K0 | K5 | Kcap |
|---|---|---|---|---|
| DADA | 0.7615 | 0.7256 | 0.7740 | 0.7624 |
| DoTA-CAP-dev | 0.8110 | 0.8002 | 0.8195 | 0.8172 |
| TAD | 0.6983 | 0.6954 | 0.7002 | 0.6928 |

| Contrast | Seed | DADA | DoTA-CAP-dev | TAD |
|---|---|---|---|---|
| Kcap − W0 (§11.5) | s2024 | **+0.0009 [−0.0107, +0.0127]** none | +0.0062 [−0.0021, +0.0144] none | −0.0055 [−0.0367, +0.0262] none |
| K5 − Kcap (§6.3) | s2024 | +0.0116 [+0.0020, +0.0210] gain | +0.0023 [−0.0058, +0.0103] none | +0.0074 [−0.0218, +0.0370] none |
| | s2025 | +0.0012 [−0.0056, +0.0080] none | −0.0063 [−0.0108, −0.0020] cost | −0.0058 [−0.0289, +0.0198] none |
| Kcap − K0 (printed) | s2024 | +0.0368 [+0.0281, +0.0457] gain | +0.0170 [+0.0106, +0.0233] gain | −0.0027 [−0.0320, +0.0239] none |
| | s2025 | +0.0371 [+0.0277, +0.0466] gain | +0.0226 [+0.0167, +0.0287] gain | −0.0024 [−0.0285, +0.0216] none |

Predictions: DADA "no difference shown" held; DoTA-CAP-dev positive held; **K5 − Kcap on DADA at s2024 "no difference shown" was wrong.**

**Reading.**

1. **No KIM row beats the DVS-on host at s2024 on any set.** K5 − W0 and Kcap − W0 both show no difference. By construction Kcap − K0 = (Kcap − W0) − (K0 − W0): on DADA, +0.0009 + 0.0359 = +0.0368. Branch + losses brings the DVS-off host back to W0's level here. That is arithmetic; it does not show the two act through one mechanism.
2. **The differences' contribution is inconsistent, not shown to be zero:** +0.0116 and +0.0012 on DADA, by seed. No pooling (§6.2), and §6.3 fails. A further seed to rescue it would change the rule after its number.
3. **K5 − Kcap's s2024 gain carries a flag:** see the mismatch below. s2025's read has the matched start, and it shows no difference.
4. **DVS on + KIM has never been measured.** Every statement here is about DVS off.

**First-batch mismatch, cause not measured.** D13 expects the K rows to start alike. `mil` at step 1 at s2024: K0 = K5 = 0.6878, **Kcap 0.6925**. At s2025 all three are 0.7057. `core/train.py` has no diff from `9e81b77`, and `kat_vad.py`'s only change picks the KIM class, with the build under a private RNG; `kim_runs` and `kim_seed` build the same command. K0 and K5 s2024 trained before the Kcap code and Kcap s2024 after it, in a later runtime, which is not proof of cause. **Checked 2026-11-07 from the downloaded run:** Kcap − K5 config = `kim.capacity_control` true, `kim.edge_mask` false, as §6.3 states; Kcap s2025 − s2024 = seed only; 1,749 rows. Step-1 losses all sit slightly off K5's, same sign and size (`mil` +0.0047, `dvs_sup` +0.0053, `dvs_sup_mil` +0.0042, `mul_mil` +0.0028), and step-2 `mil` is 0.7198 vs 0.7168. So config is excluded and the batch looks the same; the cause stays unmeasured, since torch and GPU lines were not saved. `check_run` only printed this value, so nothing stopped the run. K5 − Kcap's s2024 gain keeps its flag.

**What it touches.** K5 − Kcap and Kcap − K0 at s2024 do not have a matched start. Neither decision moves: §6.3 fails on s2025 alone, and W0's own first batch (0.6944, DVS on) never matched the K rows, so Kcap − W0 carries no matched start in any case.

## 12. K5d: KIM on the DVS-on host (dated note 2026-11-07, before any K5d number; user)

The user chose this over closing, after §11.5 read "anything else". It is a new question, not a repair of §6.3 or §11.5, and neither is reopened. Every number so far has DVS off, so K5 − W0 mixes DVS off with KIM (§9.3). K5d − W0 is the first contrast with DVS on both sides.

- **Arm K5d = W0 + K5 + `kim.seam_mask=true`:** `kim.enabled`, `kim.prototypes`, `kim.edge_mask`, `kim.seam_mask`. **D2 does not apply**: `dvs.theta` 0.7, `dvs.theta_ego` 0.85, as W0. Host kernel 3, s2024, 1,749 steps. Everything else as K5, weights unchanged.
- **Why the mask (written before any number):** DVS concatenates clips, so a spliced bag has a false jump at each join, and in a spliced abnormal bag two joins can sit on the positive span's edges. Test videos have none. Unmasked, KIM's differences could learn the joins. `seam_mask[t] = 1` when step t starts a new part; `pair[t]` is zeroed there, so `d1`, `d2` and `kim_scale` skip the join. An unmasked arm is not run.
- **What the mask cannot remove:** `state` still sees the join, and RoFormer's `vt` mixes neighbours, so differences near a join may stay high. **Printed, never decides:** `kim_seam_share` (valid steps within ±1 of a join) and `kim_anchor_seam_share` (anchors within ±1). The second far above the first means KIM may be reading the joins, and any gain carries that flag.
- **Inference:** no seams exist; `kim_seam` is `None`. Score path as K5.
- **Read:** paired per video, `kim_readout`. **K5d − W0** decides. K5d − K5 (DVS on vs off, with KIM) and K5d − K0 are printed. W0 has reads at s2024, so it is not retrained.
- **Decision, fixed now:** K5d − W0 **gain on DADA** at s2024 → the arm continues: s2025 against the W0 recipe (L0b has no DADA read; its read or a retrain comes first), and a Kcap on the same DVS-on host before any claim about the differences. **Anything else → the KIM line closes**: no unmasked arm, no λ or LR sweep, no further variant, K2 and s2026 not run. DoTA-CAP-dev is read beside it; a DoTA-only gain does not count, as §6.2. n = 1 adopts nothing.
- **Start check:** D13 expects step-1 `mil` near W0's 0.6944 (`w_out` = 0, same DVS draws). The runbook prints both, with torch version and GPU (§11.6, pending (cq)). A mismatch does not stop the run; it flags the contrast, and the read-out says so.
- **Prediction (written before the number):** K5d − W0 on DADA "no difference shown"; K5d − K5 on DADA not a gain.

### 12.1 Read-out, seed 2024 — **the KIM line closes** (2026-11-07)

Raw: `outputs/v_final/kim/REPORTS/kim_readout_s2024.{md,json}` and `outputs/v_final/kim/K5d/s2024/`, downloaded 2026-11-07; every number below was checked against the json. Per-clip reads and checkpoints stay on Drive.

**K5d − W0 on DADA = +0.0098 [+0.0010, +0.0185]: "no difference shown", so §12 reads "anything else" and the line closes.** No unmasked arm, sweep or variant; K2 and s2026 are not run. Reopening takes a new dated note from the user.

**The margin was missed by 0.0002 and must not be restated as a pass.** The interval excludes zero; the mean is below `V5_MIN_GAIN` = +0.01, the bar every read-out in this project uses. Lowering it after this number is what lesson 14 forbids. What this licenses: "+0.0098 on DADA at one seed, unmatched start, below the fixed margin". It does not license "KIM adds nothing" or "KIM improves the DVS-on host".

| Macro, s2024 | W0 | K0 | K5 | Kcap | K5d |
|---|---|---|---|---|---|
| DADA | 0.7615 | 0.7256 | 0.7740 | 0.7624 | 0.7713 |
| DoTA-CAP-dev | 0.8110 | 0.8002 | 0.8195 | 0.8172 | 0.8124 |
| TAD | 0.6983 | 0.6954 | 0.7002 | 0.6928 | 0.7055 |

| Contrast, s2024 | DADA | DoTA-CAP-dev | TAD |
|---|---|---|---|
| K5d − W0 (§12) | **+0.0098 [+0.0010, +0.0185]** none | +0.0014 [−0.0060, +0.0085] none (ego −0.0042, non-ego +0.0088) | +0.0072 [−0.0113, +0.0276] none |
| K5d − K5 (printed) | −0.0027 [−0.0121, +0.0068] none | −0.0071 [−0.0153, +0.0016] none | +0.0053 [−0.0210, +0.0322] none |
| K5d − K0 (printed) | +0.0457 [+0.0323, +0.0591] gain | +0.0122 [+0.0015, +0.0232] gain | +0.0100 [−0.0318, +0.0525] none |

Predictions held: K5d − W0 on DADA "no difference shown", by 0.0002; K5d − K5 on DADA not a gain.

**Reading, n = 1.** From K0 on DADA: DVS on alone +0.0359 (W0), KIM alone +0.0484 (K5), both +0.0457 (K5d). **The two do not add:** on top of KIM, DVS shows nothing (−0.0027). So K5 − K0's gain and DVS's gain cover much the same ground here; which mechanism is not measured. On DoTA-CAP-dev K5d − W0 is about zero, and K5d stays below the middle ruler 0.8516 and `p_T2` 0.8220.

**Three flags.**

1. **Start check fails, and the gap is small:** step-1 `mil` K5d 0.6906, W0 0.6944, past `KIM_START_TOL`, so K5d − W0 has no matched start. Every other step-1 loss sits close to W0's, far from the DVS-off rows (`dvs_sup` K5d 0.7044, W0 0.7028, K0 = K5 0.7182), and step-2 `mil` is 0.7355 vs 0.7358. So the DVS draw looks the same and the host differs slightly. Why is not measured; torch and GPU lines were not saved. A same-runtime W0 was not trained, so run-to-run spread of this size is unknown.
2. **Seam check, read from `metrics.jsonl` (printed, never decides): anchors do not concentrate on the joins.** Mean over batches, within ±1 of a join:

   | Batches | Anchors (`kim_anchor_seam_share`) | All valid steps (`kim_seam_share`) | Ratio |
   |---|---|---|---|
   | All 1,749 | 0.0545 | 0.0636 | 0.86 |
   | After warm-up, 1,530 | 0.0521 | 0.0635 | 0.82 |
   | Last quarter, 438 | 0.0521 | 0.0635 | 0.82 |

   §12's flag is the anchor share "far above" the step share. It is below, so K5d carries no seam flag. Limit: this is one seed's per-batch mean, and it says where anchors sit, not what KIM learned near a join. The Colab render omitted the table and printed `stop_rule: continue`, both fixed in the tool with a test; the downloaded report predates that, and §12 is the rule in force.
3. **Mechanics, one batch:** K5d `mil` last-epoch mean 0.2123 (K5 0.1637, W0 0.203); `kim_nkc` 2.4063 → 0.5860 (K5 2.1806 → 0.2771); ‖`w_out`‖ 0 → 0.7113.

**Config, checked:** K5d − K5 = `dvs.theta` 0.7, `dvs.theta_ego` 0.85, `kim.seam_mask` true, as §12 states (`kim.capacity_control` false is a key K5's older config lacks). K5d − W0 = the `kim` section plus `stp.*` and `model.motion_dropout` keys W0's older config lacks. `verified.json`: step 1,749; `metrics.jsonl`: 1,749 rows.

### 12.2 What the KIM line established, as closed

- K5 − K0 gain on DADA, two seeds (+0.0484, +0.0383). Against a control DVS off had lowered.
- No K row reaches a gain over W0 at s2024 on any set. K5 +0.0125, Kcap +0.0009, K5d +0.0098 on DADA.
- Explicit differences not established: K5 − Kcap on DADA +0.0116 and +0.0012 by seed, so §6.3 fails. "Sees kinematics" stays forbidden.
- Not measured: W0 at s2025 on DADA, a DVS-on Kcap, the reason a same-capacity branch helps a DVS-off host, KIM's own AUC on Nexar val (§9.5).

### 12.3 Dated note, 2026-11-07: K5 is the main model, by the user's choice

**The user chose K5 as the project's main model, replacing W0.** K5 = VideoMAE motion stream + full KIM (prototypes + NKC, first-pair `edge_mask`), DVS off, kernel 3. W0 stays a printed reference row, not a candidate.

- **This is a choice, not a rule's outcome.** §12 stands: the line is closed and no rule adopted anything. Write "chosen", never "adopted" or "won".
- **The comparison to quote against W0 is K5 − W0, s2024:** DADA +0.0125 [−0.0006, +0.0253], DoTA-CAP-dev +0.0085 [−0.0023, +0.0190], TAD +0.0019 [−0.0244, +0.0286], all "no difference shown". K5 − K0 (+0.0484, +0.0383) is against a control DVS off had lowered and does not measure K5 over W0.
- **Report K5 at both seeds:** DADA 0.7740, 0.7631; DoTA-CAP-dev 0.8195, 0.8166; TAD 0.7002, 0.6836.
- **Claims stay as in §12.2:** "sees kinematics" stays forbidden, since §6.3 failed. No K2, s2026 or sweep follows from this note.

## 13. K5 trained on T2 (dated note 2026-11-07, before any number; user)

**Not a reopening of the KIM line.** §12 stands and K5 on Nexar stays the main model (§12.3). The user asked for one more comparison after the TAD diagnosis of `K5_ZS_REPORT.md`: does the training corpus move K5's zero-shot reads? **Printed, decides nothing: no adoption, no K2, s2026 or sweep follows.**

| # | Choice | Value |
|---|---|---|
| T1 | Row | **K5-T2** = E3's A2 config (`v2_e3/A2/s2024/config.yaml`, T2, 20 × 87 = 1,740 steps, kernel 3, `mil_topk_pct` 5, `span`) + `dvs.theta = dvs.theta_ego = 1.0` + `kim.enabled`, `kim.prototypes`, `kim.edge_mask`; s2024. Nothing else differs; the notebook checks the config diff. `kim.warmup_steps` stays 219 (`round(5/40 · 1,740)` = 218; one step, not worth a key) |
| T2 | Inputs | E3's A2 bakes with **T2** statistics; TAD and DAD baked here with the same statistics. No extraction |
| T3 | Reads | DoTA-CAP-dev (protocol B), TAD (s8, raw), DAD (s7, raw, `ZS_DAD.md`), `nexar_val` (N-B: crops = endpoint, whole video printed), same tools as every earlier row |
| T4 | Nexar split | **`nexar_val`, the user's choice, 2026-11-07.** `nexar_test` keeps D-N22's one read: not touched. So these numbers never sit beside the `nexar_test` table of `RESULTS_NEXAR.md` |
| T5 | Rows on `nexar_val` | K5-T2, K5 (Nexar; **in-domain**, its own corpus's validation split) and LaGoVAD's released checkpoint, new reads. `nexar_eval` gains `--baseline` for it |
| T6 | DADA-2000 | **Not read.** The whole-video read's sources are T2's own test sources and the user did not ask for it |
| T7 | Sentence | Each set: K5-T2 − LaGoVAD and K5-T2 − K5, paired per item, 95 % bootstrap: "above / below / not separated", **PRELIMINARY (n = 1), not matched**. K5-T2 − K5 changes the corpus and the statistics together, so it never names a single cause |

**Expectations, written before the numbers:** (a) TAD macro K5-T2 above K5's 0.7002, because CLIP-only reads 0.727 trained on T2 vs 0.590 on Nexar (`RESULTS_NEXAR.md`); (b) DoTA-CAP-dev below K5's 0.8195, because A2 reads 0.7536 vs 0.8171; (c) `nexar_val` crops K5-T2 below K5, in-domain vs zero-shot. **T2 is dashcam too**, so a gain on TAD would mean the Nexar corpus matters, not that the camera does not; it does not test the fixed-camera explanation. DVS off has never been measured on T2, so (a) and (b) may fail for that reason.

Runbook `colab/final/k5_t2.ipynb`; table `core.tools.k5_t2_report`. Outputs `Thesis-Final/outputs/kim_t2/`.

### 13.1 Read-out — PRELIMINARY (n = 1), decides nothing (2026-11-07)

Raw `outputs/v_final/kim_t2/REPORTS/k5_t2_report.md`, regenerated locally from the saved scores byte for byte. K5-T2 s2024: `verified.json` and `metrics.jsonl` both step 1,740 (J10); config differs from K5 only in `data.dataset` and `train.num_epochs`.

| Set | Metric | LaGoVAD | K5 (Nexar) | K5 (T2) |
|---|---|---|---|---|
| Nexar val | macro, crops (endpoint) | 0.5859 [0.5399, 0.6314] | 0.8981 [0.8749, 0.9197] | 0.7816 [0.7447, 0.8165] |
| | macro, whole video (position test; middle ruler 0.9735) | 0.5832 | 0.9245 | 0.8201 |
| | micro / AP | 0.5224 / 0.0431 | 0.9000 / 0.4472 | 0.7644 / 0.1629 |
| DoTA-CAP-dev | macro | 0.6418 [0.6166, 0.6666] | 0.8195 [0.8048, 0.8338] | 0.8085 [0.7922, 0.8246] |
| | micro / AP | 0.6105 / 0.3838 | 0.7449 / 0.5790 | 0.7115 / 0.5309 |
| TAD | macro | 0.7579 [0.6955, 0.8147] | 0.7002 [0.6246, 0.7724] | 0.7606 [0.7015, 0.8157] |
| | micro / AP | 0.7912 / 0.2571 | 0.6781 / 0.2568 | 0.5187 / 0.2104 |
| DAD | macro | 0.6462 [0.5881, 0.7016] | 0.7916 [0.7513, 0.8294] | 0.8198 [0.7859, 0.8515] |
| | micro / AP | 0.5681 / 0.0640 | 0.7050 / 0.1156 | 0.7529 / 0.1534 |

| Set | K5-T2 − LaGoVAD (T7) | K5-T2 − K5 (T7) |
|---|---|---|
| Nexar val, crops | +0.1956 [+0.1400, +0.2504] above | −0.1165 [−0.1502, −0.0842] below |
| DoTA-CAP-dev | +0.1667 [+0.1437, +0.1902] above | −0.0110 [−0.0286, +0.0065] not separated |
| TAD | +0.0027 [−0.0642, +0.0691] not separated | +0.0604 [−0.0076, +0.1283] not separated |
| DAD | +0.1737 [+0.1124, +0.2364] above | +0.0282 [−0.0126, +0.0689] not separated |

**Expectations of §13:** (a) TAD macro above 0.7002: right direction, interval contains 0. (b) DoTA-CAP-dev below 0.8195: right direction, interval contains 0. (c) `nexar_val` crops below K5: holds, and it is in-domain against zero-shot, so it says little. **Not expected: TAD micro 0.6781 → 0.5187.**

**TAD, post-hoc, printed not gated, from the saved scores of the 100 videos.** "Inside" = accident frames vs the other frames of the abnormal videos, pooled. "Normal" = accident frames vs frames of normal videos, pooled. Clip AUC = video peak, abnormal vs normal.

| Read | LaGoVAD | K5 (Nexar) | K5 (T2) |
|---|---|---|---|
| AUC inside | 0.5393 | 0.6083 | 0.6787 |
| AUC normal | 0.8373 | 0.6909 | 0.4894 |
| Clip AUC | 0.7687 | 0.6358 | 0.5358 |
| Median score, accident frames | 0.057 | 0.044 | 0.508 |
| Median score, normal videos | 0.005 | 0.008 | 0.899 |
| Normal videos with a peak above 0.5 | 0.07 | 0.33 | 0.72 |

K5-T2 scores normal TAD videos above accident frames. Whole-video micro on `nexar_val` and DAD did not collapse with it, so "T2 has no whole normal videos" does not explain it alone. **Cause not tested;** T2 is dashcam, so this does not test the fixed-camera explanation either. By category K5-T2 is above K5 on 05_else (0.782 vs 0.665, n = 14), 06_PedestrianOnRoad (0.619 vs 0.451, n = 10), 07_RoadSpills (0.889 vs 0.584, n = 4), 01_Accident is level (0.834 vs 0.823, n = 23), and 02, 03, 04 are below (n = 3 each); no intervals.

**DAD:** placebo stays high: macro 0.8198 vs placebo 0.7146; same position, accident rows 0.7341 [0.6844, 0.7805], rows 7 up to the accident 0.7100 [0.6583, 0.7586]. The separation is there before the accident, as on K5 (Nexar). Never write that K5-T2 localizes accidents on DAD.

**Printed beside, not a contrast:** A2 trained on T2 with DVS on read DoTA-CAP-dev 0.7536 and K5-T2 reads 0.8085, but that is a five-seed average against one seed, with DVS off and KIM changed together. The DVS-off caveat of §13 stays unmeasured: there is no T2 control without KIM.

**What this licenses:** training on T2 does not rescue TAD (macro not separated, micro worse) and K5 on Nexar stays the main model. **Do not write** "T2 fixes TAD" or "the training corpus explains the TAD drop". §13 stands: no K2, s2026 or sweep follows from these numbers.

## 14. Form probe: does SA²'s two-sided, signed second difference read better than KIM's? (dated note 2026-11-07, before any number; user)

**Question (user):** SA² takes `D_t = 2P_t − P_{t−1} − P_{t+1}`, KIM's order 2 reads only earlier steps. Is SA²'s form worth trying on K5? **Inference only, no training, nothing extracted. This does not reopen the KIM line (§12), and K5 stays the main model (§12.3).**

**What the algebra already says (no data):** the two forms differ in two separate ways.

- **Two-sided vs one-sided:** `|2z_t − z_{t−1} − z_{t+1}|` at `t` equals `|z_{t+1} − 2z_t + z_{t−1}|`, the one-sided signed form at `t+1`. Once the sign is kept, two-sided is a shift by one step: 0.30 s on DoTA-CAP-dev (protocol B), 0.267 s on DADA. KP4's median peak lag on K0 is 1.80 s and 2.13 s.
- **Sign:** KIM and `vt2` take `| |Δ_t| − |Δ_{t−1}| |`, so a reversal at equal speed reads 0. The signed form does not. `kim_pilot` measured only the first.
- **Already in K5's view:** RoFormer's window is ±12 steps, the gate convolutions and the score head are symmetric kernel 3. K5's score at `t` already reads later steps; only the subtraction is one-sided.

**Not proposal R10:** R10 keeps the sign in the feature, with no magnitude at all. Every form here ends in a magnitude, so this probe neither runs nor replaces R10.

**Candidates**, on `vt` and on `clip`, one number per step, row names `<feature>2<suffix>`:

| Suffix | Form | Edges |
|---|---|---|
| none | KIM's form, `kim_pilot.change_scores` order 2, the reference | steps 0, 1 are 0 |
| `s` | signed one-sided, `mean_c |z_t − 2z_{t−1} + z_{t−2}|` | steps 0, 1 are 0 |
| `c` | signed two-sided, `mean_c |2z_t − z_{t−1} − z_{t+1}|` | first and last are 0 |
| `m` | SA²'s own `Mo`, `‖2ẑ_t − ẑ_{t−1} − ẑ_{t+1}‖₂` on L2-normalised `z`, `motion_magnitude` on one cell | as SA², the edge rows repeat their neighbour |

**Host: K0, s2024, the pilot's own host and reads.** K5's `vt` was trained through KIM's form, so it is not neutral between forms, and `kim_pilot.check_model` refuses a KIM checkpoint. **Limit: these numbers describe K0's `vt`, never K5's.** Sets, labels and inputs are the pilot's: DoTA-CAP-dev protocol B, DADA whole, TAD printed.

**Hard gate:** `kim_form_probe` recomputes `vt2`, `clip2` and `host` and raises unless every macro is within `KIM_FORM_BASELINE_TOL` of `pilot_edges/kim_pilot.json`, with the same host and global step.

**Rule, fixed now:** a candidate passes iff its macro minus the macro of the reference on the same feature is at least `KIM_PILOT_INPUT_MARGIN` (+0.02) on **both** DoTA-CAP-dev and DADA. TAD never decides.

- **No candidate passes:** the form is closed. Nothing is trained, and K5 is unchanged.
- **A `vt` candidate passes:** Claude may propose one training arm with its own dated note and rule, and the user decides. Nothing trains from this probe alone.
- **Only a `clip` candidate passes:** same, but the arm would also change KIM's input, so it would not be a pure change of form.

**Printed, never decides:** paired per-video candidate − reference with CI, DoTA-CAP-dev by ego flag, host + candidate rank average for the `vt` forms, median |peak − onset| for the `vt` forms.

**Prediction, written before any number:** no candidate passes. `c` − `s` is within ±0.005 on both endpoints, since it is a one-step shift. A wrong prediction changes no verdict.

**Limits.**

1. A step score's AUC is not a trained arm's gain: passing here does not predict K5's macro, and failing does not prove a trained branch could not use the form.
2. The bar of 0.02 is read six times (three forms × two features), with no multiplicity correction. It is KP1's margin, reused so that it was not chosen for this probe.
3. No first-pair mask, so `vt2` stays comparable with the recorded pilot; K5 has the mask. KP3's first-pair excess therefore sits in every row.
4. K5's and K0's scores on these sets have already been read, so this is not a blind look at the endpoints.
5. n = 1 seed: **PRELIMINARY**.

**Tools:** `python -m core.tools.kim_form_probe`, tests `core/tests/test_kim_form_probe.py`, runbook `colab/final/kim_form_probe.ipynb`, report `REPORTS/form_probe/kim_form_probe.{json,md}`.

### 14.1 Read-out, K0 s2024 — no form passes, the form is closed (2026-11-07)

Raw: `outputs/v_final/kim/REPORTS/form_probe/kim_form_probe.md`, pasted from Colab; the `.json` stays on Drive. The report exists, so §14's gate passed: `vt2`, `clip2` and `host` reproduce the recorded pilot.

**Decision: stop. Nothing is trained, K5 is unchanged, and no follow-up arm comes from this probe.**

| Candidate − reference (paired, 95 % CI) | DoTA-CAP-dev | DADA | TAD (never decides) |
|---|---|---|---|
| `vt2s` | −0.0011 [−0.0078, +0.0057] | −0.0000 [−0.0069, +0.0068] | −0.0050 [−0.0223, +0.0126] |
| `vt2c` | −0.0143 [−0.0234, −0.0055] | −0.0082 [−0.0170, +0.0007] | −0.0056 [−0.0341, +0.0216] |
| `vt2m` | −0.0293 [−0.0389, −0.0197] | −0.0200 [−0.0291, −0.0109] | −0.0240 [−0.0574, +0.0082] |
| `clip2s` | −0.0121 [−0.0182, −0.0059] | −0.0086 [−0.0150, −0.0024] | −0.0039 [−0.0167, +0.0085] |
| `clip2c` | −0.0338 [−0.0418, −0.0264] | −0.0222 [−0.0302, −0.0142] | −0.0079 [−0.0355, +0.0178] |
| `clip2m` | −0.0419 [−0.0498, −0.0346] | −0.0322 [−0.0407, −0.0237] | −0.0211 [−0.0523, +0.0077] |

**Predictions.** "No candidate passes" held. **"`c` − `s` within ±0.005" was wrong:** from the margins, `vt` −0.0132 and −0.0082, `clip` −0.0217 and −0.0136, DoTA-CAP-dev and DADA. No paired CI of `c` − `s` is printed. The verdict does not move.

**Reading.**

1. **Sign:** on `vt` there is no difference on any set, so the sign, which §14's algebra named as the real difference, shows no gain at the step-score level. On `clip` it costs on both endpoints.
2. **Two-sided:** reading the same magnitude one step earlier costs on DoTA-CAP-dev and on `clip`; on `vt` DADA the interval includes 0. An algebraic shift is therefore not neutral for the macro. Why is not measured.
3. **SA²'s own `Mo`:** the lowest form on each feature, with intervals below 0 on both endpoints. It also changes the normalisation and the norm, so this does not isolate which of them costs.
4. **Printed, never decides:** host + row rank average, DoTA-CAP-dev / DADA: `vt2` 0.8139 / 0.7428, `vt2s` 0.8155 / 0.7451, `vt2c` 0.8160 / 0.7461, `vt2m` 0.8126 / 0.7424. Unpaired, overlapping intervals, a K0 host: no claim, and no ground for an arm.

**Licensed:** on K0's `vt` and on `clip`, n = 1, no form reads better as a step score than KIM's. **Not licensed:** "KIM's form is better" (on `vt` the signed form ties), "two-sided is worse on K5" (K5's `vt` was not read), "a trained branch could not use the form" (§14 limit 1), or anything about R10.


## 15. Normal-text probe: does K5 care that its "Normal" definition says *surveillance*? (dated note 2026-11-09, before any number; user)

**Question (user):** `_UNIVERSAL_CLS_DEFS["Normal"]` (`core/data/definitions.py`) is a surveillance sentence ("Normal user uploaded video", "Normal surveillance footage.", "…surveillance content captured by camera…"). DoTA, DADA, Nexar and DAD all use it through `_DOTA_CLS_DEFS`. Does that matter when the model is trained on dashcam data? **Inference only on K5's two finished checkpoints, nothing trained. This does not reopen the KIM line (§12), and K5 stays the main model (§12.3).**

**Where the text acts (from the code, no data):** the class list is `[Normal, CarAccident]`. Its CLIP text features enter the co-attention `U` (each frame attends over two text tokens) and reach `H_bin` only through the **post-fusion** branch. `H_bin` blends `w · pre + (1 − w) · post` with `w = sigmoid(10 · adp_alpha)`; at init `alpha = 0.25` gives `w = 0.924`. `alpha` is learned, and no file records its trained value. `NORMAL_QUERY_CAPTION` (also "surveillance") feeds only `L_neg`, which K5 never runs.

**Step 1 (printed, reads the checkpoint):** `adp_alpha`, `w` and `1 − w` for K5 s2024 and s2025. Bar `NORMAL_PROBE_TEXT_MINOR_W` = 0.9: "the text branch is minor" iff `w ≥ 0.9` at both seeds. `w` weighs **logits**, not their variance, so step 1 describes the blend; it does not decide.

**Step 2 (decides):** K5 re-scored on DoTA-CAP-dev (protocol B) and DADA (whole) with the Normal sentences swapped, at inference only, for `NORMAL_PROBE_DASHCAM_DEFS`. These are four sentences with the same shape as the original (bare word, two short, one long), with no "surveillance":

1. `Normal driving`
2. `Normal dashcam footage of a vehicle driving on the road.`
3. `Normal driving video recorded by an in-car camera.`
4. `Normal driving recorded by a dashcam that does not involve any collision or unusual event.`

The list keeps its length, so the per-item verbalizer (C30) draws the same indices: **CarAccident's sentence is identical in both runs, and only Normal's changes** (tested).

**Hard gate:** the same in-process path with the original text must reproduce the recorded K5 read (`eval/{dota,dada}/K5` for s2024, `eval_s2025/…` for s2025) within `NORMAL_PROBE_REPRO_TOL` = 1e-4, or the report raises and writes nothing.

**Rule, fixed now.** Δ = dashcam − original, paired per clip, per seed and pooled over seeds (DoTA clustered by source video, DADA by video). Margin `NORMAL_PROBE_MARGIN` = 0.01, the margin of §12.

- **cost:** Δ ≤ −0.01 on any set at any seed. This is **uninterpretable**: K5 trained with the original text, so a swap at inference is a train/test mismatch. Nothing is concluded about training with dashcam text, and nothing is trained.
- **null:** otherwise, if the seed-mean |Δ| < 0.01 on both sets. The Normal wording does not matter at inference, and nothing is trained.
- **gain:** Δ ≥ +0.01 on both sets at both seeds, and the pooled CI's lower end > 0. Claude may propose one retrain arm (dashcam Normal text at training), with its own dated note and rule. The user decides.
- **mixed:** anything else. No claim, and nothing is trained.

**Prediction, written before any number:** `w ≥ 0.9` at both seeds, and the verdict is **null**. A wrong prediction changes no verdict.

**Limits.**

1. A null at inference does not prove training with dashcam text would not help: the fusion was fitted around the original vector. It shows only that K5's scores do not depend on the wording.
2. One candidate wording, chosen by Claude. Other wordings are not covered.
3. K5's scores on both sets were read before this probe, so this is not a blind look at the endpoints.
4. n = 2 seeds.
5. `H_mul` / `L_MIL-align` are not read: only `H_bin`'s macro decides.

**Tools:** `python -m core.tools.normal_text_probe {alpha,score,report}`, tests `core/tests/test_normal_text_probe.py`, runbook `colab/final/normal_text_probe.ipynb`, report `REPORTS/normal_text_probe/normal_text_probe.{json,md}`.

### 15.1 Read-out, K5 s2024 + s2025 — null, the Normal wording does not matter at inference (2026-11-09)

Raw: `outputs/v_final/kim/REPORTS/normal_text_probe/normal_text_probe.md`, pasted from Colab; the `.json` and the eight score dirs stay on Drive. The report exists, so §15's gate passed: with the original text, the in-process path reproduces every recorded K5 macro (DoTA-CAP-dev 0.8195 / 0.8166, DADA 0.7740 / 0.7631).

**Decision: null. Nothing is trained, and K5 is unchanged.**

| | s2024 | s2025 |
|---|---|---|
| `adp_alpha` (init 0.25) | +0.2418 | +0.2526 |
| pre-fusion `w` (init 0.9241) | 0.9182 | 0.9260 |
| text branch `1 − w` | 0.0818 | 0.0740 |

| Dashcam − original (paired, 95 % CI) | s2024 | s2025 | pooled |
|---|---|---|---|
| DoTA-CAP-dev | +0.0002 [−0.0001, +0.0005] | −0.0000 [−0.0000, +0.0000] | +0.0001 [−0.0001, +0.0003] |
| DADA | −0.0003 [−0.0009, +0.0003] | +0.0001 [+0.0000, +0.0003] | −0.0001 [−0.0004, +0.0002] |

**Predictions.** "`w ≥ 0.9` at both seeds" held, and so did "null". Every |Δ| is ≤ 0.0003, about 30× below the margin.

**Reading.**

1. **`adp_alpha` barely moved in training:** −0.008 at s2024 and +0.003 at s2025, so `w` stays within 0.006 of its init. In K5, `H_bin`'s blend is therefore close to its initial value, not something it learned. Why (gradient size, learning rate, or no signal) is not measured.
2. **Swapping the Normal sentence moves K5's frame-level macro by ≤ 0.0003 on both endpoints, at both seeds.** On K5, "surveillance" in the Normal definition is not a measurable cost at inference.

**Licensed:** K5's `H_bin` scores on DoTA-CAP-dev and DADA do not depend on the Normal wording at inference (n = 2). The branch that sees text carries about 8 % of the logit blend, the same as at init.

**Not licensed:**
- "Training with dashcam text would not help" (§15 limit 1). The fusion was fitted around the original vector. With the measured insensitivity, a gain from it is not expected, but it is not measured.
- Anything about `H_mul` / `L_MIL-align` (§15 limit 5).
- Anything about other wordings (§15 limit 2).
- "The text conditioning does nothing" in general. That needs a no-text or shuffled-text control, which was not run.

## 16. K5 trained on uncut Nexar videos (`whole`) (dated note 2026-11-09, before any number; user)

**Not a reopening of the KIM line.** §12 stands and K5 on `Nexar_window` stays the main model (§12.3). The user asked whether K5 changes when it trains on the uncut videos. **Printed, decides nothing (n = 1): no adoption follows.**

**What exists already.** The only `whole` run is the v2 pilot (s2099, `whole`/A0-ign and `whole`/A3-ign, CRN on, DVS on, `mil_topk_pct` 5; addendum §21 D-N10, commit `0e8c8f7`). On `nexar_val` its crop endpoint was 0.582 for A3-ign, below that same recipe trained on T2 and read zero-shot (0.689) and below `window`/A3-ign (0.822). W1 and W2 fired, and §29 then dropped `whole`. That pilot used a different host and `mil_topk_pct` = 5, so it does not predict this row.

**How far a row sees on `whole`** (needed for WH3–WH4):

- **The temporal encoder's band stays 25** (`model.temporal_window`, ±12 rows ≈ ±3.2 s at s8 on ~30 fps; two layers ≈ ±6.4 s). The band does not grow with the input. A `whole` video is ~151 rows (median, `nexar_train`), and `max_vis_len` = 512 does not cut it.
- **On `window` the band never bit.** 25 ≥ 20, so a mid-window row saw the whole window. `whole` is the first K5 training where train-time attention is local, as it already is at inference on long videos.
- **There is one global path.** In co-attention fusion, the text tokens attend to all T rows, and each row reads them back (`core/models/fusion.py:49-60`, two layers). Every row therefore gets a summary of the whole 40 s video. This is the clip-classifier channel (C14), and WH6 prints its symptom.

**Why `mil_topk_pct` changes** (WH4). `k = max(1, n // mil_topk_pct)` (`core/losses/mil.py:22`; `kim_mil` uses the same key, D7). The abnormal span is `t_alert … t_event + 1.4 s` (D-N1), measured on `nexar_train`'s 451 positives at s8 from `census.json`:

| `mil_topk_pct` | k (median) | span rows (median, p5) | positives with k > span rows | normal rows forced into top-k (median) |
|---|---|---|---|---|
| 5 (K5's value) | 30 | 11, 7 | **97.8 %** | 19 |
| **16** (`MIL_TOPK_PCT`, the baseline default) | 9 | 11, 7 | 20.2 % | 0 |

At 5, almost every positive bag's top-k mean must include normal rows. 16 restores the principle K5 already satisfies on `window` (k = 4 within a 20-row positive window): **top-k fits inside the span**. 16 on `window` would give k = 1, so the two arms cannot share one value.

| # | Choice | Value |
|---|---|---|
| WH1 | Row | **K5-whole** = K5's config (`outputs/v_final/kim/K5/s2024/config.yaml`) with these changes only: the training corpus `Nexar_whole` (every `nexar_train` video, video-level label, 451 positive + 449 negative), `loss.mil_topk_pct` 5 → **16** (covers `mil` and `kim_mil`), and `train.num_epochs` set so that the step count = `NEXAR_STEP_BUDGET` 1,740 (the v2 pilot had 116 × 15; the notebook recomputes this for DVS off and asserts it). **s2024 only (user).** The notebook prints and checks the config diff against K5 |
| WH2 | Kept | `model.temporal_window` 25 (user), `score_head_kernel` 3, DVS off (`theta = theta_ego = 1.0`, so `NEXAR_WHOLE_SYN_MAX_CLIPS` is moot), `kim.*` as K5, `kim.warmup_steps` 219, `v2.crn = none`, V2-S motion. `mul_mil_topk_pct` is already 16 |
| WH3 | Inputs | K5's existing Nexar s8 caches (CLIP `Nexar_s8_ncc` + the V2-S bake fitted on `nexar_train`). Both are construction-free, so there is no extraction and no new bake |
| WH4 | Negatives | All 449 negative videos of `nexar_train` (user). Length ruler AUC on Nexar is 0.537 (census), so the length leak is small but not zero; it is printed |
| WH5 | Reads | Same tools and protocols as K5's recorded rows: `nexar_val` (N-B: crop endpoint D-N2, whole-video macro, micro, clip AUC), DoTA-CAP-dev (protocol B), DADA whole (`dada_whole_eval`), TAD (s8, raw), DAD (s7, raw). `nexar_test` is not touched (D-N22) |
| WH6 | Printed beside | K5 at s2024 on each set; the `nexar_val` middle ruler 0.9735; per crop placement (0.1 … 0.9) and the W1 edge drop (centre − worse edge) for both rows; clip AUC on `nexar_val` for both, flagged if ≥ 0.95 (C14); the v2 `whole` pilot row, labelled "different host" |
| WH7 | Sentence | Each set: K5-whole − K5, paired per item, 95 % bootstrap (clusters as each tool already uses): "above / below / not separated", **PRELIMINARY (n = 1)**. The contrast changes the corpus, the top-k divisor and the steps per epoch together, so it never names a single cause and is never written "whole is better / worse" |
| WH8 | What may follow | Nothing by default. Only if K5-whole is **above K5 on `nexar_val` crops and on at least three of the four zero-shot sets** may Claude propose s2025, with its own dated note. The user decides |

**Expectations, written before the numbers:** (a) `nexar_val` crops: K5-whole below K5's s2024 read. The v2 pilot saw the same direction, and video-level labels on 151 rows are weaker supervision than span-labelled windows. (b) Clip AUC on `nexar_val`: K5-whole above K5, through the global text path. (c) W1 edge drop: K5-whole larger than K5, because 730/750 events sit mid-video, so whole-video training can learn position. (d) Zero-shot sets: no confident direction; Claude's guess is "not separated" on DoTA-CAP-dev and DAD. A wrong expectation changes nothing.

**Limits.**

1. n = 1; three things change at once (WH7).
2. pct = 16 was chosen from span statistics on `nexar_train`, not from any score. k still exceeds the span rows on 20 % of positives.
3. Band 25 is kept, so whether `whole` would want wider attention is not tested.
4. Every K5 read has been seen before, so this is not a blind comparison.

Runbook `colab/final/k5_whole.ipynb`; table `core.tools.k5_whole_report` (refuses unpaired items, a micro that disagrees with its scores, mixed or shared checkpoints, a Nexar read not on `nexar_val`; tests `core/tests/test_k5_whole_report.py`). Outputs `Thesis-Final/outputs/kim_whole/`. K5's `nexar_val` read is §13's (`kim_t2/eval/nexar/K5`); the notebook re-reads it only if it is gone.

### 16.1 Read-out, s2024 — PRELIMINARY (n = 1), decides nothing (2026-11-09)

Raw `outputs/v_final/kim_whole/REPORTS/k5_whole_report.md` (pasted by the user; the notebook's config diff and J10 passed). The v2 pilot row is absent because its Drive copy is gone (§29 disclosure).

| Set | K5 (window) | K5 (whole) | Δ, paired [95 %] | WH7 |
|---|---|---|---|---|
| Nexar val, crops (endpoint) | 0.8981 | 0.6547 | −0.2434 [−0.2848, −0.2019] | below |
| DoTA-CAP-dev | 0.8195 | 0.6408 | −0.1787 [−0.2018, −0.1547] | below |
| DADA whole | 0.7740 | 0.5922 | −0.1818 [−0.2135, −0.1505] | below |
| TAD | 0.7002 | 0.6769 | −0.0233 [−0.1042, +0.0601] | not separated |
| DAD | 0.7916 | 0.7178 | −0.0738 [−0.1371, −0.0113] | below |

Printed (WH6): Nexar val clip AUC 0.7383 vs K5 0.7735 (no flag); edge drop −0.0008 vs −0.0272; crops by placement 0.64 / 0.58 / 0.64 / 0.71 / 0.71; Nexar val AP 0.0926 vs 0.4472 (base 0.0394); TAD micro 0.4655 (< 0.5) vs 0.6781. **WH8: no s2025.**

**Expectations:** (a) crops below K5 held. (b) clip AUC above K5 was **wrong** (0.738 < 0.774). (c) a larger edge drop was **wrong** (≈ 0). (d) "not separated" on DoTA-CAP-dev and DAD was **wrong** (both below). By W3's measure (clip AUC) K5 (whole) is not a better clip classifier, and it is not a position reader; it is weaker on every read except TAD. **The post-hoc read below shows its curves are flat per video, so "not a clip classifier" holds only for clip AUC.**

**Licensed:** "K5 trained on `Nexar_whole` with `mil_topk_pct` 16 scores below K5 on Nexar val crops, DoTA-CAP-dev, DADA and DAD at s2024; TAD not separated. PRELIMINARY (n = 1)." **Not licensed:** "whole is worse", "video-level labels cannot localize", "the negative videos hurt", or any single cause (WH7: corpus, label granularity, negative pool, top-k divisor and steps per epoch change together). Nothing here speaks to windows cut from the negative videos.

**Post-hoc, printed, decides nothing (local outputs, 2026-11-09).** The full outputs (`outputs/v_final/kim_whole/`) regenerate the report byte for byte. The config differs from K5 only in `loss.mil_topk_pct`, `train.num_epochs` and two keys added later at their defaults; J10 holds at 1,740 steps.

- **Training fit.** `mil` falls below 0.1 from step 580 (epoch 38), below 0.01 from step 680 (epoch 45) and below 0.001 from step 1,216; last-100 mean 0.0005 (K5: 0.164, never below 0.1). `kim_mil` 0.0005 vs 0.160; `dvs_sup` 0.0000 vs 0.018. By epoch 45 the 900 training bags are fit, and about 60 % of the step budget runs on near-zero loss.
- **Score shape (saved curves, raw scores).** Median within-video range (max − min) for K5 (window) vs K5 (whole): DoTA-CAP-dev 0.943 vs **0.005**, TAD 0.389 vs **0.005**, DAD 0.058 vs **0.002**, DADA 0.528 vs 0.932 (DADA whole is the exception). Mean score inside vs outside the span: DoTA-CAP-dev 0.529 / 0.242 vs 0.868 / 0.812; DADA 0.467 / 0.186 vs 0.739 / 0.649. On TAD, whole scores normal videos above accident videos on average (0.681 vs 0.579), matching micro 0.4655.
- **Reading.** On three of the four zero-shot sets K5 (whole) emits a near-constant score per video: a per-video score with no frame-level shape. Under DoTA-CAP's per-clip min-max its macro of 0.64 ranks ripples of about 0.005. This is the C14 shape (`TAD_SETUP.md` §15.1). Its clip AUC of 0.738 on Nexar val says it is also a poor video classifier. Cause not tested: saturation after the fit, the fusion's global text path, or both.


## 17. K5 with windows cut from Nexar's negative videos (dated note 2026-11-09, before any number; user)

**Not a reopening of the KIM line**; K5 stays the main model unless the rule below and the user say otherwise. The user asked why `Nexar_window` cuts its normal windows only from the positive videos (pre- and post-event windows), leaving the 449 negative videos of `nexar_train` unused. The answer: the v2 plan (`katvad-v2-nexar-feasibility.md` N4) gated them behind an N-X probe that never ran. The precedent quoted for that gate (`0_Normal_Driving`, C28) belongs to the Kaggle archive, not to the original DADA-2000 release, so it does not transfer. §16 does not answer the question either: `whole` changed the labels, the top-k divisor and the steps per epoch together.

**Measured on the census and the frozen split (before any score):**

| | `Nexar_window` (K5) | `Nexar_window_neg` (K5neg) |
|---|---|---|
| positive windows | 1,689 | 1,689 |
| normal windows inside positive videos | 5,622 | 5,622 |
| windows from negative videos | 0 | **6,879** (449 videos, median 17 per video, no cap: user) |
| share of the normal pool from negative videos | 0 | **≈ 55 %** (pool sampling: user) |
| steps | 53 × 33 = 1,749 | identical (dataset length = 2 × positive windows) |

Scene and weather differ by class in `nexar_train` (Sub-urban 41 positive / 122 negative; Rain 42 / 14), so a scene shortcut is possible. NG5 prints it; it is not gated.

| # | Choice | Value |
|---|---|---|
| NG1 | Row | **K5neg** = K5's config with `--data-dir Nexar_window_neg`, built by `nexar_build corpus --constructions window_neg`: `window`'s geometry (20 rows at s8, hop 8, `WINDOW_MIN_POSITIVE`, no cap), with train windows from **every** `nexar_train` video. Val side identical to `Nexar_window`. **Config diff to K5 = 0** (asserted). s2024 only (user) |
| NG2 | Inputs | K5's bakes (`A2_S/Nexar_s8`, Nexar statistics, which already cover all 900 train videos) and K5's zero-shot bakes. No extraction, no KNN (DVS off, §16) |
| NG3 | Reads | As §16 WH5: `nexar_val` (crops endpoint, whole, micro, AP, clip AUC), DoTA-CAP-dev, DADA whole, TAD, DAD. `nexar_test` is not touched |
| NG4 | K5 on `nexar_val` | **Re-read** in this notebook, so both rows carry the new per-video stats (NG5). Gate: its crop endpoint must equal §13's recorded read (`kim_t2/eval/nexar/K5`) within 1e-4, or the report raises |
| NG5 | Printed, never gating | (a) **False alarms on `nexar_val`'s 113 negative videos**: share with peak > 0.5, mean score; (b) the same share **by scene** (Urban / Sub-urban / Highway / other): a drop concentrated in Sub-urban flags a scene shortcut; (c) median within-video score range on every set (§16.1's flat-curve check) |
| NG6 | Sentence | K5neg − K5 per set, paired per item, 95 % bootstrap: "above / below / not separated", **PRELIMINARY (n = 1)**. One factor changes (the normal pool), so a difference may be attributed to it, but with one seed it is never written as established |
| NG7 | What may follow | **Candidate** (Claude may propose s2025 + recording K5neg as a candidate main model; the user decides) iff **no set is "below"** and (**at least one set is "above"** or the false-alarm share on `nexar_val` negatives falls by ≥ 0.05). **K5 stays** if any set is "below". Otherwise: no claim |

**Expectations, written before the numbers:** macro "not separated" on most sets (macro ranks frames inside accident videos, which the new windows do not teach); the false-alarm share on `nexar_val` negatives falls; Nexar val micro and AP rise slightly. The dilution risk is that in-video contrast is drawn about 55 % less often, which could make DoTA-CAP-dev or DADA read "below". A wrong expectation changes nothing.

Runbook `colab/final/k5_neg.ipynb` (builds `Thesis-Final/data/Nexar_window_neg`, asserts K5's windows and labels unchanged, every added window all-normal and from a negative train video, val + `defs.json` byte-identical to `Nexar_window`, step plan = K5's). Corpus `nexar_build corpus --constructions window_neg` (opt-in, not in the default build). Per-video stats `nexar_eval` → `video_stats.json` (additive). Table `core.tools.k5_neg_report` (tests `core/tests/test_k5_neg_report.py`). Outputs `Thesis-Final/outputs/kim_neg/`.

**Limits.** n = 1. The scene print is descriptive: a shortcut that lowers false alarms on Sub-urban val negatives would also look like a gain. Pool sampling makes "more negatives" and "less in-video contrast per step" one factor, not two.

### 17.1 Read-out, s2024 — PRELIMINARY (n = 1), K5 stays (2026-11-10)

Raw `outputs/v_final/kim_neg/REPORTS/k5_neg_report.md` (pasted by the user). NG4 holds (K5's re-read 0.8981 vs §13's 0.8981, gap 0). Not tracked: `outputs/` is gitignored.

| Set | K5 | K5neg | Δ, paired [95 %] | NG6 |
|---|---|---|---|---|
| Nexar val, crops (endpoint) | 0.8981 | 0.8597 | −0.0384 [−0.0527, −0.0246] | below |
| DoTA-CAP-dev | 0.8195 | 0.8044 | −0.0151 [−0.0227, −0.0076] | below |
| DADA whole | 0.7740 | 0.7531 | −0.0209 [−0.0298, −0.0120] | below |
| TAD | 0.7002 | 0.7097 | +0.0095 [−0.0141, +0.0341] | not separated |
| DAD | 0.7916 | 0.8086 | +0.0170 [−0.0072, +0.0420] | not separated |

**NG7: below = Nexar, DoTA-CAP-dev, DADA → K5 stays.** The false-alarm share on `nexar_val`'s 113 negatives fell by +0.4336 (0.7168 → 0.2832, margin 0.05), but NG7 lets a drop count only when no set is below.

Printed (NG5): peak > 0.5 on the negatives by scene, K5 → K5neg: Urban 0.707 → 0.268 (n = 41), Sub-urban 0.759 → 0.310 (29), Highway 0.667 → 0.333 (36), other 0.857 → 0.000 (7). Not concentrated in Sub-urban; the print says nothing about Rain (Rain 42 / 14 in train, not printed). Nexar val micro 0.9000 → 0.8497, AP 0.4472 → 0.3573, whole macro 0.9245 → 0.8838, clip AUC 0.7735 → 0.7909. TAD micro 0.6781 → 0.5883. Median within-video range, K5 → K5neg: Nexar val 0.969 → 0.439 (confounded: 113 of the 225 videos are negatives and K5neg now keeps them low), DoTA-CAP-dev 0.943 → 0.889, DADA 0.932 → 0.720, TAD 0.389 → 0.227, DAD 0.058 → 0.048.

**Expectations:** (a) false-alarm share falls: held (−0.43). (b) macro "not separated" on most sets: **wrong** (3 of 5 below). (c) Nexar val micro and AP rise slightly: **wrong** (both fall, −0.050 and −0.090). The dilution risk named in the note (in-video contrast drawn about 55 % less often) is the reading that fits the three "below" rows, but the one-factor design cannot separate it from "more negatives".

**Post-hoc, printed, decides nothing (local `video_stats.json` of both rows, 112 positive and 113 negative `nexar_val` videos; one peak per video).**

| | K5 | K5neg |
|---|---|---|
| peak > 0.5, negatives (false alarm) | 0.717 | 0.283 |
| peak > 0.5, **positives (detection)** | **0.964** | **0.688** |
| mean peak, positives / negatives | 0.922 / 0.707 | 0.690 / 0.300 |
| positives with peak > 0.5 under one row only | only K5 0.277 | only K5neg 0.000 |
| video AUC by peak / by mean | 0.7735 / 0.6747 | 0.7909 / 0.7553 |

- The 0.5 cut moves both rates: false alarms −0.434, detections −0.277. Every positive video K5neg flags, K5 flags too; at 0.5 the change is mostly a shift of the score scale, not a new ranking.
- Matched operating points (the cut chosen on each row's own negatives): at a 0.20 false-alarm rate K5 detects 0.562 and K5neg 0.661; at 0.283, 0.607 vs 0.696; at 0.10, 0.545 vs 0.562; at 0.717 K5 0.955 vs K5neg 0.929. K5neg is ahead by about 0.09 at 0.2 – 0.3 and level at the extremes. One point per rate, no interval, n = 112 / 113 videos.
- K5's 0.964 detection at 0.5 is partly saturation: it also flags 0.717 of the negatives; its useful separation sits at peaks above 0.98.
- **Not decomposed:** the fall of Nexar val micro and AP. The per-frame scores of `nexar_val` are not saved locally (`kim_neg/eval/nexar/*` hold `video_stats.json`, `item_aucs.json`, read-outs only), so a split by frame inside positives cannot be run.

**Licensed:** "K5neg scores below K5 on Nexar val crops, DoTA-CAP-dev and DADA at s2024; TAD and DAD not separated. It lowers the share of `nexar_val` negatives with peak > 0.5 from 0.717 to 0.283, and lowers the share of positives from 0.964 to 0.688. PRELIMINARY (n = 1)." **Not licensed:** "K5neg is better / worse than K5", "negatives cause the localization loss" (one factor = negatives + less in-video contrast under pool sampling), "no scene shortcut" (Rain not printed; the 0.5 cut depends on calibration), any seed-level claim: the paired intervals are over items for one checkpoint pair, and K5's DoTA-CAP-dev macro is 0.8195 at s2024 vs 0.8058 at s2025 (≈ 0.014, about the DoTA Δ of −0.0151). Only Nexar val (−0.038) lies clearly beyond that spread. No s2025. Options for a later note, none authorized: a smaller negative share (separates the two factors), or a threshold-free false-alarm read.

## 18. Zero-shot MSAD test: K5 vs LaGoVAD's released checkpoint (dated note 2026-11-10, before any K5 number; user)

**Not a reopening of the KIM line**; nothing is trained and K5 stays the main model whatever this reads. The user asked whether K5 costs on fixed cameras (CCTV) the way TAD suggests, against both the public checkpoint and the paper. MSAD is the first set **no choice so far has seen**: K5 was chosen after DoTA-CAP-dev, DADA-2000 and TAD were read; MSAD has never been scored by any Nexar-trained row. The `mp4` were deleted from the user's Drive; the official blurred release (owner: MSAD's authors) is reached through the user's shortcut `Thesis-Final/data/MSAD Videos blurred/` (`MSAD_anomaly/<class>/`, `MSAD_normal_testing/`).

**Measured before any K5 score** (LaGoVAD's saved scores of `full_gate_a`, on record since 2026-08, `outputs/v1/MSAD/MSAD_ncc/full_gate_a`; rulers read no pixel):

| | MSAD-full | MSAD-traffic |
|---|---|---|
| videos (abnormal / two-class at s8) | 240 (119 / 96) | 140 (20 / 19) |
| LaGoVAD micro (the paper's metric) | 0.8949 | 0.9343 |
| LaGoVAD micro, every video weighted equally | 0.8558 | 0.9168 |
| LaGoVAD macro | 0.7177 | 0.6234 |
| LaGoVAD clip AUC (peak) | 0.9416 | 0.9258 |
| LaGoVAD AP | 0.6432 | 0.3340 |
| clip oracle (micro) | **0.8943** | **0.9771** |
| length ruler (micro, shorter = more anomalous) | 0.6466 | **0.9639** |
| monotone ruler `t/N` (macro) | **0.7090** | 0.5874 |

So: on MSAD-full the paper's micro (90.41) sits at the clip oracle (a clip classifier with no localization scores 0.8943); on MSAD-traffic the length ruler alone (0.9639) beats LaGoVAD's micro, so **micro on MSAD-traffic says nothing**; on MSAD-full a ramp `t/N` scores within 0.009 of LaGoVAD's macro. Of the 120 abnormal-named test videos, `Explosion_3` has no positive row at s8 (hence 119 / 121 in `RESULTS_MSAD.md`).

| # | Choice | Value |
|---|---|---|
| MS1 | Rows | **LaGoVAD** = released `best.ckpt`, KIP off (`RELEASED_CKPT_OVERRIDES`), plain `cache/clip/MSAD_ncc` (the `full_gate_a` command). **K5** s2024 = `kim/K5/s2024/checkpoint_last.pt`, input bake `A2_S/MSAD-full_s8_nexar` (`MSAD_ncc` + V2-S, Nexar's statistics, no CRN, as every K5 zero-shot read). Nothing trains |
| MS2 | Labels and text | `Thesis/data/MSAD` as `full_gate_a`: `frame_labels_test.json` at s8 (240 videos), `defs.json` = MSAD's 12 classes, `data.dataset=MSAD-full` → MSAD's own definitions, the same for both rows (proposal P4: each test set's own definitions). Raw pooling (`--score-norm auto` → none) |
| MS3 | Motion input | VideoMAE V2-S at s8, frame step 3, `assumed_fps` 30: **Nexar's geometry: the manifest equals Nexar's V2-S manifest on every key but `stride`** (asserted; `stride` is the row step of the read, 8 here). Frames decoded from the shortcut's `mp4` to JPEG q95 on the VM (`dad.decode_video`); CLIP is **not** re-extracted (`MSAD_ncc`) |
| MS4 | Gates (stop on any) | **M0** LaGoVAD re-read micro = `full_gate_a`'s 0.8949 ± 1e-3. *Note 2026-11-11, after M0 read FAIL on the per-item read (micro 0.8986, gap +0.0037; AP 0.6554 vs 0.6432; macro 0.7176 vs 0.7177), before any K5 score and before the legacy read below:* frames (18,350) and `gt` are identical on all 240 videos; scores differ on 197 / 240 (max 0.43), also on Assault_2, whose CLIP row is bit-equal to `MSAD_ncc`'s (M1). `full_gate_a/results.json` is dated 2026-08-12; `core.evaluate` sampled class definitions from ONE run-wide RNG stream (`random.Random(SEED)`, dataset order) until `b48508a` (2026-09-13, lesson C30), and has drawn them per video id since; the old code's own comment measured ±0.003 AUC and ±0.3 per-video max score from draw order alone, the size seen here. So the per-item read compares the checkpoint on other draws than `full_gate_a` did; it does not test whether the checkpoint reproduces. **M0 is unchanged (value 0.8949, tolerance 1e-3); it is now read with `core.evaluate --legacy-shared-verbalizer`** (opt-in, off by default, stamps `legacy_shared_verbalizer: true` in `results.json`), output `eval/LaGoVAD_legacy/released/`. The **table** keeps the per-item reads for both rows (LaGoVAD and K5 see the same draws per video); the per-item gap to `full_gate_a` is printed, not gated. Prediction before the read: the legacy micro lands within 1e-3 of 0.8949; if it does not, M0 still FAILS and the cause is open. **M1** identity: CLIP `_ncc` re-extracted straight from the shortcut's `mp4` for 12 test videos (first id of each of the 11 classes + first normal) equals `MSAD_ncc` row for row, max abs ≤ 1e-3; else these are other videos and every MSAD number on record stops being a reference. **M2** census: each of the 240 test ids found exactly once, and `ceil(decoded / 8) == rows(MSAD_ncc)` (same pixels). `rows ≠ len(labels)` is counted and printed, not gated: `core.evaluate` truncates both to the shorter (`dataset.py:239`), exactly as in `full_gate_a`, and both rows see the same truncation. *Note 2026-11-10, after the first census run, before any score:* M2 failed with 239 / 240 ids found twice. The shortcut also holds `VALU_MSAD_subset_2026/`, a **modified** copy of the same test videos (its README: timestamp and caption watermarks masked, some videos cut into `_clip_` files). The census and the copy now read only `MSAD_anomaly/` and `MSAD_normal_testing/` (`MSAD_ZS_SOURCE_DIRS`); M2 itself is unchanged. *Note 2026-11-11, after M1 read FAIL at max abs 0.0161, before any K5 or LaGoVAD score (user approved the amendment; raw failed read kept as `REPORTS/identity_M1_failed_atol1e-3.json`):* the atol-1e-3 bar measured bit identity, not "the same video". Measured on the 12 ids: 4 videos (Assault_2, Explosion_1, Fighting_3, Fire_2) match at ≤ 1.1e-5; 8 differ at 3.7e-3…1.6e-2 over 78–100 % of their rows (rows > 1e-3: 88/94, 88/112, 40/41, 124/155, 30/30, 42/46, 98/126, 30/31), i.e. spread over the video, not a few frames. The cause is **not identified**: the `Thesis/data/MSAD/videos` the `MSAD_ncc` was extracted from (`colab/MSAD_ncc/train.py:68`) no longer exists on Drive, so no pixel or byte comparison was possible; re-encode and decoder differences are candidates, nothing tested. Same video, however, is measured: per-row cosine(MSAD_ncc, re-extracted) ≥ **0.999992** on all 12, against ≥ 0.787 between a video's own adjacent rows and ≤ 0.779 between different videos' means; each re-extracted video-mean is nearest to its own `MSAD_ncc` mean for 12/12. **M1′ (replaces the pass rule, the 12 ids and the extraction command are unchanged):** per video, equal row count, every row cosine ≥ `MSAD_ZS_IDENTITY_MIN_COS` = 0.999, and the nearest video-mean is its own; `max_abs` and `bit_identical` (max abs ≤ 1e-3) are printed, not gated. **The 0.999 was set after the first read**; every floor between 0.8 and 0.99999 gives the same verdict on these 12, so the number was not chosen to pass. **What it does not license:** "MSAD_ncc is the blurred release" (unknown) or "the 8 videos are bit-equal" (they are not). **Consequence for the read:** LaGoVAD and K5 both take CLIP from `MSAD_ncc`; only V2-S reads the shortcut's mp4, whose frame counts match `MSAD_ncc`'s (M2, 240/240). Any pixel difference between that mp4 and the `MSAD_ncc` source reaches K5's motion stream only, and is a limit of the K5 row (add to **Limits**). **M3** V2-S rows == CLIP rows == bake rows; bake `train_ids_sha1` == Nexar's. **M4** K5 J10 + config (as `k5_dad.ipynb`) |
| MS5 | Sets | **MSAD-traffic** = 20 `Traffic_accident` + 120 normal test videos (macro over 19 two-class; the annotation gives normals no scenario, so every normal is in). **MSAD-full** = 240 videos (macro over 96) |
| MS6 | Contrast, sentence | K5 − LaGoVAD macro, paired per video, 95 % bootstrap (10,000, seed 2026, cluster = video): "above / below / not separated", each with **"PRELIMINARY (n = 1), not matched"**. The question "does K5 cost on CCTV traffic" is answered by **MSAD-traffic only**. MSAD-full is 99 of 119 abnormal videos outside the thesis scope (Fire, Robbery, Shooting, …) that K5's Nexar training never named, so a cost there is **not** attributed to the camera |
| MS7 | Printed, never gating | micro (paper's metric), micro with equal video weight, AP + base rate, clip AUC; the three rulers above; macro per class for both rows; decoded fps spread; the paper's 90.41 **beside LaGoVAD's re-read only**, never beside K5 (C8b) |
| MS8 | What may follow | Nothing is adopted. MSAD-traffic "below" → written as a limitation of K5 on fixed cameras, beside TAD. An MSAD-trained arm read on TAD (the open question of 2026-11-10: does CCTV in training close the TAD gap) needs its own dated note |

**Expectations, written before the numbers:** MSAD-full macro "below" (most abnormal classes are not traffic, and the public checkpoint trained on PreVAD's surveillance classes); MSAD-traffic macro "not separated" (19 videos give a wide interval); K5's clip AUC below LaGoVAD's on both sets, as on TAD. A wrong expectation changes nothing.

Runbook `colab/final/k5_msad.ipynb` (census → M1 → decode → V2-S → bake → M0 → K5 read → report). Census / identity `core.tools.msad_zs` (tests `core/tests/test_msad_zs.py`). Table `core.tools.k5_msad_report` (tests `core/tests/test_k5_msad_report.py`). Outputs `Thesis-Final/outputs/kim_msad/`.

**Limits.** n = 1. LaGoVAD's zero-shot status on MSAD is unverified (PreVAD's sources were never checked against MSAD's). The two rows read different pixels for motion (K5 alone has V2-S, from JPEG q95). **M1 passed only under M1′ (2026-11-11):** 8 of 12 checked videos are not bit-equal to `MSAD_ncc`'s source (max abs up to 0.0161, cosine ≥ 0.999992), cause unidentified, original files gone; the motion stream decodes the shortcut's mp4, CLIP stays `MSAD_ncc`'s. MSAD's frame rates are not checked equal to 30: the V2-S window is 16 × 3 frames, 1.6 s only at 30 fps (spread printed). MSAD-traffic has 19 two-class videos.

### 18.1 Read-out, K5 s2024 vs LaGoVAD released — PRELIMINARY (n = 1), not matched, nothing adopted (2026-11-11)

Raw: `outputs/v_final/kim_msad/REPORTS/k5_msad_report.{md,json}` (untracked; also `Thesis-Final/outputs/kim_msad/`). Gates M0–M4 passed before the table was written.

**M0 (legacy shared-verbalizer read):** LaGoVAD micro **0.894897** vs `full_gate_a` 0.8949, gap −0.00000. The checkpoint and `MSAD_ncc` reproduce; the 2026-11-11 failure was the definition-draw mode alone (C30), as the MS4 note predicted. The per-item reads the table uses are 0.8986 micro / 0.7176 macro / AP 0.6554 (legacy: 0.8949 / 0.7177 / 0.6432): macro does not move with the draws, micro by +0.0037.

| Set | Metric | LaGoVAD | K5 | K5 − LaGoVAD |
|---|---|---|---|---|
| MSAD-traffic (140 videos, macro over 19) | AUC macro | 0.6247 [0.4766, 0.7674] | 0.7424 [0.6246, 0.8507] | +0.1177 [−0.0462, +0.2789] **not separated** |
| | micro / equal-video micro | 0.9375 / 0.9226 | 0.6188 / 0.6332 | −0.3187 (printed) |
| | AP (base 0.0316) / clip AUC | 0.3361 / 0.9279 | 0.0557 / 0.5617 | printed |
| MSAD-full (240 videos, macro over 96) | AUC macro | 0.7176 [0.6625, 0.7705] | 0.6810 [0.6255, 0.7347] | −0.0366 [−0.1076, +0.0340] **not separated** |
| | micro / equal-video micro | 0.8986 / 0.8607 | 0.5386 / 0.5512 | −0.3600 (printed) |
| | AP (base 0.2300) / clip AUC | 0.6554 / 0.9450 | 0.2607 / 0.5020 | printed |

Rulers (no pixel): clip oracle 0.9771 (traffic) / 0.8943 (full); length ruler 0.9639 / 0.6466; `t/N` macro 0.5874 [0.4213, 0.7525] / 0.7090 [0.6407, 0.7750]. Census: 240 videos, 19 off 30 ± 1 fps, label rows ≠ CLIP rows on 0.

Macro per class on MSAD-full (printed, no interval, few videos per class), K5 − LaGoVAD: Assault +0.106, Robbery +0.110, Traffic_accident +0.118, Object_falling +0.039; Fighting −0.033, Shooting −0.100, Explosion −0.132, People_falling −0.144, Water_incident −0.151, Vandalism −0.197, Fire −0.400 (K5 0.4718).

**Expectations (§18) vs what read:** (a) MSAD-full macro "below" — **wrong**, not separated (−0.037, the interval holds zero). (b) MSAD-traffic macro "not separated" — held, but the interval is 0.33 wide, so it excludes no effect of either sign. (c) K5's clip AUC below LaGoVAD's on both sets — held, and by far more than expected (0.502 vs 0.945 on full; 0.562 vs 0.928 on traffic). A wrong expectation changes nothing.

**Licensed:** "On MSAD-traffic and on MSAD-full, K5 − LaGoVAD macro is not separated (traffic +0.118 over 19 two-class videos, full −0.037 over 96). K5's pooled micro and clip AUC are far below LaGoVAD's on both sets (clip AUC 0.502 / 0.562 vs 0.945 / 0.928; printed, no interval): its per-video peak does not separate abnormal from normal videos on this CCTV set. PRELIMINARY (n = 1), not matched." **Not licensed:** "K5 beats / loses to LaGoVAD" (not matched; trained on different data); "K5 costs on fixed cameras" or "K5 is fine on CCTV" (MS8 asks for a limitation only on "below"; the traffic interval is too wide for either); any sentence from a micro on MSAD-traffic (length ruler 0.9639 beats LaGoVAD's 0.9375); a per-class sentence; the 90.41 beside K5; "K5 localizes better than position" (the `t/N` ruler is 0.709 on full, within 0.009 of LaGoVAD and above K5; on traffic the K5 and ruler intervals overlap and the ruler is not paired).

**What it says, bounded.** Macro (within a video) is not separated; clip AUC and micro (across videos) collapse. The same split as K5 on TAD (§13.1: micro 0.5187). Two sets, n = 1 each, so it is a repeated sign, not a result. Cause not identified; candidates, none tested: the score scale learned on Nexar does not carry to CCTV raw pooling; V2-S reads the shortcut's `mp4`, whose pixels differ from `MSAD_ncc`'s source (M1′ limit); K5 never saw a fixed-camera negative. A test would be a clip-normalized (per-video min-max) micro of K5 beside its raw one, which separates scale from ordering; it needs a dated note before it is run.

**Mechanics.** `core.tools.k5_msad_report` first gated M0 on the per-item run and raised on the first Colab run (0.8986 vs 0.8949) although the notebook had already passed M0 on the legacy run. Fixed 2026-11-11: M0 gates `eval/LaGoVAD_legacy/released/`; the per-item micro and its gap are printed in the report and JSON (`per_item_micro`, `per_item_gap`); the report refuses if the legacy run scored other videos or labels than the table's rows. Test `test_per_item_gap_does_not_gate`; `core/tests/test_k5_msad_report.py` 9 passed. The table was produced after the fix, from the same saved scores.

**What may follow (MS8).** Nothing adopted; no limitation written (the traffic macro is not "below"). Open, each needing its own dated note: the clip-normalized micro above; more K5 seeds for the 19-video traffic interval; an MSAD-trained arm read on TAD.

## 19. K5's micro: per-video headroom and v2's §27 MSAD clip gate on K5 (2026-11-11; user)

**Question (user).** K5's macro is high but its micro is lower, because normal videos score high. Can micro rise without macro falling? **Disclosure:** Claude first proposed a video-level gate as a new idea, plus a CRN explanation; K5 has no CRN (D1), and the gate is v2's §27 (`RESULTS_TADZ.md` §3, `tad_clip_gate.py`, 2026-10-21). This section re-uses that frozen gate. It fits and trains nothing.

**Why a per-video gate cannot cost macro.** Macro ranks frames inside one video. Adding a per-video constant, or multiplying by one that is > 0, leaves every within-video order unchanged; only how videos interleave in the pooled micro ranking can move. A global calibration (temperature, sigmoid) moves neither metric.

**Step 0 (headroom, local saved scores, K5 s2024, no new run).** Tool `core.tools.k5_clip_gate`. "Offset by X" = `s + K·X(v)` with `K` = the set's score spread + 1 (exactly a lexicographic rank only for the 0/1 oracle). DoTA-CAP and DADA whole have **no normal videos** (569 / 370 all two-class), so no video-label gate can help them; they are not read.

| Set (videos / normal) | macro | micro raw | oracle per-video gate | per-video min-max | offset by K5's peak | offset by K5's mean | clip oracle | length ruler |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TAD (100 / 40) | 0.7002 | 0.6781 | **0.9394** | 0.7517 | 0.5551 | 0.6628 | 0.9226 | 0.8968 |
| DAD (466 / 301) | 0.7916 | 0.7050 | **0.8830** | 0.7758 | 0.6251 | 0.6334 | 0.8390 | 0.5000 |
| MSAD-full (240 / 121) | 0.6810 | 0.5386 | **0.9238** | 0.5555 | 0.5093 | 0.5188 | 0.8943 | 0.6466 |

- The micro gap is almost all **between videos**: a correct per-video gate lifts micro by +0.18 … +0.39.
- **K5's own curve does not carry that signal.** Clip AUC of K5's peak / mean / range: TAD 0.64 / 0.69 / 0.62, DAD 0.65 / 0.65 / 0.65, MSAD 0.50 / 0.53 / 0.50, and offsetting by them lowers micro. A gate needs information K5's score does not have.
- Per-video min-max alone raises micro on TAD (+0.074) and DAD (+0.071). It is an eval-protocol change chosen after seeing these sets: printed, never a K5 number.

**Step 1 (written before the read, here):** n = 1 (K5 s2024 × §27's frozen gate, `outputs/v2/v2_tadg/gate/c_{test,msad_test}.json`). TAD is zero-shot for both. Sentence: "the MSAD gate restores clip-level discrimination for K5 on TAD" iff the video bootstrap of micro(K5 × gate) − micro(K5) lies above 0. Expected: above (A3-ign had +0.157 with this gate at the same raw micro). MSAD-test is **in-domain for the gate** (fitted on MSAD-train): printed, not read. Macro must not move (|Δ| ≤ 1e-9, else the tool raises).

**Read (2026-11-11), `outputs/v_final/kim/REPORTS/k5_clip_gate.{md,json}` (untracked):**

| Set | row | micro | AP | clip AUC |
|---|---|---:|---:|---:|
| TAD | K5 raw | 0.6781 | 0.2568 | 0.6358 |
| | min-max only | 0.7517 | 0.2594 | 0.6146 |
| | gate only (`c(v)` on every frame) | 0.8599 | 0.2462 | 0.8512 |
| | **K5 × gate** | **0.8692** | **0.3695** | 0.8512 |
| MSAD-test (gate in-domain) | K5 raw | 0.5386 | 0.2607 | 0.5020 |
| | gate only | 0.8738 | 0.5990 | 0.9484 |
| | K5 × gate | 0.8474 | 0.5776 | 0.9484 |

- TAD: micro(K5 × gate) − micro(K5) **+0.1911 [+0.1149, +0.2693]** → **above**; the expectation held. Macro drift 0. Versus min-max only +0.1175 [+0.0712, +0.1675].
- **Most of it is the gate, not K5:** micro(K5 × gate) − micro(gate only) = +0.0093 [−0.0454, +0.0798] on TAD (not separated) and −0.0265 [−0.0498, −0.0029] on in-domain MSAD. AP is where K5 adds: 0.246 → 0.370 on TAD (localization inside the videos the gate keeps).
- Still below the length ruler (0.8968) and the clip oracle (0.9226) on TAD.

**Licensed:** "On TAD, v2's MSAD-fitted clip gate (no TAD video, no Nexar video, fitted before K5 existed) lifts K5's micro 0.678 → 0.869, Δ +0.191 [+0.115, +0.269], macro unchanged by construction; nearly all of the micro comes from the gate (gate alone 0.860). PRELIMINARY (n = 1)." **Not licensed:** "K5 detects better", "the gate detects accidents" (MSAD's 11 classes, recording style not ruled out, §27 limits); any MSAD sentence (gate in-domain); TAD micro beside 89.56 or LaGoVAD's 0.7912 without "not matched" (both rows have different training data and the system now has two components); a streaming claim (the gate needs the whole video's mean); any DAD or Nexar sentence (no gate scores yet).

**What may follow, none authorized:** (a) score §27's gate on DAD (`DAD_s7_ncc`) and `nexar_val` (Colab; CLIP mean only) — the zero-shot dashcam reads; (b) a gate fitted on `nexar_train` (dashcam, both classes; NG5's scene / weather shortcut applies), read on `nexar_val`, DAD, TAD; needs K5's `nexar_val` per-frame scores, which `nexar_eval` does not save; (c) a causal gate on a running mean.

## 20. A clip gate fitted on `nexar_train`, applied to K5 (dated note 2026-11-11, before any number; user)

**Question (user, option (b) of §19).** §19's gate was fitted on MSAD (CCTV, 11 mostly non-traffic classes). Does a gate fitted on K5's own training corpus (Nexar dashcam, both classes) lift K5's micro in-domain and zero-shot, with macro unchanged by construction? Nothing in K5 is trained; only a 513-parameter logistic gate is fitted.

| # | Choice | Value |
|---|---|---|
| NX1 | Gate `NG` | `tad_clip_gate fit` unchanged (§27 G2: standardized mean raw CLIP, class-balanced logistic, `C` = 1.0, no tuning) on **every `nexar_train` video** (451 positive / 449 negative; `--expect-counts 451 449`), features `Nexar_s1_ncc` (every native frame). Video label = census `label`. No `nexar_val` / `nexar_test` video is fitted on |
| NX2 | Gate `MS` | §27's frozen `gate.npz` (MSAD-train), printed beside `NG` on every set |
| NX3 | Scored sets | `nexar_val` (225: 112 / 113; `Nexar_s1_ncc`), TAD-test (100; `TAD_ncc`), DAD test (466; `DAD_s7_ncc`), MSAD-test (240; `MSAD_ncc`). Same `_ncc` extractor everywhere (C2); the mean is taken at each cache's own stride |
| NX4 | K5 curves | `nexar_val`: K5 s2024 re-read by `nexar_eval --save-scores` (whole videos, protocol N-B, native frames). TAD / DAD / MSAD: the saved `core.evaluate` runs of §13 / §18 (`kim/eval/{tad,dad}/K5/s2024`, `kim_msad/eval/K5/s2024`) |
| NX5 | Gates (stop on any) | **R1** the re-read's whole micro and crop endpoint equal §13's (`kim_t2/eval/nexar/K5`) within 1e-4; **R2** every gate row count equals the curve length (Nexar: s1 rows = native frames); **R3** macro of every fusion equals K5's (|Δ| ≤ 1e-9); **R4** `NG`'s fit counts are 451 / 449 |
| NX6 | Fusion, rows | `c(v) · minmax_v(s)` (§27 G3). Per set and gate: K5 raw, min-max only, gate only, K5 × gate (micro, AP, clip AUC); clip oracle and length ruler beside |
| NX7 | Sentences | Per set: micro(K5 × `NG`) − micro(K5), video bootstrap (10,000, seed 2026): "above / below / not separated". **E1 (in-domain) = `nexar_val`**; zero-shot = TAD, DAD, MSAD-full (`NG` saw none of them). For `MS`, MSAD is in-domain: printed, no sentence |
| NX8 | Printed, never gating | micro(K5 × `NG`) − micro(K5 × `MS`) and − micro(gate only) per set; on `nexar_val` the clip AUC of `NG`, `MS` and K5's peak **within each census scene and weather value** with both classes (the NG5 shortcut: `nexar_train` Sub-urban 41 / 122, Rain 42 / 14) |
| NX9 | What may follow | **Candidate add-on** (Claude may propose it; the user decides) iff E1 is "above" **and** no zero-shot set is "below". Otherwise no claim. Never a K5 change; the gate needs the whole video's mean (offline) |

**Expectations, written before the numbers:** (a) E1 above: K5's clip AUC on `nexar_val` is 0.774, and an in-domain gate should separate videos better. (b) DAD above (dashcam, zero-shot). (c) TAD and MSAD not separated, and `MS` ahead of `NG` on both (CCTV; `MS` reached 0.851 on TAD). (d) Within-scene clip AUC of `NG` lower than its pooled one (some scene shortcut). A wrong expectation changes nothing.

**Limits.** n = 1 (one K5 checkpoint, one deterministic fit). `nexar_val` already judged K5's recipe choices (in-domain, a dev set). The gate reads video-level CLIP content, so on Nexar it can learn recording conditions that differ by class (NX8 prints it, does not rule it out).

### 20.1 Read-out (2026-11-12, K5 s2024, `NG` fitted once, `MS` = §27's frozen gate; raw `outputs/v_final/kim_gate/REPORTS/k5_clip_gate.{md,json}` + `strata_nexar_val.json`; PRELIMINARY, n = 1)

**Decision (NX9): no candidate add-on, no claim.** E1 is "below", so the add-on condition fails before the zero-shot sets are read. Nothing about K5 changes.

**Gates.** R1 passed: the `--save-scores` re-read of K5 on `nexar_val` gives whole micro **0.9000** and crop endpoint **0.8981**, the same as §13 (`kim_t2/eval/nexar/K5`: 0.9000 / 0.8981). R2 passed: the tool raises when a gate's row count differs from the curve length. R3 passed: macro drift is 0.0 for every fusion. R4 passed: `NG` was fitted on 451 / 449 videos (`gates/NG/gate.json`: 900 videos). The notebook also asserts that `nexar_train` and `nexar_val` do not overlap, and that `MS` re-scored on TAD equals §27's `c_test.json` within 1e-9.

micro(K5 × gate) − micro(K5 raw), video bootstrap 10,000 draws, seed 2026:

| set | macro | K5 raw micro | `NG`: K5 × gate | Δ vs K5 raw | sentence | `MS`: K5 × gate | Δ vs K5 raw | sentence |
|---|---:|---:|---:|---|---|---:|---|---|
| `nexar_val` (E1, in-domain for `NG`) | 0.9245 | 0.9000 | 0.8571 | −0.0429 [−0.0728, −0.0161] | **below** | 0.8842 | −0.0159 [−0.0293, −0.0014] | below |
| TAD | 0.7002 | 0.6781 | 0.7185 | +0.0404 [−0.0311, +0.1116] | not separated | 0.8692 | +0.1911 [+0.1149, +0.2693] | above |
| DAD | 0.7916 | 0.7050 | 0.6973 | −0.0077 [−0.0526, +0.0358] | not separated | 0.6832 | −0.0218 [−0.0626, +0.0196] | not separated |
| MSAD-full | 0.6810 | 0.5386 | 0.4641 | −0.0745 [−0.1714, +0.0197] | not separated | 0.8474 | +0.3088 [+0.2333, +0.3832] | in-domain for `MS`: printed, no sentence |

Printed, never gating (NX8):

| set | clip AUC: K5 peak | `NG` | `MS` | `NG` vs min-max only | `MS` vs min-max only | `MS` − `NG` (K5 × gate) | min-max only − K5 raw |
|---|---:|---:|---:|---|---|---|---:|
| `nexar_val` | 0.774 | 0.677 | 0.292 | −0.0397 [−0.0691, −0.0132] | −0.0126 [−0.0217, −0.0052] | +0.0270 [−0.0017, +0.0581] | −0.0032 |
| TAD | 0.636 | 0.283 | 0.851 | −0.0332 [−0.0701, −0.0029] | +0.1175 [+0.0712, +0.1675] | +0.1507 [+0.0931, +0.2119] | +0.0736 |
| DAD | 0.652 | 0.525 | 0.513 | −0.0785 [−0.1163, −0.0431] | −0.0926 [−0.1256, −0.0587] | −0.0141 [−0.0566, +0.0295] | +0.0708 |
| MSAD-full | 0.502 | 0.304 | 0.948 | −0.0914 [−0.1519, −0.0376] | +0.2919 [+0.2446, +0.3374] | +0.3833 [+0.3212, +0.4468] | +0.0169 |

`NG`'s clip AUC on its own training videos is **0.984** (`gate.json`: `train_clip_auc`), against 0.677 on `nexar_val`. Gate-only micro on `nexar_val`: `NG` 0.5869, `MS` 0.3784. `MS` on TAD: gate alone 0.8599, fused − gate-only +0.0093 [−0.0454, +0.0798] (§19 unchanged).

Clip AUC within census strata on `nexar_val` (NX8; strata with both classes and n ≥ 10; pooled K5 peak 0.774, `NG` 0.677, `MS` 0.292):

| stratum (abnormal / videos) | K5 peak | `NG` | `MS` |
|---|---:|---:|---:|
| scene Urban (75 / 116) | 0.771 | 0.708 | 0.308 |
| scene Highway (24 / 60) | 0.817 | 0.617 | 0.343 |
| scene Sub-urban (10 / 39) | 0.697 | 0.483 | 0.238 |
| weather Clear (66 / 146) | 0.770 | 0.674 | 0.327 |
| weather Cloudy (39 / 69) | 0.797 | 0.669 | 0.240 |
| weather Rain (7 / 10) | 0.619 | 0.571 | 0.381 |
| light Normal (103 / 204) | 0.787 | 0.674 | 0.269 |
| light Twilight (5 / 11) | 0.533 | 0.667 | 0.533 |

Strata with fewer than 10 videos (Dark 3 / 9, Industrial, Other, Rural, Bright) are in the raw report and too small to read. The census field `time_to_accident` is empty for every `nexar_val` video, so its "stratum" is the pooled set.

**Expectations, scored.** (a) E1 above: **wrong**. `NG` is below K5 in-domain, and its clip AUC (0.677) is below K5's own peak (0.774). (b) DAD above: **wrong**. It is not separated, and `NG`'s DAD clip AUC is 0.525. (c) TAD and MSAD not separated with `MS` ahead of `NG` on both: **held** (`MS` − `NG` +0.151 on TAD, +0.383 on MSAD, the latter in-domain for `MS`). (d) Within-stratum clip AUC of `NG` below its pooled 0.677: **mixed**. It is lower in Highway (0.617), Sub-urban (0.483) and Rain (0.571), and higher in Urban (0.708). No interval was computed, so the shortcut is neither shown nor excluded.

**What this licenses.**
1. **A video-level CLIP gate fitted on `nexar_train` does not raise K5's micro on any set read here.** It is below in-domain and not separated on TAD, DAD and MSAD. Against min-max alone it is below on all four sets.
2. **The gate does not transfer even within Nexar.** Training clip AUC 0.984 against `nexar_val` 0.677 means a 513-parameter mean-CLIP logistic fit separates its own 900 videos and not new ones. K5's own peak separates `nexar_val` videos better (0.774) than any gate here.
3. **A gate fitted on one corpus inverts on another.** `MS` (MSAD) reads 0.292 on `nexar_val`. `NG` (Nexar) reads 0.283 on TAD and 0.304 on MSAD. The video-level CLIP content that marks an abnormal video differs in sign between dashcam and CCTV corpora. The mechanism was not tested.
4. §19's TAD result stands as written: `MS` on TAD is "above" K5, but the gate alone carries it (fused − gate-only not separated). It is a CCTV gate on a CCTV-like set. With `NG` and `MS` both below on `nexar_val` and not separated on DAD, no clip gate is proposed for K5.

**Not licensed:** "K5 needs a video-level gate", "an in-domain gate helps", any micro from a gated row beside an ungated or published number, or "the gate learns a scene shortcut" (the strata are mixed and carry no intervals).

**Parked.** The clip-gate line for K5 is parked after §19 + §20. Reopening it needs a new dated note. Candidates not authorized: a causal or streaming gate; a gate on K5's own features instead of raw CLIP.

**Limits.** n = 1 (one checkpoint, one deterministic fit). `nexar_val` is a dev set that already judged K5's recipe. The strata are printed without intervals. Lesson candidate: pending (df).

## 21. K5's between-video baseline: micro under per-video transforms (dated note 2026-11-12, before any number; user)

**Question (user).** K5's zero-shot micro is 0.7449 on DoTA-CAP-dev and 0.7222 on DADA-2000 against macro 0.8195 and 0.7740 (`K5_ZS_REPORT.md`). Every video in both sets holds normal and abnormal frames, so micro below macro can mean that the level of one video's normal frames sits near another video's abnormal frames. Is that the cause, how much would calibrating each video recover, and what does per-video min-max do on DADA?

**Read from the code before this rule (no new run).** DoTA's micro is pooled after per-clip min-max (`zs_lagovad_readout.minmax_micro`); DADA's micro is pooled on **raw** scores (`dada_whole_eval`, `tadz_readout.micro`). `core.evaluate --score-norm auto` picks min-max when under 5 % of the videos are normal (`metrics.resolve_score_norm`, C12): DADA-2000 whole sources hold none, so `auto` would have resolved DADA to min-max, and the raw DADA micro is the odd row. Both tools write per-frame curves to `clip_scores.npz`, so this note needs no GPU.

| # | Choice | Value |
|---|---|---|
| BM1 | Data | K5 s2024's saved `clip_scores.npz`: DoTA-CAP-dev protocol B (569 two-class clips, 91 source videos) and DADA-2000 whole sources (370). LaGoVAD's public checkpoint on the same two sets is printed beside (same files as `K5_ZS_REPORT.md`). Nothing is trained or extracted |
| BM2 | Rows, all computed on the raw saved curves | **Label-free:** `raw`, per-video `min-max`, `z-score`, `minus median`, `minus q20` (0.2 quantile of the video's own scores), per-video `rank`. **Oracle (read the ground truth; ceilings, never methods):** `oracle offset` = subtract the median of the video's normal frames; `oracle normal-z` = z-score on the video's normal frames. Each is increasing inside a video, so macro cannot move |
| BM3 | Intervals | micro(row) − micro(as read), bootstrap over clusters (DoTA: source video; DADA: the video), **2,000** draws (not 10,000; POSNEG precedent), seed 2026, one set of resamples for every row. Sentence "above / below / not separated" for the label-free rows only; the oracle rows print the interval |
| BM4 | Gates (stop on any) | **G1** macro recomputed from the scores equals the recorded macro within 1e-6, and the as-read micro equals the recorded micro within 1e-4 where one is recorded (DADA 0.7222; DoTA's read-out records none: its as-read micro is printed and read against 0.7449 by hand). **G2** macro under every row equals the raw macro within 1e-9. **G3** no video is single-class (the oracles need a normal frame in every video; the tool counts what it leaves out and prints it) |
| BM5 | Headroom | micro(`oracle normal-z`) − micro(as read): **shown** iff the lower bound exceeds +0.02, else not shown. Per row, the share of this headroom it recovers is printed when the headroom's point value exceeds 0.02 |
| BM6 | Printed, never gating | as-read micro split into within-video and cross-video frame pairs (the share of pairs and the AUC of each); the std of the videos' normal-level medians against the mean normal-to-abnormal gap; AP under every row |
| BM7 | What may follow | (i) a label-free row "above" on a set: a **protocol note** for that set (which pooling its micro should be read under; the same pooling is applied to LaGoVAD's row before they sit side by side), never a model result. (ii) headroom shown: a training-side calibration idea may be proposed (the user decides); not shown: no claim. (iii) never a change to K5 |

**Expectations, written before the numbers.** (a) DADA min-max above raw. (b) DoTA: the as-read row is already min-max, so `z-score` and `rank` stay near it; `oracle normal-z` above it, headroom shown. (c) Cross-video AUC below within-video AUC on both sets. (d) On DADA `minus median` / `minus q20` recover at least half of `oracle offset`. (e) LaGoVAD's micro − macro gap is smaller than K5's on DoTA (−0.031 against −0.075) and larger on DADA (−0.081 against −0.052). A wrong expectation changes nothing.

**Limits.** n = 1 checkpoint and one seed; K5 was chosen after both sets were read. Every transform except the oracles needs the whole video (offline) and is a **protocol change**: it never sits beside LaGoVAD's published 62.60, beside a raw-pooled cell, or beside another set's micro. It is not a model improvement; K5's scores are unchanged. Not position-free: a no-pixel middle ruler reaches macro 0.7342 on DADA (`dada_readout.md`), and per-video transforms that track the accident's position are not excluded. Not licensed by any outcome: "K5 localizes better", "micro 0.8x" as a model result, "the gap is a baseline problem" without the decomposition above.

**Code.** `core/tools/k5_baseline_micro.py` (+ `core/tests/test_k5_baseline_micro.py`); `tad_clip_gate.micro_bootstrap_pairs` gained `groups=` and `draws=` (defaults bit-identical to before); constants `BASEMICRO_*`. Notebook `colab/final/k5_baseline_micro.ipynb`, CPU only, outputs `Thesis-Final/outputs/kim_baseline/REPORTS`.

### 21.1 Read-out (2026-11-12, K5 s2024 and LaGoVAD public checkpoint, saved `clip_scores.npz`, `REPORTS/k5_baseline_micro.md`; PRELIMINARY, n = 1)

Gates G1 / G2 / G3 passed (the report is written only if they do): recorded macro and micro reproduce, macro is unchanged under every row, no single-class video was left out (DoTA 569, DADA 370). Bootstrap 2,000 draws, seed 2026, clusters = DoTA source video (91) / the DADA video (370). **Nothing is adopted; every row except the two oracles is an offline protocol change, printed only.**

**What micro is on these sets (BM6).** Only **0.2 % (DoTA) and 0.3 % (DADA)** of the abnormal × normal frame pairs lie inside one video, so micro is almost entirely a cross-video AUC (DoTA 0.7448, DADA 0.7221). The within-video pair-weighted AUC is 0.8033 and 0.7584; macro (mean of per-video AUCs) is 0.8195 and 0.7740 — a different quantity, not the same one measured badly.

K5, micro (Δ against the as-read row, cluster bootstrap):

| row | DoTA-CAP-dev (as read: min-max 0.7449) | DADA-2000 (as read: raw 0.7222) |
|---|---|---|
| raw | 0.6980, −0.0469 [−0.0556, −0.0387] below | as read |
| min-max | as read | **0.7635, +0.0413 [+0.0314, +0.0514] above** |
| z-score | 0.7597, +0.0148 [+0.0047, +0.0254] above | 0.7548, +0.0326 [+0.0175, +0.0474] above |
| minus median | 0.7481, +0.0032 [−0.0099, +0.0167] not separated | 0.7424, +0.0202 [+0.0042, +0.0360] above |
| minus q20 | 0.7215, −0.0235 [−0.0321, −0.0145] below | 0.7315, +0.0093 [+0.0011, +0.0166] above |
| rank | **0.7735, +0.0286 [+0.0172, +0.0396] above** | 0.7539, +0.0317 [+0.0148, +0.0481] above |
| oracle offset | 0.7369, −0.0080 [−0.0215, +0.0063] | 0.7277, +0.0055 [−0.0094, +0.0205] |
| oracle normal-z | 0.7683, +0.0234 [+0.0103, +0.0368] | 0.7463, +0.0241 [+0.0073, +0.0407] |
| headroom (BM5, lower bound > +0.02) | **not shown** | **not shown** |

Printed (no sentence): normal-level spread of the videos' medians against the mean normal-to-abnormal gap, K5 0.2733 / 0.4021 (ratio 0.68) on DoTA and 0.2291 / 0.3433 (0.67) on DADA; LaGoVAD 0.2257 / 0.1573 (1.43) on DoTA and **0.1881 / 0.0271 (6.95) on DADA**. LaGoVAD, same rows: DoTA min-max 0.6105 → z-score 0.6217 and rank 0.6214 (above), median / q20 below or not separated, headroom not shown; DADA raw 0.5388 → min-max 0.6128 (+0.0741 [+0.0495, +0.0975]), every label-free row +0.074…+0.078, oracle normal-z +0.0677 [+0.0401, +0.0944] → **headroom shown for LaGoVAD on DADA**. AP under min-max: K5 DADA 0.5557 → 0.5640; DoTA rank 0.5790 → 0.5915; rank lowers K5's DADA AP (0.5276).

**Expectations, scored.** (a) DADA min-max above raw: **held**. (b) DoTA z-score and rank near the as-read row, oracle normal-z above it with headroom shown: **wrong** (z-score and rank are above; headroom not shown, lower bounds +0.0103 / +0.0073 < +0.02). (c) Cross-video AUC below within-video AUC on both sets: **held**. (d) On DADA the label-free offsets recover at least half of the offset oracle: **held, and empty** (the oracle itself is +0.0055, not separated from 0). (e) LaGoVAD's micro − macro gap smaller than K5's on DoTA, larger on DADA (−0.031 / −0.075, −0.081 / −0.052): **held**.

**What this licenses.**
1. **No row reaches 0.8.** The best K5 micro is rank on DoTA, 0.7735, and min-max on DADA, 0.7635. Even `oracle normal-z`, which reads the ground truth, gives 0.7683 and 0.7463; the upper bounds of its intervals are +0.0368 and +0.0407 over the as-read rows. Per-video level or scale calibration is not what holds K5's micro below 0.8. "Not shown" at a +0.02 margin is not "zero": the point estimates are +0.023 and +0.024.
2. **The between-video level offset is not the lever for K5.** The offset oracle is −0.0080 and +0.0055 (both include 0). The videos' normal levels do differ (ratio 0.68), but aligning them does not move micro. The user's premise (a frame abnormal in one video scoring below a normal frame of another) holds as a statement about cross-video pairs (cross-video AUC < within-video AUC), but its cure by baseline alignment is **not supported for K5**. It is supported for **LaGoVAD on DADA** (spread 0.188 against a gap of 0.027, headroom shown). §21's expectation that DADA was where the premise was most likely to hold was wrong for K5.
3. **BM7(i), a protocol note, user decides:** on DADA-2000 whole sources (no normal video) per-video min-max is "above" raw for both K5 (0.7222 → 0.7635) and LaGoVAD (0.5388 → 0.6128), and `core.evaluate --score-norm auto` would have chosen it (C12); the raw DADA micro of `K5_ZS_REPORT.md` is the odd row beside DoTA's min-max. If the DADA micro is restated, state the pooling and re-pool LaGoVAD's cell the same way (K5 0.7635, LaGoVAD 0.6128; no paired interval was computed between them, so no sign is written). DoTA stays on LaGoVAD's min-max; `rank` / z-score are above it for both models but are a new protocol, not proposed.
   **Decided 2026-11-12 (user):** DADA-2000 micro and AP are pooled after per-video min-max from now on, because every video is positive. Applied in `zs_compare_table.gated_reads` (`TABLE_MINMAX_SETS = (DOTA, DADA)`): the raw micro of the read-out is still checked against the scores (provenance), and the table prints the min-max cells. Restated cells: `K5_ZS_REPORT.md` LaGoVAD 0.6128 / K5 0.7635 (AP 0.3532 / 0.5640); `ZS_COMPARISON.md` LaGoVAD 0.6128, W0 0.7463, R4 0.7398 (AP 0.3532 / 0.5424 / 0.5412). Read-outs written before this date (e.g. `SA2V2_RUN.md`, W0 0.6981) keep their raw cells and their own protocol. Earlier raw cells of `ZS_COMPARISON.md`: LaGoVAD 0.5388 / AP 0.3138, W0 0.6981 / 0.5268, R4 0.7048 / 0.5363; under min-max W0 sits above R4 (0.7463 vs 0.7398), the reverse of the raw order, with no interval. `dada_whole_eval` is unchanged and still records raw micro.
4. **BM7(ii):** headroom is not shown for K5, so no training-side calibration idea is proposed. BM7(iii): no change to K5.

**Not licensed:** "micro 0.8x" or "K5 localizes better" from any row; "the gap is a baseline problem" for K5; any sentence putting a transformed micro beside a raw-pooled cell or a published number; a K5 − LaGoVAD sign under min-max on DADA.

**Inferred, not tested.** On DADA, per-video min-max (0.7635) and rank sit **above** `oracle normal-z` (0.7463) although the oracle reads the labels. Both use the video's own peak, and every video here contains an accident, so they plausibly encode the prior "this video holds an anomaly" that a stream with normal videos does not have. If so, their gain is specific to an all-abnormal benchmark (C12), not a calibration that transfers. What would test it: the same rows on a set with normal videos (TAD, DAD, MSAD saved curves exist; the gate notebooks left them), not run. Likewise untested: that micro is held down by videos whose within-video AUC is weak; the saved scores can answer it (per-video AUC against that video's share of the cross-video pairs), not authorized.

**Limits.** n = 1 checkpoint, one seed; K5 was chosen after both sets were read; the transforms are offline (whole video). Lesson candidate: pending (de).

## 22. Why K5's micro stays below 0.8 after min-max, and what K5 responds to on DADA, Nexar and DAD (dated note 2026-11-12, post-hoc; user)

**Status: post-hoc diagnostics, printed only, never gating; n = 1 (K5 s2024); nothing trained, nothing adopted.** Each read was decided after the numbers before it, so none of these is a test. The user asked (1) what holds DoTA / DADA micro below 0.8 now that both are min-max pooled (§21.1 BM7(i)), (2) what to do about it, and (3) why DAD's macro and micro are below 0.8. Saved scores only. The scripts are scratch, not `core/` tools: `micro_diag.py`, `dada_phase_diag.py` and `lag_diag.py` in the session scratchpad. They would need porting before any of these numbers is re-derived.

### 22.1 Where min-max micro loses pairs (DoTA-CAP-dev protocol B, DADA-2000 whole)

| read (min-max per video) | DoTA K5 | DoTA K5-T2 | DADA K5 |
|---|---:|---:|---:|
| micro | 0.745 | 0.712 | 0.763 |
| micro if every video ranked its own frames perfectly (K5's own values reassigned) | **0.859** | 0.804 | **0.911** |
| micro without the worst 25 % of videos (by own AUC) | 0.820 | 0.762 | 0.850 |
| share of videos with own AUC < 0.5 | 0.083 | 0.081 | 0.159 |
| share of videos whose peak lies outside the span | 0.29 | 0.31 | 0.36 |
| abnormal vs normal frames within 15 % of the length from a span boundary | 0.643 | 0.613 | 0.673 |
| abnormal vs normal frames farther away | 0.818 | 0.783 | 0.820 |
| abnormal vs normal frames after the span | 0.751 | **0.634** | 0.741 |
| early half / late half of the span vs all normal | 0.759 / 0.731 | 0.679 / 0.743 | 0.736 / 0.789 |

LaGoVAD (DoTA / DADA): micro 0.611 / 0.613; the perfect-ranking ceiling is 0.888 / 0.935; own AUC < 0.5 in 0.30 / 0.33 of the videos.

**Reading.** With the between-video offset gone (§21.1), what holds micro below 0.8 is within-video ranking. A tail of videos puts its peak outside the span, and normal frames close to the span boundaries are scored like abnormal ones. K5-T2 (§13) is worse at the boundary and holds its score after the span (mean min-max score of post-span normals 0.555 against pre-span 0.352). This fits T2's label, which runs to DADA's "abnormal end"; the cause was not tested. DoTA's span definition was not checked.

### 22.2 DADA: risk phase against collision phase

Each DADA span is split at the workbook's `accident frame`: **R** = `[abnormal start, accident)` (median 6 rows = 48 frames) and **C** = `[accident, abnormal end)` (median 6 rows). Each is read against the same video's normal frames. Gate: the stride-8 labels rebuilt from `dada标注.xlsx` equal the saved labels on 370 / 370 videos. Without the census, 110 videos have an accident row that is not unique, and they are dropped. The selection is not neutral: K5 macro is 0.765 on the 260 kept videos and 0.796 on the 110 dropped.

| arm (260 videos) | R vs normal | C vs normal | R − C per video [95 %, 2,000 video draws] | peak in R / C / after span |
|---|---:|---:|---|---|
| LaGoVAD (PreVAD) | 0.611 | 0.616 | −0.005 [−0.034, +0.023] | 0.23 / 0.14 / 0.32 |
| W0 (Nexar) | 0.712 | 0.778 | −0.066 [−0.089, −0.043] | 0.16 / 0.46 / 0.26 |
| K0 (Nexar) | 0.660 | 0.765 | −0.105 [−0.134, −0.076] | 0.14 / 0.44 / 0.30 |
| **K5 (Nexar)** | 0.743 | 0.783 | **−0.040 [−0.061, −0.018]** | 0.18 / 0.43 / 0.26 |

K5's mean min-max score, from 6 rows before to 6 rows after the accident row: 0.45 0.47 0.48 0.51 0.54 0.56 **[0.57]** 0.58 0.60 0.60 0.59 0.55 0.49. The peak falls 2–3 rows (16–24 frames) after the accident.

**Claude's hypothesis of 2026-11-12, that K5 fires early because Nexar's span starts at `t_alert`, is refuted on DADA.** Every Nexar-trained arm scores C above R. **K5-T2 does not test it either:** T2's span also starts before the accident (DADA start → accident median 1.50 s, Nexar alert → event 1.43 s; `NEXAR_SETUP.md`), and D-N1 copied its post-event 1.4 s from DADA. Changing corpus therefore did not change the label definition.

### 22.3 Nexar (in-domain): where K5 peaks relative to `t_event`

`nexar_val` positives (112), per-frame scores from §20's `--save-scores` re-read. Gate: the saved span equals `[t_alert, t_event + 1.4 s]` rebuilt from `nexar_n0/census.json` within one frame at both ends on every positive (fps = scored frames / `duration_s`, as in `nexar_eval.crop_bounds`).

| read | value |
|---|---|
| peak time − `t_event`, median [95 %] | **+0.45 s [+0.32, +0.60]** (local search from `t_alert` − 2 s to `t_event` + 4 s; the whole-video argmax gives +0.44) |
| share of peaks after `t_event` / after the span end (`t_event` + 1.4 s) | 0.78 / 0.08 |
| R = `[t_alert, t_event)` vs normal, C = `[t_event, t_event + 1.4 s]` vs normal | 0.888 / 0.953; R − C per video −0.049 [−0.072, −0.026] |
| mean min-max score: pre-span / R / C / post-span | 0.037 / 0.429 / 0.738 / 0.047 |

Aligned mean profile, in 0.25 s bins: 0.12 at −2.0 s, 0.44 at −1.0 s, 0.62 at −0.5 s, **0.81 at +0.25 s**, 0.66 at +1.0 s, 0.37 at +1.5 s, 0.21 at +2.0 s. The label runs from about −1.43 s (median alert) to +1.4 s. The curve crosses 0.5 near −0.85 s on the way up and near +1.3 s on the way down.

**Reading.** In-domain there is no long aftermath tail (post-span 0.047), but **the onset is late.** The rise starts about 0.6 s after the label starts, and the peak lands about 0.45 s after the event, so the first part of the span (R) is underscored. The same shape appears zero-shot on DADA (§22.2). Suspected source, not tested: the causal 1.5 s VideoMAE window sees motion only after it has begun. The temporal encoder and score head read in both directions, so they are not the obvious cause.

### 22.4 DAD: does K5 respond at the collision rows?

DAD has 15 rows; the accident rows are 13–14 (`toa` = 90, stride 7). Read: the rise from row 12 to row 14 in positive videos against negative videos. A collision response would make the positives rise more.

| arm | rise 12→14, pos − neg [95 %] | AUC of the 12→14 rise, pos vs neg | rise 10→12 (before the accident), pos / neg |
|---|---|---:|---|
| K5 (Nexar) | +0.020 [−0.006, +0.047] | 0.529 | +0.030 / +0.000 |
| K5-T2 | +0.004 [−0.028, +0.036] | 0.467 | +0.013 / +0.012 |
| W0 (Nexar) | +0.003 [−0.016, +0.022] | 0.494 | +0.032 / +0.006 |
| LaGoVAD | **+0.006 [+0.003, +0.009]** | 0.565 | +0.006 / +0.001 |

**Reading.** K5 shows no collision-row response separable from the negatives. Its positives rise as much **before** the accident (rows 10→12) as during it. The positive − negative gap builds steadily from row 0 (0.017 → 0.179, §21.1 DAD read), and K5-T2 has the same shape with a larger gap (0.017 → 0.334). Only LaGoVAD shows a separable rise at the collision rows, and it is tiny (+0.006). The guess Claude made earlier the same day, that DAD's 0.5 s post-accident tail ends before K5's +0.45 s response builds, does not fit: if it were right, the rise at rows 13–14 would at least be starting. What K5 separates on DAD is which videos are risky, not when the collision happens. DAD macro still rides the time ramp (`ZS_DAD.md` §8) and DAD micro is still mostly clip classification (clip oracle 0.839). **No read here licenses chasing DAD 0.8.**

### 22.5 What this licenses, and what may follow (user decides; nothing authorized)

**Licensed (post-hoc, n = 1):** "K5's min-max micro on DoTA-CAP-dev and DADA is limited by within-video ranking: a tail of videos peaks outside the span, and frames near the span boundaries are confused. On Nexar and DADA, K5's response peaks after the event (Nexar +0.45 s) and underscores the risk phase before it. On DAD, K5 shows no response at the collision rows." **Not licensed:** "K5 anticipates accidents", "the Nexar label causes the DAD gap", "a centred window would fix micro" (untested), or any number from these reads beside a published one.

Candidates (each needs its own dated note before any number):
- **(a) Offline lag shift, fitted in-domain (CPU).** Fit one shift δ on `nexar_val` only (e.g. the δ that maximises within-video AUC there; the +0.45 s above is post-hoc), then read DoTA-CAP-dev and DADA once with δ fixed. This is an **offline protocol change** (it uses future frames) and is never a model result.
- **(b) Centred VideoMAE window instead of causal (GPU).** Re-extract and retrain K5, 2 seeds. Inference becomes non-causal (needs about 0.75 s of future), and the change has to be weighed against online use.
- **(c) Characterise the failing tail (CPU).** Per-video AUC by DADA type, DoTA category, length and light, to see whether the 16 % / 8 % of videos with own AUC < 0.5 share a cause.
- **Closed or not recommended:** per-video calibration (§21), clip gates (§19 / §20), retraining with a span that starts at `t_event` (§22.2 removed its premise), adding a position prior (DF8).

**Limits.** n = 1 checkpoint, one seed. The DADA phase split drops 110 videos non-neutrally. DADA's frame rate (30 fps) is assumed from the literature (A1), so the seconds quoted for DADA are approximate. `nexar_val` is a dev set that already judged K5's recipe. The scratch scripts are not committed.

## 23. Offline lag shift fitted on `nexar_val`, read once on DoTA-CAP-dev and DADA-2000 (dated note 2026-11-12, before any number; user)

**Question (user, §22.5 candidate (a)).** §22.3 found K5's in-domain response peaking about +0.45 s after `t_event`, with a late onset. If K5's curve is advanced by a lag δ fitted **on `nexar_val` only**, does its within-video ranking on the zero-shot sets improve? How large such a gain is decides whether the costly fix, a centred VideoMAE window (§22.5 (b)), is worth proposing. Nothing is trained; saved scores only (CPU).

| # | Choice | Value |
|---|---|---|
| LS1 | Shift | Advance each curve by δ seconds: `s_δ(i) = s(i + δ·r)` by linear interpolation over the step index, clamped to the last step (edge value repeated). `r` = steps per second: Nexar `len(curve) / duration_s` (native frames, as `nexar_eval.crop_bounds`), DoTA-CAP-dev `DOTA_FPS` = 10 (protocol B scores native frames), DADA-2000 `DADA_ASSUMED_FPS / FRAME_STRIDE` = 30 / 8 (stride-8 rows; A1). δ = 0 is the identity |
| LS2 | Grid | δ ∈ {0.00, 0.05, …, 1.50} s (31 values), fixed here |
| LS3 | Fit (in-domain only) | δ* = argmax over the grid of K5's **whole-video macro on the 112 `nexar_val` positives** (§20's `--save-scores` re-read); ties within 1e-9 → the smallest δ. DoTA and DADA play no part in choosing δ* |
| LS4 | Read | K5 s2024 saved scores, once at δ*: DoTA-CAP-dev protocol B (569 clips, clusters = 91 source videos) and DADA-2000 whole (370 videos). Macro: per-video paired Δ (δ* − 0), cluster bootstrap. Micro: per-video min-max pooled (§21.1 BM7(i)), Δ by `micro_bootstrap_pairs` over the same clusters. 2,000 draws, seed 2026. Sentence "above / below / not separated" for macro and micro on each set |
| LS5 | Gates (stop on any) | **G1** at δ = 0: Nexar whole macro equals `nexar_readout.json` (0.9245) within 1e-4; DoTA and DADA macro equal their read-outs within 1e-6; DADA raw micro equals 0.7222 within 1e-4. **G2** every shifted curve keeps its length; δ = 0 reproduces the curve bit for bit. **G3** at least one positive on `nexar_val` has a census row; every positive does, or the tool raises |
| LS6 | Printed, never choosing δ | The full grid on `nexar_val` (macro), and on DoTA / DADA (macro and min-max micro at every δ: shows whether δ* is near the zero-shot optimum, **post-hoc**); the `nexar_val` pooled raw micro at δ* |
| LS7 | What may follow | If macro is "above" on **both** zero-shot sets at δ*: the lag is a transferable cost, and Claude may propose §22.5 (b) (a centred window, GPU, its own note; the user decides). If "above" on one: report it, propose nothing. Otherwise: (b) loses its motivation and is recorded as not supported. Never a K5 change or a model result; never beside a published number |

**Expectations, written before the numbers.** (a) δ* lies in [0.3, 0.6] s (§22.3's peak +0.45 s). (b) DADA macro "above" (§22.2: K5 peaks 2–3 rows after the accident). (c) DoTA macro "above" but smaller (§22.1: on DoTA the late half of the span is the weaker one, so advancing could hurt the end of the span). (d) Min-max micro moves in the same direction as macro on both sets. A wrong expectation changes nothing.

**Limits.** n = 1 checkpoint. `nexar_val` is a dev set that already judged K5's recipe, and §22.3 looked at it, so δ* is fitted on a set whose lag was already seen. That is acceptable for an in-domain fit, but the zero-shot sets must not be looked at to choose δ (LS6 prints them only). The shift uses δ seconds of future frames, so it is an **offline protocol change**. Seconds on DADA rest on A1 (30 fps). A position effect is not excluded: advancing a curve moves its mass earlier in the video, and DoTA / DADA spans sit mid-video (§14.5.1 position probes).

**Code (to build after this note).** `core/tools/k5_lag_shift.py` + `core/tests/test_k5_lag_shift.py`; constants `LAGSHIFT_*`; output `outputs/v_final/kim_lag/REPORTS/k5_lag_shift.{md,json}`.

### 23.1 Read-out (2026-11-12, K5 s2024, saved scores; raw `outputs/v_final/kim_lag/REPORTS/k5_lag_shift.{md,json}`; PRELIMINARY, n = 1)

Gates G1 / G2 / G3 passed (the tool raises otherwise). At δ = 0 the scores reproduce Nexar whole macro 0.9245, DoTA macro 0.8195, DADA macro 0.7740 and DADA raw micro 0.7222. Every `nexar_val` video has a census row. Run on CPU, about 70 s.

**Fit (LS3).** **δ\* = 0.25 s.** `nexar_val` whole macro goes 0.9245 → 0.9289 (+0.0044) and pooled raw micro 0.9000 → 0.9049. The grid curve is flat near its top (0.9284–0.9289 for δ from 0.15 to 0.25 s) and falls steadily past 0.4 s (0.7983 at 1.5 s).

**Zero-shot at δ\* (LS4), cluster bootstrap 2,000 draws, seed 2026:**

| set | shift | macro δ = 0 → δ* | Δ macro [95 %] | sentence | min-max micro δ = 0 → δ* | Δ micro [95 %] | sentence |
|---|---:|---|---|---|---|---|---|
| DoTA-CAP-dev (569 / 91 clusters) | 2.5 frames | 0.8195 → 0.8136 | −0.0059 [−0.0107, −0.0013] | **below** | 0.7449 → 0.7372 | −0.0077 [−0.0109, −0.0047] | below |
| DADA-2000 (370) | 0.94 rows | 0.7740 → 0.7773 | +0.0033 [−0.0011, +0.0081] | not separated | 0.7635 → 0.7649 | +0.0014 [−0.0019, +0.0049] | not separated |

**LS7: macro is not above on either set, so §22.5 (b) (a centred VideoMAE window) is not supported by this read.**

**Printed (LS6, post-hoc, never choosing δ).** The best grid point on the zero-shot sets is DoTA δ = 0.05 s (macro 0.8197, +0.0002) and DADA δ = 0.15 s (0.7799, +0.0059). Even the shift chosen by looking at each test set buys at most +0.006 of macro.

**Expectations, scored.** (a) δ\* in [0.3, 0.6] s: **wrong** (0.25 s, against §22.3's peak offset of +0.45 s). (b) DADA macro above: **wrong** (not separated). (c) DoTA macro above but smaller: **wrong** (below). (d) Micro moves with macro: **held**.

**What this licenses.**
1. **A uniform time offset is not what costs K5 its within-video ranking.** Fitted in-domain, the shift gains +0.004 on `nexar_val`, costs DoTA 0.006 and does not separate on DADA. No offset on the grid is worth more than about +0.006 on either zero-shot set.
2. §22.3's "peak +0.45 s after the event" and "late onset" stand as descriptions, but they do not translate into a correctable lag. The curve is broad (on Nexar it takes about 2 s to rise from 0.12 to its peak and about 1.5 s to fall back). Advancing it gains at the onset and loses about as much at the end of the span. **Inferred, not tested:** the boundary loss of §22.1 is a matter of width (temporal blur), not offset. A centred window shortens the delay; it does not obviously narrow the response.
3. Do not quote any δ ≠ 0 row as K5's performance. It is an offline protocol and was read once.

**Not licensed:** "K5 has no lag", "a centred window would not help" (it was not built; only its main motivation is removed), or "sharpening would help" (untested).

**Limits.** n = 1. δ\* was fitted on a dev set whose lag §22.3 had already looked at. The seconds on DADA rest on A1. Position is not excluded (LS limits).

## 24. K5's failing tail: are the videos it ranks wrongly concentrated anywhere? (dated note 2026-11-12, before any number; user)

**Question (user, §22.5 candidate (c)).** §22.1 found that 8 % of DoTA-CAP-dev clips and 16 % of DADA-2000 videos have their own AUC below 0.5 under K5. Do these failures share a recorded attribute (night, ego involvement, accident type, scene, span geometry), and are they the same videos LaGoVAD fails on? If they cluster, that is a lead for a targeted fix. If they do not, K5's within-video ranking is probably near its ceiling for this model, and the micro line should stop. Saved scores and annotations only; nothing trained.

| # | Choice | Value |
|---|---|---|
| FT1 | Items | K5 s2024 saved scores: DoTA-CAP-dev protocol B (569 clips, clusters = source video) and DADA-2000 whole (370 videos). **Fail** = own frame AUC < 0.5 |
| FT2 | Categorical strata | DoTA (per-clip `raw_annotations/annotations/<id>.json`): `ego_involve`, `night`, `accident_name`. DADA (`dada标注.xlsx`): `light` (day / night), `weather`, `scenes`, `linear`; **ego** = the clip's `texts` names "ego-car"; **actor** = first of pedestrian / cyclist / motorbike / vehicle named in `texts`, else other. A stratum is read only with ≥ 20 items; smaller ones are pooled into "other" or printed as too small |
| FT3 | Continuous covariates | video length (steps), span share, span centre (t/T), span length in seconds (DoTA 10 fps; DADA A1); DADA also the risk-phase length (start → accident) |
| FT4 | Reads | Per stratum: n, fail share, mean own AUC; Δ fail share (stratum − rest), cluster bootstrap 2,000, seed 2026. Per covariate: AUC of the covariate separating failing from other items (Mann–Whitney), bootstrap CI. Peak location of the failing items: before / inside / after the span. Shared difficulty: Spearman ρ of per-item AUC, K5 against LaGoVAD (public checkpoint), and LaGoVAD's fail share among K5's failures against its overall fail share |
| FT5 | Lead (written before the numbers) | A **categorical lead** is a stratum with ≥ 20 items whose fail share is ≥ 2 × the set's overall share **and** whose Δ interval excludes 0. A **covariate lead** is a covariate whose AUC interval excludes 0.5 with \|AUC − 0.5\| ≥ 0.10. About 25 strata and covariates are read, so roughly one false lead at 95 % is expected by chance. A lead is a lead, not a finding |
| FT6 | What may follow | Leads found: Claude describes them and may propose one targeted check (the user decides). No lead: "K5's failures are not concentrated in any recorded attribute", and Claude recommends closing the micro line and writing it up. Never a K5 change |

**Expectations, written before the numbers.** (a) Night fails more on both sets. (b) Ego-uninvolved (DoTA `ego_involve` false, DADA no "ego-car") fails more: the motion of a third party is smaller in the frame than ego motion. (c) Short spans (span length in seconds) fail more. (d) K5's and LaGoVAD's per-item AUC correlate positively (ρ > 0.2): some videos are hard for any model. A wrong expectation changes nothing.

**Limits.** n = 1 checkpoint. Descriptive only, with many comparisons. The strata are the datasets' own annotations, not checked by eye. DADA's ego flag comes from a free-text field (`texts`, 83 strings).

**Code (built after this note).** `core/tools/k5_fail_tail.py` + `core/tests/test_k5_fail_tail.py`, constants `FAILTAIL_*`, output `outputs/v_final/kim_tail/REPORTS/k5_fail_tail.{md,json}`.

### 24.1 Read-out (2026-11-12, K5 s2024 and LaGoVAD public checkpoint, saved scores; raw `outputs/v_final/kim_tail/REPORTS/k5_fail_tail.{md,json}`; PRELIMINARY, n = 1)

K5 fails on 47 / 569 DoTA-CAP-dev clips (0.083; 91 clusters) and 59 / 370 DADA-2000 videos (0.159). Every DoTA clip has its raw annotation, and every DADA video has its workbook row.

**Leads by FT5: two, both at the edge of their bars.**
- DoTA `accident_name = leave_to_left`: n = 35, fail share 0.200 against 0.083 overall, Δ +0.125 [+0.003, +0.250]. The lower bound is 0.003 above zero.
- DoTA clip length: failing clips are longer (107 against 99 frames), AUC 0.603 [0.506, 0.695]. That is 0.003 past the 0.10 margin, and the lower bound is 0.006 above 0.5.

About 30 strata and covariates were read, so about one false lead is expected by chance. Neither lead replicates on DADA: length there is 0.557 [0.480, 0.633], and DADA has no lane-departure stratum.

**The most consistent pattern is not a lead by the 2× rule: accidents that do not involve the ego car.**

| set | ego-uninvolved fail share (n) | ego-involved fail share (n) | Δ [95 %] | mean own AUC, uninvolved / involved |
|---|---|---|---|---|
| DoTA-CAP-dev (`ego_involve`) | 0.129 (248) | 0.047 (321) | +0.082 [+0.034, +0.135] | 0.753 / 0.871 |
| DADA-2000 (`texts` without "ego-car") | 0.178 (275) | 0.105 (95) | +0.073 [−0.004, +0.147] | 0.753 / 0.835 |

The direction is the same on both sets, and on DoTA the interval excludes 0. The ratio to the overall share (1.55, 1.12) is below FT5's 2×, so by the rule it is not a lead. Recorded here as a printed pattern, not a finding.

**Where failing items peak.** DoTA: before the span 0.447, inside 0.191, after 0.362. DADA: before 0.441, inside 0.085, after 0.475. A failing curve is not flat: it peaks somewhere else, before or after the accident about equally often.

**Shared difficulty.** On DoTA, Spearman ρ(K5, LaGoVAD per-item AUC) = 0.134, and LaGoVAD fails on 0.319 of K5's failures against 0.295 overall, so K5's DoTA failures are mostly its own. On DADA ρ = 0.227, and LaGoVAD fails on 0.492 of K5's failures against 0.330 overall, so part of DADA's failing tail is hard for both models (cause not read: annotation, visibility or content).

**No effect, intervals including 0:** night or light (DoTA Δ +0.027, DADA +0.008), weather, span share, span centre, span seconds, DADA risk-phase length, DADA scene and road type (DADA `linear = 4` fails less, −0.093 [−0.168, −0.014], a single cell).

**Expectations, scored.** (a) Night fails more: **wrong** on both. (b) Ego-uninvolved fails more: **held in direction on both sets** (DoTA interval excludes 0, DADA borderline), not a lead by FT5. (c) Short spans fail more: **wrong**. (d) ρ > 0.2: **held on DADA** (0.227), **wrong on DoTA** (0.134).

**What this licenses.**
1. K5's failing tail is **not concentrated in any recorded attribute strongly enough to target.** The two formal leads are at the edge of both bars and do not replicate across sets.
2. The one pattern consistent across sets is that **accidents without the ego car fail about 1.5× as often** (DoTA). This fits frame-global features: a distant third-party collision moves few pixels compared with ego motion. That reading is inferred, not tested. Its fix would need spatial or object evidence, and the token line (`SA2V2_RUN.md` §12) and KIM's spatial forms (§14.1) are closed.
3. On DoTA, K5's failures are its own; on DADA, about half are shared with LaGoVAD.

**FT6.** Leads were found, so Claude may propose one targeted check; the user decides. Claude's recommendation, given that both leads are marginal, is to **close the micro line and write it up**, stating the ego-uninvolved pattern as a limitation. If one check is wanted, the cheapest is to look by eye at DADA's ego-uninvolved failing videos (frames exist on Drive; DoTA pixels do not) with their curves.

**Not licensed:** "K5 fails on lane departures", "longer clips fail", "K5 cannot see third-party accidents" (1.5×, one set separated), or any stratum number beside a published one.

**Limits.** n = 1; many comparisons; DADA's ego flag comes from free text; strata were not checked by eye.

### 24.2 Post-hoc (2026-11-12, user's question): did the SA²-v2 token arms help ego-uninvolved accidents?

Printed only, n = 1. Per-item AUC from each arm's `clip_aucs.json` (on W0D DoTA these equal the AUCs recomputed from `clip_scores.npz` to 1e-9; SV4 has no saved curves). Arm − W0D (the SA²-v2 control) per item, split by §24's ego flag; cluster bootstrap 2,000, seed 2026.

| set | stratum (n) | W0D macro | SV2 − W0D | SV3 − W0D | SV4 − W0D |
|---|---|---:|---|---|---|
| DoTA-CAP-dev | ego-involved (321) | 0.884 | −0.038 [−0.051, −0.026] | −0.028 [−0.038, −0.018] | −0.012 [−0.025, −0.001] |
| | **ego-uninvolved (248)** | 0.727 | −0.021 [−0.043, +0.002] | −0.031 [−0.051, −0.012] | +0.002 [−0.012, +0.015] |
| DADA-2000 | ego-involved (95) | 0.808 | −0.022 [−0.046, +0.001] | −0.013 [−0.037, +0.011] | +0.005 [−0.015, +0.026] |
| | **ego-uninvolved (275)** | 0.734 | −0.045 [−0.065, −0.025] | −0.028 [−0.045, −0.012] | −0.020 [−0.037, −0.004] |

**No SA²-v2 arm improves ego-uninvolved accidents on either set.** On DADA every arm costs there. On DoTA, SV3 costs there, and SV2 and SV4 are not separated from W0D. This agrees with `SA2V2_RUN.md` §7.2 / §11.5: SV2's map does not point at non-ego agents (hit − chance −0.006, against +0.283 for ego accidents). The ego gap is a property of the host, not of K5: W0D shows it too (0.884 against 0.727 on DoTA).

**Correction to §24.1 item 2.** "Its fix would need spatial or object evidence" overstates what is open. **Every spatial attempt made here** has failed to beat the frame-global motion prior: 7×7 tokens (SA²-v2, this table, `SA2V2_RUN.md` §12.2), crops (`LOCAL_STEP1.md` §7, `LOCAL_STEP1B.md` §5) and KIM's spatial forms (§14.1). What has not been tried is object-level evidence (detector boxes) or native-resolution input. Those are untested and costly, and `SA2V2_RUN.md` §12.2 does not license any claim about them either way.

## 25. K5ln: per-step LayerNorm on the motion columns (dated note 2026-11-14, before any number; user)

The user asked whether the motion stream's fixed Nexar-train statistics (`m_u`, `σ_u`, `c`, fitted once on Nexar-train and baked into the A2 cache) hurt zero-shot transfer, and chose to test a per-step LayerNorm. This is a new question. It does not reopen §12 or any closed line, and it is not a repair of anything.

**What is already known (written before the number).** (1) `m_u` and `σ_u` cannot over-fit by themselves: `W_u` is linear, so `W_u·c(u − m_u)/σ_u + b` reparametrizes to `W′u + b′`. A different constant changes the conditioning of the optimizer, not the family of functions. (2) Normalizing per video instead (CRN on the motion stream, A3) has already been measured on Nexar and costs on every set, −0.06 … −0.08 (`RESULTS_NEXAR.md` §4). Its training `mil` is about 10× tighter, which is a shortcut signature. (3) Architecture §5 rejected a per-token LayerNorm because it would erase the per-step deviation magnitude. That argument was made for CRN's stream; K5 is CRN-off, so on this host it is a hypothesis, not a measurement.

- **Arm K5ln = K5 s2024 + `model.motion_norm=layernorm`.** The baked rows' motion columns become `c · LN(u_t)`: LayerNorm over the 384 channels of each step, with no affine (`W_u` absorbs any gain or shift), rescaled by the cache's `c` (`motion_residual.norm_scale`, a buffer read from the input manifest at train time and carried in the checkpoint). The CLIP columns, `W_u` (zero-init), KIM, DVS off, kernel 3, s2024 and 1,749 steps are all exactly as K5. **The config differs from K5 in that one key** (the runbook asserts this).
- **Why rescale by `c`:** LN brings the per-channel RMS to 1, while the baked stream sits at about `c`. Without the rescale, every Adam step on `W_u` would move the residual about 1/c times further, and the arm would be an LR arm in disguise (`WU_ARMS.md`).
- **What LN removes:** each step's mean over channels and its norm ‖ũ_t‖. The second is "how unusual this step's motion is against Nexar-train". It may be the domain shift that hurts zero-shot, or it may be the signal. This is exactly the question.
- **Start check:** `W_u` = 0 at step 1, so the forward is identical to K5's. Step-1 `mil` within `KIM_START_TOL` (1e-4) of K5's = same start. A mismatch flags the contrast; it does not stop the run.
- **Control:** K5 s2024, whose checkpoint and reads are on Drive. It is not retrained.
- **Read:** DoTA-CAP-dev (protocol B), DADA-2000 whole video (370 sources), TAD. Paired per video, `kim_readout` with `--control K5`, cluster bootstrap 2,000, seed 2026, margin `V5_MIN_GAIN` = +0.01 (the same sentence rule as every KIM read).
- **Decision, fixed now (`ln_rule`):**
  - K5ln − K5 **gain on DADA and no cost on DoTA-CAP-dev** → K5ln s2025 against K5 s2025. Nothing is adopted at n = 1.
  - **Anything else → the LayerNorm line closes.** A DoTA-only gain does not count (§6.2). No affine variant, no un-rescaled variant, no LN on the CLIP columns, and no sweep follow.
  - At s2025: a gain on DADA → "reproduced, still nothing adopted"; anything else → closed.
- **Printed, never decides:** K5ln − W0; TAD; `motion_share` (ρ_u) and ‖W_u‖; and **the per-step norm ‖ũ_t‖ of the baked motion columns on each corpus** (Nexar, all baked ids; DoTA-CAP-dev; DADA; TAD). This is what LN erases: if the medians barely differ across corpora, LN removes no domain shift and only the magnitude.
- **Prediction (written before the number):** K5ln − K5 on DADA "no difference shown", with a cost more likely than a gain, because the per-step magnitude is part of the signal on a CRN-off host. Per-step norm medians differ by less than 20 % across corpora.

Code: `core/models/motion_residual.py` (`norm`, `set_norm_scale`, `normalize`), `core/config.py` (`model.motion_norm`, validated), `core/train.py` (scale from the manifest), `core/tools/kim_readout.py` (`ln_rule`, `start_check`, `--motion-cache`). Tests: `core/tests/test_motion_norm.py` and `core/tests/test_kim_tools.py::TestLayerNormReadout`. Runbook: `colab/final/k5_ln.ipynb`.

### 25.1 Read-out (2026-11-14, s2024; raw `outputs/v_final/kim/REPORTS/kim_readout_k5ln_s2024.{md,json}`; PRELIMINARY, n = 1)

Run checks: checkpoint step 1,749 (`verified.json`, J10); config differs from K5 in `model.motion_norm` only; start check step-1 `mil` K5ln 0.687794 = K5 0.687794 (gap 0, matched).

| set | K5 | K5ln | K5ln − K5 [95 %] | sentence |
| --- | --- | --- | --- | --- |
| DoTA-CAP-dev macro (protocol B) | 0.8195 | 0.8175 | −0.0020 [−0.0058, +0.0020] | no difference shown |
| DADA-2000 macro (370, whole) | 0.7740 | 0.7767 | +0.0027 [−0.0030, +0.0086] | no difference shown |
| TAD macro (printed) | 0.7002 | 0.6954 | −0.0048 [−0.0227, +0.0126] | no difference shown |

**`ln_rule`: no gain on DADA → the LayerNorm line CLOSES.** No s2025, no affine variant, no un-rescaled variant, no LN on the CLIP columns, no sweep (fixed in §25 before the number).

Printed, never decides:
- K5ln − W0 DADA +0.0153 [+0.0020, +0.0285] ("gain shown"), while K5 − W0 is +0.0125 [−0.0006, +0.0253] on the same seed. The difference between the two (+0.0027) is K5ln − K5, which shows nothing; the K5ln − W0 sentence is K5 − W0's interval shifted across +0.01 by noise, not an effect of LN. Never quote it as "LN gives a gain over W0".
- DoTA-CAP-dev non-ego: K5ln − K5 −0.0055 [−0.0134, +0.0021]; ego +0.0007.
- TAD micro K5 0.6781 → K5ln 0.7070 while TAD macro falls; TAD micro measures clip classification (§14.5.1) and decides nothing.
- Per-step norm ‖ũ_t‖ medians: Nexar 7.68, DoTA-CAP-dev 8.30 (+8 %), DADA 8.08 (+5 %), TAD 9.43 (+23 %). The zero-shot corpora sit only slightly above Nexar-train except TAD. The prediction "< 20 % across corpora" **failed on TAD** (+23 %) and held on DoTA-CAP-dev and DADA.
- Training: K5ln ends at a lower last-epoch `mil` (0.1572 vs 0.1637) and a higher `kim_anchor_share` (0.942 vs 0.904); it fits Nexar slightly tighter without transferring better.

**Reading.** Erasing the per-step magnitude neither helps nor hurts on any held-out set at n = 1: the fixed Nexar-train statistics are not what limits K5's zero-shot transfer, at least not through the per-step norm. The prediction's direction ("a cost more likely") was not borne out either: DADA's point estimate is +0.0027, inside the noise. K5 stays the main model unchanged.

### 25.2 Macro and micro on all four sets (dated note 2026-11-14, after §25.1; user; printed, decides nothing)

The user asked for macro **and** micro of K5ln on DoTA-CAP-dev, DADA-2000, TAD and DAD. §25.1 had no DoTA micro (`kim_readout` records none), a raw-pooled DADA micro, and no DAD read. This note adds them; it cannot reopen §25, which decided on macro before any number.

- **Read added:** K5ln on DAD (whole video, s7, raw, `core.evaluate --save-scores`), as `k5_dad.ipynb` read K5.
- **Pooling of micro / AP:** DoTA-CAP-dev and DADA-2000 after per-video min-max (every item holds an accident; §21.1, user's decision 2026-11-12); TAD and DAD raw (they hold normal videos; their micro measures clip classification, clip oracle printed).
- **Contrasts:** K5ln − K5 and W0 − K5 where W0's scores exist, paired per item. Macro: `delta_ci` (the read-outs' bootstrap). Micro: cluster bootstrap of the pooled AUC (2,000 draws, seed 2026; DoTA clusters = source video). Three-way sentence on the interval (above / below / not separated), no margin.
- **Tool:** `core/tools/zs_arm_table.py` (+ `core/tests/test_zs_arm_table.py`); it refuses rows that scored other items or labels and a score file whose as-read micro disagrees with its read-out. Local check on K5 / W0: K5 0.7449 (DoTA min-max), 0.7635 (DADA min-max), 0.6781 (TAD), 0.7050 (DAD) reproduce `K5_ZS_REPORT.md`; W0 DADA min-max 0.7463 reproduces `ZS_COMPARISON.md`.
- **Runbook:** `colab/final/k5_ln.ipynb` step 5 (DAD read) and step 7, report `REPORTS/zs_arm_table_k5ln_s2024.{md,json}`.

#### 25.2.1 Read-out (2026-11-14, s2024; raw `outputs/v_final/kim/REPORTS/zs_arm_table_k5ln_s2024.{md,json}`; PRELIMINARY, n = 1, printed only)

Macro / micro (DoTA-CAP-dev, DADA min-max; TAD, DAD raw):

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5 | 0.8195 / 0.7449 | 0.7740 / 0.7635 | 0.7002 / 0.6781 | 0.7916 / 0.7050 |
| K5ln | 0.8175 / 0.7422 | 0.7767 / 0.7668 | 0.6954 / 0.7070 | 0.7718 / 0.6980 |
| W0 | 0.8110 / 0.7255 | 0.7615 / 0.7463 | 0.6983 / 0.6034 | 0.7317 / 0.6693 |

| contrast | set | Δ macro | Δ micro |
| --- | --- | --- | --- |
| K5ln − K5 | DoTA-CAP-dev | −0.0020 [−0.0058, +0.0020] n.s. | −0.0027 [−0.0074, +0.0022] n.s. |
| K5ln − K5 | DADA | +0.0027 [−0.0030, +0.0086] n.s. | +0.0034 [−0.0019, +0.0087] n.s. |
| K5ln − K5 | TAD | −0.0048 [−0.0227, +0.0126] n.s. | **+0.0289 [+0.0073, +0.0547] above** |
| K5ln − K5 | DAD | **−0.0198 [−0.0359, −0.0054] below** | −0.0070 [−0.0141, +0.0002] n.s. |
| W0 − K5 | DoTA / DADA / TAD / DAD | n.s. / n.s. / n.s. / −0.0599 below | −0.0194 / −0.0172 / −0.0747 / −0.0357, all below |

**Reading.** §25 stays closed; nothing here could reopen it. LN changes nothing on the two all-abnormal sets under either pooling. It **costs DAD macro** (−0.020, interval excludes 0), the one new read, which is the direction §25 predicted (magnitude is signal); DAD macro is not clean localization (`ZS_DAD.md` §8: every positive is abnormal on the same last rows). The TAD micro gain (+0.029) sits beside a clip oracle of 0.9226 and a macro that does not move: a clip-level ranking change, not localization. Never write "LN helps TAD".

**Printed beside, post-hoc:** W0 − K5 micro is below on all four sets (−0.017 … −0.075), while macro separates only on DAD. This is one seed, K5 was chosen after its reads, and the contrast is not part of any rule: quote it as "K5's micro is above W0's on all four sets at s2024 (n = 1)", never as a K5 adoption result.

## 26. K5k2: MIL top-k 4 → 2 rows per window (dated note 2026-11-14, before any number; user)

**Question (user).** On a 20-row Nexar window, `mil_topk_pct = 5` gives k = 20 // 5 = 4 rows (`core/losses/mil.py:22`; the key is a divisor, pending (cv)). Are 4 rows too many, so that a positive bag's top-k mean takes in normal rows and the abnormal bag is diluted toward the normal ones?

**Measured before this rule (annotation only, no score).** A window is positive iff it holds ≥ 1 abnormal row (`WINDOW_MIN_POSITIVE = 1`). The span is `t_alert … t_event + 1.4 s` (D-N1), at s8, windows of 20 rows with hop 8. Over Nexar's 750 clean positives (the census `outputs/v2/REPORTS/nexar_n0/census.json`; all splits, so `nexar_train` may differ slightly), the result is 2,841 positive windows. The median positive window holds **8** abnormal rows (mean share 0.37). Positive windows holding fewer abnormal rows than k:

| k (pct) | share of positive windows |
|---|---|
| 4 (5, K5) | **0.205** |
| 2 (10) | 0.070 |
| 1 (20) | 0 (a plain max; the T2 EDA marked this HIGH, which is why 5 was chosen) |

So the premise holds for about one positive window in five (edge windows with 1–3 abnormal rows), not for most.

- **Arm K5k2 = K5 s2024 + `loss.mil_topk_pct=10`** (`KIM_TOPK_MIL_PCT`). k = 2. **The one key drives three things:** `mil`, `kim_mil` and KIM's anchor selection (`kim_anchors`, top-k of each abnormal bag above `anchor_thresh`). They are not separated, and the contrast names the key, not one of them. The CLIP and motion stream, `W_u`, KIM, DVS off, kernel 3, `mul_mil_topk_pct` 16 (k = 1 already), s2024 and 1,749 steps are all exactly as K5. **The config differs from K5 in that one key** (the runbook asserts this).
- **Why 10:** picked from the table above (the share falls from 0.205 to 0.070) without k = 1. No other value is run. **No sweep follows any outcome** (lesson 14).
- **No start check:** step 1's forward is identical, but its `mil` is the mean of a different top-k. Both step-1 values are printed.
- **Control:** K5 s2024, with its checkpoint and its four reads on Drive. It is not retrained.
- **Reads:** DoTA-CAP-dev (protocol B), DADA-2000 whole (370), TAD, DAD (s7). `kim_readout --control K5` gives macro and the sentence. Then `zs_arm_table` (§25.2) prints macro and micro on all four sets with paired Δ.
- **Decision, fixed now (`topk_rule`, the §25 shape):**
  - K5k2 − K5 **gain on DADA and no cost on DoTA-CAP-dev** (macro, `kim_readout` sentence, margin +0.01) → K5k2 s2025 against K5 s2025. Nothing is adopted at n = 1.
  - **Anything else → the top-k line closes.** No other divisor, no `min_positive` change, no per-head split.
  - At s2025: a gain on DADA → "reproduced, still nothing adopted"; anything else → closed.
- **Printed, never decides:** K5k2 − W0; TAD; DAD; every micro; the training diagnostics (`kim_anchor_share`, `kim_proto_cos`).
- **Prediction (written before the number):** K5k2 − K5 "no difference shown" on DoTA-CAP-dev and DADA. Only about 20 % of positive bags change composition, and in those the top-4 mean already had the right sign. K5's measured limits are the cross-video ordering (§21/§22) and the ego-uninvolved tail (§24), which k does not touch. If anything moves, it moves the curves' peakedness: within-video range up and micro on TAD / DAD down.

Code: `core/constants.py` (`KIM_TOPK_ARM`, `KIM_TOPK_MIL_PCT`), `core/tools/kim_readout.py` (`topk_rule`, through `_single_change_rule` shared with §25). Tests: `core/tests/test_kim_tools.py::TestTopkReadout`. Runbook: `colab/final/k5_k2.ipynb`.

### 26.1 Read-out (2026-11-14, s2024; raw `outputs/v_final/kim/REPORTS/kim_readout_k5k2_s2024.{md,json}`; PRELIMINARY, n = 1)

Run checks: checkpoint step 1,749 (`verified.json`, J10) = the last `global_step` of `metrics.jsonl`. The config differs from K5 in `loss.mil_topk_pct` (5 → 10) only. The three other diff lines (`model.motion_norm: none`, `kim.capacity_control: false`, `kim.seam_mask: false`) are keys added after K5 trained, all at their off defaults. No start check (§26): step-1 `mil` K5 0.6878, K5k2 0.6954 (a different top-k mean of the same forward).

| set | K5 | K5k2 | K5k2 − K5 [95 %] | sentence |
| --- | --- | --- | --- | --- |
| DoTA-CAP-dev macro (protocol B) | 0.8195 | 0.8058 | −0.0137 [−0.0213, −0.0060] | cost shown |
| DADA-2000 macro (370, whole) | 0.7740 | 0.7402 | −0.0338 [−0.0425, −0.0252] | cost shown |
| TAD macro (printed) | 0.7002 | 0.6808 | −0.0194 [−0.0493, +0.0121] | no difference shown |

**`topk_rule`: no gain on DADA and a cost on DoTA-CAP-dev → the top-k line CLOSES.** No s2025, no other divisor, no `min_positive` change, no per-head split, no sweep (fixed in §26 before the number). K5 stays the main model, with `mil_topk_pct = 5`.

**The prediction failed.** It said "no difference shown" on DoTA-CAP-dev and DADA. Both show a cost, and DADA's is the largest single-key effect measured on K5 so far.

Printed, never decides:
- K5k2 − W0: DoTA-CAP-dev −0.0052 [−0.0145, +0.0048] (no difference), DADA −0.0213 [−0.0349, −0.0079] (**cost**), TAD −0.0175 (no difference). K5k2 falls below the DVS-on host on DADA.
- DoTA-CAP-dev by ego: non-ego −0.0220 [−0.0349, −0.0093], ego −0.0073 [−0.0148, +0.0005]. The cost sits mostly in the ego-uninvolved tail (§24).
- Micro, raw-pooled as `kim_readout` records it: DADA K5 0.7222 → K5k2 0.7184 (≈ flat), TAD 0.6781 → 0.6451. TAD micro measures clip classification (§14.5.1).
- Training: K5k2 fits Nexar much tighter. Last-batch `mil` 0.1613 → 0.0639, last-epoch mean 0.1637 → 0.1270; `kim_mil` 0.1262 → 0.0746; `kim_nkc` 0.2771 → 0.1307; `kim_w_out_norm` 0.93 → 1.16; `kim_anchor_share` 0.904 → 0.962. `kim_proto_cos` is unchanged (−0.73).
- Four sets, macro + micro, the DAD read included: §26.2.

**Reading (post-hoc, printed, decides nothing).** §26 was built on a dilution worry: too many top-k rows pull a positive bag's mean toward its normal rows. The data point the other way. The median positive window holds 8 abnormal rows. k = 4 pushes up about half of the abnormal span, k = 2 only its peak, so the rest of the span is left low inside the video. That fits a large macro cost (within-video ordering) with micro on DADA nearly flat, and a lower training loss that does not transfer (an easier, peakier objective). This is the opposite of the dilution premise, so it is not licensed to propose k > 4 here. Under lesson 14, §26's "no other divisor" stands.
§26.2 qualifies this reading: DoTA-CAP-dev and DAD micro go **up** under k = 2.

### 26.2 Macro and micro on all four sets (2026-11-14, s2024; raw `outputs/v_final/kim/REPORTS/zs_arm_table_k5k2_s2024.{md,json}`; PRELIMINARY, n = 1; printed, decides nothing)

The §25.2 tool is reused: DoTA-CAP-dev and DADA micro / AP are pooled after per-video min-max, TAD and DAD raw. Δ macro uses the read-outs' bootstrap, Δ micro a cluster bootstrap (2,000, seed 2026). This table cannot reopen §26, which decided on macro before any number.

| set | K5 macro / micro | K5k2 macro / micro | Δ macro [95 %] | Δ micro [95 %] |
| --- | --- | --- | --- | --- |
| DoTA-CAP-dev (569) | 0.8195 / 0.7449 | 0.8058 / 0.7672 | −0.0137 [−0.0213, −0.0060] below | **+0.0223 [+0.0127, +0.0317] above** |
| DADA-2000 (370) | 0.7740 / 0.7635 | 0.7402 / 0.7455 | −0.0338 [−0.0425, −0.0252] below | −0.0180 [−0.0264, −0.0087] below |
| TAD (100; macro over 60) | 0.7002 / 0.6781 | 0.6808 / 0.6451 | −0.0194 [−0.0493, +0.0121] n.s. | −0.0330 [−0.0546, −0.0109] below |
| DAD (466; macro over 165) | 0.7916 / 0.7050 | 0.8042 / 0.7516 | +0.0126 [−0.0054, +0.0310] n.s. | **+0.0466 [+0.0362, +0.0570] above** |

AP: DoTA-CAP-dev 0.5790 → 0.5922, DADA 0.5640 → 0.5335, TAD 0.2568 → 0.2438, DAD 0.1156 → 0.1549. TAD's clip oracle is 0.9226 and DAD's 0.8390: micro on both measures clip classification (§14.5.1).

Reading (post-hoc, decides nothing):
- **k = 2 does not cost everywhere. It trades.** On DoTA-CAP-dev, within-video ordering (macro) falls and cross-clip pooling (micro) rises. On DAD both point up, micro clearly. On DADA both fall. So §26.1's coverage reading explains the macro side only. The peakier curves also separate across clips better on DoTA-CAP-dev and DAD, but not on DADA or TAD.
- **§26's prediction, micro half:** "micro on TAD / DAD down". It held on TAD (−0.033) and **failed on DAD** (+0.047).
- **DAD is the only set where K5k2 is not worse on any column.** It is n = 1, a set that never decides, and a micro that measures clip classification. It does not reopen the line. Never write "k = 2 helps on DAD" without all three qualifiers.
- The verdict stands: the decision sets (DoTA-CAP-dev, DADA) lose macro, DADA loses on every column, and the line is closed.

## 27. K5nc2: negative-video windows capped at 2 per video (dated note 2026-11-16, before any number; user)

**Not a reopening of the KIM line or of §17**; K5 stays the main model unless the rule below and the user say otherwise. §17.1 lowered false alarms on `nexar_val` negatives (peak > 0.5: 0.717 → 0.283) and cost macro on Nexar crops, DoTA-CAP-dev and DADA. Its one factor bundles "more negatives" with "in-video contrast drawn ≈ 55 % less often" (pool sampling). This arm keeps every negative video and shrinks their share of the normal pool, so the second part mostly goes away.

**Measured before this rule (post-hoc on §17's scores, printed, decides nothing; raw, cut 0.5).** K5's in-video penalty works far from onset: on `nexar_val` positives, 0.100 of the 5 s chunks more than 5 s before onset have max > 0.5, against 0.187 on the negatives (K5neg: 0.038 / 0.068). The negatives' median length is 40.2 s vs 13.3 s, so most of K5's 0.717 is the max over a long video; the rest is a ≈ 2× per-chunk rate on a kind of video K5 never trained on. Hard negatives of that kind exist only in the negative videos. At matched false-alarm rates K5neg already detected more (§17.1: 0.661 vs 0.562 at 0.20), so the 0.5-cut detection drop was mostly a score-scale shift.

| # | Choice | Value |
|---|---|---|
| NC1 | Row | **K5nc2** = K5's config with `--data-dir Nexar_window_negcap`, built by `nexar_build corpus --constructions window_negcap`: `window_neg`'s windows, except each negative video contributes **at most 2** windows, evenly spaced (`cap_windows`). Positive videos keep no cap, as in `window`. Val side identical to `Nexar_window`. **Config diff to K5 = 0** (asserted). s2024 only |
| NC2 | Why 2 | Chosen from the pool share, not from a score: ≤ 898 negative windows against 5,622 in-video normals ⇒ ≈ 14 % of the normal pool (K5neg ≈ 55 %), while every one of the 449 negative videos stays in. m = 1 (≈ 7 %) and other values are not run; **no sweep follows any outcome** (lesson 14). The exact count is the build's `train_items − 7,311` |
| NC3 | Inputs | As §17 NG2 (K5's bakes, the Nexar statistics cover all 900 train videos). No extraction, no KNN |
| NC4 | Reads | As §17 NG3: `nexar_val` (crops endpoint, whole, micro, AP, clip AUC) **with `--save-scores`**, DoTA-CAP-dev, DADA whole, TAD, DAD. `nexar_test` is not touched. Control = K5 s2024's existing reads (§17's NG4 re-read of K5, `kim_neg/eval/nexar/K5`); K5 is not retrained |
| NC5 | Sentence | K5nc2 − K5 per set, paired per item, 95 % bootstrap: "above / below / not separated", **PRELIMINARY (n = 1)** |
| NC6 | Printed, never gating | (a) `nexar_val` negatives with peak > 0.5; (b) **detection at matched false-alarm rates 0.20 and 0.283** (cut chosen on each row's own `nexar_val` negatives), K5 / K5neg / K5nc2; (c) share of 5 s chunks with max > 0.5, positives > 5 s before onset vs negatives; (d) peak > 0.5 in the last 5 s before onset; (e) false alarms by scene; (f) K5nc2 − K5neg macro |
| NC7 | What may follow | **Candidate** (Claude may propose s2025; the user decides) iff **no set is "below"** and (**at least one set is "above"** or **detection at matched false-alarm 0.20 rises by ≥ 0.05 over K5**). Any set "below" → K5 stays and the negative-share line closes: no m = 1, no other cap |

**Expectations, written before the numbers:** macro "not separated" from K5 on every set (the in-video contrast is ≈ 86 % of K5's); false alarms on `nexar_val` negatives fall, but less than K5neg's (between 0.717 and 0.283); detection at matched false-alarm 0.20 between K5's 0.562 and K5neg's 0.661. A wrong expectation changes nothing.

**Limits.** n = 1. The matched-rate read has one point per rate and no interval (112 / 113 videos). The cap keeps the 2 windows evenly spaced, not the hardest ones. A Nexar scene shortcut stays possible (Sub-urban 41 / 122 in train); NC6(e) only describes it. The `min_positive` change (§26's "no `min_positive` change") is **not** part of this arm and stays closed.

Code: `core/data/dada.py` (`negative_max_per_clip`, default `None` = unchanged for T2 and every other caller), `core/tools/nexar_build.py` (`window_negcap`, `train_negative_video_windows` in the build report), `core/constants.py` (`NEXAR_CONSTRUCTION_WINDOW_NEGCAP`, `NEXAR_NEG_WINDOW_CAP`, `KIM_NEGCAP_*`), `core/tools/k5_nc2_report.py` (NC5–NC7; matched detection = §17.1's arithmetic, reproduced on its numbers). Tests: `core/tests/test_dada.py::TestPlanRecordWindows`, `core/tests/test_nexar_pipeline.py` (negcap), `core/tests/test_k5_nc2_report.py`. Runbook: `colab/final/k5_nc2.ipynb`. Outputs `Thesis-Final/outputs/kim_negcap/`.

### 27.1 Read-out (2026-11-16, s2024; raw `outputs/v_final/kim_negcap/REPORTS/k5_nc2_report.{md,json}`; PRELIMINARY, n = 1)

Run checks: checkpoint step 1,749 (`verified.json`, J10) = the last `global_step` of `metrics.jsonl` = K5's step. The config equals K5neg's except `model.motion_norm: none`, a key added after K5neg trained, at its off default (against K5's original config, `kim.capacity_control: false` and `kim.seam_mask: false` also appear, likewise off): **config diff to K5 = 0** as NC1 requires. Build (`build_report_window_negcap.json`): 449 / 449 negative videos kept, 898 negative-video windows = `train_items` 8,209 − 7,311 (NC2), ≈ 13.8 % of the normal pool; 33 epochs × 53 steps = 1,749. NG4: K5's re-read on `nexar_val` crops 0.8981 = §13, gap 0.

| set | K5 | K5neg | K5nc2 | K5nc2 − K5 [95 %] | sentence (NC5) |
| --- | --- | --- | --- | --- | --- |
| Nexar val crops (endpoint, in-domain) | 0.8981 | 0.8597 | 0.8877 | −0.0103 [−0.0177, −0.0029] | **below** |
| DoTA-CAP-dev macro (protocol B) | 0.8195 | 0.8044 | 0.8187 | −0.0009 [−0.0057, +0.0041] | not separated |
| DADA-2000 macro (370, whole) | 0.7740 | 0.7531 | 0.7687 | −0.0053 [−0.0106, −0.0000] | **below** |
| TAD macro | 0.7002 | 0.7097 | 0.7024 | +0.0022 [−0.0150, +0.0194] | not separated |
| DAD macro | 0.7916 | 0.8086 | 0.7972 | +0.0056 [−0.0049, +0.0159] | not separated |

**NC7: two sets "below" (Nexar val, DADA) → K5 stays and the negative-share line CLOSES.** No m = 1, no other cap, no "hardest windows" variant, no s2025 (fixed in §27 before the number). DADA's upper bound sits at −0.0000; the verdict does not depend on it, since Nexar val is below on its own.

**Expectations:** "macro not separated on every set" **failed** (Nexar val and DADA show a small cost). "False alarms between K5 and K5neg" held (0.717 → 0.425; K5neg 0.283). "Detection at matched FA 0.20 between" held at the bottom edge: 0.5714 vs K5 0.5625 / K5neg 0.6607, Δ +0.0089 against the 0.05 margin.

Printed, never decides (NC6):
- Nexar val whole macro / micro / AP / clip AUC: K5 0.9245 / 0.9000 / 0.4472 / 0.7735; K5nc2 0.9144 / 0.8902 / 0.4347 / 0.7754; K5neg 0.8838 / 0.8497 / 0.3573 / 0.7909.
- Detection at matched FA 0.283: K5 0.6071, K5neg 0.6964, K5nc2 0.6696. One point per rate, no interval (112 / 113 videos).
- Share of 5 s chunks with max > 0.5: positives > 5 s before onset K5 0.0995 / K5neg 0.0379 / K5nc2 **0.0332**; negatives 0.1871 / 0.0680 / 0.0996; last 5 s before onset 0.3214 / 0.1607 / 0.1875.
- Negatives with peak > 0.5 by scene, K5 → K5nc2: Urban 0.707 → 0.366, Sub-urban 0.759 → 0.448, Highway 0.667 → 0.444, other (n = 7) 0.857 → 0.571. No scene stands out.
- K5nc2 − K5neg: Nexar val +0.0281, DoTA-CAP-dev +0.0143, DADA +0.0155 (all above); TAD −0.0073, DAD −0.0114 (not separated).
- Training, last batch K5 / K5neg / K5nc2: `mil` 0.161 / 0.170 / 0.147, `kim_mil` 0.126 / 0.107 / 0.119, `kim_nkc` 0.277 / 0.199 / 0.328, `kim_w_out_norm` 0.93 / 0.77 / 0.92, `kim_anchor_share` 0.904 / 0.942 / 0.933; last-epoch mean `mil` 0.164 / 0.186 / 0.189.

**Reading (post-hoc, printed, decides nothing).** The negative share moves one trade-off in one direction. At ≈ 14 % the cap removes about ¾ of K5neg's macro cost (Nexar val −0.0384 → −0.0103, DADA −0.0209 → −0.0053) and keeps about ⅔ of its false-alarm drop at cut 0.5 (0.292 of 0.434). Detection at matched FA 0.20 barely moves (+0.009), so most of the 0.5-cut drop is a score-scale shift, not better ranking, which is §17.1's reading again. Far-from-onset chunks on positives are the cleanest of the three rows, while negative-video chunks stay ≈ 3× higher (0.0996): the scenes K5 alarms on in negative videos are not taught away by a few more negative windows. A false-alarm target on Nexar is a cut chosen on negatives at deployment, not a training change. This is a remark, not a rule.

**Licensed:** "Capping negative-video windows at 2 per video (≈ 14 % of the normal pool) costs Nexar-val crops −0.010 and DADA −0.005 against K5, separates on no other set, and lowers Nexar-val false alarms at cut 0.5 from 0.717 to 0.425 with no material gain at matched false-alarm rate 0.20 (+0.009). PRELIMINARY (n = 1). K5 stays; the negative-share line is closed (§27 NC7)." **Not licensed:** "the cap lowers false alarms at no cost", "K5nc2 detects better", any streaming or calibrated-threshold claim, any m = 1 extrapolation.

## 28. K5Lneg: `L_neg` with class positives, on K5 (dated note 2026-11-16, before any number; user)

**Not a reopening of the KIM line, §17 or §27.** K5 stays the main model unless the rule below and the user say otherwise. The user asked to switch on `L_neg`. Every K run so far had it off (`loss.captions_from_definitions: false`, so `core/train.py` never builds captions).

**Why the flag alone is not run (read from the code, no data).** With `captions_from_definitions=true` on Nexar:
1. Each abnormal item gets a caption drawn from 4 `CarAccident` sentences, so ≈ 32 abnormal items per batch share 4 texts. `asymmetric_infonce_loss` asks caption *i* to prefer video *i* over every other accident video, and the texts cannot tell them apart.
2. `dvs.theta = 1.0`, so `pseudo_frame_label` is all zero. n3 mining then selects nothing, and `softmax(empty) @ empty` appends an **all-zero column** per abnormal item. There is no hard negative and no error.
3. The similarity is a raw dot product of unnormalized features (`einsum` in `CapContrastLoss`), so its scale is unknown (C37).
4. The flag also turns on a second fused caption branch (`cap_bin_logits` → a second MIL, `dvs_sup`, `mul_mil` over per-item captions with the same false negatives). That is a second factor.

| # | Choice | Value |
|---|---|---|
| NL1 | Row | **K5Lneg** = K5's config with `loss.cap_contrast_mode=class`, on K5's corpus (`Nexar_window`). **Config diff to K5 = this key only** (asserted; keys added after K5 trained stay at their off defaults). s2024 only |
| NL2 | Loss (`ClassContrastLoss`) | **Rows** = the text features of the abnormal classes present in the batch, i.e. the `class_feats` the step already encodes for `mul_mil` (no extra verbalizer draw, so the training RNG stream equals K5's). **Columns** = each video's `vis_feats` aggregated by `softmax(logit / 0.02)` (as the baseline), plus one mined "normal part" per abnormal video: `softmax(−logit / 0.02)` over its anchor span if synthesized, over all valid rows if not; skipped when its prob range < `N3_MIN_SCORE_RANGE` (0.2) or the selection is empty. **Similarity** = cosine / `LNEG_CLASS_TEMP` (0.07, CLIP's initial temperature, not tuned). **Target** = uniform over the columns of the row's class, label smoothing 0.1 (soft-target InfoNCE, the "L_out" form of supervised contrastive). Normal videos and mined parts are negatives. The video → text direction is added only when ≥ 2 abnormal classes are present; on Nexar (one class) it is absent. **No caption branch** enters the model |
| NL3 | Weight | `cap_contrastive_weight` = 1.0, the existing default (LaGoVAD's value), not derived from a number of this run. Logged per batch, never decides: `cap_contrastive`, its floor `cap_contrastive_floor` (the target's entropy, the lowest the term can reach) and `cap_mined` (mined columns) |
| NL4 | Reads | As §27 NC4: `nexar_val` (crops endpoint, whole, micro, AP, clip AUC) **with `--save-scores`**, DoTA-CAP-dev, DADA whole, TAD, DAD. `nexar_test` is not touched. Control = K5 s2024's existing reads; K5 is not retrained |
| NL5 | Sentence | K5Lneg − K5 per set, paired per item, 95 % bootstrap: "above / below / not separated", **PRELIMINARY (n = 1)** |
| NL6 | Printed, never gating | As §27 NC6 (a)–(e): false alarms on `nexar_val` negatives (peak > 0.5), detection at matched false-alarm 0.20 and 0.283, 5 s chunk rates, last 5 s before onset, false alarms by scene. Plus step-1 and last-epoch `cap_contrastive − cap_contrastive_floor` beside `mil` |
| NL7 | What may follow | **Candidate** (Claude may propose s2025 and, separately, option C: one caption per accident event, 451 calls; the user decides) iff **no set is "below"** and (**at least one set is "above"** or **detection at matched false-alarm 0.20 rises by ≥ 0.05 over K5**). Any set "below" → K5 stays and the definition-text `L_neg` line closes. Otherwise ("no claim") → K5 stays, and this arm gives no ground for option C |

**Expectations, written before the numbers:** macro "not separated" from K5 on every set. With one class the term is a text-anchored video-level contrast, close to what `mil` and `kim_mil` already ask. False alarms on `nexar_val` negatives at most slightly lower than K5's 0.717 and well above K5neg's 0.283, since K5's corpus has no negative-video windows for the term to push away. A wrong expectation changes nothing.

**Limits.** n = 1. This is **not** LaGoVAD's `L_neg`: class positives, cosine similarity and fallback mining are three deviations, fixed together because the baseline form is degenerate here (above). It cannot say anything about per-video captions (option C). τ = 0.07 is borrowed, not measured. The mined part of an unsynthesized window is "its lowest-scoring rows", which in a positive window straddling onset can include pre-onset rows that are not anomalous; that is the intended hard negative.

Code: `core/losses/contrastive.py` (`ClassContrastLoss`, `class_infonce_loss`; `CapContrastLoss` unchanged), `core/config.py` (`loss.cap_contrast_mode`), `core/constants.py` (`CAP_CONTRAST_MODE_*`, `LNEG_CLASS_TEMP`, `KIM_LNEG_*`), `core/train.py` (the `class` branch). Runbook `colab/final/k5_lneg.ipynb`. Outputs `Thesis-Final/outputs/kim_lneg/`.

### 28.1 Read-out (2026-11-16, s2024; raw `outputs/v_final/kim_lneg/REPORTS/k5_lneg_report.{md,json}`; PRELIMINARY, n = 1)

Run checks: checkpoint step 1,749 (`verified.json`, J10) = the last `global_step` of `metrics.jsonl` = K5's step. Config diff to K5 = `loss.cap_contrast_mode: class` only (plus `model.motion_norm: none`, a later key at its off default), as NL1 requires. Every logged step carries `cap_contrastive`; mined columns per batch rise from ≈ 2 (epoch 0) to ≈ 37 (last epoch). NG4: K5's re-read on `nexar_val` crops 0.8981 = §13, gap 0.

| set | K5 | K5neg | K5Lneg | K5Lneg − K5 [95 %] | sentence (NL5) |
| --- | --- | --- | --- | --- | --- |
| Nexar val crops (endpoint, in-domain) | 0.8981 | 0.8597 | 0.7779 | −0.1202 [−0.1402, −0.1009] | **below** |
| DoTA-CAP-dev macro (protocol B) | 0.8195 | 0.8044 | 0.7730 | −0.0465 [−0.0581, −0.0347] | **below** |
| DADA-2000 macro (370, whole) | 0.7740 | 0.7531 | 0.6943 | −0.0797 [−0.0938, −0.0652] | **below** |
| TAD macro | 0.7002 | 0.7097 | 0.6666 | −0.0335 [−0.0713, +0.0034] | not separated |
| DAD macro | 0.7916 | 0.8086 | 0.7296 | −0.0620 [−0.0944, −0.0305] | **below** |

**NL7: four sets "below" → K5 stays and the definition-text `L_neg` line CLOSES.** No s2025, no other τ, weight or mining (lesson 14). §29's captions have no use under this note (§29: "If §28 reads below … the captions stay unused"); reopening option C takes a new dated note that says why per-video captions would differ in kind from this result.

**Expectations:** "macro not separated on every set" **failed**, by the largest single-key cost measured on K5 (DADA −0.080 vs K5k2's −0.034). "False alarms at most slightly below 0.717" held (0.699). Detection at matched FA 0.20: 0.5804 vs K5 0.5625 (+0.018, margin 0.05).

Printed, never decides (NL6):
- Nexar val whole macro / micro / AP / clip AUC: K5 0.9245 / 0.9000 / 0.4472 / 0.7735; K5Lneg 0.7969 / 0.7801 / 0.3121 / 0.7675.
- Detection at matched FA 0.283: K5 0.6071, K5Lneg 0.6339. 5 s chunks with max > 0.5: far pre-onset 0.0995 → 0.0664, negatives 0.1871 → 0.1774, last 5 s before onset 0.3214 → 0.2232. Negatives with peak > 0.5 by scene: Urban 0.707 → 0.659, Sub-urban 0.759 → 0.724, Highway 0.667 → 0.694.
- K5Lneg − K5neg: below on all five sets (Nexar −0.0818, DoTA-CAP-dev −0.0314, DADA −0.0588, TAD −0.0431, DAD −0.0790).
- Loss, K5Lneg (K5): step 1 `mil` 0.6925 (0.6878), `cap_contrastive` 4.2651 − floor 4.1879 = +0.0772; last epoch `mil` 0.1387 (0.1637), `kim_mil` 0.1312 (0.1613), `cap_contrastive` − floor = +0.0245; `kim_nkc` last batch 0.199 (0.277).

**Reading (post-hoc, printed, decides nothing).** The cost is within-video ordering, not clip classification: Nexar whole macro falls 0.925 → 0.797 while clip AUC barely moves (0.774 → 0.768). The training objective is fitted more tightly than K5's (`mil`, `kim_mil` lower), the K5k2 pattern again: an easier objective, not a better one. The term sits near its floor throughout, yet at cosine / 0.07 its gradient still reshapes `vis_feats`; once curves sharpen, almost every accident window adds a mined "low-score part ≠ accident" negative (what MIL already teaches), and the L_out target pulls every accident window's pooled feature toward one text direction. That this collapses per-row feature diversity is a hypothesis; it is **not measured** (`grad_probe` on the checkpoint would read `rho` / `cos`).

**Licensed:** "Adding a class-positive caption contrastive term (`loss.cap_contrast_mode=class`, cosine / 0.07) to K5 costs Nexar-val crops −0.120, DoTA-CAP-dev −0.047, DADA −0.080 and DAD −0.062 (TAD not separated), with Nexar false alarms nearly unchanged (0.717 → 0.699). PRELIMINARY (n = 1). K5 stays; the definition-text `L_neg` line is closed (§28 NL7)." **Not licensed:** "L_neg hurts LaGoVAD" (this is not LaGoVAD's form), "contrastive text supervision cannot help" (one form, one seed), any mechanism sentence.

### 28.2 Probe P2: does `L_neg` collapse within-video feature diversity? (dated note 2026-11-16, before any number; user)

**Question.** §28.1's reading names a hypothesis it did not measure: the class-positive term pulls every accident window's pooled `vis_feats` toward one text direction, so rows of the same video lose diversity and the accident rows stop standing apart from the normal ones (macro falls, clip AUC does not). `grad_probe` cannot read this: it refuses `kip.enabled=false`, counts `cap_contrastive` inside "task", and measures gradients, not features. P2 reads the features directly. **Inference only; nothing trains; K5 stays.** It cannot reopen §28 or option C by itself.

| Key | Choice (fixed before reading) |
|---|---|
| FD1 checkpoints | K5 s2024 (`kim/K5/s2024`) = base, K5Lneg s2024 (`kim_lneg/K5/s2024`) = new; both J10-verified; same inputs (A2-S Nexar statistics, V2-S motion), K4 checked per checkpoint |
| FD2 videos | every positive video of `nexar_val`, whole video, protocol N-B rows (`s1[::3]`), windowed as the eval scorer (`pos_neg_gap.video_outputs`); frame labels = D-N1 (`native_labels`), used to measure only |
| FD3 tensor | `vis_feats` (the tensor `L_neg` acts on, `kat_vad.py:395`), each row L2-normalized, float64 |
| FD4 metrics, per video | (a) `row_cos` = mean off-diagonal cosine between rows; (b) `eff_rank` = exp(entropy of the normalized singular values of the centred rows), printed; (c) `sep` = ‖μ_abnormal − μ_normal‖² / mean ‖z − μ_own‖² (scale-free Fisher ratio). A video needs ≥ 2 abnormal and ≥ 2 normal rows, else skipped and counted |
| FD5 contrast | per video Δ = new − base, mean over videos, 95 % percentile bootstrap (2,000 draws, seed 2024) |
| FD6 verdict | **SUPPORTED** iff Δ`sep` CI entirely < 0 **and** Δ`row_cos` CI entirely > 0. **OPPOSITE** iff Δ`sep` CI entirely > 0. Otherwise **NOT SUPPORTED**. `eff_rank` is printed, never decides |

**Predictions (written now).** If §28.1's hypothesis holds: `sep` falls, `row_cos` rises, `eff_rank` falls.

**What each verdict licenses.** SUPPORTED: "on `nexar_val`, K5Lneg's rows are more alike within a video and its accident rows stand less apart from the normal rows than K5's (n = 1 checkpoint each)". It shows co-occurrence with the term, not a mechanism by itself, and says nothing about per-video captions beyond "a per-video text target would pull on the same pooled feature". NOT SUPPORTED / OPPOSITE: the cost is not a loss of within-video feature separation; §28.1's mechanism sentence is withdrawn and the cause stays unknown (the head, or the fusion path, not `vis_feats`).

**Limits.** n = 1 checkpoint per arm. `sep` is measured in `vis_feats`, before fusion and the head; a score can lose ordering with intact features and vice versa. Val, not train windows.

Code: `core/tools/feature_diversity.py`, constants `FEATDIV_*`, tests `core/tests/test_feature_diversity.py`, runbook `colab/final/lneg_feature_diversity.ipynb`.

#### 28.2.1 Read-out (2026-11-16; raw `kim_lneg/REPORTS/feature_diversity/feature_diversity.{md,json}`, pasted by the user; n = 1 checkpoint per arm)

Checks: both checkpoints step 1,749 (J10); 112 / 112 `nexar_val` positives scored, 0 skipped.

| metric (per video, unit `vis_feats` rows) | K5 | K5Lneg | Δ [95 %] | share of videos K5Lneg lower |
|---|---|---|---|---|
| `sep` (decides) | 2.6290 | 2.0823 | **−0.5467 [−0.6759, −0.4251]** | 0.884 |
| `row_cos` (decides) | 0.7007 | 0.7087 | **+0.0080 [+0.0046, +0.0115]** | 0.375 |
| `eff_rank` (printed) | 135.31 | 133.86 | −1.46 [−1.86, −1.02] | 0.714 |

**FD6: SUPPORTED.** The prediction (sep down, row_cos up, eff_rank down) held on all three.

**Reading (post-hoc, printed, decides nothing).** The two deciding numbers are not the same size. `sep` falls by 21 % and on 88 % of videos; `row_cos` rises by 0.008 on a base of 0.70 and on only 62.5 % of videos; `eff_rank` falls by 1 %. So the rows of a video do **not** collapse as a whole: what goes is the **accident-vs-normal direction** inside the video, with a small rise in overall row similarity. Say "K5Lneg's accident rows stand less apart from the normal rows of their own video", not "the features collapse".

**Licensed:** "On `nexar_val`, adding the class-positive caption contrastive term lowers the within-video separation of accident rows from normal rows in `vis_feats` by 21 % (Fisher ratio 2.63 → 2.08, Δ −0.55 [−0.68, −0.43], 88 % of 112 videos), with a small rise in row similarity (+0.008) (n = 1 checkpoint each)." This co-occurs with §28.1's macro cost; it does **not** prove the term causes it through this path (no intervention on the path was run). **Not licensed:** "feature collapse", "L_neg destroys features", any claim about LaGoVAD's own `L_neg`.

**Consequence for option C (remark, not a rule; corrected the same day).** P2 compares "term on" with "term off"; it cannot say which ingredient costs the separation: (A) one shared text target per class (every accident window pulled to the same vector, same-class columns positive) or (B) the pooled-feature objective itself (one attention-pooled vector per window, cosine / 0.07, mining). A per-video caption changes (A) and keeps (B), so P2 neither predicts that good captions would help nor that they would not. A further oddity: the mined `v_neg` term pushes the low-score part away from the accident text, which on its face should *raise* separation; it fell. What blocks option C today is caption quality (§29.1–§29.2 QC 3.43 < 4.0), not P2. A test that separates (A) from (B) without good captions: K5 + a fixed, unique, meaningless text target per video (not built, not authorized).

### 28.3 K5Lnegu: which ingredient of `L_neg` costs? (dated note 2026-11-16, before any number; user)

**Question.** §28.2.1 cannot say whether K5Lneg's cost comes from (A) the **one shared class text** every accident window is pulled to (same-class columns positive) or (B) the **pooled-feature objective** itself (one attention-pooled vector per window, cosine / 0.07, mining, weight 1.0). A per-video caption changes (A) and keeps (B). K5Lnegu does the same with no caption: it answers whether option C could help at all, before anyone writes 451 good captions. **Not a reopening of §28**; K5 stays the main model.

| # | Choice | Value |
|---|---|---|
| U1 | Row | **K5Lnegu** = K5's config with `loss.cap_contrast_mode=uniq`, on K5's corpus `Nexar_window`, s2024. **Config diff to K5 = this key only** (asserted) |
| U1 | Loss | `ClassContrastLoss` unchanged (columns, mining, cosine / `LNEG_CLASS_TEMP`, label smoothing, weight 1.0). Only the rows change: each **source video** present with an accident window gets one fixed unit vector `unique_target(source)` (seeded by `LNEG_UNIQ_TARGET_SEED` = 2024 and the id's sha1, own generator: the training RNG stream equals K5's). Windows of one source share it; accident windows of other sources become negatives; normal windows and mined parts stay negatives. With ≥ 2 sources in a batch the video → text direction is on (in K5Lneg, one class, it was off): the form a per-video caption would have under this loss |
| U2 | Sentences | Per set, paired, 95 % bootstrap, as NL5: **K5Lnegu − K5** and **K5Lnegu − K5Lneg** ("above / below / not separated"); K5Lneg − K5 re-derived beside them. Deciding sets S = Nexar val crops, DoTA-CAP-dev, DADA, DAD (the four K5Lneg fell on); TAD printed |
| U2 | Verdict | **A** ("the shared target costs") iff K5Lnegu − K5 is "below" on **no** set of S. **B** ("the pooled objective costs") iff "below" on **≥ 3** of S **and** K5Lnegu − K5Lneg "above" on **≤ 1** of S. Otherwise **MIXED** |
| U3 | Printed, never deciding | P2 (`feature_diversity`) K5Lnegu vs K5 on `nexar_val`; step-1 / last-epoch `cap_contrastive − floor` beside `mil`; per-frame curves K5 / K5Lneg / K5Lnegu |
| U4 | What follows | **A** → option C *may* be reopened, only with captions that pass QC (Claude-written route, fresh QC seed) and a new dated note; uniq is the extreme (orthogonal targets), real captions are correlated, so A does not promise a gain. **B** → per-video text targets under this loss form are closed, and option C with them. **MIXED** → no claim; option C stays closed unless the user decides otherwise |

**Prediction (written now):** B or MIXED, not A. Unique targets also push accident videos apart from each other, which adds a pull against MIL.

**Limits.** n = 1. Random targets are not text: they cannot test anything a caption's *content* would add. The video → text direction differs from K5Lneg (on vs off), stated in U1 as part of the per-video form.

Code: `core/losses/contrastive.py` (`unique_target`, `uniq_rows`), `core/train.py` (the `uniq` branch: items mapped to source videos by `dataset.slicer.source_of`), `core/constants.py` (`CAP_CONTRAST_MODE_UNIQ`, `LNEG_UNIQ_TARGET_SEED`, `KIM_LNEG_UNIQ_*`), `core/tools/k5_lneg_uniq_report.py`. Tests `test_class_contrast.py`, `test_train_units.py`, `test_k5_lneg_uniq_report.py`. Runbook `colab/final/k5_lneg_uniq.ipynb`. Outputs `Thesis-Final/outputs/kim_lneg_uniq/`.

#### 28.3.1 Read-out (2026-11-17, s2024; raw `outputs/v_final/kim_lneg_uniq/REPORTS/k5_lneg_uniq_report.{md,json}`; PRELIMINARY, n = 1)

Run checks: checkpoint step 1,749 (`verified.json`, J10) = the last `global_step` of `metrics.jsonl` = K5's step. Config diff to K5Lneg = `loss.cap_contrast_mode: class → uniq` only, so the diff to K5 is this key (U1). Every logged step carries `cap_contrastive`; mined columns 6 (step 1) → 28 (last step). NG4: K5's re-read on `nexar_val` crops 0.8981 = §13, gap 0. The first report run crashed on a key collision in the tool (`verdict()`'s `sets` overwrote the per-set contrasts; fixed in `0ab5a5a`, no number changed: the report is pure aggregation).

| set | K5 | K5Lneg | K5Lnegu | K5Lnegu − K5 [95 %] | K5Lnegu − K5Lneg [95 %] |
| --- | --- | --- | --- | --- | --- |
| Nexar val crops (endpoint, in-domain) | 0.8981 | 0.7779 | 0.8285 | −0.0696 [−0.0826, −0.0574] **below** | +0.0506 [+0.0342, +0.0673] **above** |
| DoTA-CAP-dev macro (protocol B) | 0.8195 | 0.7730 | 0.7944 | −0.0252 [−0.0360, −0.0142] **below** | +0.0213 [+0.0134, +0.0296] **above** |
| DADA-2000 macro (370, whole) | 0.7740 | 0.6943 | 0.7214 | −0.0526 [−0.0642, −0.0407] **below** | +0.0271 [+0.0157, +0.0387] **above** |
| TAD macro (printed) | 0.7002 | 0.6666 | 0.6694 | −0.0308 [−0.0612, −0.0007] below | +0.0028 [−0.0245, +0.0300] not separated |
| DAD macro | 0.7916 | 0.7296 | 0.7529 | −0.0387 [−0.0702, −0.0075] **below** | +0.0233 [+0.0009, +0.0455] **above** |

K5Lneg − K5, re-derived, equals §28.1 on every set.

**U2: MIXED.** K5Lnegu − K5 is "below" on all four sets of S, so not A. K5Lnegu − K5Lneg is "above" on all four, so not B (B allows ≤ 1). **U4 → no claim; option C stays closed; K5 stays the main model.** No s2025, no other target seed, τ or weight (lesson 14).

**Prediction** "B or MIXED, not A": held (MIXED).

Printed, never decides (U3):
- P2 (`feature_diversity`, K5Lnegu vs K5, 112/112 `nexar_val` positives): `sep` 2.629 → 0.903, Δ −1.726 [−2.037, −1.446] (94.6 % of videos lower; K5Lneg's Δ was −0.547); `row_cos` 0.701 → 0.632, Δ −0.069 [−0.079, −0.058]; `eff_rank` +13.6 [+12.3, +15.0]. FD6 **NOT SUPPORTED** (Δ `row_cos` < 0): unique targets spread the rows of a video (lower cosine, higher rank) while the accident-vs-normal separation inside it falls three times as far as K5Lneg's.
- Nexar val whole macro / micro / AP / clip AUC: K5 0.9245 / 0.9000 / 0.4472 / 0.7735; K5Lnegu 0.8655 / 0.8500 / 0.3402 / 0.7813.
- Micro / AP, K5Lnegu: DoTA-CAP-dev 0.7629 / 0.5792 (K5 0.7449 / 0.5790); DADA 0.7286 / 0.5063 (0.7635 / 0.5640); TAD 0.5864 / 0.2250 (0.6781 / 0.2568); DAD 0.7331 / 0.1416 (0.7050 / 0.1156).
- Loss, K5Lnegu: step 1 `mil` 0.6925, `cap_contrastive` 4.3627 − floor 0.7513 = +3.611; last epoch mean `mil` 0.209, `kim_mil` 0.194 (K5 0.164 / 0.161), `cap_contrastive` − floor = +0.660 (K5Lneg +0.025). The uniq term stays far above its floor: unlike K5Lneg's, it is never fitted.

**Reading (post-hoc, printed, decides nothing).** Replacing the shared class target with one target per video recovers 34–46 % of K5Lneg's cost on every deciding set (Nexar 42 %, DoTA 46 %, DADA 34 %, DAD 38 %) and still costs against K5 on every set. Both ingredients carry cost at this n: the shared target (what uniq removes) and the pooled objective (what uniq keeps). Unlike K5Lneg, the term is not near its floor and MIL is fitted less tightly than K5's, so the remaining cost is a pull that competes with MIL, not an easier objective. The P2 pattern (rows more diverse, accident direction weaker) fits that, but co-occurrence is not causation.

**Licensed:** "With one fixed random target per source video in place of the class text (`loss.cap_contrast_mode=uniq`), the caption contrastive term still costs K5 on Nexar-val crops (−0.070), DoTA-CAP-dev (−0.025), DADA (−0.053) and DAD (−0.039), and costs less than the class-positive form on all four (+0.021 … +0.051). PRELIMINARY (n = 1). U2 = MIXED: no claim on which ingredient costs; K5 stays." **Not licensed:** "per-video captions would help" or "would not help" (random targets are not text), "the shared target explains the cost", "the objective explains the cost", any mechanism sentence.

## 29. Nexar event captions with Qwen2.5-VL-7B (dated note 2026-11-16, before any caption is read; user)

**Status 2026-11-16: not used — §28.1 read "below", so under this note the captions have no use; the tool stays, not run.** **A data artifact, not an arm.** The user asked to build option C's input now (one caption per accident event, Qwen2.5-VL-7B on a Colab L4), before §28 is read out. Nothing here trains, and nothing here authorizes training with the captions: that needs §28 NL7 to read "candidate" **and** a new dated note with its own loss rule. If §28 reads "below" or "no claim", the captions stay unused.

| # | Choice | Value |
|---|---|---|
| CP1 | Videos | The **451 positive videos of `nexar_train`** (`core/splits/v2/nexar_train.txt` ∩ census `label = 1`; 0 census issues). No `nexar_val` / `nexar_test` video and no negative video is captioned. The manifest records `train_ids_sha1` |
| CP2 | Clip | The labelled span `[t_alert, min(duration, t_event + 1.4 s)]` (D-N1, `nexar_build.span_seconds`): 1.43 to 5.67 s, median 2.83 s. **8 frames** evenly spaced in it (`NEXAR_CAPTION_FRAMES`), from the original mp4 (`Thesis/data/Nexar/raw`), each frame resized to exactly 448 × 252 (`NEXAR_CAPTION_SIZE`, multiples of 28, so the processor's `smart_resize` keeps it), passed as one video at `fps = 8 / span`. The tool asserts the video grid `(4, 18, 32)`: 144 tokens per frame, two frames per temporal patch ⇒ **576 visual tokens** |
| CP3 | Model | `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, `sdpa` attention, safetensors only; `transformers==4.56.*` (the project pin). The revision is resolved **once** at the start of the run and passed explicitly; the tool refuses `main` (C4) and writes it to every record |
| CP4 | Prompt | `NEXAR_CAPTION_PROMPT` (in `core/constants.py`): one sentence of at most 30 words on **what happens**: which road users, their relative motion, the kind of collision or near-collision. It forbids weather, light, time of day, road type, location, colours, brands and on-screen text (the NG5 scene shortcut). Greedy decoding, `max_new_tokens` = 64; the same input gives the same caption |
| CP5 | Output | `Thesis-Final/outputs/nexar_captions/captions/{id}.json` (atomic, C11c), one per video, with frames, span, model, revision and prompt hash; `captions.json` = the manifest + all captions. A record whose model, revision or prompt hash differs is refused, not overwritten. Resume skips finished videos |
| CP6 | QC (written before any caption is read) | `nexar_caption qc` draws **30** videos with seed 2024 and writes, per video, the 8 frames as one contact sheet beside its caption. Printed beside it: the share of all 451 captions containing a forbidden term (`NEXAR_CAPTION_FORBIDDEN`), the word-count range and the number of distinct captions. **The user reads the 30 sheets and marks each caption right or wrong. Pass iff ≥ 27 / 30 describe the event correctly.** Fail → the captions are not used and the prompt is not tuned against these 30 (lesson 14); a second attempt needs a new dated note |

**Limits.** The model sees 8 frames of a few seconds: fast events may fall between frames. Captions are model output, not annotation. One QC rater (the user), 30 of 451 videos. A caption describing the collision type still carries scene information (a highway merge reads differently from an intersection); CP4 lowers it, it does not remove it.

Code: `core/tools/nexar_caption.py` (`run`, `qc`), `core/constants.py` (`NEXAR_CAPTION_*`). Tests `core/tests/test_nexar_caption.py` (fake captioner, no model download). Runbook `colab/final/nexar_captions.ipynb` (L4).

### 29.1 Amendment: follow LaGoVAD's caption protocol (dated note 2026-11-16, before any caption is generated or read; user)

**Why.** The user asked for captions made the way LaGoVAD made PreVAD's descriptions, so they can be compared with them. LaGoVAD's repository has **no caption-generation code**; it only reads the `descriptions` field (`LaGoVAD-PreVAD/src/datasets/PreVAD.py:338`). The protocol is therefore taken from the paper (supp. Sec. D and Tab. B). No caption existed when this was written, so nothing was tuned against an output.

| Key | Was (§29) | Now (§29.1) |
|---|---|---|
| CP4' prompt | own prompt: ≤ 30 words, forbids weather / light / road type / colours | LaGoVAD's constrained annotation prompt **verbatim**, `{VIDEO LABEL}` = `Vehicle Accident` (PreVAD level-1): *"You are a helpful video describing assistant. You are good at English communication. Please describe the 'Vehicle Accident' event in the given video in detail in one sentence. The description should focus on the incident. Just describe without any other output."* (`NEXAR_CAPTION_PROMPT`) |
| CP4' length | 64 new tokens | 96 new tokens (PreVAD mean 22.9 words; "in detail" may exceed 30); captions over 40 words are counted, not cut |
| NG5 scene words | forbidden in the prompt | **not forbidden**; the share of captions naming one (`NEXAR_CAPTION_FORBIDDEN`) is still printed in the manifest summary. The scene shortcut is measured, not suppressed |
| CP6' QC | y / n, pass iff ≥ 27 / 30 | LaGoVAD **Tab. B** 1–5 consistency scale (criteria printed on the QC sheet), same 30 seeded videos (2024), **pass iff the mean ≥ 4.0**. LaGoVAD's constrained prompt scored 4.3 (3 raters, 200 videos); print ours beside it |

Unchanged: CP1 (451 train positives), CP2 (8 frames over `[t_alert, t_event + 1.4 s]`, 448×252), CP3 (Qwen2.5-VL-7B, bf16, pinned revision), CP5 (atomic per-video records; the prompt hash is in the identity, so nothing made under §29's prompt can be mixed in), fail → not used and no prompt tuning on these 30. Use in training still needs the new dated note required by §28.1.

**Deviations from LaGoVAD, stated.** Model: Qwen2.5-VL-**7B**, not Qwen2-VL-**72B** (≈ 145 GB in bf16, no L4 fit). Input: 8 frames of the labelled event span of an untrimmed dashcam video; PreVAD's frame sampling is not stated, and its videos are trimmed web clips. QC: one rater (the user), 30 videos, vs three raters on 200. **Never write "PreVAD-style captions" without these three deviations.**

Code: `core/constants.py` (`NEXAR_CAPTION_LABEL`, `NEXAR_CAPTION_PROMPT`, `NEXAR_CAPTION_QC_{SCORE_MIN,SCORE_MAX,MIN_MEAN}`), `core/tools/nexar_caption.py` (`QC_CRITERIA`, `score_qc`), tests `core/tests/test_nexar_caption.py` (16), runbook `colab/final/nexar_captions.ipynb` (cells 0, 14).

### 29.2 Amendment: caption window opens before contact (dated note 2026-11-16, after the 3-video smoke, before any QC caption is generated or read; user)

**Trigger, stated honestly.** The §29.1 smoke caption of `00004` was wrong in kind ("driving past a parked car … involved in an accident"). The user viewed the frames: the camera car steers right towards a shop and a following car clips its mirror; after contact both cars stop. The annotation is right (`t_event` = 19.367 s lies inside the window). The defect is in the **input**, and it can be measured from the annotation alone:

- The §29 window `[t_alert, t_event + 1.4 s]` starts at the alert, and on `nexar_train` positives `t_event − t_alert` is short (median 1.43 s, p10 0.59 s, min 0.03 s). With 8 frames, **51 / 451 videos (11 %) have ≤ 2 frames before contact** (counts 0:5, 1:14, 2:32, 3:110, 4:146, 5:113, 6:31); `00004` has 1.
- Qwen2.5-VL merges 2 consecutive frames into one time step (temporal patch 2), so 8 frames are 4 steps; `00004`'s only pre-contact frame shares a step with a post-contact one.

The fix is chosen from these counts, not from caption wording; the prompt (CP4') is unchanged. None of the 3 smoke videos (`00003`–`00005`) is in the 30-video QC draw (checked locally), so QC stays clean.

| Key | Was (§29 / §29.1) | Now (§29.2) |
|---|---|---|
| CP2' window | `[t_alert, min(duration, t_event + 1.4 s)]`, 1.43–5.67 s | **`[max(0, t_event − 2.0 s), min(duration, t_event + 1.4 s)]`**, 3.4 s on all 451 (none clipped; earliest `t_event` 3.03 s) |
| CP2' frames | 8 (4 time steps, 576 visual tokens) | **16** (8 time steps ≈ 0.21 s apart, grid (8, 18, 32) = 1,152 tokens); **9 / 16 frames precede contact on every video** |
| CP5 identity | model, revision, prompt, frames, size, tokens | + `pre_event_s`, `post_event_s`: a record from another window is refused |

2.0 s is a round number, not tuned. The caption window is **not** the training label span: D-N1 (`[t_alert, t_event + 1.4 s]`) is untouched and no training input changes. Cost ≈ 2× (estimate ~1.5–2 h on L4, not measured). The three §29.1 smoke records are deleted by the notebook before the new smoke (step 6).

Code: `core/constants.py` (`NEXAR_CAPTION_FRAMES` = 16, `NEXAR_CAPTION_PRE_EVENT_S`), `core/tools/nexar_caption.py` (`event_frame_indices`, `identity`), tests `core/tests/test_nexar_caption.py` (18), runbook `colab/final/nexar_captions.ipynb` (cells 0, 9, 10, 13).

### 29.3 Amendment: stronger captioner, dashcam prompt, fresh QC (dated note 2026-11-17, before any §29.3 caption is generated or read; user)

**Why.** §29.1's Qwen2.5-VL-7B captions failed QC (mean 3.43 < 4.0, §28.2 status). The failures were mostly ego-perspective (cut-ins and rear approaches misread, roles inverted) and denials ("no accident", 9 %), the second partly forced by the prompt's single label `Vehicle Accident`: Nexar positives are collisions **or near-collisions**, and the metadata has no column telling which (`core/data/nexar.py` reads only `time_of_alert` / `time_of_event`). The user asked for a stronger model, a prompt written for dashcam video, and an explicit event type, with "accident" as one of the types (Nexar has real accidents too). The captions feed §30 only.

| Key | Was (§29.1 / §29.2) | Now (§29.3) |
|---|---|---|
| CP3'' model | Qwen2.5-VL-7B-Instruct, bf16, L4 | **Qwen2.5-VL-32B-Instruct, 4-bit NF4** (bitsandbytes, double quant, bf16 compute), A100 40 GB (Colab Pro). NF4 rather than the AWQ release: bitsandbytes installs cleanly on Colab and needs no extra kernels. `quant` joins the record identity (CP5) |
| CP4'' prompt | LaGoVAD's constrained prompt, label `Vehicle Accident`, verbatim | **Dashcam prompt** (`NEXAR_CAPTION_PROMPT`): the camera is on the ego vehicle; the answer **opens with exactly one of `Accident:`, `Near-collision:`, `Dangerous maneuver:`** (definitions in the prompt), then one sentence ≤ 40 words naming the road users, what each does and from which direction; "the ego vehicle" for the camera car; describe only what is visible. The type is parsed (`event_type`; anything else = `unknown`) and printed; the whole text is the caption |
| CP4'' tuning | none (prompt fixed) | **Only on a 10-video dev set** (seed 2027, `--subset dev`, one folder per prompt sha1). Claude may change the prompt after reading dev captions; the prompt is frozen before step 7, and the 451 run uses one prompt (identity check) |
| CP6'' QC draw | 30 videos, seed 2024 | **30 videos, seed 2026, drawn from the positives outside the dev set and outside the seed-2024 draw** (those 30 were read). Pass iff the Tab. B mean ≥ 4.0, unchanged. **Printed, never deciding:** `type_ok`, `roles_ok` (y / n per row, optional) and the type histogram |
| CP5 output | `outputs/nexar_captions/` | **`outputs/nexar_captions_v2/`**; §29.1's records are kept, never mixed |

Unchanged: CP1 (451 `nexar_train` positives), CP2' window (§29.2), greedy decoding, 96 new tokens, Tab. B scale.

**Deviations to state every time a caption is quoted:** not LaGoVAD's prompt (dashcam framing + event type), Qwen2.5-VL-32B NF4 not Qwen2-VL-72B, 16 event-window frames of untrimmed dashcam video, one rater on 30.

**If QC fails:** no prompt change on the QC videos, §30 is not run, and the next step is the user's (Claude-written captions or close option C).

Code: `core/constants.py` (`NEXAR_CAPTION_MODEL`, `NEXAR_CAPTION_QUANT*`, `NEXAR_CAPTION_PROMPT`, `NEXAR_CAPTION_EVENT_TYPES`, `NEXAR_CAPTION_{DEV,QC}_*`, `NEXAR_CAPTION_OUTPUT_DIRNAME`), `core/tools/nexar_caption.py` (`event_type`, `dev_sample`, `qc_sample`, `prompt_tag`, `write_sheet`, `QwenCaptioner(quant)`, `run --subset dev --quant`), tests `core/tests/test_nexar_caption.py`, runbook `colab/final/nexar_captions.ipynb` (A100; step 6 = dev set, STOP).

**§29.3 dev amendment (dated 2026-11-17, after the first 10-video dev sheet, before any QC caption; user).** Prompt `1b16ce2e` labelled 10 / 10 dev videos `Near-collision`. The user set **four types with these definitions**: **Accident** = a severe, clearly visible crash (hard impact); **Collision** = contact, but light or not clearly visible (bump, scrape); **Near-collision** = no contact, hard braking or a last-moment stop; **Dangerous maneuver** = no contact, an abnormal swerve or steering action, possibly to avoid a crash. The prompt now lists them, asks the model to decide contact first and severity second, and opens with one of the four (`NEXAR_CAPTION_EVENT_TYPES`). Dev-only change; QC (CP6'') unchanged.

**§29.3 dev round 2 read-out and round 3 (dated 2026-11-17, before any QC caption; user rated the 10 dev videos).** Prompt `6a24b2d9`: Tab. B mean **3.8** (4, 3, 3, 3, 5, 3, 5, 3, 4, 5), `type_ok` 6 / 10, `roles_ok` 7 / 10. The dominant error is **contact under-called**: 5 / 10 dev videos show contact per the user (00013 light collision, 00100 and 00125 two-car collisions, 00362 a clear rear-end into a trailer, 00546 a parked car's opened door) and the model called all five `Near-collision`; 00362 holds the trailer fixed in view for the last 8 frames. Second, wrong actors (00100 car read as a motorcycle at night, 00125 a standing pedestrian invented as crossing, 00765 the cement truck instead of the white van on the left). **Round 3 changes (user):** (1) prompt v3: decide contact first with visible cues (fills the view and stops moving relative to the camera, camera jolt, door or mirror hit, vehicles stop touching, debris), answer `Contact: yes|no. <Type>: <description>`; the contact clause is parsed (`split_contact`) and **dropped from the caption `L_neg` reads**; a contact / type mismatch is printed (`contact_inconsistent`), never corrected; name only road users that take part; (2) **frames 448 × 252 → 700 × 392** (grid (8, 28, 50) = 2,800 visual tokens; 672 × 378 is not a multiple of 28); 128 new tokens. **Tuning cap, fixed now:** at most **2 more dev rounds** (this one included); the prompt is frozen iff a round reaches **dev mean ≥ 4.0 and `type_ok` ≥ 8 / 10**; otherwise Qwen captioning stops and the user chooses Claude-written captions or closing option C. Dev scores never stand in for QC (CP6''). Risk read in the same round: the 5 near-collision / manoeuvre dev videos (00474, 00715, 00765, 00921, 00938) show whether v3 over-calls contact.

**§29.3 dev round 3 read-out and round 4, the last (dated 2026-11-17, before round 4 ran; user rated round 3).** Prompt `e40d6bb5` at 700 × 392: Tab. B mean **4.4** (≥ 4.0), `type_ok` **6 / 10** (< 8), `roles_ok` 9 / 10, `Contact: yes` 1 / 10 (00362, right), 0 contact / type mismatches → **not frozen**. Misses are side or peripheral contact (00546 door on the right, 00765 white van on the left, 00125 light intersection contact); 00921 invents a crossing pedestrian again (rounds 1 and 3). The user's own reference moved between rounds 2 and 3 (00100, 00125, 00765), so the two `type_ok` counts are not comparable; **the reference is now frozen** (`NEXAR_CAPTION_DEV_TRUTH`, user, before round 4): 00013 collision or near-collision, 00100 near-collision, 00125 collision, 00362 accident, 00474 near-collision, 00546 accident, 00715 near-collision, 00765 collision, 00921 near-collision, 00938 dangerous maneuver or near-collision. Round 4 computes `type_ok` from it (`dev_type_ok`, prefilled in the dev csv, `dev_summary.json`); the user rates `score` and `roles_ok`. Round 4 prompt adds: side contact cues (door, a vehicle beside scraping, mirror; check the frame's left and right edges) and "mention a pedestrian only if one moves into a vehicle's path". Frames, window and tokens unchanged. **This is the last dev round (cap above):** freeze iff mean ≥ 4.0 and `type_ok` ≥ 8 / 10, else Qwen captioning stops.

**§29.3 dev round 4 read-out and round 5 (dated 2026-11-17; round 5 is the user's override of the round cap, written after the round-4 read-out).** Prompt `fc5b2c90`: `type_ok` **7 / 10** against the frozen reference (misses 00125 collision, 00546 accident, 00765 collision, all called `Near-collision`, identical to round 3), 00921 invents a pedestrian a third time; not frozen, and by the cap Qwen captioning would stop. The side-contact cue changed nothing. **The user asked for one more round (round 5), focused on defining the collision type.** Freeze criteria, the frozen reference and CP6'' are unchanged; this round is recorded as beyond the pre-registered cap, and no further round follows it without a new note. Round 5 prompt (`NEXAR_CAPTION_PROMPT` v5) flips the default: the model first states the closest moment (`Closest: ...`), answers `Contact: yes` unless a clear gap is visible at that moment, and `Collision` is defined by light or partly hidden contact (bump, scrape, side swipe, door or mirror hit, camera may barely shake); `Near-collision` requires a clear gap. The `Closest:` and `Contact:` clauses are dropped from the caption (`split_contact`, `closest_clause`); 160 new tokens. Risk read in the same round: over-calling contact on 00100, 00474, 00715, 00921, 00938.

**§29.3 dev round 5 read-out and the prompt for the 451 run (dated 2026-11-17; user).** Prompt `38feb0cc` (v5): `type_ok` **6 / 10**, `Contact: yes` **0 / 10** (00362 regressed from accident to near-collision), descriptions shorter with invented actors (00474 "yellow taxi", 00921 "black sedan") and the prompt's "clear gap" echoed. Worse than round 4 on every read. **The user chose to run the 451 videos with the round-4 prompt `fc5b2c90` (v4), restored verbatim (sha1 checked).** This freezes the prompt **by the user's decision, not by the freeze rule** (round 4 `type_ok` 7 / 10 < 8); say so wherever the captions are quoted. Known weakness carried into the captions: light / side contact is called `Near-collision` (dev: 00125, 00546, 00765). **CP6'' is unchanged and still decides**: the captions feed §30 only if the fresh 30-video QC reaches a Tab. B mean ≥ 4.0. The v5 parsing (`closest_clause`, `CONTACT` searched anywhere) stays in the tool; it reads v4 answers identically.

**§29.3 dev round 6: three types (dated 2026-11-17, before round 6 ran; user, superseding the v4 freeze above).** The user replaced the four types with three: **Accident** = a severe, clearly visible crash; **Collision** = contact, or contact more likely than not (> 50 %); **No-collision** only when the model is **at least 70 % sure** there was no contact (near-collisions and swerves). Below 70 % the answer must be Collision or Accident. The type now carries the contact call (`contact_of`); no `Contact:` clause is asked for. Prompt v6 keeps v4's contact cues (front and side), actor rule, ego naming and 40-word cap. The frozen dev reference is **mapped mechanically** (near-collision, dangerous maneuver → no-collision; 00013 stays collision or no-collision). Under the mapping rounds 4 and 5 would still read 7 / 10 and 6 / 10 (merging the no-contact types changes no count: every miss was a contact event called no-contact), so only the 70 % default can move `type_ok`. Freeze criteria and CP6'' unchanged. Plan: run the dev set (step 6) with v6 before the 451 run.

**§29.3 dev round 7: Qwen2.5-VL-72B, 32 frames (dated 2026-11-17, before round 7 ran; user).** The user now has an 80 GB GPU. Model **Qwen2.5-VL-32B → Qwen2.5-VL-72B-Instruct** (NF4, ~40 GB; same class and patch size, so no code path changes; closer to LaGoVAD's Qwen2-VL-72B). Frames **16 → 32** over the same 3.4 s window (~9.4 fps; 19 of 32 before contact), so contact shown in one or two native frames is less likely to fall between samples; grid (16, 28, 50) = 5,600 visual tokens. Prompt v6 unchanged (three types, 70 % rule). Freeze criteria, the mapped dev reference and CP6'' unchanged. The revision pin is now one file per model.

## 30. K5c / K5dc: per-video caption `L_neg` with and without DVS (dated note 2026-11-17, before any number; user)

**Question.** LaGoVAD Tab. 5 / Tab. G reports that removing `L_neg` or `L_dvs` lowers zero-shot detection. Do the two help K5, and do they **complement** each other (does `L_neg` help more with DVS on)? Read from LaGoVAD Tab. G (Det. Avg., one run, no interval): `L_neg` adds +2.46 with DVS and +2.20 without; DVS adds +2.91 with `L_neg` and +2.65 without; interaction ≈ **+0.26**, i.e. LaGoVAD shows the two as roughly additive. **Gated on §29.3 QC = pass.** Not a reopening of §28 / §28.3: K5 stays the main model unless the user decides otherwise.

| # | Choice | Value |
|---|---|---|
| C1 | Arms (2 × 2, s2024) | **K5** (DVS off, no `L_neg`; existing reads) · **K5d** (DVS on as W0, `kim.seam_mask=true`; §12's checkpoint, **inference only**) · **K5c** = K5 + `loss.cap_contrast_mode=caption` · **K5dc** = K5d + `loss.cap_contrast_mode=caption`. Config diff K5c − K5 = K5dc − K5d = the caption keys only (asserted) |
| C2 | Loss (`caption` mode) | `ClassContrastLoss` as §28.3 with **rows = the CLIP text feature of each source video's §29.3 caption** (full text, type prefix included) in place of `uniq`'s random target: windows of one source share it and are its positives; accident windows of other sources, normal windows and mined parts are negatives; video → text on with ≥ 2 sources. In a DVS-synthesized abnormal bag the caption is the **anchor's**; a synthesized normal bag has none (LaGoVAD's construction). **Similarity = cosine / 0.02** (LaGoVAD's τ, supp. C.2; user, 2026-11-17), weight 1.0. **Captions enter `L_neg` only**: the definition `z` stays the class texts (user, 2026-11-17), so LaGoVAD's random "descriptions as definition" draw is not used |
| C3 | Caption input | `nexar_captions_v2/captions.json` only, with `qc/qc_summary.json` verdict = pass, its `train_ids_sha1` = the current `nexar_train`, and one identity; anything else raises |
| C4 | Reads | As §28 NL4 for K5c and K5dc; **K5d is read on `nexar_val` (with `--save-scores`) before §30 trains**, so every cell has the same four sets. `nexar_test` is not touched |
| C5 | Sentences | Per set, paired per item, 95 % bootstrap, "above / below / not separated", PRELIMINARY (n = 1): **K5c − K5** (`L_neg` alone), **K5d − K5** (DVS alone), **K5dc − K5** (both), **K5dc − K5d** (`L_neg` with DVS), and the **interaction I = (K5dc − K5d) − (K5c − K5)** over the items common to all four |
| C6 | Verdict on S = Nexar val crops, DoTA-CAP-dev, DADA, DAD (TAD printed) | **COMPLEMENTARY** iff I is "above" on ≥ 2 of S and "below" on none. **ANTAGONISTIC** iff I is "below" on ≥ 2 of S. Otherwise **NO CLAIM** (additive or not separated) |
| C7 | What may follow | Whatever C6 reads, K5 stays unless **K5dc − K5 is "above" on ≥ 1 of S and "below" on none**: then K5dc is a **candidate** and the user decides (s2025 first). No τ, weight or prompt change after a number (lesson 14) |
| C8 | Printed, never deciding | §28 NL6 (false alarms on `nexar_val` negatives, detection at matched FA 0.20 / 0.283, scene split), P2 `sep` / `row_cos` vs K5, `kim_anchor_seam_share` vs `kim_seam_share` for the DVS arms, step-1 / last-epoch `cap_contrastive − floor` beside `mil`, the caption type histogram, and LaGoVAD Tab. G's matching cells beside each contrast (direction only) |

**Predictions (written now):** I "not separated" on every set of S → NO CLAIM. K5c − K5 "below" on ≥ 1 set (K5Lneg and K5Lnegu both cost; real captions sit between them). K5dc − K5 not "above" anywhere.

**Limits.** n = 1, and I's interval is about twice a single contrast's. Training corpus Nexar windows, not PreVAD. Captions are not used as definitions (deviation C2). τ = 0.02 here vs 0.07 in §28 / §28.3, so K5c − K5Lnegu differs in two keys and is not read. K5d's start check failed in §12.1 (step-1 `mil` 0.6906 vs W0 0.6944); K5dc − K5d shares K5d's host and draws, so its own start check is K5dc's step-1 `mil` against K5d's, printed.

**Build (after §29.3 passes, with the user's approval of the `train.py` blast radius):** `cap_contrast_mode=caption` (`core/constants.py`, `core/config.py`, `core/train.py`, `core/losses/contrastive.py`), the caption loader with the C3 checks, the anchor caption for synthesized bags (`core/data/synthesis.py` must expose the anchor's source; if it cannot, the build stops and says so), a 2 × 2 report tool with the interaction bootstrap, a runbook. Outputs `Thesis-Final/outputs/kim_lneg_cap/`.

### 30.1 Amendment: captions shared across source videos (dated note 2026-11-17, before any §30 number and before the 451 captions exist; user)

Windows of one source already share one row and are its positives (`uniq_rows` mapping, §28.3; `class_infonce_loss` spreads the target over them), and a source's normal windows stay negatives (LaGoVAD's same-scene hard negative). Two **different** sources can still get the same or nearly the same caption (the §29.3 dev set already shows three near-identical pedestrian captions), which would make one event a negative of another. Fixed now:

| Case | Rule |
|---|---|
| Identical caption text after normalization (lower case, punctuation and repeated spaces removed) | **merged into one row**; every source with it is a positive |
| Different sources, CLIP text cosine of the captions **≥ 0.95** | **masked**: neither positive nor negative (logit −∞ in the loss) |
| Otherwise | negative, as §30 C2 |

0.95 is a round number chosen by the user before any number, not tuned. Printed before training (after QC passes): the cross-source caption-cosine histogram, the merged-row count, and per batch in `metrics.jsonl` the masked share. If more than 20 % of cross-source pairs are masked, Claude reports it to the user before training; the threshold is not changed. Build: an optional ignore mask on `class_infonce_loss` (default `None`, so §28 / §28.3 are unchanged).

### 30.2 Captions generated, QC waived; §30 built (dated note 2026-11-17, before any §30 number; user)

**Captions.** The user ran all 451 `nexar_train` positives with the last dev setting: Qwen2.5-VL-72B-Instruct NF4, 32 frames at 700 × 392, prompt v6 `4d213779` (three types, the 70 % rule). Dev round 7 read `type_ok` 7 / 10 (contact recall 4 / 5, one false contact on 00715). **The user waived CP6'' (the 30-video QC) and C3's QC clause.** Every §30 number carries "captions not QC-rated; dev `type_ok` 7 / 10 below the freeze bar"; the manifest is still checked for its `nexar_train` binding (`core/data/captions.py`) and for covering every source with a positive window (runbook step 5).

**Built** (`train.py` blast radius HIGH, grep-based; the codebase-memory MCP is not available in this environment):
- `loss.cap_contrast_mode=caption` + `loss.caption_manifest`; `ClassContrastLoss` at cosine / `LNEG_CAPTION_TEMP` = 0.02. Rows = the CLIP text features of the batch's captions, encoded each step through `text_encode_fn` (as the class form's texts). A DVS item's `video_id` is its anchor, so a synthesized abnormal bag takes the anchor's caption (C2).
- §30.1: `caption_rows` merges normalized-identical captions into one row; `near_duplicates` marks rows with text cosine ≥ 0.95, and `class_infonce_loss(ignore=...)` drops those columns from the softmax and from the label-smoothing mass (equal to the unmasked loss when nothing is masked, tested). `cap_masked_share` is logged per batch; the runbook warns above 0.20.
- `core/tools/k5_lneg_cap_report.py`: the C5 contrasts, the interaction I per item over the items all four rows scored (cluster bootstrap, DoTA by source video), C6 and C7, false alarms on `nexar_val` negatives printed.
- Runbook `colab/final/k5_lneg_cap.ipynb`: trains K5c and K5dc (K5dc with the KNN cache, as K5d), reads K5d, K5c, K5dc on DoTA-CAP-dev, DADA, TAD, DAD and `nexar_val` with the current code (K5d's reads are redone here so all three rows share one eval path), then the report.
- **Not built (C8, printed items dropped):** P2 feature diversity and the per-frame curves. Detection at matched FA and the scene split are not in the report; the false-alarm share is.

Tests: `test_class_contrast.py` (caption rows, merge, near-duplicates, masked InfoNCE, `dup` in the loss), `test_train_units.py` (caption mode terms, merge, missing manifest, manifest without caption mode; the RNG-stream test now covers the caption mode), `test_k5_lneg_cap_report.py`. Not run in this environment (no torch); the runbook's step 4 runs them and raises if red.

**Review, 2026-11-17 (before any §30 number; user chose A).**
- *Same-source windows.* Accident windows of one source share its row (correct, tested). Every one of `Nexar_window`'s 5,622 normal windows comes from a captioned positive video, so at batch 64 about 2 normal windows per batch (≈ 30 / 451 per normal item, estimated, not measured) are negatives of their own video's caption. §30.1 keeps them (the same-scene hard negative); the risk is a post-event window whose aftermath matches the caption. **User: keep (A), log it.** `same_source_negatives` → `cap_same_source_neg` (count per batch, never in `total`); the runbook prints it beside the masked share. No rule changes.
- *Token budget.* CLIP text gets 77 − 32 soft prompts = 45 tokens (BOS + EOS included) and the tokenizer truncates silently. §29.1's 7B captions: median 29, 3 / 451 over; v6 allows a type prefix + 40 words (≈ 47–50 tokens), so the 451 v6 captions can lose their tail. Runbook step 5 now prints the over-budget share and the cut tails; the 451 set was not measured locally (it lives on Drive only).
- *Manifest identity.* The runbook preflight now asserts model = `NEXAR_CAPTION_MODEL`, quant nf4, frames = 32 and `prompt_sha1` = the constants prompt (v6 `4d213779`); before, it only printed them.
- Tests run locally this time (`.venv` has torch): `test_class_contrast.py`, `test_train_units.py`, `test_k5_lneg_cap_report.py`, 78 passed.

### 30.3 Read-out, s2024 (2026-11-17; raw `outputs/v_final/kim_lneg_cap/` synced) — the verdict does NOT test C2

**Report verdict (as written): C6 COMPLEMENTARY, C7 K5dc candidate = False.** NG4 reproduces (K5 `nexar_val` 0.8981 = §13). Macro / micro (Nexar = val crops; DoTA-CAP-dev, DADA min-max; TAD, DAD raw):

| set | K5 | K5d | K5c | K5dc |
|---|---|---|---|---|
| Nexar val | 0.8981 / 0.9000 | 0.8896 / 0.9117 | 0.8116 / 0.8149 | 0.8607 / 0.8861 |
| DoTA-CAP-dev | 0.8195 / 0.7449 | 0.8124 / 0.7054 | 0.7684 / 0.7427 | 0.7840 / 0.7434 |
| DADA | 0.7740 / 0.7635 | 0.7713 / 0.7383 | 0.6852 / 0.6908 | 0.7225 / 0.7236 |
| TAD (printed) | 0.7002 / 0.6781 | 0.7055 / 0.6679 | 0.6729 / 0.6152 | 0.6818 / 0.5966 |
| DAD | 0.7916 / 0.7050 | 0.7329 / 0.6646 | 0.7408 / 0.7169 | 0.7557 / 0.7108 |

K5c − K5 is "below" on all four deciding sets (−0.051 … −0.089); K5dc − K5 too (−0.036 … −0.052). I is "above" on all four only because K5c costs more than K5dc: **DVS reduces caption `L_neg`'s cost, neither adds**. This is not LaGoVAD Tab. G's pattern (both terms positive, I ≈ +0.26).

**Defect found in the read-out (post-hoc, decides nothing, but voids C2):** a windowed corpus's `meta.json` gives every window its **source's** `class_name` (`core/data/dada.py` window meta, `"class_name": record.class_name`), and `DVSFeatureDataset._class_name` prefers it over the label fallback. In `Nexar_window` every window comes from a positive video, so all 5,622 normal windows carry `CarAccident` → `cls_idx ≠ 0`. Evidence in the logs: `cap_same_source_neg` = 0 on all 1,749 steps of both arms (≈ 2 / batch expected), and `cap_mined` reaches 56 / 64 (mean 34–46 > the ~32 abnormal items per batch). Consequences:
1. §30's C2 was not run: in-video normal windows were **positives of their own video's caption**, not the same-scene negatives §30.1 describes; only mined parts were negatives. The same holds for §28 K5Lneg (class form: every item one class) and §28.3 K5Lnegu (per source), so those closures rest on the same mis-specified loss.
2. `L_MIL-align` (`multi_class_mil_loss`) has had a constant target (`CarAccident`) on every T2 and `Nexar_window` item, K5 included. It is identical across the arms compared here, so contrasts are unaffected, but the term has carried no Normal/abnormal signal on these corpora (pending (ac) saw the constant `class_name` and did not connect it to the loss).
3. `masked_share` falls 0.39 (step 1) → 0.03 (last epoch): the CLIP text path's soft prompts are trained, so §30.1's near-duplicate mask moves with them.

**Licensed:** "With the current label path, caption `L_neg` costs K5 on Nexar val, DoTA-CAP-dev, DADA and DAD at s2024, and DVS reduces that cost. PRELIMINARY (n = 1)." **Not licensed:** "`L_neg` and DVS are complementary", or any claim about §30's C2 design. Figures: `outputs/v_final/kim/REPORTS/figures/train_nexar/<set>/k5_lneg_cap_2x2_s2024/` (`core.tools.zs_curves`, 4 arms, grid + onset + every video; median per-video AUC K5 / K5d / K5c / K5dc: DADA 0.865 / 0.866 / 0.737 / 0.786). Fix: §31 (label-0 items always `Normal`; K5 retrained as K5f).

## 31. K5f: K5 retrained with label-0 items labeled Normal (dated note 2026-11-17, before any number; user)

**Why.** §30.3 found that `DVSFeatureDataset._class_name` returned the window's `meta.json` `class_name`, which a windowed corpus copies from the **source** video. In `Nexar_window` every window comes from a positive video, so all 5,622 normal windows had `cls_label = CarAccident`. K5's `L_MIL-align` (`multi_class_mil_loss`, `mul_weight` 1.0) therefore had a constant target: it ends at 0.0015 in K5's log. The detection path (`mil`, `dvs_sup*`, `kim_mil`) reads `label` and is not affected. The user asked for K5 retrained on the fixed code to get the correct reference.

**Fix (code, not config).** `_class_name` returns `Normal` for every label-0 item, whatever the meta says; label-1 items keep their meta class (`CarAccident`). Test `test_dvs_dataset.py::test_normal_item_is_normal_whatever_meta_says` fails on the old code and passes on the new one. This changes every future run on T2, `Nexar_window` and the other windowed corpora; MSAD / TAD / the DADA archive (unwindowed, normal videos already `Normal`) are unchanged.

| item | K5f |
|---|---|
| F1 Row | **K5f** = K5's exact command (`kim.enabled`, `kim.prototypes`, `kim.edge_mask`, DVS off, `Nexar_window`, A2 V2-S bake, s2024, 33 epochs, 1,749 steps). **Config diff to K5 = 0** (asserted); only the code differs. Output `kim_fixed/K5f/` |
| F2 Start check | step-1 `mil` equal to K5's within `KIM_START_TOL` (same init, data and forward); step-1 `mul_mil` differs (the target changed). Printed |
| F3 Fix check | K5f's last-epoch `mul_mil` above `KIM_FIXED_MUL_MIL_FLOOR` = 0.01 (K5: 0.0015). Printed; below it flags the run to Claude |
| F4 Reads | DoTA-CAP-dev (protocol B), DADA whole, TAD, DAD, `nexar_val` crops + whole, with K5's protocols and bakes; macro and micro on all four zero-shot sets (`zs_arm_table`, control K5) |
| F5 Status | **No adoption rule: by the user's instruction K5f replaces K5 as the corrected reference whatever the sign.** K5f − K5 is printed for the record (what the mislabel cost or bought), n = 1, "not separated" if the interval spans 0 |

**Not decided here:** re-running the §28 / §30 `L_neg` arms on the fixed code (they now have true normal columns), K5d, K5neg / K5nc2, W0 and the T2 line. Each needs its own dated note; until then every number measured before §31 carries "measured with the source-class labels (§30.3)".

**Expectation, written before the number:** macro "not separated" on most sets (the detection path is unchanged and `mul_mil` saturated early in K5); a sign on Nexar val or DADA would not surprise. A wrong expectation changes nothing.

Runbook `colab/final/k5_fixed.ipynb`. Outputs `Thesis-Final/outputs/kim_fixed/`.

### 31.1 Read-out, s2024 (2026-11-17; raw `outputs/v_final/kim_fixed/` synced)

**Checks.** J10 1,749 = K5. Config diff to K5 = 0 (only keys added after K5 trained, at their defaults). **F2 same start:** step-1 `mil` 0.687794 = K5's exactly; step-1 `mul_mil` 0.6932 vs 0.6668 (target changed). **F3 PASS:** last-epoch `mul_mil` **0.0963** (K5: 0.0015). K5's term was solved by epoch 1 (0.0014); K5f's falls slowly (epoch 0 0.695, 3 0.468, 8 0.348, 16 0.191, 32 0.096), so the multi-class target now carries a Normal / accident signal. NG4: K5's `nexar_val` re-read 0.8981 = §13.

| set | K5 macro / micro | K5f macro / micro | Δ macro K5f − K5 | Δ micro |
|---|---|---|---|---|
| Nexar val (crops) | 0.8981 / 0.9000 | 0.8942 / 0.8961 | −0.0039 [−0.0077, −0.0004] below | — |
| DoTA-CAP-dev | 0.8195 / 0.7449 | 0.8188 / 0.7353 | −0.0007 [−0.0034, +0.0019] not separated | −0.0096 [−0.0132, −0.0061] below |
| DADA | 0.7740 / 0.7635 | 0.7744 / 0.7551 | +0.0004 [−0.0026, +0.0035] not separated | −0.0084 [−0.0119, −0.0046] below |
| TAD (printed) | 0.7002 / 0.6781 | 0.6981 / 0.6711 | −0.0021 [−0.0123, +0.0078] not separated | −0.0070 [−0.0280, +0.0156] not separated |
| DAD | 0.7916 / 0.7050 | 0.8070 / 0.7072 | **+0.0154 [+0.0070, +0.0240] above** | +0.0021 [−0.0045, +0.0090] not separated |

Printed: Nexar val negatives with peak > 0.5: K5 0.717 → K5f **0.549**; AP 0.447 → 0.441.

**Reading.** The expectation ("not separated on most sets") held on macro: 3 of 5 not separated, Nexar val −0.004 (below, smallest interval in the table), DAD +0.015 (above). Micro falls ≈ 0.01 on the two min-max sets. Every Δ is ≤ 0.016 in size: **the mislabel cost or bought K5's detection little**; the user's prediction "very different" is not borne out at n = 1. The false-alarm share falls 0.17 at the 0.5 cut, but a fixed cut moves with the score scale (§17.1), so it is not a ranking claim.

**Licensed:** "K5 retrained with label-0 items labeled Normal (K5f) differs from K5 by at most 0.016 macro on any of the five sets at s2024; not separated on DoTA-CAP-dev, DADA and TAD, below on Nexar val (−0.004), above on DAD (+0.015). PRELIMINARY (n = 1)." **Not licensed:** "the fix improves / hurts K5", "L_MIL-align does nothing".

**Status (F5, user's instruction):** **K5f replaces K5 as the corrected reference.** Numbers measured before §31 stay quotable with "measured with the source-class labels (§30.3)"; their contrasts were against controls carrying the same defect. Where the defect mattered is `L_neg` (§28, §28.3, §30: the labels decide its positives and negatives) — those closures are not supported until re-run on the fixed code against K5f. Not decided: which arms to re-run.

## 32. §30's 2 × 2 re-run on the fixed label path, against K5f (dated note 2026-11-17, before any number; user)

**Why.** §30 ran with every normal window labeled `CarAccident` (§30.3), so C2's same-scene negatives never existed; the user asked for K5c and K5dc again on the §31 code, compared with K5f. **From §31.1 on, K5f replaces K5 completely: no new arm is compared with K5** (user).

| item | §32 |
|---|---|
| R1 Rows | **K5f** (control, §31 run and reads, read only) · **K5df** = K5d's command (§12: DVS θ 0.7 / 0.85, seam mask, KNN cache) on the fixed code · **K5cf** = K5c's command (§30 C2) on the fixed code · **K5dcf** = K5dc's command on the fixed code. Config diffs: K5df − K5d = 0; K5cf − K5 and K5dcf − K5d = the two caption keys only (asserted). Same captions manifest as §30 (prompt v6 `4d213779`, QC waived). s2024, 1,749 steps each |
| R2 Fix guard | every row's last-epoch `mul_mil` > `KIM_FIXED_MUL_MIL_FLOOR` (runbook asserts; `k5_lneg_cap_report --fixed` raises otherwise, so a pre-§31 run can never be a row or the control); caption rows: `cap_same_source_neg` > 0 on average (it was 0 on every §30 step) |
| R3 Contrasts | §30 C5 with the renamed rows: K5cf − K5f, K5df − K5f, K5dcf − K5f, K5dcf − K5df, and I = (K5dcf − K5df) − (K5cf − K5f), paired per item, cluster bootstrap |
| R4 Verdict | §30 C6 unchanged (COMPLEMENTARY / ANTAGONISTIC / NO CLAIM on Nexar, DoTA-CAP-dev, DADA, DAD; TAD printed). C7: K5dcf a candidate iff K5dcf − K5f is above on ≥ 1 set and below on none; K5f stays unless the user decides |
| R5 Wording | if C6 says COMPLEMENTARY while K5cf and K5dcf are both below K5f (§30.3's pattern), the read-out says "DVS reduces caption `L_neg`'s cost", not "complementary" |

**Expectation, written before the number:** K5cf costs less than K5c did against K5 (the in-video normals are now negatives, as designed), but is still below K5f on most sets; I "above" on some sets; C7 not a candidate. A wrong expectation changes nothing.

Runbook `colab/final/k5_lneg_cap_fixed.ipynb`; report `core.tools.k5_lneg_cap_report --fixed` → `REPORTS/k5f_lneg_cap_report.md`. Outputs `Thesis-Final/outputs/kim_lneg_cap_fixed/`.

### 32.1 Amendment: the caption rows use LaGoVAD's full caption path (dated note 2026-11-17, before any §32 number; user, option A + (i))

**Why.** Reading `LaGoVAD-PreVAD/src/models/LaGoVAD/lagovad.py` again (user's question: "a normal window should be pulled toward the shared Normal caption"): LaGoVAD's caption set is `[Normal caption, *abnormal captions]` (l. 576–577). It feeds a **caption branch** — `cap_bin_logits` gets the MIL and DVS-supervised terms (l. 593–627) and `cap_sim_mat` gets `L_MIL-align` with normal videos → **row 0, the Normal caption** (l. 647–660) — and `L_neg` on `query_caption_feats[1:]` (l. 666–675), where normal videos are negatives only. Captions are encoded with `use_soft_prompt=False`, 77 tokens (l. 535–538). §30 (C2, "captions enter `L_neg` only") had no Normal caption anywhere, no caption branch, and soft-prompted captions (45 tokens, gradient into the prompts shared with the class texts). Claude did not say at §30 that dropping the branch drops the pull toward Normal.

| item | §32.1 |
|---|---|
| A1 Mode | new `loss.cap_contrast_mode=caption_lagovad` (`CAP_CONTRAST_MODE_CAPTION_LAGOVAD`); `caption` (§30) is kept unchanged for the record |
| A2 Caption set | `[NORMAL_QUERY_CAPTION (LaGoVAD's literal "Normal behavior, views or surveillance content …"), *per-source captions]`; windows of one source and identical captions share a row (§30.1, deviation: LaGoVAD gives each abnormal video its own index). No abnormal item in a batch → no caption branch that step (LaGoVAD: `query_captions = None`) |
| A3 Caption branch | the model fuses `v` with the caption features → `cap_bin_logits` (MIL + DVS sup added to `mil` / `dvs_sup*`, existing code path) and `cap_sim_mat` (`L_MIL-align`, label-0 items → row 0, abnormal → its source's row), weights as in the baseline |
| A4 `L_neg` | `ClassContrastLoss` on the caption branch's own features (row 0 never a positive row), cosine / 0.02, §30.1 merge + near-duplicate mask, as §30 |
| A5 Encoding | captions through `caption_encode_fn` = CLIP text **without** soft prompts, 77 tokens; class texts keep the soft prompts |
| A6 2 × 2 | interpretation (i) (user): the caption factor = "LaGoVAD caption path" (branch + `L_neg` together), so K5f and K5df stay the controls; nothing else is retrained. K5cf / K5dcf = K5f / K5df + `caption_lagovad` + the manifest. §32 R2–R5 unchanged |
| A7 Later | the user plans a second arm with a dashcam Normal caption in place of LaGoVAD's sentence; not specified here |

**Code.** `core/train.py`: `caption_lagovad` branch, `caption_encode_fn` (built with `use_soft_prompt=False`), baseline `CapContrastLoss` restricted to `cap_contrast_mode=baseline`. Tests (`test_train_units.py`): the caption encoder receives `[Normal, captions]` and the soft-prompt encoder only the class texts; `mil` and `mul_mil` differ from `caption` mode; refused without a caption encoder; **K5f's path (no captions) gives identical losses with or without the caption encoder**; the RNG-stream test covers the new mode.

**Expectation, revised before the number:** K5cf closer to K5f than §32's prediction (the Normal pull and the unprompted captions remove two of the three suspected costs); still "below" on at least one set; C7 not a candidate. A wrong expectation changes nothing.

### 32.2 Read-out, s2024 (2026-11-18; raw `outputs/v_final/kim_lneg_cap_fixed/` synced) — R5 fires: no row beats K5f

**Report verdict (as written by `k5_lneg_cap_report --fixed`): C6 COMPLEMENTARY (I above on Nexar val, DADA, DAD; below on none), C7 K5dcf candidate = False.** By R5 the read-out does not say "complementary" (below).

**Checks.** J10: every row at step 1,749. Config diffs: K5cf − K5f = K5dcf − K5df = `loss.cap_contrast_mode` (`baseline` → `caption_lagovad`) + `loss.caption_manifest` only; K5df − K5d = three keys added after K5d trained, at their defaults (`cap_contrast_mode=baseline`, `caption_manifest=""`, `model.motion_norm=none`); K5df − K5f = `dvs.theta` 0.7, `dvs.theta_ego` 0.85, `kim.seam_mask` (§12). Start: K5df step-1 `mil` 0.690560 = K5d's exactly. K5cf / K5dcf step-1 `mil` 1.3855 / 1.3815 ≈ 2 × K5f's because §32.1 A3 adds the caption branch's MIL into `mil` — not comparable to K5f's, as designed. **R2 PASS:** last-epoch `mul_mil` K5f 0.0963, K5df 0.170, K5cf 0.256, K5dcf 0.557 (all > 0.01; the caption rows include the caption branch's `L_MIL-align`); `cap_same_source_neg` mean 2.13 (K5cf) / 2.08 (K5dcf) per batch (§30: 0 on every step; §30.2 estimated ≈ 2). `cap_masked_share` 0.004 (step 1) → 0.005 (last epoch), no drift: captions no longer pass the trained soft prompts (§30.3 item 3 was 0.39 → 0.03). Last-epoch `cap_contrastive − floor`: K5cf 0.404, K5dcf 0.640.

Macro (Nexar = val crops; DoTA-CAP-dev, DADA min-max; TAD, DAD raw):

| set | K5f | K5df | K5cf | K5dcf |
|---|---|---|---|---|
| Nexar val | 0.8942 | 0.8871 | 0.8612 | 0.8785 |
| DoTA-CAP-dev | 0.8188 | 0.8164 | 0.8083 | 0.8131 |
| DADA | 0.7744 | 0.7728 | 0.7470 | 0.7564 |
| TAD (printed) | 0.6981 | 0.7033 | 0.6947 | 0.7103 |
| DAD | 0.8070 | 0.7562 | 0.8051 | 0.7800 |

Contrasts, paired per item, 95 % cluster bootstrap:

| set | K5cf − K5f | K5df − K5f | K5dcf − K5f | K5dcf − K5df | I |
|---|---|---|---|---|---|
| Nexar val | −0.0330 [−0.0412, −0.0248] below | −0.0071 [−0.0164, +0.0023] ns | −0.0158 [−0.0261, −0.0048] below | −0.0087 [−0.0175, +0.0002] ns | +0.0244 [+0.0152, +0.0337] above |
| DoTA-CAP-dev | −0.0105 [−0.0195, −0.0011] below | −0.0024 [−0.0100, +0.0058] ns | −0.0057 [−0.0149, +0.0041] ns | −0.0033 [−0.0097, +0.0030] ns | +0.0072 [−0.0010, +0.0154] ns |
| DADA | −0.0274 [−0.0377, −0.0170] below | −0.0016 [−0.0114, +0.0081] ns | −0.0181 [−0.0295, −0.0067] below | −0.0164 [−0.0258, −0.0069] below | +0.0110 [+0.0012, +0.0207] above |
| TAD (printed) | −0.0034 [−0.0276, +0.0215] ns | +0.0052 [−0.0224, +0.0337] ns | +0.0122 [−0.0157, +0.0409] ns | +0.0070 [−0.0138, +0.0278] ns | +0.0104 [−0.0118, +0.0331] ns |
| DAD | −0.0019 [−0.0275, +0.0228] ns | −0.0508 [−0.0751, −0.0273] below | −0.0270 [−0.0536, −0.0016] below | +0.0238 [−0.0023, +0.0492] ns | +0.0256 [+0.0054, +0.0462] above |

Printed: Nexar val negatives with peak > 0.5: K5f 0.549, K5df 0.673, K5cf 0.611, K5dcf 0.619 (fixed cut, scale-dependent, §17.1).

**Reading (R5).** No cell of K5cf / K5df / K5dcf − K5f is "above" on any set. I is "above" on three sets only because each factor costs less with the other present: on Nexar val and DADA, DVS reduces the caption path's cost (K5cf −0.033 / −0.027 → K5dcf −0.016 / −0.018); on DAD, the caption path reduces DVS's cost (K5df −0.051 → K5dcf −0.027). Neither factor adds over K5f. This is not LaGoVAD Tab. G's pattern (both terms positive, I ≈ +0.26). DVS alone (K5df − K5f) is not separated on three sets and below on DAD, consistent with §12's closure.

**Expectation (§32.1, revised before the number): held.** K5cf is closer to its control than K5c was to K5 (§30.3 K5c − K5: −0.087 / −0.051 / −0.089 / −0.051 on Nexar val / DoTA-CAP-dev / DADA / DAD; K5cf − K5f: −0.033 / −0.011 / −0.027 / −0.002); still below on three sets; C7 not a candidate. That cross-section comparison is post-hoc and confounds two changes (the §31 label fix and the §32.1 caption path), so the shrink is attributed to neither.

**Licensed:** "On the fixed label path, the LaGoVAD caption path (caption branch + `L_neg`) costs K5f on Nexar val, DoTA-CAP-dev and DADA at s2024; DVS costs K5f on DAD; with both on, the model is below K5f on Nexar val, DADA and DAD. Each factor reduces the other's cost; neither adds. PRELIMINARY (n = 1); captions not QC-rated (dev `type_ok` 7 / 10)." **Not licensed:** "`L_neg` and DVS are complementary", "captions help K5", or any claim beyond s2024.

**Status.** K5f stays the reference (C7 False). Not decided: closing the caption `L_neg` line, the A7 dashcam-Normal-caption arm, or a second seed; each needs the user and a dated note.


## 33. K5cfn: the dashcam Normal caption (§32.1 A7), last arm of the caption `L_neg` line (dated note 2026-11-18, before any number; user)

**Line closure (user, 2026-11-18).** The caption `L_neg` line (§28–§32) **closes**: no further caption, `L_neg` or caption-branch variant is specified or run. The single exception is this arm (§32.1 A7). After §33's read-out the line is closed whatever the number, unless R4 names K5cfn a candidate; even then K5f stays the reference unless the user decides. No second seed for §32 or §33 (user). Reopening takes a new dated note.

**Why.** In `caption_lagovad` mode, row 0 of the caption set is LaGoVAD's literal `NORMAL_QUERY_CAPTION` ("Normal behavior, views or surveillance content captured by camera …", `core/data/definitions.py:23`). It is the target of every label-0 item in the caption branch's `L_MIL-align` and a row of `cap_bin`. Nexar, DoTA, DADA and DAD are dashcam footage. §15.1 measured that swapping the *class definition* at inference does nothing (text branch `1 − w` ≈ 0.08), but that is not this question: here the sentence is a training target.

| item | §33 |
|---|---|
| R1 Rows | **K5f** (control, read only, `kim_fixed/K5f`) · **K5cf** (§32, read only) · **K5cfn** = K5cf's command + `loss.normal_caption=dashcam`. s2024, 1,749 steps. Outputs `Thesis-Final/outputs/kim_lneg_cap_fixed/K5cfn/` |
| R2 Sentence | `"Normal driving recorded by a dashcam that does not involve any collision or unusual event."` = `NORMAL_PROBE_DASHCAM_DEFS[3]` (§15, the long sentence, same shape as LaGoVAD's); not a new wording. Encoded like every caption (`caption_encode_fn`, no soft prompts, 77 tokens). Class texts unchanged |
| R3 Contrasts | **primary K5cfn − K5cf** (one sentence differs: does the "surveillance" target cost?) and **K5cfn − K5f**; paired per item, 95 % cluster bootstrap (DoTA: source video), Nexar val crops, DoTA-CAP-dev / DADA min-max, DAD raw, TAD printed. Decision sets S = `KIM_LNEG_CAP_SETS` (Nexar val, DoTA-CAP-dev, DADA, DAD) |
| R4 Verdict | **candidate** iff K5cfn − K5f is above on ≥ 1 set of S and below on none (§30 C7's form). Otherwise the line is closed with no candidate. K5cfn − K5cf is reported as "the Normal sentence matters on {sets}" (above / below per set) and decides nothing on its own |
| R5 Guards | J10 (ckpt step = metrics' last step = 1,749); config diff K5cfn − K5cf = `loss.normal_caption` only (asserted in the runbook); last-epoch `mul_mil` > `KIM_FIXED_MUL_MIL_FLOOR` (every row); **`cap_same_source_neg` equal to K5cf's at every step** (same seed, same batches, the sentence does not touch batch composition: the report raises otherwise) |

**Expectation, written before the number:** K5cfn − K5cf not separated on ≥ 3 of the 4 sets of S; K5cfn not a candidate (§32.2: the whole caption path costs, one row-0 sentence cannot carry that). A wrong expectation changes nothing.

**Limits.** n = 1; one wording (Claude's §15 sentence, approved by the user); captions still not QC-rated (§30.2); K5cf and K5f were read before this note, so the controls are not blind.

**Code.** `loss.normal_caption` ∈ `NORMAL_CAPTION_CHOICES` (`lagovad` default, `dashcam`); `dashcam` outside `cap_contrast_mode=caption_lagovad` raises. Report `python -m core.tools.k5_a7_report` → `REPORTS/k5cfn_report.{md,json}`; runbook `colab/final/k5_a7_dashcam_normal.ipynb`.

### 33.1 Read-out, s2024 (2026-11-18; raw `outputs/v_final/kim_lneg_cap_fixed/K5cfn/` + `REPORTS/k5cfn_report.{md,json}` synced) — not a candidate: the caption `L_neg` line is CLOSED

**Report verdict (R4, as written by `k5_a7_report`): K5cfn candidate = False** (K5cfn − K5f above on none, below on Nexar val, DoTA-CAP-dev, DADA) → **the caption `L_neg` line is closed** (§33). K5f stays the reference.

**Checks (R5).** J10: 1,749 steps. Config diff K5cfn − K5cf = `loss.normal_caption` (`dashcam`) + the staged `caption_manifest` VM path (same Drive manifest). Last-epoch `mul_mil` K5f 0.0963, K5cf 0.2555, K5cfn 0.2276 (> 0.01). **`cap_same_source_neg` equal to K5cf's at all 1,749 steps** (same batches). Wiring, from the logs: step 1 `kim_mil` and `cap_contrastive` are identical to K5cf's (same init; `L_neg` drops row 0 by design, §32.1 A4), while `mul_mil` (4.2775 → 4.3211), `mil` and `dvs_sup` differ (row 0 reaches the caption branch's `L_MIL-align` / MIL only, as specified); the two runs stay equal on `kim_mil` until step 7 (AMP's GradScaler skips the first updates). `cap_masked_share` 0.0048 in both.

| set | K5f | K5cf | K5cfn | K5cfn − K5cf (primary) | K5cfn − K5f |
|---|---|---|---|---|---|
| Nexar val | 0.8942 | 0.8612 | 0.8585 | −0.0027 [−0.0061, +0.0008] ns | −0.0357 [−0.0442, −0.0275] below |
| DoTA-CAP-dev | 0.8188 | 0.8083 | 0.8024 | −0.0058 [−0.0085, −0.0032] below | −0.0164 [−0.0255, −0.0069] below |
| DADA | 0.7744 | 0.7470 | 0.7443 | −0.0027 [−0.0064, +0.0008] ns | −0.0301 [−0.0400, −0.0204] below |
| TAD (printed) | 0.6981 | 0.6947 | 0.6980 | +0.0033 [−0.0081, +0.0176] ns | −0.0001 [−0.0241, +0.0249] ns |
| DAD | 0.8070 | 0.8051 | 0.7972 | −0.0079 [−0.0221, +0.0054] ns | −0.0098 [−0.0368, +0.0156] ns |

Printed, decides nothing: Nexar val negatives with peak > 0.5: K5f 0.549, K5cf 0.611, K5cfn 0.558 (fixed cut, scale-dependent, §17.1). Last-epoch `mul_mil` is lower with the dashcam sentence (0.2276 vs 0.2555): post-hoc, the dashcam row is an easier `L_MIL-align` target for normal windows; it does not reach detection.

**Expectation (§33, before the number): held.** K5cfn − K5cf is not separated on 3 of the 4 sets of S (below on DoTA-CAP-dev); K5cfn is not a candidate.

**Licensed:** "At s2024, replacing LaGoVAD's surveillance Normal caption with a dashcam sentence does not recover the LaGoVAD caption path's cost against K5f (still below on Nexar val, DoTA-CAP-dev, DADA); against K5cf it is not separated on Nexar val, DADA and DAD and costs −0.006 on DoTA-CAP-dev. PRELIMINARY (n = 1), one wording, captions not QC-rated." **Not licensed:** "the Normal wording never matters" (one wording, one seed), "dashcam text hurts", or any claim about the class definitions (§15 is that question).

**Status: caption `L_neg` line CLOSED (§28–§33).** No further caption / `L_neg` / caption-branch arm; reopening takes a new dated note. K5f stays the reference and the main model.

## 34. K5dfi / K5dfie: DVS anchor mode `ignore` on the DVS-on host, and the ego DVS setting (dated note 2026-11-19, before any number; user)

**Why.** Every KIM row so far trains with `loss.dvs_anchor_mode=span` (13 saved configs, K0 … K5dcf; the LaGoVAD default, never chosen for this corpus: pending (bj), C29). `span` marks the whole spliced anchor positive in the dense `L_dvs-sup` BCE; `ignore` drops it from that BCE and leaves the positive pressure to `pseudo_sup_mil_loss`. On the v2 host it was worth +0.040 macro on DoTA-CAP-dev (A3 → A3-ign, `RESULTS_E4_0.md`); on the DADA archive Phase 1 it moved macro −0.0056 (`RESULTS_DADA_PHASE1.md`, FALSIFIED). Neither is this host, so it is a new question. The user asked for it on K5f, then added the ego setting. Plan: `.project/plans/katvad-kim-k5dfi-dvs-ignore.md`.

**"K5f + ignore" is not run: it is the same loss.** K5f has DVS off (`dvs.theta = dvs.theta_ego = 1.0`): `dataset.py:183` never splices, `synthesis.py:80-84` leaves `pseudo` all zero, and `ignore_positive` masks only positives, so the mask is empty (`test_baseline_losses.py::test_ignore_positive_is_a_noop_on_an_all_normal_row`). The arm that can show an effect has DVS on: K5df (§32).

**What `is_egocentric=true` changes (read from `dataset.py:107-111`, the only consumer, `train.py:1069`).** `theta` 0.7 → `theta_ego` 0.85 (splice probability 0.30 → 0.15) and `max_clips` `min(5, delta_m)` → `min(5, delta_m_ego)` = 2, so a spliced item has exactly one filler and one seam. K5dfie therefore carries **lighter DVS plus `ignore`**, not an ego-specific mechanism; read it as "less DVS". Nothing else reads the flag (`motion_aware_knn_key` is a separate knob and stays false).

| item | §34 |
|---|---|
| R1 Rows | **K5f** (reference, read only, `kim_fixed/K5f`) · **K5df** (control, read only, `kim_lneg_cap_fixed/K5df`) · **K5dfi** = K5df's command + `loss.dvs_anchor_mode=ignore` · **K5dfie** = K5dfi + `data.is_egocentric=true`. s2024, 1,749 steps, `Nexar_window`, A2 V2-S bake, KNN cache as K5df. Outputs `Thesis-Final/outputs/kim_dvs_ignore/<row>/` |
| R2 Contrasts | **primary K5dfi − K5df**; **secondary K5dfie − K5dfi** (ego setting, both `ignore`); printed: K5dfie − K5df, K5dfi − K5f, K5dfie − K5f, K5df − K5f. Paired per item, 95 % cluster bootstrap; Nexar val crops, DoTA-CAP-dev / DADA min-max, DAD raw, TAD printed; **macro and micro** (`zs_arm_table`) on the four zero-shot sets |
| R3 Margin | `V5_MIN_GAIN` = +0.01, not lowered after the number (lesson 14) |
| R4 Verdict (primary and secondary, same rule) | **GAIN** iff the contrast is ≥ +0.01 with the interval above 0 on DoTA-CAP-dev or DADA **and** it is a COST on none of Nexar val / DoTA-CAP-dev / DADA / DAD; **COST** iff ≤ −0.01 with the interval below 0 on any of those four; else **NO DIFFERENCE**. TAD never decides |
| R5 Candidate | a row is a candidate iff its contrast with K5f is above on ≥ 1 of the four sets and below on none (C7's form). K5f stays the reference and main model unless the user decides, whatever the sign |
| R6 Guards | J10 (ckpt step = metrics' last step = 1,749); config diff to K5df = `loss.dvs_anchor_mode` (K5dfi) / + `data.is_egocentric` (K5dfie), asserted in the runbook; last-epoch `mul_mil` > `KIM_FIXED_MUL_MIL_FLOOR`. **Mask live:** K5dfi's step-1 `dvs_sup` differs from K5df's (`k5_dvs_ignore_report` raises if equal). **Start check (flag, never stops):** K5dfi's step-1 `mil` and `dvs_sup_mil` equal K5df's within `KIM_START_TOL` (same draws; `ignore` changes neither). K5dfie has no matched start (its splice draws differ). If the flag fires, the read-out says the contrast has no matched start; a same-runtime K5df retrain is then the user's call (§12.1 flag 1) |
| R7 Wording | if K5dfi is above K5df but below K5f: "`ignore` reduces DVS's cost", not "improves the model". K5dfie − K5dfi mixes `is_egocentric`'s two effects (probability, clips per splice): never "ego helps". `ignore` under the ego setting without `ignore` under `span` is not measured (no K5dfe) |

**Expectation, written before the number:** K5dfi − K5df small, NO DIFFERENCE on most sets (DVS on top of KIM already showed nothing: K5d − K5 on DADA −0.0027, §12.1); a sign on DoTA-CAP-dev or DADA would not surprise. K5dfie lands between K5f and K5dfi (fewer splices); neither row a candidate; K5f stays. A wrong expectation changes nothing.

**Limits.** n = 1; no s2025 unless R4 says GAIN and the user asks. The share of an anchor that is truly anomalous on `Nexar_window` is **not measured** (the plan's P0 print was dropped: windows carry only source/start/end, and the ignored share is not logged); lesson (bh): `ignore` lowers raw positive scores, so any raw "pre-anomaly mean" falls for free — min-max per clip first. K5df's reads exist from §32 and are not blind.

**Code.** No model or loss change: the flag and `data.is_egocentric` exist. New: `constants.KIM_DVS_IGNORE_*`, `core/tools/k5_dvs_ignore_report.py` (`python -m core.tools.k5_dvs_ignore_report` → `REPORTS/k5dfi_report.{md,json}`), test `core/tests/test_k5_dvs_ignore_report.py` (21 pass; the report was not rendered on real reads: per-clip scores of the new rows do not exist yet), runbook `colab/final/k5_dvs_ignore.ipynb`. Not committed.

### 34.1 Read-out, s2024 (2026-11-19; raw `outputs/v_final/kim_dvs_ignore/` + `REPORTS/k5dfi_report.{md,json}` synced) — R4 primary: COST (Nexar val only); ego: NO DIFFERENCE; neither row a candidate

**Guards (checked on the synced files).** J10 1,749 both rows (`verified.json`, last `metrics.jsonl` row). Config diff to K5df: K5dfi = `loss.dvs_anchor_mode` (+ `loss.normal_caption: lagovad`, a key added after K5df trained, at its default); K5dfie = + `data.is_egocentric`. Last-epoch `mul_mil`: K5f 0.0963, K5df 0.1700, K5dfi 0.1519, K5dfie 0.1228 (> 0.01). **Start check:** K5dfi step-1 `mil` 0.6906 and `dvs_sup_mil` 0.7803 equal K5df's (same draws, matched start); step-1 `dvs_sup` 0.7044 → 0.7039 (barely over `KIM_START_TOL`: at init every logit ≈ 0 so BCE ≈ ln 2 whatever the target; the weak step-1 guard is not the evidence that the mask is live). **Mask live, from the logs:** epoch-mean `dvs_sup` K5df 0.447 → 0.186 (epochs 0 → 32), K5dfi 0.292 → 0.042, K5dfie 0.260 → 0.034; `dvs_sup_mil` last epoch 0.156 / 0.123 / 0.094.

| macro, s2024 | K5f | K5df | K5dfi | K5dfie |
|---|---|---|---|---|
| Nexar val (crops) | 0.8942 | 0.8871 | 0.8698 | 0.8723 |
| DoTA-CAP-dev | 0.8188 | 0.8164 | 0.8259 | 0.8263 |
| DADA | 0.7744 | 0.7728 | 0.7677 | 0.7724 |
| TAD (printed) | 0.6981 | 0.7033 | 0.7198 | 0.7214 |
| DAD | 0.8070 | 0.7562 | 0.7494 | 0.7508 |

| contrast (paired, 95 % cluster bootstrap) | Nexar val | DoTA-CAP-dev | DADA | DAD | TAD (printed) |
|---|---|---|---|---|---|
| **K5dfi − K5df (primary)** | **−0.0174 [−0.0267, −0.0088] below** | +0.0095 [+0.0033, +0.0159] above | −0.0051 [−0.0132, +0.0033] ns | −0.0068 [−0.0284, +0.0145] ns | +0.0165 [−0.0064, +0.0399] ns |
| K5dfie − K5dfi (secondary) | +0.0025 ns | +0.0004 ns | +0.0047 ns | +0.0014 ns | +0.0016 ns |
| K5dfi − K5f | −0.0245 below | +0.0071 ns | −0.0067 ns | −0.0576 below | +0.0217 ns |
| K5dfie − K5f | −0.0220 below | +0.0075 above | −0.0020 ns | −0.0562 below | +0.0233 above |
| K5df − K5f | −0.0071 ns | −0.0024 ns | −0.0016 ns | −0.0508 below | +0.0052 ns |

**R4 verdicts.** Primary **COST**: Nexar val is ≤ −0.01 with the interval below 0 (the rule lets one set decide). DoTA-CAP-dev +0.0095 is **0.0005 under the +0.01 margin** (§12.1's miss again: not restated as a pass), and it could not have made a GAIN anyway with a COST on Nexar. Secondary (ego) **NO DIFFERENCE** on all five sets. **R5:** K5dfi not a candidate (below K5f on Nexar val and DAD); K5dfie not a candidate (above on DoTA-CAP-dev, below on Nexar val and DAD). K5f stays.

**Printed, post-hoc, never decides.**
- **DAD's cost is DVS's, not `ignore`'s:** K5df − K5f −0.0508 already; K5dfi − K5df −0.0068 ns.
- **Raw-pooled micro (point values from `results.json`, no interval):** TAD K5f 0.6711, K5df 0.6538, K5dfi **0.6119**, K5dfie 0.6546; K5dfi − K5f −0.0592 [−0.0943, −0.0224] below (`zs_arm_table`, K5dfie ns). DAD micro K5df 0.6613, K5dfi 0.7091. TAD micro measures clip classification (oracle 0.9226): "the floor drop of §bh shows on a raw-pooled micro" is a hypothesis, not a measured mechanism.
- **Micro vs K5f, `zs_arm_table` (control K5f; per-clip scores of K5df are not local, so no micro vs K5df):** DoTA-CAP-dev K5dfi +0.0261 [+0.0188, +0.0335] above, K5dfie +0.0227 above; DADA +0.0040 / +0.0047 ns; DAD +0.0019 / +0.0010 ns.
- Nexar val whole macro K5df 0.9305 → K5dfi 0.9043; clip AUC 0.7522 → 0.7709; false alarms at cut 0.5 on negatives K5f 0.549, K5df 0.673, K5dfi 0.681, K5dfie 0.584 (fixed-cut shares move with the score scale, §17.1: not a ranking claim).

**Gap to state.** The notebook's report step ran `k5_dvs_ignore_report` only (macro); §34 R2 promised macro and micro on all four sets. Micro of K5dfi − K5df is not computed because K5df's per-clip scores are not in the local sync; K5f's are, hence the micro vs K5f above.

**Expectation check (written before the number).** "NO DIFFERENCE on most sets, a sign on DoTA-CAP-dev or DADA would not surprise, neither row a candidate, K5f stays": held for 4 of 5 sets, DoTA-CAP-dev showed the sign, and both rows are not candidates. Not predicted: a **COST on Nexar val** (the training domain), −0.0174.

**Licensed (n = 1, PRELIMINARY).** "On the K5df host at s2024 with a matched start, `dvs_anchor_mode=ignore` costs Nexar val −0.0174 [−0.0267, −0.0088], moves DoTA-CAP-dev +0.0095 (below the +0.01 margin), and shows no difference on DADA, DAD and TAD. `is_egocentric=true` on top (θ_ego 0.85, ≤ 2 clips per splice) shows no difference on any set. Neither row separates above K5f where it counts." **Not licensed:** "`ignore` hurts the model" (one seed, one set decides), "the anchor label is not a lever" (the cost on Nexar val is also what losing the dense positive pressure would produce: `dvs_sup` falls 0.186 → 0.042 and only one frame per anchor, k = max(1, n_pos // 4), is pushed up by `dvs_sup_mil`), "ego helps / hurts" (R7), any mechanism for the TAD micro drop.

**Status.** §34 closed at n = 1: K5dfi and K5dfie are not candidates, K5f stays the reference and the main model. No s2025 (R4 did not say GAIN). **Not decided (user):** a K5dfi + `loss.sup_mil_topk_pct=2` arm (more anchor frames pushed up through the existing MIL term, no code change; it also widens the top-k on normal rows, so it moves two things) or an `ignore_top` loss; neither is specified or run. Reopening takes a new dated note.

### 34.2 Micro and AP, now with K5df's per-clip scores (2026-11-19; `zs_arm_table`, controls K5df and K5f; printed, R4 decides on macro only, so §34.1's verdicts do not change)

Raw: `outputs/v_final/kim_dvs_ignore/REPORTS/zs_arm_table_dvs_ignore_vs_k5df.{md,json}` and `..._vs_k5f.{md,json}` (run locally after K5df's `clip_scores.npz` were synced; closes §34.1's "Gap").

| micro, s2024 | DoTA-CAP-dev | DADA | TAD | DAD |
|---|---|---|---|---|
| K5f | 0.7353 | 0.7551 | 0.6711 | 0.7072 |
| K5df | 0.7038 | 0.7349 | 0.6538 | 0.6613 |
| K5dfi | 0.7613 | 0.7591 | 0.6119 | 0.7091 |
| K5dfie | 0.7580 | 0.7598 | 0.6546 | 0.7081 |

| Δ micro, paired, 95 % cluster bootstrap | DoTA-CAP-dev | DADA | TAD | DAD |
|---|---|---|---|---|
| K5dfi − K5df | +0.0576 [+0.0494, +0.0655] above | +0.0241 [+0.0164, +0.0317] above | −0.0419 [−0.0781, −0.0050] below | +0.0478 [+0.0355, +0.0591] above |
| K5dfie − K5df | +0.0542 [+0.0445, +0.0636] above | +0.0248 [+0.0160, +0.0332] above | +0.0008 [−0.0324, +0.0327] ns | +0.0468 [+0.0358, +0.0576] above |
| K5df − K5f | −0.0315 below | −0.0201 below | −0.0173 ns | −0.0459 below |
| K5dfi − K5f | +0.0261 above | +0.0040 ns | −0.0592 below | +0.0019 ns |

**Reading (n = 1, printed).** Under DVS the pooled micro falls against K5f on three sets (K5df − K5f); `ignore` raises it again on the same three (K5dfi − K5df +0.058 / +0.024 / +0.048, intervals clear of 0) while macro stays flat (§34.1). §34 R7's wording applies: "`ignore` reduces DVS's micro cost on DoTA-CAP-dev, DADA and DAD", not "improves the model". The one set where it goes the other way is TAD (−0.042, K5dfie ns). **§34.1's hypothesis that a lower positive floor shows on raw-pooled micro is not supported as a rule:** DAD (raw) micro rises +0.048 under `ignore`; only TAD micro falls (TAD micro measures clip classification, oracle 0.9226). K5dfie's micro matches K5dfi's within 0.007 on every set. The primary verdict stays COST (Nexar val −0.0174, macro, R4); micro was not in R4.

**What the micro gain is (decomposition, 2026-11-19, post-hoc, printed; ad-hoc computation on the synced `clip_scores.npz`, not a tracked tool).** Pooled micro AUC = Mann-Whitney over all (positive frame, negative frame) pairs of the per-clip min-max curves. Split the pairs into same-clip pairs and cross-clip pairs: **cross-clip pairs are 99.8 % of the pairs on DoTA-CAP-dev and 99.7 % on DADA**, so micro is, to rounding, a cross-clip AUC, and macro (same-clip ranking) is what micro does not measure. Min-max removes each clip's offset and scale, so what is left to compare across clips is the **shape** of the normalized curve (broad plateau vs narrow peak): a broad clip's negative frames stay high after min-max and outrank a narrow clip's positive frames.

| set | arm | micro | AUC, same-clip pairs | AUC, cross-clip pairs | negative frames > 0.5 after min-max | mean normalized score, positive / negative frames |
|---|---|---|---|---|---|---|
| DoTA-CAP-dev | K5f | 0.7353 | 0.8018 | 0.7352 | 0.325 | 0.650 / 0.351 |
| DoTA-CAP-dev | K5df | 0.7038 | 0.8022 | 0.7036 | 0.497 | 0.740 / 0.507 |
| DoTA-CAP-dev | K5dfi | 0.7613 | 0.8125 | 0.7613 | 0.236 | 0.601 / 0.269 |
| DoTA-CAP-dev | K5dfie | 0.7580 | 0.8091 | 0.7579 | 0.241 | 0.593 / 0.270 |
| DADA | K5f | 0.7551 | 0.7581 | 0.7551 | 0.226 | 0.580 / 0.261 |
| DADA | K5df | 0.7349 | 0.7574 | 0.7349 | 0.336 | 0.664 / 0.377 |
| DADA | K5dfi | 0.7591 | 0.7510 | 0.7591 | 0.161 | 0.514 / 0.202 |
| DADA | K5dfie | 0.7598 | 0.7522 | 0.7598 | 0.174 | 0.526 / 0.211 |

**Reading (n = 1).** Under `span` (K5df) the normalized curves are broad (49.7 % of DoTA-CAP-dev negative frames above 0.5), which is what the onset-aligned curves show (`figures/train_nexar/{dota,dada}/dvs_ignore_s2024/onset_aligned.png`: K5df high before the onset and slow to fall after the GT ends); `ignore` narrows them (23.6 %) and the cross-clip AUC rises with it. **The same-clip AUC does not follow:** DoTA-CAP-dev +0.010, DADA **−0.006** (K5df → K5dfi). So the micro gain is a curve-width effect and says nothing about better localization within a clip, which is what macro (flat, §34.1) measures. Corrects the wording of this section's earlier discussion: the micro difference is **not** a raw-scale difference between clips (min-max removes it); it is shape. Licensed: "`ignore` narrows the normalized score curves of K5df; pooled micro on DoTA-CAP-dev and DADA rises with that narrowing while same-clip ranking is unchanged". Not licensed: "`ignore` localizes better", "K5dfi beats K5f on DoTA-CAP-dev / DADA" (macro vs K5f not separated: +0.0071, −0.0067), and any selection of K5dfi on the micro of these two sets (R5 is stated on four sets, §34).

## 35. K5f, K5dfi, K5dfie and the released LaGoVAD checkpoint on whole DoTA-CAP (dated note 2026-11-21, before any number; user)

**Why.** Every DoTA number of the KIM line is DoTA-CAP-dev (569 clips). The user asked for the whole DoTA-CAP, test part included, to compare the three rows on the set the thesis reports (dev + eval = 1,129 clips, `DOTA_CAP_MANIFEST`). `dota_cap_eval` (560) is in `V2_SEALED_SPLITS`; v2's Final step opened it once for A0-A3 (2026-10-06, `OPENED.json`). **This is the first read of a KIM arm on it, opened by the user's request**, with `protocol_b_eval --final`. The user then asked (same day, before any number) for the **released LaGoVAD checkpoint** as the comparison row on the same clips: its DoTA-CAP-dev read exists (`zs_lagovad/eval/dota/LaGoVAD/B`, `ZS_TAD_DADA.md` §7, protocol B, macro 0.6418); its eval read is new here. Notebook `colab/final/k5_cap_eval.ipynb`; marker `Thesis-Final/outputs/kim_cap_eval/OPENED.json`.

**No new extraction (expected).** v2 §29 (`nexar_zeroshot.ipynb`) already baked `A2_S/dota_cap_eval_s3_from_s1_nexar` (CLIP `DoTA_s1_ncc` + V2-S `DoTA_CAP_s1_squash`, Nexar statistics, `s1[::3]`) and read it with A2 (560 clips): that is K5's input. The notebook verifies the bake (manifest `crn` none, `motion` V2-S, `train_ids_sha1` of `Nexar_s8`, 560 ids) and rebuilds it on CPU only if missing. If the V2-S rows of the eval clips are absent it stops; their extraction (`colab/v2/dota_cap_videomae.ipynb`, CAP tars) is not part of this note.

| # | Choice |
|---|---|
| R1 Rows | **K5f** (reference and control), **K5dfi**, **K5dfie**; s2024, step 1,749, checkpoints read only, no retraining. K5dfi / K5dfie are printed beside K5f, not as candidates (§34 closed them). **LaGoVAD** = the released `Thesis/ckpts/best.ckpt` (`--baseline released`, the file of Gate T0 and of its dev read), on the plain CLIP rows `DoTA_s1_ncc[::3]`; no bake, no motion stream (it has none) |
| R2 Sets | **whole DoTA-CAP only** (user, 2026-11-21) = the row's dev read (569, its own earlier run) + the new eval read (560), concatenated per clip, no re-scoring. The eval read is the missing half, not a reported set: no eval-only number is printed or tabled |
| R3 Metrics | macro = mean per-clip frame-level AUC over two-class clips, 95 % source-video cluster CI; micro and AP after per-clip min-max (every clip holds an accident, C12); paired Δ vs K5f by `zs_arm_table`. Named **DoTA-CAP (n/1397)** (D14), never beside LaGoVAD's 62.60 or a protocol-A number |
| R4 Guards | J10 (ckpt step = metrics' last = `verified.json`, 1,749), config of each row as trained (K5f `dvs.theta` 1.0; K5dfi / K5dfie `ignore`, K5dfie ego), reads carry the checkpoint's own `global_steps`, dev and eval ids disjoint (569 + 560). LaGoVAD: dev and eval reads both `released_baseline` on the same file path, protocol `s1[::3]`; before it is scored, `zs_lagovad_readout check-inputs` asserts its plain rows equal the bake's CLIP columns on all 560 eval clips (`ZS_TAD_DADA.md` §7.3-1), so every row sees the same CLIP frames; `zs_arm_table` refuses rows with other clips or other frame labels |
| R5 Decision | **none.** n = 1, printed only. K5f stays the reference and the main model whatever the numbers; a row above K5f here changes nothing (it was not above on the sets §34 decided on) and a row below it is not a new finding. No arm is selected, retrained or dropped from this read; reopening a line takes its own dated note |
| R6 Wording | "above / below / not separated" with the interval; never "K5dfi beats K5f on DoTA" from a micro gain (§34.2: micro follows curve width, same-clip ranking is what macro measures). The LaGoVAD row is named "LaGoVAD (public ckpt, protocol B)" and never sits beside its printed 62.60 (lesson 8b, D14) |

**Expectation, written before the number:** K5dfi and K5dfie close to K5f on macro (dev: +0.0071 ns / +0.0075 above), micro above K5f by ~0.02-0.03 as on dev; whole-set macro within ~0.02 of each row's dev macro (v2's eval was within ±0.02 of dev). LaGoVAD well below K5f on whole DoTA-CAP, as on dev (dev: LaGoVAD − K5f macro −0.177 [−0.206, −0.149], micro −0.125, computed locally 2026-11-21 from the two dev reads with `zs_arm_table` before any eval number). A wrong expectation changes nothing.

### 35.1 Read-out (2026-11-21, n = 1, s2024; printed, decides nothing)

Raw: `outputs/v_final/kim_cap_eval/` (synced; `REPORTS/zs_arm_table_cap_all_s2024.{md,json}`, `OPENED.json` 2026-11-21T06:36Z). **Guards pass:** every eval read is `dota_cap_eval`, 560 clips, all two-class, protocol `s1[::3]`; KIM rows at step 1,749 with `crn` none / `motion` V2-S, LaGoVAD `released_baseline` on `Thesis/ckpts/best.ckpt`. Checked locally: in each row's `dota_cap_all` file the 569 dev curves are bit-identical to the row's own dev read and the 560 eval curves to its eval read (569 + 560 = 1,129, disjoint); `zs_arm_table` re-run locally on the four `dota_cap_all` dirs reproduces the report to the fourth decimal.

**DoTA-CAP (1,129/1397), protocol B, micro / AP after per-clip min-max, control K5f**

| row | macro [95 %] | micro | AP |
|---|---|---|---|
| K5f | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 |
| K5dfi | 0.8240 [0.8136, 0.8342] | 0.7603 | 0.5899 |
| K5dfie | 0.8270 [0.8165, 0.8371] | 0.7587 | 0.5931 |
| LaGoVAD (public ckpt, protocol B) | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 |

Base rate 0.3260.

| contrast | Δ macro | | Δ micro | |
|---|---|---|---|---|
| K5dfi − K5f | +0.0014 [−0.0042, +0.0069] | not separated | +0.0242 [+0.0192, +0.0293] | above |
| K5dfie − K5f | +0.0044 [+0.0007, +0.0084] | above | +0.0225 [+0.0181, +0.0268] | above |
| LaGoVAD − K5f | −0.1761 [−0.1943, −0.1579] | below | −0.1164 [−0.1352, −0.0983] | below |

**Reading.** (1) **K5f is above the released LaGoVAD checkpoint on whole DoTA-CAP** by +0.176 macro [+0.158, +0.194], +0.116 micro and +0.175 AP, on the same 1,129 clips and the same CLIP frames (check-inputs). This is the comparison the thesis's DoTA row can carry (as "DoTA-CAP, protocol B, n = 1"), not beside LaGoVAD's printed 62.60. (2) K5dfi / K5dfie sit where they sat on dev: macro not separated / just above (+0.0044, below §34's 0.01 margin), micro above by ~0.02, i.e. §34.2's curve-width effect; nothing reopens §34, and per R5 K5f stays the reference and the main model. (3) Whole-set macro of each KIM row is within 0.004 of its dev macro (K5f 0.8188 → 0.8226, K5dfi 0.8259 → 0.8240, K5dfie 0.8263 → 0.8270); LaGoVAD 0.6418 → 0.6465.

**Expectation check.** Written before the number: KIM rows close to K5f on macro, micro above by ~0.02-0.03, whole within ~0.02 of dev, LaGoVAD well below K5f. All held.

**Licensed:** "on DoTA-CAP (dev + eval, 1,129 clips), K5f is above the released LaGoVAD checkpoint by +0.176 macro (n = 1)". **Not licensed:** any eval-only number (R2), "K5dfie beats K5f" (+0.0044 is under the margin and §34 decided on four sets), "+17.6 points over LaGoVAD's paper number", or a seed claim (n = 1; the LaGoVAD row has no seed).

### 35.2 DADA-2000 with the LaGoVAD row (2026-11-21, user; post-hoc table, no new read, decides nothing)

The user asked for the same four rows on DADA-2000 after §35.1. **No model was run:** every row's DADA read already existed (K5f §31, K5dfi / K5dfie §34, released LaGoVAD `zs_lagovad/eval/dada/LaGoVAD`, `ZS_TAD_DADA.md` §7); all four are the same protocol (370 T2-test source videos, whole video, every s8 row, labels = annotation span). `zs_arm_table` run locally → `outputs/v_final/kim_cap_eval/REPORTS/zs_arm_table_dada_lagovad_s2024.{md,json}`.

**DADA-2000 (370 whole sources, s8), micro / AP after per-video min-max, control K5f, n = 1**

| row | macro [95 %] | micro | AP |
|---|---|---|---|
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 |
| K5dfie | 0.7724 [0.7487, 0.7960] | 0.7598 | 0.5607 |
| LaGoVAD (public ckpt) | 0.6202 [0.5925, 0.6480] | 0.6128 | 0.3532 |
| *position only: middle ruler (no pixels)* | *0.7342 [0.7076, 0.7594]* | | |
| *position only: monotone ruler `t/L`* | *0.5514 [0.5247, 0.5782]* | | |

Base rate 0.2885.

| contrast | Δ macro | | Δ micro | |
|---|---|---|---|---|
| K5dfi − K5f | −0.0067 [−0.0156, +0.0024] | not separated | +0.0040 [−0.0044, +0.0128] | not separated |
| K5dfie − K5f | −0.0020 [−0.0086, +0.0046] | not separated | +0.0047 [−0.0011, +0.0110] | not separated |
| LaGoVAD − K5f | −0.1542 [−0.1865, −0.1220] | below | −0.1423 [−0.1734, −0.1127] | below |

**Reading.** (1) K5f is above the released LaGoVAD checkpoint on DADA-2000 by +0.154 macro, +0.142 micro, +0.199 AP. (2) **Print the ruler beside it:** a no-pixel middle-of-video ruler scores 0.7342 on the same 370 sources; LaGoVAD's 0.6202 is *below* it and K5f's 0.7744 is above it by only ~0.04. So the contrast licenses "K5f ranks accident frames better than LaGoVAD here", not "K5f localizes accidents well on DADA" — most of K5f's macro is reachable by position alone. (3) K5dfi / K5dfie not separated from K5f on either metric (as §34.1); K5f stays.

**Licensed:** "on DADA-2000 (370 whole sources), K5f is above the released LaGoVAD checkpoint by +0.154 macro (n = 1); a position-only ruler scores 0.734". **Not licensed:** any DADA number beside a published frame-level number, or K5f's DADA macro quoted without the ruler.

## 36. L0dfi: the LaGoVAD trunk on K5dfi's recipe and corpus (dated note 2026-11-21, before any number; user)

**Why.** §35.1 / §35.2 compare K5f with the *released* LaGoVAD checkpoint, which was trained on other data. The user asked for LaGoVAD trained on our Nexar corpus with **K5dfi's recipe** (the user's choice over K5f's, 2026-11-21), read zero-shot on DoTA-CAP and DADA-2000, so that new vs old model is compared at the same data and recipe. v2's Nexar A0 (5 seeds) is close but not the same recipe (DVS `span`, pre-§31 window labels: last-epoch `mul_mil` ≈ 0.001); it is printed beside, not used as the control.

| # | Choice |
|---|---|
| R1 Row | **L0dfi** = K5dfi's training command (§34: Nexar_window, DVS θ 0.7 / θ_ego 0.85, `loss.dvs_anchor_mode=ignore`, §31 labels, 33 epochs = 1,749 steps, s2024) with **`kim.enabled=false`, `v2.motion=none`, `model.motion_dim=0`**; `kim.prototypes` / `kim.edge_mask` / `kim.seam_mask` return to their defaults (false) because the config refuses them without KIM (`seam_mask` only masks KIM's pairs). Input = the plain CLIP cache `Nexar_s8_ncc` (no manifest = A0 input) |
| R2 Sets | **whole DoTA-CAP** (dev 569 + eval 560 = 1,129; protocol B on plain `DoTA_s1_ncc[::3]`; dev and eval read, then pooled by `core.tools.dota_cap_pool`; `dota_cap_eval` read with `--final` under this note) and **DADA-2000** (370 whole T2-test sources, s8, plain `DADA2000_orig` CLIP). Nexar val, TAD, DAD are not read (the user asked for the two sets) |
| R3 Metrics | as §35: macro [95 % CI], micro and AP after per-clip / per-video min-max, `zs_arm_table` paired Δ. Rows: **K5dfi (control)**, L0dfi, K5f, v2 Nexar A0 (5-seed average, printed), LaGoVAD (public ckpt, protocol B). DADA beside its middle ruler 0.7342 |
| R4 Guards | J10 (1,749); config diff to K5dfi = exactly {`kim.enabled`, `kim.prototypes`, `kim.edge_mask`, `kim.seam_mask`, `v2.motion`, `model.motion_dim`}; last-epoch `mul_mil` > 0.01 (§31 fix); `zs_lagovad_readout check-inputs`: `Nexar_s8_ncc` == the A2 Nexar bake's CLIP columns on all 900 train videos, `DADA2000_orig` == the A2 DADA bake's on 370, `DoTA_s1_ncc[::3]` == the A2 DoTA-CAP bakes' on 569 + 560 (every row sees the same CLIP frames); step-1 numbers vs K5dfi printed only (no matched start: the model and its parameter draws differ) |
| R5 Decision | **none.** The contrast **K5dfi − L0dfi** is printed as "the model's effect (VideoMAE motion + KIM) at fixed data and recipe"; K5f stays the reference and the main model whatever the number. n = 1 |
| R6 Wording | "above / below / not separated" with the interval. Never "KIM gives +x" from this contrast: it moves motion and KIM together (A2 − A0 on Nexar already shows V2-S alone at +0.08…+0.15). Never put L0dfi beside LaGoVAD's 62.60 |

**Expectation, written before the number:** L0dfi close to v2 Nexar A0 (DoTA-CAP 0.719, DADA 0.617), within ~0.03; K5dfi − L0dfi ≈ +0.10 on DoTA-CAP and ≈ +0.15 on DADA, both above; L0dfi on DADA near or below the middle ruler (0.734). A wrong expectation changes nothing.

Built: `constants.KIM_LAGOVAD_NEXAR_*`, `DOTA_CAP_*`; `core/tools/dota_cap_pool.py` + `core/tests/test_dota_cap_pool.py` (13 tests; one re-pools K5f's §35 reads and reproduces the committed file); notebook `colab/final/l0dfi_lagovad_nexar.ipynb`.

### 36.1 Read-out (2026-11-21, n = 1, s2024; printed, decides nothing)

Raw: `outputs/v_final/kim_lagovad_nexar/` (`REPORTS/zs_arm_table_l0dfi_s2024.{md,json}`, `OPENED.json` 2026-11-21T08:36Z, user). **Guards checked locally:** J10 = `verified.json` step 1,749 (= last `metrics.jsonl` step); config diff to `outputs/v_final/kim_dvs_ignore/K5dfi/K5/s2024/config.yaml` = exactly {`model.motion_dim` 384 → 0, `v2.motion` → none, `kim.enabled`, `kim.prototypes`, `kim.edge_mask`, `kim.seam_mask` → false} (R4); last-epoch `mul_mil` 0.30–0.35 (> 0.01, §31 fix live). **Not re-checked here:** the `check-inputs` CLIP-frame equality and the step-1 print (taken from the notebook).

**DoTA-CAP (dev + eval, 1,129 clips, protocol B), micro / AP after per-clip min-max, control K5dfi; base rate 0.3260**

| row | macro [95 %] | micro | AP |
|---|---|---|---|
| K5dfi (control) | 0.8240 [0.8136, 0.8342] | 0.7603 | 0.5899 |
| **L0dfi** | **0.6770 [0.6610, 0.6921]** | **0.6423** | **0.4738** |
| K5f | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 |
| LaGoVAD (public ckpt) | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 |
| v2 Nexar A0 (5-seed, span, pre-§31) | 0.7187 [0.7025, 0.7344] | 0.6678 | 0.5008 |

**DADA-2000 (370 whole sources, s8), micro / AP after per-video min-max; base rate 0.2885**

| row | macro [95 %] | micro | AP |
|---|---|---|---|
| K5dfi (control) | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 |
| **L0dfi** | **0.5752 [0.5544, 0.5962]** | **0.5748** | **0.3623** |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 |
| LaGoVAD (public ckpt) | 0.6202 [0.5925, 0.6480] | 0.6128 | 0.3532 |
| v2 Nexar A0 (5-seed, span, pre-§31) | 0.6167 [0.5907, 0.6433] | 0.6070 | 0.3794 |
| *position only: middle ruler (no pixels)* | *0.7342 [0.7076, 0.7594]* | | |

| contrast (paired) | Δ macro | | Δ micro | |
|---|---|---|---|---|
| DoTA-CAP: L0dfi − K5dfi | −0.1470 [−0.1611, −0.1334] | below | −0.1180 [−0.1307, −0.1063] | below |
| DADA: L0dfi − K5dfi | −0.1925 [−0.2151, −0.1694] | below | −0.1842 [−0.2031, −0.1639] | below |

**Reading.** (1) At the same corpus and recipe, the motion model (K5dfi: VideoMAE motion + KIM) is above the plain-CLIP trunk by +0.147 macro on DoTA-CAP and +0.193 on DADA (AP +0.116 / +0.193) — §36 R5's "model's effect" contrast. Per R6 it moves motion and KIM together; never "KIM gives +x". (2) **L0dfi is not close to A0** (expectation missed): 0.042 below A0 on both sets (0.677 vs 0.719; 0.575 vs 0.617) — outside the ±0.03 the note predicted. A0 differs in DVS (`span`, θ defaults) and in pre-§31 labels, so no cause is licensed; no paired L0dfi − A0 contrast was run (intervals above are unpaired). (3) **L0dfi vs the released LaGoVAD:** point estimates 0.677 vs 0.647 on DoTA-CAP (L0dfi higher) and 0.575 vs 0.620 on DADA (L0dfi lower); both intervals overlap on DADA and nearly touch on DoTA, and no paired contrast was printed — "not separated" is the only licensed wording. (4) On DADA L0dfi sits **below the position-only ruler** (0.734): it ranks frames worse than position alone; K5dfi / K5f sit ~0.03–0.04 above it. (5) R5: decides nothing; K5f stays the reference and the main model.

**Expectation check.** Written before the number: L0dfi within ~0.03 of A0 (**missed**, −0.042 on both); K5dfi − L0dfi ≈ +0.10 DoTA-CAP / +0.15 DADA, both above (**direction held, sizes larger**: +0.147 / +0.193); L0dfi on DADA at or below the middle ruler (**held**, 0.575 vs 0.734). A wrong expectation changes nothing.

**Licensed:** "trained on Nexar with K5dfi's recipe, the plain-CLIP trunk L0dfi scores 0.677 macro on DoTA-CAP and 0.575 on DADA-2000 (n = 1); K5dfi is above it by +0.147 / +0.193". **Not licensed:** "KIM gives +0.15", "new data fixes LaGoVAD" or "LaGoVAD retrained is worse/better than released" (no paired contrast), any DADA number without the ruler, any seed claim.

## 37. K5fT2 and K5dfiT2: K5f and K5dfi retrained on T2, read on whole DoTA-CAP (dated note 2026-11-21, before any number; user)

**Why.** The only T2-trained KIM row is K5T2 (§13), trained before the §31 label fix (last-epoch `mul_mil` 0.0026) and read on DoTA-CAP-dev only. The user asked for K5f and K5dfi trained on T2 and read zero-shot on DoTA-CAP. DADA-2000 is **not** read: T2 is cut from the DADA-2000 original release, so DADA is in-domain (held-out sources) for these rows, not zero-shot (§13 T6).

| # | Choice |
|---|---|
| R1 Rows | **K5fT2** = K5T2's training command (§13: `DADA2000_orig_v2`, A2 bake with T2 statistics, T2 KNN, 20 epochs = 1,740 steps, s2024) run on the §31 code; its config equals K5T2's (K5T2 vs K5f differs only in `data.dataset`, `train.num_epochs`, checked locally 2026-11-21). **K5dfiT2** = K5fT2 + K5dfi's four keys `dvs.theta=0.7`, `dvs.theta_ego=0.85`, `kim.seam_mask=true`, `loss.dvs_anchor_mode=ignore` |
| R2 Sets | **whole DoTA-CAP only** (dev 569 + eval 560, protocol B, the T2-statistics bakes `A2_S/DoTA_CAP_s3_from_s1` (dev, E3's) and `A2_S/DoTA_CAP_eval_s3_from_s1` (eval, v2 Final's; rebuilt on CPU with `build_v2_inputs apply` if missing); eval read with `--final` under this note; pooled by `dota_cap_pool`). K5T2's existing checkpoint also gets its eval read so its row pools |
| R3 Metrics | as §35. Rows: **K5fT2 (control)**, K5dfiT2, K5T2 (pre-§31, printed), K5f and K5dfi (Nexar-trained, §35 reads), LaGoVAD (public ckpt, protocol B). Same clips and frame labels (`zs_arm_table` refuses otherwise); inputs differ only in the normalization statistics (T2 vs Nexar) |
| R4 Guards | J10 (1,740); config diff to K5T2 = {} (K5fT2) / exactly the four keys above (K5dfiT2); last-epoch `mul_mil` > 0.01 (§31 fix live; K5T2 0.0026); bakes' manifests `crn` none, `motion` V2-S, `train_ids_sha1` = the T2 bake's; step-1 numbers vs K5T2 printed (K5fT2 should start like K5T2 except the classification target) |
| R5 Decision | **none.** n = 1, printed. K5f (Nexar) stays the reference and the main model; a T2 row above it changes nothing (§13.1: training corpus is not a lever we select on). Contrasts printed: K5dfiT2 − K5fT2 (DVS + ignore on T2), K5fT2 − K5T2 (§31 fix on T2), K5fT2 − K5f and K5dfiT2 − K5dfi (training corpus, same recipe) |
| R6 Wording | "above / below / not separated" with the interval; micro differences read as §34.2 (curve width). Never "T2 is a better training set" from one seed and one set |

**Expectation, written before the number:** K5fT2 within ±0.01 of K5T2 (the fix moved K5 → K5f by −0.0007 on DoTA-CAP-dev); K5dfiT2 − K5fT2 macro not separated, micro above (as on Nexar); K5fT2 − K5f not separated or slightly below (§13.1: K5T2 − K5 −0.011 ns on dev). A wrong expectation changes nothing.

Notebook `colab/final/k5_t2_fixed.ipynb`; outputs `Thesis-Final/outputs/kim_t2_fixed/`.

### 37.1 Read-out (2026-11-21, s2024, n = 1; printed, decides nothing)

Raw outputs synced to `outputs/v_final/kim_t2_fixed/` (`OPENED.json` 09:24Z, `dota_cap_eval` opened with `--final` under this note for K5fT2, K5dfiT2 and K5T2). Report `REPORTS/zs_arm_table_t2_fixed_s2024.md`.

**Guards (R4), checked locally on the synced files.** J10: both rows `verified.json` step 1,740 = last `metrics.jsonl` step. Config diff K5dfiT2 − K5fT2 = exactly `dvs.theta` 1.0 → 0.7, `dvs.theta_ego` 1.0 → 0.85, `kim.seam_mask` false → true, `loss.dvs_anchor_mode` span → ignore. K5fT2 − K5T2: no value differs; the diff holds only four keys added to the config after K5T2 was trained, each at its default (`motion_norm: none`, `cap_contrast_mode: baseline`, `caption_manifest: ''`, `normal_caption: lagovad`). §31 fix live: last-epoch mean `mul_mil` K5fT2 **0.0483**, K5dfiT2 **0.1060** (K5T2 0.0019). Step 1: `mil` K5fT2 = K5T2 = 0.7389.

**Whole DoTA-CAP (dev 569 + eval 560 = 1,129, protocol B), control K5fT2:**

| row | trained on | macro [95 %] | micro | AP |
| --- | --- | --- | --- | --- |
| **K5fT2** (control) | T2 | 0.8091 [0.7973, 0.8208] | 0.7173 | 0.5399 |
| K5dfiT2 | T2 | 0.7969 [0.7850, 0.8087] | 0.7236 | 0.5400 |
| K5T2 (pre-§31 labels) | T2 | 0.8118 [0.7996, 0.8237] | 0.7206 | 0.5402 |
| K5f | Nexar | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 |
| K5dfi | Nexar | 0.8240 [0.8136, 0.8342] | 0.7603 | 0.5899 |
| LaGoVAD (public ckpt) | PreVAD | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 |

| contrast (R5) | Δ macro | | Δ micro | |
| --- | --- | --- | --- | --- |
| K5dfiT2 − K5fT2 (DVS + ignore on T2) | −0.0121 [−0.0174, −0.0068] | below | +0.0064 [+0.0005, +0.0122] | above |
| K5T2 − K5fT2 (pre-fix − fix, on T2) | +0.0027 [+0.0006, +0.0050] | above | +0.0033 [+0.0004, +0.0063] | above |
| K5f − K5fT2 (Nexar − T2, K5f recipe) | +0.0136 [+0.0020, +0.0249] | above | +0.0189 [+0.0073, +0.0314] | above |
| K5dfi − K5fT2 | +0.0150 [+0.0051, +0.0247] | above | +0.0431 [+0.0317, +0.0550] | above |
| LaGoVAD − K5fT2 | −0.1625 [−0.1796, −0.1456] | below | −0.0975 [−0.1147, −0.0799] | below |
| K5dfiT2 − K5dfi (T2 − Nexar, K5dfi recipe; local `zs_arm_table`, control K5dfi, same seed / draws) | −0.0271 [−0.0368, −0.0173] | below | −0.0367 [−0.0486, −0.0257] | below |

**Against the expectations.** K5fT2 within ±0.01 of K5T2: met (−0.0027). K5dfiT2 − K5fT2 macro "not separated": **wrong, below by 0.012**; micro above: met. K5fT2 − K5f "not separated or slightly below": met, below by 0.014.

**Licensed:** "Trained on T2 with the same recipe, K5f scores below its Nexar-trained self on whole DoTA-CAP (macro −0.014 [−0.025, −0.002], micro −0.019), K5dfi's recipe likewise (K5dfiT2 − K5dfi macro −0.027 [−0.037, −0.017]); both T2 rows stay far above the public LaGoVAD checkpoint (macro +0.16). On T2, adding DVS + `ignore` (K5dfiT2) lowers DoTA-CAP macro by 0.012 [0.007, 0.017] and raises micro by 0.006. The §31 label fix moves the T2 row by −0.003 macro. PRELIMINARY (n = 1), one set." **Not licensed:** "Nexar is a better training set" or "T2 is worse" (one seed, one set; §13.1: the training corpus is not a lever we select on); "DVS hurts on T2" as a recipe claim (one seed; on Nexar the same keys were not separated on DoTA-CAP, §35.1); any DADA sentence (in-domain for T2 rows, §13 T6). K5f stays the reference and the main model (R5).

## 38. K5fno: normals only from Nexar's negative videos — S0 source-shortcut probe first (dated note 2026-11-21, before any number; user)

**Why.** TAD micro falls whenever Nexar's negative videos enter training (K5 0.678 → K5nc2 0.643 → K5neg 0.588 → K5-whole 0.466), but every one of those arms kept K5's in-video normals or changed the positive side too (§16, §17, §27), so "TAD micro is low because K5's normals are cut from accident videos" has never been tested alone. The user proposed **K5fno** = K5f with the positive windows unchanged and the normal pool drawn **only** from the 449 negative `nexar_train` videos, its size matched to K5f's (≈ 5,622; the sampler already pairs one positive with one normal per item, so the class ratio per step does not change). **Not built.** It is built only if S0 passes.

**Risk S0 tests.** In K5f a normal window shares its video's scene, weather and camera with the positive one. In K5fno it does not, and `nexar_train` is scene-skewed by class (Sub-urban 41 positive / 122 negative, Rain 42 / 14, §17). If the two normal pools separate on the model's input, K5fno's label is readable from the video, not the event (C38: every imported normal pool so far had shortcut AUC 1.000).

| # | Choice |
|---|---|
| S0-1 Data | the built `Nexar_window_neg` (§17) train split: **A** = 5,622 normal windows inside positive videos, **B** = 6,879 negative-video windows, **P** = 1,689 positive windows (`meta.json` `source_is_abnormal`, `positive_frames`) |
| S0-2 Reads | (i) plain CLIP `Nexar_s1_ncc[::8]` (the corpus rows); (ii) the A2 V2-S bake `A2_S/Nexar_s8` = K5f's training input. Window = mean of its 20 rows |
| S0-3 Probe | `core.tools.nexar_source_probe`: the EDA logistic probe (`transfer_scores`: standardized, class-balanced), 5-fold GroupKFold by source video, seed 2024 |
| S0-4 Gate | **S1 = AUC(A vs B). S1 ≥ 0.90 on either read → shortcut, K5fno is not built** (C38's claim gate). Below on both → K5fno may be built with B drawn stratified to the positives' scene × weather shares |
| S0-5 Printed | S2 = S1 with B matched to A's scene × weather shares; S3 = scene × weather one-hot alone (no pixels); S4 = P vs A (K5f's contrast) and P vs B (K5fno's). Never gating |
| S0-6 Wording | S1 < 0.90 says "no source shortcut at this gate on a linear probe", never "the pools are the same": a non-linear model can still find one |

**Expectation, written before the number:** S1 between 0.70 and 0.90 on plain CLIP (same camera type and city mix, different scenes); S3 ≈ 0.60–0.65; S4 P vs B above P vs A. A wrong expectation changes nothing.

Notebook `colab/final/nexar_source_probe.ipynb` (CPU only); outputs `Thesis-Final/outputs/kim_source_probe/REPORTS/nexar_source_probe_{clip,a2}.md`.

### 38.1 S0 read-out (2026-11-21, user's Colab run; raw `outputs/v_final/kim_source_probe/REPORTS/` synced)

Pools as planned: A 5,622 windows / 451 videos, B 6,879 / 449, P 1,689 / 451.

| probe | plain CLIP `s1[::8]` (512 d) | A2 V2-S bake (896 d) | role |
|---|---|---|---|
| **S1 A vs B** | **0.6724** | **0.6935** | decides (gate 0.90) |
| S2 A vs B, B matched to A's scene × weather (5,589 / 1,331) | 0.6412 | 0.6768 | printed |
| S3 scene × weather one-hot alone | 0.6396 | 0.6396 | printed |
| S4 P vs A (K5f's contrast) | 0.6223 | 0.7082 | printed |
| S4 P vs B (K5fno's contrast) | 0.7210 | 0.7673 | printed |

**S0-4: PASS on both reads** (S1 < 0.90). Licensed: "no source shortcut at this gate on a linear probe". Not licensed: "the two normal pools are the same" (S0-6).

**Against the expectations.** S1 0.70–0.90 on plain CLIP: **wrong, below** (0.672). S3 ≈ 0.60–0.65: met (0.640). S4 P vs B above P vs A: met (+0.099 plain CLIP, +0.059 bake).

**Printed, decides nothing.** Metadata alone (S3 0.640) explains most of S1 (0.672 / 0.694), and matching scene × weather moves S1 by only −0.03 (plain CLIP) / −0.02 (bake). S4: on the model's input a linear probe separates K5fno's positive-vs-normal pair 0.059 more easily than K5f's, a margin that is not the event; K5fno's in-domain numbers must be read with this in mind.

**Pool size under S0-4's stratification (computed from the report's strata, before any build).** Matching B to the positives' scene × weather shares leaves **980** windows (limited by Highway|Rain: 17 B windows); to A's shares 1,338; scene only ≈ 2,490; K5f's normal pool is 5,622. The written rule would make K5fno's normal pool 5.7× smaller than K5f's, a second change beside the normals' source. **Sampling choice put to the user before any build (§38.2).**

### 38.2 Amendment (2026-11-21, user, after S0 and before any K5fno number): pool by per-video cap, not by stratification; the K5fno rule

S0-4's stratified draw would leave 980 normal windows (§38.1), so pool size would change beside the source. **The user chose a per-video cap instead:** every negative `nexar_train` video keeps at most `c` evenly spaced windows (`cap_windows`), `c` = the cap whose total is closest to the number of in-video normal windows it replaces (5,622 on the real corpus; ties → the smaller cap). `c` is computed by the build from window counts, never from a score. The scene × weather shift is kept and printed (S3 0.640, S2 moved S1 by ≤ 0.03); this is not advisor-reviewed.

| # | Choice |
|---|---|
| N1 Row | **K5fno** = K5f's exact command (`kim.enabled`, `kim.prototypes`, `kim.edge_mask`, DVS off, A2 V2-S bake, s2024, 33 epochs, 1,749 steps) with `--data-dir Nexar_window_negonly`. **Config diff to K5f = 0** (the corpus is a CLI path). Output `kim_negonly/K5fno/` |
| N2 Corpus | `nexar_build corpus --constructions window_negonly`: `window_neg`'s windows, minus **every** normal window of a positive video, negative videos capped at `c`. Positive windows and labels identical to `Nexar_window` (asserted); val, `defs.json` byte-identical; every kept normal window from a negative train video; step plan = K5f's |
| N3 Guards | J10 (1,749); config diff to K5f = {}; last-epoch `mul_mil` > 0.01 (§31 fix live); step-1 `mil` printed beside K5f's (the normal items differ, so they need not match) |
| N4 Reads | K5f's sets and protocols: `nexar_val` (crops endpoint, whole macro / micro / clip AUC), DoTA-CAP-dev (protocol B), DADA whole, TAD (s8, raw), DAD (s7, raw); `zs_arm_table` control **K5f**, macro and micro; TAD also by category (`01_Accident` 23 videos beside the rest), micro, clip AUC |
| N5 Question | **Primary: TAD micro, K5fno − K5f, paired cluster bootstrap.** "above" = in-video normals were part of K5's TAD micro cost; "not separated" / "below" = the normals' source is not the cause on this host. Macro on every set printed with its sentence |
| N6 Decision | **none for the model:** K5f stays the reference whatever the sign. n = 1, printed |
| N7 Wording | never "negative-video normals fix TAD", never a cause from one contrast without the S4 caveat (K5fno's pair is 0.06 easier on the model's input for a reason that is not the event) |

**Expectation, written before the number:** macro below K5f on Nexar val crops and DADA (hard same-scene negatives gone; K5neg already paid −0.038 / −0.021 for mixing them in); TAD micro not separated or below (every arm that added negative-video normals lowered it). A wrong expectation changes nothing.

Runbook `colab/final/k5fno_negonly.ipynb`; outputs `Thesis-Final/outputs/kim_negonly/`.

### 38.3 K5fno read-out (2026-11-22, user's Colab run; raw `outputs/v_final/kim_negonly/` + `REPORTS/{k5fno_report,zs_arm_table_k5fno_s2024,tad_category_k5fno_s2024}.md` synced; s2024, n = 1, printed, decides nothing) — N5: **below**

**Build (N2).** `c` = 14 → 5,781 normal windows from 449 negative videos (in-video normals dropped: 5,622; K5f's pool size within +2.8 %); 1,689 abnormal windows, val 1,788 windows (434 two-class) unchanged; step plan 33 epochs × 53 = 1,749, = K5f's.

**Guards (N3).** J10: `verified.json` step 1,749 = last `metrics.jsonl` row. Config diff to K5f = `loss.normal_caption: lagovad` only, a key added after K5f trained, at its default and read only by `caption_lagovad` (same case as K5dfi, §34.1) → effective diff {}. Last-epoch mean `mul_mil` **0.0112** (> 0.01, but K5f 0.0963: only just over the floor). Step-1 `mil` 0.7153 vs K5f 0.6878 (normal items differ, need not match).

**Training became easy.** Last-epoch means K5fno vs K5f: `mil` **0.025 vs 0.184**, `kim_mil` 0.023 vs 0.146, total 0.074 vs 0.546 (≈ 7× lower). With no same-video normal, the positive and the negative bag differ by more than the event (S4: P vs B is 0.06–0.10 easier than P vs A on the model's input), and the objective is satisfied by that difference. This is the reading the S4 caveat (N7) asked to keep in view; it is a reading of the logs, not a second test.

**Results (macro / micro, Δ = K5fno − K5f, paired).**

| set | K5f | K5fno | Δ macro | Δ micro |
|---|---|---|---|---|
| Nexar val (crops macro; whole macro / micro / AP / clip AUC) | 0.8942 (0.9210 / 0.8961 / 0.4411 / 0.7659) | 0.8000 (0.8348 / 0.7853 / 0.1771 / 0.7698) | **−0.0942 [−0.1234, −0.0663] below** | — |
| DoTA-CAP-dev | 0.8188 / 0.7353 | 0.7476 / 0.6398 | **−0.0712 [−0.0876, −0.0546] below** | −0.0954 [−0.1121, −0.0792] below |
| DADA-2000 | 0.7744 / 0.7551 | 0.7013 / 0.6561 | **−0.0731 [−0.0928, −0.0534] below** | −0.0990 [−0.1176, −0.0795] below |
| TAD | 0.6981 / 0.6711 | 0.6951 / 0.5008 | −0.0030 [−0.0555, +0.0510] not separated | **−0.1704 [−0.3013, −0.0430] below (N5)** |
| DAD | 0.8070 / 0.7072 | 0.7906 / 0.6556 | −0.0164 [−0.0617, +0.0301] not separated | −0.0515 [−0.0853, −0.0172] below |

Nexar val FA share (negatives, peak > 0.5) 0.549 → 0.478; clip AUC flat (0.766 → 0.770) while every within-video number falls: K5fno still tells accident videos from negative videos, it localizes worse inside them.

**N5 (primary): TAD micro K5fno − K5f = −0.1704 [−0.3013, −0.0430] → "below": the in-video normals' source is not the cause of K5's TAD micro cost on this host.** Removing them makes TAD micro worse, not better. TAD macro does not move (−0.003), so the micro loss is clip-level: TAD clip AUC (peak of abnormal vs Normal videos) **0.625 → 0.429**, below chance; AP 0.261 → 0.085. By category (N4, n per category 3–23): clip AUC falls in all seven categories (`01_Accident` 0.590 → 0.494); macro rises on `05_else` (+0.069, 14 videos) and `06_PedestrianOnRoad` (+0.049, 10) and falls on the other five — printed, not read.

**Expectation check.** Macro below K5f on Nexar val crops and DADA: **held** (−0.094 / −0.073, larger than K5neg's −0.038 / −0.021). TAD micro not separated or below: **held** (below). DoTA-CAP-dev below (−0.071) was not written down.

**N6.** No decision: **K5f stays the reference.** Licensed sentence: "On the Nexar-trained KIM host, replacing in-video normals with negative-video normals (pool size matched, n = 1) lowered macro on Nexar val, DoTA-CAP-dev and DADA and lowered TAD micro; it did not raise TAD micro." Not licensed (N7): "negative-video normals fix / break TAD", or any single cause without the S4 caveat (the K5fno pair is easier on the model's input for a reason that is not the event). The read is one more measurement that same-video normals are where this host's frame-level signal comes from (cf. K5neg §17.1, K5nc2 §27.1).

## 39. K5dfiACC and L0dfiACC: K5dfi's and L0dfi's recipes trained on ACCIDENT (real CCTV), read on TAD (dated note 2026-11-22, before any number; user)

**Why.** §38.3 closed the in-video-normal hypothesis: TAD macro sits at ≈ 0.70 on every dashcam-trained row (K5f 0.698, K5fno 0.695, K5T2 not separated from K5). The remaining hypothesis is **H-dom**: TAD is CCTV and every training corpus here is ego dashcam. ACCIDENT (CVPRW 2026, Kaggle `picekl/accident`, CC BY-NC-SA 4.0) ships **2,027 real CCTV accident clips** with one impact time each and **no normal video**. It can test H-dom on TAD **macro** (localization inside a video). It cannot test TAD micro: micro there measures clip classification (oracle 0.9226, §38.3), and no corpus without normal videos teaches it. Plan: pending (dq). User's choices 2026-11-22: base = **K5dfi**; post-impact span **2 s**; **L0 arm yes**; split and dedup threshold "Claude's call" (S, D below).

| # | Choice |
|---|---|
| A1 Rows | **K5dfiACC** = K5dfi's §34 command (KIM + prototypes + edge / seam mask, DVS θ 0.7 / θ_ego 0.85, `loss.dvs_anchor_mode=ignore`, `is_egocentric=false`, V2-S motion, s2024) with `data.dataset=ACCIDENT`, `--data-dir ACCIDENT_window`, `--clip-dir` = the ACCIDENT A2 bake, `--knn-cache` = ACCIDENT's, and `train.num_epochs` from the step plan (A6). **L0dfiACC** = L0dfi's §36 command (KIM off, motion off, plain CLIP s8) with the same four changes. Config diff to K5dfi / L0dfi = {`data.dataset`, `train.num_epochs`}; outputs `kim_accident/<row>/` |
| A2 Ingest | Kaggle `real_videos/*.mp4` (4.92 GB; synthetic CARLA clips **not used**). Native fps varies 3.9–49.7 (median 15), so every clip is **resampled to 30 fps CFR by timestamp** (frame `k` = the last decoded frame with `t ≤ k/30`, `t` from the first decoded frame) in memory, no re-encode; then Nexar's transforms unchanged: CLIP `no_center_crop` every CFR frame, V2-S causal 16 × step 3 (1.5 s). Caches `ACCIDENT_s1_ncc`, `video/<V2-S>/ACCIDENT_s1_squash` |
| A3 Labels | By time (`accident_time`; 494 clips have `accident_frame` ≠ time × fps, so the frame column is not used). A CFR frame `f` is abnormal iff `t_impact ≤ f/30 ≤ t_impact + 2 s`; **frames after `t_impact + 2 s` are cut** (the aftermath is unlabeled: median 19.7 s, no end time). Label-0 = `Normal` (§31). Windows = Nexar's: 20 rows at s8, hop 8, no per-clip cap, `min_positive` 1; a clip shorter than 20 rows up to `t_impact + 2 s` gives no window (metadata simulation: 1,501 of 2,027 clips, 1,482 positive / 1,930 normal windows before the split and dedup) |
| S Split | **Own split, not ACCIDENT's.** Its IID / geo splits train on only 507 / 454 clips (25 %) because they are benchmark splits, and this project never reports on ACCIDENT's own test. Grouped by upload (`id` minus its `_NN` suffix and, on 257 ids, a segment index `_<k>`: `3Xd6PxEvbNk_3_00` → `3Xd6PxEvbNk`; 1,789 sources, 92 with ≥ 2 clips, 645 same-source pairs), stratified by `type` (5 classes), **90 / 10 train / val**, seed 2026, frozen in `core/splits/accident/` with the metadata sha1. Val is in-domain, read and printed, never decides |
| D Dedup vs TAD | Against the **100 TAD test videos read here** (Lv 2021; ACCIDENT's "14 TAD overlaps" are Xu 2025's set, pending (dq)). Score(a, b) = max cosine over CLIP frames (ACCIDENT `s1[::8][::4]`, TAD `s8[::4]`). **Threshold τ = the 10th percentile of the same-source control**: per upload with ≥ 2 clips (92), its best clip pair (two clips of one upload = the same-camera positive control; the best pair only, because a large upload is often a compilation of several cameras and its cross-camera pairs would drag the control down); printed beside it: the cross-source nearest-neighbour distribution (300 clips) and every clip's TAD top-1. Every clip with TAD top-1 ≥ τ is flagged and **its whole source is excluded** (train and val). A contact sheet of the flagged pairs is drawn; the user may un-exclude a source only before training, recorded here. **If > 10 % of clips would be excluded, stop** (τ does not separate same camera from same kind of scene) |
| B Bake | ACCIDENT rows baked with **Nexar's A2 statistics** (`build_v2_inputs apply --stats-dir A2_S/Nexar_s8 --stride 8`, CRN off). One factor changes vs K5dfi (the training data), and every zero-shot bake K5dfi was read on is reused as is. ACCIDENT's own statistics would also move the input scale of every read |
| A6 Steps | `epoch_plan` with Nexar's budget (1,740): `steps/epoch = ceil(2 × positives / 64)`; expected ≈ 42–47 × 37–41 ≈ 1,740 (K5dfi 1,749). Printed |
| R Reads | K5dfiACC: TAD (A2 `TAD_s8_nexar`, raw), DoTA-CAP-dev (protocol B), DADA whole, DAD (K5dfi's bakes), ACCIDENT val (printed). L0dfiACC and **L0dfi** (no TAD read before): TAD on plain `TAD_ncc`; L0dfiACC also DoTA-CAP-dev / DADA on plain CLIP. Tables: `zs_arm_table` control **K5dfi** (rows K5dfiACC, K5f; 4 sets), control **L0dfi** (row L0dfiACC; TAD, DoTA-CAP-dev, DADA); `tad_category_read` for K5dfi, K5dfiACC, L0dfi, L0dfiACC |
| Q Question | **Primary P1: TAD macro, K5dfiACC − K5dfi** (paired, the read-outs' bootstrap). "above" = H-dom supported on this host: training on CCTV accidents instead of dashcam accidents raises TAD localization. "not separated" / "below" = H-dom not supported at this data size. **Secondary P2: TAD macro, L0dfiACC − L0dfi.** Everything else printed |
| G Guards | J10 (ckpt step = metrics' last = plan steps); config diff as A1; last-epoch `mul_mil` > 0.01 (§31); the ACCIDENT bake's manifest = Nexar's `train_ids_sha1` / V2-S / CRN off; no excluded source and no val source in the train windows; no window row past `t_impact + 2 s` |
| N Decision | **None for the model.** K5f stays the reference and the main model. n = 1 |
| W Wording | Never "CCTV data fixes TAD" and never a cause from P1 alone: ACCIDENT also changes the label rule (impact + 2 s, no alert time), the amount of data (normals ≈ 1/3 of Nexar's), the fps (upsampled from a median of 15), and the camera. Never use a TAD micro number as evidence for or against H-dom (A1 note). Never compare with ACCIDENT's leaderboard (other split, other task) |

**Expectation, written before the number:** P1 above by +0.02…+0.06, possibly not separated (TAD macro has 60 videos and K5fno's interval was ±0.08); the dashcam sets (DoTA-CAP-dev, DADA, DAD) below K5dfi; TAD micro not above (no normal videos); dedup excludes < 2 % of clips. A wrong expectation changes nothing.

Runbooks: `colab/final/accident_ingest.ipynb` (download, resample, extract, dedup; stops for the contact sheet), then `colab/final/k5dfi_acc.ipynb` (corpus, bake, KNN, the two rows, reads, tables). Outputs `Thesis-Final/outputs/kim_accident/`.

**Built (2026-11-22, not run).** `core/tools/accident_build.py` (`split`, `corpus`; `source_of` strips `_NN` and a segment index), `core/tools/accident_extract.py` (`cfr_index`, PyAV decode with timestamps, Nexar's `clip_rows` / `motion_rows`; `nexar_extract.check_clip_manifest` takes an optional manifest, default unchanged), `core/tools/accident_dedup.py`, `core/tests/test_accident.py` (16 tests), constants `ACCIDENT_*` / `KIM_ACCIDENT_*`, `definitions.DATASET_NAME_TO_ABBR["ACCIDENT"] = "dada"`. **Split frozen** `core/splits/accident/` (metadata sha1 `771cd751…`): train 1,824 clips / 1,610 uploads, val 203 / 179. **Corpus smoke on the real metadata** (rows = duration × 30, no dedup): 3,068 train windows (**1,320 positive**, 1,748 normal) from 1,337 clips, val 344 windows (162 two-class); 485 train clips too short for one window (abnormal-source retention 73 %, the C32 warning fires: kept, it is A3's 20-row geometry); plan **42 steps × 41 epochs = 1,722**. The real counts come from the build report after extraction and dedup. **Extract to the cut (2026-11-22, user: GPU budget; before any extraction, changes no number):** `accident_extract` keeps frames only to `t_impact + 2 s` (decode stops one frame past it; cached rows == full decode + `cut_frames`), runs V2-S only at the s8 rows (`nexar_extract.motion_rows(stride=8)` == stride-1 rows `[::8]`; cache `ACCIDENT_s8_squash`, manifest stride 8; the bake then reads CLIP s8 + V2-S s8 at `--stride 1`, the same rows as Nexar's s1 bake at `--stride 8`), and skips clips whose cut is shorter than one window (`accident_build.windowable`, metadata duration, margin 2 rows; the corpus leaves the same clips out, `skipped_short_clips` in the build report). Measured on the metadata: **1,666 of 2,027 clips extracted; ≈ 0.46 M CLIP frames and ≈ 0.05 M V2-S clips instead of 1.33 M each** (≈ 23× less VideoMAE). No subset of clips is drawn: every clip that can give a window trains. Tests `test_decoding_to_the_cut_gives_the_full_decode_cut`, `test_windowable_skips_only_clips_too_short_for_a_window`, `test_extract_keeps_the_cut_runs_motion_at_s8_and_skips_short_clips`, `test_motion_rows_at_a_stride_are_the_stride1_rows_subsampled`. **TAD pixels (2026-11-22, user deleted TAD frames from Drive):** no score needs them (dedup and every TAD read use `TAD_ncc` + the label json); the contact sheet downloads only the images it draws from Kaggle `nikanvasei/traffic-anomaly-dataset-tad` (the Lv 2021 / WSAL release, same `TAD/frames/{abnormal,normal}/<id>.mp4/<k>.jpg` layout) via `kagglehub.dataset_download(handle, path=<one jpg>)`.

**Dedup read-out 1 (2026-11-22, user's Colab run; before any training): STOP.** τ = p10 of the same-source control (60 sources) = **0.7911**; 848 clips flagged → 753 sources → **869 / 1,666 clips (52.2 %)** excluded, over the 10 % stop (expectation < 2 %: wrong). Cause: τ sits **below the cross-source noise floor** (ACCIDENT-vs-ACCIDENT nearest neighbour p10 0.854, p50 0.906) and at the median of the TAD top-1 (p50 0.792, p99 0.896, i.e. below the cross-source p50): the control cannot separate one camera from one kind of scene. TAD top-1 concentrates on hubs (`Normal_187` 99 clips, `Normal_128` 80, `Normal_178` 65, `01_Accident_025` 61); 457 of the 848 flags point at TAD Normal. The first 6 pairs of the old sheet (alphabetical, not by score) are the same kind of scene, not the same camera. Nothing is excluded, nothing trained.

**Amendment §39 D′ (2026-11-22, user: "hướng 2"; before any training):** the threshold is chosen **by eye on a score-sorted contact sheet**, not by the percentile. Sheet = every clip with TAD top-1 ≥ **0.87** (88 clips / 86 sources), highest score first, 8 pairs per PNG. The user (or Claude on the user's word) names the ranks that are the same physical camera; τ′ is set at the score where they stop and recorded here **before** `k5dfi_acc.ipynb` runs; the 10 % stop is re-applied at τ′. Built: `accident_dedup.high_score_pairs` (+ `pair_record`, `sheet_pairs` unchanged: output equal on the real json), constants `ACCIDENT_DEDUP_SHEET_*`, test in `test_accident.py`, notebook `colab/final/accident_dedup_sheet.ipynb` (CPU, reads `accident_dedup.json`, writes `REPORTS/high/`). Not run.

**Resolution of D′ (2026-11-22, user: "ok làm đi"; before any training): no clip is excluded.** The user's Colab run of `accident_dedup_sheet.ipynb` gave 88 clips / 86 sources with TAD top-1 ≥ 0.87 (0.923 … 0.870, `outputs/v_final/kim_accident/REPORTS/high/`). Claude read all 88 pairs by eye (11 pages, one best frame per pair, 900 px): **none is the same physical camera**. Groups, all same kind of scene and different footage: Chinese DVR compilation `3Xd6PxEvbNk` vs `01_Accident_086/072` / `Normal_105` (ranks 1, 3–6, 18); night Traffic-Cam-Watch / MGM-CAM intersections vs the Nest camera of `01_Accident_025` (10, 11, 19, 33, 51, 87); wide intersections vs `Normal_187` (21, 31, 47, 48, 66: the pair Claude was least sure about; `Normal_187` has a signal head at lower left and cracked asphalt, none of the clips does); MnDOT / highway vs `Normal_178` / `Normal_128` (24, 65, 85); every dashcam TAD video vs overhead CCTV. Shared **uploader watermarks** (CCTUBE, Live Leak, Viral06, Crash Daily, YouDM) appear on both sides: the same compilation channels, different footage; a same incident re-cut at another angle is not what this test measures. **τ′ = above the highest score (0.923) → 0 / 1,666 clips excluded (< 10 %).** Limits, stated: one best frame per pair, thumbnails, the criterion is Claude's reading; clips below 0.87 were not inspected. Rough bound: the control puts 0.87 near its p38, so about 60 % of real same-camera pairs would score ≥ 0.87; had there been 3 duplicate clips, the chance that the sheet shows none is ≈ 5 % (assumes the ACCIDENT-vs-ACCIDENT control transfers to ACCIDENT-vs-TAD). **Wording for every P1 / P2 sentence:** "dedup vs TAD read by eye above 0.87, nothing excluded; below 0.87 not inspected". `k5dfi_acc.ipynb`: `DEDUP_NO_EXCLUSION = True` (asserts status STOP, `KEEP_SOURCES` empty, `EXCLUDED = []`) and the report prints the resolution; `False` restores the original rule. P1 / P2, the expectation and every other choice unchanged. Not run.

**Read-out of §39 (2026-11-23, user's Colab run; s2024, n = 1; printed, decides nothing).** Raw: `outputs/v_final/kim_accident/` (`REPORTS/k5dfiacc_report.md`, `zs_arm_table_{k5dfiacc,l0dfiacc}_s2024.md`, `tad_category_k5dfiacc_s2024.md`).

*Guards, all passed.* J10: ckpt step 1,722 = last `metrics.jsonl` step = plan (42 × 41); config diff to K5dfi = {`data.dataset`, `train.num_epochs` 41 vs 33}; `mul_mil` of the last two batches 0.26 and 0.46 (> 0.01); `excluded_ids_sha1` = sha1 of the empty list (D′, nothing excluded). Corpus: 1,322 positive / 3,038 train windows from 1,339 clips (1,494 clips windowable of 1,824; 361 skipped short), val 375 windows (161 two-class).

*P1 / P2 (TAD macro).* **P1 K5dfiACC − K5dfi −0.0211 [−0.0921, +0.0483], not separated** (expectation +0.02…+0.06: wrong; H-dom not supported on this host at this data size, the sentence §39 Q fixed). **P2 L0dfiACC − L0dfi +0.1001 [+0.0299, +0.1667], above**, but L0dfi starts at 0.5902 and L0dfiACC (0.6904) reaches K5dfi (0.7198) without passing it (L0dfiACC − K5dfi −0.0294, not separated).

| TAD (60 two-class, raw) | macro [95 %] | micro | AP |
|---|---|---|---|
| L0dfi | 0.5902 [0.5312, 0.6488] | 0.6733 | 0.1956 |
| L0dfiACC | 0.6904 [0.6294, 0.7463] | 0.7389 | 0.2684 |
| K5dfi | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 |
| K5dfiACC | 0.6987 [0.6335, 0.7609] | 0.6170 | 0.2397 |
| LaGoVAD (public ckpt = Gate T0) | 0.7579 [0.6955, 0.8147] | 0.7912 | 0.2571 |

Clip oracle micro 0.9226, so TAD micro is clip classification. Δ vs K5dfi: LaGoVAD macro +0.0381 (not separated), micro +0.1793 [+0.0553, +0.2932] (above); L0dfiACC micro +0.1270 [+0.0206, +0.2237] (above); K5dfiACC micro +0.0051 (not separated). L0dfi has its first TAD read here (plain `TAD_ncc`).

*Other sets (Δ ACC − non-ACC, macro / micro).* K5: DoTA-CAP-dev **−0.0887 / −0.0798 below**, DADA **−0.1084 / −0.1171 below**, DAD **+0.0772 above / +0.0299 not separated** (expectation "dashcam sets below": right on DoTA / DADA, **wrong on DAD**; K5f − K5dfi on DAD is +0.0576, the DVS cost of §34). L0: DoTA −0.0131 n.s. / −0.0253 below, DADA −0.0024 / −0.0139 n.s., TAD +0.1001 above / +0.0656 n.s. (L0 has no DAD read). ACCIDENT val (in-domain, printed): K5dfiACC 0.7667 / 0.6934, L0dfiACC 0.5996 / 0.5700. TAD by category (3–23 videos, descriptive): K5dfiACC − K5dfi macro `01_Accident` −0.10 (23 videos; clip AUC 0.576 → 0.727), `05_else` +0.05, `06_Pedestrian` +0.06, `07_RoadSpills` +0.28 (4), `03_IllegalOccupation` −0.18 and `04_Retrograde` −0.28 (3 each).

*Effect of VideoMAE motion + KIM at fixed data and recipe (user, 2026-11-23: ACCIDENT, `ignore`, nothing else differs; no `span` arm built: it would add a second change).* K5dfiACC − L0dfiACC: DoTA-CAP-dev **+0.0744 [+0.0538, +0.0953] / +0.0675 [+0.0504, +0.0843]**, DADA **+0.0865 [+0.0649, +0.1077] / +0.0810 [+0.0645, +0.0997]**, TAD macro +0.0083 [−0.0432, +0.0591] not separated, **TAD micro −0.1220 [−0.1988, −0.0427] below** (TAD clip AUC 0.652 vs 0.686, no interval). On Nexar the same TAD micro pair is not separated (L0dfi − K5dfi +0.0614 [−0.0260, +0.1393]). Sentence: "trained on ACCIDENT, adding VideoMAE + KIM raises macro and micro on DoTA and DADA, leaves TAD macro unchanged and lowers TAD micro (−0.12) (n = 1)". It is the gross effect of motion and KIM together (R6 of §36), the inputs differ (K5 reads the A2 bake, L0 plain CLIP), and TAD micro is clip classification.

*Wording (W, binding).* Never "CCTV data improves TAD": it did for the L0 host (macro +0.10, n = 1), not for K5. ACCIDENT also changes the label rule (impact + 2 s), the normals (≈ 1/3 of Nexar's) and the fps (median native 15, ≈ 70 % of clips under 20 fps, frames repeated to 30 fps CFR); a repeated-frame distortion of the motion input is a hypothesis (it would fit K5 falling on DoTA / DADA while L0 does not), never a finding. Dedup: "read by eye above 0.87, nothing excluded; below 0.87 not inspected". **Decision: none; K5f stays the reference and the main model.**

## 40. EDA of the Nexar training corpus K5f reads, `Nexar_window` (dated note 2026-11-22, user; dataset properties only)

**Why.** Every Nexar corpus so far was characterized by the raw-release census (`NEXAR_SETUP.md` §1.1) and by `build_report.json` counts; `core.tools.eda` had never been run on one, and it reads **test-side** frame labels, which a windowed corpus's train windows do not have (one weak label per window). The geometry the MIL loss trains on (abnormal rows per positive window, where they sit, how far the normal windows lie from the span) was therefore unmeasured.

| # | Choice |
|---|---|
| E1 Corpus | `Thesis-V2/data/Nexar_window` = K5f's corpus (7,311 train windows = 1,689 positive + 5,622 normal, 451 positive videos; asserted). K5dfi reads the same corpus (DVS splices at train time), so the report serves both. User wrote "K5fi"; read as K5f / K5dfi. |
| E2 Labels | Rebuilt from `meta.json` (`source`, `start`, `end`, `normalized_span`, `total_frames`) with `sampled_frame_labels` at stride 8 and **gated**: every window's rebuilt positive count must equal `positive_frames`, and `label == (positive_frames >= WINDOW_MIN_POSITIVE)`; otherwise the tool raises. |
| E3 Read | counts · abnormal rows per positive window (percentiles, share ≤ k rows) · share of positive windows abnormal at each of the 20 row indices, train beside val · rows between each normal window and its own span · windows per source · scene / weather per class · AUC of source length, window centre and a hump `-\|c - 0.5\|` as window-label predictors (raw and inside one source) · the val windows' constant-score oracle. Optional: `core.tools.eda report --sections corpus,labels,protocol --no-probe` on the same files. |
| E4 Status | Printed, never gating. No arm, no decision, no change to K5f. The hump AUC is a position ruler, not a model; it is printed beside any macro per the cubic-probe rule. |

**Built (2026-11-22, not run on real data).** `core/tools/nexar_eda.py` (`load_corpus`, `window_rows`, `build_report`, `render_markdown`), `core/tests/test_nexar_eda.py` (4 passed; ruff / mypy / pyright / bandit clean), constants `NEXAR_EDA_*`, notebook `colab/final/nexar_eda.ipynb` (CPU, parses). Smoke-tested on a synthetic corpus built by `nexar_build.build_window` with both `nexar_eda` and `core.tools.eda report` (the generic run works on a Nexar-built corpus). Not committed.

**Read-out of §40 (2026-11-23, user's Colab run; printed, decides nothing).** Raw: `outputs/v_final/nexar_eda/REPORTS/` (`nexar_eda_report.{md,json}`, `eda/eda_report.{md,json}`; commits `e172d5f`, `a217f23`). Gate passed: rebuilt positive rows equal `meta.json` on all 7,311 train windows.

| Item | Measured |
|---|---|
| Counts | train 7,311 windows = 1,689 positive + 5,622 normal from 451 positive videos (every video has both classes; 3.7 positive / 12.5 normal windows per video); val 1,788 = 434 two-class + 1,354 all-normal + 0 all-abnormal |
| Positive windows | abnormal rows / window: mean 7.45, p50 8 of 20; at most 1 row 6.6 %, at most 3 rows 19.3 %, whole window abnormal 0.1 %. Share abnormal per row index is flat 0.36-0.38 (train and val): the position inside the window carries no label |
| Normal windows | 2,868 before the span, 2,754 after; rows to the span p50 25, p25 12, p75 39; within 2 rows of the span 6.3 % (few hard negatives) |
| No-pixel AUC over train windows | source length 0.461; window centre raw 0.510; **hump `-\|c - 0.5\|` 0.973** (0.979 inside one source) |
| Scene / weather | positive rate Urban 0.222, Highway 0.255, Sub-urban 0.236; Clear 0.230, Cloudy 0.234, Rain 0.228: the strata do not separate the classes |
| Val oracle | constant score per window: micro AUC **0.9156**, AP 0.3662, macro 0.5 by construction; 99.97 % of positive / negative frame pairs span two windows. 27,080 of 35,760 val frames (75.7 %) sit in all-normal windows; of the 8,680 mixed-window frames 3,179 are positive and 5,501 normal, and those 5,501 tie with the positives, which is what holds the oracle at 0.9156 instead of 1.0 |
| Length leak | window length is 20 rows everywhere, length AUC 0.5000 |
| MIL k | `mil_topk_pct = 5` (K5f's trained value): **k = 4 for every window**, 0 % at k = 1; `mul_mil_topk_pct` stays 16, so the multi-class MIL has k = 1 |

**Correction inside this read-out.** The first run of `core.tools.eda report` (`e172d5f`) was read at the profiler's default `mil_topk_pct = 16` and printed a HIGH verdict "MIL top-k degenerates to a plain max (100 % of clips k = 1)". K5f / K5dfi / K5dfiACC trained with `mil_topk_pct = 5` (their `config.yaml`), so that verdict did not apply to them. The notebook's cell 10 now passes `--mil-topk-pct constants.DADA_ORIGIN_MIL_TOPK_PCT` (= 5, `e09635d`), the re-run (`a217f23`) shows k = 4 and no HIGH verdict. Only the CRITICAL verdict (micro is clip classification, C12) remains. Pending (du).

**Second correction (2026-11-23, Claude, from the raw `eda_report.md`).** The same run read the score-head coverage (§1.2 of the generic report) at the profiler's default `--score-head-kernel 9`; K5f / K5dfi trained with `model.score_head_kernel = 3` (`outputs/v_final/kim_fixed/K5f/K5/s2024/config.yaml`). At kernel 3 one output timestep sees 3 / 20 = **15 %** of a window (the report printed 45 %); windows entirely inside the kernel stay **0** at both kernels, so no verdict changes. The notebook's cell 10 now also passes `--score-head-kernel constants.DADA_ORIGIN_SCORE_HEAD_KERNEL` (= 3); not re-run (the coverage is arithmetic on a fixed 20-row window). Also in the raw report, not in the table above: 36 val positive windows have exactly one abnormal row and 73 have at most two; DVS length 3,378 = 53 steps / epoch at batch 64; every two-class val window has one span. Same pending (du), second occurrence.

**Figures (2026-11-23, user; re-run read 2026-11-23: kernel 3 → 15 %, every other number unchanged; PNGs in `REPORTS/figures/`).** `nexar_eda --plots` writes six PNGs to `REPORTS/figures/` from the same gated rows (no new number, nothing gates): `abnormal_rows_per_positive_window` (train vs val, share), `row_profile` (§2's row table as lines), `normal_gap_to_span` (before / after stacked, bin = `NEXAR_EDA_NEAR_ROWS + 1` rows so the "near" line is a bin edge), `window_center_by_class` (normal vs positive centre density, the hump of §4's 0.973), `scene_weather_positive_rate` (rate per stratum, corpus rate line), `val_frame_share` (val rows by window kind with the mixed windows split into normal / abnormal rows, oracle in the title). The notebook's cell 8 passes `--plots`, copies the figures and shows them inline; the re-run also gives the generic report at kernel 3. Tests `test_nexar_eda.py` 6 passed. Read from the figures (printed): normal windows avoid window-centre 0.40–0.60 almost entirely and tile the rest of the video evenly; their distance to the span is flat from 0 to ~52 rows, then stops (the video ends), so the 6.3 % within 2 rows is what even tiling gives, not a deficit of the cut. Only ~172 before-span normal windows (bin 0–2 rows) exist for 451 videos: for most accident videos the last ~0.5 s before the alert is covered only by positive windows, whose non-top-k rows get no target. This is the supervision side of §43's pre-onset rise; not a cause.

**Licensed sentences.** "Nexar's val micro AUC is mostly clip classification: a constant score per window scores 0.9156, so only the part above it and `auc_macro` can show localization." "In the K5f corpus the positive windows' abnormal rows are not tied to a row position, and a source's window-centre hump predicts the window label at AUC 0.973 (a property of the corpus, not of a model)."
**Not licensed.** Any claim that K5f uses the position prior: the model sees 20 rows, not the window centre; the hump AUC says a position prior exists in the corpus, not that it is learned (test it with `position_shift same-frame`, never from this table). Never write "k = 1" for K5f's `L_MIL`. No arm, decision or change to K5f follows from this read-out.

## 41. K5f, K5dfi, L0f and L0dfi trained on MSAD-full (whole videos), kernel 3 and 9 (dated note 2026-11-23, before any number; user)

**Why.** §18 MS8 left one question open: does CCTV in training (here MSAD, fixed cameras, normals present) change what the KIM recipes do on fixed-camera test sets (TAD, MSAD's own test)? §39 asked it with ACCIDENT (no normal videos, TAD macro only). MSAD has both classes of video, an in-domain test set (240 videos) and TAD-like cameras. The user's choices 2026-11-23: train on **whole videos, no cutting**; recipes **K5f, K5dfi, L0f, L0dfi**; **both** head kernels (3 and 9) as rows "to see which fits"; MIL top-k **16**; **125 epochs**.

**Amendment 1 (2026-11-23, user, before any run; after the first commit `497c6e8`): the point is a clean CCTV-trained version, not a comparison with the Nexar-trained rows.** So (a) the A2 motion statistics (`m_u`, `sigma_u`, `c`) are **fitted on MSAD train** (`build_v2_inputs fit-ids`, CRN off) and the MSAD test and TAD inputs are baked with those statistics; (b) the Nexar-trained controls (K5f, K5dfi, L0dfi read zero-shot) and the "in-domain vs zero-shot" contrasts are **dropped**: they would need Nexar-scaled bakes beside MSAD-scaled ones and answer a question nobody asked; (c) R1's recipes still copy the Nexar rows' switches and the config guard still compares with their saved `config.yaml`, as a recipe check only (no Nexar checkpoint or score is read). R3 / R4 / R5 / R7 / W / Expectations below are the amended text.

**Corpus, measured from the user's Drive before any number (read-only).** Official blurred release `MSAD Videos blurred/`: `MSAD_anomaly/<Class>/<Class>_<n>.mp4` (240 videos, train and test together), `MSAD_normal_training/<scenario>/<scenario><n>.mp4` (360 = the 14 scenarios of `video_msad_all_720.xlsx`; **one is an `.avi`**, `perdestrian_street7`), `MSAD_normal_testing/MSAD_normal_testing_<n>.mp4` (120), and `VALU_MSAD_subset_2026` (the modified copy: **never read**). Train = 120 abnormal (the 240 minus the 120 in `frame_labels_test.json`) + 360 normal = `V2_TADG_MSAD_TRAIN`. Train normal ids are the file stems (`frontdoor1`, `restaurant&bar1`), not `MSAD_normal_testing_<n>`; the census therefore matches by stem and gates each label against the folder. From the table's frame counts the normal-train videos cluster near 600 frames and many abnormal videos are shorter, so a clip-length shortcut may sit in the supervision (lesson C28); the census prints the exact ruler (R7).

**Why the geometry is not K5f's.** `loss.mil_topk_pct` is a divisor: `k = max(1, n // pct)` (`core/losses/mil.py`). K5f's 5 gives k = 4 on a 20-row Nexar window (20 %) but **k = 17 on MSAD's median 86-row video** (20 % of the whole video, most of it normal frames inside a positive bag). LaGoVAD / v1's 16 gives k = 5 there (≈ 6 %). Step budget: a balanced epoch draws 2 × positives; with 120 positives that is `ceil(240 / 64)` = **4 steps/epoch**, so v1's 125 epochs = **500 steps** (RESULTS_MSAD.md) while K5f's 1,749 steps were Nexar's (1,689 positives). The geometry is fixed **a priori from the corpus and v1's record, not fitted on any MSAD score**.

| # | Choice |
|---|---|
| R1 Rows | Recipes = the Nexar rows' own switches, nothing else. **K5f**: KIM + prototypes + edge mask, `dvs.theta` / `theta_ego` 1.0 (DVS off), `dvs_anchor_mode=span`, V2-S motion. **K5dfi**: + `kim.seam_mask`, `dvs.theta` 0.7 / `theta_ego` 0.85, `dvs_anchor_mode=ignore`. **L0f**: K5f with `kim.enabled=false`, `v2.motion=none`, `model.motion_dim=0` (plain CLIP). **L0dfi**: K5dfi with the same three switches off. Common: `train.stage=2`, `train.seed=2024`, `train.amp=true`, `kip.enabled=false`, `data.is_egocentric=false`, `data.dataset=MSAD-full`. A row is `<arm>M<kernel>` (`K5fM3` … `L0dfiM9`): **8 rows**. `core.tools.msad_train.train_sets` is the single source of these lists; `test_msad_train.py` checks each against the Nexar rows' saved `config.yaml` (K5f, K5dfi, L0dfi) |
| R2 Geometry | `loss.mil_topk_pct=16`; `model.score_head_kernel` ∈ {3, 9}, **both are rows**; `train.num_epochs=125` (= 500 steps). Kernel 3 is the inherited host (§3.1 / §9: kernel 9 gained on DoTA-CAP-dev but not on DADA, so the both-endpoints rule failed and kernel 3 stayed); kernel 9 is LaGoVAD's and v1's MSAD value. `mul_mil_topk_pct` (16) and `sup_mil_topk_pct` (4) stay at their defaults, as in the Nexar rows |
| R3 Inputs | CLIP = `cache/clip/MSAD_ncc` (the cache `full_gate_a` and §18 read). Motion (K5 rows): VideoMAE V2-S at s8, frame step 3, assumed 30 fps, decoded from the blurred release's mp4 / avi (as §18 MS3), manifest equal to Nexar's V2-S manifest on every key but `stride`; baked with the A2 statistics **fitted on the 480 MSAD train videos, CRN off** (`fit-ids`; manifest `train_ids_sha1` = the sha1 of `labels_train.json`'s ids) into `cache/v2/A2_S/MSAD-full_train_s8_msad`; the same statistics bake the 240 test videos (`MSAD-full_test_s8_msad`) and TAD (`TAD_s8_msad`). L0 rows read plain `MSAD_ncc`. KNN (DVS rows only): `cache/knn/MSAD_ncc/knn_cache.npz`, v1's if present, else rebuilt with `core.data.knn_cache --dataset MSAD-full`, no motion key (fixed camera) |
| R4 Reads | **MSAD test** (240, MSAD-full and MSAD-traffic as §18 MS5; raw scores) for the 8 rows (K5 rows: the MSAD-statistics test bake; L0 rows: plain `MSAD_ncc`). **TAD** (100 videos, macro over 60; K5 rows: `TAD_s8_msad`, L0: `TAD_ncc`, raw) for the 8 rows. DoTA-CAP-dev, DADA and DAD are **not** read here, and no Nexar-trained row is read |
| R5 Contrasts | Paired per video, bootstrap 10,000, seed 2026, cluster = video, per kernel: K5f − L0f; K5dfi − L0dfi; K5dfi − K5f; L0dfi − L0f; and, printed only, kernel 9 − 3 per arm. Macro on MSAD-full, MSAD-traffic and TAD; micro, AP, clip AUC and the no-pixel rulers beside |
| R6 Decision | **None.** No kernel is picked, no row is adopted, K5f stays the reference and the main model. The report never prints a best-of. A headline kernel, if the user wants one, needs a dated note whose rule does not use the MSAD test set (MSAD has no validation split, and 120 positives are too few to carve one) |
| R7 Guards (stop on any) | **M2 (train census):** each of the 480 `labels_train.json` ids found exactly once under `MSAD_anomaly` / `MSAD_normal_training` (never VALU), 120 label-1 ids, **each label agrees with the folder it was found in**, `ceil(decoded / 8) == rows(MSAD_ncc)`; the census prints rows by label and the **train length ruler** (P(abnormal shorter than normal)), the fps spread, and the test ids ∩ train ids = ∅ check is asserted in the runbook. **V2-S** manifest == Nexar's except `stride`, rows == CLIP rows; **bake** manifest `crn` none, `motion` V2-S, `train_ids_sha1` == the sha1 of `labels_train.json`'s ids (the test and TAD bakes carry the same sha1), rows == CLIP rows. **Per row:** J10 (ckpt `global_step` == last `metrics.jsonl` step == the plan's 500), config diff to the Nexar parent ⊆ {`data.dataset`, `train.num_epochs`, `loss.mil_topk_pct`, `model.score_head_kernel`} (L0f also {`kim.*`, `model.motion_dim`, `v2.motion`}) with exactly the §41 values (`msad_train.check_config`), last-epoch `mul_mil` > 0.01 (§31 floor) |
| W Wording (binding) | Never "the better kernel" and never "kernel 9 / 3 is right for MSAD" (it would be chosen on the set it is read on). Never "CCTV data fixes TAD" (§39 W). Never write a Nexar-vs-MSAD sentence from this section: no Nexar-trained row is read here. MSAD micro is clip classification (clip oracle 0.8943 on MSAD-full, §18) and MSAD-traffic micro says nothing (length ruler 0.9639): macro first, micro printed with the oracle. TAD micro is never evidence (oracle 0.9226). Every sentence carries "PRELIMINARY (n = 1)" |

**Note 2026-11-23 (after the first train census, before any score or training; user's Colab run).** The census found 454 of the 480 ids; **all 26 missing ids are `pedestrian_street<n>`**: the release's files are spelled `perdestrian_street<n>` (as `video_msad_all_720.xlsx` lists them, one is an `.avi`), while `labels_train.json` and `MSAD_ncc` use `pedestrian_street<n>`. M2 did its job (the gate failed, nothing ran). Fix: `constants.MSAD_FILE_STEM_ALIASES` maps the typo'd prefix to the id (`msad_zs.canonical_id`); a file spelled exactly as the id wins over an alias, and the census prints the alias-matched ids (`alias_matched`). Nothing else changes. The other 454 videos passed M2 as written: rows == `MSAD_ncc` rows on all of them, `label_rows_differ` 0, 24 off 30 ± 1 fps (6.28 … 30.29). **Partial read, printed only (454 videos, 334 normal):** rows median 52.5 abnormal vs 78 normal, **train length ruler 0.6308** (an abnormal video is shorter than a normal one 63 % of the time): a mild clip-length shortcut sits in the supervision (lesson C28), as the test split has one too (§18: length ruler 0.6466 on MSAD-full). It changes no rule; it is a limit of every row and is reported with them.

**Expectations, written before the numbers (a wrong one changes nothing).** MSAD-traffic intervals wide (19 two-class videos); no prediction on kernel 9 − 3; the train length ruler above 0.5 (the abnormal videos are the shorter ones); no prediction on TAD macro or on motion + KIM − plain CLIP.

**Limits.** n = 1 (s2024); 120 positive videos. The motion stream decodes the blurred release while CLIP stays `MSAD_ncc`'s (§18 M1′: 8 of 12 checked test videos were not bit-equal, cosine ≥ 0.999992); the train videos' CLIP identity is **not** re-checked (M2's row count is the gate). MSAD's frame rates are not all 30 (the table lists 6, 10, 12.5, 15, 23.98, 24, 25, 29.97, 30); the V2-S window is 1.6 s only at 30 fps and the census prints the spread. A repeated-frame distortion of motion at low fps is a hypothesis, never a finding. Abnormal train videos often span most of the video (the annotation's windows are not used: weak supervision), so MIL sees few clean negatives inside a positive bag.

**Runbook.** `colab/final/k5_msad_train.ipynb` (train census + gate, copy / decode, V2-S, bake, KNN, 8 rows, reads, table); table `core.tools.msad_train report` (tests `core/tests/test_msad_train.py`); census `core.tools.msad_zs census --split train` (tests `core/tests/test_msad_zs.py::TestTrainSplit`). Outputs `Thesis-Final/outputs/kim_msad_train/`. **Built 2026-11-23; run and READ OUT 2026-11-23 (below).**

### 41.1 Read-out (2026-11-23, user's Colab run; raw `outputs/v_final/kim_msad_train/`, report `REPORTS/k5msad_train_report.{md,json}`). PRELIMINARY (n = 1, s2024)

**Guards (R7), checked locally on the synced outputs.** Train census PASS: 480 / 480 found (26 through `MSAD_FILE_STEM_ALIASES`), 120 abnormal / 360 normal, `label_rows_differ` 0, no failure; **train length ruler 0.6320** (rows median 52.5 abnormal vs 78 normal); 24 videos off 30 ± 1 fps. Per row: `global_step` 500 = last `metrics.jsonl` step = `verified.json` = the plan (J10); `config.yaml` carries exactly the §41 values (kernel 3 / 9, `mil_topk_pct` 16, 125 epochs, `MSAD-full`; dfi rows `dvs.theta` 0.7 + `ignore`, f rows 1.0 + `span`; L0 rows `kim.enabled` false, `v2.motion` none, `motion_dim` 0); last-epoch `mul_mil` 0.126–0.167 (> the §31 floor 0.01). Last-epoch `mil` 0.001–0.006: the MIL loss is near 0 on MSAD train (the train set is nearly fitted). K5 rows: `motion_share` 0.072–0.088, `w_u_norm` 0.60–0.63.

**MSAD-full (240 videos, macro over 96; rulers: clip oracle micro 0.8943, length ruler micro 0.6466, `t/N` macro 0.7090).** Macro k3 / k9: K5dfi 0.7793 / 0.7787, K5f 0.7666 / 0.7474, L0dfi 0.7183 / 0.7221, L0f 0.7252 / 0.7187. Position alone (`t/N` 0.709) is within ≈ 0.01–0.02 of every L0 row (point values; no paired interval vs the ruler) and ≈ 0.04–0.07 below the K5 rows. K5dfi − L0dfi **+0.0610 [+0.0135, +0.1076] / +0.0566 [+0.0090, +0.1030] above**; K5f − L0f +0.0414 / +0.0287 not separated. Micro 0.89–0.91 ≈ the clip oracle: clip classification, not read.

**MSAD-traffic (140 videos, macro over 19; length ruler micro 0.9639, clip oracle 0.9771, `t/N` 0.5874).** Every motion + KIM − L0 contrast not separated (intervals ≈ ±0.12–0.15). Micro (0.961–0.966) sits at the length ruler: says nothing.

**DVS `ignore` (R5), the one pattern that repeats.** K5dfi − K5f: MSAD-full **+0.0127 [+0.0050, +0.0209] / +0.0313 [+0.0207, +0.0426] above**, MSAD-traffic **+0.0379 [+0.0191, +0.0595] / +0.0570 [+0.0317, +0.0856] above**, TAD +0.0060 / +0.0092 not separated (k3 / k9). On plain CLIP L0dfi − L0f: MSAD-traffic k3 **−0.0246 [−0.0467, −0.0011] below**, every other cell not separated.

**TAD zero-shot (100 videos, macro over 60; `t/N` 0.5872, clip oracle micro 0.9226, length ruler micro 0.8968).** Macro k3 / k9: L0f 0.7676 / 0.7569, L0dfi 0.7603 / 0.7646, K5dfi 0.7228 / 0.7509, K5f 0.7168 / 0.7417 (all ≈ 0.13–0.18 above `t/N`). **K5fM3 − L0fM3 −0.0508 [−0.0992, −0.0052] below**; K5dfiM3 − L0dfiM3 −0.0375, K5fM9 − L0fM9 −0.0152, K5dfiM9 − L0dfiM9 −0.0137, not separated. Same direction as §39 (K5dfiACC − L0dfiACC TAD macro +0.0083, not separated): on neither CCTV corpus does motion + KIM raise TAD macro. TAD micro 0.876–0.893 is below the length ruler (never evidence, W); clip AUC 0.843–0.880 printed (no Nexar row beside it, W). Context, **not matched** (other training data): the released LaGoVAD checkpoint's TAD macro is 0.7578 (Gate T0, §8.1 of `TAD_SETUP.md`).

**Kernel 9 − 3 (printed, never a selection, R6).** 11 / 12 cells not separated; K5f on MSAD-full −0.0191 [−0.0374, −0.0019] below. On TAD the K5 rows lean to 9 (+0.0249, +0.0281), not separated.

**Expectation check.** MSAD-traffic intervals wide: **held**. Train length ruler above 0.5: **held** (0.632). No prediction on kernel, TAD macro, or motion + KIM − plain CLIP.

**Decision (R6): none. K5f stays the reference and the main model; no kernel is picked.** Licensed (PRELIMINARY, n = 1): "Trained on MSAD-full, VideoMAE + KIM with `dvs_anchor_mode=ignore` raises MSAD-full macro by +0.06 over plain CLIP with the same recipe; on zero-shot TAD, motion + KIM does not raise macro, and lowers it in one configuration (kernel 3, DVS off: −0.051). `ignore` raises macro on the KIM host on MSAD-full and MSAD-traffic, not on plain CLIP." **Not licensed (W):** "CCTV data fixes TAD", "KIM helps on CCTV", "kernel 9 / 3 fits MSAD", any Nexar-vs-MSAD sentence (e.g. against §34.1's `ignore` cost on Nexar val), TAD micro as evidence, "plain CLIP localizes on MSAD" (it sits ≈ 0.01–0.02 above `t/N`). **Limits:** one seed; 120 positive videos; train length shortcut 0.632; the MIL loss is near 0 at the end; MSAD fps not all 30 (repeated-frame distortion of motion = hypothesis only). The DVS-ignore × KIM pattern is the only one worth more seeds; a new arm would not settle it. Next steps need their own dated note.

### 41.2 Why TAD micro is above the public checkpoint for the MSAD rows and below it for K5dfiACC (2026-11-24, user's question; post-hoc, printed, decides nothing; n = 1, s2024)

Computed from the saved TAD score files (no new run); raw output `outputs/v_final/kim_msad_train/REPORTS/tad_micro_decomposition.txt`. Written after the numbers. It explains what the micro column measures; it is **not** a ranking of MSAD against ACCIDENT (§42 W) and TAD micro stays no evidence (W).

**TAD test:** 60 accident videos (2,385 rows, median 36) and 40 normal videos (8,660 rows, median 139). The negatives are 8,660 rows of normal videos (84.5 %) and 1,585 normal rows inside accident videos (15.5 %), so micro = 0.845 · AUC(accident rows vs normal-video rows) + 0.155 · AUC(accident rows vs normal rows of accident videos, pooled over the 60 videos).

| row | micro | vs normal-video rows | vs in-video normal rows (pooled) | macro | accident videos pooled after min-max | median in-video score range |
|---|---|---|---|---|---|---|
| LaGoVAD public | 0.791 | 0.837 | 0.539 | 0.758 | 0.731 | 0.064 |
| L0fM3 (MSAD) | 0.892 | 0.953 | 0.557 | 0.768 | 0.709 | 0.014 |
| K5fM3 (MSAD) | 0.878 | 0.937 | 0.554 | 0.717 | 0.687 | 0.029 |
| K5dfiM3 (MSAD) | 0.876 | 0.935 | 0.555 | 0.723 | 0.687 | 0.011 |
| L0dfiACC (ACCIDENT) | 0.739 | 0.753 | 0.660 | 0.690 | 0.674 | 0.888 |
| K5dfiACC (ACCIDENT) | 0.617 | 0.607 | 0.672 | 0.699 | 0.682 | 0.953 |
| K5f (Nexar) | 0.671 | 0.682 | 0.612 | 0.698 | 0.650 | 0.487 |
| K5fno (Nexar, negative-video normals) | 0.501 | 0.483 | 0.599 | 0.695 | 0.631 | 0.184 |

Δ vs public, paired video bootstrap (2,000 draws, seed 2026): micro L0fM3 +0.100 [+0.029, +0.177], K5fM3 +0.087 [+0.024, +0.156] above; K5dfiACC −0.174 [−0.276, −0.068] below; L0dfiACC −0.052 [−0.148, +0.049] not separated. **Macro, all not separated:** L0fM3 +0.010 [−0.054, +0.078], K5fM3 −0.041 [−0.094, +0.015], K5dfiM3 −0.035 [−0.088, +0.022], K5dfiACC −0.059 [−0.142, +0.018], L0dfiACC −0.068 [−0.145, +0.008].

**Reading.** (1) The micro gap sits in the normal-video term: the MSAD rows score normal videos low (mean 0.07–0.09) and whole accident videos high (normal rows 0.66–0.80, accident rows 0.75–0.85); K5dfiACC scores normal videos 0.359, above the normal rows of its accident videos (0.300). (2) **The pooled in-video column is not localization.** It mixes score levels across 60 videos; within a video the MSAD rows rank at 0.72–0.77 (macro) and ACCIDENT's at 0.69–0.70. What the MSAD rows and the public checkpoint lack is score **contrast**: in-video range < 0.1 on 36–40 and 35 of the 60 accident videos (K5dfiACC 4), so one absolute alarm threshold separates videos, not moments. (3) Length does not explain the video separation: on accident / normal pairs within ×1.25 in length (155 pairs) the video AUC is 0.86–0.92 on MSAD rows, 0.81 public, 0.46 K5dfiACC. (4) It needs neither KIM nor DVS: L0fM3 (plain CLIP, DVS off) has the highest macro and near-highest micro, and a logistic gate on mean CLIP fitted on MSAD-train reached video AUC 0.851 on TAD (`RESULTS_TADZ.md` §3.1). (5) With ACCIDENT, VideoMAE + KIM lowers micro further (K5dfiACC − L0dfiACC −0.122 [−0.199, −0.043], §39).

**Candidate reason, not established.** MSAD-train has 360 stand-alone normal videos; ACCIDENT's normals are pre-impact windows of the accident clips, so its objective never asks whether a whole video is normal (the T2 finding, `RESULTS_TADZ.md` §2.3). Stand-alone normals alone are not enough: K5fno (Nexar negative videos, dashcam) reads 0.501. The corpora also differ in label rule, bag (whole video vs 20-row window), steps, fps and source, and recording style (continuous recording vs compiled accident clip) is not ruled out.

**Licensed (PRELIMINARY, n = 1).** "On TAD, the MSAD-trained rows separate videos with an incident from normal videos better than the released checkpoint (micro +0.09 to +0.10); their within-video ranking (macro) is not separated from it." Always print the length ruler (0.897) and clip oracle (0.923) beside a micro number; every MSAD row is below both. **Never:** "MSAD is better than the public checkpoint on TAD" without "micro, video separation"; "MSAD beats ACCIDENT"; "MSAD does not localize" (macro says it ranks); "normal videos fix TAD micro".

## 42. K5dfiACCs: K5dfiACC with the A2 statistics fitted on ACCIDENT train (dated note 2026-11-23, before any number; user)

**Why.** §39 B baked ACCIDENT with **Nexar's** A2 statistics so that K5dfiACC − K5dfi changes one factor (the training data) and reuses every zero-shot bake. §41 Amendment 1 fits MSAD's own statistics, so the two CCTV lines are not symmetric. The user asked to try ACCIDENT with its own statistics (2026-11-23). The open question: does K5dfiACC's input carry a scale mismatch (ACCIDENT's motion moments normalized by Nexar's)?

| # | Choice |
|---|---|
| R1 Row | **K5dfiACCs** = K5dfiACC's §39 command, unchanged (`ACCIDENT_window`, same dedup = none, same KNN cache, same plan steps / epochs, s2024); **config diff to K5dfiACC = {}** (guarded). Only `--clip-dir` differs: `cache/v2/A2_S/ACCIDENT_s8_acc`. Control **K5dfiACC**; K5dfi printed. L0 rows are not re-run: they read plain CLIP, which no A2 statistic touches |
| R2 Statistics | `build_v2_inputs fit-ids` on the ACCIDENT **train** sources that hold a train window (val excluded), CRN off, V2-S, stride 1 over the s8 caches (= §39's bake input). Fitted: `c` (CLIP RMS), `m_u`, `σ_u`; CLIP columns are `x` itself under CRN off |
| R3 Zero-shot bakes | The four bakes K5dfi / K5dfiACC read (DADA test sources, DoTA-CAP-dev s3-from-s1, TAD s8, DAD) are **re-expressed** with ACCIDENT's statistics by `build_v2_inputs restat` (new): under CRN off the motion columns `c (u − m_u) / σ_u` are a per-channel affine map of `u`, so `u` is recovered and re-scaled exactly; CLIP columns copied. No re-extraction. Outputs `cache/v2/A2_S/<name>_acc` |
| R4 Gate | **R2-gate (HARD):** restat of §39's ACCIDENT bake (Nexar statistics) to ACCIDENT's == the direct `fit-ids` bake on every ACCIDENT clip, max \|diff\| < `V2_RESTAT_GATE_ATOL` = 1e-4, CLIP columns bit-equal. Fails → stop. Plus the usual J10, `mul_mil` floor, manifest `train_ids_sha1` = ACCIDENT's on every new bake |
| R5 Reads | TAD (raw), DoTA-CAP-dev (protocol B), DADA whole, DAD, ACCIDENT val (printed), each on the ACCIDENT-statistics bake; K5dfi's protocols |
| R6 Table | `zs_arm_table` control **K5dfiACC**, rows K5dfiACCs and K5dfi, four sets, macro + micro; the statistics line (`c` Nexar → ACCIDENT, σ_u ratio, \|Δm_u\|/σ_u) printed beside it |
| N Status | **No primary, printed only, decides nothing.** K5f stays the reference and the main model |

**What a difference means.** K5dfiACCs − K5dfiACC changes the statistics at **train and at read together** (every zero-shot input moves with them), so it is "the effect of normalizing with ACCIDENT's moments instead of Nexar's", never "ACCIDENT's statistics fit the test sets better", and never evidence on the repeated-frame hypothesis of §39 W by itself.

**Not licensed.** Comparing K5dfiACCs with any §41 MSAD row as "CCTV A vs CCTV B" (different data and different fit sets); any cause from n = 1.

**Runbook.** `colab/final/k5dfi_acc_stats.ipynb` (run after `k5dfi_acc.ipynb`; preflight, tests, fit + gate + restat, train, reads, table). Code: `core.data.v2_inputs.restat_rows`, `core.tools.build_v2_inputs restat`; tests `core/tests/test_v2_restat.py` (7). Constants `KIM_ACCIDENT_STATS_*`, `ACCIDENT_SELF_BAKE_DIRNAME`, `{NEXAR,ACCIDENT}_STATS_BAKE_SUFFIX`, `V2_RESTAT_GATE_ATOL`. **Built 2026-11-23, run by the user (read out 2026-11-24, §42.1).**

### 42.1 Read-out (2026-11-24, user's Colab run; raw `outputs/v_final/kim_accident_stats/`; s2024, n = 1, printed, decides nothing)

**Guards.** J10: ckpt step 1,722 = last `metrics.jsonl` step = K5dfiACC's; `config.yaml` identical to K5dfiACC's (diff {}); gate R2 max |diff| 2.4e-07 < 1e-4; last-epoch mean `mul_mil` 0.313 (> 0.01; K5dfiACC 0.318). Fitted on 1,339 ACCIDENT train clips. Statistics Nexar → ACCIDENT: `c` 0.4293 → 0.4073, σ_u ratio median 1.088 [p10 0.915, p90 1.291], |Δm_u| / σ_u median **0.517**. Last-epoch motion share 0.151 (K5dfiACC 0.228).

| macro / micro | DoTA-CAP-dev | DADA | TAD | DAD |
|---|---|---|---|---|
| K5dfiACC (control) | 0.7372 / 0.6815 | 0.6593 / 0.6420 | 0.6987 / 0.6170 | 0.8266 / 0.7391 |
| **K5dfiACCs** | 0.7452 / 0.6872 | 0.6720 / 0.6491 | 0.7148 / 0.6193 | 0.8228 / 0.7420 |
| K5dfi (Nexar) | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |
| K5f (reference) | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |

| K5dfiACCs − K5dfiACC | macro | micro |
|---|---|---|
| DoTA-CAP-dev | +0.0079 [+0.0042, +0.0115] above | +0.0057 [+0.0029, +0.0084] above |
| DADA | +0.0127 [+0.0087, +0.0168] above | +0.0071 [+0.0039, +0.0103] above |
| TAD | +0.0161 [+0.0029, +0.0361] above | +0.0023 [−0.0089, +0.0148] not separated |
| DAD | −0.0037 [−0.0156, +0.0077] not separated | +0.0029 [−0.0033, +0.0092] not separated |

ACCIDENT val (in-domain, printed): macro 0.7682 vs 0.7667, micro 0.6982 vs 0.6934.

**K5dfiACCs − K5dfi (printed; `REPORTS/zs_arm_table_k5dfiaccs_vs_k5dfi.md`, local):** macro DoTA **−0.0808 [−0.1013, −0.0601] below**, DADA **−0.0957 [−0.1220, −0.0696] below**, TAD −0.0050 [−0.0715, +0.0612] not separated, DAD **+0.0734 [+0.0275, +0.1193] above**; micro −0.0741 / −0.1100 below, TAD and DAD not separated.

**Post-hoc, printed (after the table).** (a) TAD's +0.0161 rests on one of 60 videos: `01_Accident_072` (+0.50); without it +0.0079, without the top 5 +0.0038; median +0.0025; 30 videos up, 19 down. DoTA and DADA are broad (341 up / 213 down, median +0.0055; 223 / 129, +0.0076). (b) The intervals cover items, **not seeds**: the same recipe at two seeds differed by 0.003 (K5 DoTA-CAP-dev) and 0.011 (K5 DADA) (§10.2), so DADA's +0.013 is of the size of the one seed gap measured.

**Reading.** Fitting the statistics on ACCIDENT's own train clips gives a small, broad gain on DoTA-CAP-dev and DADA and no separated change on DAD. It recovers 9 % of K5dfiACC's deficit against K5dfi on DoTA-CAP-dev (0.0079 of 0.0887) and 12 % on DADA (0.0127 of 0.1084). The input-scale mismatch therefore is not why ACCIDENT-trained K5 falls on dashcam sets; about nine tenths of the deficit remains, and `H-dom` (§39 P1) is unchanged.

**Licensed (PRELIMINARY, n = 1).** "Normalizing with ACCIDENT's own moments instead of Nexar's (train and read together) changes ACCIDENT-trained K5dfi by +0.008 [0.004, 0.012] macro on DoTA-CAP-dev and +0.013 [0.009, 0.017] on DADA; it stays 0.08 and 0.10 below the Nexar-trained row." **Not licensed (§42):** "ACCIDENT's statistics fit the test sets better"; "the statistics fix CCTV training"; any sentence on the repeated frames (§39 W); any TAD gain as a finding (one video); DAD's +0.073 over K5dfi as a training-corpus gain (K5f − K5dfi there is +0.058, the DVS cost of §34); any comparison with a §41 MSAD row. **Decision: none; K5f stays the reference and the main model.**

## 43. Pre-onset rise on Nexar val: does K5f's score climb before the alert? (dated note 2026-11-23, before any number; user)

**Why.** The user reads K5f's curves as rising before the accident. §40 shows the supervision that would produce it: only 6.3 % of normal windows lie within 2 rows of their span, and the ~12.5 normal rows inside a positive window get no target (MIL top-k lifts only k = 4; `dvs_sup` runs on normal and spliced rows only, `core/train.py:470`), while `dvs_anchor_mode=span` trains every row of a spliced anchor to 1 (C29). §27 NC6(d) printed one number of this kind for pre-§31 K5 (5 s before onset, share with max > 0.5: 0.3214 vs 0.1871 on negative 5 s chunks); nothing event-locked, and nothing on K5f. This section measures it on saved reads; **no model is loaded, nothing is trained, nothing gates.**

| # | Choice |
|---|---|
| P1 Data | `nexar_val` whole videos (225, 112 positive), the saved `scores/*.npz` of protocol N-B (s1[::3] rows, native frames). Onset = first `gt = 1` frame = `time_of_alert` (the span is `[alert, event + 1.4 s]`, `nexar_build.span_seconds`). Time = frames / `NEXAR_NOMINAL_FPS` |
| P2 Rows | Control **K5f**; printed beside: **K5df** (DVS `span`), **K5dfi** (DVS `ignore`), **K5fno** (normals from negative videos only, §38), **LaGoVAD** (released, zero-shot). All rows must carry identical videos and labels (`tadz_readout.check_labels`) |
| P3 Reads | (a) event-locked mean raw score, 0.5 s bins from −10 s to +3 s, per-bin 95 % interval over videos; each row's negative-video mean beside it. (b) Zone AUC vs negative-video frames (pooled frames, the row's own ranking): far (< −5 s), [−5, −2), [−2, −1), [−1, 0), span. 0.5 = the zone scores like a video with no accident. (c) Per positive video, scale-free **rise** = (mean[−2 s, 0) − far) / (span mean − far), far = mean before −5 s; mean and interval over videos with span − far > `PRE_ONSET_MIN_CONTRAST`. (d) Paired per video: rise(row) − rise(K5f). Intervals: cluster bootstrap (cluster = video) 10,000, seed 2026 |
| P4 Status | **Printed only, decides nothing.** K5f stays the reference and the main model. A rise is a property of the scores; it does not say which loss term makes it |

**Expectations, written before the numbers (a wrong one changes nothing).** E1: K5f's rise > 0 with an interval above 0 and its [−1, 0) zone AUC above 0.5 (the user's reading). E2: K5df's rise > K5dfi's (`span` lifts the whole anchor). E3: K5fno's rise ≥ K5f's (its negatives are all far from any span). No prediction for LaGoVAD.

**Not licensed.** "K5f anticipates accidents" (a rise before the alert is unlabeled positive score, not foresight, and Nexar's alert is itself an anticipation mark). Any cause from a row contrast at n = 1. The head's kernel at the read's 0.1 s rows spans ±0.1 s, so bins before −0.5 s are not the head kernel's smear; the encoder's attention may still carry them.

**Runbook.** `core.tools.pre_onset_read` (tests `core/tests/test_pre_onset_read.py`), notebook `colab/final/pre_onset_read.ipynb` (CPU, reads Drive's saved scores). Report `outputs/kim_fixed/REPORTS/pre_onset/`. **Built 2026-11-23.**

**Read-out of §43 (2026-11-23, Claude, on the saved reads synced from Drive; printed, decides nothing; the user's Colab run of `pre_onset_read.ipynb` reproduces every number of the JSON exactly, only the source paths differ).** Raw: `outputs/v_final/kim_fixed/REPORTS/pre_onset/` (`pre_onset_read.{md,json}`, `pre_onset_curves.png`). Same 225 videos and labels on all five rows (checked).

| row | negative videos | far (< −5 s) | last 2 s | span | rise | zone AUC far / [−5,−2) / [−2,−1) / [−1,0) / span |
|---|---|---|---|---|---|---|
| K5f | 0.046 | 0.025 | 0.092 | 0.511 | **0.179 [0.050, 0.397]** | 0.437 / 0.525 / 0.590 / **0.694** / 0.887 |
| K5df | 0.111 | 0.081 | 0.215 | 0.650 | 0.143 [0.017, 0.245] | 0.485 / 0.577 / 0.641 / 0.741 / 0.895 |
| K5dfi | 0.053 | 0.029 | 0.093 | 0.495 | 0.173 [0.051, 0.372] | 0.458 / 0.550 / 0.613 / 0.708 / 0.878 |
| K5fno | 0.077 | 0.200 | 0.258 | 0.519 | (unstable) | **0.639** / 0.659 / 0.663 / 0.717 / 0.839 |
| LaGoVAD | 0.249 | 0.258 | 0.270 | 0.280 | (unstable) | 0.517 / 0.520 / 0.531 / 0.533 / 0.530 |

Paired rise − K5f: K5df +0.010 [−0.169, +0.120], K5dfi +0.005 [−0.034, +0.050]. Event-locked K5f (raw): 0.026 at −5 s, 0.038 at −3 s, 0.053 at −2 s, 0.092 at −1 s, 0.159 at −0.5 s, 0.284 at 0, peak 0.561 at +1.5 s.

* **E1 holds.** K5f leaves its far level about 2 s before the alert: rise above 0, [−1, 0) frames rank above negative-video frames at 0.694 ([−2, −1) 0.590; [−5, −2) 0.525 ≈ none). Most of the climb is in the last second; at −0.5 s K5f is already 3.5× its negative-video mean.
* **E2: direction on zone AUC, not on rise.** K5df ranks every pre-onset zone higher than K5dfi (0.741 vs 0.708 on [−1, 0)) but also lifts its far level and its negatives (0.111 vs 0.053), so its scale-free rise is not higher (0.143 vs 0.173); the zone AUCs carry no interval. K5dfi − K5f ≈ 0 on every read: under `ignore`, DVS leaves the pre-onset shape as K5f's.
* **E3 does not hold as written: K5fno is not "more rise", it lifts the whole accident video.** Its far frames rank above negative-video frames (0.639; K5f 0.437) and its zones are flat (0.66 → 0.72). Without in-video normals the model separates *which video* rather than *when*; with them (K5f) the far part of an accident video scores below a negative video. Same direction as §38's costs.
* **LaGoVAD (released, zero-shot)** does not localize on Nexar val: span AUC 0.530, flat curve.
* **Defect in P3(c), recorded, rule not changed after the numbers.** The rise is a ratio per video; where span − far is small it explodes (median span − far: K5f 0.504, K5fno 0.293, LaGoVAD 0.009; videos with span − far < 0.05: 12 %, 36 %, 68 %), so K5fno's and LaGoVAD's rise intervals (±2) are not read. `PRE_ONSET_MIN_CONTRAST` = 1e-3 was too small; the zone AUCs are the robust read. The rescaled figure panel is clipped to [−1, 2] (display only).

**Licensed.** "On Nexar val K5f's score starts climbing about 2 s before the alert, mostly in the last second: frames in the last second before the alert rank above negative-video frames at AUC 0.694 (PRELIMINARY, n = 1)." "Without in-video normals (K5fno) the whole accident video is lifted (far-zone AUC 0.639 vs K5f 0.437)." **Not licensed:** "K5f anticipates accidents"; "DVS `span` causes the early rise" (no interval on the zone AUCs, rise not higher); any statement on TAD / DAD / DADA from this read.

### 43.1 Pre-onset rise on DoTA-CAP and DADA (dated note 2026-11-24, user; post-hoc, printed, decides nothing)

**Why.** The user read `zs_curves`' onset-aligned figure (K5dfi vs LaGoVAD, `kim/REPORTS/figures/train_nexar/dota_dada/k5dfi_lagovad_kim_vs_lagovad_s2024/onset_aligned.png`) as rising ~4 s before the GT on DoTA and DADA, much earlier than on Nexar val. That figure and the Nexar one differ in three ways that move a mean curve: the model (Nexar: pre-§31 K5; DoTA / DADA: K5dfi), the pooling (raw vs per-video min-max) and the composition per bin (DoTA's pre-roll median 3.5 s; only 177 / 1,129 clips have >= 5 s, so the −4 s bin is another set of clips than the −1 s bin). **Post-hoc:** the numbers below were first computed in a scratch script and shown to the user, then put into the tool unchanged; nothing here was fixed before a number was seen.

| # | Choice |
|---|---|
| Q1 Sets / rows | Nexar val (§43's five rows), whole DoTA-CAP (1,129, protocol B, native 10 fps) and DADA whole (370, s8 = 0.267 s), each with K5f, K5dfi, L0dfi (plain CLIP, no motion / KIM) and the released LaGoVAD (`constants.PRE_ONSET_ZS_ROWS`) |
| Q2 Fixed composition | Clips with >= 5 s before onset; 1 s zones from −5 s to 0, the same clips in every zone. Rescaled per clip: 0 = mean before −4 s, 1 = span mean, clips with span − far < 0.05 left out (§43's 1e-3 guard was the defect found there). Min-max per clip printed beside it (what the figure shows) |
| Q3 Macro cost | Per-clip AUC with and without the **normal** frames of [−2, 0) s, paired, mean over two-class clips (> 0 = those frames cost macro) |
| Q4 Pre-roll strata | Min-max level of the last 1 s by pre-roll < 2 / 2–4 / >= 4 s (a short pre-roll puts the clip's minimum near the onset) |
| Q5 Status | Printed only, n = 1; intervals: cluster bootstrap over clips 10,000, seed 2026. No arm, no decision; K5f stays |

**GT onsets are comparable on Nexar and DADA, not checked on DoTA.** Nexar's span starts at `time_of_alert`, median 1.43 s before `time_of_event`; DADA's at "abnormal start", median 1.50 s before the accident frame (`NEXAR_SETUP.md` §1.1). Where DoTA's `anomaly_start` sits relative to contact is **not** measured here.

**Read-out (2026-11-24, Claude, on saved reads synced from Drive).** Raw: `outputs/v_final/kim_fixed/REPORTS/pre_onset/pre_onset_read{,_dota,_dada}.{md,json}`, `pre_onset_curves{,_dota,_dada}.png`. The macros reproduce the recorded reads (DoTA K5f 0.8226 / K5dfi 0.8240, §35.1; DADA K5f 0.7744; Nexar K5f whole 0.9210, §43).

Rescaled level on the fixed composition (0 = far, 1 = span):

| set | row | clips | [−4, −3) s | [−3, −2) | [−2, −1) | [−1, 0) |
|---|---|---|---|---|---|---|
| Nexar val | K5f | 99 | 0.01 | 0.02 | 0.04 [0.00, 0.09] | 0.15 [0.10, 0.22] |
| Nexar val | K5dfi | 98 | −0.00 | 0.01 | 0.03 | 0.14 |
| DoTA-CAP | **K5f** | 143 | 0.37 | 0.61 | **0.81 [0.63, 1.01]** | 0.96 |
| DoTA-CAP | K5dfi | 148 | 0.24 | 0.41 | 0.58 [0.50, 0.66] | 0.76 |
| DoTA-CAP | L0dfi | 128 | 0.03 | 0.15 | 0.23 [−0.05, 0.50] | 0.48 |
| DoTA-CAP | LaGoVAD | 76 | 0.11 | 0.23 | 0.44 [0.32, 0.57] | 0.71 |
| DADA | K5f | 109 | 0.16 | 0.19 | 0.36 [0.25, 0.47] | 0.54 |
| DADA | K5dfi | 113 | 0.19 | 0.19 | 0.33 [0.11, 0.61] | 0.44 |
| DADA | L0dfi | 81 | −0.02 | 0.24 | 0.20 | 0.34 |
| DADA | LaGoVAD | 47 | 0.17 | 0.28 | 0.45 [0.28, 0.64] | 0.62 |

Macro cost of the normal frames in [−2, 0) s (macro without them − macro): DoTA K5f **+0.0204 [+0.0142, +0.0266]**, K5dfi +0.0046 [−0.0009, +0.0100], L0dfi −0.0038, LaGoVAD −0.0111; DADA K5f −0.0018 [−0.0094, +0.0056], K5dfi −0.0072, L0dfi −0.0041, LaGoVAD +0.0072; Nexar K5f +0.0016 [−0.0006, +0.0039], K5dfi +0.0033 [+0.0013, +0.0054].

* **On the same definitions, the rise is a DoTA property of the KIM rows.** One to two seconds before the GT, K5f on DoTA already sits at 0.81 of its span level (Nexar 0.04, DADA 0.36); K5dfi 0.58. Plain CLIP trained the same way (L0dfi) sits at 0.23 and the released LaGoVAD at 0.44, so the early part is carried by the motion + KIM path, not by CLIP.
* **It costs macro only on DoTA and only for K5f (+0.020).** K5dfi +0.005 (interval includes 0); on DADA and Nexar the zone costs nothing (dropping it does not raise macro). For scale, K5f − L0dfi on DoTA-CAP macro is +0.146.
* **DADA's figure overstates it.** On the fixed composition K5f's DADA rise (0.36) is not above LaGoVAD's (0.45); the figure's gap comes from K5f's low far level under min-max beside LaGoVAD's high one. Short pre-roll lowers the min-max level of the last 1 s (DoTA K5f 0.33 at < 2 s vs 0.62 at >= 4 s): part of any min-max onset curve is the clip's own length.
* **It reaches past the Nexar label gap.** The unlabeled stretch before a Nexar span is at most 1.87 s (§40); the DoTA rise starts ~4 s before the GT (rescaled 0.37 at [−4, −3) s), so an edge-window fix aimed at that gap is not expected to move DoTA's early part. Not tested.

**Licensed.** "On whole DoTA-CAP, K5f's score reaches 0.81 of its span level 1–2 s before the annotated onset (fixed composition, rescaled; Nexar val 0.04), and those pre-onset normal frames cost 0.020 macro (PRELIMINARY, n = 1, post-hoc)." "The early rise is carried by the motion + KIM rows: plain CLIP trained the same way reaches 0.23." **Not licensed:** "K5f anticipates accidents"; "VideoMAE causes the early rise" (L0dfi changes motion and KIM together); "DoTA's annotation starts late" (not measured); any claim from the min-max figure alone.

**Runbook.** `core.tools.pre_onset_read --set {nexar,dota,dada}` (§4 of each report = Q2–Q4; tests `core/tests/test_pre_onset_read.py`, 6); notebook `colab/final/pre_onset_read.ipynb` now runs the three sets.

## 44. DoTA-CAP and DADA micro: raw against min-max, and where raw micro is lost (dated note 2026-11-24, user)

**Why.** The user asked how micro AUC on DoTA and DADA could rise, what the raw micro is, and how far raw sits from min-max for K5f and K5dfi. Both sets hold an accident in every item (C12), so micro there = ranking inside each clip + how level the scores are across clips; the clip mean alone scores 0.51–0.53.

### 44.1 Raw against min-max (post-hoc, printed, decides nothing)

**Post-hoc:** computed in a scratch script on the saved reads and shown to the user before this note; the min-max column reproduces `zs_arm_table` (`kim_cap_eval/REPORTS/zs_arm_table_cap_all_s2024.md`, `…_dada_lagovad_s2024.md`) to 4 decimals. Items: whole DoTA-CAP (1,129, protocol B) and DADA whole (370, s8), s2024, n = 1.

| set | row | macro | micro raw | micro min-max | raw − min-max | p10 of clip max (raw) |
|---|---|---|---|---|---|---|
| DoTA-CAP | K5f | 0.8226 | 0.6839 | 0.7362 | −0.052 | 0.039 |
| DoTA-CAP | K5dfi | 0.8240 | 0.7207 | 0.7603 | −0.040 | 0.092 |
| DADA | K5f | 0.7744 | 0.7065 | 0.7551 | −0.049 | 0.044 |
| DADA | K5dfi | 0.7677 | 0.7218 | 0.7591 | −0.037 | 0.092 |

Printed beside, **not a protocol** (choosing a pooling after seeing its number is lesson 12/14): per-clip z-score micro DoTA 0.759 / 0.776, DADA 0.752 / 0.754; per-clip rank micro DoTA 0.779 / 0.782, DADA 0.753 / 0.747 (K5f / K5dfi). Clip mean as the only score: 0.51–0.53.

**Licensed:** "Raw micro sits 0.04–0.05 below min-max micro on DoTA-CAP and DADA for K5f and K5dfi; K5dfi's raw micro is higher than K5f's (DoTA +0.037, DADA +0.015, no interval) and its gap is smaller (PRELIMINARY, n = 1, post-hoc)." **Not licensed:** any per-clip normalization as "the" micro; "`ignore` raises micro" as a cause (no interval here; §35.1's paired min-max micro K5dfi − K5f on DoTA-CAP is +0.024 [+0.019, +0.029]).

### 44.2 Check 2: which clips sit low across the set? (rule written before any number of this check)

| # | Choice |
|---|---|
| L1 Items / rows | As §44.1: K5f and K5dfi, whole DoTA-CAP (1,129; cluster = source video, `dota_group`) and DADA whole (370; cluster = video). Raw scores |
| L2 Level of a clip | `A_c` = mean over the clip's **positive** frames of the share of the set's **negative** frames (all clips) scored below it, ties ½. Raw micro = the positive-frame-weighted mean of `A_c` (gate: recomputed equal to the pooled micro within 1e-6, or the tool raises). **Low clip** = `A_c` < 0.5: its accident frames rank below the set's median normal frame. 0.5 is fixed by construction, not fitted. Printed beside: `A_c` under per-clip min-max |
| L3 Normal side (printed, no strata) | `N_c` = mean over the clip's negative frames of the share of the set's positive frames scored below it; **loud clip** = `N_c` > 0.5 |
| L4 Reads | Per set × row: share of low clips, share of positive frames in low clips, share of loud clips. Calibration vs ranking: among low clips, mean own AUC and share with own AUC ≥ 0.5, against the rest. Overlap of K5f's and K5dfi's low sets (both, K5f only, K5dfi only, Jaccard) |
| L5 Strata | §24 FT2's (DoTA `ego_involve`, `night`, `accident_name`; DADA `light`, `weather`, `scenes`, `linear`, ego from `texts`, actor), pooled as §24; covariates §24 FT3 (length, span share, span centre, span seconds). Low share per stratum, Δ vs rest, cluster bootstrap 2,000, seed 2026. **Lead = §24 FT5** (≥ 20 items, ≥ 2 × the overall low share, Δ interval excludes 0; covariate \|AUC − 0.5\| ≥ 0.10 with its interval excluding 0.5) |
| L6 Status | **Printed only, decides nothing, n = 1.** No arm. K5f stays the reference and the main model. Leads found → Claude may propose one targeted check; the user decides |

**Expectations, written before the numbers (a wrong one changes nothing).** E1: K5dfi has fewer low clips than K5f on DoTA-CAP. E2: ego-uninvolved accidents are over-represented among low clips on both sets (§24's direction). E3: most low clips (> 50 %) still have own AUC ≥ 0.5, i.e. raw micro is lost to level, not to ranking. E4: K5f's and K5dfi's low sets overlap with Jaccard > 0.5.

**Limits.** n = 1; about 25 strata and covariates per set and row, so about one false lead per set and row at 95 %; strata are the datasets' own annotations (DADA ego from free text). Code: `core/tools/micro_level_read.py` (reuses `k5_fail_tail`'s strata, pooling and bootstrap), tests `core/tests/test_micro_level_read.py`, output `outputs/v_final/kim_fixed/REPORTS/micro_level/`.

### 44.3 Read-out of check 2 (2026-11-24, Claude, saved reads synced from Drive; printed, decides nothing, n = 1)

Raw `outputs/v_final/kim_fixed/REPORTS/micro_level/micro_level_read.{md,json}`. L2 gate passed on all four set × row reads (weighted `A_c` = pooled raw micro within 1e-6); every clip two-class; DoTA annotations 1,129 / 1,129, DADA workbook rows 370 / 370.

| set | row | micro raw | low clips | low (min-max) | positive frames in low | loud clips | own AUC low / rest | low with own AUC ≥ 0.5 |
|---|---|---|---|---|---|---|---|---|
| DoTA-CAP | K5f | 0.6839 | 0.244 (275) | 0.094 | 0.250 | 0.176 | 0.688 / 0.866 | 0.785 |
| DoTA-CAP | K5dfi | 0.7207 | 0.190 (215) | 0.066 | 0.200 | 0.086 | 0.665 / 0.861 | 0.763 |
| DADA | K5f | 0.7065 | 0.235 (87) | 0.103 | 0.241 | 0.170 | 0.546 / 0.845 | 0.529 |
| DADA | K5dfi | 0.7218 | 0.186 (69) | 0.103 | 0.189 | 0.116 | 0.524 / 0.824 | 0.551 |

Overlap of low sets (K5f, K5dfi): DoTA both 200, K5f only 75, K5dfi only 15, Jaccard 0.690; DADA both 66 / 21 / 3, Jaccard 0.733.

**Leads by L5 (= §24 FT5): one, `span centre` on DADA K5dfi** (AUC 0.383 [0.313, 0.457]; low clips' spans sit earlier, centre 0.46 vs 0.54; K5f 0.409 [0.343, 0.479], 0.009 short of the margin). About one false lead per set and row is expected; it is not read further.

**The largest pattern is not a lead by the 2× rule: accidents without the ego car.**

| set | row | low share, ego-uninvolved (n) | low share, ego-involved (n) | Δ [95 %] | ratio to overall |
|---|---|---|---|---|---|
| DoTA-CAP | K5f | 0.434 (502) | 0.091 (627) | +0.343 [+0.290, +0.396] | 1.78 |
| DoTA-CAP | K5dfi | 0.359 (502) | 0.056 (627) | +0.303 [+0.249, +0.357] | 1.89 |
| DADA | K5f | 0.269 (275) | 0.137 (95) | +0.132 [+0.040, +0.218] | 1.14 |
| DADA | K5dfi | 0.215 (275) | 0.105 (95) | +0.109 [+0.027, +0.187] | 1.16 |

The interval excludes 0 in all four. FT5's 2× bar cannot be reached by a stratum holding 44 % of DoTA's clips unless almost every low clip falls in it (ceiling 1 / 0.445 = 2.25); recorded as a property of the rule, **the rule is not changed after the numbers**.

**Post-hoc, printed (computed after the table, not in L4):** DoTA raw micro by ego stratum, K5f / K5dfi: ego-involved 0.741 / 0.783, ego-uninvolved **0.632 / 0.660**; median clip max (raw) ego-involved 0.993 / 0.998, ego-uninvolved **0.578 / 0.631**. The model saturates on ego accidents and fires at about half that level on third-party ones.

Other strata with an interval excluding 0 (none a lead): DoTA `leave_to_left` / `leave_to_right` more low (+0.11…+0.14, both rows), `oncoming` less (−0.15…−0.19), night less (K5f only, −0.080); DADA daylight (`light` 1) more (K5f only), `linear` 3 more / 4 less, motorbike more (K5f only).

**Expectations, scored.** E1 K5dfi fewer low clips than K5f on DoTA: **held** (215 vs 275). E2 ego-uninvolved over-represented: **held on both sets and both rows** (interval excludes 0), not a lead by FT5. E3 most low clips still rank well inside: **held on DoTA** (0.785 / 0.763), **at the edge on DADA** (0.529 / 0.551; their mean own AUC 0.55 / 0.52): on DADA a low clip is about as often a ranking failure as a level failure. E4 overlap Jaccard > 0.5: **held** (0.69, 0.73).

**Reading.** On DoTA-CAP raw micro is lost mostly to **level, not ranking**: a quarter of K5f's clips (25 % of the positive frames) put their accident frames below the set's median normal frame, and 79 % of those clips still rank their own accident above their own normal frames. These clips are concentrated in accidents that do not involve the ego car (43 % low vs 9 %). K5dfi removes 75 of K5f's low clips and adds 15; its low set is mostly K5f's. The fit with frame-global features, Nexar's ego-involved training events and §24.1's ranking pattern is inferred, not tested.

**Licensed (PRELIMINARY, n = 1).** "On whole DoTA-CAP, 24 % of K5f's clips (19 % for K5dfi) have accident frames that rank below the set's median normal frame under raw scores, and most of them (79 % / 76 %) still rank their own accident above their own normal frames. Accidents not involving the ego car are low about 4.8× (K5f) and 6.4× (K5dfi) as often as ego-involved ones; DADA shows the same direction (about 2×)." **Not licensed:** "K5f cannot detect third-party accidents" (it ranks them inside the clip at own AUC 0.75); "`ignore` fixes level" (no interval on the row difference); any cause; any per-clip normalization as the micro. **What could follow (not authorized, the user decides):** a check whether ego-uninvolved accidents are rare in Nexar's training spans (`nexar_train` annotation, no training); nothing about the model changes from this read.

## 45. Are ego-uninvolved events rare in `nexar_train`'s labelled spans? (dated note 2026-11-24, before any sheet is rated; user)

**Question (user, §44.3 "what could follow").** K5f's low clips on DoTA-CAP sit in accidents without the ego car (43 % vs 9 %). Does K5f's training corpus hold such events at all? Dataset property only: nothing is trained or scored, no arm, `nexar_val` / `nexar_test` are not touched.

**Read before this note (not blind).** Nexar's metadata has no ego column (census: `file_name, light_conditions, scene, time_of_alert, time_of_event, time_to_accident, weather`). The dataset paper (Moura et al., CVPRW 2025, §3 and §3.1) defines *collision* as "an accident event between the ego vehicle and a third party", *near-collision* as "a close call event between the ego vehicle and a third party", and accepts a positive only if it involves "cars or trucks (excluding pedestrians, bicycles, motorcycles, animals, or stationary objects)" with "the (potential) impact … on the front side of the ego vehicle". So by definition the share is 0; this note checks whether the released videos follow the definition.

| # | Choice |
|---|---|
| NE1 Sample | The 30 contact sheets of §29.2's QC draw (seed 2024, 451 `nexar_train` positives; 16 frames from 2 s before `t_event` to 1.4 s after). Already on disk; the draw does not depend on the event. Membership in `nexar_train` asserted |
| NE2 Label per sheet | **E** ego is a party: contact, or another road user moves into / stops in the ego path so that ego is at risk. **T** third-party only: the event is between other road users and ego is not at risk. **M** both. **U** cannot tell from 16 frames. Rater: Claude, by eye, one pass, without reading the sheets' captions or the user's QC ratings first |
| NE3 Reading | Primary count T; sensitivity T + M + U. One-sided 95 % Clopper-Pearson upper bound on n = 30. **RARE** iff the bound on T is < 0.222 (half of DoTA-CAP's uninvolved share, 502 / 1,129 = 0.445), i.e. T ≤ 2 (bounds for 0 / 1 / 2 / 3: 0.095 / 0.149 / 0.195 / 0.239). **NOT RARE** iff T ≥ 3. If T ≤ 2 but T + M + U ≥ 3: "RARE on T, not settled by the sheets" |
| NE4 Printed, never deciding | The share of the 451 Qwen2.5-VL-7B captions (§29.2, QC FAIL 3.43, prompt does not ask for the ego vehicle) that name the ego / camera car. Weak: the captioner's known failure is the ego perspective |
| NE5 Status | A dataset property. RARE does **not** show that rarity causes the low level on DoTA: no training run varies it. K5f stays |

**Expectation (written now).** T = 0 and U ≤ 5: "RARE".

### 45.1 Read-out (2026-11-24, Claude, local; dataset property, decides nothing)

Ratings `outputs/v_final/kim_fixed/REPORTS/nexar_ego/sheet_ratings.csv`. All 30 ids are in `nexar_train` (NE1).

| label | n of 30 | share, one-sided 95 % upper bound |
|---|---|---|
| E ego is a party | 28 (3 low confidence: 00301, 00839, 00856) | |
| **T third-party only** | **0** | **0 %, ≤ 0.095** |
| M both | 0 | |
| U cannot tell | 2 (00351, 00403) | |

**NE3: RARE** (T = 0 ≤ 2; T + M + U = 2 < 3, so settled). Stricter, not in the rule: if the 3 low-confidence E also count as U, the sensitivity count is 5 and the sheets alone would not settle it; in those 3 what is uncertain is whether the event shows, not that another pair of road users collides. No sheet shows two other road users colliding. NE4: 41 / 451 captions (9.1 %) name a camera / dashcam viewpoint; 2 of the 30 sheets. As expected, that text is no ego label and is not used.

**The definition is stronger than the sheets.** Nexar accepts a positive only if (a) ego and a third party collide or nearly do, (b) the other party is a **car or truck** (no pedestrian, bicycle, motorcycle, animal, stationary object), and (c) the (potential) impact is on ego's **front side**. The third-party accident is therefore outside the training support by construction, not merely rare; so are pedestrian / two-wheeler / obstacle accidents and impacts on ego's rear. Negatives are "no event" videos.

**Expectation, scored.** T = 0 and U ≤ 5: **held**.

**Licensed.** "By the dataset's own definition Nexar's positives are ego-involved front-side (near-)collisions with a car or truck; in a random 30 of the 451 training positives no third-party-only event was seen (upper bound 9.5 %, one rater, 16 frames each). About 44 % of DoTA-CAP's clips (ego-uninvolved) and 74 % of DADA's (no `ego-car` in `texts`) are events of a kind K5f had no positive example of." **Not licensed:** "the low level on third-party accidents is **caused** by this" (nothing varied it; small distant motion is the rival reading, §24.1); any claim on §43's pre-onset rise. **Limits.** One rater (Claude), not blind to the definition; 16 frames per event; the flags for DoTA / DADA are the datasets' own.

**What could follow (not authorized, the user decides).** A CPU read on saved scores: DADA's videos are 76 % non-ego by `texts` (472 / 1,945 ego, whole workbook; T2's own train list is not local), so the T2-trained rows (§37, K5fT2 / K5dfiT2, whole DoTA-CAP saved) saw third-party accidents and the Nexar rows did not. Run `micro_level_read` on K5fT2 vs K5f and compare the low-share gap (ego-uninvolved − ego-involved). If the missing support drives the level, the gap should shrink on the T2 rows; if it does not, the motion-size reading stands. Confounded (corpus, windows and normals all differ), so it can weaken or support the reading, not prove it. Rule and bars in a dated note first.

## 46. Does K5f's ego gap in level shrink when the corpus holds third-party accidents? (dated note 2026-11-24, before any number of this check; user)

**Question (user, §45.1).** §45: Nexar holds no third-party accident positive. §44.3: K5f's low clips (accident frames below the set's median normal frame) are 43 % of DoTA-CAP's ego-uninvolved clips against 9 % of the ego-involved. Two readings are open: **support** (the corpus never showed such positives) and **size** (third-party accidents move few pixels at distance). K5fT2 / K5dfiT2 (§37) are the same recipes trained on T2, cut from DADA-2000, where about 3 / 4 of the accident videos do not name an ego-car (§45.1; counted on T2's own window list below). No training, no inference: saved whole DoTA-CAP reads (1,129 clips, protocol B), s2024.

**Read before this note (not blind).** §37.1's macro / micro of these rows (K5fT2 0.809 / 0.717 min-max, K5f 0.823 / 0.736); §44.3's strata of K5f and K5dfi; no ego stratum of any T2 row.

| # | Choice |
|---|---|
| EG1 Items / rows | Whole DoTA-CAP, raw scores. **Primary pair K5fT2 vs K5f**; secondary K5dfiT2 vs K5dfi. Ego flag = DoTA `ego_involve` (502 false / 627 true). DADA is not read (in-domain for the T2 rows, §37) |
| EG2 Level | §44.2 L2's `A_c` and low flag (`A_c` < 0.5), computed per row from that row's own pooled negatives (a global offset of the scores cancels). Gate: L2's, and K5f / K5dfi must reproduce §44.3 (0.434 / 0.091; 0.359 / 0.056) to 0.001 or the tool raises |
| EG3 Reads | Per row: low share per ego group, gap `G` = low share(uninvolved) − low share(involved), gap `G_A` = mean `A_c`(involved) − mean `A_c`(uninvolved), own AUC per group. Paired differences T2 − Nexar per group (`Δ_U`, `Δ_I`) and of the gaps (`D` = G(T2) − G(Nexar); `D_A` likewise). Cluster bootstrap over source videos (`dota_group`), 2,000 draws, seed 2026, same draws for every row |
| EG4 Reading (primary pair) | **SUPPORT-CONSISTENT** iff `D`'s interval < 0, `D_A`'s interval < 0 and `Δ_U`'s (low share) interval < 0: the gap narrows because third-party clips rise. **NARROWED BY THE OTHER SIDE** iff `D` < 0 but `Δ_U` is not below 0 (the involved clips got worse). **NO SHRINK SHOWN** iff `D`'s interval includes 0: then a shrink of ≥ ⅓ of K5f's G (0.114) is **excluded** if `D`'s lower bound > −0.114, else **not settled**. **WIDENED** iff `D`'s interval > 0. `D` < 0 with `D_A` not below 0 = **MIXED** |
| EG5 Secondary | Same reads on the K5dfi pair; printed, never deciding; a different verdict is stated |
| EG6 Status | Printed, decides nothing, n = 1. K5f stays. **No verdict says T2 is the better corpus, or that support is the cause:** the corpora also differ in windows, normals, domain, frame rate, statistics |

**Expectations, written before the numbers.** `D` < 0 with interval, `Δ_U` < 0; size: −0.05 to −0.20 (part of the gap, not all). Own AUC of the uninvolved group does not rise by more than 0.02.

### 46.1 Read-out (2026-11-24, Claude, saved reads; printed, decides nothing, n = 1)

Raw `outputs/v_final/kim_fixed/REPORTS/ego_gap/ego_gap_read.{md,json}`. EG2 gates passed (K5f 0.434 / 0.091, K5dfi 0.359 / 0.056 reproduced). T2 share, counted on `outputs/EDA/DADA2000_orig_T2_w20s8/meta.json` (assumed to be the geometry K5fT2 trained on, W = 20 / hop 8; corrects §45.1's "T2's train list is not local"): **2,770 abnormal train windows (T2-val out), 73.8 % with no `ego-car`**, 75.5 % of the sources.

| row | trained on | low U / I | G | mean `A_c` U / I | own AUC U / I |
|---|---|---|---|---|---|
| K5f | Nexar | 0.434 / 0.091 | 0.343 | 0.550 / 0.818 | 0.749 / 0.881 |
| K5fT2 | T2 | 0.380 / 0.105 | 0.275 | 0.581 / 0.804 | 0.758 / 0.850 |
| K5dfi | Nexar | 0.359 / 0.056 | 0.303 | 0.603 / 0.848 | 0.751 / 0.882 |
| K5dfiT2 | T2 | 0.327 / 0.094 | 0.233 | 0.617 / 0.795 | 0.747 / 0.837 |

| T2 − Nexar | primary K5fT2 − K5f | secondary K5dfiT2 − K5dfi |
|---|---|---|
| `D` (gap in low share) | **−0.068 [−0.128, −0.005]** | −0.070 [−0.130, −0.012] |
| `D_A` (gap in `A_c`) | −0.046 [−0.077, −0.016] | −0.067 [−0.096, −0.039] |
| `Δ_U` low share, uninvolved | −0.054 [−0.111, +0.006] | −0.032 [−0.085, +0.026] |
| `Δ_I` low share, involved | +0.014 [−0.013, +0.044] | +0.038 [+0.012, +0.065] |
| mean `A_c` uninvolved | +0.032 [+0.005, +0.058] | +0.014 [−0.011, +0.038] |
| mean `A_c` involved | −0.014 [−0.030, +0.001] | −0.053 [−0.066, −0.038] |
| own AUC uninvolved | +0.009 [−0.012, +0.029] | −0.004 [−0.021, +0.012] |
| own AUC involved | −0.032 [−0.044, −0.019] | −0.046 [−0.056, −0.034] |

**EG4 verdict (primary): NARROWED BY THE OTHER SIDE** (`D` < 0, `Δ_U` includes 0); same on the secondary pair. The rule's wording is sharper than the primary numbers: "not below 0" is not "the involved side moved". On the primary pair neither side's low share is separated from 0 (−0.054 and +0.014 add to −0.068) and the continuous level of the uninvolved group is above (+0.032); on the secondary pair the involved side is clearly worse (−0.053 in level) and the uninvolved side is not separated. The rule is not changed after the numbers.

**Expectations, scored.** `D` < 0 with its interval: **held**. `Δ_U` < 0 with its interval: **wrong** (just misses). Size −0.05 to −0.20: **held** (−0.068). Uninvolved own AUC not up by more than 0.02: **held** (+0.009).

**Reading.** A corpus where 74 % of the positives are third-party narrows the gap by about a fifth (0.343 → 0.275), part of it because the ego-involved level is not higher on T2, and the interval's best end still leaves a gap of 0.215. Inside the clip T2 does not rank third-party accidents better (own AUC +0.009, interval includes 0). So the missing support in Nexar (§45) can explain **at most a minority** of the level gap here, and most of it survives in a corpus that has the support. That favors the rival reading (third-party accidents move few pixels; frame-global features) but does not test it.

**Licensed (PRELIMINARY, n = 1, whole DoTA-CAP).** "Trained on T2 (74 % third-party positives), K5f's recipe still puts 38 % of ego-uninvolved clips low against 10 % of ego-involved; the gap is 0.07 [0.005, 0.13] smaller than on Nexar (0.34) and the uninvolved clips' own AUC is not separated." **Not licensed:** "Nexar's missing third-party positives are the cause" / "are not a cause"; "T2 fixes third-party accidents"; "motion size is the cause"; T2 as the better corpus (§37). **Limits.** T2 differs from Nexar in windows, normals, frame rate, country and statistics; DADA's flag is free text; one seed. **Next (user's call):** nothing follows from this alone; a size read on saved tracks (box size at the accident frame vs `A_c` within the uninvolved group) would test the rival, if boxes are on disk.

## 47. Does box size explain K5f's ego gap in level on DoTA-CAP? (dated note 2026-11-24, before any score is paired with a box; user)

**Question (user, §46.1).** §46 left the rival reading open: third-party accidents score low because they are small in the frame. DoTA annotates the boxes of the objects in the accident. No training, no inference; saved whole DoTA-CAP reads (protocol B, raw scores, s2024).

**Read before this note (dataset property only, no score).** Boxes exist only inside `[anomaly_start, anomaly_end)`; all 502 ego-uninvolved clips have some, 564 of 627 ego-involved do (63 have none). Median box share of the frame (median over annotated frames of the largest box, 1280 × 720): uninvolved 0.018, involved 0.101. Also read: all of §44–§46 (K5f low shares by ego; no read by size).

| # | Choice |
|---|---|
| BS1 Items / rows | The 1,066 boxed clips (502 U / 564 I). Levels `A_c`, low flags and own AUC are computed on **all 1,129** clips first (§44.2, §46 EG2 gate), then restricted. **K5f primary**, K5dfi secondary, K5fT2 / K5dfiT2 printed. Low shares of the 63 unboxed involved clips printed |
| BS2 Size | `size` = median over the clip's annotated accident frames of the largest box's share of the 1280 × 720 frame. Printed beside: `speed` = median centre displacement per frame (px) of same-`obj_track_id` boxes, and `size × speed` |
| BS3 Relation | Within U: Spearman `ρ_A` = ρ(size, `A_c`) and ρ(size, own AUC), cluster bootstrap (`dota_group`, 2,000, seed 2026). Printed: within I; ρ with the linear rank effect of §24 FT3's covariates (length, span share, span centre, span seconds) removed; the same for speed and size × speed |
| BS4 Size-matched gap | Bins on `size` with fixed edges 0.01 / 0.02 / 0.04 / 0.08 / 0.16 (I counts 20 / 29 / 70 / 109 / 189 / 147, U 156 / 116 / 94 / 73 / 42 / 21). `G_raw` = low share U − I on boxed clips. `G_adj` = Σ_b w_b (low_U,b − low_I,b), w_b = U's share of bin b. `R` = `G_raw` − `G_adj`: how much of the gap the size mix carries. Same draws; a draw with an empty U or I bin is dropped and counted |
| BS5 Verdict (primary; every state on its own interval) | **SIZE-CONSISTENT** iff ρ_A low > 0 **and** R low > 0 **and** R ≥ ⅓ `G_raw`. **RELATED, GAP NOT EXPLAINED** iff ρ_A low > 0 **and** R high < ⅓ `G_raw`. **NO SIZE RELATION** iff ρ_A's interval includes 0 with high < 0.2 **and** R high < ⅓ `G_raw`. Anything else: **NOT SETTLED**. A verdict on the secondary row that differs is stated |
| BS6 Status | Printed, decides nothing, n = 1. K5f stays. Size is not motion: SIZE-CONSISTENT does not show that motion energy or a frame-global feature is the cause |

**Expectations, written before the numbers.** ρ_A > 0 with its interval, 0.15 to 0.35. ρ(size, own AUC) > 0. R ≥ ⅓ `G_raw`: SIZE-CONSISTENT. Within I the relation is weaker than within U. Limits known in advance: the smallest involved bin holds 20 clips; "involved" also differs in ways size does not carry.

### 47.1 Read-out (2026-11-24, Claude, saved reads and annotations; printed, decides nothing, n = 1)

Raw `outputs/v_final/kim_fixed/REPORTS/box_size/box_size_read.{md,json}`. EG2 gates passed; no bootstrap draw dropped. Unboxed involved clips (63) low: K5f 0.190, K5dfi 0.159 (0.080 / 0.044 for the boxed ones), so the boxed `G_raw` (0.354 / 0.314) sits a little above §46's.

| row | ρ_A within U | ρ(size, own AUC) within U | `G_raw` | `G_adj` | `R` | R / `G_raw` | verdict |
|---|---|---|---|---|---|---|---|
| **K5f (primary)** | +0.181 [+0.090, +0.269] | +0.255 [+0.165, +0.335] | 0.354 | 0.289 [0.206, 0.371] | **+0.066 [+0.006, +0.127]** | 0.19 | **NOT SETTLED** |
| K5dfi | +0.226 [+0.133, +0.313] | +0.275 [+0.185, +0.353] | 0.314 | 0.254 [0.177, 0.334] | +0.060 [+0.008, +0.120] | 0.19 | NOT SETTLED |
| K5fT2 (printed) | +0.194 [+0.119, +0.273] | +0.228 [+0.140, +0.310] | 0.287 | 0.181 [0.089, 0.265] | +0.105 [+0.038, +0.183] | 0.37 | SIZE-CONSISTENT |
| K5dfiT2 (printed) | +0.225 [+0.153, +0.297] | +0.249 [+0.156, +0.334] | 0.243 | 0.188 [0.108, 0.263] | +0.056 [+0.001, +0.121] | 0.23 | NOT SETTLED |

Why NOT SETTLED on K5f: ρ_A is separated from 0 and R is separated from 0, but R's point (0.066) is under ⅓ `G_raw` (0.118) while its upper bound (0.127) is above it. K5fT2's verdict is printed only and is the one of four; it is not the result.

Low share by size bin (K5f, U / I; n U / I): < 0.01 0.551 / 0.250 (156 / 20); 0.01–0.02 0.405 / 0.103 (116 / 29); 0.02–0.04 0.351 / 0.129 (94 / 70); 0.04–0.08 0.315 / 0.083 (73 / 109); 0.08–0.16 0.476 / 0.079 (42 / 189); ≥ 0.16 0.429 / 0.027 (21 / 147).

Other printed reads (K5f, U): size × speed ρ_A +0.236 [+0.149, +0.315] (+0.225 with §24 FT3's covariates removed), speed +0.188, size +0.165 with the covariates removed. Box speed carries camera motion, so it is not an object speed. Within I the size relation is at least as strong (ρ_A +0.233 [+0.159, +0.305]).

**Expectations, scored.** ρ_A > 0 in 0.15–0.35: **held** (0.181). ρ(size, own AUC) > 0: **held**. R ≥ ⅓ `G_raw`, SIZE-CONSISTENT: **wrong** (0.19). Weaker within I than U: **wrong** for K5f and K5dfi.

**Reading.** Size matters a little, also for ranking (own AUC 0.686 below 1 % of the frame vs 0.75–0.81 above), but it carries about a fifth of the ego gap (2 % to 36 % of it at the interval's ends). At matched size uninvolved clips are still low 0.29 [0.21, 0.37] more often, and big third-party accidents are not spared: at ≥ 8 % of the frame K5f puts 0.43–0.48 of the uninvolved clips low against 0.03–0.08 of the involved (63 uninvolved clips). With §46 (the T2 corpus narrows the gap by about a fifth, at most 37 % at the interval's end), **neither reading carries the majority of the gap**; what remains is that a collision with the camera car shakes, stops or turns the camera, which a frame-global feature reads and a third-party accident does not. That is inferred, not tested.

**Licensed (PRELIMINARY, n = 1, whole DoTA-CAP, 1,066 boxed clips).** "Within ego-uninvolved accidents, the level of K5f's score rises weakly with the size of the accident's boxes (ρ 0.18 [0.09, 0.27]). At matched size the ego-uninvolved clips are still low 0.29 [0.21, 0.37] more often than the ego-involved ones; size carries about 0.07 of a 0.35 gap." **Not licensed:** "small motion is the cause" / "size is not a factor"; any sentence on motion energy; K5fT2's SIZE-CONSISTENT as the result. **Limits.** n = 1; box size is the annotator's largest involved object, not the moving pixels; the smallest involved bin holds 20 clips.

**Close (Claude's recommendation, user decides).** Two rivals were tested on saved data and neither explains the gap. A test of the remaining reading (camera shake / ego motion) needs a new read of the optical flow or camera motion; DoTA's frames are gone, so it is not cheap. Claude recommends closing the line and writing the third-party gap into the limitations, with these two bounded reads.

## 48. The ego-uninvolved line is CLOSED; it is a limitation (dated note 2026-11-24, user)

The user closed the line after §45–§47 ("ok đóng line đi"). Nothing follows: no arm, no corpus change. K5f stays. DAD and CCD as a source of third-party positives were discussed and not taken: DAD's label is a fixed last 0.4 s on 15-row clips (`t/N` macro 1.000, `ZS_DAD.md` §8), has no ego flag and is one of the four zero-shot sets; CCD stays PARKED (2026-09-26). Reopening takes a new dated note.

**Limitation text (for the report; n = 1, s2024, whole DoTA-CAP, raw scores).** "K5f scores accidents that do not involve the camera car at a lower level: the accident frames of 43 % of the ego-uninvolved DoTA-CAP clips rank below the set's median normal frame, against 9 % of the ego-involved ones. Nexar's positives are by definition ego-involved front-side (near-)collisions with a car or truck. Training on T2, where 74 % of the abnormal windows do not name an ego car, narrows the gap only by 0.07 [0.005, 0.13] of 0.34, and matching the accident's box size leaves 0.29 [0.21, 0.37] of 0.35. Neither the missing positives nor the box size explains most of the gap; the cause is not established." **Never write:** that either factor is, or is not, the cause; that the two shares add up; that K5f cannot detect third-party accidents (own AUC 0.75 within the clip).

## 49. K5fepre / K5fepost (edge normal windows, once per epoch) and K5fbk (bottom-k MIL) on K5f (dated note 2026-11-24, before any number of these rows; user)

**Why.** The user reads K5f's curves as rising before the accident (§43, §43.1) and wants (a) normal windows right beside the span ("edge windows") and (b) the bottom-k MIL term. Claude advised against (b); **the user keeps it: its two earlier failures were on the DADA archive and TAD, corpora that are not clean training sets (C28, C14), so they are not counted as evidence about Nexar.** This note reopens nothing closed: §26 closed `min_positive` and other top-k divisors; §27 / §38 changed the *source* of the normals. Here the positive windows, the random normal pool and every other key are K5f's.

**Printed before this rule (post-hoc, saved K5f reads, n = 1; they motivate the arms and choose none of their geometry).**

| read | value |
|---|---|
| macro gained by dropping the normal frames of [−2, 0) s: Nexar val / DoTA-CAP / DADA | +0.002 ns / +0.021 / −0.002 ns (K5dfi DoTA +0.005 ns) |
| same, (0, 2] s **after** the span | **+0.006 / +0.020 / +0.030**, every interval above 0 (K5dfi DoTA +0.037, DADA +0.040) |
| Nexar val windows with an abnormal row (433, rows cut from the whole-video reads): bottom-4 rows' mean score, median | 0.001 (93 % of the windows below 0.05); other normal rows 0.151, abnormal rows 0.494 |
| lost AUC carried by the bottom-4 rows (Nexar val) / by the lowest 20 % of frames (DoTA, DADA) | 1.7 % / 0.9 %, 1.4 % |
| share of those bottom-4 rows that are abnormal | 8.6 % |
| edge windows the random pool already holds (gap 0; builder on the census, exactly K5f's 7,311 / 1,689) | before 62 videos, after 61 of 451 (nearest normal window: mean gap ~3.5 rows, from an approximate rebuild) |
| videos with room for a 20-row edge window: before / after | 449 / 449 |
| positive train windows with fewer than 4 normal rows | 1.3 % |

| # | Choice |
|---|---|
| ED1 Rows | **K5fepre**, **K5fepost**, **K5fbk**, each = K5f's exact command (`kim.enabled`, `kim.prototypes`, `kim.edge_mask`, DVS off, A2 V2-S bake, 33 epochs, 1,749 steps), **s2024**, control **K5f** (not retrained). Never combined in one run. Order: K5fepre, K5fepost, K5fbk |
| ED2 Edge window | Per `nexar_train` positive video, 20 rows at s8: **pre** = the rows ending right before the first abnormal row, **post** = the rows starting right after the last one (gap 0, no guard). Videos without room contribute none. Every row must be normal (asserted) |
| ED3 Dose (user: option C) | Each edge window is drawn **exactly once per epoch**, as every positive window is: normal slots `0 … n_edge − 1` are the edge windows, the other 1,689 − n_edge slots draw at random from K5f's unchanged 5,622. ~449 / 1,689 ≈ 27 % of the normal items, ~8.5 of ~32 per batch; nothing is tuned. `rng.choice` is still called on every slot, so the random slots draw K5f's items |
| ED4 Corpus | `nexar_build corpus --constructions window_edgepre window_edgepost` = `window` plus `windows.json` entries for the edge windows and `forced_normals.json`. `labels_train.json`, `frame_labels_test.json`, `defs.json`, `meta.json` byte-identical to `Nexar_window`. **Config diff to K5f = {}** (the corpus is a CLI path) |
| ED5 Bottom-k | `loss.bottomk_weight=1.0` (symmetry with `L_MIL`), `loss.bottomk_topk_pct=5` (k = 4, `L_MIL`'s k; the default 16 gives k = 1). **Config diff = those two keys.** Acts on the host score only, not on KIM's. No other weight or k |
| ED6 Guards | J10 (1,749); config diff as above; last-epoch `mul_mil` > 0.01; K5fbk step-1 `mil` = K5f's within `KIM_START_TOL` (same draws, else a flag). Edge rows' step-1 `mil` must **differ** from K5f's (equal = the forced windows were not read; the notebook raises) |
| ED7 Reads | K5f's five: Nexar val with `--save-scores`, DoTA-CAP-dev protocol B, DADA whole, TAD s8, DAD s7. Macro **and** micro on all four zero-shot sets (`zs_arm_table`) |
| ED8 Early-rise target | **T1** Nexar val: zone AUC of [−1, 0) s frames against every negative-video frame (§43's definition), row − K5f, video bootstrap (positive and negative videos resampled): **hit** iff the interval is below 0. **T2** DoTA-CAP-dev: per clip, AUC of the span's frames against the **normal frames of [−2, 0) s** ("onset AUC"), row − K5f, source-video clusters: **hit** iff the interval is above 0. Printed beside: the onset AUC on Nexar val and DADA; §43.1's rescaled level; the post-span twin (span vs (0, 2] s). **Change from the form agreed in chat:** T2 was "the rescaled level at [−2, −1) s", which only 58 dev clips define; checked on the known pair K5dfi − K5f (dev), its interval is [−0.44, −0.08] against the onset AUC's [+0.034, +0.059] on 568 clips. Chosen before any number of these rows |
| ED9 Cost / gain | §34's label per set (margin 0.01 and the interval past 0). **COST** on any of Nexar val crops, DoTA-CAP-dev, DADA, DAD; **GAIN** only on DoTA-CAP-dev or DADA. TAD printed |
| ED10 K5fepre | T1 and T2 hit, no COST → **REDUCED**, s2025 may run. T1 or T2 hit and a COST → **TRADE-OFF**, closed. Exactly one hit, no COST → **PARTIAL**, closed. No hit → **NOT REDUCED**, closed |
| ED11 K5fepost, K5fbk | GAIN and no COST → s2025 may run; anything else → closed. T1 / T2 printed |
| ED12 After | Nothing is adopted at n = 1; K5f stays the reference. Closed = no other dose, gap, guard, weight or k, and no combination (lesson 14) |

**Expectations, written before the numbers (a wrong one changes nothing).** K5fepre: T1 hit, T2 not (§43.1: the DoTA rise starts ~4 s before the GT, the edge window adds ~1 s); DADA not separated; a possible COST on Nexar val, because the rows just before and just after the alert look alike. K5fepost: no prediction of sign (the post-span cost is real on three sets, but Nexar's 1.4 s tail is borrowed from DADA and the causal 1.5 s VideoMAE window still holds accident motion in the first ~6 rows after it, so those rows may teach "accident motion is normal"). K5fbk: no difference on any set (the rows it pushes are already near 0).

**Not licensed, whatever the outcome.** "Edge windows reduce the early rise" from one hit or from TRADE-OFF; "K5f anticipates"; which loss acts (`mil`, `kim_mil`, `mul_mil` and NKC all see the edge windows); any sentence about bottom-k beyond this host, weight and k.

Code: `core/data/dataset.py` (`forced_normals.json`, opt-in), `core/tools/nexar_build.py` (`build_window_edge`), `core/tools/k5_edge_report.py`; tests `core/tests/test_dvs_dataset.py`, `test_nexar_pipeline.py`, `test_k5_edge_report.py`; runbook `colab/final/k5_edge.ipynb`; outputs `Thesis-Final/outputs/kim_edge/`.

### 49.1 Read-out (s2024, n = 1): nothing adopted, all three rows CLOSED, K5f stays (2026-11-24)

Raw: `outputs/v_final/kim_edge/REPORTS/{k5_edge_report,zs_arm_table_k5_edge_s2024,k5_edge_summary}.md`. **Re-checked locally 2026-11-24 after the full sync:** `k5_edge_report` re-run from the saved scores is identical to the Colab report; every row has 1,749 metrics rows and `verified.json` step 1,749; config diff to K5f = `loss.normal_caption` (added later, at its default) for the edge rows, plus the two bottom-k keys for K5fbk. Per-video curves and onset figures: `outputs/v_final/kim/REPORTS/figures/train_nexar/<set>/kim_edge_s2024/` (gitignored). Last-epoch `mul_mil` 0.1335 / 0.1173 / 0.0730 (floor 0.01).

| row − K5f | Nexar val crops | DoTA-CAP-dev | DADA | DAD | TAD (printed) | T1 Nexar [−1, 0) | T2 DoTA onset AUC | verdict |
|---|---|---|---|---|---|---|---|---|
| K5fepre | **−0.0149 [−0.0214, −0.0087]** COST | +0.0026 n.s. | −0.0086 [−0.0158, −0.0014], inside the margin | −0.0037 n.s. | +0.0035 n.s. | −0.0119 [−0.0277, +0.0045] no | **+0.0663 [+0.0546, +0.0787]** hit | **TRADE-OFF, closed** |
| K5fepost | +0.0020 n.s. | −0.0062 n.s. | +0.0059 n.s. | **−0.0942 [−0.1161, −0.0732]** COST | +0.0086 n.s. | +0.0172 n.s. | −0.0416 (below) | **closed** |
| K5fbk | **−0.0732 [−0.0883, −0.0587]** COST | −0.0063 n.s. | **−0.0331 [−0.0427, −0.0232]** COST | **−0.0513 [−0.0732, −0.0301]** COST | −0.0055 n.s. | −0.0720 hit | +0.0460 hit | **closed** |

**The edge rows move the curve in time, they do not sharpen it.** Onset AUC against the 2 s on each side (row − K5f, every interval past 0 except the one marked):

| set | K5fepre: pre / post | K5fepost: pre / post |
|---|---|---|
| Nexar val | −0.006 n.s. / −0.030 | −0.028 / **+0.036** |
| DoTA-CAP-dev | **+0.066** / −0.071 | −0.042 / **+0.043** |
| DADA | **+0.019** / −0.049 | −0.023 / **+0.049** |

Each row gains on its own side and loses on the other; macro does not move on DoTA for either. The [−2, 0) s frames cost K5f 0.0192 of DoTA macro and cost K5fepre −0.0084, (negative = those frames now help its macro), so the pre-span cost is gone on DoTA; the post-span side got worse by the onset AUC above. A drop in the rise measure is separated on DoTA (−0.210 [−0.369, −0.090]) and DADA (−0.239 [−0.525, −0.032]); on Nexar it is not (+0.061 [−0.049, +0.257]).

**K5fbk: on DoTA the early rise is visibly smaller** (onset-aligned min-max mean at the GT onset about 0.28 against K5f's 0.62, read by eye from `onset_aligned.png`; consistent with T2 +0.0460 and the micro gain), **but the curve also starts late inside the span, and Nexar val, DADA and DAD lose macro.** On Nexar T1 is met together with a lower span, so there it is not a repair. Nexar span zone AUC falls 0.887 → 0.813, the span's mean raw score 0.511 → 0.295, and Nexar onset AUC falls on both sides (−0.091 / −0.039). Min-max micro on DoTA-CAP-dev rises +0.0440 [+0.0343, +0.0535] while macro does not move: the pooled number improves without better ranking inside a clip (one reading, not measured: lower min-max level on the far frames, see the event-locked table). DAD micro +0.0329, TAD micro −0.0519 (both printed). **This is the first bottom-k number on a clean training corpus**, so the user's reason for running it was right to test; on this host, weight 1.0 and k = 4 it costs three sets.

**Flag, K5fbk − K5f (ED6): the starts do not match.** Step-1 `mil` 0.692545 vs 0.687794. The value is not random: Kcap, K5Lneg, K5Lnegu and K5c read exactly 0.6925445795, with the same `kim_mil` (0.864495) and step-2 `mil` (0.7198), and K0 / K5 / K5ln / K5f read 0.687794 and 0.7168. Two clusters, same data and config family; "two runtimes" is an inference, the cause was never measured (§11.6, pending (cq), torch and GPU lines not saved here either). **It likely affects the edge rows too** (same session), so every row − K5f above may carry it. It does not readily explain the mirror, whose two rows sit on opposite sides of K5f (K5fepre − K5fepost = +0.108 on DoTA pre, −0.114 on DoTA post, whatever K5f's start). **Its size is unknown: no pair of runs here differs by the cluster alone**, and n = 1 has no seed spread either. **Update after the sync:** step-1 `mil` K5fepre 0.6973, K5fepost 0.6913, K5fbk 0.6925 (plain pool), so the edge rows differ from the plain-pool row of the same notebook, which is stronger evidence that the forced windows were read; it assumes the three shared a runtime, which is not logged. **ED6's edge guard was weak:** an edge row whose step-1 `mil` differs from K5f's could differ because of the runtime alone, so "differs" does not prove the forced windows were read. That they were is supported by the mirror and, if the log is kept, by its "449 forced normals" line (not seen here).

**Expectations.** K5fepre: wrong on both targets (T1 predicted hit, T2 predicted not; the opposite happened), right on a possible Nexar cost, right that DADA is not a COST. K5fepost: no sign was predicted; it hurts DAD and raises Nexar false alarms (peak > 0.5: 0.549 → 0.708). K5fbk: **wrong** ("no difference on any set").

**Licensed:** the table rows and the verdicts; "K5fepre reaches T2 and costs Nexar val at s2024". **Never write:** that edge windows reduce the early rise, or fix DoTA; that K5fbk anticipates less, or reduces early alarms; "bottom-k fails" (one weight, one k, one seed, one unmatched start); anything from the micro gains. Closed by ED12: no other dose, gap, weight or k, and no pre + post combination.

### 49.2 Where K5fbk loses (post-hoc, printed, decides nothing; s2024, n = 1; 2026-11-24)

The user read the K5fbk figures (§49.1) as "rises at the right time and falls well, only the peak is a little lower". k = 20 // 5 = 4 rows, weight 1.0. Per video on the saved scores, two-class videos, means; K5fbk − K5f with a video bootstrap (1,000; DoTA not clustered by source here).

| read | Nexar val | DoTA-CAP-dev | DADA | DAD |
|---|---|---|---|---|
| AUC of the span's **first third** vs all normal rows | 0.871 → 0.753 (**−0.118**) | 0.827 → 0.787 (**−0.040**) | 0.750 → 0.680 (**−0.070**) | not read (span too short) |
| ... last third | −0.032 | +0.020 | +0.001 n.s. | not read |
| span vs [−2, 0) s before | −0.091 | +0.046 | −0.025 | −0.090 |
| span vs (0, 2] s after | −0.039 | −0.055 | −0.051 | no post zone |
| raw max inside the span | 0.805 → 0.752 | 0.686 → 0.691 n.s. | 0.621 → 0.572 | 0.389 → 0.323 |
| raw mean inside the span | 0.511 → 0.295 | 0.565 → 0.446 | 0.486 → 0.321 | 0.369 → 0.294 |
| min-max level inside the span | 0.549 → 0.315 | 0.682 → 0.516 | 0.592 → 0.392 | 0.744 → 0.583 |

The per-video peak does not move on DoTA (n.s.) and falls 0.05 to 0.07 elsewhere (Nexar val, DADA, DAD); the mean level inside the span falls on every set. So a K5fbk curve is **narrower**: the start of the accident scores low, which is what shows as a clean onset in a min-max picture. A lower mean peak in the onset-aligned figure is a mean over curves that peak at different times, not a lower peak. **On DADA the picture misleads:** the pre-onset level looks lower, but the span ranks *worse* against the 2 s before it (−0.025). Never "bottom-k fixes the early rise" from the min-max picture; the DoTA onset gain (+0.046) is real but it comes with a lower first third of the span on Nexar val, DoTA and DADA. Why (the loss on k rows at 1.0 falls to 0.002 and yet moves the whole curve) is not established.

## 50. K5fbk25 (bottom-k weight 0.25) against a same-notebook K5f (K5frr) (dated note 2026-11-24, before any number of these rows; user)

**Why.** ED12 closed bottom-k's weight and k; **the user reopens it with one value**, weight **0.25** (k stays 4: `bottomk_topk_pct=5`), and asks for K5f re-trained in the same notebook. §49.2: K5fbk (weight 1.0) loses the first third of the span (Nexar −0.118, DoTA −0.040, DADA −0.070) and gains the DoTA onset (+0.046). K5fbk − K5f had unmatched starts (0.6925 vs 0.6878, §49.1), so the control is **not** the old K5f. **Post-hoc:** both targets below were chosen after §49.2 on the sets they are read on; 0.25 was named as an example, not derived or swept.

| # | Choice |
|---|---|
| BQ1 Rows | **K5frr** = K5f's exact command (config diff to K5f {}); **K5fbk25** = K5frr + `loss.bottomk_weight=0.25`, `loss.bottomk_topk_pct=5` (those two keys). s2024, 33 epochs, 1,749 steps, `Nexar_window`. Trained in one notebook; GPU name, torch and CUDA version saved beside each run. **K5f stays the project reference**; the control of every contrast here is K5frr |
| BQ2 Start gate | Step-1 `mil` of K5fbk25 = K5frr's within `KIM_START_TOL`. Not equal → the report writes "UNMATCHED START" instead of a verdict. K5frr vs the old K5f printed |
| BQ3 Guards | J10 (1,749), the config diffs, last-epoch `mul_mil` > 0.01 |
| BQ4 Runtime read | K5frr − K5f on all five sets, printed, decides nothing. It is **one** observation of the re-run spread: it says whether a difference of the size of K5fepre's Nexar −0.0149 is inside that spread, and no more |
| BQ5 Targets | **A** first third of the span (`np.array_split` of the abnormal rows in three; at least 3 rows) against **all** normal rows, per video, K5fbk25 − K5frr, on Nexar val and DADA: the interval must not lie wholly below 0 on either set. **B** DoTA-CAP-dev onset AUC against the [−2, 0) s before (§49's): interval above 0. Intervals as §49 (videos on Nexar; source clusters on DoTA / DADA) |
| BQ6 Cost | §34's label (margin 0.01 and the interval past 0) on Nexar val crops, DoTA-CAP-dev, DADA, DAD against K5frr |
| BQ7 Verdict | A and B and no COST → **HELD**, s2025 may run. B, with A failed or a COST → **TRADE-OFF, closed**. Otherwise **closed**. Nothing is adopted at n = 1 |
| BQ8 After | Closed = no other weight, no other k, no bottom-k restricted to part of the window, no combination with edge windows |

**A is lenient** ("not wholly below 0" is not non-inferiority); the report prints, beside it, whether the lower bound is above −0.01. **Expectation (a wrong one changes nothing):** TRADE-OFF; weight 0.25 sits between K5frr and K5fbk, so B is met and A fails on Nexar.

**Not licensed, whatever the outcome.** "Bottom-k fixes the early rise"; any read of K5fbk (weight 1.0) against K5frr, or of K5fbk25 against the old K5f; a runtime conclusion from one re-run.

Code: `core/tools/k5_bottomk_report.py`; tests `core/tests/test_k5_bottomk_report.py`; runbook `colab/final/k5_bottomk25.ipynb`; outputs `Thesis-Final/outputs/kim_bottomk25/`.

### 50.1 Read-out (s2024, n = 1): TRADE-OFF, closed; K5f stays (2026-11-24)

Guards pass: step-1 `mil` 0.687794 in both rows (same draws), same GPU / torch / CUDA (NVIDIA L4, 2.14.1+cu130), config diff exactly the two bottom-k keys, J10, last-epoch `mul_mil` 0.0963 / 0.0932. Control K5frr.

| set | macro K5frr → K5fbk25 | Δ macro | label | Δ micro |
|---|---|---|---|---|
| Nexar val crops | 0.8942 → 0.8679 | −0.0263 [−0.0324, −0.0206] | **COST** | not read |
| DoTA-CAP-dev | 0.8188 → 0.8268 | +0.0080 [+0.0041, +0.0121] | above 0, under the 0.01 margin | +0.0362 [+0.0312, +0.0414] (min-max) |
| DADA | 0.7744 → 0.7624 | −0.0120 [−0.0170, −0.0069] | **COST** | +0.0032 n.s. (min-max) |
| TAD (printed) | 0.6981 → 0.7073 | +0.0092 n.s. | – | −0.0150 below (raw) |
| DAD | 0.8070 → 0.8063 | −0.0007 n.s. | no difference | +0.0232 above (raw) |

Target **A fails**: first third of the span Nexar −0.0482 [−0.0595, −0.0373], DADA −0.0301 [−0.0385, −0.0219] (strict bound fails too; DoTA, printed, −0.0087). Target **B holds**: DoTA onset AUC, pre side, +0.0454 [+0.0375, +0.0541]. Two COSTs. As expected in §50. BQ8 applies: no other weight, no other k, no partial bottom-k, no combination with edge windows.

**The re-run is bit-identical to K5f.** K5frr's `metrics.jsonl` equals K5f's byte for byte over 1,749 steps, and the five sets' macros and the TAD / DAD scores are equal to the last digit. The notebook trains it from scratch (its skip reads only the new folder, no path copies K5f; its `config.yaml` carries `normal_caption`, a key K5f's does not have). So training is deterministic where K5frr and K5f ran, and no code change since K5f moved it. It does **not** give a runtime spread: a spread of 0 in one environment says nothing about another, and K5f's GPU was not saved. BQ4 is therefore unanswered. K5fbk's 0.6925 start came from something other than the code; the GPU difference stays a suspicion. K5fepre's −0.0149 stays unplaced.

**Post-hoc, printed, decides nothing** (per video on the saved scores; video bootstrap; DoTA not clustered; `scratchpad` script, not a repo tool):

| read (K5frr → K5fbk25) | Nexar val | DoTA-CAP-dev | DADA |
|---|---|---|---|
| raw max inside the span | 0.805 → 0.802 n.s. | 0.686 → 0.681 | 0.621 → 0.611 |
| width of the min-max curve above 0.5 (s) | 2.51 → 1.94 | 4.31 → 3.19 | 3.76 → 2.90 |
| first crossing of 0.5, relative to onset (s; from −5 s) | +0.51 → +0.79 | −1.03 → −0.18 | −0.28 → +0.29 |
| last crossing of 0.5, relative to the span's end (s) | +1.90 → +1.75 n.s. | +0.64 → +0.36 | +0.76 → +0.62 |
| span vs [−2, 0) s before | −0.027 | +0.045 | +0.008 n.s. |
| span vs (0, 2] s after | −0.027 | −0.027 | −0.026 |
| last third of the span vs all normal | −0.003 | +0.024 | +0.003 n.s. |
| macro gained by dropping the [−2, 0) normals | +0.0016 → +0.0018 | +0.0192 → +0.0021 | −0.0018 → −0.0069 n.s. both |

The curve is **narrower with the same peak**, not shifted: delaying K5frr by 0.25 to 2 s matches K5fbk25 worse than no delay on every set (MAE of min-max curves). The same narrowing helps where K5frr rose early (DoTA, first crossing 1.0 s before onset) and hurts where it did not (Nexar val, already 0.5 s after the alert). On DADA the early rise is visibly smaller, but the span's ranking against [−2, 0) does not separate. Nexar val false alarms at 0.5: 0.549 → 0.575.

**Not licensed.** "Bottom-k fixes the early rise" (DoTA only, with a Nexar and DADA cost); "bottom-k 0.25 is adopted / better on DoTA" (below the margin, closed); any dose trend from K5fbk (unmatched start); "re-run spread = 0" beyond the same environment.

## 51. K5dfibk25 (bottom-k weight 0.25 on K5dfi) against a same-notebook K5dfi (K5dfirr) (dated note 2026-11-24, before any number of these rows; user)

**Why.** BQ8 (§50) closed every further bottom-k arm, including combinations; **the user reopens it with one row**: bottom-k on K5dfi. Claude recommended against it, after these reads of rows that already exist (printed, post-hoc, n = 1): K5dfi − K5f looks much like K5fbk25 − K5f on DoTA (onset AUC +0.046 vs +0.045, first third −0.017 vs −0.009) and DADA (onset +0.007 vs +0.008, first third −0.032 vs −0.030); both cost Nexar val about 0.025; K5dfi also costs DAD −0.058, which is DVS's (§34.1). So there is little onset left to gain, and the first-third losses may add. Not measured: whether the two act on the same rows (on a spliced window bottom-k's lowest rows are mostly filler normals that `L_dvs-sup` already pushes down).

| # | Choice |
|---|---|
| BD1 Rows | **K5dfirr** = K5dfi's exact command (config diff to K5dfi {}); **K5dfibk25** = K5dfirr + `loss.bottomk_weight=0.25`, `loss.bottomk_topk_pct=5` (the two keys of §50; `k = n // 5`, so 4 on a plain window and larger on a spliced one). s2024, 33 epochs, 1,749 steps, `Nexar_window`, K5df's KNN cache. One session; GPU, torch, CUDA saved. **K5f stays the project reference**; the control of the deciding contrast is K5dfirr |
| BD2 Start gate | Step-1 `mil` of K5dfibk25 = K5dfirr's within `KIM_START_TOL`; otherwise "UNMATCHED START". K5dfirr vs the old K5dfi printed |
| BD3 Guards | J10 (1,749), config diffs, last-epoch `mul_mil` > 0.01, `dvs.theta` 0.7 / 0.85, `dvs_anchor_mode=ignore`, seam mask on |
| BD4 Targets | §50's A and B, on K5dfibk25 − K5dfirr: **A** first third not wholly below 0 on Nexar val and DADA; **B** DoTA-CAP-dev onset AUC above 0 |
| BD5 Cost | §34's label on Nexar val crops, DoTA-CAP-dev, DADA, DAD, against K5dfirr |
| BD6 Reference | The same label against **K5f**. A set that is a COST against either K5dfirr **or** K5f fails the arm. (Starts not matched: DVS changes the draws.) |
| BD7 Verdict | B fails → **closed**. B holds, and A fails or any COST (BD5 or BD6) → **TRADE-OFF, closed**. A, B and no COST → **HELD**, s2025 may run. Nothing is adopted at n = 1 |
| BD8 After | Closed = no other weight, k, host or combination with DVS, edge windows or captions. A further arm needs a new dated note naming what it tests |

**Expectation (a wrong one changes nothing):** TRADE-OFF or closed. K5dfi already fails BD6 on Nexar val (−0.0245) and DAD (−0.0576) and bottom-k is not a fix for either; HELD needs the pair to repair both, which is not expected.

**Not licensed, whatever the outcome.** "Bottom-k and DVS are complementary / redundant" (one row, no 2 × 2: K5fbk25 is on another control and start); "bottom-k fixes the early rise"; a comparison of K5dfibk25 with K5fbk25; a runtime conclusion.

Code: `core/tools/k5_bottomk_report.py --spec k5dfi` (§50's report re-run on the synced reads: identical); tests `core/tests/test_k5_bottomk_report.py`; runbook `colab/final/k5dfi_bottomk25.ipynb`; outputs `Thesis-Final/outputs/kim_dfi_bottomk25/`.

### 51.1 Read-out (s2024, n = 1): CLOSED (target B not met); K5f stays (2026-11-24)

Guards pass: step-1 `mil` 0.690560 in both rows (same draws), same GPU / torch / CUDA (Tesla T4, 2.14.1+cu130), K5dfirr's config diff to K5dfi {} and K5dfibk25's exactly the two bottom-k keys, J10, last-epoch `mul_mil` 0.1519 / 0.1510. Control K5dfirr.

| set | macro K5dfirr → K5dfibk25 | Δ vs K5dfirr | label | Δ vs K5f | label |
|---|---|---|---|---|---|
| Nexar val crops | 0.8698 → 0.8611 | −0.0087 [−0.0130, −0.0048] below | no difference (under 0.01) | −0.0332 [−0.0433, −0.0236] | **COST** |
| DoTA-CAP-dev | 0.8259 → 0.8252 | −0.0007 n.s. | no difference | +0.0064 n.s. | no difference |
| DADA | 0.7677 → 0.7632 | −0.0045 [−0.0070, −0.0018] below | no difference (under 0.01) | −0.0112 [−0.0206, −0.0018] | **COST** |
| DAD | 0.7494 → 0.7515 | +0.0021 n.s. | no difference | −0.0555 [−0.0760, −0.0357] | **COST** |
| TAD (printed) | 0.7198 → 0.7160 | −0.0038 n.s. | – | +0.0179 n.s. | – |

Micro Δ vs K5dfirr: DoTA +0.0077 [+0.0054, +0.0101] above; DADA −0.0003, TAD −0.0094, DAD +0.0013, none separated.

Target **B fails**: DoTA onset AUC, pre side, −0.0005 [−0.0036, +0.0025]. Target **A fails**: first third Nexar −0.0107 [−0.0173, −0.0045], DADA −0.0058 [−0.0099, −0.0018] (DADA passes the strict bound; DoTA +0.0017 n.s.). Verdict by BD7: **CLOSED**. BD8 applies: no other weight, k, host or combination. The Nexar and DAD COST vs K5f was already K5dfi's (−0.0245, −0.0576); the DADA one is new (K5dfi −0.0067 n.s.).

**Post-hoc, printed, decides nothing** (per video on the saved scores; video bootstrap; scratch script): bottom-k still narrows the curve on K5dfi, but less than on K5f: width of the min-max curve above 0.5 −0.28 s on DoTA (−1.12 on K5f), −0.25 on DADA (−0.87), −0.16 on Nexar (−0.57); first crossing of 0.5 relative to onset −0.39 → −0.25 s on DoTA (K5f host −1.03 → −0.18). The onset AUC vs [−2, 0) does not move on DoTA and falls on Nexar (−0.019) and DADA (−0.006). Nexar val false alarms at 0.5: 0.681 → 0.717. So the DoTA onset gain of K5fbk25 − K5frr (+0.045) is not seen on this host, where K5dfi's onset was already that good; "redundant" is not licensed (no 2 × 2, different start and GPU).

**Re-run.** K5dfirr's `metrics.jsonl` is byte-identical to the old K5dfi's (T4 here; §50's K5frr was identical to K5f on an L4). Deterministic again; it says nothing of a cross-GPU spread.

**Expectation check.** TRADE-OFF or closed: held (closed). "First-third losses may add": held in sign on Nexar and DADA, small.

**Not licensed.** "Bottom-k and DVS are redundant / complementary"; "bottom-k fixes the early rise"; any comparison of K5dfibk25 with K5fbk25 as rows.

## 52. DoTA **train** frames located in CAP-DATA: construction and pixel gate (dated note 2026-11-24; user)

**Why.** The user wants to train on DoTA's train split. The project recorded "DoTA's train pixels are gone" without a test: MM-AU Phase 0 queried only the 1,397 **val** clips (they alone have surviving CLIP rows). CAP-DATA also holds the train clips. This note fixes how they are found, packed and checked. **No model runs under this note; the training recipe, the windows and the reads get their own note before any number.**

**Order of events, stated honestly.** The matching below was developed on the 1,248 exact val pairs and its counts were read before this note was written. DT7 (the pixel gate) was written before any train frame was looked at.

| # | Choice |
|---|---|
| DT1 Annotation | `DoTA_annotations.zip` (4,677 clips; train = 3,275 = the zip minus `val_split.txt` = the ids of `metadata_train.json`). Official GitHub copy = the local copy (sha256 in the map) |
| DT2 Order | Inside three CAP id ranges the exact val pairs are in DoTA file-name order **as Windows Explorer sorts it** (case-blind, `-` ignored, `_` before digits): 0 breaks inside a range (case-blind ASCII order: 11). A = [800, 1509) and B = [1509, 6450) at 10 fps, C = [9000, 11800) at 30 fps |
| DT3 Matching | Per range, a maximum-weight monotone matching of CAP ids to **all** 4,677 DoTA names; a pair is allowed iff CAP's total frames = `n` (A, B) or `3n − 2` / `3n − 3` (C); weight 1 + exp(−start gap / 8 frames), so the annotated start ranks but never filters; exact val pairs are anchors. A clip found twice takes B, then A, then C |
| DT4 No val pixels | A CAP video that hit **any** val clip (exact or near, all retrieved candidates) is never given to a train clip. One CAP video serves one train clip |
| DT5 Clock | 10 fps: frame `d`. 30 fps: frame `3d − 1` clamped (the mode of the frozen val maps; within 2 CAP frames, 67 ms, of 305 / 310 of them) |
| DT6 Pack | `dota_cap_frames pack` unchanged (original CAP bytes, stored). A clip whose streamed CAP video has another frame count than annotated is dropped and counted; every other missing clip raises |
| DT7 Pixel gate | DoTA's boxes on the recovered frames, judged by eye by the user, blind: 30 real train clips per rate group, 6 val controls (pixels already verified by CLIP), 10 decoys (a neighbouring clip of the same video under this clip's boxes), seed 2024, shuffled. A row is wrong if at least one of its three frames has no participant under a box. **Valid** iff ≥ 8 / 10 decoys are flagged and ≤ 1 / 6 controls. Valid → a rate group with > 2 / 30 flagged rows is dropped from the corpus. 10 fps dropped → STOP. Not valid → redraw once with 5 frames per row and seed 2025 (dated note), then STOP |
| DT8 Name | The corpus is **DoTA-CAP-train (n/3275)**. A read on DoTA-CAP val by a model trained on it is **in-domain**, never zero-shot: 177 of 188 train YouTube videos are also val videos (3,246 / 3,275 clips). This is DoTA's official split; no train clip overlaps a val clip in time (1,033 end or start within 5 frames of one). Nexar val, DADA, TAD, DAD stay zero-shot |
| DT9 Held-out rule | "DoTA stays held out" ends for rows trained on this corpus only. Every earlier row and K5f keep DoTA as zero-shot; K5f stays the reference |

**Map (frozen, `core/splits/dota_train/DOTA_TRAIN_CAP.json`, ids sha1 `64d796b0…`): 3,023 / 3,275 clips (0.923)**: A 258, B 2,045, C 720; 186 YouTube videos; 307,874 DoTA frames, about 57 GB as CAP JPEGs (estimate from the val archive's bytes per frame).

| check | A | B | C | all |
|---|---|---|---|---|
| val pairs hidden from the anchors (2 folds), right / wrong | 116 / 2 | 846 / 5 | 246 / 2 | **1,208 / 9 of 1,248** (precision 0.993, recall 0.968) |
| train: night flag agrees (not used by DT3) | 0.985 | 0.975 | 0.961 | |
| train: span end within 15 frames (not used) | 0.864 | 0.774 | 0.769 | |

On the first, looser alignment the true val pairs read 0.96–0.98 on the night flag and 0.75–0.85 on the span end, shuffled pairs 0.77–0.88 and 0.40–0.62: the train pairs agree like true pairs.

**Known before training (counts from the annotation; decide nothing yet).** DoTA has no normal video and its median clip is 99 frames (33 rows at s3). On the first alignment's 2,721 clips: windows W = 20 / hop 8 give 5,724 positive and only **504** normal (353 clips hold both); W = 16 / 8: 6,487 / 1,172; W = 12 / 6: 8,479 / 3,004 (1,803 clips hold both). Cutting them uses the annotated span, as T2 and `Nexar_window` do. A cubic position probe scores 0.852 on DoTA-CAP-dev with no pixel: print the rulers beside every in-domain number.

**Not licensed.** "DoTA's train pixels are verified" before DT7; any zero-shot claim on DoTA for a row trained here; a comparison of this corpus with DoTA's published train split as if complete (252 clips are missing, not at random).

Code: `core/tools/dota_train_match.py`, `core/tools/dota_train_sheet.py`, `dota_cap_frames verify` (`--dota-clip-dir` optional); tests `core/tests/test_dota_train.py`; runbook `colab/final/dota_train_pack_frames.ipynb`; frames `Thesis-Final/data/DoTA_CAP_train_frames/`, reports `Thesis-Final/outputs/REPORTS/dota_train/`. `--cap-csv` = CAP's `cap_text_annotations.xls` (HF `JeffreyChou/MM-AU`, sha256 `1b5b1fa1…`), columns video, type, light, abnormal start, abnormal end, total frames.

### 52.1 Read-out of the first sheet and the redraw (dated note 2026-11-25, before the redraw is drawn; user agreed)

**Pack (DT6): pass.** 3,023 / 3,023 clips, 307,874 frames, 56.69 GB, 0 failed, 0 dropped.

**Pixel gate (DT7), first sheet (seed 2024, 3 frames): NOT VALID.** Decoys caught **7 / 10** (bar 8); controls flagged 0 / 6; real rows flagged 10 fps 2 / 30 (`gZbGLa253Ak_002191`, `nvVk7L3GrJQ_002650`), 30 fps 2 / 30 (`UOjQsfDXjO0_004067`, `gZbGLa253Ak_001152`). The two groups' `pass: true` in `box_gate.json` is not a verdict because the gate is not valid (`usable_groups` is empty). **Pixels of DoTA-CAP-train are not verified; nothing trains on it.** The flagged row numbers were not stored (tool gap, fixed) and the user no longer has them, so which 3 decoys were missed is unknown.

**Post-hoc, unblinded (Claude, with the key open; printed, changes no verdict).** Of the 60 real rows, 58 look right, including both flagged 30 fps rows (their first two frames follow the vehicle exactly; the last is the blurred impact). 10 fps #9 (boxes on trees) looks wrong and #27 (nothing visible under the boxes even brightened) probably is. Decoys #13, #54, #55, #64, #69 have a box that overlaps a vehicle or a rider by chance.

**Redraw, as DT7 fixed it: seed 2025, 5 frames per row, once, then STOP if not valid.** Every DT7 threshold is unchanged: valid iff >= 8 / 10 decoys and <= 1 / 6 controls; then a rate group with > 2 / 30 flagged rows is dropped; 10 fps dropped means STOP. DT7's "one of its three frames" becomes "one of its five frames". Two additions, neither lowers a bar:

| # | Addition | Why |
|---|---|---|
| DT10 | Every clip the first sheet showed (boxes or pixels, 76 rows) is excluded from the redraw | its answers have been read |
| DT11 | Judging text: **every box** of every frame must frame a participant; one wrong box makes the row wrong; a frame where nothing is visible under a box counts as wrong | DT7 did not say which box counts when a frame has several. Written **after** the first sheet's key was read; it makes the judge stricter on real rows too |

Not added: an exemption for blurred impact frames. It would clear the two flagged 30 fps rows, so it would relax the rule after its number was read.

**Expectation.** Decoys are easier to catch (5 frames, every box). If real rows are flagged at the first sheet's rate (2 / 30), each group exceeds 2 / 30 with probability about 1/3 (binomial, n = 30, p = 0.067: 0.32), and 5 frames can only add flags. A dropped group would therefore not by itself show a bad map; it would still be dropped.

Code: `dota_train_sheet sheet --frames --exclude-key`; `score` now stores `flagged` and `decoys_missed`. Runbook `colab/final/dota_train_box_redraw.ipynb` (no re-pack); outputs `REPORTS/dota_train/box_sheet_redraw/`, `box_gate_redraw.json`.

### 52.2 Read-out of the redraw: NOT VALID, line STOPPED (dated note 2026-11-25; judge: the user, n = 1 judge)

Seed 2025, 5 frames, 76 rows, none from the first sheet. **Decoys caught 6 / 10 (bar 8): the gate is NOT VALID, and DT7's STOP applies.** Controls flagged 0 / 6. Real rows flagged: 10 fps 3 / 30 (`0qfbmt4G8Rw_001843`, `fWJbp43k644_002781`, `fb5i7JlmJdg_000281`), 30 fps 5 / 30 (`JmwUVoOrgk8_001359`, `TNZv-NBcV5U_004857`, `JkYzYrJpSoQ_004398`, `6E2N9ld0eHg_000290`, `90gyBengKDs_001407`). Decoys missed: #4 `P5MjgFmqouA_005080`, #8 `xjbbH74SwTg_003057`, #27 `t_bCDlNOhJc_004165`, #40 `y1vGuUK0db4_000391`. Flagged rows: 3, 10, 35, 39, 46, 51, 53, 54, 57, 59, 61, 64, 71, 74. Both groups' `pass: false` is not a verdict either (the gate is not valid); had it been valid, both groups were above 2 / 30 and 10 fps dropped would also have been STOP.

**Consequence.** The pixels of DoTA-CAP-train are **not verified**; nothing trains on it; DoTA stays held out for every row. No third sheet, no re-score. The frozen map, the 3,023 zips and the reports stay on Drive as they are (deleting them is the user's decision). Reopening takes a new dated note written before any new number. *(Reopened by the user the same day: §55.)*

**What it does not say.** Not "the map is wrong": a judge who misses 4 of 10 known-wrong rows cannot decide. The gate measured itself as too weak, on both sheets.

**Post-hoc (printed, changes no verdict).** Both sheets, same judge: decoys 13 / 20 flagged (0.65, exact 95 % [0.41, 0.85]), real 12 / 120 (0.10 [0.05, 0.17]), controls 0 / 12 ([0, 0.26]). Real rows are flagged far less than decoys (Fisher p = 3e-7; redraw alone p = 0.003), so real rows do not behave like one-off-neighbour errors; real against controls is not separated (p = 0.60). If a wrong clip is flagged as often as a decoy and a right one never is, at most about 0.10 / 0.65 = 15 % of the real rows are wrong; this is a rough reading of two sheets, not an estimate of the map's error. The redraw caught fewer decoys than the first sheet (6 against 7), contrary to §52.1's expectation; the samples are too small to say why. The likely design weakness, not tested: a decoy is a neighbouring clip of the **same** video, so it shows the same kind of road with vehicles, and a box can land on one by chance (§52.1 saw this in five decoys).

**Post-hoc re-read by Claude of the redraw's pages (asked by the user; unblinded: the missed decoys and flagged ids were known; printed, changes no verdict, no re-score).** Pages from `colab/final/dota_train_box_redraw_result.ipynb`. All 10 decoys look wrong, the four missed ones included (#27 on empty road, #40's first box on empty road, #4 and #8 far larger than any vehicle). The 8 flagged real rows follow their vehicle or person through frames 1-4; in the last frame it leaves the picture or is blurred (#39, #57, #59, #64 a visible part; #35, #46, #54 blurred or dark; #61 the car is not clearly visible). DT11 says a part is enough, so most of these flags come from the rule's "only clips the edge" wording as Claude explained it, not from the map. **One real row looks wrong and was not flagged: #11 `V3--0ubJkNE_005461` (10 fps, CAP 005720): rider and bike boxes, no rider or bike.** So 59 / 60 real rows look right here and 58 / 60 on the first sheet: 117 / 120, 3 wrong (0.025, exact 95 % [0.005, 0.071]), all in the 10 fps group. One unblinded reader; the fingerprint agrees with the val cross-check (1,208 / 9) but does not replace the gate.

**Process facts.** (i) Judged with Claude's written restatement of the rule in front of the user, given before scoring; two of its three "SAI" sketches (a small box on a large vehicle's body, a large box mostly on road) are Claude's reading of DT11's "about where and how big", not DT11's own text. (ii) The run was slow because `choose_rows` opens all 3,023 zips to count frames. A notebook cell replaced the count by `verify.json`'s; checked locally, it reproduces the first sheet's 76 rows exactly, so it did not change the draw (the repo code was not edited).

## 53. KIM ablation on six arms: K5f, K5dfi, K5dfie, K5fbk25, K5dfibk25, K5fepre (dated note 2026-11-24, before any number of the KIM-off rows; user)

**Why.** The user asks which of the six arms benefits from KIM. Metrics in this order: **micro AUC (primary), macro AUC (secondary)**. No valid KIM-off twin exists: K0 trained before §31 (source-class labels), L0f / L0dfi also drop the motion input, so a gap to them is KIM + VideoMAE. Six KIM-off rows are trained; the KIM-on rows are the saved ones.

| rule | content |
|---|---|
| KA1 Rows | KIM-off row = its arm's exact command with `kim.enabled`, `kim.prototypes`, `kim.edge_mask`, `kim.seam_mask` all false and nothing else: same motion stream (V2-S), corpus, DVS, KNN cache, bottom-k, 33 epochs = 1,749 steps, s2024. K5f → **K0f**, K5dfi → **K0dfi**, K5dfie → **K0dfie**, K5fbk25 → **K0fbk25**, K5dfibk25 → **K0dfibk25**, K5fepre → **K0fepre** (corpus `Nexar_window_edgepre`). The factor is **all of KIM** (branch, scorer, prototypes, `kim_mil`, `kim_nkc`), not its differences: Kcap (§11) says the differences do not carry K5's gain |
| KA2 Guards (raise) | J10 on every row; config diff on → off exactly the true `kim.*` flags of the on row; last-epoch `mul_mil` > 0.01 on every row; one checkpoint per row over its five reads, none shared; same items and labels on every set; micro of the score file = micro of the read-out; K0fepre's step-1 `mil` ≠ K0f's (else the edge windows were not read) |
| KA3 Start gate | KIM is built under a private RNG and `W_out` starts at 0, so the off row's step-1 `mil` must equal its on row's within `KIM_START_TOL`. Where it does not (a second start value, 0.692545 for K5f, is on record with no measured cause, §49.1; K5fepre trained in that session), the notebook re-trains the **on** row's exact command in the same session (`<arm>rr`) and that re-run is the on side. If the re-run does not match either, the arm reads **UNMATCHED START** and is not decided. The choice looks at step 1 only, never at a score |
| KA4 Contrast | **KIM-on − KIM-off**, one arm at a time, on Nexar val, DoTA-CAP-dev, DADA, TAD, DAD. Micro: pooled AUC, DoTA and DADA after per-clip min-max, Nexar val (whole video), TAD and DAD raw; cluster bootstrap (2,000 draws, source video on DoTA, else the video), the same resamples for all rows. Macro: the read-outs' endpoints and bootstrap (Nexar = crop means) |
| KA5 Per-set label | GAIN: mean ≥ +0.01 and interval above 0. COST: mean ≤ −0.01 and interval below 0. Else NO DIFFERENCE. Same labels on micro and on macro |
| KA6 Arm verdict | On the deciding sets Nexar val, DoTA-CAP-dev, DADA, DAD (as §34; **TAD printed**: its micro measures video separation, oracle printed). **BENEFITS**: ≥ 1 GAIN, no COST. **HURT**: ≥ 1 COST, no GAIN. **MIXED**: both. **NO DIFFERENCE**: neither. The headline is the micro verdict; the macro verdict stands beside it, never averaged in. A micro BENEFITS whose macro is HURT or MIXED is written "micro only" |
| KA7 "Which arm benefits more" | Only from a difference of differences: (on − off)_arm − (on − off)_K5f, same resamples (micro) or same items (macro). "Arm X gains more from KIM than K5f on set S" needs an interval off 0. Two arms with different labels but an overlapping difference of differences are **not** different. The ranking by mean micro Δ over the deciding sets is printed, decides nothing |
| KA8 What it cannot say | n = 1 seed, 6 arms × 4 sets × 2 metrics = 48 labels: about two intervals would leave 0 by chance, so a single-set label is a lead, not a finding. Nothing is adopted or dropped; K5f stays the reference; `nexar_test` and the DoTA-CAP eval half are not read. Micro on DoTA / DADA mixes ranking inside a clip with the level across clips (§44); micro on Nexar val and DAD also holds video separation |

**Expectation (before any number).** KIM's gain was measured on DADA (K5 − K0 +0.048, pre-§31, DVS off; n = 2) and nearly vanished with DVS on (K5d − W0 +0.010), so the DVS-off arms should show the larger DADA Δ. No expectation is stated for micro. K0dfi may equal K5dfi within noise: that reads NO DIFFERENCE, not "KIM does not work".

Code: `core/tools/kim_ablation_report.py`; tests `core/tests/test_kim_ablation_report.py`; runbook `colab/final/kim_ablation.ipynb`; outputs `Thesis-Final/outputs/kim_ablation/<row>/`.

### 53.1 Read-out (2026-11-25; user's Colab run, s2024, n = 1; decides nothing, K5f stays)

**Guards pass** (the tool raises otherwise). **Start gate (KA3):** every arm is decided. K5f and K5fbk25 match their saved on rows (step-1 `mil` 0.687794); K5dfi, K5dfie, K5dfibk25 and K5fepre did not, so their on rows are the re-runs (`<arm>rr`), which match (0.689790 / 0.697346 / 0.689790 / 0.693477). All rows trained in this campaign ran on an L4.

**Verdicts (KA6).** Micro first; macro BENEFITS on all six arms.

| arm | micro verdict | Δ micro Nexar | DoTA | DADA | DAD | TAD (printed) |
|---|---|---|---|---|---|---|
| K5f | MIXED | +0.0569 | −0.0396 COST | +0.0100 n.s. | −0.0318 COST | +0.1235 |
| K5dfi | BENEFITS | +0.0330 | −0.0041 n.s. | +0.0229 | −0.0036 n.s. | +0.0671 |
| K5dfie | MIXED | +0.0490 | −0.0133 COST | +0.0256 | −0.0199 COST | +0.0994 |
| K5fbk25 | MIXED | +0.0670 | −0.0021 n.s. | +0.0258 | −0.0157 COST | +0.1202 |
| K5dfibk25 | BENEFITS | +0.0456 | +0.0037 n.s. | +0.0370 | −0.0075 n.s. | +0.0797 |
| K5fepre | MIXED | +0.0745 | −0.0462 COST | +0.0081 n.s. | −0.0373 COST | +0.1275 |

Unmarked = GAIN. Macro on − off is above 0 on Nexar and DADA for all six, on DoTA for all but K5f (+0.0087 [−0.0016, +0.0186]), on DAD for K5f, K5fbk25, K5dfibk25, K5fepre. Ranking by mean micro Δ over the four deciding sets (printed): K5dfibk25 > K5fbk25 > K5dfi > K5dfie > K5fepre > K5f. Full tables with intervals: `kim_ablation/REPORTS/kim_ablation_summary.md`.

**Post-hoc (user's question: DoTA + DADA only, micro first, then macro and AP; not in §53's rule; printed).** Mean of the two sets, on − off; cluster bootstrap, 2,000 draws, same resamples as the tool; the AP and pairwise numbers come from a scratch script built on the tool's loader (its micro, macro and AP cells equal the reports').

| arm | Δ micro | Δ macro | Δ AP |
|---|---|---|---|
| **K5dfibk25** | **+0.0204** [+0.0152, +0.0252] | +0.0313 [+0.0266, +0.0361] | +0.0218 [+0.0154, +0.0280] |
| **K5fbk25** | **+0.0118** [+0.0055, +0.0180] | +0.0297 [+0.0233, +0.0358] | +0.0150 [+0.0064, +0.0230] |
| K5dfi | +0.0094 [+0.0052, +0.0136] | +0.0249 [+0.0208, +0.0289] | +0.0085 [+0.0028, +0.0139] |
| K5dfie | +0.0061 [+0.0004, +0.0117] | +0.0247 [+0.0187, +0.0305] | +0.0080 [−0.0003, +0.0159] |
| K5f | −0.0148 [−0.0226, −0.0073] | +0.0195 [+0.0121, +0.0271] | −0.0109 [−0.0211, −0.0010] |
| K5fepre | −0.0190 [−0.0269, −0.0112] | +0.0254 [+0.0184, +0.0327] | −0.0145 [−0.0242, −0.0051] |

Top two = **K5dfibk25, K5fbk25**. Differences of differences, mean of the two sets: K5dfibk25 − K5fbk25 micro +0.0085 [+0.0028, +0.0140], AP +0.0068 [−0.0012, +0.0143], macro +0.0016 [−0.0042, +0.0073]; K5dfibk25 − K5dfi above 0 on all three; **K5fbk25 − K5dfi not separated** (micro +0.0024 [−0.0024, +0.0074]). Second place is a point-estimate rank.

**What the numbers are.**
1. On DoTA **micro**, KIM helps no arm (best +0.0037 n.s.; K5f, K5dfie, K5fepre COST) while DoTA macro rises on five of six: KIM improves the ranking inside a clip; on K5f, K5dfie and K5fepre the pooled micro falls anyway, which points at the level across clips (§44; inferred, not measured here). The two top arms' micro gain is DADA's.
2. The largest Δ sit on the **weakest KIM-off rows**: bottom-k lowers DADA micro without KIM (K0f 0.7450, K0fbk25 0.7325, K0dfibk25 0.7211) and KIM gives part of it back. Mean micro of DoTA and DADA: K5fbk25 0.7649, K5dfibk25 0.7646, K5dfi 0.7630, **K0f (no KIM, no bottom-k) 0.7600**.
3. "Largest contribution" is not "best model": DADA micro is highest on K5dfie (0.7647), DADA macro on K5dfie and K5f (0.7754, 0.7744).

**Expectation check.** "The DVS-off arms show the larger DADA Δ": not held on micro (K5f +0.0100 against K5dfi +0.0229) and level on macro (+0.0302 / +0.0287).

**Re-run spread (post-hoc, printed): the same command on another GPU moves a row by up to about 0.01 micro.** Re-run − saved:

| row | Nexar micro / macro | DoTA | DADA | TAD | DAD |
|---|---|---|---|---|---|
| K5dfi | −0.0016 / −0.0029 | +0.0044 / +0.0005 | +0.0012 / −0.0012 | −0.0030 / +0.0015 | −0.0015 / +0.0051 |
| K5dfie | −0.0021 / +0.0015 | −0.0012 / −0.0022 | +0.0049 / +0.0030 | −0.0072 / −0.0055 | +0.0000 / −0.0072 |
| K5dfibk25 | −0.0021 / −0.0026 | +0.0019 / −0.0021 | −0.0005 / −0.0006 | +0.0023 / −0.0004 | −0.0011 / +0.0023 |
| K5fepre | +0.0040 / +0.0078 | −0.0099 / +0.0040 | +0.0010 / +0.0043 | +0.0049 / +0.0028 | −0.0090 / +0.0186 |

Recorded environments: K5dfirr and K5dfibk25 of §51 on a **T4** start at 0.690560, the same commands here on an **L4** at 0.689790, with the same torch and CUDA. The GPU model is the recorded difference (other differences are not excluded); K5dfi's saved run has no `env.json` and starts at the T4 value. K5f's second value 0.692545 is not measured this way. No interval in this project contains this spread. Consequence for §49.1: K5fepre's Nexar macro reads −0.0070 against K5f with the matched-GPU re-run (saved −0.0149); §49.1's TRADE-OFF label is not re-opened by this, but do not quote its Nexar cost without this line.

**Not licensed.** "KIM helps / hurts on micro" without the set; "bottom-k + KIM is the best model"; "K5fbk25 benefits more than K5dfi"; a difference under about 0.01 micro between rows trained on different GPUs; "KIM hurts DAD" without macro beside it (K5f DAD macro +0.0559).

## 54. K5dfibk25 on whole DoTA-CAP (dated note 2026-11-25, before any number; user)

**Why.** The user wants the onset figure of K5dfibk25 against LaGoVAD on whole DoTA-CAP (1,129 clips), as drawn for K5dfi. K5dfibk25 has only its dev read (569). The eval half (`dota_cap_eval`, 560) is opened for this one row by the user's request, with `protocol_b_eval --final`; §35 / §36 / §37 opened it before for other rows.

| # | Choice |
|---|---|
| CB1 Row | `K5dfibk25rr` (§53's on row, L4, step 1,749), checkpoint read only. LaGoVAD (public checkpoint) and K5f: §35's whole-set reads, not re-scored |
| CB2 Set | whole DoTA-CAP = the row's dev read + the new eval read, pooled by `dota_cap_pool`. No eval-only number is printed or tabled (as §35 R2) |
| CB3 Output | `zs_curves` onset figure (DoTA 1,129, DADA 370; legend "LaGoVAD + KIM" / "LaGoVAD"; panel title "DoTA", the user's naming for the thesis) and `zs_arm_table` (control K5f). The table keeps the name DoTA-CAP |
| CB4 Decision | none. §51.1 closed K5dfibk25 and §53.1 adopts nothing; K5f stays whatever the numbers. The KIM-off twin K0dfibk25 is not read here, so the figure compares whole systems, not KIM alone |

**Expectation (before the number).** Whole-set macro within about 0.01 of the dev macro (0.8231), LaGoVAD's 0.6465 far below.

Runbook `colab/final/k5dfibk25_cap_eval.ipynb`; outputs `kim_cap_eval/K5dfibk25rr/`, `kim_cap_eval/REPORTS/`.

### 54.1 Read-out (2026-11-26, user's Colab run; s2024, n = 1; printed, decides nothing, K5f stays)

Raw `outputs/v_final/kim_cap_eval/{K5dfibk25rr/,REPORTS/zs_arm_table_cap_all_k5dfibk25_s2024.md}`; `OPENED.json` 2026-11-26T09:32Z (`dota_cap_eval`, row K5dfibk25rr, checkpoint `kim_ablation/K5dfibk25rr`). Pooled read 1,129 clips (dev + eval).

| whole DoTA-CAP (1,129, protocol B) | macro [95 %] | micro | AP |
|---|---|---|---|
| K5f (control) | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 |
| **K5dfibk25rr** | **0.8210 [0.8096, 0.8319]** | **0.7692** | **0.6070** |
| K5dfi (§35.1, printed) | 0.8240 [0.8136, 0.8342] | 0.7603 | 0.5899 |
| LaGoVAD (public ckpt) | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 |

Paired vs K5f: macro −0.0016 [−0.0074, +0.0042] not separated; **micro +0.0330 [+0.0275, +0.0388] above**. LaGoVAD − K5f macro −0.176, micro −0.116, below.

**Reading.** (1) Expectation held: whole-set macro 0.8210 is 0.002 below the dev macro 0.8231; LaGoVAD far below. (2) Against K5f the gain is micro only (level across clips, §44), macro level — the §53.1 DoTA pattern on the whole set. (3) K5dfibk25rr vs K5dfi is **not** a contrast here: no paired number was printed, the point gap is +0.009 micro / −0.003 macro, inside the GPU re-run spread of §53.1 (K5dfi saved on a T4, K5dfibk25rr on an L4), and §51.1 found bottom-k under the margin on dev. (4) The local copy of the onset figure (`kim/REPORTS/figures/train_nexar/dota_dada/k5dfibk25_lagovad_kim_vs_lagovad_s2024/`, 2026-11-25) is the **dev-only** draft (DoTA n = 569); the 1,129-clip figure this run drew is on Drive and not synced.

**Licensed:** "on whole DoTA-CAP, K5dfibk25 is level with K5f on macro (−0.002 n.s.) and above it on micro (+0.033 [+0.028, +0.039]), n = 1". **Not licensed:** "bottom-k improves on K5dfi"; any eval-only number (CB2); a change of reference.

## 55. K5dfiDoTA: K5dfi's recipe trained on the recovered DoTA train clips (dated note 2026-11-25, before any feature or number; user)

**Override.** §52's gate DT7 was NOT VALID twice and its rule said STOP (§52.2). **The user decided on 2026-11-25 to train anyway**, after the unblinded re-read of both sheets (117 of 120 real rows right, 3 wrong; not a gate). This note reopens the line; it does not change §52's verdicts. The pixels are **not verified by the pre-registered gate**: about 2.5 % [0.5, 7.1] of the rows looked wrong, and only the 3 seen ones are removed.

| # | Choice |
|---|---|
| DC1 Corpus | `DOTA_TRAIN_CAP.json` (3,023 clips) minus `V3--0ubJkNE_005461`, `gZbGLa253Ak_002191`, `nvVk7L3GrJQ_002650` = **3,020 clips**. DoTA annotation, span `[anomaly_start, anomaly_end)`; row j at stride 3 is abnormal iff `start <= 3j < end` |
| DC2 Windows (user) | **W = 12 rows (3.6 s), hop 6, stride 3**, `WINDOW_MIN_POSITIVE`, no cap, no padding: **12,937 windows = 9,386 positive + 3,551 normal**; 3,019 clips give a window, 2,051 both kinds. All normals come from accident clips (DoTA has no normal video) |
| DC3 Features | CLIP `no_center_crop` and V2-S squash at stride 3 from the packed frames; V2-S clip = 16 frames, step 1 (1.5 s at 10 fps, the geometry of the DoTA-CAP cache). **A2 statistics (CRN off) fitted on DoTA train (user, 2026-11-25, before any feature; the first draft said Nexar's)**: `build_v2_inputs fit-ids` on the 3,019 clips that give a window. The four zero-shot bakes K5dfi was read on are re-expressed with them by `restat` (exact for CRN off), as §42; Nexar val is baked at read time from the new statistics. Gate R2 first: restat of a Nexar-statistics bake of the train clips == the direct bake, max abs diff < 1e-4, CLIP columns identical |
| DC4 Recipe | **K5dfi's command (user, 2026-11-25; a first draft of this note named K5f, changed before any feature was extracted)**: KIM, DVS `theta` 0.7, `dvs_anchor_mode=ignore`, `seam_mask`, s2024. Config diff to K5dfi = exactly {`data.dataset` = DoTA_train, `data.frame_stride` = 3, `train.num_epochs` = 6}. 294 steps/epoch x 6 = **1,764 steps** (K5dfi 1,749). `mil_topk_pct` 5 gives k = 2 of 12 un-spliced rows (K5dfi 4 of 20); `mul_mil_topk_pct` 16 gives k = 1 of 12. KNN fillers = `knn_cache` on the plain CLIP s3 rows, as K5dfi's on s8. Fillers can come from the anchor's own clip, as for K5dfi; hop 6 of 12 means they can overlap the anchor by 6 rows (K5dfi: 12 of 20) |
| DC5 Guards | J10; the config diff above; last-epoch `mul_mil` > 0.01 (§31 F3); bake manifests (CRN off, V2-S, DoTA `train_ids_sha1` on the train bake and the four restated ones); gate R2; corpus counts = DC2 |
| DC6 Reads | Nexar val, DoTA-CAP-dev (protocol B, **in-domain**), DADA whole, TAD, DAD. The eval half of DoTA-CAP is not opened by this note |
| DC7 Table | `zs_arm_table` control K5dfi (macro + micro, four sets; K5f printed as a row), Nexar val K5dfiDoTA − K5dfi paired per video. Two rulers beside the DoTA row |
| DC8 Decision | none (n = 1). K5f stays the reference and the zero-shot model whatever the numbers |

**Rulers, computed before any feature (annotation only, 569 dev clips).** A cubic position probe fitted on the 3,020 train clips scores DoTA-CAP-dev macro **0.8545 [0.8386, 0.8695]** with no pixel; `t/N` 0.5573. K5dfi (zero-shot) reads 0.8259 [0.8119, 0.8398], K5f 0.8188. So an in-domain macro up to about 0.85 can be position alone; windows of 12 rows hide the clip position from the model at training, but nothing hides it at the whole-clip read. **566 of the 569 dev clips share their YouTube id with a train clip** (DoTA's official split is by clip): an in-domain gain can also be camera and scene.

**What the contrast is.** K5dfiDoTA − K5dfi changes the corpus (Nexar positives to DoTA) **and** the window (5.33 s of 20 rows to 3.6 s of 12), the row spacing (0.267 s to 0.3 s), the positive share, the input statistics (Nexar's to DoTA's, at train and at every read) and the pixels' checking. A difference is not attributable to "more data" or to DoTA. At the reads, DADA / TAD rows are 0.267 s and Nexar val 0.1 s, as for K5dfi.

**Expectation (before the number).** DoTA-CAP-dev macro above K5dfi's 0.8259, uncertain whether above the 0.8545 ruler; Nexar val below K5dfi's 0.8698 (zero-shot now, in-domain for K5dfi); DADA (K5dfi 0.7677) not predictable; TAD / DAD near K5dfi.

**Not licensed.** "Zero-shot on DoTA" for this row; "DoTA pixels verified"; "state of the art on DoTA"; "more data helps" from one contrast with several changes; any comparison with a method trained on DoTA's full train split (252 clips missing, not at random).

Code: `core/tools/dota_train_extract.py`, `core/tools/dota_train_build.py`, `nexar_extract.motion_rows(clip_step=)`; tests `core/tests/test_dota_train_corpus.py`; runbook `colab/final/k5dfidota.ipynb`; outputs `Thesis-Final/outputs/kim_dota_train/`, caches `clip/DoTA_train_s3_ncc`, `video/<V2-S>/DoTA_train_s3_squash`, `v2/A2_S/DoTA_train_s3_dota` and the four `*_dota` restats, corpus `Thesis-Final/data/DoTA_train_window`.

### 55.1 Read-out (2026-11-26, s2024, n = 1; decides nothing, K5f stays)

Guards: J10 (step 1,764 = last step of `metrics.jsonl`), config diff = DC4, last-epoch `mul_mil` 0.26–0.48 (> 0.01), gate R2 max |diff| 2.38e-07, corpus counts = DC2. NVIDIA L4. Raw `outputs/v_final/kim_dota_train/REPORTS/{k5dfidota_report,zs_arm_table_k5dfidota_s2024}.md`.

| set | K5dfi | K5dfiDoTA | K5f | K5dfiDoTA − K5dfi macro | micro |
|---|---|---|---|---|---|
| Nexar val (crops macro) | 0.8698 | 0.7363 | — | −0.1335 [−0.1635, −0.1037] | — |
| DoTA-CAP-dev (**in-domain** for K5dfiDoTA) | 0.8259 / 0.7613 | 0.8590 / 0.8209 | 0.8188 / 0.7353 | +0.0331 [+0.0212, +0.0451] | +0.0595 [+0.0463, +0.0730] |
| DADA-2000 | 0.7677 / 0.7591 | 0.7659 / 0.7522 | 0.7744 / 0.7551 | −0.0018 [−0.0207, +0.0173] n.s. | −0.0069 n.s. |
| TAD | 0.7198 / 0.6119 | 0.7579 / 0.6488 | 0.6981 / 0.6711 | +0.0381 [−0.0141, +0.0887] n.s. | +0.0369 n.s. |
| DAD | 0.7494 / 0.7091 | 0.9287 / 0.9109 | 0.8070 / 0.7072 | +0.1793 [+0.1452, +0.2149] | +0.2018 [+0.1708, +0.2335] |

(cells = macro / micro; DoTA / DADA micro after per-video min-max, TAD / DAD raw.)

**Reading.**
1. **DoTA-CAP-dev:** 0.8590 lies inside the no-pixel cubic position ruler's interval 0.8545 [0.8386, 0.8695], and 566 / 569 dev clips share a YouTube id with train. The in-domain gain is not separated from position + scene; no capability claim.
2. **DADA, the cleanest zero-shot read: no difference.** Training on DoTA does not transfer to another dashcam set here.
3. **TAD:** not separated (macro or micro).
4. **Nexar val −0.134:** zero-shot for this row, in-domain for K5dfi; a corpus switch, not a cost of DoTA.
5. **DAD +0.179 is mostly a time ramp (post-hoc, printed, not gated).** `dad_position_diag` with K5dfi / K5dfiDoTA in place of W0 / R4 (`REPORTS/dad_position_diag.md`): the same accident mask on the 301 negatives (no accident) scores **placebo 0.8916 [0.8717, 0.9100]** for K5dfiDoTA (K5dfi 0.6222, LaGoVAD 0.5134); gap macro − placebo **+0.0370** (K5dfi +0.1273). Mean score on DAD negatives by fifth of the video: K5dfiDoTA 0.011 / 0.040 / 0.064 / 0.178 / 0.337, last > first in 99 % of videos (K5dfi 0.021 → 0.161, 84 %) (`REPORTS/ramp_posthoc.py`). Same-position, accident rows: K5dfiDoTA − K5dfi **+0.0804 [+0.0272, +0.1323]** — separates accident videos from normal ones, does not localize; chosen after the numbers. On TAD's 38 normal videos K5dfiDoTA's level is higher (0.13–0.22 vs 0.05–0.09 per fifth).
6. **Mechanism not measured.** Candidate: DoTA's normals all come from clip starts, where the causal V2-S window is padded (first full window = row 7 on DAD), so "padded = normal" could be learned; the ramp continues past row 7, so this does not explain it all.

**Expectations (§55):** DoTA above K5dfi — right; above the 0.8545 ruler — no (inside its CI); Nexar below — right; DADA — flat; TAD / DAD near K5dfi — DAD **wrong** (+0.179, a ramp).

**Write:** "trained on DoTA: DADA no difference (−0.002 n.s.); DoTA in-domain 0.859 inside the position ruler's CI; DAD +0.179 mostly a time ramp (placebo 0.892)". **Never write:** "DoTA training improves DAD / TAD", "K5dfiDoTA localizes better", a DoTA number of this row beside a zero-shot one, "more data helps". Claude's proposal (not decided): close the DoTA-train line; another seed does not remove the YouTube-id overlap or the ramp.

## 56. Three arms on the §55 DoTA-train corpus: K5dfieDoTA, K5dfibk25DoTA, K5dfiebk25DoTA (dated note 2026-11-26, before any number of these rows; user)

**Why.** The user wants to see how the two §34 / §51 knobs behave when the host is trained on DoTA train (§55) instead of Nexar. The line is in-domain on DoTA and §55.1 showed a time ramp on DAD; these rows inherit both caveats. Claude had proposed closing the line (§55.1); the user chose to run these arms.

| # | Choice |
|---|---|
| DA1 Rows | Control **K5dfiDoTA** (§55, checkpoint reused, not retrained). **K5dfieDoTA** = + `data.is_egocentric=true` (θ_ego 0.85, ≤ 2 clips per splice: §34). **K5dfibk25DoTA** = + `loss.bottomk_weight=0.25`, `loss.bottomk_topk_pct=5` (§51's two keys; `k = n // 5` = 2 on a plain 12-row window, 4 on Nexar's 20). **K5dfiebk25DoTA** = both. Together a 2 × 2 (ego × bottom-k) with K5dfiDoTA at the origin. `constants.KIM_DOTA_TRAIN_ARMS` |
| DA2 Corpus / inputs | §55's exactly: `DoTA_train_window` (12,937 windows), the DoTA-statistics bake, the four restated zero-shot bakes, the same KNN cache; nothing re-extracted or re-fitted |
| DA3 Recipe | K5dfiDoTA's command + DA1's sets; s2024, 6 epochs, 1,764 steps. Guards: J10, config diff to K5dfiDoTA = exactly the keys of DA1 (bottomk_topk_pct shows because its default is 16), last-epoch `mul_mil` > 0.01, bake manifests (DoTA `train_ids_sha1`) |
| DA4 Reads | As §55 DC6: Nexar val, DoTA-CAP-dev (protocol B, **in-domain**), DADA whole, TAD, DAD. DoTA-CAP eval stays sealed |
| DA5 Table | `zs_arm_table` control K5dfiDoTA, rows the three arms + K5dfi (Nexar-trained, printed); Nexar val each arm − K5dfiDoTA paired per video. **DAD position diagnostics for every row** (`dad_position_diag --row`, control K5dfiDoTA, LaGoVAD printed): placebo and same-position accident rows beside every DAD number. Interaction (ego × bottom-k) is not tested; the 2 × 2 cells are printed |
| DA6 Decision | none (n = 1). K5f stays the reference and the zero-shot model whatever the numbers |

**Expectation (before the number; a wrong one changes nothing).** On Nexar, K5dfie − K5dfi showed no difference on any set (§34.1) and K5dfibk25 − K5dfirr was under the margin (§51.1), so: ego ≈ 0 on every set; bottom-k narrows the curve (§50.1) and may lower DoTA-CAP-dev micro slightly; DADA not separated for all three; whether bottom-k shrinks DAD's ramp (placebo 0.892) is not predicted.

**Not licensed.** Anything from §55.1's list; "ego / bottom-k helps on DoTA" as a zero-shot claim; reading a DAD macro without its placebo.

Runbook `colab/final/k5dfidota_arms.ipynb`; outputs `Thesis-Final/outputs/kim_dota_train/<row>/`, report `REPORTS/k5dfidota_arms_report.md`.

## 57. EDA of the cut DoTA train corpus `DoTA_train_window` (dated note 2026-11-26, user; dataset properties only)

**Why.** The user asked for insight into the corpus §55 / §56 train on. Mirror of §40 (Nexar) for DoTA: no model, nothing gates, no arm follows.

| # | Choice |
|---|---|
| DE1 Corpus | §55's 3,020 clips and 12,937 windows, **rebuilt in memory** from DoTA's annotation zip (sha256 = the frozen map's) and the frozen map by `dota_train_build.plan` (the planner that wrote the corpus; `build` now calls it). With `--data-dir` the rebuilt windows are gated against the corpus on disk (ids, labels, start / end, positive rows) |
| DE2 Read | **train only** (user: no dev): clips: length, span length and position, span reaching the end / starting at 0, ego / night / class · row-level time profile (relative bins, rows from the end / start) and no-pixel rulers (late `t/N`, hump; micro and within-clip macro) · windows: counts, abnormal rows per positive window, row profile, gap to the span, centre rulers, V2-S padded rows (rows < 5 at s3) and clip-end windows by class, positive rate by stratum, MIL k. Six figures |
| DE3 Status | Printed, never gating. Rulers are corpus priors, not model results |

Tool `core/tools/dota_eda.py` (train only; a first draft also profiled DoTA-CAP-dev, removed by the user), tests `core/tests/test_dota_eda.py` (8), runbook `colab/final/dota_eda.ipynb` (CPU, gate against Drive's corpus), constants `DOTA_EDA_*`.

### 57.1 Local read (2026-11-26, Claude; same annotation zip, **gate not run** — the Colab run adds it)

**Gate (2026-11-26, user's Colab run): PASSED** — rebuilt windows == the corpus on Drive, 12,937 windows checked; every number below reproduced in the synced `REPORTS/dota_eda_report.md`.

Raw `outputs/v_final/dota_eda/REPORTS/{dota_eda_report.{md,json},figures/}`.

| Item | Measured |
|---|---|
| Clips | 3,020 from 186 YouTube ids; median 9.9 s (33 rows, 3.3–30 s); abnormal share of a clip median 0.30; span start / end median 0.38 / 0.71 of the clip; turning 38 %, lateral 15 %, moving ahead 14 %, oncoming 11 %; ego-involved 61 %, night 12 % |
| Where the span sits | mid-clip: abnormal share per tenth of the clip 0.01 → 0.70 (0.5–0.6) → 0.09; span reaches the clip end in **10.5 %** of clips, starts at row 0 in 0 %. No-pixel row rulers: hump micro 0.807 / macro 0.832; late `t/N` 0.585 / 0.594 |
| Windows | 12,937 = 9,386 positive (**73 %**) + 3,551 normal; 944 of 3,019 clips (31 %) give **no** normal window (42 % have < 12 normal rows before the span); 1 clip too short (`r6_ZhT7rmhM_004291`), 2 clips without an abnormal row at s3 |
| Positive windows | abnormal rows median 6 of 12; ≤ 1 row 7.2 %, ≤ 2 15.2 %, whole window 12.9 %. Row profile inside a positive window rises 0.44 → 0.58 (abnormal rows sit later in the window) |
| Normal windows | 2,500 before the span, 1,044 after (**70 % before**); gap to the span median 3 rows, **42 % within 2 rows** (Nexar 6.3 %: DoTA's negatives are near-span, hard) |
| Position inside a clip | window centre within-clip AUC raw **0.760** (later windows are positive), hump 0.859; pooled raw 0.677, hump 0.790 |
| V2-S padding | rows < 5 see a clamped first frame: **49.3 % of normal windows hold padded rows vs 13.5 % of positive** (padded-rows AUC 0.321, i.e. 0.679 for "normal"); clip-end windows 4.2 % / 3.6 % (AUC 0.497) |
| Strata | positive rate ego 0.726 vs other 0.725; day 0.729, night 0.700; classes 0.64 (pedestrian) – 0.76 (leave_to_left) |
| MIL k (12 rows) | `L_MIL` k = 2 (pct 5), multi-class k = 1, §56 bottom-k k = 2 |

**Reading (printed).** Two corpus priors point the way §55.1's DAD ramp went: inside a training clip the positive windows sit later than the normal ones (within-clip raw 0.760), and half the normal windows carry the V2-S padded rows of the clip start against 13.5 % of the positives. A model can only see a window's 12 rows, but the padded rows are visible in its V2-S input. **This is the corpus side of the hypothesis, not a test of it:** whether K5dfiDoTA uses either cue needs a model read (e.g. its score on padded vs unpadded rows of DAD negatives, or a corpus whose normal windows avoid rows < 5).

**Licensed:** "In DoTA train, 70 % of the normal windows lie before the span and 49 % of them hold V2-S padded rows (positives 13.5 %)". **Not licensed:** "K5dfiDoTA learned the padding / late shortcut" (no model was read); "DoTA has hard negatives so it is a better corpus" (near-span negatives also hold label noise at the boundary).

## 58. Ablation on the §55 DoTA-train corpus: K5dfiDoTA, K0dfiDoTA (KIM off), L0dfiDoTA (LaGoVAD trunk) (dated note 2026-11-26, before any number of the new rows; user)

**Why.** The user wants KIM's contribution when the host trains on DoTA train, and where the trunk without motion lands. §53 measured KIM on − off on Nexar-trained hosts; §36 / §39 trained the LaGoVAD trunk on Nexar / ACCIDENT. Nothing has done both on DoTA train. Reads: **DoTA-CAP-dev (in-domain) and DADA (zero-shot) only** (user).

| # | Choice |
|---|---|
| AB1 Rows | **K5dfiDoTA** (§55, checkpoint and reads reused). **K0dfiDoTA** = K5dfiDoTA's command with `kim.enabled`, `kim.prototypes`, `kim.edge_mask`, `kim.seam_mask` false and nothing else (§53 KA1): same V2-S bake with DoTA statistics, corpus, DVS (θ 0.7, `ignore`), KNN, 1,764 steps, s2024. **L0dfiDoTA** = K0dfiDoTA + `v2.motion=none`, `model.motion_dim=0` (§36 R1), input = plain CLIP `DoTA_train_s3_ncc`, same KNN |
| AB2 Contrasts | **KIM** = K5dfiDoTA − K0dfiDoTA. **VideoMAE motion** = K0dfiDoTA − L0dfiDoTA. **Total** = K5dfiDoTA − L0dfiDoTA. The three add up exactly on the means; each interval is its own paired bootstrap. Two `zs_arm_table`s: control K0dfiDoTA (rows K5dfiDoTA, L0dfiDoTA) and control L0dfiDoTA (rows K0dfiDoTA, K5dfiDoTA) |
| AB3 Start gate | §53 KA3: K0dfiDoTA's step-1 `mil` must equal K5dfiDoTA's within `KIM_START_TOL`; otherwise **K5dfiDoTArr** (the exact command, same session) is the on side, and if that does not match either, KIM reads UNMATCHED START and is not labelled. L0dfiDoTA has no matched start (other model): printed only (§36 R4) |
| AB4 Guards (raise) | J10 on every new row; config diff K5dfiDoTA → K0dfiDoTA = exactly the four KIM flags; K0dfiDoTA → L0dfiDoTA = exactly {`v2.motion`, `model.motion_dim`}; last-epoch `mul_mil` > 0.01; `zs_lagovad_readout check-inputs`: plain `DoTA_train_s3_ncc` == the train bake's CLIP columns (3,019 sources), `DoTA_s1_ncc[::3]` == the restated DoTA-CAP-dev bake's (569), `DADA2000_orig` == the restated DADA bake's (370), so every row sees the same CLIP frames |
| AB5 Reads | DoTA-CAP-dev protocol B (K rows on `*_dota` bakes, L0 on plain `DoTA_s1_ncc` stride 3), DADA whole (K rows on `*_dota`, L0 on plain `DADA2000_orig`). Micro after per-clip min-max (§53 KA4), macro from the read-outs |
| AB6 Labels | §53 KA5 on each contrast, set and metric: GAIN mean ≥ +0.01 and interval above 0; COST mean ≤ −0.01 and interval below 0; else NO DIFFERENCE. Micro first, macro beside it. Rulers printed beside: DoTA-CAP-dev cubic position 0.8545 [0.8386, 0.8695] (§55), DADA middle ruler 0.7342 (§36.1) |
| AB7 Decision | none (n = 1). K5f stays the reference. DoTA-CAP-dev is in-domain for all three rows (566 / 569 dev clips share a YouTube id with train): a DoTA contrast measures what each component adds in-domain, never zero-shot |

**Expectation (before any number; a wrong one changes nothing).** KIM: on Nexar (§53.1, K5dfi) DoTA micro −0.004 n.s. and DADA micro +0.023; here DADA KIM ≈ +0.01…+0.03, DoTA in-domain NO DIFFERENCE. VideoMAE: on Nexar K5dfi − L0dfi was +0.147 / +0.193 macro on DoTA-CAP / DADA (motion and KIM together); here K0dfiDoTA − L0dfiDoTA GAIN on DADA (≥ +0.10), smaller on DoTA in-domain (the position ruler already reaches 0.8545, so L0dfiDoTA may sit near it). L0dfiDoTA on DADA near or below the middle ruler (L0dfi 0.575).

**Not licensed.** "KIM gives +x on DoTA" as zero-shot; a contrast to a Nexar-trained row as KIM's effect (corpus changes too); "VideoMAE alone" from K5 − L0 (that is the total); anything from §55.1's list.

Runbook `colab/final/k5dfidota_ablation.ipynb`; outputs `Thesis-Final/outputs/kim_dota_train/{K0dfiDoTA,L0dfiDoTA[,K5dfiDoTArr]}/`, report `REPORTS/k5dfidota_ablation_report.md`; constants `KIM_DOTA_TRAIN_OFF_ROW`, `KIM_DOTA_TRAIN_L0_ROW`, `KIM_DOTA_TRAIN_ABL_*`.

### 58.1 Read-out (2026-11-26, user's Colab run, NVIDIA L4, s2024, n = 1; printed, decides nothing, K5f stays)

Raw `outputs/v_final/kim_dota_train/REPORTS/{k5dfidota_ablation_report.md,ablation/}`. **Guards (checked again locally on the synced files):** J10 = 1,764 on all three rows; config diff K5dfiDoTA → K0dfiDoTA = exactly the four `kim.*` flags, K0dfiDoTA → L0dfiDoTA = exactly {`v2.motion`, `model.motion_dim`}; last-epoch `mul_mil` 0.361 / 0.360 / 0.475 (> 0.01). **Start gate (AB3): matched** — step-1 `mil` 0.731826 on both K rows, so the saved K5dfiDoTA is the on side (no `K5dfiDoTArr`). L0dfiDoTA starts at 0.780400 (other model, printed).

| row | DoTA-CAP-dev (in-domain) macro [95 %] / micro / AP | DADA (zero-shot) macro [95 %] / micro / AP |
|---|---|---|
| K5dfiDoTA | 0.8590 [0.8468, 0.8709] / 0.8209 / 0.6776 | 0.7659 [0.7469, 0.7850] / 0.7522 / 0.5715 |
| K0dfiDoTA | 0.8532 [0.8409, 0.8651] / 0.8156 / 0.6817 | 0.7675 [0.7488, 0.7866] / 0.7553 / 0.5753 |
| L0dfiDoTA | 0.6682 [0.6536, 0.6826] / 0.6412 / 0.5010 | 0.5911 [0.5763, 0.6063] / 0.5849 / 0.3892 |
| *position only* | *cubic ruler 0.8545 [0.8386, 0.8695]* | *middle ruler 0.7342 [0.7076, 0.7594]* |

| component (AB2) | set | Δ micro | Δ macro | label (KA5) |
|---|---|---|---|---|
| KIM = K5 − K0 | DoTA-CAP-dev | +0.0053 [+0.0007, +0.0098] | +0.0058 [+0.0015, +0.0101] | NO DIFFERENCE (interval above 0, under the +0.01 margin) |
| KIM | DADA | −0.0031 [−0.0098, +0.0034] | −0.0016 [−0.0083, +0.0050] | NO DIFFERENCE |
| VideoMAE = K0 − L0 | DoTA-CAP-dev | +0.1745 [+0.1627, +0.1864] | +0.1850 [+0.1720, +0.1980] | GAIN |
| VideoMAE | DADA | +0.1704 [+0.1524, +0.1878] | +0.1764 [+0.1585, +0.1944] | GAIN |
| total = K5 − L0 | DoTA-CAP-dev | +0.1797 | +0.1908 | GAIN |
| total | DADA | +0.1674 | +0.1748 | GAIN |

**Reading.**
1. **On the DoTA-trained host, KIM adds nothing measurable.** DADA (the zero-shot read) −0.002 macro / −0.003 micro, not separated; DoTA in-domain +0.006, its interval clears 0 but not the margin. The total is VideoMAE's.
2. **VideoMAE V2-S carries the whole gain at fixed data and recipe:** +0.17…+0.19 on both sets, both metrics. On DADA it lifts the trunk from **below** the middle ruler (0.591 < 0.734) to ~0.03 above it (0.768).
3. **DoTA in-domain, read against position:** L0dfiDoTA (0.668) sits far below the no-pixel cubic ruler (0.8545); K0 / K5 (0.853 / 0.859) sit **inside** its interval. The motion rows reach what position alone reaches; this read cannot say whether they learn position, content or both (§55.1 ramp, §57 padding prior: the CLIP-only L0 has no padded motion rows, a later read can use that).
4. **Against the Nexar-trained host (other corpus, printed, not a contrast):** §53.1 measured KIM on K5dfi at DADA micro +0.0229 GAIN; here −0.0031. KIM's DADA gain does not reappear when the host trains on DoTA; the corpus, window (20 → 12 rows), row spacing and input statistics all change between the two, so no cause is licensed. §36.1's K5dfi − L0dfi on DADA (+0.193 macro, motion + KIM) and this K5dfiDoTA − L0dfiDoTA (+0.175) agree in size.

**Expectations (§58):** KIM on DADA +0.01…+0.03 — **wrong** (−0.002); KIM on DoTA NO DIFFERENCE — held; VideoMAE GAIN ≥ +0.10 on DADA — held (+0.176); L0dfiDoTA on DADA near or below the middle ruler — held (0.591, below).

**Licensed (n = 1):** "Trained on DoTA, the motion stream accounts for the whole gain over the LaGoVAD trunk (DADA +0.176 macro); KIM adds nothing separable on DADA (−0.002) or in-domain DoTA (+0.006, under the margin)." **Not licensed:** "KIM does not work" (one host, one seed; on the Nexar host it gains on DADA); "KIM only helps on Nexar" (the hosts differ in five ways); any DoTA number of these rows as zero-shot; "VideoMAE learns accidents on DoTA" without the position ruler beside it.

### 58.2 L0dfiDoTA on DAD + DAD placebo (2026-11-27, user's Colab run `colab/final/l0dfidota_dad_figures.ipynb`, s2024, n = 1; POST-HOC, printed, decides nothing, K5f stays)

Raw `outputs/v_final/kim_dota_train/L0dfiDoTA/eval/dad/K5/s2024/results.json` (466 videos, `score_norm` none) and `outputs/v_final/kim_dota_train/REPORTS/dad_position_l0dfidota/dad_position_diag.{md,json}` (control K5dfiDoTA). Same CLIP frames as K5dfiDoTA's DAD read (check-inputs, step 5). **DAD's no-pixel `t/N` ruler macro = 1.0000** (accident = rows 13–14 of 15 in every positive): DAD macro alone measures position, so it is read only beside the placebo (§56 DA5).

| row | DAD macro / micro / AP (raw) | placebo, all rows | gap beyond placebo, all rows | gap, rows 7+ | same position: accident rows (pos vs neg) |
|---|---|---|---|---|---|
| K5dfiDoTA | 0.9287 / 0.9109 / 0.3487 | 0.8916 [0.8717, 0.9100] | +0.0370 [+0.0121, +0.0621] | +0.0334 [−0.0085, +0.0736] | 0.7553 [0.7077, 0.8005] |
| L0dfiDoTA | 0.6697 / 0.6732 / 0.1283 | 0.6048 [0.5709, 0.6384] | +0.0649 [+0.0100, +0.1185] | +0.0384 [−0.0212, +0.0979] | 0.6341 [0.5793, 0.6881] |

Accident rows, paired (position cannot help): **L0dfiDoTA − K5dfiDoTA −0.1212 [−0.1768, −0.0645]**. Accident rows − rows 7 up to the accident: K5dfiDoTA +0.0388 [−0.0007, +0.0777], L0dfiDoTA −0.0106 [−0.0694, +0.0472]. Median video AUC (`zs_curves`, global raw): DAD 0.947 / 0.731, DADA 0.789 / 0.587, DoTA-CAP-dev (in-domain) 0.865 / 0.665 (K5dfiDoTA / L0dfiDoTA).

**Reading.**
1. **The +0.259 DAD macro gap between the two rows is mostly ramp.** K5dfiDoTA's placebo is 0.892, L0dfiDoTA's 0.605; beyond its own placebo each row gains only +0.04 / +0.06 (all rows), and neither interval clears 0 on rows 7+. Within a video, neither row is shown to point at the accident beyond the time ramp.
2. **Across videos at the same rows, the DoTA-trained motion host separates better:** accident rows of positives vs the same rows of negatives 0.755 vs 0.634, paired +0.121 [+0.065, +0.177]. This is the DAD read position cannot reach. It is motion + KIM together (no K0dfiDoTA DAD read; on DADA §58.1 put the whole gain on VideoMAE).
3. L0dfiDoTA's same-position AUC (0.634) sits between the released LaGoVAD (0.588) and the Nexar-trained K5dfi (0.675) of the §56 placebo table — other runs, printed beside, not a contrast.

**Licensed (n = 1):** "On DAD, trained on DoTA, the motion host ranks accident rows of positive videos above the same rows of negative videos better than the LaGoVAD trunk (+0.121 [+0.065, +0.177])." **Not licensed:** any DAD macro difference without the placebo ("+0.26 on DAD"); "KIM helps on DAD" (no KIM-off DAD read); "either row localizes the accident within a DAD video" (rows 7+ gaps cross 0); "DoTA training improves DAD".

**Figures:** `outputs/v_final/kim/REPORTS/figures/train_dota/dota_dada_dad/k5dfidota_vs_l0dfidota_s2024/` (onset, grids DoTA / DADA / DAD, two models). The notebook in the tree names the tag `k5dfidota_vs_l0dfidota_vs_lagovad_s2024` and adds the released checkpoint to the onset; the synced folder has neither, so the Drive copy that ran was older (pending: re-run step 7 only if the three-line onset is wanted).

## 59. K5dfice: KIM reads its own CLIP-only temporal encoder, on K5dfi; capacity control K5dficx (dated note 2026-11-28, before any number of these rows; user)

**Why.** LAS-VAD's IAM differences X_f, i.e. CLIP features after its own temporal modelling (local Transformer + GCN), CLIP only. KIM differences `v^t` = RoFormer(`x + W_u u`), so its differences already contain the VideoMAE stream. The user wants KIM's differences taken on CLIP after temporal modelling, while the fused path (CLIP + VideoMAE → RoFormer → fusion) stays as it is. This needs a **second temporal encoder** that reads CLIP only and feeds KIM alone. Host = **K5dfi** (user). Prior read (printed, training-free, §5 KP1 on K0): `‖Δ² clip‖` − `‖Δ² vt‖` macro +0.013 on DoTA-CAP-dev, +0.005 on DADA, +0.002 on TAD, all under KP1's 0.02. This arm is **not** R7 / K7 (R7 differences raw `c_t`); it is a new forking path, declared here.

| # | Choice |
|---|---|
| CE1 Rows | **K5dfice** = K5dfi + `kim.input_encoder=clip`: a `TemporalEncoder` with the host's configuration (2 layers, 4 heads, window 25, same max positions, same `temp_gate`) reads the CLIP columns of each row (`rows[..., :512]`, = `c_t` under CRN off) and its output `h^c_t` replaces `v^t` as KIM's input. KIM still returns onto `v^t`: `ṽ^t = v^t + W_out k(h^c)`. **K5dficx** (capacity control) = K5dfi + `kim.input_encoder=fused`: the same second encoder reads the fused rows `x_t + W_u u_t` (the main encoder's input). Both: s2024, 33 epochs, 1,749 steps, `Nexar_window`, K5df's KNN cache, DVS θ 0.7 / θ_ego 0.85, `ignore`, seam mask, edge mask, prototypes. Config diff to K5dfi = exactly {`kim.input_encoder`} |
| CE2 Init and RNG | The second encoder is built under KIM's private RNG (D13), **after** KIM's own layers, so the host's init, KIM's init and the data order equal K5dfi's. Its dropout draws come from a private stream (its own generator state, swapped in around its forward, training mode only; seed `train.seed + KIM_ENCODER_RNG_OFFSET`), so every host draw equals K5dfi's. With `W_out = 0` at init, step 1's forward equals K5dfi's. The private stream is not checkpointed (a resumed run restarts it: these runs must not resume) |
| CE3 Control | **K5dfirr** (§51, K5dfi's exact command, trained on Tesla T4, byte-identical to K5dfi). If this session's GPU is a Tesla T4, K5dfirr's checkpoint and reads are reused from `kim_dfi_bottomk25/`; otherwise K5dfirr is trained again in this notebook (user, 2026-11-28). The GPU, torch and CUDA of every row are saved and printed |
| CE4 Start gate | Step-1 `mil` of K5dfice and of K5dficx = K5dfirr's within `KIM_START_TOL`. A row that misses it reads UNMATCHED START and is not labelled |
| CE5 Trains check | Logged per optimizer step: `kim_enc_grad_norm` (unscaled) and `kim_enc_drift` = ‖θ − θ₀‖ / ‖θ₀‖ of the second encoder. **DID NOT TRAIN** if the last-epoch mean grad norm is 0 or not finite, or the final drift is below `KIM_ENC_MIN_DRIFT` (1e-3); such a row is not labelled (an untrained encoder makes "no difference" meaningless, cf. STP's prefix, `STP_BUILD.md` §5.1). Drift of the main encoder is not logged (no comparison is made) |
| CE6 Guards (raise) | J10 (1,749), config diffs as CE1, last-epoch `mul_mil` > 0.01 (§31), `dvs.theta` 0.7 / 0.85, `ignore`, seam mask on, the new rows' state dicts = K5dfi's keys + `kim.encoder.*` only |
| CE7 Contrasts | **P** (module) = K5dfice − K5dfirr. **C** (input, beyond capacity) = K5dfice − K5dficx. **X** (the second encoder alone, printed) = K5dficx − K5dfirr. **R** (reference) = K5dfice − K5f, starts not matched (DVS changes the draws). Each paired per item as §51 (`k5_whole_report` readers); macro on Nexar val crops, DoTA-CAP-dev (protocol B), DADA, DAD decide, TAD printed; `zs_arm_table` prints macro **and** micro on the four zero-shot sets (control K5dfirr; K5dficx, K5dfi, K5f beside) |
| CE8 Labels | §34's label per set (GAIN mean ≥ +0.01 and interval above 0; COST mean ≤ −0.01 and interval below 0; else NO DIFFERENCE) |
| CE9 Verdict | **Module (P):** any COST → **COST (closed)**; else any GAIN → **HELD** (s2025 may run); else **NO DIFFERENCE (closed)**. **Attribution (C), only when P is HELD:** C GAIN on a set where P gains → "the CLIP-only input carries it there"; otherwise "not separated from the extra encoder". **Candidate for the main model:** HELD and R has no COST; otherwise K5f stays. Nothing is adopted at n = 1 |
| CE10 After | Closed = no other encoder size, depth, input mix, host or seed on this line without a new dated note naming what it tests |

**Expectation (before any number; a wrong one changes nothing).** P: NO DIFFERENCE (the training-free gap is +0.005…+0.013, near the 0.01 margin; trained heads tend to close such gaps). C: NO DIFFERENCE. R: COST on Nexar val and DAD, inherited from the host (K5dfi − K5f −0.0245 / −0.0576, §51 BD6), so "candidate" is not expected whatever P reads.

**Not licensed, whatever the outcome.** "KIM now matches LAS-VAD's IAM" (RoFormer in place of Transformer + GCN, a VideoMAE-fused return path, binary normal-anchored losses); "pure CLIP is the better input for KIM" unless C separates; "KIM works / does not work" from P alone (one host, n = 1); any K5f-hosted claim (K5dfi is the host); a runtime conclusion.

Code: `kim.input_encoder` (`core/config.py`, `core/kim/module.py`, `core/models/kat_vad.py`, logs in `core/train.py`); report `core/tools/k5_clip_encoder_report.py`; tests `core/tests/test_kim_input_encoder.py`, `core/tests/test_k5_clip_encoder_report.py`; runbook `colab/final/k5dfi_clip_encoder.ipynb`; outputs `Thesis-Final/outputs/kim_dfi_clip_encoder/`; constants `KIM_INPUT_ENCODER_*`, `KIM_CLIP_ENC_*`, `KIM_ENCODER_RNG_OFFSET`, `KIM_ENC_MIN_DRIFT`.

### 59.1 Read-out (2026-11-28, user's Colab run, Tesla T4 / torch 2.14.1+cu130 / CUDA 13.0 on all three rows, s2024, n = 1; K5f stays)

Raw `outputs/v_final/kim_dfi_clip_encoder/REPORTS/{k5dfi_clip_encoder_report.md,k5dfi_clip_encoder_summary.md,zs_arm_table_k5dfi_clip_encoder_s2024.md}`. **Control:** K5dfirr reused from `kim_dfi_bottomk25/` (CE3: env equal). **Guards (checked again locally on the synced files):** last metrics step 1,749 on all three rows; config diff to K5dfirr = exactly {`kim.input_encoder`} (`clip` / `fused`); last-epoch `mul_mil` 0.1519 / 0.1442 / 0.1455. **CE4 start: matched** (step-1 `mil` 0.690560 on all three). **CE5: both encoders trained** (last-epoch grad norm 4.52 / 5.50, final drift 0.098 / 0.114).

| set (macro) | K5dfirr | K5dfice | K5dficx | P = ce − rr | C = ce − cx | X = cx − rr | R = ce − K5f |
|---|---|---|---|---|---|---|---|
| Nexar val | 0.8698 | 0.8498 | 0.8588 | **−0.0199 [−0.0279, −0.0122] COST** | −0.0089 below (n.d.) | **−0.0110 COST** | **−0.0444 COST** |
| DoTA-CAP-dev | 0.8259 | 0.8202 | 0.8154 | −0.0057 below (n.d.) | +0.0049 above (n.d.) | **−0.0106 COST** | +0.0014 n.s. |
| DADA | 0.7677 | 0.7500 | 0.7494 | **−0.0177 [−0.0239, −0.0116] COST** | +0.0006 n.s. | **−0.0183 COST** | **−0.0244 COST** |
| DAD | 0.7494 | 0.7573 | 0.7786 | +0.0079 n.s. | **−0.0212 COST** | **+0.0291 GAIN** | **−0.0497 COST** |
| TAD (printed) | 0.7198 | 0.7149 | 0.7145 | −0.0049 n.s. | +0.0004 n.s. | −0.0053 n.s. | +0.0168 n.s. |

(n.d. = interval excludes 0 but |mean| < 0.01, NO DIFFERENCE by §34.)

* **Verdict (CE9): COST (closed).** P costs on Nexar val and DADA. C is not read (P not HELD); candidate = False (R COST on Nexar val, DADA, DAD). The expectation (P NO DIFFERENCE) was wrong in sign; that changes nothing.
* **Printed, decides nothing.** X (the second encoder fed the fused rows) costs on Nexar val, DoTA and DADA by about the same size as P, and C is not separated on DADA: **on DADA the cost goes with adding a second encoder, not with what it reads.** On Nexar val about half of P is X (−0.011) and half is C (−0.009, under the margin). The only set where the input separates beyond the margin is DAD, against the CLIP-only input (C −0.021).
* **Micro (printed, `zs_arm_table`):** K5dfice − K5dfirr DoTA-CAP-dev **+0.0174 [+0.0117, +0.0233] above**, DAD +0.0211 above, DADA −0.0127 below, TAD −0.0631 below (TAD micro = clip classification, oracle 0.9226). Nexar val negatives with peak > 0.5: 0.681 → 0.690 (K5dficx 0.735).
* **Not tested:** why a second encoder costs (capacity on 1,749 steps, the extra gradient path into KIM, a changed optimizer trajectory). Nothing here says which.

**Licensed:** "on K5dfi, s2024, n = 1, giving KIM its own temporal encoder costs Nexar val and DADA macro; on DADA the CLIP-only input is not separated from the fused input." **Not licensed:** "pure CLIP input hurts KIM" (C is not separated on DADA and not read by rule); "KIM does not work"; "LAS-VAD's IAM design does not transfer"; any K5f-hosted claim; "the second encoder improves DoTA" (macro −0.006; micro is printed).

## 60. KIM on − off on the T2-trained host: K5dfiT2 vs K0dfiT2 (dated note 2026-11-28, before any number of the new rows; user)

**Why.** The user asked how much KIM contributes to K5dfiT2 (§37) on DoTA micro, macro and AP. No KIM-off row trained on T2 exists: §53 measured KIM on − off on Nexar-trained hosts only, §58 on the DoTA-trained host. A2 of E3 (`v2_e3/A2`) is not a KIM-off control (it differs in DVS too, predates the §31 label fix, and has no whole DoTA-CAP read). Reads: **whole DoTA-CAP and DAD only; TAD is not read** (user).

| # | Choice |
|---|---|
| TK1 Rows | **K0dfiT2** = K5dfiT2's command (§37: K5T2's `--set` list + `dvs.theta=0.7`, `dvs.theta_ego=0.85`, `kim.seam_mask=true`, `loss.dvs_anchor_mode=ignore`) with `kim.enabled`, `kim.prototypes`, `kim.edge_mask`, `kim.seam_mask` false and nothing else (§53 KA1); same T2 corpus, A2 V2-S bake with T2 statistics, T2 KNN, 20 epochs = 1,740 steps, s2024. **K5dfiT2rr** = K5dfiT2's exact command, trained in the same session |
| TK2 Why the re-run | The saved K5dfiT2 has no `env.json`, so §59 CE3's reuse condition (same GPU) cannot be checked; §53.1 measured the same command moving by up to ~0.01 micro between T4 and L4. **The on side is always K5dfiT2rr.** The saved K5dfiT2 is read on DAD too and printed beside it as a reproduction check, never as the on side |
| TK3 Contrast | **KIM = K5dfiT2rr − K0dfiT2**, paired per item (`zs_arm_table`, control K0dfiT2). K5dfiT2 (saved) − K0dfiT2 printed only |
| TK4 Start gate | §53 KA3: step-1 `mil` of K0dfiT2 = K5dfiT2rr's within `KIM_START_TOL`; otherwise KIM reads UNMATCHED START and is not labelled. Step-1 of the saved K5dfiT2 (0.749036) printed |
| TK5 Guards (raise) | J10 = 1,740 on both new rows; config diff K5dfiT2 → K5dfiT2rr = {}; K5dfiT2rr → K0dfiT2 = exactly the four KIM flags (`kim_ablation_report.check_kim_only`); last-epoch `mul_mil` > 0.01; DoTA-CAP and DAD bakes: `crn` none, `motion` V2-S, `train_ids_sha1` = the T2 bake's; GPU / torch / CUDA saved per row (`env.json`) and printed |
| TK6 Reads | **Whole DoTA-CAP** (dev 569 + eval 560 = 1,129, protocol B, `DoTA_s1_ncc[::3]`): dev, then eval with `--final` under this note (marker `OPENED_s60.json` beside §37's `OPENED.json`), pooled by `dota_cap_pool`; micro / AP after per-clip min-max. **DAD** (466 test videos, s7, raw, the §13 bake `DAD_s7_t2`). DADA is not read (T2 is cut from it: in-domain, §13 T6); Nexar val and TAD are not read |
| TK7 Labels | §53 KA5 on each set and metric: GAIN mean ≥ +0.01 and interval above 0; COST mean ≤ −0.01 and interval below 0; else NO DIFFERENCE. Micro first, macro beside, AP printed. **DAD macro is read only beside the placebo** (§56 DA5; DAD's no-pixel `t/N` ruler = 1.0000): `dad_position_diag` (control K0dfiT2) prints placebo, gap beyond placebo and the same-position accident-row AUC with its paired contrast. DAD micro = clip classification (constant-per-clip oracle 0.8390): printed, never "localization" |
| TK8 Decision | **none** (n = 1). K5f stays the reference and the main model. A T2 row is not a candidate (§37 R5) |

**Expectation (before any number; a wrong one changes nothing).** On the Nexar host (§53.1, K5dfi) KIM on − off was DoTA micro −0.004 n.s., DAD micro −0.004 n.s., macro above 0 on DoTA and DADA; on the DoTA host (§58.1) NO DIFFERENCE everywhere. Here: DoTA-CAP micro and macro NO DIFFERENCE (|Δ| < 0.01); DAD macro NO DIFFERENCE beyond its placebo. K5dfiT2rr within ±0.005 of the saved K5dfiT2 on whole DoTA-CAP macro.

**Not licensed.** "KIM adds x on T2" without the set and metric; any contrast to a Nexar- or DoTA-trained row as KIM's effect (corpus, statistics and window change too); a DAD macro difference without the placebo; "KIM does not work" from n = 1.

Runbook `colab/final/k5dfit2_ablation.ipynb`; outputs `Thesis-Final/outputs/kim_t2_fixed/{K0dfiT2,K5dfiT2rr}/`, K5dfiT2's DAD read under `kim_t2_fixed/K5dfiT2/eval/dad/`, report `kim_t2_fixed/REPORTS/k5dfit2_ablation_report.md`; constants `KIM_T2_OFF_ROW`, `KIM_T2_RERUN_ROW`, `KIM_T2_ABL_*`.

### 60.1 Read-out (2026-11-28, s2024, n = 1; Tesla T4 both rows; decides nothing, K5f stays)

**Gates.** TK4 start matched (step-1 `mil` 0.749036 on = off = saved K5dfiT2). TK5: J10 1,740 both rows; config diff K5dfiT2rr → K0dfiT2 = the four `kim.*` flags only; last-epoch mean `mul_mil` 0.100 (off) / 0.106 (on). TK2 reproduction: K5dfiT2rr = saved K5dfiT2 on every printed cell (macro Δ +0.0000). Last-epoch mean trunk losses equal (`mil` 0.139 / 0.140); the last-batch values (0.137 vs 0.308) are one batch, not a difference.

| set | K0dfiT2 macro / micro / AP | K5dfiT2rr macro / micro / AP | Δ micro (KIM) | Δ macro (KIM) | Δ AP |
|---|---|---|---|---|---|
| DoTA-CAP (1,129) | 0.7840 / 0.7405 / 0.5605 | 0.7969 / 0.7236 / 0.5400 | −0.0168 [−0.0230, −0.0110] **COST** | +0.0130 [+0.0079, +0.0181] **GAIN** | −0.0205 |
| DAD (466) | 0.8242 / 0.7807 / 0.2183 | 0.8107 / 0.7516 / 0.1491 | −0.0291 [−0.0435, −0.0140] **COST** (clip classification, oracle 0.8390) | −0.0135 [−0.0338, +0.0077] n.d. | −0.0692 |

**DAD beside the placebo (§56 DA5, post-hoc).** KIM raises the placebo (0.681 → 0.718) and narrows the gap beyond it (+0.143 → +0.093; rows 7+ +0.119 → +0.079). Same-position accident rows, positives vs negatives: 0.749 → 0.726, paired −0.024 [−0.045, −0.003] below.

**Reading.** The expectation (NO DIFFERENCE on DoTA-CAP) was wrong in both metrics, in opposite directions: on the T2 host KIM gains within-clip ranking (macro) and loses cross-clip level (micro, AP). The Nexar host showed the same sign pattern, smaller (§53.1: macro above, DoTA micro −0.004 n.s.). On DAD the macro difference is not separated, and the position-free reads go down.

**Licensed.** "On T2, s2024, KIM on − off: DoTA-CAP macro +0.013 GAIN, micro −0.017 COST, AP −0.021; DAD micro −0.029 COST, DAD macro n.d." **Not licensed.** "KIM improves DoTA" without the metric; "KIM hurts" without the metric; "KIM sharpens localization" (macro only, n = 1); any cause of the micro cost (level vs ranking not decomposed here: `micro_level_read` was not run on these rows).

## 61. Class prototypes on `v^t` in the main score, KIM off: K0dfiP, K0dfiL, K0dfiPL, K0dfiPLb on host K0dfi (dated note 2026-11-28, before any number of these rows; user)

**Why.** The user asked whether KIM's prototype idea (typical features per class, kept as buffers) can enter the main flow as an extra confidence term with KIM disabled, so that `H_bin` does not decide alone, and whether a loss like `L_nkc` should teach it. Facts read in the tree before this note: KIM's prototypes live on `kin_feat`, only weight `kim_score`, and `kim_score` is train-only (`core/inference.py:197` scores `cls_bin_logits`), so **no prototype has ever reached the inference score**; §11 put KIM's gain on "branch + losses". Logged `kim_proto_cos` falls from ~0.9 to −0.3…−0.7, so the two means separate on `kin_feat`. Host = **K0dfi** (§53: K5dfi with KIM off; V2-S motion, DVS θ 0.7 / θ_ego 0.85, `ignore`). Reads: **Nexar val, DoTA-CAP-dev, DADA, DAD; no TAD** (user). No training-free probe first (user).

| # | Choice |
|---|---|
| PR1 Prototypes | Two buffers on `v^t` (the RoFormer output, D = 512; KIM off, so `vis_feats = v^t`), no gradient, saved in the checkpoint and frozen at inference. μ0 ← mean `v^t` over valid steps of label-0 bags; μ1 ← mean over the anchors = top-k valid steps of each abnormal bag by the main score `sigmoid(cls_bin_logits)` (detached) above 0.5, k = the host's `n // mil_topk_pct` (`kim_anchors` on the main score). EMA momentum 0.1, first update sets a prototype, updates from step 219 (warm-up), after the loss of the step. All values = KIM D12 / D11, not tuned |
| PR2 Score term (P) | `cls_bin_logits += λ · (cos(v^t, μ1) − cos(v^t, μ0))` on valid steps, the same term on the caption path. λ = one learnable scalar, **init 0** (AdamW with the host's lr and weight decay), so the row starts equal to K0dfi; the term is 0 until both prototypes exist |
| PR3 Loss (L) | **Normal-anchored, one-sided:** `kim_nkc_loss` with roles swapped — anchors = valid steps of label-0 bags, positive μ0, negatives μ1 + each anchor's 16 most similar abnormal anchors of the batch, temperature 0.1, weight 0.1, from step 219 once both prototypes exist. Abnormal anchors are never pulled to μ1 (μ1 is the mean of the head's own picks: self-confirming) |
| PR4 Bottom-k (b) | μ0's update set also takes the bottom-k valid steps of each abnormal bag by the main score (same k). Prototype only, never a loss term (bottom-k MIL is closed, §49–§51; this is not it) |
| PR5 Rows | **K0dfiP** = K0dfi + P. **K0dfiL** = K0dfi + L (prototypes built, no score term). **K0dfiPL** = K0dfi + P + L. **K0dfiPLb** = K0dfiPL + b. Everything else = K0dfi's command (Nexar_window, A2 Nexar bake, KNN, 1,749 steps, s2024) |
| PR6 Control | **K0dfi** (§53, NVIDIA L4, torch 2.14.1+cu130, CUDA 13.0, step-1 `mil` 0.689790), checkpoint and reads reused **only** if this session's GPU / torch / CUDA equal its `env.json`; otherwise **K0dfirr** = its exact command, trained first in this session, is the control |
| PR7 Gates | Start: step-1 `mil` of every new row = the control's within `KIM_START_TOL` (λ = 0 and warm-up make step 1 identical); a row that misses it reads UNMATCHED START and is not labelled. Live: `proto_ready` = 1 on the last step of every new row (raise otherwise). Printed: final λ, `proto_cos`, anchor share |
| PR8 Guards (raise) | J10 = 1,749; config diff to the control = exactly the row's `proto.*` keys; last-epoch `mul_mil` > 0.01; state dict = the control's keys + `proto.*` only; bakes `crn` none, `motion` V2-S, Nexar statistics |
| PR9 Contrasts | **D = K0dfiPL − control decides.** Printed: P = K0dfiP − control, L = K0dfiL − control, PL − P (the loss on top of the term), b = K0dfiPLb − K0dfiPL, R = K0dfiPL − K5f (candidacy), KIM = K0dfiPL − K5dfi (the prototype head against KIM on the same host; envs may differ). Macro by the read-outs' bootstrap (Nexar val per-video crop means), micro / AP by `zs_arm_table` |
| PR10 Labels | §34: GAIN mean ≥ +0.01 and interval above 0; COST mean ≤ −0.01 and interval below 0; else NO DIFFERENCE. **Deciding sets: Nexar val, DoTA-CAP-dev, DADA.** DAD is labelled but read only beside `dad_position_diag` (placebo, same-position AUC; DAD `t/N` ruler 1.0000, §56 DA5) and never decides |
| PR11 Decision | D: any COST on a deciding set → **COST (closed)**; else any GAIN → **HELD** (s2025 may run; K5f stays); else **NO DIFFERENCE (closed)**. Candidate for the main model only if HELD and R shows no COST on a deciding set; the user decides. n = 1 |

**Expectation (before any number; a wrong one changes nothing).** λ stays small (|λ| < 1) and P is NO DIFFERENCE: the head can already read `cos(v^t, μ)` from `v^t`, so the term adds no information, only a path. L may move the trunk: DADA / DoTA ±0.01, Nexar val NO DIFFERENCE. b changes μ0 little on Nexar_window (in-video negatives already sit in label-0 windows): b NO DIFFERENCE. R: COST on DAD inherited from K0dfi / K5dfi (§34.1, −0.05 is DVS's).

**Not licensed.** "Prototypes make the model confident" (λ and the term are measured, confidence is not); "the prototype head replaces KIM" from KIM = PL − K5dfi alone (envs, n = 1); any DAD macro difference without the placebo; "μ1 helps / hurts" (no row has μ0 without μ1).

Code: config `proto.*` (`core/config.py`, default all false = every earlier arm unchanged, refused with `kim.enabled`), `core/models/proto_bank.py`, `core/models/kat_vad.py` (`_fuse_and_head`), `core/train.py` (`_proto_losses`), report `core/tools/k0dfi_proto_report.py`; runbook `colab/final/k0dfi_proto.ipynb`; outputs `Thesis-Final/outputs/kim_dfi_proto/`.

### 61.1 Read-out (2026-11-28, user's Colab run, Tesla T4 all rows, s2024, n = 1): **D = COST (closed)**, K5f stays

Raw `outputs/v_final/kim_dfi_proto/REPORTS/k0dfi_proto_summary.md`. **Gates.** Control = **K0dfirr** (K0dfi's env was L4, this session T4, PR6). PR7 start matched on all four rows (step-1 `mil` 0.690560); `proto_ready` = 1; J10 1,749 on all five; config diff = the row's `proto.*` keys; last-epoch `mul_mil` 0.149…0.152; state dict = control + `proto.*` (notebook check fixed the same day: the first version's assertion could never pass). Last step λ: P +0.0091, PL +0.0034, PLb +0.0033; cos(μ0, μ1) +0.26 without L, −0.44 with L.

| contrast (macro; Nexar val per-video crop means) | Nexar val | DoTA-CAP-dev | DADA | DAD (never decides) |
|---|---|---|---|---|
| **D = K0dfiPL − K0dfirr** | **−0.0240 [−0.0307, −0.0177] COST** | +0.0033 n.d. (above, under margin) | +0.0011 n.d. | −0.0082 n.d. (below, under margin) |
| P = K0dfiP − K0dfirr | −0.0005 n.d. | +0.0001 | +0.0001 | +0.0012 |
| L = K0dfiL − K0dfirr | −0.0247 COST | +0.0034 | +0.0012 | −0.0089 |
| PL − P | −0.0236 COST | +0.0032 | +0.0010 | −0.0093 |
| b = K0dfiPLb − K0dfiPL | +0.0002 | +0.0001 | −0.0000 | +0.0000 |
| R = K0dfiPL − K5f | −0.0758 COST | −0.0088 n.d. | −0.0353 COST | −0.0758 COST |
| KIM = K0dfiPL − K5dfi (envs differ) | −0.0513 COST | −0.0159 COST | −0.0285 COST | −0.0182 COST |

Micro (printed): L / PL raise DoTA-CAP-dev micro +0.005 and DADA micro +0.004 (intervals above 0, under the margin), DAD micro −0.005. Nexar val negatives with peak > 0.5: 0.646 (rr, P) → 0.673 (L, PL, PLb). DAD placebo / same-position reads: no row separates from K0dfirr (accident rows paired |Δ| ≤ 0.003).

**Reading.**
1. **The score term does nothing:** λ stays ≤ 0.01 from its zero init, so the term moves the logit by ≤ ~0.02; P is equal to the control on every set (expectation held).
2. **Every effect is the normal-anchored loss**, and it costs Nexar val −0.024 (expectation "Nexar NO DIFFERENCE" wrong) while DoTA / DADA move by < +0.005. The loss separates the prototypes in `v^t` (cos +0.26 → −0.44) without that separation reaching the score.
3. **Bottom-k into μ0 changes nothing** (expectation held).
4. Not a candidate: R costs Nexar val, DADA, DAD; the prototype head sits below K5dfi everywhere (printed, envs differ).

**Licensed (n = 1):** "On K0dfi, a zero-init prototype score term stays at λ ≈ 0 and changes nothing; the normal-anchored prototype loss costs Nexar val −0.024 and moves DoTA-CAP-dev / DADA by < 0.005." **Not licensed:** "prototypes do not work" (one host, one weight 0.1, one τ); "the loss hurts Nexar because …" (no cause measured); "μ1 helps / hurts"; "the prototype term replaces KIM".

## 62. Where KIM's min-max micro goes: within-clip vs cross-clip pairs (dated note 2026-11-28, before any number of this read; user)

**Why.** §53.1, §58.1 and §60.1 share one pattern on the all-abnormal sets: KIM on − off raises macro (within-clip ranking) and lowers, or raises less, the min-max micro. §60.1 left the cause open ("level vs ranking not decomposed"). The user asked for the micro-level read. §44.2's `micro_level_read` reads *raw* micro of one row; the reports' DoTA / DADA micro is pooled *after per-clip min-max* and the question is an on − off difference, so this read splits that micro exactly.

| # | Choice |
|---|---|
| MP1 Split | Per row, scores min-max per clip (the reports' pooling). Over all (positive frame, negative frame) pairs: micro = w_W · AUC_W + (1 − w_W) · AUC_X, with w_W = Σ_c P_c N_c / (P N) the within-clip pair share, AUC_W the pair-weighted within-clip AUC and AUC_X the cross-clip pairs. AUC_X is split by the negative frame's place in its clip: **pre** (before the clip's first positive frame), **post** (after its last), gap (between positives, printed if any). Levels printed: mean min-max value of positive, pre and post frames. Macro (equal-clip mean) printed beside |
| MP2 Gates (raise) | Reconstruction of micro from its parts within 1e-6; every clip two-class; on and off rows score the same clip ids with the same labels |
| MP3 Pairs | Nexar host, §53's six arms with §53.1's on sides (K5f, K5dfirr, K5dfierr, K5fbk25, K5dfibk25rr, K5feprerr vs K0f … K0fepre) on DoTA-CAP-dev and DADA; T2 host K5dfiT2rr − K0dfiT2 on whole DoTA-CAP (dev + eval pooled); DoTA host K5dfiDoTA − K0dfiDoTA on DADA (its DoTA-CAP-dev read is in-domain, printed) |
| MP4 Intervals | Δ on − off per part, paired cluster bootstrap (DoTA by source video, DADA by clip), 2,000 draws, §24's seed; a resampled clip's copies count as one clip for the within / cross split |
| MP5 Reading (declared now) | "**Cross-clip**": Δ AUC_X interval below 0 while Δ AUC_W is not; then the place is the part (pre / post) whose Δ AUC_X interval is below 0 and whose negative level rises. "**Within-clip**": Δ AUC_W below 0. Else "not located". Printed, decides nothing, n = 1 |

**Expectation (before any number; a wrong one changes nothing).** w_W is small (≲ 0.01 at 569 clips), so micro ≈ AUC_X and KIM's micro loss sits in AUC_X; on K5f and K5fepre it sits on pre-onset negatives (§43.1: K5f is at 0.81 of its span 1–2 s before the GT).

**Not licensed.** "KIM causes the early rise" (on − off locates a difference, not a mechanism inside KIM); any reading of a pair whose start gate did not match in §53.1 / §60.1; DoTA-host DoTA numbers as zero-shot.

Code: `core/tools/micro_pair_split.py`, tests `core/tests/test_micro_pair_split.py`; output `outputs/v_final/kim/REPORTS/micro_pair_split/`.

### 62.1 Read-out (2026-11-28, local CPU on saved scores, n = 1; printed, decides nothing, K5f stays)

Raw `outputs/v_final/kim/REPORTS/micro_pair_split/micro_pair_split.{md,json}`. MP2 gates passed on all 15 pairs (parts rebuild micro within 1e-6, every clip two-class, labels equal on − off). Every Δ micro equals the §53.1 / §58.1 / §60.1 reports' cell. No gap negatives on any set.

**1. Min-max micro on DoTA-CAP / DADA is a cross-clip number.** w_W (within-clip pair share) = 0.0016 (DoTA-CAP-dev), 0.0008 (whole DoTA-CAP), 0.0028 (DADA): ≥ 99.7 % of micro's pairs compare a positive frame of one clip with a negative frame of another. Within-clip ranking does not enter micro.

**2. KIM never lowers within-clip ranking.** Δ AUC_W is above 0 on every pair except K5f DoTA (+0.0006 n.s.) and the DoTA host on DADA (+0.0006 n.s.); MP5's "within-clip" reading fires nowhere.

**3. On DoTA zero-shot the micro loss sits in cross-clip pairs against pre-onset negatives.**

| pair (DoTA-CAP) | Δ micro | Δ within | Δ x_pre | Δ x_post | pre-onset level off → on | positive level off → on | MP5 |
|---|---|---|---|---|---|---|---|
| K5f | −0.040 | +0.001 n.s. | **−0.081** | +0.007 n.s. | 0.112 → 0.368 | 0.489 → 0.650 | cross-clip (pre) |
| K5fepre | −0.046 | +0.013 | **−0.081** | −0.008 n.s. | 0.087 → 0.376 | 0.484 → 0.680 | cross-clip (pre) |
| K5dfie | −0.013 | +0.009 | **−0.041** | +0.018 | 0.141 → 0.272 | 0.501 → 0.585 | cross-clip (pre) |
| K5fbk25 | −0.002 n.s. | +0.019 | **−0.022** | +0.020 | 0.080 → 0.208 | 0.447 → 0.577 | not located (pre and post cancel) |
| K5dfi | −0.004 n.s. | +0.020 | **−0.011** | +0.004 n.s. | 0.123 → 0.231 | 0.480 → 0.593 | not located |
| K5dfibk25 | +0.004 n.s. | +0.023 | −0.003 n.s. | +0.011 | 0.110 → 0.197 | 0.457 → 0.565 | not located |
| K5dfiT2 (T2 host, 1,129) | −0.017 | +0.012 | **−0.020** | **−0.013** | 0.143 → 0.248 | 0.606 → 0.677 | cross-clip (pre, post) |

KIM lifts the pre-onset frames inside each clip's min-max range (×1.7…×4.3) more than it lifts the accident frames (×1.1…×1.4). Within a clip the accident still outranks its own pre-onset frames (Δ within ≥ 0), but across clips the lifted pre-onset frames now outrank other clips' accident frames. x_pre falls on 6 / 7 DoTA pairs (5 intervals below 0), and the fall is largest on the rows with the largest pre-onset lift (K5f, K5fepre). This is the §43.1 early rise seen through micro.

**4. On DADA the same lift does not cost:** pre-onset levels rise less (e.g. K5dfi 0.106 → 0.167) and positives more (0.377 → 0.503), so x_pre and x_post both rise and micro gains on 4 / 6 Nexar arms. **The DoTA host does not lift pre-onset frames** (DoTA in-domain 0.073 → 0.075, DADA 0.106 → 0.131); its small DADA change sits in x_post (−0.016).

**Expectation check.** w_W small: held. Micro loss in AUC_X: held. Pre-onset on K5f / K5fepre: held (and on K5dfie, T2).

**Licensed (n = 1):** "On DoTA-CAP, KIM's min-max micro cost is cross-clip: KIM raises pre-onset frames within each clip's range more than accident frames, so they outrank other clips' accident frames, while within-clip ranking does not fall." "Min-max micro on DoTA-CAP / DADA measures cross-clip comparisons (≥ 99.7 % of pairs)." **Not licensed:** "KIM anticipates accidents" or "KIM causes the early rise" (on − off locates a difference, not a mechanism; §43.1's cause H1 untested); "the DoTA host avoids the early rise because of DoTA training" (corpus, window, spacing, statistics differ); a reading of a pre-onset lift as a level shift of the whole curve (min-max removes level; this is a shape change).

## 63. DoTA train captions with Qwen2.5-VL: a 30-clip pilot (dated note 2026-11-28, before any caption is generated or read; user)

**A data artifact, not an arm.** The user asked for captions of the recovered DoTA train clips (§52 / §55 corpus), tried first on 30 clips to see the QC, then decide. Nothing here trains, and nothing authorizes using the captions: that needs the pilot read-out **and** a new dated note with its own use and loss rule (the caption `L_neg` line is closed since §33; a reopening is the user's call).

| # | Choice | Value |
|---|---|---|
| DC1 Clips | the frozen §52 map minus `DOTA_TRAIN_EXCLUDED` = **3,020**, read from `Thesis-Final/data/DoTA_CAP_train_frames/{dota_id}.zip` (DoTA-native 10 fps). Pixels not verified (DT7, §52.2): a wrong map row shows up as a caption of another clip |
| DC2 Pilot | **30 clips, seed 2024**: one per DoTA `accident_name` (10 types, seeded pick), then 20 at random from the rest (`dota_caption.pilot_sample`). Drawn: turning 9, oncoming 5, lateral 4, moving-ahead 4, start-stop 2, obstacle 2, pedestrian / leave-left / leave-right / unknown 1 each; ego-involved 19 / 30 |
| DC3 Window | 16 frames at the bin centres of `[anomaly_start − 2.0 s, min(anomaly_start + 3.0 s, anomaly_end))`, widened inside the clip when shorter than 16 frames; each frame 700 × 392 (Nexar's size, both 16:9) |
| DC4 Model | Nexar's §29.3 round-7 setting: `Qwen/Qwen2.5-VL-72B-Instruct`, NF4, greedy, 128 new tokens; the revision reuses Nexar's pin when the model is the same. 32B NF4 allowed on a 40 GB GPU (own folder) |
| DC5 Prompt | `DOTA_CAPTION_PROMPT`: dashcam framing, **DoTA's annotated type** (`DOTA_CAPTION_TYPE_TEXT`, Yao et al. Tab. 1 wording) **and ego involvement** (`DOTA_CAPTION_EGO_TEXT`; non-ego clips are told not to say the ego vehicle crashes) as a hint, the way LaGoVAD's prompt carries the video label; one sentence ≤ 40 words: who collides, what each does before the impact, from where, what follows; describe the video if it disagrees with the label; no weather / lighting / time of day. No type prefix (DoTA's type is known, unlike Nexar's §29.3 contact call) |
| DC6 Output | `Thesis-Final/outputs/dota_captions/pilot/<prompt sha1[:8]>_<model>_16f/` — per-clip JSON (atomic, identity-checked as §29 CP5, filled prompt stored), `pilot_summary.json`, `qc/` (contact sheets, `qc_ratings.csv`, `qc_sheet.md`) |
| DC7 QC | the user rates the 30 on LaGoVAD Tab. B 1–5 (`nexar_caption.score_qc`, unchanged); `type_ok` / `roles_ok` optional y / n. Printed: mean, histogram, word counts, scene-word share, non-ego clips whose caption names "ego vehicle" (a heuristic, not a verdict). **Read as a pilot: nothing passes or fails here.** |
| DC8 After the pilot | the 30 pilot clips become the prompt's **dev set**: the prompt may be changed after reading them, and any later QC that decides anything is a fresh seeded draw outside these 30 (lesson 14) |

**Limits.** One rater; the label hint makes type errors rare by construction, so the pilot measures roles, direction and invention, not type recognition; DoTA's boxes and spans are loose (§52.2). Not LaGoVAD's prompt or model (Qwen2-VL-72B), 16 frames of a 5 s window.

Code: `core/tools/dota_caption.py` (`pilot`, `qc-score`; reuses `nexar_caption`'s captioner, cleaning, identity check and QC scorer), `core/constants.py` (`DOTA_CAPTION_*`), tests `core/tests/test_dota_caption.py` (12). Runbook `colab/final/dota_captions_pilot.ipynb` (80 GB GPU). **Status: built, NOT RUN.**
