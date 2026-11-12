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
