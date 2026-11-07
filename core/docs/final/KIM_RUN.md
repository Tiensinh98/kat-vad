# KIM — build and run rules (LaGoVAD-KIM)

> Branch `v_final_las_vad`. Proposal: `core/docs/final/Thesis proposal … (LaGoVAD-KIM.md`.
> Plan: `.project/plans/katvad-kim.md`. **Status: RULE, written 2026-11-06 before any KIM number.
> K-1 BUILT 2026-11-06 (`core/kim/`, wiring, `core/tests/test_kim.py`); pilot `core/tools/kim_pilot.py`,
> read-out `core/tools/kim_readout.py`, runbook `colab/final/kim_runs.ipynb` built the same day.** Every later change to this file is a dated note
> below the text it changes, never an edit of a rule after its number is seen.
>
> **Run state (2026-11-07):** K0, K0k9, pilot and K5 read at s2024 (§9). K0 and K5 read at s2025 (§10): **§6.2 reads "reproduced",
> n = 2, nothing adopted.** Kcap read at s2025 (§11): **§6.3 fails, K5's gain is not attributed to the differences.** Next: Kcap s2024 with paired
> Kcap − W0 (§11.5, built, not run); it cannot reverse §6.3. K2 and s2026 are held until the user reads §11.5.

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
