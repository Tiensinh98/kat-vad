# Plan — LaGoVAD-KIM (Kinematic Intention Module)

> Branch `v_final_las_vad` (tip `7eefc17`). Source: `core/docs/final/Thesis proposal … (LaGoVAD-KIM.md`.
> Status (2026-11-07): **K-1 to K-3 done at s2024, n = 1, nothing adopted; next = s2025 (`KIM_RUN.md` §6.2).** Written 2026-11-06. Every decision marked **[U]** was asked before the build.

## 0. Status

| Step | What | State |
|---|---|---|
| K-0 | Notation map + rule doc `core/docs/final/KIM_RUN.md` (before any number) | **done 2026-11-06**; proposal corrected the same day (§2 items 1–8, DVS off, Nexar-only, pilot on endpoints) |
| K-1 | Build `core/kim/` + wiring + tests (CPU, data-free) | **built 2026-11-06** (K2/K5 only; ladder flags not built); `test_kim.py` 27 green. Added arm **K0k9** (host kernel 9, `KIM_RUN.md` §3.1) |
| K-2 | Pilot (inference only, **decided control K0 / K0k9**: W0's checkpoint deleted, `KIM_RUN.md` §4 note, §5.1) KP1–KP4 | **read 2026-11-07 on K0** (`KIM_RUN.md` §9.2): KP1, KP2 pass; KP3 fail → first-pair mask (§5.3); KP4 fail → H6 dropped |
| K-3 | First training read: K0 / K0k9 → K5 (K2 waits, §6.1), n = 1, Nexar | **read 2026-11-07, s2024** (`KIM_RUN.md` §9): kernel 3 stays; K5 passes §6.1 against K0; K5 − W0 no difference shown on any set |
| K-4 | s2025 K0 / K5 (§6.2), then Kcap (§6.3), K2, s2026 if reproduced | **s2025 read 2026-11-07: reproduced, n = 2** (`KIM_RUN.md` §10: K5 − K0 DADA +0.0383 [+0.0293, +0.0473]). **Kcap s2025 read 2026-11-07: §6.3 fails** (`KIM_RUN.md` §11: K5 − Kcap DADA +0.0012 [−0.0056, +0.0080]), gain = branch + losses. **Kcap s2024 read: Kcap − W0 no difference on any set (§11.6, DADA +0.0009 [−0.0107, +0.0127])**, step-1 mismatch flagged, cause not measured. **K5d read: K5d − W0 DADA +0.0098 [+0.0010, +0.0185], below the +0.01 margin, so the KIM LINE CLOSED 2026-11-07 (§12.1).** K2, s2026 not run. **User chose K5 as main model in place of W0 (§12.3), not a rule's outcome** |
| K-5 | Rest of the ladder (K4, K6–K8, ...) | not built; only after K-4 |

## 1. Notation collisions (proposal ↔ code). READ BEFORE WRITING ANY CODE

The proposal reuses symbols the code and records already use for **different things**. Code
uses the right-hand column names; the proposal symbol never becomes a code identifier.

### 1.1 Losses

| Proposal | Proposal meaning | Already in code / records | Code name for KIM |
|---|---|---|---|
| **L_kin**, weight λ_k | top-k MIL BCE on KIM's own score q_t | `losses["kin"]` = KIP v1 `kinematic_loss` (`core/kip/losses.py:98`): MIL on the motion head ŷ_O **+ β·smooth-L1 consistency** to the main curve; weight `loss.gamma_kin` = 0.2; switch `kip.use_lkin` | `losses["kim_mil"]`, `kim.mil_weight`. **Do not call `kinematic_loss`** (it adds the consistency term and reads `kip.*`) |
| L_nkc, λ_c, τ_c = 0.1 | normal-anchored InfoNCE | no NKC; but `loss.lambda_rec` / `lambda_align` / `tau_align` are KIP's, `CONTRASTIVE_TEMP` = 0.02 is L_neg's | `losses["kim_nkc"]`, `kim.nkc_weight`, `kim.nkc_temp` |
| "top-k MIL, k = max(⌊L/16⌋, 1)" | host MIL | host on Nexar runs `mil_topk_pct` **5** (k = L // 5), not 16 (`STP_BUILD.md` D2). Also: code averages the top-k **logits** then BCE-with-logits (`core/losses/mil.py:41`); the proposal averages **probabilities** | `kim.topk_pct` defaults to the host's `loss.mil_topk_pct`; averaging form **[U1]** |
| L_MIL-align (LaGoVAD) | multi-class MIL alignment | = code `mul_mil`. **Not** `kip_align` (= L_KIP_align, PMG↔flow InfoNCE) | unchanged |
| L_dvs | DVS pseudo-label loss | = `dvs_sup` + `dvs_sup_mil`, `dvs_anchor_mode=span` on W0 | unchanged |
| L_neg ("h̃ feeds L_neg's pooling") | caption contrast | **never runs on Nexar**: `captions_from_definitions: false` → `train.py` skips it (`STP_BUILD.md` D8). The proposal's §3.2.6 point 6 (LaGoVAD's −6.3 TAD without L_neg) is LaGoVAD's host, not ours | nothing to wire; proposal text must say L_neg is off |

### 1.2 Symbols

| Proposal | Proposal meaning | Code / records meaning | KIM code name |
|---|---|---|---|
| **u_t** | velocity feature | **`u` = VideoMAE motion vector** (`ũ`, `σ_u`, `c·ũ/σ_u`, "u adds +0.043", `motion_share`) | `vel` |
| **p_t**, "position" | learned 128-d state | **`p` = cubic position probe, `p_T2` = no-pixel position prior** (G6/G7, every read-out prints it) | `state` (never "position" in code or read-outs) |
| **W_p** (128×512) | state projection | `module.w_p`, `stp.sv2_proj`, `sv2_wp_norm` = SV4's 512×512 SA²-v2 projection | `kim.state_proj`, log `kim_w_state_norm` |
| a_t | acceleration | head blend `a`, `adaptive_fuse_alpha0` | `acc` |
| k, k_t | top-k count **and** the 384-d KIM feature (same letter, proposal-internal clash) | k = top-k count | feature = `kin_feat` |
| c_t | CLIP CLS | `c` = fixed motion scale in `c·ũ/σ_u`; SA²-v2 `c_t`/`c̃` | `clip` / host `x` |
| ṽ_t = (v̄−m_u)/σ_u | motion input | the baked `c·ũ/σ_u` of A2 (no CRN) | read the baked cache as is — **no new normalisation** |
| β = 0.1 | prototype momentum | `loss.beta_cons` = 0.5 (KIP consistency); β₁…β₄ SA²-v2 scorer | `kim.proto_momentum` |
| α = 0.5 | anchor threshold | `adaptive_fuse_alpha0` | `kim.anchor_thresh` |
| ρ_V | per-video median change | `rho` = KIP grad ratio (C37), `ρ_u` (WI), Spearman ρ | `kim_scale` |
| q_t | KIM step score | SA²-v2 "top-3 `q`" | `kim_score` |
| r_t | KIM logits | CRN `r = d − f` | `kim_logits` |
| M = 16 | NKC negatives | v5 `M = 8` (queries) | `kim.nkc_negatives` |
| m_t | seam/cut/pad mask | `mask` / `padding_mask` (float, 1 = valid) — padding only | `kim_step_mask` (pairs, 1 = valid difference) |
| h_t, h̃_t | RoFormer out, after return | `vt` (v^t), `vk` (v^k = KIP splice slot, `kat_vad.py:351`) | same slot, new module |
| R0…R14, R-cap | KIM ladder | **R1/R4/R5/R6 = STP arms (`ZS_COMPARISON` "LaGoVAD + R4" = SA²), v5 R1/R3/R5/R5d/R7**, W0/WI/WL, SV2–SV4 | **K0…K14, Kcap** (K0 = W0) |
| B0, B1, B2 | baseline rows | **`B` = deployed A3-ign; protocol B (E1); E1 arm B** | KB0, KB1 (KB2 = K0 = W0) |
| Step 0, K1–K4 | pilot checks | v5 Stage 0; v2 kill-switch K, K-pos | pilot **KP1–KP4** |

### 1.3 Concept trap

"Kinematic" was already used for **KIP** (ego-kinematics refuted 10/10 arm-seeds; KIP's +0.09 = smoothing).
KIM is a different module (differences of `vt`, no flow target, no gate-shift). Every doc says
"KIM", never "kinematic module" alone, and never cites KIP's numbers as KIM evidence.

## 2. Proposal claims that do not match the record (fix in the proposal, or say why not)

1. **"0.81 macro / 0.75 micro zero-shot on DoTA, Nexar + CCD, 1,402 clips"** — the measured W0 is
   **DoTA-CAP-dev (569 two-class clips), protocol B, macro 0.8110, micro 0.7255, Nexar only**
   (`ZS_COMPARISON.md`; `STP_BUILD.md` D2/D3). DoTA pixels are gone, the 1,402 list cannot be scored
   by retrained rows, and CLAUDE.md forbids putting a protocol-B number beside 62.60. Same-protocol
   LaGoVAD public ckpt = **0.6418**.
2. **Host recipe**: W0 runs `score_head_kernel` **3** (proposal: 9), `mil_topk_pct` 5 (16), stage 2,
   **1,749 steps** (40 epochs). §3.2.6 point 3 ("kernel-9 linear conv cannot form |Δ|") still holds
   for kernel 3, but the text must say 3. Warm-up "5 of 40 epochs" → steps.
3. **CCD is not in the repo** (no adapter, no cache). P3 = Nexar only unless CCD is downloaded and built **[U2]**.
4. **Pilot on DoTA's training split is impossible** (no pixels → no CLIP/VideoMAE rows). DoTA-CAP-dev and
   DADA T2-test are the endpoints, so the pilot cannot read them either. Pilot set = **Nexar val**
   (already cached, has event/alert times for KP4) **[U3]**.
5. **P1 TAD in-domain**: in-domain TAD training collapses to a clip classifier (C14, macro 0.7578 → 0.6174,
   every objective repair failed). KIM on P1 inherits that; P1 should come last or be dropped **[U4]**.
6. **VideoMAE input** "16 frames centred, 1.6 s at 10 fps" ≠ the cached V2-S (causal ≈ 1.5 s). Changing it
   invalidates every cache and every number (C2). Keep the cache; fix the text.
7. **"Raw CLIP differences as inputs did not help in preliminary tests"** — no record found on this branch.
   Cite the run or remove the sentence.
8. **Shot cuts (PySceneDetect)**: needs pixels and a new tool; Nexar dashcam clips should have ~none.
   Phase 1 masks **DVS seams + padding only**; cuts deferred to TAD/MSAD **[U5]**.

## 3. Design decisions for the build (my picks, each with the alternative)

- **D-1 Splice.** KIM sits in KIP's slot (`vt → vk`), as a new `core/kim/` module with its own
  `KIMConfig` (`kim.enabled`, default off). `validate_kim`: `kim.enabled` and `kip.enabled` are exclusive.
  *Alt:* reuse `KIPConfig` → rejected, it would mix L_kin meanings.
- **D-2 Seams.** `SynthesizedSample` gets `segment_ids (T,)` from `compose_sequence` (lengths of parts are
  already there); `truncate_sample` truncates it; collate pads with −1. Difference valid iff same segment,
  both valid. Default path: unused field, no number moves.
- **D-3 ρ_V scope.** Per **segment** (source clip), not per synthesized bag: a bag mixes clips with different
  ego speeds. *Alt:* per bag as written → R9-style side row if wanted.
- **D-4 Return path.** `W_o` 512×384 zero-init, bias-free, base LR, one AdamW group (SV3/SV4 lesson:
  Σlr ≈ 0.044 per coordinate over 1,749 steps — fine for a full matrix, W_u trains the same way). Logged
  `kim_w_o_norm` from step 0. No LR multiplier unless a pre-registered note derives it.
- **D-5 NKC negatives.** "Steps of normal videos in the batch" = steps of bags with label 0 **plus DVS
  filler segments** (they come from normal videos, exact labels). Non-synthesized abnormal bags never give
  negatives.
- **D-6 Anchors 𝒜.** top-k of `kim_score` per abnormal bag, above `anchor_thresh`; on synthesized abnormal
  bags inside the anchor span (`pseudo_label == 1`). Empty 𝒜 → NKC = 0 for that batch (logged share).
- **D-7 Warm-up.** NKC + prototype updates off for the first `kim.warmup_steps` = round(5/40 · 1,749) = 219.
- **D-8 No hyperparameter search in the first pass** (λ_k = 1, λ_c = 0.1, M = 16, α = 0.5, β = 0.1 fixed),
  as STP D5. The 15 % held-out search is a later step, on Nexar train only.
- **D-9 Test-time score** = host score only; `kim_score` and ‖acc‖ saved as diagnostics.

## 4. Steps

### K-0 Rule doc (before any number)
`core/docs/final/KIM_RUN.md`: notation map (§1 above), deviations (§2), arms, endpoints, read-out rule:
endpoint DoTA-CAP-dev protocol-B macro + DADA T2-test macro, Δ vs K0 with cluster bootstrap, "gain" iff
low > 0 and ≥ +0.01, always PRELIMINARY at n = 1; `p_T2` / cubic `p` printed beside every macro;
mechanics printed (‖W_o‖, gate means, |𝒜| share, cos(μ0, μ1), KIM video AUC on Nexar val). Prediction
written first (honest prior: no difference, as MQR/LSP/SA²).

### K-1 Build (CPU, data-free)
- `core/kim/module.py`: state proj + LN, masked |Δ| / ρ, gated vel / acc (Conv1d k=3, Xavier, bias 0),
  2-layer scorer, prototype buffers, `W_o`. Order flag `kim.order ∈ {state, state_vel, full}` (K6),
  `kim.signed` (K10), `kim.scale` on/off (K9), `kim.mask` on/off (K8), `kim.input ∈ {vt, x, clip}` (K7),
  `kim.detach_input` (K12), `kim.capacity_control` (Kcap).
- `core/kim/losses.py`: `kim_mil_loss`, `kim_nkc_loss` (positive in the denominator).
- Wiring: `KATVAD.from_config/forward`, `train.py` loss dict + total + logs, `segment_ids` through
  `synthesis` / `dataset` / collate, `constants.KIM_*`, eval tools keep host score.
- Tests `core/tests/test_kim.py`: kim off ⇒ identical outputs and state-dict keys; W_o = 0 ⇒ scores equal
  K0 at step 0; seam/pad mask zeroes cross-segment differences; ρ detached and per segment; NKC hand-computed
  on a toy; prototypes no grad; warm-up gate; exclusivity with KIP; ckpt round-trip.
- Quality gate §11. (codebase-memory / detect_changes MCP not present in this container — run
  `git diff --stat` scope check instead, say so.)

### K-2 Pilot KP1–KP4 (inference only, W0 s2024 ckpt, Nexar val)
`core/tools/kim_pilot.py`: per-video AUC of ‖Δ vt‖/ρ and ‖Δ² vt‖/ρ vs the same on CLIP and motion rows (KP1);
rank-average with W0's score (KP2, on Nexar val, not an endpoint); top-5 % ‖Δ‖ near DVS seams — needs a
synthesized dump (KP3); peak lag vs Nexar event time (KP4). Gates decide K7's default and how H6 is worded,
never stop the plan (as the proposal says).

### K-3 First training read (n = 1, s2024, Nexar)
K0 = W0 (existing ckpt, config diff asserted = kim section only), **K2** (KIM + kim_mil, no prototypes),
**K5** (full: prototypes + NKC). Endpoints as K-0. ≈ 2 runs × one W0 cost.

### K-4 Ladder (only if K-3 is not a cost)
Kcap, K4 (LAS-VAD contrast), K6, K7, K8 first at n = 1; 3 seeds for K0/K2/K5/Kcap. K1, K3, K9–K14 by budget.
Stop rule written in K-0: K5 − K0 cost on both endpoints → close the line, no variants (SA²-v2 §11.4 lesson).

## 5. Risks

- **Redundancy with W_u / VideoMAE** (≈ MQR/LSP): KIM differences `vt`, which already contains `W_u u`.
  Kcap + K13 separate it; prior = no difference.
- **Smoothing in disguise** (KIP lesson): gated |Δ| through a k=3 conv + W_o can act as a temporal filter.
  Kcap is the control that matters; never write "kinematic evidence" without it.
- **Position**: |Δ| near clip ends / padding; `p_T2` printed, padding masked.
- **Overfit tell**: train `mil` far below W0's 0.30 (SV2/SV4 pattern).
- **Adam budget**: small LR × 1,749 steps; KIM's own scorer trains only through kim_mil at init (W_o = 0).

## 6. Questions for the user

**Answered 2026-11-06:** U1 = proposal form (mean of top-k **probabilities** `kim_score`, then BCE;
differs from the host's logit average, recorded as a deviation-free choice in `KIM_RUN.md`).
U2 = **Nexar only, no CCD** (P3 = Nexar; CCD rows, CCD onset and Nexar+CCD comparisons leave scope;
onset analysis uses Nexar event times + DoTA-CAP-dev anomaly start). U3 = **pilot on Nexar val**.
U4 = **no training on TAD (nor MSAD: Nexar only)**; TAD = zero-shot test only, as `ZS_TAD_DADA.md`.
U6 (pilot) = **Step 0 reads DoTA-CAP-dev, DADA-2000 T2-test (370 sources) and TAD test**, with the W0
s2024 ckpt on the existing A2 Nexar-stat bakes (`dota_cap_dev_s3_from_s1_nexar`,
`A2_S/DADA2000_orig_test_sources_nexar`, `A2_S/TAD_s8_nexar`). These are KIM's endpoints, so the pilot's
decision rule is frozen in `KIM_RUN.md` before it runs and its read is disclosed as "seen before KIM
training" in every KIM read-out. Nexar val printed beside.
**DVS OFF for every KIM row and its control** (user: cleanest KIM path): `dvs.theta = dvs.theta_ego = 1.0`
→ no splice, no `dvs_sup` / `dvs_sup_mil`, no seam. Consequences: D-2 (`segment_ids`) and D-3 (per-segment
ρ) are dropped (ρ per video, mask = padding only); D-5 negatives = label-0 bags only; D-6 anchors = top-k
of label-1 bags only; **K0 is no longer W0** → new control **K0 = W0 with DVS off**, retrained (W0 printed
beside); KP3 (seam check) dropped. Shot cuts (U5): not built — no TAD/MSAD training, Nexar has ~none.

- **[U1]** KIM MIL: average top-k logits (host code) or top-k probabilities (proposal)?
- **[U2]** CCD: download + adapter now, or Nexar-only like every v_final row?
- **[U3]** Pilot on Nexar val instead of DoTA-train (impossible)?
- **[U4]** P1 TAD in-domain: drop or last?
- **[U5]** Shot cuts deferred (seams + padding only first)?
- **[U6]** Fix proposal text for §2 items 1, 2, 6, 7 now, or after K-3?
