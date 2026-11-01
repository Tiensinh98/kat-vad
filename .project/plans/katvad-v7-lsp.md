# KAT-VAD v7 — LaGoVAD-LSP (v5 with MQR replaced by the Latent Surprise Predictor), Nexar-only

Proposal: `core/docs/final/Thesis proposal … .md` §3.2–§4.5 (user-edited, uncommitted).
Status: **PLAN — awaiting user sign-off. Nothing built.**

**Lineage (user, 2026-11-01):** v7 = **v5** (LaGoVAD-MQR) with MQR, which v5 measured as useless, replaced by LSP.
**No relation to v6** (agent prompt): no v6 code, data, control or result is used.
**Precedence (user):** where proposal and current code disagree, **the code wins**; proposal text is fixed afterwards.

## 0. Status

| Phase | What | State |
|---|---|---|
| B0 | re-base branch `v7` on `origin/v5-r7@fc3624d` (drop v6 commits) | **done 2026-11-01**: `v7` = `fc3624d` + `707594c` (papers); old branch kept as `v7-on-v6@5fd86fd`; proposal working copy restored (sha1 `583a077…`, identical) |
| P0 | prereg `core/docs/v7/PREREG_V7.md` §1 (Step 0 rules L0-1…L0-12) | **written 2026-11-01** |
| P1 | Step 0: D1 + LSP stage 1 + D2–D5 → GO / NO-GO (no detector) | **READ OUT 2026-11-01: NO-GO** (D2 gain +0.0981 < 0.10; `core/docs/v7/RESULTS_V7_STEP0.md`) → P2–P4 not started |
| P1-C | Step 0 on CLIP frame vectors (PREREG_V7 §1.1, Amendment 1, post-hoc) | **READ OUT 2026-11-01: NO-GO** (D2 −0.0041 vs context-mean; D3 DoTA s − p −0.012) — `RESULTS_V7_STEP0.md` §5–§6 |
| P2 | build: e_t cache + `v7.*` config + `LSPEvidence` in the v5 model | not started |
| P3 | pilot one seed: R1′ / R3 / R5 vs R0 (= v5 R1 s2024) | not started |
| P4 | 3 seeds on kept rows; CAP-eval once | not started |

## 1. B0 — branch

Today `v7` = `fc3624d` (v5-r7 tip) + `f4e1f82` + `a8d4475` (v6 code, ~5.9 k lines) + `5fd86fd` (paper markdown).
Plan: keep the current branch as `v7-on-v6` (backup, not deleted) → new `v7` from `fc3624d` → cherry-pick `5fd86fd`
→ restore the uncommitted proposal working copy and this plan. Memory bank on `v7` = v5-r7's copy (no v6 entries);
add a `v7` row to CLAUDE.md §14.0.1 + INIT + activeContext.

## 2. What v7 is (code wins)

Host = **v5 R1 exactly** (PREREG_V5 §2, A2 recipe): CLIP rows + V7 grid rows (`--grid-dir`, causal window),
`v5.readout = mean` → frozen V2-S `fc_norm` → `c (· − m_u)/σ_u` with Nexar A2's baked stats → `MotionResidual`
(`W_u` zero-init) → v5/LaGoVAD unchanged. `v2.crn = none`, `dvs_anchor_mode = span`, `is_egocentric = false`,
`L_neg` off, Nexar `window` corpus, 1,749 steps, J10. MQR code (`readout = mqr | tile | change_topk`, sMIL) stays in
the tree untouched and unused.

v7 adds one term after the host's motion residual:

`h_t = x_t + W_u·motion_t + tanh(β)·W_e·e_t` — **the proposal's gated fusion, literally** (user, 2026-11-01): tanh gate,
β = 0, W_e ∈ R^{512×386} Xavier, **no extra LN** on e_t (its residual block is already `LN(sg r_t)`), β no weight decay,
LR × 10 (proposal §3.2.2–§3.2.3; built on v5's `query_lr_mult` optimizer-group path). The tanh gate applies to the
e_t term only; the host motion term stays v5 R1's (X3), so R0 = the existing v5 R1 s2024 checkpoint.

| # | Proposal | v7 (= code) | Why |
|---|---|---|---|
| X1 | P3 trains on Nexar + CCD | **Nexar only**, `Nexar_window` | user; no CCD cache |
| X2 | R0 = "motion stream, 0.81 / 0.75 on DoTA 1,402" | **R0 = v5 R1 s2024**: DoTA-CAP-dev protocol B macro **0.8102**, `nexar_val` crop endpoint **0.8858** | the measured host (RESULTS_V5_STAGE2) |
| X3 | host `c_t + tanh(α) W LN(v_t)` | `x + W_u (c·(fc_norm(mean G) − m_u)/σ_u)`, zero-init, no gate | v5 R1 code |
| X4 | centred window | **causal** (16 frames every 3rd, ending at the row, 1.5 s) | user (a); v5 R1's grids |
| X5 | `u_t = LN_no-affine(mean tokens)` | `u_t = (s1_squash − b)/γ` with V2-S `fc_norm` (γ, b); fp32 cache ⇒ exact | no re-extraction |
| X6 | Δ, K in detector steps | LSP on a **fixed 10 fps lattice** (Nexar frames f, f−3, …; DoTA native). **Δ = 16** (1.6 s ≥ 1.53 s span); K ∈ {16, 32} | user (b) |
| X7 | 15 % P3 hold-out | **`nexar_val`** (225, frozen) | user (c) |
| X8 | 10 % normals for early stop / ŝ stats | 45 of 449 `nexar_train` negatives, seeded id file | proposal |
| X9 | D3/D5 on DoTA train split | **DoTA-CAP-dev (569)**; CAP-eval sealed | no V2-S outside DoTA-CAP |
| X10 | D4 sources Nexar / CCD / DoTA | Nexar-val normals vs DoTA-CAP-dev pre-anomaly | no CCD |
| X11 | 3 seeds from the start | **pilot s2024** paired with R0 s2024 (v5 B11 practice); 3 seeds for kept rows | v5 R1 exists for s2024 only |
| X12 | proposal row names R1 = copy-last | v7 calls it **R1′** to avoid clashing with v5's R1 (= R0 here) | naming |

## 3. Inputs (all exist — no new extraction)

| Input | Drive (read-only) | Used for |
|---|---|---|
| V2-S causal `fc_norm(mean)`, every native frame, all 1,500 Nexar videos | `Thesis-V2/cache/video/vit_s_k710_dl_from_giant/Nexar_s1_squash/` | LSP training (negatives), e_t bake |
| DoTA-CAP V2-S every frame | `Thesis-V2/cache/video/…/DoTA_CAP_s1_squash/` | D3/D5, e_t for eval |
| CLIP, A2 stats, `Nexar_window`, KNN | `Thesis-V2/…` (as v5 Stage 1) | host |
| V7 causal grids: `nexar_train` positives s8, `nexar_val` s1, DoTA-CAP s3 | `Thesis-V5/cache/video_grid/…` | host (R1 readout) |
| R0 ckpt + protocol-B scores + `v5_head.npz` | `Thesis-V5/outputs/v5_stage1/R1/s2024/` | control, read only |

Trigger-only extraction (not in this plan): V-JEPA 2 (iff D3 fails), VideoMAE-S K400 / DAPT (encoder ablation),
ClearCLIP (R12).

## 4. P1 — Step 0 (cheap; no detector)

Tool `core/tools/v7_lsp.py` (`d1`, `train`, `diag`), model `core/models/lsp.py`, notebook
`colab/v7/lsp_step0.ipynb`, outputs `Thesis-V7/outputs/v7_lsp/step0/`.

**LSP (proposal §3.2.2):** 2 pre-LN causal transformer layers, width 384, 6 heads, MLP 4×, learned positions over K,
linear readout from the last position, **zero-init** (starts as copy-last). `û_t = u_{t−Δ} + g(u_{t−Δ−K+1..t−Δ})`.
Smooth-L1 β = 1, mean over 384, targets stop-grad. AdamW 3e-4, wd 0.05, cosine, 50 epochs, batch 256 random windows
from 404 training negatives (all three 30→10 fps phases), early stop on the 45 held-out negatives. Seed 2024.

**Evidence:** `r_t = u_t − û_t`, `s_t = mean ℓ(r_t)`, `p_t = mean ℓ(u_t − u_{t−Δ})`, `ρ_t = log((s+ε)/(p+ε))`,
ε = 1e-6; `ŝ_t` standardized with the 45 held-out negatives; `e_t = m_t·[LN(sg r_t); ŝ_t; ρ_t]` (386-d); `m_t = 0` on
the first Δ + K − 1 lattice steps of each **source** video.

| Diag | Read | Rule (fixed now) |
|---|---|---|
| D1 (before training) | `cos(u_t, u_{t+k})`, k = 1…40, training negatives | printed; **fails only if** cos at k = 16 is within 0.01 of k = 1 |
| K | held-out L_rec, K ∈ {16, 32} | lower wins; < 1 % rel. difference → 16 |
| D2 | held-out L_rec: LSP vs copy-last, const-velocity `2u_{t−Δ} − u_{t−2Δ}`, context-mean | **LSP ≥ 10 % below the best fixed predictor** |
| D3 | `nexar_val` video AUC of `max_t ŝ` vs `max_t p`; DoTA-CAP-dev per-clip frame AUC of ŝ vs p (cluster bootstrap) | **ŝ > p on both** |
| D4 | 5-fold logistic source classifier on per-video surprise stats | printed; source AUC − D3 AUC > 0.2 ⇒ add median-subtracted ŝ variant |
| D5 | lead-in-max share (ŝ vs p), corr(ŝ, p) on normals, ego / non-ego | printed |

Cubic position prior + ego / non-ego printed beside every DoTA-CAP number.
**GO** iff D1 not failed ∧ D2 ∧ D3. Otherwise NO-GO → record, stop (V-JEPA 2 control = separate decision).

## 5. P2–P4

**P2 build (after GO).** `core/tools/build_v7_inputs.py` bakes, per source video on its lattice, at exactly the rows
the v5 grid caches hold (Nexar train s8, `nexar_val` s1, DoTA-CAP s3): (i) e_t of the frozen LSP (R5); (ii) copy-last
`[LN(u_t − u_{t−Δ}); p̂_t]` (R1′); (iii) raw context `u_{t−Δ−K+1..t}` fp16 (R3, predictor trained inside the graph).
`--lsp-dir` + manifest (`lsp_sha`, Δ, K, stride, `train_ids_sha1`), refused on mismatch (v5 grid pattern). DVS splices
e_t with its rows, so seams never get a cross-video context. Config `v7.lsp = none | copy_last | frozen | mil`.
Tests: `v7.lsp=none` == v5 R1 s2024 (strict load, identical scores); β = 0 ⇒ == R0; `mil` at init == `copy_last`;
e_t of a window row == that row of the whole video. β fallback: tanh(β) < 0.01 after epoch 1 → one restart at
tanh(β₀) = 0.1, disclosed.

**P3 pilot (s2024).** Rows R1′ (copy-last), **R3** (LSP via MIL, decisive), **R5** (frozen LSP = LaGoVAD-LSP).
Harness = v5 Stage 2's (protocol B on DoTA-CAP-dev, `nexar_val` crop guardrail). **PILOT-PASS** iff R5 − R0 ≥ 0 ∧
R5 − R3 ≥ 0 ∧ `nexar_val` ≥ R0 − 0.01. Printed: micro, ego / non-ego, window thirds, position prior, tanh(β), ‖W_e‖.

**P4 (on pass).** R0 s2025/s2026 via v5 path + kept rows; primary R5 − R0, R5 − R3, 3/3 seeds, CI excluding 0;
CAP-eval opened once. R4, R6–R11 on the user's word.

## 6. Cost and Drive

P1 one short GPU session (3.5 M params, ~0.5 GB vectors). P2 one pass over cached vectors. P3 three A2-sized runs.
`Thesis-V7/` = code + outputs + v7 caches; `Thesis/`, `Thesis-V2/`, `Thesis-V5/` read-only.

## 7. Risks

* 10 fps rows share 15/16 frames → copy-last is strong at small k; Δ = 16 guards, D1/D2 check.
* Ego-motion surprise everywhere (Asleep at the Wheel) → D5; remedy R9 not in plan.
* Nexar negatives = one dashcam network → DoTA surprise may be domain shift (D4).
* `fc_norm(mean grid)` vs `s1_squash` drifted on some Nexar rows (v5 B9, min cos 0.9995): host reads grids, LSP reads
  s1_squash → printed per split, not gated.
* R3 context bake ~1–2 GB fp16 at K = 32; fallback = gather from s1 in the dataset.
