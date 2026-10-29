# v5 Stage 2b — MQR with the query LR × 100 (B13), mechanics gate on the training logs

Rules: `core/docs/v5/PREREG_V5.md` §2.4 **B13** (post-hoc to Stage 2, disclosed; deviation D10), written and built
before any Stage 2b number. Plan: `.project/plans/katvad-v5-mqr.md`. Runbook: `colab/v5/stage2b_query_lr.ipynb`.
Raw outputs: `outputs/v5/v5_stage2b/` (`REPORTS/{g4_provenance,g6_collapse}.{md,json}`,
`<row>/s2024/{config.yaml,metrics.jsonl,verified.json}`); untracked.
Rows run: R5q (= R5, content keys) and R5dq (= R5Δ, change keys), each with `v5.query_lr_mult = 100`; seed 2024.
Control (never re-run here): Stage 1's R1 and Stage 2's R5 / R5Δ at × 1 (`RESULTS_V5_STAGE2.md`).

## 1. Outcome: **B13 G0 FAIL → stopped before any evaluation**

G0 (B13 (c)): for **every** R5q/R5dq run, `mil`/`sMIL` finite **and** some query below 0.95·log 49 = **3.697** at the
last step. R5q's lowest query ends at **3.714** → the run fails, so the notebook stopped as pre-registered. No
DoTA-CAP-eval, DoTA-CAP-dev or `nexar_val` number exists for R5q/R5dq, and none will be computed for them (DoTA-CAP-eval
stays unread by v5).

Per B13 (c) the outcome reported to the user is **"MQR does not train at × 100"**, with the reading in §3: the queries
do move at × 100, but they stop at ≈ 40 of 49 tiles; that is not the step-budget failure Stage 2 diagnosed.

## 2. Validity

- S1-G4 provenance: both runs end at step 1749 = the last `metrics.jsonl` step; config differs from A2 only in v5
  keys. PASS.
- S1-G6: `mil`, `sMIL` finite in both. "Did not leave the mean" = **no** for both, unlike Stage 2's R5 / R5Δ.
- R5dq passes G0 alone, by 0.002 (3.695 < 3.697). The rule needs every run. The threshold is not re-read after the
  number (lesson 14).

## 3. Reads (training logs only; printed, the gate is §1)

### 3.1 Last step (1749)

| run | entropy per query (log 49 = 3.892) | min | logit std per query | ‖u_m‖ range | G0 |
|---|---|---|---|---|---|
| R5q | 3.733, 3.779, 3.714, 3.843, 3.763, 3.822, 3.863, 3.789 | 3.714 | 0.23–0.58 | 4.5–9.3 | fail |
| R5dq | 3.739, 3.745, 3.695, 3.813, 3.730, 3.753, 3.810, 3.792 | 3.695 | 0.37–0.60 | 7.1–15.1 | pass |

Stage 2 at × 1, for contrast: every query 3.8907–3.8918 at every step, maps' tile spread ≤ 8 % of their mean.

### 3.2 Trajectory (computed locally from `metrics.jsonl`)

| run | step | min entropy | mean entropy | effective tiles e^H (min query) | max logit std | ‖u_m‖ max |
|---|---|---|---|---|---|---|
| R5q | 437 | 3.775 | 3.826 | 43.6 | 0.48 | 5.79 |
| R5q | 874 | 3.740 | 3.800 | 42.1 | 0.55 | 8.43 |
| R5q | 1311 | 3.697 | 3.784 | 40.3 | 0.59 | 9.15 |
| R5q | 1749 | 3.714 | 3.788 | 41.0 | 0.58 | 9.26 |
| R5dq | 437 | 3.750 | 3.796 | 42.5 | 0.50 | 10.74 |
| R5dq | 874 | 3.710 | 3.766 | 40.9 | 0.58 | 13.95 |
| R5dq | 1311 | 3.683 | 3.755 | 39.8 | 0.60 | 14.90 |
| R5dq | 1749 | 3.695 | 3.760 | 40.2 | 0.60 | 15.06 |

Mean of the min-query entropy over the last 10 % of steps: R5q 3.707, R5dq 3.687.

## 4. Diagnosis: the queries move, then saturate (printed, not gated)

- **The × 100 group did its job.** Logit std goes from 0 to 0.2–0.6 within the first quarter, and every query leaves
  the mean. The step-budget defect of Stage 2 (`RESULTS_V5_STAGE2.md` §4) is fixed mechanically; lesson candidate
  (ca)'s fix works.
- **Concentration stops at ≈ 40 effective tiles.** Entropy falls only in the first half (R5q min 3.775 → 3.697 by
  step 1311, R5dq 3.750 → 3.683), then sits or rises slightly while LR decays. The toy with one planted tile reached
  entropy 2.89 / mass 0.993 under the same optimiser, so the readout can concentrate when a tile carries the label.
- **The norm grows, the spread does not.** From step 437 to 1749, R5q's max ‖u_m‖ grows ×1.6 (5.8 → 9.3) and R5dq's
  ×1.4 (10.7 → 15.1), while max logit std grows only ×1.2 (0.48 → 0.58, 0.50 → 0.60). Growing a query along a
  direction where tiles barely differ buys no spread. Reading: in the directions the loss rewards, V2-S's 7×7 keys
  differ little from tile to tile. This is an inference from logged scalars, not a measured key spectrum.
- **Consistent with Stage 0.** V2-S token change did not point at the agent (G-B ✗, `RESULTS_V5_STAGE0.md`), and check L
  attributes most of the miss to the 16-frame window overlap. A video-level MIL + sMIL objective has no reason to
  push queries onto tiles that do not carry more anomaly evidence than their neighbours.
- **What would not fix it.** × 1000 (needs a new amendment) scales a gradient that is already being spent, and the
  entropy plateau is not limited by step size. An entropy penalty would force sharpness without saying where.

## 5. What is licensed

- **Licensed:** with V2-S 7×7 token grids trained on Nexar under the A2 recipe (λ 0.5, M 8, seed 2024), an MQR readout
  whose queries get LR × 100 leaves the tile mean but settles at ≈ 40 of 49 effective tiles and fails B13 G0
  (min entropy 3.714 vs 3.697 required). Combined with Stage 2: at × 1 the queries cannot move; at × 100 they move and
  stay diffuse.
- **Not licensed:** any detection, map or guardrail claim about R5q/R5dq (none was read); "MQR does not help detection"
  (untested at the readout's intended operating point); "R5dq trains" (one run passing by 0.002 under a rule that
  needs both).
- v5's detection record stays Stage 2's one-seed preliminary read: R5/R5Δ − R1 +0.004 (ns), measured on the tile
  mean + `W_V` + sMIL.
- Next: plan `.project/plans/katvad-v5-mqr.md` §0, Stage 2 row. User's choice of (A) close v5 as a diagnosed null, or
  (C) R7 = MQR on CLIP patch grids (§2 signed it out; needs a new amendment and a Nexar CLIP-patch cache).
