# v5 Stage 2 — MQR on Nexar, read on DoTA-CAP-dev (one-seed preliminary results)

Rules: `core/docs/v5/PREREG_V5.md` §2 (S2-7 … S2-14, signed 2026-10-27), §2.4 B10 (read-out choices), **B11 (one seed,
`SKIP_R3`)** and **B12 (AP printed)**, all written before any Stage 2 number.
Plan: `.project/plans/katvad-v5-mqr.md`. Runbook: `colab/v5/stage2_train_read.ipynb`.
Raw outputs: `outputs/v5/v5_stage2/` (`REPORTS/{stage2_readout,maps_readout,g4_provenance,g6_collapse}.{md,json}`,
`eval/{dota/dota_cap_dev,nexar_val}/<row>/`, `maps/`, `<row>/s2024/metrics.jsonl`).
Rows run: R1 (Stage 1, mean readout), R5 (MQR, 8 queries, content keys, λ 0.5), R5Δ (change keys); seed 2024 only.

## 1. Outcome: **PRELIMINARY NOT-ADOPTED (S2-9) (n = 1 seed)**

Candidate R5Δ (R5 had no sentence; R5Δ had the maps sentence). Detection: no difference shown at n = 1. Maps: R5Δ
points, R5 does not. Guardrail S2-9 fails for both rows by 0.002. Per B11 this adopts nothing and does not open
S2-11 (`nexar_test`).

**But the run did not test the readout it was meant to test (§4): the queries never left the mean.**

## 2. Validity

- S1-G4 provenance: R5/s2024 and R5Δ/s2024 both end at step 1749 = last `metrics.jsonl` step; config differs from A2
  only in v5 keys. PASS.
- S1-G6: `mil` and `sMIL` finite; **every query of both runs "did not leave the mean"** (§4).
- Reproduced locally (2026-10-29): `python -m core.tools.v5_stage2` on the raw outputs gives the same read-out to the
  last digit; that local run also adds the B12 DoTA AP rows (§3.4), which the Colab run predates.

## 3. Reads

### 3.1 Detection (S2-7) — DoTA-CAP-dev protocol B, 569 two-class clips, cluster bootstrap 97.5 %

| row | macro AUC | Δ vs R1 [97.5 %] | sentence |
|---|---|---|---|
| R1 | 0.8102 | — | control |
| R5 | 0.8140 | +0.0038 [−0.0038, +0.0116] | no difference shown at n = 1 |
| R5Δ | 0.8139 | +0.0037 [−0.0039, +0.0115] | no difference shown at n = 1 |

Printed: `p_T2` 0.8220 (above every row); v2 Nexar A2 0.8171 (5-seed curve), 0.8110 (s2024); R0 = v2 Nexar A0
0.7138 / 0.6886. `f_2` (DoTA-CAP placement) 0.8528 for all three rows, Δ −0.0001.

### 3.2 Maps (S2-8) — 528 boxed CAP-dev clips, 95 % cluster bootstrap

| row | hit − chance | lift − lift_pre (Am. 1) | hit − centre prior | points? | STAUC |
|---|---|---|---|---|---|
| R5 | +0.307 [+0.282, +0.333] | +0.104 [+0.074, +0.134] | +0.004 [−0.036, +0.045] | no | 0.3168 |
| R5Δ | +0.358 [+0.336, +0.383] | +0.168 [+0.142, +0.195] | **+0.056 [+0.018, +0.093]** | **yes** | 0.3261 |

R5's maps are explained by a centre prior; R5Δ's are not. Both are stronger by ego (+0.33 / +0.37) than non-ego
(+0.29 / +0.34) and grow with box share (<0.5 %: +0.18 / +0.25; >8 %: +0.36 / +0.42). Stage 0's V7 change at ℓ = 6
read +0.076.

### 3.3 Guardrails (S2-9) — `nexar_val` crop endpoint, seed 2024

| row | endpoint | Δ vs R1 | hold (≥ −0.01) |
|---|---|---|---|
| R1 | 0.8858 | — | — |
| R5 | 0.8737 | −0.0121 | no |
| R5Δ | 0.8739 | −0.0119 | no |

Context (printed here, not a rule): v2 Nexar A2's five seeds span 0.866–0.886 on the same endpoint, so a one-seed
−0.012 is inside seed spread. R5 and R5Δ lose the same amount, so the cost sits in what they share (`W_V`, sMIL),
not in the keys.

### 3.4 AP (B12, printed, never gating) — computed locally from `clip_scores.npz`

| row | DoTA per-clip AP macro | Δ vs R1 [95 %] | DoTA pooled AP (per-clip min-max) |
|---|---|---|---|
| R1 | 0.6808 | — | 0.5584 |
| R5 | 0.6930 | +0.0122 [+0.0036, +0.0214] | 0.5743 |
| R5Δ | 0.6928 | +0.0120 [+0.0034, +0.0211] | 0.5742 |
| `p_T2` | 0.6930 | — | 0.5922 |

Positive-frame share (an all-equal score's pooled AP): 0.3271. **`nexar_val` pooled AP is not computable from the
outputs**: `nexar_eval` saved crop/video AUCs only, no frame scores. It needs a Colab re-evaluation.

## 4. Diagnosis: the queries did not learn (a design defect, not a bug)

- **Entropy.** Every query of both runs sits at 3.8907–3.8918 from step 0 to step 1749 (log 49 = 3.8918). R5Δ's
  per-step maps differ across tiles by a median **4.4 %** (max 8.3 %) of their mean (`maps/maps_R5d_s2024.npz`).
- **Cause: step budget versus logit scale.** `u_m` starts at 0. AdamW's per-coordinate step is ≈ the LR, here
  5e-5 under warmup + cosine (mean multiplier ≈ 0.5), so over 1,749 steps each coordinate moves at most ≈ 0.044.
  The logit is `u_m · LN(K) / √d_K` with √d_K = 19.6 (27.7 for R5Δ), and tiles of one step share most of their
  layer-normalised key, so the logit spread across tiles stays ≈ 0.05. Proposal §4.4 lists an LR multiplier of
  1 or 10 for MQR; S2-13 D2 fixed it at 1. Efficient Probing, the source of the readout, trains the probe alone at a
  probe's learning rate.
- **Toy check (local CPU run, synthetic grids, one planted tile; script in the session scratchpad, not tracked).** With
  `GridReadout` + `SpatialMILHead` + sMIL under the same optimiser and schedule: ×1 entropy 3.8918, query mass on the
  planted tile 0.021 (uniform 0.020) → the Nexar run reproduced; ×10 entropy ≥ 3.8907, mass 0.033 (still the mean);
  **×100 (queries only, no weight decay) entropy min 2.890, mass 0.993**; ×1 with `u_m ~ N(0, 0.02²)` 3.8918 / 0.021
  (a random start does not help: the step budget, not symmetry, is the limit). The toy's signal strength is arbitrary,
  so it shows the scale at which the queries *can* move, not what they will find on Nexar.
- **What was measured, then.** R5 and R5Δ = "tile mean + learned `W_V` + the sMIL auxiliary loss". The +0.004 AUC /
  +0.012 AP and the −0.012 Nexar guardrail belong to that, not to query selection. R5Δ's maps "point" because the
  hit read ranks tiles, so a few-percent tilt counts. The tilt is real, but weak.

## 5. What is licensed

- **Licensed:** at seed 2024 with λ 0.5, M 8 and LR ×1, an MQR-initialised readout stays at the tile mean; the sMIL
  auxiliary moves DoTA-CAP-dev per-clip AP by +0.012 [+0.004, +0.021] (printed, not a sentence) and AUC by an amount
  that cannot be told from zero; R5Δ's near-uniform maps rank the agent's tile above a centre prior.
- **Not licensed:** any statement that "MQR helps / does not help". The readout under test did not train. No
  "maps point" claim without "near-uniform (≤ 8 % tile spread), one seed".
- Next: see plan `.project/plans/katvad-v5-mqr.md` §0, Stage 2 row (direction B, to be decided by the user).
