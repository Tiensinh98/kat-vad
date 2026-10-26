# v5 Stage 0 — do VideoMAE V2-S tokens carry local anomaly evidence? (results)

Rules: `core/docs/v5/PREREG_V5.md` §1 (S0-1 … S0-12, signed before any grid existed), §1.1 Amendment 1 (before any
grid), **§1.2 Amendment 2 and §1.3 check L (both written after the first read-out, disclosed there)**.
Plan: `.project/plans/katvad-v5-mqr.md`. Runbooks: `colab/v5/stage0_dota_cap_tokens.ipynb` (first run),
`colab/v5/stage0b_amend2_lag.ipynb` (Amendment 2 re-extract + re-read + check L).
Local copy of the first read-out: `outputs/v5/v5_stage0/stage0_readout.md` (gitignored; this file is the record).

## 1. First read-out (2026-10-26): INVALID — a defect in the gate, not in the data

- **Validity.** G0 min cos **0.99999999998** over every stored step: the new pass reproduces v2's
  `DoTA_CAP_s1_squash` exactly. G2: no clip failed. **G1-pixel: 15 / 569 dev clips failed** → 554 dev maps → INVALID.
- **Cause (Amendment 2).** The tool re-took v2's clip-level pixel gate (`passes_frame_gate`: mean ≥ 0.99, min ≥ 0.95)
  on the Stage 0 rows `s1[::3]` only. v2 took it on **every** frame and passed all 15: their v2 all-frame means are
  **0.9901 – 0.9918** (`outputs/v2/REPORTS/dota_cap/dota_cap_alignment.json`), mins 0.964 – 0.983. A third of the rows
  of a clip on the 0.99 line falls under it by sampling. Same pixels v2 accepted.
- **Fix.** G1-pixel = v2's all-frame verdict + per-row min ≥ 0.95 (`passes_row_gate`). The 15 clips' V7 grids were
  never written (Stage 2 needs them), so they are re-extracted (`--retry g1_pixel`), then the read-out is re-run.

## 2. What the 554 valid clips show (read under §1; formally void until the re-run)

Cluster bootstrap over DoTA source videos, B = 10,000, 95 %.

| read | V7 | V14 | C14 CLIP patches (printed) |
|---|---|---|---|
| D1 `e = ρ − ρ_pre` (G-A) | **+0.059** [+0.024, +0.092] (507) | +0.083 [+0.049, +0.116] | +0.414 [+0.375, +0.454] |
| D1 `e`, non-ego | +0.030 [−0.023, +0.084] (240) | +0.048 [−0.008, +0.107] | +0.466 [+0.398, +0.533] |
| D1 raw `ρ` (inside vs outside) | **−0.083** [−0.114, −0.053] | −0.143 [−0.173, −0.115] | +0.530 [+0.496, +0.567] |
| D3 hit − chance (G-B) | **+0.002** [−0.014, +0.018] (514) | −0.010 [−0.023, +0.004] | +0.286 [+0.267, +0.305] |
| D3 lift − lift_pre (G-B, Am. 1) | +0.011 [−0.009, +0.032] | +0.002 [−0.013, +0.017] | +0.215 [+0.197, +0.233] |
| D2 top-k − global, non-ego (G-C) | **+0.035** [+0.021, +0.050] (244) | +0.076 [+0.057, +0.099] | +0.007 [−0.009, +0.025] |
| D2 macro top-k (training-free) | 0.580 | 0.630 | 0.715 |
| D2 macro global | 0.538 | 0.538 | 0.699 |

Gates: V7 and V14 both **G-A pass, G-B fail, G-C pass** → the S0-11 table would read **PARTIAL**. `t/N` ruler 0.557;
v2's `p_T2` 0.822 on CAP-dev. D0 (CLIP, 8 frames apart): within 0.966 vs across 0.753 raw; 0.857 vs 0.000 centred.

By ego (printed, confounded with box share, S0-12): V7 `e` ego **+0.084** [+0.040, +0.127] vs non-ego +0.030
[−0.023, +0.084]; V7 hit − chance ≈ 0 in every box-share bin.

**Reading (what is licensed).**
1. VideoMAE token change **does not point at the agent**: the most-changed tile hits a box at chance (G-B, both grids).
2. Raw `ρ < 0`: boxed tiles change **less** than the rest of the frame; the periphery (ego-motion) dominates token
   change. Only relative to the same mask before the anomaly do boxed tiles rise (G-A), and mostly on **ego** clips —
   not on the non-ego clips the proposal uses to motivate MQR (§4.3).
3. Local tokens do carry signal the global mean loses for *detection* (G-C), but at a low level (V7 top-k 0.580, close
   to the `t/N` ruler and far below `p_T2`).
4. **V14 > V7** on G-A and D2: the proposal's 2 × 2 pooling to 7 × 7 loses evidence.
5. CLIP patches carry strong local evidence (pointing +0.29) — the opposite of the proposal's expectation (S0-12:
   "a ratio near 1 confirms CLIP patches are the wrong place to read"). **Not a fair encoder comparison yet:** CLIP's
   change compares two single frames at `t`; consecutive V2-S windows share 13 of their 16 frames, **and** each window
   is causal (centre `t − 7.5` frames, ~0.75 s behind the box it is scored against — S0-3 deviated from the proposal's
   centred window to keep G0). Check L (§1.3) tests both: L-1 lag (overlap), L-3 centre shift (offset).

## 3. Amendment 2 re-read and check L — pending (`stage0b_amend2_lag.ipynb`)

Expected before the run (§1.2): **PARTIAL**. Check L interpretation fixed in §1.3 L-2 (lag) and L-3 (centre shift → Stage 1 window).

| | result |
|---|---|
| re-extract 15 clips (`--retry g1_pixel`) | pending |
| §1 read-out, Amendments 1–2 | pending |
| check L, lags 1 / 3 / 6 × shifts 0 / 2 / 3 | pending |
| L-3 window for Stage 1 (centred / causal) | pending |
