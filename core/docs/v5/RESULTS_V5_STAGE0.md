# v5 Stage 0 — do VideoMAE V2-S tokens carry local anomaly evidence? (results)

Rules: `core/docs/v5/PREREG_V5.md` §1 (S0-1 … S0-12, signed before any grid existed), §1.1 Amendment 1 (before any
grid), **§1.2 Amendment 2 and §1.3 check L (both written after the first read-out, disclosed there)**.
Plan: `.project/plans/katvad-v5-mqr.md`. Runbooks: `colab/v5/stage0_dota_cap_tokens.ipynb` (first run),
`colab/v5/stage0b_amend2_lag.ipynb` (Amendment 2 re-extract + re-read + check L).
Raw outputs: `outputs/v5/v5_stage0/REPORTS/stage0_{readout,lag}.{md,json}` (gitignored; this file is the record). The first
read-out's local copy was replaced by the Amendment 2 re-read.

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

## 3. Amendment 2 re-read (2026-10-27): **PARTIAL** — valid, as recorded in §1.2 before the run

`colab/v5/stage0b_amend2_lag.ipynb` → `outputs/v5/v5_stage0/REPORTS/stage0_readout.{md,json}` (gitignored).
Validity: dev maps **569 / 569**, failed 0, missing 0; G0 min cos **0.99999999998**. Gates (S0-11): V7 and V14 both
**G-A ✓ · G-B ✗ · G-C ✓** → **PARTIAL** (the user decides; §1 not changed).

| read (n clips) | V7 | V14 | C14 CLIP patches (printed) |
|---|---|---|---|
| D1 `e = ρ − ρ_pre` (G-A) | **+0.059** [+0.025, +0.092] (521) | +0.086 [+0.052, +0.118] | +0.415 [+0.376, +0.455] |
| D1 `e`, non-ego | +0.032 [−0.021, +0.085] (244) | +0.050 [−0.006, +0.109] | +0.467 [+0.398, +0.533] |
| D1 raw `ρ` | **−0.083** [−0.114, −0.053] (528) | −0.142 [−0.171, −0.114] | +0.532 [+0.497, +0.567] |
| D3 hit − chance (G-B) | **+0.003** [−0.013, +0.020] (528) | −0.012 [−0.025, +0.002] | +0.288 [+0.269, +0.307] |
| D3 lift − lift_pre (G-B, Am. 1) | +0.013 [−0.007, +0.035] (521) | +0.001 [−0.013, +0.016] | +0.216 [+0.197, +0.234] |
| D2 top-k − global, non-ego (G-C) | **+0.036** [+0.021, +0.051] (248) | +0.077 [+0.058, +0.100] | +0.009 [−0.007, +0.026] |
| D2 top-k − global, all | +0.042 [+0.032, +0.051] (569) | +0.090 [+0.077, +0.103] | +0.018 [+0.007, +0.029] |
| D2 macro top-k / max / global | 0.581 / 0.573 / 0.540 | 0.630 / 0.580 / 0.540 | 0.718 / 0.682 / 0.700 |

`t/N` ruler 0.557, `p_T2` 0.822. By ego (printed, confounded with box share): V7 `e` ego +0.082 [+0.039, +0.124] vs
non-ego +0.032 [−0.021, +0.085]; V7 hit − chance ≈ 0 in every box-share bin. The 15 re-extracted clips moved no gate;
every number is within 0.003 of §2.

## 4. Check L (§1.3, printed, never gating) — `REPORTS/stage0_lag.{md,json}`

Validity: lag 1 (fp16 grid) vs stored fp32 `d7` max |Δ| **1.15e-4** (regression passes). V7 only (V14 grids are not
stored). Shift `s` = grid row read against step `k` is `k + s` (window centre `t − 7.5`, `t − 1.5`, `t + 1.5` frames).

| lag ℓ (frames) | s | D1 `e` | D1 `e` non-ego | D3 hit − chance | D3 lift − lift_pre | D2 top-k − global non-ego | macro top-k | G-A | G-B | G-C |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (3) | 0 | +0.059 [+0.025, +0.092] | +0.032 [−0.021, +0.085] | +0.003 [−0.013, +0.020] | +0.013 [−0.007, +0.034] | +0.035 [+0.021, +0.051] | 0.581 | ✓ | ✗ | ✓ |
| 1 (3) | 2 | +0.076 [+0.038, +0.113] | +0.085 [+0.030, +0.142] | −0.003 [−0.020, +0.013] | +0.009 [−0.010, +0.028] | +0.031 [+0.015, +0.049] | 0.640 | ✓ | ✗ | ✓ |
| 1 (3) | 3 | +0.087 [+0.050, +0.124] | +0.096 [+0.041, +0.153] | −0.000 [−0.017, +0.017] | +0.017 [+0.000, +0.034] | +0.028 [+0.011, +0.046] | 0.643 | ✓ | ✗ | ✓ |
| 3 (9) | 0 | +0.153 [+0.120, +0.184] | +0.140 [+0.086, +0.193] | +0.055 [+0.036, +0.076] | +0.049 [+0.028, +0.071] | +0.030 [+0.015, +0.045] | 0.486 | ✓ | ✓ | ✓ |
| 3 (9) | 2 | +0.143 [+0.110, +0.177] | +0.163 [+0.108, +0.220] | +0.058 [+0.040, +0.077] | +0.046 [+0.026, +0.066] | +0.033 [+0.019, +0.048] | 0.580 | ✓ | ✓ | ✓ |
| 3 (9) | 3 | +0.119 [+0.086, +0.151] | +0.150 [+0.098, +0.206] | +0.047 [+0.029, +0.067] | +0.033 [+0.016, +0.051] | +0.032 [+0.018, +0.047] | 0.622 | ✓ | ✓ | ✓ |
| **6 (18)** | 0 | +0.171 [+0.138, +0.203] | +0.173 [+0.115, +0.230] | **+0.076** [+0.057, +0.096] | **+0.058** [+0.034, +0.082] | +0.026 [+0.009, +0.042] | **0.391** | ✓ | ✓ | ✓ |
| 6 (18) | 2 | +0.195 [+0.160, +0.230] | +0.216 [+0.157, +0.277] | +0.102 [+0.081, +0.123] | +0.074 [+0.049, +0.098] | +0.031 [+0.014, +0.048] | 0.476 | ✓ | ✓ | ✓ |
| 6 (18) | 3 | +0.192 [+0.158, +0.226] | +0.222 [+0.167, +0.277] | +0.106 [+0.084, +0.128] | +0.078 [+0.053, +0.103] | +0.031 [+0.015, +0.047] | 0.523 | ✓ | ✓ | ✓ |

**L-2 (fixed before the run): fires.** At ℓ = 6 (no shared frame) both G-B intervals have low > 0 → Stage 0's pointing
failure is **attributed to window overlap**; the next design must compare non-overlapping windows.
**L-3 (fixed before the run): causal.** `s = 0` passes G-B at ℓ = 3 and 6 as well, so the failure is not the causal
offset → Stage 1 keeps the causal window (v2-comparable; Nexar is not re-extracted centred; row 1 can be borrowed).

## 5. Reading (what is licensed)

1. **Decision stays PARTIAL.** Check L is post-hoc and printed; write "PARTIAL; a post-hoc check attributes the
   pointing failure to window overlap", never "Stage 0 GO".
2. **VideoMAE V7 change between non-overlapping windows points at the agent:** hit − chance +0.08 … +0.11, `e` +0.17 …
   +0.22, and **stronger on non-ego clips** (+0.17 … +0.22) than at ℓ = 1 (+0.03, ns). The effect grows monotonically
   with ℓ in all six ℓ ≥ 3 cells, so it is not one cell picked out of nine.
3. **"Where" improves with lag, "when" gets worse.** Training-free top-k macro falls 0.581 → 0.486 → **0.391**
   (ℓ = 1 → 3 → 6, `s = 0`; below chance at ℓ = 6). Lagged change is high where the agent is, but its magnitude does not
   time the anomaly within a clip. Stage 0 gives no support to change magnitude as a detection score; MQR's "when" has to
   come from the trained heads.
4. **CLIP patches still point harder** (+0.29 at 3 frames vs V7 +0.08 … +0.11 at 18 frames). Not matched in lag (C14 at
   ℓ = 6 needs pixels; not run). The proposal's §4.3 expectation ("CLIP patches are the wrong place to read") is not
   supported by any Stage 0 number.
5. **Stage 0 tested change, not content.** No Stage 0 number says whether the static grid `P_t` (what the proposal's
   MQR reads) carries pointing evidence. Stage 2 (`PREREG_V5.md` §2) tests both a content-keyed and a change-keyed
   readout for that reason.
6. All reads are training-free and far below `p_T2` 0.822: Stage 0 says nothing about detection.
