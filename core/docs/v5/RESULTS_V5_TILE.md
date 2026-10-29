# v5.1 — tile-scored MQR, one-seed pilot on DoTA-CAP-dev: **PILOT-FAIL**

Rules: `core/docs/v5/PREREG_V5.md` **§3** (signed 2026-10-29, before any v5.1 number). Plan:
`.project/plans/katvad-v5-tile-mqr.md`. Setup: `core/docs/v5/V51_TILE_SETUP.md`. Runbook:
`colab/v5/v51_tile_pilot.ipynb`. Raw outputs: `outputs/v5/v5_tile/REPORTS/` (`v51_pilot_readout`, `maps_readout`,
`g4_provenance`, `g_c_dota_tile`, `g1_extract_tile_s{3,8}`, `head`, `extract_tile_*`; `.md` + `.json`); untracked.
Rows: R1c (mean of the 196 cells), R3c (untrained change top-K), R6c (tile scorer + `L_tile`, dense negatives); seed
2024; λ 0.5, K 4, β 0.01 fixed (§3 T7). R5c built, **not run** (§3, closed at signing). Outcome label:
**PRELIMINARY (n = 1 seed)**.

## 1. Outcome: **PILOT-FAIL for R6c and R3c → v5.1 is a negative result; no tuning run follows (§3.3)**

| row | (i) maps beat centre **and** change | (ii) `nexar_val` ≥ R1c − 0.01 | (iii) pos. (b) Δ high ≥ 0 | outcome |
|---|---|---|---|---|
| **R6c** | ✗ hit − centre **−0.345** [−0.381, −0.312]; hit − change **−0.046** [−0.067, −0.023] | ✗ 0.8746 vs 0.8985 (**−0.024**) | ✓ +0.0024 [−0.0019, +0.0067] | **PILOT-FAIL** |
| R3c | ✗ hit − centre −0.299 [−0.341, −0.259] | ✗ 0.7963 (−0.102) | ✗ −0.0120 [−0.0176, −0.0063] | PILOT-FAIL |

§3.4 (seeds 2025/2026, DoTA-CAP-eval) is not run. **DoTA-CAP-eval stays unread by every v5 row.**

## 2. Validity

- S1-G4 provenance PASS: R1c / R3c / R6c end at step 1749 = their last `metrics.jsonl` step (33 epochs × 53); config
  differs from v2 Nexar A2 only in `v5.*`.
- G-c (DoTA-CAP centred 4 × 7 × 7): 1129 / 1129, min cos 0.99999999999 vs v2 `DoTA_CAP_s1_squash` row `t + 7`.
- S1-G1: `nexar_train` s8 563 / 563, `nexar_val` s3 225 / 225 (length-gated; min cos 0.99946 / 0.99952 printed, B9).
- Shift test: peak at 0 for R1c and R6c, −1 for R3c — all within tolerance, so the centred window is not mis-centred.
- Losses finite in all three runs.

## 3. Reads (DoTA-CAP-dev, protocol B, 569 clips; 95 % cluster bootstrap over source videos)

### 3.1 Detection

| row | macro | Δ vs R1c | TOST ±0.01 | pos. (a) | pos. (b) | gain beyond `p` | Δ (b) vs R1c | AP clip-macro | `nexar_val` |
|---|---|---|---|---|---|---|---|---|---|
| R1c | 0.8124 | — | — | 0.6400 | 0.8755 | +0.0260 | — | 0.7137 | 0.8985 |
| R3c | 0.7505 | −0.0618 [−0.0776, −0.0467] | different | 0.6477 | 0.8636 | +0.0140 | −0.0120 [−0.0176, −0.0063] | 0.6459 | 0.7963 |
| R6c | 0.8047 | −0.0076 [−0.0168, +0.0017] | inconclusive | 0.6580 | 0.8779 | +0.0284 | +0.0024 [−0.0019, +0.0067] | 0.7123 | 0.8746 |

- **Position alone (cubic `p`, fitted on CAP-dev) = 0.8496 — above every row's macro.** R1c adds +0.026 beyond it.
- Time-reversed − forward macro: R1c −0.036, R3c −0.025, R6c −0.024 → all rows use temporal order (printed).
- Subsets (ego / box share): R6c − R1c is inside ±0.011 with CIs on 0 in every subset; R3c's point estimate is below R1c's in all of them.

### 3.2 Maps (528 boxed clips)

| row | hit − chance | hit − centre | hit − change | mass − chance | mass − change | STAUC | uniform | centre | change |
|---|---|---|---|---|---|---|---|---|---|
| R3c | +0.003 [−0.016, +0.021] | −0.299 [−0.341, −0.259] | 0 (its map *is* the change map) | +0.0035 | 0 | 0.2127 | 0.2014 | **0.2642** | 0.2127 |
| R6c | **−0.043** [−0.053, −0.032] | −0.345 [−0.381, −0.312] | −0.046 [−0.067, −0.023] | +0.0061 [+0.0040, +0.0083] | +0.0026 [+0.0004, +0.0050] | 0.2110 | 0.2009 | **0.2641** | 0.2109 |

- The centre prior's own hit − chance on these 528 clips is **+0.303** (exact: (hit − chance) − (hit − centre), same
  clips, both rows agree to 1e-4).
- R6c's map points **away from** the boxes (hit − chance < 0) on ego (−0.029) and non-ego (−0.058) clips and in
  three of four box-share bins.
- Amendment 1's lift − lift_pre: R3c +0.032 [+0.014, +0.050] (the change map rises inside the box during the
  anomaly, as Stage 0's G-A), R6c +0.012 [−0.001, +0.025].

### 3.3 Mechanics (last step; printed)

| run | mil | tile | `p` abnormal | `p` normal | T | entropy / log 196 | motion share |
|---|---|---|---|---|---|---|---|
| R1c | 0.1126 | — | — | — | — | — | 0.1432 |
| R3c | 0.2085 | — | — | — | — | 0.2626 | 0.2589 |
| R6c | 0.1315 | 0.1818 | **0.1220** | **0.1258** | 1.0153 | 0.6979 | 0.1906 |

## 4. Diagnosis (printed, not gated; inference marked)

1. **The tile scorer does not separate abnormal from normal cells.** Mean `p` is 0.122 on abnormal vs 0.126 on normal
   sequences at the last step, and `T` stayed at ≈ 1.0. `L_tile`'s dense negatives pushed every cell down rather than
   making in-event cells stand out. The readout is not uniform (entropy 0.70·log 196), so it concentrates on
   something — and the maps say that something is off the boxes. *(Inference: periphery / ego-motion tiles, as Stage 0's
   raw ρ < 0 showed for V2-S token change.)*
2. **Detection is R1c's.** R6c ≈ R1c on macro, AP and position-controlled AUC; the extra motion share (0.19 vs 0.14)
   buys nothing and costs `nexar_val` 0.024.
3. **The untrained change map on the centred 4 × 7 × 7 grid does not point** (+0.003), unlike Stage 0's lag-6 V7
   change (+0.076). *(Hypothesis, not measured: Δ between consecutive kept slots spans 4 frames — 0.4 s at DoTA step 1
   — far shorter than Stage 0's 18-frame lag, and the `|R|` term is dominated by static salient regions.)* Picking the
   most-changed cells (R3c) costs 0.06 macro — consistent with Stage 0 §3 point 3 ("where" ≠ "when").
4. **The centre prior beats every map read in v5** (STAUC 0.264 vs ≤ 0.213; hit − chance +0.30). On DoTA-CAP-dev boxes,
   any pointing claim must be read against it, never against chance alone.

## 5. What is licensed

- "The v5.1 tile-scored readout (R6c), trained on Nexar with weak labels, failed its pre-registered pilot gate: its
  maps point no better than chance and far below a centre prior on DoTA-CAP-dev, and it does not change detection
  (Δ −0.008, inconclusive at ±0.01, n = 1 seed)."
- "In the v5.1 pilot, selecting the most-changed VideoMAE cells (R3c) lowers detection (−0.062)."

**Not licensed:** any "MQR / tile scoring improves detection" or "localizes the agent" sentence; any equivalence claim
(TOST inconclusive, one seed); any statement about DoTA-CAP-eval; any R5c or tuned-λ/K/β statement (not run).

## 6. What follows

- §3.3: no tuning run, no further seeds, R5c not run.
- **Check C** (`PREREG_V5.md` §3.7, written after this read-out, post-hoc): CLIP-patch change maps (Stage 0's C14)
  vs the centre prior, paired on the same grid — the only remaining argument for R7. Training-free;
  `core/tools/v5_centre_check.py`, `colab/v5/check_c_centre.ipynb`. **CENTRE-WINS → v5 closed; CLIP-BEATS-CENTRE → R7
  eligible for an amendment (MAPS-ONLY at most), on the user's word.** Result: *pending.*
- Model of record unchanged: **`B` = A3-ign** (v2).
