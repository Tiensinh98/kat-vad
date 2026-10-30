# v5 R7 fast pilot — MQR on CLIP 14 × 14 patches: **NULL-PILOT** (PRELIMINARY, n = 1)

Rule: `core/docs/v5/PREREG_V5.md` **§4** (§4.2 recommended column, §4.4 fast pilot F1–F4, §4.5 R7-A1), fixed
2026-10-30 before any R7 number. Runbook `colab/v5/r7_fast_pilot.ipynb`, plan `.project/plans/katvad-v5-r7-clip.md`.
Raw read-outs `outputs/v5/v5_r7/REPORTS/{r7_pilot_readout,maps_readout,g4_provenance,g_r7_dota_cgrid}.md` and
`g_r7c_repro/check_c_readout.md`; maps dump `outputs/v5/v5_r7/maps/maps_R7t_s2024.npz`. Read out 2026-10-31.

Rows (s2024, A2's recipe, 1,749 steps, `h = x + W_u·u + W_c·LN(r)`, `W_c` zero-init): **R7m** = mean of the 196
CLIP cells (control), **R7t** = tile scorer on `[LN(G − med) ; LN(ΔG − med)]`, ΔG lag 1 row, `L_tile` dense
negatives, λ 0.5 / K 4 / β 0.01. Read on DoTA-CAP-dev (569 clips; 528 with boxes), protocol B. `nexar_val` not read.

## 1. Outcome

**NULL-PILOT (maps)** — none of P-e's four 95 % lows is > 0; three are < 0. Per §4.4 F4 this is reported as a
**one-seed negative**; P-g's seeds 2025 / 2026 and CAP-eval are not run (they follow a PASS only).

| R7t map read (14 × 14, 528 clips) | value [95 %] | P-e needs |
|---|---|---|
| hit − chance | **−0.083** [−0.096, −0.069] | low > 0 ✗ |
| lift − lift_pre | −0.020 [−0.030, −0.010] | low > 0 ✗ |
| hit − C14 change map | **−0.371** [−0.395, −0.346] | low > 0 ✗ |
| mass − centre prior | −0.033 [−0.039, −0.028] | low > 0 ✗ |
| mass − chance (printed) | +0.002 [−0.000, +0.005] | — |
| STAUC (printed) | 0.196 vs uniform 0.205, centre 0.269 | — |

By subset (hit − chance, printed): ego −0.091, non-ego −0.073; box share < 0.5 % −0.023, 0.5–2 % −0.043, 2–8 %
−0.080, **> 8 % −0.120** — the larger the box, the further below chance.

## 2. Validity (all pass)

- **G1** Nexar train s8: 563 / 563 videos, 81,538 rows, min cos vs `Nexar_s1_ncc` 0.99994, max |CLS − `image_embeds`| 0.
  DoTA-CAP-dev: 569 / 569 clips, 19,180 rows, G1-pixel min cos 0.954 (≥ 0.95).
- **G-R7c** (printed per the user, §4.5): check C re-run on the R7 cache reproduces C14 hit − chance **+0.2877** vs
  Stage 0's +0.2878 (|Δ| 1.1e-4, tol 0.01) and again reads **CLIP-BEATS-CENTRE** (hit − centre +0.131, mass − centre
  +0.020). The R7 maps are therefore read against the same C14 that check C measured.
- **S1-G4** provenance: R7m / R7t checkpoints at step 1749 = their last `metrics.jsonl` step. Last-step losses finite.

## 3. Detection (printed only, §4.2 P-h — never a claim)

Position alone (cubic `p`, fitted on CAP-dev): **0.8496**.

| row | macro | Δ vs R7m | pos. (b) | Δ (b) vs R7m | gain beyond `p` | reversed − forward | AP clip-macro |
|---|---|---|---|---|---|---|---|
| R7m | 0.7804 | — | 0.8656 | — | +0.0160 | −0.051 | 0.6612 |
| R7t | 0.8000 | +0.0197 [+0.0103, +0.0295] | 0.8668 | **+0.0012 [−0.0040, +0.0062]** | +0.0172 | −0.079 | 0.6800 |

- R7t − R7m is entirely position: the raw gain (+0.020, TOST "different") vanishes under position control (b).
- Both rows sit **below v5.1's R1c** (0.8124, pos. (b) 0.8755; same seed and recipe, separate run — printed cross-run
  reference from `RESULTS_V5_TILE.md`, not a paired contrast). The CLIP branch takes 38–44 % of the residual
  (`clip_share` R7t 0.376 / R7m 0.438) and the trunk reads worse with it.
- Reversed − forward < 0 for both: temporal order is used.

## 4. Why the map fails — post-hoc diagnostic (printed, not gated; written after the read-out)

Computed by Claude on `maps_R7t_s2024.npz` (`E_*` = R7t tile map, `C_*` = the C14 change map), 569 clips,
18,611 steps:

1. **The map lives on the frame border.** 93 % of per-step argmaxes fall in rows 0–2 (59 %: sky / top-left corner,
   tile (1, 0) alone 15.5 %) or rows 11–13 (39 %: hood / dashboard); rows 3–10, where road users are, get ≈ 1 %.
   Per clip, the median share of border argmaxes is 1.00 and 97.9 % of clips have ≥ 80 %. Mean mass on those six rows
   is 0.571 vs 0.429 for a uniform map. The C14 change map's argmax is spread (top tile ≈ 1 %, mid rows).
2. **It is not a sign flip.** Per-step Spearman(R7t map, C14) median **−0.18** (p10 −0.36, p90 +0.01): weakly
   anti-correlated, not ≈ −1 as a sign bug would give.
3. **The scorer did not learn abnormal vs normal at the tile level.** Last training step: `p_abnormal` 0.117 vs
   `p_normal` 0.116; map entropy on DoTA-CAP-dev 0.91 × log 196 (0.63 on train).

**Hypothesis (untested):** the scorer keys on CLIP ViT's high-norm "artifact" patch tokens, which sit in
low-information background (sky, hood) and change position frame to frame — `LN(G − med)` and `ΔG` make them the most
salient cells. A per-row patch-norm print on CAP-dev would test it; it would explain the failure, not change it.

## 5. What is licensed

- "A tile scorer trained with video-level weak labels on CLIP 14 × 14 patches (Nexar, one seed) does not point at the
  agent on DoTA-CAP-dev: its map is below chance (hit − chance −0.083), below the centre prior and far below the
  untrained CLIP change map (−0.371); its mass concentrates on the frame border."
- Across v5: **no trained map** (V2-S token queries, v5.1 tile, R7 CLIP tile) points beyond a centre prior on
  DoTA-CAP-dev; the only map that does is the **untrained** CLIP-patch change map C14, and only at 14 × 14
  (`RESULTS_V5_CHECK_C.md`).

**Not licensed:** any detection gain for R7t (position-explained); any statement about CLIP patches "not containing"
spatial evidence (C14 shows they do); the artifact-token mechanism (untested); anything at n > 1 seed or on CAP-eval.

## 6. Consequence

v5's spatial track is closed by its own rules: §3.7 C-6 capped R7 at MAPS-ONLY, and R7 reads NULL. Re-running R7 with
border masking, registers or other features after seeing this map would be tuning on the read-out (lesson 14); it
needs a new amendment declared as a forking path. `r7_clip_grids.ipynb` (`nexar_val` + CAP-eval grids, ≈ 41 GB) is
not needed. The thesis' adopted model stays v2's `B` = A3-ign.
