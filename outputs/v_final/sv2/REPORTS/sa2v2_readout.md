# SA²-v2 read-out: SV2 vs W0, Nexar-trained, seed 2024

Rule: `core/docs/final/SA2V2_RUN.md §3-§4`, written before any number. PRELIMINARY (n = 1 seed): adopts nothing.

## DADA-2000 (370 sources, whole video, s8)

| Row | Macro | Micro | Clip AUC | Middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] | 0.6981 | — | 0.7342 [0.7076, 0.7594] |
| SV2 | 0.7133 [0.6883, 0.7379] | 0.6661 | — | 0.7342 [0.7076, 0.7594] |

**SV2 - W0 (per video): -0.0482 [-0.0674, -0.0290] → cost shown, PRELIMINARY (n = 1 seed).**

Printed: first SA² macro 0.7579 [0.7334, 0.7820]; SV2 - R4 -0.0446 [-0.0625, -0.0268].

Check: dump vs frame read, max |Δ score| = 0.00e+00.

## Controls

* **C1** head on `x^AS` 0.6488 [0.6275, 0.6703] vs same head on the cell mean 0.4855 [0.4639, 0.5074]; Δ +0.1634 [+0.1382, +0.1885] → gain shown (on DADA, rule D5).
* **C3** top-3 q on abnormal 0.5526 vs normal rows 0.3737; Δ +0.1789 [+0.1584, +0.1994] over 370 videos.

### C2 / C4: training diagnostics (first → mean of the last tenth)

| Diagnostic | SV2 | Shuffled | Moved further than shuffled? |
| --- | --- | --- | --- |
| sv2_beta1 | 1.0000 → 0.9941 | pending | pending |
| sv2_beta2 | 0.0000 → -0.0004 | pending | pending |
| sv2_beta3 | 0.0000 → -0.0035 | pending | pending |
| sv2_beta4 | 0.0000 → 0.0001 | pending | pending |
| sv2_gamma | 0.0000 → -0.0010 | pending | pending |
| sv2_temperature | 1.0000 → 0.9794 | pending | pending |
| sv2_bias | -2.0000 → -2.0006 | pending | pending |
| sv2_w_norm | 0.0000 → 0.1448 | pending | pending |
| sv2_entropy | 0.8280 → 0.6247 | pending | pending |
| sv2_spread | 1.0000 → 3.1971 | pending | pending |
| sv2_xas_ratio | 0.5547 → 0.5909 | pending | pending |
| sv2_top3q_abnormal | 0.5333 → 0.6629 | pending | pending |
| sv2_top3q_normal | 0.5468 → 0.3476 | pending | pending |
| sv2_patch | 0.7420 → 0.5446 | pending | pending |
| sv2_branch | 0.6928 → 0.6062 | pending | pending |

## DoTA-CAP-dev

Macro: SV2 0.7848 [0.7681, 0.8016], W0 0.8110 [0.7949, 0.8271].

**SV2 - W0 (per clip): -0.0262 [-0.0400, -0.0122] → cost shown, PRELIMINARY (n = 1 seed).**

# v5 S2-8 maps read-out

Rule: core/docs/v5/PREREG_V5.md S2-8 + S1-G5

| row | D3 hit - chance | lift - lift_pre (Am. 1) | hit - centre prior | points? | STAUC | AUC |
|---|---|---|---|---|---|---|
| Mo | +0.293 [+0.271, +0.316] (528) | +0.250 [+0.228, +0.272] (522) | -0.010 [-0.051, +0.030] (528) | no | 0.3343 | 0.7225 |
| SV2 | +0.148 [+0.120, +0.176] (528) | +0.200 [+0.173, +0.226] (522) | -0.154 [-0.201, -0.107] (528) | no | 0.2844 | 0.7225 |
| centre | +0.303 [+0.270, +0.337] (528) | -0.024 [-0.056, +0.007] (522) | +0.000 [+0.000, +0.000] (528) | no | 0.2677 | 0.7225 |
| uniform | -0.151 [-0.163, -0.139] (528) | -0.033 [-0.043, -0.022] (522) | -0.453 [-0.492, -0.416] (528) | no | 0.2544 | 0.7225 |

Printed beside: Stage 0 V7 change at lag 6 hit - chance +0.076; uniform map (R1) = 0.

- Mo by ego: ego +0.329 [+0.298, +0.360] (282), non_ego +0.251 [+0.214, +0.287] (246); by bin: <0.5% +0.060 [+0.022, +0.104] (41), 0.5-2% +0.238 [+0.190, +0.286] (116), 2-8% +0.359 [+0.313, +0.404] (166), >8% +0.316 [+0.283, +0.352] (205)
- SV2 by ego: ego +0.283 [+0.247, +0.318] (282), non_ego -0.006 [-0.029, +0.017] (246); by bin: <0.5% -0.032 [-0.048, -0.012] (41), 0.5-2% -0.005 [-0.027, +0.018] (116), 2-8% +0.090 [+0.052, +0.128] (166), >8% +0.319 [+0.277, +0.359] (205)
- centre by ego: ego +0.252 [+0.212, +0.292] (282), non_ego +0.361 [+0.315, +0.409] (246); by bin: <0.5% +0.365 [+0.234, +0.508] (41), 0.5-2% +0.282 [+0.224, +0.339] (116), 2-8% +0.300 [+0.241, +0.364] (166), >8% +0.304 [+0.257, +0.352] (205)
- uniform by ego: ego -0.185 [-0.203, -0.165] (282), non_ego -0.112 [-0.124, -0.099] (246); by bin: <0.5% -0.049 [-0.055, -0.043] (41), 0.5-2% -0.086 [-0.092, -0.080] (116), 2-8% -0.143 [-0.151, -0.134] (166), >8% -0.215 [-0.242, -0.186] (205)

