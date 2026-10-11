# v2.1 E4a — SG-NM gate on B = A3ign, DoTA-CAP (n/1397) dev

Rule: core/docs/v2.1/KAT_VAD_PROPOSAL_v2.1.md §11.2 step 5; plan katvad-v2.1-e4a.md §5. **SG-NM DROPPED — v2.1 is B.**

| check | result |
|---|---|
| delta_t95_above_0 | FAIL |
| f2_point_ge_0 | FAIL |

- Δ = macro(rank-avg(y^bin, S~)) - macro(y^bin), paired t95: -0.0119 [-0.0161, -0.0077] (n = 5); per seed s2024 -0.0140, s2025 -0.0165, s2026 -0.0081, s2027 -0.0093, s2028 -0.0116
- `f_2` Δ (seed-averaged, cluster CI): -0.0033 [-0.0060, -0.0006]

## Printed, never gated

| seed-averaged macro | value |
|---|---|
| y^bin (B) | 0.7912 |
| S~ alone | 0.7524 |
| rank-avg | 0.7845 |
| `p_T2` | 0.8220 |

- corr(AUC(S~), AUC(`p_T2`)) = 0.140 (§7: > 0.4 reads as position; B's y^bin: 0.147) · corr with y^bin = 0.850
- T2-val Δ per seed: -0.0097, -0.0175, -0.0083, -0.0123, -0.0133 (mean -0.0122)
- prototype usage (max share, mean over clips; > 0.95 = collapse): s2024 0.754, s2025 0.618, s2026 0.714, s2027 0.750, s2028 0.780
- G0 |Δ| vs protocol B: s2024 2.8e-06, s2025 9.1e-07, s2026 2.1e-06, s2027 3.1e-06, s2028 1.4e-06

The gate's Δ is also a test-time ensemble's gain; v2.1 does not deploy it (§5.3).
