# v2 Amendment 20 — A2ign vs A3ign on DoTA-CAP (n/1397) dev (drop CRN?)

Rule: core/docs/v2/PREREG_ADDENDUM.md §31 (fixed before any A2-ign number). Margin 0.01. **Base = A3ign.**

> Removing CRN is not shown to be harmless on DoTA-CAP-dev at n = 5.

## Decision (Z7)

| check | result |
|---|---|
| z4a_paired_t95_low_gt_minus_margin | FAIL |
| z4b_cluster_ci_low_gt_minus_margin | FAIL |
| z5_f2_ci_low_gt_minus_margin | FAIL |
| z6_guardrails_hold | FAIL |

- macro A3ign 0.7912 · A2ign 0.7898 · `p_T2` 0.8220 · E3 A2 (span, printed) 0.7536
- paired t95 (A2ign - A3ign): +0.0023 [-0.0151, +0.0197] (n = 5); per seed s2024 -0.0057, s2025 -0.0003, s2026 +0.0267, s2027 -0.0015, s2028 -0.0078
- cluster bootstrap, per-clip Δ: -0.0014 [-0.0165, +0.0135]
- `f_2` Δ: -0.0119 [-0.0204, -0.0033] (holds for DoTA-CAP's placement only, pending (bn))

## Guardrails (Z6, T2-val seed means)

| read | A3ign | A2ign | pass |
|---|---|---|---|
| macro | 0.7338 | 0.7068 | False |
| normal-window peak (printed) | 0.6151 | 0.4271 | — |
| O1' every seed | — | {'s2024': True, 's2025': True, 's2026': True, 's2027': True, 's2028': True} | True |

## Printed only (Z8)

- per-clip Δ by native-fps group: other +0.0255 [-0.1134, +0.1664]; r1 +0.0012 [-0.0168, +0.0188]; r3 -0.0079 [-0.0293, +0.0128]
- `f_2` macros: A3ign 0.8526, A2ign 0.8407
- source shortcut AUC (`v^t`): A3ign 0.5838, A2ign 0.9997
- position R² (`v^t`): A3ign -0.0070, A2ign -0.0820
