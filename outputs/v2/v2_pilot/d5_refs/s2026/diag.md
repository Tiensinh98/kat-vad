# v2 pilot diagnostics — p4_s2026

Arm crn=none motion=none · step 2040 · T2-val windows 645 · DoTA-dev clips read unlabelled 702 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.7567 |
| macro | 0.6524 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.9613 |
| A0 micro | — |

Guardrails: micro_below_clip_oracle **FAIL**, macro_at_least_micro **FAIL** → **FAIL**

Position R² on V^t -0.2198 (input -0.2139)

Source-shortcut AUC (T2-val vs DoTA-dev): V^t 0.9999, y^bin 0.5429
