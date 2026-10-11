# v2 pilot diagnostics — A0_s2099

Arm crn=none motion=none · step 1740 · T2-val windows 645 · DoTA-dev clips read unlabelled 702 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.6586 |
| macro | 0.6737 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.6863 |
| A0 micro | — |

Guardrails: micro_below_clip_oracle PASS, macro_at_least_micro PASS → **PASS**

Position R² on V^t -0.2009 (input -0.2139)

Source-shortcut AUC (T2-val vs DoTA-dev): V^t 0.9999, y^bin 0.5375
