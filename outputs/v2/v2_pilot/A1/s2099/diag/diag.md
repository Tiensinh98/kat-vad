# v2 pilot diagnostics — A1_s2099

Arm crn=R2 motion=none · step 1740 · T2-val windows 645 · DoTA-dev clips read unlabelled 702 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.6683 |
| macro | 0.6705 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.6828 |
| A0 micro | 0.6586 |

Guardrails: micro_below_clip_oracle PASS, macro_at_least_micro PASS, micro_within_a0_margin PASS → **PASS**

Position R² on V^t -0.0295 (input -0.0205, A0 -0.2009)

Source-shortcut AUC (T2-val vs DoTA-dev): V^t 0.5507, y^bin 0.5201
