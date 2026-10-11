# v2 pilot diagnostics — A0_s2027

Arm crn=none motion=none · step 1740 · T2-val windows 645 · `dota_cap_dev` clips read unlabelled 569 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.6454 |
| macro | 0.6638 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.6877 |
| A0 micro | — |

Guardrails: micro_below_clip_oracle PASS, macro_at_least_micro PASS → **PASS**

Position R² on V^t -0.2156 (input -0.2139)

Source-shortcut AUC (T2-val vs dota_cap_dev): V^t 0.9999, y^bin 0.5524
