# v2 pilot diagnostics — A3_s2099

Arm crn=R2 motion=vit_s_k710_dl_from_giant · step 1740 · T2-val windows 645 · `dota_cap_dev` clips read unlabelled 569 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.7161 |
| macro | 0.7052 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.7637 |
| A0 micro | 0.6586 |

Guardrails: micro_below_clip_oracle **FAIL**, macro_at_least_micro **FAIL**, micro_within_a0_margin PASS → **FAIL**

Position R² on V^t 0.0096 (input 0.0494, A0 -0.2009)

Source-shortcut AUC (T2-val vs dota_cap_dev): V^t 0.5830, y^bin 0.5372

| motion | first | last | max |
|---|---|---|---|
| motion_share | 0 | 0.1075 | 0.1223 |
| w_u_norm | 0 | 1.102 | 1.102 |
