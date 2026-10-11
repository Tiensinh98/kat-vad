# v2 pilot diagnostics — A3ign_s2024

Arm crn=R2 motion=vit_s_k710_dl_from_giant · step 1740 · T2-val windows 645 · `dota_cap_dev` clips read unlabelled 569 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.7444 |
| macro | 0.7366 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.7824 |
| A0 micro | 0.6446 |

Guardrails: micro_below_clip_oracle **FAIL**, macro_at_least_micro **FAIL**, micro_within_a0_margin PASS → **FAIL**

O1' (Amendment 8): Δ window AUC +0.1093, Δ macro +0.0893, Δ micro +0.0998; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**

Position R² on V^t -0.0011 (input 0.0494, A0 -0.2265)

Source-shortcut AUC (T2-val vs dota_cap_dev): V^t 0.5881, y^bin 0.5174

| motion | first | last | max |
|---|---|---|---|
| motion_share | 0 | 0.1293 | 0.1438 |
| w_u_norm | 0 | 1.161 | 1.161 |
