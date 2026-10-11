# v2 pilot diagnostics — A3_s2027

Arm crn=R2 motion=vit_s_k710_dl_from_giant · step 1740 · T2-val windows 645 · `dota_cap_dev` clips read unlabelled 569 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.7104 |
| macro | 0.7104 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.7474 |
| A0 micro | 0.6454 |

Guardrails: micro_below_clip_oracle **FAIL**, macro_at_least_micro **FAIL**, micro_within_a0_margin PASS → **FAIL**

O1' (Amendment 8): Δ window AUC +0.0597, Δ macro +0.0466, Δ micro +0.0651; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**

Position R² on V^t 0.0070 (input 0.0494, A0 -0.2156)

Source-shortcut AUC (T2-val vs dota_cap_dev): V^t 0.5739, y^bin 0.5381

| motion | first | last | max |
|---|---|---|---|
| motion_share | 0 | 0.1037 | 0.1242 |
| w_u_norm | 0 | 1.094 | 1.094 |
