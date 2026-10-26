# v2 pilot diagnostics — A2ign_s2025

Arm crn=none motion=vit_s_k710_dl_from_giant · step 1740 · T2-val windows 645 · `dota_cap_dev` clips read unlabelled 569 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.7240 |
| macro | 0.7071 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.7762 |
| A0 micro | 0.6591 |

Guardrails: micro_below_clip_oracle **FAIL**, macro_at_least_micro **FAIL**, micro_within_a0_margin PASS → **FAIL**

O1' (Amendment 8): Δ window AUC +0.0906, Δ macro +0.0454, Δ micro +0.0649; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**

Position R² on V^t -0.0960 (input -0.1191, A0 -0.2585)

Source-shortcut AUC (T2-val vs dota_cap_dev): V^t 0.9997, y^bin 0.5007

| motion | first | last | max |
|---|---|---|---|
| motion_share | 0 | 0.145 | 0.1612 |
| w_u_norm | 0 | 1.313 | 1.313 |
