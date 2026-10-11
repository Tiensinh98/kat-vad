# v2 pilot diagnostics — A2_s2025

Arm crn=none motion=vit_s_k710_dl_from_giant · step 1740 · T2-val windows 645 · `dota_cap_dev` clips read unlabelled 569 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.6939 |
| macro | 0.6847 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.7685 |
| A0 micro | 0.6591 |

Guardrails: micro_below_clip_oracle PASS, macro_at_least_micro **FAIL**, micro_within_a0_margin PASS → **FAIL**

O1' (Amendment 8): Δ window AUC +0.0829, Δ macro +0.0230, Δ micro +0.0348; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**

Position R² on V^t -0.0962 (input -0.1191, A0 -0.2585)

Source-shortcut AUC (T2-val vs dota_cap_dev): V^t 0.9998, y^bin 0.5237

| motion | first | last | max |
|---|---|---|---|
| motion_share | 0 | 0.1253 | 0.1436 |
| w_u_norm | 0 | 1.249 | 1.249 |
