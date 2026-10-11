# v2 pilot diagnostics — A1_s2026

Arm crn=R2 motion=none · step 1740 · T2-val windows 645 · `dota_cap_dev` clips read unlabelled 569 (no DoTA metric computed, D4).

| T2-val | value |
|---|---|
| micro (norm none) | 0.6615 |
| macro | 0.6469 |
| clip oracle micro | 0.6986 |
| window-level AUC (C14 signature) | 0.6996 |
| A0 micro | 0.6417 |

Guardrails: micro_below_clip_oracle PASS, macro_at_least_micro **FAIL**, micro_within_a0_margin PASS → **FAIL**

O1' (Amendment 8): Δ window AUC +0.0279, Δ macro +0.0142, Δ micro +0.0198; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**

Position R² on V^t -0.0314 (input -0.0205, A0 -0.2206)

Source-shortcut AUC (T2-val vs dota_cap_dev): V^t 0.5530, y^bin 0.5454
