# v2 pilot batch 1 (seed 2099) — T2-val only (D4)

| arm | T2-val micro | macro | clip oracle | window AUC | guardrails | pos R² V^t | shortcut V^t | shortcut y^bin |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.6586 | 0.6737 | 0.6986 | 0.6863 | PASS | -0.2009 | 0.9999 | 0.5375 |
| A1 | 0.6683 | 0.6705 | 0.6986 | 0.6828 | PASS | -0.0295 | 0.5507 | 0.5201 |

D5 (O6): FAIL -- micro Δ -0.1089, macro Δ -0.0073 vs the phase-4 mean (in-sample for them)
