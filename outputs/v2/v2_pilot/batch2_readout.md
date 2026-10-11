# v2 pilot batch 2 (seed 2099, encoder vit_s_k710_dl_from_giant) — T2-val only (D4)

| arm | T2-val micro | macro | clip oracle | window AUC | guardrails | pos R² V^t | shortcut V^t (CAP) | shortcut y^bin (CAP) | ρ_u last / max | ‖W_u‖ last |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.6586 | 0.6737 | 0.6986 | 0.6863 | PASS | -0.2009 | 0.9997 | 0.5426 | — / — | — |
| A1 | 0.6683 | 0.6705 | 0.6986 | 0.6828 | PASS | -0.0295 | 0.5573 | 0.5270 | — / — | — |
| A2 | 0.7120 | 0.6882 | 0.6986 | 0.7767 | FAIL | -0.0447 | 0.9998 | 0.5343 | 0.1325 / 0.1525 | 1.271 |
| A3 | 0.7161 | 0.7052 | 0.6986 | 0.7637 | FAIL | 0.0096 | 0.5830 | 0.5372 | 0.1075 / 0.1223 | 1.102 |

O5 for all four arms on `dota_cap_dev` (569 clips, unlabelled; Amendment 6 G5). A0/A1 rows = batch 1 checkpoints re-read; their T2-val columns reproduce batch 1.
