# v7 Step 0 — Latent Surprise Predictor, stream clip (no detector)

Rule: core/docs/v7/PREREG_V7.md §1.1 (Amendment 1). **Decision: NO-GO** — checks {'d1_not_failed': True, 'd2': False, 'd3_nexar': True, 'd3_dota': False}

LSP: K = 16, Δ = 16 (10 fps lattice); ŝ stats mean 0.04114, std 0.01793.

D1: {'cos_k1': 0.9835178714208778, 'cos_horizon': 0.949372435156953, 'failed': False}

## D2 — held-out L_rec (45 negatives, t ≥ Δ + max K - 1)

| predictor | L_rec |
|---|---|
| lsp | 0.04127 |
| copy_last | 0.04915 |
| const_velocity | 0.12895 |
| context_mean | 0.04110 |

Best fixed = context_mean; relative gain -0.0041 (bar ≥ 0.1) → **FAIL**

## D3 — standalone surprise

nexar_val video AUC of max (n = 225): ŝ 0.5357, p 0.5137, rho 0.6299; ŝ - p +0.0220 [-0.0079, +0.0526]

DoTA-CAP-dev per-clip frame AUC on valid frames (read 550 of 569; missing caches 0):

| subset | ŝ | p | rho | cubic position | ŝ - p |
|---|---|---|---|---|---|
| all | 0.7244 [0.7077, 0.7405] (n = 550) | 0.7362 [0.7197, 0.7524] (n = 550) | 0.3207 [0.3049, 0.3368] (n = 550) | 0.8201 [0.7998, 0.8397] (n = 550) | -0.0118 [-0.0180, -0.0055] (n = 550) |
| ego | 0.7692 [0.7503, 0.7875] (n = 312) | 0.7739 [0.7558, 0.7916] (n = 312) | 0.2981 [0.2789, 0.3184] (n = 312) | 0.8146 [0.7884, 0.8393] (n = 312) | -0.0048 [-0.0124, +0.0034] (n = 312) |
| non_ego | 0.6657 [0.6390, 0.6911] (n = 238) | 0.6867 [0.6605, 0.7120] (n = 238) | 0.3503 [0.3257, 0.3746] (n = 238) | 0.8273 [0.7982, 0.8541] (n = 238) | -0.0210 [-0.0325, -0.0101] (n = 238) |

## D4 — source classifier (printed)

normals 113, lead-ins 220: source AUC 0.9992, gap vs D3 +0.4635, flag True

## D5 — printed

lead-in max share (n = 338): ŝ 0.1272, p 0.1124; corr(s, p) on nexar_val normals 0.9274
