# v7 Step 0 — Latent Surprise Predictor on V2-S vectors (no detector)

Rule: core/docs/v7/PREREG_V7.md §1. **Decision: NO-GO** — checks {'d1_not_failed': True, 'd2': False, 'd3_nexar': True, 'd3_dota': True}

LSP: K = 16, Δ = 16 (10 fps lattice); ŝ stats mean 0.05876, std 0.03241.

D1: {'cos_k1': 0.9765621891789469, 'cos_horizon': 0.9325235261777649, 'failed': False}

## D2 — held-out L_rec (45 negatives, t ≥ Δ + max K - 1)

| predictor | L_rec |
|---|---|
| lsp | 0.05870 |
| copy_last | 0.06916 |
| const_velocity | 0.17609 |
| context_mean | 0.06509 |

Best fixed = context_mean; relative gain +0.0981 (bar ≥ 0.1) → **FAIL**

## D3 — standalone surprise

nexar_val video AUC of max (n = 225): ŝ 0.7006, p 0.6854, rho 0.4125; ŝ - p +0.0152 [-0.0185, +0.0497]

DoTA-CAP-dev per-clip frame AUC on valid frames (read 550 of 569; missing caches 0):

| subset | ŝ | p | rho | cubic position | ŝ - p |
|---|---|---|---|---|---|
| all | 0.4454 [0.4221, 0.4689] (n = 550) | 0.4295 [0.4057, 0.4537] (n = 550) | 0.5447 [0.5238, 0.5664] (n = 550) | 0.8201 [0.7998, 0.8397] (n = 550) | +0.0159 [+0.0067, +0.0253] (n = 550) |
| ego | 0.4519 [0.4228, 0.4808] (n = 312) | 0.4205 [0.3907, 0.4494] (n = 312) | 0.5726 [0.5458, 0.5999] (n = 312) | 0.8146 [0.7884, 0.8393] (n = 312) | +0.0314 [+0.0206, +0.0420] (n = 312) |
| non_ego | 0.4369 [0.4057, 0.4667] (n = 238) | 0.4414 [0.4110, 0.4713] (n = 238) | 0.5080 [0.4811, 0.5367] (n = 238) | 0.8273 [0.7982, 0.8541] (n = 238) | -0.0045 [-0.0196, +0.0109] (n = 238) |

## D4 — source classifier (printed)

normals 113, lead-ins 220: source AUC 0.9930, gap vs D3 +0.2924, flag True

## D5 — printed

lead-in max share (n = 338): ŝ 0.0621, p 0.0562; corr(s, p) on nexar_val normals 0.9078
