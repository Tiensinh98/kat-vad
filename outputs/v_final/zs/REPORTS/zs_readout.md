# Zero-shot TAD / DADA-2000 — R4 (SA²) vs W0 (host) — **PRELIMINARY (n = 1 seed)**

Rule: `core/docs/final/ZS_TAD_DADA.md §3`. Both Nexar-trained, seed 2024, prefix OFF (released prompts).

## TAD test (100 videos, s8, §26's protocol, raw scores)

| row | macro [95 % CI over videos] | micro | clip AUC |
|---|---|---|---|
| W0 s2024 | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.5942 |
| R4 s2024 | 0.7220 [0.6515, 0.7889] | 0.7167 | 0.6567 |
| v2 A2, 5-seed per-frame mean (printed) | 0.7058 [0.6276, 0.7797] | 0.6447 | 0.6208 |

* **R4 - W0 (paired, per video): +0.0237 [+0.0005, +0.0499] (n=60) → gain shown — PRELIMINARY (n = 1 seed)**
* printed: R4 - v2 A2 5-seed mean: +0.0162 [-0.0072, +0.0429] (n=60) (biased toward A2: averaging)
* reproduction: W0 0.6983 vs v2 A2 s2024 0.6983 (gap +0.00001)
* rulers (no pixels): middle ruler 0.6831 [0.6097, 0.7517] · clip oracle micro 0.9226 · length ruler micro 0.8968

## DADA-2000 whole videos (370 sources of T2-test, s8)

| row | macro [95 % CI over videos] | micro | clip AUC |
|---|---|---|---|
| W0 s2024 | 0.7615 [0.7369, 0.7859] | 0.6981 | — |
| R4 s2024 | 0.7579 [0.7334, 0.7820] | 0.7048 | — |
| v2 A2, 5-seed per-frame mean (printed) | 0.7652 [0.7417, 0.7884] | 0.6987 | — |

* **R4 - W0 (paired, per video): -0.0036 [-0.0161, +0.0089] (n=370) → no difference shown — PRELIMINARY (n = 1 seed)**
* printed: R4 - v2 A2 5-seed mean: -0.0073 [-0.0185, +0.0039] (n=370) (biased toward A2: averaging)
* reproduction: W0 0.7615 vs v2 A2 s2024 0.7615 (gap +0.00000)
* rulers (no pixels): middle ruler 0.7342 [0.7076, 0.7594] · monotone ruler 0.5514 [0.5247, 0.5782]

Training diagnostics (first → last logged batch):

| run | step | SA² entropy / log K | ‖W_q‖ | prompt norm | rho_u | ‖W_u‖ | mil |
|---|---|---|---|---|---|---|---|
| W0 | 1749 | — | — | — | 0.0000 → 0.1459 | 0.0000 → 1.2610 | 0.6944 → 0.1760 |
| R4 | 1749 | 1.0000 → 0.9111 | 0.0000 → 1.4333 | 126.4213 → 126.3666 | 0.0000 → 0.0909 | 0.0000 → 1.4073 | 0.7257 → 0.2278 |
