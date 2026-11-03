# W_u arms read-out — DoTA-CAP (n/1397) dev, protocol B — **PRELIMINARY (n = 1 seed)**

Rule: `core/docs/final/WU_ARMS.md §3`. Control = **W0** (same code, W_u zero-init, LR x1). Seeds: 2024.

| row | macro [95 % cluster CI] | per-run macro | t/N ruler | arm |
|---|---|---|---|---|
| W0 | 0.8110 [0.7948, 0.8269] | {'s2024': 0.8110072204788573} | 0.5573 [0.5380, 0.5765] | `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}` |
| WI | 0.8106 [0.7948, 0.8260] | {'s2024': 0.8105686538665703} | 0.5573 [0.5380, 0.5765] | `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}` |
| WL | 0.7693 [0.7491, 0.7889] | {'s2024': 0.7693285658615107} | 0.5573 [0.5380, 0.5765] | `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}` |
| R1 | 0.8102 [0.7942, 0.8260] | {'s2024': 0.810195687446355} | 0.5573 [0.5380, 0.5765] | `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'v5_keys': 'content', 'v5_queries': 1, 'v5_readout': 'mean', 'v5_smil_weight': 0.0}` |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220.

| contrast | Δ all | sentence | Δ ego | Δ non-ego |
|---|---|---|---|---|
| WI - W0 | -0.0004 [-0.0032, +0.0021] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | -0.0016 [-0.0047, +0.0014] (n=321) | +0.0010 [-0.0033, +0.0049] (n=248) |
| WL - W0 | -0.0417 [-0.0538, -0.0297] (n=569) | cost shown — PRELIMINARY (n = 1 seed) | -0.0274 [-0.0395, -0.0159] (n=321) | -0.0602 [-0.0807, -0.0391] (n=248) |
| R1 - W0 | -0.0008 [-0.0026, +0.0009] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | -0.0014 [-0.0030, +0.0003] (n=321) | -0.0001 [-0.0037, +0.0033] (n=248) |

Training diagnostics (first → last logged batch):

| run | step | SA² entropy / log K | ‖W_q‖ | prompt norm | rho_u | ‖W_u‖ | mil |
|---|---|---|---|---|---|---|---|
| W0/s2024 | 1749 | — | — | — | 0.0000 → 0.1459 | 0.0000 → 1.2610 | 0.6944 → 0.1760 |
| WI/s2024 | 1749 | — | — | — | 0.0579 → 0.1574 | 1.3289 → 1.8358 | 0.6952 → 0.1772 |
| WL/s2024 | 1749 | — | — | — | 0.0000 → 0.5701 | 0.0000 → 6.5762 | 0.6944 → 0.0613 |
