# LaGoVAD-STP read-out — DoTA-CAP (n/1397) dev, protocol B — **PRELIMINARY (n = 1 seed)**

Rule: `core/docs/final/STP_BUILD.md §3`. Control = **R1** (v5 Stage 1 motion-stream host). Seeds: 2024.

| row | macro [95 % cluster CI] | per-run macro | t/N ruler | arm |
|---|---|---|---|---|
| R1 | 0.8102 [0.7942, 0.8260] | {'s2024': 0.810195687446355} | 0.5573 [0.5380, 0.5765] | `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'v5_keys': 'content', 'v5_queries': 1, 'v5_readout': 'mean', 'v5_smil_weight': 0.0}` |
| R4 | 0.8111 [0.7927, 0.8293] | {'s2024': 0.8111191434950765} | 0.5573 [0.5380, 0.5765] | `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'stp_motion': 'raw', 'stp_sa2': True, 'stp_text_prompt': 'released', 'stp_top_k': 12}` |
| R6 | 0.8124 [0.7944, 0.8302] | {'s2024': 0.8124289574112392} | 0.5573 [0.5380, 0.5765] | `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'stp_motion': 'raw', 'stp_sa2': True, 'stp_text_prompt': 'prefix', 'stp_top_k': 12}` |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220.

| contrast | Δ all | sentence | Δ ego | Δ non-ego |
|---|---|---|---|---|
| R4 - R1 | +0.0009 [-0.0093, +0.0106] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | +0.0116 [+0.0014, +0.0200] (n=321) | -0.0129 [-0.0307, +0.0042] (n=248) |
| R6 - R1 | +0.0022 [-0.0083, +0.0122] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | +0.0092 [-0.0012, +0.0189] (n=321) | -0.0068 [-0.0245, +0.0100] (n=248) |
| R6 - R4 (prefix ON - OFF) | +0.0013 [-0.0045, +0.0073] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | -0.0024 [-0.0087, +0.0038] (n=321) | +0.0061 [-0.0043, +0.0172] (n=248) |

Training diagnostics (first → last logged batch):

| run | step | SA² entropy / log K | ‖W_q‖ | prompt norm | rho_u | mil |
|---|---|---|---|---|---|---|
| R4/s2024 | 1749 | 1.0000 → 0.9111 | 0.0000 → 1.4333 | 126.4213 → 126.3666 | 0.0000 → 0.0909 | 0.7257 → 0.2278 |
| R6/s2024 | 1749 | 1.0000 → 0.8961 | 0.0000 → 1.5173 | 1.2612 → 1.2616 | 0.0000 → 0.0953 | 0.9564 → 0.2267 |
