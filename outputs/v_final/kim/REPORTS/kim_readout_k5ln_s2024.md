# KIM first-pass read-out, Nexar-trained, seed 2024 — **PRELIMINARY (n = 1)**

Rule: `core/docs/final/KIM_RUN.md §3.1, §6, §6.1, §6.2, §6.3, §12, §25`. Control = **K5**; W0 printed. The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1).

## DoTA-CAP-dev (protocol B, per-clip min-max)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] | — | 0.5573 [0.5380, 0.5765] | — |
| K5 | 0.8195 [0.8047, 0.8337] | — | 0.5573 [0.5380, 0.5765] | — |
| K5ln | 0.8175 [0.8030, 0.8317] | — | 0.5573 [0.5380, 0.5765] | — |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0085 [-0.0023, +0.0190] | no difference shown — PRELIMINARY (n = 1) | -0.0012 [-0.0126, +0.0096] | +0.0211 [+0.0040, +0.0380] |
| K5ln - K5 | per-step LayerNorm on the motion columns (§25) | -0.0020 [-0.0058, +0.0020] | no difference shown — PRELIMINARY (n = 1) | +0.0007 [-0.0041, +0.0053] | -0.0055 [-0.0134, +0.0021] |
| K5ln - W0 | K5ln vs the DVS-on host (printed) | +0.0065 [-0.0048, +0.0174] | no difference shown — PRELIMINARY (n = 1) | -0.0005 [-0.0129, +0.0110] | +0.0155 [-0.0032, +0.0336] |

## DADA-2000 (370 sources, whole video, s8, raw)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] | 0.6981 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5 | 0.7740 [0.7492, 0.7982] | 0.7222 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5ln | 0.7767 [0.7522, 0.8004] | 0.7188 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0125 [-0.0006, +0.0253] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5ln - K5 | per-step LayerNorm on the motion columns (§25) | +0.0027 [-0.0030, +0.0086] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5ln - W0 | K5ln vs the DVS-on host (printed) | +0.0153 [+0.0020, +0.0285] | gain shown — PRELIMINARY (n = 1) | — | — |

## TAD test (100 videos, s8, raw) — printed, never decides

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5 | 0.7002 [0.6246, 0.7724] | 0.6781 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5ln | 0.6954 [0.6188, 0.7688] | 0.7070 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0019 [-0.0244, +0.0286] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5ln - K5 | per-step LayerNorm on the motion columns (§25) | -0.0048 [-0.0227, +0.0126] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5ln - W0 | K5ln vs the DVS-on host (printed) | -0.0029 [-0.0395, +0.0330] | no difference shown — PRELIMINARY (n = 1) | — | — |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220, in-domain cubic `p` = 0.8520.

## Decisions (fixed before the numbers)

- **ln_rule**: no gain on DADA, or a cost on DoTA-CAP-dev: the LayerNorm line closes (§25)

Start check (printed, never decides): step-1 `mil` K5ln 0.687794 vs K5 0.687794, gap 0.00e+00 (tol 1e-04) — matched.

## Per-step norm of the baked motion columns (what LN erases; printed)

| cache | videos | steps | mean | q10 | median | q90 |
| --- | --- | --- | --- | --- | --- | --- |
| nexar (all baked ids) | 1125 | 159650 | 8.1132 | 5.7160 | 7.6764 | 11.0502 |
| DoTA-CAP-dev | 569 | 19180 | 8.8626 | 6.2799 | 8.3048 | 12.3327 |
| DADA | 370 | 16090 | 8.6860 | 6.4132 | 8.0808 | 11.8933 |
| TAD | 100 | 11045 | 9.7682 | 7.2281 | 9.4299 | 12.7943 |

## Training diagnostics (first → last logged batch)

| run | step | mil last-epoch mean | mil | kim_mil | kim_nkc | kim_w_out_norm | kim_vel_gate | kim_acc_gate | kim_anchor_share | kim_proto_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K5 | 1749 | 0.1637 | 0.6878 → 0.1613 | 0.8593 → 0.1262 | 2.1806 → 0.2771 | 0.0000 → 0.9335 | 0.5044 → 0.3638 | 0.5034 → 0.4993 | 0.0076 → 0.9038 | 0.0000 → -0.7319 |
| K5ln | 1749 | 0.1572 | 0.6878 → 0.1518 | 0.8593 → 0.1095 | 2.1438 → 0.2797 | 0.0000 → 0.9168 | 0.5044 → 0.3635 | 0.5034 → 0.5000 | 0.0076 → 0.9423 | 0.0000 → -0.7078 |
