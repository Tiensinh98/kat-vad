# KIM first-pass read-out, Nexar-trained, seed 2024 — **PRELIMINARY (n = 1)**

Rule: `core/docs/final/KIM_RUN.md §3.1, §6, §6.1, §6.2, §6.3, §12, §25, §26`. Control = **K5**; W0 printed. The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1).

## DoTA-CAP-dev (protocol B, per-clip min-max)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] | — | 0.5573 [0.5380, 0.5765] | — |
| K5 | 0.8195 [0.8047, 0.8337] | — | 0.5573 [0.5380, 0.5765] | — |
| K5k2 | 0.8058 [0.7902, 0.8210] | — | 0.5573 [0.5380, 0.5765] | — |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0085 [-0.0023, +0.0190] | no difference shown — PRELIMINARY (n = 1) | -0.0012 [-0.0126, +0.0096] | +0.0211 [+0.0040, +0.0380] |
| K5k2 - K5 | MIL top-k 4 -> 2 rows per window (§26) | -0.0137 [-0.0213, -0.0060] | cost shown — PRELIMINARY (n = 1) | -0.0073 [-0.0148, +0.0005] | -0.0220 [-0.0349, -0.0093] |
| K5k2 - W0 | K5k2 vs the DVS-on host (printed) | -0.0052 [-0.0145, +0.0048] | no difference shown — PRELIMINARY (n = 1) | -0.0084 [-0.0192, +0.0023] | -0.0009 [-0.0175, +0.0161] |

## DADA-2000 (370 sources, whole video, s8, raw)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] | 0.6981 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5 | 0.7740 [0.7492, 0.7982] | 0.7222 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5k2 | 0.7402 [0.7170, 0.7629] | 0.7184 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0125 [-0.0006, +0.0253] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5k2 - K5 | MIL top-k 4 -> 2 rows per window (§26) | -0.0338 [-0.0425, -0.0252] | cost shown — PRELIMINARY (n = 1) | — | — |
| K5k2 - W0 | K5k2 vs the DVS-on host (printed) | -0.0213 [-0.0349, -0.0079] | cost shown — PRELIMINARY (n = 1) | — | — |

## TAD test (100 videos, s8, raw) — printed, never decides

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5 | 0.7002 [0.6246, 0.7724] | 0.6781 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5k2 | 0.6808 [0.6102, 0.7496] | 0.6451 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0019 [-0.0244, +0.0286] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5k2 - K5 | MIL top-k 4 -> 2 rows per window (§26) | -0.0194 [-0.0493, +0.0121] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5k2 - W0 | K5k2 vs the DVS-on host (printed) | -0.0175 [-0.0475, +0.0129] | no difference shown — PRELIMINARY (n = 1) | — | — |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220, in-domain cubic `p` = 0.8520.

## Decisions (fixed before the numbers)

- **topk_rule**: no gain on DADA, or a cost on DoTA-CAP-dev: the top-k line closes (§26)

## Training diagnostics (first → last logged batch)

| run | step | mil last-epoch mean | mil | kim_mil | kim_nkc | kim_w_out_norm | kim_vel_gate | kim_acc_gate | kim_anchor_share | kim_proto_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K5 | 1749 | 0.1637 | 0.6878 → 0.1613 | 0.8593 → 0.1262 | 2.1806 → 0.2771 | 0.0000 → 0.9335 | 0.5044 → 0.3638 | 0.5034 → 0.4993 | 0.0076 → 0.9038 | 0.0000 → -0.7319 |
| K5k2 | 1749 | 0.1270 | 0.6954 → 0.0639 | 0.8256 → 0.0746 | 2.1373 → 0.1307 | 0.0000 → 1.1572 | 0.5044 → 0.3767 | 0.5034 → 0.4985 | 0.0000 → 0.9615 | 0.0000 → -0.7356 |
