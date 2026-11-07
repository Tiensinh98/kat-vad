# KIM first-pass read-out, Nexar-trained, seed 2024 — **PRELIMINARY (n = 1)**

Rule: `core/docs/final/KIM_RUN.md §3.1, §6`. Control = **K0** (W0 + DVS off); W0 printed. The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1).

## DoTA-CAP-dev (protocol B, per-clip min-max)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] | — | 0.5573 [0.5380, 0.5765] | — |
| K0 | 0.8002 [0.7848, 0.8155] | — | 0.5573 [0.5380, 0.5765] | — |
| K0k9 | 0.8187 [0.8042, 0.8331] | — | 0.5573 [0.5380, 0.5765] | — |
| K5 | 0.8195 [0.8047, 0.8337] | — | 0.5573 [0.5380, 0.5765] | — |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0108 [-0.0200, -0.0016] | cost shown — PRELIMINARY (n = 1) | -0.0188 [-0.0291, -0.0083] | -0.0004 [-0.0175, +0.0163] |
| K0k9 - K0 | host kernel 9 (§3.1) | +0.0185 [+0.0108, +0.0268] | gain shown — PRELIMINARY (n = 1) | +0.0190 [+0.0108, +0.0270] | +0.0179 [+0.0039, +0.0323] |
| K5 - K0 | full KIM | +0.0193 [+0.0091, +0.0292] | gain shown — PRELIMINARY (n = 1) | +0.0176 [+0.0071, +0.0275] | +0.0215 [+0.0045, +0.0384] |

## DADA-2000 (370 sources, whole video, s8, raw)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] | 0.6981 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K0 | 0.7256 [0.7033, 0.7473] | 0.7099 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K0k9 | 0.7233 [0.7006, 0.7457] | 0.7014 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5 | 0.7740 [0.7492, 0.7982] | 0.7222 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0359 [-0.0490, -0.0232] | cost shown — PRELIMINARY (n = 1) | — | — |
| K0k9 - K0 | host kernel 9 (§3.1) | -0.0022 [-0.0144, +0.0100] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - K0 | full KIM | +0.0484 [+0.0367, +0.0599] | gain shown — PRELIMINARY (n = 1) | — | — |

## TAD test (100 videos, s8, raw) — printed, never decides

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K0 | 0.6954 [0.6314, 0.7566] | 0.5508 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K0k9 | 0.7163 [0.6476, 0.7808] | 0.6102 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5 | 0.7002 [0.6246, 0.7724] | 0.6781 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0029 [-0.0388, +0.0331] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K0k9 - K0 | host kernel 9 (§3.1) | +0.0208 [-0.0094, +0.0538] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - K0 | full KIM | +0.0047 [-0.0247, +0.0340] | no difference shown — PRELIMINARY (n = 1) | — | — |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220, in-domain cubic `p` = 0.8520.

## Training-free baseline: K0 + `vt2` rank average (§5.2, printed, never decides)

| set | host macro | rank avg macro | Δ (paired) |
| --- | --- | --- | --- |
| dota | 0.8002 [0.7847, 0.8157] | 0.8139 [0.7998, 0.8283] | +0.0137 [+0.0061, +0.0211] |
| dada | 0.7256 [0.7033, 0.7474] | 0.7428 [0.7214, 0.7635] | +0.0172 [+0.0081, +0.0262] |
| tad | 0.6954 [0.6314, 0.7566] | 0.7244 [0.6723, 0.7751] | +0.0290 [-0.0001, +0.0598] |

## Decisions (fixed before the numbers)

- **kernel**: host keeps kernel 3
- **stop_rule**: continue to §3's second group

## Training diagnostics (first → last logged batch)

| run | step | mil last-epoch mean | mil | kim_mil | kim_nkc | kim_w_out_norm | kim_vel_gate | kim_acc_gate | kim_anchor_share | kim_proto_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K0 | 1749 | 0.1288 | 0.6878 → 0.0861 | — | — | — | — | — | — | — |
| K0k9 | 1749 | 0.1163 | 0.6959 → 0.0934 | — | — | — | — | — | — | — |
| K5 | 1749 | 0.1637 | 0.6878 → 0.1613 | 0.8593 → 0.1262 | 2.1806 → 0.2771 | 0.0000 → 0.9335 | 0.5044 → 0.3638 | 0.5034 → 0.4993 | 0.0076 → 0.9038 | 0.0000 → -0.7319 |
