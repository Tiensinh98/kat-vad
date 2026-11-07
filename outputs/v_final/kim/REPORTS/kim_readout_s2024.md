# KIM first-pass read-out, Nexar-trained, seed 2024 — **PRELIMINARY (n = 1)**

Rule: `core/docs/final/KIM_RUN.md §3.1, §6, §6.1, §6.2, §6.3`. Control = **K0** (W0 + DVS off); W0 printed. The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1).

## DoTA-CAP-dev (protocol B, per-clip min-max)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] | — | 0.5573 [0.5380, 0.5765] | — |
| K0 | 0.8002 [0.7848, 0.8155] | — | 0.5573 [0.5380, 0.5765] | — |
| K0k9 | 0.8187 [0.8042, 0.8331] | — | 0.5573 [0.5380, 0.5765] | — |
| K5 | 0.8195 [0.8047, 0.8337] | — | 0.5573 [0.5380, 0.5765] | — |
| Kcap | 0.8172 [0.8013, 0.8327] | — | 0.5573 [0.5380, 0.5765] | — |
| K5d | 0.8124 [0.7978, 0.8270] | — | 0.5573 [0.5380, 0.5765] | — |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0108 [-0.0200, -0.0016] | cost shown — PRELIMINARY (n = 1) | -0.0188 [-0.0291, -0.0083] | -0.0004 [-0.0175, +0.0163] |
| K0k9 - K0 | host kernel 9 (§3.1) | +0.0185 [+0.0108, +0.0268] | gain shown — PRELIMINARY (n = 1) | +0.0190 [+0.0108, +0.0270] | +0.0179 [+0.0039, +0.0323] |
| K5 - K0 | full KIM | +0.0193 [+0.0091, +0.0292] | gain shown — PRELIMINARY (n = 1) | +0.0176 [+0.0071, +0.0275] | +0.0215 [+0.0045, +0.0384] |
| K5d - K0 | full KIM + seam mask, DVS on, vs DVS off (printed) | +0.0122 [+0.0015, +0.0232] | gain shown — PRELIMINARY (n = 1) | +0.0146 [+0.0029, +0.0257] | +0.0092 [-0.0098, +0.0288] |
| Kcap - K0 | capacity + losses, no differences | +0.0170 [+0.0106, +0.0233] | gain shown — PRELIMINARY (n = 1) | +0.0213 [+0.0141, +0.0281] | +0.0114 [+0.0006, +0.0223] |
| K5 - Kcap | the differences (§6.3) | +0.0023 [-0.0058, +0.0103] | no difference shown — PRELIMINARY (n = 1) | -0.0037 [-0.0106, +0.0028] | +0.0100 [-0.0051, +0.0251] |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0085 [-0.0023, +0.0190] | no difference shown — PRELIMINARY (n = 1) | -0.0012 [-0.0126, +0.0096] | +0.0211 [+0.0040, +0.0380] |
| Kcap - W0 | branch + losses, DVS off, vs the DVS-on host (printed) | +0.0062 [-0.0021, +0.0144] | no difference shown — PRELIMINARY (n = 1) | +0.0025 [-0.0069, +0.0113] | +0.0110 [-0.0039, +0.0258] |
| K5 - K0k9 | full KIM vs host kernel 9, no KIM (printed) | +0.0008 [-0.0116, +0.0130] | no difference shown — PRELIMINARY (n = 1) | -0.0014 [-0.0141, +0.0107] | +0.0035 [-0.0184, +0.0250] |
| K5d - W0 | full KIM + seam mask, both DVS on (§12) | +0.0014 [-0.0060, +0.0085] | no difference shown — PRELIMINARY (n = 1) | -0.0042 [-0.0127, +0.0034] | +0.0088 [-0.0031, +0.0204] |
| K5d - K5 | DVS on vs off, both full KIM (printed) | -0.0071 [-0.0153, +0.0016] | no difference shown — PRELIMINARY (n = 1) | -0.0030 [-0.0113, +0.0056] | -0.0123 [-0.0259, +0.0019] |

## DADA-2000 (370 sources, whole video, s8, raw)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] | 0.6981 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K0 | 0.7256 [0.7033, 0.7473] | 0.7099 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K0k9 | 0.7233 [0.7006, 0.7457] | 0.7014 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5 | 0.7740 [0.7492, 0.7982] | 0.7222 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| Kcap | 0.7624 [0.7385, 0.7859] | 0.7236 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5d | 0.7713 [0.7466, 0.7955] | 0.6915 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0359 [-0.0490, -0.0232] | cost shown — PRELIMINARY (n = 1) | — | — |
| K0k9 - K0 | host kernel 9 (§3.1) | -0.0022 [-0.0144, +0.0100] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - K0 | full KIM | +0.0484 [+0.0367, +0.0599] | gain shown — PRELIMINARY (n = 1) | — | — |
| K5d - K0 | full KIM + seam mask, DVS on, vs DVS off (printed) | +0.0457 [+0.0323, +0.0591] | gain shown — PRELIMINARY (n = 1) | — | — |
| Kcap - K0 | capacity + losses, no differences | +0.0368 [+0.0281, +0.0457] | gain shown — PRELIMINARY (n = 1) | — | — |
| K5 - Kcap | the differences (§6.3) | +0.0116 [+0.0020, +0.0210] | gain shown — PRELIMINARY (n = 1) | — | — |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0125 [-0.0006, +0.0253] | no difference shown — PRELIMINARY (n = 1) | — | — |
| Kcap - W0 | branch + losses, DVS off, vs the DVS-on host (printed) | +0.0009 [-0.0107, +0.0127] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - K0k9 | full KIM vs host kernel 9, no KIM (printed) | +0.0507 [+0.0335, +0.0678] | gain shown — PRELIMINARY (n = 1) | — | — |
| K5d - W0 | full KIM + seam mask, both DVS on (§12) | +0.0098 [+0.0010, +0.0185] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5d - K5 | DVS on vs off, both full KIM (printed) | -0.0027 [-0.0121, +0.0068] | no difference shown — PRELIMINARY (n = 1) | — | — |

## TAD test (100 videos, s8, raw) — printed, never decides

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| W0 | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K0 | 0.6954 [0.6314, 0.7566] | 0.5508 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K0k9 | 0.7163 [0.6476, 0.7808] | 0.6102 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5 | 0.7002 [0.6246, 0.7724] | 0.6781 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| Kcap | 0.6928 [0.6207, 0.7602] | 0.5629 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5d | 0.7055 [0.6284, 0.7772] | 0.6679 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0029 [-0.0388, +0.0331] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K0k9 - K0 | host kernel 9 (§3.1) | +0.0208 [-0.0094, +0.0538] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - K0 | full KIM | +0.0047 [-0.0247, +0.0340] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5d - K0 | full KIM + seam mask, DVS on, vs DVS off (printed) | +0.0100 [-0.0318, +0.0525] | no difference shown — PRELIMINARY (n = 1) | — | — |
| Kcap - K0 | capacity + losses, no differences | -0.0027 [-0.0320, +0.0239] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - Kcap | the differences (§6.3) | +0.0074 [-0.0218, +0.0370] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - W0 | full KIM, DVS off, vs the DVS-on host (printed) | +0.0019 [-0.0244, +0.0286] | no difference shown — PRELIMINARY (n = 1) | — | — |
| Kcap - W0 | branch + losses, DVS off, vs the DVS-on host (printed) | -0.0055 [-0.0367, +0.0262] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5 - K0k9 | full KIM vs host kernel 9, no KIM (printed) | -0.0161 [-0.0555, +0.0229] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5d - W0 | full KIM + seam mask, both DVS on (§12) | +0.0072 [-0.0113, +0.0276] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K5d - K5 | DVS on vs off, both full KIM (printed) | +0.0053 [-0.0210, +0.0322] | no difference shown — PRELIMINARY (n = 1) | — | — |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220, in-domain cubic `p` = 0.8520.

## Training-free baseline: K0 + `vt2` rank average (§5.2, printed, never decides)

| set | host macro | rank avg macro | Δ (paired) |
| --- | --- | --- | --- |
| dota | 0.8002 [0.7847, 0.8157] | 0.8139 [0.7998, 0.8283] | +0.0137 [+0.0061, +0.0211] |
| dada | 0.7256 [0.7033, 0.7474] | 0.7428 [0.7214, 0.7635] | +0.0172 [+0.0081, +0.0262] |
| tad | 0.6954 [0.6314, 0.7566] | 0.7244 [0.6723, 0.7751] | +0.0290 [-0.0001, +0.0598] |

## Decisions (fixed before the numbers)

- **kernel**: host keeps kernel 3
- **dvs_host**: no gain on DADA vs the DVS-on host: the KIM line closes (§12)
- **stop_rule**: continue to §3's second group

## Training diagnostics (first → last logged batch)

| run | step | mil last-epoch mean | mil | kim_mil | kim_nkc | kim_w_out_norm | kim_vel_gate | kim_acc_gate | kim_anchor_share | kim_proto_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K0 | 1749 | 0.1288 | 0.6878 → 0.0861 | — | — | — | — | — | — | — |
| K0k9 | 1749 | 0.1163 | 0.6959 → 0.0934 | — | — | — | — | — | — | — |
| K5 | 1749 | 0.1637 | 0.6878 → 0.1613 | 0.8593 → 0.1262 | 2.1806 → 0.2771 | 0.0000 → 0.9335 | 0.5044 → 0.3638 | 0.5034 → 0.4993 | 0.0076 → 0.9038 | 0.0000 → -0.7319 |
| Kcap | 1749 | 0.0925 | 0.6925 → 0.0584 | 0.8490 → 0.0693 | 2.2651 → 0.1644 | 0.0000 → 1.6011 | 0.0000 → 0.0000 | 0.0000 → 0.0000 | 0.0000 → 0.9519 | 0.0000 → -0.4860 |
| K5d | 1749 | 0.2123 | 0.6906 → 0.1597 | 0.8723 → 0.1288 | 2.4063 → 0.5860 | 0.0000 → 0.7113 | 0.5054 → 0.3862 | 0.5034 → 0.5112 | 0.0045 → 0.9464 | 0.0000 → -0.5870 |
