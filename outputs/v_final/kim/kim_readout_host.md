# KIM first-pass read-out, Nexar-trained, seed 2024 — **PRELIMINARY (n = 1)**

Rule: `core/docs/final/KIM_RUN.md §3.1, §6`. Control = **K0** (W0 + DVS off); W0 printed. The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1).

## DoTA-CAP-dev (protocol B, per-clip min-max)

| row | macro | micro | t/N ruler |
| --- | --- | --- | --- |
| W0 | 0.8110 [0.7949, 0.8271] | — | 0.5573 [0.5380, 0.5765] |
| K0 | 0.8002 [0.7848, 0.8155] | — | 0.5573 [0.5380, 0.5765] |
| K0k9 | 0.8187 [0.8042, 0.8331] | — | 0.5573 [0.5380, 0.5765] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0108 [-0.0200, -0.0016] | cost shown — PRELIMINARY (n = 1) | -0.0188 [-0.0291, -0.0083] | -0.0004 [-0.0175, +0.0163] |
| K0k9 - K0 | host kernel 9 (§3.1) | +0.0185 [+0.0108, +0.0268] | gain shown — PRELIMINARY (n = 1) | +0.0190 [+0.0108, +0.0270] | +0.0179 [+0.0039, +0.0323] |

## DADA-2000 (370 sources, whole video, s8, raw)

| row | macro | micro | t/N ruler |
| --- | --- | --- | --- |
| W0 | 0.7615 [0.7369, 0.7859] | 0.6981 | 0.5514 [0.5247, 0.5782] |
| K0 | 0.7256 [0.7033, 0.7473] | 0.7099 | 0.5514 [0.5247, 0.5782] |
| K0k9 | 0.7233 [0.7006, 0.7457] | 0.7014 | 0.5514 [0.5247, 0.5782] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0359 [-0.0490, -0.0232] | cost shown — PRELIMINARY (n = 1) | — | — |
| K0k9 - K0 | host kernel 9 (§3.1) | -0.0022 [-0.0144, +0.0100] | no difference shown — PRELIMINARY (n = 1) | — | — |

## TAD test (100 videos, s8, raw) — printed, never decides

| row | macro | micro | t/N ruler |
| --- | --- | --- | --- |
| W0 | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.5872 [0.5126, 0.6611] |
| K0 | 0.6954 [0.6314, 0.7566] | 0.5508 | 0.5872 [0.5126, 0.6611] |
| K0k9 | 0.7163 [0.6476, 0.7808] | 0.6102 | 0.5872 [0.5126, 0.6611] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K0 - W0 | DVS off alone (printed) | -0.0029 [-0.0388, +0.0331] | no difference shown — PRELIMINARY (n = 1) | — | — |
| K0k9 - K0 | host kernel 9 (§3.1) | +0.0208 [-0.0094, +0.0538] | no difference shown — PRELIMINARY (n = 1) | — | — |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220.

## Decisions (fixed before the numbers)

- **kernel**: host keeps kernel 3
- **stop_rule**: pending

## Training diagnostics (first → last logged batch)

| run | step | mil last-epoch mean | mil | kim_mil | kim_nkc | kim_w_out_norm | kim_vel_gate | kim_acc_gate | kim_anchor_share | kim_proto_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K0 | 1749 | 0.1288 | 0.6878 → 0.0861 | — | — | — | — | — | — | — |
| K0k9 | 1749 | 0.1163 | 0.6959 → 0.0934 | — | — | — | — | — | — | — |
