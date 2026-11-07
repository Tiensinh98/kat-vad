# KIM first-pass read-out, Nexar-trained, seed 2025 — **PRELIMINARY (n = 1)**

Rule: `core/docs/final/KIM_RUN.md §3.1, §6, §6.1, §6.2, §6.3`. Control = **K0** (W0 + DVS off); W0 printed. The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1).

## DoTA-CAP-dev (protocol B, per-clip min-max)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| K0 | 0.8002 [0.7847, 0.8157] | — | 0.5573 [0.5380, 0.5765] | — |
| K5 | 0.8166 [0.8009, 0.8319] | — | 0.5573 [0.5380, 0.5765] | — |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - K0 | full KIM | +0.0163 [+0.0102, +0.0223] | gain shown — PRELIMINARY (n = 1) | +0.0159 [+0.0087, +0.0232] | +0.0169 [+0.0069, +0.0272] |

## DADA-2000 (370 sources, whole video, s8, raw)

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| K0 | 0.7248 [0.7019, 0.7468] | 0.7110 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |
| K5 | 0.7631 [0.7389, 0.7863] | 0.7219 | 0.5514 [0.5247, 0.5782] | 0.7342 [0.7076, 0.7594] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - K0 | full KIM | +0.0383 [+0.0293, +0.0473] | gain shown — PRELIMINARY (n = 1) | — | — |

## TAD test (100 videos, s8, raw) — printed, never decides

| row | macro | micro | t/N ruler | middle ruler |
| --- | --- | --- | --- | --- |
| K0 | 0.6918 [0.6260, 0.7558] | 0.5841 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |
| K5 | 0.6836 [0.6081, 0.7543] | 0.6448 | 0.5872 [0.5126, 0.6611] | 0.6831 [0.6097, 0.7517] |

| contrast | what | Δ | sentence | Δ ego | Δ non-ego |
| --- | --- | --- | --- | --- | --- |
| K5 - K0 | full KIM | -0.0082 [-0.0393, +0.0193] | no difference shown — PRELIMINARY (n = 1) | — | — |

Printed: v2's cubic position prior `p_T2` on CAP-dev = 0.8220, in-domain cubic `p` = 0.8520.

## Decisions (fixed before the numbers)

- **replication**: reproduced at n = 2: the line goes on (§6.2)

## Training diagnostics (first → last logged batch)

| run | step | mil last-epoch mean | mil | kim_mil | kim_nkc | kim_w_out_norm | kim_vel_gate | kim_acc_gate | kim_anchor_share | kim_proto_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K0 | 1749 | 0.1181 | 0.7057 → 0.0601 | — | — | — | — | — | — | — |
| K5 | 1749 | 0.1454 | 0.7057 → 0.0753 | 0.7584 → 0.0719 | 2.0888 → 0.2179 | 0.0000 → 0.9082 | 0.5078 → 0.3442 | 0.5044 → 0.5015 | 0.2016 → 0.9667 | 0.0000 → -0.7829 |
