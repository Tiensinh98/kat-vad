# l0dfi_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5dfi**. Pooling of micro / AP: DoTA-CAP (dev+eval) and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP (dev+eval) | DADA-2000 |
| --- | --- | --- |
| K5dfi | 0.8240 / 0.7603 | 0.7677 / 0.7591 |
| L0dfi | 0.6770 / 0.6423 | 0.5752 / 0.5748 |
| K5f | 0.8226 / 0.7362 | 0.7744 / 0.7551 |
| LaGoVAD | 0.6465 / 0.6198 | 0.6202 / 0.6128 |
| A0 | 0.7187 / 0.6678 | 0.6167 / 0.6070 |

## DoTA-CAP (dev+eval) (1129 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfi | 0.8240 [0.8136, 0.8342] | 0.7603 | 0.5899 | 0.3260 |
| L0dfi | 0.6770 [0.6610, 0.6921] | 0.6423 | 0.4738 | 0.3260 |
| K5f | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 | 0.3260 |
| LaGoVAD | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 | 0.3260 |
| A0 | 0.7187 [0.7025, 0.7344] | 0.6678 | 0.5008 | 0.3260 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| L0dfi - K5dfi | -0.1470 [-0.1611, -0.1334] | below | -0.1180 [-0.1307, -0.1063] | below |
| K5f - K5dfi | -0.0014 [-0.0069, +0.0042] | not separated | -0.0242 [-0.0293, -0.0192] | below |
| LaGoVAD - K5dfi | -0.1775 [-0.1954, -0.1599] | below | -0.1406 [-0.1579, -0.1243] | below |
| A0 - K5dfi | -0.1053 [-0.1197, -0.0911] | below | -0.0925 [-0.1039, -0.0814] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| L0dfi | 0.5752 [0.5544, 0.5962] | 0.5748 | 0.3623 | 0.2885 |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| LaGoVAD | 0.6202 [0.5925, 0.6480] | 0.6128 | 0.3532 | 0.2885 |
| A0 | 0.6167 [0.5907, 0.6433] | 0.6070 | 0.3794 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| L0dfi - K5dfi | -0.1925 [-0.2151, -0.1694] | below | -0.1842 [-0.2031, -0.1639] | below |
| K5f - K5dfi | +0.0067 [-0.0024, +0.0156] | not separated | -0.0040 [-0.0128, +0.0044] | not separated |
| LaGoVAD - K5dfi | -0.1475 [-0.1797, -0.1160] | below | -0.1463 [-0.1758, -0.1165] | below |
| A0 - K5dfi | -0.1510 [-0.1762, -0.1256] | below | -0.1521 [-0.1745, -0.1307] | below |

DADA-2000 position only (no pixels): middle ruler macro 0.7342 [0.7076, 0.7594].
