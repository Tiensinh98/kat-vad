# dada_lagovad_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5f**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DADA-2000 |
| --- | --- |
| K5f | 0.7744 / 0.7551 |
| K5dfi | 0.7677 / 0.7591 |
| K5dfie | 0.7724 / 0.7598 |
| LaGoVAD | 0.6202 / 0.6128 |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5dfie | 0.7724 [0.7487, 0.7960] | 0.7598 | 0.5607 | 0.2885 |
| LaGoVAD | 0.6202 [0.5925, 0.6480] | 0.6128 | 0.3532 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfi - K5f | -0.0067 [-0.0156, +0.0024] | not separated | +0.0040 [-0.0044, +0.0128] | not separated |
| K5dfie - K5f | -0.0020 [-0.0086, +0.0046] | not separated | +0.0047 [-0.0011, +0.0110] | not separated |
| LaGoVAD - K5f | -0.1542 [-0.1865, -0.1220] | below | -0.1423 [-0.1734, -0.1127] | below |
