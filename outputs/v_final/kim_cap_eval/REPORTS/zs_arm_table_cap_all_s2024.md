# cap_all_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5f**. Pooling of micro / AP: DoTA-CAP (dev+eval) after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP (dev+eval) |
| --- | --- |
| K5f | 0.8226 / 0.7362 |
| K5dfi | 0.8240 / 0.7603 |
| K5dfie | 0.8270 / 0.7587 |
| LaGoVAD | 0.6465 / 0.6198 |

## DoTA-CAP (dev+eval) (1129 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 | 0.3260 |
| K5dfi | 0.8240 [0.8136, 0.8342] | 0.7603 | 0.5899 | 0.3260 |
| K5dfie | 0.8270 [0.8165, 0.8371] | 0.7587 | 0.5931 | 0.3260 |
| LaGoVAD | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 | 0.3260 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfi - K5f | +0.0014 [-0.0042, +0.0069] | not separated | +0.0242 [+0.0192, +0.0293] | above |
| K5dfie - K5f | +0.0044 [+0.0007, +0.0084] | above | +0.0225 [+0.0181, +0.0268] | above |
| LaGoVAD - K5f | -0.1761 [-0.1943, -0.1579] | below | -0.1164 [-0.1352, -0.0983] | below |
