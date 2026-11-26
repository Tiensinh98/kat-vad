# cap_all_k5dfibk25_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5f**. Pooling of micro / AP: DoTA-CAP (dev+eval) after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP (dev+eval) |
| --- | --- |
| K5f | 0.8226 / 0.7362 |
| K5dfibk25rr | 0.8210 / 0.7692 |
| LaGoVAD | 0.6465 / 0.6198 |

## DoTA-CAP (dev+eval) (1129 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 | 0.3260 |
| K5dfibk25rr | 0.8210 [0.8096, 0.8319] | 0.7692 | 0.6070 | 0.3260 |
| LaGoVAD | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 | 0.3260 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfibk25rr - K5f | -0.0016 [-0.0074, +0.0042] | not separated | +0.0330 [+0.0275, +0.0388] | above |
| LaGoVAD - K5f | -0.1761 [-0.1943, -0.1579] | below | -0.1164 [-0.1352, -0.0983] | below |
