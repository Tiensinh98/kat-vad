# t2_fixed_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5fT2**. Pooling of micro / AP: DoTA-CAP (dev+eval) after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP (dev+eval) |
| --- | --- |
| K5fT2 | 0.8091 / 0.7173 |
| K5dfiT2 | 0.7969 / 0.7236 |
| K5T2 | 0.8118 / 0.7206 |
| K5f | 0.8226 / 0.7362 |
| K5dfi | 0.8240 / 0.7603 |
| LaGoVAD | 0.6465 / 0.6198 |

## DoTA-CAP (dev+eval) (1129 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5fT2 | 0.8091 [0.7973, 0.8208] | 0.7173 | 0.5399 | 0.3260 |
| K5dfiT2 | 0.7969 [0.7850, 0.8087] | 0.7236 | 0.5400 | 0.3260 |
| K5T2 | 0.8118 [0.7996, 0.8237] | 0.7206 | 0.5402 | 0.3260 |
| K5f | 0.8226 [0.8116, 0.8329] | 0.7362 | 0.5652 | 0.3260 |
| K5dfi | 0.8240 [0.8136, 0.8342] | 0.7603 | 0.5899 | 0.3260 |
| LaGoVAD | 0.6465 [0.6295, 0.6631] | 0.6198 | 0.3903 | 0.3260 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiT2 - K5fT2 | -0.0121 [-0.0174, -0.0068] | below | +0.0064 [+0.0005, +0.0122] | above |
| K5T2 - K5fT2 | +0.0027 [+0.0006, +0.0050] | above | +0.0033 [+0.0004, +0.0063] | above |
| K5f - K5fT2 | +0.0136 [+0.0020, +0.0249] | above | +0.0189 [+0.0073, +0.0314] | above |
| K5dfi - K5fT2 | +0.0150 [+0.0051, +0.0247] | above | +0.0431 [+0.0317, +0.0550] | above |
| LaGoVAD - K5fT2 | -0.1625 [-0.1796, -0.1456] | below | -0.0975 [-0.1147, -0.0799] | below |
