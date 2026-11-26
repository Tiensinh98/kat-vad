# k5dfidota_abl_vs_k0_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K0dfiDoTA**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 |
| --- | --- | --- |
| K0dfiDoTA | 0.8532 / 0.8156 | 0.7675 / 0.7553 |
| K5dfiDoTA | 0.8590 / 0.8209 | 0.7659 / 0.7522 |
| L0dfiDoTA | 0.6682 / 0.6412 | 0.5911 / 0.5849 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfiDoTA | 0.8532 [0.8409, 0.8651] | 0.8156 | 0.6817 | 0.3271 |
| K5dfiDoTA | 0.8590 [0.8468, 0.8709] | 0.8209 | 0.6776 | 0.3271 |
| L0dfiDoTA | 0.6682 [0.6536, 0.6826] | 0.6412 | 0.5010 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiDoTA - K0dfiDoTA | +0.0058 [+0.0015, +0.0101] | above | +0.0053 [+0.0007, +0.0098] | above |
| L0dfiDoTA - K0dfiDoTA | -0.1850 [-0.1980, -0.1720] | below | -0.1745 [-0.1864, -0.1627] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfiDoTA | 0.7675 [0.7488, 0.7866] | 0.7553 | 0.5753 | 0.2885 |
| K5dfiDoTA | 0.7659 [0.7469, 0.7850] | 0.7522 | 0.5715 | 0.2885 |
| L0dfiDoTA | 0.5911 [0.5763, 0.6063] | 0.5849 | 0.3892 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiDoTA - K0dfiDoTA | -0.0016 [-0.0083, +0.0050] | not separated | -0.0031 [-0.0098, +0.0034] | not separated |
| L0dfiDoTA - K0dfiDoTA | -0.1764 [-0.1944, -0.1585] | below | -0.1704 [-0.1878, -0.1524] | below |
