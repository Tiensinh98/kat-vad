# k5dfidota_abl_vs_l0_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **L0dfiDoTA**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 |
| --- | --- | --- |
| L0dfiDoTA | 0.6682 / 0.6412 | 0.5911 / 0.5849 |
| K0dfiDoTA | 0.8532 / 0.8156 | 0.7675 / 0.7553 |
| K5dfiDoTA | 0.8590 / 0.8209 | 0.7659 / 0.7522 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| L0dfiDoTA | 0.6682 [0.6536, 0.6826] | 0.6412 | 0.5010 | 0.3271 |
| K0dfiDoTA | 0.8532 [0.8409, 0.8651] | 0.8156 | 0.6817 | 0.3271 |
| K5dfiDoTA | 0.8590 [0.8468, 0.8709] | 0.8209 | 0.6776 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K0dfiDoTA - L0dfiDoTA | +0.1850 [+0.1720, +0.1980] | above | +0.1745 [+0.1627, +0.1864] | above |
| K5dfiDoTA - L0dfiDoTA | +0.1908 [+0.1773, +0.2043] | above | +0.1797 [+0.1674, +0.1928] | above |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| L0dfiDoTA | 0.5911 [0.5763, 0.6063] | 0.5849 | 0.3892 | 0.2885 |
| K0dfiDoTA | 0.7675 [0.7488, 0.7866] | 0.7553 | 0.5753 | 0.2885 |
| K5dfiDoTA | 0.7659 [0.7469, 0.7850] | 0.7522 | 0.5715 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K0dfiDoTA - L0dfiDoTA | +0.1764 [+0.1585, +0.1944] | above | +0.1704 [+0.1524, +0.1878] | above |
| K5dfiDoTA - L0dfiDoTA | +0.1748 [+0.1558, +0.1937] | above | +0.1674 [+0.1480, +0.1856] | above |
