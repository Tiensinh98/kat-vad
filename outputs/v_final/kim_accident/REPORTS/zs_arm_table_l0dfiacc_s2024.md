# l0dfiacc_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **L0dfi**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD |
| --- | --- | --- | --- |
| L0dfi | 0.6758 / 0.6393 | 0.5752 / 0.5748 | 0.5902 / 0.6733 |
| L0dfiACC | 0.6628 / 0.6140 | 0.5728 / 0.5610 | 0.6904 / 0.7389 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| L0dfi | 0.6758 [0.6564, 0.6951] | 0.6393 | 0.4721 | 0.3271 |
| L0dfiACC | 0.6628 [0.6430, 0.6818] | 0.6140 | 0.4420 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| L0dfiACC - L0dfi | -0.0131 [-0.0357, +0.0093] | not separated | -0.0253 [-0.0464, -0.0044] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| L0dfi | 0.5752 [0.5544, 0.5962] | 0.5748 | 0.3623 | 0.2885 |
| L0dfiACC | 0.5728 [0.5518, 0.5948] | 0.5610 | 0.3387 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| L0dfiACC - L0dfi | -0.0024 [-0.0274, +0.0227] | not separated | -0.0139 [-0.0382, +0.0100] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| L0dfi | 0.5902 [0.5312, 0.6488] | 0.6733 | 0.1956 | 0.0724 |
| L0dfiACC | 0.6904 [0.6294, 0.7463] | 0.7389 | 0.2684 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| L0dfiACC - L0dfi | +0.1001 [+0.0299, +0.1667] | above | +0.0656 [-0.0303, +0.1616] | not separated |
