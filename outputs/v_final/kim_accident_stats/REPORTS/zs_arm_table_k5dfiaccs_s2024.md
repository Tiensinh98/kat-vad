# k5dfiaccs_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5dfiACC**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5dfiACC | 0.7372 / 0.6815 | 0.6593 / 0.6420 | 0.6987 / 0.6170 | 0.8266 / 0.7391 |
| K5dfiACCs | 0.7452 / 0.6872 | 0.6720 / 0.6491 | 0.7148 / 0.6193 | 0.8228 / 0.7420 |
| K5dfi | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiACC | 0.7372 [0.7169, 0.7582] | 0.6815 | 0.4886 | 0.3271 |
| K5dfiACCs | 0.7452 [0.7263, 0.7646] | 0.6872 | 0.4968 | 0.3271 |
| K5dfi | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACCs - K5dfiACC | +0.0079 [+0.0042, +0.0115] | above | +0.0057 [+0.0029, +0.0084] | above |
| K5dfi - K5dfiACC | +0.0887 [+0.0670, +0.1102] | above | +0.0798 [+0.0578, +0.1024] | above |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiACC | 0.6593 [0.6371, 0.6824] | 0.6420 | 0.4027 | 0.2885 |
| K5dfiACCs | 0.6720 [0.6500, 0.6944] | 0.6491 | 0.4102 | 0.2885 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACCs - K5dfiACC | +0.0127 [+0.0087, +0.0168] | above | +0.0071 [+0.0039, +0.0103] | above |
| K5dfi - K5dfiACC | +0.1084 [+0.0816, +0.1352] | above | +0.1171 [+0.0912, +0.1422] | above |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiACC | 0.6987 [0.6335, 0.7609] | 0.6170 | 0.2397 | 0.0724 |
| K5dfiACCs | 0.7148 [0.6504, 0.7747] | 0.6193 | 0.2378 | 0.0724 |
| K5dfi | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACCs - K5dfiACC | +0.0161 [+0.0029, +0.0361] | above | +0.0023 [-0.0089, +0.0148] | not separated |
| K5dfi - K5dfiACC | +0.0211 [-0.0483, +0.0921] | not separated | -0.0051 [-0.1132, +0.1094] | not separated |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiACC | 0.8266 [0.7897, 0.8611] | 0.7391 | 0.1624 | 0.0472 |
| K5dfiACCs | 0.8228 [0.7874, 0.8569] | 0.7420 | 0.1594 | 0.0472 |
| K5dfi | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACCs - K5dfiACC | -0.0037 [-0.0156, +0.0077] | not separated | +0.0029 [-0.0033, +0.0092] | not separated |
| K5dfi - K5dfiACC | -0.0772 [-0.1247, -0.0298] | below | -0.0299 [-0.0740, +0.0123] | not separated |
