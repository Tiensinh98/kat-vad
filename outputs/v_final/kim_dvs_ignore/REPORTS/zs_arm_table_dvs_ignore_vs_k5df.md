# dvs_ignore_vs_k5df — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5df**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |
| K5df | 0.8164 / 0.7038 | 0.7728 / 0.7349 | 0.7033 / 0.6538 | 0.7562 / 0.6613 |
| K5dfi | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |
| K5dfie | 0.8263 / 0.7580 | 0.7724 / 0.7598 | 0.7214 / 0.6546 | 0.7508 / 0.7081 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |
| K5df | 0.8164 [0.8025, 0.8299] | 0.7038 | 0.5337 | 0.3271 |
| K5dfi | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |
| K5dfie | 0.8263 [0.8121, 0.8400] | 0.7580 | 0.5923 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5f - K5df | +0.0024 [-0.0058, +0.0100] | not separated | +0.0315 [+0.0226, +0.0402] | above |
| K5dfi - K5df | +0.0095 [+0.0033, +0.0159] | above | +0.0576 [+0.0494, +0.0655] | above |
| K5dfie - K5df | +0.0098 [+0.0030, +0.0161] | above | +0.0542 [+0.0445, +0.0636] | above |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| K5df | 0.7728 [0.7488, 0.7967] | 0.7349 | 0.5285 | 0.2885 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5dfie | 0.7724 [0.7487, 0.7960] | 0.7598 | 0.5607 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5f - K5df | +0.0016 [-0.0081, +0.0114] | not separated | +0.0201 [+0.0114, +0.0290] | above |
| K5dfi - K5df | -0.0051 [-0.0132, +0.0033] | not separated | +0.0241 [+0.0164, +0.0317] | above |
| K5dfie - K5df | -0.0004 [-0.0092, +0.0084] | not separated | +0.0248 [+0.0160, +0.0332] | above |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |
| K5df | 0.7033 [0.6289, 0.7738] | 0.6538 | 0.2482 | 0.0724 |
| K5dfi | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |
| K5dfie | 0.7214 [0.6528, 0.7875] | 0.6546 | 0.2386 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5f - K5df | -0.0052 [-0.0337, +0.0224] | not separated | +0.0173 [-0.0136, +0.0508] | not separated |
| K5dfi - K5df | +0.0165 [-0.0064, +0.0399] | not separated | -0.0419 [-0.0781, -0.0050] | below |
| K5dfie - K5df | +0.0182 [-0.0039, +0.0403] | not separated | +0.0008 [-0.0324, +0.0327] | not separated |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |
| K5df | 0.7562 [0.7184, 0.7925] | 0.6613 | 0.0947 | 0.0472 |
| K5dfi | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |
| K5dfie | 0.7508 [0.7100, 0.7895] | 0.7081 | 0.1120 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5f - K5df | +0.0508 [+0.0273, +0.0751] | above | +0.0459 [+0.0355, +0.0564] | above |
| K5dfi - K5df | -0.0068 [-0.0284, +0.0145] | not separated | +0.0478 [+0.0355, +0.0591] | above |
| K5dfie - K5df | -0.0054 [-0.0280, +0.0172] | not separated | +0.0468 [+0.0358, +0.0576] | above |
