# k5_bottomk25_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5frr**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5frr | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |
| K5fbk25 | 0.8268 / 0.7714 | 0.7624 / 0.7583 | 0.7073 / 0.6562 | 0.8063 / 0.7304 |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |
| K5fbk25 | 0.8268 [0.8127, 0.8404] | 0.7714 | 0.6044 | 0.3271 |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | +0.0080 [+0.0041, +0.0121] | above | +0.0362 [+0.0312, +0.0414] | above |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| K5fbk25 | 0.7624 [0.7381, 0.7862] | 0.7583 | 0.5557 | 0.2885 |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | -0.0120 [-0.0170, -0.0069] | below | +0.0032 [-0.0017, +0.0086] | not separated |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |
| K5fbk25 | 0.7073 [0.6367, 0.7754] | 0.6562 | 0.2634 | 0.0724 |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | +0.0092 [-0.0030, +0.0216] | not separated | -0.0150 [-0.0268, -0.0034] | below |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |
| K5fbk25 | 0.8063 [0.7681, 0.8427] | 0.7304 | 0.1304 | 0.0472 |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | -0.0007 [-0.0084, +0.0070] | not separated | +0.0232 [+0.0184, +0.0279] | above |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
