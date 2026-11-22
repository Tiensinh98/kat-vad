# k5dfiacc_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5dfi**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5dfi | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |
| K5dfiACC | 0.7372 / 0.6815 | 0.6593 / 0.6420 | 0.6987 / 0.6170 | 0.8266 / 0.7391 |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfi | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |
| K5dfiACC | 0.7372 [0.7169, 0.7582] | 0.6815 | 0.4886 | 0.3271 |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACC - K5dfi | -0.0887 [-0.1102, -0.0670] | below | -0.0798 [-0.1024, -0.0578] | below |
| K5f - K5dfi | -0.0071 [-0.0153, +0.0009] | not separated | -0.0261 [-0.0335, -0.0188] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5dfiACC | 0.6593 [0.6371, 0.6824] | 0.6420 | 0.4027 | 0.2885 |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACC - K5dfi | -0.1084 [-0.1352, -0.0816] | below | -0.1171 [-0.1422, -0.0912] | below |
| K5f - K5dfi | +0.0067 [-0.0024, +0.0156] | not separated | -0.0040 [-0.0128, +0.0044] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfi | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |
| K5dfiACC | 0.6987 [0.6335, 0.7609] | 0.6170 | 0.2397 | 0.0724 |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACC - K5dfi | -0.0211 [-0.0921, +0.0483] | not separated | +0.0051 [-0.1094, +0.1132] | not separated |
| K5f - K5dfi | -0.0217 [-0.0485, +0.0018] | not separated | +0.0592 [+0.0224, +0.0943] | above |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfi | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |
| K5dfiACC | 0.8266 [0.7897, 0.8611] | 0.7391 | 0.1624 | 0.0472 |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiACC - K5dfi | +0.0772 [+0.0298, +0.1247] | above | +0.0299 [-0.0123, +0.0740] | not separated |
| K5f - K5dfi | +0.0576 [+0.0378, +0.0779] | above | -0.0019 [-0.0119, +0.0086] | not separated |
