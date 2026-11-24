# k5_edge_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5f**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |
| K5fepre | 0.8214 / 0.7409 | 0.7658 / 0.7512 | 0.7016 / 0.6647 | 0.8033 / 0.7228 |
| K5fepost | 0.8126 / 0.7471 | 0.7803 / 0.7730 | 0.7067 / 0.6366 | 0.7128 / 0.6911 |
| K5fbk | 0.8125 / 0.7793 | 0.7413 / 0.7452 | 0.6926 / 0.6192 | 0.7557 / 0.7401 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |
| K5fepre | 0.8214 [0.8069, 0.8354] | 0.7409 | 0.5700 | 0.3271 |
| K5fepost | 0.8126 [0.7969, 0.8273] | 0.7471 | 0.5847 | 0.3271 |
| K5fbk | 0.8125 [0.7976, 0.8272] | 0.7793 | 0.6110 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | +0.0026 [-0.0047, +0.0103] | not separated | +0.0056 [-0.0011, +0.0123] | not separated |
| K5fepost - K5f | -0.0062 [-0.0130, +0.0009] | not separated | +0.0119 [+0.0059, +0.0178] | above |
| K5fbk - K5f | -0.0063 [-0.0143, +0.0017] | not separated | +0.0440 [+0.0343, +0.0535] | above |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| K5fepre | 0.7658 [0.7410, 0.7896] | 0.7512 | 0.5455 | 0.2885 |
| K5fepost | 0.7803 [0.7568, 0.8034] | 0.7730 | 0.5783 | 0.2885 |
| K5fbk | 0.7413 [0.7178, 0.7642] | 0.7452 | 0.5380 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | -0.0086 [-0.0158, -0.0014] | below | -0.0039 [-0.0103, +0.0025] | not separated |
| K5fepost - K5f | +0.0059 [-0.0026, +0.0145] | not separated | +0.0179 [+0.0099, +0.0261] | above |
| K5fbk - K5f | -0.0331 [-0.0427, -0.0232] | below | -0.0099 [-0.0194, +0.0001] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |
| K5fepre | 0.7016 [0.6274, 0.7723] | 0.6647 | 0.2519 | 0.0724 |
| K5fepost | 0.7067 [0.6408, 0.7694] | 0.6366 | 0.2056 | 0.0724 |
| K5fbk | 0.6926 [0.6299, 0.7536] | 0.6192 | 0.2218 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | +0.0035 [-0.0167, +0.0259] | not separated | -0.0064 [-0.0594, +0.0513] | not separated |
| K5fepost - K5f | +0.0086 [-0.0170, +0.0348] | not separated | -0.0346 [-0.0722, +0.0038] | not separated |
| K5fbk - K5f | -0.0055 [-0.0338, +0.0238] | not separated | -0.0519 [-0.0881, -0.0176] | below |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |
| K5fepre | 0.8033 [0.7674, 0.8375] | 0.7228 | 0.1255 | 0.0472 |
| K5fepost | 0.7128 [0.6720, 0.7527] | 0.6911 | 0.0926 | 0.0472 |
| K5fbk | 0.7557 [0.7170, 0.7930] | 0.7401 | 0.1417 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | -0.0037 [-0.0198, +0.0124] | not separated | +0.0156 [+0.0053, +0.0253] | above |
| K5fepost - K5f | -0.0942 [-0.1161, -0.0732] | below | -0.0161 [-0.0262, -0.0065] | below |
| K5fbk - K5f | -0.0513 [-0.0732, -0.0301] | below | +0.0329 [+0.0194, +0.0456] | above |
