# K5fno vs K5f (KIM_RUN.md §38.2), s2024, n = 1

K5fno = K5f config; corpus `Nexar_window_negonly`: 5781 normal windows from 449 negative videos (cap 14), none from a positive video. Printed only: K5f stays the reference (N6).

## Nexar val (crops macro, paired per video)

| row | macro crops | macro whole | micro | AP | clip AUC | FA share (peak > 0.5, negatives) |
|---|---|---|---|---|---|---|
| K5f | 0.8942 | 0.9210 | 0.8961 | 0.4411 | 0.7659 | 0.549 |
| K5fno | 0.8000 | 0.8348 | 0.7853 | 0.1771 | 0.7698 | 0.478 |

K5fno - K5f: -0.0942 [-0.1234, -0.0663] -> below

## Zero-shot, four sets (N5 primary: TAD micro)

# k5fno_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5f**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |
| K5fno | 0.7476 / 0.6398 | 0.7013 / 0.6561 | 0.6951 / 0.5008 | 0.7906 / 0.6556 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |
| K5fno | 0.7476 [0.7297, 0.7652] | 0.6398 | 0.4492 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fno - K5f | -0.0712 [-0.0876, -0.0546] | below | -0.0954 [-0.1121, -0.0792] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| K5fno | 0.7013 [0.6758, 0.7264] | 0.6561 | 0.4223 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fno - K5f | -0.0731 [-0.0928, -0.0534] | below | -0.0990 [-0.1176, -0.0795] | below |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |
| K5fno | 0.6951 [0.6164, 0.7683] | 0.5008 | 0.0848 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fno - K5f | -0.0030 [-0.0555, +0.0510] | not separated | -0.1704 [-0.3013, -0.0430] | below |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |
| K5fno | 0.7906 [0.7482, 0.8319] | 0.6556 | 0.1098 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fno - K5f | -0.0164 [-0.0617, +0.0301] | not separated | -0.0515 [-0.0853, -0.0172] | below |


# TAD by category (`KIM_RUN.md` §38.2 N4) — printed, decides nothing

Macro = mean frame AUC over the category's two-class videos. Clip AUC = peak score of the category's videos vs every Normal video's peak. Micro = the category's frames pooled with every Normal video's frames (raw). n = 1 seed; per-category counts are small.

| category | videos | K5f macro / clip AUC / micro | K5fno macro / clip AUC / micro |
| --- | --- | --- | --- |
| 01_Accident | 23 (23 two-class) | 0.8230 / 0.5902 / 0.6711 | 0.7881 / 0.4935 / 0.5659 |
| 02_IllegalTurn | 3 (3 two-class) | 0.8544 / 0.6750 / 0.8171 | 0.8409 / 0.4500 / 0.4898 |
| 03_IllegalOccupation | 3 (3 two-class) | 0.6425 / 0.6833 / 0.8181 | 0.5668 / 0.5750 / 0.7210 |
| 04_Retrograde | 3 (3 two-class) | 0.7883 / 0.7917 / 0.6986 | 0.6886 / 0.3667 / 0.2632 |
| 05_else | 14 (14 two-class) | 0.6626 / 0.6214 / 0.6324 | 0.7316 / 0.3357 / 0.4135 |
| 06_PedestrianOnRoad | 10 (10 two-class) | 0.4437 / 0.6150 / 0.6256 | 0.4929 / 0.3500 / 0.3906 |
| 07_RoadSpills | 4 (4 two-class) | 0.5974 / 0.6562 / 0.7465 | 0.5303 / 0.5062 / 0.5629 |
| all abnormal | 60 (60 two-class) | 0.6981 / 0.6250 / 0.6711 | 0.6951 / 0.4292 / 0.5008 |

