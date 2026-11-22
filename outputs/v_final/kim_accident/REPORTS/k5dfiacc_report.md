# K5dfiACC / L0dfiACC (KIM_RUN.md §39), s2024, n = 1 — printed, decides nothing

ACCIDENT_window: 1322 positive / 3038 train windows from 1339 clips (1494 after dedup of 1824); val 375 windows (161 two-class). Dedup: tau 0.7911 (STOP, resolved by §39 D′: top-1 >= 0.87 read by eye, no same camera), 0 clips excluded, 0 sources kept by the user. Plan {'epochs': 41, 'items_per_epoch': 2644, 'steps': 1722, 'steps_per_epoch': 42}. Baked with Nexar A2 statistics (§39 B).

## ACCIDENT val (in-domain, printed only)

| row | macro | micro | AP | score norm |
|---|---|---|---|---|
| K5dfiACC | 0.7667 | 0.6934 | 0.3055 | none |
| L0dfiACC | 0.5996 | 0.5700 | 0.1610 | none |

## Control K5dfi (P1 = TAD macro K5dfiACC − K5dfi)

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


## Control L0dfi (P2 = TAD macro L0dfiACC − L0dfi)

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


# TAD by category (`KIM_RUN.md` §38.2 N4) — printed, decides nothing

Macro = mean frame AUC over the category's two-class videos. Clip AUC = peak score of the category's videos vs every Normal video's peak. Micro = the category's frames pooled with every Normal video's frames (raw). n = 1 seed; per-category counts are small.

| category | videos | K5dfi macro / clip AUC / micro | K5dfiACC macro / clip AUC / micro | L0dfi macro / clip AUC / micro | L0dfiACC macro / clip AUC / micro |
| --- | --- | --- | --- | --- | --- |
| 01_Accident | 23 (23 two-class) | 0.8383 / 0.5761 / 0.6166 | 0.7350 / 0.7272 / 0.7329 | 0.6399 / 0.6065 / 0.6857 | 0.7022 / 0.7152 / 0.7964 |
| 02_IllegalTurn | 3 (3 two-class) | 0.7903 / 0.6083 / 0.6901 | 0.7935 / 0.8833 / 0.7768 | 0.6913 / 0.4167 / 0.5553 | 0.7142 / 0.5750 / 0.6837 |
| 03_IllegalOccupation | 3 (3 two-class) | 0.6529 / 0.7333 / 0.8242 | 0.4735 / 0.3750 / 0.3568 | 0.5666 / 0.9167 / 0.9661 | 0.5047 / 0.5750 / 0.6577 |
| 04_Retrograde | 3 (3 two-class) | 0.7689 / 0.8167 / 0.6566 | 0.4914 / 0.4417 / 0.2760 | 0.7547 / 0.9333 / 0.8668 | 0.6896 / 0.7333 / 0.6908 |
| 05_else | 14 (14 two-class) | 0.7194 / 0.5696 / 0.5882 | 0.7709 / 0.5750 / 0.6004 | 0.4429 / 0.5661 / 0.6238 | 0.7127 / 0.6393 / 0.7562 |
| 06_PedestrianOnRoad | 10 (10 two-class) | 0.4746 / 0.5950 / 0.5012 | 0.5361 / 0.5950 / 0.5183 | 0.5795 / 0.7225 / 0.6107 | 0.6533 / 0.6725 / 0.7115 |
| 07_RoadSpills | 4 (4 two-class) | 0.6137 / 0.6687 / 0.6624 | 0.8973 / 0.8188 / 0.8887 | 0.6657 / 0.6312 / 0.7659 | 0.7588 / 0.8500 / 0.9140 |
| all abnormal | 60 (60 two-class) | 0.7198 / 0.6054 / 0.6119 | 0.6987 / 0.6517 / 0.6170 | 0.5902 / 0.6404 / 0.6733 | 0.6904 / 0.6863 / 0.7389 |

