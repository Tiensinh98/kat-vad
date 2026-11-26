# §56 arms on DoTA train (KIM_RUN.md §56), s2024, n = 1 — printed, decides nothing; K5f stays

Corpus {'clips': 3020, 'clips_giving_normals': 2075, 'clips_with_both_kinds': 2051, 'clips_with_windows': 3019, 'row_abnormal_share': 0.323, 'train_abnormal': 9386, 'train_items': 12937, 'train_normal': 3551}; 1764 steps each; control K5dfiDoTA (§55, reused). Every row is IN-DOMAIN on DoTA-CAP-dev (position ruler 0.8545 [0.8386, 0.8695]; 566 / 569 dev clips share a YouTube id with train).

| row | sets beyond K5dfiDoTA | GPU |
|---|---|---|
| K5dfieDoTA | data.is_egocentric=true | NVIDIA L4 |
| K5dfibk25DoTA | loss.bottomk_weight=0.25, loss.bottomk_topk_pct=5 | NVIDIA L4 |
| K5dfiebk25DoTA | data.is_egocentric=true, loss.bottomk_weight=0.25, loss.bottomk_topk_pct=5 | NVIDIA L4 |

## Nexar val (zero-shot for every row here; crops macro, paired per video vs K5dfiDoTA)

| row | macro crops | macro whole | micro | AP | Δ crops vs K5dfiDoTA |
|---|---|---|---|---|---|
| K5dfiDoTA | 0.7363 | 0.7733 | 0.7113 | 0.1239 | — |
| K5dfieDoTA | 0.7154 | 0.7527 | 0.6975 | 0.1090 | -0.0209 [-0.0267, -0.0151] |
| K5dfibk25DoTA | 0.7328 | 0.7681 | 0.7091 | 0.1215 | -0.0035 [-0.0058, -0.0011] |
| K5dfiebk25DoTA | 0.7114 | 0.7467 | 0.6934 | 0.1034 | -0.0248 [-0.0312, -0.0186] |

## Four sets (control K5dfiDoTA; K5dfi = Nexar-trained, printed)

# k5dfidota_arms_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5dfiDoTA**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5dfiDoTA | 0.8590 / 0.8209 | 0.7659 / 0.7522 | 0.7579 / 0.6488 | 0.9287 / 0.9109 |
| K5dfieDoTA | 0.8470 / 0.8111 | 0.7462 / 0.7337 | 0.7583 / 0.6591 | 0.9394 / 0.9192 |
| K5dfibk25DoTA | 0.8605 / 0.8231 | 0.7677 / 0.7528 | 0.7589 / 0.6487 | 0.9140 / 0.9074 |
| K5dfiebk25DoTA | 0.8513 / 0.8157 | 0.7508 / 0.7363 | 0.7560 / 0.6605 | 0.9189 / 0.9098 |
| K5dfi | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiDoTA | 0.8590 [0.8468, 0.8709] | 0.8209 | 0.6776 | 0.3271 |
| K5dfieDoTA | 0.8470 [0.8340, 0.8595] | 0.8111 | 0.6536 | 0.3271 |
| K5dfibk25DoTA | 0.8605 [0.8485, 0.8722] | 0.8231 | 0.6837 | 0.3271 |
| K5dfiebk25DoTA | 0.8513 [0.8386, 0.8635] | 0.8157 | 0.6665 | 0.3271 |
| K5dfi | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfieDoTA - K5dfiDoTA | -0.0120 [-0.0149, -0.0091] | below | -0.0098 [-0.0127, -0.0068] | below |
| K5dfibk25DoTA - K5dfiDoTA | +0.0015 [+0.0001, +0.0028] | above | +0.0022 [+0.0011, +0.0034] | above |
| K5dfiebk25DoTA - K5dfiDoTA | -0.0077 [-0.0109, -0.0046] | below | -0.0052 [-0.0084, -0.0020] | below |
| K5dfi - K5dfiDoTA | -0.0331 [-0.0451, -0.0212] | below | -0.0595 [-0.0730, -0.0463] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiDoTA | 0.7659 [0.7469, 0.7850] | 0.7522 | 0.5715 | 0.2885 |
| K5dfieDoTA | 0.7462 [0.7274, 0.7651] | 0.7337 | 0.5479 | 0.2885 |
| K5dfibk25DoTA | 0.7677 [0.7491, 0.7866] | 0.7528 | 0.5754 | 0.2885 |
| K5dfiebk25DoTA | 0.7508 [0.7323, 0.7693] | 0.7363 | 0.5548 | 0.2885 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfieDoTA - K5dfiDoTA | -0.0197 [-0.0233, -0.0161] | below | -0.0185 [-0.0225, -0.0146] | below |
| K5dfibk25DoTA - K5dfiDoTA | +0.0018 [+0.0002, +0.0034] | above | +0.0006 [-0.0008, +0.0020] | not separated |
| K5dfiebk25DoTA - K5dfiDoTA | -0.0152 [-0.0189, -0.0114] | below | -0.0159 [-0.0200, -0.0119] | below |
| K5dfi - K5dfiDoTA | +0.0018 [-0.0173, +0.0207] | not separated | +0.0069 [-0.0128, +0.0256] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiDoTA | 0.7579 [0.6965, 0.8141] | 0.6488 | 0.2359 | 0.0724 |
| K5dfieDoTA | 0.7583 [0.6995, 0.8128] | 0.6591 | 0.2422 | 0.0724 |
| K5dfibk25DoTA | 0.7589 [0.6980, 0.8152] | 0.6487 | 0.2287 | 0.0724 |
| K5dfiebk25DoTA | 0.7560 [0.6965, 0.8113] | 0.6605 | 0.2373 | 0.0724 |
| K5dfi | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfieDoTA - K5dfiDoTA | +0.0004 [-0.0085, +0.0092] | not separated | +0.0103 [-0.0064, +0.0267] | not separated |
| K5dfibk25DoTA - K5dfiDoTA | +0.0011 [-0.0036, +0.0057] | not separated | -0.0002 [-0.0062, +0.0055] | not separated |
| K5dfiebk25DoTA - K5dfiDoTA | -0.0018 [-0.0111, +0.0074] | not separated | +0.0117 [-0.0043, +0.0279] | not separated |
| K5dfi - K5dfiDoTA | -0.0381 [-0.0887, +0.0141] | not separated | -0.0369 [-0.1139, +0.0403] | not separated |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfiDoTA | 0.9287 [0.9121, 0.9441] | 0.9109 | 0.3487 | 0.0472 |
| K5dfieDoTA | 0.9394 [0.9238, 0.9538] | 0.9192 | 0.3864 | 0.0472 |
| K5dfibk25DoTA | 0.9140 [0.8953, 0.9308] | 0.9074 | 0.3310 | 0.0472 |
| K5dfiebk25DoTA | 0.9189 [0.9009, 0.9354] | 0.9098 | 0.3447 | 0.0472 |
| K5dfi | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfieDoTA - K5dfiDoTA | +0.0107 [+0.0056, +0.0163] | above | +0.0083 [+0.0040, +0.0120] | above |
| K5dfibk25DoTA - K5dfiDoTA | -0.0147 [-0.0198, -0.0100] | below | -0.0035 [-0.0056, -0.0014] | below |
| K5dfiebk25DoTA - K5dfiDoTA | -0.0098 [-0.0177, -0.0023] | below | -0.0012 [-0.0055, +0.0030] | not separated |
| K5dfi - K5dfiDoTA | -0.1793 [-0.2149, -0.1452] | below | -0.2018 [-0.2335, -0.1708] | below |


## DAD: placebo and same-position (post-hoc kind of read, printed for every row by §56 DA5)

# DAD position diagnostics (POST-HOC, printed not gated, n = 1 seed)

165 positives, 301 negatives, 15 rows each; accident rows [13, 14]; first V2-S window without a repeated frame = row 7. **No-pixel `t/N` ruler macro = 1.0000.**

## Within video: positives vs the same mask on negatives (placebo)

| row | macro, all rows | placebo, all rows | gap | macro, rows 7+ | placebo, rows 7+ | gap |
|---|---|---|---|---|---|---|
| LaGoVAD | 0.6462 | 0.5134 [0.4709, 0.5564] | +0.1327 [+0.0615, +0.2020] | 0.6439 | 0.5260 [0.4817, 0.5692] | +0.1179 [+0.0440, +0.1895] |
| K5dfiDoTA | 0.9287 | 0.8916 [0.8717, 0.9100] | +0.0370 [+0.0121, +0.0621] | 0.8571 | 0.8236 [0.7946, 0.8519] | +0.0334 [-0.0085, +0.0736] |
| K5dfieDoTA | 0.9394 | 0.8971 [0.8775, 0.9153] | +0.0423 [+0.0186, +0.0666] | 0.8778 | 0.8295 [0.8009, 0.8571] | +0.0483 [+0.0079, +0.0880] |
| K5dfibk25DoTA | 0.9140 | 0.8818 [0.8611, 0.9010] | +0.0322 [+0.0055, +0.0588] | 0.8293 | 0.8048 [0.7757, 0.8336] | +0.0245 [-0.0201, +0.0661] |
| K5dfiebk25DoTA | 0.9189 | 0.8810 [0.8605, 0.9002] | +0.0378 [+0.0119, +0.0636] | 0.8399 | 0.7993 [0.7697, 0.8281] | +0.0406 [-0.0031, +0.0825] |

## Same position: positive vs negative videos (position cannot help)

| row | accident rows | rows 7 up to the accident | difference |
|---|---|---|---|
| LaGoVAD | 0.5881 [0.5339, 0.6419] | 0.5762 [0.5220, 0.6308] | +0.0118 [+0.0025, +0.0212] |
| K5dfiDoTA | 0.7553 [0.7077, 0.8005] | 0.7165 [0.6684, 0.7631] | +0.0388 [-0.0007, +0.0777] |
| K5dfieDoTA | 0.7596 [0.7116, 0.8049] | 0.7203 [0.6719, 0.7676] | +0.0393 [-0.0018, +0.0797] |
| K5dfibk25DoTA | 0.7522 [0.7040, 0.7974] | 0.7243 [0.6767, 0.7705] | +0.0279 [-0.0133, +0.0680] |
| K5dfiebk25DoTA | 0.7510 [0.7028, 0.7967] | 0.7301 [0.6823, 0.7770] | +0.0209 [-0.0213, +0.0622] |

Accident rows, paired:

* LaGoVAD - K5dfiDoTA: -0.1673 [-0.2294, -0.1045]
* K5dfieDoTA - K5dfiDoTA: +0.0043 [-0.0075, +0.0162]
* K5dfibk25DoTA - K5dfiDoTA: -0.0031 [-0.0096, +0.0035]
* K5dfiebk25DoTA - K5dfiDoTA: -0.0043 [-0.0185, +0.0099]

