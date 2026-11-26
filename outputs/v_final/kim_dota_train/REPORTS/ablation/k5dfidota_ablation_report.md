# §58 ablation on DoTA train (KIM_RUN.md §58), s2024, n = 1 — printed, decides nothing; K5f stays

Corpus {'clips': 3020, 'clips_giving_normals': 2075, 'clips_with_both_kinds': 2051, 'clips_with_windows': 3019, 'row_abnormal_share': 0.323, 'train_abnormal': 9386, 'train_items': 12937, 'train_normal': 3551}; 1764 steps per row. KIM-on side **K5dfiDoTA** — start gate (KA3): **matched (saved K5dfiDoTA)** (step-1 mil off 0.731826, on 0.731826). DoTA-CAP-dev is IN-DOMAIN for every row (566 / 569 dev clips share a YouTube id with train).

| row | train | input |
|---|---|---|
| K5dfiDoTA | VideoMAE V2-S + KIM | NVIDIA L4 |
| K0dfiDoTA | VideoMAE V2-S, KIM off | NVIDIA L4 |
| L0dfiDoTA | LaGoVAD trunk (CLIP only) | NVIDIA L4 |

## Cells (macro [95 %] / micro / AP; micro and AP after per-clip min-max)

| row | DoTA-CAP-dev (in-domain) | DADA-2000 (zero-shot) |
|---|---|---|
| K5dfiDoTA | 0.8590 [0.8468, 0.8709] / 0.8209 / 0.6776 | 0.7659 [0.7469, 0.7850] / 0.7522 / 0.5715 |
| K0dfiDoTA | 0.8532 [0.8409, 0.8651] / 0.8156 / 0.6817 | 0.7675 [0.7488, 0.7866] / 0.7553 / 0.5753 |
| L0dfiDoTA | 0.6682 [0.6536, 0.6826] / 0.6412 / 0.5010 | 0.5911 [0.5763, 0.6063] / 0.5849 / 0.3892 |
| *position only (no pixels)* | *cubic ruler 0.8545 [0.8386, 0.8695]* | *middle ruler 0.7342 [0.7076, 0.7594]* |

## Contrasts (paired; labels §53 KA5: GAIN ≥ +0.01 and interval above 0, COST mirror, else NO DIFFERENCE)

| component | contrast | set | Δ micro | Δ macro |
|---|---|---|---|---|
| KIM | K5dfiDoTA − K0dfiDoTA | DoTA-CAP-dev (in-domain) | +0.0053 [+0.0007, +0.0098] NO DIFFERENCE | +0.0058 [+0.0015, +0.0101] NO DIFFERENCE |
| KIM | K5dfiDoTA − K0dfiDoTA | DADA-2000 (zero-shot) | -0.0031 [-0.0098, +0.0034] NO DIFFERENCE | -0.0016 [-0.0083, +0.0050] NO DIFFERENCE |
| VideoMAE motion | K0dfiDoTA − L0dfiDoTA | DoTA-CAP-dev (in-domain) | +0.1745 [+0.1627, +0.1864] GAIN | +0.1850 [+0.1720, +0.1980] GAIN |
| VideoMAE motion | K0dfiDoTA − L0dfiDoTA | DADA-2000 (zero-shot) | +0.1704 [+0.1524, +0.1878] GAIN | +0.1764 [+0.1585, +0.1944] GAIN |
| total (motion + KIM) | K5dfiDoTA − L0dfiDoTA | DoTA-CAP-dev (in-domain) | +0.1797 [+0.1674, +0.1928] GAIN | +0.1908 [+0.1773, +0.2043] GAIN |
| total (motion + KIM) | K5dfiDoTA − L0dfiDoTA | DADA-2000 (zero-shot) | +0.1674 [+0.1480, +0.1856] GAIN | +0.1748 [+0.1558, +0.1937] GAIN |

Nexar-trained reference (printed, other corpus): §53.1 K5dfi KIM on − off micro DoTA −0.0041 n.s., DADA +0.0229; §36.1 K5dfi − L0dfi (motion + KIM) macro DoTA-CAP (whole) +0.147, DADA +0.193.

# k5dfidota_abl_vs_k0_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K0dfiDoTA**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 |
| --- | --- | --- |
| K0dfiDoTA | 0.8532 / 0.8156 | 0.7675 / 0.7553 |
| K5dfiDoTA | 0.8590 / 0.8209 | 0.7659 / 0.7522 |
| L0dfiDoTA | 0.6682 / 0.6412 | 0.5911 / 0.5849 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfiDoTA | 0.8532 [0.8409, 0.8651] | 0.8156 | 0.6817 | 0.3271 |
| K5dfiDoTA | 0.8590 [0.8468, 0.8709] | 0.8209 | 0.6776 | 0.3271 |
| L0dfiDoTA | 0.6682 [0.6536, 0.6826] | 0.6412 | 0.5010 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiDoTA - K0dfiDoTA | +0.0058 [+0.0015, +0.0101] | above | +0.0053 [+0.0007, +0.0098] | above |
| L0dfiDoTA - K0dfiDoTA | -0.1850 [-0.1980, -0.1720] | below | -0.1745 [-0.1864, -0.1627] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfiDoTA | 0.7675 [0.7488, 0.7866] | 0.7553 | 0.5753 | 0.2885 |
| K5dfiDoTA | 0.7659 [0.7469, 0.7850] | 0.7522 | 0.5715 | 0.2885 |
| L0dfiDoTA | 0.5911 [0.5763, 0.6063] | 0.5849 | 0.3892 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiDoTA - K0dfiDoTA | -0.0016 [-0.0083, +0.0050] | not separated | -0.0031 [-0.0098, +0.0034] | not separated |
| L0dfiDoTA - K0dfiDoTA | -0.1764 [-0.1944, -0.1585] | below | -0.1704 [-0.1878, -0.1524] | below |


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


