# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5_lneg/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8219 [0.7931, 0.8488] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7887 [0.7587, 0.8174] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7751 [0.7457, 0.8032] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7641 [0.7318, 0.7948] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7443 [0.7080, 0.7799] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7779 [0.7485, 0.8056]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7969 [0.7689, 0.8243] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7801 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3121 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7675
* per-run whole macro: {'s2024': 0.7969}
* per-run crop endpoint: {'s2024': 0.7779}
