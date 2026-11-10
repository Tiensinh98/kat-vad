# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5_neg/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8964 [0.8644, 0.9241] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8495 [0.8183, 0.8774] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8412 [0.8127, 0.8676] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8525 [0.8242, 0.8787] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.8649 [0.8308, 0.8967] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8597 [0.8333, 0.8841]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8838 [0.8585, 0.9068] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8497 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3573 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7909
* per-run whole macro: {'s2024': 0.8838}
* per-run crop endpoint: {'s2024': 0.8597}
