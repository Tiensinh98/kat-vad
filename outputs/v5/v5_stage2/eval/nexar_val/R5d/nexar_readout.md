# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'v5_readout': 'mqr', 'v5_queries': 8, 'v5_keys': 'change', 'v5_smil_weight': 0.5}`, data `/content/v5s2/stage/data_window`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.9021 [0.8746, 0.9261] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8611 [0.8338, 0.8861] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8462 [0.8187, 0.8715] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8783 [0.8530, 0.9017] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.8868 [0.8524, 0.9172] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8739 [0.8490, 0.8965]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9264 [0.9084, 0.9426] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.9009 · clip oracle 0.7611
* clip-level AUC (max score): 0.7452
* per-run whole macro: {'s2024': 0.9264}
* per-run crop endpoint: {'s2024': 0.8739}
