# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'none'}`, data `/content/kim_clip_host_nexar/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8005 [0.7656, 0.8326] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7804 [0.7475, 0.8110] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7436 [0.7129, 0.7736] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7173 [0.6806, 0.7535] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7296 [0.6858, 0.7699] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7527 [0.7228, 0.7817]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7766 [0.7463, 0.8052] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7598 · clip oracle 0.7611
* micro AP (all videos, raw): 0.1633 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7085
* per-run whole macro: {'s2024': 0.7766}
* per-run crop endpoint: {'s2024': 0.7527}
