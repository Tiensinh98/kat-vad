# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar/stage/data_window`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8455 [0.8165, 0.8720] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8210 [0.7905, 0.8497] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8138 [0.7843, 0.8416] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8064 [0.7745, 0.8361] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7718 [0.7285, 0.8129] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8101 [0.7844, 0.8359]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8640 [0.8422, 0.8849] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8438 · clip oracle 0.7611
* clip-level AUC (max score): 0.6907
* per-run whole macro: {'s2024': 0.8594, 's2025': 0.8591, 's2026': 0.8409, 's2027': 0.8509, 's2028': 0.8583}
* per-run crop endpoint: {'s2024': 0.8097, 's2025': 0.799, 's2026': 0.7992, 's2027': 0.8001, 's2028': 0.8045}
