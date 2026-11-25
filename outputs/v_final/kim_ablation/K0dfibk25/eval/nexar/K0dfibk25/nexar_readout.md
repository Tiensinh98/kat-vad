# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/kimabl/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8553 [0.8311, 0.8784] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8430 [0.8186, 0.8660] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8227 [0.7975, 0.8468] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8088 [0.7809, 0.8349] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7919 [0.7598, 0.8224] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8238 [0.7996, 0.8469]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8508 [0.8289, 0.8718] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8305 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3339 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7564
* per-run whole macro: {'s2024': 0.8508}
* per-run crop endpoint: {'s2024': 0.8238}
