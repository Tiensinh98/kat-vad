# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/kimabl/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8436 [0.8168, 0.8693] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8143 [0.7864, 0.8411] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8022 [0.7727, 0.8301] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7994 [0.7678, 0.8291] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7726 [0.7364, 0.8066] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8064 [0.7780, 0.8328]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8337 [0.8079, 0.8587] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8222 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3584 · positive-frame share 0.0394
* clip-level AUC (max score): 0.8025
* per-run whole macro: {'s2024': 0.8337}
* per-run crop endpoint: {'s2024': 0.8064}
