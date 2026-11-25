# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/kimabl/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8566 [0.8308, 0.8814] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8350 [0.8081, 0.8604] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8187 [0.7920, 0.8439] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8035 [0.7728, 0.8322] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7923 [0.7584, 0.8243] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8205 [0.7942, 0.8452]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8490 [0.8253, 0.8719] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8288 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3314 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7582
* per-run whole macro: {'s2024': 0.849}
* per-run crop endpoint: {'s2024': 0.8205}
