# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/kimabl/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8389 [0.8121, 0.8646] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8153 [0.7879, 0.8418] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7975 [0.7683, 0.8245] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7800 [0.7476, 0.8099] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7554 [0.7203, 0.7890] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7973 [0.7693, 0.8234]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8214 [0.7962, 0.8459] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8084 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3181 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7732
* per-run whole macro: {'s2024': 0.8214}
* per-run crop endpoint: {'s2024': 0.7973}
