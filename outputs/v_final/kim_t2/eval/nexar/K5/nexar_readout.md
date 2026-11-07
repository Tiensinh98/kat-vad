# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5_t2/stage/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.9251 [0.8961, 0.9498] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8849 [0.8545, 0.9123] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8811 [0.8535, 0.9067] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8977 [0.8716, 0.9200] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.9083 [0.8781, 0.9351] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8981 [0.8749, 0.9197]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9245 [0.9034, 0.9431] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.9000 · clip oracle 0.7611
* micro AP (all videos, raw): 0.4472 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7735
* per-run whole macro: {'s2024': 0.9245}
* per-run crop endpoint: {'s2024': 0.8981}
