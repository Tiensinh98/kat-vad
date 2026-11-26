# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5dfidota_arms/nexar_defs`, runs ['s2024'] (steps {'s2024': 1764}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.7704 [0.7337, 0.8048] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7678 [0.7347, 0.7990] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7091 [0.6792, 0.7376] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.6627 [0.6328, 0.6922] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.6527 [0.6184, 0.6855] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7114 [0.6834, 0.7382]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7467 [0.7204, 0.7728] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.6934 · clip oracle 0.7611
* micro AP (all videos, raw): 0.1034 · positive-frame share 0.0394
* clip-level AUC (max score): 0.5533
* per-run whole macro: {'s2024': 0.7467}
* per-run crop endpoint: {'s2024': 0.7114}
