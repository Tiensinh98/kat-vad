# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5dfidota/nexar_defs`, runs ['s2024'] (steps {'s2024': 1764}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.7781 [0.7414, 0.8126] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7925 [0.7586, 0.8242] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7366 [0.7044, 0.7665] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.6937 [0.6612, 0.7249] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.6876 [0.6496, 0.7242] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7363 [0.7063, 0.7641]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7733 [0.7449, 0.8005] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7113 · clip oracle 0.7611
* micro AP (all videos, raw): 0.1239 · positive-frame share 0.0394
* clip-level AUC (max score): 0.5977
* per-run whole macro: {'s2024': 0.7733}
* per-run crop endpoint: {'s2024': 0.7363}
