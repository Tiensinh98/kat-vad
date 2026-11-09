# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5_whole/nexar_defs`, runs ['s2024'] (steps {'s2024': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.6397 [0.5828, 0.6945] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.5840 [0.5360, 0.6323] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.6389 [0.6011, 0.6765] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7100 [0.6697, 0.7493] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7082 [0.6512, 0.7627] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.6547 [0.6154, 0.6938]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7012 [0.6614, 0.7406] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.6895 · clip oracle 0.7611
* micro AP (all videos, raw): 0.0926 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7383
* per-run whole macro: {'s2024': 0.7012}
* per-run crop endpoint: {'s2024': 0.6547}
