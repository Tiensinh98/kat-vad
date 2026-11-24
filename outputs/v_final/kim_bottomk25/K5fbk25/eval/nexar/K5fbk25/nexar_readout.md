# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5bk25/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.9007 [0.8724, 0.9263] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8609 [0.8309, 0.8884] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8534 [0.8253, 0.8799] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8642 [0.8367, 0.8891] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.8677 [0.8360, 0.8970] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8679 [0.8429, 0.8913]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8977 [0.8748, 0.9187] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8754 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3978 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7645
* per-run whole macro: {'s2024': 0.8977}
* per-run crop endpoint: {'s2024': 0.8679}
