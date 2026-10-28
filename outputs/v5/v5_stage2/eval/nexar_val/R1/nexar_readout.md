# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'v5_readout': 'mean', 'v5_queries': 1, 'v5_keys': 'content', 'v5_smil_weight': 0.0}`, data `/content/v5s2/stage/data_window`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.9156 [0.8895, 0.9381] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8686 [0.8413, 0.8934] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8627 [0.8360, 0.8877] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8876 [0.8600, 0.9119] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.8994 [0.8645, 0.9303] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8858 [0.8610, 0.9083]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9303 [0.9111, 0.9475] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.9084 · clip oracle 0.7611
* clip-level AUC (max score): 0.7699
* per-run whole macro: {'s2024': 0.9303}
* per-run crop endpoint: {'s2024': 0.8858}
