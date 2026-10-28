# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant', 'v5_readout': 'mqr', 'v5_queries': 8, 'v5_keys': 'content', 'v5_smil_weight': 0.5}`, data `/content/v5s2/stage/data_window`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.9019 [0.8743, 0.9259] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8609 [0.8335, 0.8859] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8460 [0.8185, 0.8714] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8782 [0.8529, 0.9015] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.8868 [0.8525, 0.9172] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8737 [0.8489, 0.8965]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9263 [0.9084, 0.9425] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.9008 · clip oracle 0.7611
* clip-level AUC (max score): 0.7442
* per-run whole macro: {'s2024': 0.9263}
* per-run crop endpoint: {'s2024': 0.8737}
