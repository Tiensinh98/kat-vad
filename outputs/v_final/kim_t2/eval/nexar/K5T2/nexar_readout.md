# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5_t2/stage/nexar_defs`, runs ['s2024'] (steps {'s2024': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8334 [0.7936, 0.8711] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8004 [0.7614, 0.8364] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7772 [0.7394, 0.8133] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7668 [0.7266, 0.8052] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7430 [0.6910, 0.7929] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7816 [0.7447, 0.8165]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8201 [0.7864, 0.8520] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7644 · clip oracle 0.7611
* micro AP (all videos, raw): 0.1629 · positive-frame share 0.0394
* clip-level AUC (max score): 0.6970
* per-run whole macro: {'s2024': 0.8201}
* per-run crop endpoint: {'s2024': 0.7816}
