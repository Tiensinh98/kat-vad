# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5_lneg_cap_fixed/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8979 [0.8676, 0.9244] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8617 [0.8303, 0.8898] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8576 [0.8312, 0.8823] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8827 [0.8582, 0.9050] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.8988 [0.8671, 0.9269] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8785 [0.8549, 0.9006]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9210 [0.9010, 0.9388] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8883 · clip oracle 0.7611
* micro AP (all videos, raw): 0.3729 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7515
* per-run whole macro: {'s2024': 0.921}
* per-run crop endpoint: {'s2024': 0.8785}
