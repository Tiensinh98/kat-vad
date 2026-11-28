# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'none'}`, data `/content/kim_clip_host_nexar/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.6781 [0.6482, 0.7075] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.6685 [0.6403, 0.6958] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.6683 [0.6437, 0.6930] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.6670 [0.6373, 0.6964] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.6651 [0.6328, 0.6968] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.6676 [0.6431, 0.6917]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.6858 [0.6590, 0.7125] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.6873 · clip oracle 0.7611
* micro AP (all videos, raw): 0.1108 · positive-frame share 0.0394
* clip-level AUC (max score): 0.6397
* per-run whole macro: {'s2024': 0.6858}
* per-run crop endpoint: {'s2024': 0.6676}
