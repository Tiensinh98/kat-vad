# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5bk25/nexar_defs`, runs ['s2024'] (steps {'s2024': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.9210 [0.8926, 0.9459] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8833 [0.8539, 0.9099] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8781 [0.8499, 0.9039] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8933 [0.8660, 0.9169] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.9018 [0.8707, 0.9292] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8942 [0.8703, 0.9164]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9210 [0.8991, 0.9402] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8961 · clip oracle 0.7611
* micro AP (all videos, raw): 0.4411 · positive-frame share 0.0394
* clip-level AUC (max score): 0.7659
* per-run whole macro: {'s2024': 0.921}
* per-run crop endpoint: {'s2024': 0.8942}
