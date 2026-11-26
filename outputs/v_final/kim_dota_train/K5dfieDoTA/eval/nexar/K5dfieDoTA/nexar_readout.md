# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/k5dfidota_arms/nexar_defs`, runs ['s2024'] (steps {'s2024': 1764}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.7569 [0.7201, 0.7913] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7743 [0.7403, 0.8062] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7174 [0.6867, 0.7466] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.6719 [0.6410, 0.7020] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.6614 [0.6256, 0.6955] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7154 [0.6864, 0.7428]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7527 [0.7261, 0.7790] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.6975 · clip oracle 0.7611
* micro AP (all videos, raw): 0.1090 · positive-frame share 0.0394
* clip-level AUC (max score): 0.5810
* per-run whole macro: {'s2024': 0.7527}
* per-run crop endpoint: {'s2024': 0.7154}
