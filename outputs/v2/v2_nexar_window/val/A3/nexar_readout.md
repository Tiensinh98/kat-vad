# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar/stage/data_window`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8656 [0.8309, 0.8959] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8081 [0.7736, 0.8407] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8252 [0.7923, 0.8559] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8238 [0.7890, 0.8563] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7963 [0.7441, 0.8454] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8224 [0.7926, 0.8514]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8894 [0.8668, 0.9109] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8589 · clip oracle 0.7611
* clip-level AUC (max score): 0.6826
* per-run whole macro: {'s2024': 0.8931, 's2025': 0.8905, 's2026': 0.8637, 's2027': 0.8773, 's2028': 0.8844}
* per-run crop endpoint: {'s2024': 0.8209, 's2025': 0.8168, 's2026': 0.8057, 's2027': 0.8083, 's2028': 0.8163}
