# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar/stage/data_window`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.9160 [0.8890, 0.9391] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8741 [0.8470, 0.8985] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8645 [0.8391, 0.8881] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8930 [0.8675, 0.9155] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.9020 [0.8677, 0.9322] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8887 [0.8648, 0.9102]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9324 [0.9147, 0.9483] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.9092 · clip oracle 0.7611
* clip-level AUC (max score): 0.7549
* per-run whole macro: {'s2024': 0.9298, 's2025': 0.929, 's2026': 0.9265, 's2027': 0.91, 's2028': 0.9288}
* per-run crop endpoint: {'s2024': 0.8859, 's2025': 0.8853, 's2026': 0.879, 's2027': 0.8657, 's2028': 0.8785}
