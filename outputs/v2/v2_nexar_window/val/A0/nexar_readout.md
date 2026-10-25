# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'none'}`, data `/content/nexar/stage/data_window`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8302 [0.7906, 0.8680] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7926 [0.7538, 0.8278] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7824 [0.7479, 0.8146] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7822 [0.7449, 0.8175] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7816 [0.7358, 0.8238] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7919 [0.7589, 0.8230]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8159 [0.7821, 0.8469] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7919 · clip oracle 0.7611
* clip-level AUC (max score): 0.6881
* per-run whole macro: {'s2024': 0.8148, 's2025': 0.8091, 's2026': 0.8099, 's2027': 0.751, 's2028': 0.815}
* per-run crop endpoint: {'s2024': 0.7873, 's2025': 0.7839, 's2026': 0.7947, 's2027': 0.7261, 's2028': 0.787}
