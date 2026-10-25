# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'R2', 'motion': 'none'}`, data `/content/nexar/stage/data_window`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.7165 [0.6647, 0.7654] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7049 [0.6586, 0.7478] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7532 [0.7153, 0.7890] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7206 [0.6748, 0.7642] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.6768 [0.6216, 0.7302] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7132 [0.6758, 0.7497]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7741 [0.7377, 0.8084] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7399 · clip oracle 0.7611
* clip-level AUC (max score): 0.6049
* per-run whole macro: {'s2024': 0.7608, 's2025': 0.7668, 's2026': 0.7597, 's2027': 0.765, 's2028': 0.7512}
* per-run crop endpoint: {'s2024': 0.6991, 's2025': 0.7049, 's2026': 0.7049, 's2027': 0.7023, 's2028': 0.7054}
