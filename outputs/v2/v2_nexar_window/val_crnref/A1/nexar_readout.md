# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'R2', 'motion': 'none'}`, data `/content/drive/MyDrive/Thesis-V2/data/Nexar_window`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN over the source video (§30), native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.7495 [0.6975, 0.7973] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7161 [0.6714, 0.7587] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.7295 [0.6894, 0.7687] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.7279 [0.6807, 0.7710] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.7540 [0.7052, 0.8016] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.7338 [0.6949, 0.7716]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7741 [0.7377, 0.8084] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7399 · clip oracle 0.7611
* clip-level AUC (max score): 0.6049
* per-run whole macro: {'s2024': 0.7608, 's2025': 0.7668, 's2026': 0.7597, 's2027': 0.765, 's2028': 0.7512}
* per-run crop endpoint: {'s2024': 0.7183, 's2025': 0.7298, 's2026': 0.7201, 's2027': 0.7243, 's2028': 0.718}
