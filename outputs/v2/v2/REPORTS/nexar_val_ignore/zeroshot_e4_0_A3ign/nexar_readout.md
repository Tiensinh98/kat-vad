# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar/stage/data_whole`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1740, 's2025': 1740, 's2026': 1740, 's2027': 1740, 's2028': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.7436 [0.6954, 0.7891] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.7238 [0.6759, 0.7696] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.6961 [0.6527, 0.7382] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.6545 [0.6064, 0.7008] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.6248 [0.5679, 0.6813] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.6888 [0.6486, 0.7276]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7387 [0.7014, 0.7743] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.7352 · clip oracle 0.7611
* clip-level AUC (max score): 0.6772
* per-run whole macro: {'s2024': 0.7387, 's2025': 0.7158, 's2026': 0.7406, 's2027': 0.7135, 's2028': 0.7497}
