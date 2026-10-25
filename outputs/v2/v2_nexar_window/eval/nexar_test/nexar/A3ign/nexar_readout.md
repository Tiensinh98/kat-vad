# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.8465 [0.8258, 0.8661] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.8261 [0.8049, 0.8465] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.8208 [0.8004, 0.8404] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.7973 [0.7721, 0.8215] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.7815 [0.7510, 0.8113] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.8144 [0.7946, 0.8330]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8775 [0.8600, 0.8939] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.8616 · clip oracle 0.7584
* clip-level AUC (max score): 0.7410
* per-run whole macro: {'s2024': 0.8758, 's2025': 0.8664, 's2026': 0.8701, 's2027': 0.8647, 's2028': 0.8663}
* per-run crop endpoint: {'s2024': 0.8099, 's2025': 0.8004, 's2026': 0.81, 's2027': 0.8054, 's2028': 0.8061}
