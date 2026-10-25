# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'R2', 'motion': 'none'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.7496 [0.7141, 0.7832] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.7449 [0.7142, 0.7750] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.7848 [0.7603, 0.8090] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.7474 [0.7155, 0.7779] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.7135 [0.6739, 0.7526] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.7491 [0.7242, 0.7737]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8016 [0.7749, 0.8273] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.7744 · clip oracle 0.7584
* clip-level AUC (max score): 0.6684
* per-run whole macro: {'s2024': 0.7918, 's2025': 0.7906, 's2026': 0.7875, 's2027': 0.7915, 's2028': 0.7859}
* per-run crop endpoint: {'s2024': 0.7388, 's2025': 0.7379, 's2026': 0.7376, 's2027': 0.7397, 's2028': 0.7313}
