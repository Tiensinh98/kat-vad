# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.8505 [0.8263, 0.8732] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.8035 [0.7788, 0.8277] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.8269 [0.8040, 0.8489] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.8149 [0.7867, 0.8413] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.8082 [0.7723, 0.8431] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.8207 [0.7981, 0.8415]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8959 [0.8771, 0.9134] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.8768 · clip oracle 0.7584
* clip-level AUC (max score): 0.7625
* per-run whole macro: {'s2024': 0.8974, 's2025': 0.894, 's2026': 0.8831, 's2027': 0.8814, 's2028': 0.8826}
* per-run crop endpoint: {'s2024': 0.8205, 's2025': 0.8115, 's2026': 0.8112, 's2027': 0.8087, 's2028': 0.8078}
