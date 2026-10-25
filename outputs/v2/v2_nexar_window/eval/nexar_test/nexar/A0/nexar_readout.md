# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'none', 'motion': 'none'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.8445 [0.8158, 0.8711] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.8078 [0.7832, 0.8315] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.7929 [0.7681, 0.8166] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.7955 [0.7673, 0.8225] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.8141 [0.7827, 0.8442] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.8116 [0.7881, 0.8335]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8406 [0.8171, 0.8630] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.8263 · clip oracle 0.7584
* clip-level AUC (max score): 0.6949
* per-run whole macro: {'s2024': 0.8353, 's2025': 0.8332, 's2026': 0.8418, 's2027': 0.7729, 's2028': 0.8326}
* per-run crop endpoint: {'s2024': 0.8062, 's2025': 0.8008, 's2026': 0.808, 's2027': 0.7422, 's2028': 0.7991}
