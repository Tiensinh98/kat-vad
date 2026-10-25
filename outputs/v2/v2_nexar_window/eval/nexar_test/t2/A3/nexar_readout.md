# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1740, 's2025': 1740, 's2026': 1740, 's2027': 1740, 's2028': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.7851 [0.7513, 0.8173] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.7580 [0.7258, 0.7889] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.7587 [0.7298, 0.7870] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.7249 [0.6892, 0.7588] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.7061 [0.6638, 0.7478] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.7474 [0.7193, 0.7742]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7872 [0.7602, 0.8128] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.7746 · clip oracle 0.7584
* clip-level AUC (max score): 0.6437
* per-run whole macro: {'s2024': 0.7657, 's2025': 0.7769, 's2026': 0.7851, 's2027': 0.7732, 's2028': 0.7937}
* per-run crop endpoint: {'s2024': 0.7348, 's2025': 0.7352, 's2026': 0.7344, 's2027': 0.7348, 's2028': 0.7552}
