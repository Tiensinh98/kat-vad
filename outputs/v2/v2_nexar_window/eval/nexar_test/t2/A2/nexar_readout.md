# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1740, 's2025': 1740, 's2026': 1740, 's2027': 1740, 's2028': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.8453 [0.8183, 0.8705] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.7999 [0.7744, 0.8253] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.7829 [0.7569, 0.8084] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.7851 [0.7562, 0.8127] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.7998 [0.7650, 0.8328] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.8055 [0.7819, 0.8285]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8415 [0.8191, 0.8640] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.7796 · clip oracle 0.7584
* clip-level AUC (max score): 0.6803
* per-run whole macro: {'s2024': 0.8304, 's2025': 0.8294, 's2026': 0.8428, 's2027': 0.8255, 's2028': 0.8431}
* per-run crop endpoint: {'s2024': 0.7963, 's2025': 0.7883, 's2026': 0.8093, 's2027': 0.7901, 's2028': 0.8044}
