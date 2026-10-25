# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'none', 'motion': 'none'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1740, 's2025': 1740, 's2026': 1740, 's2027': 1740, 's2028': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.7085 [0.6692, 0.7470] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.6862 [0.6544, 0.7183] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.6879 [0.6581, 0.7173] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.6851 [0.6486, 0.7210] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.7022 [0.6586, 0.7439] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.6974 [0.6680, 0.7258]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.6706 [0.6375, 0.7026] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.6637 · clip oracle 0.7584
* clip-level AUC (max score): 0.6141
* per-run whole macro: {'s2024': 0.6541, 's2025': 0.643, 's2026': 0.6706, 's2027': 0.6724, 's2028': 0.6474}
* per-run crop endpoint: {'s2024': 0.6766, 's2025': 0.6652, 's2026': 0.6855, 's2027': 0.6962, 's2028': 0.6727}
