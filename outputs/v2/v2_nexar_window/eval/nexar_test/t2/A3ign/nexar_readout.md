# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1740, 's2025': 1740, 's2026': 1740, 's2027': 1740, 's2028': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.7811 [0.7494, 0.8113] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.7581 [0.7271, 0.7881] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.7470 [0.7185, 0.7748] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.7136 [0.6797, 0.7467] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.6999 [0.6608, 0.7388] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.7411 [0.7137, 0.7676]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.7767 [0.7504, 0.8021] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.7663 · clip oracle 0.7584
* clip-level AUC (max score): 0.6356
* per-run whole macro: {'s2024': 0.7645, 's2025': 0.7589, 's2026': 0.7801, 's2027': 0.753, 's2028': 0.7772}
* per-run crop endpoint: {'s2024': 0.7368, 's2025': 0.72, 's2026': 0.736, 's2027': 0.726, 's2028': 0.7463}
