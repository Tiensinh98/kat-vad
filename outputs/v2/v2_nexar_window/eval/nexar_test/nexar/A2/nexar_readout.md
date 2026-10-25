# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'none', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.9150 [0.8983, 0.9300] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.8630 [0.8428, 0.8819] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.8454 [0.8227, 0.8668] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.8792 [0.8596, 0.8976] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.9034 [0.8802, 0.9247] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.8815 [0.8637, 0.8979]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.9352 [0.9219, 0.9473] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.9105 · clip oracle 0.7584
* clip-level AUC (max score): 0.7892
* per-run whole macro: {'s2024': 0.9328, 's2025': 0.9319, 's2026': 0.9291, 's2027': 0.9142, 's2028': 0.9345}
* per-run crop endpoint: {'s2024': 0.8795, 's2025': 0.8754, 's2026': 0.8746, 's2027': 0.8554, 's2028': 0.8777}
