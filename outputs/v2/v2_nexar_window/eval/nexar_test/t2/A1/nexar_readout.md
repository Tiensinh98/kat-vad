# Nexar read-out — nexar_test (375 videos, 187 positive)

Arm `{'crn': 'R2', 'motion': 'none'}`, data `/content/nexar_full/stage/nexar_defs`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1740, 's2025': 1740, 's2026': 1740, 's2027': 1740, 's2028': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 185 | 0.6664 [0.6243, 0.7083] | 0.2080 [0.1993, 0.2172] |
| 0.3 | 186 | 0.6649 [0.6269, 0.7011] | 0.6589 [0.6561, 0.6619] |
| 0.5 | 186 | 0.6585 [0.6284, 0.6885] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 187 | 0.5996 [0.5595, 0.6396] | 0.6604 [0.6574, 0.6635] |
| 0.9 | 186 | 0.5781 [0.5302, 0.6259] | 0.2163 [0.2078, 0.2255] |
| **mean (videos with all 5)** | 184 | **0.6362 [0.6045, 0.6662]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 2, '0.3': 1, '0.5': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.6597 [0.6263, 0.6917] · middle ruler 0.9751 [0.9630, 0.9836]
* micro (all videos): 0.6614 · clip oracle 0.7584
* clip-level AUC (max score): 0.5393
* per-run whole macro: {'s2024': 0.6502, 's2025': 0.6581, 's2026': 0.6664, 's2027': 0.6296, 's2028': 0.6501}
* per-run crop endpoint: {'s2024': 0.6261, 's2025': 0.6402, 's2026': 0.6387, 's2027': 0.6133, 's2028': 0.6251}
