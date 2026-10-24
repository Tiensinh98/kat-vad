# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'none', 'motion': 'none'}`, data `/content/nexar/stage/data_whole`, runs ['s2099'] (steps {'s2099': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.5641 [0.4993, 0.6278] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.5383 [0.4834, 0.5917] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.5773 [0.5314, 0.6228] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.6417 [0.5953, 0.6875] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.6378 [0.5802, 0.6929] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.5876 [0.5416, 0.6328]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.5841 [0.5406, 0.6276] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.6153 · clip oracle 0.7611
* clip-level AUC (max score): 0.6515
* per-run whole macro: {'s2099': 0.5841}
