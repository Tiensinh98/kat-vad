# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/nexar/stage/data_whole`, runs ['s2099'] (steps {'s2099': 1740}). N-B: s1[::3] per item, CRN per item, native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.6915 [0.6295, 0.7514] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.6844 [0.6327, 0.7350] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.5802 [0.5318, 0.6294] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.4832 [0.4266, 0.5410] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.4654 [0.4000, 0.5308] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.5819 [0.5347, 0.6279]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.5811 [0.5318, 0.6299] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.6048 · clip oracle 0.7611
* clip-level AUC (max score): 0.6954
* per-run whole macro: {'s2099': 0.5811}
