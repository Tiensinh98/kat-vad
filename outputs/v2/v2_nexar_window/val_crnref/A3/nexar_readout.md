# Nexar read-out — nexar_val (225 videos, 112 positive)

Arm `{'crn': 'R2', 'motion': 'vit_s_k710_dl_from_giant'}`, data `/content/drive/MyDrive/Thesis-V2/data/Nexar_window`, runs ['s2024', 's2025', 's2026', 's2027', 's2028'] (steps {'s2024': 1749, 's2025': 1749, 's2026': 1749, 's2027': 1749, 's2028': 1749}). N-B: s1[::3] per item, CRN over the source video (§30), native frames, seed-averaged.

## Crops (endpoint, D-N2)

| event centre at | items | macro | middle ruler |
|---|---:|---|---|
| 0.1 | 111 | 0.8738 [0.8363, 0.9073] | 0.2020 [0.1928, 0.2117] |
| 0.3 | 112 | 0.8134 [0.7766, 0.8483] | 0.6562 [0.6531, 0.6594] |
| 0.5 | 112 | 0.8121 [0.7772, 0.8455] | 1.0000 [1.0000, 1.0000] |
| 0.7 | 112 | 0.8236 [0.7884, 0.8567] | 0.6580 [0.6550, 0.6611] |
| 0.9 | 111 | 0.8573 [0.8134, 0.8980] | 0.2106 [0.2015, 0.2205] |
| **mean (videos with all 5)** | 110 | **0.8342 [0.8022, 0.8647]** | |

Crops skipped (did not fit): {'0.9': 1, '0.1': 1}

## Whole videos (printed — a position test on Nexar)

* macro (positives): 0.8894 [0.8668, 0.9109] · middle ruler 0.9735 [0.9608, 0.9838]
* micro (all videos): 0.8589 · clip oracle 0.7611
* clip-level AUC (max score): 0.6826
* per-run whole macro: {'s2024': 0.8931, 's2025': 0.8905, 's2026': 0.8637, 's2027': 0.8773, 's2028': 0.8844}
* per-run crop endpoint: {'s2024': 0.8409, 's2025': 0.8301, 's2026': 0.8091, 's2027': 0.8209, 's2028': 0.8238}
