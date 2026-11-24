# Pre-onset rise on dada (KIM_RUN.md §43 / §43.1)

370 videos; onset = first `gt = 1` frame; 3.75 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | - | 0.081 [0.058, 0.106] | 0.289 [0.256, 0.324] | 0.486 [0.447, 0.526] | 0.769 [0.425, 1.250] | - |
| K5fepre | - | 0.090 [0.065, 0.117] | 0.290 [0.260, 0.322] | 0.534 [0.496, 0.572] | 0.019 [-1.148, 0.706] | -0.239 [-0.525, -0.032] |
| K5fepost | - | 0.044 [0.029, 0.061] | 0.224 [0.195, 0.255] | 0.412 [0.375, 0.451] | 1.511 [0.400, 3.526] | 0.599 [-0.313, 2.164] |
| K5fbk | - | 0.023 [0.015, 0.032] | 0.096 [0.077, 0.116] | 0.321 [0.290, 0.354] | 1.098 [0.232, 2.720] | 0.421 [-0.230, 1.625] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5f | - | - | - | - | - |
| K5fepre | - | - | - | - | - |
| K5fepost | - | - | - | - | - |
| K5fbk | - | - | - | - | - |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5f | K5fepre | K5fepost | K5fbk |
|---|---|---|---|---|
| -10.0 | 0.076 | 0.088 | 0.026 | 0.028 |
| -9.5 | 0.088 | 0.097 | 0.043 | 0.038 |
| -9.0 | 0.096 | 0.105 | 0.051 | 0.031 |
| -8.5 | 0.128 | 0.128 | 0.078 | 0.036 |
| -8.0 | 0.110 | 0.119 | 0.061 | 0.022 |
| -7.5 | 0.106 | 0.116 | 0.048 | 0.029 |
| -7.0 | 0.110 | 0.126 | 0.058 | 0.029 |
| -6.5 | 0.106 | 0.122 | 0.065 | 0.032 |
| -6.0 | 0.114 | 0.124 | 0.068 | 0.038 |
| -5.5 | 0.127 | 0.137 | 0.077 | 0.049 |
| -5.0 | 0.125 | 0.143 | 0.075 | 0.039 |
| -4.5 | 0.132 | 0.155 | 0.086 | 0.043 |
| -4.0 | 0.132 | 0.150 | 0.085 | 0.043 |
| -3.5 | 0.131 | 0.151 | 0.089 | 0.045 |
| -3.0 | 0.132 | 0.154 | 0.093 | 0.048 |
| -2.5 | 0.173 | 0.191 | 0.119 | 0.068 |
| -2.0 | 0.230 | 0.241 | 0.163 | 0.084 |
| -1.5 | 0.276 | 0.280 | 0.208 | 0.091 |
| -1.0 | 0.330 | 0.330 | 0.267 | 0.111 |
| -0.5 | 0.380 | 0.381 | 0.320 | 0.128 |
| +0.0 | 0.408 | 0.428 | 0.358 | 0.166 |
| +0.5 | 0.453 | 0.488 | 0.401 | 0.237 |
| +1.0 | 0.488 | 0.532 | 0.426 | 0.318 |
| +1.5 | 0.501 | 0.554 | 0.422 | 0.366 |
| +2.0 | 0.495 | 0.551 | 0.392 | 0.374 |
| +2.5 | 0.451 | 0.522 | 0.323 | 0.331 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

370 two-class videos; 152 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5f | rescaled | 109 | 0.12 [0.07, 0.18] | 0.16 [0.08, 0.24] | 0.19 [0.10, 0.29] | 0.36 [0.25, 0.47] | 0.54 [0.42, 0.65] |
| K5f | minmax | 152 | 0.19 [0.16, 0.24] | 0.21 [0.17, 0.25] | 0.23 [0.18, 0.27] | 0.33 [0.28, 0.38] | 0.43 [0.38, 0.49] |
| K5fepre | rescaled | 120 | 0.19 [0.08, 0.33] | 0.22 [0.12, 0.32] | 0.25 [0.12, 0.38] | 0.32 [0.19, 0.46] | 0.44 [0.31, 0.55] |
| K5fepre | minmax | 152 | 0.20 [0.16, 0.24] | 0.22 [0.18, 0.26] | 0.23 [0.19, 0.27] | 0.31 [0.27, 0.36] | 0.40 [0.35, 0.45] |
| K5fepost | rescaled | 108 | 0.13 [0.01, 0.27] | 0.20 [0.10, 0.31] | 0.24 [0.14, 0.35] | 0.43 [0.29, 0.58] | 0.64 [0.47, 0.82] |
| K5fepost | minmax | 152 | 0.13 [0.10, 0.17] | 0.15 [0.11, 0.18] | 0.15 [0.12, 0.19] | 0.26 [0.21, 0.31] | 0.36 [0.31, 0.42] |
| K5fbk | rescaled | 105 | 0.09 [0.03, 0.16] | 0.12 [0.04, 0.21] | 0.14 [0.06, 0.22] | 0.23 [0.12, 0.34] | 0.32 [0.17, 0.51] |
| K5fbk | minmax | 152 | 0.08 [0.06, 0.10] | 0.09 [0.06, 0.12] | 0.11 [0.08, 0.14] | 0.13 [0.10, 0.17] | 0.16 [0.13, 0.21] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5f | 0.7744 [0.7500, 0.7982] | -0.0018 [-0.0094, 0.0056] |
| K5fepre | 0.7658 [0.7413, 0.7894] | -0.0112 [-0.0190, -0.0036] |
| K5fepost | 0.7803 [0.7566, 0.8033] | 0.0083 [0.0011, 0.0153] |
| K5fbk | 0.7413 [0.7183, 0.7643] | -0.0052 [-0.0124, 0.0014] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5f | 0.28 (n=72) | 0.46 (n=86) | 0.45 (n=198) |
| K5fepre | 0.26 (n=72) | 0.45 (n=86) | 0.43 (n=198) |
| K5fepost | 0.22 (n=72) | 0.41 (n=86) | 0.37 (n=198) |
| K5fbk | 0.08 (n=72) | 0.17 (n=86) | 0.18 (n=198) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
