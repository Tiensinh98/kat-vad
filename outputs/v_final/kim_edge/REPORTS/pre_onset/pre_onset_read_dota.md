# Pre-onset rise on dota (KIM_RUN.md §43 / §43.1)

569 videos; onset = first `gt = 1` frame; 10 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | - | 0.053 [0.031, 0.079] | 0.412 [0.382, 0.441] | 0.565 [0.533, 0.596] | 1.028 [0.761, 1.361] | - |
| K5fepre | - | 0.049 [0.027, 0.074] | 0.364 [0.337, 0.390] | 0.608 [0.578, 0.638] | 0.798 [0.625, 1.015] | -0.210 [-0.369, -0.090] |
| K5fepost | - | 0.033 [0.017, 0.053] | 0.338 [0.311, 0.365] | 0.506 [0.475, 0.536] | 1.636 [0.796, 2.799] | 0.610 [-0.013, 1.494] |
| K5fbk | - | 0.005 [0.003, 0.007] | 0.111 [0.095, 0.127] | 0.446 [0.417, 0.474] | 0.542 [0.369, 0.759] | -0.491 [-0.732, -0.310] |

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
| -10.0 | 0.073 | 0.082 | 0.019 | 0.032 |
| -9.5 | 0.104 | 0.112 | 0.015 | 0.040 |
| -9.0 | 0.095 | 0.092 | 0.025 | 0.033 |
| -8.5 | 0.074 | 0.066 | 0.028 | 0.020 |
| -8.0 | 0.058 | 0.065 | 0.036 | 0.008 |
| -7.5 | 0.064 | 0.083 | 0.062 | 0.005 |
| -7.0 | 0.066 | 0.070 | 0.051 | 0.004 |
| -6.5 | 0.073 | 0.056 | 0.047 | 0.003 |
| -6.0 | 0.083 | 0.064 | 0.057 | 0.004 |
| -5.5 | 0.089 | 0.078 | 0.070 | 0.009 |
| -5.0 | 0.098 | 0.095 | 0.076 | 0.023 |
| -4.5 | 0.096 | 0.091 | 0.069 | 0.030 |
| -4.0 | 0.108 | 0.099 | 0.071 | 0.035 |
| -3.5 | 0.137 | 0.114 | 0.089 | 0.032 |
| -3.0 | 0.200 | 0.166 | 0.131 | 0.035 |
| -2.5 | 0.267 | 0.223 | 0.186 | 0.051 |
| -2.0 | 0.337 | 0.280 | 0.249 | 0.062 |
| -1.5 | 0.394 | 0.335 | 0.309 | 0.084 |
| -1.0 | 0.440 | 0.390 | 0.369 | 0.118 |
| -0.5 | 0.491 | 0.463 | 0.438 | 0.185 |
| +0.0 | 0.544 | 0.543 | 0.497 | 0.304 |
| +0.5 | 0.579 | 0.603 | 0.538 | 0.422 |
| +1.0 | 0.592 | 0.637 | 0.549 | 0.496 |
| +1.5 | 0.586 | 0.645 | 0.520 | 0.531 |
| +2.0 | 0.556 | 0.624 | 0.462 | 0.503 |
| +2.5 | 0.497 | 0.571 | 0.372 | 0.411 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

569 two-class videos; 76 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5f | rescaled | 59 | 0.27 [0.09, 0.56] | 0.45 [0.28, 0.66] | 0.60 [0.43, 0.82] | 0.74 [0.41, 1.04] | 0.85 [0.50, 1.17] |
| K5f | minmax | 76 | 0.21 [0.16, 0.28] | 0.34 [0.27, 0.41] | 0.46 [0.39, 0.53] | 0.55 [0.48, 0.63] | 0.64 [0.56, 0.71] |
| K5fepre | rescaled | 61 | 0.18 [0.09, 0.29] | 0.42 [0.25, 0.63] | 0.56 [0.39, 0.76] | 0.65 [0.47, 0.84] | 0.75 [0.49, 0.98] |
| K5fepre | minmax | 76 | 0.20 [0.14, 0.26] | 0.29 [0.23, 0.37] | 0.38 [0.31, 0.46] | 0.47 [0.39, 0.54] | 0.57 [0.49, 0.65] |
| K5fepost | rescaled | 56 | 0.11 [0.05, 0.18] | 0.23 [0.16, 0.31] | 0.43 [0.32, 0.53] | 0.64 [0.51, 0.77] | 0.86 [0.74, 0.98] |
| K5fepost | minmax | 76 | 0.17 [0.11, 0.23] | 0.24 [0.18, 0.31] | 0.34 [0.27, 0.40] | 0.45 [0.38, 0.52] | 0.56 [0.49, 0.64] |
| K5fbk | rescaled | 61 | 0.09 [0.03, 0.18] | 0.24 [0.09, 0.42] | 0.25 [0.09, 0.41] | 0.35 [0.19, 0.51] | 0.71 [0.40, 1.14] |
| K5fbk | minmax | 76 | 0.08 [0.05, 0.12] | 0.14 [0.09, 0.20] | 0.13 [0.09, 0.18] | 0.18 [0.13, 0.23] | 0.32 [0.25, 0.40] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5f | 0.8188 [0.8032, 0.8338] | 0.0192 [0.0108, 0.0274] |
| K5fepre | 0.8214 [0.8065, 0.8360] | -0.0084 [-0.0161, -0.0009] |
| K5fepost | 0.8126 [0.7971, 0.8273] | 0.0384 [0.0303, 0.0465] |
| K5fbk | 0.8125 [0.7965, 0.8275] | -0.0046 [-0.0115, 0.0022] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5f | 0.33 (n=68) | 0.57 (n=298) | 0.60 (n=202) |
| K5fepre | 0.27 (n=68) | 0.49 (n=298) | 0.53 (n=202) |
| K5fepost | 0.27 (n=68) | 0.50 (n=298) | 0.54 (n=202) |
| K5fbk | 0.05 (n=68) | 0.16 (n=298) | 0.27 (n=202) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
