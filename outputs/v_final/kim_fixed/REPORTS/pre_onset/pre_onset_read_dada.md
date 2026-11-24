# Pre-onset rise on dada (KIM_RUN.md §43 / §43.1)

370 videos; onset = first `gt = 1` frame; 3.75 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | - | 0.081 [0.058, 0.106] | 0.289 [0.256, 0.324] | 0.486 [0.447, 0.526] | 0.769 [0.425, 1.250] | - |
| K5dfi | - | 0.051 [0.036, 0.068] | 0.204 [0.177, 0.233] | 0.442 [0.405, 0.478] | 0.500 [0.310, 0.715] | -0.147 [-0.449, 0.087] |
| L0dfi | - | 0.254 [0.220, 0.290] | 0.277 [0.254, 0.302] | 0.367 [0.343, 0.392] | -0.862 [-4.515, 1.352] | -1.814 [-5.468, 0.240] |
| LaGoVAD | - | 0.106 [0.082, 0.132] | 0.147 [0.128, 0.168] | 0.165 [0.145, 0.186] | 0.291 [0.028, 0.514] | -0.584 [-1.304, -0.059] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5f | - | - | - | - | - |
| K5dfi | - | - | - | - | - |
| L0dfi | - | - | - | - | - |
| LaGoVAD | - | - | - | - | - |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5f | K5dfi | L0dfi | LaGoVAD |
|---|---|---|---|---|
| -10.0 | 0.076 | 0.036 | 0.229 | 0.136 |
| -9.5 | 0.088 | 0.056 | 0.205 | 0.154 |
| -9.0 | 0.096 | 0.060 | 0.257 | 0.143 |
| -8.5 | 0.128 | 0.083 | 0.262 | 0.142 |
| -8.0 | 0.110 | 0.070 | 0.244 | 0.120 |
| -7.5 | 0.106 | 0.063 | 0.266 | 0.110 |
| -7.0 | 0.110 | 0.062 | 0.285 | 0.107 |
| -6.5 | 0.106 | 0.065 | 0.236 | 0.108 |
| -6.0 | 0.114 | 0.077 | 0.239 | 0.110 |
| -5.5 | 0.127 | 0.089 | 0.255 | 0.109 |
| -5.0 | 0.125 | 0.094 | 0.271 | 0.114 |
| -4.5 | 0.132 | 0.104 | 0.263 | 0.123 |
| -4.0 | 0.132 | 0.103 | 0.240 | 0.117 |
| -3.5 | 0.131 | 0.105 | 0.263 | 0.116 |
| -3.0 | 0.132 | 0.107 | 0.271 | 0.123 |
| -2.5 | 0.173 | 0.142 | 0.279 | 0.129 |
| -2.0 | 0.230 | 0.173 | 0.276 | 0.132 |
| -1.5 | 0.276 | 0.193 | 0.273 | 0.140 |
| -1.0 | 0.330 | 0.231 | 0.275 | 0.150 |
| -0.5 | 0.380 | 0.275 | 0.292 | 0.154 |
| +0.0 | 0.408 | 0.318 | 0.306 | 0.159 |
| +0.5 | 0.453 | 0.379 | 0.334 | 0.163 |
| +1.0 | 0.488 | 0.441 | 0.377 | 0.165 |
| +1.5 | 0.501 | 0.475 | 0.398 | 0.165 |
| +2.0 | 0.495 | 0.479 | 0.363 | 0.166 |
| +2.5 | 0.451 | 0.441 | 0.365 | 0.164 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

370 two-class videos; 152 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5f | rescaled | 109 | 0.12 [0.07, 0.18] | 0.16 [0.08, 0.24] | 0.19 [0.10, 0.29] | 0.36 [0.25, 0.47] | 0.54 [0.42, 0.65] |
| K5f | minmax | 152 | 0.19 [0.16, 0.24] | 0.21 [0.17, 0.25] | 0.23 [0.18, 0.27] | 0.33 [0.28, 0.38] | 0.43 [0.38, 0.49] |
| K5dfi | rescaled | 113 | 0.21 [0.10, 0.35] | 0.19 [0.05, 0.34] | 0.19 [0.04, 0.34] | 0.33 [0.11, 0.61] | 0.44 [0.24, 0.67] |
| K5dfi | minmax | 152 | 0.15 [0.11, 0.18] | 0.17 [0.13, 0.21] | 0.18 [0.15, 0.22] | 0.24 [0.20, 0.29] | 0.30 [0.25, 0.35] |
| L0dfi | rescaled | 81 | 0.10 [-0.07, 0.28] | -0.02 [-0.25, 0.19] | 0.24 [-0.01, 0.49] | 0.20 [-0.06, 0.45] | 0.34 [0.06, 0.65] |
| L0dfi | minmax | 152 | 0.30 [0.26, 0.34] | 0.28 [0.24, 0.32] | 0.30 [0.27, 0.34] | 0.29 [0.25, 0.33] | 0.28 [0.24, 0.32] |
| LaGoVAD | rescaled | 47 | 0.08 [0.01, 0.16] | 0.17 [0.07, 0.27] | 0.28 [0.16, 0.42] | 0.45 [0.28, 0.64] | 0.62 [0.42, 0.84] |
| LaGoVAD | minmax | 152 | 0.33 [0.29, 0.38] | 0.34 [0.30, 0.38] | 0.35 [0.31, 0.39] | 0.37 [0.33, 0.41] | 0.41 [0.37, 0.46] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5f | 0.7744 [0.7500, 0.7982] | -0.0018 [-0.0094, 0.0056] |
| K5dfi | 0.7677 [0.7443, 0.7906] | -0.0072 [-0.0154, 0.0002] |
| L0dfi | 0.5752 [0.5539, 0.5961] | -0.0041 [-0.0109, 0.0028] |
| LaGoVAD | 0.6202 [0.5922, 0.6482] | 0.0072 [-0.0053, 0.0198] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5f | 0.28 (n=72) | 0.46 (n=86) | 0.45 (n=198) |
| K5dfi | 0.14 (n=72) | 0.34 (n=86) | 0.32 (n=198) |
| L0dfi | 0.34 (n=72) | 0.33 (n=86) | 0.29 (n=198) |
| LaGoVAD | 0.48 (n=72) | 0.42 (n=86) | 0.42 (n=198) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
