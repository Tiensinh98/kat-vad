# Pre-onset rise on dada (KIM_RUN.md §43 / §43.1)

370 videos; onset = first `gt = 1` frame; 3.75 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5dfirr.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5dfirr |
|---|---|---|---|---|---|---|
| K5dfirr | - | 0.051 [0.036, 0.068] | 0.204 [0.177, 0.233] | 0.442 [0.405, 0.478] | 0.500 [0.310, 0.715] | - |
| K5dfibk25 | - | 0.045 [0.033, 0.060] | 0.181 [0.156, 0.207] | 0.424 [0.389, 0.460] | 0.371 [0.022, 0.684] | -0.129 [-0.361, 0.020] |
| K5dfi | - | 0.051 [0.036, 0.068] | 0.204 [0.177, 0.233] | 0.442 [0.405, 0.478] | 0.500 [0.310, 0.715] | 0.000 [0.000, 0.000] |
| K5f | - | 0.081 [0.058, 0.106] | 0.289 [0.256, 0.324] | 0.486 [0.447, 0.526] | 0.769 [0.425, 1.250] | 0.147 [-0.087, 0.449] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5dfirr | - | - | - | - | - |
| K5dfibk25 | - | - | - | - | - |
| K5dfi | - | - | - | - | - |
| K5f | - | - | - | - | - |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5dfirr | K5dfibk25 | K5dfi | K5f |
|---|---|---|---|---|
| -10.0 | 0.036 | 0.035 | 0.036 | 0.076 |
| -9.5 | 0.056 | 0.053 | 0.056 | 0.088 |
| -9.0 | 0.060 | 0.052 | 0.060 | 0.096 |
| -8.5 | 0.083 | 0.074 | 0.083 | 0.128 |
| -8.0 | 0.070 | 0.058 | 0.070 | 0.110 |
| -7.5 | 0.063 | 0.055 | 0.063 | 0.106 |
| -7.0 | 0.062 | 0.054 | 0.062 | 0.110 |
| -6.5 | 0.065 | 0.056 | 0.065 | 0.106 |
| -6.0 | 0.077 | 0.069 | 0.077 | 0.114 |
| -5.5 | 0.089 | 0.085 | 0.089 | 0.127 |
| -5.0 | 0.094 | 0.086 | 0.094 | 0.125 |
| -4.5 | 0.104 | 0.096 | 0.104 | 0.132 |
| -4.0 | 0.103 | 0.094 | 0.103 | 0.132 |
| -3.5 | 0.105 | 0.096 | 0.105 | 0.131 |
| -3.0 | 0.107 | 0.098 | 0.107 | 0.132 |
| -2.5 | 0.142 | 0.132 | 0.142 | 0.173 |
| -2.0 | 0.173 | 0.154 | 0.173 | 0.230 |
| -1.5 | 0.193 | 0.169 | 0.193 | 0.276 |
| -1.0 | 0.231 | 0.205 | 0.231 | 0.330 |
| -0.5 | 0.275 | 0.244 | 0.275 | 0.380 |
| +0.0 | 0.318 | 0.290 | 0.318 | 0.408 |
| +0.5 | 0.379 | 0.356 | 0.379 | 0.453 |
| +1.0 | 0.441 | 0.427 | 0.441 | 0.488 |
| +1.5 | 0.475 | 0.462 | 0.475 | 0.501 |
| +2.0 | 0.479 | 0.465 | 0.479 | 0.495 |
| +2.5 | 0.441 | 0.421 | 0.441 | 0.451 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

370 two-class videos; 152 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5dfirr | rescaled | 113 | 0.21 [0.10, 0.35] | 0.19 [0.05, 0.34] | 0.19 [0.04, 0.34] | 0.33 [0.11, 0.61] | 0.44 [0.24, 0.67] |
| K5dfirr | minmax | 152 | 0.15 [0.11, 0.18] | 0.17 [0.13, 0.21] | 0.18 [0.15, 0.22] | 0.24 [0.20, 0.29] | 0.30 [0.25, 0.35] |
| K5dfibk25 | rescaled | 112 | 0.20 [0.09, 0.33] | 0.19 [0.08, 0.30] | 0.16 [0.06, 0.26] | 0.23 [0.12, 0.34] | 0.37 [0.25, 0.49] |
| K5dfibk25 | minmax | 152 | 0.13 [0.10, 0.17] | 0.15 [0.12, 0.19] | 0.17 [0.14, 0.21] | 0.22 [0.17, 0.26] | 0.27 [0.23, 0.32] |
| K5dfi | rescaled | 113 | 0.21 [0.10, 0.35] | 0.19 [0.05, 0.34] | 0.19 [0.04, 0.34] | 0.33 [0.11, 0.61] | 0.44 [0.24, 0.67] |
| K5dfi | minmax | 152 | 0.15 [0.11, 0.18] | 0.17 [0.13, 0.21] | 0.18 [0.15, 0.22] | 0.24 [0.20, 0.29] | 0.30 [0.25, 0.35] |
| K5f | rescaled | 109 | 0.12 [0.07, 0.18] | 0.16 [0.08, 0.24] | 0.19 [0.10, 0.29] | 0.36 [0.25, 0.47] | 0.54 [0.42, 0.65] |
| K5f | minmax | 152 | 0.19 [0.16, 0.24] | 0.21 [0.17, 0.25] | 0.23 [0.18, 0.27] | 0.33 [0.28, 0.38] | 0.43 [0.38, 0.49] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5dfirr | 0.7677 [0.7443, 0.7906] | -0.0072 [-0.0154, 0.0002] |
| K5dfibk25 | 0.7632 [0.7399, 0.7861] | -0.0062 [-0.0136, 0.0007] |
| K5dfi | 0.7677 [0.7443, 0.7906] | -0.0072 [-0.0154, 0.0002] |
| K5f | 0.7744 [0.7500, 0.7982] | -0.0018 [-0.0094, 0.0056] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5dfirr | 0.14 (n=72) | 0.34 (n=86) | 0.32 (n=198) |
| K5dfibk25 | 0.14 (n=72) | 0.30 (n=86) | 0.29 (n=198) |
| K5dfi | 0.14 (n=72) | 0.34 (n=86) | 0.32 (n=198) |
| K5f | 0.28 (n=72) | 0.46 (n=86) | 0.45 (n=198) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
