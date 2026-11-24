# Pre-onset rise on dada (KIM_RUN.md §43 / §43.1)

370 videos; onset = first `gt = 1` frame; 3.75 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5frr.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5frr |
|---|---|---|---|---|---|---|
| K5frr | - | 0.081 [0.058, 0.106] | 0.289 [0.256, 0.324] | 0.486 [0.447, 0.526] | 0.769 [0.425, 1.250] | - |
| K5fbk25 | - | 0.056 [0.038, 0.077] | 0.171 [0.146, 0.198] | 0.424 [0.387, 0.461] | 0.696 [0.231, 1.527] | 0.028 [-0.205, 0.453] |
| K5f | - | 0.081 [0.058, 0.106] | 0.289 [0.256, 0.324] | 0.486 [0.447, 0.526] | 0.769 [0.425, 1.250] | 0.000 [0.000, 0.000] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5frr | - | - | - | - | - |
| K5fbk25 | - | - | - | - | - |
| K5f | - | - | - | - | - |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5frr | K5fbk25 | K5f |
|---|---|---|---|
| -10.0 | 0.076 | 0.053 | 0.076 |
| -9.5 | 0.088 | 0.060 | 0.088 |
| -9.0 | 0.096 | 0.068 | 0.096 |
| -8.5 | 0.128 | 0.098 | 0.128 |
| -8.0 | 0.110 | 0.079 | 0.110 |
| -7.5 | 0.106 | 0.074 | 0.106 |
| -7.0 | 0.110 | 0.077 | 0.110 |
| -6.5 | 0.106 | 0.076 | 0.106 |
| -6.0 | 0.114 | 0.084 | 0.114 |
| -5.5 | 0.127 | 0.095 | 0.127 |
| -5.0 | 0.125 | 0.088 | 0.125 |
| -4.5 | 0.132 | 0.098 | 0.132 |
| -4.0 | 0.132 | 0.094 | 0.132 |
| -3.5 | 0.131 | 0.095 | 0.131 |
| -3.0 | 0.132 | 0.091 | 0.132 |
| -2.5 | 0.173 | 0.105 | 0.173 |
| -2.0 | 0.230 | 0.125 | 0.230 |
| -1.5 | 0.276 | 0.156 | 0.276 |
| -1.0 | 0.330 | 0.202 | 0.330 |
| -0.5 | 0.380 | 0.252 | 0.380 |
| +0.0 | 0.408 | 0.299 | 0.408 |
| +0.5 | 0.453 | 0.359 | 0.453 |
| +1.0 | 0.488 | 0.424 | 0.488 |
| +1.5 | 0.501 | 0.455 | 0.501 |
| +2.0 | 0.495 | 0.454 | 0.495 |
| +2.5 | 0.451 | 0.409 | 0.451 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

370 two-class videos; 152 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5frr | rescaled | 109 | 0.12 [0.07, 0.18] | 0.16 [0.08, 0.24] | 0.19 [0.10, 0.29] | 0.36 [0.25, 0.47] | 0.54 [0.42, 0.65] |
| K5frr | minmax | 152 | 0.19 [0.16, 0.24] | 0.21 [0.17, 0.25] | 0.23 [0.18, 0.27] | 0.33 [0.28, 0.38] | 0.43 [0.38, 0.49] |
| K5fbk25 | rescaled | 108 | 0.10 [0.04, 0.16] | 0.12 [0.04, 0.20] | 0.09 [0.01, 0.17] | 0.15 [0.05, 0.25] | 0.31 [0.20, 0.42] |
| K5fbk25 | minmax | 152 | 0.15 [0.12, 0.18] | 0.16 [0.12, 0.19] | 0.15 [0.12, 0.19] | 0.19 [0.15, 0.23] | 0.27 [0.22, 0.32] |
| K5f | rescaled | 109 | 0.12 [0.07, 0.18] | 0.16 [0.08, 0.24] | 0.19 [0.10, 0.29] | 0.36 [0.25, 0.47] | 0.54 [0.42, 0.65] |
| K5f | minmax | 152 | 0.19 [0.16, 0.24] | 0.21 [0.17, 0.25] | 0.23 [0.18, 0.27] | 0.33 [0.28, 0.38] | 0.43 [0.38, 0.49] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5frr | 0.7744 [0.7500, 0.7982] | -0.0018 [-0.0094, 0.0056] |
| K5fbk25 | 0.7624 [0.7385, 0.7858] | -0.0069 [-0.0143, 0.0003] |
| K5f | 0.7744 [0.7500, 0.7982] | -0.0018 [-0.0094, 0.0056] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5frr | 0.28 (n=72) | 0.46 (n=86) | 0.45 (n=198) |
| K5fbk25 | 0.17 (n=72) | 0.31 (n=86) | 0.29 (n=198) |
| K5f | 0.28 (n=72) | 0.46 (n=86) | 0.45 (n=198) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
