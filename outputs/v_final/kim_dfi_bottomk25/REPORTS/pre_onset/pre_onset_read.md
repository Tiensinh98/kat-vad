# Pre-onset rise on nexar (KIM_RUN.md §43 / §43.1)

225 videos; onset = first `gt = 1` frame (= `time_of_alert`); 30 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5dfirr.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5dfirr |
|---|---|---|---|---|---|---|
| K5dfirr | 0.053 [0.042, 0.066] | 0.029 [0.021, 0.038] | 0.093 [0.067, 0.124] | 0.495 [0.440, 0.551] | 0.173 [0.051, 0.372] | - |
| K5dfibk25 | 0.051 [0.040, 0.064] | 0.027 [0.020, 0.036] | 0.089 [0.064, 0.118] | 0.474 [0.421, 0.528] | 0.327 [0.059, 0.808] | 0.154 [-0.008, 0.450] |
| K5dfi | 0.053 [0.042, 0.066] | 0.029 [0.021, 0.038] | 0.093 [0.067, 0.124] | 0.495 [0.440, 0.551] | 0.173 [0.051, 0.372] | 0.000 [0.000, 0.000] |
| K5f | 0.046 [0.036, 0.057] | 0.025 [0.016, 0.035] | 0.092 [0.062, 0.125] | 0.511 [0.452, 0.570] | 0.179 [0.050, 0.397] | -0.005 [-0.050, 0.034] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5dfirr | 0.458 | 0.878 | 0.550 | 0.613 | 0.708 |
| K5dfibk25 | 0.461 | 0.871 | 0.551 | 0.612 | 0.704 |
| K5dfi | 0.458 | 0.878 | 0.550 | 0.613 | 0.708 |
| K5f | 0.437 | 0.887 | 0.525 | 0.590 | 0.694 |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5dfirr | K5dfibk25 | K5dfi | K5f |
|---|---|---|---|---|
| -10.0 | 0.029 | 0.028 | 0.029 | 0.022 |
| -9.5 | 0.032 | 0.034 | 0.032 | 0.023 |
| -9.0 | 0.035 | 0.037 | 0.035 | 0.021 |
| -8.5 | 0.035 | 0.034 | 0.035 | 0.031 |
| -8.0 | 0.042 | 0.040 | 0.042 | 0.036 |
| -7.5 | 0.047 | 0.045 | 0.047 | 0.034 |
| -7.0 | 0.034 | 0.031 | 0.034 | 0.030 |
| -6.5 | 0.040 | 0.040 | 0.040 | 0.034 |
| -6.0 | 0.035 | 0.033 | 0.035 | 0.029 |
| -5.5 | 0.032 | 0.028 | 0.032 | 0.027 |
| -5.0 | 0.031 | 0.030 | 0.031 | 0.026 |
| -4.5 | 0.035 | 0.035 | 0.035 | 0.029 |
| -4.0 | 0.033 | 0.032 | 0.033 | 0.030 |
| -3.5 | 0.040 | 0.043 | 0.040 | 0.038 |
| -3.0 | 0.045 | 0.045 | 0.045 | 0.038 |
| -2.5 | 0.041 | 0.043 | 0.041 | 0.038 |
| -2.0 | 0.054 | 0.054 | 0.054 | 0.053 |
| -1.5 | 0.068 | 0.063 | 0.068 | 0.064 |
| -1.0 | 0.100 | 0.096 | 0.100 | 0.092 |
| -0.5 | 0.152 | 0.146 | 0.152 | 0.159 |
| +0.0 | 0.261 | 0.242 | 0.261 | 0.284 |
| +0.5 | 0.375 | 0.346 | 0.375 | 0.406 |
| +1.0 | 0.469 | 0.444 | 0.469 | 0.500 |
| +1.5 | 0.540 | 0.522 | 0.540 | 0.561 |
| +2.0 | 0.549 | 0.533 | 0.549 | 0.537 |
| +2.5 | 0.460 | 0.434 | 0.460 | 0.428 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

112 two-class videos; 112 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5dfirr | rescaled | 98 | -0.00 [-0.03, 0.03] | -0.00 [-0.04, 0.05] | 0.01 [-0.04, 0.06] | 0.03 [-0.03, 0.09] | 0.14 [0.07, 0.21] |
| K5dfirr | minmax | 112 | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] | 0.04 [0.03, 0.07] | 0.07 [0.04, 0.09] | 0.13 [0.10, 0.17] |
| K5dfibk25 | rescaled | 99 | 0.01 [-0.03, 0.06] | 0.00 [-0.03, 0.05] | 0.00 [-0.05, 0.05] | 0.03 [-0.02, 0.08] | 0.14 [0.08, 0.21] |
| K5dfibk25 | minmax | 112 | 0.04 [0.02, 0.05] | 0.04 [0.02, 0.06] | 0.05 [0.03, 0.07] | 0.06 [0.04, 0.09] | 0.13 [0.09, 0.17] |
| K5dfi | rescaled | 98 | -0.00 [-0.03, 0.03] | -0.00 [-0.04, 0.05] | 0.01 [-0.04, 0.06] | 0.03 [-0.03, 0.09] | 0.14 [0.07, 0.21] |
| K5dfi | minmax | 112 | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] | 0.04 [0.03, 0.07] | 0.07 [0.04, 0.09] | 0.13 [0.10, 0.17] |
| K5f | rescaled | 99 | 0.00 [-0.02, 0.02] | 0.01 [-0.02, 0.04] | 0.02 [-0.01, 0.05] | 0.04 [0.00, 0.09] | 0.15 [0.10, 0.22] |
| K5f | minmax | 112 | 0.03 [0.02, 0.05] | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] | 0.06 [0.04, 0.09] | 0.13 [0.09, 0.18] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5dfirr | 0.9043 [0.8825, 0.9250] | 0.0033 [0.0013, 0.0054] |
| K5dfibk25 | 0.8946 [0.8718, 0.9159] | 0.0039 [0.0020, 0.0060] |
| K5dfi | 0.9043 [0.8825, 0.9250] | 0.0033 [0.0013, 0.0054] |
| K5f | 0.9210 [0.8998, 0.9406] | 0.0016 [-0.0006, 0.0039] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5dfirr | - | - | 0.13 (n=112) |
| K5dfibk25 | - | - | 0.13 (n=112) |
| K5dfi | - | - | 0.13 (n=112) |
| K5f | - | - | 0.13 (n=112) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
