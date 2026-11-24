# Pre-onset rise on dota (KIM_RUN.md §43 / §43.1)

569 videos; onset = first `gt = 1` frame; 10 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5dfirr.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5dfirr |
|---|---|---|---|---|---|---|
| K5dfirr | - | 0.026 [0.015, 0.040] | 0.286 [0.263, 0.311] | 0.549 [0.519, 0.579] | 0.772 [0.603, 0.975] | - |
| K5dfibk25 | - | 0.024 [0.014, 0.035] | 0.247 [0.224, 0.269] | 0.538 [0.509, 0.568] | 0.758 [0.568, 0.987] | -0.014 [-0.065, 0.047] |
| K5dfi | - | 0.026 [0.015, 0.040] | 0.286 [0.263, 0.311] | 0.549 [0.519, 0.579] | 0.772 [0.603, 0.975] | 0.000 [0.000, 0.000] |
| K5f | - | 0.053 [0.031, 0.079] | 0.412 [0.382, 0.441] | 0.565 [0.533, 0.596] | 1.028 [0.761, 1.361] | 0.252 [0.129, 0.403] |

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
| -10.0 | 0.059 | 0.058 | 0.059 | 0.073 |
| -9.5 | 0.067 | 0.047 | 0.067 | 0.104 |
| -9.0 | 0.085 | 0.060 | 0.085 | 0.095 |
| -8.5 | 0.083 | 0.046 | 0.083 | 0.074 |
| -8.0 | 0.051 | 0.043 | 0.051 | 0.058 |
| -7.5 | 0.059 | 0.058 | 0.059 | 0.064 |
| -7.0 | 0.058 | 0.050 | 0.058 | 0.066 |
| -6.5 | 0.036 | 0.030 | 0.036 | 0.073 |
| -6.0 | 0.037 | 0.030 | 0.037 | 0.083 |
| -5.5 | 0.042 | 0.037 | 0.042 | 0.089 |
| -5.0 | 0.063 | 0.061 | 0.063 | 0.098 |
| -4.5 | 0.068 | 0.066 | 0.068 | 0.096 |
| -4.0 | 0.076 | 0.071 | 0.076 | 0.108 |
| -3.5 | 0.088 | 0.078 | 0.088 | 0.137 |
| -3.0 | 0.122 | 0.107 | 0.122 | 0.200 |
| -2.5 | 0.172 | 0.145 | 0.172 | 0.267 |
| -2.0 | 0.218 | 0.178 | 0.218 | 0.337 |
| -1.5 | 0.258 | 0.214 | 0.258 | 0.394 |
| -1.0 | 0.308 | 0.263 | 0.308 | 0.440 |
| -0.5 | 0.379 | 0.345 | 0.379 | 0.491 |
| +0.0 | 0.472 | 0.452 | 0.472 | 0.544 |
| +0.5 | 0.546 | 0.534 | 0.546 | 0.579 |
| +1.0 | 0.586 | 0.579 | 0.586 | 0.592 |
| +1.5 | 0.601 | 0.596 | 0.601 | 0.586 |
| +2.0 | 0.575 | 0.565 | 0.575 | 0.556 |
| +2.5 | 0.507 | 0.491 | 0.507 | 0.497 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

569 two-class videos; 76 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5dfirr | rescaled | 63 | 0.13 [0.05, 0.23] | 0.40 [0.21, 0.63] | 0.51 [0.35, 0.70] | 0.67 [0.53, 0.82] | 0.85 [0.65, 1.12] |
| K5dfirr | minmax | 76 | 0.15 [0.10, 0.20] | 0.25 [0.19, 0.32] | 0.33 [0.27, 0.40] | 0.40 [0.34, 0.47] | 0.50 [0.42, 0.58] |
| K5dfibk25 | rescaled | 64 | 0.21 [0.05, 0.48] | 0.51 [0.21, 0.94] | 0.57 [0.33, 0.90] | 0.70 [0.50, 0.96] | 0.93 [0.66, 1.28] |
| K5dfibk25 | minmax | 76 | 0.14 [0.09, 0.19] | 0.23 [0.17, 0.29] | 0.29 [0.22, 0.35] | 0.36 [0.29, 0.43] | 0.47 [0.39, 0.54] |
| K5dfi | rescaled | 63 | 0.13 [0.05, 0.23] | 0.40 [0.21, 0.63] | 0.51 [0.35, 0.70] | 0.67 [0.53, 0.82] | 0.85 [0.65, 1.12] |
| K5dfi | minmax | 76 | 0.15 [0.10, 0.20] | 0.25 [0.19, 0.32] | 0.33 [0.27, 0.40] | 0.40 [0.34, 0.47] | 0.50 [0.42, 0.58] |
| K5f | rescaled | 59 | 0.27 [0.09, 0.56] | 0.45 [0.28, 0.66] | 0.60 [0.43, 0.82] | 0.74 [0.41, 1.04] | 0.85 [0.50, 1.17] |
| K5f | minmax | 76 | 0.21 [0.16, 0.28] | 0.34 [0.27, 0.41] | 0.46 [0.39, 0.53] | 0.55 [0.48, 0.63] | 0.64 [0.56, 0.71] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5dfirr | 0.8259 [0.8112, 0.8400] | 0.0020 [-0.0054, 0.0093] |
| K5dfibk25 | 0.8252 [0.8102, 0.8396] | 0.0017 [-0.0055, 0.0087] |
| K5dfi | 0.8259 [0.8112, 0.8400] | 0.0020 [-0.0054, 0.0093] |
| K5f | 0.8188 [0.8032, 0.8338] | 0.0192 [0.0108, 0.0274] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5dfirr | 0.15 (n=68) | 0.40 (n=298) | 0.47 (n=202) |
| K5dfibk25 | 0.13 (n=68) | 0.34 (n=298) | 0.43 (n=202) |
| K5dfi | 0.15 (n=68) | 0.40 (n=298) | 0.47 (n=202) |
| K5f | 0.33 (n=68) | 0.57 (n=298) | 0.60 (n=202) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
