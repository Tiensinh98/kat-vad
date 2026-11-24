# Pre-onset rise on dota (KIM_RUN.md §43 / §43.1)

569 videos; onset = first `gt = 1` frame; 10 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5frr.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5frr |
|---|---|---|---|---|---|---|
| K5frr | - | 0.053 [0.031, 0.079] | 0.412 [0.382, 0.441] | 0.565 [0.533, 0.596] | 1.028 [0.761, 1.361] | - |
| K5fbk25 | - | 0.023 [0.012, 0.036] | 0.234 [0.211, 0.256] | 0.510 [0.479, 0.540] | 0.766 [0.511, 1.107] | -0.262 [-0.350, -0.180] |
| K5f | - | 0.053 [0.031, 0.079] | 0.412 [0.382, 0.441] | 0.565 [0.533, 0.596] | 1.028 [0.761, 1.361] | 0.000 [0.000, 0.000] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5frr | - | - | - | - | - |
| K5fbk25 | - | - | - | - | - |
| K5f | - | - | - | - | - |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5frr | K5fbk25 | K5f |
|---|---|---|---|
| -10.0 | 0.073 | 0.057 | 0.073 |
| -9.5 | 0.104 | 0.090 | 0.104 |
| -9.0 | 0.095 | 0.077 | 0.095 |
| -8.5 | 0.074 | 0.053 | 0.074 |
| -8.0 | 0.058 | 0.035 | 0.058 |
| -7.5 | 0.064 | 0.020 | 0.064 |
| -7.0 | 0.066 | 0.017 | 0.066 |
| -6.5 | 0.073 | 0.021 | 0.073 |
| -6.0 | 0.083 | 0.031 | 0.083 |
| -5.5 | 0.089 | 0.044 | 0.089 |
| -5.0 | 0.098 | 0.059 | 0.098 |
| -4.5 | 0.096 | 0.057 | 0.096 |
| -4.0 | 0.108 | 0.060 | 0.108 |
| -3.5 | 0.137 | 0.063 | 0.137 |
| -3.0 | 0.200 | 0.082 | 0.200 |
| -2.5 | 0.267 | 0.112 | 0.267 |
| -2.0 | 0.337 | 0.148 | 0.337 |
| -1.5 | 0.394 | 0.199 | 0.394 |
| -1.0 | 0.440 | 0.258 | 0.440 |
| -0.5 | 0.491 | 0.339 | 0.491 |
| +0.0 | 0.544 | 0.434 | 0.544 |
| +0.5 | 0.579 | 0.507 | 0.579 |
| +1.0 | 0.592 | 0.545 | 0.592 |
| +1.5 | 0.586 | 0.554 | 0.586 |
| +2.0 | 0.556 | 0.523 | 0.556 |
| +2.5 | 0.497 | 0.453 | 0.497 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

569 two-class videos; 76 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5frr | rescaled | 59 | 0.27 [0.09, 0.56] | 0.45 [0.28, 0.66] | 0.60 [0.43, 0.82] | 0.74 [0.41, 1.04] | 0.85 [0.50, 1.17] |
| K5frr | minmax | 76 | 0.21 [0.16, 0.28] | 0.34 [0.27, 0.41] | 0.46 [0.39, 0.53] | 0.55 [0.48, 0.63] | 0.64 [0.56, 0.71] |
| K5fbk25 | rescaled | 59 | 0.11 [0.05, 0.19] | 0.31 [0.13, 0.59] | 0.39 [0.23, 0.61] | 0.48 [0.35, 0.64] | 0.74 [0.56, 0.95] |
| K5fbk25 | minmax | 76 | 0.15 [0.10, 0.20] | 0.21 [0.15, 0.27] | 0.25 [0.19, 0.31] | 0.32 [0.26, 0.38] | 0.48 [0.40, 0.55] |
| K5f | rescaled | 59 | 0.27 [0.09, 0.56] | 0.45 [0.28, 0.66] | 0.60 [0.43, 0.82] | 0.74 [0.41, 1.04] | 0.85 [0.50, 1.17] |
| K5f | minmax | 76 | 0.21 [0.16, 0.28] | 0.34 [0.27, 0.41] | 0.46 [0.39, 0.53] | 0.55 [0.48, 0.63] | 0.64 [0.56, 0.71] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5frr | 0.8188 [0.8032, 0.8338] | 0.0192 [0.0108, 0.0274] |
| K5fbk25 | 0.8268 [0.8113, 0.8414] | 0.0021 [-0.0056, 0.0096] |
| K5f | 0.8188 [0.8032, 0.8338] | 0.0192 [0.0108, 0.0274] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5frr | 0.33 (n=68) | 0.57 (n=298) | 0.60 (n=202) |
| K5fbk25 | 0.17 (n=68) | 0.36 (n=298) | 0.42 (n=202) |
| K5f | 0.33 (n=68) | 0.57 (n=298) | 0.60 (n=202) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
