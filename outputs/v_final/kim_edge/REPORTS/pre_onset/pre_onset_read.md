# Pre-onset rise on nexar (KIM_RUN.md §43 / §43.1)

225 videos; onset = first `gt = 1` frame (= `time_of_alert`); 30 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | 0.046 [0.036, 0.057] | 0.025 [0.016, 0.035] | 0.092 [0.062, 0.125] | 0.511 [0.452, 0.570] | 0.179 [0.050, 0.397] | - |
| K5fepre | 0.033 [0.026, 0.041] | 0.018 [0.012, 0.025] | 0.062 [0.039, 0.089] | 0.466 [0.410, 0.521] | 0.224 [0.007, 0.616] | 0.061 [-0.049, 0.257] |
| K5fepost | 0.052 [0.043, 0.063] | 0.030 [0.019, 0.043] | 0.117 [0.083, 0.154] | 0.522 [0.468, 0.575] | 0.144 [0.031, 0.275] | -0.031 [-0.145, 0.047] |
| K5fbk | 0.019 [0.015, 0.025] | 0.006 [0.004, 0.009] | 0.028 [0.014, 0.045] | 0.295 [0.251, 0.339] | 0.036 [-0.056, 0.122] | -0.027 [-0.076, 0.048] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5f | 0.437 | 0.887 | 0.525 | 0.590 | 0.694 |
| K5fepre | 0.447 | 0.886 | 0.535 | 0.588 | 0.682 |
| K5fepost | 0.429 | 0.882 | 0.525 | 0.600 | 0.712 |
| K5fbk | 0.439 | 0.813 | 0.527 | 0.561 | 0.622 |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5f | K5fepre | K5fepost | K5fbk |
|---|---|---|---|---|
| -10.0 | 0.022 | 0.017 | 0.026 | 0.005 |
| -9.5 | 0.023 | 0.016 | 0.035 | 0.009 |
| -9.0 | 0.021 | 0.019 | 0.034 | 0.011 |
| -8.5 | 0.031 | 0.025 | 0.037 | 0.008 |
| -8.0 | 0.036 | 0.035 | 0.046 | 0.008 |
| -7.5 | 0.034 | 0.029 | 0.039 | 0.008 |
| -7.0 | 0.030 | 0.023 | 0.036 | 0.002 |
| -6.5 | 0.034 | 0.024 | 0.037 | 0.009 |
| -6.0 | 0.029 | 0.018 | 0.032 | 0.009 |
| -5.5 | 0.027 | 0.015 | 0.036 | 0.007 |
| -5.0 | 0.026 | 0.015 | 0.029 | 0.009 |
| -4.5 | 0.029 | 0.016 | 0.028 | 0.012 |
| -4.0 | 0.030 | 0.021 | 0.026 | 0.009 |
| -3.5 | 0.038 | 0.024 | 0.036 | 0.013 |
| -3.0 | 0.038 | 0.025 | 0.044 | 0.011 |
| -2.5 | 0.038 | 0.025 | 0.055 | 0.010 |
| -2.0 | 0.053 | 0.035 | 0.076 | 0.018 |
| -1.5 | 0.064 | 0.046 | 0.081 | 0.016 |
| -1.0 | 0.092 | 0.059 | 0.119 | 0.030 |
| -0.5 | 0.159 | 0.107 | 0.193 | 0.047 |
| +0.0 | 0.284 | 0.199 | 0.307 | 0.088 |
| +0.5 | 0.406 | 0.320 | 0.430 | 0.176 |
| +1.0 | 0.500 | 0.436 | 0.540 | 0.260 |
| +1.5 | 0.561 | 0.530 | 0.590 | 0.356 |
| +2.0 | 0.537 | 0.541 | 0.518 | 0.353 |
| +2.5 | 0.428 | 0.431 | 0.394 | 0.261 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

112 two-class videos; 112 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5f | rescaled | 99 | 0.00 [-0.02, 0.02] | 0.01 [-0.02, 0.04] | 0.02 [-0.01, 0.05] | 0.04 [0.00, 0.09] | 0.15 [0.10, 0.22] |
| K5f | minmax | 112 | 0.03 [0.02, 0.05] | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] | 0.06 [0.04, 0.09] | 0.13 [0.09, 0.18] |
| K5fepre | rescaled | 97 | -0.00 [-0.02, 0.02] | 0.00 [-0.02, 0.02] | 0.00 [-0.04, 0.04] | 0.02 [-0.04, 0.07] | 0.08 [0.02, 0.14] |
| K5fepre | minmax | 112 | 0.02 [0.01, 0.04] | 0.03 [0.02, 0.04] | 0.03 [0.02, 0.04] | 0.05 [0.02, 0.07] | 0.09 [0.06, 0.12] |
| K5fepost | rescaled | 103 | -0.01 [-0.03, 0.02] | -0.02 [-0.07, 0.02] | 0.00 [-0.05, 0.06] | 0.04 [-0.03, 0.10] | 0.19 [0.10, 0.29] |
| K5fepost | minmax | 112 | 0.03 [0.02, 0.05] | 0.03 [0.02, 0.05] | 0.05 [0.03, 0.08] | 0.08 [0.05, 0.12] | 0.16 [0.12, 0.21] |
| K5fbk | rescaled | 86 | -0.00 [-0.03, 0.02] | 0.02 [-0.02, 0.08] | 0.01 [-0.02, 0.05] | 0.04 [-0.02, 0.10] | 0.07 [0.02, 0.12] |
| K5fbk | minmax | 112 | 0.02 [0.01, 0.03] | 0.01 [0.01, 0.03] | 0.01 [0.00, 0.02] | 0.02 [0.01, 0.04] | 0.04 [0.02, 0.07] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5f | 0.9210 [0.8998, 0.9406] | 0.0016 [-0.0006, 0.0039] |
| K5fepre | 0.9095 [0.8864, 0.9309] | 0.0012 [-0.0009, 0.0034] |
| K5fepost | 0.9220 [0.9037, 0.9386] | 0.0036 [0.0016, 0.0058] |
| K5fbk | 0.8443 [0.8186, 0.8695] | 0.0029 [0.0010, 0.0047] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5f | - | - | 0.13 (n=112) |
| K5fepre | - | - | 0.09 (n=112) |
| K5fepost | - | - | 0.16 (n=112) |
| K5fbk | - | - | 0.04 (n=112) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
