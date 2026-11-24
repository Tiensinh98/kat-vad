# Pre-onset rise on dota (KIM_RUN.md §43 / §43.1)

1129 videos; onset = first `gt = 1` frame; 10 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | - | 0.062 [0.042, 0.084] | 0.419 [0.397, 0.441] | 0.568 [0.545, 0.591] | 0.953 [0.803, 1.131] | - |
| K5dfi | - | 0.036 [0.023, 0.052] | 0.294 [0.277, 0.312] | 0.550 [0.528, 0.572] | 0.888 [0.679, 1.177] | -0.196 [-0.297, -0.102] |
| L0dfi | - | 0.218 [0.183, 0.255] | 0.222 [0.209, 0.235] | 0.387 [0.371, 0.403] | 0.705 [0.256, 1.203] | -0.463 [-0.861, -0.043] |
| LaGoVAD | - | 0.256 [0.217, 0.296] | 0.188 [0.174, 0.202] | 0.227 [0.211, 0.244] | 0.499 [0.236, 0.713] | -0.447 [-0.770, -0.171] |

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
| -10.0 | 0.073 | 0.059 | 0.306 | 0.368 |
| -9.5 | 0.075 | 0.045 | 0.264 | 0.356 |
| -9.0 | 0.066 | 0.043 | 0.244 | 0.222 |
| -8.5 | 0.062 | 0.034 | 0.217 | 0.264 |
| -8.0 | 0.098 | 0.039 | 0.256 | 0.215 |
| -7.5 | 0.110 | 0.050 | 0.238 | 0.227 |
| -7.0 | 0.129 | 0.085 | 0.226 | 0.228 |
| -6.5 | 0.107 | 0.068 | 0.234 | 0.250 |
| -6.0 | 0.097 | 0.062 | 0.231 | 0.271 |
| -5.5 | 0.095 | 0.059 | 0.212 | 0.257 |
| -5.0 | 0.094 | 0.062 | 0.214 | 0.245 |
| -4.5 | 0.092 | 0.059 | 0.218 | 0.222 |
| -4.0 | 0.107 | 0.065 | 0.211 | 0.216 |
| -3.5 | 0.144 | 0.085 | 0.206 | 0.199 |
| -3.0 | 0.213 | 0.130 | 0.209 | 0.191 |
| -2.5 | 0.281 | 0.184 | 0.206 | 0.185 |
| -2.0 | 0.346 | 0.226 | 0.198 | 0.183 |
| -1.5 | 0.402 | 0.266 | 0.206 | 0.185 |
| -1.0 | 0.448 | 0.316 | 0.222 | 0.189 |
| -0.5 | 0.494 | 0.384 | 0.259 | 0.199 |
| +0.0 | 0.543 | 0.474 | 0.325 | 0.209 |
| +0.5 | 0.579 | 0.545 | 0.393 | 0.219 |
| +1.0 | 0.597 | 0.588 | 0.422 | 0.227 |
| +1.5 | 0.591 | 0.603 | 0.412 | 0.231 |
| +2.0 | 0.561 | 0.578 | 0.382 | 0.233 |
| +2.5 | 0.504 | 0.510 | 0.342 | 0.233 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

1129 two-class videos; 177 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5f | rescaled | 143 | 0.19 [0.10, 0.32] | 0.37 [0.27, 0.50] | 0.61 [0.49, 0.77] | 0.81 [0.63, 1.01] | 0.96 [0.77, 1.16] |
| K5f | minmax | 177 | 0.18 [0.15, 0.22] | 0.29 [0.25, 0.33] | 0.44 [0.39, 0.49] | 0.57 [0.52, 0.62] | 0.65 [0.60, 0.70] |
| K5dfi | rescaled | 148 | 0.09 [0.05, 0.14] | 0.24 [0.16, 0.35] | 0.41 [0.32, 0.51] | 0.58 [0.50, 0.66] | 0.76 [0.66, 0.90] |
| K5dfi | minmax | 177 | 0.13 [0.10, 0.16] | 0.20 [0.16, 0.23] | 0.32 [0.28, 0.36] | 0.43 [0.38, 0.47] | 0.52 [0.47, 0.57] |
| L0dfi | rescaled | 128 | -0.01 [-0.10, 0.07] | 0.03 [-0.12, 0.17] | 0.15 [-0.08, 0.39] | 0.23 [-0.05, 0.50] | 0.48 [0.22, 0.74] |
| L0dfi | minmax | 177 | 0.23 [0.19, 0.26] | 0.24 [0.20, 0.27] | 0.27 [0.23, 0.30] | 0.28 [0.24, 0.32] | 0.35 [0.31, 0.39] |
| LaGoVAD | rescaled | 76 | 0.01 [-0.05, 0.06] | 0.11 [0.03, 0.18] | 0.23 [0.14, 0.33] | 0.44 [0.32, 0.57] | 0.71 [0.59, 0.85] |
| LaGoVAD | minmax | 177 | 0.30 [0.26, 0.34] | 0.33 [0.29, 0.37] | 0.36 [0.32, 0.40] | 0.41 [0.37, 0.45] | 0.48 [0.44, 0.52] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5f | 0.8226 [0.8116, 0.8335] | 0.0204 [0.0142, 0.0266] |
| K5dfi | 0.8240 [0.8133, 0.8346] | 0.0046 [-0.0009, 0.0100] |
| L0dfi | 0.6770 [0.6632, 0.6908] | -0.0038 [-0.0090, 0.0017] |
| LaGoVAD | 0.6465 [0.6306, 0.6621] | -0.0111 [-0.0198, -0.0020] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5f | 0.33 (n=137) | 0.57 (n=568) | 0.62 (n=423) |
| K5dfi | 0.16 (n=137) | 0.40 (n=568) | 0.48 (n=423) |
| L0dfi | 0.24 (n=137) | 0.26 (n=568) | 0.32 (n=423) |
| LaGoVAD | 0.41 (n=137) | 0.41 (n=568) | 0.45 (n=423) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
