# Pre-onset rise on nexar_val (KIM_RUN.md §43)

225 videos; onset = first `gt = 1` frame (= `time_of_alert`); 30 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | 0.046 [0.036, 0.057] | 0.025 [0.016, 0.035] | 0.092 [0.062, 0.125] | 0.511 [0.452, 0.570] | 0.179 [0.050, 0.397] | - |
| K5df | 0.111 [0.090, 0.134] | 0.081 [0.061, 0.103] | 0.215 [0.172, 0.260] | 0.650 [0.592, 0.707] | 0.143 [0.017, 0.245] | 0.010 [-0.169, 0.120] |
| K5dfi | 0.053 [0.042, 0.066] | 0.029 [0.021, 0.038] | 0.093 [0.067, 0.124] | 0.495 [0.440, 0.551] | 0.173 [0.051, 0.372] | 0.005 [-0.034, 0.050] |
| K5fno | 0.077 [0.050, 0.108] | 0.200 [0.146, 0.258] | 0.258 [0.192, 0.330] | 0.519 [0.448, 0.590] | 0.237 [-2.125, 2.483] | 0.023 [-2.334, 2.260] |
| LaGoVAD | 0.249 [0.211, 0.288] | 0.258 [0.215, 0.305] | 0.270 [0.224, 0.318] | 0.280 [0.234, 0.329] | 0.857 [0.048, 1.712] | 0.489 [-0.315, 1.374] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5f | 0.437 | 0.887 | 0.525 | 0.590 | 0.694 |
| K5df | 0.485 | 0.895 | 0.577 | 0.641 | 0.741 |
| K5dfi | 0.458 | 0.878 | 0.550 | 0.613 | 0.708 |
| K5fno | 0.639 | 0.839 | 0.659 | 0.663 | 0.717 |
| LaGoVAD | 0.517 | 0.530 | 0.520 | 0.531 | 0.533 |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5f | K5df | K5dfi | K5fno | LaGoVAD |
|---|---|---|---|---|---|
| -10.0 | 0.022 | 0.088 | 0.029 | 0.212 | 0.272 |
| -9.5 | 0.023 | 0.080 | 0.032 | 0.214 | 0.273 |
| -9.0 | 0.021 | 0.086 | 0.035 | 0.228 | 0.271 |
| -8.5 | 0.031 | 0.100 | 0.035 | 0.254 | 0.268 |
| -8.0 | 0.036 | 0.102 | 0.042 | 0.259 | 0.259 |
| -7.5 | 0.034 | 0.107 | 0.047 | 0.251 | 0.254 |
| -7.0 | 0.030 | 0.101 | 0.034 | 0.252 | 0.251 |
| -6.5 | 0.034 | 0.097 | 0.040 | 0.256 | 0.248 |
| -6.0 | 0.029 | 0.089 | 0.035 | 0.260 | 0.255 |
| -5.5 | 0.027 | 0.091 | 0.032 | 0.246 | 0.266 |
| -5.0 | 0.026 | 0.089 | 0.031 | 0.240 | 0.268 |
| -4.5 | 0.029 | 0.093 | 0.035 | 0.241 | 0.269 |
| -4.0 | 0.030 | 0.094 | 0.033 | 0.239 | 0.266 |
| -3.5 | 0.038 | 0.099 | 0.040 | 0.244 | 0.256 |
| -3.0 | 0.038 | 0.110 | 0.045 | 0.236 | 0.255 |
| -2.5 | 0.038 | 0.114 | 0.041 | 0.217 | 0.267 |
| -2.0 | 0.053 | 0.135 | 0.054 | 0.228 | 0.274 |
| -1.5 | 0.064 | 0.163 | 0.068 | 0.229 | 0.268 |
| -1.0 | 0.092 | 0.233 | 0.100 | 0.265 | 0.267 |
| -0.5 | 0.159 | 0.328 | 0.152 | 0.310 | 0.271 |
| +0.0 | 0.284 | 0.457 | 0.261 | 0.345 | 0.283 |
| +0.5 | 0.406 | 0.564 | 0.375 | 0.431 | 0.287 |
| +1.0 | 0.500 | 0.643 | 0.469 | 0.518 | 0.283 |
| +1.5 | 0.561 | 0.701 | 0.540 | 0.575 | 0.279 |
| +2.0 | 0.537 | 0.692 | 0.549 | 0.574 | 0.269 |
| +2.5 | 0.428 | 0.643 | 0.460 | 0.518 | 0.266 |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno and LaGoVAD; no cause follows from n = 1.
