# K5fbk25 vs K5frr (K5f re-trained in the same notebook) -- s2024

Rule: `core/docs/final/KIM_RUN.md §50`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the project reference. Last-epoch mul_mil: K5frr 0.0963, K5fbk25 0.0932, K5f 0.0963.

* **Verdict: TRADE-OFF (closed).** Target A (first third, Nexar val and DADA) False; target B (DoTA onset) True.
* A, strict (lower bound >= -0.01, printed): nexar False, dada False.
* Start gate: step-1 mil 0.687794 vs K5frr 0.687794, same draws = True.
* K5frr vs the old K5f: step-1 mil 0.687794 vs 0.687794, same draws = True.
* Environment (saved beside the runs): {'K5frr': {'gpu': 'NVIDIA L4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K5fbk25': {'gpu': 'NVIDIA L4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}}.

## First third of the span vs all normal rows (per video)

| set | videos | K5frr | K5fbk25 | K5fbk25 - K5frr |
|---|---|---|---|---|
| nexar | 112 | 0.8706 | 0.8224 | -0.0482 [-0.0595, -0.0373] |
| dota (printed) | 568 | 0.8275 | 0.8187 | -0.0087 [-0.0133, -0.0040] |
| dada | 370 | 0.7495 | 0.7194 | -0.0301 [-0.0385, -0.0219] |

## Onset AUC: span vs the normal rows within 2 s

| set | side | K5frr | K5fbk25 | K5fbk25 - K5frr |
|---|---|---|---|---|
| nexar | pre | 0.8907 | 0.8634 | -0.0273 [-0.0388, -0.0162] |
| nexar | post | 0.8441 | 0.8174 | -0.0267 [-0.0346, -0.0191] |
| dota | pre | 0.7812 | 0.8266 | +0.0454 [+0.0375, +0.0541] |
| dota | post | 0.7657 | 0.7387 | -0.0270 [-0.0335, -0.0201] |
| dada | pre | 0.7844 | 0.7929 | +0.0085 [-0.0007, +0.0174] |
| dada | post | 0.6877 | 0.6620 | -0.0257 [-0.0334, -0.0177] |

## Macro (Nexar = val crops): K5fbk25 - K5frr decides; K5frr - K5f is the re-run spread (printed)

| set | K5frr | K5fbk25 | K5fbk25 - K5frr | label | K5frr - K5f |
|---|---|---|---|---|---|
| nexar | 0.8942 | 0.8679 | -0.0263 [-0.0324, -0.0206] below | COST | +0.0000 [+0.0000, +0.0000] not separated |
| dota | 0.8188 | 0.8268 | +0.0080 [+0.0041, +0.0121] above | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dada | 0.7744 | 0.7624 | -0.0120 [-0.0170, -0.0069] below | COST | +0.0000 [+0.0000, +0.0000] not separated |
| tad (printed) | 0.6981 | 0.7073 | +0.0092 [-0.0030, +0.0216] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dad | 0.8070 | 0.8063 | -0.0007 [-0.0084, +0.0070] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |

Printed: Nexar val negatives with peak > 0.5: K5frr 0.549, K5fbk25 0.575, K5f 0.549.

Never from this report: "bottom-k fixes the early rise"; any read of K5fbk (weight 1.0) against K5frr; a runtime conclusion from one re-run.


# Four sets, macro and micro

# k5_bottomk25_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5frr**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5frr | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |
| K5fbk25 | 0.8268 / 0.7714 | 0.7624 / 0.7583 | 0.7073 / 0.6562 | 0.8063 / 0.7304 |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |
| K5fbk25 | 0.8268 [0.8127, 0.8404] | 0.7714 | 0.6044 | 0.3271 |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | +0.0080 [+0.0041, +0.0121] | above | +0.0362 [+0.0312, +0.0414] | above |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| K5fbk25 | 0.7624 [0.7381, 0.7862] | 0.7583 | 0.5557 | 0.2885 |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | -0.0120 [-0.0170, -0.0069] | below | +0.0032 [-0.0017, +0.0086] | not separated |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |
| K5fbk25 | 0.7073 [0.6367, 0.7754] | 0.6562 | 0.2634 | 0.0724 |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | +0.0092 [-0.0030, +0.0216] | not separated | -0.0150 [-0.0268, -0.0034] | below |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5frr | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |
| K5fbk25 | 0.8063 [0.7681, 0.8427] | 0.7304 | 0.1304 | 0.0472 |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fbk25 - K5frr | -0.0007 [-0.0084, +0.0070] | not separated | +0.0232 [+0.0184, +0.0279] | above |
| K5f - K5frr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |


# Pre-onset rise on nexar (KIM_RUN.md §43 / §43.1)

225 videos; onset = first `gt = 1` frame (= `time_of_alert`); 30 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5frr.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5frr |
|---|---|---|---|---|---|---|
| K5frr | 0.046 [0.036, 0.057] | 0.025 [0.016, 0.035] | 0.092 [0.062, 0.125] | 0.511 [0.452, 0.570] | 0.179 [0.050, 0.397] | - |
| K5fbk25 | 0.036 [0.028, 0.044] | 0.017 [0.011, 0.025] | 0.053 [0.031, 0.080] | 0.425 [0.372, 0.478] | 0.155 [-0.011, 0.430] | -0.025 [-0.073, 0.042] |
| K5f | 0.046 [0.036, 0.057] | 0.025 [0.016, 0.035] | 0.092 [0.062, 0.125] | 0.511 [0.452, 0.570] | 0.179 [0.050, 0.397] | 0.000 [0.000, 0.000] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5frr | 0.437 | 0.887 | 0.525 | 0.590 | 0.694 |
| K5fbk25 | 0.426 | 0.862 | 0.517 | 0.574 | 0.650 |
| K5f | 0.437 | 0.887 | 0.525 | 0.590 | 0.694 |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5frr | K5fbk25 | K5f |
|---|---|---|---|
| -10.0 | 0.022 | 0.016 | 0.022 |
| -9.5 | 0.023 | 0.019 | 0.023 |
| -9.0 | 0.021 | 0.015 | 0.021 |
| -8.5 | 0.031 | 0.019 | 0.031 |
| -8.0 | 0.036 | 0.026 | 0.036 |
| -7.5 | 0.034 | 0.024 | 0.034 |
| -7.0 | 0.030 | 0.018 | 0.030 |
| -6.5 | 0.034 | 0.026 | 0.034 |
| -6.0 | 0.029 | 0.023 | 0.029 |
| -5.5 | 0.027 | 0.022 | 0.027 |
| -5.0 | 0.026 | 0.020 | 0.026 |
| -4.5 | 0.029 | 0.021 | 0.029 |
| -4.0 | 0.030 | 0.019 | 0.030 |
| -3.5 | 0.038 | 0.028 | 0.038 |
| -3.0 | 0.038 | 0.025 | 0.038 |
| -2.5 | 0.038 | 0.026 | 0.038 |
| -2.0 | 0.053 | 0.035 | 0.053 |
| -1.5 | 0.064 | 0.042 | 0.064 |
| -1.0 | 0.092 | 0.051 | 0.092 |
| -0.5 | 0.159 | 0.083 | 0.159 |
| +0.0 | 0.284 | 0.171 | 0.284 |
| +0.5 | 0.406 | 0.287 | 0.406 |
| +1.0 | 0.500 | 0.409 | 0.500 |
| +1.5 | 0.561 | 0.480 | 0.561 |
| +2.0 | 0.537 | 0.482 | 0.537 |
| +2.5 | 0.428 | 0.376 | 0.428 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

112 two-class videos; 112 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5frr | rescaled | 99 | 0.00 [-0.02, 0.02] | 0.01 [-0.02, 0.04] | 0.02 [-0.01, 0.05] | 0.04 [0.00, 0.09] | 0.15 [0.10, 0.22] |
| K5frr | minmax | 112 | 0.03 [0.02, 0.05] | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] | 0.06 [0.04, 0.09] | 0.13 [0.09, 0.18] |
| K5fbk25 | rescaled | 98 | 0.01 [-0.01, 0.03] | 0.01 [-0.02, 0.05] | 0.01 [-0.02, 0.04] | 0.03 [-0.01, 0.07] | 0.07 [0.03, 0.12] |
| K5fbk25 | minmax | 112 | 0.03 [0.01, 0.04] | 0.03 [0.01, 0.05] | 0.03 [0.01, 0.04] | 0.04 [0.02, 0.07] | 0.07 [0.04, 0.11] |
| K5f | rescaled | 99 | 0.00 [-0.02, 0.02] | 0.01 [-0.02, 0.04] | 0.02 [-0.01, 0.05] | 0.04 [0.00, 0.09] | 0.15 [0.10, 0.22] |
| K5f | minmax | 112 | 0.03 [0.02, 0.05] | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] | 0.06 [0.04, 0.09] | 0.13 [0.09, 0.18] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5frr | 0.9210 [0.8998, 0.9406] | 0.0016 [-0.0006, 0.0039] |
| K5fbk25 | 0.8977 [0.8753, 0.9190] | 0.0018 [-0.0003, 0.0040] |
| K5f | 0.9210 [0.8998, 0.9406] | 0.0016 [-0.0006, 0.0039] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5frr | - | - | 0.13 (n=112) |
| K5fbk25 | - | - | 0.07 (n=112) |
| K5f | - | - | 0.13 (n=112) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.


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
