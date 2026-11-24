# K5dfibk25 vs K5dfirr (K5dfi re-trained in the same notebook) -- s2024

Rule: `core/docs/final/KIM_RUN.md §51`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the project reference. Last-epoch mul_mil: K5dfirr 0.1519, K5dfibk25 0.1510, K5dfi 0.1519, K5f 0.0963.

* **Verdict: CLOSED (target B not met).** Target A (first third, Nexar val and DADA) False; target B (DoTA onset) False.
* A, strict (lower bound >= -0.01, printed): nexar False, dada True.
* Start gate: step-1 mil 0.690560 vs K5dfirr 0.690560, same draws = True.
* K5dfirr vs the old K5dfi: step-1 mil 0.690560 vs 0.690560, same draws = True.
* Environment (saved beside the runs): {'K5dfirr': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K5dfibk25': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}}.

## First third of the span vs all normal rows (per video)

| set | videos | K5dfirr | K5dfibk25 | K5dfibk25 - K5dfirr |
|---|---|---|---|---|
| nexar | 112 | 0.8457 | 0.8350 | -0.0107 [-0.0173, -0.0045] |
| dota (printed) | 568 | 0.8107 | 0.8124 | +0.0017 [-0.0013, +0.0047] |
| dada | 370 | 0.7180 | 0.7122 | -0.0058 [-0.0099, -0.0018] |

## Onset AUC: span vs the normal rows within 2 s

| set | side | K5dfirr | K5dfibk25 | K5dfibk25 - K5dfirr |
|---|---|---|---|---|
| nexar | pre | 0.8548 | 0.8363 | -0.0185 [-0.0264, -0.0112] |
| nexar | post | 0.8152 | 0.8226 | +0.0074 [+0.0023, +0.0123] |
| dota | pre | 0.8276 | 0.8271 | -0.0005 [-0.0036, +0.0025] |
| dota | post | 0.7262 | 0.7312 | +0.0050 [+0.0014, +0.0088] |
| dada | pre | 0.7913 | 0.7854 | -0.0059 [-0.0098, -0.0019] |
| dada | post | 0.6529 | 0.6544 | +0.0015 [-0.0028, +0.0058] |

## Macro (Nexar = val crops): K5dfibk25 - K5dfirr decides; K5dfirr - K5dfi is the re-run spread (printed)

| set | K5dfirr | K5dfibk25 | K5dfibk25 - K5dfirr | label | K5dfirr - K5dfi |
|---|---|---|---|---|---|
| nexar | 0.8698 | 0.8611 | -0.0087 [-0.0130, -0.0048] below | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dota | 0.8259 | 0.8252 | -0.0007 [-0.0027, +0.0012] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dada | 0.7677 | 0.7632 | -0.0045 [-0.0070, -0.0018] below | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| tad (printed) | 0.7198 | 0.7160 | -0.0038 [-0.0123, +0.0042] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dad | 0.7494 | 0.7515 | +0.0021 [-0.0079, +0.0121] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |

## Macro against K5f (the project reference): a COST here also fails (BD6); starts not matched (another recipe)

| set | K5f | K5dfibk25 | K5dfibk25 - K5f | label |
|---|---|---|---|---|
| nexar | 0.8942 | 0.8611 | -0.0332 [-0.0433, -0.0236] below | COST |
| dota | 0.8188 | 0.8252 | +0.0064 [-0.0017, +0.0147] not separated | NO DIFFERENCE |
| dada | 0.7744 | 0.7632 | -0.0112 [-0.0206, -0.0018] below | COST |
| tad (printed) | 0.6981 | 0.7160 | +0.0179 [-0.0047, +0.0424] not separated | NO DIFFERENCE |
| dad | 0.8070 | 0.7515 | -0.0555 [-0.0760, -0.0357] below | COST |

Printed: Nexar val negatives with peak > 0.5: K5dfirr 0.681, K5dfibk25 0.717, K5dfi 0.681, K5f 0.549.

Never from this report: "bottom-k fixes the early rise"; any read of K5fbk (weight 1.0) against K5dfirr; a runtime conclusion from one re-run.


# Four sets, macro and micro

# k5dfi_bottomk25_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5dfirr**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |
| K5dfibk25 | 0.8252 / 0.7690 | 0.7632 / 0.7587 | 0.7160 / 0.6025 | 0.7515 / 0.7104 |
| K5dfi | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |
| K5dfibk25 | 0.8252 [0.8106, 0.8394] | 0.7690 | 0.6045 | 0.3271 |
| K5dfi | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfibk25 - K5dfirr | -0.0007 [-0.0027, +0.0012] | not separated | +0.0077 [+0.0054, +0.0101] | above |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | -0.0071 [-0.0153, +0.0009] | not separated | -0.0261 [-0.0335, -0.0188] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5dfibk25 | 0.7632 [0.7397, 0.7862] | 0.7587 | 0.5583 | 0.2885 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfibk25 - K5dfirr | -0.0045 [-0.0070, -0.0018] | below | -0.0003 [-0.0028, +0.0020] | not separated |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | +0.0067 [-0.0024, +0.0156] | not separated | -0.0040 [-0.0128, +0.0044] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |
| K5dfibk25 | 0.7160 [0.6491, 0.7810] | 0.6025 | 0.2245 | 0.0724 |
| K5dfi | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfibk25 - K5dfirr | -0.0038 [-0.0123, +0.0042] | not separated | -0.0094 [-0.0215, +0.0024] | not separated |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | -0.0217 [-0.0485, +0.0018] | not separated | +0.0592 [+0.0224, +0.0943] | above |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |
| K5dfibk25 | 0.7515 [0.7119, 0.7893] | 0.7104 | 0.1168 | 0.0472 |
| K5dfi | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfibk25 - K5dfirr | +0.0021 [-0.0079, +0.0121] | not separated | +0.0013 [-0.0029, +0.0051] | not separated |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | +0.0576 [+0.0378, +0.0779] | above | -0.0019 [-0.0119, +0.0086] | not separated |


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
