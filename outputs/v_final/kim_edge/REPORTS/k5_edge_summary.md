# K5fepre / K5fepost / K5fbk vs K5f -- s2024

Rule: `core/docs/final/KIM_RUN.md §49`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the reference. Last-epoch mul_mil: K5f 0.0963, K5fepre 0.1335, K5fepost 0.1173, K5fbk 0.0730.

* **K5fepre: TRADE-OFF (closed).** T1 hit False, T2 hit True.
* **K5fepost: closed (no gain, or a cost).** T1 hit False, T2 hit False (printed).
* **K5fbk: closed (no gain, or a cost).** T1 hit True, T2 hit True (printed).
* K5fbk step-1 mil 0.692545 vs K5f 0.687794; same draws = False. **FLAG: not the same draws as K5f.**

## T1: Nexar val zone AUC [-1, 0) s vs negative-video frames (lower = less early rise)

| row | AUC | row - K5f |
|---|---|---|
| K5f | 0.6943 | - |
| K5fepre | 0.6825 | -0.0119 [-0.0277, +0.0045] |
| K5fepost | 0.7115 | +0.0172 [-0.0009, +0.0353] |
| K5fbk | 0.6224 | -0.0720 [-0.1050, -0.0404] |

## Onset AUC: span vs the normal frames within 2 s (higher = sharper edge; T2 = dota, pre)

| set | side | row | videos | AUC | row - K5f |
|---|---|---|---|---|---|
| nexar | pre | K5f | 112 | 0.8907 | - |
| nexar | pre | K5fepre | 112 | 0.8851 | -0.0055 [-0.0171, +0.0066] |
| nexar | pre | K5fepost | 112 | 0.8630 | -0.0276 [-0.0418, -0.0133] |
| nexar | pre | K5fbk | 112 | 0.7999 | -0.0908 [-0.1146, -0.0677] |
| nexar | post | K5f | 112 | 0.8441 | - |
| nexar | post | K5fepre | 112 | 0.8137 | -0.0305 [-0.0418, -0.0188] |
| nexar | post | K5fepost | 112 | 0.8797 | +0.0356 [+0.0218, +0.0513] |
| nexar | post | K5fbk | 112 | 0.8056 | -0.0386 [-0.0610, -0.0167] |
| dota | pre | K5f | 568 | 0.7812 | - |
| dota | pre | K5fepre | 568 | 0.8475 | +0.0663 [+0.0546, +0.0787] |
| dota | pre | K5fepost | 568 | 0.7395 | -0.0416 [-0.0544, -0.0293] |
| dota | pre | K5fbk | 568 | 0.8272 | +0.0460 [+0.0322, +0.0600] |
| dota | post | K5f | 542 | 0.7657 | - |
| dota | post | K5fepre | 542 | 0.6949 | -0.0709 [-0.0817, -0.0599] |
| dota | post | K5fepost | 542 | 0.8088 | +0.0430 [+0.0304, +0.0566] |
| dota | post | K5fbk | 542 | 0.7112 | -0.0546 [-0.0665, -0.0419] |
| dada | pre | K5f | 356 | 0.7844 | - |
| dada | pre | K5fepre | 356 | 0.8036 | +0.0192 [+0.0081, +0.0304] |
| dada | pre | K5fepost | 356 | 0.7619 | -0.0225 [-0.0367, -0.0085] |
| dada | pre | K5fbk | 356 | 0.7594 | -0.0251 [-0.0414, -0.0089] |
| dada | post | K5f | 354 | 0.6877 | - |
| dada | post | K5fepre | 354 | 0.6385 | -0.0492 [-0.0602, -0.0382] |
| dada | post | K5fepost | 354 | 0.7369 | +0.0492 [+0.0355, +0.0634] |
| dada | post | K5fbk | 354 | 0.6367 | -0.0510 [-0.0654, -0.0365] |

## Macro per row (Nexar = val crops) and row - K5f, paired, 95 % cluster bootstrap

| set | K5f | K5fepre | K5fepre - K5f | label | K5fepost | K5fepost - K5f | label | K5fbk | K5fbk - K5f | label |
|---|---|---|---|---|---|---|---|---|---|---|
| nexar | 0.8942 | 0.8794 | -0.0149 [-0.0214, -0.0087] below | COST | 0.8962 | +0.0020 [-0.0067, +0.0105] not separated | NO DIFFERENCE | 0.8210 | -0.0732 [-0.0883, -0.0587] below | COST |
| dota | 0.8188 | 0.8214 | +0.0026 [-0.0047, +0.0103] not separated | NO DIFFERENCE | 0.8126 | -0.0062 [-0.0130, +0.0009] not separated | NO DIFFERENCE | 0.8125 | -0.0063 [-0.0143, +0.0017] not separated | NO DIFFERENCE |
| dada | 0.7744 | 0.7658 | -0.0086 [-0.0158, -0.0014] below | NO DIFFERENCE | 0.7803 | +0.0059 [-0.0026, +0.0145] not separated | NO DIFFERENCE | 0.7413 | -0.0331 [-0.0427, -0.0232] below | COST |
| tad (printed) | 0.6981 | 0.7016 | +0.0035 [-0.0167, +0.0259] not separated | NO DIFFERENCE | 0.7067 | +0.0086 [-0.0170, +0.0348] not separated | NO DIFFERENCE | 0.6926 | -0.0055 [-0.0338, +0.0238] not separated | NO DIFFERENCE |
| dad | 0.8070 | 0.8033 | -0.0037 [-0.0198, +0.0124] not separated | NO DIFFERENCE | 0.7128 | -0.0942 [-0.1161, -0.0732] below | COST | 0.7557 | -0.0513 [-0.0732, -0.0301] below | COST |

Printed: Nexar val negatives with peak > 0.5: K5f 0.549, K5fepre 0.504, K5fepost 0.708, K5fbk 0.549.

Never from this report: that edge windows reduce the early rise (one hit or a TRADE-OFF), which loss acts, "K5f anticipates".


# Four sets, macro and micro

# k5_edge_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5f**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |
| K5fepre | 0.8214 / 0.7409 | 0.7658 / 0.7512 | 0.7016 / 0.6647 | 0.8033 / 0.7228 |
| K5fepost | 0.8126 / 0.7471 | 0.7803 / 0.7730 | 0.7067 / 0.6366 | 0.7128 / 0.6911 |
| K5fbk | 0.8125 / 0.7793 | 0.7413 / 0.7452 | 0.6926 / 0.6192 | 0.7557 / 0.7401 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |
| K5fepre | 0.8214 [0.8069, 0.8354] | 0.7409 | 0.5700 | 0.3271 |
| K5fepost | 0.8126 [0.7969, 0.8273] | 0.7471 | 0.5847 | 0.3271 |
| K5fbk | 0.8125 [0.7976, 0.8272] | 0.7793 | 0.6110 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | +0.0026 [-0.0047, +0.0103] | not separated | +0.0056 [-0.0011, +0.0123] | not separated |
| K5fepost - K5f | -0.0062 [-0.0130, +0.0009] | not separated | +0.0119 [+0.0059, +0.0178] | above |
| K5fbk - K5f | -0.0063 [-0.0143, +0.0017] | not separated | +0.0440 [+0.0343, +0.0535] | above |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |
| K5fepre | 0.7658 [0.7410, 0.7896] | 0.7512 | 0.5455 | 0.2885 |
| K5fepost | 0.7803 [0.7568, 0.8034] | 0.7730 | 0.5783 | 0.2885 |
| K5fbk | 0.7413 [0.7178, 0.7642] | 0.7452 | 0.5380 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | -0.0086 [-0.0158, -0.0014] | below | -0.0039 [-0.0103, +0.0025] | not separated |
| K5fepost - K5f | +0.0059 [-0.0026, +0.0145] | not separated | +0.0179 [+0.0099, +0.0261] | above |
| K5fbk - K5f | -0.0331 [-0.0427, -0.0232] | below | -0.0099 [-0.0194, +0.0001] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |
| K5fepre | 0.7016 [0.6274, 0.7723] | 0.6647 | 0.2519 | 0.0724 |
| K5fepost | 0.7067 [0.6408, 0.7694] | 0.6366 | 0.2056 | 0.0724 |
| K5fbk | 0.6926 [0.6299, 0.7536] | 0.6192 | 0.2218 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | +0.0035 [-0.0167, +0.0259] | not separated | -0.0064 [-0.0594, +0.0513] | not separated |
| K5fepost - K5f | +0.0086 [-0.0170, +0.0348] | not separated | -0.0346 [-0.0722, +0.0038] | not separated |
| K5fbk - K5f | -0.0055 [-0.0338, +0.0238] | not separated | -0.0519 [-0.0881, -0.0176] | below |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |
| K5fepre | 0.8033 [0.7674, 0.8375] | 0.7228 | 0.1255 | 0.0472 |
| K5fepost | 0.7128 [0.6720, 0.7527] | 0.6911 | 0.0926 | 0.0472 |
| K5fbk | 0.7557 [0.7170, 0.7930] | 0.7401 | 0.1417 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5fepre - K5f | -0.0037 [-0.0198, +0.0124] | not separated | +0.0156 [+0.0053, +0.0253] | above |
| K5fepost - K5f | -0.0942 [-0.1161, -0.0732] | below | -0.0161 [-0.0262, -0.0065] | below |
| K5fbk - K5f | -0.0513 [-0.0732, -0.0301] | below | +0.0329 [+0.0194, +0.0456] | above |


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


# Pre-onset rise on dota (KIM_RUN.md §43 / §43.1)

569 videos; onset = first `gt = 1` frame; 10 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | - | 0.053 [0.031, 0.079] | 0.412 [0.382, 0.441] | 0.565 [0.533, 0.596] | 1.028 [0.761, 1.361] | - |
| K5fepre | - | 0.049 [0.027, 0.074] | 0.364 [0.337, 0.390] | 0.608 [0.578, 0.638] | 0.798 [0.625, 1.015] | -0.210 [-0.369, -0.090] |
| K5fepost | - | 0.033 [0.017, 0.053] | 0.338 [0.311, 0.365] | 0.506 [0.475, 0.536] | 1.636 [0.796, 2.799] | 0.610 [-0.013, 1.494] |
| K5fbk | - | 0.005 [0.003, 0.007] | 0.111 [0.095, 0.127] | 0.446 [0.417, 0.474] | 0.542 [0.369, 0.759] | -0.491 [-0.732, -0.310] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5f | - | - | - | - | - |
| K5fepre | - | - | - | - | - |
| K5fepost | - | - | - | - | - |
| K5fbk | - | - | - | - | - |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5f | K5fepre | K5fepost | K5fbk |
|---|---|---|---|---|
| -10.0 | 0.073 | 0.082 | 0.019 | 0.032 |
| -9.5 | 0.104 | 0.112 | 0.015 | 0.040 |
| -9.0 | 0.095 | 0.092 | 0.025 | 0.033 |
| -8.5 | 0.074 | 0.066 | 0.028 | 0.020 |
| -8.0 | 0.058 | 0.065 | 0.036 | 0.008 |
| -7.5 | 0.064 | 0.083 | 0.062 | 0.005 |
| -7.0 | 0.066 | 0.070 | 0.051 | 0.004 |
| -6.5 | 0.073 | 0.056 | 0.047 | 0.003 |
| -6.0 | 0.083 | 0.064 | 0.057 | 0.004 |
| -5.5 | 0.089 | 0.078 | 0.070 | 0.009 |
| -5.0 | 0.098 | 0.095 | 0.076 | 0.023 |
| -4.5 | 0.096 | 0.091 | 0.069 | 0.030 |
| -4.0 | 0.108 | 0.099 | 0.071 | 0.035 |
| -3.5 | 0.137 | 0.114 | 0.089 | 0.032 |
| -3.0 | 0.200 | 0.166 | 0.131 | 0.035 |
| -2.5 | 0.267 | 0.223 | 0.186 | 0.051 |
| -2.0 | 0.337 | 0.280 | 0.249 | 0.062 |
| -1.5 | 0.394 | 0.335 | 0.309 | 0.084 |
| -1.0 | 0.440 | 0.390 | 0.369 | 0.118 |
| -0.5 | 0.491 | 0.463 | 0.438 | 0.185 |
| +0.0 | 0.544 | 0.543 | 0.497 | 0.304 |
| +0.5 | 0.579 | 0.603 | 0.538 | 0.422 |
| +1.0 | 0.592 | 0.637 | 0.549 | 0.496 |
| +1.5 | 0.586 | 0.645 | 0.520 | 0.531 |
| +2.0 | 0.556 | 0.624 | 0.462 | 0.503 |
| +2.5 | 0.497 | 0.571 | 0.372 | 0.411 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

569 two-class videos; 76 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5f | rescaled | 59 | 0.27 [0.09, 0.56] | 0.45 [0.28, 0.66] | 0.60 [0.43, 0.82] | 0.74 [0.41, 1.04] | 0.85 [0.50, 1.17] |
| K5f | minmax | 76 | 0.21 [0.16, 0.28] | 0.34 [0.27, 0.41] | 0.46 [0.39, 0.53] | 0.55 [0.48, 0.63] | 0.64 [0.56, 0.71] |
| K5fepre | rescaled | 61 | 0.18 [0.09, 0.29] | 0.42 [0.25, 0.63] | 0.56 [0.39, 0.76] | 0.65 [0.47, 0.84] | 0.75 [0.49, 0.98] |
| K5fepre | minmax | 76 | 0.20 [0.14, 0.26] | 0.29 [0.23, 0.37] | 0.38 [0.31, 0.46] | 0.47 [0.39, 0.54] | 0.57 [0.49, 0.65] |
| K5fepost | rescaled | 56 | 0.11 [0.05, 0.18] | 0.23 [0.16, 0.31] | 0.43 [0.32, 0.53] | 0.64 [0.51, 0.77] | 0.86 [0.74, 0.98] |
| K5fepost | minmax | 76 | 0.17 [0.11, 0.23] | 0.24 [0.18, 0.31] | 0.34 [0.27, 0.40] | 0.45 [0.38, 0.52] | 0.56 [0.49, 0.64] |
| K5fbk | rescaled | 61 | 0.09 [0.03, 0.18] | 0.24 [0.09, 0.42] | 0.25 [0.09, 0.41] | 0.35 [0.19, 0.51] | 0.71 [0.40, 1.14] |
| K5fbk | minmax | 76 | 0.08 [0.05, 0.12] | 0.14 [0.09, 0.20] | 0.13 [0.09, 0.18] | 0.18 [0.13, 0.23] | 0.32 [0.25, 0.40] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5f | 0.8188 [0.8032, 0.8338] | 0.0192 [0.0108, 0.0274] |
| K5fepre | 0.8214 [0.8065, 0.8360] | -0.0084 [-0.0161, -0.0009] |
| K5fepost | 0.8126 [0.7971, 0.8273] | 0.0384 [0.0303, 0.0465] |
| K5fbk | 0.8125 [0.7965, 0.8275] | -0.0046 [-0.0115, 0.0022] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5f | 0.33 (n=68) | 0.57 (n=298) | 0.60 (n=202) |
| K5fepre | 0.27 (n=68) | 0.49 (n=298) | 0.53 (n=202) |
| K5fepost | 0.27 (n=68) | 0.50 (n=298) | 0.54 (n=202) |
| K5fbk | 0.05 (n=68) | 0.16 (n=298) | 0.27 (n=202) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.


# Pre-onset rise on dada (KIM_RUN.md §43 / §43.1)

370 videos; onset = first `gt = 1` frame; 3.75 rows/s. **Printed only, decides nothing; PRELIMINARY (n = 1).** Control K5f.

## 1. Levels and rise (raw scores; intervals over videos)

rise = (mean[-2 s, 0) - far) / (span - far), far = mean before -5 s: 0 = still at the far level 2 s before the alert, 1 = already at the span level.

| row | negative videos | far | last 2 s | span | rise | rise - K5f |
|---|---|---|---|---|---|---|
| K5f | - | 0.081 [0.058, 0.106] | 0.289 [0.256, 0.324] | 0.486 [0.447, 0.526] | 0.769 [0.425, 1.250] | - |
| K5fepre | - | 0.090 [0.065, 0.117] | 0.290 [0.260, 0.322] | 0.534 [0.496, 0.572] | 0.019 [-1.148, 0.706] | -0.239 [-0.525, -0.032] |
| K5fepost | - | 0.044 [0.029, 0.061] | 0.224 [0.195, 0.255] | 0.412 [0.375, 0.451] | 1.511 [0.400, 3.526] | 0.599 [-0.313, 2.164] |
| K5fbk | - | 0.023 [0.015, 0.032] | 0.096 [0.077, 0.116] | 0.321 [0.290, 0.354] | 1.098 [0.232, 2.720] | 0.421 [-0.230, 1.625] |

## 2. Zone AUC against every negative-video frame (0.5 = like no accident)

| row | far | span | [-5, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|
| K5f | - | - | - | - | - |
| K5fepre | - | - | - | - | - |
| K5fepost | - | - | - | - | - |
| K5fbk | - | - | - | - | - |

## 3. Event-locked mean raw score (bin start, s)

| bin | K5f | K5fepre | K5fepost | K5fbk |
|---|---|---|---|---|
| -10.0 | 0.076 | 0.088 | 0.026 | 0.028 |
| -9.5 | 0.088 | 0.097 | 0.043 | 0.038 |
| -9.0 | 0.096 | 0.105 | 0.051 | 0.031 |
| -8.5 | 0.128 | 0.128 | 0.078 | 0.036 |
| -8.0 | 0.110 | 0.119 | 0.061 | 0.022 |
| -7.5 | 0.106 | 0.116 | 0.048 | 0.029 |
| -7.0 | 0.110 | 0.126 | 0.058 | 0.029 |
| -6.5 | 0.106 | 0.122 | 0.065 | 0.032 |
| -6.0 | 0.114 | 0.124 | 0.068 | 0.038 |
| -5.5 | 0.127 | 0.137 | 0.077 | 0.049 |
| -5.0 | 0.125 | 0.143 | 0.075 | 0.039 |
| -4.5 | 0.132 | 0.155 | 0.086 | 0.043 |
| -4.0 | 0.132 | 0.150 | 0.085 | 0.043 |
| -3.5 | 0.131 | 0.151 | 0.089 | 0.045 |
| -3.0 | 0.132 | 0.154 | 0.093 | 0.048 |
| -2.5 | 0.173 | 0.191 | 0.119 | 0.068 |
| -2.0 | 0.230 | 0.241 | 0.163 | 0.084 |
| -1.5 | 0.276 | 0.280 | 0.208 | 0.091 |
| -1.0 | 0.330 | 0.330 | 0.267 | 0.111 |
| -0.5 | 0.380 | 0.381 | 0.320 | 0.128 |
| +0.0 | 0.408 | 0.428 | 0.358 | 0.166 |
| +0.5 | 0.453 | 0.488 | 0.401 | 0.237 |
| +1.0 | 0.488 | 0.532 | 0.426 | 0.318 |
| +1.5 | 0.501 | 0.554 | 0.422 | 0.366 |
| +2.0 | 0.495 | 0.551 | 0.392 | 0.374 |
| +2.5 | 0.451 | 0.522 | 0.323 | 0.331 |

## 4. Fixed composition and macro cost (§43.1, post-hoc)

370 two-class videos; 152 with >= 5 s before onset (the same videos in every zone). Rescaled: 0 = mean before -4 s, 1 = span mean, per video (span - far < 0.05 left out). Min-max: per video, as the `zs_curves` figure shows it.

| row | read | n | [-5, -4) s | [-4, -3) s | [-3, -2) s | [-2, -1) s | [-1, 0) s |
|---|---|---|---|---|---|---|---|
| K5f | rescaled | 109 | 0.12 [0.07, 0.18] | 0.16 [0.08, 0.24] | 0.19 [0.10, 0.29] | 0.36 [0.25, 0.47] | 0.54 [0.42, 0.65] |
| K5f | minmax | 152 | 0.19 [0.16, 0.24] | 0.21 [0.17, 0.25] | 0.23 [0.18, 0.27] | 0.33 [0.28, 0.38] | 0.43 [0.38, 0.49] |
| K5fepre | rescaled | 120 | 0.19 [0.08, 0.33] | 0.22 [0.12, 0.32] | 0.25 [0.12, 0.38] | 0.32 [0.19, 0.46] | 0.44 [0.31, 0.55] |
| K5fepre | minmax | 152 | 0.20 [0.16, 0.24] | 0.22 [0.18, 0.26] | 0.23 [0.19, 0.27] | 0.31 [0.27, 0.36] | 0.40 [0.35, 0.45] |
| K5fepost | rescaled | 108 | 0.13 [0.01, 0.27] | 0.20 [0.10, 0.31] | 0.24 [0.14, 0.35] | 0.43 [0.29, 0.58] | 0.64 [0.47, 0.82] |
| K5fepost | minmax | 152 | 0.13 [0.10, 0.17] | 0.15 [0.11, 0.18] | 0.15 [0.12, 0.19] | 0.26 [0.21, 0.31] | 0.36 [0.31, 0.42] |
| K5fbk | rescaled | 105 | 0.09 [0.03, 0.16] | 0.12 [0.04, 0.21] | 0.14 [0.06, 0.22] | 0.23 [0.12, 0.34] | 0.32 [0.17, 0.51] |
| K5fbk | minmax | 152 | 0.08 [0.06, 0.10] | 0.09 [0.06, 0.12] | 0.11 [0.08, 0.14] | 0.13 [0.10, 0.17] | 0.16 [0.13, 0.21] |

Macro AUC (two-class videos) and its change when the normal frames of [-2, 0) s are dropped (paired per video; > 0 = those frames cost macro):

| row | macro | without the zone - macro |
|---|---|---|
| K5f | 0.7744 [0.7500, 0.7982] | -0.0018 [-0.0094, 0.0056] |
| K5fepre | 0.7658 [0.7413, 0.7894] | -0.0112 [-0.0190, -0.0036] |
| K5fepost | 0.7803 [0.7566, 0.8033] | 0.0083 [0.0011, 0.0153] |
| K5fbk | 0.7413 [0.7183, 0.7643] | -0.0052 [-0.0124, 0.0014] |

Min-max level of the last 1 s before onset, by pre-roll length (a short pre-roll puts the video's minimum near the onset):

| row | <2 s | 2-4 s | >=4 s |
|---|---|---|---|
| K5f | 0.28 (n=72) | 0.46 (n=86) | 0.45 (n=198) |
| K5fepre | 0.26 (n=72) | 0.45 (n=86) | 0.43 (n=198) |
| K5fepost | 0.22 (n=72) | 0.41 (n=86) | 0.37 (n=198) |
| K5fbk | 0.08 (n=72) | 0.17 (n=86) | 0.18 (n=198) |

## How to read this

* A rise above 0 says the score leaves its far level before the alert; it is unlabeled positive score, not foresight (Nexar's alert is itself an anticipation mark). Never "anticipates accidents".
* Zone AUC uses the row's own ranking, so rows with different calibration compare; the rise is scale-free per video. Raw levels in §1/§3 are not comparable across rows.
* At the read's 0.1 s rows the head's kernel spans about +-0.1 s; bins before -0.5 s are not its smear (the encoder's attention can still carry them).
* A row contrast changes more than one training factor for K5fno, L0dfi and LaGoVAD; no cause follows from n = 1.
* §4 is post-hoc (§43.1): the same definitions on every set, so it is the table to compare sets with. Zone AUC (§2) needs negative videos and is `-` on DoTA / DADA.
