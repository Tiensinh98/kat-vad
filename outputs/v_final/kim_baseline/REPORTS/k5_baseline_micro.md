# K5 between-video baseline: micro under per-video transforms (`KIM_RUN.md` §21)

PRELIMINARY (n = 1 seed). Saved scores only; nothing is trained. Macro is unchanged by every row (G2). Bootstrap: 2000 cluster draws, seed 2026. Oracle rows use the ground truth: ceilings, never methods. Every other transform needs the whole video (offline) and is a **protocol change, printed only**: it is not a model improvement and never sits beside a published number.

## DoTA-CAP-dev/K5

569 two-class videos (0 single-class left out), 91 bootstrap clusters; frame base rate 0.3271. As read: **min-max**.

* within-video frame pairs: 0.002 of all abnormal x normal pairs, pair-weighted AUC 0.8033; the rest cross two videos.
* normal-level spread on the as-read curves: std of the videos' normal median 0.2733 against a mean normal-to-abnormal gap 0.4021 (ratio 0.68).
* calibration headroom (`oracle normal-z` minus as-read, lower bound > 0.02): **not shown**.

| row | micro | AP | macro | cross-video AUC | Δ micro vs as-read [95 %] | sentence | share of headroom |
|---|---:|---:|---:|---:|---|---|---:|
| raw | 0.6980 | 0.5459 | 0.8195 | 0.6979 | -0.0469 [-0.0556, -0.0387] | below | -2.00 |
| min-max (as read) | 0.7449 | 0.5790 | 0.8195 | 0.7448 | — | — | — |
| z-score | 0.7597 | 0.5565 | 0.8195 | 0.7596 | +0.0148 [+0.0047, +0.0254] | above | 0.63 |
| minus median | 0.7481 | 0.5555 | 0.8195 | 0.7480 | +0.0032 [-0.0099, +0.0167] | not separated | 0.14 |
| minus q20 | 0.7215 | 0.5445 | 0.8195 | 0.7213 | -0.0235 [-0.0321, -0.0145] | below | -1.00 |
| rank | 0.7735 | 0.5915 | 0.8195 | 0.7734 | +0.0286 [+0.0172, +0.0396] | above | 1.22 |
| oracle offset (oracle) | 0.7369 | 0.5807 | 0.8195 | 0.7368 | -0.0080 [-0.0215, +0.0063] | — | -0.34 |
| oracle normal-z (oracle) | 0.7683 | 0.6807 | 0.8195 | 0.7683 | +0.0234 [+0.0103, +0.0368] | — | 1.00 |

## DADA-2000/K5

370 two-class videos (0 single-class left out), 370 bootstrap clusters; frame base rate 0.2885. As read: **raw**.

* within-video frame pairs: 0.003 of all abnormal x normal pairs, pair-weighted AUC 0.7584; the rest cross two videos.
* normal-level spread on the as-read curves: std of the videos' normal median 0.2291 against a mean normal-to-abnormal gap 0.3433 (ratio 0.67).
* calibration headroom (`oracle normal-z` minus as-read, lower bound > 0.02): **not shown**.

| row | micro | AP | macro | cross-video AUC | Δ micro vs as-read [95 %] | sentence | share of headroom |
|---|---:|---:|---:|---:|---|---|---:|
| raw (as read) | 0.7222 | 0.5557 | 0.7740 | 0.7221 | — | — | — |
| min-max | 0.7635 | 0.5640 | 0.7740 | 0.7635 | +0.0413 [+0.0314, +0.0514] | above | 1.71 |
| z-score | 0.7548 | 0.5106 | 0.7740 | 0.7548 | +0.0326 [+0.0175, +0.0474] | above | 1.35 |
| minus median | 0.7424 | 0.5401 | 0.7740 | 0.7423 | +0.0202 [+0.0042, +0.0360] | above | 0.84 |
| minus q20 | 0.7315 | 0.5452 | 0.7740 | 0.7314 | +0.0093 [+0.0011, +0.0166] | above | 0.39 |
| rank | 0.7539 | 0.5276 | 0.7740 | 0.7538 | +0.0317 [+0.0148, +0.0481] | above | 1.31 |
| oracle offset (oracle) | 0.7277 | 0.5578 | 0.7740 | 0.7276 | +0.0055 [-0.0094, +0.0205] | — | 0.23 |
| oracle normal-z (oracle) | 0.7463 | 0.6263 | 0.7740 | 0.7463 | +0.0241 [+0.0073, +0.0407] | — | 1.00 |

## DoTA-CAP-dev/LaGoVAD

569 two-class videos (0 single-class left out), 91 bootstrap clusters; frame base rate 0.3271. As read: **min-max**.

* within-video frame pairs: 0.002 of all abnormal x normal pairs, pair-weighted AUC 0.6353; the rest cross two videos.
* normal-level spread on the as-read curves: std of the videos' normal median 0.2257 against a mean normal-to-abnormal gap 0.1573 (ratio 1.43).
* calibration headroom (`oracle normal-z` minus as-read, lower bound > 0.02): **not shown**.

| row | micro | AP | macro | cross-video AUC | Δ micro vs as-read [95 %] | sentence | share of headroom |
|---|---:|---:|---:|---:|---|---|---:|
| raw | 0.4996 | 0.3274 | 0.6418 | 0.4994 | -0.1109 [-0.1373, -0.0826] | below | -4.56 |
| min-max (as read) | 0.6105 | 0.3838 | 0.6418 | 0.6104 | — | — | — |
| z-score | 0.6217 | 0.4002 | 0.6418 | 0.6217 | +0.0113 [+0.0043, +0.0186] | above | 0.46 |
| minus median | 0.5914 | 0.3763 | 0.6418 | 0.5913 | -0.0191 [-0.0315, -0.0063] | below | -0.78 |
| minus q20 | 0.6036 | 0.3738 | 0.6418 | 0.6036 | -0.0068 [-0.0219, +0.0086] | not separated | -0.28 |
| rank | 0.6214 | 0.4004 | 0.6418 | 0.6214 | +0.0110 [+0.0022, +0.0200] | above | 0.45 |
| oracle offset (oracle) | 0.5876 | 0.4082 | 0.6418 | 0.5875 | -0.0229 [-0.0380, -0.0074] | — | -0.94 |
| oracle normal-z (oracle) | 0.6348 | 0.5584 | 0.6418 | 0.6348 | +0.0243 [+0.0076, +0.0409] | — | 1.00 |

## DADA-2000/LaGoVAD

370 two-class videos (0 single-class left out), 370 bootstrap clusters; frame base rate 0.2885. As read: **raw**.

* within-video frame pairs: 0.003 of all abnormal x normal pairs, pair-weighted AUC 0.6287; the rest cross two videos.
* normal-level spread on the as-read curves: std of the videos' normal median 0.1881 against a mean normal-to-abnormal gap 0.0271 (ratio 6.95).
* calibration headroom (`oracle normal-z` minus as-read, lower bound > 0.02): **shown**.

| row | micro | AP | macro | cross-video AUC | Δ micro vs as-read [95 %] | sentence | share of headroom |
|---|---:|---:|---:|---:|---|---|---:|
| raw (as read) | 0.5388 | 0.3138 | 0.6202 | 0.5385 | — | — | — |
| min-max | 0.6128 | 0.3532 | 0.6202 | 0.6128 | +0.0741 [+0.0495, +0.0975] | above | 1.09 |
| z-score | 0.6152 | 0.3534 | 0.6202 | 0.6151 | +0.0764 [+0.0506, +0.1020] | above | 1.13 |
| minus median | 0.6157 | 0.3642 | 0.6202 | 0.6157 | +0.0769 [+0.0499, +0.1017] | above | 1.14 |
| minus q20 | 0.6147 | 0.3574 | 0.6202 | 0.6147 | +0.0760 [+0.0590, +0.0916] | above | 1.12 |
| rank | 0.6167 | 0.3561 | 0.6202 | 0.6166 | +0.0779 [+0.0524, +0.1022] | above | 1.15 |
| oracle offset (oracle) | 0.5987 | 0.3804 | 0.6202 | 0.5987 | +0.0600 [+0.0363, +0.0836] | — | 0.89 |
| oracle normal-z (oracle) | 0.6065 | 0.4754 | 0.6202 | 0.6064 | +0.0677 [+0.0401, +0.0944] | — | 1.00 |
