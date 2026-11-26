# DAD position diagnostics (POST-HOC, printed not gated, n = 1 seed)

165 positives, 301 negatives, 15 rows each; accident rows [13, 14]; first V2-S window without a repeated frame = row 7. **No-pixel `t/N` ruler macro = 1.0000.**

## Within video: positives vs the same mask on negatives (placebo)

| row | macro, all rows | placebo, all rows | gap | macro, rows 7+ | placebo, rows 7+ | gap |
|---|---|---|---|---|---|---|
| LaGoVAD | 0.6462 | 0.5134 [0.4709, 0.5564] | +0.1327 [+0.0615, +0.2020] | 0.6439 | 0.5260 [0.4817, 0.5692] | +0.1179 [+0.0440, +0.1895] |
| K5dfiDoTA | 0.9287 | 0.8916 [0.8717, 0.9100] | +0.0370 [+0.0121, +0.0621] | 0.8571 | 0.8236 [0.7946, 0.8519] | +0.0334 [-0.0085, +0.0736] |
| K5dfieDoTA | 0.9394 | 0.8971 [0.8775, 0.9153] | +0.0423 [+0.0186, +0.0666] | 0.8778 | 0.8295 [0.8009, 0.8571] | +0.0483 [+0.0079, +0.0880] |
| K5dfibk25DoTA | 0.9140 | 0.8818 [0.8611, 0.9010] | +0.0322 [+0.0055, +0.0588] | 0.8293 | 0.8048 [0.7757, 0.8336] | +0.0245 [-0.0201, +0.0661] |
| K5dfiebk25DoTA | 0.9189 | 0.8810 [0.8605, 0.9002] | +0.0378 [+0.0119, +0.0636] | 0.8399 | 0.7993 [0.7697, 0.8281] | +0.0406 [-0.0031, +0.0825] |

## Same position: positive vs negative videos (position cannot help)

| row | accident rows | rows 7 up to the accident | difference |
|---|---|---|---|
| LaGoVAD | 0.5881 [0.5339, 0.6419] | 0.5762 [0.5220, 0.6308] | +0.0118 [+0.0025, +0.0212] |
| K5dfiDoTA | 0.7553 [0.7077, 0.8005] | 0.7165 [0.6684, 0.7631] | +0.0388 [-0.0007, +0.0777] |
| K5dfieDoTA | 0.7596 [0.7116, 0.8049] | 0.7203 [0.6719, 0.7676] | +0.0393 [-0.0018, +0.0797] |
| K5dfibk25DoTA | 0.7522 [0.7040, 0.7974] | 0.7243 [0.6767, 0.7705] | +0.0279 [-0.0133, +0.0680] |
| K5dfiebk25DoTA | 0.7510 [0.7028, 0.7967] | 0.7301 [0.6823, 0.7770] | +0.0209 [-0.0213, +0.0622] |

Accident rows, paired:

* LaGoVAD - K5dfiDoTA: -0.1673 [-0.2294, -0.1045]
* K5dfieDoTA - K5dfiDoTA: +0.0043 [-0.0075, +0.0162]
* K5dfibk25DoTA - K5dfiDoTA: -0.0031 [-0.0096, +0.0035]
* K5dfiebk25DoTA - K5dfiDoTA: -0.0043 [-0.0185, +0.0099]
