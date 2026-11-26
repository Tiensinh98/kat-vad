# DAD position diagnostics (POST-HOC, printed not gated, n = 1 seed)

165 positives, 301 negatives, 15 rows each; accident rows [13, 14]; first V2-S window without a repeated frame = row 7. **No-pixel `t/N` ruler macro = 1.0000.**

## Within video: positives vs the same mask on negatives (placebo)

| row | macro, all rows | placebo, all rows | gap | macro, rows 7+ | placebo, rows 7+ | gap |
|---|---|---|---|---|---|---|
| LaGoVAD | 0.6462 | 0.5134 [0.4709, 0.5564] | +0.1327 [+0.0615, +0.2020] | 0.6439 | 0.5260 [0.4817, 0.5692] | +0.1179 [+0.0440, +0.1895] |
| W0 = K5dfi | 0.7494 | 0.6222 [0.5887, 0.6555] | +0.1273 [+0.0770, +0.1757] | 0.6051 | 0.4956 [0.4543, 0.5365] | +0.1095 [+0.0434, +0.1743] |
| R4 = K5dfiDoTA | 0.9287 | 0.8916 [0.8717, 0.9100] | +0.0370 [+0.0121, +0.0621] | 0.8571 | 0.8236 [0.7946, 0.8519] | +0.0334 [-0.0085, +0.0736] |

## Same position: positive vs negative videos (position cannot help)

| row | accident rows | rows 7 up to the accident | difference |
|---|---|---|---|
| LaGoVAD | 0.5881 [0.5339, 0.6419] | 0.5762 [0.5220, 0.6308] | +0.0118 [+0.0025, +0.0212] |
| W0 = K5dfi | 0.6749 [0.6242, 0.7247] | 0.6495 [0.5966, 0.7007] | +0.0254 [+0.0047, +0.0462] |
| R4 = K5dfiDoTA | 0.7553 [0.7077, 0.8005] | 0.7165 [0.6684, 0.7631] | +0.0388 [-0.0007, +0.0777] |

Accident rows, paired:

* W0 - LaGoVAD: +0.0868 [+0.0237, +0.1504]
* R4 - LaGoVAD: +0.1673 [+0.1045, +0.2294]
* R4 - W0: +0.0804 [+0.0272, +0.1323]
