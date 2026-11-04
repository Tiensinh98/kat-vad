# DAD position diagnostics (POST-HOC, printed not gated, n = 1 seed)

165 positives, 301 negatives, 15 rows each; accident rows [13, 14]; first V2-S window without a repeated frame = row 7. **No-pixel `t/N` ruler macro = 1.0000.**

## Within video: positives vs the same mask on negatives (placebo)

| row | macro, all rows | placebo, all rows | gap | macro, rows 7+ | placebo, rows 7+ | gap |
|---|---|---|---|---|---|---|
| LaGoVAD | 0.6462 | 0.5134 [0.4709, 0.5564] | +0.1327 [+0.0615, +0.2020] | 0.6439 | 0.5260 [0.4817, 0.5692] | +0.1179 [+0.0440, +0.1895] |
| W0 | 0.7317 | 0.6260 [0.5933, 0.6586] | +0.1057 [+0.0523, +0.1578] | 0.5980 | 0.5042 [0.4648, 0.5432] | +0.0938 [+0.0282, +0.1589] |
| R4 | 0.7571 | 0.6477 [0.6141, 0.6797] | +0.1094 [+0.0574, +0.1607] | 0.6530 | 0.5349 [0.4939, 0.5742] | +0.1181 [+0.0536, +0.1828] |

## Same position: positive vs negative videos (position cannot help)

| row | accident rows | rows 7 up to the accident | difference |
|---|---|---|---|
| LaGoVAD | 0.5881 [0.5339, 0.6419] | 0.5762 [0.5220, 0.6308] | +0.0118 [+0.0025, +0.0212] |
| W0 | 0.6590 [0.6068, 0.7106] | 0.6363 [0.5823, 0.6884] | +0.0227 [+0.0058, +0.0396] |
| R4 | 0.6871 [0.6355, 0.7379] | 0.6632 [0.6113, 0.7144] | +0.0238 [+0.0056, +0.0418] |

Accident rows, paired:

* W0 - LaGoVAD: +0.0709 [+0.0072, +0.1344]
* R4 - LaGoVAD: +0.0990 [+0.0318, +0.1656]
* R4 - W0: +0.0281 [+0.0001, +0.0557]
