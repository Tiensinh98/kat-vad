# DAD position diagnostics (POST-HOC, printed not gated, n = 1 seed)

165 positives, 301 negatives, 15 rows each; accident rows [13, 14]; first V2-S window without a repeated frame = row 7. **No-pixel `t/N` ruler macro = 1.0000.**

## Within video: positives vs the same mask on negatives (placebo)

| row | macro, all rows | placebo, all rows | gap | macro, rows 7+ | placebo, rows 7+ | gap |
|---|---|---|---|---|---|---|
| K0dfiT2 | 0.8242 | 0.6808 [0.6430, 0.7172] | +0.1434 [+0.0909, +0.1955] | 0.7520 | 0.6329 [0.5919, 0.6714] | +0.1191 [+0.0545, +0.1831] |
| K5dfiT2rr | 0.8107 | 0.7177 [0.6850, 0.7493] | +0.0930 [+0.0453, +0.1402] | 0.6859 | 0.6069 [0.5653, 0.6465] | +0.0790 [+0.0132, +0.1447] |
| K5dfiT2 | 0.8107 | 0.7177 [0.6850, 0.7493] | +0.0930 [+0.0453, +0.1402] | 0.6859 | 0.6069 [0.5653, 0.6465] | +0.0790 [+0.0132, +0.1447] |

## Same position: positive vs negative videos (position cannot help)

| row | accident rows | rows 7 up to the accident | difference |
|---|---|---|---|
| K0dfiT2 | 0.7494 [0.6997, 0.7954] | 0.6938 [0.6422, 0.7418] | +0.0556 [+0.0248, +0.0868] |
| K5dfiT2rr | 0.7256 [0.6750, 0.7729] | 0.6904 [0.6381, 0.7392] | +0.0351 [+0.0136, +0.0576] |
| K5dfiT2 | 0.7256 [0.6750, 0.7729] | 0.6904 [0.6381, 0.7392] | +0.0351 [+0.0136, +0.0576] |

Accident rows, paired:

* K5dfiT2rr - K0dfiT2: -0.0238 [-0.0448, -0.0032]
* K5dfiT2 - K0dfiT2: -0.0238 [-0.0448, -0.0032]
