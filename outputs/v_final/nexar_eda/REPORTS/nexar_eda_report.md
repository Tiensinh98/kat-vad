# Nexar training corpus EDA

`/content/nexar_eda/data` · stride 8 · dataset properties only, no model.
Gate: the positive rows rebuilt from `normalized_span` equal `meta.json` on all 7311 train windows.

## 1. Counts

- train: **7311** windows = 1689 positive + 5622 normal, from 451 sources (5622 normal windows come from abnormal sources)
- sources without a positive window: 0; without a normal window: 0
- rows per window: mean 20.000; p0 20.000 / p25 20.000 / **p50 20.000** / p75 20.000 / p100 20.000
- normal windows / source: mean 12.466; p0 2.000 / p25 13.000 / **p50 13.000** / p75 14.000 / p100 23.000
- positive windows / source: mean 3.745; p0 2.000 / p25 3.000 / **p50 4.000** / p75 4.000 / p100 5.000
- val: 1788 windows = 434 two-class + 1354 all-normal + 0 all-abnormal

## 2. Positive windows

- abnormal rows / positive window: mean 7.451; p0 1.000 / p25 4.000 / **p50 8.000** / p75 10.000 / p100 20.000
- share with at most k abnormal rows: k=1: 0.066, k=3: 0.193, k=5: 0.329, k=10: 0.782
- whole window abnormal: 0.001
- first abnormal row: mean 6.205; p0 0.000 / p25 0.000 / **p50 4.000** / p75 12.000 / p100 19.000
- last abnormal row: mean 12.655; p0 0.000 / p25 7.000 / **p50 15.000** / p75 19.000 / p100 19.000

Share of positive windows abnormal at each row index (train | val):

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 0.380 | 0.370 | 0.370 | 0.360 | 0.370 | 0.370 | 0.370 | 0.380 | 0.380 | 0.370 | 0.370 | 0.360 | 0.370 | 0.380 | 0.370 | 0.380 | 0.380 | 0.370 | 0.370 | 0.360 |
| val | 0.360 | 0.360 | 0.360 | 0.370 | 0.370 | 0.370 | 0.370 | 0.370 | 0.360 | 0.360 | 0.360 | 0.370 | 0.370 | 0.370 | 0.370 | 0.370 | 0.360 | 0.360 | 0.360 | 0.370 |

## 3. Normal windows against their own span

- with a span in their source: 5622 (before the span 2868, after 2754)
- rows between the window and the span: mean 25.980; p0 0.000 / p25 12.000 / **p50 25.000** / p75 39.000 / p100 161.000
- within 2 rows of the span: 0.063

## 4. What no-pixel properties score (AUC over train windows; 0.5 = nothing)

- source length (rows): 0.461
- window center, raw: 0.510; hump `-|c - 0.5|`: 0.973
- hump inside one source (mean over 451 sources with both classes): 0.979

## 5. Scene and weather

| scene | normal | positive | sources | share of normal | share of positive | positive rate |
|---|---|---|---|---|---|---|
| Urban | 3751 | 1068 | 286 | 0.667 | 0.632 | 0.222 |
| Highway | 1200 | 410 | 110 | 0.213 | 0.243 | 0.255 |
| Sub-urban | 509 | 157 | 41 | 0.091 | 0.093 | 0.236 |
| Other | 147 | 46 | 12 | 0.026 | 0.027 | 0.238 |
| Rural | 12 | 4 | 1 | 0.002 | 0.002 | 0.250 |
| Nature | 3 | 4 | 1 | 0.001 | 0.002 | 0.571 |

| weather | normal | positive | sources | share of normal | share of positive | positive rate |
|---|---|---|---|---|---|---|
| Clear | 3317 | 988 | 265 | 0.590 | 0.585 | 0.230 |
| Cloudy | 1761 | 537 | 143 | 0.313 | 0.318 | 0.234 |
| Rain | 541 | 160 | 42 | 0.096 | 0.095 | 0.228 |
| Snow | 3 | 4 | 1 | 0.001 | 0.002 | 0.571 |

## 6. Val windows (where every Nexar number is read)

- frames: 35760; clips by type {'all_normal': 1354, 'all_positive': 0, 'mixed': 434}
- constant-score-per-window oracle: micro AUC 0.916, AP 0.366 (macro 0.5 by construction)
- abnormal rows / positive window: mean 7.325; p0 1.000 / p25 4.000 / **p50 8.000** / p75 10.000 / p100 17.000
