# DoTA train corpus EDA (`DoTA_train_window`, KIM_RUN.md §57)

Stride 3 (0.3 s / row) · windows 12 rows, hop 6 · dataset properties only, no model, nothing gates.

Gate: not run (no `--data-dir`); windows rebuilt from the annotations only.

## 1. Clips

|  | value |
|---|---|
| clips | 3020 |
| clips with no abnormal row | 2 |
| YouTube ids | 186 |
| span reaches the clip end (<= 1 row) | 0.105 |
| span starts at row 0 | 0.000 |
| normal rows before span < one window (12) | 0.420 |
| ego-involved | 0.608 |
| night | 0.121 |

- seconds: mean 10.187; p0 3.300 / p25 8.300 / **p50 9.900** / p75 11.700 / p100 30.000
- rows: mean 34.285; p0 11.000 / p25 28.000 / **p50 33.000** / p75 39.000 / p100 100.000
- span_rows: mean 11.081; p0 1.000 / p25 7.000 / **p50 10.000** / p75 14.000 / p100 64.000
- span_share: mean 0.325; p0 0.022 / p25 0.212 / **p50 0.300** / p75 0.410 / p100 0.938
- start_rel: mean 0.391; p0 0.000 / p25 0.298 / **p50 0.379** / p75 0.471 / p100 0.950
- end_rel: mean 0.716; p0 0.176 / p25 0.606 / **p50 0.714** / p75 0.828 / p100 1.000
- center_rel: mean 0.553; p0 0.118 / p25 0.468 / **p50 0.548** / p75 0.635 / p100 0.975
- pre_rows: mean 13.180; p0 0.000 / p25 9.250 / **p50 12.000** / p75 16.000 / p100 77.000
- post_rows: mean 10.028; p0 0.000 / p25 5.250 / **p50 9.000** / p75 13.000 / p100 77.000

Accident class (share of clips):

| class | share |
|---|---|
| turning | 0.380 |
| lateral | 0.152 |
| moving_ahead_or_waiting | 0.144 |
| oncoming | 0.105 |
| leave_to_left | 0.077 |
| leave_to_right | 0.072 |
| start_stop_or_stationary | 0.019 |
| pedestrian | 0.018 |
| obstacle | 0.017 |
| unknown | 0.016 |

## 2. Time profile of the abnormal rows

Row abnormal share: 0.323.

| relative bin | abnormal share |
|---|---|
| 0.0-0.1 | 0.006 |
| 0.1-0.2 | 0.051 |
| 0.2-0.3 | 0.185 |
| 0.3-0.4 | 0.430 |
| 0.4-0.5 | 0.648 |
| 0.5-0.6 | 0.703 |
| 0.6-0.7 | 0.579 |
| 0.7-0.8 | 0.362 |
| 0.8-0.9 | 0.187 |
| 0.9-1.0 | 0.086 |

| row (1 = last / first) | from the clip end | from the clip start |
|---|---|---|
| 1 | 0.082 | 0.000 |
| 2 | 0.105 | 0.004 |
| 3 | 0.125 | 0.009 |
| 4 | 0.152 | 0.018 |
| 5 | 0.182 | 0.031 |
| 6 | 0.229 | 0.055 |
| 7 | 0.277 | 0.084 |
| 8 | 0.335 | 0.125 |
| 9 | 0.394 | 0.168 |
| 10 | 0.459 | 0.246 |

No-pixel row rulers (AUC; 0.5 = nothing):

| micro late | micro hump | macro late | macro hump | two-class clips |
|---|---|---|---|---|
| 0.585 | 0.807 | 0.594 | 0.832 | 3018 |

## 3. Windows

12937 windows = 9386 positive + 3551 normal from 3019 clips; clips without a normal window 944.

- normal / clip: mean 1.176; p0 0.000 / p25 0.000 / **p50 1.000** / p75 2.000 / p100 12.000
- positive / clip: mean 3.109; p0 0.000 / p25 2.000 / **p50 3.000** / p75 4.000 / p100 10.000
- abnormal rows / positive window: mean 6.417; p0 1.000 / p25 4.000 / **p50 6.000** / p75 9.000 / p100 12.000; <= 1: 0.072, <= 2: 0.152, <= 3: 0.243, <= 6: 0.532; whole window: 0.129
- row profile inside a positive window: 0.442 0.470 0.489 0.509 0.529 0.547 0.560 0.568 0.571 0.575 0.579 0.577
- normal windows: 2500 before the span, 1044 after; gap rows mean 5.366; p0 0.000 / p25 1.000 / **p50 3.000** / p75 7.000 / p100 65.000; within 2 rows 0.420

V2-S padding: rows < 5 of a clip see a clamped (repeated) first frame.

|  | normal windows | positive windows |
|---|---|---|
| share holding padded rows | 0.493 | 0.135 |
| share touching the clip end | 0.042 | 0.036 |

No-pixel window rulers (AUC of the window label):

| centre raw | centre hump | raw within clip | hump within clip | padded rows | touches end |
|---|---|---|---|---|---|
| 0.677 | 0.790 | 0.760 | 0.859 | 0.321 | 0.497 |

Positive rate by stratum (windows):

| ego | normal | positive | clips | positive rate |
|---|---|---|---|---|
| ego | 2074 | 5491 | 1834 | 0.726 |
| other | 1477 | 3895 | 1185 | 0.725 |

| night | normal | positive | clips | positive rate |
|---|---|---|---|---|
| day | 3066 | 8257 | 2655 | 0.729 |
| night | 485 | 1129 | 364 | 0.700 |

| accident | normal | positive | clips | positive rate |
|---|---|---|---|---|
| turning | 1224 | 3307 | 1149 | 0.730 |
| lateral | 525 | 1512 | 459 | 0.742 |
| moving_ahead_or_waiting | 525 | 1333 | 434 | 0.717 |
| oncoming | 428 | 876 | 318 | 0.672 |
| leave_to_left | 288 | 907 | 232 | 0.759 |
| leave_to_right | 264 | 805 | 217 | 0.753 |
| start_stop_or_stationary | 84 | 168 | 56 | 0.667 |
| pedestrian | 90 | 158 | 54 | 0.637 |
| obstacle | 65 | 161 | 52 | 0.712 |
| unknown | 58 | 159 | 48 | 0.733 |

MIL k on a 12-row window: `L_MIL` k = 2 (pct 5), multi-class k = 1 (pct 16), bottom-k arms (§56) k = 2.

