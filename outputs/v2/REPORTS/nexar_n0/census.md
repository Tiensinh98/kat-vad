# Nexar N0 census

- videos 1500 (per label {'0': 750, '1': 750}), probed 1500, clean positives 750
- metadata columns: ['file_name', 'light_conditions', 'scene', 'time_of_alert', 'time_of_event', 'time_to_accident', 'weather']
- issues: none
- header frames total 1701166; header vs duration x fps mismatched 0
- decode check: {'checked': 20, 'max_abs_diff': 0}
- length ruler (duration -> label): {'auc': 0.5366, 'separability': 0.5366}

| quantity | label | p0 | p5 | p25 | p50 | p75 | p95 | p100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| duration_s | 0 | 15.000 | 18.000 | 39.333 | 40.100 | 40.336 | 41.020 | 60.000 |
| duration_s | 1 | 14.000 | 18.000 | 39.733 | 40.133 | 40.361 | 41.037 | 60.000 |
| t_event / duration | 1 | 0.097 | 0.462 | 0.482 | 0.496 | 0.511 | 0.555 | 0.947 |
| time_of_event_s | 1 | 3.032 | 10.111 | 19.133 | 19.802 | 20.333 | 21.385 | 56.800 |
| time_of_alert_s | 1 | 1.966 | 8.531 | 17.290 | 18.259 | 18.977 | 20.067 | 55.467 |
| alert -> event (s) | 1 | 0.033 | 0.401 | 0.986 | 1.433 | 2.090 | 3.283 | 4.466 |
| share [alert, event+0s] | 1 | 0.001 | 0.011 | 0.025 | 0.037 | 0.056 | 0.100 | 0.191 |
| share [alert, event+1s] | 1 | 0.024 | 0.036 | 0.050 | 0.062 | 0.082 | 0.130 | 0.246 |
| share [alert, event+2s] | 1 | 0.048 | 0.059 | 0.075 | 0.088 | 0.108 | 0.178 | 0.302 |

Corpus abnormal frame share by post-event extension: +0s 0.0212, +1s 0.0344, +2s 0.0477

**t_event / duration histogram (positives)**

| bin | n |
|---|---:|
| [0,0.1) | 1 |
| [0.1,0.2) | 1 |
| [0.2,0.3) | 1 |
| [0.3,0.4) | 5 |
| [0.4,0.5) | 418 |
| [0.5,0.6) | 312 |
| [0.6,0.7) | 8 |
| [0.7,0.8) | 1 |
| [0.8,0.9) | 2 |
| [0.9,1) | 1 |

**fps**

| value | label 0 | label 1 |
|---|---:|---:|
| 23.6 | 1 | 0 |
| 23.9 | 1 | 0 |
| 24.1 | 1 | 0 |
| 24.6 | 1 | 0 |
| 25.3 | 0 | 1 |
| 25.6 | 1 | 0 |
| 26.4 | 1 | 1 |
| 26.7 | 0 | 1 |
| 27.1 | 2 | 0 |
| 27.3 | 1 | 0 |
| 27.7 | 0 | 1 |
| 28.1 | 0 | 1 |
| 28.3 | 0 | 1 |
| 28.4 | 1 | 0 |
| 28.5 | 0 | 1 |
| 28.6 | 1 | 0 |
| 28.7 | 3 | 0 |
| 28.9 | 7 | 3 |
| 29.0 | 5 | 3 |
| 29.1 | 6 | 3 |
| 29.2 | 8 | 7 |
| 29.3 | 6 | 5 |
| 29.4 | 7 | 3 |
| 29.5 | 2 | 3 |
| 29.6 | 7 | 1 |
| 29.7 | 6 | 5 |
| 29.8 | 12 | 10 |
| 29.9 | 53 | 50 |
| 30.0 | 442 | 450 |
| 30.1 | 1 | 2 |
| 30.2 | 2 | 1 |
| 30.3 | 1 | 6 |
| 30.4 | 1 | 9 |
| 30.5 | 46 | 40 |
| 30.6 | 124 | 132 |
| 31.0 | 0 | 10 |

**resolution**

| value | label 0 | label 1 |
|---|---:|---:|
| 1280x720 | 750 | 750 |

**codec**

| value | label 0 | label 1 |
|---|---:|---:|
| h264 | 750 | 750 |

**meta_light_conditions**

| value | label 0 | label 1 |
|---|---:|---:|
| Bright | 3 | 7 |
| Dark | 25 | 22 |
| Normal | 675 | 680 |
| Twilight | 47 | 41 |

**meta_scene**

| value | label 0 | label 1 |
|---|---:|---:|
| Highway | 208 | 171 |
| Industrial | 10 | 3 |
| Nature | 0 | 1 |
| Other | 12 | 22 |
| Rural | 15 | 3 |
| Sub-urban | 196 | 75 |
| Urban | 309 | 475 |

**meta_time_to_accident**

| value | label 0 | label 1 |
|---|---:|---:|
|  | 750 | 750 |

**meta_weather**

| value | label 0 | label 1 |
|---|---:|---:|
|  | 1 | 1 |
| Clear | 474 | 445 |
| Cloudy | 252 | 243 |
| Rain | 23 | 60 |
| Snow | 0 | 1 |

