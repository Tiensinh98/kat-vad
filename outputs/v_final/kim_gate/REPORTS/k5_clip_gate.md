# K5 micro headroom and clip gates (`KIM_RUN.md` §19 / §20)

PRELIMINARY (n = 1: K5 s2024, one deterministic fit per gate). Macro is unchanged by construction.

## nexar_val

225 videos (113 normal); macro 0.9245; clip oracle 0.7611; length ruler 0.5251

| step 0: micro under | value |
|---|---:|
| K5 raw | 0.9000 |
| oracle per-video gate | 0.9578 |
| per-video min-max | 0.8968 |
| offset by K5's peak | 0.8102 |
| offset by K5's mean | 0.8255 |

Clip AUC of K5's own video statistics: peak 0.774, mean 0.675, range 0.773

### gate NG — **below**

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.9000 | 0.4472 | 0.7735 |
| min-max only | 0.8968 | 0.4120 | 0.7735 |
| gate only | 0.5869 | 0.0519 | 0.6768 |
| K5 x gate | 0.8571 | 0.3794 | 0.6768 |

- micro(K5 x gate) - micro(K5 raw): -0.0429 [-0.0728, -0.0161]
- micro(K5 x gate) - micro(min-max only): -0.0397 [-0.0691, -0.0132]
- micro(K5 x gate) - micro(gate only): +0.2702 [+0.2380, +0.3035]
- macro drift 0.0e+00

### gate MS — **below**

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.9000 | 0.4472 | 0.7735 |
| min-max only | 0.8968 | 0.4120 | 0.7735 |
| gate only | 0.3784 | 0.0296 | 0.2922 |
| K5 x gate | 0.8842 | 0.3382 | 0.2922 |

- micro(K5 x gate) - micro(K5 raw): -0.0159 [-0.0293, -0.0014]
- micro(K5 x gate) - micro(min-max only): -0.0126 [-0.0217, -0.0052]
- micro(K5 x gate) - micro(gate only): +0.5057 [+0.4604, +0.5493]
- macro drift 0.0e+00

- micro(K5 x MS) - micro(K5 x NG): +0.0270 [-0.0017, +0.0581] (printed)

### clip AUC within strata (printed, never gating)

- light_conditions = Bright (1 / 1 abnormal): one class only
- light_conditions = Dark (3 / 9 abnormal): K5 peak 0.778, NG 0.611, MS 0.444
- light_conditions = Normal (103 / 204 abnormal): K5 peak 0.787, NG 0.674, MS 0.269
- light_conditions = Twilight (5 / 11 abnormal): K5 peak 0.533, NG 0.667, MS 0.533
- scene = Highway (24 / 60 abnormal): K5 peak 0.817, NG 0.617, MS 0.343
- scene = Industrial (1 / 3 abnormal): K5 peak 0.500, NG 0.000, MS 0.000
- scene = Other (1 / 4 abnormal): K5 peak 0.667, NG 0.000, MS 0.667
- scene = Rural (1 / 3 abnormal): K5 peak 1.000, NG 1.000, MS 1.000
- scene = Sub-urban (10 / 39 abnormal): K5 peak 0.697, NG 0.483, MS 0.238
- scene = Urban (75 / 116 abnormal): K5 peak 0.771, NG 0.708, MS 0.308
- time_to_accident =  (112 / 225 abnormal): K5 peak 0.774, NG 0.677, MS 0.292
- weather = Clear (66 / 146 abnormal): K5 peak 0.770, NG 0.674, MS 0.327
- weather = Cloudy (39 / 69 abnormal): K5 peak 0.797, NG 0.669, MS 0.240
- weather = Rain (7 / 10 abnormal): K5 peak 0.619, NG 0.571, MS 0.381

## TAD

100 videos (40 normal); macro 0.7002; clip oracle 0.9226; length ruler 0.8968

| step 0: micro under | value |
|---|---:|
| K5 raw | 0.6781 |
| oracle per-video gate | 0.9394 |
| per-video min-max | 0.7517 |
| offset by K5's peak | 0.5551 |
| offset by K5's mean | 0.6628 |

Clip AUC of K5's own video statistics: peak 0.636, mean 0.687, range 0.615

### gate NG — **not separated**

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.6781 | 0.2568 | 0.6358 |
| min-max only | 0.7517 | 0.2594 | 0.6146 |
| gate only | 0.3497 | 0.0520 | 0.2833 |
| K5 x gate | 0.7185 | 0.2357 | 0.2833 |

- micro(K5 x gate) - micro(K5 raw): +0.0404 [-0.0311, +0.1116]
- micro(K5 x gate) - micro(min-max only): -0.0332 [-0.0701, -0.0029]
- micro(K5 x gate) - micro(gate only): +0.3688 [+0.1882, +0.5453]
- macro drift 0.0e+00

### gate MS — **above**

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.6781 | 0.2568 | 0.6358 |
| min-max only | 0.7517 | 0.2594 | 0.6146 |
| gate only | 0.8599 | 0.2462 | 0.8512 |
| K5 x gate | 0.8692 | 0.3695 | 0.8512 |

- micro(K5 x gate) - micro(K5 raw): +0.1911 [+0.1149, +0.2693]
- micro(K5 x gate) - micro(min-max only): +0.1175 [+0.0712, +0.1675]
- micro(K5 x gate) - micro(gate only): +0.0093 [-0.0454, +0.0798]
- macro drift 0.0e+00

- micro(K5 x MS) - micro(K5 x NG): +0.1507 [+0.0931, +0.2119] (printed)

## DAD

466 videos (301 normal); macro 0.7916; clip oracle 0.8390; length ruler 0.5000

| step 0: micro under | value |
|---|---:|
| K5 raw | 0.7050 |
| oracle per-video gate | 0.8830 |
| per-video min-max | 0.7758 |
| offset by K5's peak | 0.6251 |
| offset by K5's mean | 0.6334 |

Clip AUC of K5's own video statistics: peak 0.652, mean 0.647, range 0.647

### gate NG — **not separated**

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.7050 | 0.1156 | 0.6523 |
| min-max only | 0.7758 | 0.1578 | 0.6471 |
| gate only | 0.5168 | 0.0496 | 0.5248 |
| K5 x gate | 0.6973 | 0.1275 | 0.5248 |

- micro(K5 x gate) - micro(K5 raw): -0.0077 [-0.0526, +0.0358]
- micro(K5 x gate) - micro(min-max only): -0.0785 [-0.1163, -0.0431]
- micro(K5 x gate) - micro(gate only): +0.1805 [+0.1489, +0.2121]
- macro drift 0.0e+00

### gate MS — **not separated**

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.7050 | 0.1156 | 0.6523 |
| min-max only | 0.7758 | 0.1578 | 0.6471 |
| gate only | 0.5088 | 0.0474 | 0.5130 |
| K5 x gate | 0.6832 | 0.1038 | 0.5130 |

- micro(K5 x gate) - micro(K5 raw): -0.0218 [-0.0626, +0.0196]
- micro(K5 x gate) - micro(min-max only): -0.0926 [-0.1256, -0.0587]
- micro(K5 x gate) - micro(gate only): +0.1744 [+0.1498, +0.1985]
- macro drift 0.0e+00

- micro(K5 x MS) - micro(K5 x NG): -0.0141 [-0.0566, +0.0295] (printed)

## MSAD

240 videos (121 normal); macro 0.6810; clip oracle 0.8943; length ruler 0.6466

| step 0: micro under | value |
|---|---:|
| K5 raw | 0.5386 |
| oracle per-video gate | 0.9238 |
| per-video min-max | 0.5555 |
| offset by K5's peak | 0.5093 |
| offset by K5's mean | 0.5188 |

Clip AUC of K5's own video statistics: peak 0.502, mean 0.525, range 0.503

### gate NG — **not separated**

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.5386 | 0.2607 | 0.5020 |
| min-max only | 0.5555 | 0.2698 | 0.5027 |
| gate only | 0.3721 | 0.1859 | 0.3035 |
| K5 x gate | 0.4641 | 0.2285 | 0.3040 |

- micro(K5 x gate) - micro(K5 raw): -0.0745 [-0.1714, +0.0197]
- micro(K5 x gate) - micro(min-max only): -0.0914 [-0.1519, -0.0376]
- micro(K5 x gate) - micro(gate only): +0.0920 [+0.0428, +0.1416]
- macro drift 0.0e+00

### gate MS — in-domain for this gate: printed, no sentence

| row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.5386 | 0.2607 | 0.5020 |
| min-max only | 0.5555 | 0.2698 | 0.5027 |
| gate only | 0.8738 | 0.5990 | 0.9484 |
| K5 x gate | 0.8474 | 0.5776 | 0.9484 |

- micro(K5 x gate) - micro(K5 raw): +0.3088 [+0.2333, +0.3832]
- micro(K5 x gate) - micro(min-max only): +0.2919 [+0.2446, +0.3374]
- micro(K5 x gate) - micro(gate only): -0.0265 [-0.0498, -0.0029]
- macro drift 0.0e+00

- micro(K5 x MS) - micro(K5 x NG): +0.3833 [+0.3212, +0.4468] (printed)
