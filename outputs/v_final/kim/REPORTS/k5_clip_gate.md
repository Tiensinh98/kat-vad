# K5 micro headroom and the §27 MSAD clip gate (`KIM_RUN.md` §19)

PRELIMINARY (n = 1: K5 s2024, one frozen gate). Macro is unchanged by construction.

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

| step 1 row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.6781 | 0.2568 | 0.6358 |
| min-max only | 0.7517 | 0.2594 | 0.6146 |
| gate only | 0.8599 | 0.2462 | 0.8512 |
| K5 x gate | 0.8692 | 0.3695 | 0.8512 |

- micro(K5 x gate) - micro(K5 raw): +0.1911 [+0.1149, +0.2693]
- micro(K5 x gate) - micro(min-max only): +0.1175 [+0.0712, +0.1675]
- micro(K5 x gate) - micro(gate only): +0.0093 [-0.0454, +0.0798]
- macro drift 0.0e+00

## MSAD — **gate in-domain (fitted on MSAD-train): printed, not read**

240 videos (121 normal); macro 0.6810; clip oracle 0.8943; length ruler 0.6466

| step 0: micro under | value |
|---|---:|
| K5 raw | 0.5386 |
| oracle per-video gate | 0.9238 |
| per-video min-max | 0.5555 |
| offset by K5's peak | 0.5093 |
| offset by K5's mean | 0.5188 |

Clip AUC of K5's own video statistics: peak 0.502, mean 0.525, range 0.503

| step 1 row | micro | AP | clip AUC |
|---|---:|---:|---:|
| K5 raw | 0.5386 | 0.2607 | 0.5020 |
| min-max only | 0.5555 | 0.2698 | 0.5027 |
| gate only | 0.8738 | 0.5990 | 0.9484 |
| K5 x gate | 0.8474 | 0.5776 | 0.9484 |

- micro(K5 x gate) - micro(K5 raw): +0.3088 [+0.2333, +0.3832]
- micro(K5 x gate) - micro(min-max only): +0.2919 [+0.2446, +0.3374]
- micro(K5 x gate) - micro(gate only): -0.0265 [-0.0498, -0.0029]
- macro drift 0.0e+00

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
