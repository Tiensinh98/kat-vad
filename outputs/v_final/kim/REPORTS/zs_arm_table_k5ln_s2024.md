# k5ln_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5 | 0.8195 / 0.7449 | 0.7740 / 0.7635 | 0.7002 / 0.6781 | 0.7916 / 0.7050 |
| K5ln | 0.8175 / 0.7422 | 0.7767 / 0.7668 | 0.6954 / 0.7070 | 0.7718 / 0.6980 |
| W0 | 0.8110 / 0.7255 | 0.7615 / 0.7463 | 0.6983 / 0.6034 | 0.7317 / 0.6693 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5 | 0.8195 [0.8048, 0.8338] | 0.7449 | 0.5790 | 0.3271 |
| K5ln | 0.8175 [0.8028, 0.8316] | 0.7422 | 0.5769 | 0.3271 |
| W0 | 0.8110 [0.7948, 0.8269] | 0.7255 | 0.5562 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5ln - K5 | -0.0020 [-0.0058, +0.0020] | not separated | -0.0027 [-0.0074, +0.0022] | not separated |
| W0 - K5 | -0.0085 [-0.0190, +0.0023] | not separated | -0.0194 [-0.0297, -0.0097] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5 | 0.7740 [0.7492, 0.7982] | 0.7635 | 0.5640 | 0.2885 |
| K5ln | 0.7767 [0.7522, 0.8004] | 0.7668 | 0.5657 | 0.2885 |
| W0 | 0.7615 [0.7369, 0.7859] | 0.7463 | 0.5424 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5ln - K5 | +0.0027 [-0.0030, +0.0086] | not separated | +0.0034 [-0.0019, +0.0087] | not separated |
| W0 - K5 | -0.0125 [-0.0253, +0.0006] | not separated | -0.0172 [-0.0278, -0.0065] | below |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5 | 0.7002 [0.6246, 0.7724] | 0.6781 | 0.2568 | 0.0724 |
| K5ln | 0.6954 [0.6188, 0.7688] | 0.7070 | 0.2742 | 0.0724 |
| W0 | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.2117 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5ln - K5 | -0.0048 [-0.0227, +0.0126] | not separated | +0.0289 [+0.0073, +0.0547] | above |
| W0 - K5 | -0.0019 [-0.0286, +0.0244] | not separated | -0.0747 [-0.1172, -0.0302] | below |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5 | 0.7916 [0.7513, 0.8294] | 0.7050 | 0.1156 | 0.0472 |
| K5ln | 0.7718 [0.7305, 0.8110] | 0.6980 | 0.1107 | 0.0472 |
| W0 | 0.7317 [0.6897, 0.7716] | 0.6693 | 0.1102 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5ln - K5 | -0.0198 [-0.0359, -0.0054] | below | -0.0070 [-0.0141, +0.0002] | not separated |
| W0 - K5 | -0.0599 [-0.0918, -0.0287] | below | -0.0357 [-0.0508, -0.0207] | below |
