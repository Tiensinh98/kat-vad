# KIM pilot KP1-KP4 (host K0 s2024, step 1749, no training)

Rule: `core/docs/final/KIM_RUN.md §5`. **The KP pilot reads DoTA-CAP-dev / DADA / TAD with the decided control (K0 or K0k9), after the kernel read-out and before K2 / K5 train (KIM_RUN.md §5.1).**

## dota_cap_dev (569 two-class of 569; t/N ruler 0.5573 [0.5383, 0.5764])

| candidate | macro | ego | non-ego |
| --- | --- | --- | --- |
| vt1 | 0.7337 [0.7199, 0.7476] | 0.7713 [0.7556, 0.7864] | 0.6852 [0.6630, 0.7051] |
| vt2 | 0.7597 [0.7454, 0.7737] | 0.7980 [0.7824, 0.8133] | 0.7101 [0.6871, 0.7315] |
| x1 | 0.7349 [0.7195, 0.7501] | 0.7774 [0.7583, 0.7952] | 0.6800 [0.6553, 0.7027] |
| x2 | 0.7671 [0.7530, 0.7810] | 0.8072 [0.7896, 0.8240] | 0.7151 [0.6922, 0.7366] |
| clip1 | 0.7420 [0.7264, 0.7573] | 0.7823 [0.7635, 0.7999] | 0.6898 [0.6647, 0.7128] |
| clip2 | 0.7731 [0.7588, 0.7875] | 0.8104 [0.7923, 0.8274] | 0.7249 [0.7011, 0.7464] |
| motion1 | 0.5729 [0.5556, 0.5901] | 0.6029 [0.5812, 0.6256] | 0.5341 [0.5073, 0.5588] |
| motion2 | 0.5766 [0.5591, 0.5940] | 0.6027 [0.5812, 0.6252] | 0.5429 [0.5169, 0.5673] |
| host | 0.8002 [0.7847, 0.8157] | 0.8537 [0.8367, 0.8705] | 0.7310 [0.7035, 0.7565] |
| host+vt2 (rank avg) | 0.8139 [0.7998, 0.8283] | 0.8633 [0.8491, 0.8776] | 0.7500 [0.7242, 0.7741] |

KP2 (rank avg - host, per video): +0.0137 [+0.0061, +0.0211]
KP3 edge share of top-5 % vt1 steps: 0.130 (chance 0.066, 569 videos)
KP4 vt2 peak - onset: median |lag| 1.80 s, 0.13 of 569 clips within ±0.5 s

## dada (370 two-class of 370; t/N ruler 0.5514 [0.5238, 0.5779])

| candidate | macro |
| --- | --- |
| vt1 | 0.6743 [0.6569, 0.6914] |
| vt2 | 0.7026 [0.6846, 0.7206] |
| x1 | 0.6702 [0.6531, 0.6877] |
| x2 | 0.7029 [0.6844, 0.7213] |
| clip1 | 0.6763 [0.6588, 0.6938] |
| clip2 | 0.7075 [0.6887, 0.7261] |
| motion1 | 0.5552 [0.5363, 0.5739] |
| motion2 | 0.5566 [0.5365, 0.5765] |
| host | 0.7256 [0.7033, 0.7474] |
| host+vt2 (rank avg) | 0.7428 [0.7214, 0.7635] |

KP2 (rank avg - host, per video): +0.0172 [+0.0081, +0.0262]
KP3 edge share of top-5 % vt1 steps: 0.156 (chance 0.053, 370 videos)
KP4 vt2 peak - onset: median |lag| 2.13 s, 0.07 of 370 clips within ±0.5 s

## tad (60 two-class of 100; t/N ruler 0.5872 [0.5126, 0.6611])

| candidate | macro |
| --- | --- |
| vt1 | 0.6664 [0.6220, 0.7077] |
| vt2 | 0.6903 [0.6462, 0.7323] |
| x1 | 0.6738 [0.6278, 0.7174] |
| x2 | 0.6959 [0.6508, 0.7384] |
| clip1 | 0.6758 [0.6290, 0.7208] |
| clip2 | 0.6927 [0.6475, 0.7366] |
| motion1 | 0.5511 [0.5045, 0.5987] |
| motion2 | 0.5713 [0.5230, 0.6199] |
| host | 0.6954 [0.6314, 0.7566] |
| host+vt2 (rank avg) | 0.7244 [0.6723, 0.7751] |

KP2 (rank avg - host, per video): +0.0290 [-0.0001, +0.0598]
KP3 edge share of top-5 % vt1 steps: 0.187 (chance 0.072, 99 videos)

## Decisions (fixed in §5 before the pilot)

- **KP1**: {'vt2_ego_macro': 0.798040750055447, 'verdict': 'pass'}
- **KP2**: {'delta': {'mean': 0.013713022196359044, 'low': 0.006082199367968008, 'high': 0.021133022550961122, 'clips': 569, 'clusters': 91}, 'verdict': 'pass'}
- **input**: vt
- **KP3**: {'verdict': 'fail', 'consequence': 'mask the last valid pair (KIM_EDGE_MASK)'}
- **KP4**: {'verdict': 'fail', 'consequence': 'H6 dropped from the claims'}
