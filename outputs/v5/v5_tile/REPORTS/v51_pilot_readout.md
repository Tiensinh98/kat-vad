# v5.1 tile-scored MQR -- pilot read-out

Rule: core/docs/v5/PREREG_V5.md §3. Control R1c. Seeds s2024. DoTA-CAP-dev clips 569.

**PRELIMINARY (n = 1 seed)** -- a pilot read; it adopts nothing.


## Detection

Position alone (cubic in t / (n - 1)): macro 0.8496.

| row | macro | Δ vs control (95 %) | TOST ±0.01 | pos. (a) cross-clip | pos. (b) z(y)+2z(p) | gain beyond position | Δ (b) vs control | shift peak | nexar_val crop |
|---|---|---|---|---|---|---|---|---|---|
| R1c | 0.8124 | — | — | 0.6400 | 0.8755 | +0.0260 | — | +0 (ok) | 0.8985 |
| R3c | 0.7505 | -0.0618 [-0.0776, -0.0467] | different | 0.6477 | 0.8636 | +0.0140 | -0.0120 [-0.0176, -0.0063] | -1 (ok) | 0.7963 |
| R6c | 0.8047 | -0.0076 [-0.0168, +0.0017] | inconclusive | 0.6580 | 0.8779 | +0.0284 | +0.0024 [-0.0019, +0.0067] | +0 (ok) | 0.8746 |

Time-reversed control (printed; reversed - forward should be ≤ 0): R1c +0.7766 [+0.7558, +0.7966] (-0.0358), R3c +0.7261 [+0.7068, +0.7446] (-0.0245), R6c +0.7807 [+0.7600, +0.8011] (-0.0240)

AP (positive frame share 0.327): R1c clip-macro 0.7137 / pooled 0.5809, R3c clip-macro 0.6459 / pooled 0.5608, R6c clip-macro 0.7123 / pooled 0.5926; Δ clip-macro vs control R3c -0.0678 [-0.0876, -0.0482], R6c -0.0015 [-0.0127, +0.0099]

## Subsets (proposal §4.4; printed, 95 % cluster CI of Δ vs control)

| read | group | clips | R1c | R3c | R6c | Δ R3c | Δ R6c |
|---|---|---|---|---|---|---|---|
| auc_by_ego | ego_involved | 321 | 0.8831 | 0.8260 | 0.8740 | -0.0570 [-0.0745, -0.0396] | -0.0090 [-0.0193, +0.0007] |
| auc_by_ego | ego_uninvolved | 248 | 0.7208 | 0.6528 | 0.7151 | -0.0680 [-0.0966, -0.0413] | -0.0058 [-0.0246, +0.0119] |
| pos_b_by_ego | ego_involved | 321 | 0.8847 | 0.8828 | 0.8875 | -0.0019 [-0.0099, +0.0062] | +0.0028 [-0.0016, +0.0073] |
| pos_b_by_ego | ego_uninvolved | 248 | 0.8637 | 0.8387 | 0.8656 | -0.0249 [-0.0362, -0.0139] | +0.0019 [-0.0067, +0.0103] |
| auc_by_box_share | 0.5-2% | 116 | 0.7278 | 0.6421 | 0.7168 | -0.0857 [-0.1214, -0.0517] | -0.0109 [-0.0383, +0.0147] |
| auc_by_box_share | 2-8% | 169 | 0.8173 | 0.7764 | 0.8116 | -0.0408 [-0.0690, -0.0134] | -0.0056 [-0.0257, +0.0157] |
| auc_by_box_share | <0.5% | 41 | 0.6148 | 0.5645 | 0.6233 | -0.0502 [-0.1319, +0.0209] | +0.0086 [-0.0336, +0.0479] |
| auc_by_box_share | >8% | 208 | 0.8940 | 0.8303 | 0.8843 | -0.0637 [-0.0904, -0.0385] | -0.0097 [-0.0205, +0.0012] |
| pos_b_by_box_share | 0.5-2% | 116 | 0.8499 | 0.8244 | 0.8532 | -0.0255 [-0.0417, -0.0088] | +0.0034 [-0.0075, +0.0147] |
| pos_b_by_box_share | 2-8% | 169 | 0.8727 | 0.8681 | 0.8693 | -0.0046 [-0.0149, +0.0051] | -0.0034 [-0.0141, +0.0076] |
| pos_b_by_box_share | <0.5% | 41 | 0.8260 | 0.7995 | 0.8359 | -0.0265 [-0.0625, +0.0048] | +0.0099 [-0.0097, +0.0277] |
| pos_b_by_box_share | >8% | 208 | 0.9018 | 0.8943 | 0.9053 | -0.0075 [-0.0184, +0.0028] | +0.0035 [-0.0020, +0.0090] |

## Maps on DoTA-CAP-dev boxes (proposal §4.4 H2)

| row | hit - chance | hit - centre | hit - change | mass - chance | mass - change | STAUC | uniform | centre | change |
|---|---|---|---|---|---|---|---|---|---|
| R3c | +0.0033 [-0.0155, +0.0214] | -0.2993 [-0.3414, -0.2592] | +0.0000 [+0.0000, +0.0000] | +0.0035 [+0.0026, +0.0043] | +0.0000 [+0.0000, +0.0000] | 0.2127 | 0.2014 | 0.2642 | 0.2127 |
| R6c | -0.0427 [-0.0533, -0.0316] | -0.3453 [-0.3807, -0.3119] | -0.0460 [-0.0673, -0.0230] | +0.0061 [+0.0040, +0.0083] | +0.0026 [+0.0004, +0.0050] | 0.2110 | 0.2009 | 0.2641 | 0.2109 |

## Mechanics (last step; printed)

| run | finite | mil | tile | p abnormal | p normal | T | entropy / log 196 | motion share |
|---|---|---|---|---|---|---|---|---|
| R1c/s2024 | yes | 0.1126 | — | — | — | — | — | 0.1432 |
| R3c/s2024 | yes | 0.2085 | — | — | — | — | 0.2626 | 0.2589 |
| R6c/s2024 | yes | 0.1315 | 0.1818 | 0.1220 | 0.1258 | 1.0153 | 0.6979 | 0.1906 |

## Pilot gate (§3)

| row | maps beat centre + change | nexar guard | pos. (b) not below | outcome |
|---|---|---|---|---|
| R3c | no | no | no | **PILOT-FAIL (maps_beat_centre_and_change, nexar_guard, position_controlled_not_below)** |
| R6c | no | no | yes | **PILOT-FAIL (maps_beat_centre_and_change, nexar_guard)** |
