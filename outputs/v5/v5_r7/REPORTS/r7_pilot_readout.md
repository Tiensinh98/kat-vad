# v5 R7 -- MQR on CLIP 14 x 14 patches, fast pilot read-out

Rule: core/docs/v5/PREREG_V5.md §4 (R7 fast pilot). Control R7m. Seeds s2024. DoTA-CAP-dev clips 569.

**PRELIMINARY (n = 1 seed)** -- a pilot read; it adopts nothing.


## Detection

Position alone (cubic in t / (n - 1)): macro 0.8496.

| row | macro | Δ vs control (95 %) | TOST ±0.01 | pos. (a) cross-clip | pos. (b) z(y)+2z(p) | gain beyond position | Δ (b) vs control | shift peak | nexar_val crop |
|---|---|---|---|---|---|---|---|---|---|
| R7m | 0.7804 | — | — | 0.6583 | 0.8656 | +0.0160 | — | -1 (ok) | not read |
| R7t | 0.8000 | +0.0197 [+0.0103, +0.0295] | different | 0.6368 | 0.8668 | +0.0172 | +0.0012 [-0.0040, +0.0062] | -1 (ok) | not read |

Time-reversed control (printed; reversed - forward should be ≤ 0): R7m +0.7294 [+0.7093, +0.7487] (-0.0510), R7t +0.7210 [+0.7009, +0.7401] (-0.0790)

AP (positive frame share 0.327): R7m clip-macro 0.6612 / pooled 0.5791, R7t clip-macro 0.6800 / pooled 0.5594; Δ clip-macro vs control R7t +0.0188 [+0.0082, +0.0299]

## Subsets (proposal §4.4; printed, 95 % cluster CI of Δ vs control)

| read | group | clips | R7m | R7t | Δ R7t |
|---|---|---|---|---|---|
| auc_by_ego | ego_involved | 321 | 0.8567 | 0.8672 | +0.0105 [+0.0021, +0.0191] |
| auc_by_ego | ego_uninvolved | 248 | 0.6815 | 0.7131 | +0.0316 [+0.0148, +0.0500] |
| pos_b_by_ego | ego_involved | 321 | 0.8859 | 0.8836 | -0.0023 [-0.0083, +0.0035] |
| pos_b_by_ego | ego_uninvolved | 248 | 0.8392 | 0.8450 | +0.0058 [-0.0025, +0.0142] |
| auc_by_box_share | 0.5-2% | 116 | 0.6937 | 0.7170 | +0.0232 [-0.0053, +0.0525] |
| auc_by_box_share | 2-8% | 169 | 0.7763 | 0.7944 | +0.0181 [+0.0023, +0.0343] |
| auc_by_box_share | <0.5% | 41 | 0.5992 | 0.6497 | +0.0505 [+0.0050, +0.0974] |
| auc_by_box_share | >8% | 208 | 0.8650 | 0.8784 | +0.0135 [+0.0018, +0.0249] |
| pos_b_by_box_share | 0.5-2% | 116 | 0.8302 | 0.8346 | +0.0044 [-0.0089, +0.0182] |
| pos_b_by_box_share | 2-8% | 169 | 0.8595 | 0.8592 | -0.0002 [-0.0076, +0.0071] |
| pos_b_by_box_share | <0.5% | 41 | 0.8092 | 0.8176 | +0.0084 [-0.0073, +0.0261] |
| pos_b_by_box_share | >8% | 208 | 0.8983 | 0.8988 | +0.0005 [-0.0068, +0.0079] |

## Maps on DoTA-CAP-dev boxes (proposal §4.4 H2)

| row | hit - chance | hit - centre | hit - change | mass - chance | mass - change | STAUC | uniform | centre | change |
|---|---|---|---|---|---|---|---|---|---|
| R7t | -0.0830 [-0.0959, -0.0693] | -0.2395 [-0.2806, -0.2000] | -0.3706 [-0.3945, -0.3456] | +0.0023 [-0.0004, +0.0053] | -0.0528 [-0.0571, -0.0482] | 0.1964 | 0.2049 | 0.2691 | nan |

## Mechanics (last step; printed)

| run | finite | mil | tile | p abnormal | p normal | T | entropy / log 196 | motion share | CLIP share (R7) |
|---|---|---|---|---|---|---|---|---|---|
| R7m/s2024 | yes | 0.1346 | — | — | — | — | — | 0.1537 | 0.4382 |
| R7t/s2024 | yes | 0.0768 | 0.1690 | 0.1171 | 0.1158 | 1.0165 | 0.6307 | 0.1462 | 0.3758 |

## Pilot gate (§4.2)

| row | maps: R7 rule (§4.2 P-e) | nexar guard | pos. (b) not below | outcome |
|---|---|---|---|---|
| R7t | no | not read | yes | **NULL-PILOT (maps) -- nexar_val guardrail not read** |
