# Nexar pilot summary (seeds [2099]; both arms dvs_anchor_mode=ignore, §28)

val middle ruler (whole videos, position alone): 0.9735

| construction | arm | crops mean (endpoint) | crop @0.1 | @0.5 | @0.9 | whole macro | whole micro | clip AUC | DoTA-CAP-dev (2nd look) |
|---|---|---|---|---|---|---|---|---|---|
| whole | A0 | **0.5876 [0.5416, 0.6328]** | 0.5641 [0.4993, 0.6278] | 0.5773 [0.5314, 0.6228] | 0.6378 [0.5802, 0.6929] | 0.5841 [0.5406, 0.6276] | 0.6153 | 0.6515 | 0.5680 [0.5463, 0.5899] |
| whole | A3 | **0.5819 [0.5347, 0.6279]** | 0.6915 [0.6295, 0.7514] | 0.5802 [0.5318, 0.6294] | 0.4654 [0.4000, 0.5308] | 0.5811 [0.5318, 0.6299] | 0.6048 | 0.6954 | 0.5333 [0.5143, 0.5521] |
| zero-shot (E4-0 T2) | A3-ign | **0.6888 [0.6486, 0.7276]** | 0.7436 [0.6954, 0.7891] | 0.6961 [0.6527, 0.7382] | 0.6248 [0.5679, 0.6813] | 0.7387 [0.7014, 0.7743] | 0.7352 | 0.6772 | (E4-0) |

middle ruler per crop placement: 0.1: 0.202, 0.3: 0.656, 0.5: 1.000, 0.7: 0.658, 0.9: 0.211
* last metrics whole/A0 s2099: step 1740 {'lr': 0.0, 'mil': 0.0581, 'dvs_sup': 0.0526, 'dvs_sup_mil': 0.0078, 'mul_mil': 0.0009, 'total': 0.1193}
* last metrics whole/A3 s2099: step 1740 {'lr': 0.0, 'motion_share': 0.0936, 'w_u_norm': 0.8058, 'mil': 0.0117, 'dvs_sup': 0.0063, 'dvs_sup_mil': 0.0006, 'mul_mil': 0.0002, 'total': 0.0188}

# Nexar window trigger (§21 D-N10)

**BUILD `window`** — fired: W1_position, W2_no_gain

| check | fires | read | bar |
|---|---|---|---|
| W1 position | True | centre - edge crop macro 0.1147 (zero-shot, printed: 0.0713) | > 0.05 |
| W2 no gain | True | whole/A3 0.5819 vs zero-shot A3 0.6888 | ≤ |
| W3 clip classifier | False | clip AUC 0.6954; whole/A3 0.5819 vs whole/A0 0.5876 | ≥ 0.95 and A3 < A0 |

Sources: whole_A3 `/content/drive/MyDrive/Thesis-V2/outputs/v2/REPORTS/nexar_val_ignore/pilot_whole_A3`, whole_A0 `/content/drive/MyDrive/Thesis-V2/outputs/v2/REPORTS/nexar_val_ignore/pilot_whole_A0`, zero_shot_A3 `/content/drive/MyDrive/Thesis-V2/outputs/v2/REPORTS/nexar_val_ignore/zeroshot_e4_0_A3ign`

