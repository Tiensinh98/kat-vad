# Nexar window trigger (§21 D-N10)

**BUILD `window`** — fired: W1_position, W2_no_gain

| check | fires | read | bar |
|---|---|---|---|
| W1 position | True | centre - edge crop macro 0.1147 (zero-shot, printed: 0.0713) | > 0.05 |
| W2 no gain | True | whole/A3 0.5819 vs zero-shot A3 0.6888 | ≤ |
| W3 clip classifier | False | clip AUC 0.6954; whole/A3 0.5819 vs whole/A0 0.5876 | ≥ 0.95 and A3 < A0 |

Sources: whole_A3 `/content/drive/MyDrive/Thesis-V2/outputs/v2/REPORTS/nexar_val_ignore/pilot_whole_A3`, whole_A0 `/content/drive/MyDrive/Thesis-V2/outputs/v2/REPORTS/nexar_val_ignore/pilot_whole_A0`, zero_shot_A3 `/content/drive/MyDrive/Thesis-V2/outputs/v2/REPORTS/nexar_val_ignore/zeroshot_e4_0_A3ign`
