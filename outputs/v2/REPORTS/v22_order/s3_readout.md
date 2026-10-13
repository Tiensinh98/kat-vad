# v2.2 S3 — κ-probe

Bar (§6.1): cluster-bootstrap 95 % low > 0 **and** mean ≥ +0.01, DoTA-CAP-dev, protocol B. DoTA-CAP (n/1397).

κ width 131 (Δu not admitted). **VALUE FAIL · ORDER FAIL · S5 eligible: False**

| contrast | a (DoTA-CAP-dev macro) | b | **Δ a - b** (decides) | `f_2` Δ | T2-val Δ (reported) |
|---|---|---|---|---|---|
| VALUE: `F0+K` - `F0` | 0.8531 [0.8413, 0.8646] | 0.8538 [0.8418, 0.8659] | **-0.0007 [-0.0063, +0.0047]** | +0.0006 [-0.0026, +0.0038] | -0.0004 [-0.0108, +0.0101] |
| ↳ per-seed Δ (DoTA, stability only) | 2024: +0.0048, 2025: +0.0133, 2026: -0.0058, 2027: -0.0050, 2028: -0.0098 | | | | |
| ORDER: `F0+K` - `F0+K^π` | 0.8531 [0.8413, 0.8646] | 0.8451 [0.8328, 0.8577] | **+0.0080 [+0.0021, +0.0140]** | +0.0019 [-0.0014, +0.0053] | -0.0034 [-0.0123, +0.0055] |
| ↳ per-seed Δ (DoTA, stability only) | 2024: +0.0365, 2025: +0.0119, 2026: -0.0006, 2027: +0.0044, 2028: -0.0054 | | | | |

Printed: DADA↔DoTA shortcut AUC on κ 0.905. Sequences: {'train': {'sequences': 1272, 'under_12': 0, 'under_3_excluded': 0, 'gap_counts': {6: 1272}}, 'val': {'sequences': 219, 'under_12': 0, 'under_3_excluded': 0, 'gap_counts': {6: 219}}, 'dota': {'sequences': 569, 'under_12': 0, 'under_3_excluded': 0, 'gap_counts': {6: 569}}}.
