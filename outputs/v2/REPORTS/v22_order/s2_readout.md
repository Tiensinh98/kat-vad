# v2.2 S2 — D6-step probe (true order vs adjacency-free block shuffle)

Bar (§6.1): cluster-bootstrap 95 % low > 0 **and** mean ≥ +0.01, DoTA-CAP-dev, protocol B. DoTA-CAP (n/1397).

**Verdict (on `F0`): PASS**

| contrast | a (DoTA-CAP-dev macro) | b | **Δ a - b** (decides) | `f_2` Δ | T2-val Δ (reported) |
|---|---|---|---|---|---|
| `F0 = [x̃;ũ;p]`: true - shuffled (decides) | 0.8532 [0.8416, 0.8651] | 0.8250 [0.8140, 0.8358] | **+0.0283 [+0.0212, +0.0356]** | +0.0138 [+0.0103, +0.0174] | +0.0165 [+0.0060, +0.0267] |
| ↳ per-seed Δ (DoTA, stability only) | 2024: +0.0268, 2025: +0.0300, 2026: +0.0301, 2027: +0.0264, 2028: +0.0267 | | | | |
| `[x̃;p]` CLIP only: true - shuffled (printed) | 0.7260 [0.7084, 0.7433] | 0.7140 [0.6969, 0.7313] | **+0.0119 [+0.0015, +0.0230]** | +0.0087 [+0.0035, +0.0138] | +0.0252 [+0.0103, +0.0401] |
| ↳ per-seed Δ (DoTA, stability only) | 2024: +0.0090, 2025: +0.0128, 2026: +0.0273, 2027: +0.0018, 2028: +0.0096 | | | | |

Unshuffled short sequences (n ≤ 3): 0. Estimand: order finer than ≈ 1.5 s, beyond coarse position and beyond VideoMAE's 1.5 s bag; lag-2 neighbours remain.
