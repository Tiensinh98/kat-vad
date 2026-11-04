# LaGoVAD-OSTP Step 1 read-out — DoTA-CAP (n/1397) dev, protocol B — **PRELIMINARY (n = 1 seed)**

Rule: `core/docs/final/LOCAL_STEP1.md §4-§5`. Fusion: 0.5 minmax(a) + 0.5 minmax(b), per clip, native frames.

## Verdict: **G1 FAIL without the overfit tell: the patch input itself is weak** — PRELIMINARY (n = 1 seed)

| gate | read | bar | pass |
|---|---|---|---|
| G1 L1 alone | 0.6122 [0.5976, 0.6268] (n=569) | low > 0.6, mean > 0.7132 | False |
| G2 (L0+L1) - (L0+L0b) | -0.0493 [-0.0584, -0.0402] (n=569) | low > 0, mean ≥ 0.01 | False |
| G3 oracle max(L0,L1) - max(L0,L0b) | +0.0003 [-0.0064, +0.0071] (n=569) | low > 0 | False |

| row | macro [95 % cluster CI] |
|---|---|
| L0 | 0.8110 [0.7948, 0.8269] (n=569) |
| L0b | 0.8058 [0.7895, 0.8221] (n=569) |
| L1 | 0.6122 [0.5976, 0.6268] (n=569) |
| L0+L0b | 0.8127 [0.7968, 0.8283] (n=569) |
| L0+L1 | 0.7634 [0.7480, 0.7785] (n=569) |

Printed: `p_T2` (cubic position prior, no pixels) on CAP-dev = 0.8220.

| contrast (printed) | Δ all | sentence | Δ ego | Δ non-ego |
|---|---|---|---|---|
| L0+L1 - L0+L0b (G2) | -0.0493 [-0.0584, -0.0402] (n=569) | cost shown — PRELIMINARY (n = 1 seed) | -0.0547 [-0.0670, -0.0428] (n=321) | -0.0424 [-0.0555, -0.0294] (n=248) |
| L1 - L0 | -0.1988 [-0.2187, -0.1792] (n=569) | cost shown — PRELIMINARY (n = 1 seed) | -0.2170 [-0.2409, -0.1931] (n=321) | -0.1752 [-0.2087, -0.1411] (n=248) |
| L0+L1 - L0 | -0.0476 [-0.0564, -0.0390] (n=569) | cost shown — PRELIMINARY (n = 1 seed) | -0.0524 [-0.0644, -0.0413] (n=321) | -0.0414 [-0.0548, -0.0285] (n=248) |
| L0+L0b - L0 | +0.0017 [-0.0024, +0.0058] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | +0.0023 [-0.0021, +0.0061] (n=321) | +0.0009 [-0.0060, +0.0080] (n=248) |
| L0b - L0 (seed spread) | -0.0052 [-0.0134, +0.0031] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | -0.0016 [-0.0093, +0.0056] (n=321) | -0.0098 [-0.0244, +0.0052] (n=248) |

D2 (printed): per-clip AUC vs L0; a clip is won at a lead ≥ 0.05.

| row | clips | Spearman with L0 | row wins | L0 wins |
|---|---|---|---|---|
| L1 | 569 | 0.259 | 0.120 | 0.768 |
| L0b | 569 | 0.887 | 0.167 | 0.202 |

Mean per-clip Spearman of the native scores with `p_T2` (printed; high = the row tracks position):

| row | rho | clips |
|---|---|---|
| L0 | 0.561 | 569 |
| L0b | 0.557 | 569 |
| L1 | 0.152 | 569 |

Training diagnostics (first → last logged batch; W0 s2024 mil last = 0.176):

| run | step | mil | SA² entropy / log K | ‖W_q‖ | rho_u |
|---|---|---|---|---|---|
| L0 | 1749 | 0.6944 → 0.1760 | — | — | 0.0000 → 0.1459 |
| L0b | 1749 | 0.7102 → 0.1788 | — | — | 0.0000 → 0.1370 |
| L1 | 1749 | 0.7168 → 0.5907 | 1.0000 → 0.9473 | 0.0000 → 1.7342 | — |
