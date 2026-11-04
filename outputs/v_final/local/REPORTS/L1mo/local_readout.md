# LaGoVAD-OSTP Step 1 read-out (L1mo) — DoTA-CAP (n/1397) dev, protocol B — **PRELIMINARY (n = 1 seed)**

Rule: `core/docs/final/LOCAL_STEP1.md §4-§5`. Fusion: 0.5 minmax(a) + 0.5 minmax(b), per clip, native frames.

## Verdict: **G1 FAIL without the overfit tell: the patch input itself is weak** — PRELIMINARY (n = 1 seed)

| gate | read | bar | pass |
|---|---|---|---|
| G1 L1mo alone | 0.7041 [0.6841, 0.7230] (n=569) | low > 0.6, mean > 0.7132 | False |
| G2 (L0+L1mo) - (L0+L0b) | -0.0121 [-0.0212, -0.0031] (n=569) | low > 0, mean ≥ 0.01 | False |
| G3 oracle max(L0,L1mo) - max(L0,L0b) | +0.0133 [+0.0060, +0.0208] (n=569) | low > 0 | True |

| row | macro [95 % cluster CI] |
|---|---|
| L0 | 0.8110 [0.7948, 0.8269] (n=569) |
| L0b | 0.8058 [0.7895, 0.8221] (n=569) |
| L1mo | 0.7041 [0.6841, 0.7230] (n=569) |
| L0+L0b | 0.8127 [0.7968, 0.8283] (n=569) |
| L0+L1mo | 0.8006 [0.7825, 0.8182] (n=569) |
| L1 (printed) | 0.6122 [0.5976, 0.6268] (n=569) |

Printed: `p_T2` (cubic position prior, no pixels) on CAP-dev = 0.8220.

| contrast (printed) | Δ all | sentence | Δ ego | Δ non-ego |
|---|---|---|---|---|
| L0+L1mo - L0+L0b (G2) | -0.0121 [-0.0212, -0.0031] (n=569) | cost shown — PRELIMINARY (n = 1 seed) | -0.0034 [-0.0147, +0.0075] (n=321) | -0.0233 [-0.0368, -0.0094] (n=248) |
| L1mo - L0 | -0.1069 [-0.1252, -0.0893] (n=569) | cost shown — PRELIMINARY (n = 1 seed) | -0.0958 [-0.1179, -0.0745] (n=321) | -0.1213 [-0.1509, -0.0876] (n=248) |
| L0+L1mo - L0 | -0.0104 [-0.0191, -0.0017] (n=569) | cost shown — PRELIMINARY (n = 1 seed) | -0.0012 [-0.0115, +0.0086] (n=321) | -0.0223 [-0.0359, -0.0073] (n=248) |
| L0+L0b - L0 | +0.0017 [-0.0024, +0.0058] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | +0.0023 [-0.0021, +0.0061] (n=321) | +0.0009 [-0.0060, +0.0080] (n=248) |
| L0b - L0 (seed spread) | -0.0052 [-0.0134, +0.0031] (n=569) | no difference shown — PRELIMINARY (n = 1 seed) | -0.0016 [-0.0093, +0.0056] (n=321) | -0.0098 [-0.0244, +0.0052] (n=248) |
| L1mo - L1 | +0.0919 [+0.0737, +0.1096] (n=569) | gain shown — PRELIMINARY (n = 1 seed) | +0.1212 [+0.0982, +0.1436] (n=321) | +0.0539 [+0.0265, +0.0819] (n=248) |

D2 (printed): per-clip AUC vs L0; a clip is won at a lead ≥ 0.05.

| row | clips | Spearman with L0 | row wins | L0 wins |
|---|---|---|---|---|
| L1mo | 569 | 0.368 | 0.200 | 0.573 |
| L0b | 569 | 0.887 | 0.167 | 0.202 |

Mean per-clip Spearman of the native scores with `p_T2` (printed; high = the row tracks position):

| row | rho | clips |
|---|---|---|
| L0 | 0.561 | 569 |
| L0b | 0.557 | 569 |
| L1mo | 0.300 | 569 |

Training diagnostics (first → last logged batch; W0 s2024 mil last = 0.176):

| run | step | mil | SA² entropy / log K | ‖W_q‖ | rho_u |
|---|---|---|---|---|---|
| L0 | 1749 | 0.6944 → 0.1760 | — | — | 0.0000 → 0.1459 |
| L0b | 1749 | 0.7102 → 0.1788 | — | — | 0.0000 → 0.1370 |
| L1mo | 1749 | 0.7443 → 0.6876 | 1.0000 → 0.9042 | 0.0000 → 1.9970 | — |
