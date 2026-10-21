# TAD clip-level gate — `s = c(v) · minmax_v(y_A3ign)`, `c` fitted on MSAD-train

Rule: PREREG_ADDENDUM.md §27 (Amendment 16, G1-G10). `B` = A3-ign is not re-decided; macro is unchanged by construction (G3).

**Sentence (G6): a clip-level gate trained on MSAD restores clip-level discrimination on TAD.**

## Dev gate (G5) — TAD-train, never fitted on

* 410 videos (200 abnormal): gate clip-level AUC **0.8051** (bar 0.60 → PASS) · length ruler 0.6541
* MSAD-test (in-domain sanity): gate clip-level AUC 0.9484

## Endpoint (G6) — TAD-test micro, Δ = micro(s) - micro(A3-ign)

| paired t95 over seeds | video bootstrap (seed-averaged) |
|---|---|
| +0.1361 [+0.1323, +0.1398] (excludes 0) | +0.1569 [+0.1024, +0.2037] |

## Levels (seed-averaged; macro 0.7594 for every row built on A3-ign)

| | micro | clip-level AUC |
|---|---:|---:|
| A3ign | 0.6779 | 0.5440 |
| c x minmax(A3ign) | 0.8347 | 0.8512 |
| T0 (LaGoVAD released) | 0.7912 | 0.7687 |
| mean(T0) x minmax(A3ign) (form seen on this set) | 0.7524 | 0.8000 |

## Printed (G7)

* gate alone, clip-level AUC on TAD-test 0.8512 · length ruler 0.6940
* micro(s) - micro(T0), video bootstrap +0.0435 [-0.0313, +0.1294] — not matched (T6)
* micro rulers (§26): clip oracle 0.9226 · length ruler 0.8968; never beside 89.56
