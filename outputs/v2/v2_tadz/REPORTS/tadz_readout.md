# TAD zero-shot — 100 videos (60 two-class), TAD test, stride 8, whole video, raw scores

Rule: PREREG_ADDENDUM.md §26 (Amendment 15, T1-T10). Seeds ['s2024', 's2025', 's2026', 's2027', 's2028']. `B` = A3-ign is not re-decided here.

**Endpoint (T5): no transfer gain of v2's components is shown on TAD at n = 5.**

**LaGoVAD (T6): A3-ign and LaGoVAD's released checkpoint are not separated on TAD.**

## Contrasts (macro)

| Δ | paired t95 over seeds | per-clip bootstrap (seed-averaged) |
|---|---|---|
| **A3ign - A0** (endpoint) | +0.0218 [-0.0032, +0.0467] | +0.0327 [-0.0486, +0.1155] |
| A3-A0 (printed) | +0.0324 [+0.0038, +0.0610] (excludes 0) | +0.0429 [-0.0437, +0.1313] |
| A3ign-A3 (printed) | -0.0107 [-0.0263, +0.0050] | -0.0102 [-0.0362, +0.0160] |
| A3ign - T0 (printed, T6) | — | +0.0015 [-0.0631, +0.0663] |

## Levels (seed-averaged scores)

| | macro | clip-level AUC | micro (raw) | per-seed macro |
|---|---|---:|---:|---|
| A0 | 0.7268 [0.6568, 0.7930] | 0.5779 | 0.4932 | 0.7196, 0.7381, 0.7255, 0.7092, 0.7299 |
| A3 | 0.7696 [0.7118, 0.8227] | 0.4996 | 0.6851 | 0.7303, 0.7719, 0.7880, 0.7553, 0.7388 |
| A3ign | 0.7594 [0.7074, 0.8077] | 0.5440 | 0.6779 | 0.7399, 0.7512, 0.7679, 0.7486, 0.7234 |
| T0 (LaGoVAD released) | 0.7579 [0.6974, 0.8144] | 0.7687 | 0.7912 | — |

## Rulers — what no pixel scores (printed beside every number)

* clip oracle micro **0.9226** · length ruler micro **0.8968** — micro on TAD is clip classification (C14); never put it beside 89.56
* `p_T2` macro 0.6885 [0.6151, 0.7578] · monotone `t/N` 0.5872 [0.5113, 0.6611] · middle 0.6831 [0.6103, 0.7534]

## Collapse print (O1' shape: clip-level up while macro down), vs A0

* A3: clip-level AUC -0.0783, macro +0.0429
* A3ign: clip-level AUC -0.0340, macro +0.0327
