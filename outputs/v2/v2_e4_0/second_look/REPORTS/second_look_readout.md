# v2 second look — A3ign on the opened sets (second look at an opened set)

Rule: core/docs/v2/PREREG_ADDENDUM.md §23 (Amendment 12, S1-S8). Nothing re-decides (S2): `B` = A3ign stands; §19's numbers stay.

## DoTA-CAP (n/1397) eval — 560 clips

| arm | macro (seed-averaged) | dev macro | optimism (dev - eval) |
|---|---|---|---|
| A3ign | 0.8120 | 0.7912 | -0.0209 |
| A3 | 0.7785 | 0.7576 | -0.0209 |
| A0 | 0.6979 | — | — |
| `p_T2` | 0.8256 | — | — |

- **A3ign - A3, paired t95 (S3): +0.0376 [+0.0326, +0.0426]** (n = 5); per seed s2024 +0.0422, s2025 +0.0394, s2026 +0.0396, s2027 +0.0337, s2028 +0.0330
- A3ign - A0 (printed): +0.1276 [+0.0991, +0.1561]; clip-level A3ign - A3 (printed): +0.0336 [+0.0255, +0.0415]
- `f_2` (S4): A3ign 0.8640 · A3 0.8484 · Δ +0.0157 [+0.0104, +0.0207]

## T2-test (S6) — seed means

| arm | macro | micro | clip oracle | window AUC |
|---|---|---|---|---|
| A3ign | 0.7202 | 0.7334 | 0.7037 | 0.7714 |
| A3 | 0.6950 | 0.7095 | 0.7037 | 0.7706 |
| A0 | 0.6307 | 0.6228 | 0.7037 | 0.6667 |

- A3ign - A3 T2-test macro, paired t95 (printed): +0.0252 [+0.0170, +0.0333]
- O1' vs A0, every seed: PASS {'s2024': True, 's2025': True, 's2026': True, 's2027': True, 's2028': True}

## Sentences (S5)

- Dropping the whole-anchor DVS label improves DoTA-CAP (n/1397) detection, on dev and on a second look at the opened eval set.
- ... beyond the position prior (second look).
