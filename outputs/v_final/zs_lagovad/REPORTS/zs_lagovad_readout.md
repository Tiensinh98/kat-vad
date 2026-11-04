# Zero-shot summary — LaGoVAD vs W0 vs R4 — **PRELIMINARY (n = 1 seed)**

Rule: `core/docs/final/ZS_TAD_DADA.md §7`. Arms: Nexar-trained, seed 2024, prefix OFF. LaGoVAD = the released `best.ckpt`, KIP off, plain CLIP `_ncc` rows (the arms' own CLIP columns, checked row for row). **Not a matched comparison**: LaGoVAD trained on PreVAD.

## Summary (macro AUC [95 % cluster CI])

| row | DoTA-CAP-dev (n/1397) | TAD test | DADA-2000 | DAD |
|---|---|---|---|---|
| LaGoVAD (released ckpt, PreVAD-trained) | 0.6418 [0.6166, 0.6666] | 0.7579 [0.6955, 0.8147] | 0.6202 [0.5925, 0.6480] | 0.6462 [0.5881, 0.7016] |
| W0 (no SA², Nexar, s2024) | 0.8110 [0.7948, 0.8269] | 0.6983 [0.6222, 0.7695] | 0.7615 [0.7369, 0.7859] | 0.7317 [0.6897, 0.7716] |
| R4 (SA², Nexar, s2024) | 0.8111 [0.7927, 0.8293] | 0.7220 [0.6515, 0.7889] | 0.7579 [0.7334, 0.7820] | 0.7571 [0.7175, 0.7963] |

## DoTA-CAP-dev (n/1397) · protocol B (s1[::3], whole clip, native frames)

| row | macro | micro (per-clip min-max, pooled (lesson 12: DoTA-CAP is all-abnormal)) | clip AUC |
|---|---|---|---|
| LaGoVAD (released ckpt, PreVAD-trained) | 0.6418 [0.6166, 0.6666] | 0.6105 | — |
| W0 (no SA², Nexar, s2024) | 0.8110 [0.7948, 0.8269] | 0.7255 | — |
| R4 (SA², Nexar, s2024) | 0.8111 [0.7927, 0.8293] | 0.7291 | — |

* W0 - LaGoVAD (paired, per item): +0.1692 [+0.1438, +0.1958] (n=569) → **above** — PRELIMINARY (n = 1 seed), not matched
* R4 - LaGoVAD (paired, per item): +0.1693 [+0.1400, +0.1996] (n=569) → **above** — PRELIMINARY (n = 1 seed), not matched
* **R4 - W0 (paired, per item): +0.0001 [-0.0100, +0.0098] (n=569) → no difference shown — PRELIMINARY (n = 1 seed)**
* printed: LaGoVAD at protocol A (s1[::8], its training rate): macro 0.6185 [0.5966, 0.6405], micro 0.5932
* rulers (no pixels): position ruler 0.5573 [0.5380, 0.5765]

## TAD test · 100 videos, s8, raw scores (§26)

| row | macro | micro (raw, pooled (clip oracle 0.9226: measures clip classification)) | clip AUC |
|---|---|---|---|
| LaGoVAD (released ckpt, PreVAD-trained) | 0.7579 [0.6955, 0.8147] | 0.7912 | 0.7687 |
| W0 (no SA², Nexar, s2024) | 0.6983 [0.6222, 0.7695] | 0.6034 | 0.5942 |
| R4 (SA², Nexar, s2024) | 0.7220 [0.6515, 0.7889] | 0.7167 | 0.6567 |

* W0 - LaGoVAD (paired, per item): -0.0596 [-0.1460, +0.0259] (n=60) → **not separated** — PRELIMINARY (n = 1 seed), not matched
* R4 - LaGoVAD (paired, per item): -0.0359 [-0.1184, +0.0470] (n=60) → **not separated** — PRELIMINARY (n = 1 seed), not matched
* **R4 - W0 (paired, per item): +0.0237 [+0.0005, +0.0499] (n=60) → gain shown — PRELIMINARY (n = 1 seed)**
* integrity: LaGoVAD TAD micro 0.7912 vs Gate T0 0.7912 (gap +0.00003) → PASS
* rulers (no pixels): clip oracle micro 0.9226 · length ruler micro 0.8968 · middle ruler 0.6831 [0.6097, 0.7517]

## DADA-2000 · 370 T2-test sources, whole video, s8 (D-N20)

| row | macro | micro (raw, pooled) | clip AUC |
|---|---|---|---|
| LaGoVAD (released ckpt, PreVAD-trained) | 0.6202 [0.5925, 0.6480] | 0.5388 | — |
| W0 (no SA², Nexar, s2024) | 0.7615 [0.7369, 0.7859] | 0.6981 | — |
| R4 (SA², Nexar, s2024) | 0.7579 [0.7334, 0.7820] | 0.7048 | — |

* W0 - LaGoVAD (paired, per item): +0.1413 [+0.1093, +0.1738] (n=370) → **above** — PRELIMINARY (n = 1 seed), not matched
* R4 - LaGoVAD (paired, per item): +0.1377 [+0.1057, +0.1696] (n=370) → **above** — PRELIMINARY (n = 1 seed), not matched
* **R4 - W0 (paired, per item): -0.0036 [-0.0161, +0.0089] (n=370) → no difference shown — PRELIMINARY (n = 1 seed)**
* rulers (no pixels): monotone ruler 0.5514 [0.5247, 0.5782] · middle ruler 0.7342 [0.7076, 0.7594]

## DAD · testing split, whole video, s7 at 20 fps (ZS_DAD.md)

| row | macro | micro (raw, pooled (equal-length videos: the length ruler is uninformative)) | clip AUC |
|---|---|---|---|
| LaGoVAD (released ckpt, PreVAD-trained) | 0.6462 [0.5881, 0.7016] | 0.5681 | 0.5808 |
| W0 (no SA², Nexar, s2024) | 0.7317 [0.6897, 0.7716] | 0.6693 | 0.6451 |
| R4 (SA², Nexar, s2024) | 0.7571 [0.7175, 0.7963] | 0.7020 | 0.6729 |

* W0 - LaGoVAD (paired, per item): +0.0855 [+0.0196, +0.1517] (n=165) → **above** — PRELIMINARY (n = 1 seed), not matched
* R4 - LaGoVAD (paired, per item): +0.1110 [+0.0466, +0.1758] (n=165) → **above** — PRELIMINARY (n = 1 seed), not matched
* **R4 - W0 (paired, per item): +0.0254 [-0.0084, +0.0592] (n=165) → no difference shown — PRELIMINARY (n = 1 seed)**
* rulers (no pixels): clip oracle micro 0.8390 · length ruler micro 0.5000 · middle ruler 0.1538 [0.1538, 0.1538]

