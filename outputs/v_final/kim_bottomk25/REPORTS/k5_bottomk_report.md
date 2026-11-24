# K5fbk25 vs K5frr (K5f re-trained in the same notebook) -- s2024

Rule: `core/docs/final/KIM_RUN.md §50`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the project reference. Last-epoch mul_mil: K5frr 0.0963, K5fbk25 0.0932, K5f 0.0963.

* **Verdict: TRADE-OFF (closed).** Target A (first third, Nexar val and DADA) False; target B (DoTA onset) True.
* A, strict (lower bound >= -0.01, printed): nexar False, dada False.
* Start gate: step-1 mil 0.687794 vs K5frr 0.687794, same draws = True.
* K5frr vs the old K5f: step-1 mil 0.687794 vs 0.687794, same draws = True.
* Environment (saved beside the runs): {'K5frr': {'gpu': 'NVIDIA L4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K5fbk25': {'gpu': 'NVIDIA L4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}}.

## First third of the span vs all normal rows (per video)

| set | videos | K5frr | K5fbk25 | K5fbk25 - K5frr |
|---|---|---|---|---|
| nexar | 112 | 0.8706 | 0.8224 | -0.0482 [-0.0595, -0.0373] |
| dota (printed) | 568 | 0.8275 | 0.8187 | -0.0087 [-0.0133, -0.0040] |
| dada | 370 | 0.7495 | 0.7194 | -0.0301 [-0.0385, -0.0219] |

## Onset AUC: span vs the normal rows within 2 s

| set | side | K5frr | K5fbk25 | K5fbk25 - K5frr |
|---|---|---|---|---|
| nexar | pre | 0.8907 | 0.8634 | -0.0273 [-0.0388, -0.0162] |
| nexar | post | 0.8441 | 0.8174 | -0.0267 [-0.0346, -0.0191] |
| dota | pre | 0.7812 | 0.8266 | +0.0454 [+0.0375, +0.0541] |
| dota | post | 0.7657 | 0.7387 | -0.0270 [-0.0335, -0.0201] |
| dada | pre | 0.7844 | 0.7929 | +0.0085 [-0.0007, +0.0174] |
| dada | post | 0.6877 | 0.6620 | -0.0257 [-0.0334, -0.0177] |

## Macro (Nexar = val crops): K5fbk25 - K5frr decides; K5frr - K5f is the re-run spread (printed)

| set | K5frr | K5fbk25 | K5fbk25 - K5frr | label | K5frr - K5f |
|---|---|---|---|---|---|
| nexar | 0.8942 | 0.8679 | -0.0263 [-0.0324, -0.0206] below | COST | +0.0000 [+0.0000, +0.0000] not separated |
| dota | 0.8188 | 0.8268 | +0.0080 [+0.0041, +0.0121] above | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dada | 0.7744 | 0.7624 | -0.0120 [-0.0170, -0.0069] below | COST | +0.0000 [+0.0000, +0.0000] not separated |
| tad (printed) | 0.6981 | 0.7073 | +0.0092 [-0.0030, +0.0216] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dad | 0.8070 | 0.8063 | -0.0007 [-0.0084, +0.0070] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |

Printed: Nexar val negatives with peak > 0.5: K5frr 0.549, K5fbk25 0.575, K5f 0.549.

Never from this report: "bottom-k fixes the early rise"; any read of K5fbk (weight 1.0) against K5frr; a runtime conclusion from one re-run.
