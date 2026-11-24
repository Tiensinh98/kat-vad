# K5dfibk25 vs K5dfirr (K5dfi re-trained in the same notebook) -- s2024

Rule: `core/docs/final/KIM_RUN.md §51`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the project reference. Last-epoch mul_mil: K5dfirr 0.1519, K5dfibk25 0.1510, K5dfi 0.1519, K5f 0.0963.

* **Verdict: CLOSED (target B not met).** Target A (first third, Nexar val and DADA) False; target B (DoTA onset) False.
* A, strict (lower bound >= -0.01, printed): nexar False, dada True.
* Start gate: step-1 mil 0.690560 vs K5dfirr 0.690560, same draws = True.
* K5dfirr vs the old K5dfi: step-1 mil 0.690560 vs 0.690560, same draws = True.
* Environment (saved beside the runs): {'K5dfirr': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K5dfibk25': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}}.

## First third of the span vs all normal rows (per video)

| set | videos | K5dfirr | K5dfibk25 | K5dfibk25 - K5dfirr |
|---|---|---|---|---|
| nexar | 112 | 0.8457 | 0.8350 | -0.0107 [-0.0173, -0.0045] |
| dota (printed) | 568 | 0.8107 | 0.8124 | +0.0017 [-0.0013, +0.0047] |
| dada | 370 | 0.7180 | 0.7122 | -0.0058 [-0.0099, -0.0018] |

## Onset AUC: span vs the normal rows within 2 s

| set | side | K5dfirr | K5dfibk25 | K5dfibk25 - K5dfirr |
|---|---|---|---|---|
| nexar | pre | 0.8548 | 0.8363 | -0.0185 [-0.0264, -0.0112] |
| nexar | post | 0.8152 | 0.8226 | +0.0074 [+0.0023, +0.0123] |
| dota | pre | 0.8276 | 0.8271 | -0.0005 [-0.0036, +0.0025] |
| dota | post | 0.7262 | 0.7312 | +0.0050 [+0.0014, +0.0088] |
| dada | pre | 0.7913 | 0.7854 | -0.0059 [-0.0098, -0.0019] |
| dada | post | 0.6529 | 0.6544 | +0.0015 [-0.0028, +0.0058] |

## Macro (Nexar = val crops): K5dfibk25 - K5dfirr decides; K5dfirr - K5dfi is the re-run spread (printed)

| set | K5dfirr | K5dfibk25 | K5dfibk25 - K5dfirr | label | K5dfirr - K5dfi |
|---|---|---|---|---|---|
| nexar | 0.8698 | 0.8611 | -0.0087 [-0.0130, -0.0048] below | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dota | 0.8259 | 0.8252 | -0.0007 [-0.0027, +0.0012] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dada | 0.7677 | 0.7632 | -0.0045 [-0.0070, -0.0018] below | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| tad (printed) | 0.7198 | 0.7160 | -0.0038 [-0.0123, +0.0042] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |
| dad | 0.7494 | 0.7515 | +0.0021 [-0.0079, +0.0121] not separated | NO DIFFERENCE | +0.0000 [+0.0000, +0.0000] not separated |

## Macro against K5f (the project reference): a COST here also fails (BD6); starts not matched (another recipe)

| set | K5f | K5dfibk25 | K5dfibk25 - K5f | label |
|---|---|---|---|---|
| nexar | 0.8942 | 0.8611 | -0.0332 [-0.0433, -0.0236] below | COST |
| dota | 0.8188 | 0.8252 | +0.0064 [-0.0017, +0.0147] not separated | NO DIFFERENCE |
| dada | 0.7744 | 0.7632 | -0.0112 [-0.0206, -0.0018] below | COST |
| tad (printed) | 0.6981 | 0.7160 | +0.0179 [-0.0047, +0.0424] not separated | NO DIFFERENCE |
| dad | 0.8070 | 0.7515 | -0.0555 [-0.0760, -0.0357] below | COST |

Printed: Nexar val negatives with peak > 0.5: K5dfirr 0.681, K5dfibk25 0.717, K5dfi 0.681, K5f 0.549.

Never from this report: "bottom-k fixes the early rise"; any read of K5fbk (weight 1.0) against K5dfirr; a runtime conclusion from one re-run.
