# K5dfice vs K5dfirr (KIM on its own CLIP-only temporal encoder) -- s2024

Rule: `core/docs/final/KIM_RUN.md §59`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the project reference. Last-epoch mul_mil: K5dfirr 0.1519, K5dfice 0.1442, K5dficx 0.1455, K5f 0.0963.

* **Verdict: COST (closed).** Attribution (C): not read. Candidate for the main model: False.
* K5dfice: step-1 mil 0.690560 vs K5dfirr 0.690560, same draws = True; encoder last-epoch grad norm 4.519e+00, final drift 0.0982 (floor 0.001), trained = True.
* K5dficx: step-1 mil 0.690560 vs K5dfirr 0.690560, same draws = True; encoder last-epoch grad norm 5.501e+00, final drift 0.1137 (floor 0.001), trained = True.
* Environment (saved beside the runs): {'K5dfirr': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K5dfice': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K5dficx': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}}.

## P (decides): K5dfice - K5dfirr

| set | K5dfirr | K5dfice | K5dfice - K5dfirr | label |
|---|---|---|---|---|
| nexar | 0.8698 | 0.8498 | -0.0199 [-0.0279, -0.0122] below | COST |
| dota | 0.8259 | 0.8202 | -0.0057 [-0.0102, -0.0012] below | NO DIFFERENCE |
| dada | 0.7677 | 0.7500 | -0.0177 [-0.0239, -0.0116] below | COST |
| tad (printed) | 0.7198 | 0.7149 | -0.0049 [-0.0210, +0.0115] not separated | NO DIFFERENCE |
| dad | 0.7494 | 0.7573 | +0.0079 [-0.0086, +0.0242] not separated | NO DIFFERENCE |

## C (input beyond capacity; read only when P is HELD): K5dfice - K5dficx

| set | K5dficx | K5dfice | K5dfice - K5dficx | label |
|---|---|---|---|---|
| nexar | 0.8588 | 0.8498 | -0.0089 [-0.0140, -0.0040] below | NO DIFFERENCE |
| dota | 0.8154 | 0.8202 | +0.0049 [+0.0025, +0.0071] above | NO DIFFERENCE |
| dada | 0.7494 | 0.7500 | +0.0006 [-0.0028, +0.0040] not separated | NO DIFFERENCE |
| tad (printed) | 0.7145 | 0.7149 | +0.0004 [-0.0116, +0.0121] not separated | NO DIFFERENCE |
| dad | 0.7786 | 0.7573 | -0.0212 [-0.0331, -0.0103] below | COST |

## X (the second encoder alone, printed): K5dficx - K5dfirr

| set | K5dfirr | K5dficx | K5dficx - K5dfirr | label |
|---|---|---|---|---|
| nexar | 0.8698 | 0.8588 | -0.0110 [-0.0167, -0.0054] below | COST |
| dota | 0.8259 | 0.8154 | -0.0106 [-0.0145, -0.0065] below | COST |
| dada | 0.7677 | 0.7494 | -0.0183 [-0.0242, -0.0124] below | COST |
| tad (printed) | 0.7198 | 0.7145 | -0.0053 [-0.0190, +0.0085] not separated | NO DIFFERENCE |
| dad | 0.7494 | 0.7786 | +0.0291 [+0.0156, +0.0431] above | GAIN |

## R (the project reference; a COST = not a candidate; starts not matched): K5dfice - K5f

| set | K5f | K5dfice | K5dfice - K5f | label |
|---|---|---|---|---|
| nexar | 0.8942 | 0.8498 | -0.0444 [-0.0573, -0.0324] below | COST |
| dota | 0.8188 | 0.8202 | +0.0014 [-0.0092, +0.0123] not separated | NO DIFFERENCE |
| dada | 0.7744 | 0.7500 | -0.0244 [-0.0362, -0.0126] below | COST |
| tad (printed) | 0.6981 | 0.7149 | +0.0168 [-0.0157, +0.0512] not separated | NO DIFFERENCE |
| dad | 0.8070 | 0.7573 | -0.0497 [-0.0739, -0.0261] below | COST |

Printed: Nexar val negatives with peak > 0.5: K5dfirr 0.681, K5dfice 0.690, K5dficx 0.735, K5f 0.549.

Never from this report: "KIM now matches LAS-VAD's IAM"; "pure CLIP is the better input for KIM" unless C separates; "KIM works / does not work" from P alone; any claim about a K5f-hosted KIM; a runtime conclusion.
