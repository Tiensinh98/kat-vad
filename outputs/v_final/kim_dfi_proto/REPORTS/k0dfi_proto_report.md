# Class prototypes in the main score (§61) vs K0dfirr -- s2024

Rule: `core/docs/final/KIM_RUN.md §61`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the project reference. Last-epoch mul_mil: K0dfirr 0.1486, K0dfiP 0.1485, K0dfiL 0.1523, K0dfiPL 0.1516, K0dfiPLb 0.1521, K5f 0.0963, K5dfi 0.1519.

* **Verdict (D): COST (closed).** Candidate for the main model: False. Deciding sets: nexar, dota, dada; DAD labelled, never decides (read beside `dad_position_diag`).
* K0dfiP: step-1 mil 0.690560 vs K0dfirr 0.690560, same draws = True; last step λ +0.0091, cos(μ0, μ1) +0.258, anchor share 0.661.
* K0dfiL: step-1 mil 0.690560 vs K0dfirr 0.690560, same draws = True; last step λ n/a, cos(μ0, μ1) -0.437, anchor share 0.643.
* K0dfiPL: step-1 mil 0.690560 vs K0dfirr 0.690560, same draws = True; last step λ +0.0034, cos(μ0, μ1) -0.438, anchor share 0.638.
* K0dfiPLb: step-1 mil 0.690560 vs K0dfirr 0.690560, same draws = True; last step λ +0.0033, cos(μ0, μ1) -0.450, anchor share 0.643.
* Environment (saved beside the runs): {'K0dfirr': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K0dfiP': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K0dfiL': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K0dfiPL': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}, 'K0dfiPLb': {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'}}.

## D (decides): K0dfiPL - K0dfirr

| set | K0dfirr | K0dfiPL | K0dfiPL - K0dfirr | label |
|---|---|---|---|---|
| nexar | 0.8424 | 0.8184 | -0.0240 [-0.0307, -0.0177] below | COST |
| dota | 0.8067 | 0.8100 | +0.0033 [+0.0015, +0.0052] above | NO DIFFERENCE |
| dada | 0.7381 | 0.7392 | +0.0011 [-0.0011, +0.0033] not separated | NO DIFFERENCE |
| dad (never decides) | 0.7394 | 0.7312 | -0.0082 [-0.0156, -0.0005] below | NO DIFFERENCE |

## P (score term alone): K0dfiP - K0dfirr

| set | K0dfirr | K0dfiP | K0dfiP - K0dfirr | label |
|---|---|---|---|---|
| nexar | 0.8424 | 0.8420 | -0.0005 [-0.0007, -0.0002] below | NO DIFFERENCE |
| dota | 0.8067 | 0.8068 | +0.0001 [-0.0001, +0.0003] not separated | NO DIFFERENCE |
| dada | 0.7381 | 0.7382 | +0.0001 [-0.0003, +0.0004] not separated | NO DIFFERENCE |
| dad (never decides) | 0.7394 | 0.7406 | +0.0012 [-0.0002, +0.0026] not separated | NO DIFFERENCE |

## L (loss alone): K0dfiL - K0dfirr

| set | K0dfirr | K0dfiL | K0dfiL - K0dfirr | label |
|---|---|---|---|---|
| nexar | 0.8424 | 0.8177 | -0.0247 [-0.0313, -0.0184] below | COST |
| dota | 0.8067 | 0.8101 | +0.0034 [+0.0016, +0.0052] above | NO DIFFERENCE |
| dada | 0.7381 | 0.7392 | +0.0012 [-0.0010, +0.0034] not separated | NO DIFFERENCE |
| dad (never decides) | 0.7394 | 0.7305 | -0.0089 [-0.0166, -0.0012] below | NO DIFFERENCE |

## PL - P (the loss on top of the term)

| set | K0dfiP | K0dfiPL | K0dfiPL - K0dfiP | label |
|---|---|---|---|---|
| nexar | 0.8420 | 0.8184 | -0.0236 [-0.0302, -0.0173] below | COST |
| dota | 0.8068 | 0.8100 | +0.0032 [+0.0015, +0.0050] above | NO DIFFERENCE |
| dada | 0.7382 | 0.7392 | +0.0010 [-0.0012, +0.0032] not separated | NO DIFFERENCE |
| dad (never decides) | 0.7406 | 0.7312 | -0.0093 [-0.0170, -0.0014] below | NO DIFFERENCE |

## b (bottom-k in mu0): K0dfiPLb - K0dfiPL

| set | K0dfiPL | K0dfiPLb | K0dfiPLb - K0dfiPL | label |
|---|---|---|---|---|
| nexar | 0.8184 | 0.8186 | +0.0002 [-0.0003, +0.0007] not separated | NO DIFFERENCE |
| dota | 0.8100 | 0.8101 | +0.0001 [-0.0002, +0.0003] not separated | NO DIFFERENCE |
| dada | 0.7392 | 0.7391 | -0.0000 [-0.0005, +0.0004] not separated | NO DIFFERENCE |
| dad (never decides) | 0.7312 | 0.7312 | +0.0000 [-0.0014, +0.0014] not separated | NO DIFFERENCE |

## R (candidacy; a COST = not a candidate; starts not matched): K0dfiPL - K5f

| set | K5f | K0dfiPL | K0dfiPL - K5f | label |
|---|---|---|---|---|
| nexar | 0.8942 | 0.8184 | -0.0758 [-0.0917, -0.0607] below | COST |
| dota | 0.8188 | 0.8100 | -0.0088 [-0.0194, +0.0020] not separated | NO DIFFERENCE |
| dada | 0.7744 | 0.7392 | -0.0353 [-0.0468, -0.0238] below | COST |
| dad (never decides) | 0.8070 | 0.7312 | -0.0758 [-0.1019, -0.0508] below | COST |

## KIM (printed; same host, KIM on; envs may differ): K0dfiPL - K5dfi

| set | K5dfi | K0dfiPL | K0dfiPL - K5dfi | label |
|---|---|---|---|---|
| nexar | 0.8698 | 0.8184 | -0.0513 [-0.0637, -0.0393] below | COST |
| dota | 0.8259 | 0.8100 | -0.0159 [-0.0209, -0.0110] below | COST |
| dada | 0.7677 | 0.7392 | -0.0285 [-0.0349, -0.0221] below | COST |
| dad (never decides) | 0.7494 | 0.7312 | -0.0182 [-0.0350, -0.0021] below | COST |

Printed: Nexar val negatives with peak > 0.5: K0dfirr 0.646, K0dfiP 0.646, K0dfiL 0.673, K0dfiPL 0.673, K0dfiPLb 0.673, K5f 0.549, K5dfi 0.681.

Never from this report: "prototypes make the model confident"; "the prototype term replaces KIM" from KIM alone; a DAD macro difference without the placebo; "μ1 helps / hurts" (no row has μ0 without μ1).
