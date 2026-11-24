# K5fepre / K5fepost / K5fbk vs K5f -- s2024

Rule: `core/docs/final/KIM_RUN.md §49`. PRELIMINARY, n = 1; nothing is adopted, K5f stays the reference. Last-epoch mul_mil: K5f 0.0963, K5fepre 0.1335, K5fepost 0.1173, K5fbk 0.0730.

* **K5fepre: TRADE-OFF (closed).** T1 hit False, T2 hit True.
* **K5fepost: closed (no gain, or a cost).** T1 hit False, T2 hit False (printed).
* **K5fbk: closed (no gain, or a cost).** T1 hit True, T2 hit True (printed).
* K5fbk step-1 mil 0.692545 vs K5f 0.687794; same draws = False. **FLAG: not the same draws as K5f.**

## T1: Nexar val zone AUC [-1, 0) s vs negative-video frames (lower = less early rise)

| row | AUC | row - K5f |
|---|---|---|
| K5f | 0.6943 | - |
| K5fepre | 0.6825 | -0.0119 [-0.0277, +0.0045] |
| K5fepost | 0.7115 | +0.0172 [-0.0009, +0.0353] |
| K5fbk | 0.6224 | -0.0720 [-0.1050, -0.0404] |

## Onset AUC: span vs the normal frames within 2 s (higher = sharper edge; T2 = dota, pre)

| set | side | row | videos | AUC | row - K5f |
|---|---|---|---|---|---|
| nexar | pre | K5f | 112 | 0.8907 | - |
| nexar | pre | K5fepre | 112 | 0.8851 | -0.0055 [-0.0171, +0.0066] |
| nexar | pre | K5fepost | 112 | 0.8630 | -0.0276 [-0.0418, -0.0133] |
| nexar | pre | K5fbk | 112 | 0.7999 | -0.0908 [-0.1146, -0.0677] |
| nexar | post | K5f | 112 | 0.8441 | - |
| nexar | post | K5fepre | 112 | 0.8137 | -0.0305 [-0.0418, -0.0188] |
| nexar | post | K5fepost | 112 | 0.8797 | +0.0356 [+0.0218, +0.0513] |
| nexar | post | K5fbk | 112 | 0.8056 | -0.0386 [-0.0610, -0.0167] |
| dota | pre | K5f | 568 | 0.7812 | - |
| dota | pre | K5fepre | 568 | 0.8475 | +0.0663 [+0.0546, +0.0787] |
| dota | pre | K5fepost | 568 | 0.7395 | -0.0416 [-0.0544, -0.0293] |
| dota | pre | K5fbk | 568 | 0.8272 | +0.0460 [+0.0322, +0.0600] |
| dota | post | K5f | 542 | 0.7657 | - |
| dota | post | K5fepre | 542 | 0.6949 | -0.0709 [-0.0817, -0.0599] |
| dota | post | K5fepost | 542 | 0.8088 | +0.0430 [+0.0304, +0.0566] |
| dota | post | K5fbk | 542 | 0.7112 | -0.0546 [-0.0665, -0.0419] |
| dada | pre | K5f | 356 | 0.7844 | - |
| dada | pre | K5fepre | 356 | 0.8036 | +0.0192 [+0.0081, +0.0304] |
| dada | pre | K5fepost | 356 | 0.7619 | -0.0225 [-0.0367, -0.0085] |
| dada | pre | K5fbk | 356 | 0.7594 | -0.0251 [-0.0414, -0.0089] |
| dada | post | K5f | 354 | 0.6877 | - |
| dada | post | K5fepre | 354 | 0.6385 | -0.0492 [-0.0602, -0.0382] |
| dada | post | K5fepost | 354 | 0.7369 | +0.0492 [+0.0355, +0.0634] |
| dada | post | K5fbk | 354 | 0.6367 | -0.0510 [-0.0654, -0.0365] |

## Macro per row (Nexar = val crops) and row - K5f, paired, 95 % cluster bootstrap

| set | K5f | K5fepre | K5fepre - K5f | label | K5fepost | K5fepost - K5f | label | K5fbk | K5fbk - K5f | label |
|---|---|---|---|---|---|---|---|---|---|---|
| nexar | 0.8942 | 0.8794 | -0.0149 [-0.0214, -0.0087] below | COST | 0.8962 | +0.0020 [-0.0067, +0.0105] not separated | NO DIFFERENCE | 0.8210 | -0.0732 [-0.0883, -0.0587] below | COST |
| dota | 0.8188 | 0.8214 | +0.0026 [-0.0047, +0.0103] not separated | NO DIFFERENCE | 0.8126 | -0.0062 [-0.0130, +0.0009] not separated | NO DIFFERENCE | 0.8125 | -0.0063 [-0.0143, +0.0017] not separated | NO DIFFERENCE |
| dada | 0.7744 | 0.7658 | -0.0086 [-0.0158, -0.0014] below | NO DIFFERENCE | 0.7803 | +0.0059 [-0.0026, +0.0145] not separated | NO DIFFERENCE | 0.7413 | -0.0331 [-0.0427, -0.0232] below | COST |
| tad (printed) | 0.6981 | 0.7016 | +0.0035 [-0.0167, +0.0259] not separated | NO DIFFERENCE | 0.7067 | +0.0086 [-0.0170, +0.0348] not separated | NO DIFFERENCE | 0.6926 | -0.0055 [-0.0338, +0.0238] not separated | NO DIFFERENCE |
| dad | 0.8070 | 0.8033 | -0.0037 [-0.0198, +0.0124] not separated | NO DIFFERENCE | 0.7128 | -0.0942 [-0.1161, -0.0732] below | COST | 0.7557 | -0.0513 [-0.0732, -0.0301] below | COST |

Printed: Nexar val negatives with peak > 0.5: K5f 0.549, K5fepre 0.504, K5fepost 0.708, K5fbk 0.549.

Never from this report: that edge windows reduce the early rise (one hit or a TRADE-OFF), which loss acts, "K5f anticipates".
