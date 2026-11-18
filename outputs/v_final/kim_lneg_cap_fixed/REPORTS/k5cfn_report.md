# K5f / K5cf / K5cfn: dashcam Normal caption -- s2024

Rule: `core/docs/final/KIM_RUN.md §33`. PRELIMINARY, n = 1. Row 0 = "Normal driving recorded by a dashcam that does not involve any collision or unusual event.". Last-epoch mul_mil: K5f 0.0963, K5cf 0.2555, K5cfn 0.2276; cap_same_source_neg equal to K5cf's at all 1749 steps.

**R4: K5cfn candidate = False -> caption L_neg line CLOSED** (K5cfn - K5f above on [], below on ['nexar', 'dota', 'dada']; sets ['nexar', 'dota', 'dada', 'dad']).

Primary (K5cfn - K5cf, decides nothing): the Normal sentence is above on [], below on ['dota'].

## Macro per row (Nexar = val crops)

| set | K5f | K5cf | K5cfn |
|---|---|---|---|
| nexar | 0.8942 | 0.8612 | 0.8585 |
| dota | 0.8188 | 0.8083 | 0.8024 |
| dada | 0.7744 | 0.7470 | 0.7443 |
| tad (printed) | 0.6981 | 0.6947 | 0.6980 |
| dad | 0.8070 | 0.8051 | 0.7972 |

## Contrasts, paired per item, 95 % cluster bootstrap

| set | K5cfn - K5cf | K5cfn - K5f |
|---|---|---|
| nexar | -0.0027 [-0.0061, +0.0008] not separated | -0.0357 [-0.0442, -0.0275] below |
| dota | -0.0058 [-0.0085, -0.0032] below | -0.0164 [-0.0255, -0.0069] below |
| dada | -0.0027 [-0.0064, +0.0008] not separated | -0.0301 [-0.0400, -0.0204] below |
| tad | +0.0033 [-0.0081, +0.0176] not separated | -0.0001 [-0.0241, +0.0249] not separated |
| dad | -0.0079 [-0.0221, +0.0054] not separated | -0.0098 [-0.0368, +0.0156] not separated |

Printed: Nexar val negatives with peak > 0.5: K5f 0.549, K5cf 0.611, K5cfn 0.558.
