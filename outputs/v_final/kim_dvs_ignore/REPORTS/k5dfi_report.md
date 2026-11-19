# K5f / K5df / K5dfi / K5dfie: DVS anchor mode and ego setting -- s2024

Rule: `core/docs/final/KIM_RUN.md §34`. PRELIMINARY, n = 1. Last-epoch mul_mil: K5f 0.0963, K5df 0.1700, K5dfi 0.1519, K5dfie 0.1228.

**R4 primary (K5dfi - K5df, `dvs_anchor_mode` ignore vs span): COST.** Secondary (K5dfie - K5dfi, `is_egocentric` true vs false): NO DIFFERENCE.

Candidate (K5dfi): not a candidate (K5f stays) (above K5f on [], below on ['nexar', 'dad']).
Candidate (K5dfie): not a candidate (K5f stays) (above K5f on ['dota'], below on ['nexar', 'dad']).

Start check (K5dfi vs K5df, step 1): mask live (dvs_sup 0.7044 -> 0.7039); same draws = True (mil True, dvs_sup_mil True). K5dfie has no matched start (`theta_ego` / `delta_m_ego` change the splice draws).

## Macro per row (Nexar = val crops)

| set | K5f | K5df | K5dfi | K5dfie |
|---|---|---|---|---|
| nexar | 0.8942 | 0.8871 | 0.8698 | 0.8723 |
| dota | 0.8188 | 0.8164 | 0.8259 | 0.8263 |
| dada | 0.7744 | 0.7728 | 0.7677 | 0.7724 |
| tad (printed) | 0.6981 | 0.7033 | 0.7198 | 0.7214 |
| dad | 0.8070 | 0.7562 | 0.7494 | 0.7508 |

## Contrasts, paired per item, 95 % cluster bootstrap

| set | K5dfi - K5df | K5dfie - K5dfi | K5dfie - K5df | K5dfi - K5f | K5dfie - K5f | K5df - K5f |
|---|---|---|---|---|---|---|
| nexar | -0.0174 [-0.0267, -0.0088] below | +0.0025 [-0.0052, +0.0099] not separated | -0.0149 [-0.0247, -0.0056] below | -0.0245 [-0.0335, -0.0160] below | -0.0220 [-0.0293, -0.0149] below | -0.0071 [-0.0164, +0.0023] not separated |
| dota | +0.0095 [+0.0033, +0.0159] above | +0.0004 [-0.0046, +0.0053] not separated | +0.0098 [+0.0030, +0.0161] above | +0.0071 [-0.0009, +0.0153] not separated | +0.0075 [+0.0021, +0.0132] above | -0.0024 [-0.0100, +0.0058] not separated |
| dada | -0.0051 [-0.0132, +0.0033] not separated | +0.0047 [-0.0020, +0.0114] not separated | -0.0004 [-0.0092, +0.0084] not separated | -0.0067 [-0.0156, +0.0024] not separated | -0.0020 [-0.0086, +0.0046] not separated | -0.0016 [-0.0114, +0.0081] not separated |
| tad | +0.0165 [-0.0064, +0.0399] not separated | +0.0016 [-0.0118, +0.0144] not separated | +0.0182 [-0.0039, +0.0403] not separated | +0.0217 [-0.0018, +0.0485] not separated | +0.0233 [+0.0069, +0.0412] above | +0.0052 [-0.0224, +0.0337] not separated |
| dad | -0.0068 [-0.0284, +0.0145] not separated | +0.0014 [-0.0159, +0.0189] not separated | -0.0054 [-0.0280, +0.0172] not separated | -0.0576 [-0.0779, -0.0378] below | -0.0562 [-0.0695, -0.0431] below | -0.0508 [-0.0751, -0.0273] below |

## R4 labels (margin 0.01; GAIN needs one of ['dota', 'dada']; deciding sets ['nexar', 'dota', 'dada', 'dad'])

| set | ignore | ego |
|---|---|---|
| nexar | COST | NO DIFFERENCE |
| dota | NO DIFFERENCE | NO DIFFERENCE |
| dada | NO DIFFERENCE | NO DIFFERENCE |
| tad | NO DIFFERENCE | NO DIFFERENCE |
| dad | NO DIFFERENCE | NO DIFFERENCE |

Printed: Nexar val negatives with peak > 0.5: K5f 0.549, K5df 0.673, K5dfi 0.681, K5dfie 0.584.
