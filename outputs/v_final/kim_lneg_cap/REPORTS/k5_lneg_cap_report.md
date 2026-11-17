# K5 / K5d / K5c / K5dc: caption L_neg x DVS -- s2024

Rule: `core/docs/final/KIM_RUN.md §30`. PRELIMINARY, n = 1. Captions: §29.3 prompt v6, Qwen2.5-VL-72B NF4, 32 frames, **QC waived by the user** (2026-11-17). NG4: K5's nexar_val re-read 0.8981 vs §13 0.8981.

**C6 verdict: COMPLEMENTARY** -- COMPLEMENTARY iff I is above on >= 2 of ['nexar', 'dota', 'dada', 'dad'] and below on none; ANTAGONISTIC iff below on >= 2; else NO CLAIM. I above on ['nexar', 'dota', 'dada', 'dad'], below on [].

**C7: K5dc candidate = False** (K5dc - K5 above on [], below on ['nexar', 'dota', 'dada', 'dad']). K5 stays unless this is True and the user decides.

## Macro per row (Nexar = val crops)

| set | K5 | K5d | K5c | K5dc |
|---|---|---|---|---|
| nexar | 0.8981 | 0.8896 | 0.8116 | 0.8607 |
| dota | 0.8195 | 0.8124 | 0.7684 | 0.7840 |
| dada | 0.7740 | 0.7713 | 0.6852 | 0.7225 |
| tad (printed) | 0.7002 | 0.7055 | 0.6729 | 0.6818 |
| dad | 0.7916 | 0.7329 | 0.7408 | 0.7557 |

## Contrasts, paired per item, 95 % cluster bootstrap

| set | K5c - K5 | K5d - K5 | K5dc - K5 | K5dc - K5d | I |
|---|---|---|---|---|---|
| nexar | -0.0865 [-0.1043, -0.0691] below | -0.0085 [-0.0182, +0.0004] not separated | -0.0374 [-0.0518, -0.0218] below | -0.0288 [-0.0434, -0.0130] below | +0.0577 [+0.0400, +0.0761] above |
| dota | -0.0511 [-0.0646, -0.0378] below | -0.0071 [-0.0153, +0.0016] not separated | -0.0355 [-0.0490, -0.0219] below | -0.0284 [-0.0401, -0.0171] below | +0.0227 [+0.0104, +0.0354] above |
| dada | -0.0888 [-0.1046, -0.0728] below | -0.0027 [-0.0121, +0.0068] not separated | -0.0515 [-0.0676, -0.0351] below | -0.0488 [-0.0647, -0.0328] below | +0.0400 [+0.0268, +0.0535] above |
| tad | -0.0272 [-0.0622, +0.0080] not separated | +0.0053 [-0.0210, +0.0322] not separated | -0.0183 [-0.0564, +0.0194] not separated | -0.0236 [-0.0626, +0.0131] not separated | +0.0036 [-0.0339, +0.0410] not separated |
| dad | -0.0508 [-0.0895, -0.0133] below | -0.0587 [-0.0841, -0.0340] below | -0.0359 [-0.0713, -0.0005] below | +0.0228 [-0.0145, +0.0597] not separated | +0.0737 [+0.0427, +0.1070] above |

Printed: Nexar val negatives with peak > 0.5: K5 0.717, K5d 0.770, K5c 0.558, K5dc 0.850.

Direction only, for comparison: LaGoVAD Tab. G Det. Avg. (one run, PreVAD): L_neg +2.46 with DVS, +2.20 without; DVS +2.91 with L_neg, +2.65 without; interaction about +0.26.
