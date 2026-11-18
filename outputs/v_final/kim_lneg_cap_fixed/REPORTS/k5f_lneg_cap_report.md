# K5f / K5df / K5cf / K5dcf: caption L_neg x DVS -- s2024

Rule: `core/docs/final/KIM_RUN.md §32`. PRELIMINARY, n = 1. Captions: §29.3 prompt v6, Qwen2.5-VL-72B NF4, 32 frames, **QC waived by the user** (2026-11-17). **Fixed label path (§31): control = K5f, which replaces K5; every row's last-epoch mul_mil above the floor: K5f 0.0963, K5df 0.1700, K5cf 0.2555, K5dcf 0.5566.**

**C6 verdict: COMPLEMENTARY** -- COMPLEMENTARY iff I is above on >= 2 of ['nexar', 'dota', 'dada', 'dad'] and below on none; ANTAGONISTIC iff below on >= 2; else NO CLAIM. I above on ['nexar', 'dada', 'dad'], below on [].

**C7: K5dcf candidate = False** (K5dcf - K5f above on [], below on ['nexar', 'dada', 'dad']). K5f stays unless this is True and the user decides.

## Macro per row (Nexar = val crops)

| set | K5f | K5df | K5cf | K5dcf |
|---|---|---|---|---|
| nexar | 0.8942 | 0.8871 | 0.8612 | 0.8785 |
| dota | 0.8188 | 0.8164 | 0.8083 | 0.8131 |
| dada | 0.7744 | 0.7728 | 0.7470 | 0.7564 |
| tad (printed) | 0.6981 | 0.7033 | 0.6947 | 0.7103 |
| dad | 0.8070 | 0.7562 | 0.8051 | 0.7800 |

## Contrasts, paired per item, 95 % cluster bootstrap

| set | K5cf - K5f | K5df - K5f | K5dcf - K5f | K5dcf - K5df | I |
|---|---|---|---|---|---|
| nexar | -0.0330 [-0.0412, -0.0248] below | -0.0071 [-0.0164, +0.0023] not separated | -0.0158 [-0.0261, -0.0048] below | -0.0087 [-0.0175, +0.0002] not separated | +0.0244 [+0.0152, +0.0337] above |
| dota | -0.0105 [-0.0195, -0.0011] below | -0.0024 [-0.0100, +0.0058] not separated | -0.0057 [-0.0149, +0.0041] not separated | -0.0033 [-0.0097, +0.0030] not separated | +0.0072 [-0.0010, +0.0154] not separated |
| dada | -0.0274 [-0.0377, -0.0170] below | -0.0016 [-0.0114, +0.0081] not separated | -0.0181 [-0.0295, -0.0067] below | -0.0164 [-0.0258, -0.0069] below | +0.0110 [+0.0012, +0.0207] above |
| tad | -0.0034 [-0.0276, +0.0215] not separated | +0.0052 [-0.0224, +0.0337] not separated | +0.0122 [-0.0157, +0.0409] not separated | +0.0070 [-0.0138, +0.0278] not separated | +0.0104 [-0.0118, +0.0331] not separated |
| dad | -0.0019 [-0.0275, +0.0228] not separated | -0.0508 [-0.0751, -0.0273] below | -0.0270 [-0.0536, -0.0016] below | +0.0238 [-0.0023, +0.0492] not separated | +0.0256 [+0.0054, +0.0462] above |

Printed: Nexar val negatives with peak > 0.5: K5f 0.549, K5df 0.673, K5cf 0.611, K5dcf 0.619.

Direction only, for comparison: LaGoVAD Tab. G Det. Avg. (one run, PreVAD): L_neg +2.46 with DVS, +2.20 without; DVS +2.91 with L_neg, +2.65 without; interaction about +0.26.
