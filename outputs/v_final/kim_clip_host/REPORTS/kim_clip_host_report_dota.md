# §64 KIM without VideoMAE — host dota, s2024 — **PRELIMINARY (n = 1 seed); decides nothing, K5f stays the reference**

Rows: K5 = **K5dfiDoTA**, K0 = **K0dfiDoTA**, L5 = **L5dfiDoTA**, L5cap = **L5dficapDoTA**, L0 = **L0dfiDoTArr**.
Deciding sets: dota, dada.

## Start gates (CH3: step-1 `mil`)

| pair | control | new | status |
|---|---|---|---|
| L5dfiDoTA vs L0dfiDoTArr | 0.778298 | 0.778298 | matched |
| L5dficapDoTA vs L0dfiDoTArr | 0.778298 | 0.778298 | matched |
| K5dfiDoTA vs K0dfiDoTA | 0.731826 | 0.731826 | matched |

## Verdicts (CH8)

| question | micro | macro |
|---|---|---|
| Q1 KIM helps the CLIP host | NO | NO |
| Q2 VideoMAE masks KIM | REVERSED | REVERSED |
| Q3 what carries A | DIFFERENCES COST | DIFFERENCES COST |

## Cells (micro / macro)

| row | dota | dada |
|---|---|---|
| L5 = L5dfiDoTA | 0.4281 / 0.4001 | 0.4691 / 0.4465 |
| L5cap = L5dficapDoTA | 0.6464 / 0.6781 | 0.5993 / 0.6040 |
| L0 = L0dfiDoTArr | 0.6372 / 0.6638 | 0.5829 / 0.5880 |
| K5 = K5dfiDoTA | 0.8209 / 0.8590 | 0.7522 / 0.7659 |
| K0 = K0dfiDoTA | 0.8156 / 0.8532 | 0.7553 / 0.7675 |

## Contrasts, micro (labels §53 KA5; printed rows never decide)

| contrast | dota | dada |
|---|---|---|
| L5 - L0 (KIM, CLIP host) | -0.2091 [-0.2298, -0.1904] COST | -0.1137 [-0.1301, -0.0978] COST |
| K5 - K0 (KIM, VideoMAE host) | +0.0053 [+0.0007, +0.0098] NO DIFFERENCE | -0.0031 [-0.0098, +0.0034] NO DIFFERENCE |
| I = A - B (interaction) | -0.2144 [-0.2362, -0.1956] COST | -0.1107 [-0.1287, -0.0930] COST |
| L5 - L5cap (differences) | -0.2183 [-0.2403, -0.1987] COST | -0.1302 [-0.1480, -0.1130] COST |
| L5cap - L0 (branch alone) *(printed)* | +0.0092 [+0.0032, +0.0149] NO DIFFERENCE | +0.0165 [+0.0114, +0.0216] GAIN |
| K0 - L0 (VideoMAE alone) *(printed)* | +0.1784 [+0.1670, +0.1909] GAIN | +0.1724 [+0.1544, +0.1897] GAIN |
| K5 - L5 (VideoMAE given KIM) *(printed)* | +0.3928 [+0.3741, +0.4148] GAIN | +0.2831 [+0.2582, +0.3082] GAIN |

## Contrasts, macro (labels §53 KA5; printed rows never decide)

| contrast | dota | dada |
|---|---|---|
| L5 - L0 (KIM, CLIP host) | -0.2637 [-0.2854, -0.2413] COST | -0.1415 [-0.1607, -0.1226] COST |
| K5 - K0 (KIM, VideoMAE host) | +0.0058 [+0.0015, +0.0101] NO DIFFERENCE | -0.0016 [-0.0083, +0.0050] NO DIFFERENCE |
| I = A - B (interaction) | -0.2695 [-0.2918, -0.2463] COST | -0.1399 [-0.1606, -0.1194] COST |
| L5 - L5cap (differences) | -0.2779 [-0.2994, -0.2557] COST | -0.1575 [-0.1779, -0.1375] COST |
| L5cap - L0 (branch alone) *(printed)* | +0.0143 [+0.0088, +0.0195] GAIN | +0.0160 [+0.0109, +0.0213] GAIN |
| K0 - L0 (VideoMAE alone) *(printed)* | +0.1894 [+0.1763, +0.2026] GAIN | +0.1795 [+0.1614, +0.1978] GAIN |
| K5 - L5 (VideoMAE given KIM) *(printed)* | +0.4588 [+0.4373, +0.4795] GAIN | +0.3194 [+0.2944, +0.3452] GAIN |

Never: "KIM models motion"; "VideoMAE hides KIM's motion signal" unless Q2 is SUPPORTED and Q3 is DIFFERENCES on the same host and metric; a sentence from a labelled-only set; a seed claim.
