# §64 KIM without VideoMAE — host nexar, s2024 — **PRELIMINARY (n = 1 seed); decides nothing, K5f stays the reference**

Rows: K5 = **K5dfirr**, K0 = **K0dfi**, L5 = **L5dfi**, L5cap = **L5dficap**, L0 = **L0dfirr**.
Deciding sets: nexar, dota, dada; labelled only: dad.

## Start gates (CH3: step-1 `mil`)

| pair | control | new | status |
|---|---|---|---|
| L5dfi vs L0dfirr | 0.720915 | 0.720915 | matched |
| L5dficap vs L0dfirr | 0.720915 | 0.720915 | matched |
| K5dfirr vs K0dfi | 0.689790 | 0.689790 | matched |

## Verdicts (CH8)

| question | micro | macro |
|---|---|---|
| Q1 KIM helps the CLIP host | YES | YES |
| Q2 VideoMAE masks KIM | SUPPORTED | SUPPORTED |
| Q3 what carries A | DIFFERENCES | DIFFERENCES |

## Cells (micro / macro)

| row | nexar | dota | dada | dad |
|---|---|---|---|---|
| L5 = L5dfi | 0.7598 / 0.7527 | 0.6615 / 0.7385 | 0.6306 / 0.6572 | 0.5681 / 0.4923 |
| L5cap = L5dficap | 0.7340 / 0.7353 | 0.6290 / 0.6740 | 0.5901 / 0.5950 | 0.5869 / 0.5427 |
| L0 = L0dfirr | 0.6873 / 0.6676 | 0.6354 / 0.6714 | 0.5698 / 0.5735 | 0.6014 / 0.5478 |
| K5 = K5dfirr | 0.8834 / 0.8669 | 0.7657 / 0.8264 | 0.7603 / 0.7665 | 0.7076 / 0.7545 |
| K0 = K0dfi | 0.8505 / 0.8415 | 0.7697 / 0.8053 | 0.7374 / 0.7378 | 0.7112 / 0.7375 |

## Contrasts, micro (labels §53 KA5; printed rows never decide)

| contrast | nexar | dota | dada | dad |
|---|---|---|---|---|
| L5 - L0 (KIM, CLIP host) | +0.0725 [+0.0544, +0.0913] GAIN | +0.0260 [+0.0147, +0.0372] GAIN | +0.0608 [+0.0491, +0.0725] GAIN | -0.0333 [-0.0562, -0.0107] COST |
| K5 - K0 (KIM, VideoMAE host) | +0.0330 [+0.0263, +0.0405] GAIN | -0.0041 [-0.0103, +0.0019] NO DIFFERENCE | +0.0229 [+0.0164, +0.0297] GAIN | -0.0036 [-0.0146, +0.0068] NO DIFFERENCE |
| I = A - B (interaction) | +0.0395 [+0.0213, +0.0584] GAIN | +0.0301 [+0.0195, +0.0403] GAIN | +0.0379 [+0.0265, +0.0503] GAIN | -0.0297 [-0.0559, -0.0051] COST |
| L5 - L5cap (differences) | +0.0258 [+0.0070, +0.0438] GAIN | +0.0324 [+0.0210, +0.0437] GAIN | +0.0404 [+0.0278, +0.0535] GAIN | -0.0188 [-0.0433, +0.0048] NO DIFFERENCE |
| L5cap - L0 (branch alone) *(printed)* | +0.0467 [+0.0351, +0.0581] GAIN | -0.0064 [-0.0139, +0.0014] NO DIFFERENCE | +0.0204 [+0.0122, +0.0287] GAIN | -0.0145 [-0.0308, +0.0039] NO DIFFERENCE |
| K0 - L0 (VideoMAE alone) *(printed)* | +0.1632 [+0.1365, +0.1909] GAIN | +0.1343 [+0.1186, +0.1494] GAIN | +0.1676 [+0.1475, +0.1862] GAIN | +0.1098 [+0.0692, +0.1534] GAIN |
| K5 - L5 (VideoMAE given KIM) *(printed)* | +0.1237 [+0.0949, +0.1525] GAIN | +0.1042 [+0.0871, +0.1205] GAIN | +0.1297 [+0.1113, +0.1476] GAIN | +0.1395 [+0.1002, +0.1799] GAIN |

## Contrasts, macro (labels §53 KA5; printed rows never decide)

| contrast | nexar | dota | dada | dad |
|---|---|---|---|---|
| L5 - L0 (KIM, CLIP host) | +0.0851 [+0.0660, +0.1044] GAIN | +0.0671 [+0.0540, +0.0804] GAIN | +0.0837 [+0.0701, +0.0972] GAIN | -0.0555 [-0.0958, -0.0159] COST |
| K5 - K0 (KIM, VideoMAE host) | +0.0254 [+0.0178, +0.0332] GAIN | +0.0211 [+0.0160, +0.0264] GAIN | +0.0287 [+0.0224, +0.0347] GAIN | +0.0170 [-0.0007, +0.0354] NO DIFFERENCE |
| I = A - B (interaction) | +0.0597 [+0.0414, +0.0781] GAIN | +0.0460 [+0.0332, +0.0591] GAIN | +0.0550 [+0.0417, +0.0686] GAIN | -0.0725 [-0.1152, -0.0301] COST |
| L5 - L5cap (differences) | +0.0175 [+0.0001, +0.0341] GAIN | +0.0645 [+0.0512, +0.0781] GAIN | +0.0622 [+0.0486, +0.0758] GAIN | -0.0503 [-0.0914, -0.0089] COST |
| L5cap - L0 (branch alone) *(printed)* | +0.0676 [+0.0525, +0.0828] GAIN | +0.0027 [-0.0049, +0.0104] NO DIFFERENCE | +0.0215 [+0.0123, +0.0309] GAIN | -0.0051 [-0.0294, +0.0193] NO DIFFERENCE |
| K0 - L0 (VideoMAE alone) *(printed)* | +0.1738 [+0.1494, +0.1981] GAIN | +0.1339 [+0.1164, +0.1517] GAIN | +0.1643 [+0.1424, +0.1860] GAIN | +0.1897 [+0.1359, +0.2427] GAIN |
| K5 - L5 (VideoMAE given KIM) *(printed)* | +0.1142 [+0.0860, +0.1426] GAIN | +0.0879 [+0.0716, +0.1047] GAIN | +0.1093 [+0.0902, +0.1283] GAIN | +0.2622 [+0.2112, +0.3126] GAIN |

Never: "KIM models motion"; "VideoMAE hides KIM's motion signal" unless Q2 is SUPPORTED and Q3 is DIFFERENCES on the same host and metric; a sentence from a labelled-only set; a seed claim.
