Control: K0dfirr (/content/drive/MyDrive/Thesis-Final/outputs/kim_dfi_proto/K0dfirr)

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


# DoTA-CAP-dev, DADA, DAD: macro and micro

# k0dfi_proto_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K0dfirr**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | DAD |
| --- | --- | --- | --- |
| K0dfirr | 0.8067 / 0.7697 | 0.7381 / 0.7389 | 0.7394 / 0.7137 |
| K0dfiP | 0.8068 / 0.7697 | 0.7382 / 0.7390 | 0.7406 / 0.7140 |
| K0dfiL | 0.8101 / 0.7748 | 0.7392 / 0.7431 | 0.7305 / 0.7087 |
| K0dfiPL | 0.8100 / 0.7749 | 0.7392 / 0.7428 | 0.7312 / 0.7082 |
| K0dfiPLb | 0.8101 / 0.7747 | 0.7391 / 0.7431 | 0.7312 / 0.7084 |
| K5dfi | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7494 / 0.7091 |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.8070 / 0.7072 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfirr | 0.8067 [0.7910, 0.8221] | 0.7697 | 0.6009 | 0.3271 |
| K0dfiP | 0.8068 [0.7912, 0.8223] | 0.7697 | 0.6007 | 0.3271 |
| K0dfiL | 0.8101 [0.7943, 0.8257] | 0.7748 | 0.6075 | 0.3271 |
| K0dfiPL | 0.8100 [0.7942, 0.8256] | 0.7749 | 0.6077 | 0.3271 |
| K0dfiPLb | 0.8101 [0.7943, 0.8257] | 0.7747 | 0.6073 | 0.3271 |
| K5dfi | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K0dfiP - K0dfirr | +0.0001 [-0.0001, +0.0003] | not separated | -0.0000 [-0.0002, +0.0002] | not separated |
| K0dfiL - K0dfirr | +0.0034 [+0.0016, +0.0052] | above | +0.0051 [+0.0032, +0.0072] | above |
| K0dfiPL - K0dfirr | +0.0033 [+0.0015, +0.0052] | above | +0.0052 [+0.0032, +0.0073] | above |
| K0dfiPLb - K0dfirr | +0.0034 [+0.0016, +0.0052] | above | +0.0051 [+0.0030, +0.0072] | above |
| K5dfi - K0dfirr | +0.0192 [+0.0140, +0.0245] | above | -0.0083 [-0.0146, -0.0020] | below |
| K5f - K0dfirr | +0.0121 [+0.0009, +0.0232] | above | -0.0344 [-0.0453, -0.0233] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfirr | 0.7381 [0.7156, 0.7599] | 0.7389 | 0.5368 | 0.2885 |
| K0dfiP | 0.7382 [0.7157, 0.7600] | 0.7390 | 0.5366 | 0.2885 |
| K0dfiL | 0.7392 [0.7165, 0.7614] | 0.7431 | 0.5426 | 0.2885 |
| K0dfiPL | 0.7392 [0.7163, 0.7614] | 0.7428 | 0.5422 | 0.2885 |
| K0dfiPLb | 0.7391 [0.7163, 0.7614] | 0.7431 | 0.5423 | 0.2885 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K0dfiP - K0dfirr | +0.0001 [-0.0003, +0.0004] | not separated | +0.0001 [-0.0001, +0.0003] | not separated |
| K0dfiL - K0dfirr | +0.0012 [-0.0010, +0.0034] | not separated | +0.0042 [+0.0012, +0.0070] | above |
| K0dfiPL - K0dfirr | +0.0011 [-0.0011, +0.0033] | not separated | +0.0039 [+0.0009, +0.0067] | above |
| K0dfiPLb - K0dfirr | +0.0010 [-0.0012, +0.0033] | not separated | +0.0042 [+0.0013, +0.0070] | above |
| K5dfi - K0dfirr | +0.0296 [+0.0233, +0.0360] | above | +0.0202 [+0.0134, +0.0269] | above |
| K5f - K0dfirr | +0.0364 [+0.0246, +0.0481] | above | +0.0162 [+0.0042, +0.0275] | above |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfirr | 0.7394 [0.6972, 0.7783] | 0.7137 | 0.1248 | 0.0472 |
| K0dfiP | 0.7406 [0.6986, 0.7795] | 0.7140 | 0.1249 | 0.0472 |
| K0dfiL | 0.7305 [0.6895, 0.7692] | 0.7087 | 0.1191 | 0.0472 |
| K0dfiPL | 0.7312 [0.6904, 0.7697] | 0.7082 | 0.1192 | 0.0472 |
| K0dfiPLb | 0.7312 [0.6900, 0.7697] | 0.7084 | 0.1194 | 0.0472 |
| K5dfi | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K0dfiP - K0dfirr | +0.0012 [-0.0002, +0.0026] | not separated | +0.0002 [-0.0001, +0.0006] | not separated |
| K0dfiL - K0dfirr | -0.0089 [-0.0166, -0.0012] | below | -0.0050 [-0.0092, -0.0009] | below |
| K0dfiPL - K0dfirr | -0.0082 [-0.0156, -0.0005] | below | -0.0055 [-0.0099, -0.0013] | below |
| K0dfiPLb - K0dfirr | -0.0082 [-0.0156, -0.0005] | below | -0.0053 [-0.0095, -0.0011] | below |
| K5dfi - K0dfirr | +0.0100 [-0.0075, +0.0282] | not separated | -0.0046 [-0.0149, +0.0053] | not separated |
| K5f - K0dfirr | +0.0676 [+0.0410, +0.0958] | above | -0.0066 [-0.0224, +0.0103] | not separated |


# DAD position diagnostics (control K0dfirr)

# DAD position diagnostics (POST-HOC, printed not gated, n = 1 seed)

165 positives, 301 negatives, 15 rows each; accident rows [13, 14]; first V2-S window without a repeated frame = row 7. **No-pixel `t/N` ruler macro = 1.0000.**

## Within video: positives vs the same mask on negatives (placebo)

| row | macro, all rows | placebo, all rows | gap | macro, rows 7+ | placebo, rows 7+ | gap |
|---|---|---|---|---|---|---|
| K0dfirr | 0.7394 | 0.5834 [0.5488, 0.6182] | +0.1560 [+0.1007, +0.2085] | 0.5990 | 0.4729 [0.4319, 0.5141] | +0.1261 [+0.0584, +0.1931] |
| K0dfiP | 0.7406 | 0.5834 [0.5487, 0.6182] | +0.1571 [+0.1018, +0.2098] | 0.6000 | 0.4720 [0.4308, 0.5136] | +0.1280 [+0.0598, +0.1948] |
| K0dfiL | 0.7305 | 0.5799 [0.5454, 0.6137] | +0.1507 [+0.0968, +0.2029] | 0.5879 | 0.4679 [0.4272, 0.5086] | +0.1200 [+0.0547, +0.1856] |
| K0dfiPL | 0.7312 | 0.5801 [0.5456, 0.6142] | +0.1511 [+0.0975, +0.2029] | 0.5889 | 0.4668 [0.4261, 0.5075] | +0.1221 [+0.0564, +0.1875] |
| K0dfiPLb | 0.7312 | 0.5797 [0.5451, 0.6140] | +0.1515 [+0.0973, +0.2037] | 0.5889 | 0.4690 [0.4280, 0.5097] | +0.1199 [+0.0544, +0.1858] |
| K5dfi | 0.7494 | 0.6222 [0.5887, 0.6555] | +0.1273 [+0.0770, +0.1757] | 0.6051 | 0.4956 [0.4543, 0.5365] | +0.1095 [+0.0434, +0.1743] |

## Same position: positive vs negative videos (position cannot help)

| row | accident rows | rows 7 up to the accident | difference |
|---|---|---|---|
| K0dfirr | 0.6709 [0.6205, 0.7209] | 0.6339 [0.5813, 0.6858] | +0.0370 [+0.0093, +0.0645] |
| K0dfiP | 0.6713 [0.6209, 0.7213] | 0.6342 [0.5814, 0.6861] | +0.0371 [+0.0095, +0.0646] |
| K0dfiL | 0.6696 [0.6192, 0.7205] | 0.6354 [0.5827, 0.6870] | +0.0342 [+0.0067, +0.0619] |
| K0dfiPL | 0.6683 [0.6175, 0.7194] | 0.6341 [0.5812, 0.6859] | +0.0342 [+0.0065, +0.0622] |
| K0dfiPLb | 0.6690 [0.6186, 0.7199] | 0.6355 [0.5827, 0.6869] | +0.0334 [+0.0060, +0.0611] |
| K5dfi | 0.6749 [0.6242, 0.7247] | 0.6495 [0.5966, 0.7007] | +0.0254 [+0.0047, +0.0462] |

Accident rows, paired:

* K0dfiP - K0dfirr: +0.0004 [-0.0005, +0.0013]
* K0dfiL - K0dfirr: -0.0013 [-0.0074, +0.0049]
* K0dfiPL - K0dfirr: -0.0026 [-0.0087, +0.0035]
* K0dfiPLb - K0dfirr: -0.0019 [-0.0082, +0.0042]
* K5dfi - K0dfirr: +0.0040 [-0.0098, +0.0180]
