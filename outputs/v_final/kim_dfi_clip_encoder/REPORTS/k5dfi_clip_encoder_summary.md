Control K5dfirr: /content/drive/MyDrive/Thesis-Final/outputs/kim_dfi_bottomk25/K5dfirr

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


# Four sets, macro and micro

# k5dfi_clip_encoder_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K5dfirr**. Pooling of micro / AP: DoTA-CAP-dev and DADA-2000 after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP-dev | DADA-2000 | TAD | DAD |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |
| K5dfice | 0.8202 / 0.7788 | 0.7500 / 0.7464 | 0.7149 / 0.5488 | 0.7573 / 0.7302 |
| K5dficx | 0.8154 / 0.7693 | 0.7494 / 0.7427 | 0.7145 / 0.5563 | 0.7786 / 0.7348 |
| K5dfi | 0.8259 / 0.7613 | 0.7677 / 0.7591 | 0.7198 / 0.6119 | 0.7494 / 0.7091 |
| K5f | 0.8188 / 0.7353 | 0.7744 / 0.7551 | 0.6981 / 0.6711 | 0.8070 / 0.7072 |

## DoTA-CAP-dev (569 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |
| K5dfice | 0.8202 [0.8058, 0.8346] | 0.7788 | 0.6119 | 0.3271 |
| K5dficx | 0.8154 [0.8010, 0.8297] | 0.7693 | 0.6011 | 0.3271 |
| K5dfi | 0.8259 [0.8120, 0.8398] | 0.7613 | 0.5904 | 0.3271 |
| K5f | 0.8188 [0.8043, 0.8327] | 0.7353 | 0.5693 | 0.3271 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfice - K5dfirr | -0.0057 [-0.0102, -0.0012] | below | +0.0174 [+0.0117, +0.0233] | above |
| K5dficx - K5dfirr | -0.0106 [-0.0145, -0.0065] | below | +0.0079 [+0.0036, +0.0123] | above |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | -0.0071 [-0.0153, +0.0009] | not separated | -0.0261 [-0.0335, -0.0188] | below |

## DADA-2000 (370 whole sources, s8; micro / AP after per-video min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5dfice | 0.7500 [0.7277, 0.7716] | 0.7464 | 0.5480 | 0.2885 |
| K5dficx | 0.7494 [0.7271, 0.7713] | 0.7427 | 0.5412 | 0.2885 |
| K5dfi | 0.7677 [0.7439, 0.7905] | 0.7591 | 0.5548 | 0.2885 |
| K5f | 0.7744 [0.7499, 0.7986] | 0.7551 | 0.5524 | 0.2885 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfice - K5dfirr | -0.0177 [-0.0239, -0.0116] | below | -0.0127 [-0.0190, -0.0063] | below |
| K5dficx - K5dfirr | -0.0183 [-0.0242, -0.0124] | below | -0.0163 [-0.0219, -0.0109] | below |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | +0.0067 [-0.0024, +0.0156] | not separated | -0.0040 [-0.0128, +0.0044] | not separated |

## TAD (100 videos (macro over the 60 two-class), s8; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |
| K5dfice | 0.7149 [0.6524, 0.7753] | 0.5488 | 0.1660 | 0.0724 |
| K5dficx | 0.7145 [0.6473, 0.7794] | 0.5563 | 0.1695 | 0.0724 |
| K5dfi | 0.7198 [0.6508, 0.7864] | 0.6119 | 0.2212 | 0.0724 |
| K5f | 0.6981 [0.6250, 0.7675] | 0.6711 | 0.2612 | 0.0724 |

Clip oracle micro (a constant score per video): 0.9226. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfice - K5dfirr | -0.0049 [-0.0210, +0.0115] | not separated | -0.0631 [-0.0904, -0.0346] | below |
| K5dficx - K5dfirr | -0.0053 [-0.0190, +0.0085] | not separated | -0.0556 [-0.0823, -0.0308] | below |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | -0.0217 [-0.0485, +0.0018] | not separated | +0.0592 [+0.0224, +0.0943] | above |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K5dfirr | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |
| K5dfice | 0.7573 [0.7186, 0.7944] | 0.7302 | 0.1224 | 0.0472 |
| K5dficx | 0.7786 [0.7413, 0.8140] | 0.7348 | 0.1266 | 0.0472 |
| K5dfi | 0.7494 [0.7117, 0.7851] | 0.7091 | 0.1167 | 0.0472 |
| K5f | 0.8070 [0.7685, 0.8429] | 0.7072 | 0.1149 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfice - K5dfirr | +0.0079 [-0.0086, +0.0242] | not separated | +0.0211 [+0.0121, +0.0298] | above |
| K5dficx - K5dfirr | +0.0291 [+0.0156, +0.0431] | above | +0.0257 [+0.0198, +0.0316] | above |
| K5dfi - K5dfirr | +0.0000 [+0.0000, +0.0000] | not separated | +0.0000 [+0.0000, +0.0000] | not separated |
| K5f - K5dfirr | +0.0576 [+0.0378, +0.0779] | above | -0.0019 [-0.0119, +0.0086] | not separated |
