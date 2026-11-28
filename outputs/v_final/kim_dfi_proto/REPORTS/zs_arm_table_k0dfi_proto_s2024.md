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
