# K5 vs LaGoVAD public checkpoint on the zero-shot sets

Status 2026-11-07: all four sets read; DAD by `colab/final/k5_dad.ipynb`. Raw report
`outputs/v_final/kim/REPORTS/k5_zs_report.md`, regenerated locally from the saved scores byte for
byte. **PRELIMINARY (n = 1 seed, s2024).** K5 is the user's chosen
main model (`KIM_RUN.md` §12.3), not a rule's outcome. Requested by the user for a quick report to
the advisor: K5 and the public checkpoint only.

Table generator: `core.tools.k5_zs_report`, pure aggregation with the protocols of
`ZS_COMPARISON.md`. It refuses to write when the rows scored different items, when a recomputed
micro disagrees with its read, when K5's reads come from different checkpoints, or when LaGoVAD
misses Gate T0 on TAD. DAD adds `ZS_DAD.md` §8's ruler, placebo and same-position reads.

| Dataset | Metric | LaGoVAD (public ckpt) | K5 | Base rate |
|---|---|---|---|---|
| **DoTA-CAP-dev** | AUC macro | 0.6418 [0.6166, 0.6666] | 0.8195 [0.8048, 0.8338] | |
| | AUC micro | 0.6105 | 0.7449 | |
| | AP | 0.3838 | 0.5790 | 0.3271 |
| **DADA-2000** | AUC macro | 0.6202 [0.5925, 0.6480] | 0.7740 [0.7492, 0.7982] | |
| | AUC micro | 0.6128 | 0.7635 | |
| | AP | 0.3532 | 0.5640 | 0.2885 |
| **TAD** | AUC macro | 0.7579 [0.6955, 0.8147] | 0.7002 [0.6246, 0.7724] | |
| | AUC micro | 0.7912 | 0.6781 | |
| | AP | 0.2571 | 0.2568 | 0.0724 |
| **DAD** | AUC macro | 0.6462 [0.5881, 0.7016] | 0.7916 [0.7513, 0.8294] | |
| | AUC micro | 0.5681 | 0.7050 | |
| | AP | 0.0640 | 0.1156 | 0.0472 |

**DADA-2000 micro and AP are pooled after per-video min-max since 2026-11-12** (user's decision on
`KIM_RUN.md` §21.1 BM7(i)). DADA-2000 whole sources hold no normal video, so this is the pooling
DoTA-CAP-dev already uses and the one `core.evaluate --score-norm auto` picks (C12). Before that the
cells were raw: LaGoVAD 0.5388 / AP 0.3138, K5 0.7222 / AP 0.5557. Never put a raw-pooled DADA micro
beside these. Macro does not change. No paired micro interval was computed, so no K5 - LaGoVAD sign
is written for micro.

K5 - LaGoVAD, macro, paired per item: DoTA-CAP-dev +0.1777 [+0.1495, +0.2067] above; DADA-2000
+0.1538 [+0.1212, +0.1861] above; TAD -0.0577 [-0.1389, +0.0239] not separated; DAD +0.1455 [+0.0814, +0.2100] above, read the next
section before quoting it.

## DAD: what the macro does not measure

Printed, post-hoc, n = 1, never a verdict (`ZS_DAD.md` §8). The no-pixel `t/N` ruler scores macro
1.000. K5's scores rise over the video without an accident: mean on negatives 0.019 to 0.164 over
the 15 rows; LaGoVAD 0.050 to 0.049.

| read | LaGoVAD | K5 | K5 - LaGoVAD |
|---|---|---|---|
| macro, all rows | 0.6462 | 0.7916 | +0.1455 [+0.0814, +0.2100] |
| placebo, all rows | 0.5134 [0.4709, 0.5564] | 0.6877 [0.6544, 0.7204] | +0.1743 |
| macro - placebo, all rows | +0.1327 [+0.0615, +0.2020] | +0.1039 [+0.0517, +0.1542] | -0.0288 [-0.1126, +0.0560] |
| macro, rows 7+ | 0.6439 | 0.6864 | +0.0424 [-0.0288, +0.1157] |
| macro - placebo, rows 7+ | +0.1179 [+0.0440, +0.1895] | +0.0839 [+0.0183, +0.1491] | -0.0340 [-0.1268, +0.0606] |
| same position, accident rows | 0.5881 [0.5339, 0.6419] | 0.6678 [0.6160, 0.7185] | +0.0797 [+0.0162, +0.1427] |
| same position, rows 7 up to the accident | 0.5762 [0.5220, 0.6308] | 0.6506 [0.5990, 0.7021] | not computed |

The K5 - LaGoVAD cells of the macro-minus-placebo rows and of macro, rows 7+ are not in the tool's
report: computed once from the saved scores with the tool's bootstrap seed and draw order, resampling
positives and negatives separately. Write: "K5 separates accident videos from normal ones better
than LaGoVAD; the macro gain on DAD is not shown to come from reacting to the accident". Never
write that K5 detects or localizes accidents on DAD.

**Not a matched comparison:** the public checkpoint trained on PreVAD, K5 on Nexar, and its
zero-shot status on these sets is unverified (`ZS_TAD_DADA.md` §7). Never write "beats LaGoVAD".
K5 was chosen after DoTA-CAP-dev, DADA-2000 and TAD were read, so they are not an independent test of it. Micro on TAD
and DAD measures clip classification (clip oracle 0.9226, 0.8390). DAD was read after K5 was chosen. The DoTA bracket differs from
`protocol_b_readout.md` (0.8047, 0.8337) in the last digit: this table uses the bootstrap seed of
`zs_lagovad_readout`, the same one as the LaGoVAD cell.

## Added comparison: K5 trained on T2

The user asked for this after the TAD drop. Rule and full read-out: `KIM_RUN.md` §13 and §13.1. Raw `outputs/v_final/kim_t2/REPORTS/k5_t2_report.md`. **PRELIMINARY (n = 1), decides nothing;** K5 on Nexar stays the main model. DADA-2000 is not read here, and `nexar_val` is new to this table.

| Set | Metric | LaGoVAD | K5 (Nexar) | K5 (T2) |
|---|---|---|---|---|
| Nexar val | macro, crops | 0.5859 | 0.8981 (in-domain) | 0.7816 |
| DoTA-CAP-dev | macro | 0.6418 | 0.8195 | 0.8085 |
| TAD | macro | 0.7579 | 0.7002 | 0.7606 |
| TAD | micro | 0.7912 | 0.6781 | 0.5187 |
| DAD | macro | 0.6462 | 0.7916 | 0.8198 |

K5 (T2) − K5 (Nexar), macro: only Nexar val separates (−0.1165 [−0.1502, −0.0842]); TAD +0.0604 [−0.0076, +0.1283], DoTA-CAP-dev and DAD are not separated either. On TAD, K5 (T2) scores normal videos above accident frames (median 0.899 vs 0.508). Never write "T2 fixes TAD".
