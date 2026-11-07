# K5 vs LaGoVAD public checkpoint on the zero-shot sets

Status 2026-11-07: DoTA-CAP-dev, DADA-2000 and TAD read from the saved scores; **DAD not read yet**
(runbook `colab/final/k5_dad.ipynb`). **PRELIMINARY (n = 1 seed, s2024).** K5 is the user's chosen
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
| | AUC micro | 0.5388 | 0.7222 | |
| | AP | 0.3138 | 0.5557 | 0.2885 |
| **TAD** | AUC macro | 0.7579 [0.6955, 0.8147] | 0.7002 [0.6246, 0.7724] | |
| | AUC micro | 0.7912 | 0.6781 | |
| | AP | 0.2571 | 0.2568 | 0.0724 |

K5 - LaGoVAD, macro, paired per item: DoTA-CAP-dev +0.1777 [+0.1495, +0.2067] above; DADA-2000
+0.1538 [+0.1212, +0.1861] above; TAD -0.0577 [-0.1389, +0.0239] not separated.

**Not a matched comparison:** the public checkpoint trained on PreVAD, K5 on Nexar, and its
zero-shot status on these sets is unverified (`ZS_TAD_DADA.md` §7). Never write "beats LaGoVAD".
K5 was chosen after these sets were read, so they are not an independent test of it. Micro on TAD
measures clip classification (clip oracle 0.9226). The DoTA bracket differs from
`protocol_b_readout.md` (0.8047, 0.8337) in the last digit: this table uses the bootstrap seed of
`zs_lagovad_readout`, the same one as the LaGoVAD cell.
