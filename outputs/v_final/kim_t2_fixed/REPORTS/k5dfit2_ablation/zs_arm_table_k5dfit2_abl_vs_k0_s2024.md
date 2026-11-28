# k5dfit2_abl_vs_k0_s2024 — macro and micro on the zero-shot sets, s2024 — **PRELIMINARY (n = 1 seed); printed, decides nothing**

Control = **K0dfiT2**. Pooling of micro / AP: DoTA-CAP (dev+eval) after per-video min-max (all-abnormal), TAD and DAD raw. Δ = arm - control, paired per item; macro by the read-outs' bootstrap, micro by a cluster bootstrap (seed 2026).

## Summary (macro / micro)

| arm | DoTA-CAP (dev+eval) | DAD |
| --- | --- | --- |
| K0dfiT2 | 0.7840 / 0.7405 | 0.8242 / 0.7807 |
| K5dfiT2rr | 0.7969 / 0.7236 | 0.8107 / 0.7516 |
| K5dfiT2 | 0.7969 / 0.7236 | 0.8107 / 0.7516 |

## DoTA-CAP (dev+eval) (1129 clips, protocol B; micro / AP after per-clip min-max)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfiT2 | 0.7840 [0.7708, 0.7970] | 0.7405 | 0.5605 | 0.3260 |
| K5dfiT2rr | 0.7969 [0.7850, 0.8087] | 0.7236 | 0.5400 | 0.3260 |
| K5dfiT2 | 0.7969 [0.7850, 0.8087] | 0.7236 | 0.5400 | 0.3260 |

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiT2rr - K0dfiT2 | +0.0130 [+0.0079, +0.0181] | above | -0.0168 [-0.0230, -0.0110] | below |
| K5dfiT2 - K0dfiT2 | +0.0130 [+0.0079, +0.0181] | above | -0.0168 [-0.0230, -0.0110] | below |

## DAD (466 videos (macro over the 165 positives), s7; raw)

| arm | macro [95 %] | micro | AP | base rate |
| --- | --- | --- | --- | --- |
| K0dfiT2 | 0.8242 [0.7862, 0.8604] | 0.7807 | 0.2183 | 0.0472 |
| K5dfiT2rr | 0.8107 [0.7755, 0.8445] | 0.7516 | 0.1491 | 0.0472 |
| K5dfiT2 | 0.8107 [0.7755, 0.8445] | 0.7516 | 0.1491 | 0.0472 |

Clip oracle micro (a constant score per video): 0.8390. Micro here measures clip classification.

| contrast | Δ macro | sentence | Δ micro | sentence |
| --- | --- | --- | --- | --- |
| K5dfiT2rr - K0dfiT2 | -0.0135 [-0.0338, +0.0077] | not separated | -0.0291 [-0.0435, -0.0140] | below |
| K5dfiT2 - K0dfiT2 | -0.0135 [-0.0338, +0.0077] | not separated | -0.0291 [-0.0435, -0.0140] | below |
