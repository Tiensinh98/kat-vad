# §60 KIM on − off on the T2-trained host (KIM_RUN.md §60), s2024, n = 1 — printed, decides nothing; K5f stays

Corpus T2 (DADA2000_orig), 1740 steps per row. KIM-on side **K5dfiT2rr**; start gate (TK4): **matched** (step-1 mil on 0.749036, off 0.749036; saved K5dfiT2 0.749036).

| row | what | env |
|---|---|---|
| K0dfiT2 | VideoMAE V2-S, KIM off (control) | {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'} |
| K5dfiT2rr | K5dfiT2 re-run, KIM on | {'gpu': 'Tesla T4', 'torch': '2.14.1+cu130', 'cuda': '13.0'} |
| K5dfiT2 | K5dfiT2 saved (§37), printed | not recorded |

## Cells (macro [95 %] / micro / AP)

| row | DoTA-CAP (dev+eval, 1,129) | DAD (466, raw; macro = position, read beside the placebo) |
|---|---|---|
| K0dfiT2 | 0.7840 [0.7708, 0.7970] / 0.7405 / 0.5605 | 0.8242 [0.7862, 0.8604] / 0.7807 / 0.2183 |
| K5dfiT2rr | 0.7969 [0.7850, 0.8087] / 0.7236 / 0.5400 | 0.8107 [0.7755, 0.8445] / 0.7516 / 0.1491 |
| K5dfiT2 | 0.7969 [0.7850, 0.8087] / 0.7236 / 0.5400 | 0.8107 [0.7755, 0.8445] / 0.7516 / 0.1491 |

## KIM = K5dfiT2rr − K0dfiT2 (paired; §53 KA5: GAIN ≥ +0.01 and interval above 0, COST mirror, else NO DIFFERENCE)

| contrast | set | Δ micro | Δ macro | Δ AP (point) |
|---|---|---|---|---|
| K5dfiT2rr − K0dfiT2 (KIM) | DoTA-CAP (dev+eval, 1,129) | -0.0168 [-0.0230, -0.0110] COST | +0.0130 [+0.0079, +0.0181] GAIN | -0.0205 |
| K5dfiT2rr − K0dfiT2 (KIM) | DAD (466, raw; macro = position, read beside the placebo) | -0.0291 [-0.0435, -0.0140] COST | -0.0135 [-0.0338, +0.0077] printed | -0.0692 |
| K5dfiT2 (saved) − K0dfiT2, printed | DoTA-CAP (dev+eval, 1,129) | -0.0168 [-0.0230, -0.0110] printed | +0.0130 [+0.0079, +0.0181] printed | -0.0205 |
| K5dfiT2 (saved) − K0dfiT2, printed | DAD (466, raw; macro = position, read beside the placebo) | -0.0291 [-0.0435, -0.0140] printed | -0.0135 [-0.0338, +0.0077] printed | -0.0692 |

Reproduction (TK2, printed): K5dfiT2rr − K5dfiT2 whole DoTA-CAP macro +0.0000 (expectation |Δ| ≤ 0.005).
Nexar-host reference (other corpus, printed, not a contrast): §53.1 K5dfi KIM on − off micro DoTA −0.0041 n.s., DAD −0.0036 n.s.; §58.1 DoTA host: NO DIFFERENCE on DoTA-CAP-dev and DADA.

## DAD position diagnostics (control K0dfiT2; §56 DA5)

# DAD position diagnostics (POST-HOC, printed not gated, n = 1 seed)

165 positives, 301 negatives, 15 rows each; accident rows [13, 14]; first V2-S window without a repeated frame = row 7. **No-pixel `t/N` ruler macro = 1.0000.**

## Within video: positives vs the same mask on negatives (placebo)

| row | macro, all rows | placebo, all rows | gap | macro, rows 7+ | placebo, rows 7+ | gap |
|---|---|---|---|---|---|---|
| K0dfiT2 | 0.8242 | 0.6808 [0.6430, 0.7172] | +0.1434 [+0.0909, +0.1955] | 0.7520 | 0.6329 [0.5919, 0.6714] | +0.1191 [+0.0545, +0.1831] |
| K5dfiT2rr | 0.8107 | 0.7177 [0.6850, 0.7493] | +0.0930 [+0.0453, +0.1402] | 0.6859 | 0.6069 [0.5653, 0.6465] | +0.0790 [+0.0132, +0.1447] |
| K5dfiT2 | 0.8107 | 0.7177 [0.6850, 0.7493] | +0.0930 [+0.0453, +0.1402] | 0.6859 | 0.6069 [0.5653, 0.6465] | +0.0790 [+0.0132, +0.1447] |

## Same position: positive vs negative videos (position cannot help)

| row | accident rows | rows 7 up to the accident | difference |
|---|---|---|---|
| K0dfiT2 | 0.7494 [0.6997, 0.7954] | 0.6938 [0.6422, 0.7418] | +0.0556 [+0.0248, +0.0868] |
| K5dfiT2rr | 0.7256 [0.6750, 0.7729] | 0.6904 [0.6381, 0.7392] | +0.0351 [+0.0136, +0.0576] |
| K5dfiT2 | 0.7256 [0.6750, 0.7729] | 0.6904 [0.6381, 0.7392] | +0.0351 [+0.0136, +0.0576] |

Accident rows, paired:

* K5dfiT2rr - K0dfiT2: -0.0238 [-0.0448, -0.0032]
* K5dfiT2 - K0dfiT2: -0.0238 [-0.0448, -0.0032]


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


