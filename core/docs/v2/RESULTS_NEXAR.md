# KAT-VAD v2 — Nexar results (Amendment 18, §29) and the CRN diagnostic (Amendment 19, §30)

Pre-registration: `PREREG_ADDENDUM.md` §21 (D-N1…D-N10), §28, **§29 (D-N16…D-N22)**, **§30 (C1…C9)**. Setup and
runbooks: `NEXAR_SETUP.md` §3; `colab/v2/nexar_train.ipynb`, `colab/v2/nexar_zeroshot.ipynb`,
`colab/v2/nexar_crn_ref.ipynb`. Raw outputs (gitignored; this file is the durable record):
`outputs/v2/v2_nexar_window/{REPORTS/transfer_readout.{md,json}, val/val_summary.md, eval/, REPORTS/crn_diag/}`.
Colab read-outs carry `commit: UNKNOWN` by design.

**Provenance.** 25 Nexar-trained runs (`window` × {A0, A1, A2, A3, A3-ign} × seeds 2024–2028), every checkpoint at
`global_step` **1,749** (the `window` plan: 33 × 53; T2's runs are 1,740, +0.5 %).
Three runs were resumed from their own Drive checkpoint after a lost upload (`resume_stale`, `NEXAR_SETUP.md` §3):
**A0/s2025, A1/s2025, A3/s2027** (each keeps its `metrics.jsonl.stale-<stamp>`). `nexar_test` was opened once (`NEXAR_TEST_OPENED.json`) and is not re-scored.

## 1. Decision (D-N18)

**"No difference shown at n = 5."** A3-ign − A0 on `nexar_test`, per-seed crop endpoint: paired t95
**+0.0151 [−0.0184, +0.0487]**; video bootstrap +0.0028 [−0.0196, +0.0255]. `B` stays A3-ign (T2-trained).
Everything below is printed only (D-N19).

## 2. Nexar test (in-domain; endpoint = shifted-crop mean, D-N2)

| arm | Nexar-trained macro | micro | clip AUC | T2-trained macro | Nexar − T2 (video bootstrap) |
|---|---|---|---|---|---|
| A0 | 0.8116 [0.7881, 0.8335] | 0.8263 | 0.6949 | 0.6974 | +0.114 [+0.083, +0.145] |
| A1 | 0.7491 [0.7242, 0.7737] | 0.7744 | 0.6684 | 0.6362 | +0.113 [+0.080, +0.147] |
| **A2** | **0.8815 [0.8637, 0.8979]** | 0.9105 | 0.7892 | **0.8055** | +0.076 [+0.055, +0.097] |
| A3 | 0.8207 [0.7981, 0.8415] | 0.8768 | 0.7625 | 0.7474 | +0.073 [+0.045, +0.101] |
| A3-ign | 0.8144 [0.7946, 0.8330] | 0.8616 | 0.7410 | 0.7411 | +0.073 [+0.047, +0.099] |

Rulers: clip oracle micro 0.7584; middle ruler per placement 0.21 / 0.66 / 1.00 / 0.66 / 0.22 (0.1 … 0.9).

## 3. Contrasts, Nexar-trained (paired t95 over seeds; item bootstrap beside in `transfer_readout.md`)

| contrast | Nexar test | TAD | DADA whole | DoTA-CAP dev+eval |
|---|---|---|---|---|
| A1 − A0 (CRN alone) | **−0.054 [−0.090, −0.018]** | −0.024 [−0.063, +0.016] | **−0.029 [−0.047, −0.012]** | **−0.054 [−0.066, −0.043]** |
| A2 − A0 (V2-S alone) | **+0.081 [+0.059, +0.104]** | **+0.117 [+0.098, +0.136]** | **+0.151 [+0.140, +0.162]** | **+0.103 [+0.091, +0.115]** |
| A3 − A2 (CRN on top of V2-S) | **−0.061 [−0.071, −0.050]** | **−0.084 [−0.109, −0.059]** | **−0.082 [−0.100, −0.065]** | **−0.074 [−0.087, −0.062]** |
| A3 − A0 | +0.021 [−0.012, +0.053] | +0.032 [+0.002, +0.062] | +0.069 [+0.057, +0.081] | +0.029 [+0.016, +0.042] |
| A3-ign − A3 (`ignore`) | −0.006 [−0.012, +0.001] | **+0.034 [+0.021, +0.046]** | **+0.018 [+0.013, +0.024]** | **+0.041 [+0.035, +0.048]** |
| A3-ign − A0 | +0.015 [−0.018, +0.049] | +0.066 [+0.037, +0.095] | +0.087 [+0.080, +0.095] | +0.070 [+0.060, +0.080] |

Zero-shot macros (Nexar-trained): TAD A0 0.590 · A1 0.556 · **A2 0.706** · A3 0.612 · A3-ign 0.649 (middle ruler
0.683); DADA whole A0 0.617 · A1 0.587 · **A2 0.765** · A3 0.685 · A3-ign 0.702 (middle ruler 0.734, monotone 0.551);
DoTA-CAP pooled A0 0.719 · A1 0.661 · **A2 0.815** · A3 0.740 · A3-ign 0.783. TAD clip-level AUC of A3 / A3-ign
**0.501 / 0.500** (A0 0.633, A2 0.621).

## 4. Readings

1. **CRN costs on every set when the model is trained on Nexar.** A3 − A2 is −0.06 … −0.08 with every interval below
   0, and A1 − A0 is below 0 on three of four sets. On T2 (E3) the same contrasts were ≈ 0 (A1 − A0 DoTA-dev −0.015
   ns; A3 − A2 ≈ −0.007 by seed; interaction +0.005): **CRN has never helped a detection number in this project**;
   E3 adopted A3 over A2 by §10.3's order, not by a measured gap. Its one measured effect remains mechanical
   (`V^t` source-shortcut AUC 0.9998 → 0.569, E3 §2).
2. **CRN hurts on Nexar test whichever corpus trained the model.** T2-trained A3 0.747 vs A2 0.806 and A1 0.636 vs
   A0 0.697 (point estimates, unpaired): part of the cost belongs to the Nexar eval protocol (8 s crops, each its own
   CRN reference), not only to Nexar training. §30 tests the reference unit.
3. **CRN arms fit Nexar's training bags ~10× tighter and transfer worse** — a shortcut signature. Last-step `mil`
   per seed: A0 0.35–0.49 · A2 0.18–0.29 · **A1 0.02–0.09 · A3 0.02–0.06 · A3-ign 0.04–0.11**.
4. **V2-S without CRN (A2) is the best arm on every set**, both training corpora. On DoTA-CAP-dev it reads **0.8171**,
   the highest v2 number on that set — and still **≤ `p_T2` 0.8220**, the no-pixel position prior (dev; eval A2
   0.8132 vs 0.8256). It ties position, it does not beat it.
5. **`ignore` helps transfer again, on a second training corpus** (TAD +0.034, DADA +0.018, DoTA +0.041; Nexar
   in-domain −0.006 ns), the direction of E4-0 (DoTA-CAP-dev +0.040). On T2-trained models TAD read −0.011 (§26).
6. **Training on Nexar instead of T2** (same set, seed-averaged headline macros, unpaired):

   | DoTA-CAP-dev | A0 | A1 | A2 | A3 | A3-ign |
   |---|---|---|---|---|---|
   | T2-trained (E3 / E4-0) | 0.6768 | 0.6577 | 0.7536 | 0.7576 | 0.7912 |
   | Nexar-trained | 0.7138 | 0.6609 | **0.8171** | 0.7311 | 0.7693 |
   | Δ | +0.037 | +0.003 | **+0.064** | **−0.027** | **−0.022** |

   CAP-eval: A3 0.7785 → 0.7485, A3-ign 0.8120 → 0.7958. TAD macro: A0 0.727 → 0.590, A3 0.770 → 0.612, A3-ign
   0.759 → 0.649 (T2-trained from `RESULTS_TADZ.md` §26). **Nexar training lifts the arms without CRN on DoTA and
   lowers the arms with CRN; it lowers every arm on TAD.**

## 5. Why does CRN cost? (§30)

### 5.1 C7 — per-clip CRN cost against the event share (printed; seen before §30 was written)

H1 (C1): training references each window by its 40 s source (event ≈ 7 % of the rows at the eval stride), eval
references each item by itself; a reference that contains more event should cost more (Spearman < 0, the high-share
bin most negative). `crn_cost_diag share`, `REPORTS/crn_diag/crn_share.md`:

| set | contrast | clips | mean Δ | median share | Spearman(share, Δ) | Δ by share quartile (low → high) |
|---|---|---:|---:|---:|---:|---|
| DoTA-CAP-dev | A3 − A2 | 569 | −0.086 | 0.303 | +0.132 | −0.097 / −0.115 / −0.095 / −0.037 |
| DoTA-CAP-eval | A3 − A2 | 560 | −0.065 | 0.293 | +0.092 | −0.081 / −0.076 / −0.066 / −0.035 |
| DADA whole | A3 − A2 | 370 | −0.080 | 0.291 | +0.019 | −0.104 / −0.039 / −0.101 / −0.075 |
| TAD | A3 − A2 | 60 | −0.093 | 0.323 | −0.016 | −0.116 / −0.071 / −0.109 / −0.077 |
| Nexar val whole | A3 − A2 | 112 | −0.043 | 0.073 | −0.022 | −0.029 / −0.043 / −0.083 / −0.017 |
| Nexar val crops | A3 − A2 | 558 | −0.066 | 0.325 | −0.127 | −0.048 / −0.064 / −0.091 / −0.061 |
| Nexar test whole | A3 − A2 | 187 | −0.039 | 0.073 | +0.074 | −0.059 / −0.040 / −0.018 / −0.040 |
| Nexar test crops | A3 − A2 | 930 | −0.060 | 0.325 | +0.105 | −0.079 / −0.069 / −0.066 / −0.028 |

(A1 − A0 and A3-ign − A2 in the same file, same pattern.) **What it says:** on the transfer sets the cost does not grow
with the share — the highest-share quartile is the *cheapest*. On Nexar, about **⅔ of the crop cost is already there
on whole videos** (−0.043 of −0.066 on val), where the reference matches training's unit. So the reference unit can
explain at most ≈ ⅓ of the crop cost; the rest belongs to CRN itself. Per-placement, A3 − A2 on val crops is
−0.050 / −0.066 / −0.039 / −0.069 / −0.106 (0.1 … 0.9): smallest at 0.5, largest at the edges.

### 5.2 C4–C6 — the source-video reference on `nexar_val` (read out 2026-10-26)

`colab/v2/nexar_crn_ref.ipynb` → `REPORTS/crn_diag/crn_ref.md`. **Gate C4: pass.** Primary arm A3.

**Sentence (C6): "the reference unit explains part of it."**

| arm | endpoint, item ref | endpoint, source ref | twin (no CRN) | Δ\_ref paired t95 | f (share of twin gap) |
|---|---:|---:|---:|---|---:|
| **A3** (twin A2) | 0.8136 | 0.8249 | 0.8789 | **+0.0113 [+0.0035, +0.0192]** | **0.17** |
| A1 (twin A0, printed) | 0.7033 | 0.7221 | 0.7758 | +0.0188 [+0.0126, +0.0249] | 0.26 |
| A3-ign (twin A2, printed) | 0.8025 | 0.8045 | 0.8789 | +0.0020 [−0.0031, +0.0071] | 0.03 |

Crop macro per placement (C7, printed; seed-averaged scores):

| arm | path | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---:|---:|---:|---:|---:|
| A3 | item | 0.8656 | 0.8081 | 0.8252 | 0.8238 | 0.7963 |
| A3 | source | 0.8738 | 0.8134 | 0.8121 | 0.8236 | 0.8573 |
| A1 | item | 0.7165 | 0.7049 | 0.7532 | 0.7206 | 0.6768 |
| A1 | source | 0.7495 | 0.7161 | 0.7295 | 0.7279 | 0.7540 |
| A3-ign | item | 0.8455 | 0.8210 | 0.8138 | 0.8064 | 0.7718 |
| A3-ign | source | 0.8507 | 0.8110 | 0.7983 | 0.8029 | 0.8150 |
| twin A2 | — | 0.9160 | 0.8741 | 0.8645 | 0.8930 | 0.9020 |
| twin A0 | — | 0.8302 | 0.7926 | 0.7824 | 0.7822 | 0.7816 |

**Readings.**

1. **The protocol explains ≈ ⅙ of CRN's crop cost; the rest is CRN.** Giving A3 the reference it was trained with
   lifts it by +0.011 (interval > 0), but it still sits **0.054 below A2** (0.825 vs 0.879). C7 predicted `f` ≤ ⅓.
2. **The protocol share sits where the event ends the crop** (placement 0.9: A3 0.796 → 0.857, A1 0.677 → 0.754): a
   self-referenced 8 s crop whose last third is the event has its median pulled toward the event. In the middle
   (0.5) the source reference is slightly *worse*.
3. **With `ignore`, the reference unit does not matter** (A3-ign `f` 0.03, interval ∋ 0).
4. **Per C8 CRN is not rescued:** a deployed model scores a clip without its source video, and even the most
   favourable reference leaves the cost. H2 (centring removes absolute scene content the task needs) is untested.
5. **What it does not license:** dropping CRN from `B`, which is T2-trained. That is a T2 question, tested by
   Amendment 20 (§31): **A3-ign stays** (`RESULTS_CRN_DROP.md`).
