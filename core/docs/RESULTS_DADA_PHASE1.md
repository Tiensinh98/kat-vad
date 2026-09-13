# DADA-2000 Phase 1 — the two loss fixes do not help, and that was the useful answer

**Branch:** `main` (KAT-VAD **v1**), tip `814c177`. Every arm here is a
**KIP-off trunk** — no flow cache, no stage 1, `kip.enabled=false`.
**Measured:** 2026-09-12, from `outputs/v1/DADA2000/2024/**` (5 arms × up to 3
eval protocols = 13 `results.json` + 4,672 per-clip `.npz`).
**Runbook:** `core/docs/DADA_SETUP.md` §10.2. **Notebook:** `colab/DADA/v1/train.py`.
**Answers:** the pre-registered readings in
`core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §6 Phase 1.

> **One line.** Both Phase 1 losses (`loss.dvs_anchor_mode=ignore` for C29,
> `loss.bottomk_weight=1.0`) **failed their own pre-registered predictions**:
> `auc_macro` did not rise (0.5190 → 0.5134 / 0.5104 / 0.5097), the
> within-abnormal-clip gap did not turn positive (+0.0085 → +0.0038 / +0.0045 /
> +0.0013), and the within-clip score range **narrowed** by 20–37 % instead of
> widening. Zero-shot DoTA transfer degrades monotonically with intervention
> strength (macro 0.6254 → 0.6132 → 0.6053 → 0.5952). Under the C28 length
> control every arm's `auc_macro` is **below chance** (0.4237–0.4333). By §6's
> own exit condition this **confirms R2 (the receptive field) as the binding
> constraint and makes Phase 2 mandatory**. Keep all three flags default-off.

---

## 1. What ran

| dir under `outputs/v1/DADA2000/2024/` | `loss.dvs_anchor_mode` | `loss.bottomk_weight` | evals present |
|---|---|---|---|
| `p1_ctrl` — **P0 control** | `span` | 0.0 | dada, dada_eq5, dota |
| `p1_dvsignore` — **P1** (C29) | **`ignore`** | 0.0 | dada, dada_eq5, dota |
| `p1_bottomk` — **P2** | `span` | **1.0** | dada, dada_eq5, dota |
| `p1_both` — **P3** | **`ignore`** | **1.0** | dada, dada_eq5, dota |
| `kipoff` — duplicate of P0, see §6 | `span` | 0.0 | dada_eq5 only |

All five: seed **2024**, 20 epochs, `kip.enabled=false`, `data.is_egocentric=true`,
`bottomk_topk_pct=16`, `mil_topk_pct` untouched (item 1.4 deferred on purpose).
Flags verified from each arm's own `stage2/config.yaml` (lesson **C17**).

Eval protocols: **dada** = 383 test clips, raw pooling; **dada_eq5** =
`--equalize-length 5 --equalize-anchor end` (the C28 control); **dota** =
zero-shot DoTA, per-clip min-max (1,397 clips).

`--equalize-length 5` was used, not the `7` still printed in §6's table — `7`
leaves only 105 two-class clips. The eq5 geometry reproduced the pre-measured
values exactly on all five runs: **331 kept / 52 dropped, 346/476 positives
(72.7 %), anchor=end**.

## 2. Validity gates

| Gate | Result |
|---|---|
| Every shipped metric reproducible offline from the saved `.npz` (lesson **C22b**) | **PASS** — all 13 `results.json` reproduce `auc`/`ap`/`auc_raw`/`ap_raw`/`auc_macro` to < 5e-7 through `evaluate`'s own float32 path, with matching clip and macro-clip counts |
| Arms differ only in the flag under test | **PASS** — configs diffed; identical but for `dvs_anchor_mode` / `bottomk_weight` |
| eq5 crop geometry identical across arms | **PASS** — same 331/52 split, same 346/476 positives, same dropped-id list |
| Cross-branch comparability of the control | **PASS, and stronger than expected** — §6 |
| Raw-vs-eq5 deltas free of eval noise | **FAIL** — §5, a real defect (lesson **C30**) |
| Arm deltas resolvable at n=1 seed | **FAIL** — §7, read signs not magnitudes |

## 3. Headline — DADA-2000 in-domain, 383 clips, raw pooling

Chance macro = 0.5. Baselines that do **no** localization: length-only ruler
**0.8654**, clip-level oracle **0.9086** (both from `RESULTS_DADA.md` §4.1, both
re-derived here from these `.npz` and identical).

| arm | micro | AP | **macro** | **micro, clip-mean removed** | clip-level AUC | argmax-hit |
|---|---:|---:|---:|---:|---:|---:|
| **P0** `p1_ctrl` | 0.7050 | 0.1966 | **0.5190** | **0.4912** | 0.7665 | 0.274 |
| **P1** `p1_dvsignore` | 0.6896 | 0.1836 | 0.5134 | 0.4794 | 0.7528 | 0.258 |
| **P2** `p1_bottomk` | 0.6902 | 0.1862 | 0.5104 | 0.4858 | 0.7591 | 0.284 |
| **P3** `p1_both` | 0.6764 | 0.1740 | 0.5097 | 0.4785 | 0.7466 | 0.274 |
| length-only ruler | **0.8654** | 0.2630 | 0.5000 | 0.5000 | 0.8105 | — |
| clip oracle | **0.9086** | 0.3531 | 0.5000 | 0.5000 | 1.0000 | — |

`argmax-hit` is the fraction of two-class abnormal clips whose highest-scoring
frame is a true positive; **base rate 0.348**. Every arm is at or below it.
Every arm is far below the ruler on micro — the KIP-off trunk was already the
worst micro arm of the v3 campaign, which is the *honest* end of that column.

## 4. The mechanism: both losses shrink the score scale, they do not create contrast

This is the finding. Read the last column, not the third.

| arm | mean σ within abnormal clip | mean within-clip range | pos−neg gap | **d = gap / σ** | mean score (pos) |
|---|---:|---:|---:|---:|---:|
| **P0** | 0.0521 | 0.1473 | +0.0085 | **+0.164** | 0.1160 |
| **P1** `ignore` | 0.0425 | 0.1198 | +0.0038 | +0.091 | 0.0771 |
| **P2** `bottomk` | 0.0414 | 0.1179 | +0.0045 | +0.108 | 0.0953 |
| **P3** both | 0.0326 | 0.0925 | +0.0013 | +0.039 | 0.0635 |

Both interventions do exactly what their gradients say they do and nothing more:

* `dvs_anchor_mode=ignore` removes the dense BCE target from **64.9 %** of anchor
  frames, so fewer frames are pushed up — `mean_pos` 0.1160 → 0.0771.
* `bottomk_weight=1.0` adds a term that pushes the lowest-k frames of an abnormal
  clip **down**, with nothing pushing the rest up harder — `mean_pos` → 0.0953.

Neither buys separation. The pre-registered signal for 1.2 was "within-clip score
range widens"; it **narrowed 0.1473 → 0.1179**. And the effect survives
normalization: `d = gap/σ` falls monotonically **+0.164 → +0.091 / +0.108 →
+0.039**, so this is not a rescaling artifact. See lesson **C31**.

## 5. The C28 length control, and a below-chance macro

| arm | micro | AP | **macro** | clip-mean removed | gap | **d** |
|---|---:|---:|---:|---:|---:|---:|
| **P0** `p1_ctrl` | 0.6403 | 0.2984 | **0.4237** | 0.4244 | −0.0109 | −0.226 |
| **P1** `p1_dvsignore` | 0.6277 | 0.2854 | 0.4253 | 0.4200 | −0.0124 | −0.305 |
| **P2** `p1_bottomk` | 0.6302 | 0.2907 | 0.4258 | 0.4244 | −0.0098 | −0.253 |
| **P3** `p1_both` | 0.6194 | 0.2796 | **0.4333** | 0.4232 | −0.0104 | −0.332 |
| length-only ruler | — | — | 0.5000 | — | — | — |
| clip oracle | 0.8342 | — | 0.5000 | — | — | — |

**The control works**: the length-only ruler drops from 0.8654 to **0.5000
exactly**, and the clip oracle from 0.9086 to 0.8342 — so eq5 removes **C28, not
C12**, and `auc_macro` remains the honest metric.

**New information, not in the diagnosis.** With length neutralized, `auc_macro`
is **below chance on every arm** and `d` is genuinely negative (−0.23 to −0.33):
inside the last 5 sampled frames the model ranks the *negatives* above the
positives. The mechanism follows from R2 — the only within-clip variation a
whole-clip receptive field can express is a monotone positional ramp toward the
clip end, which inverts once the window is cropped to that end. `p1_both` has the
best eq5 macro (0.4333); it is still below chance, so it means nothing.

**Caveat on this table (a real defect, now lesson C30).** `core/evaluate.py`
seeds the definition verbalizer **once per run** while sampling per window, so
the 52 clips dropped by `--equalize-length` shift the RNG stream for every later
clip. Measured: of the 34 clips where `T == 5` exactly — where the eq5 window
*is* the whole clip, i.e. byte-identical model input — **32/34 curves differ**
from the raw run, by up to **0.0033**, on all four arms. The code's own comment
prices that at **±0.003 AUC**. So do not subtract §3 from §5 and call the
remainder a crop effect. Arm-vs-arm comparisons *within* one protocol share the
drop set and the RNG stream, so §3, §4, §5 and §6 are internally valid.

## 6. Zero-shot DoTA — transfer degrades monotonically

| arm | micro (minmax) | AP | **macro** | d |
|---|---:|---:|---:|---:|
| **P0** `p1_ctrl` | 0.6069 | 0.4165 | **0.6254** | +0.413 |
| **P1** `p1_dvsignore` | 0.5956 | 0.4072 | 0.6132 | +0.378 |
| **P2** `p1_bottomk` | 0.5921 | 0.4031 | 0.6053 | +0.345 |
| **P3** `p1_both` | 0.5821 | 0.3954 | 0.5952 | +0.313 |

No arm improved anything. The ordering is the same on all four columns, and the
same as on DADA in-domain — unlike the v3 campaign, where in-domain fit and
transfer were *anti*-correlated (`RESULTS_DADA.md` §4.2). Here nothing is trading
off: the interventions are simply worse.

### The control reproduces v3's A0 bit-for-bit — cross-branch Δ is safe for KIP-off

`DADA_SETUP.md` §10.2.1 warns that diffing a `main` arm against
`RESULTS_DADA.md`'s A0 is a cross-branch comparison. **For the KIP-off trunk it
is not.** `p1_ctrl` scores 0.7050 / macro 0.5190 in-domain and 0.6254 DoTA macro
— A0's numbers to 4 dp. Stronger: the `kipoff` arm's eval was pointed at the
**v3** checkpoint (`$KATVAD_OUTPUT_ROOT_V3/DADA2000/kipoff_s2024/`) by mistake,
and all **331/331** of its eq5 score curves are **bitwise identical** to
`p1_ctrl`'s, max |Δ| = 0. Two independently trained checkpoints, two branches,
identical float32 outputs.

Consequence: the `kipoff` arm was a wasted GPU run (its config is byte-identical
to `p1_ctrl`'s, and `main`'s own `kipoff/stage2` was never evaluated), but it
bought a free reproducibility control. **This holds for `kip.enabled=false`
only** — nothing here says a KIP-on arm is cross-branch comparable, and on `main`
it cannot be (no `gate_type`).

## 7. Pre-registered verdict (§6's table, answered)

| Arm | Prediction | Measured | Verdict |
|---|---|---|---|
| **1.3** eq-length, anchor `end` | ruler → chance; micro falls; macro ~unchanged | ruler → **0.5000 exactly** ✅; micro 0.7050 → 0.6403 ✅; macro 0.5190 → **0.4237** ❌ | **partly falsified** — macro *fell below chance*; §5 |
| **1.1** `dvs_anchor_mode=ignore` | gap turns positive; macro rises; in-domain micro falls | gap **+0.0085 → +0.0038**; macro **−0.0056**; micro falls ✅ | **FALSIFIED** → §6's branch: "the whole-anchor label was not the binding constraint; R2 dominates" |
| **1.2** `bottomk_weight > 0` | within-clip range widens; macro rises | range **0.1473 → 0.1179** (−20 %); macro **−0.0086** | **FALSIFIED** → §6's branch: "confirms R2 is binding" |

**Statistical honesty.** n = **1 seed**. The v3 campaign needed n=3 for ±0.035
CIs on `auc_macro`; the per-arm deltas here (0.005–0.010 macro) are **not
resolvable** at n=1 and must not be published as effect sizes. What *is* solid is
the sign pattern: **7 metrics × 3 protocols, all monotone in intervention
strength, no exceptions**, plus the scale-invariant `d` collapse in §4, which is
a 76 % relative change on P3 — well outside any plausible eval noise.

**C29 is not refuted.** DVS really does label the whole anchor positive; that
description is unchanged and the lesson stands. What Phase 1 shows is that it is
**not the bottleneck** — fixing it changes the loss exactly as predicted and buys
no localization, because the architecture cannot express localization at T = 9.
Do not delete C29; do not re-litigate it.

## 8. Defects found

1. **`--equalize-length` perturbs the text conditioning** (§5). `core/evaluate.py:159-164`.
   → lesson **C30**, fixed in this cycle.
2. **`kipoff` eval pointed at the v3 output root** (§6), so one arm was trained
   on `main` and never scored while a v3 checkpoint was scored in its place.
   Harmless here, and informative, but it is the `$KATVAD_OUTPUT_ROOT_V3` variable
   doing the damage — drop it from `colab/DADA/v1/train.py`.
3. **Eval-command drift.** Only `kipoff` and `p1_dvsignore` have eval cells in the
   notebook; P0/P2/P3 were run by hand-editing, so their flags exist nowhere
   (**C17**). `results.json` records the checkpoint but not `--score-norm`,
   `--equalize-*` or the data dirs.
4. **Version skew across evals.** 5 of 13 `results.json` lack `num_videos_total`
   → run on a pre-`814c177` checkout. Diffed that commit: the only scoring-path
   change is the equalize branch, skipped when the flag is absent. **Numerically
   harmless**, but it is luck, not design.
5. **`checkpoint_every_steps` 400 (P0) vs 100 (P1–P3).** No effect on
   `checkpoint_last.pt`; rules out a convergence-matched intermediate probe (**C16**).

## 9. Limitations — read before citing

* **n = 1 seed**, one corpus, one architecture. §7.
* **No number here is comparable to a published frame-level AUC.** DADA micro is
  bounded above by a ruler (0.8654) and an oracle (0.9086); `RESULTS_DADA.md` §9
  applies unchanged.
* The raw-vs-eq5 difference carries a **±0.003 AUC** artifact from C30. Within-protocol
  comparisons are clean.
* The arms are **KIP-off**. Phase 1 says nothing about KIP, and on `main` a KIP-on
  DADA arm is a fixed ~50 % channel shift by construction (**C24**).
* `auc_macro` here is over **190** two-class clips (raw) / **155** (eq5). The eq5
  macro is a different clip population, not a re-measurement of the same one.

## 10. What this means for Phase 2

§6's Phase 1 exit condition — *"if `auc_macro` stays at chance under both arms,
R2 is confirmed as the binding constraint and Phase 2 is mandatory rather than
optional"* — is **met**, in its stronger form: macro is at chance raw
(0.5097–0.5190) and **below** chance once length is controlled (0.4237–0.4333).

Plan: `.project/plans/katvad-dada-phase2-corpus-rebuild.md`. It splits §6's
Phase 2 into **2a** (fixed-length windows + a fresh cache, then re-baseline) and
**2b** (the four resolution knobs as a small attributable ladder), because
bundling six changes into one arm would be unattributable — the mistake the v3
campaign spent three weeks undoing (**C14**).

## 11. Lessons written from this cycle

* **C30 [HIGH]** — an eval-time sampler seeded once per run couples every item's
  draw to which other items were scored. Promoted (all 5 gates); trigger-map row added.
* **C31 [MEDIUM]** — a loss that lowers scores is not a loss that creates
  contrast; report `gap / σ`, never the raw gap. Promoted (all 5 gates).

## 12. Provenance

* Score curves: `outputs/v1/DADA2000/2024/{p1_ctrl,p1_dvsignore,p1_bottomk,p1_both}/eval_{dada,dada_eq5,dota}/scores/*.npz` and `outputs/v1/DADA2000/2024/kipoff/eval_dada_eq5/scores/*.npz`.
* Metrics: each arm's `eval_*/results.json`; configs from `stage2/config.yaml`; loss traces in `stage2/metrics.jsonl`.
* Baselines (ruler 0.8654, oracle 0.9086, eq5 ruler 0.5000, eq5 oracle 0.8342): re-derived from these same `.npz`, and identical to `core/docs/v3/RESULTS_DADA.md` §4.1 / `DIAGNOSIS_...md` §6 Phase 0.
* Reference A0 row: `core/docs/v3/RESULTS_DADA.md` §3 (branch `v3`).
* **`outputs/` is gitignored** — this document is the durable record.
