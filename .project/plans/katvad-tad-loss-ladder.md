# Plan — TAD loss ladder (the "T-ladder")

**Branch:** `main` (KAT-VAD v1). **Created:** 2026-09-14.
**Status:** code-complete before it started — every flag already ships on `main`,
default-off. No code change, no re-extraction, no cache invalidation (**C2** does
not fire).

---

## 1. Why this exists

`m0` (TAD in-domain, KIP-off, seed 2024) **collapsed into a clip classifier**.
Measured 2026-09-14 from the saved `.npz` of `outputs/v1/TAD/2024/m0/eval_tad`
and `outputs/v1/TAD/gate_t0`:

| | `gate_t0` (`best.ckpt`, zero-shot) | `m0` (ours, in-domain) | Δ |
|---|---:|---:|---:|
| micro AUC (as reported) | 0.7912 | **0.9237** | +0.1325 |
| **clip-level AUC** (max score vs clip label) | 0.7687 | **0.9975** | **+0.2288** |
| **macro AUC** (n = 60) | **0.7578** | 0.6174 | **−0.1404** |
| **d = gap / within-clip σ** (C31) | **+0.903** | +0.419 | **−0.484** |
| micro, per-clip mean removed | 0.7059 | 0.6803 | −0.0256 |
| **DoTA zero-shot macro** | **0.6158** | 0.5496 | **−0.0662** |

Training in-domain bought near-perfect clip ranking and paid for it in
localization **and** in transfer. `m0`'s micro 0.9237 sits just above the
corpus's constant-score-per-clip oracle of **0.9226** (`outputs/EDA/TAD`
§3.2) — it is, to three decimals, a clip classifier.

**Mechanism.** Under clip-level MIL nothing lowers a frame inside a positive
bag: `mil_loss` raises the top-k, `pseudo_sup_mil_loss` raises the in-span
top-k, `multi_class_mil_loss` raises the class top-k. TAD's abnormal clips are
only **33 %** positive (EDA §2), so **67 % of the frames of every abnormal clip
receive no downward pressure at all**. A per-clip constant satisfies the
objective completely. This is the C29 hole, one level up.

---

## 2. Why the loss arms and not a corpus rebuild

The obvious alternative — re-shard TAD into fixed-length windows, as DADA
Phase 2a does — was simulated against the real `gt` of all 100 test clips before
being rejected. At stride 8:

| W | hop | cap | keeps abnormal | two-class windows | ratio abn:nor | all-positive windows |
|---:|---:|---:|---:|---:|---|---:|
| 12 | 6 | 4 | **93.3 %** ✅ | 109 | 1 : 1.59 ✅ | **17** ⚠ |
| 16 | 8 | 4 | 88.3 % ⚠ | 111 | 1 : 1.41 ✅ | 7 |
| 24 | 12 | 4 | **78.3 %** ❌ | 88 | 1 : 1.43 ✅ | 0 |
| 32 | 16 | 4 | **61.7 %** ❌ | 60 | 1 : 1.78 ✅ | 0 |
| 12 | 6 | — | 93.3 % ✅ | 163 | **1 : 7.8** ❌ | 30 |

Against the model's own geometry (`score_head_kernel=9`, `temporal_window=25`,
`mil_topk_pct=16`) and TAD's positive-span median of **11** sampled frames:

| W | kernel 9 spans | `temporal_window` 25 spans | MIL k | span 11 fills | keeps abnormal |
|---:|---:|---:|---:|---:|---:|
| 12 | **75 %** ❌ | **208 %** ❌ | 2 | **92 %** ❌ | 93 % ✅ |
| 24 | 38 % ⚠ | **104 %** ❌ | 4 | 46 % ✅ | 78 % ❌ |
| 32 | 28 % ✅ | 78 % ⚠ | 5 | 34 % ✅ | 62 % ❌ |

**No window at stride 8 keeps ≥ 90 % of abnormal clips without putting the score
head's kernel over ≥ 75 % of it** — i.e. reproducing DADA's own disease (C27)
through a different door. Fixing that needs stride-2 re-extraction (~269 k
frames, 3–5 h, and **C2 voids `gate_t0` and `m0` and every number measured on
them**) — and stride only buys resolution, not balance: TAD's positive span is
33 % of its clip at *every* stride, so a window sized at 3× the span keeps ~45 %
of abnormal clips and lands back on **C32**.

Third reason: **DADA's own windowed rebuild is not yet validated.** It failed
Gate W once (C32), and the corrected `w24s1` geometry has never been trained.
Gate W measures corpus health, not localization. Cloning an unproven remedy onto
a second corpus is C14 at the programme level.

**The loss arms attack the same hole for minutes of GPU and zero risk.** They
failed on DADA (Phase 1, `RESULTS_DADA_PHASE1.md`) — but under a known-open
precondition defect: DADA's median clip is **9** sampled frames under a kernel of
9, so there was no within-clip resolution for either term to act on (R2 / C27).
Per **C14** that makes DADA Phase 1 *unmeasured*, not *refuted*. **TAD is the
corpus where that confound is absent**: median 43 frames, kernel 9 spans 20.9 %.
This is a genuine re-test.

---

## 3. The ladder — 3 new runs, seed 2024

Names mirror DADA Phase 1 deliberately, so the two tables can be read side by
side.

| arm | `loss.dvs_anchor_mode` | `loss.bottomk_weight` | output dir | status |
|---|---|---|---|---|
| `t1_ctrl` | `span` | `0.0` | **= the existing `TAD/2024/m0`** | ✅ already trained |
| `t1_dvsignore` | **`ignore`** | `0.0` | `TAD/2024/t1_dvsignore/stage2` | to run |
| `t1_bottomk` | `span` | **`1.0`** | `TAD/2024/t1_bottomk/stage2` | to run |
| `t1_both` | **`ignore`** | **`1.0`** | `TAD/2024/t1_both/stage2` | to run |

**`m0` *is* the control** — its `config.yaml` already carries
`dvs_anchor_mode: span`, `bottomk_weight: 0.0`, `seed: 2024`,
`num_epochs: 72`. Do not retrain it. Verify with the config diff in §6 before
reading any Δ (`RESULTS_DADA_PHASE1.md` §3 did the same check).

**`E = 72` is fixed** (§11 of `TAD_SETUP.md`: 200 abnormal train clips → DVS
length 400 → 7 steps/epoch at batch 64 → 504 steps). Every arm uses 72. An arm
trained for a different number of steps is not a control.

**`bottomk_weight = 1.0`, `bottomk_topk_pct = 16` — identical to DADA Phase 1.**
Do **not** sweep them (lesson 14): the comparison is the point, and 1.0 is a
symmetry prior against `L_MIL`, not a tuned value. Note for the record that
`_topk_k` computes `k = max(1, n // topk_pct)`, so at TAD's median n = 43 this
pushes down only **k = 2** frames per abnormal clip — weak pressure, stated in
advance so it cannot become a post-hoc excuse.

---

## 4. Readout — micro AUC is BANNED on this ladder

TAD's micro AUC is a length-weighted clip classifier (EDA CRITICAL ×2; the
constant-per-clip oracle scores **0.9226**, a frame-count-only ruler **0.8968**).
Report it for the record only. **The four columns that decide this ladder:**

| column | `t1_ctrl` (= m0) | direction if the hypothesis holds |
|---|---:|---|
| **clip-level AUC** | 0.9975 | **↓** — this is the collapse |
| **macro AUC** (n = 60) | 0.6174 | **↑** toward `gate_t0`'s 0.7578 |
| **d = gap/σ** | +0.419 | **↑** toward `gate_t0`'s +0.903 |
| **DoTA zero-shot macro** | 0.5496 | **↑** toward `gate_t0`'s 0.6158 |

### Pre-registered, written before the first run (C33 — all four are reachable)

* **H-T1.** `t1_bottomk` moves **all four** columns in the predicted direction
  simultaneously. One column alone is noise at n = 1.
* **H-T2.** `t1_dvsignore` moves macro and d up; its effect is smaller than
  `t1_bottomk`'s, because it removes a wrong target rather than adding the
  missing one.
* **H-T3.** `t1_both` ≥ `t1_bottomk` on macro.
* **Worth-pursuing bar.** `t1_bottomk` or `t1_both` reaches **macro ≥ 0.65**
  (halfway from 0.6174 to `gate_t0`'s 0.7578) **and** clip-level AUC **≤ 0.95**.

**Falsification branch.** If all four columns stay flat across all three arms,
the missing-downward-term hypothesis is refuted on a corpus where the C27
confound is absent — two corpora then agree, and the stride-2 re-extraction plus
windowed rebuild of §2 becomes the justified next expense rather than a guess.

**n = 1 seed.** Report the **sign pattern across 4 columns × 3 arms**, never an
individual Δ and never a CI. Seeds 2025/2026 only if the sign pattern is there.

---

## 5. Order of work

```
[ ] §13.0 of TAD_SETUP.md — train t1_dvsignore, t1_bottomk, t1_both (seed 2024)
[ ] §14.1a               — 6 evals (TAD + DoTA per arm)
[ ] §14.4                — rescore + score-count assertion (C11b, C22b)
[ ] §15.1                — the 4-column analysis cell. STOP AND READ IT.
[ ] decide: seeds 2025/2026, or the §2 rebuild, or drop TAD to zero-shot only
[ ] core/docs/RESULTS_TAD.md + memory bank (outputs/ is gitignored)
```

## 6. Risks

| Risk | Mitigation |
|---|---|
| `m0` is not actually a valid control | config diff in `TAD_SETUP.md` §15.1, run it before any Δ |
| Reading micro AUC and declaring victory | micro is banned in §4; the analysis cell does not print it as a headline |
| `k = 2` too weak to move anything | pre-registered in §3; a null is then "weak pressure", not "no mechanism" — say so |
| n = 1 over-read | §4 forbids per-Δ claims; sign pattern only |
| Tuning `bottomk_weight` against macro | lesson **14** — forbidden; 1.0 matches DADA or the tables do not join |
| Someone joins a TAD micro to 89.56 | `TAD_SETUP.md` §0.1 + §8.1; the reference is **0.7912** (C8b) |
