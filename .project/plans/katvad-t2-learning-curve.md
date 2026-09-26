# KAT-VAD: T2 learning curve (step B), "is data the bottleneck on T2?"

**Status:** **AUTHORIZED 2026-09-26.** The user chose D-1 fixed steps, D-2 micro primary
with the macro sign, D-3 25 % + 50 % (6 runs). **Branch:** `main` (KAT-VAD v1). **Corpus:** `DADA2000_orig` T2 (W=20 hop 8).
**Parent:** the CCD reopen condition in `activeContext.md` (2026-09-26, CCD PARKED).
**Arms:** KIP-off only. Phase 5 read KIP-v1 as neutral on T2
(`core/docs/RESULTS_DADA_ORIG_T2.md`), so the cheaper arm answers the data question.

> Every bar in §5 is written before any subset is built or any step is trained. No bar is
> moved after the fact (C33, lesson 14).

---

## 1. The question, in one paragraph

CCD (`data/CarCrash/`) was parked because its expected gain over T2 is a prior, not a
measurement. The measurement that decides it is cheap: train the phase-4 KIP-off config
on **nested fractions of T2's train source videos** at a **fixed compute budget** and read
the slope.
- If halving the data costs AUC, data is a bottleneck, and more in-domain-like data (CCD,
  after G-dup and G-pos) is worth its price.
- If the curve is flat, adding more of the same kind of data will not move T2. Then CCD
  closes and the project moves to the write-up (D).

## 2. What already exists (read from the tree, 2026-09-26)

| fact | where | consequence |
|---|---|---|
| The training set is exactly the ids in `labels_train.json` (window id → 0/1) | `core/data/dataset.py:75-82` | A subset is a new data dir with a filtered `labels_train.json`. Every other file is copied unchanged |
| `windows.json` maps each window to its `source` video | `core/data/windows.py` | Subsetting **by source** is a set filter. No window can straddle the subset boundary |
| Epoch length = `ceil(2 · N_abnormal_windows / batch)` | `core/data/dataset.py:97`, `core/train.py:210-213` | Fewer windows means fewer steps per epoch. The cosine horizon is `steps_per_epoch · num_epochs` |
| The KNN cache is built from `labels_train.json` | `core/data/knn_cache.py:183-195` | It must be **rebuilt per subset**, otherwise DVS splices in normals the arm never trains on |
| Test split, CLIP cache, DoTA assets | phase 4 | untouched and shared |
| 100 % arm | `outputs/REPORTS/DADA2000_orig_phase4/` (KIP-off, s2024/25/26) | **reused, not re-run** |

## 3. Decisions (debate done; the three marked ◆ need the user)

| # | Question | Options | Recommendation | Why |
|---|---|---|---|---|
| ◆ D-1 | Budget | **fixed steps** (2,040 at every fraction, epochs scaled up) · fixed epochs (20, steps scale down) | **fixed steps** | Holds compute constant, so the curve measures data, not training time. Fixed epochs at 25 % is 510 steps and would confound "less data" with "under-trained". Price: 80 epochs at 25 % can overfit, and that overfitting is part of what a data curve should show |
| ◆ D-2 | Primary metric | T2 **micro** · T2 macro | **micro primary, macro must agree in sign** | Power (below). The CCD note says "`auc_macro` still rising", but at n = 3 macro can only detect a Δ ≥ ~0.044. The change is made here, **before** any number exists. Macro still gates RISING (§5) |
| ◆ D-3 | Fractions | 50 % only · **25 % + 50 %** | **25 % + 50 %** (6 runs) | With two points only a slope can be read, not a shape. The 25 % point tells a flat curve apart from one that saturates between 50 % and 100 %. Cost: +3 runs |
| D-4 | Subset draw | one fixed subset for all seeds · **one per seed, nested** (25 % ⊂ 50 % ⊂ 100 % within a seed) | **per seed, nested** | "Which videos" is real uncertainty in a data claim, and a single draw would hide it. Nesting within a seed keeps the fraction contrast paired |
| D-5 | Stratification | none · **by accident type** | **by type** | The same rule as `split_by_type`. Types too small to round to ≥ 1 video at a fraction are reported, never forced in |
| D-6 | Where the subset code lives | inline in the notebook · **`core/tools/subset_train.py` + tests** | **core tool** | It defines the training data. A leak or class wipe-out here is silent (C10, C28), and it needs tests. It is small (~100 LOC), numpy/json only, and CPU-local |
| D-7 | Seeds | 3 · 5 | **3** (2024/25/26) | These are the seeds of the reused 100 % arm. More seeds would need a new 100 % arm too |

**Power, measured on this corpus (phase 5 paired Δs as a proxy for the paired-diff sd):**
- micro: sd 0.0031 → t95 half-width ≈ **±0.008**
- macro: sd 0.0177 → t95 half-width ≈ **±0.044**

Micro can resolve a 1-point slope; macro cannot resolve anything below 4 points.

## 4. Deliverables

### 4.1 Code (the only `core/` change): `core/tools/subset_train.py`

```
python -m core.tools.subset_train --data-dir <T2> --out-dir <T2_f50_s2024> \
    --fraction 0.5 --seed 2024 [--nest-in <T2_f..._s2024>]
```

- Reads `labels_train.json`, `windows.json`, `meta.json`. Groups train windows by `source`
  and sources by type (`class_name` in `meta.json`). Draws `round(n_type · fraction)`
  sources per type with `random.Random(seed)`.
- `--nest-in` restricts the draw to a larger subset's sources, so that 25 % ⊂ 50 % holds.
- Writes the out dir: filtered `labels_train.json`. Every other dataset file is **copied**,
  never symlinked (Drive FUSE, C10). Also writes `subset_manifest.json` with fraction, seed,
  parent, source/window/class counts, dropped types and `sha1` of the kept source ids.
- **Raises** if the out dir has no abnormal or no normal window, if any kept source is a
  test source, or if a `--nest-in` source is outside its parent.
- Constants go in `core/constants.py` (`SUBSET_MANIFEST_FILENAME`).

### 4.2 Tests: `core/tests/test_subset_train.py`

Build a synthetic T2-shaped dir and check:
- the fraction per type is within rounding;
- no test source appears in the subset;
- nesting holds;
- the draw is deterministic per seed;
- the class wipe-out raises;
- the other dataset files are byte-identical;
- the manifest counts are correct.

### 4.3 Runbook: `colab/DADA2000Origin/phase_6_learning_curve.ipynb`

Modelled on `phase_4.ipynb`:
- stage the caches;
- build the subsets (s × {50, 25});
- build a KNN cache per subset;
- print the §5.1 gates;
- train stage 2 KIP-off with `num_epochs = round(2040 / steps_per_epoch)`;
- eval T2 and DoTA;
- print the §5.2 table and verdict;
- persist to `outputs/REPORTS/DADA2000_orig_lcurve/` with a run manifest (C17).

### 4.4 Docs

`core/docs/TRAINING.md` gets one paragraph on the subset tool. The results go into a new
`§8` of `RESULTS_DADA_ORIG_T2.md` after the run.

## 5. Pre-registered bars

### 5.1 Build gates (per subset, HARD STOP on fail)

| id | quantity | bar |
|---|---|---|
| G-S1 | kept sources ∩ test sources | = ∅ |
| G-S2 | abnormal and normal train windows | both ≥ 1; report counts and ratio vs 100 % |
| G-S3 | nesting (if D-3 = 25 %) | sources(25 %) ⊂ sources(50 %), per seed |
| G-S4 | total optimizer steps | within ±2 % of 2,040 |
| G-S5 | KNN cache anomaly ids | == the subset's abnormal window ids |
| G-S6 | window fraction realized | within ±0.03 of the source fraction (reported; the windows-per-source spread) |

### 5.2 Read-out

Δ50 = AUC(100 %) − AUC(50 %), paired by seed. The 100 % values are phase-4 KIP-off.

| verdict | condition | next |
|---|---|---|
| **RISING** | Δ50 micro t95 **lower > 0** **and** Δ50 macro point estimate > 0 | Data is a bottleneck → write the CCD plan (G-dup vs DoTA, then G-pos) |
| **FLAT** | Δ50 micro t95 **upper < +0.010** | Doubling T2's data buys < 1 point → **close CCD**, go to D |
| **INCONCLUSIVE** | anything else | CCD stays parked, go to D. **No seed extension** (optional stopping) |

Reported, never decided on:
- Δ25 = AUC(50 %) − AUC(25 %) (shape: diminishing if Δ25 > Δ50);
- the DoTA 0-shot Δs;
- every arm's micro beside the 0.7037 clip oracle. A subset arm with **macro < micro by more
  than 0.02** is flagged as a C14-style collapse and excluded from the verdict row, with the
  reason printed.

## 6. Risks

1. **Overfitting at 25 % under fixed steps** (80 epochs). It is part of the answer, not a
   bug. But if the 25 % arm's training MIL loss falls far below the 100 % arm's, the report
   says "overfit" beside Δ25.
2. **Type dropout.** Types with 1–2 train sources vanish at 25 %. They are reported in the
   manifest. The test set is unchanged, so this is a real property of "less data".
3. **The DVS neighbour pool shrinks** with the normal windows. That is intended (it is the
   subset's own data), but it changes augmentation strength as well as data size. It is
   stated, not separated.
4. **Reusing the 100 % arm.** Its config must equal the subset arms' except `num_epochs`
   and the data/KNN paths. The notebook asserts the diff (as phase 5 §4 did).
5. **Micro is partly clip-level on T2** (oracle 0.7037). A micro rise with a flat macro
   would be "better clip ranking", which is why RISING also needs the macro sign.

## 7. Phases

| phase | where | work | exit |
|---|---|---|---|
| L0 ✅ | local | user authorizes D-1…D-3 (2026-09-26) | this file's status line |
| L1 ✅ | local | §4.1 tool + §4.2 tests + TRAINING.md/COLAB.md §4.3c; quality gate (ruff/mypy/bandit/pyright clean on the changed files) | suite green: **598 collected, 0 fail** (2026-09-26) |
| L2 ✅ | local | §4.3 notebook written (22 cells); §1–§3 + §7–§8 dry-run on a synthetic T2 (real `subset_train` + `knn_cache`, fake evals) | cells run end to end |
| L3 | Colab | build subsets + KNN (CPU), then 6 (or 3) KIP-off runs on GPU, then eval | §5.1 all pass |
| L4 | local | score §5.2; fill Appendix A; RESULTS §8; memory bank; Thesis_Report | verdict row written |

---

## Appendix A: Results

*(empty; filled in L3/L4, bars above left as written)*
