# KAT-VAD — DADA-2000 original, **Phase 2: the T2 corpus**

**Parent plan:** `.project/plans/katvad-dada-original-corpus.md` §5.
**Branch:** `main` (KAT-VAD v1). **Opened:** 2026-09-16, after Gate D0 PASSED
(`auc_macro` 0.6518 on 400 clips, `outputs/EDA/DADA2000Origin/gate_d0_report.md`).

---

## 0. One paragraph

Phases 0 and 1 established that the DADA-2000 **original** release is real,
untrimmed data whose frozen-CLIP features carry frame-level signal (0.6518,
against DoTA's 0.6708 and the trimmed archive's 0.5228). Phase 2 turns it into a
trainable corpus: **T2** — 16-sampled-frame windows at hop 8, cut out of the
accident videos themselves so the negatives share camera, weather and scene with
the positives. This is the phase where `core/` is finally touched. Phase 3
(Gate W) then checks the construction held; Phase 4 trains and evaluates
zero-shot on DoTA against the bar of **0.6408**.

---

## 1. Blast radius, measured before writing anything

`trace_path`, depth 3, `include_tests=true`, on the codebase-memory graph
(`nodes: 4603, status: ready`):

| symbol | real hop-1 callers | risk | decision |
|---|---|---|---|
| `dada._write_windowed` | **1** (`dada.preprocess`) + `test_dada.py` fixtures | MEDIUM | **promote to public `write_windowed`**, add one keyword-only arg with a default |
| `dada.plan_record_windows` | `_write_windowed` + 6 direct tests | MEDIUM | **not touched** — signature frozen |
| `dada.preprocess` | `dada.main` + tests | HIGH | **not touched** |
| `video_io.video_id_from_path` | 5 CRITICAL (DoTA + TAD + both extractors) | **CRITICAL** | **not touched** — the symlink farm is the supported route |

**Graph noise, checked by hand:** the trace reports `core/tests/test_tad.py::*`
as callers of `_write_windowed`. `core/data/tad.py` contains no such function
(`grep` confirms); the graph collapsed the two same-named `preprocess` nodes.
The real blast radius is DADA-only.

No HIGH or CRITICAL symbol is modified, so no §6 hard stop applies.

---

## 2. Options considered

| | approach | verdict |
|---|---|---|
| **A** | add `--layout origin` to `dada.preprocess` | **rejected.** `preprocess` already takes 18 parameters and is the HIGH-risk symbol holding every trimmed-corpus number. The original release differs in metadata format (xlsx vs CSV), directory layout, the absence of `Fault_Label` and the absence of `0_Normal_Driving` — five conditionals inside one function, any of which can silently change the archive corpus (C2) |
| **B** | **new `core/data/dada_origin.py`** reusing `DadaRecord`, `sampled_frame_labels`, `plan_record_windows`, `window_label`, `split_records`, `write_windowed` | **CHOSEN.** One-line change to existing code; the archive corpus stays bit-for-bit reproducible |
| **C** | keep it outside `core/`, as Phase 1 did | **rejected.** Phase 1 stayed out because D0 was a hard stop. D0 passed. Phase 4 needs tests, a manifest and reproducibility |

---

## 3. Three decisions taken (user-confirmed 2026-09-16)

### 3.1 Labels are built from **absolute** frame indices

Phase 1 measured `d0_frac` 0.6518 vs `d0_abs` 0.6492 (|Δ| 0.0026, bar 0.01) —
equivalent *on aggregate*. But `t05_v040` proved the fraction convention is
**wrong** wherever the release is trimmed: on-disk 382 vs annotation 482, so the
`D/A = 0.79` rescale drags an end-of-clip anomaly into the middle, 17 sampled
frames off (parent plan §6.1).

T2 therefore builds `span` against the **on-disk** count:

```
span = (start / frames_on_disk, min(end, frames_on_disk) / frames_on_disk)
```

This is the absolute convention expressed in the existing `DadaRecord.span`
field, so `sampled_frame_labels` is reused unchanged — **no new label code
path**, and the 3/400 disagreeing clips are fixed rather than dropped.

> **C33 note.** The threshold that decides is the *measured* one: any clip whose
> derived window is empty after clamping is dropped and logged by id. The
> condition is in the code, not in prose beside it.

### 3.2 The split stratifies by accident **`type`**

The original release ships **zero normal videos**, so `split_records`'
`(is_abnormal, fault_label)` key collapses to one group and the 52-type taxonomy
can drift between train and test. `dada_origin` stratifies by `type` before
delegating, and still splits **by source video, never by window** (parent plan
§8 risk 1 — `write_windowed` re-asserts this and raises).

### 3.3 Cache name is `clip/DADA2000_orig`

The parent plan §5.4 says `DADA2000_orig_w16s8`; **that is superseded.** Features
are keyed by *source clip* and a window is a slice (`core/data/windows.py`
`FeatureSlicer`, `core/eda/corpus.py:43`), so one full-clip cache serves Phase 1
and Phase 2 alike — **and Phase 1's 400 `.npy` on Drive are reused, not
re-extracted.** Never `DADA2000_ncc`: that is the trimmed archive (C2).

### 3.4 Seven annotation columns go into `meta.json` — and only there

`texts`, `causes`, `measures`, `weather`, `light`, `scenes`, `linear`, plus the
raw `type`/`video` key. Cost is near zero and it keeps the `L_neg` door open
without paying 94 GiB twice. **They never reach `labels_train.json`.**

**Why `L_neg` is not activated in this phase (measured 2026-09-16 on the real
xlsx):**

| caption source | unique / 1,962 | rows colliding | % rows colliding at S=64 |
|---|---:|---:|---:|
| `texts` alone | **81** (4.1 %) | 99.0 % | **78 %** |
| `texts+causes`+ 4 scene attrs | 1,124 (57.3 %) | 58.1 % | 10 % |

`asymmetric_infonce_loss` (`core/losses/contrastive.py:20-38`) treats each
caption's own video as the **only** positive, so two clips sharing a string give
`ano_sim` two identical rows against a diagonal target — an unsatisfiable
objective whose gradient pushes two videos of the *same* accident type apart.
With `texts` alone that is 78 % of a batch at `BATCH_SIZE = 64`.

The composite caption is **viable** (10 % at S=64) and carries no timing leak,
which corrects the memory bank's blanket *"`L_neg` from this file: ruled out"* —
that verdict holds for `texts` alone, not for the file. It is still out of scope
here, for four reasons:

1. **G4 is code, not a flag.** `core/train.py:270` only has
   `captions_from_definitions`, which verbalizes *class names*; DADA has one
   class, so flipping it on gives every abnormal clip an identical caption —
   100 % collision, strictly worse than off. `DVSFeatureDataset` must carry a
   `caption` field first.
2. **T2 manufactures its own collisions.** `--window-max-per-clip 4` puts up to
   four windows of one source clip in the pool with, by construction, identical
   captions. Needs a sampler constraint.
3. **Comparability.** `causes` is human-written causal annotation LaGoVAD's
   weakly-supervised setting never had. Undeclared, it makes the comparison
   against DoTA 0.6408 apples-to-oranges.
4. **C14.** Stacking an unattributed component on an untrained trunk measures
   neither.

If it is ever run: a **declared extra-supervision side arm**, after Gate W and
after the T2 trunk has a baseline. Not the headline.

> Minor correction while here: `N3_MIN_SCORE_RANGE = 0.2`
> (`core/losses/contrastive.py:91`) gates only the **mining** branch. When it
> fires, `mined` is empty and `L_neg` falls back to vanilla contrastive — it is
> still computed. The memory bank's phrasing overstates it as a blocker.

---

## 4. Construction

```
sampled frame:  0    5   10   15   20   25   30   35   40
frame label:    0000000000000000000011111111111100000000
                                    +-- [tai, tae] --+
win0 offset  0: [================]                          0 pos -> NORMAL
win1 offset  8:         [================]                  4 pos -> ABNORMAL
win2 offset 16:                 [================]         12 pos -> ABNORMAL
win3 offset 24:                         [================]  8 pos -> ABNORMAL
```

| flag | value |
|---|---|
| `--window-length` | **16** |
| `--window-stride` | **8** |
| `--window-max-per-clip` | 4 |
| `--window-min-positive` | 1 |
| `--window-weak-mode` | drop |
| `--stride` (frame) | 8 |
| `model.score_head_kernel` | **3** (from 9; kernel 9 over W=16 covers 56.2 % — C27) |
| normal source | **none** — negatives are cut from inside the accident videos |

---

## 5. Work items

### 5.1 `core/constants.py` (additive only)

`DADA_ORIGIN_DATASET = "DADA2000_orig"`, `DADA_ORIGIN_ANNOTATION_FILENAME`,
`DADA_ORIGIN_ROOT_DIRNAME`, `DADA_ORIGIN_IMAGES_SUBDIR`,
`DADA_ORIGIN_FAULT_SENTINEL`, and the T2 window geometry as named constants (no
magic numbers, §10).

### 5.2 `core/data/definitions.py`

One entry: `"DADA2000_orig": "dada"` in `DATASET_NAME_TO_ABBR`. Without it
`dataset_abbr` raises and neither the verbalizer nor `check_definition_coverage`
resolves.

### 5.3 `core/data/dada.py` — the only change to existing code

`_write_windowed` → `write_windowed`, plus a keyword-only
`extra_meta: dict[str, dict[str, object]] | None = None` merged into each
window's meta row by **source id**. Additive with a default: the one existing
caller is unaffected.

### 5.4 `core/data/dada_origin.py` — new

* xlsx read with **sheet detection**, never a hardcoded name — this workbook's
  sheets are named the opposite of their contents (`text` holds the taxonomy,
  `Sheet1` holds the 1,962-row table). Ported from `pick_probe.find_sheet`,
  which is the version that has actually run. Header keys normalize NBSP.
* `video_id = t{type:02d}_v{video:03d}` — matches Phase 1's ids exactly, so the
  existing feature cache is reused.
* Resolver for `{root}/{type}/{video:03d}/images`.
* Symlink farm pointing at the **`images` directory itself** (C26), because
  `pathlib` will not recurse into a symlinked directory — a clip-level farm plus
  `--frames-subdir images` raises *"No frame folders found"* (measured, Phase 1).
* Type-stratified split, then `write_windowed`.
* argparse CLI (§14.6).

### 5.5 `core/tests/test_dada_origin.py` — new

Tiny synthetic xlsx (written with `zipfile` + minimal XML, no new dependency),
PNG frame folders, CPU. Covers: sheet detection incl. a decoy sheet · id
uniqueness across types (C26) · the absolute-span clamp on a trimmed clip ·
no source clip in both splits · type stratification · reproducing the Phase 1
label convention · the symlink farm resolving through `list_frame_folders`.

### 5.6 Gates

`ruff · mypy · bandit · pycycle · pyright`, then `detect_changes` before commit.

---

## 6. Phase 3 — Gate W (unchanged from the parent plan §6)

| criterion | threshold |
|---|---|
| length leak (clip AUC) | ≤ 0.55 |
| clip oracle (micro) | ≤ 0.75 |
| abnormal source retention | ≥ 90 % |
| class ratio | within 1:3 either way |
| kernel coverage (median) | ≤ 35 % |
| two-class test clips | ≥ 300 |

**The parent plan's §5.2 simulation was computed on the annotation alone**, before
Phase 0/1 measured that on-disk counts differ from annotated ones. Its 0.6631
oracle and 98.1 % retention are *predictions*. Gate W re-measures; nobody quotes
the simulated figures as results.

### 6.1 RESULT — Gate W ran twice. W=16 FAILED, **W=20 PASSED** (2026-09-16)

Record: `outputs/EDA/DADA2000_orig_T2/` (failing) and
`outputs/EDA/DADA2000_orig_T2_w20s8/` (passing), both on the real 1,945-clip
archive census, both with `--score-head-kernel 3`.

| criterion | bar | W=16 hop 8 | **W=20 hop 8** |
|---|---|---:|---:|
| length leak (clip AUC) | ≤ 0.55 | 0.5000 | **0.5000** |
| clip oracle (micro) | ≤ 0.75 | **0.7529 FAIL** | **0.7037** |
| abnormal source retention | ≥ 0.90 | 0.9830 | **0.9568** (1,861/1,945) |
| class ratio | within 1:3 | 2.06 | **2.80** |
| kernel coverage (median, k=3) | ≤ 0.35 | 0.1875 | **0.1500** |
| two-class test clips | ≥ 300 | 787 | **798** |
| | | **1 FAIL** | **6 PASS** |

The W=16 miss was **151 frames of 19,536 (0.77 %)** — C33's closed form needs
`F_norm ≤ X` and measured 6,528 vs 6,377 — and it was **not** waved through. At
W=20 the EDA's CRITICAL verdict *"micro AUC here is mostly clip classification"*
also disappears, so the criterion had content behind it rather than being a round
number. `core/constants.py:DADA_ORIGIN_WINDOW_LENGTH` is **20** as of this result;
it was 16 until the gate passed, never before.

> **§7 risk 3 was wrong and is corrected there.** The per-clip cap is not the
> lever for the oracle: caps 2/3/4/6 at W=16 span **0.7527–0.7529**, while the
> window length spans 0.61–0.75. Lesson **C35**.

---

## 6.2 Phase 4 — PRE-REGISTRATION (written 2026-09-16, before any arm ran)

Everything in this section is fixed **before** the first training step, because a
criterion chosen after seeing a DoTA delta is not a criterion (C14, lesson 14).

### 6.2.1 The objective, and the one declared deviation

| flag | value | why |
|---|---|---|
| `model.score_head_kernel` | **3** | kernel 9 spans 45 % of a 20-frame window; a head whose kernel spans the clip is a clip classifier (C27) |
| `loss.mil_topk_pct` | **5** (k = 4) | **the deviation.** The default 16 gives `k = max(1, 20 // 16) = 1` for *every* window — `L_MIL` collapses to a plain max, one supervised frame per bag per step (EDA verdict, HIGH). The bag shrank, not the formula: MSAD's median clip is 86 frames, where the same pct gives k = 5. `pct=5` holds the **number** of supervised frames per bag (4) comparable to the MSAD campaign rather than its fraction (20 % vs 6.25 %). Declared, recorded in every run manifest, never tuned against a result |
| `data.frame_stride` | 8 | the cache's stride; changing it voids every `.npy` (C2) |
| everything else | package default | `kip.gate_signal=flow_norm` is the only gate on `main` (C24) |

`sup_mil_topk_pct` and `mul_mil_topk_pct` keep their defaults: `L_MIL` is the term
the EDA flagged, and moving three knobs to fix one makes the arm unattributable.

### 6.2.2 The ceilings, recorded now so a low number is not misread

| quantity | value | what it is |
|---|---:|---|
| **in-domain frame-level ceiling** | **0.5983** | supervised linear probe, `auc_macro`, on the T2 windows the arms will train on (5-fold, grouped by clip) |
| in-domain clip-level probe | 0.5551 AUC | the same features, mean-pooled per window |
| clip constant oracle | 0.7037 micro | what a model that ranks windows perfectly and localizes nothing scores |
| length-only ruler | 0.5000 | any arm below this is unmeasured |
| Gate D0, full clips | 0.6518 | the same features, same transform, **un-windowed** |
| DoTA frame probe | 0.6708 | the held-out benchmark's own ceiling |

**No arm can be expected to exceed 0.5983 in-domain at frame level**, and the drop
from D0's 0.6518 is the price of a 20-frame window (lag-1 autocorrelation 0.968),
**not** evidence against frozen CLIP. The EDA fires CRITICAL *"the features carry
no frame-level signal"* here because `PROBE_SIGNAL_AUC = 0.60`
(`core/eda/report.py:34`) and 0.5983 misses it by **0.0017** — that is a binary
threshold on a 5-fold estimate with no CI, and its prescribed remedy (swap the
backbone) voids every cache (C2, C13). **Do not act on it.**

### 6.2.3 What decides the phase

| | metric | bar |
|---|---|---|
| **headline** | DoTA **zero-shot** `auc_macro`, per-clip min-max pooling | **> 0.6408** |
| attribution | Δ(KIP on − off), paired over seeds 2024/2025/2026, t-interval | CI excluding zero |
| in-domain | T2 `auc_macro` over its **798** two-class windows | reported beside 0.5983, never beside a published frame-level AUC |
| sanity | T2 micro AUC | reported **only** beside the 0.7037 oracle (C12) |

The in-domain micro number is **not** a result and never appears in a table with a
published figure. `--score-norm auto` resolves to `none` on T2 (27.2 % normal
windows) and to `minmax` on DoTA (all-abnormal) — that difference is the protocol,
not a choice (C8, C12).

### 6.2.4 Falsification, stated in advance

* **KIP is refuted on this corpus** if Δ(on − off) over three seeds has a
  t-interval containing zero. That is the expected outcome: the +0.09 on MSAD is
  already attributed to temporal smoothing, and the ordering inverted on the DADA
  archive (`v3/RESULTS_DADA.md`).
* **The corpus is refuted** if the KIP-off trunk cannot beat **0.5000**
  in-domain or **0.6408** zero-shot — then T2 is another TAD (C14) and the phase
  stops rather than trying a KIP variant.
* **The arms are void** if any of them trains without `model.score_head_kernel=3`
  or `loss.mil_topk_pct=5`. Both are in `config.yaml` and the run manifest; check
  before reading any number.

### 6.2.5 Housekeeping fixed in advance

* **Exclude at scoring:** `t05_v048`, `t10_v014` — abnormal sources whose windows
  hold no positive frame; they read as genuine normal clips (`DADA_SETUP.md`
  §5.1). **The ids are per-build** — at W=16 they were `t10_v100`, `t48_v056`.
  Read them from the build's own `eda_report.md` §2, never from this list.
* **`commit: UNKNOWN`** in both Gate-W manifests: the Drive copy of the repo is
  not a git checkout. Fix before the campaign, or every arm inherits a corpus
  build that cannot be identified (C17).
* **KIP-on needs a RAFT pass the corpus does not have.** The CLIP extraction
  deleted its frames; `L_KIP_rec` needs `cache/flow/v1/DADA2000_orig`, which needs
  the frames back. Flow is **train-only**, so the pass covers `train_ids.txt`
  (1,491 sources), not all 1,861. Until it exists, `kip.enabled=true` raises —
  **the KIP-off trunk is what is trainable today.**

---

## 7. Risks

1. **94.01 GiB of `images` over 6 spanned-zip volumes** (Phase 0 §measured). No
   runtime holds it. Phase 2's extraction **must** shard: ~200 clips ≈ 9.6 GiB,
   extract → encode → delete → next. Reuse `build_d0_dataset.py`'s
   `full`/`shard`/`labels` pattern.
2. **C17.** Copy `meta.json` and the frame census off the VM *in the same cell
   that writes the report*. Phase 1 lost its census to a recycled runtime.
3. ~~**Class ratio 2.07:1 toward positives** while `L_MIL` wants negative bags. The
   per-clip cap is what holds it — if Gate W reports drift, move the **cap**, not
   the window length (moving the length re-fires C32 retention).~~
   **CORRECTED 2026-09-16, measured (lesson C35).** The cap is the lever for the
   *class ratio* and for nothing else. For the **clip oracle** it is inert: caps
   2/3/4/6 at W=16 measure 0.7765 / 0.7527 / 0.7529 / 0.7580, a 0.025 span, while
   the window length moves it 0.75 → 0.61. Following this risk as written would
   have cost a rebuild cycle to discover. The oracle's lever is `--window-length`,
   paid for in retention; §6.1 has the curve. Ratio at the adopted W=20 is
   **2.80:1**, still inside the 1:3 bar but nearer the edge — *that* is what the
   cap is for.
4. **C14.** TAD proved in-domain training can destroy localization. T2 may do the
   same. The bar is DoTA **0.6408** and **the DoTA delta is not a tuning signal**
   (lesson 14).
5. **C2.** New corpus ⇒ new cache. No existing DADA number is invalidated; they
   remain the record of the trimmed archive. Do not overwrite.
