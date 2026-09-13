# DADA-2000 Phase 2 — fixed-length windows, then the resolution ladder

**Status:** 2a code shipped 2026-09-12. **The first rebuild FAILED Gate W on 2026-09-13**; geometry corrected below and four code fixes landed (lessons **C32**, **C33**). 508 tests pass; ruff / mypy / pyright / pycycle clean, bandit 0 High. Remaining in 2a: the rebuild, the re-extraction and the W0 run — all GPU/user. 2b is config-only. No training run is part of this plan.
**Branch:** `main` (KAT-VAD v1). All arms are **KIP-off trunks**.
**Source of truth:** `core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §6 Phase 2 (the six changes), §7 (the backbone decision).
**Why now:** Phase 1 is measured (`core/docs/RESULTS_DADA_PHASE1.md`) and its exit condition fired — `auc_macro` at chance raw (0.5097–0.5190) and **below** chance length-controlled (0.4237–0.4333) under both loss arms. R2 is the binding constraint.

> **One line.** Split §6's six simultaneous changes into **2a** (kill the length
> leak: fixed-length windows into a *new* cache, then re-baseline) and **2b** (the
> four resolution knobs as a 5-arm ladder on that cache). Six changes in one arm
> produce one unattributable number — lesson **C14**, and the mistake the v3
> campaign spent three weeks undoing.

---

## 1. Exit criteria (pre-registered, lesson 14)

Phase 2 succeeds or fails on these, written before any run:

**Revised 2026-09-13 after the first rebuild failed.** E2 as originally written
("clip oracle < 0.75") was **unreachable by arithmetic** — see §1.1 — and is
replaced by three criteria that measure what it was reaching for. E1 is unchanged
and passed on the first attempt.

| # | Claim | Measurement | Pass |
|---|---|---|---|
| **W-1** (was E1) | The length leak is closed | `core.tools.eda` §3.3 | clip-length AUC < 0.55 **and** length-only micro < 0.55, C28 verdict gone |
| **W-2** (replaces E2) | The test set can still support `auc_macro` | EDA §3.4 | **two-class test windows >= 150** (stride-8 corpus had 190; failed build had 57) |
| **W-3** (new) | The rebuild kept the class it exists to measure | preprocessor log | **abnormal-source retention >= 90 %** |
| **W-4** (new) | The rebuild did not manufacture an imbalance | preprocessor log | abnormal:normal windows no worse than **1:3** |
| E3 | There is real frame-level signal | `auc_macro` on the rebuilt test split | **> 0.60** under *any* 2b arm |
| E4 | The model localizes, not classifies | micro with clip-mean removed | **> 0.60** |
| E5 | The gain is attributable | 2b ladder, one knob at a time | each arm's Δ reported separately, n >= 2 seeds on the winner |

The clip oracle is **printed, not gated** (lesson **C12** still requires printing it).

### 1.1 Why E2 was unreachable — the derivation that should have preceded it

`oracle = (F_norm + 0.5·X) / (F_norm + X)`, with `F_norm` = frames in all-normal
clips and `X` = negative frames inside abnormal clips. It reproduces both measured
corpora exactly: `(3896+436)/(3896+872) = 0.9086` and
`(22496+552.5)/(22496+1105) = 0.9766`. `< 0.75` requires `F_norm < X`, i.e.
abnormal clips holding **>= 62 %** of all test frames; a perfectly balanced test set
scores **0.811**, and the target 1:1.3 ratio scores **0.840**. Lesson **C33**.

### 1.2 What the first rebuild measured (2026-09-13)

`--window-length 32 --stride 2`, the geometry §4.2 originally prescribed:

| | result | |
|---|---|---|
| W-1 length leak | clip-length AUC **0.5000**, micro **0.5000**, 0 separable clips, C28 verdict **gone** | ✅ |
| E2 clip oracle | **0.9766** (was 0.9086) | ❌ unreachable criterion |
| W-2 two-class test windows | **57** (was 190) | ❌ |
| W-3 abnormal retention | **25.3 %** of abnormal clips kept | ❌ |
| W-4 class ratio | **327 abnormal : 3,244 normal** windows | ❌ |
| — C27 head span | kernel 9 covers **28.1 %** of a clip, 0 clips inside it (was 100 %) | ✅ genuine win |
| — MIL top-k floor | **0 %** of clips at k=1 (was 90 %) | ✅ genuine win |

Cause: DADA's accident clips are trimmed to a raw median of **49** frames, so a
32-frame window at stride 2 (= 64 raw) deleted three-quarters of them, and a fixed
hop let 3x-longer normal clips flood the corpus. Lesson **C32**.

**If E1/E2 pass but E3/E4 fail**, the ceiling is representational or the labels
are too coarse → Phase 3 (probe on the stride-2 cache) decides the backbone, per
§7's pre-registered rule. **That is a valid outcome, not a failure of the plan.**

## 2. Constraints (hard)

- **C-a** Lesson **C2/C13**: every change here invalidates the DADA caches. Write to **new** paths (`clip/DADA2000_w32s2`, `knn/DADA2000_w32s2`), never overwrite. Every number in `core/docs/v3/RESULTS_DADA.md` and `RESULTS_DADA_PHASE1.md` describes the **old** corpus and stays valid for it.
- **C-b** Default behavior must not change. Windowing is opt-in per corpus; MSAD / DoTA / TAD / PreVAD artifacts and the 439-test suite stay byte-identical.
- **C-c** Lesson **C30**: fix the verbalizer RNG coupling *before* running equalized evals on the new corpus. Part of 2a.
- **C-d** Lesson **C31**: pre-register `d = gap/σ`, not the raw gap.
- **C-e** `LaGoVAD-PreVAD/` read-only; §10 code standards; §11 quality gate; ≤20 files per commit.
- **C-f** Lesson **C17**: every arm records the flags that define it. The runbook ships eval cells for *every* arm — Phase 1 lost three (`RESULTS_DADA_PHASE1.md` §8.3).

## 3. Blast radius (traced 2026-09-12)

`trace_path` + grep, because the graph under-reports callers on this repo
(`activeContext` 2026-09-09):

| Symbol to change | Callers | Real risk |
|---|---|---|
| `DVSFeatureDataset._load_features` / `_load_flow` | `_load_clip` → `__getitem__`; constructed only at `core/train.py:623` | **LOW** — gated on a file that does not exist for any current corpus |
| `FeatureEvalDataset.__getitem__` | `core/evaluate.py:153` only | **LOW** — same gate |
| `knn_cache.central_frame_key` / `motion_descriptor` | `build_key` → `build_knn_cache` → CLI `main`; tests `fixtures.py:128`, `test_dada.py:450`, `test_tad.py:346` | **LOW** — same gate |
| `core/evaluate.py:main` (verbalizer) | entry point | **LOW** — changes eval determinism *by design*; see §4.1 |
| `core/data/dada.py:preprocess` / `build_frame_labels` / `_meta` | CLI `main` only | **MEDIUM** — new output file; old flags unchanged |

The tool labels hop-1 callers CRITICAL by distance. That is not meaningful here:
every new path is behind `windows.json` existing, which no shipped corpus has.

## 4. Phase 2a — kill the leak (code + one re-extraction + re-baseline)

### 4.1 The C30 fix (do this first, it is 5 lines)

`core/evaluate.py` seeds the verbalizer once per run while sampling per window,
so `--equalize-length`'s `continue` shifts every later clip's conditioning
(±0.003 AUC). Rebuild it per item from `SEED + index`. Test: score a set, score a
subset, assert the shared curves are bit-identical.

**This changes existing eval numbers by ≤0.003.** Record that: `RESULTS_DADA_PHASE1.md`
§3–§6 were measured pre-fix, and any re-run of those arms will differ in the 3rd
decimal. Do not re-run them to "clean up" — that buys nothing and costs GPU.

### 4.2 Fixed-length windows (2.1) — design

DADA's leak is that a clip's *length* is its label. The cure is to make every
scored and trained item the same length. Two ways to do it:

| Option | How | Verdict |
|---|---|---|
| **A. Window ids in the feature cache** | extract one `.npy` per window | ❌ 4–8× storage, re-extract on every window-geometry change, and the same frame lands in several files |
| **B. Window ids over clip features** (chosen) | one `.npy` per source clip (as today); a new `windows.json` maps `window_id → (source, start, end)`; loaders slice | ✅ one extraction per *stride*, geometry is a cheap re-run of the preprocessor, no duplicated pixels |

**The contract.** A fifth, **optional** dataset file beside the existing four
(`core/docs/DATA_LAYOUT.md`):

```
windows.json   {window_id: {"source": video_id, "start": int, "end": int}}
```

`labels_train.json`, `frame_labels_test.json` and `meta.json` are keyed by
**window_id** when it is present. Absent, everything behaves exactly as today.

**Loader change** — one resolver, three call sites:

```python
# core/data/windows.py  (new)
Window = namedtuple-ish frozen dataclass: source, start, end
load_windows(data_dir) -> dict[str, Window] | None
class FeatureSlicer:  # holds clip_dir + windows|None
    def load(self, item_id) -> np.ndarray   # slices when windowed
    def source_of(self, item_id) -> str     # for flow + knn keys
```

Used by `DVSFeatureDataset._load_features`/`_load_flow`, `FeatureEvalDataset`,
and `knn_cache.build_key`. Flow must be sliced with the **same** window — a
`(source, start, end)` mismatch between appearance and flow is lesson **C13** in
miniature.

**Window geometry** (`core/data/dada.py` new flags) — **corrected 2026-09-13**:

```
--stride 1                # frame stride; 1, not 2 -- see the trade-off below
--window-length 24        # sampled frames per window; T is now constant
--window-stride 12        # hop between consecutive windows (50 % overlap)
--window-max-per-clip 4   # cap per SOURCE clip, evenly spaced (lesson C32)
--window-min-positive 1   # a window is abnormal iff it holds >= this many positive frames
--window-weak-mode drop   # abnormal clips with no annotated span
```

Sized from the **abnormal** length distribution (n = 975 abnormal / 938 normal
source clips, from `data/DADA2000/meta.json`):

| | raw p5 | raw p25 | **raw p50** | raw p75 |
|---|---:|---:|---:|---:|
| abnormal | 22 | 35 | **49** | 64 |
| normal | 29 | 79 | **139** | 209 |

A 24-frame window needs 24 raw frames → **94.2 % abnormal retention**. Why stride 1
rather than 2: at stride 2 the only window short enough is 12 frames, and at T = 12
`score_head_kernel=9` (75 % of the window), `temporal_window=9` (75 %) and
`mil_topk_pct=8` (k = 1) are all degenerate — there would be nothing left to ablate.
A 24-frame window at stride 1 keeps the **same clips** with twice the resolution
inside each, at 2x the extraction (~209 k frames).

Rules, each of which is a test:
1. **No padding, ever.** A source clip shorter than `--window-length` is either dropped (`--window-drop-short`) or raises. Padding would fabricate frames and interact with `ConvScoreHead`'s `padding_mode="replicate"`.
2. **Windows never cross a source clip.** `end <= num_sampled_frames(total, stride)`.
3. **The accident is not centered.** Abnormal windows are emitted at every valid hop, so the positive span lands at varying offsets. A fixed offset would replace the length leak with a position leak.
4. **Train/test split is by SOURCE clip, not by window.** Two windows of one clip must never straddle the split — that is a leak of a different kind.
5. **A window inherits its source's class name** for `H_mul` and for the definition conditioning.
6. **`meta.json` records `source`, `start`, `end`, `positive_frames`, `sampled_frames`** so every downstream table can re-derive the geometry (**C17**).

**Expected geometry at `--window-length 32 --window-stride 16 --stride 2`**
(median source clip is 9 stride-8 frames = ~36 stride-2 frames): ~1–3 windows per
clip, constant T=32, and — by construction — a length-only baseline of exactly
**0.5000**. Verify with `core.tools.eda` before training (E1).

### 4.3 Re-extraction (2.6) and the new cache paths

| Artifact | Old | New |
|---|---|---|
| dataset dir | `data/DADA2000` | `data/DADA2000_w24s1` |
| CLIP cache | `cache/clip/DADA2000` | `cache/clip/DADA2000_s1` |
| KNN cache | `cache/knn/DADA2000/knn_cache.npz` | `cache/knn/DADA2000_w24s1/knn_cache.npz` |
| outputs | `outputs/DADA2000/2024/*` | `outputs/DADA2000_w24s1/2024/*` |

The CLIP cache is keyed by **stride only** (windows slice it), so `_s2` is the
honest name; the dataset and KNN dirs carry the window geometry. Cost: 4× the
frames of the current cache (~21 k test + ~84 k train sampled frames).
`no_center_crop` is already the pipeline default — do not change the transform in
the same step (**C2**).

### 4.4 Re-baseline (the only 2a arm)

One arm: **W0**, the 2a corpus with **today's** hyperparameters
(`frame_stride=2` forced by the cache; `score_head_kernel=9`,
`temporal_window=25`, `mil_topk_pct=16` unchanged). Seed 2024.

Its job is *not* to be good. Its job is to give 2b a baseline measured on the
same corpus, and to answer E1/E2 with `core.tools.eda`. Diffing a 2b arm against
`RESULTS_DADA_PHASE1.md`'s P0 would be a cross-corpus Δ, i.e. not a Δ.

### 4.5 Deliverables (2a)

| # | File | Change |
|---|---|---|
| 1 | `core/evaluate.py` | ✅ per-item verbalizer seed (C30) |
| 2 | `core/constants.py` | ✅ `WINDOWS_FILENAME`, window defaults |
| 3 | `core/data/windows.py` | ✅ **new** — `Window`, `load_windows`, `FeatureSlicer` |
| 4 | `core/data/dataset_files.py` | ✅ `write_windows` + docstring contract |
| 5 | `core/data/dataset.py` | ✅ both datasets take the slicer |
| 6 | `core/data/knn_cache.py` | ✅ keys through the slicer |
| 7 | `core/data/dada.py` | ✅ `--window-*` flags, window records, split-by-source |
| 8 | `core/tests/test_windows.py` | ✅ **new** — geometry, no-padding, no-crossing, split-by-source |
| 9 | `core/tests/test_dada.py`, `test_data_utils.py`, `test_e2e_synthetic.py` | ✅ windowed build/train/eval, the C30 subset test (verified to fail pre-fix) and the shared-RNG pin |
| 10 | `core/docs/DATA_LAYOUT.md`, `DADA_SETUP.md` §10.3 | ✅ the contract + the 9-step runbook, every `--set` parse-tested |

## 5. Phase 2b — the resolution ladder (config only, no code)

All four knobs already exist in `core/config.py`. Verified on this branch:
`model.temporal_window`, `model.score_head_kernel`, `loss.mil_topk_pct`,
`data.frame_stride`. **No code change is needed for 2b** — which is exactly why
it should not be bundled into 2a.

| Arm | Change from W0 | Isolates | Pre-registered reading |
|---|---|---|---|
| **W0** | — (2a baseline) | the corpus rebuild alone | E1, E2 must pass here |
| **W1** | `model.temporal_window` 25 → **9** | the encoder's global receptive field — **the largest of the three spans and the one nobody has ever changed** | if `auc_macro` moves most here, R2 is an *encoder* problem |
| **W2** | `model.score_head_kernel` 9 → **3** | the score head's span | §5 of `RESULTS_DADA.md` blames this one; test it alone |
| **W3** | `loss.mil_topk_pct` 16 → **8** | k > 1 in MIL top-k (item 1.4, deferred from Phase 1 because it does nothing at T = 9) | small effect expected; it is free |
| **W4** | W1 + W2 + W3 | the joint effect | if W4 ≫ W1+W2+W3, the knobs interact and the ladder was necessary |

Judge every arm on **`auc_macro`**, the **clip-mean-removed micro**, and
**`d = gap/σ`** (C31) — never raw micro (C12, C28). Run the winner at seeds
2025/2026 before any claim (E5). `frame_stride` is **not** an arm: it is fixed by
which cache you load (**C2**).

## 6. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Windowing changes the train-set size and so the effective LR schedule | high | report `len(dataset)` and steps/epoch for W0 beside every metric; keep `num_epochs` fixed across the ladder |
| DVS composes windows from different source clips, so a "clip" is now a splice of splices | medium | unchanged behavior — DVS already splices; but `dvs_anchor_mode` semantics now apply to a 32-frame window whose positive fraction is much closer to 1, which is C29's precondition. **Re-measure the positive fraction on the new corpus and reconsider `ignore`** |
| Two windows of one source clip land in train and test | would void everything | rule 4 above + a test |
| The new cache is built with a different transform by accident | high (it is a fresh path) | one `--set` block in the runbook, and `core.tools.eda` prints the transform |
| E3/E4 fail and the result is "the backbone is the ceiling" | ~40 % | that is Phase 3's question, pre-registered in §7. Do not turn it into a VideoMAE swap inside Phase 2 — that voids `H_mul` and every cache |
| Scope creep into 2b while 2a is unfinished | medium | 2b is config-only; nothing in 2b needs a commit |

## 7. Cost

| | GPU | Wall |
|---|---|---|
| 2a code + tests | 0 | this session |
| 2a re-extraction (CLIP, **stride 1**, ~209 k frames) | 1× | ~2–4 h Colab |
| 2a KNN rebuild | ~0 | minutes |
| W0 train + 3 evals | 1 arm | ~1 h |
| 2b: W1–W4 train + 3 evals each | 4 arms | ~4 h |
| E5 seed replication of the winner | 2 arms | ~2 h |

Flow/RAFT is **not** re-extracted: every arm here is KIP-off.

## 8. Out of scope

* Any KIP-on DADA arm. On `main` KIP is a fixed ~50 % shift (**C24**); attribution needs branch `v3`.
* The backbone swap (§7). Phase 3, and only after E3/E4.
* TAD and PreVAD. Untouched — but **run `core.tools.eda` on TAD before training it**, because C28 is a property of a corpus build and TAD has never been checked.
