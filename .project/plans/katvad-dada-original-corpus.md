# KAT-VAD — DADA-2000 **original** corpus (T2 windowing)

**Branch:** `main` (KAT-VAD v1). **Status:** planned, nothing built, nothing trained.
**Author:** sinhpham · **Opened:** 2026-09-15
**Supersedes:** `.project/plans/katvad-dada-phase2-corpus-rebuild.md` (Phase 2 re-sharded
the *trimmed* archive; measured worse on every column — §2.3).

---

## 0. One paragraph

Train on the **original LOTVS-DADA-2000 release** (full-length accident videos,
raw median **322** frames) instead of the reconstructed `archive.zip` this project
has been using (abnormal trimmed to raw median ~56). Re-shard it with the
**window machinery that already ships on `main`** (`plan_record_windows`), taking
**both** classes from inside the same accident videos: a window overlapping
`[tai, tae]` is abnormal, one that does not is normal. Evaluate **zero-shot on
DoTA val**, which stays held out, against the numbers already in `outputs/v1/DoTA/`.

**Nothing here is measured on pixels yet.** Every number below is simulated from
`data/DADA/dada标注.xlsx` (1,945 valid accident rows) plus the real per-clip
lengths in `outputs/v3/DADA2000/kipoff_s2024/eval_dada/results.json`. Two gates
(§3, §4) must pass before any download beyond the probe.

---

## 1. Why the current DADA corpus cannot be fixed in place

| defect | measurement | source |
|---|---:|---|
| C28 length leak | length-only ruler: clip AUC 0.8105, micro **0.8654** | `outputs/EDA/DADA2000` |
| C12 clip oracle | constant-score-per-clip micro **0.9086** | 〃 |
| C27 receptive field | median clip **9** sampled frames, kernel 9 covers **100%** | 〃 |
| representation | supervised frame linear probe `auc_macro` **0.5228** | 〃 |

The archive trims accident videos around the accident and keeps normal-driving
videos longer. **Both classes are trimmed, just unequally** — measured 2026-09-15:

```
original DADA video (xlsx)        raw median 322 frames
0_Normal_Driving (archive)        raw median 152        (19 sampled × 8)
abnormal          (archive)       raw median  56        ( 7 sampled × 8)
```

So `0_Normal_Driving` is **not** untrimmed footage. Swapping the original
full-length abnormal clips in beside it does not close the leak — it inverts it:

| abnormal source | vs the same 192 normal clips | direction |
|---|---:|---|
| archive (trimmed) | **0.8105** | abnormal shorter |
| **original (xlsx)** | **0.8539** | abnormal longer |

(Method reproduces the EDA's published 0.8105 exactly on the trimmed data.)

---

## 2. Options considered, all measured

### 2.1 The ledger

| option | length leak | clip oracle | kernel-9 coverage | abnormal retained | trainable |
|---|---:|---:|---:|---:|---|
| original full-length, all-abnormal | *undefined* (1 class) | 0.5000 under min-max | **22.5%** | 100% | **no — no negative bags** |
| original + `0_Normal_Driving` | **0.8539** | — | 22.5% | 100% | yes |
| prefix-split at `tai` | **0.6782** | — | ok | 91% | yes |
| length-matched subsample (T5) | 0.5143 | **0.7517** | 34.6% | **28%** | yes |
| **T2 — windows from inside the accident videos** | **0.5000** | **0.6631** | 18.8% at kernel 3 | **98.1%** | yes |

### 2.2 Why every corpus in this project is blocked one way or the other

| corpus | negative bags | clean |
|---|---|---|
| MSAD | yes | yes, but CCTV — not motion/traffic |
| TAD | yes | no — clip-classifier collapse, oracle 0.9226 |
| DADA (archive) | yes | no — leak 0.8105, probe 0.5228 |
| DoTA | **no** (3 normal clips, `train_clips: 0`) | yes — oracle 0.5017, probe **0.6708** |
| DADA original | **no** (0 normal clips) | yes — kernel 22.5%, 100% mixed clips |

**Every corpus with negative bags is degenerate; every clean corpus has no
negative bags.** T2 is the only construction that produces negative bags without
importing a second, differently-distributed pool.

### 2.3 Why the earlier rebuilds failed — corrected 2026-09-15

Both live in `outputs/EDA/`:

| corpus | test abn / nor | leak | oracle | macro population |
|---|---:|---:|---:|---:|
| `DADA2000` | 191 / 192 | 0.8105 | 0.9086 | 190 |
| `DADA2000_w32s2` (W=32, stride 2) | **57 / 703** | 0.5000 | **0.9766** | **57** |
| `DADA2000_w24s1` (W=24, stride 1) | 406 / 755 | 0.5000 | 0.8965 | 370 |

The obvious reading — "the clips were too short" — is only true of `w32s2`
(32 × 2 = 64 raw frames needed against a raw median of 49; C32). **`w24s1` kept
406 abnormal test clips and its oracle is still 0.8965**, so retention is not the
root cause.

The root cause is **where the negatives came from**. Both rebuilds drew them from
`0_Normal_Driving`, a separate pool ~3× longer. A fixed hop yields windows in
proportion to clip length, so the long pool dominates:

```
w32s2:    57 abnormal / 703 normal   =  1 : 12
w24s1:   406 abnormal / 755 normal   =  1 : 1.9
T2   :  4053 abnormal / 1962 normal  =  2.07 : 1
```

and the oracle follows the **frame share**, not the leak (C33's formula
`oracle = (F_norm + 0.5X)/(F_norm + X)`):

```
w32s2: abnormal hold  7.5% of frames -> 0.9766
w24s1: abnormal hold   35% of frames -> 0.8965
T2   : abnormal hold   67% of frames -> 0.6631
```

> **Closing the leak and lowering the oracle are two different jobs.** Fixed-length
> windows close the leak — both rebuilds reached exactly 0.5000. Lowering the
> oracle requires changing the frame share between classes, and the only way to do
> that is to take the negatives from inside the abnormal videos.

---

## 3. Phase 0 — download probe (**gate P**)

~30–50 videos spread across the 52 `type` values. ~1–2 GB. Answers what the
annotation cannot.

| id | question | method | fail |
|---|---|---|---|
| **P1** | does `total frames` match the decoded frame count? | `ffprobe -count_frames` vs the xlsx column | **STOP** |
| **P2** | does `(type, video)` resolve to the release's directory layout? | list dirs, join | write a resolver first |
| **P3** | do frames decode at a resolution `preprocess_frames` accepts? | run the extractor on 5 clips | adjust extractor |

**P1 is the hard gate.** `core/data/dada.py:290-299` maps the anomaly window as a
*fraction* of the annotation's `total_frames` and applies it to the on-disk frame
count, warning but never raising on a mismatch. On the trimmed archive that
mismatch is 322 vs 68 and the fraction assumption is silently wrong — a trim
"around the accident" is not proportional. If the original release mismatches
too, every frame label is misplaced and this whole plan is void.

**Pass:** ≥95% of probed clips match within ±2 frames.

---

## 4. Phase 1 — **Gate D0**, the representation ceiling

Extract CLIP features for ~400 original clips and run
`python -m core.tools.eda report --sections features`. **No training.**

The current DADA EDA carries a CRITICAL verdict — *"the features carry no
frame-level signal"*, frame linear probe `auc_macro` **0.5228**. A supervised
linear probe is the ceiling for any head, so if that holds on the original
corpus, no corpus surgery, loss arm or KIP variant can work.

**Two reasons to expect it does not hold:**

1. It was measured on abnormal clips of median **7** sampled frames — ~2 positives
   against ~5 negatives, 10 pairs, macro AUC grid step ≈ 0.1. Nearly no resolution.
2. **DoTA's frame probe is `auc_macro` 0.6708** (`outputs/EDA/DoTA`) on the same
   frozen CLIP features, same transform. The 0.5228 verdict is a property of the
   degenerate DADA build, not of CLIP.

Attainable range (C33): on a 16-frame window with ~8 positives and ~8 negatives the
grid step is ≈0.016, so 0.60 is measurable.

| probe `auc_macro` | decision |
|---|---|
| **≥ 0.60** | PASS → build the corpus, run the full ladder |
| 0.55 – 0.60 | marginal → **one** arm, nothing more |
| **< 0.55** | **STOP** — representation is the ceiling; ship the negative result and the backbone recommendation |

---

## 5. Phase 2 — the T2 corpus

### 5.1 Construction

Cut each original accident video into **16 sampled-frame windows at hop 8**
(`frame_stride` 8, so 128 raw frames per window). A window is abnormal iff it
holds ≥1 positive frame; otherwise it is a genuine negative bag from the same
video, the same camera, the same weather, seconds apart.

```
sampled frame:  0    5   10   15   20   25   30   35   40
                |----|----|----|----|----|----|----|----|
frame label:    0000000000000000000011111111111100000000
                                    +-- [tai, tae] --+

win0 offset  0: [================]                          0 positive  -> NORMAL
win1 offset  8:         [================]                  4 positive  -> ABNORMAL
win2 offset 16:                 [================]         12 positive  -> ABNORMAL
win3 offset 24:                         [================]  8 positive  -> ABNORMAL
```

### 5.2 Simulated shape (annotation only — to be re-measured after Phase 0)

| W | hop | cap | min-pos | win+ | win− | ratio | oracle | 2-class videos | abnormal retained | pos-frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **16** | **8** | **4** | **1** | **4,053** | **1,962** | 2.07 | **0.6631** | 1,159 | **98.1%** | 0.500 |
| 16 | 8 | 4 | 4 | 3,468 | 2,547 | 1.36 | 0.7117 | 1,340 | 98.1% | 0.562 |
| 24 | 12 | 4 | 1 | 2,870 | 843 | 3.40 | 0.6135 | 588 | 88.6% ⚠ | 0.417 |
| 32 | 16 | 4 | 1 | 1,819 | 349 | 5.21 | 0.5805 | 249 | **70.8%** ✗ | 0.344 |

**Chosen: W=16, hop=8, cap=4, min-positive=1.** Longer windows lower the oracle
but fail C32's 90% retention warning and skew the class ratio.

### 5.3 The code already exists on `main`

`b48508a` shipped the machinery; it was built for the trimmed corpus and failed
there for §2.3's reason. Its own docstring states T2's premise:

> *"A window's label is derived from its own sliced frame labels, so an abnormal
> clip's normal prefix becomes genuine negative windows, which is the whole point."*
> — `core/data/dada.py:plan_record_windows`

| | current default | T2 |
|---|---|---|
| frame source | trimmed archive | **original release** |
| `--window-length` | 32 | **16** |
| `--window-stride` | 16 | **8** |
| `--window-max-per-clip` | 4 | 4 |
| `--window-min-positive` | 1 | 1 |
| `--window-weak-mode` | drop | drop |
| `model.score_head_kernel` | 9 | **3** |
| `0_Normal_Driving` | used | **not used** |

Only `score_head_kernel` is a model config; the rest are existing CLI flags.
Kernel 9 over a 16-frame window covers 56.2% (C27 fires again); kernel 3 covers
18.8%. The EDA's own DADA verdict already recommends "try 3".

### 5.4 Open work in `core/`

Nothing is written until `trace_call_path` has been run and its blast radius
reported. Expected touch points, smallest first:

1. `core/data/dada.py` — a resolver for the original `frames/{type}/{video}`
   layout beside the existing fault-dir one (the xlsx has **no `Fault_Label`**;
   its key is `(type, video)`).
2. `core/constants.py` — a `DADA_ORIG_*` cache/dataset name. **New cache dir**
   (C2): `clip/DADA2000_orig_w16s8`, never reuse `DADA2000_ncc`.
3. Optional: carry the xlsx's `texts` / `causes` / `measures` and the 52-type
   taxonomy into `meta.json` only — see `.project/plans/` note on `L_neg` (G4).
   **Not** into `labels_train.json` in this plan.

---

## 6. Phase 3 — EDA and **Gate W**

Run the full profiler on the built corpus and check the simulation held.

| criterion | threshold | derivation |
|---|---|---|
| length leak (clip AUC) | ≤ 0.55 | 0.5000 by construction; slack for the split |
| clip oracle (micro) | ≤ 0.75 | C33 formula; T2's frame share puts it at 0.6631, so **attainable** — unlike the trimmed corpus, where C33 proved < 0.75 impossible |
| abnormal source retention | ≥ 90% | C32; simulated 98.1% |
| class ratio | within 1:3 either way | simulated 2.07:1 |
| kernel coverage (median) | ≤ 35% | kernel 3 over W=16 = 18.8% |
| two-class clips for macro | ≥ 300 | `w24s1` reached 370; T2 should exceed it |

Any miss → re-open §5.2's table, do not proceed.

---

## 7. Phase 4 — training, and how the result is compared

### 7.1 The comparison rule

**A training corpus does not need a published number. The evaluation benchmark
does.** MSAD already works this way here: MSAD is fuel, DoTA is the benchmark.

So: train on T2, evaluate **zero-shot on DoTA val**, which stays held out and
untouched. The in-domain T2 number is a sanity check only, printed beside the
oracle and the length ruler, and **never** placed next to a published
frame-level AUC (C8, C8b, C12).

### 7.2 The DoTA comparison table this fills in

| reference | micro (min-max) | source |
|---|---:|---|
| LaGoVAD published | 0.6260 | paper |
| LaGoVAD released ckpt (**the real gate**, C8b) | 0.6142 / 0.6012 `_ncc` | `RESULTS_DOTA.md` |
| ours, MSAD-trained kip_off | 0.5492 | `outputs/v1/DoTA/DoTA_ncc_s{2024,2025,2026}` |
| ours, MSAD-trained kip_on | **0.6408** | 〃 |
| ours, PreVAD-trained kip_off / on | 0.5868 / 0.5843 | `DoTA_ncc_pv_s*` |
| **ours, T2-trained** | **?** | this plan |

The question it answers, which nobody has: **does in-domain traffic-dashcam
training beat out-of-domain CCTV training (MSAD) on a traffic benchmark?**
The bar to beat is **0.6408**.

### 7.3 Arms

Seeds 2024/2025/2026, KIP-off and KIP-on, `gate_signal=flow_norm` (the only gate
`main` has — **no `kip.gate_type`**; the v3 ladder cannot be run here).
Report `auc_macro` as the headline on T2, min-max micro on DoTA.

**Unlike PreVAD, T2 can host a real KIP A/B**: the original release ships pixels,
so RAFT targets for `L_KIP_rec` are buildable.

---

## 8. Risks and pre-registered cautions

1. **Split by SOURCE VIDEO, never by window.** T2 yields ~3.8 windows per video;
   splitting on windows puts the same accident in train and test. Not yet a
   lesson — candidate filed in `lessons-learned/pending.md`.
2. **Class ratio 2.07:1 toward positives.** `L_MIL` wants negative bags; cap
   windows per clip or mine extra negatives from long pre-accident stretches.
3. **C2 — a new corpus means a new cache.** No existing DADA number is
   invalidated; they remain the record of the trimmed corpus. Do not overwrite.
4. **C17 — write a run manifest.** `--init-weights`, data dir, cache paths, git
   commit, `sys.argv`. `t2_warm` on TAD is unprovable precisely because this was
   skipped.
5. **C30 — seed per item** in any eval that filters or subsets.
6. **Do not tune on the DoTA delta** (C14, lesson 14). The MSAD +0.09 is
   attributed to smoothing and does not replicate on DADA; a third fit to the
   same benchmark is fitting the benchmark.
7. **BDD100K as a normal source is out of scope here.** Different camera,
   country and codec: a source probe would separate it from DADA in a single
   frame, which is strictly worse than the length leak. If it is ever tried, the
   pre-registered test is a logistic regression on CLIP features predicting the
   source — **≥0.90 AUC means the corpus cannot support an in-domain claim.**

---

## 9. Order of work

```
Phase 0  probe P1/P2/P3      ~30-50 videos, ~1-2 GB     GATE P
   |
Phase 1  Gate D0             ~400 clips, features only  GATE D0  <- hard stop
   |
Phase 2  full download, build T2 corpus
   |
Phase 3  EDA + Gate W
   |
Phase 4  train ladder -> eval zero-shot on DoTA vs 0.6408
```

Phases 0 and 1 together are ~2 GB and one evening. If D0 fails, that is a
publishable negative result and a week saved.
