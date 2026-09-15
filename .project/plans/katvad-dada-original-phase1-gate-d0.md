# KAT-VAD — DADA-2000 original, **Phase 1 / Gate D0** (branch `main`)

**Written 2026-09-15**, after Phase 0 passed P1/P2/P3
(`core/docs/DADA_ORIGIN_PHASE0.md` §8, `outputs/EDA/DADA2000Origin/phase0_report.md`).
Parent plan: `.project/plans/katvad-dada-original-corpus.md` §4.
This document refines that §4; where they disagree, **this one wins** — §4 was
written before anyone read `core/tools/extract_clip_features.py`.

> **Approach chosen: B — zero changes to `core/`.** Reasoning in §2.

---

## 0. The gate, in one line

Extract frozen-CLIP features for **~400 original DADA-2000 clips** and measure
the **supervised frame linear probe** `auc_macro`. **No training, no model.**

| probe `auc_macro` | decision |
|---|---|
| **≥ 0.60** | **PASS** → Phase 2, build the T2 corpus |
| 0.55 – 0.60 | marginal → **one** arm, nothing more |
| **< 0.55** | **STOP** — the representation is the ceiling; ship the negative result |

> ## ✅ RESULT 2026-09-15 — **PASS. `auc_macro` = 0.6518** (`d0_abs` 0.6492, Δ 0.0026)
>
> 400 clips, 52/52 types, 15,961 frames, coverage 400/400.
> **DoTA 0.6708 · this corpus 0.6518 · DADA trimmed archive 0.5228** — the
> original release is 0.019 off the held-out benchmark and 0.129 above the
> corpus this project had been training on. The archive's CRITICAL *"features
> carry no frame-level signal"* verdict is confirmed as a property of that build.
> P1 re-ran at n = 400: **397 exact (99.2 %)**, superseding Phase 0's n = 30.
> Full record: `outputs/EDA/DADA2000Origin/gate_d0_report.md`.

Reference points on the same frozen CLIP, same `_ncc` transform:
**DoTA 0.6708** (`outputs/EDA/DoTA`) · **DADA trimmed archive 0.5228**
(`outputs/EDA/DADA2000`). D0 asks which of those two the original release
resembles.

**A probe is a ceiling, not a promise.** Passing D0 says a linear head *could*
localize on these features. It says nothing about whether WS-MIL training will —
TAD passed nothing like this and still collapsed (C14's family). Do not let a
PASS grow into a claim.

---

## 1. What Phase 0 left, and what blocks §4 as written

Five findings, all measured this session. They are why §4's two-line recipe
does not run.

### 1.1 The command in §4 is wrong

`python -m core.tools.eda run …` — the subcommand is **`report`**
(`core/tools/eda.py:39`); `run` dies in argparse. Fixed in the parent plan, the
runbook, `phase0_report.md` and `activeContext.md`.

### 1.2 D0 needs labels, and the parent plan schedules the label code for Phase 2

`core/eda/corpus.py:load_dataset_files` **requires four files** and raises on any
missing: `labels_train.json`, `frame_labels_test.json`, `defs.json`, `meta.json`
(`windows.json` is optional — `load_windows` returns `None`).
`core/eda/features.py:346` reads `files.frame_labels_test` to length-check the
cache, and both probes key on it.

So D0 cannot run on a bare feature cache. The parent plan puts the original-layout
resolver in **Phase 2** §5.4 — a dependency inversion.

### 1.3 C26 — the extractor collapses 1,962 clips into 255, silently

`core/tools/extract_clip_features.py:194`

```python
folders = select_ids(
    {video_id_from_path(p): p for p in list_frame_folders(frames_dir, subdir)}, …)
```

`video_id_from_path` returns `path.name` (`core/data/video_io.py:63`). On the
original layout `DADA2000/{type}/{video:03d}/images/`, `path.name` is the
**video number alone**:

```
1,962 unique (type, video)  ->  255 distinct ids
video 001 collides across 52 types, 002 across 41, 003 across 40
```

A dict comprehension keeps the last writer. Nothing raises. Then `pending_items`
skips what exists (C11), so a re-run reports a clean resume over a cache that is
87 % missing. **This is exactly lesson C26.**

**Do not fix it in `video_id_from_path`.** `trace_path` inbound, depth 3:

| hop | risk | callers |
|---|---|---|
| 1 | **CRITICAL** | `extract_clip_features.extract_frame_directory`, `raft_extract.extract_frame_directory`, `dota.resolve_frame_counts`, `tad.parse_annotation_file`, `tad.frame_folders_by_id` |
| 2 | HIGH | `extract_clip_features.main`, `raft_extract.main`, `dota.{preprocess,resolve_frame_counts}`, `tad.{preprocess,resolve_records,resolve_train_records}` |
| 3 | MEDIUM | `dota.main`, `tad.main`, both extractor modules |

Changing it touches every corpus in the project, DoTA and TAD included. **Per
§13's rule, HIGH/CRITICAL is a stop, not a speed bump.**

**The remedy already exists and needs no edit:** `core/data/dada.py:401`
`materialize_flat_dir` — a symlink farm `flat/{video_id} -> real folder`, whose
docstring names this exact problem:

> *"`extract_clip_features.py`/`raft_extract.py --frames-dir` key their
> (dataset-generic) folder scan on bare `Path.name`, so they cannot resolve a
> fault-dir-prefixed `video_id` … Pointing them at this flat dir instead needs no
> change to either tool."*

### 1.4 The existing DADA adapter cannot read the original release

| symbol | assumes | original release |
|---|---|---|
| `_FOLDER_RE` (`dada.py:88`) | flat `type<N>_vid<N>` folder names | nested `{type}/{video:03d}` → **raises** |
| `_make_video_id` (`:164`) | a `fault_label` prefix | **no `Fault_Label` column exists** |
| `frame_folders_by_type_vid` (`:179`) | one fault directory | 52 type directories |
| `parse_metadata_csv` (`:214`) | `Cleaned_Metadata.csv`, 975 rows | the xlsx, **1,962** rows |
### 1.5 MEASURED — a symlink farm must point at the **images** directory

Found while building §3's script, not predicted. `pathlib` refuses to recurse
**into** a symlinked directory (cycle guard; verified on this tree's Python
3.10.6, and the behaviour is unchanged through 3.13). `list_frame_folders`
(`core/data/video_io.py:79`) walks `root.rglob(subdir)`, so:

| farm layout | extractor call | result |
|---|---|---|
| `flat/{id} -> …/{video:03d}` | `--frames-subdir images` | **`ValueError: No frame folders found`** — `rglob("images")` never descends through the link |
| **`flat/{id} -> …/{video:03d}/images`** | **no `--frames-subdir`** | ✅ 30 folders, 30 unique ids — the link is matched by the *final* component of `rglob("*")`, and `list_frame_images` follows it |

It fails loudly, which is the good case. But it also means
`core.data.dada.materialize_flat_dir` works for the **trimmed** archive only
because that layout has no image subdir. **Phase 2's adapter must link at the
images level too**, or it inherits this.

**Measured on the 30-clip fixture:** the raw tree yields **20 unique ids for 30
folders** (10 clips silently lost); the flat farm yields **30 for 30**.

---

## 2. Internal debate — A vs B vs C

### Option A — write the real adapter in `core/data/dada.py` now

Original-layout resolver + xlsx parser + flat dir, then extract, then D0.

* **Pro:** the honest artifact; reused verbatim by Phase 2; tests can cover it.
* **Con:** it modifies a module with the caller graph of §1.3 **before a hard-stop
  gate**. If D0 reads < 0.55 the plan stops and that work is dead. This
  contradicts the parent plan's own §9: *"Phases 0 and 1 together are ~2 GB and
  one evening. If D0 fails, that is a publishable negative result and a week
  saved."* Spending the week before the gate defeats the gate.
* **Risk:** a partial adapter that "mostly works" becomes the thing Phase 2
  inherits without review — the shape of C24 (a plausible module nobody re-read).

### Option B — a Phase-1-only builder outside `core/` ← **CHOSEN**

A script beside `pick_probe.py`, living in the runbook and `colab/`, that emits
the four dataset files from the xlsx and builds the symlink farm. `core/` is
**read-only**: `extract_clip_features` and `core.tools.eda report` run unmodified.

* **Pro:** zero blast radius. Nothing to review, nothing to revert, no cache or
  checkpoint touched. If D0 fails, the cost is one evening and ~20 GiB of
  scratch. If it passes, Phase 2 writes the adapter knowing the gate is clear.
* **Con:** the script duplicates a slice of `dada.py`'s job, and **a label
  convention that drifts from the pipeline's would make D0 measure the wrong
  thing** — the real risk here, and a C2/C13-shaped one.
* **Mitigation (the reason B is safe):** the script **imports** the convention
  rather than restating it —
  `core.data.dada.sampled_frame_labels`, `core.data.dataset_files.num_sampled_frames`,
  `core.data.dada.DadaRecord`, `class_name_list`, `constants.*`. It constructs
  real `DadaRecord` objects and calls the project's own label function. Import is
  read-only; the risk collapses to "did we fill the dataclass correctly", which
  §4.3's assertions check.

### Option C — skip D0, build the corpus, train, see what happens

Rejected. It is the DADA-archive campaign again: three arms, a diagnosis document
and a retracted claim. D0 exists precisely because that happened.

### Decision

**B.** A + B differ only in *when* the adapter is written; B writes it after the
gate instead of before, at no measurement cost, because the label convention is
imported rather than reimplemented.

---

## 3. What gets built (all outside `core/`)

```
core/docs/
  DADA_ORIGIN_PHASE1.md  NEW  the runbook AND the durable copy of the builder
                              (inline %%writefile, as Phase 0 does) -- `colab/`
                              is gitignored (.gitignore:46), so a script that
                              lives only there does not survive
colab/DADA2000Origin/      (gitignored -- working copies only)
  pick_probe.py          (exists, Phase 0)  --n 400 --seed 2024
  build_d0_dataset.py    NEW  clips json + on-disk counts -> 4 JSON files + farm
  phase_1.ipynb          NEW  the session, cell by cell
outputs/EDA/DADA2000Origin/
  d0_frac/ d0_abs/       the two dataset dirs (small JSON, committable)
  eda_report.{json,md}   the gate's own record
```

### 3.1 `build_d0_dataset.py` — contract

**Input:** the xlsx, the extracted frame tree, `--stride 8`.
**Output**, into `--out-dir`:

| file | content |
|---|---|
| `frame_labels_test.json` | `{video_id: [0/1] * num_sampled_frames}` — **every** clip; D0 is a test-only measurement |
| `labels_train.json` | `{}` — empty on purpose; no training happens in Phase 1 |
| `defs.json` | `class_name_list({constants.DADA_CLASS_NAME})` = `["Normal", "CarAccident"]` |
| `meta.json` | per clip: `type`, `video`, `total_frames` (annotation), `frames_on_disk`, `sampled_frames`, `positive_frames`, `normalized_span`, `split: "test"` — C17 |
| `flat/{video_id}` | symlink → `DADA2000/{type}/{video:03d}/`**`images`** — the images dir itself, see §1.5 |

`video_id` = **`t{type:02d}_v{video:03d}`** — unique by construction, no
`Fault_Label`, and it carries the `(type, video)` key so Phase 2 can re-derive it.
It deliberately differs from `dada.py`'s `{fault_label}__{folder}` scheme: these
two corpora must never share a cache directory (C2).

### 3.2 Two label constructions, both written

* **`d0_frac/` — primary.** `core.data.dada.sampled_frame_labels`, i.e. the span
  as a **fraction** of `total_frames` applied to the sampled length. This is what
  the pipeline will do in Phase 4, so it is what the ceiling must be measured
  under.
* **`d0_abs/` — sanity arm.** Absolute indices, `start // stride … end // stride`.
  Strictly more correct on an untrimmed release.

**Attainable range, derived before the threshold (C33) — corrected after
measuring.** An earlier draft of this plan pre-registered "≤ 1 sampled frame of
disagreement". That was underived and wrong: `sampled_frame_labels` **rounds**
each boundary while the absolute arm floors the start and ceils the end, so the
two constructions differ by up to **one frame per boundary = 2 per clip** by
arithmetic alone. Measured on the 30-clip fixture: **15 of 30 clips differ, worst
2 frames, 16 frames total** — exactly the rounding, nothing else.

**Pre-registered, corrected:**

* per-clip label disagreement **> 2 frames** ⇒ the frame counts disagree, not the
  rounding. Investigate before extracting.
* `|auc_macro(frac) − auc_macro(abs)| > 0.01` ⇒ **both numbers are void** until
  explained. This is the bar that matters; the frame-level one is its early
  warning.

> ### CORRECTED AGAIN 2026-09-16, after the run — the "2 frames" bound was still under-derived
>
> The D0 run fired the warning: **256 clips differ, worst 17 frames**. The bound
> above is right **only when the on-disk count equals the annotation's**. The
> fraction mapping computes `round((start/A) · L)` with `L = ceil(D/s)`, so when
> `D ≠ A` the whole window is rescaled by `D/A` and the shift grows as
>
> ```
> shift ≈ (start / s) · |D − A| / A
> ```
>
> For `t05_v040` (**−100** frames) that is ~10–15 sampled frames — which is the
> observed 17. 253 of the 256 differing clips differ by the rounding-only 1–2;
> the tail is the three clips of the 400 whose frame counts are inexact.
>
> **Corrected bound:** *≤ 2 sampled frames for clips whose on-disk count matches
> the annotation; otherwise ~`(start/s)·|D−A|/A`.*
>
> **This is the second under-derived threshold in this plan** (the first was the
> "≤ 1 frame" bar, corrected on 2026-09-15) and the third project-wide instance of
> **C33**. The AUC bar, which is the one that decides, was met with room to spare:
> **0.0026 vs 0.01**, on 311 disagreeing labels out of 15,961 (**1.9 %**). The
> verdict does not move.
>
> **Rule for Phase 2:** when a threshold is conditional, write the condition into
> the threshold, not into the prose beside it.

Running the probe twice costs seconds and shares one feature cache.

### 3.3 Sample

`--n 400 --seed 2024`, type-stratified exactly as Phase 0 (one per type, then
fill). ~400 × ~49 MiB ≈ **19.6 GiB** of PNGs.

> **CORRECTED 2026-09-15, after it failed.** This paragraph said "against 87 GiB
> free on `/content` → one pass, no sharding". **Wrong, and the error was
> mine:** the 87 GiB was measured on Phase 0's **CPU** runtime, while §4 asks for
> a **GPU** runtime — a different VM with a much smaller disk. The one-pass
> extraction died with `System ERROR: errno=28 : No space left on device` after
> `7z` had written most of ~20 GiB.
>
> **Phase 1 shards too.** 40 clips per shard (~2 GiB): extract → build shard farm
> → encode → **delete frames** → next. The builder gained `--mode shard`
> (census out, no dataset dirs) and `--mode labels` (dataset dirs from the merged
> censuses, no frames on disk). Verified byte-identical to a one-pass `full` run
> on a 30-clip fixture split into three shards with the frames deleted between
> each — all four JSON files match exactly.
>
> **The persisted artifact was always tiny:** 400 clips × ~43 sampled frames ×
> 512 float32 ≈ **35 MB**. The 20 GiB is scratch, and now it is treated as such.

**Two clips per shard is not a sample-size change** — the 400 picked clips and
their labels are identical either way; only the disk lifetime differs.

**Every clip is abnormal — that is intended, not a defect.** The original release
has zero normal videos (parent plan §2.2). The frame probe needs two classes *in
frames*, which it has: positive fraction median **0.376**, so a median clip gives
~16 positive and ~27 negative sampled frames.

**Consequence to pre-register:** `clip_linear_probe` will return
`{"ran": false, "reason": "every clip has the same clip-level label"}`
(`core/eda/features.py:276`). **That is the expected output, not a failure** — and
it is exactly why the original release is interesting: there is no clip-level
shortcut to find. Do not "fix" it by importing `0_Normal_Driving`; that is the
pool whose length inverted the leak (`activeContext` 2026-09-15 §5).

---

## 4. Execution

### 4.1 Order

```
1  pick 400 clips            pick_probe.py --n 400 --seed 2024   (needs the fill
                               pass -- the old script capped at 52, see runbook §2)
2  measure free disk         shutil.disk_usage('/content')  -- GPU != CPU runtime
3  per shard of 40 clips:
     3a extract images/        7z x, ~2 GiB
     3b shard farm + census    build_d0_dataset.py --mode shard
     3c CLIP features          extract_clip_features --frames-dir farm/flat
                                 --no-center-crop --stride 8  (NO --frames-subdir)
     3d DELETE the frames      the cache is the artifact, ~35 MB for all 400
4  dataset dirs              build_d0_dataset.py --mode labels --counts-in ...
5  the gate                  core.tools.eda report --sections features --probe
6  read auc_macro            against 0.60 / 0.55
```

With > 25 GiB free, 3a–3d collapse into one pass (`--mode full`); the output is
the same.

### 4.2 The two commands that must not be typed from memory

```bash
python -m core.tools.extract_clip_features \
  --frames-dir  "$D0/flat" \
  --dataset     DADA2000_orig \
  --stride      8 \
  --no-center-crop \
  --output-dir  "$KATVAD_CACHE_ROOT/clip/DADA2000_orig"
```

* **No `--frames-subdir`.** The farm links at the images level (§1.5); passing
  `--frames-subdir images` makes the scan raise *"No frame folders found"*.
* **`--no-center-crop` is mandatory.** The whole tree is `_ncc` since 2026-08-12
  (C2, C13). Omitting it silently builds a cache on the wrong transform.
* **Cache name `DADA2000_orig`, not `DADA2000_orig_w16s8`.** The parent plan §5.4
  names the latter, but `core/eda/corpus.py:43` and `core/data/windows.py`
  `FeatureSlicer` show features are keyed by **source clip** and windows slice
  into them. One full-clip cache serves Phase 1 **and** Phase 2's T2 windows.
  A window suffix on a clip-keyed cache is a lie that costs a re-extraction.
* **Never `DADA2000_ncc`** — that is the trimmed archive's cache (C2).

```bash
python -m core.tools.eda report \
  --dataset    DADA2000_orig \
  --data-dir   "$D0/d0_frac" \
  --clip-dir   "$KATVAD_CACHE_ROOT/clip/DADA2000_orig" \
  --sections   features \
  --probe \
  --output-dir outputs/EDA/DADA2000Origin/d0_frac
```

### 4.3 Assertions the builder must make (fail loud, C5's spirit)

1. Every `video_id` unique; count equals the number of picked clips.
2. `len(frame_labels[id]) == num_sampled_frames(frames_on_disk, 8)` for **every**
   clip — this is the cache/label alignment `features.py:346` checks later, and
   catching it here is cheaper than after a 20 GiB extraction.
3. Per-clip `|frames_on_disk − total_frames|`: report the distribution. This is
   **P1 re-run at n ≈ 400**, closing §8.1's Wilson-interval gap for free.
4. Every symlink target exists and holds ≥ 1 PNG.
5. `0 ≤ start < end ≤ total` for every row, and ≥ 1 positive sampled frame after
   labelling — count and name the clips whose window vanishes at stride 8
   (`build_frame_labels` warns about exactly this).

---

## 5. Risks

| risk | impact | control |
|---|---|---|
| Label convention drifts from the pipeline's | D0 measures a ceiling the model will never see | import `sampled_frame_labels`, never restate it; §3.2's two-construction check |
| C26 collision reappears (someone points `--frames-dir` at the raw tree) | cache 87 % missing, no error | the symlink farm is the only supported path; assertion 1 + a coverage log |
| `--no-center-crop` forgotten | wrong transform, unusable number (C2/C13) | it is in §4.2's block; the extractor logs which transform it used — read that line |
| Reading a PASS as "KIP will work" | a claim the data cannot support | §0's last paragraph; D0 is a ceiling |
| `clip_linear_probe` "not ran" read as a failure | a correct null misfiled as a bug | pre-registered in §3.3 |
| 400 clips is still a sample | corpus-wide claims unproven | assertion 3 makes it the largest P1 yet; Phase 2 still re-runs corpus-wide |
| `/content` fills mid-extraction | **HAPPENED** 2026-09-15: `errno=28` on the one-pass 400-clip extract, because the GPU runtime is not the CPU runtime Phase 0 measured | shard at 40 clips (~2 GiB) and delete after encoding; `shutil.disk_usage` printed before every shard; the loop is resumable per shard |
| `--n` silently capped at the number of strata | **HAPPENED**: `--n 400` returned **52**, every downstream check passed on the wrong sample size | `pick_probe.py` now fills and prints `picked N` + `types covered: X/52`; runbook §2 says stop if it prints 52 |

---

## 6. What does NOT happen in Phase 1

* **No `core/` edit.** Not one line. If something in `core/` turns out to be
  needed, that is a finding to report, not a patch to slip in.
* **No training, no checkpoint, no DoTA evaluation.**
* **No windowing.** T2 is Phase 2. D0 runs on full-length clips, whose median 43
  sampled frames give a *finer* probe grid than the 16-frame windows the parent
  plan §4 worried about.
* **No `L_neg` / caption work.** Gap G4 stands; `texts` stays out
  (`activeContext` 2026-09-15 §2).
* **No definition-coverage work (C19).** D0 touches no text path; `defs.json` is
  loaded and counted only. Coverage is Phase 2's.

---

## 7. Deliverables

1. `core/docs/DADA_ORIGIN_PHASE1.md` — the runbook, scripts inline.
2. `colab/DADA2000Origin/build_d0_dataset.py` **(written 2026-09-15)** +
   `phase_1.ipynb`. Quality gate clean (`ruff`, `mypy`, `bandit`, `pyright`), and
   exercised on a 30-clip fixture rebuilt from `probe_clips.json`: it reproduces
   P1 (29 exact, one −3), both dataset dirs load through
   `core.eda.corpus.load_dataset_files`, and the flat farm resolves 30/30 unique
   ids where the raw tree gives 20/30.
3. `outputs/EDA/DADA2000Origin/{d0_frac,d0_abs}/` + the two `eda_report.*`.
4. A `RESULTS` section appended to `phase0_report.md`'s sibling, or a new
   `gate_d0_report.md`, carrying: the two `auc_macro` values, their delta against
   §3.2's 0.01 bar, the per-clip P1 distribution at n ≈ 400, and the decision.
5. `activeContext.md` + a lesson candidate if anything new is learned.

**Then stop.** Phase 2 is a separate decision, taken on the number.
