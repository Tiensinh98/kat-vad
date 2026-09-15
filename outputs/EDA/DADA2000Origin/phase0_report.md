# DADA-2000 **original** release — Phase 0 probe report

**Run 2026-09-15 on Colab (CPU runtime), branch `main` (KAT-VAD v1).**
Notebook: `colab/DADA2000Origin/phase_0.ipynb`.
Runbook: `core/docs/DADA_ORIGIN_PHASE0.md` (§8 carries the same verdicts; that
file is the one to cite — this one is the working record beside the artifact).
Plan: `.project/plans/katvad-dada-original-corpus.md` Phase 0.

---

## 0. Verdict

| gate | question | result | verdict |
|---|---|---|---|
| **P2** | archive layout, does `(type, video)` resolve? | `DADA2000/{type}/{video:03d}/images/{frame:04d}.png`; **52/52** types match the xlsx; `missing on disk 0/30` | ✅ **PASS** |
| **P1** | does `total frames` match the real frame count? | **29/30 = 96.7 %** within ±2 — and all 29 match **exactly**; the single miss is `36/002` at **−3** on 345 (−0.87 %) | ✅ **PASS** (§6.1 row 1) |
| **P3** | do frames decode at a resolution `preprocess_frames` accepts? | **5/5 ok, 0 failed**; native **660 × 1584** → `(8, 3, 224, 224)` float32 | ✅ **PASS** |

**All three pass. The plan is not void; the next gate is D0** (frame linear
probe, PASS ≥ 0.60, STOP < 0.55) — `katvad-dada-original-corpus.md` §4.

**Phase 0 is a probe, not a corpus-wide proof.** See §5 for exactly what these
30 clips do and do not establish.

---

## 1. Provenance of the probe

```
pick_probe.py --xlsx "$DADA_ORIG/dada标注.xlsx" --n 30 --seed 2024
  sheet: 'Sheet1'  (19 columns)          <- DETECTED, not hardcoded
  1945 annotated clips over 52 types -> picked 30
```

Selection: one clip per `type` (all 52), shuffled with `Random(2024)`, truncated
to 30. So the sample is **30 distinct types, one clip each** — 22 types are
unrepresented.

Artifact: `probe_clips.json` (this directory), 30 rows of
`{type, video, start, end, total}`. Verified locally: 30 unique `(type, video)`,
30 unique types, **0 rows violating `0 ≤ start < end ≤ total`**.

| property of the 30 picked clips | value |
|---|---|
| `total frames` — min / p25 / median / p75 / max | 162 / 243 / **346** / 420 / 632 |
| sampled length at `FRAME_STRIDE = 8` — min / median / max | 20 / **43** / 79 |
| positive fraction `(end−start)/total` — min / median / max | 0.116 / **0.376** / 0.629 |

Two readings worth carrying forward:

* **Median 43 sampled frames** vs the trimmed archive's **9**. `score_head_kernel=9`
  spans 20.9 % of the median clip here instead of 100 % — which is the whole
  point of switching corpora (C27). The plan's `9 → 3` still applies to the T2
  **windows** (W = 16), not to these full-length clips.
* **Positive fraction median 0.376**, close to the archive's 0.351. **C29 is
  untouched by the corpus swap**: DVS would still train ~62 % of anchor frames to
  1 against a 0 annotation. `loss.dvs_anchor_mode=ignore` remains the relevant
  arm; nothing here retires that lesson.

---

## 2. P2 — layout (`7z l`, no extraction)

```
DADA2000/{type}/{video:03d}/{subdir}/{frame:04d}.png
52 types on disk: [1..24, 30, 33, 34, 36..45, 47..61]  == the 52 in the xlsx
```

| subdir | files | GiB | needed |
|---|---:|---:|---|
| **`images`** | 651,320 | **94.01** | ✅ RGB frames |
| `maps` | 651,325 | 11.96 | ❌ driver-attention |
| `seg` | 651,320 | 10.33 | ❌ |
| `semantic` | 651,320 | 4.04 | ❌ |
| `fixation` | 1,302,640 | 3.16 | ❌ gaze, 2 files/frame |

Aggregate: **651,320 on disk vs 649,399 annotated = +0.30 %** (~+1 frame/clip),
against **−79 %** for the trimmed archive. An index convention, not a trim.

**Sizing (measured, cell 5):** `/content` free **87 G of 108 G**; archive
**117 G** over 6 volumes; `images` 94.01 GiB. Neither fits →
**Phase 2 must shard** (~200 clips ≈ 9.6 GiB: extract → features → delete →
next). Persisted output ≈ `651,320 / 8 × 512 × 4 B` ≈ **167 MB**.

---

## 3. P1 — the hard gate

```
checked   30   missing on disk 0
matched within +-2: 29/30 (96.7%)

clip                               annotation  on disk   delta
DADA2000/36/002/images                    345      342      -3
```

### 3.1 An independent cross-check, stronger than the 96.7 %

`sum(total)` over the 30 picked rows = **10,326**.
`7z x` reported `Files: 10323`. Difference **exactly −3** — the one `36/002`
miss, and nothing else.

Two conclusions in one number: **29 of 30 clips match the annotation to the
frame** (the ±2 tolerance was never used), and **the extraction pulled `images/`
and nothing else** (had a second subdir leaked in, the count would be ~5× off).

### 3.2 ⚠️ The warning line in the raw output is a FALSE ALARM

The original `check_p1.py` closed with:

> *"A systematic one-sided delta means the clips were TRIMMED … the fraction
> mapping in `core/data/dada.py:296` is then WRONG."*

It printed that over **one** mismatch of **−0.87 %**. The defect it describes
measures **−79 %** on the trimmed archive. §6.1 row 1 (≥ 95 % matched) fires;
row 3 (< 80 %, **or** a strongly one-sided mean) does not. **P1 PASSES.**

The script emitted the prose unconditionally whenever `deltas` was non-empty —
a tool defect, not a data defect. **Fixed** in `DADA_ORIGIN_PHASE0.md` §6: the
trim verdict now requires all three of `≥ 3` mismatches, `≥ 90 %` sign agreement
and `≥ 2 %` relative mean, and the script prints §6.1's decision table verbatim
as a `VERDICT:` line. Re-run against a fixture reproducing this exact probe:

```
matched EXACTLY (delta 0): 29/30
mismatches 1   mean signed delta -3.0 (0.87% of the median clip)   sign agreement 100%
NOT a trim signature: needs >= 3 mismatches, >= 90% sign agreement and >= 2% relative mean.
VERDICT: PASS (96.7% >= 95%) -- the annotation describes the real files.
```

Four fixtures exercised: this probe (PASS), a −79 % trim (STOP), an empty root
(BACK TO P2), and 83 % matched with two-sided ±5 deltas (PASS WITH A NOTE).
Lesson candidate filed in `.project/memory-bank/lessons-learned/pending.md`.

### 3.3 The xlsx on Drive was never `md5sum`-ed

§1.2's precaution was skipped: cell 5's `ls -la "$DADA_ORIG"` printed the
**symlink**, not the directory behind it. A valid substitute landed by accident —
`pick_probe.py` ran against the **Drive** copy and printed
`1945 annotated clips over 52 types`, identical to the count measured on the
committed `data/DADA/dada标注.xlsx`. The two agree on the **population**; that is
not a claim that they are byte-identical.

---

## 4. P3 — decode, preprocess, and a new fact

```
images: 8 frames, native (660, 1584) -> (8, 3, 224, 224) torch.float32 range[-1.748,2.146]
...  (5 clips)
P3: 5 ok, 0 failed
```

Ranges ≈ [−1.79, 2.15] are CLIP-normalized output, as expected.
`center_crop=False` was used — mandatory, the whole tree is `_ncc` (C2, C13).

### 4.1 Native **1584 × 660** = **2.40 : 1** — recorded nowhere before now

`preprocess_frames(..., center_crop=False)` resizes **anisotropically** to a 224
square (`core/tools/extract_clip_features.py:57-72`, the `else` branch). It does
not preserve aspect ratio, deliberately — that is LaGoVAD's `no_center_crop`.

| corpus | native `H × W` | aspect | horizontal squash |
|---|---|---:|---:|
| DoTA | 720 × 1280 | 1.78 : 1 | 1.78× |
| **DADA-2000 original** | **660 × 1584** | **2.40 : 1** | **2.40×** |

**DADA-original frames are distorted 1.35× more than DoTA's.**

* **Not a C2/C13 violation.** One transform, one cache dir — this corpus gets
  `clip/DADA2000_orig_w16s8`; no existing cache or checkpoint is touched.
* **It is a domain-shift term on the DoTA-transfer column**, which is the column
  this project is built around. Named here, *before* the measurement, so it
  cannot be produced afterwards as an excuse.
* **Do not "fix" it** by moving this corpus to `center_crop=True`: that makes it
  the only corpus on a different transform, breaks the appearance/flow
  field-of-view match (C13), and voids comparison with every `_ncc` artifact
  since 2026-08-12.

---

## 5. What these 30 clips do NOT establish

1. **A corpus-wide P1 rate.** 29/30 has a Wilson 95 % lower bound of **≈ 0.83**.
   "≥ 95 % across 1,962 clips" is unproven, and P1 is a **hard stop** — re-run it
   over every clip in Phase 2.
2. **Resolution uniformity.** 5 clips × 8 frames, all 1584 × 660. The original
   release is not guaranteed uniform across 52 types. Follow-up (a) in
   `DADA_ORIGIN_PHASE0.md` §7.1 sweeps all 30 from PNG headers, no decode.
3. **That all 1,962 clip directories exist.** Cell 10 counted *types*, not
   *videos*. §2's +0.30 % aggregate assumes full presence; a shortfall and an
   overcount would cancel inside it. Follow-up (b) in §7.1 counts directories
   from the already-written `/content/listing.txt`.
4. **Anything about Gate D0.** Phase 0 says the data is what the annotation
   claims. It says nothing about whether frozen CLIP features carry frame-level
   signal here (`DADA_ORIGIN_PHASE0.md` §0.2).

Also worth noting for reproducibility (C17): P3's log printed `folder.name`,
which is always the literal `"images"`, so the raw output does not identify which
5 clips were probed. They are the first 5 rows of `probe_clips.json`:
types 33/001, 44/002, 42/013, 47/001, 39/016.

---

## 6. Next

1. §7.1 follow-ups (a) resolution sweep and (b) clip-directory count — cheap,
   run at the top of the Phase 1 session.
2. **Gate D0** — extract CLIP features for ~400 clips, run
   `python -m core.tools.eda run --sections features`. Bar: frame linear probe
   `auc_macro` **≥ 0.60**; **< 0.55 stops the plan** and the deliverable becomes
   the negative result. Reference points: DoTA 0.6708 on the same features and
   transform family; the DADA *archive* 0.5228.
3. Only then Phase 2 (T2 windowing, W = 16 hop 8, `score_head_kernel` 9 → 3),
   sharded per §2's sizing.

**Do not read a P1 pass as permission to skip D0** (`DADA_ORIGIN_PHASE0.md` §0.2,
trap 7).
