# DADA-2000 **original** release — Phase 0 probe on Colab (branch `main`)

**Written 2026-09-15.** Phase 0 of `.project/plans/katvad-dada-original-corpus.md`.
Companion docs: `DADA_SETUP.md` (the *trimmed* archive runbook — different corpus,
do not mix commands), `DATA_LAYOUT.md`, `COLAB.md` (session mechanics), `EDA.md`.

> ## ⚠️ Branch check — run this first
>
> ```bash
> git branch --show-current      # must print: main
> ```

---

## 0. What Phase 0 is, and what it is not

Phase 0 downloads **nothing new** and extracts **~30 clips**. It answers the three
questions the annotation cannot, before anyone spends a day on a full extraction.

| id | question | fail ⇒ |
|---|---|---|
| **P2** | what is the archive's internal layout, and does `(type, video)` resolve to it? | write a resolver before extracting |
| **P1** | does the annotation's `total frames` match the real frame count? | **STOP — the whole plan is void** |
| **P3** | do the frames decode at a resolution `preprocess_frames` accepts? | adjust the extractor |

### 0.1 Why P1 is a hard stop

`core/data/dada.py:290-299` maps the anomaly window as a **fraction** of the
annotation's `total_frames`, then applies that fraction to the **on-disk** frame
count. It logs a warning on a mismatch and **never raises**:

```python
if on_disk != row.total_frames:
    LOGGER.warning("%s: %d images on disk but annotation says %d frames; using disk", ...)
...
span=(row.start / row.total_frames, row.end / row.total_frames),
```

A fraction is only correct if the clip was trimmed **proportionally at both ends**.
On the reconstructed archive this project has been using, the mismatch is
**322 vs 68 raw frames** and the trim is *around the accident* — i.e. head-heavy,
not proportional. Every frame label there may be misplaced, silently.

If the original release mismatches too, its labels are misplaced the same way and
nothing downstream is worth building. **Measure it before you download 100 GB.**

### 0.2 Phase 0 is not Gate D0

Gate D0 (the supervised linear probe, `plan §4`) comes *after* Phase 0 and needs
~400 clips of CLIP features. Do not conflate them.

---

## 1. What is on Drive

The shortcut added 2026-09-15:

```
$KATVAD_DATA_ROOT/DADA2000Origin/DADA2000/
    DADA2000.z01   DADA2000.z02   DADA2000.z03
    DADA2000.z04   DADA2000.z05   DADA2000.zip
    dada标注.xlsx
```

### 1.1 ⚠️ These six files are ONE archive, not six

`.zip` + `.z01`…`.z05` is a **spanned (split) PKZIP archive**. Two consequences
that cost an hour each if you learn them the hard way:

1. **`unzip DADA2000.zip` fails.** By the PKZIP spanning convention the `.zip`
   part is the **last** segment and holds the central directory; the payload is
   spread across `.z01`–`.z05`. All six must sit in the same directory under
   their exact original names.
2. **`zip -s 0 DADA2000.zip --out joined.zip` needs 2× the space.** Recombining a
   ~100 GB archive wants another ~100 GB free. Colab will not have it.

**Use `7z`.** It reads spanned archives natively — no recombine, no second copy,
and it can list or extract *selected* entries without touching the rest.

```bash
apt-get -qq install -y p7zip-full
```

### 1.2 The annotation is already in the repo

`data/DADA/dada标注.xlsx` is committed locally, and the archive on Drive carries
its own copy. They have **not** been compared; the numbers below are measured on
the local one. A one-line precaution, not a blocker:

```bash
%%bash
md5sum "$DADA_ORIG/dada标注.xlsx"   # vs: md5sum data/DADA/dada标注.xlsx
```

### 1.3 ⚠️ The two sheets are named the opposite of what they contain

```
name="text"    -> rId1 -> worksheets/sheet1.xml   the type 1-38 taxonomy
name="Sheet1"  -> rId2 -> worksheets/sheet2.xml   the 1,962-row data table
```

**The per-clip annotation lives in `Sheet1`. `text` is the category catalogue.**
The first draft of §4 hardcoded `SHEET = "text"` and parsed zero rows; the failure
then looked like a corrupt file on Drive, which it was not.

`pick_probe.py` now **detects** the sheet by looking for the required columns and
prints which one it chose, so the name cannot matter again. Do not reintroduce a
hardcoded sheet name for this workbook.

Its 1,962 rows are a strict superset of
`Cleaned_Metadata.csv`'s 975 — **0/975 mismatches** on `texts` and on
`(abnormal start frame, abnormal end frame, total frames)`. The CSV adds only a
`Fault_Label` column, which **the xlsx does not have** — the xlsx's key is
`(type, video)`, which is what P2 has to resolve.

1,945 rows carry `whether an accident occurred = 1` and a window satisfying
`0 <= start < end <= total`. Those are the Phase 0 population.

---

## 2. Colab session setup

### 2.0 What this phase needs, and what it does not

**Drive must be mounted** — the archive and the repo both live there. But Phase 0
only **reads** Drive; everything it produces goes to `/content` (§5's frames,
`probe_clips.json`, the helper scripts). Only §8's report copy writes back, and
that is optional.

**No GPU.** Nothing here loads a model: `encode_frame_dir` — the one function
that needs one — is Phase 1's business. Run Phase 0 on a **CPU runtime**: it
costs no GPU quota and you will not lose the session to a timeout midway through
a `7z` extraction.

**Minimal install for §3–§6** (no torch, no CLIP):

```bash
%%bash
apt-get -qq install -y p7zip-full
pip install -q openpyxl
```

**§7 alone** needs the project environment (it imports `core.tools.
extract_clip_features`), so run §2.1 of `TAD_SETUP.md` before it — or defer §7
into the Phase 1 session, where that environment is set up anyway.

### 2.1 Full session setup

§2.1 and §2.2 of `TAD_SETUP.md` verbatim — same mount, same install cell, same
restart discipline, same four env vars. Not repeated here. Then:

```python
import os
os.environ['DADA_ORIG'] = f"{os.environ['KATVAD_DATA_ROOT']}/DADA2000Origin/DADA2000"
os.environ['PROBE']     = '/content/probe'          # VM-local NVMe, never Drive
os.makedirs(os.environ['PROBE'], exist_ok=True)
```

```bash
%%bash
apt-get -qq install -y p7zip-full
pip install -q openpyxl        # Colab usually has it; the project venv does not
ls -la "$DADA_ORIG"
df -h /content | tail -1       # how much VM disk you actually have
du -ch "$DADA_ORIG"/DADA2000.z* | tail -1   # total archive size
```

> **Record the two numbers.** If the archive is larger than free `/content`
> space, Phase 2 must extract in shards rather than in one pass. That decision
> belongs in the plan, not in a cell you improvise later.

---

## 3. **P2** — the layout, without extracting anything

`7z l` reads only the archive's directory, so this costs seconds and zero disk.

```bash
%%bash
cd "$DADA_ORIG"
7z l DADA2000.zip | head -60
echo '--- entry count ---'
7z l DADA2000.zip | tail -3
```

```bash
%%bash
cd "$DADA_ORIG"
# distinct top-level and second-level path prefixes
7z l -slt DADA2000.zip | grep '^Path = ' | sed 's/^Path = //' \
  | awk -F/ '{print $1"/"$2}' | sort -u | head -40
```

### 3.1 MEASURED 2026-09-15 — this is the answer, do not re-derive it

```
DADA2000/{type}/{video:03d}/{subdir}/{frame:04d}.png
```

`(type, video)` resolves **directly**, with no `Fault_Label` — 52 types on disk,
`[1..24, 30, 33, 34, 36..45, 47..61]`, **exactly the 52 the xlsx carries**, none
missing and none extra.

**Five subdirs per clip. Only one is video frames:**

| subdir | files | GiB | needed |
|---|---:|---:|---|
| **`images`** | 651,320 | **94.01** | ✅ the RGB frames |
| `maps` | 651,325 | 11.96 | ❌ driver-attention maps |
| `seg` | 651,320 | 10.33 | ❌ segmentation |
| `semantic` | 651,320 | 4.04 | ❌ |
| `fixation` | 1,302,640 | 3.16 | ❌ gaze fixation points (2 per frame) |

**Always extract `…/images/*` and nothing else** — the other four are DADA's own
*driver-attention prediction* task, which this project does not use. Skipping
them saves **29.5 GiB (24%)**.

> `fixation` sorts first alphabetically, so a truncated `7z l` shows only it.
> Reading that as "the archive holds fixation maps" is the easy mistake here.

**Aggregate frame-count check (a pre-signal for P1, not a substitute):**

```
images on disk        651,320 frames
annotation total      649,399 frames   (1,962 rows)
delta                  +1,921   = +0.30%,  about +1 frame per clip
```

Compare the reconstructed archive this project has been training on: 322 vs 68
raw frames, **−79%**. A 0.30% aggregate delta has the shape of an index
convention, not a trim. **§6 still runs** — an aggregate can hide per-clip errors
that cancel.

### 3.2 Archive size and the sharding decision

```
Total Physical Size   125,355,616,726 B = 116.7 GiB compressed, 6 volumes
images uncompressed    94.01 GiB  (~49 MiB per clip)
/content free           87 GB   (of 108 GB, measured)
```

**Neither the archive nor `images` fits on `/content`.** Phase 2 extracts in
shards of ~200 clips (~9.6 GiB each), and per shard: extract → CLIP features at
stride 8 → **delete the frames** → next shard. Persisted output is
`651,320 / 8 ≈ 81,415` sampled frames × 512 float32 ≈ **167 MB**.

> Freeing space **on Drive** does not help: the constraint is the VM's overlay
> filesystem at `/content`, not the Drive quota.

Measured in the Phase 0 session (`colab/DADA2000Origin/phase_0.ipynb`, cell 5):
`df -h /content` → **87 G free of 108 G**; `du -ch "$DADA_ORIG"/DADA2000.z*` →
**117 G**. Both numbers confirm the shard plan above; neither was improvised.

> **`Fault_Label` does not exist in the original release.** It was added by
> whoever repackaged the trimmed archive into
> `0_Non_Ego_Fault / 1_Ego_Fault / 0_Normal_Driving`. The original key is
> `(type, video)` and the xlsx confirms it is unique across all 1,962 rows.
> Any command copied from `DADA_SETUP.md` that assumes a fault directory
> **will not work here.**

### 3.3 MEASURED 2026-09-15 — native resolution **1584 x 660**, and why it matters

P3 (§7) read the real pixels: every probed frame is **H = 660, W = 1584**, an
aspect ratio of **2.40 : 1**. This was recorded nowhere before Phase 0. It is not
a blocker, but it belongs in every later reading of a DoTA-transfer number.

`preprocess_frames(..., center_crop=False)` — the transform this whole project
uses — resizes **anisotropically** to a 224 square
(`core/tools/extract_clip_features.py:57-72`, the `else` branch). It does not
preserve aspect ratio, deliberately: that is LaGoVAD's `no_center_crop`.
So each corpus is squashed horizontally by its own aspect ratio:

| corpus | native `H x W` | aspect | horizontal squash into 224 sq. |
|---|---|---:|---:|
| DoTA | 720 x 1280 | 1.78 : 1 | 1.78x |
| **DADA-2000 original** | **660 x 1584** | **2.40 : 1** | **2.40x** |

**DADA-original frames are distorted 1.35x more than DoTA's.** Consequences, in
order of how likely they are to bite:

1. **It is a domain-shift term on the one column this project is built around**
   — DADA-trained → DoTA zero-shot transfer. If that number disappoints, this is
   a named suspect, not a post-hoc excuse. Name it *before* the measurement.
2. **It does not violate C2/C13.** One transform, one cache dir: this corpus gets
   `clip/DADA2000_orig_w16s8` of its own, and no existing cache or checkpoint is
   touched. The rule is broken only if someone scores a DADA-original checkpoint
   on a cache built any other way.
3. **Do not "fix" it by switching this corpus to `center_crop=True`.** That would
   make it the only corpus in the tree on a different transform, break the
   appearance/flow field-of-view match (C13), and void the comparison with every
   `*_ncc` artifact measured since 2026-08-12.

> **Caveat on the evidence: 5 clips, 8 frames each, all 1584 x 660.** That is a
> sample, not a survey. The original release is not guaranteed uniform across its
> 52 types. §7.1's follow-up sweeps all 30 probe clips from PNG headers alone.

---

## 4. Pick the 30 probe clips

Stratified across `type` so no accident category is unrepresented. Run this
**after** §3 tells you the layout; set `LAYOUT` accordingly.

```python
%%writefile /content/pick_probe.py
"""Pick a type-stratified probe sample from the DADA-2000 annotation.

The sheet is **detected**, not hardcoded: this workbook is a hand-maintained
export and its sheet names and header spelling are not a contract. On failure the
script prints every sheet's header so the mismatch is visible in one run.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
from openpyxl import load_workbook

REQUIRED = ("type", "video", "abnormal start frame",
            "abnormal end frame", "total frames")
COL_ACCIDENT = "whether an accident occurred (1/0)"


def norm(value: object) -> str:
    """Case/whitespace-insensitive header key (the file uses NBSP in places)."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip().lower()


def find_sheet(xlsx: Path) -> tuple[list[tuple], list[str]]:
    """``(rows, header)`` for the first sheet carrying every REQUIRED column."""
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    seen: list[tuple[str, list[str]]] = []
    for name in wb.sheetnames:
        it = wb[name].iter_rows(values_only=True)
        try:
            header = [norm(c) for c in next(it)]
        except StopIteration:
            seen.append((name, []))
            continue
        seen.append((name, header))
        if all(norm(col) in header for col in REQUIRED):
            print(f"sheet: {name!r}  ({len(header)} columns)")
            return list(it), header
    lines = [f"  [{n}] {h}" for n, h in seen]
    sys.exit(
        f"No sheet in {xlsx} carries all of {REQUIRED}.\nSheets found:\n"
        + "\n".join(lines)
    )


def read_rows(xlsx: Path) -> list[dict]:
    body, header = find_sheet(xlsx)
    idx = {c: header.index(norm(c)) for c in REQUIRED}
    acc = header.index(norm(COL_ACCIDENT)) if norm(COL_ACCIDENT) in header else None
    if acc is None:
        print(f"WARNING: no {COL_ACCIDENT!r} column; treating every row with a "
              "valid window as an accident row")
    out: list[dict] = []
    for raw in body:
        def cell(i: int) -> object:
            return raw[i] if i < len(raw) else None
        if acc is not None and str(cell(acc)).strip() != "1":
            continue
        try:
            typ, vid = int(cell(idx["type"])), int(cell(idx["video"]))
            start = int(cell(idx["abnormal start frame"]))
            end = int(cell(idx["abnormal end frame"]))
            total = int(cell(idx["total frames"]))
        except (TypeError, ValueError):
            continue
        if not 0 <= start < end <= total:
            continue
        out.append({"type": typ, "video": vid, "start": start,
                    "end": end, "total": total})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = read_rows(args.xlsx)
    if not rows:
        sys.exit("No valid accident rows parsed -- check the sheet name")
    by_type: dict[int, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for typ in sorted(by_type):                       # one per type first
        picked.append(rng.choice(by_type[typ]))

    # THEN fill at random to --n. Without this the sample is capped at the number
    # of types (52), silently: `--n 400` returned 52 on 2026-09-15 and the Phase 1
    # runbook's claim that it "fills" was simply untrue. The guard below is
    # entered only when more than one-per-type was asked for, so every sample
    # with --n <= 52 -- Phase 0's 30 among them -- is bit-for-bit unchanged.
    if args.n > len(picked):
        taken = {(r["type"], r["video"]) for r in picked}
        rest = [r for r in rows if (r["type"], r["video"]) not in taken]
        rng.shuffle(rest)
        picked.extend(rest[: args.n - len(picked)])

    rng.shuffle(picked)
    picked = picked[: args.n]
    args.out.write_text(json.dumps(picked, indent=2), encoding="utf-8")
    print(f"{len(rows)} annotated clips over {len(by_type)} types "
          f"-> picked {len(picked)}")
    if len(picked) < args.n:
        print(f"WARNING: asked for {args.n} but only {len(picked)} clips exist; "
              "every later population number must say so")
    covered = len({r["type"] for r in picked})
    print(f"types covered: {covered}/{len(by_type)}")
    for r in picked[:5]:
        print("  ", r)


if __name__ == "__main__":
    main()
```

```bash
%%bash
python /content/pick_probe.py \
  --xlsx "$DADA_ORIG/dada标注.xlsx" --n 30 --seed 2024 \
  --out /content/probe_clips.json
```

Expect `1945 annotated clips over 52 types -> picked 30`, then
`types covered: 30/52`.

> **`--n` above 52 needs the fill pass.** Until 2026-09-15 this script stopped at
> one clip per type, so `--n 400` silently returned **52**. Check the `picked N`
> line against the `--n` you asked for — it is printed for exactly this reason.

---

## 5. Selective extraction — ~30 clips only

Pattern confirmed by §3.1. **`/images/` is not optional** — without it you pull
five subdirs and 5.3× the bytes.

```bash
%%bash
set -e
cd "$DADA_ORIG"
python - <<'PY' > /content/probe_patterns.txt
import json
PAT = "DADA2000/{t}/{v:03d}/images/*"     # measured layout, see section 3.1
for r in json.load(open('/content/probe_clips.json')):
    print(PAT.format(t=r['type'], v=r['video']))
PY
wc -l /content/probe_patterns.txt          # expect 30
head -3 /content/probe_patterns.txt

7z x DADA2000.zip -o"$PROBE" -y $(tr '\n' ' ' < /content/probe_patterns.txt)
du -sh "$PROBE"                            # expect ~1.5 GB (30 x ~49 MiB)
find "$PROBE" -mindepth 3 -maxdepth 3 -type d | wc -l   # expect 30
```

> `7z` reads the central directory from the last volume (`.zip`) and then seeks
> into `.z01`–`.z05` for the payload, all over Drive's FUSE mount. 30 clips is a
> couple of minutes; do **not** extrapolate that rate to Phase 2 — §3.2's shard
> plan exists because the full pass is ~94 GiB through the same mount.

> **Extract to `/content`, never to Drive.** `DADA_SETUP.md` §3 settled this:
> frames are an intermediate, only `clip/*.npy` is worth persisting, and Drive's
> FUSE mount charges per file open — you would pay it twice.

> **Do not glob the whole archive into a shell argument list** (lesson **C20**).
> 30 patterns is fine; 1,945 is not. Phase 2 extracts by directory, not by glob.

---

## 6. **P1** — the hard gate

```python
%%writefile /content/check_p1.py
"""P1: does the annotation's `total frames` match the real frame count?

The verdict below is section 6.1's decision table, in code. It is written that way
because the first version printed the "clips were TRIMMED" warning unconditionally
whenever *any* mismatch existed: on the 2026-09-15 probe that fired over ONE clip
off by 3 frames (-0.87%) while the table said PASS. A diagnostic whose prose
contradicts its own pre-registered table will be obeyed instead of the table.
"""
from __future__ import annotations
import argparse, json, statistics, subprocess
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

# --- section 6.1's thresholds, as constants so they cannot drift in prose ---
PASS_RATE = 0.95          # >= this fraction matched within tolerance -> PASS
NOTE_RATE = 0.80          # [NOTE_RATE, PASS_RATE) -> PASS with a note
MAX_MISSING = 2           # more than this -> P2 failed, not P1
# A trim verdict needs ALL THREE. One clip off by 3 frames is not a trim.
TRIM_MIN_N = 3            # too few mismatches to call anything systematic
TRIM_MIN_AGREE = 0.90     # share of mismatches sharing the sign
TRIM_MIN_REL = 0.02       # |mean delta| / median annotated length


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def count_video_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--template", required=True,
                    help="path under --root, e.g. 'DADA2000/{t}/{v:03d}/images'")
    ap.add_argument("--tolerance", type=int, default=2)
    args = ap.parse_args()

    rows = json.loads(args.clips.read_text(encoding="utf-8"))
    checked = matched = missing = 0
    deltas: list[tuple[str, int, int]] = []
    per_clip: list[tuple[str, int, int]] = []   # every checked clip, match or not
    for r in rows:
        rel = args.template.format(t=r["type"], v=r["video"])
        target = args.root / rel
        if target.is_dir():
            real = count_images(target)
        elif target.is_file() and target.suffix.lower() in VIDEO_SUFFIXES:
            real = count_video_frames(target)
        else:
            cand = [p for p in (args.root / rel).parent.glob(f"{Path(rel).name}*")
                    if p.suffix.lower() in VIDEO_SUFFIXES] if (args.root / rel).parent.is_dir() else []
            if len(cand) == 1:
                real = count_video_frames(cand[0])
            else:
                missing += 1
                continue
        checked += 1
        per_clip.append((rel, r["total"], real))
        delta = real - r["total"]
        if abs(delta) <= args.tolerance:
            matched += 1
        else:
            deltas.append((rel, r["total"], real))

    print(f"checked   {checked}   missing on disk {missing}")
    if checked:
        print(f"matched within +-{args.tolerance}: {matched}/{checked} "
              f"({100 * matched / checked:.1f}%)")
    exact = sum(1 for _, a, r in per_clip if r == a)
    print(f"matched EXACTLY (delta 0): {exact}/{checked}")

    trimmed = False
    if deltas:
        print(f"\n{'clip':34s} {'annotation':>10s} {'on disk':>8s} {'delta':>7s}")
        for rel, ann, real in sorted(deltas, key=lambda d: abs(d[2] - d[1]))[:20]:
            print(f"{rel:34s} {ann:10d} {real:8d} {real - ann:+7d}")
        signed = [r - a for _, a, r in deltas]
        mean = sum(signed) / len(signed)
        agree = max(sum(1 for d in signed if d > 0),
                    sum(1 for d in signed if d < 0)) / len(signed)
        scale = statistics.median(a for _, a, _ in per_clip) or 1
        rel_mean = abs(mean) / scale
        print(f"\nmismatches {len(deltas)}   mean signed delta {mean:+.1f} "
              f"({rel_mean:.2%} of the median clip)   sign agreement {agree:.0%}")
        trimmed = (len(deltas) >= TRIM_MIN_N
                   and agree >= TRIM_MIN_AGREE
                   and rel_mean >= TRIM_MIN_REL)
        if not trimmed:
            print(f"NOT a trim signature: needs >= {TRIM_MIN_N} mismatches, "
                  f">= {TRIM_MIN_AGREE:.0%} sign agreement and "
                  f">= {TRIM_MIN_REL:.0%} relative mean. Treat as re-encode / "
                  "index-convention noise.")

    # --- the verdict, section 6.1 verbatim ---
    rate = matched / checked if checked else 0.0
    print()
    if missing > MAX_MISSING:
        print(f"VERDICT: BACK TO P2 -- {missing} clips missing on disk "
              f"(> {MAX_MISSING}). The layout template is wrong; this is not a "
              "P1 result at all.")
    elif trimmed:
        print("VERDICT: STOP -- the clips are TRIMMED, like the reconstructed "
              "archive. Frame labels cannot be derived by fraction "
              "(core/data/dada.py:296). The original-corpus plan is void as "
              "written; report and re-plan.")
    elif rate >= PASS_RATE:
        print(f"VERDICT: PASS ({rate:.1%} >= {PASS_RATE:.0%}) -- the annotation "
              "describes the real files. Proceed to P3 (section 7).")
    elif rate >= NOTE_RATE:
        print(f"VERDICT: PASS WITH A NOTE ({rate:.1%} in "
              f"[{NOTE_RATE:.0%}, {PASS_RATE:.0%})) -- record the delta "
              "distribution and re-check on the full corpus in Phase 2.")
    else:
        print(f"VERDICT: STOP ({rate:.1%} < {NOTE_RATE:.0%} matched) -- too few "
              "clips match to build labels on. Report and re-plan.")
    print(f"NOTE: n = {checked}. A probe cannot prove a corpus-wide rate; "
          "P1 is a hard stop, so re-run it over every clip in Phase 2.")


if __name__ == "__main__":
    main()
```

```bash
%%bash
python /content/check_p1.py \
  --clips /content/probe_clips.json \
  --root "$PROBE" \
  --template 'DADA2000/{t}/{v:03d}/images' \
  --tolerance 2
```

### 6.1 Decision table — pre-registered, do not renegotiate after seeing the output

| result | reading | action |
|---|---|---|
| **≥ 95% matched within ±2** | annotation describes the real files | **PASS** → §7 |
| 80–95% matched, deltas both signs, small | re-encode / off-by-one on decode | **PASS with a note**; record the distribution and re-check on the full corpus in Phase 2 |
| **< 80%, or mean signed delta strongly one-sided** | clips are **trimmed** — same defect as the reconstructed archive | **STOP.** Frame labels cannot be derived by fraction. Report and re-plan |
| `missing on disk` > 2 | §3's layout guess is wrong | back to **P2**, not a P1 failure |

> The third row is the one this probe exists for. A mean signed delta of, say,
> −250 frames over 30 clips is not noise: it means the release you have is
> *also* a trimmed repackaging, and `.project/plans/katvad-dada-original-corpus.md`
> is void as written.

---

## 7. **P3** — decode and preprocess

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
python - <<'PY'
import json, itertools
from pathlib import Path
from core.tools.extract_clip_features import preprocess_frames
from core.data.video_io import list_frame_images, read_images

root = Path('/content/probe')
rows = json.loads(Path('/content/probe_clips.json').read_text())
ok = bad = 0
for r in itertools.islice(rows, 5):
    folder = root / f"DADA2000/{r['type']}/{r['video']:03d}/images"
    if not folder.is_dir():
        print('skip (layout)', folder); continue
    paths = list_frame_images(folder)[:8]
    try:
        frames = read_images(paths)                   # (N, H, W, 3) uint8
        batch = preprocess_frames(frames, center_crop=False)   # <-- _ncc, see below
        print(f"{folder.name}: {len(paths)} frames, native {frames.shape[1:3]} "
              f"-> {tuple(batch.shape)} {batch.dtype} "
              f"range[{batch.min():.3f},{batch.max():.3f}]")
        ok += 1
    except Exception as exc:                      # probe only; report, don't mask
        print('FAIL', folder, type(exc).__name__, exc); bad += 1
print(f"\nP3: {ok} ok, {bad} failed")
PY
```

> **`center_crop=False` is not optional.** `preprocess_frames`'s signature is
> `(frames: np.ndarray, crop_size=224, center_crop=True)` — it takes a decoded
> **array**, not a list of paths, and its default centre-crops. Every artifact in
> this project since 2026-08-12 is `no_center_crop` (`*_ncc`), so probing with the
> default would measure a transform the pipeline does not use — and a transform
> mismatch does not raise, it just returns a worse number that reads as a result
> (lessons **C2**, **C13**).

**Pass:** every probed clip yields a `(N, 3, 224, 224)` float tensor with no
exception. Note the **native resolution** of the source images too — the
original release is not guaranteed to match the trimmed archive's, and a
resolution change is a transform change (lessons **C2**, **C13**): it invalidates
no existing cache here only because this corpus gets its **own** cache directory,
`clip/DADA2000_orig_w16s8`.

### 7.1 Follow-up — sweep resolution and clip count before Phase 2

§7 as run sampled **5 clips x 8 frames**, all 1584 x 660 (§3.3). Two cheap checks
close the gap; run them at the top of the Phase 1 session, before any extraction.
Neither decodes a pixel nor re-reads the archive payload.

```bash
%%bash
# (a) native resolution across ALL 30 probe clips, from PNG headers only
python - <<'PROBE_RES'
import json
from collections import Counter
from pathlib import Path
from PIL import Image                       # header read, no decode

root = Path('/content/probe')
sizes = Counter()
for r in json.loads(Path('/content/probe_clips.json').read_text()):
    folder = root / f"DADA2000/{r['type']}/{r['video']:03d}/images"
    paths = sorted(folder.glob('*.png'))
    if not paths:
        print('MISSING', folder)
        continue
    for pth in (paths[0], paths[len(paths) // 2], paths[-1]):
        with Image.open(pth) as im:         # (W, H)
            sizes[im.size] += 1
    with Image.open(paths[0]) as im:
        first = im.size
    print(f"{r['type']:>3}/{r['video']:03d}  {len(paths):>4} frames  {first}")
print()
print('distinct (W, H):', dict(sizes))
PROBE_RES
```

**PASS:** exactly one distinct `(W, H)`. More than one and `preprocess_frames`
still succeeds — it resizes anything — but §3.3's squash factor becomes per-type,
and that table needs a distribution rather than a single row.

```bash
%%bash
# (b) how many CLIP DIRECTORIES are on disk, vs the xlsx's 1,962?
# /content/listing.txt is the section 3 archive listing -- already written, costs nothing.
grep '^Path = DADA2000/' /content/listing.txt | awk -F/ 'NF==3 {print $2"/"$3}' | sort -u | wc -l
```

**Why (b) matters:** §3.1's aggregate check (651,320 frames on disk vs 649,399
annotated, +0.30 %) silently assumes all 1,962 clips are present. If the disk
holds fewer, that aggregate is hiding a shortfall and an overcount that cancel.
P1 covered 30 clips; **nobody has counted the directories.** Expect **1,962**. A
smaller number is a Phase 2 sizing input, not a P1 failure — record it, do not
stop on it.

> If the Phase 0 VM is already gone, `/content/listing.txt` is gone with it (it
> was never copied to Drive). Regenerating it costs the 1 m 35 s `7z l -slt` of
> §3 — cheap, but do it in the same session as (a).

---

## 8. Reporting Phase 0

Write the three answers into the plan's §3 as a table, with the probe's seed and
sample size, then stop and decide. **Do not start Phase 1 in the same session
without recording P1's number** — lesson **C17**: a run that does not record what
defines it cannot be cited later.

```bash
%%bash
mkdir -p "$KATVAD_OUTPUT_ROOT/EDA/DADA2000Origin"
cp /content/probe_clips.json "$KATVAD_OUTPUT_ROOT/EDA/DADA2000Origin/"
# paste the P1 / P2 / P3 outputs into phase0_report.md beside it
```

| | result | verdict |
|---|---|---|
| **P2** layout | `DADA2000/{type}/{video:03d}/images/{frame:04d}.png`; 52/52 types match the xlsx | **PASS** (2026-09-15) |
| **P2** aggregate frames | 651,320 on disk vs 649,399 annotated = **+0.30%** | pre-signal only |
| archive size | 116.7 GiB compressed / 94.01 GiB `images`; `/content` free **87 GB** | **shard Phase 2: yes**, ~200 clips per shard |
| **P1** per-clip frame match | **29/30 = 96.7 %** within ±2 — and 29 match **exactly**, delta 0; the one miss is `36/002`, **−3** on 345 (−0.87 %). `missing on disk 0`. | **PASS** (2026-09-15), §8.1 |
| **P3** decode | **5/5 ok, 0 failed**; native **660 × 1584** (2.40 : 1) → `(8, 3, 224, 224)` float32, range ≈ [−1.79, 2.15] | **PASS**, see §3.3 and §7.1 |

### 8.1 P1 — the record, and the false alarm in its own output

Probe: `--n 30 --seed 2024`, type-stratified,
`outputs/EDA/DADA2000Origin/probe_clips.json`; full write-up beside it in
`phase0_report.md`. Four things a later reader needs.

**1. An independent cross-check, stronger than the 96.7 %.**
`sum(total frames)` over the 30 picked rows = **10,326**. `7z x` reported
`Files: 10323`. The difference is **exactly −3** — the single `36/002` miss. So
29 of 30 clips match the annotation *to the frame* and the ±2 tolerance was never
used. It also proves the extraction pulled `images/` and nothing else.

**2. The output's closing line is a FALSE ALARM. Do not act on it.**
`check_p1.py` ends with *"A systematic one-sided delta means the clips were
TRIMMED … the fraction mapping in `core/data/dada.py:296` is then WRONG"*. In
this run that sentence printed over **one** mismatch of **−0.87 %**. The trimmed
archive's defect is **−79 %**. §6.1 row 1 (≥ 95 % matched) fires; row 3 does not.
**P1 PASSES.** The script emitted the prose unconditionally whenever any mismatch
existed — a defect in the tool, fixed in §6 by the `n_mismatch` / relative-delta
guard. Logged as a lesson candidate in
`.project/memory-bank/lessons-learned/pending.md`.

**3. What 30 clips do NOT establish.** 29/30 has a Wilson 95 % lower bound of
**≈ 0.83**, so "≥ 95 % across all 1,962 clips" is *not* proven — and P1 is a hard
stop, so it must be re-run corpus-wide in Phase 2. The probe also covers only
**30 of 52 types**: `pick_probe.py` takes one clip per type, shuffles, then
truncates to `--n`, so 22 types were never touched. Adequate for layout and frame
counts; **not** a resolution survey (§7.1).

**4. The xlsx on Drive was never `md5sum`-ed** — §1.2's precaution was skipped,
because cell 5's `ls -la "$DADA_ORIG"` printed the symlink rather than the
directory behind it. A valid substitute landed anyway: `pick_probe.py` run
against the **Drive** copy printed `1945 annotated clips over 52 types`,
identical to the count measured on the committed `data/DADA/dada标注.xlsx`. That
establishes the two agree on the **population**; it does not establish they are
byte-identical.

**Next on a full PASS:** Phase 1 / Gate D0 —
`.project/plans/katvad-dada-original-corpus.md` §4. Extract CLIP features for
~400 clips and run `python -m core.tools.eda report --sections features`. The bar is
frame linear probe `auc_macro` **≥ 0.60**; below **0.55** the plan stops and the
deliverable becomes the negative result.

---

## 9. Traps, collected

1. `unzip DADA2000.zip` — **fails**, spanned archive (§1.1). Use `7z`.
2. `zip -s 0 … --out` — needs **2×** the disk. Do not.
3. Extracting to Drive instead of `/content` — pays FUSE per-file cost twice
   (`DADA_SETUP.md` §3).
4. Copying any command from `DADA_SETUP.md` that names
   `0_Non_Ego_Fault` / `1_Ego_Fault` / `0_Normal_Driving` — **those directories do
   not exist in the original release** (§3).
5. A dataset-sized glob in a shell cell — overflows `ARG_MAX` and dies *after* a
   successful extraction (lesson **C20**).
6. Reading `--window-length 32` defaults from `DADA_SETUP.md` §10.3 — that sizing
   was for the trimmed corpus and fails C32 here. This plan uses **16** at hop 8
   (plan §5.2).
7. Treating a P1 pass as permission to skip Gate D0 (§0.2).
