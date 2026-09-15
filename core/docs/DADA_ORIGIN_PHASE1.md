# DADA-2000 **original** release — Phase 1 / **Gate D0** on Colab (branch `main`)

**Written 2026-09-15.** Phase 1 of
`.project/plans/katvad-dada-original-phase1-gate-d0.md` (read §1 and §2 before
running anything — they say why this looks different from the parent plan's §4).
Prerequisite: **Phase 0 passed** — `DADA_ORIGIN_PHASE0.md` §8.
Companion docs: `EDA.md` (what the probe is), `COLAB.md` (session mechanics),
`DATA_LAYOUT.md` (the four dataset files).

> ## ⚠️ Branch check — run this first
>
> ```bash
> git branch --show-current      # must print: main
> ```

---

## 0. What Gate D0 is

**One number:** the supervised **frame linear probe** `auc_macro` over ~400
original DADA-2000 clips, on frozen CLIP features. **No training, no model, no
DoTA.**

| `auc_macro` | decision |
|---|---|
| **≥ 0.60** | **PASS** → Phase 2 (build the T2 corpus) |
| 0.55 – 0.60 | marginal → **one** arm, nothing more |
| **< 0.55** | **STOP** — the representation is the ceiling; the negative result is the deliverable |

Same features, same `_ncc` transform, for scale: **DoTA 0.6708**, **DADA trimmed
archive 0.5228**. D0 asks which one the original release looks like.

**A probe is a ceiling, not a promise.** It says a linear head *could* localize
here. It does not say WS-MIL training will — TAD is the counter-example
(`TAD_SETUP.md` §15.1). Do not let a PASS become a claim.

### 0.1 Nothing in `core/` changes

Phase 1 is option **B** of the plan: the builder is a throwaway script (§4 holds
it), `core/` is **read-only**, and the two stock tools run unmodified. The builder
*imports* the label convention from `core.data.dada` rather than restating it, so
D0 measures the ceiling of the pipeline as it will actually be built.

If you find yourself editing `core/`, **stop** and re-read plan §2 — the real
adapter is Phase 2's decision, taken after the gate, and `video_id_from_path` in
particular has 5 CRITICAL hop-1 callers across DoTA and TAD (plan §1.3).

---

## 1. Session setup

§2.1 and §2.2 of `TAD_SETUP.md` verbatim (mount, install, restart, the four env
vars). **A GPU runtime this time** — §5 runs CLIP over ~35 k frames. Then:

```python
import os
os.environ['DADA_ORIG'] = f"{os.environ['KATVAD_DATA_ROOT']}/DADA2000Origin/DADA2000"
os.environ['D0']        = '/content/d0'       # VM-local NVMe, never Drive
os.makedirs(os.environ['D0'], exist_ok=True)
```

```bash
%%bash
apt-get -qq install -y p7zip-full
pip install -q openpyxl          # pick_probe.py only; the project venv lacks it
df -h /content | tail -1         # need ~20 GiB free for section 3
```

---

## 2. Pick ~400 clips

> ### ⚠️ Re-write `pick_probe.py` first — the copy on `/content` is stale
>
> Until 2026-09-15 the script picked **one clip per type and stopped**: 52 rows,
> whatever `--n` said. A first Phase 1 attempt asked for 400 and got **52**, and
> an earlier draft of this section claimed the script "fills", which it did not.
> The fill pass is now in `DADA_ORIGIN_PHASE0.md` §4 — **re-run that
> `%%writefile` cell before this one.** Samples with `--n ≤ 52`, Phase 0's 30
> among them, are bit-for-bit unchanged (the fill branch is entered only when
> more than one-per-type is asked for); verified against the old code at
> n = 5 / 30 / 52.

```bash
%%bash
python /content/pick_probe.py \
  --xlsx "$DADA_ORIG/dada标注.xlsx" --n 400 --seed 2024 \
  --out /content/d0_clips.json
```

Expect:

```
1945 annotated clips over 52 types -> picked 400
types covered: 52/52
```

**If it prints `picked 52`, you are running the old script. Stop** — do not
extract, do not build. A 52-clip probe is a smoke test, not the gate (§2.1).

> **Phase 0's 30 are not necessarily a subset of these 400.** `pick_probe`
> reshuffles after the one-per-type pass, and the fill pass consumes the same
> RNG stream, so containment is not guaranteed at `--n 400`. It does not matter
> for D0; it would matter if you tried to reuse the Phase 0 extraction, which §3
> does not.

### 2.1 Why 52 clips cannot decide this gate

52 is above `EDA_PROBE_MIN_CLIPS` (10), so the probe **runs** and returns a
number. That number is not comparable to the references it would be read against:

| corpus | clips in the probe |
|---|---:|
| DoTA — the 0.6708 reference | **1,397** |
| DADA trimmed archive — the 0.5228 reference | **383** |
| a 52-clip Phase 1 | **52** |

`auc_macro` is a mean of per-clip AUCs, so its standard error scales as
`1/√N`: at 52 clips it is **√(383/52) = 2.71×** wider than the archive's and
**5.2×** wider than DoTA's. The gate's decision band is 0.55–0.60 — **0.05
wide**. A number whose own error bar spans the band decides nothing, whichever
side it lands on (lesson **C33**: know the attainable resolution before reading
the threshold).

**Use the 52 as a smoke test if it is already built** — it exercises §5 and §6
end to end for the price of ~2,265 frames, and catches a broken extractor flag
before 20 GiB is spent. **Record it as a smoke test, never as Gate D0.**

---

## 3. ⚠️ Disk: a GPU runtime is NOT the machine Phase 0 measured

Phase 0 measured **87 GiB free** on a **CPU** runtime. §1 asks for a **GPU**
runtime, which is a different VM with a different (much smaller) disk — and on
2026-09-15 a one-pass 400-clip extraction died on it with
`System ERROR: errno=28 : No space left on device`, after `7z` had already
written most of ~20 GiB.

**Measure before extracting, every session:**

```python
import os, shutil
total, used, free = shutil.disk_usage('/content')
print(f"/content: {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB")
print(f"one-pass 400 clips needs ~20 GiB; a 40-clip shard needs ~2 GiB")
```

```python
# clean anything a failed run left behind
import shutil, os
for sub in ('frames',):
    shutil.rmtree(os.path.join(os.environ['D0'], sub), ignore_errors=True)
```

| free at `/content` | do |
|---|---|
| **> 25 GiB** | one pass is fine — §4.1 |
| **2–25 GiB** | **shard** — §4.2, the default |
| < 2 GiB | nothing will work; restart the runtime |

**The persisted artifact is tiny either way.** 400 clips × ~43 sampled frames ×
512 float32 ≈ **35 MB** of features. The ~20 GiB of PNGs is scratch, and the
sharded loop treats it as such: extract → encode → **delete** → next.

---

## 4. Extract, build and encode

§4.0 is the script (the durable copy). Then pick **§4.1 one pass** or
**§4.2 sharded** by §3's table.

### 4.0 The builder

> ## ⚠️ `PYTHONPATH="$REPO"` — the builder imports `core`
>
> The script is written to `/content`, and **`python /content/foo.py` puts
> `/content` on `sys.path`, not the working directory** — so `cd "$REPO"` does
> nothing for the import, and `from core import constants` raises
> `ModuleNotFoundError: No module named 'core'`, exit 1. Measured 2026-09-16.
>
> On a dev machine with the project pip-installed editable this never shows up;
> on Colab, where it is not installed, it fails on the first shard. **Every call
> to this script needs `PYTHONPATH="$REPO"`** (the loop in §4.2 sets it for the
> whole child environment).

> **`colab/` is gitignored** (`.gitignore:46`), so a copy living only there does
> not survive. **This section is the durable copy of the script** — the same
> arrangement Phase 0 uses for `pick_probe.py` and `check_p1.py`. The working
> copy at `colab/DADA2000Origin/build_d0_dataset.py` is convenience, not record.

It has three modes, because of §3:

| mode | reads frames | builds farm | writes dataset dirs |
|---|---|---|---|
| `full` | yes, all at once | yes | yes |
| `shard` | yes, this shard | yes | no — dumps a `{video_id: frames_on_disk}` census |
| `labels` | **no** | no | yes, from the merged censuses |

`labels` mode exists so the dataset dirs can be built **after every frame has
been deleted**. Verified byte-identical to `full` on a 30-clip fixture split into
three shards, frames removed between each: all four JSON files match exactly.

```python
%%writefile /content/build_d0_dataset.py
"""Build the Gate-D0 dataset directories for the DADA-2000 **original** release.

Phase 1 of `.project/plans/katvad-dada-original-phase1-gate-d0.md`. This script
lives OUTSIDE ``core/`` on purpose: D0 is a hard-stop gate, so nothing is written
into the package until the gate is clear (plan §2, option B).

It writes the four files ``core.eda.corpus.load_dataset_files`` requires, plus a
symlink farm that lets the stock extractor see globally unique ids.

**It does not restate the label convention — it imports it.** ``DadaRecord`` and
``sampled_frame_labels`` come from ``core.data.dada``, so ``d0_frac`` carries
exactly the labels the Phase 4 pipeline would build:
``total_frames`` is the **on-disk** count and ``span`` is the fraction implied by
the annotation's own total (``core/data/dada.py:resolve_annotated_records``).
``d0_abs`` is the sanity arm: absolute indices, ``start // stride``.

Why the symlink farm (lesson **C26**): ``extract_clip_features.py:194`` keys its
folder scan on ``Path.name``, which on ``DADA2000/{type}/{video:03d}/images`` is
the video number alone — **1,962 clips collapse to 255 ids**, silently, and
``pending_items`` then reports a clean resume over a cache that is 87 % missing.

Three modes, because a GPU Colab runtime has far less disk than the CPU one
Phase 0 measured, and 400 clips of frames (~20 GiB) do not fit beside everything
else:

``full``
    resolve from disk, build the farm, write both dataset dirs. One pass; needs
    every clip's frames present at once.
``shard``
    resolve **this shard** from disk, build a shard farm, dump
    ``{video_id: frames_on_disk}`` to ``--counts-out``. Writes no dataset dir, so
    the caller may delete the frames straight after encoding features.
``labels``
    read the merged ``--counts-in`` files, touch **no** frames and build **no**
    farm, write both dataset dirs. Run once, after every shard is encoded.

Usage::

    # one pass (plenty of disk)
    python build_d0_dataset.py --mode full \\
        --clips /content/d0_clips.json --root /content/d0/frames \\
        --out-dir /content/d0/dataset --stride 8

    # per shard
    python build_d0_dataset.py --mode shard \\
        --clips /content/shard_03.json --root /content/d0/frames \\
        --out-dir /content/d0/shard_03 --counts-out /content/counts/03.json

    # once, at the end
    python build_d0_dataset.py --mode labels \\
        --clips /content/d0_clips.json --out-dir /content/d0/dataset \\
        --counts-in /content/counts/*.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core import constants
from core.data.dada import DadaRecord, sampled_frame_labels
from core.data.dataset_files import class_name_list, num_sampled_frames
from core.data.video_io import list_frame_folders, list_frame_images, video_id_from_path

LOGGER = logging.getLogger("build_d0_dataset")

IMAGES_SUBDIR = "images"
FRAC_DIRNAME = "d0_frac"
ABS_DIRNAME = "d0_abs"
FLAT_DIRNAME = "flat"
#: Pre-registered bar of plan §3.2, with its derivation (lesson **C33**: derive
#: the attainable range BEFORE writing the threshold).
#: ``sampled_frame_labels`` rounds each boundary (``round(frac * length)``) while
#: the absolute arm floors the start and ceils the end, so the two can differ by
#: at most **one sampled frame per boundary** = 2 per clip. More than that means
#: the frame counts, not the rounding, disagree.
MAX_LABEL_DISAGREEMENT_FRAMES = 2


def video_id(row: dict[str, int]) -> str:
    """``t{type:02d}_v{video:03d}`` — unique, and it keeps the (type, video) key.

    Deliberately unlike ``core.data.dada._make_video_id``'s
    ``{fault_label}__{folder}``: the original release has no ``Fault_Label``, and
    the two corpora must never share a feature cache (lesson **C2**).
    """
    return f"t{int(row['type']):02d}_v{int(row['video']):03d}"


def clip_folder(root: Path, row: dict[str, int]) -> Path:
    """The measured on-disk layout (``DADA_ORIGIN_PHASE0.md`` §3.1)."""
    return root / "DADA2000" / str(int(row["type"])) / f"{int(row['video']):03d}"


def resolve_rows(
    rows: list[dict[str, int]], root: Path, allow_missing: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach the on-disk frame count to every picked row.

    Mirrors ``core.data.dada.resolve_annotated_records``: a clip with no readable
    images is dropped, not silently zero-length (lesson **C10** — an existing
    directory is not evidence of data).
    """
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        folder = clip_folder(root, row)
        images = list_frame_images(folder / IMAGES_SUBDIR)
        if not images:
            missing.append(f"{video_id(row)} ({folder})")
            continue
        entry: dict[str, Any] = dict(row)
        entry["video_id"] = video_id(row)
        entry["folder"] = folder
        entry["frames_on_disk"] = len(images)
        resolved.append(entry)
    if missing and not allow_missing:
        raise ValueError(
            f"{len(missing)}/{len(rows)} picked clips have no readable images "
            f"under {root}: {sorted(missing)[:5]}. Finish the extraction, or "
            "pass --allow-missing-frames"
        )
    if missing:
        LOGGER.warning("%d/%d clips dropped (no images): %s",
                       len(missing), len(rows), sorted(missing)[:5])
    if not resolved:
        raise ValueError(f"No picked clip has readable images under {root}")
    return resolved, missing


def entries_from_counts(
    rows: list[dict[str, int]], counts: dict[str, int]
) -> list[dict[str, Any]]:
    """Rebuild the resolved entries from a shard-time frame census.

    ``labels`` mode runs after the frames are gone, so the on-disk count cannot
    be re-measured — it is read back from what the shards recorded. A clip absent
    from the census was never encoded and is dropped, loudly: a label row with no
    feature row would fail ``core/eda/features.py:346`` much later (lesson C10 —
    the census, not the directory, is the evidence).
    """
    out: list[dict[str, Any]] = []
    absent: list[str] = []
    for row in rows:
        vid = video_id(row)
        if vid not in counts:
            absent.append(vid)
            continue
        entry: dict[str, Any] = dict(row)
        entry["video_id"] = vid
        entry["folder"] = None
        entry["frames_on_disk"] = int(counts[vid])
        out.append(entry)
    if absent:
        LOGGER.warning("%d/%d clips missing from the counts census (never "
                       "encoded?): %s", len(absent), len(rows), sorted(absent)[:5])
    if not out:
        raise ValueError("No clip survived the counts census")
    return out


def make_record(entry: dict[str, Any]) -> DadaRecord:
    """A real ``DadaRecord``, so the project's own label function applies.

    ``accident_frac`` is ``None``: the original annotation's accident-frame column
    is not carried by ``pick_probe.py`` and nothing in the features section reads
    it. ``fault_label`` is a sentinel — the original release has no such column.
    """
    total = int(entry["total"])
    return DadaRecord(
        video_id=entry["video_id"],
        folder_name=entry["video_id"],
        class_name=constants.DADA_CLASS_NAME,
        fault_label="origin",
        total_frames=int(entry["frames_on_disk"]),
        span=(int(entry["start"]) / total, int(entry["end"]) / total),
        accident_frac=None,
    )


def absolute_frame_labels(entry: dict[str, Any], stride: int) -> list[int]:
    """Labels from absolute frame indices — the sanity arm of plan §3.2.

    Strictly more correct on an untrimmed release, and therefore the check that
    the fraction mapping (which P1 validated to ±3 frames on one clip of 30) has
    not moved the window.
    """
    length = num_sampled_frames(int(entry["frames_on_disk"]), stride)
    labels = [0] * length
    start = max(0, min(length, int(entry["start"]) // stride))
    end = max(0, min(length, -(-int(entry["end"]) // stride)))
    for index in range(start, end):
        labels[index] = 1
    return labels


def build_meta(
    entries: list[dict[str, Any]],
    frame_labels: dict[str, list[int]],
    records: dict[str, DadaRecord],
) -> dict[str, dict[str, object]]:
    """Per-clip diagnostics — every field that defines the run (lesson **C17**)."""
    meta: dict[str, dict[str, object]] = {}
    for entry in entries:
        vid = entry["video_id"]
        labels = frame_labels[vid]
        record = records[vid]
        meta[vid] = {
            "class_name": record.class_name,
            "split": "test",
            "type": int(entry["type"]),
            "video": int(entry["video"]),
            "total_frames": int(entry["frames_on_disk"]),
            "annotation_total_frames": int(entry["total"]),
            "frame_delta": int(entry["frames_on_disk"]) - int(entry["total"]),
            "sampled_frames": len(labels),
            "positive_frames": sum(labels),
            "annotation_span_frames": [int(entry["start"]), int(entry["end"])],
            "normalized_span": list(record.span) if record.span else None,
        }
    return meta


def write_dataset_dir(
    out_dir: Path,
    frame_labels: dict[str, list[int]],
    meta: dict[str, dict[str, object]],
) -> None:
    """The four files ``load_dataset_files`` demands; ``labels_train`` is empty.

    Phase 1 trains nothing, so an empty ``labels_train.json`` is the honest
    statement — not a copy of the test ids, which would read as a split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, object] = {
        constants.FRAME_LABELS_TEST_FILENAME: frame_labels,
        constants.LABELS_TRAIN_FILENAME: {},
        constants.DEFS_FILENAME: class_name_list({constants.DADA_CLASS_NAME}),
        constants.META_FILENAME: meta,
    }
    for filename, payload in payloads.items():
        target = out_dir / filename
        part = target.with_suffix(target.suffix + ".part")
        part.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        part.replace(target)          # atomic, lesson C11c
    LOGGER.info("wrote %s (%d test clips)", out_dir, len(frame_labels))


def materialize_flat(entries: list[dict[str, Any]], flat_dir: Path) -> int:
    """``flat/{video_id} -> DADA2000/{type}/{video:03d}/images`` (lesson **C26**).

    The same trick as ``core.data.dada.materialize_flat_dir``, with one
    difference that is **not** cosmetic: the link points at the ``images``
    directory itself, and the extractor is then run **without**
    ``--frames-subdir``.

    ``pathlib`` refuses to recurse into a symlinked directory (cycle guard, every
    version incl. 3.10 and 3.13), so ``list_frame_folders(flat, "images")`` walks
    ``flat.rglob("images")``, never descends through the link, and raises
    *"No frame folders found"*. A link at the images level is matched by the
    final component of ``rglob("*")`` instead, and ``list_frame_images`` follows
    it happily. Measured, not assumed.
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in entries:
        target = flat_dir / entry["video_id"]
        if target.is_symlink() or target.exists():
            continue
        source = (Path(entry["folder"]) / IMAGES_SUBDIR).resolve()
        target.symlink_to(source, target_is_directory=True)
        written += 1
    LOGGER.info("flat dir: %d new symlinks, %d entries total under %s",
                written, len(entries), flat_dir)
    return written


def check(
    entries: list[dict[str, Any]],
    frac: dict[str, list[int]],
    absolute: dict[str, list[int]],
    flat_dir: Path | None,
    stride: int,
) -> None:
    """Plan §4.3's five assertions. Loud, before 20 GiB of extraction is spent."""
    ids = [e["video_id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("video_id collision — the (type, video) key is not unique")

    for entry in entries:
        vid = entry["video_id"]
        expected = num_sampled_frames(int(entry["frames_on_disk"]), stride)
        if len(frac[vid]) != expected or len(absolute[vid]) != expected:
            raise ValueError(
                f"{vid}: label length {len(frac[vid])}/{len(absolute[vid])} != "
                f"{expected} sampled frames — cache and labels would not align"
            )

    deltas = [int(e["frames_on_disk"]) - int(e["total"]) for e in entries]
    exact = sum(1 for d in deltas if d == 0)
    within2 = sum(1 for d in deltas if abs(d) <= 2)
    LOGGER.info("P1 at n=%d: %d exact (%.1f%%), %d within +-2 (%.1f%%), "
                "mean signed delta %+.2f",
                len(deltas), exact, 100 * exact / len(deltas),
                within2, 100 * within2 / len(deltas),
                sum(deltas) / len(deltas))
    worst = sorted(zip(ids, deltas, strict=True), key=lambda p: -abs(p[1]))[:5]
    LOGGER.info("largest frame deltas: %s", worst)

    if flat_dir is not None:
        broken = [e["video_id"] for e in entries
                  if not list_frame_images(flat_dir / e["video_id"])]
        if broken:
            raise ValueError(
                f"{len(broken)} symlinks do not resolve to a folder of images: "
                f"{broken[:5]}"
            )
        scanned = list_frame_folders(flat_dir)
        scanned_ids = {video_id_from_path(f) for f in scanned}
        if len(scanned_ids) != len(entries):
            raise ValueError(
                f"the extractor would see {len(scanned_ids)} unique ids for "
                f"{len(entries)} clips — lesson C26 is not solved"
            )
        LOGGER.info("extractor scan of the flat dir: %d folders, %d unique ids",
                    len(scanned), len(scanned_ids))

    vanished = sorted(v for v in frac if not any(frac[v]))
    if vanished:
        LOGGER.warning("%d clips lose their window at stride %d: %s",
                       len(vanished), stride, vanished[:5])

    disagree = [
        (v, sum(a != b for a, b in zip(frac[v], absolute[v], strict=True)))
        for v in frac
    ]
    worst_frames = max(n for _, n in disagree)
    total_diff = sum(n for _, n in disagree)
    LOGGER.info("frac vs abs labels: %d clips differ, worst %d frames, %d frames total",
                sum(1 for _, n in disagree if n), worst_frames, total_diff)
    if worst_frames > MAX_LABEL_DISAGREEMENT_FRAMES:
        LOGGER.warning(
            "worst-case disagreement %d frames exceeds the pre-registered %d "
            "(plan §3.2) — compare the two eda reports before trusting either",
            worst_frames, MAX_LABEL_DISAGREEMENT_FRAMES,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--mode", choices=("full", "shard", "labels"), default="full")
    parser.add_argument("--clips", type=Path, required=True,
                        help="pick_probe.py output: [{type, video, start, end, total}]")
    parser.add_argument("--root", type=Path, default=None,
                        help="extraction root holding DADA2000/{type}/{video:03d}/images "
                             "(full and shard modes)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=constants.FRAME_STRIDE)
    parser.add_argument("--counts-out", type=Path, default=None,
                        help="shard mode: where to record {video_id: frames_on_disk}")
    parser.add_argument("--counts-in", type=Path, nargs="+", default=None,
                        help="labels mode: the shard censuses to merge")
    parser.add_argument("--allow-missing-frames", action="store_true",
                        help="drop picked clips with no images instead of raising")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mode in ("full", "shard") and args.root is None:
        parser.error(f"--root is required in {args.mode} mode")
    if args.mode == "shard" and args.counts_out is None:
        parser.error("--counts-out is required in shard mode")
    if args.mode == "labels" and not args.counts_in:
        parser.error("--counts-in is required in labels mode")

    rows = json.loads(args.clips.read_text(encoding="utf-8"))
    LOGGER.info("mode %s: %d clips, stride %d", args.mode, len(rows), args.stride)

    if args.mode == "labels":
        counts: dict[str, int] = {}
        for path in args.counts_in:
            counts.update(json.loads(path.read_text(encoding="utf-8")))
        LOGGER.info("merged %d shard censuses -> %d clips",
                    len(args.counts_in), len(counts))
        entries = entries_from_counts(rows, counts)
        flat_dir: Path | None = None
    else:
        entries, _ = resolve_rows(rows, args.root, args.allow_missing_frames)
        farm = args.out_dir / FLAT_DIRNAME
        materialize_flat(entries, farm)
        flat_dir = farm

    records = {e["video_id"]: make_record(e) for e in entries}
    frac = {vid: sampled_frame_labels(rec, args.stride) for vid, rec in records.items()}
    absolute = {e["video_id"]: absolute_frame_labels(e, args.stride) for e in entries}
    check(entries, frac, absolute, flat_dir, args.stride)

    if args.mode == "shard":
        args.counts_out.parent.mkdir(parents=True, exist_ok=True)
        census = {e["video_id"]: int(e["frames_on_disk"]) for e in entries}
        args.counts_out.write_text(json.dumps(census, indent=2), encoding="utf-8")
        LOGGER.info("census -> %s (%d clips). Encode features from %s, then "
                    "DELETE the frames.", args.counts_out, len(census), flat_dir)
        return

    meta = build_meta(entries, frac, records)
    write_dataset_dir(args.out_dir / FRAC_DIRNAME, frac, meta)
    write_dataset_dir(args.out_dir / ABS_DIRNAME, absolute,
                      build_meta(entries, absolute, records))

    positives = sum(sum(v) for v in frac.values())
    sampled = sum(len(v) for v in frac.values())
    LOGGER.info("D0 population: %d clips, %d sampled frames, %d positive (%.1f%%)",
                len(frac), sampled, positives, 100 * positives / sampled)
    if args.mode == "full":
        LOGGER.info("next: extract features from %s (NO --frames-subdir, see "
                    "materialize_flat), then core.tools.eda report --sections "
                    "features --probe", flat_dir)
    else:
        LOGGER.info("next: core.tools.eda report --sections features --probe")


if __name__ == "__main__":
    main()
```

### 4.1 One pass (> 25 GiB free)

```bash
%%bash
set -e
cd "$DADA_ORIG"
python - <<'PY' > /content/d0_patterns.txt
import json
PAT = "DADA2000/{t}/{v:03d}/images/*"
for r in json.load(open('/content/d0_clips.json')):
    print(PAT.format(t=r['type'], v=r['video']))
PY
wc -l /content/d0_patterns.txt                  # expect 400
7z x DADA2000.zip -o"$D0/frames" -y $(tr '\n' ' ' < /content/d0_patterns.txt)
du -sh "$D0/frames"
```

```bash
%%bash
cd "$REPO"
PYTHONPATH="$REPO" python /content/build_d0_dataset.py --mode full \
  --clips /content/d0_clips.json --root "$D0/frames" \
  --out-dir "$D0/dataset" --stride 8
```

Then §5's extractor command once, against `$D0/dataset/flat`.

### 4.2 Sharded — the default

One Python cell does extract → build → encode → delete, per shard. It is
**resumable**: a shard whose census exists *and* whose ids are all in the feature
cache is skipped, so a lost runtime costs one shard (lesson **C11**).

```python
import json, os, shutil, subprocess, sys
from pathlib import Path

REPO  = Path(os.environ['REPO'])
D0    = Path(os.environ['D0'])
ORIG  = Path(os.environ['DADA_ORIG'])
CACHE = Path(os.environ['KATVAD_CACHE_ROOT']) / 'clip' / 'DADA2000_orig'
SHARD = 40                     # clips per shard; ~2 GiB of PNGs each

clips  = json.loads(Path('/content/d0_clips.json').read_text())
counts = D0 / 'counts'
counts.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# `python /content/build_d0_dataset.py` puts **/content** on sys.path, not the
# cwd -- so `from core import ...` fails however you `cd`. The repo is not
# pip-installed on Colab, so PYTHONPATH is what makes it importable. Passing it
# to the extractor too is harmless and keeps the two calls identical.
ENV = {**os.environ, 'PYTHONPATH': str(REPO)}


def free_gib() -> float:
    return shutil.disk_usage('/content').free / 2**30


def run(cmd: list[str], cwd: Path, capture: bool = False) -> None:
    """Run a child, and make its failure legible.

    `capture=True` for the short steps: on failure the child's own stdout and
    stderr are printed here, instead of a bare `CalledProcessError` that says
    only `returned non-zero exit status 1`.
    """
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=ENV,
                          capture_output=capture, text=True)
    if proc.returncode:
        if capture:
            print(proc.stdout or '', proc.stderr or '', sep='\n')
        raise RuntimeError(
            f'exit {proc.returncode}: {" ".join(str(c) for c in cmd)}'
            + ('' if capture else '  -- scroll up for the child output')
        )
    if capture and proc.stdout:
        print(proc.stdout.rstrip())


def vid(r: dict) -> str:
    return f"t{r['type']:02d}_v{r['video']:03d}"


for k in range(0, len(clips), SHARD):
    shard, tag = clips[k:k + SHARD], f'{k // SHARD:03d}'
    census = counts / f'{tag}.json'
    if census.exists() and all((CACHE / f'{vid(r)}.npy').exists() for r in shard):
        print(f'shard {tag}: already encoded, skipping')
        continue

    frames, farm = D0 / 'frames', D0 / f'farm_{tag}'
    shutil.rmtree(frames, ignore_errors=True)
    shutil.rmtree(farm, ignore_errors=True)

    patterns = [f"DADA2000/{r['type']}/{r['video']:03d}/images/*" for r in shard]
    shard_json = D0 / f'clips_{tag}.json'
    shard_json.write_text(json.dumps(shard), encoding='utf-8')

    print(f'--- shard {tag}: {len(shard)} clips, {free_gib():.1f} GiB free')
    run(['7z', 'x', 'DADA2000.zip', f'-o{frames}', '-y', '-bso0', '-bsp0',
         *patterns], cwd=ORIG)
    run([sys.executable, '/content/build_d0_dataset.py', '--mode', 'shard',
         '--clips', shard_json, '--root', frames, '--out-dir', farm,
         '--counts-out', census, '--stride', '8'], cwd=REPO, capture=True)
    run([sys.executable, '-m', 'core.tools.extract_clip_features',
         '--frames-dir', farm / 'flat', '--dataset', 'DADA2000_orig',
         '--stride', '8', '--batch-size', '32', '--device', 'auto',
         '--no-center-crop', '--output-dir', CACHE], cwd=REPO)

    shutil.rmtree(frames, ignore_errors=True)
    shutil.rmtree(farm, ignore_errors=True)
    print(f'    cache: {len(list(CACHE.glob("*.npy")))} clips, '
          f'{free_gib():.1f} GiB free')

print('\nshards done:', len(list(counts.glob('*.json'))),
      '| cached clips:', len(list(CACHE.glob('*.npy'))))
```

Then build the dataset dirs from the censuses — **no frames needed**:

```bash
%%bash
cd "$REPO"
PYTHONPATH="$REPO" python /content/build_d0_dataset.py --mode labels \
  --clips /content/d0_clips.json \
  --out-dir "$D0/dataset" \
  --counts-in "$D0"/counts/*.json \
  --stride 8
```

### 4.3 Three lines to read before continuing

| line | meaning | act if |
|---|---|---|
| `P1 at n=400: … exact (…%), … within +-2` | **P1 re-run at 13× Phase 0's sample** — closes §8.1's Wilson gap for free | < 95 % within ±2 → **P1 failure at scale**; stop, re-read `DADA_ORIGIN_PHASE0.md` §6.1 |
| `extractor scan of the flat dir: N folders, N unique ids` (per shard) | lesson **C26** is neutralized | the two numbers differ → farm is wrong, do **not** encode |
| `frac vs abs labels: … worst N frames` | the two label constructions (plan §3.2) | `N > 2` → frame counts disagree, not rounding. Investigate first |

Plus, in `labels` mode: `merged N shard censuses -> 400 clips`. **A number below
400 means shards are missing** — the warning names them.

> **Why a symlink farm at all** (C26): `extract_clip_features.py:194` keys its
> scan on `Path.name`, which here is the bare video number — **1,962 clips
> collapse to 255 ids**, silently, and `pending_items` then reports a clean
> resume over a cache that is 87 % missing.
>
> **Why the link points at `…/images`:** `pathlib` will not recurse *into* a
> symlinked directory, so `--frames-subdir images` against a clip-level farm
> raises *"No frame folders found"*. Measured, plan §1.5. **Hence: no
> `--frames-subdir` anywhere in this runbook.**

---

## 5. The extractor flags, wherever you call it

§4.2's loop already passes these; §4.1 needs them typed once:

```bash
%%bash
cd "$REPO"
python -m core.tools.extract_clip_features \
  --frames-dir "$D0/dataset/flat" \
  --dataset    DADA2000_orig \
  --stride     8 \
  --batch-size 32 \
  --device     auto \
  --no-center-crop \
  --output-dir "$KATVAD_CACHE_ROOT/clip/DADA2000_orig"
```

Three things that are wrong by default and must be right here:

1. **`--no-center-crop`.** Every artifact in this project since 2026-08-12 is
   `no_center_crop` (lessons **C2**, **C13**). Omitting it builds a cache on a
   transform nothing else uses and nothing raises. **The extractor logs which
   transform it chose — read that line**, it should say *anisotropic resize*.
2. **No `--frames-subdir`.** The farm already points at the images directory
   (§4.3).
3. **Cache name `DADA2000_orig`.** Not `DADA2000_ncc` (that is the *trimmed*
   archive — C2), and not `DADA2000_orig_w16s8`: features are keyed by **source
   clip** and windows slice into them (`core/eda/corpus.py:43`,
   `core/data/windows.py:FeatureSlicer`), so this one cache serves Phase 1 and
   Phase 2's T2 windows alike.

```bash
%%bash
ls "$KATVAD_CACHE_ROOT/clip/DADA2000_orig" | wc -l     # expect 400
du -sh "$KATVAD_CACHE_ROOT/clip/DADA2000_orig"         # expect ~35 MB
```

---

## 6. **Gate D0**

```bash
%%bash
cd /content/drive/MyDrive/Thesis/kat-vad
for ARM in d0_frac d0_abs; do
  python -m core.tools.eda report \
    --dataset    DADA2000_orig \
    --data-dir   "$D0/dataset/$ARM" \
    --clip-dir   "$KATVAD_CACHE_ROOT/clip/DADA2000_orig" \
    --sections   features \
    --probe \
    --output-dir "outputs/EDA/DADA2000Origin/$ARM"
done
```

> The subcommand is **`report`**. `run` does not exist — it is what the parent
> plan said, and it dies in argparse (plan §1.1).

```bash
%%bash
python - <<'PY'
import json
from pathlib import Path
for arm in ('d0_frac', 'd0_abs'):
    p = Path(f'outputs/EDA/DADA2000Origin/{arm}/eda_report.json')
    probe = json.loads(p.read_text())['features']['frame_linear_probe']
    print(arm, {k: probe.get(k) for k in ('ran', 'reason', 'auc', 'auc_macro', 'folds', 'clips')})
PY
```

### 6.1 Reading the result — pre-registered, do not renegotiate

| observation | reading |
|---|---|
| `auc_macro ≥ 0.60` on **both** arms, and they agree within 0.01 | **PASS** → Phase 2 |
| 0.55 – 0.60 | marginal → **one** arm only, and say so in every later table |
| `< 0.55` | **STOP.** The representation is the ceiling. Ship the negative result and the backbone recommendation (`DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §7) |
| the two arms differ by **> 0.01** | **both numbers void** until explained — the frame counts, not the rounding, disagree (plan §3.2) |

**Expected and NOT a failure:** `clip_linear_probe` returns
`{"ran": false, "reason": "every clip has the same clip-level label"}`. The
original release has **zero normal videos**, so every clip is abnormal and there
is no clip-level target. That is the point of this corpus — there is no
clip-level shortcut to find, which is precisely what the trimmed archive and TAD
both had (C12's mirror, C28). **Do not "fix" it by importing
`0_Normal_Driving`**: that is the ~3× longer pool whose length *inverted* the
leak (0.8105 → 0.8539).

---

## 6.2 ⚠️ Persist BEFORE the runtime dies

`$D0` is `/content/d0` — **VM-local**. On 2026-09-15 the runtime was recycled
before `meta.json` was copied and the per-clip frame census was lost
(`gate_d0_report.md` §2.1). §7's cell now copies it in the same step that writes
the report; **run §7 in the same session as §6**, never "later".

What survives a recycle by itself: the feature cache (it is on Drive) and
anything written under `$REPO/outputs` (the repo is on Drive). Everything under
`$D0` does not.

**If it is already too late**, most of it is recoverable without re-extracting a
byte:

* **the clip sample** — the pick is deterministic:
  `pick_probe.py --n 400 --seed 2024` against the committed
  `data/DADA/dada标注.xlsx` reproduces it exactly (verified; committed as
  `outputs/EDA/DADA2000Origin/d0_clips_400.json`);
* **`sampled_frames` per clip** — the first axis of each cached `.npy`, which
  pins `frames_on_disk` to `[8(L−1)+1, 8L]` and makes a large mismatch
  unmistakable. The recovery cell is in `gate_d0_report.md` §4.1;
* **the label JSONs** — rebuildable from those two with `--mode labels`.

What is *not* recoverable is each clip's exact on-disk count.

---

## 7. Record the gate

`outputs/EDA/DADA2000Origin/gate_d0_report.md`, carrying at minimum:

* both `auc_macro` values and their delta against §6.1's 0.01 bar;
* the P1 distribution at n = 400 from §4 — **this supersedes Phase 0's n = 30**;
* clip count, sampled frames, positive fraction, folds actually used;
* the seed (2024), the sample size, the cache name, and the extractor's
  transform line (lesson **C17** — a run that does not record what defines it
  cannot be cited);
* the decision, in the words of §6.1.

Then update `.project/memory-bank/activeContext.md` and file a lesson candidate
if anything new was learned. **Then stop.** Phase 2 is a separate decision, taken
on the number.

---

## 8. Traps, collected

1. `eda run` — **does not exist**. It is `eda report` (§6).
2. Omitting `--no-center-crop` — silently wrong transform (C2, C13).
3. Passing `--frames-subdir images` in §5 — *"No frame folders found"*, because
   the farm already links at the images level (plan §1.5).
4. Pointing `--frames-dir` at `$D0/frames` instead of the farm — **C26**, a cache
   87 % missing with no error.
5. Naming the cache `DADA2000_ncc` — that is the trimmed archive's (C2).
6. Adding `_w16s8` to the cache name — features are clip-keyed, not window-keyed
   (§5).
7. Reading `clip_linear_probe: ran=false` as a bug (§6.1).
8. Treating a D0 pass as evidence that training will localize (§0).
9. Editing `core/` to make any of this work (§0.1).
16. Letting the runtime die before §7 runs — everything under `$D0` is
    VM-local and goes with it (§6.2). Happened 2026-09-15.
10. Running the **stale** `pick_probe.py` — `--n 400` silently yields **52**
    (§2). Check the `picked N` and `types covered` lines every time.
11. Reading a 52-clip `auc_macro` as the gate — its error bar is 2.7× the
    archive's and spans the whole decision band (§2.1).
12. Assuming the GPU runtime has Phase 0's 87 GiB. It does not — a one-pass
    400-clip extraction died with `errno=28` on 2026-09-15 (§3). Measure,
    then shard.
13. Re-extracting a shard that is already encoded — §4.2 skips it; deleting
    `$D0/counts` throws that away.
14. Calling `/content/build_d0_dataset.py` without `PYTHONPATH="$REPO"` —
    `cd` does **not** put the repo on `sys.path` (§4.0).
15. Wrapping a child process without surfacing its stderr — a bare
    `CalledProcessError: exit status 1` cost a whole round trip on
    2026-09-16. §4.2's `run(..., capture=True)` prints the child's output.
