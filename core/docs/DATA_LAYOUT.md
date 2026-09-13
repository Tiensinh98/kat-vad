# KAT-VAD On-Disk Data Layout Contract

Defined once in `core/constants.py`; download CLIs, extractors and dataloaders all
resolve paths through those constants. Roots are overridable via environment
variables so Colab can point everything at Google Drive:

| Env var | Default | Holds |
|---|---|---|
| `KATVAD_DATA_ROOT` | `./data` | raw videos + label files |
| `KATVAD_CACHE_ROOT` | `./cache` | CLIP features, flow embeddings, KNN cache |
| `KATVAD_CKPT_ROOT` | `./ckpts` | checkpoints (ours + LaGoVAD `best.ckpt`) |
| `KATVAD_OUTPUT_ROOT` | `./outputs` | eval results, visualizations, logs |

## Layout

```
data/{DATASET}/videos/...                    raw videos (as downloaded)
data/{DATASET}/annotations/...               shipped annotation files (as downloaded)
data/{DATASET}/labels_train.json             {video_id: 0|1}            video-level train labels
data/{DATASET}/frame_labels_test.json        {video_id: [0,1,...]}      per SAMPLED frame (stride-8 aligned)
data/{DATASET}/defs.json                     {video_id: [desc,...]} or global class-name list
data/{DATASET}/meta.json                     per-video scenario/class/split/window (diagnostics
                                             only — NEVER training supervision)
data/{DATASET}/windows.json                  OPTIONAL {window_id: {source, start, end}}
                                             present only for a corpus re-sharded into
                                             fixed-length windows (lesson C28)

cache/clip/{DATASET}/{video_id}.npy          (L, 512) float32 CLIP ViT-B/16 features
cache/flow/v1/{DATASET}/{video_id}.npy       (L, 256) float32 RAFT flow embeddings e_O (train only)
cache/flow/v1/{DATASET}/{video_id}.stats.npy (L, 23) float32 raw pooling stats (motion-aware KNN key
                                             needs magnitude/direction, unrecoverable from e_O)
cache/flow/v1/flow_projection.npz            seeded fixed linear map for flow pooling (A10)
cache/knn/{DATASET}/knn_cache.npz            DVS KNN filler cache (abnormal -> top-K normal ids)

ckpts/lagovad_best.ckpt                      LaGoVAD reference checkpoint (gate a)
ckpts/{run_name}/epoch_{N}.pt                our training checkpoints (full resume state)
```

`{DATASET}` ∈ `TAD`, `PreVAD`, `DoTA`, `DADA2000`, `MSAD`, `UCF-Crime` (constants in
`core/constants.py`).

## Rules

- The flow cache is **versioned** (`flow/v1/`): any change to the pooling method or
  projection bumps the version directory — never overwrite in place.
- The flow projection matrix ships **with** the cache it produced; loaders must fail
  loudly if `flow_projection.npz` is missing for the cache version they read.
- Frame-level label arrays are aligned to the **sampled** frames (stride 8, or native
  fps for DoTA/DADA), not to raw video frames.
- Extractors are resumable: one `.npy` per video, skip-if-exists.

## Fixed-length windows — the optional fifth file (Phase 2a, lesson C28)

A corpus whose **clip length predicts its label** cannot be measured, however good
the model is: on the reconstructed DADA-2000, abnormal test clips run to at most
17 sampled frames while all 107 clips of 18+ frames are normal, so a detector
reading only the frame count scores micro AUC **0.8654**
(`DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §2). The cure is to make every trained
and scored item the same length.

```
data/{DATASET}/windows.json   {"{source}__w000": {"source": ..., "start": 0, "end": 32}, ...}
```

**The feature cache is not re-sharded.** A window is a *slice* of its source
clip's `.npy`, so `cache/clip/{DATASET}/` keeps exactly one file per source clip
and no frame is stored twice. Changing the window geometry is a re-run of a
preprocessor, not a re-extraction.

Rules:

- **The file's presence is the switch.** When `windows.json` exists, every id in
  `labels_train.json`, `frame_labels_test.json` and `meta.json` is a **window**
  id; when it is absent, every loader behaves exactly as before. There is no
  config flag — `core.data.windows.load_windows` keys off existence, so an *empty*
  `windows.json` is refused at write time rather than making every id unresolvable.
- **`train_ids.txt` / `test_ids.txt` stay SOURCE ids.** They feed
  `extract_clip_features --ids-file` and `raft_extract --ids-file`, which write one
  `.npy` per source clip and know nothing about windows.
- **No padding, ever.** A clip shorter than one window contributes no window and is
  logged; padding would fabricate frames that `ConvScoreHead`'s
  `padding_mode="replicate"` then smooths over (lesson **C27**).
- **A window never crosses a clip boundary**, and **appearance and flow are sliced
  by the same window** — an offset between the two branches is lesson **C13** in
  miniature. `FeatureSlicer.load(..., suffix=)` covers `.stats.npy` for the same reason.
- **The train/test split is by source clip.** Two windows of one clip in different
  splits is a leak; `core.data.dada` raises on it.
- **A window's label is derived from its own sliced frame labels**, so an abnormal
  clip's normal stretch becomes genuine negative windows. An abnormal clip with no
  annotated span has no derivable window label: `--window-weak-mode drop` (default)
  leaves it out, `all-positive` marks all its windows abnormal and accepts the noise.
- **The windows are part of the cache identity** (lesson **C2**): a dataset dir
  carries the geometry in its name (`data/DADA2000_w32s2`), the feature cache
  carries only the stride (`cache/clip/DADA2000_s2`), and the KNN cache carries the
  geometry because its keys are window centres.

Built by `python -m core.data.dada ... --window-length 32 --window-stride 16`;
runbook in `DADA_SETUP.md` §10.3.

## MSAD annotation source (user-provided, 2026-07-08)

`core/data/msad.py` builds the label files from the 5-column table

```
name    scenario    total_frames    anomaly_start_frame    anomaly_end_frame
```

(tab-, comma- or 2+-space-separated, optional header; normal rows have empty
or negative window fields). Default assumptions, overridable by CLI flags:
frame indices are 0-based (`--one-indexed`) and the anomaly end frame is
inclusive (`--end-exclusive`). Frame-label rule: sampled frame `i` covers raw
frame `i*stride` and is 1 iff `start <= i*stride <= end`. The default split
follows MSAD protocol ii ratios (abnormal 0.5 / normal 0.25 to test), seeded
and stratified by (label, scenario); `--split-file` (test ids) overrides.
Train-video anomaly windows are written to `meta.json` only — weak supervision
is preserved.
