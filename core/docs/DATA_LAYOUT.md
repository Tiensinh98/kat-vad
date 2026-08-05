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
