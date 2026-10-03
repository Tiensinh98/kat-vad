# KAT-VAD v2 — training the four arms (A0–A3)

**Scope:** how to build the inputs and train/evaluate each v2 arm with the code on branch `v2`.
The network is `KAT_VAD_v2_ARCHITECTURE.md`; the rules are `PREREG_ADDENDUM.md` (§10 = the
implementation choices K1–K6 this document follows).

## 1. What changes versus a KIP-off run

Nothing in the trunk, heads, losses, windowing or DVS. Two things are added:

| | Where | Trainable |
|---|---|---|
| CRN `x̃ = s·(x − μ^x)` and the motion scaling `c·ũ⊘σ_u` | baked into the **v2 input cache** offline (`core/data/v2_inputs.py`, `core/tools/build_v2_inputs.py`) | no |
| `h = x̃ + W_u(c·ũ⊘σ_u)`, `W_u` zero-init | `core/models/motion_residual.py`, called first in `KATVAD.forward` when `model.motion_dim > 0` | `W_u` only (≈ 0.39 M for B, 0.20 M for S) |

| Arm | `--clip-dir` | `v2.crn` | `v2.motion` | `model.motion_dim` |
|---|---|---|---|---|
| A0 | the plain CLIP cache (`cache/clip/DADA2000_orig`) | `none` | `none` | 0 |
| A1 | a baked A1 cache | `R2` (E2's choice at s3, D11) | `none` | 0 |
| A2 | a baked A2 cache | `none` | encoder | 768 (B) / 384 (S) |
| A3 | a baked A3 cache | `R2` | encoder | 768 / 384 |

Every v2 arm also needs `kip.enabled=false` (validated; v2 retires KIP). The reference is **R2** (median):
E2 re-read at stride 3 under D11 picked it over R1 by a tie-level margin (`RESULTS_E2_CRN.md` §D11);
the code accepts any of R1–R4, and the manifest records which one a cache was baked with.

## 2. Build the input cache

`fit` fits `s`, `c`, `m_u`, `σ_u` on **T2-train minus T2-val** and bakes every source video the T2
corpus references (train, T2-val, test). A T2 window is a slice of its baked source, so its CRN
reference is the source video in training and in evaluation (K2).

```bash
# A1 (CRN only; no motion features needed)
python -m core.tools.build_v2_inputs fit \
  --t2-dir data/DADA2000_orig --clip-dir cache/clip/DADA2000_orig \
  --crn R2 --motion none --out-dir cache/v2/A1_R2/DADA2000_orig

# A3 (needs the full-T2 VideoMAE cache from P4)
python -m core.tools.build_v2_inputs fit \
  --t2-dir data/DADA2000_orig --clip-dir cache/clip/DADA2000_orig \
  --video-dir cache/video/vit_b_k710_dl_from_giant/DADA2000_orig_s8_squash \
  --crn R2 --motion vit_b_k710_dl_from_giant --out-dir cache/v2/A3_R2_B/DADA2000_orig
```

`apply` bakes another corpus with the **fitted** statistics; each clip is its own reference. On full
DoTA only CLIP-only arms (A0/A1) can be scored — it has no pixels. Motion arms are scored on
**DoTA-CAP** (addendum §11, Amendment 4): the clips whose pixels were recovered from CAP-DATA, VideoMAE
cache `cache/video/<encoder>/DoTA_CAP_s1_squash/` (stride 1, row-aligned with `DoTA_s1_ncc`) built by
`core.tools.dota_cap`. `apply --stride 3` does not yet subsample a motion cache (it raises); that path
is still to build.
DoTA is read at protocol B (E1): `--stride 3` over the stride-1 cache, so the clip reference is
taken over the stride-3 rows the model sees.

```bash
python -m core.tools.build_v2_inputs apply \
  --stats-dir cache/v2/A1_R2/DADA2000_orig --clip-dir cache/clip/DoTA_s1_ncc --stride 3 \
  --ids-file core/splits/v2/dota_dev.txt --out-dir cache/v2/A1_R2/DoTA_s3_from_s1
```

**Open (before P6):** `core.evaluate` scores DoTA at stride 8 against `labels_s8`, and
`core.tools.rate_matched_eval` reads the raw CLIP cache. Neither yet scores a baked stride-3 DoTA
cache at native frames, so protocol B for a v2 arm needs that path first.

Each output directory holds `{id}.npy` (float32 rows, width 512 or 512 + d_v), `v2_input_stats.npz`
and `v2_input_manifest.json` (`arm`, `crn`, `motion`, `s`, `c`, `train_ids_sha1`, source dirs).
**A cache is bound to the split it was fitted on**: rebuilding T2-val means rebuilding the cache.

## 3. Train and evaluate

Same command as a phase-4 KIP-off arm, plus the v2 flags and the baked `--clip-dir`:

```bash
python -m core.train \
  --set train.stage=2 --set train.seed=2099 --set train.num_epochs=20 --set train.amp=true \
  --set data.dataset=DADA2000_orig \
  --set model.score_head_kernel=3 --set loss.mil_topk_pct=5 --set kip.enabled=false \
  --set v2.crn=R2 --set v2.motion=vit_b_k710_dl_from_giant --set model.motion_dim=768 \
  --data-dir data/DADA2000_orig --clip-dir cache/v2/A3_R2_B/DADA2000_orig \
  --knn-cache cache/knn/DADA2000_orig/knn_cache.npz --output-dir runs/A3_s2099
```

`core.evaluate` takes the architecture **and** `v2` from the checkpoint; point `--clip-dir` at the
matching baked cache (T2 or DoTA). Both CLIs raise on a manifest mismatch (K4):

| Run config | Cache | Result |
|---|---|---|
| v2 on | plain CLIP cache (no manifest) | `ValueError: … plain CLIP cache` |
| v2 on | cache baked for another `crn`/`motion` | `ValueError: … baked for crn=…` |
| v2 off (A0) | a baked v2 cache | `ValueError: … baked for …` |

## 4. What to read in `metrics.jsonl`

For A2/A3 every batch record carries, besides the losses:

| Key | Meaning | Expect |
|---|---|---|
| `motion_share` | `ρ_u = ‖W_u(c·ũ⊘σ_u)‖ / ‖x̃‖` over valid steps | 0 at step 0, rising if the stream is used |
| `w_u_norm` | `‖W_u‖_F` | 0 at step 0 |

Neither enters `total`. `ρ_u ≈ 0` for a whole run means the stream is unused: record it, do not
tune lr or `c` (plan P6).

## 5. Checkpoint compatibility

- v1 / phase-4 / A0 checkpoints have no `model.motion_dim` or `v2` section; they load as A0.
- A2/A3 checkpoints carry `motion_residual.proj.*`; loading an A0 state into a motion model raises
  (strict `load_state_dict`, lesson C5).
