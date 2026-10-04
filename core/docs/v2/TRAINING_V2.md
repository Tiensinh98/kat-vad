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

**Encoder = V2-S (`vit_s_k710_dl_from_giant`, 384-d)** — E2(d)'s pick (`RESULTS_E2D.md`, addendum §15 G1).

Every v2 arm also needs `kip.enabled=false` (validated; v2 retires KIP). The reference is **R2** (median):
E2 re-read at stride 3 under D11 picked it over R1 by a tie-level margin (`RESULTS_E2_CRN.md` §D11);
the code accepts any of R1–R4, and the manifest records which one a cache was baked with.

## 2. The training dataset dir: T2-train minus T2-val

Proposal §7.2: every v2 arm trains on T2-train **minus T2-val**, and T2-val is the in-domain decision
set. The T2 dataset dir trains on every T2-train window (T2-val included) and evaluates T2-test, so
v2 runs use a derived dir:

```bash
python -m core.tools.v2_dataset --data-dir data/DADA2000_orig --out-dir data/DADA2000_orig_v2
python -m core.data.knn_cache --data-dir data/DADA2000_orig_v2 --dataset DADA2000_orig \
  --clip-dir cache/clip/DADA2000_orig --output cache/knn/DADA2000_orig_v2/knn_cache.npz
```

`labels_train.json` drops T2-val's windows; `frame_labels_test.json` holds **T2-val's windows**,
labelled by the arithmetic that labelled T2-test (the tool first rebuilds every parent T2-test
window's labels from `meta.json` and refuses to write on a single mismatch). `core.evaluate` on this
dir therefore scores T2-val; T2-test stays unread until the final report. The KNN cache is keyed to
the train split, so it is rebuilt (on raw CLIP, shared by every arm). **A phase-4 checkpoint scored
on this dir's T2-val is scored in-sample** — it trained on those windows (addendum §14 O6).

## 3. Build the input cache

`fit` fits `s`, `c`, `m_u`, `σ_u` on **T2-train minus T2-val** and bakes every source video the T2
corpus references (train, T2-val, test). A T2 window is a slice of its baked source, so its CRN
reference is the source video in training and in evaluation (K2).

```bash
# A1 (CRN only; no motion features needed)
python -m core.tools.build_v2_inputs fit \
  --t2-dir data/DADA2000_orig --clip-dir cache/clip/DADA2000_orig \
  --crn R2 --motion none --out-dir cache/v2/A1_R2/DADA2000_orig

# A3 (needs the full-T2 V2-S cache: colab/v2/p4_s_full_t2_d6.ipynb step 2); A2 = the same with --crn none
python -m core.tools.build_v2_inputs fit \
  --t2-dir data/DADA2000_orig_v2 --clip-dir cache/clip/DADA2000_orig \
  --video-dir cache/video/vit_s_k710_dl_from_giant/DADA2000_orig_s8_squash \
  --crn R2 --motion vit_s_k710_dl_from_giant --out-dir cache/v2/A3_R2_S/DADA2000_orig
```

`apply` bakes another corpus with the **fitted** statistics; each clip is its own reference. On full
DoTA only CLIP-only arms (A0/A1) can be scored — it has no pixels. Motion arms are scored on
**DoTA-CAP** (addendum §11, Amendment 4): the clips whose pixels were recovered from CAP-DATA, VideoMAE
cache `cache/video/<encoder>/DoTA_CAP_s1_squash/` (stride 1, row-aligned with `DoTA_s1_ncc`) built by
`core.tools.dota_cap`. `apply --stride 3` subsamples a motion cache too, provided it is stride-1 and row-aligned with
the CLIP one (its `video_manifest.json` says `stride: 1`; anything else raises).
DoTA is read at protocol B (E1): `--stride 3` over the stride-1 cache, so the clip reference is
taken over the stride-3 rows the model sees.

```bash
python -m core.tools.build_v2_inputs apply \
  --stats-dir cache/v2/A1_R2/DADA2000_orig --clip-dir cache/clip/DoTA_s1_ncc --stride 3 \
  --ids-file core/splits/v2/dota_dev.txt --out-dir cache/v2/A1_R2/DoTA_s3_from_s1
```

**Scoring DoTA at protocol B** (E1's protocol) for any arm: `core.tools.protocol_b_eval`. It loads
each checkpoint after the J10 gate, checks the input cache against the checkpoint's `v2` (K4) and
its baked stride, scores whole clips, interpolates to native frames, seed-averages and writes
per-clip AUCs (`clip_aucs.json`, the input of every paired Δ) with a cluster-bootstrap macro.
`--split dota_dev` for CLIP-only contrasts and `F`; `--split dota_cap_dev` for every motion
contrast (D13). A0 reads the plain `DoTA_s1_ncc` (subsampled in the tool); A1–A3 read their
`apply --stride 3` cache.

```bash
python -m core.tools.protocol_b_eval --run s2099 runs/A1_s2099/checkpoint_last.pt \
  --input-dir cache/v2/A1_R2/DoTA_s3_from_s1 --s1-dir cache/clip/DoTA_s1_ncc \
  --metadata data/DoTA/metadata_val.json --split-file data/DoTA/val_split.txt \
  --data-dir data/DoTA/labels_s8 --split dota_dev --out-dir outputs/v2/REPORTS/pb_A1
```

Each output directory holds `{id}.npy` (float32 rows, width 512 or 512 + d_v), `v2_input_stats.npz`
and `v2_input_manifest.json` (`arm`, `crn`, `motion`, `s`, `c`, `train_ids_sha1`, source dirs).
**A cache is bound to the split it was fitted on**: rebuilding T2-val means rebuilding the cache.

## 4. Train and evaluate

Same command as a phase-4 KIP-off arm, plus the v2 flags and the baked `--clip-dir`:

```bash
python -m core.train \
  --set train.stage=2 --set train.seed=2099 --set train.num_epochs=20 --set train.amp=true \
  --set data.dataset=DADA2000_orig \
  --set model.score_head_kernel=3 --set loss.mil_topk_pct=5 --set kip.enabled=false \
  --set v2.crn=R2 --set v2.motion=vit_s_k710_dl_from_giant --set model.motion_dim=384 \
  --data-dir data/DADA2000_orig_v2 --clip-dir cache/v2/A3_R2_S/DADA2000_orig \
  --knn-cache cache/knn/DADA2000_orig_v2/knn_cache.npz --output-dir runs/A3_s2099
```

`core.evaluate` takes the architecture **and** `v2` from the checkpoint; point `--clip-dir` at the
matching baked cache (T2 or DoTA). Both CLIs raise on a manifest mismatch (K4):

| Run config | Cache | Result |
|---|---|---|
| v2 on | plain CLIP cache (no manifest) | `ValueError: … plain CLIP cache` |
| v2 on | cache baked for another `crn`/`motion` | `ValueError: … baked for crn=…` |
| v2 off (A0) | a baked v2 cache | `ValueError: … baked for …` |

## 5. What to read in `metrics.jsonl`

For A2/A3 every batch record carries, besides the losses:

| Key | Meaning | Expect |
|---|---|---|
| `motion_share` | `ρ_u = ‖W_u(c·ũ⊘σ_u)‖ / ‖x̃‖` over valid steps | 0 at step 0, rising if the stream is used |
| `w_u_norm` | `‖W_u‖_F` | 0 at step 0 |

Neither enters `total`. `ρ_u ≈ 0` for a whole run means the stream is unused: record it, do not
tune lr or `c` (plan P6).

## 6. Pilot diagnostics (addendum §4, §14 O1–O7)

`core.tools.v2_diagnostics` — one checkpoint per call, one forward pass per item, on the v2 dataset
dir's T2-val windows and (unlabelled) DoTA-dev at protocol B through the arm's own input:

| Read | What | Decides |
|---|---|---|
| O1 guardrails | micro < clip oracle, macro ≥ micro, micro ≥ A0 − 0.01 | P7 |
| O2 | window-level AUC beside macro (C14 signature) | printed |
| O3 | `motion_share` / `w_u_norm` first, last, max | printed |
| O4 | position R² on `V^t` (ridge → `t/T`, folds by source), beside the input's and A0's | printed |
| O5 | source-shortcut AUC, T2-val vs DoTA-dev, on `V^t` and on `y^bin` | printed |
| O6 / D5 | A0 vs the phase-4 KIP-off mean, micro and macro within 0.02 (`v2_diagnostics.d5`) | P7 |

`--dota-split dota_cap_dev` moves O5's DoTA side to DoTA-CAP-dev (addendum §15 G5): batch 2 reads all four arms
there, because a motion arm has DoTA rows only on DoTA-CAP; A0/A1 are re-read into `diag_cap/` and their T2-val
columns must reproduce batch 1.

Runbooks: `colab/v2/p6_pilot_batch1.ipynb` (A0 + A1, D16) and `colab/v2/p6_pilot_batch2.ipynb` (A2 + A3 on V2-S,
bakes them first; needs batch 1 and the full-T2 V2-S cache).

## 6b. D6 shuffle control (addendum §2 D6, §12 N11, §15 G3)

`dota_cap extract --shuffle-seed 2024 --dota-ids-file <dota_cap_dev> --encoders vit_s_k710_dl_from_giant` writes
`DoTA_CAP_s1_squash_shuf2024`: the same pixel-gated DoTA-CAP clips, every window's 16 frames permuted by one fixed
permutation (`extract_video_features.shuffled_frame_order`). `python -m core.tools.e2d_shuffle` re-reads the
in-domain probe on ordered vs shuffled `u`. Printed only; runbook `colab/v2/p4_s_full_t2_d6.ipynb`.

## 7. Checkpoint compatibility

- v1 / phase-4 / A0 checkpoints have no `model.motion_dim` or `v2` section; they load as A0.
- A2/A3 checkpoints carry `motion_residual.proj.*`; loading an A0 state into a motion model raises
  (strict `load_state_dict`, lesson C5).
