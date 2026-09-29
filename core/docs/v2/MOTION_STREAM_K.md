# v2 motion stream — extractor contract and kill-switch K (P1)

Pre-registration: `PREREG_ADDENDUM.md` §3 and **§6.1** (Amendment 1, D7). Runbook:
`colab/v2/p1_kill_switch.ipynb`. Plan: `.project/plans/katvad-v2-e0-e2.md` P1.

## 1. The encoder (`core/models/videomae_v2.py`)

- **VideoMAE V2 distilled** (`vit_b_k710_dl_from_giant`, 768-d; `vit_s_…`, 384-d). These
  checkpoints are **not in `transformers`**. The inference path of upstream
  `OpenGVLab/VideoMAEv2` `models/modeling_finetune.py` is vendored with its parameter names
  unchanged, so the checkpoint loads with `strict=True`. Only the K710 `head.*` is dropped.
- **Weights pinned twice:** HF repo `OpenGVLab/VideoMAE2` at commit `706cc172…`, and a
  sha256 per file (`constants.VIDEOMAE_WEIGHTS_SHA256`). A mismatch raises. The file is
  `{"module": state_dict}` and loads with `torch.load(weights_only=True)` (C4, C15).
- **Parity measured 2026-09-28** against upstream (timm 0.9.16) on the real ViT-S weights:
  strict load clean, max |Δ| **1.5e-6** on `forward_features`.
- `u` = `fc_norm(mean over tokens)`, the feature the upstream K710 head reads.

## 2. The extractor (`core/tools/extract_video_features.py`)

| Item | Value |
|---|---|
| Row `i` | Aligned with CLIP step `i`: raw frame `stride·i`, so `L = ceil(frames / stride)`, the same as `cache/clip/…` |
| Clip | 16 frames, every 3rd raw frame, **ending at** frame `stride·i`: 1.5 s at the **assumed** 30 fps (plan A1). No lookahead (tested) |
| Clip start | Indices are clamped to 0, i.e. frame 0 is repeated |
| Transform | Anisotropic resize of the full frame to 224² (the CLIP `_ncc` field of view), then ImageNet mean/std |
| Cache | `cache/video/<encoder>/<dataset>_s<stride>_squash/{id}.npy` plus `video_manifest.json` |
| Rerun under a different manifest | Raises (C2); `--force` rebuilds |
| Writes | Atomic and resumable (`feature_cache`, C11) |

DADA-original frames need the C26 symlink farm (`flat/{t..}_v..` pointing at `…/images`). The notebook builds it per shard.

## 3. Kill-switch K (`core/tools/kill_switch_probe.py`)

- `pick`: about 300 T2-train sources with **no T2-val source**, drawn per `type` with the
  same `draw_subset` and pool as the frozen T2-val. Writes `k_sources.txt` and a manifest (sha1).
- `run`: runs the frame linear probe on whole sources at stride 8. Labels follow the annotation
  span (Gate D0's rule). It probes three feature sets, `x`, `u` and `[x;u]`, under
  (i′) source-grouped CV and (ii′) type-grouped CV. Metric = mean per-source AUC; CIs are
  bootstraps over sources. Output is `k_readout.{json,md}` with a verdict:
  **GO / KILL / SUSPECT_PIPELINE**.
- **Rule = addendum §6.1 as amended (option A, 2026-09-28):** KILL iff both Δ upper < +0.03.
  The P0 rule's extra "point ≤ 0" leg was dropped. If `u` adds nothing to `x`, Δ ≈ ±1e-4 and
  that leg made the verdict a sign coin-flip (KILL on 1 of 4 synthetic seeds). With the leg
  gone, a redundant `u` is KILLed deterministically (tested on seeds 0–3).

## 4. Result and the K-pos diagnostic

- **K ran 2026-09-28: GO** (295 sources, 12,409 frames). `u` 0.760 · `x` 0.615 · Δ(`[x;u]` − `x`)
  **+0.132 [+0.109, +0.155]** (i′), **+0.126 [+0.101, +0.152]** (ii′). The control passed (lower bound 0.740).
  `[x;u]` < `u` alone: the 1280-d linear probe loses a little when CLIP is added.
- **Label-only rulers on the same sources:** absolute `t` 0.529, padded-clip flag 0.573, relative-position
  tent (fit in-sample) **0.729**. So the gap is not yet known to be motion.
- `kill_switch_probe diag` (addendum **§6.2**, printed and not gated): (P) drop the
  `padded_steps(stride)` leading steps (6 at stride 8) → `PAD_EXPLAINS` / `NOT_PAD`; (Q) `p` = cubic
  in `(t+0.5)/L`, `[x;u;p]` vs `[x;p]` → `POSITION_PROXY` / `BEYOND_POSITION`. Same bar (+0.03), probe
  and bootstrap as K. Writes `k_diag.{json,md}` and leaves `k_readout.*` alone. Runbook: notebook §4b.
- **K-pos ran 2026-09-28.** (P) **NOT_PAD**: without the 6 padded steps `u` is unchanged (0.758); `x` rises
  0.615 → 0.646; Δ +0.106 [+0.080, +0.130] (i′), +0.100 [+0.074, +0.126] (ii′). (Q) **BEYOND_POSITION**:
  `p` 0.730 out-of-fold (matches the in-sample tent 0.729), `[x;p]` 0.734, `[x;u;p]` 0.777; Δ **+0.043
  [+0.023, +0.064]** (i′), **+0.039 [+0.018, +0.061]** (ii′).
- **Reading.** `u` carries a frame signal that position does not explain, but it is about a third of K's Δ:
  +0.13 → +0.04. **CLIP adds +0.004 over position** on whole DADA sources. Whole-source DADA macro is
  mostly a position number. On T2-val **windows** (the unit the model trains on) the best position
  ruler is only **0.575** (anomaly start/end medians 25 % / 90 % of the window).
