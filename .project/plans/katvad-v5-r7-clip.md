# KAT-VAD v5 R7 — MQR on CLIP 14 × 14 patches (after check C)

Rules: `core/docs/v5/PREREG_V5.md` **§4** (DRAFT). Branch `v5-r7` (off `v5-probe@a6d61df`). Predecessors:
`katvad-v5-mqr.md` (Stages 0–2b), `katvad-v5-tile-mqr.md` (v5.1, PILOT-FAIL), check C (§3.7, CLIP-BEATS-CENTRE).

## 0. Status

| phase | what | state |
|---|---|---|
| 0 | user: R7 at native 14 × 14 × 512; store the grids unpooled for later pooling; new branch | **decided 2026-10-30** |
| 1 | CLIP grid extraction (Nexar train s8 / val s3, DoTA-CAP all 1,129 s3) + gates G1 / G-R7c | **BUILT 2026-10-30**: `v5_stage1 extract --clip-grid`, `v5_stage0 extract --clip-grid`, `v5_grids.clip_grid_manifest`, constants `V5_CLIP_*` / `V5_R7_*`, tests `test_v5_r7_grids.py` (7); notebook `colab/v5/r7_clip_grids.ipynb`. **Not run.** |
| 2 | PREREG §4.2 (recommended column, user's "fastest result" instruction) → readout/trainer on 196 CLIP cells, `W_c` path, maps vs C14 | **BUILT 2026-10-30**: `v5.window=frame` (config `_validate_v5_frame`), `TileReadout(frame=True)` + `frame_tile_features`, R7m `GridReadout` at 512 × 196, `KATVAD` residual width = readout width (A2's `u` skipped), `frame_lag` / `eval_grid` frame path, no `v5_head.npz`, `v5_maps.frame_change_map` + 14 × 14 centre prior + `mass_minus_centre` + `r7_points`, `v5_tile_read --gate r7` (no `--nexar-dir`), `v5_stage0 extract --dev-only`; constants `V5_R7_ROWS` etc.; tests `test_v5_r7_grids.py` (23) |
| 3 | fast pilot s2024 on DoTA-CAP-dev (§4.4) → read-out | **notebook `colab/v5/r7_fast_pilot.ipynb` built, not run** |

## 1. Why 14 × 14, why store unpooled

Check C: C14 beats the centre prior only at 14 × 14 (hit − centre +0.131); pooled to 7 × 7 it ties (−0.002). A 7 × 7
cache could never recover the 14 × 14 read; a 14 × 14 cache gives 7 × 7 / 2 × 2 by a reshape-mean. Storing is fp16,
200 KB per frame.

**Not stored:** the 768-d hidden states before `visual_projection`; other strides (Nexar s1 ≈ 182 GB). Either needs a
new extraction.

## 2. Phase 1 design

| item | choice | reason |
|---|---|---|
| tensor | `visual_projection(post_layernorm(patch))`, 14 × 14 × 512 fp16 | = C14's space and `image_embeds`' (same path as CLS) |
| Nexar pixels | N2 `decode_resized` + CLIP mean/std on the CPU (`clip_grid_rows`) | reproduces `Nexar_s1_ncc` (`nexar_extract.clip_rows`) |
| Nexar TF32 | torch defaults (no `exact_fp32`) | N2 built `Nexar_s1_ncc` with the defaults |
| DoTA-CAP pixels | Stage 0's `clip_patches` on the rebuilt CAP folders | reproduces `dc14` (G-R7c) |
| rows | v5.1's (s8 / s3 / s3) | grids line up with the v5.1 and A2 caches row for row |
| DoTA-CAP set | all 1,129 (Stage 0 had CAP-dev only, maps only) | CAP-eval needs no second extraction |
| shared code | `v5_stage0.clip_patch_tokens` (one batch) used by `clip_patches` and `clip_grid_rows` | DRY |

Cache dirs: `Thesis-V5/cache/video_grid/clip_vit_b16/{Nexar_s8,Nexar_s3,DoTA_CAP_s3}_cgrid14/` + `video_manifest.json`
(`stride`, `frame_step` 1, `slots` 1, `window_kind` `frame`, model + revision, transform).

## 3. Phase 2 — as built (fast pilot, §4.4)

- Rows carry A2's bake `[x ; u]` + `G` (+ `ΔG` for R7t, lag 1 row, per item before DVS). The model drops `u` and
  feeds `LN(r)` into `MotionResidual(512, 512)` = `W_c`, zero-init.
- Memory: an R7t row is 201,600 floats (806 KB fp32), ≈ 2.7 × a v5.1 row; a batch of 64 windows × 20 rows is
  ≈ 1–2 GB on the host. Use a High-RAM runtime.
- Not built yet: `nexar_val` at s3 needs Δ at 3 rows (0.1 s / row); the fast pilot does not read it.

## 4. Risks

- Prior evidence says a video-level MIL objective does not move a readout onto the agent (Stage 2b, v5.1). The bar
  (beat the untrained C14 map) may be unreachable; the outcome is then NULL at ≈ v5.1's GPU cost.
- G1 0.999 on Nexar assumes `Nexar_s1_ncc` was computed under the same TF32 defaults; a failure there is a numerics
  question first (paste the report), not a data one.
