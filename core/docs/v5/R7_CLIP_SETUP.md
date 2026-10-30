# v5 R7 — CLIP 14 × 14 × 512 grids: run setup (PREREG_V5 §4, Phase 1)

Runbook: `colab/v5/r7_clip_grids.ipynb` (A100, `/content` ≥ 90 GB). Upload branch `v5-r7` to `Thesis-V5/kat-vad`.

| step | tool | output (Drive, `Thesis-V5/`) |
|---|---|---|
| 1 Nexar | `python -m core.tools.v5_stage1 extract --clip-grid --clip-s1-dir <Nexar_s1_ncc> --stride {8,3} ...` | `cache/video_grid/clip_vit_b16/Nexar_s{8,3}_cgrid14/` |
| 1 DoTA-CAP | `python -m core.tools.v5_stage0 extract --clip-grid --dota-clip-dir <DoTA_s1_ncc> --stage0-maps-dir <v5_stage0/maps> ...` (5 CAP groups) | `cache/video_grid/clip_vit_b16/DoTA_CAP_s3_cgrid14/` |
| 2 summary | notebook cell | `outputs/v5/v5_r7/REPORTS/r7_phase1.md` |

**Files.** One `<id>.npy` per video / clip, `(rows, 14, 14, 512)` float16; row `k` = native frame `k · stride`
(DoTA-CAP: `step_end_frames(n, 3)`). Tokens are `visual_projection(post_layernorm(patch))` — the 512-d space of
`image_embeds`. Pool later with a reshape-mean, e.g. 7 × 7: `g.reshape(T, 7, 2, 7, 2, 512).mean((2, 4))`.

**Gates.** Nexar: length == `Nexar_s1_ncc`, max |projected CLS − `image_embeds`| ≤ 1e-5, every row's `image_embeds`
cos ≥ 0.999 vs `Nexar_s1_ncc`. DoTA-CAP: aspect, projected CLS, every row ≥ 0.95 vs `DoTA_s1_ncc`, and on CAP-dev
**G-R7c**: the stored grid's lag-1 change map == Stage 0's `dc14` (max |Δ| ≤ 5e-3). A failing item stores nothing.

**Resume.** Grids go to VM disk and are mirrored to Drive every 5 min; a re-run restores Drive's copy and skips
finished items (per-item report + readable `.npy`). CUDA OOM → `MAX_GPU_JOBS = 1`.
