# KAT-VAD v5.1 — Tile-scored MQR (proposal 2026-10-29 rewrite)

Proposal: `core/docs/final/Thesis proposal Deep learning-based traffic video anomaly detection.md` (uncommitted diff vs
`bc58a64`, +139 / −89). Predecessor: `.project/plans/katvad-v5-mqr.md` (Stage 0–2b; Stage 2b G0 FAIL, free queries
saturate ≈ 40/49 tiles). Branch to build on: new branch off `v5-probe` (suggested `v5.1-tile`).

## 0. Status

| phase | what | state |
|---|---|---|
| 0 | user decisions D-a…D-e (§3) + PREREG_V5 §3 drafted & signed | 2026-10-29: D-c = **fixed HP, 1 seed** (λ 0.5, K 4, β 0.01, s2024); §2 mapping + Nexar-only **approved**; D-a (s3) and D-b pending the user's reply; PREREG §3 not drafted |
| 1 | code: centred 4×7×7 extraction, tile scorer, tile-level MIL, maps, eval reads | **BUILT 2026-10-29** on `v5-probe` (user: no new branch): see §8; PREREG_V5 §3 **drafted, not signed**; D-a = s3, D-b = row 4 dropped |
| 2 | Colab extraction (Nexar train s8, Nexar val, DoTA-CAP) + gates | **DONE 2026-10-30**: G-c 1129/1129, S1-G1 563/563 + 225/225, S1-G4 PASS |
| 3 | pilot, 1 seed (rows R1c/R3c/R6c; R5c not run; no λ/K/β selection) | **READ OUT 2026-10-30: PILOT-FAIL** (R6c fails (i) maps, (ii) `nexar_val` −0.024; R3c fails all three) → `core/docs/v5/RESULTS_V5_TILE.md`; check C (PREREG §3.7) built, not run |
| 4 | confirmation, 3 seeds, decision gates | **not run** (pilot failed; §3.3) |
| 5 | Stage 3 protocols (MSAD/TAD/CCD/DADA, full ladder) | only if Phase 4 supports MQR |

## 1. What changed in the proposal (code-relevant only)

| # | proposal change | current code (`v5-probe`) | consequence |
|---|---|---|---|
| C1 | **Tile scorer** `h(F) = w₂ᵀ GELU(W₁F + b₁) + b₂`, `W₁ ∈ R^{128×2D}`, last layer zero-init | free queries `u_m` (`core/models/mqr.py` `GridReadout`) | new module |
| C2 | features `F = [LN(R); LN(Δ)]`, `R` = cell − **median** over the 49 tiles of the slot, `Δ` = slot change − its median; Δ repeated on the last slot | content keys `LN(G)` / change keys `LN([G; G − G_{t−ℓ}])` across *rows* | new feature fn, computed on GPU; `grid_change` / lag path becomes legacy |
| C3 | **grid 4×7×7** (pairs of the 8 slots kept), 196 cells, 150 KB/clip fp16 | 7×7 time-averaged, 37.6 KB/row | **re-extract every grid** (×4 storage) |
| C4 | **centred clip** (t−8…t+7 at 10 fps; Nexar 30 fps, step 3 → shift ≈ 23 frames ≈ 0.75 s), edge repeat; shift test | causal (Stage 0 check L picked causal) | extraction window + manifest field `window: centred`; R1 must be retrained centred |
| C5 | readout `a = softmax_i(e/T)`, `T = exp(θ)`, θ₀ = 0, T ≥ 0.05; `r = W_V Σ a·Q`; gate `x = c + tanh(α)·W·LN(r)` (unchanged, α₀ = 0) | softmax over `u·LN(K)/√d` | new readout; head path (fc_norm, c, m_u, σ_u, `MotionResidual`) kept (D1) |
| C6 | **tile-level MIL**: abnormal → BCE(mean top-K cells of top-k steps, 1); normal → dense BCE(p, 0) on every valid cell; β·(L1 abn / L2 norm), β = 0.01; K = 4 | `spatial_mil_loss` (max over queries, top-k steps, BCE) | new loss in `core/losses/smil.py` |
| C7 | DVS synthetic videos: abnormal for MIL; their pseudo-normal steps enter the dense negatives except ±8 frames of a boundary | sMIL sees DVS items with their label only | needs `pseudo_frame_label` + boundary mask per step in the loss |
| C8 | evidence map `E_t(j) = max_s p_{t,s,j}` (the classifier, not attention) | `Σ_m s·a` | `v5_maps` change |
| C9 | baselines: uniform, centre, **untrained change-magnitude map**; metric **mass-in-box** added | uniform, centre | add map + metric |
| C10 | **position-controlled AUC** (cross-clip same-relative-position pairs; model + position over position-only), time-reversed control, shift test | `p_T2` / `f_2` prints | new eval read |
| C11 | equivalence TOST ±0.01 AUC; decision gates (pilot 1 seed → 3-seed confirm) | S2-7 / S2-9 | new readout tool |
| C12 | ladder rows: 3 = untrained change top-K, 4 = free-query **repaired** (cosine logits, κ₀ = log₂(N²−N) ≈ 15, random/k-means init, LSE, overlap penalty), 5 = tile scorer w/o dense negatives, 6 = full MQR | R1/R3/R5/R5d(+q) | new row table `V5_ROWS` (v5.1) |
| C13 | λ by held-out 15 % P3 video AUC; K ∈ {2,4,8}, β ∈ {0, 0.01} by pointing on the "DoTA-train pilot subset" | λ = 0.5 fixed | selection runs |

## 2. Conflicts with what the project can actually run (must go into PREREG §3)

- **DoTA-train has no pixels** (memory `dota-frames-unrecoverable`). The proposal's "500-clip DoTA-train pilot subset" and
  "1,305 / 1,402 official list" cannot be built. Map: pilot/selection set = **DoTA-CAP-dev (569)**, test = **DoTA-CAP-eval
  (560)**, as v5 so far. Report rule: DoTA-CAP = DoTA (memory `report-dota-cap-is-dota`).
- **P3 = Nexar + CCD** in the proposal; v5 user decision = **Nexar-only**. Keep Nexar-only through Phase 4; CCD only in Phase 5.
- This is a **post-hoc redesign after Stage 2b's failure** — PREREG §3 must say so and freeze gates before any v5.1 number.
- `p_T2` 0.822 still beats every model on CAP-dev → the position-controlled AUC (C10) is the only honest detection read.

## 3. Decisions for the user (Phase 0)

- **D-a nexar_val grid stride.** s1 (current B1 rule: crops start on any phase) = **≈ 39 GB**, ≈ 3–5 h extraction; or
  s3 aligned crops (protocol N-B rows are s1[::3]) = **≈ 13 GB**, ≈ 1–1.5 h. Recommendation: **s3** + snap crop starts to
  the s3 phase (disclosed deviation; the crop endpoint's variance is dominated by videos, not phase).
- **D-b free-query row 4** in the pilot or deferred. Recommendation: **defer** (Stage 2/2b already measured free queries
  twice; row 4 only if R6c passes).
- **D-c selection budget.** Full λ × K × β = 18 runs; sequential (λ at K=4, β=0.01, then K × β at chosen λ) = 7 runs.
  Recommendation: **sequential**.
- **D-d 8-slot side ablation** (doubles grids) — recommendation: **skip until Phase 4**.
- **D-e** keep v5 Stage 0–2b code paths (R1/R3/R5/R5d) for reproducibility — recommendation: **keep**, add `readout=tile`
  beside them, no deletion.

## 4. Phases

### Phase 0 — PREREG_V5 §3 (0.5 day, no GPU)
- [ ] Write §3: rows R1c/R3c/R5c/R6c(/R4c), sets (§2 mapping), centred window, gates (proposal "Decision gates"), TOST,
  position-controlled AUC definition, shift-test pass rule (peak at offset 0 ± 1 step), extraction gates.
- [ ] User signs §3 before Phase 2.

### Phase 1 — Code (2–3 days, CPU, data-free)
Every edited symbol gets `trace_call_path` first (CLAUDE.md §13); lessons meta-index loaded.
- [ ] **Constants** `V5_GRID_SLOTS = 4`, `V5_GRID_CELLS = 196`, `V5_WINDOW_CENTRED`, `V5_TILE_HIDDEN = 128`, `V5_TILE_K = 4`,
  `V5_TILE_BETA = 0.01`, `V5_TEMP_MIN = 0.05`, `V5_DVS_BOUNDARY_FRAMES = 8`, `V5_TOST_MARGIN = 0.01`, v5.1 row table.
- [ ] **Extraction** (`core/tools/v5_stage1.py extract` + `VideoMAEv2Encoder.forward_tokens`): keep slot axis
  (8 → 4 by pair mean, 14×14 → 7×7), centred window with edge repeat, manifest `slots`, `window`, `stride`, `frame_step`.
  Gate **G-c**: mean over slots+tiles == `fc_norm`-free token mean (cos ≥ 0.9999 vs the causal cache re-indexed by the
  shift, on frames where both exist).
- [ ] **Data** (`core/data/v5_grids.py`): grid width = 4·49·384; drop ΔG row block for `readout=tile` (features on GPU);
  `read_grid_geometry` reads `window`; refuse causal grids for a centred checkpoint and back.
- [ ] **Model** (`core/models/mqr.py`): `tile_features` (median-residual, slot change, LN), `TileScorer` (zero last layer),
  `TileReadout` (softmax(e/T), W_V = I, same head path), stop-grad option; row 3 `ChangeTopKReadout` (no params besides
  W_V); row 4 repaired free-query (only if D-b = pilot).
- [ ] **Loss** (`core/losses/smil.py`): `tile_mil_loss(p, labels, lengths, pseudo, boundary_mask, k_pct, K, beta)` in
  fp32 / logits; dense-negatives switch (row 5 vs 6).
- [ ] **Trainer** (`core/train.py`): `v5.readout=tile` branch, θ (+ queries) in no-decay group, logs: entropy (diagnostic),
  tile-score sparsity, `p` mean on normal / abnormal, T.
- [ ] **Config** `V5Config`: `readout ∈ {none, mean, mqr, tile, change_topk}`, `tile_k`, `tile_beta`, `dense_negatives`,
  `stop_grad`, `window`; `validate_v5`.
- [ ] **Maps** (`core/tools/v5_maps.py`): E = max over slots of p; untrained change map; mass-in-box; STAUC unchanged.
- [ ] **Reads** (new `core/tools/v5_stage3.py` or extend `v5_stage2.py`): position-controlled AUC (two forms), time-reversed
  control, shift test, paired bootstrap over clips, TOST, decision table.
- [ ] Tests: equality at init (R6c == R1c at step 0, ≤ 1e-6), median-residual invariants, loss on toy bags (dense
  negatives push every normal cell down), DVS boundary mask, centred-window index math, map max-over-slots, TOST.
- [ ] Docs: `core/docs/v5/` setup note for v5.1; update `TRAINING`-style runbook section.
- [ ] Quality gate (ruff/mypy/pyright on touched files) + full suite.

### Phase 2 — Extraction on Colab (1 session, A100)
- [ ] Notebook `colab/v5/stage3a_centred_grids.ipynb` → `Thesis-V5/cache/video_grid/.../{Nexar_s8_grid4x7, Nexar_s{1|3}_grid4x7, DoTA_CAP_s3_grid4x7}`.
- [ ] Gates G-c (consistency), length, finite; `exact_fp32` (TF32 off, lesson (by)); atomic writes, resume by manifest.
- [ ] Shift test on R1c-equivalent precompute where possible.

### Phase 3 — Pilot, 1 seed (s2024) — user 2026-10-29: fixed HP, no selection runs, result fast
- Rows: **R1c + R6c** (minimum), **R3c** recommended (cheap H4 baseline); untrained change map costs no training.
- The selection bullets below are **dropped** for the fast read (λ 0.5, K 4, β 0.01 frozen in §3).
- [ ] R1c (mean, centred), R3c (change top-K), R5c (tile, bag negatives), R6c (full MQR); [+R4c per D-b].
- [ ] Selection on R6c: λ ∈ {0.1, 0.5, 1} by held-out 15 % Nexar-train video AUC; then K × β by pointing on CAP-dev.
- [ ] Pilot gate (proposal): beats centre prior **and** untrained change map on pointing; no Nexar-val drop beyond R1c's
  seed spread; not below R1c on position-controlled AUC. Fail → stop, write negative result.

### Phase 4 — Confirmation, 3 seeds
- [ ] R1c, R3c, R6c (+R5c) × s2024–2026; DoTA-CAP-eval opened once; supported iff 95 % paired CI on position-controlled
  AUC **or** (STAUC and pointing) excludes 0 with no in-domain regression; TOST printed.

### Phase 5 — only if Phase 4 supports
- MSAD / TAD / CCD / DADA protocols, full ladder rows 0–8, 8-slot and 14×14 ablations.

## 5. Cost

### Storage (fp16, 4×7×7×384 = 150.5 KB / row; measured row counts from Stage 1 / Stage 0)

| cache | rows | old V7 (37.6 KB) | new 4×7×7 |
|---|---|---|---|
| Nexar train s8 (563 videos) | 81,538 | 3.1 GB | **12.3 GB** |
| Nexar val s1 (225 videos) | 257,191 | 9.7 GB | **38.7 GB** |
| Nexar val s3 (option D-a) | ≈ 85,700 | — | **12.9 GB** |
| DoTA-CAP s3 (1,129 clips) | ≈ 37,000 | 1.4 GB | **5.6 GB** |
| **total (val s1 / val s3)** | | 14.2 GB | **≈ 57 GB / ≈ 31 GB** |

Plus checkpoints (~1.4 GB each in the v2 / v5 format) × ≈ 20 runs ≈ 28 GB unless only `best` is kept. 8-slot ablation
doubles the grid rows it covers. Phase 5 adds the proposal's §4.2 table (TAD 10, CCD 34, DoTA-all 110, DADA 12.4 GB).
Old V7 caches (14 GB) can be deleted after Phase 2 passes — only on the user's word.

### GPU (A100 on Colab; training numbers are estimates until the Phase 3 timing run)

| item | basis | estimate |
|---|---|---|
| Nexar train extraction | measured 87 min A100 for the same 81.5 k windows (Stage 1) | ≈ 1.5 h |
| Nexar val extraction | linear in windows from the above | s1 ≈ 4–5 h / s3 ≈ 1.5 h |
| DoTA-CAP extraction | Stage 0: GPU tens of minutes, I/O-bound (267 GB CAP tar stream) | ≈ 2–4 h wall |
| one training run | 1,749 steps, batch 64 × L = 20 rows × 150 KB ≈ 190 MB/batch; scorer ≈ 0.1 M params → I/O-bound; needs the 12 GB train cache on VM disk / page cache | ≈ 0.3–0.7 h |
| eval per run | nexar_val crops + CAP-dev + maps | ≈ 0.2–0.3 h |
| Phase 3 pilot | 4–5 rows + 7 selection runs ≈ 11–12 runs | ≈ 6–12 h |
| Phase 4 confirm | 3–4 rows × 2 extra seeds = 6–8 runs | ≈ 3–8 h |
| **total Phases 2–4** | | **≈ 15–30 A100-hours** |

## 6. Risks

| risk | effect | mitigation |
|---|---|---|
| tile scorer learns the source (normal tiles easy to reject) | maps light the scene | Nexar-only (same source for pos / neg); centre + change-map baselines |
| ego-motion survives median subtraction | change map = ego-motion | D4 tile-similarity read; residual-only / change-only ablation later |
| 7×7 too coarse for distant agents | small-box bins null | 14×14 PCA ablation in Phase 5; box-size bins printed |
| centred window leaks future frames | not causal for deployment | disclosed; detection is offline in this thesis |
| Drive I/O / torn ckpts (lessons (bi), C10, C39) | lost runs | reuse atomic publish + `verified.json` + VM-local cache copy |
| position prior explains gains | false GO | position-controlled AUC is the gate, not CAP-dev macro |
| post-hoc redesign | forking-paths critique | PREREG §3 signed before any v5.1 number; disclose Stage 2/2b nulls |

## 7. Next steps for the user

1. Answer D-a…D-e (§3).
2. Approve the §2 mapping (DoTA-CAP-dev pilot / CAP-eval test, Nexar-only).
3. Say go → Claude drafts PREREG_V5 §3, then builds Phase 1 on a new branch.

## 8. Phase 1 build record (2026-10-29)

- `core/constants.py`: `V5_READOUT_TILE` / `V5_READOUT_CHANGE_TOPK`, v5.1 block (`V5_SLOTS_TILE`, `V5_WINDOW_CENTRED`,
  `V5_TILE_*`, rows `V5_TILE_ROWS` R1c/R3c/R5c/R6c, pilot rows R1c/R3c/R6c, read constants).
- `core/config.py`: `V5Config.grid_slots / window / dense_negatives / tile_topk_cells / tile_beta` + `_validate_v5_tile`.
- `core/tools/extract_video_features.py`: `centred_clip_indices`.
- `core/data/v5_grids.py`: `GRID_WIDTHS` (49·D | 196·D), `GridGeometry.slots / window` (manifest `slots`, `window_kind`),
  `centred_reference` (gate G-c), `tile_boundary_rows`, `item_grid` phase snap for centred caches only.
- `core/tools/v5_stage0.py`: `tile_grid`, `video_tile_grids`, `Stage0Extractor(tile=True)`, `extract --tile` (DoTA-CAP).
- `core/tools/v5_stage1.py`: `grid_rows(tile=True)`, `extract --tile` (Nexar, G-c printed), `ALL_ROWS` for row_config /
  provenance / collapse.
- `core/models/mqr.py`: `tile_residuals`, `tile_features`, `change_magnitude`, `TileScorer`, `TileReadout`,
  `ChangeTopKReadout`, `build_readout`; `GridReadout.motion`; `GridReadoutOutput.tile_logits`.
- `core/models/kat_vad.py`: `build_readout`, cell count from the readout, `v5_tile_logits` output.
- `core/losses/smil.py`: `tile_mil_loss`, `boundary_steps`.
- `core/train.py`: θ no-decay group, `L_tile` + `v5_temp` / `tile_p_*` logs, slot / window check vs the cache,
  `Trainer(tile_boundary_rows=…)`.
- `core/tools/v5_maps.py`: tile map (slot max of `p`), untrained change map (`C_` in dumps), mass-in-box,
  `beats_centre_and_change`.
- `core/tools/protocol_b_eval.py`: `--reverse-time`.
- `core/tools/v5_tile_read.py` (new): TOST, position-controlled (a)/(b), shift test, pilot gate.
- Tests `core/tests/test_v5_tile.py` (42).
- Not built: Colab notebook (Phase 2), 8-slot / 14×14 ablations, free-query row 4, stop-grad variant.
- 2026-10-29 (later): **PREREG_V5 §3 SIGNED** (PASS rule as written, R5c not run, cubic `p` on CAP-dev) + §3.6
  (printed reads). Added: `v5_maps` STAUC of uniform / centre / change maps (`baseline_staucs`, `uniform_tarr`),
  `v5_tile_read` subsets (ego, box share), AP, maps table, mechanics; fix: `v5_stage0.todo` used the 2-D-only
  `is_complete`, so a resumed DoTA-CAP extraction redid every finished 4-D/5-D grid (`grid_complete` moved to
  `v5_stage0`). **Phase 2 notebook** `colab/v5/v51_tile_pilot.ipynb` (parallel GPU jobs, VM-disk grids mirrored to
  Drive), doc `core/docs/v5/V51_TILE_SETUP.md`. Next: the user runs the notebook, pastes `v51_pilot_readout.md`.

