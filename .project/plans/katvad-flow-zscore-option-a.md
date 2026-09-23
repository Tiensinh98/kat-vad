# KAT-VAD — Option A: z-scored flow target (`flow/v2_zscore`)

**Status:** **P1 DONE 2026-09-23** (code + tests + docs, uncommitted). P0/P2–P4
pending (Colab). **Authorized** by the user 2026-09-23 (construction **A1** chosen).
**Branch:** `main` (KAT-VAD v1). **Corpus:** `DADA2000_orig` T2 (W=20 hop 8).
**Parent:** `.project/plans/katvad-kip-loss-scale-diagnosis.md` §5 **A**, Appendix A.3.
**Fires:** **C2** (new flow cache → every KIP-on number re-measured). **Never
overwrites `flow/v1`.**

> Every bar in §6 is written **before** anything is built or trained. Where a bar
> has a condition, the condition is in the bar (C33). No bar is tuned after the
> fact, and `lambda_rec` is **derived, never swept** (lesson 14).

---

## 1. What is being fixed, in one paragraph

`L_KIP_rec` is a bare MSE against `e_O = s @ M`, where `s` is 23 **raw**
frame-global flow statistics (pixel units) and `M` a fixed seeded `N(0, 1/23)`
map to 256-d. D1 measured the consequence on T2: the global-mean predictor scores
`V = 31.642`, `mag_max` alone carries **83.0 %** of `E[s²]`, so at
`lambda_rec = 1.0` the term is ~32× the rest of the objective, and D2 measured it
**capturing the trunk orthogonally** (`rho` 3.105, `cos` −0.001). A1 standardizes
each of the 23 statistics with **train-split** moments, then re-applies the
**same** `M`. The loss returns to O(1), every statistic gets equal weight, and
the target keeps its 23-dimensional content.

## 2. Nothing needs frames or RAFT

`cache/flow/v1/DADA2000_orig/` already holds, per source video:

| file | shape | role in v2 |
|---|---|---|
| `{id}.stats.npy` | `(L, 23)` raw stats | **input** to the z-score; **copied byte-identical** into v2 |
| `{id}.npy` | `(L, 256)` `e_O` | used only for the integrity gate G0 (`npy ≈ stats @ M`) |
| `../flow_projection.npz` | `(23, 256)` | loaded via `load_projection` (fails loud); **copied** into v2 |

`.stats.npy` stays **raw** in v2 because `core/data/knn_cache.py:motion_descriptor`
reads raw magnitudes and angle histograms from it. The only consumer of `{id}.npy`
is `kip_reconstruction_loss` (`core/train.py:256,402`); `L_KIP_align`
L2-normalizes its projections (`core/kip/losses.py`) and `L_kin` never reads `e_O`.

## 3. Decisions (debate done; one line each)

| # | Question | Options | Decision | Why |
|---|---|---|---|---|
| D-1 | Where to standardize | **A1** 23 raw stats, then re-project · A2 the 256-d `e_O` | **A1** (user, 2026-09-23) | A2 fixes the units but every output dim is still a mixture dominated by `mag_max`; A1 fixes the content too |
| D-2 | Population for `μ, σ` | train **windows** via `FeatureSlicer` · unique frames of train source videos | **train windows** | It is exactly the population `L_KIP_rec` averages over, and the one D1's `V` was measured on (4,401 windows, 88,020 rows). Overlap weighting is then correct by construction |
| D-3 | Where the tool lives | `core/flow/zscore_cache.py` · a flag on `raft_extract` | **new module** `core/flow/zscore_cache.py` | `raft_extract` needs torch + RAFT; this is numpy-only and runs on CPU in minutes. Separate CLI, no RAFT import |
| D-4 | Moment code | new accumulator · reuse `core/eda/features.py:_MomentAccumulator` | **reuse** (promote to `MomentAccumulator`, keep `_MomentAccumulator` alias) | DRY; the same code that measured `V` measures `μ, σ` |
| D-5 | Cache version name | `v2` · `v2_zscore` | **`v2_zscore`**, new constant `FLOW_ZSCORE_CACHE_VERSION` | Says what changed. `FLOW_CACHE_VERSION` default stays `v1` — MSAD/DoTA/TAD are untouched; T2 runs pass `--flow-dir` explicitly |
| D-6 | Round-trip check (D1-d) on v2 | tool-only · also teach `eda.flow_stats` | **both** — `flow_stats` standardizes raw stats when `zscore_stats.npz` sits in `flow_dir` | Otherwise `eda report` on v2 prints `predicted_second_moment_from_raw` from **raw** stats against a **z-scored** target — a ~30× false alarm waiting for the next reader |
| D-7 | `σ ≈ 0` dimension | epsilon floor · raise | **raise** (`FLOW_ZSCORE_MIN_STD`) | D1 measured `dead_dims = 0`; a dead dim on rebuild means the input changed, which is a stop, not a clamp |

## 4. Deliverables (code — Phase 1) — ✅ done 2026-09-23

*As built:* the numpy-only pieces (`MomentAccumulator`, `ZScoreStats`,
`fit_zscore`, `standardize`, load/save) live in a leaf module
**`core/flow/zscore.py`**, not in `zscore_cache.py` and not in `core/eda/`:
`eda.features` imports them and `zscore_cache` imports `eda.features`, so any
other placement is an import cycle. `_MomentAccumulator` moved there as
`MomentAccumulator` (D-4). G1–G2 are scored inside the tool from
`flow_stats(dst)` and written to the manifest; the tool exits non-zero on a HARD
failure *after* writing it. §6.2's inertness test is a **NaN-poison** test
(`lambda_rec = nan` must leave `total` finite): `Trainer.compute_losses` is not
bitwise repeatable on the stub fixture, so an equality test would be flaky.

**New**
- `core/flow/zscore_cache.py` — argparse CLI, `logging`, type hints:
  ```
  python -m core.flow.zscore_cache \
      --data-dir  $KATVAD_DATA_ROOT/DADA2000_orig  --dataset DADA2000_orig \
      --src-root  $KATVAD_CACHE_ROOT/flow/v1 \
      --dst-root  $KATVAD_CACHE_ROOT/flow/v2_zscore \
      [--force]
  ```
  1. `load_dataset_files(data_dir, dataset)` → `train_ids`, `slicer`.
  2. Pass 1 (train windows only): accumulate `μ, σ` over `slicer.load(src, id, ".stats.npy")`.
     Raise on any `σ_j < FLOW_ZSCORE_MIN_STD` or non-finite row.
  3. Pass 2 (**every** source file under `src/{dataset}/`, not only train — the
     transform is per-frame and the set of files must match v1's):
     - **G0** `max|npy_v1 − stats @ M| ≤ FLOW_ZSCORE_G0_ATOL` else raise;
     - write `((stats − μ) / σ) @ M` as float32 via `feature_cache.save_array`
       (atomic, C11); copy `.stats.npy` byte-identical; resumable, skip-if-done.
  4. Write `dst/flow_projection.npz` (copy; assert equal to src), and
     `dst/{dataset}/zscore_stats.npz` = `{mean, std, n_windows, n_rows,
     src_version, train_ids_sha1, stat_names}`, plus `zscore_manifest.json`
     (counts, G0 max error, file coverage `N/M` — C10).
  5. Refuse if `dst_root == src_root`, or if `dst` exists with a different
     `train_ids_sha1` (a normalization is bound to its split, like C2's transform).
- `core/tests/test_flow_zscore.py` — CPU, data-free, synthetic cache (see §5).

**Changed** (run `trace_path` on each before editing; report blast radius)
- `core/constants.py` — `FLOW_ZSCORE_CACHE_VERSION`, `FLOW_ZSCORE_STATS_FILENAME`,
  `FLOW_ZSCORE_MANIFEST_FILENAME`, `FLOW_ZSCORE_MIN_STD`, `FLOW_ZSCORE_G0_ATOL`.
- `core/eda/features.py` — promote `_MomentAccumulator` (D-4); `flow_stats`
  standardizes raw stats for `predicted_second_moment_from_raw` when
  `zscore_stats.npz` is present, and records `"target_normalized": true`.
- Docs: `core/docs/TRAINING.md` (the flow-target contract: v1 raw vs v2 z-scored,
  how `lambda_rec` is derived), `core/docs/COLAB.md` (the v2 build command),
  `core/flow/raft_extract.py` module docstring (cache layout gains `v2_zscore`).

**Not changed:** `core/train.py`, `core/data/dataset.py`, `core/kip/*`, any config
default. `lambda_rec` is passed as an override on the T2 runs only.

## 5. Tests (must exist before Phase 2 runs)

| test | asserts |
|---|---|
| moments from train only | adding a test-split source with outlier stats leaves `μ, σ` unchanged |
| windowed moments | on a windowed synthetic corpus, `μ, σ` equal a numpy reference over the sliced rows (overlap counted) |
| transform | `npy_v2 == ((s − μ)/σ) @ M` to float32 tolerance, for every file |
| raw copy | `.stats.npy` in v2 is byte-identical to v1 |
| G0 | a v1 `.npy` perturbed off `stats @ M` raises and names the file |
| dead dim | a constant stat column raises |
| guards | `dst == src` raises; mismatched `train_ids_sha1` raises; missing projection raises |
| resume | a planted `.part` file is redone; a completed file is skipped |
| eda round-trip | `flow_stats` on a synthetic v2 reports round-trip ≈ 1 and `V ≈ Z` |

Suite stays green: 565 + new. Quality gate §11 before commit.

## 6. Pre-registered bars

### 6.1 Build gates (Phase 2 — CPU, Colab or local)

| id | quantity | bar | on fail |
|---|---|---|---|
| **G0** | per-file `max|npy_v1 − stats@M|` | ≤ `FLOW_ZSCORE_G0_ATOL` on **every** file | **HARD STOP** — v1 `.npy` and `.stats.npy` disagree; nothing downstream is valid |
| **G0-c** | coverage | v2 file count == v1 file count, 0 non-finite | HARD STOP (C10) |
| **G1** | train-window moments of the z-scored stats | `|mean_j| ≤ 1e-3`, `std_j ∈ [0.999, 1.001]` for all 23 | HARD STOP — code bug |
| **G2-a** | `V_v2` (global-mean MSE on v2, via `eda report --sections features --no-probe`) | **∈ [0.80, 1.25]** — predicted ≈ 1: `E_M[e_d²] = (1/23) Σ_j E[z_j²] = 1` | outside → the stats are strongly correlated through `M`; **report, do not re-weight**; `lambda_rec` still = `1/V_v2` |
| **G2-b** | `Z_v2 / V_v2` | ∈ [0.99, 1.01] (target is centred) | HARD STOP — centring failed |
| **G2-c** | round-trip (D1-d) on v2 | ∈ [0.9, 1.1] | **HARD STOP** (plan A.4 item 2) |
| **G2-d** | `between_item_share` on v2 | reported beside v1's **0.488**; trip-wire 0.5 | if ≥ 0.5: record; the target is then mostly *which clip*, not *which frame* |
| **G2-e** | `R²_item`-oracle `W_v2 / V_v2` | reported | — |

> **Clarification (2026-09-23, before any measurement):** on the train windows
> the standardized stats have `mean_j E[z_j²] = 1` exactly, so G2-c's round-trip
> equals `1 / Z_v2` and, with G2-b, `≈ 1 / V_v2`. **G2-c therefore binds `V_v2`
> to ≈ [0.909, 1.111]** and supersedes G2-a's wider band as the operative stop.
> On v2 the round-trip no longer tests "cache vs projection" (G0 does that per
> file, exactly); it tests how far the stats' correlations through `M` move
> `V` off 1. The bar is kept as written; if it is the only failure, the reading
> is "correlated stats", not "corrupt cache", and the user decides.

**`lambda_rec` for every v2 run = `1 / V_v2`, rounded to 4 decimals, written into
`zscore_manifest.json` and passed as `loss.lambda_rec=<value>`.** Expected ≈ 1.0.
Do not reuse **0.0316** — that is `1/V` of the **v1** cache and would switch
`L_KIP_rec` off on v2.

### 6.2 Training (Phase 3 — C2 re-measure)

KIP-on, **stage 1 and stage 2**, seeds **2024 / 2025 / 2026**, T2 config
unchanged except `--flow-dir …/flow/v2_zscore/DADA2000_orig` and
`loss.lambda_rec = 1/V_v2`. **KIP-off arms are re-used, not re-run**: with
`kip.enabled=false`, `kip_rec` is never computed and flow is never read
(`core/train.py`, `require_flow=False`) — Phase 1 adds a test asserting both
fields are inert under KIP-off, so the pairing stays one *effective* line.

### 6.3 Read-out bars (Phase 4)

| id | quantity | bar | reading |
|---|---|---|---|
| **R-1** | `R²_v2 = 1 − kip_rec / V_v2`, stage-2 end, 3 seeds | reported; **> 0** expected | ≤ 0 → PMG cannot fit the standardized target (see risk 1) |
| **R-1b** | per-block `R²` via `ê_O @ pinv(M)` (magnitude 0–6 / histogram 7–22) | reported | histogram ≪ magnitude → risk 1 is live |
| **R-2** | D2 re-run (`core.tools.grad_probe`, same protocol) `rho` at stage-2 end | **< 1.0** on every seed mean → capture **removed**; ≥ 1.0 → A did not fix capture | prediction ≈ 0.5 (target scale ÷ √V ≈ ÷5.6, weight ≈ ×1) |
| **R-3** | paired Δ T2 `auc` micro, KIP-on(v2) − KIP-off, t95 n=3 | see decision table | the pre-registered primary |
| **R-4** | paired Δ T2 `auc_macro` | reported, same table | secondary |
| **R-5** | DoTA 0-shot Δ | **reported, never decided on** (v1 rows were ~20× wider than their Δ) | — |

**Decision table (R-2 × R-3):**

| R-2 | R-3 t95 | verdict | next |
|---|---|---|---|
| < 1.0 | excludes 0, **negative** | loss scale was not the (whole) cost | **C24** (dead gate → fixed 50 % shift) is the next suspect → that work is on branch `v3`; or §5 **D** |
| < 1.0 | includes 0 | cost removed; KIP-v1 neutral on T2 | write a bounded null; stop spending seeds on v1 KIP |
| < 1.0 | excludes 0, **positive** | A repaired KIP on T2 | replicate on a second seed triple before any claim |
| ≥ 1.0 | any | A did not remove capture | re-open D2; do **not** lower `lambda_rec` off a Δ |

## 7. Risks

1. **The target becomes direction-dominated.** 16 of the 23 stats are the
   L1-normalized angle histogram, with raw σ ≈ 0.05–0.18 (D1). Equal weighting
   gives the histogram block **16/23 = 69.6 %** of the target, and on near-static
   frames those bins are the direction of noise. This is A1's price, stated now
   so it is not discovered later. Read-out **R-1b**: `M` is `(23, 256)` with rank
   23 and `e_O` lies in its row space, so `ŝ = ê_O @ pinv(M)` recovers the 23
   standardized stats exactly; report `R²` per block (magnitude dims 0–6 vs
   histogram 7–22) at stage-2 end. If R-3 comes out negative, this is a named
   suspect. **Block re-weighting is not authorized and must not be chosen off a
   Δ** (lesson 14).
2. **Normalization is corpus-bound.** `μ, σ` are T2-train statistics. Using
   `v2_zscore` for another corpus is a C2 violation; the `train_ids_sha1` guard
   enforces it per dataset directory.
3. **`lambda_rec ≈ 1.0` reads like "nothing changed".** The weight is almost the
   same number; the loss is ~30× smaller because the **target** changed.
   `TRAINING.md` says so in one sentence.
4. **Drive FUSE.** Read and write on a staged local copy (as
   `diag_kip_loss_scale.ipynb:stage()`), then `rsync` to Drive; count files after
   (C10). Atomic writes (C11).
5. **Pairing drift.** v2 KIP-on configs differ from KIP-off in three lines
   (`kip.enabled`, `lambda_rec`, flow dir). §6.2's inertness test is what keeps
   the Δ interpretable; if it cannot be written, re-run KIP-off on the same
   config file.
6. **C24 is untouched.** Every KIP-on arm on `main` remains a fixed ~50 % shift.
   A positive R-3 is attributable to the **loss target**, not to motion gating.

## 8. Phases and order

| phase | where | work | exit |
|---|---|---|---|
| **P0** | Colab | Count `.npy` / `.stats.npy` under `flow/v1/DADA2000_orig`, confirm `flow_projection.npz`, confirm `train_ids` ⊂ sources | counts printed, equal |
| **P1** ✅ | local | §4 code + §5 tests + docs; quality gate; `detect_changes`; commit | suite green — **582 collected, 0 fail** (2026-09-23) |
| **P2** | Colab (CPU) | build `v2_zscore`; `eda report --sections features --no-probe --flow-dir v2`; score §6.1; derive `lambda_rec` | G0–G2 pass; value recorded |
| **P3** | Colab (GPU) | `colab/DADA2000Origin/phase_5_zscore.ipynb`: KIP-on stage 1 + 2 × 3 seeds on v2 | 6 checkpoints + `metrics.jsonl` |
| **P4** | Colab + local | eval T2 + DoTA; `grad_probe` D2 re-run; score §6.3; update memory bank + lesson (C37 gains its outcome) | decision row written into Appendix A of this file |

**Runbook for P0 + P2–P4: `colab/DADA2000Origin/phase_5_zscore.ipynb`** (written 2026-09-23, not yet run) — §1 P0, §2 P2 + EDA cross-check + Drive persist, §3 P3, §4 the config-diff pairing check, §5–§8 P4 (R-1, R-1b, R-2, R-3–R-5 and the decision row), §9 `REPORTS/DADA2000_orig_zscore/run_manifest.json`.

P1 is the only phase that touches the repo's code. P2–P4 touch Drive and
`outputs/v1/DADA2000_orig_zscore/` only.

---

## Appendix A — Results

*(empty — filled in P2 and P4, bars above left as written)*
