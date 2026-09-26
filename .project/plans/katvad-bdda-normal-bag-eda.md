# BDD-A as a negative-bag pool for T2 (DADA-2000 original) — EDA plan

**Status:** **CLOSED 2026-09-26 — NO-GO** (G-X FAIL on `calm`, Δ(X−R0) −0.127; shortcut 1.000).
Write-up: `core/docs/BDDA_EDA.md`. Planned 2026-09-25 (branch `main`). Runbook: `colab/BDDA/eda_normal_bags.ipynb`.
Template: the D2City EDA (`.project/plans/katvad-d2city-normal-bag-eda.md`,
`core/docs/D2CITY_EDA.md`) — same probes, same bars, same DADA side. No `core/` change,
no model trained.

## 1. Question

Can BDD-A clips be **added to T2 as negative bags** (T2 = W=20 hop 8 windows cut from the
DADA-2000 original accident videos, in-video negatives)? As with D2City, the question is
not "do they look alike" but **whether BDD-A negatives teach a linear reader of frozen
CLIP the accident or the source** (C38).

**Prior (stated before the run):** NO-GO is the expected outcome. D2City (China, the same
country as DADA) failed G-X at Δ −0.090 with shortcut 1.000, and DADA vs DoTA separates at
0.9999. BDD-A is US footage (Berkeley / SF). The EDA exists because it is cheap: 3.1 GB
already on disk, one feature pass, no training.

## 2. What the local copy is (measured 2026-09-25, `data/BDDA/training/`)

- **BDD-A = Berkeley DeepDrive *Attention*** (Xia et al., 2018), **not** BDD100K. It was
  collected around **braking events**. That makes it a near-miss-enriched pool, not a
  normal-driving one.
- 926 `camera_videos/*.mp4` + 926 `gps_jsons/*.json` (the BDD-A training split).
  `922.mp4` has no moov atom and cannot be decoded (1/926).
- 1280×720 on 925/925. fps: **89.3 % ≈ 30** (29.97/30), **10.6 % ≈ 60**.
  Duration: p5–p75 = 10.0 s, p95 14 s, max 32 s; 88.8 % fall in 9.5–10.5 s.
- GPS: 1 Hz, `{timestamp ms, speed m/s, course, lat, lon, accuracy}`. 821 clips have
  ≥ 1 consecutive pair with dt ∈ [0.5, 1.5] s. The other 97 have no such pair and 8 have
  fewer than 2 samples. Minimum acceleration over consecutive pairs: p50 −2.59 m/s²,
  p10 −4.73. **Calm (≥ −3 m/s²) = 509 clips**, or 62 % of the 821 that have usable GPS.
  Mean speed p50 5.5 m/s (urban, slow).
- The validation split (200 clips + GPS) is **not needed** unless G-I's calm floor fails.
  The notebook already reads `data/BDDA/validation/` if it is present, with ids prefixed by
  split (C26).

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Per-clip stride** = `round(fps × 8/30)` → 8 @ 30 fps, 16 @ 60 fps; step error ≤ 5 % or the clip is dropped with a count | One time step for DADA (A1: 30 fps) and every BDD-A clip. 60 fps at stride 8 would halve the step |
| D2 | **One field-of-view arm (V0, `_ncc`)**. No band-crop arm | The V1 crop moved nothing on D2City, and C38 says a crop cannot remove a source cue. Adding it here would be arm-shopping (lesson 14) |
| D3 | **Two GPS arms:** `all` (every decodable clip) and `calm` (GPS-known, min accel ≥ **−3.0 m/s²** ≈ 0.3 g) | BDD-A is braking-enriched, so `all` mixes near-misses into "normal". **The gate is read on `calm`.** `all` is descriptive |
| D4 | **Length construction L-T2** (chosen): BDD-A cut into T2's own W=20 hop 8 windows; lever table against **T2 test windows** | The target corpus is T2, whose bags are all length 20. L-raw (whole clip against a whole DADA source) is printed for contrast |
| D5 | Probes use **whole clips** as negatives (grouped by clip) | A frame probe does not see bag length. Overlapping windows would count frames twice |
| D6 | R0 computed **once** and shared by both arms | R0 never reads BDD-A. With the same DADA folds and seed it must **reproduce D2City's R0 = 0.676258** — a free pipeline check |

## 4. Pre-registered gates (edit nothing after the first run)

| gate | bar |
|---|---|
| **G-I** (HARD) | decodable ≥ 99 % · step-matched ≥ 99 % · duration ≥ one T2 window (20 × 0.267 s = 5.33 s) on ≥ 99 % · GPS-known ≥ 80 % · calm clips ≥ **300** (below → add the validation split) |
| **G-L** (HARD) | L-T2 clip-level length AUC at mix 1.0 ∈ [0.45, 0.55] |
| sanity | R0 `auc_macro` ∈ [0.62, 0.68]; soft check: \|R0 − 0.676258\| < 1e-4 (D2City run); soft check: T2 mix-0 oracle ≈ 0.7037 |
| **G-X** (on `calm`) | PASS: X ≥ 0.60 and Δ(X−R0) ≥ −0.03 · FAIL: X < 0.55 or Δ < −0.06 · else MARGINAL |
| **G-M** (on `calm`) | Δ(M−R0) ≥ −0.01 |
| descriptive | S (DADA pre-accident vs BDD-A), S-ref (vs DoTA), shortcut AUC (red flag > 0.90), the L-T2 oracle rise, per-second autocorrelation |

Verdict table: R0 outside sanity → PIPELINE BUG · G-X FAIL → NO-GO · G-M fail → NO-GO
(additive) · G-X MARGINAL → GO with in-video negatives only (M-style) · otherwise → GO,
and write the corpus-build plan.

## 5. Costs if it passes (out of scope here)

RAFT `e_O` for BDD-A (C13), a new z-score cache (`train_ids_sha1` changes), and the T2 builder
learning to read a second, negative-only source. None of this starts before the numbers
exist.

## Appendix A — results

Run on Colab GPU, read out 2026-09-26 (`outputs/EDA/BDDA/`; durable record `core/docs/BDDA_EDA.md`).

| gate | measured | |
|---|---|---|
| G-I | 925/926 decodable · step-matched 99.9 % · ≥ 1 window 99.9 % · GPS-known 88.7 % · calm **508** | PASS |
| G-L | L-T2 length AUC at mix 1.0 **0.5000** (1,588 BDD-A windows) | PASS |
| sanity | R0 **0.676227** (D2City 0.676258, \|Δ\| 3.1e-5) · T2 mix-0 oracle 0.7037 | PASS |
| **G-X (calm)** | X **0.5492**, Δ(X−R0) **−0.1270 [−0.1532, −0.1008]**, 5/5 folds < −0.06 | **FAIL** |
| G-M (calm) | Δ(M−R0) −0.0029 [−0.0062, +0.0004] | pass (≈ 0, no gain) |
| descriptive | shortcut R0 / X / M = 0.374 / **1.000** / 0.999 · S 1.0000 vs S-ref 0.9999 · L-T2 oracle 0.7037 → **0.8813** at mix 1.0 · autocorr lag 1: BDD-A 0.977 vs DADA 0.967 | |
| `all` arm | X 0.5375, Δ −0.1387 [−0.1535, −0.1240]; M Δ −0.0048 | descriptive |

**Row taken: G-X FAIL → NO-GO.** T2 stays; Option A remains the live track. No further BDD
downloads (the §2 condition was a pass). R0 put BDD-A *above* DADA normals in 63 % of pairs
before training (shortcut 0.374, the reverse of D2City's 0.820); once BDD-A was used as the
negatives the probe separated it perfectly (1.000) — different start, same end (C38).
