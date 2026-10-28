# KAT-VAD v5 — LaGoVAD-MQR, staged (plan B)

Proposal: `core/docs/final/Thesis proposal Deep learning-based traffic video anomaly detection.md`.
Rules: `core/docs/v5/PREREG_V5.md`. User decisions 2026-10-26: plan B (staged, early kill), **Nexar-only** training,
**3 seeds** first, proposal **not** edited until feasibility is known, CRN dropped (v2 §29–§31 record).

## 0. Status

| stage | what | state |
|---|---|---|
| 0a | DoTA box annotations local (`data/DoTA/raw_annotations/`), coverage on CAP-dev | done 2026-10-26: 534 / 569 clips boxed, 80 % of window frames |
| 0b | §1 S0-1…S0-12 signed | done 2026-10-26 |
| 0c | tool `core/tools/v5_stage0.py` (+ `forward_tokens`, tests) + notebook `colab/v5/stage0_dota_cap_tokens.ipynb` | built |
| 0d | run (Colab: one CAP-DATA stream, V7 grids 1,129 clips + dev change maps) | done 2026-10-26 |
| 0e | readout → GO-7 / GO-14 / PARTIAL / KILL | **INVALID** 2026-10-26: 15 dev clips failed `g1_pixel` (gate re-taken on `s1[::3]`, tool defect). On the 554 valid: G-A ✓ G-B ✗ G-C ✓ on V7 and V14 (≈ PARTIAL); CLIP patches point (+0.29), VideoMAE does not (+0.002). `core/docs/v5/RESULTS_V5_STAGE0.md` |
| 0f | Amendment 2 (§1.2) + check L (§1.3, lag × centre shift; L-3 picks Stage 1's window): re-extract the 15, re-read, lagged/shifted V7 change; `colab/v5/stage0b_amend2_lag.ipynb` | **done 2026-10-27: PARTIAL** (569/569 valid; G-A ✓ G-B ✗ G-C ✓, V7 and V14). Check L: **L-2 fires** (ℓ = 6, no shared frame: hit − chance +0.076…+0.106, non-ego `e` +0.17…+0.22) → pointing failure = window overlap; **L-3 = causal**. Lagged change times the anomaly badly (top-k macro 0.391 at ℓ = 6). `RESULTS_V5_STAGE0.md` §3–§5 |
| 0g | user decision after PARTIAL: go to Stage 1/2 under a new pre-registration (option a′) | done 2026-10-27 |
| 1 | MQR module (content-keyed + change-keyed) + sMIL loss; Nexar token-grid extraction; STAUC/pointing tool; gates S1-G1…S1-G6 (`PREREG_V5.md` §2) | **§2 SIGNED 2026-10-27** (R7 not run; λ 0.5 / M 8 fixed; `span`), §2.4 build notes B1–B7. **BUILT 2026-10-27**: `core/models/mqr.py`, `core/losses/smil.py`, `core/data/v5_grids.py`, `core/tools/v5_stage1.py` (extract/head/init-check/provenance/g3/collapse), `core/tools/v5_maps.py` (STAUC port, dump/read/g5), trainer/eval wiring (`--grid-dir`, `--grid-s1-dir`), 40 tests; suite 1292 / 0 fail. S1-G5(c) passes locally (Δ 0). Notebook `colab/v5/stage1_nexar_grids_gates.ipynb` (S1-G1…G5 + R1 s2024). **Not run.** Stage 2 notebook (R3/R5/R5d × 3, reads S2-7/S2-8) after Stage 1 passes |
| 2 | Nexar-only rows R0/R1 (borrowed from v2 Nexar A0/A2) + R3, R5, R5Δ (+ R7 CLIP patches, if the user approves its cache) × 3 seeds; DoTA-CAP-dev protocol B; pointing/STAUC on CAP-dev boxes; decision table S2-9 | **BUILT 2026-10-28, not run**: `core/tools/v5_stage2.py` (S2-7/S2-9/S2-10 + S2-14 prints), PREREG §2.4 B10, tests 14, notebook `colab/v5/stage2_train_read.ipynb`; R0 printed only, R1 from Stage 1 (B9), R7 not run. **RUN 2026-10-29 (s2024, B11): PRELIMINARY NOT-ADOPTED (S2-9)**; queries never left the mean (entropy = log 49) → readout untested; `core/docs/v5/RESULTS_V5_STAGE2.md`. Next: direction B (user) |
| 3 | MSAD / TAD / CCD protocols | only if stage 2 is positive |

## 1. Why staged

- MQR's premise (local tokens carry evidence the mean loses) is unmeasured; Stage 0 tests it for ~1–2 GPU-hours plus
  one CAP-DATA stream, before any training or Nexar re-extraction.
- Known traps in the proposal's own protocols: TAD in-domain collapse (C14); CCD normals from BDD100K (C38, shortcut
  1.000 measured on D2City/BDD-A); DoTA pixels only via DoTA-CAP (1,129 / 1,397, D14 forbids 62.60 beside it);
  position prior `p_T2` 0.822 > every v2 model on CAP-dev; V2-S is not temporal-order-sensitive (D6).
  These go into the proposal **after** Stage 0, if it is GO.

## 2. Cost of Stage 0

- GPU: V2-S on ≈ 37 k windows (1,129 clips at stride 3) + CLIP ViT-B/16 on ≈ 19 k dev frames — tens of minutes.
- I/O dominates: CAP-DATA tar groups on Drive (267 GB total) streamed once; same path as `colab/v2/dota_cap_videomae.ipynb`.
- Disk: V7 grids ≈ 37 k × 37.6 KB ≈ 1.4 GB (Drive); dev change maps ≈ 20 MB (`outputs/v5/v5_stage0/maps/`).

## 3. Risks

| risk | effect | mitigation |
|---|---|---|
| CAP frame geometry ≠ 1280×720 aspect | boxes misplaced | G2 per clip |
| token extraction differs from v2's cache | every read invalid | G0 per step (cos ≥ 0.9999 vs `DoTA_CAP_s1_squash`) |
| ego-motion makes boxed tiles change more anyway | false GO | `ρ^pre` with the same fixed mask (S0-6), `h^pre` (S0-7) |
| box share ⟂ ego confounded | bin reads uninterpretable | disclosed (S0-12), printed only |
| GO but training gains nothing | Stage 2 null | rows 1/3/5 only first, 3 seeds |
