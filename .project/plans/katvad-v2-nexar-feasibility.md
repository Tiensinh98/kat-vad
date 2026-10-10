# KAT-VAD v2 on Nexar Collision Prediction — feasibility + training plan

> Status: **DRAFT, not pre-registered, nothing built.** Branch `v2`. Written 2026-10-09.
> Source: <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction>,
> paper arXiv 2503.03848 (CVPRW 2025). Field names below come from the dataset card and must be
> re-checked against the real files in N0 before any code reads them.

## 0. Status

| Item | State |
|---|---|
| Scope (user, 2026-10-09) | **N3 → train (R-B then R-A)**; negative videos **in scope behind N-X**; `HF_TOKEN` ready |
| Repo listing (2026-10-09) | `train/{positive,negative}/` 750 + 750 mp4 + `metadata.csv` each; `test-public/`, `test-private/`; **`solution.csv` is public** (official test labels, clip-level anticipation — still not a detection set) |
| Raw storage | **changed:** mirror the whole repo to `Thesis/data/Nexar/raw/` (~31 GB) instead of deleting mp4 after features — DoTA's pixels were lost with their links |
| N0 code | built 2026-10-09 (uncommitted): `core/data/nexar.py`, `core/tools/nexar_census.py`, `core/tools/nexar_splits.py`, `core/tests/test_nexar.py` (15), `colab/v2/nexar_n0_census.ipynb`, `core/docs/v2/NEXAR_SETUP.md`; `nexar_test` added to `V2_SEALED_SPLITS` |
| N0 census | **RUN 2026-10-09** — 1,500 videos, 0 issues, length AUC 0.537, **no collision/near-miss column**, `t_event/duration` 730/750 in [0.4, 0.6) (`NEXAR_SETUP.md` §1.1) |
| N0 split | **FROZEN 2026-10-10** — 900 / 225 / 375 (`nexar_test` sealed), `core/splits/v2/NEXAR_MANIFEST.json`, `--check` passes |
| Scope change (user, 2026-10-10) | train **two constructions**: `whole` (uncut videos, both pools) vs `window` (T2 windows, positives only), arms A0/A3 |
| N1 | addendum **§21 DRAFT** (D-N1…D-N10) — val reads may run, `nexar_test` needs the signature |
| N2–N5 code | built 2026-10-10 (uncommitted): `nexar_extract`, `nexar_build`, `nexar_eval`, `build_v2_inputs fit-ids`, `test_nexar_pipeline.py` (16), `colab/v2/nexar_train.ipynb` |
| Whole first (user, 2026-10-10) | train `whole` only; `window` iff the pilot trigger fires (D-N10: W1 edge drop > 0.05 · W2 ≤ E3-A3 zero-shot · W3 clip AUC ≥ 0.95 and A3 < A0). D-N9 split into (a) whole-only: `whole`/A3 − `whole`/A0, (b) both. Built: `nexar_trigger` (+5 tests), notebook step 5b (zero-shot) + trigger in step 7 |
| Next | user uploads → runs `nexar_train.ipynb` pilot (s2099, `whole` × {A0, A3}, 2 runs + E3 zero-shot) → pastes `summary_pilot.md` (trigger verdict inside) |

## 1. Summary

- Nexar = **1,500 labelled US dashcam videos** (750 positive = 400 collision + 350 near-miss; 750 negative),
  ~40 s, 1280×720, ~30 fps, ~1.70 M frames, 31.4 GB, gated HF repo, Nexar Open Data License.
- Positives carry `time_of_alert` and `time_of_event` (mean alert→event 1.6 s, max 4.5 s). **No frame spans,
  no end-of-anomaly time.** The official test set (1,344 × 10 s clips) **ends 0.5–1.5 s before the event** — it is
  an *anticipation* task, not detection, and its labels are not known to be public.
- **Verdict: feasible, with three non-negotiable changes to the "just train on it" idea** (§2.3): (1) only the 1,500
  train videos are usable, so we freeze our own split; (2) the frame-level span must be *defined* and pre-registered;
  (3) the 750 negative videos are a separate pool and must pass a G-X gate (C38) before they are used as negatives.
- **Recommended order:** first use Nexar as a **fresh sealed endpoint** for the existing 20 T2-trained v2 checkpoints
  (no training, answers the "DoTA-CAP is already opened" problem of REPORT §8), **then** train A0/A3 on Nexar.

## 2. Assumptions & Constraints

### 2.1 Facts (from card + paper)

| Item | Value | Consequence for v2 |
|---|---|---|
| fps | ~30 (paper: "~30") | Matches T2 assumption A1; VideoMAE 16×3 = 1.5 s holds. **Verify per video** — some dashcam models may differ |
| Length | bimodal, mostly ~40 s; "depends on dashcam model" | Length leak risk (C28) if the modes differ by class → measure in N0 |
| Recording trigger | IMU hard brake / sudden accel, **for negatives too** | Negatives are hard (braking without event) — good; but positives may share a braking signature at mid-clip |
| Event position | "typically near the middle" | **Position prior will be huge in-domain** on whole videos (REPORT §6.2 again). Long context makes H1-style shifted crops possible |
| Ego involvement | all ego, car/truck, front impact | Narrower than DoTA (REPORT §6.4: non-ego is v2's weak class). Nexar-only training will not fix non-ego |
| Categories | collision / near-miss + weather, light, scene | No DADA/DoTA taxonomy → `H_mul` needs a new definition set |
| Test set | 10 s, ends before event, labels unknown | **Unusable for frame-level detection.** Out of scope |
| Pixels | full mp4 | Unlike DoTA, every extractor (CLIP, V2-S, crops) is possible |

### 2.2 Project constraints that bind

- Real-data runs ship as Colab notebooks; Drive = `Thesis/` (read-only data) + `Thesis-V2/` (code + v2 outputs).
- Lesson 14: no tuning on a Δ. Every decision rule is committed **before** the number is read (pending (aw)).
- C2: a cache is bound to its transform + stride (`_ncc` CLIP, squash V2-S).
- C28/C33/C35: windows close the length leak; the clip oracle follows the frame share; sweep geometry on the real
  census, never predict it from prose.
- C38: an imported normal pool is admitted by a transfer probe (G-X), not by similarity.
- DoTA-CAP-eval / DoTA-eval / T2-test are **opened**. Any Nexar-trained read on them is a "second look".

### 2.3 What "training v2 on Nexar" can and cannot mean

| Role | Clean? | Cost | Value |
|---|---|---|---|
| **R-B: Nexar-test as a new zero-shot endpoint for the T2-trained 20 ckpts** | **Yes — sealed, never seen** | Extraction + forward passes | Highest for the thesis: a fresh read of A0–A3 and `p_T2`, plus H1 for free |
| R-A: Nexar as a training corpus (A0_N, A3_N), eval in-domain on Nexar-test | Yes, if split frozen first | + 10 training runs | Answers "does v2 work on a US ego-dashcam corpus" |
| R-A′: Nexar-trained → DoTA-CAP / T2-test transfer | **No** — opened sets | forward only | Descriptive, labelled "second look" |
| R-C: T2 + Nexar mixed training | Risky (C38 source shortcut 16:9 vs 2.40:1) | + G-X gate | Do not plan now |
| R-D: Kaggle anticipation mAP | Off-task (v2 is detection, encoder non-causal) | — | Out of scope |

## 3. Plan Metadata

- **Plan type:** new corpus onboarding + experiment (feasibility → pre-registered campaign)
- **Size / scope:** Medium (≈ 5–6 new modules/tools, 3–4 notebooks, ~10 GPU training runs)
- **Estimated duration:** ~2–2.5 weeks of wall time (Colab-bound)
- **Storage path:** `@.project/plans/katvad-v2-nexar-feasibility.md`

## 4. Phases Overview

| Phase | Goal | Est. | Depends on |
|---|---|---|---|
| N0 – Access + census | Get the data, measure it, freeze the split | 1 d | user accepts HF license |
| N1 – Pre-registration | Span rule, gates, arms, decision rules — committed before any score | 0.5–1 d | N0 census (no labels on scores) |
| N2 – Ingest + features | CLIP s1 `_ncc`, V2-S at s8 and s3, census JSON | 2–3 d | N0 |
| N3 – Zero-shot read (R-B) | T2-trained A0–A3 + `p_T2` on Nexar-test, + H1 shifted crops | 1 d | N1, N2 |
| N4 – Corpus build + probe gates | Windows, N-L / N-D0 / N-X / N-W / K-N gates | 2 d | N2 |
| N5 – Training (R-A) | Pilot 1 seed, then A0_N / A3_N × 5 seeds | 3–5 d | N4 all PASS |
| N6 – Read-out + docs | Results doc, memory bank, lessons | 1 d | N3, N5 |

## 5. Detailed Tasks by Phase

### Phase N0 – Access + census

- **Goal:** know exactly what is on disk before designing anything on it.
- **Tasks:**
  - [ ] User accepts the gated license on HF; add `HF_TOKEN` as a Colab secret (never committed).
  - [ ] List the repo files (`train/`, metadata csv/jsonl, LICENSE). Record the real field names and the event-type
        field (collision vs near-miss). Confirm whether `test/` labels exist (expected: no).
  - [ ] Census notebook (`colab/v2/nexar_n0_census.ipynb`), per video: duration, frame count, **fps**, resolution,
        codec, label, event type, `time_of_alert`, `time_of_event`, light/weather/scene. Metadata-only where possible
        (ffprobe on streamed files), no feature extraction yet.
  - [ ] Print, do not decide on: length AUC by class (C28), `time_of_event / duration` histogram by class,
        alert→event gap, fps histogram, dashcam-model proxy (resolution/codec) × label.
  - [ ] Freeze the split **from metadata only**: `core/splits/v2/nexar_{train,val,test}.txt`, stratified on
        label × event type × length mode, by video id (no source overlap is possible — one id = one recording).
        Proposed 60 / 15 / 25 (900 / 225 / 375). Commit with `train_ids_sha1` before N2.
- **Deliverables:** `core/docs/v2/NEXAR_SETUP.md` §1 (census), frozen split files, census JSON in
  `outputs/v2/REPORTS/nexar_n0/` (tracked).

### Phase N1 – Pre-registration (new addendum section, next free number after §20)

- **Goal:** fix every rule before any Nexar score exists (pending (aw): rule in git before the number).
- **Tasks:**
  - [ ] **Frame span rule.** Primary: abnormal = `[t_alert, t_event + Δ_post]`. Pick `Δ_post` by a rule *not* read
        from Nexar scores — derive it from how DADA's/DoTA's span ends relative to the accident frame (read their
        annotation convention, T2-train median). Print Δ_post ∈ {0, 1 s, 2 s} as a sensitivity row, never adopt from it.
  - [ ] **Eval protocol N-B** (mirror of E1 protocol B): CLIP `s1[::3]`, whole video, scores interpolated to native
        frames, macro over two-class videos, bootstrap by video. Micro + clip oracle printed beside (C12 mirror).
        Negatives contribute to micro / clip-level only.
  - [ ] **Position prints (mandatory):** `p_T2` (frozen T2 fit), a cubic `p_N` fitted on Nexar-train, `t/N`.
  - [ ] **H1 on Nexar (pending (ay) design guard):** from each test positive, cut fixed-length crops (e.g. 10 s) with
        the event centre at {0.1, 0.3, 0.5, 0.7, 0.9}, plus the same-length crop at its original relative position as
        control; CRN re-baked per crop; `p_T2` frozen. Kill criterion as REPORT §8 H1.
  - [ ] **Gates for training (N4)** — thresholds written now: N-L (length AUC on windows = 0.5000), N-D0 (frozen-CLIP
        frame probe macro on Nexar-val ≥ 0.60), N-X (§N4), N-W (retention ≥ 0.95, class ratio band, oracle printed),
        K-N (`u` beyond `[x;p]` on Nexar, printed).
  - [ ] **Training arms + rule:** A0_N, A3_N (CRN R2 + V2-S, stats fitted on Nexar-train − Nexar-val), 5 seeds,
        LaGoVAD losses minus L_neg (= the E3 recipe). Primary contrast: A3_N − A0_N macro on Nexar-test, paired t95
        over seeds; adoption sentence + the `f_2` beyond-position read pre-written. Secondary (descriptive):
        A3_N − A3_T2 (zero-shot) on Nexar-test.
  - [ ] **Collapse guard:** O1′ (Amendment 8) on Nexar-val windows.
  - [ ] **Definition set for `H_mul`:** new `nexar` verbalizer key with 2 classes (ego collision with car/truck,
        ego near-collision) — wording written now, never edited after a read.
- **Deliverables:** `core/docs/v2/PREREG_ADDENDUM.md` new section (signed by the user), this plan's §0 table.

### Phase N2 – Ingest + features

- **Goal:** Nexar CLIP + V2-S caches on Drive, bound to the v2 transforms, read from the N0 mirror.
- **Tasks:**
  - [ ] Ingest tool (`core/tools/nexar_ingest.py`, argparse): per video from `Thesis/data/Nexar/raw/` → decode once with PyAV
        (`core/data/video_io.py`) → CLIP `_ncc` at **s1** (any stride derivable later, as `DoTA_s1_ncc`) → V2-S squash
        at **s8** (training rows) and **s3** (eval rows). Resumable via `feature_cache`, census JSON,
        manifest per cache (C2). Reuse `extract_clip_features` / `extract_video_features` code paths, do not fork
        the transforms.
  - [ ] Decide whether `extract_video_features` can read mp4 directly or needs a frame-dir; prefer reading decoded
        arrays to avoid writing 1.7 M PNGs.
  - [ ] Compute check: ~1.7 M decoded frames; CLIP s1 ≈ 1.7 M images; V2-S ≈ 212 K (s8) + 567 K (s3) 16-frame clips.
        Expected a few GPU-hours on A100, multi-session on T4 → notebook must resume by video.
  - [ ] Storage: CLIP s1 fp32 ≈ 3.5 GB; V2-S ≈ 1.2 GB — fits `Thesis/cache/`.
  - [ ] Regression gates: `s1[::8]` == an independently extracted s8 sample (J9′ analogue); V2-S rows == CLIP rows.
- **Deliverables:** `cache/clip/Nexar_s1_ncc/`, `cache/video/<V2-S>/Nexar_s{3,8}_squash/`, `colab/v2/nexar_n2_extract.ipynb`.

### Phase N3 – Zero-shot read of the T2-trained v2 (R-B)

- **Goal:** a fresh, sealed transfer endpoint for A0–A3 — the thing REPORT §8 says v2 lacks.
- **Tasks:**
  - [ ] `build_v2_inputs apply` with the **T2** statistics (nothing refitted) → `cache/v2/A{1,3}_R2/Nexar_s3`.
  - [ ] Generalize `protocol_b_eval` / `position_prior` / `final_readout` to a non-DoTA corpus (a `--corpus`
        adapter, not a copy) with `nexar_test` sealed like `dota_eval`.
  - [ ] Read once: macro A0–A3, contrasts A3 − A0 / A3 − A1, `p_T2`, `f_2`, ego-only by construction, collision vs
        near-miss split, H1 shifted crops.
- **Deliverables:** `outputs/v2/REPORTS/nexar_zeroshot/`, `core/docs/v2/RESULTS_NEXAR.md` §1.

### Phase N4 – Corpus build + probe gates

- **Goal:** a Nexar training corpus that cannot leak, decided by gates, not by looks.
- **Tasks:**
  - [ ] `core/data/nexar.py` adapter → records reusing `dada.DadaRecord`-style fields, `plan_record_windows`,
        `window_label`, `write_windowed`, `check_definition_coverage` (no fork of the window code).
  - [ ] **Construction N-T2 (default):** T2-style windows at s8 cut from the **positive videos only** (pre-alert
        windows = in-video negatives). Sweep W/hop and per-video caps on the real census (C33/C35): pre-alert context
        is ~18 s ⇒ windows per positive video ≈ 15 vs ~2 abnormal → class ratio needs a cap.
  - [ ] **N-X gate for the 750 negative videos** (C38 protocol, same probe/folds as D2City/BDD-A): R0 = in-video
        negatives, X = negative videos only, M = both. Admit only if Δ(X − R0) within 0.03 and shortcut AUC well below
        1.0. Same fleet/camera makes a pass plausible — but 0_Normal_Driving was "same dataset" too and leaked.
  - [ ] N-L, N-D0, N-W, K-N as pre-registered.
  - [ ] `build_v2_inputs fit` generalized to fit CRN / `c` / `σ_u` on Nexar-train − Nexar-val (today it is T2-only).
- **Deliverables:** `data/Nexar_v2/` dataset dir, `outputs/v2/REPORTS/nexar_n4/` gate read-outs.

### Phase N5 – Training

- **Goal:** A0_N vs A3_N, 5 seeds, the E3 recipe.
- **Tasks:**
  - [ ] Pilot: A0_N + A3_N, 1 seed → O1′ guardrail, J10 (ckpt step == last metrics step), `ρ_u` printed.
  - [ ] Check `score_head_kernel` / `mil_topk_pct` against Nexar window length (lesson: kernel spanning the clip =
        clip classifier). With W=20 they carry over from T2.
  - [ ] Full: 10 runs (`colab/v2/nexar_n5_train.ipynb`, resumable, sync ckpts only when final).
- **Deliverables:** `outputs/v2/nexar_*/`, run manifests (cache dirs, `--flow-dir` n/a, split sha1).

### Phase N6 – Read-out + docs

- **Tasks:**
  - [ ] Open `nexar_test` once (in-domain A0_N/A3_N + zero-shot A3_T2), write `RESULTS_NEXAR.md`.
  - [ ] Descriptive "second look" transfer of A3_N on DoTA-CAP-dev/eval and T2-test, labelled as such.
  - [ ] Memory bank (activeContext, progress), REPORT_V2_RESULTS §8 update, lesson candidates.
  - [ ] Tests for every new module (adapter, ingest, corpus build, generalized eval tools); suite green.

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Event near clip middle → position prior wins in-domain (as `p_T2` 0.82 > A3 on DoTA) | Headline number is position | Windows for training; `p_T2`/`p_N`/`f_2` printed beside every macro; H1 shifted crops are the real test |
| Span undefined (no end time) | Frame metric depends on an arbitrary Δ_post | Rule fixed in N1 from DADA/DoTA convention; sensitivity printed only |
| Length modes differ by class (dashcam model) | C28 leak on whole videos | Windows (leak 0.5000 by construction); length AUC printed on whole videos |
| Negative videos separable by source/trigger (C38) | MIL learns "which video", not "what" | N-X gate; default construction uses in-video negatives only |
| Ego/car-only, US-only, 2 classes | Nexar-trained model weaker on non-ego DoTA | Claim scope = "ego collision, US dashcam"; no cross-corpus mixing |
| fps not exactly 30 on some cameras | V2-S 1.5 s geometry and s3 = 0.1 s drift | Census fps; resample-by-time or exclude, rule fixed in N1 |
| Gated license / ToS | Cannot redistribute | Raw mirror stays private on Drive, attribute in thesis, no re-identification |
| Colab disconnects in a 1.7 M-frame extraction | Lost days | Per-video resume, atomic writes (C11), census as progress ledger |
| Selection optimism from reusing Nexar-val | Inflated in-domain number | Nexar-test sealed; val only for O1′ and gates |
| Pending (aw): rule written after the number | Read-out not pre-registered | Commit addendum + gates before running N3/N4 notebooks |

## 7. Next Steps for the User

1. Accept the license on the HF page and add `HF_TOKEN` to Colab secrets.
2. Choose scope: **R-B only** (zero-shot endpoint, ~1 week) or **R-B → R-A** (full training, ~2–2.5 weeks). Recommended: R-B → R-A.
3. Say whether the 750 negative videos are in scope (needs the N-X gate) or in-video negatives only.
4. Give the go for N0: Claude writes `nexar_n0_census.ipynb` + split freezer; you run it and paste the census read-out.
5. After the census: sign the N1 addendum before any feature is scored.
