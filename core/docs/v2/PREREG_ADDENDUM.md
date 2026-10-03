# KAT-VAD v2 — Pre-registration addendum (P0, 2026-09-27)

**Scope.** This addendum records the frozen splits and the run-order changes of plan
`.project/plans/katvad-v2-e0-e2.md` **before any v2 number is read**. The proposal
(`KAT_VAD_PROPOSAL_v2.md`) remains the authority for components, eligibility rules and
adoption rules. This file changes only the **order** of the checks, adds **engineering checks**,
and makes the **bootstrap unit** explicit. The commit that adds this file is the hash every v2
read-out cites.

---

## 1. Frozen splits

Written by `python -m core.tools.freeze_splits` into `core/splits/v2/`. Verify them with
`--check`, and read them only through `core.data.v2_splits.load_split`. That loader checks each
file's sha1 against `SPLITS_MANIFEST.json` and raises `SealedSplitError` for DoTA-eval unless it
is called with `final=True`.

| Split | Unit | Size | Rule |
|---|---|---|---|
| **T2-val** | T2-train source video | **219 / 1,491 sources**, 645 windows (472 abnormal / 173 normal) | 15 % per accident `type`, drawn with `subset_train.draw_subset`, seed 2024. Types with fewer than 4 train sources get no val source (17 types) |
| **DoTA-dev** | DoTA val clip, grouped by YouTube video | **702 clips / 93 videos** | 50 % of clips, stratified by the share bin of each video's median clip, seed 2024 |
| **DoTA-eval** | same | **700 clips / 86 videos** | the complement. **Sealed** until the final report |

Measured by the freeze (DoTA clips per share bin, dev / eval): `<30` 343 / 343 · `30–50`
247 / 247 · `50–70` 93 / 90 · `>70` **19 / 20**.

Facts the freeze surfaced:
- **At native frames every DoTA val clip has an accident.** The "3 all-normal clips" seen in
  stride-8 results are anomaly windows that round away at stride 8 (`dota.py:256`). The
  manifest's `all_normal_clips` is empty.
- **The `>70` bin holds only 19 dev clips.** The proposal's no-reversal rule (§4.2 step 4)
  reads AUC ≥ 0.5 in both `>50 %` bins. On `>70` that point estimate rests on about 19 clips
  from a handful of videos. The rule is **applied as written**. Its cluster-bootstrap interval
  is printed beside it, and a pass or fail decided by `>70` alone is flagged to the advisor
  rather than silently accepted.
- **T2 `class_name` is constant (`CarAccident`) over all 5,507 windows.** A `class_name`
  stratum is therefore a single group, so T2-val is stratified by `type`. See pending lesson (ac).

## 2. Run-order changes (vs proposal §10.2)

| # | Change | Why |
|---|---|---|
| D1 | **Kill-switch K** (§3) runs right after this freeze, before E1 / E0b | The largest risk to v2 is a frozen motion stream with no signal. It should be known first. |
| D2 | E2(b) runs at **both** stride 8 and stride 3. The reference is taken at the stride E1 adopts | E1 changes DoTA's step rate, which changes `t/T`, the deviation and the share-bin content. |
| D3 | Every bootstrap on DoTA resamples **source videos**, not clips (cluster bootstrap). This includes E1's decision interval | 1,402 clips come from 179 videos (up to 19 clips each). A clip bootstrap understates the variance. |
| D4 | **Pilot**: 1 seed (**2099**, outside 2024–2028) × 4 arms. It reads mechanics and T2-val only, and **DoTA-dev is not printed** | Catches bugs before the 20 E3 runs without touching the pre-registered seeds or inviting tuning on a Δ (lesson 14). |
| D5 | **A0 regression**: A0 built with v2 code, re-scored on T2-val, must be within 0.02 of the phase-4 KIP-off checkpoint re-scored on T2-val | The v2 code changes the forward pass and the data layer. A0 has to be shown to still be the baseline. |
| D6 | **Temporal-shuffle control** for the chosen encoder: shuffle the 16 frames inside each clip, then re-probe. **Reported only** | Separates "adds motion" from "is a second appearance encoder". It limits what the thesis may call motion. |

## 3. Kill-switch K (fixed now)

- **Encoder:** VideoMAE V2-**B** (`vit_b_k710_dl_from_giant`), the strongest candidate. If B
  carries no signal, S does not either.
- **Subset:** about 200 DoTA-dev clips, grouped and stratified by share bin, plus about 300
  T2-train sources (none from T2-val). Stride 8, so the features pair with the existing CLIP cache.
- **Probes:** `core/eda/features.py` frame linear probe.
  - (i) in-domain DoTA-dev, video-grouped CV: CLIP-only vs `u`-only vs `[x ; u]`;
  - (ii) transfer, T2 subset → DoTA-dev subset.
  - All intervals are cluster bootstraps over videos.
- *(Superseded by §6.1, 2026-09-28: the point-estimate leg is dropped.)*
  **KILL the motion stream** if **both** Δ(`[x;u]` − CLIP) have a point estimate ≤ 0 **and** an
  upper CI bound < +0.03.
- **Positive control:** the `u`-only in-domain AUC must have a CI that excludes 0.5. If it does
  not, the pipeline is suspected and fixed, and **K does not KILL**.
- **Otherwise GO.** K never makes an encoder eligible; that stays E2(d)'s rule (proposal §10.2).

## 4. Pilot read-out (P6, fixed now)

Read on T2-val only:
- the guardrails (micro < clip oracle, macro ≥ micro, micro ≥ A0 − 0.01);
- `ρ_u` and `‖W_u‖` over training;
- clip-level AUC vs macro (the C14 collapse signature);
- position R² on `V^t` and the source-shortcut AUC.

`ρ_u` ≈ 0 is recorded as a result, **not** fixed by changing `lr` or `c`.

## 5. Go / no-go for E3 (P7, fixed now)

E3 runs only if all of the following hold:
- the splits pass `--check`;
- the protocol is fixed by E1;
- the CRN reference is chosen, or CRN is dropped;
- the encoder is eligible, or the motion stream is dropped;
- D5 passes;
- every pilot arm passes the guardrails, with no collapse;
- the test suite and the quality gate are green.

The E3 arms are the survivors of E2. Seeds, decision interval and adoption rules follow proposal §10.

---

## 6. Amendment 1 (2026-09-28) — DoTA pixels are gone: D7–D9

Written **before any v2 number is read**; nothing of P1–P3 had run. Cite this file's commit
beside `7422975` in every v2 read-out.

**Fact.** The project holds no DoTA pixels. The Drive copy of the frames was deleted, the
official links in `MoonBlvd/Detection-of-Traffic-Anomaly` return 404, and all 184 source
YouTube videos in `DoTA_urls.txt` are unavailable. What survives on Drive (confirmed by the
user 2026-09-28) is the stride-8 CLIP cache `clip/DoTA_s8_ncc` and `labels_s8`. So every
number that needs a new DoTA feature is unmeasurable: a stride-3 CLIP cache, and any
VideoMAE feature. Plan assumption A2 was wrong for DoTA; it holds for DADA-original.

| # | Change | Why |
|---|---|---|
| D7 | **Kill-switch K runs on T2 alone** (design below). Probe (i) on DoTA-dev and probe (ii) T2 → DoTA are replaced by (i′) and (ii′) | (i) and (ii) both need VideoMAE features on DoTA |
| D8 | **E1 is dropped.** The DoTA protocol stays stride 8, whole clip, per-clip min-max (today's). D2's stride-3 branch is dropped. **E0 is kept as a record only**: it can no longer switch anything | E1's arms B and C need a stride-3 DoTA cache, which needs pixels |
| D9 | **The motion arms (A2, A3) have no DoTA endpoint. This amendment does not choose one.** If K = KILL, D9 is moot: v2 keeps A0/A1, both scorable on the DoTA s8 cache. If K = GO, the advisor chooses the motion endpoint **before P4**, and the choice is committed as Amendment 2 before any motion number is read as a decision | The proposal's E2(d) eligibility rule (DoTA-dev in-domain ≥ +0.10 or transfer ≥ +0.03) and the E3 decision interval for A2/A3 cannot be computed as written. Choosing a replacement after seeing K would be choosing it on the data |

### 6.1 Kill-switch K on T2 (D7, fixed now)

- **Encoder:** unchanged, VideoMAE V2-B distilled (`vit_b_k710_dl_from_giant`), weights pinned
  by HF commit and sha256 (`core/constants.py`).
- **Subset:** about 300 **T2-train** sources with **no T2-val source** (`draw_subset`,
  stratified by accident `type`, seed 2024). The drawn list and its sha1 are written beside the read-out.
- **Unit: the whole source video at stride 8**, with frame labels from the annotation span,
  built through `dada_origin.make_record` + `sampled_frame_labels` (Gate D0's convention).
  Windows are not used because T2's train windows carry no frame labels.
- **Features:** CLIP `x` (the existing `clip/DADA2000_orig` cache), `u` (VideoMAE on the causal
  1.5 s clip that ends at each CLIP step), and `[x ; u]`. If `len(u) ≠ len(x)` for any source, the probe raises.
- **Probe:** the `core.eda` frame linear probe (standardized logistic regression, balanced), 5 folds:
  - (i′) **source-grouped** CV: in-domain;
  - (ii′) **type-grouped** CV: every test fold holds accident types absent from its train fold.
    **This is not a domain transfer.** It is the stricter of the two generalization reads left
    without DoTA pixels, and the read-out says so.
- **Metric and interval:** macro = mean per-source frame AUC over two-class sources. Δ is paired
  per source. The 95 % CI is a percentile bootstrap over **sources** (B = 2000, seed 2024).
- **Rule (amends §3, chosen 2026-09-28 before any K number, option A):** **KILL** if **both**
  Δ(`[x;u]` − `x`) have an upper bound < +0.03, i.e. no gain ≥ 0.03 is compatible with the data.
  §3's extra leg "point estimate ≤ 0" is **dropped**. On a synthetic `u` that adds nothing to `x`,
  Δ was about ±1e-4, and that leg made the verdict a sign coin-flip (KILL on 1 of seeds 0–3). A
  rule that cannot reliably kill the null it exists for is not a rule. Price: a real gain below
  +0.03 with a tight CI is killed too. That gain is below the proposal's own +0.03 transfer
  eligibility bar, so it would not have become eligible anyway. **Positive control:** `u`-only under (i′) must have a CI lower
  bound > 0.5. If it does not, the pipeline is suspected and fixed, and **K does not KILL**.
  Otherwise **GO**. K still never makes an encoder eligible.
- **Printed, not gated:** the `x`-only (i′) macro beside Gate D0's 0.6518 (different subset,
  same convention), and the extraction throughput.

### 6.2 K-pos diagnostic (2026-09-28, written after K = GO, before any diagnostic number)

**Status: printed, not gated.** It cannot change K's verdict (GO stands, §6.1) and it makes no
encoder eligible. It exists to give the advisor data for the D9 endpoint choice.

**What was seen before writing this.** K's read-out (§6.1: `u` 0.760, `x` 0.615, Δ +0.13) and
three **label-only** rulers on the same 295 sources, no feature read: absolute step index
0.529; "the step's causal clip is padded" 0.573; a position tent `−|t/L − c|` with `c` fit
in-sample 0.729 (`c` = 0.525). DADA accidents sit mid-video (median span 36 %–71 % of `L`).

**Two questions, one subcommand** (`kill_switch_probe diag`, same inputs, sources and seed as
`run`; writes `k_diag.{json,md}` and never touches `k_readout.*`):

- **(P) Pad drop.** The first `ceil(3·15 / stride)` steps (6 at stride 8) have a causal clip
  that repeats frame 0; they are always normal. Drop them from `x`, `u` and the labels,
  re-filter to two-class sources, re-run `x`, `u`, `[x;u]` under (i′) and (ii′).
  Read: **PAD_EXPLAINS** iff both Δ(`[x;u]` − `x`) have upper < +0.03; else **NOT_PAD**.
- **(Q) Beyond position.** `p_t = [τ, τ², τ³]`, `τ = (t + 0.5)/L` (the cubic of E2(b)).
  Probe `p`, `[x;p]`, `[x;u;p]` on all steps under (i′) and (ii′).
  Read: **POSITION_PROXY** iff both Δ(`[x;u;p]` − `[x;p]`) have upper < +0.03; else
  **BEYOND_POSITION**. `p` alone is the out-of-fold version of the tent ruler above.

Probe, metric, bootstrap and the +0.03 bar are §6.1's, unchanged.

---

## 7. E0 / E2(a–c) implementation choices (2026-09-28, before any E2 number)

The rule is proposal §4.2 steps 1–6, applied mechanically by `python -m core.tools.crn_select`
(references in `core/crn/reference.py`). The proposal leaves these points open; they are fixed here.
Already seen before writing: K, K-pos (§6.2) and label-only position rulers on T2 (whole
sources 0.73, T2-val windows 0.575). No deviation `d_t` of any reference has been computed.

| # | Choice | Why |
|---|---|---|
| I1 | **T2-val unit = the 219 whole source videos** at s8, labels from the annotation span (Gate D0 / K convention) | CRN's reference is computed over the source video in training (§4.2); T2 windows are not the unit CRN sees |
| I2 | Share = native annotation span for DoTA (the freeze's stratum, so the bins match §1's counts); the s8 label mean for T2 sources | DoTA-dev bins must match the frozen stratification |
| I3 | Position ruler and `f` use `t/T`, `t = 0..T−1` | §4.2 step 2 as written |
| I4 | R4 on a clip shorter than `N_w` = 8 steps: the warm-up is all its steps (= R1) | DoTA's median clip is ~13 steps; the proposal does not define this case |
| I5 | Steps 4–5 read on **DoTA-dev**: both `>50 %` bins (`50-70`, `>70`) of the decision metric ≥ 0.5 **and** DoTA-dev stratified AUC ≥ 0.5. An empty bin is a fail. T2-val is printed, not decided on | Step 4 names DoTA-dev for the overall; the bin check is read on the same set |
| I6 | Coverage fallback (step 3) is read on the `f`-fit set: T2-val normal steps with `t/T ≥ 0.8` | The fallback exists because `f` lacks late normal steps |
| I7 | Intervals: cluster bootstrap, B = 2000, seed 2024; DoTA clusters = source video (D3), T2 clusters = source. Printed, not decided on (steps 4–5 are point rules), except E2(c) | §1: a `>70` pass/fail is flagged (`decided_by_gt70`), not silently accepted |
| I8 | E2(c) probe = the `core.eda` logistic frame probe fitted on T2-train sources (all T2-train minus T2-val, whole, s8), scored on DoTA-dev; CRN input `x − μ^ref` (the scalar `s` is absorbed by standardization). Veto iff the chosen reference's paired Δ vs raw has upper < 0 | Step 6 as written; D3 for the interval |
| I9 | E0 fps: DoTA **10** (dataset release), DADA **30** (assumption A1). Record only (D8) | E1 is dropped; E0 switches nothing |

## 8. Amendment 2 (2026-09-29) — a DoTA stride-1 CLIP cache exists: E1 and D2 restored (D10–D11)

Written **before any E1 number is read**. No DoTA score at stride 3 or at native frames has been
computed by any code in this tree. **Authorized by the user on 2026-09-29; not reviewed by the
advisor** (the user's instruction). D9 (the motion endpoint) stays open; this amendment does not
touch it.

**Fact.** Drive holds `clip/DoTA_s1_ncc` (1,397 clips). `s1[::8]` equals `DoTA_s8_ncc` on 30/30
random clips (`np.allclose`, atol 1e-4), so it was built with the same `no_center_crop`
transform, the same CLIP backbone and the same `range(0, N, stride)` sampling. Every CLIP stride
is therefore derivable without pixels. Pixels are still gone, so D7 and D9 stand.

| # | Change | Why |
|---|---|---|
| D10 | **D8 is reversed: E1 runs as proposal §10.2 wrote it**, on DoTA-dev, on the three phase-4 KIP-off stage-2 checkpoints (seeds 2024–2026, `checkpoint_last.pt`). E0 read 3.00×, so E1's skip condition does not fire | D8's only reason was "a stride-3 cache needs pixels"; `s1[::3]` is that cache |
| D11 | **D2 is restored, conditionally.** If E1 adopts B or C, E2(b) and E2(c) are re-read on DoTA-dev at stride 3 before P5 fixes the CRN reference; the T2 side is unchanged. If E1 keeps A, the P2 read-out stands as is | The reference must be taken at the stride E1 adopts (D2) |

**D11 executed (2026-09-30; a record, not an amendment).** E1 adopted B, so E2(b)/(c) were re-read on
DoTA-dev at stride 3 (`colab/v2/p2_e2_s3.ipynb`, s8 == s1@8 gate passed). The §4.2 rule, unchanged,
picks **R2** (DoTA-dev `r` 0.7092 vs R1 0.7081); the E2(c) veto passes (+0.037 [+0.027, +0.047]).
**The CRN reference for every CRN arm is R2.** R1 and R2 are within noise; no tie clause is added
after the fact. Record: `core/docs/v2/RESULTS_E2_CRN.md` §D11.

### 8.1 E1 implementation choices (fixed now)

`python -m core.tools.rate_matched_eval`, runbook `colab/v2/p3_e1.ipynb`.

| # | Choice | Why |
|---|---|---|
| J1 | Native frame count `N` = rows of `DoTA_s1_ncc`. Native labels = `sampled_frame_labels` at stride 1 with `total_frames = N` (the baseline's arithmetic; equal to the raw `[start, end)` window when `N` matches the annotation). Clips where `N` differs from the annotation's `num_frames` are counted and printed | One ground truth for every arm, and it is the one the s8 labels were rounded from |
| J2 | A = `s1[::8]`, whole clip. B = `s1[::3]`, whole clip. C = `s1[::3]` in windows of W = 20 steps, hop 4 (starts 0, 4, …, plus one window ending at the last step if the hop misses it); a step's score is the mean over the windows that cover it. A clip of ≤ 20 steps is one window (C = B there). Whole clip = one forward pass (the longest dev clip is < 100 steps at s3, below `max_vis_len` 512) | Proposal §7.3 |
| J3 | Step `t` sits at native frame `t · stride`; scores are linearly interpolated to frames `0 … N−1` and held constant past the last sampled frame. All three arms go through the same interpolation | "Scores are interpolated to native frames, and the baseline is re-scored under the same evaluator" (§7.3) |
| J4 | Seed averaging: per native frame, the mean of the three checkpoints' sigmoid scores | "Seed-averaged scores" (§10.1) |
| J5 | Decision metric: macro AUC over the two-class DoTA-dev clips at native frames. Per-clip AUC is rank-based, so per-clip min-max does not change it. Micro AUC (per-clip min-max) is printed, never decided on | The v2 DoTA endpoint is macro (§10.1) |
| J6 | Interval: per-clip paired Δ = AUC(arm) − AUC(A) on seed-averaged scores; cluster bootstrap over source videos (D3), 10,000 resamples, seed 2024, 95 % percentile | §10.1's E1 exception, with D3's clusters |
| J7 | **Rule.** Arm X ∈ {B, C} is eligible iff its Δ interval's lower bound is > 0. None eligible → keep A. One → adopt it. Both → the larger mean Δ, unless the two means differ by < 0.01, in which case B (no windowing) | §10.1: "adopt B or C if the interval excludes 0"; the proposal names no tie rule and a negative interval is not a reason to switch |
| J8 | Printed, never decided on: per-seed macro per arm, macro per share bin, micro, and the no-pixel position ruler `t/N` at native frames (identical for every arm, since the labels are shared) | Print position beside every frame-level macro (pending (ag)) |
| J9 | ~~**Hard regression gate.** Arm A at step level (before interpolation) must reproduce every DoTA-dev clip's `max_score` in that seed's phase-4 `eval_dota_kip_off/results.json` within 1e-4. A failure stops the run before B or C is scored. Only DoTA-dev entries are read; DoTA-eval labels are never loaded~~ **Replaced by J9′ (Amendment 3, §9)** | A harness that cannot reproduce the old protocol cannot be trusted to compare against it |

## 9. Amendment 3 (2026-09-29) — two of the three phase-4 KIP-off checkpoints are mid-training snapshots (D12, J9′, J10)

Written **before any E1 number is read**: the first E1 run stopped at J9 before B or C was scored
and before any macro, micro or Δ was computed. **Authorized by the user on 2026-09-29; not reviewed
by the advisor.**

**Fact (measured on Drive, 2026-09-29).** `outputs/DADA2000_orig_phase4/s{seed}/stage2_kip_off/checkpoint_last.pt`:

| seed | checkpoint `global_step` | `metrics.jsonl` last step | J9 |
|---|---|---|---|
| 2024 | 2040 | 2040 | **passed** on all 702 DoTA-dev clips |
| 2025 | **510** (epoch 5) | 2040 | failed on all 702 clips |
| 2026 | **1530** (epoch 15) | 2040 | not reached |

Phase 4 trained on VM-local disk and synced to Drive after every 5-epoch chunk; the final
checkpoints of s2025 and s2026 never landed on Drive while their `metrics.jsonl` did, and the
notebook's "already trained" check reads `metrics.jsonl`, not the checkpoint. The phase-4 DoTA and
T2 numbers were computed on the local, finished checkpoints and stay valid; the finished weights of
s2025 and s2026 are lost. s2024's J9 pass shows the harness's scoring reproduces `core.evaluate`
exactly on a finished checkpoint.

| # | Change | Why |
|---|---|---|
| D12 | **E1's checkpoints = s2024 (phase 4, finished) + s2025 and s2026 retrained** with phase 4's exact stage-2 KIP-off command (`colab/DADA2000Origin/phase_4.ipynb` `train_cmd`: `train.num_epochs=20`, `train.amp=true`, `model.score_head_kernel=3`, `loss.mil_topk_pct=5`, `kip.enabled=false`, same T2 corpus, CLIP cache and KNN cache), runbook `colab/v2/p3_retrain_kipoff.ipynb`, written to `Thesis-V2/outputs/v2_p4_retrain/`. The retrained seeds are not bit-identical to the lost ones; their phase-4 DoTA macros (0.6466, 0.5928) are printed beside the retrained ones, never gated | J4 needs three checkpoints; a 1-seed E1 would drop the training-variance averaging the rule assumes. The retrained checkpoints also serve P6's A0 regression (D5) |
| J9′ | **Regression gate, replacing J9.** For every checkpoint, arm A's step-level curve from `s1[::8]` must equal, within 1e-4 on every step of every DoTA-dev clip, the curve the same checkpoint gives through the evaluator's input path (the `DoTA_s8_ncc` cache, whole clip, per-item verbalizer). No old `results.json` is read and no label is read. A failure stops the run before B or C is scored | J9 tied the gate to one historical checkpoint file; J9′ tests what J9 was for (the harness input path equals the evaluator's) and works for any checkpoint. The scoring code itself is already validated against `core.evaluate` by s2024's J9 pass |
| J10 | **Finished-checkpoint gate.** A checkpoint is used only if its stored `global_step` equals the last `global_step` in the `metrics.jsonl` beside it; otherwise the run raises before scoring | The defect behind D12; a mid-training snapshot otherwise scores silently |

## 10. P5 implementation choices (2026-09-30, before any v2 model is trained)

Fixed while building the v2 model (plan P5), **before P6 or any v2 training run**; no v2 checkpoint
exists. Authorized by the user on 2026-09-29/30; not reviewed by the advisor. Architecture:
`core/docs/v2/KAT_VAD_v2_ARCHITECTURE.md`; code and commands: `core/docs/v2/TRAINING_V2.md`.

| # | Choice | Why |
|---|---|---|
| K1 | **CRN and the motion scaling are baked offline** (`core.tools.build_v2_inputs`) into a v2 input cache whose rows are `[x or s·(x − μ^x) ; c·ũ⊘σ_u]`; the model adds only `W_u` (`core/models/motion_residual.py`). A0 reads the plain CLIP cache | Both transforms are parameter-free given fixed statistics (architecture §4–§5). Baking keeps windowing, DVS splicing ("CRN per segment before splicing", §9), collate and evaluation untouched |
| K2 | **Reference unit: the source video for every T2 item** (training windows *and* T2-val/T2-test windows, which are slices of the baked source file); **the clip for DoTA** | Architecture §4 says "source video in training, whole clip at test"; for T2 the evaluated window is a slice of one source video, and a 20-step reference would differ from training's. DoTA clips are the test unit |
| K3 | Statistics fitted on **T2-train minus T2-val** sources: `c = E‖x‖/√512` (raw CLIP); `s = E‖x‖ / E‖x − μ^x‖` (so `E‖x̃‖ = E‖x‖`); `m_u` = channel mean of `u`; `σ_u` = per-channel std of the arm's centred motion stream (`u − m_u` for A2, `u − μ^u` for A3), floored at 1e-6. The manifest records `train_ids_sha1` | Architecture §4–§5, §11 |
| K4 | A cache carries `v2_input_manifest.json`; `train` and `evaluate` refuse a cache whose `crn`/`motion` differ from the run's `v2` config (evaluate reads `v2` from the checkpoint), and refuse a v2 cache under a v2-off config | A plain CLIP cache under a v2 config would silently train A0 |
| K5 | `W_u` weight and bias zero-initialized; padded steps get no motion term; `ρ_u` and `‖W_u‖_F` logged on every batch in `metrics.jsonl` (every 50 steps is a subset) and never enter the loss | Architecture §5 |
| K6 | v2 inputs require `kip.enabled=false`; `model.motion_dim` must equal the encoder's width (768 B, 384 S) | Architecture §13: KIP is removed in v2 |

## 11. Amendment 4 (2026-10-03) — D9 decided: the motion endpoint is DoTA-CAP (D13–D15, L1–L7)

Written **before any motion-arm score exists**: no v2 model has been trained and no VideoMAE feature
of any DoTA clip has been computed. **Authorized by the user on 2026-10-03; not reviewed by the
advisor.** It replaces plan `katvad-mmau-phase0.md` §4's sentence "partial coverage is not used as a
DoTA subset for the motion arms" — the user overrode it, and the conditions below are what make the
override defensible.

**Fact (MM-AU Phase 0, full CAP, 2026-10-02; `outputs/v2/REPORTS/mmau_p0/all/`).** Branch **C** by
§4's rule: exact coverage 0.893 of all DoTA (1,248 / 1,397) and 0.890 of DoTA-dev (625 / 702), below
the 0.95 bar; exact ∪ near 0.935 / 0.930. Null flag 0.006 (reliable). Counting the 7 pairs demoted by
Amendment P0b-1 as hits would give 0.940, so the branch does not depend on P0b-1. DADA-2000 is
essentially absent from CAP (3 near / 1,945). Exact matches align at CAP/DoTA frame rate ≈ 1 (842) or
≈ 3 (399): CAP stores some DoTA clips at 10 fps and some at 30.

| # | Change | Why |
|---|---|---|
| D13 | **D9 = DoTA-CAP.** The motion endpoint is DoTA restricted to the clips whose pixels are recovered from CAP by L1–L7 below. Every contrast that involves a motion arm (A2 − A0, A3 − A0, A3 − A1, A2 − `F`, A3 − `F`, and E2(d)'s DoTA-dev probes) is computed with **all arms on the same DoTA-CAP clips**, decided on **DoTA-CAP-dev** = DoTA-dev ∩ DoTA-CAP and reported once on DoTA-CAP-eval (still sealed). Protocol B (E1) unchanged. The CRN contrast A1 − A0 and the choice of `F` stay on full DoTA-dev as pre-registered. The adoption rules of proposal §10.3 are otherwise unchanged | Pairing inside one clip set keeps the contrast fair: the subset is chosen by whether CAP contains the clip, which does not depend on any model's score |
| D14 | **Naming and placement.** It is written "DoTA-CAP (n/1397)" everywhere; a DoTA-CAP number is never put beside a full-DoTA number, LaGoVAD's 62.60 or a phase-4 DoTA number | The 149 missing clips are not missing at random |
| D15 | **Representativeness, printed, never decided on:** A0 and A1 macro on full DoTA-dev vs on DoTA-CAP-dev (same checkpoints, seed-averaged); per-clip length, accident share and DoTA category of kept vs dropped DoTA-dev clips. DoTA-eval's composition is not read | Lets a reader judge how far DoTA-CAP generalizes to DoTA; it cannot change a verdict |

### 11.1 DoTA-CAP construction (L1–L7, fixed now) — `python -m core.tools.dota_cap`, runbook `colab/v2/dota_cap_videomae.ipynb`

| # | Choice |
|---|---|
| L0 | Candidates = the P0 `exact` pairs only (κ ≥ 0.99). `near` pairs (58) are excluded: CAP re-processed them, so VideoMAE would see different pixels |
| L1 | Per-frame map: for each DoTA frame `d` the CAP frame `j_d`, strictly increasing in `d`, maximizing `Σ_d cos(DoTA_s1[d], CAP_s1[j_d])` (dynamic programming on the two s1 `_ncc` CLIP caches; no rate is assumed). A CAP clip shorter than the DoTA clip fails |
| L2 | Mean aligned cosine ≥ 0.99 **and** every aligned frame ≥ 0.95 |
| L3 | Time axis: at most 5 % of the CAP steps `Δj` may differ from the clip's median step by more than 1 (a look-alike frame in a static stretch would warp VideoMAE's time axis) |
| L4 | The streamed CAP folder has exactly as many frames as the CAP CLIP cache has rows (the map indexes the same sorted frame list) |
| L5 | Pixel gate: CLIP (`no_center_crop`) of the rebuilt frame sequence vs `DoTA_s1_ncc`, same thresholds as L2. A clip that fails gets no feature file |
| L6 | VideoMAE geometry on the rebuilt sequence: DoTA's native 10 fps, 16 consecutive frames (`DOTA_VIDEOMAE_FRAME_STEP` = 1, the same causal 1.5 s as DADA's every 3rd at 30 fps — architecture §3), ending at the step, clamped at DoTA frame 0 (no CAP frame outside the DoTA clip is used), squash 224², stride 1 (row-aligned with `DoTA_s1_ncc`; any stride is `[::s]`). Encoders V2-B and V2-S. Cache `cache/video/<encoder>/DoTA_CAP_s1_squash/`, manifest with `frame_source` and `alignment_sha256` |
| L7 | DoTA-CAP = clips passing L1–L6 with a feature file for every encoder. The id list (`dota_cap_ids.txt`, sha1 in its read-out) is frozen into `core/splits/v2/` and committed **before any motion-arm score is read** |
