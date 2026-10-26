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

### 11.2 Amendment 4a (2026-10-03) — L1 uses a fitted line, not the DP path (L1′)

Written after the `align` read-out (counts of kept clips only) and **before any VideoMAE feature or
motion score exists**. **Authorized by the user on 2026-10-03; not reviewed by the advisor.** The
change was made after seeing how many clips survived; it is justified by the mechanism below, which
was measured, not by the count.

**Fact (Colab, 2026-10-03, `outputs/v2/REPORTS/dota_cap/`).** Under L1–L3 as first written: kept 895 /
1,248 exact (DoTA-dev 439 / 702); `irregular_steps` dropped 323, `mean_cos` 15, `min_cos` 7,
`cap_shorter` 8. Kept median CAP step 1 → 817, 3 → 77: almost every CAP-at-30-fps clip (399 at P0)
failed L3. Diagnostics (printed, not gated):

| | n | Theil–Sen rate q05/q50/q95 | aligned mean cos q50 | adjacent-CAP duplicate share q50 |
|---|---|---|---|---|
| kept | 895 | 1.0 / 1.0 / 3.0 | 0.9942 | 0.0 |
| `irregular_steps` | 323 | **3.0 / 3.0 / 3.0** | 0.9937 | 0.16 |

The failed clips are uniform 30 fps streams (rate exactly 3) whose content matches as well as the kept
ones. CAP did not duplicate frames (the duplicate-collapse hypothesis rescued 1 clip). The DP path
picks a CAP frame 1–2 frames (33–67 ms) off the true one because CLIP cannot tell two adjacent 30 fps
frames apart: the defect was in L1's construction, and L3 correctly refused to hand VideoMAE a
jittered clock.

| # | Change |
|---|---|
| L1′ | The frame map is `j_d = ⌊rate · d + offset + ½⌋`, with `(rate, offset)` the Theil–Sen fit (median pairwise slope; intercept `median(j) − rate · median(d)`) on the L1 DP path. The line must stay inside the CAP clip (`out_of_range`) and be strictly increasing (`not_increasing`). L2 and L3 are then evaluated on this map, unchanged. A CAP stream that is not a uniform resample of DoTA fails L2. Rounding is half-up: numpy's half-to-even rounding alternates the steps 4, 2, 4, 2 on an odd integer rate with a .5 offset (caught by a test) |

Expected (Colab diagnostic with numpy rounding, so indicative only): about 886 of the 895 + 256 of the
323 pass L2. The tool's own read-out is the record.

### 11.3 Amendment 4b (2026-10-03) — the line may move by less than one DoTA frame

Written after the L1′ `align` read-out (counts only) and **before any VideoMAE feature or motion
score exists**. **Authorized by the user on 2026-10-03; not reviewed by the advisor.**

**Fact.** Under L1′: kept 852 (DoTA-dev 409 / 702); `out_of_range` **347**, `mean_cos` 32, `min_cos` 9,
`cap_shorter` 8; kept median step 1 → 817, 3 → 34. L1′ moved the 30 fps clips from `irregular_steps`
to `out_of_range`, and it also lost 43 of the 77 rate-3 clips that L1 had kept. The Colab diagnostic
that predicted ~1,142 clipped the line to the CAP clip; the tool did not. A CAP clip that ends on DoTA's
last frame leaves no room for a fitted offset above 0 (the DP's jitter pulls the median offset toward
+1), so the rounded line ends one or two CAP frames past the clip.

| # | Change |
|---|---|
| L1″ | Keep L1′'s rate. The line may move by an integer `k` CAP frames with `|k| < rate` (strictly less than one DoTA frame interval, so a DoTA frame can never take its neighbour's content; at rate 1 only `k = 0`). Among the shifts that stay inside the CAP clip, the one with the highest mean cosine is used; none inside → `out_of_range`. L2/L3 are unchanged. Each clip records `overshoot` (CAP frames the unshifted line leaves the clip by) and `phase_shift`; the read-out prints both histograms |

Not chosen: clamping the line to the clip, which makes two DoTA frames share one CAP frame (`not_increasing`).

### 11.4 Amendment 4c (2026-10-03) — each DoTA frame may sit within a band of the line

Written after the L1″ `align` read-out and a printed diagnostic, **before any VideoMAE feature or motion
score exists**. **Authorized by the user on 2026-10-03; not reviewed by the advisor.**

**Fact.** Under L1″: kept 906 (dev 441); `out_of_range` 46 (overshoot 1 → 26, 2 → 12, ≥ 3 → 8), `mean_cos`
**274**, `min_cos` 14. Diagnostic on the cached CLIP rows (q50; `ceiling` = each DoTA frame's best CAP frame
anywhere, `near3` = best within ±3 of the line, `drift` = Theil–Sen slope of that best offset × clip length):

| group | n | line | DP path | ceiling | near3 | drift | q1 / mid / q4 (line) |
|---|---|---|---|---|---|---|---|
| `mean_cos`, rate 3 | 261 | 0.9863 | 0.9934 | 0.9935 | 0.9934 | 0 | 0.986 / 0.985 / 0.990 |
| `ok`, rate 3 | 88 | 0.9920 | 0.9948 | 0.9949 | 0.9948 | 0 | 0.991 / 0.992 / 0.993 |
| `mean_cos`, rate 1 | 12 | 0.9897 | 0.9897 | 0.9907 | 0.9904 | 0 | — |

Rate-3 content is present (ceiling ≥ 0.99) within ±3 frames of the line, with no drift and no edge effect:
the true map is the line plus a bounded residual that a straight line cannot follow. Mechanism (consistent,
not proven): the adjacent-duplicate share of these CAP streams is 0.16 ≈ 1/6, i.e. a 25 fps source resampled
to 30 fps, while DoTA took 10 fps from that source, so two nearest-frame resamplings leave a periodic ±1 residual;
rate-2.5 CAP clips also exist. The rate-1 failures have ceiling ≈ 0.991 — genuinely different content; they stay out.

| # | Change |
|---|---|
| L1‴ | After L1″'s shift, each DoTA frame takes the CAP frame with the highest cosine within `band = ⌈rate/2⌉ − 1` frames of the line (strictly under half a DoTA interval: 1 at rate 3 or 2.5, 0 at rate ≤ 2), strictly increasing (DP restricted to the band). The shift is chosen by the banded path's mean cosine. The line still sets the clock, so no drift is possible; the band only absorbs the residual |
| L3′ | Steps are compared with the fitted rate, not their median: irregular iff `|Δj − rate| > DOTA_CAP_STEP_TOL + 2·band`. At band 0 this is L3 (line steps equal the rate). A median of alternating 4, 2 steps is 4 or 2, which made L3 reject the residual L1‴ is meant to keep |

L2 thresholds, L4–L7 unchanged; L5 (pixel CLIP vs `DoTA_s1_ncc`) still re-checks every rebuilt clip. Each clip
records `line_mean_cos`; the read-out prints line → band q50 per reason. Not chosen: lowering L2 (a rule tuned on a count).

## 12. E2(d) on DoTA-CAP — implementation choices (2026-10-03, before any motion number)

Written **after** DoTA-CAP was frozen (`b6d937f`, `core/docs/v2/DOTA_CAP.md`) and **before any VideoMAE
feature of a DoTA clip has been probed, labelled or scored**. The construction run read only CLIP cosines
(L2/L5), never a label. Already seen before writing: K and K-pos on T2 (§6.1–6.2), E2(b)/(c) at s3 (§8,
CLIP only), E1 (§8.1). **Authorized by the user on 2026-10-03, N4, N6 and N7 as written; not
reviewed by the advisor.** Proposal §10.2 E2(d) is applied as written; this section fixes what it leaves
open and what D13 changes.

| # | Choice | Why |
|---|---|---|
| N1 | **Decision set = `dota_cap_dev` (569 clips), protocol B rows**: CLIP `DoTA_s1_ncc[::3]` and VideoMAE `DoTA_CAP_s1_squash[::3]`, whole clip. Labels = J1's native labels at stride 3 (`resized_frame_labels(record, N, 3)`, the D11 loader). Every clip's VideoMAE row count must equal its `DoTA_s1_ncc` row count (L6), else the run raises. Metric = **macro**: mean per-clip step AUC over the two-class clips, no interpolation (as E2(c) at s3) | D13: all motion contrasts on DoTA-CAP, protocol B (E1). A probe has no native-frame output to interpolate; E2(c)'s s3 read is the precedent |
| N2 | **Representation per arm** (proposal: "the representation each arm feeds the trunk"). A2 = `[x ; u]` vs `x`; A3 = `[x − μ^R2(x) ; u − μ^R2(u)]` vs `x − μ^R2(x)`, `μ^R2` = the per-clip median (R2, §8) of that stream at the probe's stride (DoTA clip / T2 source unit, K2). The scalars `s`, `c`, the centring `m_u` and `σ_u` are **not applied**: the probe standardizes every channel on its train side, which absorbs a per-channel affine map exactly, so they cannot change a probe score | The comparator of each arm is its own CLIP-only input (A0's `x`, A1's `x̃`), so Δ isolates the motion stream |
| N3 | **In-domain probe** = the `core.eda` logistic frame probe (standardized, class-balanced, `C` = `EDA_PROBE_C`), 5-fold `GroupKFold` on `dota_cap_dev` with **group = source YouTube video** (D3), seed 2024. Out-of-fold scores → per-clip AUC | D3: clips cut from one video are correlated; a clip-grouped CV would leak |
| N4 | **Transfer probe** = the same probe fitted on **the K sources** (`outputs/v2/REPORTS/v2_K/k_sources.txt`, 300 T2-train sources, no T2-val source, frozen with its sha1 in `k_sources_manifest.json`), whole sources at s8, labels from the annotation span (K convention), scored on `dota_cap_dev` (N1). VideoMAE-B exists there (K); **VideoMAE-S is extracted for the same 300 sources** with the same geometry (16 frames, every 3rd at 30 fps = 1.5 s, causal, squash, s8). CLIP-only comparators fitted on the same 300 sources | Plan P4: "T2-train subsample". Re-using K's frozen draw needs one S extraction, not 2 × 1,272 sources; the train side is identical for every arm and encoder, so the contrast is paired. Full-T2 extraction follows only for the chosen encoder (plan P4 last step) |
| N5 | **Δ and interval** per arm × encoder × probe: per-clip paired Δ = AUC(with `u`) − AUC(CLIP-only), mean over clips; cluster bootstrap over source videos, 10,000 resamples, seed 2024, 95 % percentile | J6's construction (D3) |
| N6 | **Eligibility = proposal §10.2 thresholds on the point estimate**: encoder E is eligible for arm a iff in-domain mean Δ ≥ **+0.10** or transfer mean Δ ≥ **+0.03**. E is eligible iff it is eligible for A2 or A3. Intervals are printed, not gated | The proposal writes thresholds, not interval bounds; adding a bound now would be a new rule |
| N7 | **Pick**: among eligible encoders, the larger **A3 transfer mean Δ**; if B and S differ by < 0.02 there, **S** (cheaper, 384-d). None eligible → the Motion Stream is dropped: A2/A3 leave E3 and v2 = A0/A1 (proposal §10.2) | A3 is the full-v2 arm the thesis claims; one ranking quantity avoids a choice between arms after the numbers |
| N8 | **Candidates = V2-B and V2-S only.** SimpleTAD's DAPT encoder is not probed: no DAPT-only release has been verified, and a DoTA/DADA-fine-tuned weight is forbidden (proposal §4.1). Recorded as a limitation | Proposal: "only if the released weight is DAPT-only" — unverified, so the condition is not met |
| N9 | **Printed, never decided on**: (a) `u` alone; (b) position: K-pos's position features `p` (`position_features`: `τ, τ², τ³`, `τ = (t + 0.5)/L`) alone and `[x;p]` vs `[x;u;p]` per arm under both probes (pending (ag)); (c) macro per share bin (I2's bins); (d) T2-side in-domain read on the K sources (source-grouped CV) for S, beside K's B | Position explains most of K (§6.2); a motion Δ is read beside it |
| N10 | **D15 representativeness (printed)**: (i) CLIP-only in-domain probe macro on `dota_dev` vs `dota_cap_dev` (same folds rule, N3); (ii) A0 = E1's three KIP-off checkpoints at protocol B, macro on `dota_dev` vs `dota_cap_dev`, re-scored by `rate_matched_eval` with per-clip AUCs written (E1's read-out kept only aggregates); (iii) kept vs dropped `dota_dev` clips: length, accident share, DoTA category. A1's half of D15 waits for A1 checkpoints (P6/E3). DoTA-eval is not read | D15 as written; (ii) uses checkpoints that exist |
| N11 | **D6 shuffle control** runs only for the picked encoder, after the pick: VideoMAE re-extracted on `dota_cap_dev` with the 16 frames of each window permuted (fixed seed 2024, same permutation for every window), in-domain probe re-read, Δ(ordered − shuffled) printed. Needs the CAP stream again, so it is its own notebook | D6 is a diagnostic; it cannot move the pick |
| N12 | Hard gates before any probe: `dota_cap_dev` via `load_split` (sha1); every id has a CLIP s1 row file and a VideoMAE file per encoder with equal row counts; the VideoMAE manifests name `alignment_sha256` `5e8690e…` and the DoTA geometry (L6); the K-source list matches its manifest sha1; the S cache on the K sources has the same manifest geometry as B's | C2: a cache is bound to the transform and geometry that built it |

## 13. Amendment 5 (2026-10-03) — the pilot runs in two batches (D16)

Written before any v2 model is trained. **Authorized by the user on 2026-10-03; not reviewed by the
advisor.**

| # | Change | Why |
|---|---|---|
| D16 | **P6 runs in two batches, same seed 2099.** Batch 1 = **A0 and A1**, as soon as A1 is baked; batch 2 = **A2 and A3**, after E2(d) picks an encoder and the full-T2 VideoMAE cache exists. §4's read-out is unchanged and applied per arm; batch 1 may be read out alone (T2-val only, DoTA-dev still not printed, D4). D5 is read in batch 1. P7 waits for both batches. If E2(d) drops the Motion Stream, batch 2 never runs and the pilot is batch 1 | A0 and A1 need no motion feature, so they do not depend on E2(d); waiting would only idle the GPU. Nothing is read earlier than D4 allows |

**Record (not an amendment).** Proposal §7.2: "All arms train on T2-train minus T2-val". Every v2 arm,
pilot and E3 alike, trains on a dataset directory whose `labels_train.json` drops the T2-val sources'
windows and whose evaluation file holds T2-val's windows only (`python -m core.tools.v2_dataset`). Phase 4
trained on the full T2-train, T2-val included: a phase-4 checkpoint scored on T2-val is scored
**in-sample**.

## 14. P6 pilot implementation choices (2026-10-03, before any v2 model is trained) — O1–O7

Fixes how §4's read-out and D5 are computed (`python -m core.tools.v2_diagnostics`, one checkpoint
per call) and how batch 1 (D16) trains. No v2 checkpoint exists. **Authorized by the user on
2026-10-03; not reviewed by the advisor.**

| # | Choice | Why |
|---|---|---|
| O1 | **Guardrails on T2-val windows** of the v2 dataset dir (`core.tools.v2_dataset`: train = T2-train minus T2-val, evaluation file = T2-val's windows, labelled by the arithmetic that labelled T2-test, self-checked on the parent's T2-test windows). Micro with `score_norm=auto`, macro over two-class windows, clip oracle = `clip_constant_oracle`. Pass iff micro < oracle **and** macro ≥ micro **and** (A1–A3) micro ≥ A0's micro − 0.01, A0 = batch 1's A0 | Proposal §10.1 guardrails, on the decision set §4 names |
| O2 | Window-level AUC (max score vs window label) printed beside macro | §4: "clip-level AUC vs macro (the C14 collapse signature)" |
| O3 | `motion_share` (`ρ_u`) and `w_u_norm`: first / last / max over `metrics.jsonl`. `ρ_u` ≈ 0 is recorded, never fixed by `lr` or `c` (§4) | Architecture §5 |
| O4 | **Position probe on `V^t`** (`vis_feats`, KIP off): standardized ridge, α = 1, target `t/T` inside each T2-val window, 5 folds grouped by source video, out-of-fold R². Printed beside the same probe on the arm's input rows and beside A0's | Proposal §10.1: "linear, predicting `t/T`, R² reported"; an R² above A0's is the caveat, not a verdict |
| O5 | **Source-shortcut AUC = corpus separability**, as lesson C38 and `REPORT_T2_OPTICAL_FLOW_FOR_ADVISOR.md` §2.4 used the term: T2-val steps vs DoTA-dev steps. On `V^t`: the `core.eda` logistic probe, 5 folds grouped by source video. On `y^bin`: the score alone, direction-free `max(AUC, 1 − AUC)`. DoTA-dev rows go through the arm's own input at protocol B (plain `DoTA_s1_ncc[::3]` for A0, `apply --stride 3` cache otherwise) and are read **without labels**: no DoTA metric is computed, so D4 holds | The proposal names the measurement but not the two classes; CRN's stated purpose is to remove per-source appearance, and this project measured "source" as corpus every time |
| O6 | **D5:** A0 (v2 code, v2 dataset dir, seed 2099) vs the mean of the three finished phase-4 KIP-off checkpoints (s2024 phase 4; s2025, s2026 retrained under D12), all scored by O1 on T2-val. Pass iff micro **and** macro are each within 0.02. **Known bias:** the references trained on the parent dir, so T2-val is in-sample for them and they are expected to read higher; a FAIL with A0 below them blocks P7 until explained, and is never fixed by changing A0 | D5 as written (§2); the bias is stated before the number |
| O7 | **Batch 1 training (D16):** A0 and A1 on the v2 dataset dir, seed 2099, phase 4's stage-2 recipe (`train.num_epochs=20`, `train.amp=true`, `model.score_head_kernel=3`, `loss.mil_topk_pct=5`, `kip.enabled=false`), KNN cache rebuilt on the v2 dir. A1's input: `build_v2_inputs fit --crn R2 --motion none` (statistics on T2-train minus T2-val), then `apply --stride 3` on `dota_dev` for O5 only. The J10 gate holds for every checkpoint read | Proposal §8 hyper-parameters = phase 4's; nothing is tuned |

## 15. Amendment 6 (2026-10-04) — after E2(d): what is read next, and how position is printed (G1–G7)

Written **after** E2(d)'s read-out (`outputs/v2/REPORTS/v2_E2d/`, `core/docs/v2/RESULTS_E2D.md`) and **before** any
D6, batch-2 or E3 number exists. **Authorized by the user on 2026-10-04; not reviewed by the advisor.** Nothing here
changes a decision already taken or any adoption rule of proposal §10: E2(d)'s pick stands as the rule made it.
What E2(d) taught is that on DoTA-CAP-dev **position alone** (`p`, cubic in `τ`) reaches **0.852** in-domain and
**0.847** in transfer, above `u` and above `[x;u]`, while E1's `t/N` ruler read 0.566 on the same clips: a monotone
ruler cannot see an accident that sits mid-clip. G6–G7 make every later motion number carry that reading.

| # | Choice | Why |
|---|---|---|
| G1 | **Encoder = `vit_s_k710_dl_from_giant` (V2-S)**, as N7 picked it (A3 transfer Δ B 0.1508 vs S 0.1325, gap 0.0183 < 0.02 → cheaper). **Recorded limitation, not acted on:** on the printed position-controlled read (`[x;u;p]` − `[x;p]`) B ranks above S in transfer (A2 +0.037 vs +0.018; A3 +0.029 vs +0.026), and on the T2 side (+0.137 vs +0.107). Switching to B now would be a rule chosen after the number (lesson 14) | N7 was committed in `7b1889a` before the first probe |
| G2 | **Full-T2 V2-S cache** = every source `build_v2_inputs.t2_all_sources` lists (train, T2-val, test), K's geometry (16 frames, every 3rd at 30 fps, causal, squash, s8), **the same cache dir** as the K-source S cache (manifest compared, C2). Rows must equal the CLIP cache's rows per source, or the notebook stops | `fit` bakes every source it lists; one cache, one manifest |
| G3 | **D6 / N11 as built:** `dota_cap extract --shuffle-seed 2024 --dota-ids-file dota_cap_dev --encoders V2-S` re-streams only those CAP videos, re-runs the L4–L5 pixel gate (a clip that now fails stops the notebook), and encodes with one fixed permutation of the 16 frame slots (`extract_video_features.shuffled_frame_order(2024)`, never the identity) applied to every window. Own cache `DoTA_CAP_s1_squash_shuf2024`, manifest with `frame_order` and `shuffle_seed`; an ordered cache in the shuffle slot (or the reverse) raises. Read by `core.tools.e2d_shuffle`: N3's in-domain probe (same folds, seed 2024, cluster bootstrap B = 10,000) on `u`, `with` and `with_p` for A2 and A3, ordered vs shuffled; Δ(ordered − shuffled) and Δ(shuffled `with` − CLIP-only) **printed**. **Reading fixed now:** if Δ(ordered − shuffled) on `with_p` has a CI that contains 0, the thesis may not call the stream's gain "motion" on DoTA-CAP; it is "a second (video) appearance encoder". It never moves G1 | N11 fixed *what*; this fixes the build and the sentence the number licenses |
| G4 | **Bake A2/A3:** `build_v2_inputs fit --motion V2-S` with `--crn none` (A2) and `--crn R2` (A3) on **the same v2 dataset dir as A1** (`train_ids_sha1` must equal A1's); `apply --stride 3` on `dota_cap_dev` from the stride-1 DoTA-CAP V2-S cache. `model.motion_dim = 384`. Batch 2 trains them with O7's recipe; the config of A2 must differ from A0's, and A3's from A1's, **only** in `v2.motion` and `model.motion_dim` | O7 for the motion arms |
| G5 | **Batch-2 O5 on `dota_cap_dev`** for all four arms (`v2_diagnostics --dota-split dota_cap_dev`): a motion arm has DoTA rows only there, so A0 and A1 are re-read on the same 569 clips (their T2-val side must reproduce batch 1). Batch 1's `dota_dev` O5 stays as read. O1–O4 unchanged; A2/A3's O1 margin uses batch 1's A0 | Same clips for the four arms; nothing labelled on DoTA (D4) |
| G6 | **E3 position reads, printed beside every DoTA-CAP-dev macro of A0–A3** (never decide, never move an adoption): (a) `p_T2` = the `core.eda` logistic probe on `position_features` (`τ, τ², τ³`) fitted on T2-train-minus-val **window** frames with their labels, scored on `dota_cap_dev` at protocol B — what a model could score by having learned T2's position prior alone; (b) `p_CAP` = E2(d)'s in-domain 0.852, as the ceiling of position on these clips; (c) per arm, the mean over clips of the Spearman correlation between `y^bin` and (a)'s score, and the paired Δ of that correlation for A2 − A0 and A3 − A1 (cluster bootstrap). **Sentence rule:** when a motion contrast's Δ macro has CI > 0 **and** its Δ correlation has CI > 0, the read-out must print "this gain co-moves with the position prior" beside it | E2(d) showed position dominates DoTA-CAP; T2 windows carry little (0.575), so a trained model is cleaner than a probe — G6 shows whether it stayed so |
| G7 | **The position number** printed beside any DoTA protocol-B macro is now (a) of G6 (cubic, fitted on T2); E1's `t/N` ruler is still printed for continuity, labelled "monotone". A cubic in-domain `p` is never a decision quantity | `t/N` read 0.566 where cubic `p` reads 0.85 on the same clips |

## 16. Amendment 7 (2026-10-04) — D5 failed on T2-val as O6 predicted; how it is explained (H1–H4)

Written **after** batch 1's read-out (`outputs/v2/v2_pilot/`, `core/docs/v2/RESULTS_P6_PILOT.md`) and **before**
any T2-test number of a v2 checkpoint exists. **Authorized by the user on 2026-10-04; not reviewed by the advisor.**
D5 stays **FAIL** as read (micro Δ −0.1089, macro Δ −0.0073 vs the phase-4 mean on T2-val). O6 says such a FAIL
"blocks P7 until explained" and is never fixed by changing A0; this amendment fixes **what counts as the
explanation**, before the number. Evidence already in hand, printed not gated: all three references fail O1 on
T2-val (micro 0.757–0.774 > the clip oracle 0.6986) with window-level AUC 0.961–0.972 vs A0's 0.686 — the signature of
windows seen in training — while macro agrees within 0.0073.

| # | Choice | Why |
|---|---|---|
| H1 | **Code identity.** The phase-4 s2024 KIP-off checkpoint, re-scored by the v2 tree's `core.evaluate` exactly as phase 4 called it (`--set data.dataset=DADA2000_orig`, parent T2 dir, plain `DADA2000_orig` CLIP cache, `score_norm` default), must reproduce `REPORTS/DADA2000_orig_phase4/results_s2024_kip_off_t2.json` — micro 0.6230, macro 0.6334 — to \|Δ\| ≤ 1e-3 each, over the same 1,106 T2-test windows | Separates "v2 changed the v1 scoring path" from "A0 trained differently" |
| H2 | **Out-of-sample D5.** A0 (s2099) vs the mean of the same three references, all scored as H1 on **T2-test** — unseen by every one of them. Pass iff micro **and** macro each within `V2_D5_MARGIN` (0.02) | D5 as written, on a set where O6's known bias does not exist |
| H3 | **Verdict.** H1 ∧ H2 → D5 reads "**FAIL on T2-val, explained: in-sample bias of the references**" and no longer blocks P7. ¬H1 → the v2 tree changed v1's forward or scoring path: stop, debug, P7 blocked. H1 ∧ ¬H2 → A0's training differs (v2 dataset dir, KNN cache): stop, debug, P7 blocked. No outcome changes A0, its recipe or the T2-val read-out | Fixed before the number (lesson 14) |
| H4 | **Printed, never gated:** s2025/s2026 T2-test beside their lost phase-4 numbers (0.6164/0.6352, 0.6152/0.6057; retrained under D12, not bit-identical); each checkpoint's T2-val − T2-test micro gap (the in-sample inflation, references vs A0). **T2-test use:** this is a code/training identity check, not an arm comparison — A1–A3 are **not** scored on T2-test here, and no v2 decision reads T2-test | T2-test was read throughout v1; using it to check the baseline leaks nothing into a v2 choice |

## 17. Amendment 8 (2026-10-05) — the O1 guardrail fails an improving arm; collapse is redefined (Q1–Q7)

Written **after** batch 2's T2-val read-out (`outputs/v2/v2_pilot/batch2_readout.md`, `RESULTS_P6_PILOT.md`) and
**before** any T2-test number of A1–A3 and before any E3 number exists. **Authorized by the user on 2026-10-05; not
reviewed by the advisor. This amendment is post-hoc for the pilot** — it was written because two arms failed, and the
thesis must say so wherever Q5's verdict is cited. It is **pre-registered for E3** (Q6).

**What was read.** A2 (V2-S, no CRN) and A3 (V2-S + CRN R2), seed 2099, T2-val:

| arm | micro | macro | clip oracle | window AUC | O1 |
|---|---|---|---|---|---|
| A0 | 0.6586 | 0.6737 | 0.6986 | 0.6863 | PASS |
| A2 | 0.7120 | 0.6882 | 0.6986 | 0.7767 | **FAIL** (micro > oracle, macro < micro) |
| A3 | 0.7161 | 0.7052 | 0.6986 | 0.7637 | **FAIL** (micro > oracle, macro < micro) |

**Why O1 is defective.** O1's first two legs were written to catch C14 — a model that ranks windows (clip
classification) while going flat inside them. They test the *level* of micro, not that failure:

* "micro < clip oracle" is failed by **any** sufficiently good frame scorer — a perfect one scores micro = 1.0
  > every oracle. It cannot tell "learned the oracle" from "learned more than the oracle".
* "macro ≥ micro" fails whenever window-level separation improves faster than within-window ranking, even when the
  within-window ranking improves too.
* C14's measured signature is **two-sided**: TAD `m0` raised clip-level AUC 0.7687 → **0.9975** (+0.229) *while*
  macro **fell** 0.7578 → 0.6174 (`TAD_SETUP.md` §15.1). A2/A3 raise window AUC (+0.090 / +0.077 over A0) **and**
  raise macro (+0.015 / +0.032). The second half of the signature is absent.

| # | Choice | Why |
|---|---|---|
| Q1 | **The record stands.** A2 and A3 **FAIL O1 on T2-val, as registered.** Every table that cites the pilot prints this verdict beside Q5's. No recipe, `c`, `lr`, `σ_u` or checkpoint changes | Lesson 14: a rule is never re-read to make a number pass |
| Q2 | **O1′ (collapse rule), for an arm X ≠ A0, read against A0 of the same seed on the same set:** X is **collapsed** iff window AUC(X) > window AUC(A0) **and** macro(X) < macro(A0) − `V2_GUARD_A0_MARGIN` (0.01, reused). X **passes O1′** iff it is not collapsed **and** micro(X) ≥ micro(A0) − 0.01 (O1's third leg, unchanged). A0 itself has no reference and stays under O1 as registered | Encodes C14's two-sided signature; the margin is O1's own 0.01, not a new number |
| Q3 | **Retro-check of O1′ on recorded numbers (printed, done before Q4):** TAD `m0` vs `gate_t0` → clip AUC +0.229, macro −0.140 → **collapsed** (caught). T2 phase-4 references on T2-val (in-sample, §16): s2026 window +0.275, macro 0.6524 < 0.6637 → collapsed; s2024 / s2025 macro ≥ A0 → not collapsed. **Known limit:** O1′ does not detect memorization of seen windows (s2024/s2025 pass it). That job belongs to the dataset dir (§13 record, pending (aq)), which holds T2-val out of every v2 arm; O1′ must never be read on a set the arm trained on | Shows O1′ still fires on the one in-domain collapse this project measured, and states what it cannot see |
| Q4 | **Out-of-sample confirmation on T2-test.** A0–A3 (s2099) scored by `core.evaluate` on the parent dir's 1,106 T2-test windows (unseen by all four; A1–A3 through their own baked v2 input cache; **hard gate:** every T2-test source has a row file in that cache, else the notebook stops and the cache is re-applied, never refit) with O2's window AUC beside micro, macro and the T2-test clip oracle. **Gate:** A2 and A3 each pass O1′ against A0 **on T2-test**. Printed, never gated: O1 as registered on T2-test; Δ(X − A0) on every column ("one seed, not a result"). **T2-test use:** a guardrail check only. §16 H4's "no v2 decision reads T2-test" is amended to "no adoption reads T2-test"; E3's adoption rules (proposal §10.3) read DoTA-dev only and are untouched | Same move as Amendment 7: a set where nothing was chosen. T2-val already shaped Q2, so it cannot confirm it |
| Q5 | **Verdict per arm.** O1′ passes on T2-val **and** on T2-test → "O1 FAIL as registered; **not collapsed** under O1′ (post-hoc, Amendment 8), confirmed on T2-test" — the arm is a pilot survivor and P7 is not blocked by it. O1′ fails on either set → the arm is **collapsed**, it does not enter E3; if A2 and A3 both fail, the Motion Stream is dropped and E3 = {A0, A1}. No outcome changes an arm's recipe | Fixed before Q4's number |
| Q6 | **E3 guardrail = O1′**, replacing proposal §10.1's "T2 micro < the clip oracle, and macro ≥ micro" for A1–A3; "the guardrails hold" in §10.3's adoption rules means O1′ on T2-val, per seed, against A0 of that seed. A0 seeds stay under O1. O1's two retired legs, the clip oracle and window AUC are **printed** for every arm and seed | Pre-registered: no E3 number exists |
| Q7 | **Printed, never gated (pilot and E3):** per-epoch mean training loss per arm. In the pilot the CRN arms fit T2-train far harder than the raw ones (last-epoch mean `total` A0 0.742, A2 0.451, **A1 0.153, A3 0.136**; `mil` A1 0.027, A3 0.019). A1 passed O1 regardless, so this does not explain A2/A3's FAIL; it is a capacity/overfit caveat for reading CRN arms' T2 numbers, not a defect | Seen while reading batch 2; recorded so it cannot become a post-hoc reason later |

---

## 18. Amendment 9 (2026-10-05) — how E3 is computed (M1–M10)

Written **after** the pilot's last read-out (Q4 on T2-test, D6) and **before any E3 checkpoint exists**: no seed of
2024–2028 has been trained for any v2 arm, and no v2 arm has been scored on DoTA-dev with a label. **Authorized by
the user on 2026-10-06; not reviewed by the advisor.** **Provenance, stated plainly (cite it wherever E3 is cited):**
the text of M1–M10 was drafted 2026-10-05 and committed in `59ee17e` (2026-10-06 08:04 +0700) and has not been edited
since; the user reports running the 20 E3 runs from that upload with the notebook's unsigned-§18 guard disabled by
hand, and signed only after `e3_readout.md` existed. The rules were therefore **frozen in git before any E3 number
but signed after it**; no rule here was changed after the read-out. It fixes the computation of proposal §10.1 / §10.3 and §17 Q6;
it changes no adoption rule. Harness: `python -m core.tools.position_prior` (G6 a) + `python -m core.tools.e3_readout`;
runbook `colab/v2/e3_factorial.ipynb`.

**P7 checklist (§5), read on 2026-10-05:** splits pass `--check` (`7422975`) · protocol = B (E1) · CRN = R2 (D11) ·
encoder V2-S eligible (E2(d)) · D5 explained (§16) · every pilot arm passes the guardrails with no collapse — **A1 by
O1; A2/A3 by O1′ only (§17, post-hoc for the pilot)** · suite 896 / 0 fail (with this amendment's harness) · quality gate: ruff clean, mypy/pyright
clean after a test-only typing fix, bandit B614 ×8 in `core/tests/` only (pre-existing, `torch.load` of the tests'
own temp checkpoints). → **GO**, with the two caveats §17 Q5 and §15 G3 (D6: the stream is "a second (video)
appearance encoder", not "motion").

| # | Choice | Why |
|---|---|---|
| M1 | **Runs.** A0–A3 × seeds **2024–2028** = 20 runs, each with O7's recipe on the v2 dataset dir; a run's config differs from its pilot twin (s2099) **only** in `train.seed` (asserted). Inputs = the pilot's baked T2 caches (`v2_inputs/A{1,2,3}_T2`, seed-independent, `train_ids_sha1` == A1's). The checkpoint read is `checkpoint_last.pt` after 20 epochs and must pass J10. A crashed run is re-run **with the same seed**; a seed is never replaced or added. The pilot seed 2099 is never pooled into E3 | §10.1: n = 5, paired by seed, no seeds added after a result |
| M2 | **Guardrails per seed (Q6).** `v2_diagnostics` per arm × seed on T2-val, `--a0-diag` = A0's diag **of the same seed**, `--dota-split dota_cap_dev` (O5). For X ∈ {A1, A2, A3}, "**the guardrails hold**" ⇔ X passes O1′ on **all 5** seeds. A0 is printed under O1 per seed and decides nothing (it is the fallback arm). O1 as registered, the clip oracle and window AUC are printed for every arm × seed | One collapsed seed is a collapse; "most seeds" would be a rule chosen to pass |
| M3 | **DoTA scoring.** `protocol_b_eval` per arm with its 5 checkpoints named `s<seed>`: **A0 and A1 on `dota_dev`** (A1 through the pilot's `dota_dev` bake) and **A0–A3 on `dota_cap_dev`** (A1 through the same `dota_dev` bake — CRN is per clip, so restricting the clip set changes no row; A2/A3 through the pilot's CAP bakes). Each read-out's `global_steps` must equal M1's J10 steps. `protocol_b_eval` additionally writes the seed-averaged native scores and labels (`clip_scores.npz`), read by M6 only | D13: motion contrasts on DoTA-CAP-dev with all four arms on the same clips; CRN and `F` on full DoTA-dev |
| M4 | **Decision interval (§10.1).** For a contrast X − Y on a set: Δ_s = macro_s(X) − macro_s(Y) from `per_run_macro`, paired by seed name; interval = mean ± 2.776 · SD / √5 (SD with n − 1). "Excludes 0" ⇔ both bounds strictly on the same side of 0; **for adoption** (§10.3) it must be the positive side (lower bound > 0). Contrasts: A1 − A0 on `dota_dev`; A2 − A0, A3 − A0, A3 − A1, A2 − `F`, A3 − `F` and the interaction (descriptive) on `dota_cap_dev`. When `F` = A0, X − `F` is X − A0 | Proposal §10.1 / §10.3 verbatim, D13 for the sets |
| M5 | **The free rule for A1 (`F`).** A1 passes ⇔ (i) mean Δ(A1 − A0) on DoTA-dev macro ≥ 0, **and** (ii) mean over seeds of T2-val macro(A1) − macro(A0) (M2's diags) ≥ 0, **and** (iii) **no reversal in the share bins**: no bin of `V2_SHARE_BIN_LABELS` in which the clip-level paired Δ(A1 − A0) on DoTA-dev (seed-averaged clip AUCs, cluster bootstrap, B = 10,000, seed 2024) has its 95 % upper bound < 0. Every bin counts, small ones included (a small bin's wide interval cannot reverse by noise alone) | §10.3 says "no reversal" without defining it; a point-estimate rule on four bins of 19–300 clips would fire on noise |
| M6 | **Printed, never decided on.** (a) The clip-level paired cluster bootstrap Δ for every contrast (§10.1). (b) Macro per share bin per arm (§10.1 breakdown). (c) Mean over seeds of the source-shortcut AUC (`V^t`, `y^bin`, on CAP), position R² (`V^t`) and `ρ_u`. (d) The monotone `t/N` ruler, labelled "monotone" (G7). (e) **G6**: `p_T2` = `core.eda`'s logistic probe on cubic `position_features` fitted on every **training window** of the v2 dataset dir with its frame labels (`v2_dataset.window_frame_labels`), scored at each DoTA clip's native frames; its macro on `dota_dev` and `dota_cap_dev`; `p_CAP` = 0.852 (E2(d)); per arm the mean over two-class clips of Spearman(seed-averaged `y^bin`, `p_T2`) (a clip with a constant score is dropped and counted); paired Δ of that correlation for A2 − A0 and A3 − A1 (cluster bootstrap). G6's sentence rule is applied by the tool | §10.1's required breakdowns and Amendment 6 G6, computed one way |
| M7 | **MDE.** E0b was never run (its v1 contrasts need phase-5 / learning-curve checkpoints whose `global_step` is unverified, pending (aj)). E3 prints instead **MDE_E3** = 2.776 · SD_pooled / √5, SD_pooled pooled over the four main contrasts' per-seed Δ (A1 − A0, A2 − A0, A3 − A0, A3 − A1; ≤ 16 df, correlated through shared arms), labelled "E3-internal, not E0b". A non-significant contrast is written "**not detectable at MDE_E3**", never "no effect" | §10.1 requires a printed MDE; this is the closest one that exists without new runs |
| M8 | **Wording the tool prints.** A significant motion contrast (A3 − A1 if A3 is adopted, A2 − A0 if A2 is) licenses "**the video stream (V2-S) improves DoTA-CAP (n/1397)**" — never "motion" (G3 / D6). An A3 − A0 gain alone licenses only "v2 as a whole improves DoTA-CAP". A1 − A0 is a full-DoTA-dev number and is never printed beside a DoTA-CAP one in the same table (D14) | The pre-registered claims of §10.3 under D6's verdict |
| M9 | **Sealed sets.** E3 reads T2-val, DoTA-dev and DoTA-CAP-dev only. DoTA-eval, DoTA-CAP-eval and T2-test stay unread until the Final step (§10.2), which is its own notebook after the adoption is recorded | §10.2 |
| M10 | **Reading order.** The notebook trains all 20 runs, then diagnoses (M2), then scores (M3), then calls `e3_readout` once. No contrast is printed before every run of every arm is finished and verified | Avoids a partial read steering the rest |

---

## 19. Amendment 10 (2026-10-06) — the Final step, and how position is controlled there (P1–P8)

Written **after** E3's read-out (`RESULTS_E3.md`; adoption recorded: **A3**, `F` = A0) and **before** any E3
checkpoint has been scored on a sealed set (DoTA-eval, DoTA-CAP-eval) or on T2-test. **Authorized by the user on
2026-10-06, before the Final notebook exists and before any sealed set is read by an E3 checkpoint; not reviewed by
the advisor** (the advisor brief `ADVISOR_BRIEF_E3.md` is sent with it; an objection becomes Amendment 11, not an edit here).

**Why it exists.** E3's G6 read printed `p_T2` — a cubic position probe fitted on T2 training windows, no pixels —
at **0.8220** on DoTA-CAP-dev, above every arm (A3 0.7576). A post-hoc read on dev (`RESULTS_E3.md` §5) adds each
arm's within-clip z-scored score to `w · z(p_T2)`: A3 − A0 shrinks **0.081 → 0.045 (w = 1) → 0.023 (w = 2)** and
stays > 0, while A2 − A0 falls to **+0.007 [−0.005, +0.019]**. That read was chosen after seeing dev, so on dev it is
post-hoc. This amendment fixes its Final version before the eval set exists for any E3 checkpoint.

| # | Choice | Why |
|---|---|---|
| P1 | **Adoption is closed.** A3 (`F` = A0), recorded in `RESULTS_E3.md`. Nothing read at Final re-opens it; a Final number that disagrees with dev is reported as selection optimism (§10.2), never as a reason to adopt another arm | §10.2: Final reports, it does not decide |
| P2 | **What is opened, once.** The 20 E3 checkpoints of M1 (`checkpoint_last.pt`, J10 = step 1740), no retraining, no seed added: **DoTA-CAP-eval** (560 clips) for A0–A3 under protocol B through the same bakes as M3; **DoTA-eval** (700 clips) for A0 and A1 only (A2/A3 have no V2-S features outside CAP, D13); **T2-test** (1,106 windows, `core.evaluate`, as §17 Q4) for A0–A3, every seed. `load_split(..., final=True)` is called by the Final notebook only | §10.2 "open T2-test and DoTA-eval once"; D13/D14 for which arm can be read where |
| P3 | **Printed per arm and set.** Seed-mean macro with its t95; the five §10.3 contrasts paired by seed (t95, descriptive); the clip-level paired cluster bootstrap Δ; macro per share bin; **optimism = dev − eval** per arm; `p_T2` and the monotone `t/N` on the same clips; DoTA micro and AP (DoTA-CAP-labelled, never beside 62.60, D14). T2-test: micro, macro, clip oracle, window AUC, O1′ per seed vs A0 of that seed. MCC/AUCMCC and efficiency are **not measured** in v2 and are listed as such | §10.1's reported set, minus what no harness computes — said, not hidden |
| P4 | **The position-controlled read.** Per arm X and clip c: `f_w(X)_c = z_c(ȳ_X) + w · z_c(p_T2)`, `ȳ_X` = seed-averaged `y^bin` at native frames (M3's `clip_scores.npz`), `z_c` = standardize within the clip (a constant series → 0), `p_T2` = G6 (a)'s probe, refit by `core.tools.position_prior` exactly as for E3 (same T2 training windows, same cubic features, seed `constants.SEED`; no DoTA label enters the fit), scored on the eval clips. Macro over two-class clips. **w ∈ {1, 2}, w = 2 primary.** Δ by clip-level paired cluster bootstrap (clusters = source video, B = 10,000, seed 2024). Reads: `f_2(A3) − f_2(A0)`, `f_2(A3) − f_2(A1)`, `f_2(A2) − f_2(A0)`, and `f_2(X) − p_T2` per arm. Computed on DoTA-CAP-eval and re-printed on DoTA-CAP-dev beside it | w = 2 is where, on dev, A0 + position ≈ position alone (0.819 vs 0.822): the weakest arm carries nothing beyond position, so any arm's surplus there is content. **Chosen on dev — stated, and the reason eval is the test** |
| P5 | **Sentence rule (printed, never moves P1).** (a) `f_2(A3) − f_2(A0)` lower bound > 0 on DoTA-CAP-eval → "**A3's gain over A0 survives the T2 position prior**"; else "A3's gain is not separable from the T2 position prior at w = 2". (b) The same for `f_2(A3) − f_2(A1)` with "the video stream's gain on top of CRN". (c) `f_2(A3) − p_T2` lower bound > 0 → "A3 carries signal beyond position"; else "A3 adds nothing measurable to position". Every DoTA-CAP macro in the thesis carries `p_T2` beside it | Fixes the sentence before the number (lesson 14), as G3 did for D6 |
| P6 | **Why `p_T2`, not `p_CAP`.** `p_T2` is fitted on T2 training windows only — the prior a T2-trained model could have absorbed. `p_CAP` is fitted on DoTA-CAP labels; fitting it on CAP-eval would be in-sample. `p_CAP` = 0.852 (dev) stays a printed ceiling only | G6 (a)/(b) |
| P7 | **Harness gate before anything sealed is read.** The Final tool recomputes P4 on DoTA-CAP-dev from E3's `clip_scores.npz` and must reproduce `RESULTS_E3.md` §5's **means** to 1e-4 (the means are deterministic; the scratch intervals used B = 5,000 and another seed, so only means are gated). Fail → stop, no sealed set is opened | The dev numbers came from a scratch script; the Final number must come from the tool that is tested |
| P8 | **What E3 still may not say.** Unchanged: never "motion" (G3 / D6); "v2 as a whole" for A3 − A0; "the video stream (V2-S)" for A3 − A1; A2/A3 carry "O1 FAIL as registered; not collapsed under O1′" (§17 Q5); §18's provenance sentence | Carried forward so the Final write-up cannot drop them |

---

## 20. Amendment 11 (2026-10-06) — five DoTA-eval clips have no features (R1–R4)

Written **after** the first Final attempt stopped in its preflight and **before** any score, label, position prior or
metric on a sealed set was computed. **Authorized by the user on 2026-10-06, before the Final notebook is re-run and before any
sealed-set number exists; not reviewed by the advisor.**

| # | Choice | Why |
|---|---|---|
| R1 | **The fact.** `dota_eval` was frozen (`7422975`) from `metadata_val.json`'s 1,402 clips; the CLIP caches hold 1,397. The five missing clips — `TNZv-NBcV5U_002389`, `TNZv-NBcV5U_002660`, `W6YrlYyWguc_005597`, `W6YrlYyWguc_005927`, `nADqn-DZ-Dc_000075` — were lost in v1 to a FUSE unzip that wrote empty folders (`REPORT_KIP_MSAD_DOTA_PREVAD.md`, coverage note: every v1 DoTA number is on 1,397). DoTA pixels are gone, so they cannot be re-extracted. All five fall in `dota_eval`; `dota_dev`, `dota_cap_dev` and `dota_cap_eval` have **0** missing, so E3 is untouched | Measured on the local `DoTA_s8_ncc` and on Drive's `DoTA_s1_ncc` (the failed staging) |
| R2 | **DoTA-eval is read on 695 / 700 clips.** The five ids are fixed in `constants.V2_FINAL_DOTA_EVAL_NO_FEATURES`. `protocol_b_eval` / `position_prior` drop them only with `--final --exclude-featureless` on `dota_eval`, and **refuse** a listed clip that has a CLIP file, a listed clip not in the split, and any other featureless clip. The notebook asserts the featureless set equals the list before anything is opened. The read-out prints "695 / 700" and the ids | A fixed list, checked against the disk, cannot become a choice made after a score exists |
| R3 | **What the failed attempt read.** It wrote `v2_final/OPENED.json` (the marker of §19 P2) and loaded the eval id lists, then failed staging the first featureless clip. No checkpoint was run on a sealed set, no label or `p_T2` was computed there, and — by the notebook's order, the error being in step 2 of 7 — nothing was written under `v2_final/REPORTS/` (the user's traceback, 2026-10-06). The marker stays (it dates the opening); the re-run resumes from it | The record of when the sets were opened is kept, not reset |
| R4 | **Nothing else changes.** DoTA-CAP-eval stays 560 / 560, T2-test 1,106 windows; P1–P8 stand. A DoTA-eval number is "DoTA-eval (695 / 700)" and is never put beside a full-DoTA or 62.60 number (already D14 for CAP; now also for DoTA-eval) | Same rule, one more set |

## 21. Nexar (2026-10-10) — span rule, constructions, endpoint, what is opened (D-N1–D-N10)

**Authorized by the user on 2026-10-10, before any Nexar feature, score or position prior exists; not reviewed by the
advisor.** Written after the N0 census (metadata only); D-N10 and the D-N9 split were added the same day, also before any
Nexar number. Frozen from here: a change is a numbered Amendment, never an edit. `nexar_test` is opened once, by
`nexar_eval --split nexar_test --final`, after the full seeds of D-N9. Plan `.project/plans/katvad-v2-nexar-feasibility.md`; census `core/docs/v2/NEXAR_SETUP.md` §1.1.

| # | Choice | Why |
|---|---|---|
| D-N1 | **Span.** A positive's frame is abnormal iff `t_alert ≤ t ≤ t_event + 1.4 s` (clipped to the video); negatives have none. 1.4 s = median (accident → abnormal end) over DADA's 1,945 annotated rows at 30 fps (`NEXAR_POST_EVENT_S`). Sensitivity Δ_post ∈ {0, 1, 2} s may be printed, never adopted | Nexar has no end time; the convention comes from the corpus v2 trained on. Measured alert → event median 1.43 s ≈ DADA start → accident 1.50 s, so `t_alert` is DADA's "abnormal start" |
| D-N2 | **Endpoint = shifted crops.** From each positive, an 8 s crop with the span centre at {0.1, 0.3, 0.5, 0.7, 0.9}; CRN re-referenced per crop; macro per placement, endpoint = mean over the videos that fit all five (739 / 750 on the census). Whole-video macro, micro, clip AUC are printed only | 730 / 750 events sit at 0.4–0.6 of their video; `-|t/T − 0.5|` alone scores 0.974 macro on `nexar_train` positives (annotation only). A crop moves the event, so the middle ruler wins only at 0.5 |
| D-N3 | **Protocol N-B.** Rows `s1[::3]` of the N2 caches (CLIP `_ncc`, V2-S squash, one decode, bit-exact to both reference transforms), whole item, scores interpolated to native frames, seed-averaged; bootstrap clusters = video. Frame rates are not resampled (31 / 1,500 below 29 fps, printed by `nexar_build`) | E1's protocol, one corpus over |
| D-N4 | **Text.** `data.dataset=Nexar` maps to T2's definition set; positives are `CarAccident`. No collision / near-miss split (the release has no such column) | Census: columns are file name, times, light, scene, weather only |
| D-N5 | **Two constructions, one recipe; `whole` is trained first (D-N10).** `whole` = every `nexar_train` video, video label (the 750-video negative pool included — the arm exists to measure it). `window` = T2's windows (20 rows at s8, hop 8, no per-video cap) from positive videos only; pre/post-event windows are the negatives. Both: E3 recipe (kernel 3, top-k 5 %, stage 2, KIP off, AMP), arms A0 and A3 (CRN R2 + V2-S, statistics fitted on `nexar_train` by `build_v2_inputs fit-ids`, one bake for both constructions) | The user's question (2026-10-10): does training on the uncut videos change what the model learns? The N-X gate for adding the negative pool to `window` is not part of this step. Order changed 2026-10-10 (user, before any Nexar feature existed): `whole` alone is the cheaper first step |
| D-N6 | **Budget.** Every run spends `NEXAR_STEP_BUDGET` = 1,740 optimizer steps (= E3's 20 × 87); epochs = budget / steps per epoch. `whole` sets `dvs.syn_max_num_clips = 3` (5 spliced 40 s videos exceed the 512-row cap and `truncate_sample` can drop the anchor) | Step-matched, so a difference is not "more updates"; the DVS cap is a construction necessity, recorded |
| D-N7 | **Seeds.** Pilot s2099 (`whole` × {A0, A3} = 2 runs), then 2024–2028 (`whole`: 10 runs; +10 for `window` iff D-N10 fires, which also gets its own s2099 pilot), paired by seed. J10 on every checkpoint | As E3 (M1) |
| D-N8 | **Second look.** DoTA-CAP-dev at protocol B with the Nexar statistics is printed for every arm. It was opened by E3: descriptive, never a decision | Lesson (aw): a reused set cannot decide |
| D-N9 | **Decision rule (written now, read on `nexar_test` once, after the full seeds).** **(a) D-N10 did not fire** (`whole` only): primary = per-seed endpoint (D-N2 mean) of `whole`/A3 − `whole`/A0, paired t95 over 5 seeds; "A3 helps on Nexar" iff the CI low > 0, "A3 hurts" iff the CI high < 0, otherwise "no difference shown". The E3 T2-trained A3 zero-shot endpoint (5 ckpts, seed-averaged) and the middle ruler per placement are printed beside. Nothing is claimed about windowing: without the `window` arm, "whole is enough" is **not** a reachable verdict. **(b) D-N10 fired** (both constructions): primary = per-seed endpoint of `window`/A3 − `whole`/A3, paired t95; checked in order: "window is needed" iff the CI low > 0; "whole is better" iff the CI high < 0; "whole is enough" iff the CI lies inside [−0.01, +0.01]; otherwise "undecided". Printed beside: A3 − A0 per construction, each placement, the middle ruler. `nexar_val` reads (pilot and full) never change this rule | Rule in git before the number |
| D-N10 | **Window trigger (pilot only, `nexar_val`, `core.tools.nexar_trigger`).** Read on the s2099 `whole` pilot; `window` is trained iff **any** fires, on seed-averaged point estimates: **W1 position** — `whole`/A3 crop macro at 0.5 minus that at 0.1 or at 0.9 > **0.05** (`NEXAR_TRIGGER_MAX_EDGE_DROP`; the zero-shot drop is printed beside, not gated); **W2 no gain** — `whole`/A3 crop endpoint ≤ the endpoint of E3's five T2-trained A3 checkpoints scored zero-shot on `nexar_val` with their own T2 statistics; **W3 clip classifier** — `whole`/A3 clip-level AUC ≥ **0.95** (`NEXAR_TRIGGER_CLIP_AUC`) **and** its crop endpoint < `whole`/A0's (C14). The verdict only schedules the `window` arm; it is not a result and is not re-read on the full seeds | User, 2026-10-10: train on uncut videos first (cheaper), cut only if the result is not OK. "Not OK" fixed here before any Nexar score exists (lesson 14). Known: W1 can fire from crop-edge effects (CRN reference, causal motion warm-up) rather than learned position — a false fire costs only the `window` arm |

**Signature:** signed by the user, 2026-10-10 (rules D-N1–D-N10 as written above; no Nexar feature or score existed).

## 22. E4-0 (2026-10-11) — A3-ign and the base rule (E1–E4)

**Authorized by the user on 2026-10-11, before any A3-ign number; not reviewed by the advisor.** First committed as
proposal v2.1 §11.4 / §11.1 / §7 in `b2b6512` and `2395e39` (both 2026-10-11 01:28 +0700); the E4-0 read-out was
committed in `f9fe5fa` (12:04 +0700). The v2.1 documents were removed on 2026-10-11 after SG-NM closed, and the E4-0
part of their text was moved here **unchanged**, except for section cross-references and the omitted SG-NM lines
(marked). E4-0 is a **post-Final v2 step on DoTA-CAP-dev only**: it does not re-decide §19 (Final read A3), and any
sealed-set number for A3-ign is a second look at an opened set. Read-out `core/docs/v2/RESULTS_E4_0.md`.

### E1. Why this step exists

- A3 was trained with `dvs_anchor_mode = span`, LaGoVAD's default, which v2 never chose on purpose (Gap S: `REPORT_V2_RESULTS.md` §6.3).
- On a spliced abnormal sequence this labels **every** step of the anchor window `y^p` = 1, although ≈ 54 % of an abnormal T2 window's steps are normal (C29). It is a candidate cause of Gap S that needs no new module.
- *[Three lines on the interaction with SG-NM omitted: SG-NM closed 2026-10-11 (its E4a gate failed); full text in
  `b2b6512`.]*

**What A3-ign changes, and what it does not.**
- **Only** spliced abnormal sequences are affected: about 30 % of abnormal anchors (θ = 0.7), i.e. ≈ 15 % of batch rows.
- On those, the anchor's steps leave the dense `L_dvs-sup` BCE; the fillers stay hard negatives (0).
- Positive pressure on the anchor still comes from `L_dvs-mil` (top-k inside the anchor, k = ⌊n_pos/4⌋) and from `L_MIL`.
- Unspliced windows and every normal sequence are trained exactly as in A3.
- No parameter, no code change: the flag exists, with tests, since 2026-09-09.

**Prior evidence, and why it does not decide.**
- `ignore` was measured twice, both times with v1 code, KIP off, n = 1 seed, and in regimes where the model could not localize:
  - DADA-2000 archive Phase 1: macro 0.5190 (`span`) vs 0.5134 (`ignore`); the score head spanned the whole clip (C27/C28).
  - TAD T-ladder: macro 0.6174 vs 0.6209 (+0.0035, below the metric's resolution); the model had collapsed into a clip classifier (C14).
- T2 is the first corpus here where the model localizes (kernel 3, macro ≥ micro), so the anchor label can matter for the first time. **No directional prediction is made on macro.**

**The run.** A3-ign × 5 seeds (2024–2028), 1,740 steps each, config-diffed against the E3 A3 run of the same seed (exactly one line differs: `loss.dvs_anchor_mode`). Read with E3's harness (§18) and the endpoints and diagnostics of E2, plus the `L_dvs-sup` / `L_dvs-mil` trajectories against A3's.

### E2. Endpoints, guardrails and predictions

- **Decision interval:** the paired t95 over the 5 seeds (2024–2028, paired with E3's A3 runs) of the per-seed
  DoTA-CAP-dev macro Δ (t = 2.776).
- **Position control:** `f_2 = z(ȳ) + 2·z(p_T2)`, the score the Final used (§19 P4), on seed-averaged scores, with a
  cluster bootstrap over source videos (B = 10,000). `p_T2` is printed beside every macro.
- **Guardrails** (T2-val, against A3): T2-val macro no more than 0.01 below A3's; the normal-window peak (mean top-k
  score on T2-val normal windows) rises by no more than 0.02; O1′ (§17).

| Prediction (stated before the run) | Drop or qualify if… |
|---|---|
| **The pre-anomaly level falls:** A3's 0.55 mean before the span falls, and the 0.86 inside does not | the inside mean falls with it → a global rescale, not a contrast; no Gap S claim |
| **More of the span is covered:** A3's 61 % argmax-in-span share rises | it does not rise → the whole-anchor label is not what shapes the peak |
| **No hold failure:** the tail-clip macro (277 of 569) does not fall | it falls → the change repeats the hold's failure mode |
| **In-domain positive pressure survives:** T2-val macro no more than 0.01 below A3's | guardrail fails: `L_dvs-mil` and `L_MIL` alone are too weak a positive signal |
| DoTA-CAP-dev macro: **no directional prediction** | — (the base rule, E3, decides) |

### E3. Base rule

**Base rule.** `B` = **A3-ign** if all of the following hold; otherwise `B` = **A3** (`span`):
1. the paired t95 of the (A3-ign − A3) DoTA-CAP-dev macro Δ lies **above 0**;
2. the `f_2` Δ point estimate is **≥ 0**;
3. the guardrails (E2) hold against A3.

- No tie clause, margin or tie-break is added after the number is seen (lesson 14).
- If `B` = A3-ign, A3-ign is also the new best adoptable free arm `F`, and every later costly arm must beat it.

**What the thesis may claim from E4-0:**
- **"Dropping the whole-anchor DVS label improves DoTA-CAP detection"** requires rule 1. The eval number is labelled a second look.
- **"… beyond the position prior"** requires the `f_2` Δ **interval** to exclude 0.
- **"The whole-anchor label causes (part of) Gap S"** requires all three mechanism rows (E2) for A3-ign to come out as predicted: the pre-anomaly mean falls while the inside mean does not, the argmax-in-span share rises, and the tail-clip macro does not fall.
- **A non-significant E4-0** is "not detectable at this size". It is not evidence that C29 is harmless on T2.

### E4. Operational definitions

**Operational definitions** (fixed now; implemented by `core.tools.e4_0_readout`):
- **Span reads** use the seed-averaged DoTA-CAP-dev scores, **min-max normalized per clip**, so a global rescale cannot move them. `ignore` lowers positive scores everywhere, so a raw-scale read would show a "falling pre-anomaly level" for free.
  - argmax-in-span: 1 if the clip's argmax is an anomalous frame;
  - pre mean: the mean over frames before the span's first frame, on clips that have such frames;
  - inside mean: the mean over anomalous frames.
  - A3's values are **recomputed by the same code**. The results report's 61 % / 0.55 / 0.86 came from an ad hoc script, so they are context, not the reference.
- **Each Δ** is A3-ign − A3 per clip, with a cluster bootstrap over source videos (B = 10,000, 95 %).
  - "falls" means the interval lies **below** 0;
  - "rises" means the interval lies **above** 0;
  - "does not fall" means the interval does **not** lie entirely below 0.
- **Tail clips:** two-class clips with ≥ w = 30 normal frames after the span. w is H2(a)'s T2-train median span, 3.02 s at 10 fps. The read-out refuses to run unless it finds H2(a)'s 277 clips.
- **Normal-window peak:** the mean over all-normal T2-val windows of the mean top-k `σ(y^bin)`, k = max(1, ⌊L/5⌋). These are the steps `L_MIL` pushes down.
- **Guardrails** compare seed means of T2-val macro and normal-window peak against A3, re-diagnosed with the same code. O1′ must pass on every A3-ign seed, against E3's A0 of the same seed.
- **T2-val span profile** (the in-domain replicate) is printed per seed, never gated.

## 23. Amendment 12 (2026-10-11) — the second look of A3-ign on the opened sets (S1–S8)

**Authorized by the user on 2026-10-11, before any sealed-set score of A3-ign exists; not reviewed by the advisor.**
§19 opened DoTA-CAP-eval and T2-test once (2026-10-06, `OPENED.json`) for A0–A3. E4-0 (§22) later adopted A3-ign on
DoTA-CAP-dev. This amendment fixes, before the read, how A3-ign is read on those already-opened sets. It is a
**second look at an opened set**, and every number it produces carries that label. Tool `core.tools.second_look_readout`;
runbook `colab/v2/e4_0_second_look.ipynb`.

| # | Choice | Why |
|---|---|---|
| S1 | **What is read.** A3-ign's five E4-0 checkpoints (seeds 2024–2028, step 1740) on **DoTA-CAP-eval** (560 clips, `protocol_b_eval --final`, input = the Final's A3 CAP-eval bake: same T2-train statistics, same CRN R2 + V2-S) and on **T2-test** (1,106 windows, `v2_guard_test`, A0 of the same seed as the O1′ reference). Nothing else: no DoTA-eval (A3-ign has the video stream; DoTA pixels are gone, Amendment 1), no other arm, no other setting | The only arm whose sealed numbers are missing is the one the project now adopts |
| S2 | **Nothing re-decides.** `B` = A3-ign stands whatever this read shows (§22 E3); §19's numbers for A0–A3 stay as recorded; no later step selects, tunes or seeds on these numbers | The eval set has been opened; a second selection on it would be fitting it |
| S3 | **Contrast.** A3-ign − A3, paired by seed with the Final's A3 runs, on DoTA-CAP-eval macro: the paired t95 of the per-seed Δ (t = 2.776). Printed only: clip-level Δ (cluster bootstrap over source videos, B = 10,000), A3-ign − A0, optimism (dev − eval) of A3-ign beside A3's | Same statistic as §22 and §19, so dev and eval read alike |
| S4 | **Position control.** `f_2 = z(ȳ) + 2·z(p_T2)` on seed-averaged scores, `p_T2` = the Final's DoTA-CAP-eval prior (fit on T2 only); Δ (A3-ign − A3) with the cluster bootstrap; `p_T2` macro printed beside every macro | §19 P4/P5 |
| S5 | **Sentences, by outcome.** t95 entirely above 0 → "Dropping the whole-anchor DVS label improves DoTA-CAP (n/1397) detection, on dev and on a second look at the opened eval set." t95 contains 0 → "On the second look, A3-ign's gain over A3 is not detectable at this size (dev: +0.040)." t95 entirely below 0 → "On the second look, A3-ign scores below A3: the dev gain did not hold. A3-ign stays the recipe by §22 (no re-decision); both numbers are reported." "… beyond the position prior (second look)" only if the `f_2` Δ interval lies entirely above 0 | Fixed before the read (lesson 14) |
| S6 | **T2-test guardrail.** O1′ against A0 on every seed; the paired t95 of T2-test macro Δ (A3-ign − A3, against the Final's A3 rows) is printed. A failure is reported as "guardrail failed on the second look"; nothing re-decides | §17 Q4–Q5 |
| S7 | **Integrity gates, before any number is printed.** A3-ign's eval steps == its E4-0 dev steps (1740, J10); its eval clips == the Final's A3 CAP-eval clips; the Final's A3 eval steps == E3's; every T2-test read has 1,106 windows; A0's T2-test macro re-read here == the Final's within `V2_FINAL_DEV_TOL` (1e-4). Any failure stops the read-out | The pairing is only valid on the same clips, windows and checkpoints |
| S8 | **Once.** The notebook writes `SECOND_LOOK_OPENED.json` before scoring; a re-run resumes and skips every output that exists; nothing is re-scored with other settings | §19 "once means once" |

## 24. Amendment 13 (2026-10-14) — DF8: the position-swap read-out (SC-1; X1–X10)

**Authorized by the user on 2026-10-14 ("decide what you can and run"), before any crop is built or scored; not
reviewed by the advisor.** Plan `.project/plans/katvad-v2-df8-position-shift.md` (choices D1–D6 taken as
recommended); v2.2 proposal §R7 D8 (with the sign correction in X7). Tool `core.tools.position_shift`; runbook
`colab/v2/df8_position_shift.ipynb`. **No training. Nothing adopted, nothing re-decided** (v2's `B` = A3-ign stands).

| # | Choice | Why |
|---|---|---|
| X1 | **Set.** DoTA-CAP-dev only (569 clips). DoTA-CAP-eval is not read (D1) | dev is open; nothing is selected here |
| X2 | **Geometry** (D4, D5). Per two-class clip, from the native labels: span `[s, e)` (must be contiguous), `pre = s`, `post = N − e`. Eligible iff `min(pre, post) ≥ 10` frames (1 s). Budget `m = min(pre, post) + K`, K = 3; crop length `L = (e − s) + m`. Four arms: **full** (the whole clip), **control** (`round(m·pre/(pre+post))` frames before the span, the rest after), **early** (K before), **late** (`m − K` before, K after). Measured on the frozen labels before any score: **491 clips**, L median 57, centre early 0.30 / control 0.51 / late 0.70. The tool refuses another count | the only same-length swap the data allows (plan §2) |
| X3 | **Inputs per crop.** CLIP rows = `DoTA_s1_ncc[c0 : c0 + L]`. V2-S rows = the DoTA-CAP V2-S cache sliced the same way, **except** the protocol-B rows whose 1.5 s causal window reaches before `c0` (crop steps `k < 15`, `k ≡ 0 mod 3`, `c0 > 0`): those are **re-extracted** from the CAP frames with the extractor's own rule clamped at the crop's first frame (D2). CRN R2 and the motion scaling are applied to the crop's own stride-3 rows with E3's T2-train statistics (never refitted). Labels = the native labels sliced to the crop | what a clip starting at `c0` would see; CRN's reference must be the crop's (pending (ay)) |
| X4 | **Scoring.** Protocol B on the crop: `crop[::3]`, whole crop, interpolated to the crop's native frames, per-clip AUC, macro over the 491. Checkpoints: **A3-ign × 5 (E4-0) primary**; A3 × 5 and A0 × 5 (E3) printed (D3). Seed-averaged per frame; per-seed macros printed. `p_T2` = the T2-fit cubic (E3's fit, `position_prior`) at `t / L` of each crop, never refitted | §4 of the plan |
| X5 | **R1 (decides the SC-1 sentence).** Per clip, shifted AUC = mean of the early and late AUCs (D6). Δ = shifted(A3-ign) − shifted(`p_T2`), cluster bootstrap over source videos (B = 10,000, 95 %) | the mirror average cancels "aftermath vs pre-accident" content |
| X6 | **R2 (the model's own position cost).** For A3-ign: `early − control`, `late − control`, `early − late`, same bootstrap. "Position-sensitive" only if an interval excludes 0; never "position-invariant" | — |
| X7 | **Non-inferiority for the strong sentence (D8, sign corrected).** "… whereas the detector's score does" requires **both** `early − control` and `late − control` (A3-ign) to have lower bound **> −0.01**. D8 wrote "`control − early` … lower bound > −0.01", which would test whether moving *helps*; the intended test is that moving costs less than 0.01. Corrected here, before any read | a detector that loses > 0.01 when moved cannot be said to survive the swap |
| X8 | **Sentences (fixed now).** All carry "(DoTA-CAP-dev shift subset, n = 491; accident centre 0.30–0.70)". R1 above 0 **and** X7 holds → "The position prior's advantage is a property of where DoTA-CAP places its accidents; it does not survive a position swap, whereas the detector's score does." R1 above 0, X7 fails → "The prior's advantage does not survive a position swap, but the detector also depends on where the accident sits." R1 below 0 → "A position prior remains competitive even when positions are swapped; the detector's use of position is an open problem" (the "position is an artefact" argument is withdrawn). R1 contains 0 → "Once the accident is moved, neither the detector nor the position prior is ahead at this size." | lesson 14 |
| X9 | **Printed, never decided on.** R3 `full − control` (context cost); R4 `p_T2` per arm; A3 / A0 per arm; `f_2 = z(score) + 2 z(p_T2)` per arm; R1 by span-length bin (< 2 s, 2–4 s, ≥ 4 s); D2 sensitivity (the leaky cache slice instead of the re-extracted heads); full-clip macros of the 491 kept vs the 78 dropped clips | context for the sentence |
| X10 | **Integrity gates, before any number is printed.** (a) every checkpoint passes J10; (b) each model's **full** arm reproduces its saved protocol-B scores on DoTA-CAP-dev (E4-0 for A3-ign, E3 for A3/A0) within `V2_FINAL_DEV_TOL` per frame; (c) every re-extracted head is gated by re-extracting, with the same code, one unclamped window per shifted arm (`c0 + 15`) and comparing it to the cache row: min cosine ≥ 0.999; (d) X2's clip count. Any failure stops the read-out | the crop path must be the protocol-B path, and the re-extraction must be the extractor |

**Read-out (2026-10-15, appended after the run; the rule above is unchanged).** X10 (a)–(d) pass. Outcome **"R1 > 0,
X7 fails"**: R1 +0.1201 [+0.0974, +0.1413]; A3-ign early − control −0.1251, late − control +0.0744. Sentence as fixed in
X8. A post-hoc same-frame read (`position_shift same-frame`, printed only, decides nothing) is in
`core/docs/v2.2/RESULTS_DF8.md` §4.

## 25. Amendment 14 (2026-10-15) — SC-2: A3-B vs A3-ign, the adoption rule (Y1–Y11)

**Authorized by the user on 2026-10-15 ("commit đi rồi làm batch 3"), under round 7's standing authority ("decide what
you can and run"); written before any V2-B T2 feature beyond K's 300 sources is extracted and before any A3-B run;
not reviewed by the advisor.** Design: v2.2 proposal §3.4, §7.1, §8 (SC-2a/b), §R7 D5 and D9. Plan
`.project/plans/katvad-v2.2-sc2-a3b.md`. Tool `core.tools.sc2_readout`; runbook `colab/v2/sc2_a3b.ipynb`.

| # | Choice | Why |
|---|---|---|
| Y1 | **Arm.** A3-B = A3-ign (E4-0's recipe, `loss.dvs_anchor_mode=ignore`) with `v2.motion = vit_b_k710_dl_from_giant` and `model.motion_dim = 768`. Its config differs from E4-0's A3-ign of the same seed in **exactly** those two keys (asserted before any GPU time). Input cache `A3_R2_B`: `build_v2_inputs fit --crn R2 --motion B` on the same v2 dataset dir (manifest `train_ids_sha1` == A3's), `apply --stride 3` on `dota_cap_dev` from the DoTA-CAP V2-B cache that E2(d) read | one change: the encoder. D9: this re-opens N7 (a cost rule, not evidence of equivalence) |
| Y2 | **Control.** E4-0's five A3-ign runs (seeds 2024–2028), their protocol-B read and their diags, read from `v2_e4_0/`, **never retrained**. Pairing by seed | the deployed `B` |
| Y3 | **V2-B on T2 (SC-2a).** K's geometry (16 frames every 3rd at the assumed 30 fps, causal, squash 224², stride 8), the **same** cache dir as K's 300 V2-B sources (manifest checked, C2); every T2 source (train + T2-val + test, `t2_all_sources`) | what `fit` bakes; identical to how V2-S was extended (§15 G2) |
| Y4 | **Primary (span).** DoTA-CAP-dev macro, protocol B (`s1[::3]`, whole clip, native frames), 569 clips. Δ = A3-B − A3-ign | proposal §8 SC-2b |
| Y5 | **Trained-arm bar (§7.1).** (a) paired t95 over the 5 per-seed macro Δ: lower bound **> 0**; (b) cluster bootstrap (source videos, B = 10,000, 95 %) of the per-clip Δ on seed-averaged scores: lower bound **> 0**; (c) point = the paired mean **≥ +0.01** | as registered |
| Y6 | **`f_2`.** `z(score) + 2 z(p_T2)` per clip, Δ point **≥ 0** (E3's `p_T2`). Qualified by DF8: it holds for DoTA-CAP's placement only (pending (bn)) | proposal §8 |
| Y7 | **Guardrails.** Seed-mean T2-val macro(A3-B) ≥ A3-ign's − 0.01; **O1′** (§17 Q2) on every A3-B seed against E3's A0 of the same seed. The normal-window peak is printed, not gated (it is not in SC-2b's row) | proposal §8 |
| Y8 | **Decision, mechanical.** Y5 (a)(b)(c) + Y6 + Y7 all hold → **adopt A3-B**: it becomes `B`, the deployed model, and the base of any later arm; sentence "A larger frozen video encoder improves DoTA-CAP-dev detection." Otherwise **A3-ign stays**; sentence "A larger frozen video encoder does not improve DoTA-CAP-dev detection at this size" (never "V2-B is no better"). No tie clause | lesson 14 |
| Y9 | **Printed, never decided on.** Macros with `p_T2` beside them; clip-level raw Δ; `f_2` macros; per native-fps group (`r1` / `r3` / other, D7); T2-val span profile; source-shortcut and position R² of `v^t`; final-epoch losses; extraction wall time | context |
| Y10 | **DF8 forward (printed only), only if Y8 adopts.** `position_shift score --model A3B` with V2-B heads re-extracted for the crops, then `readout`'s macros per arm and `same-frame`. If A3-ign stays, DF8 already describes the deployed model and nothing is run | the position story is told for the deployed model; no GPU spent on a model that is not deployed |
| Y11 | **Integrity, before any number.** (a) J10 on every A3-B checkpoint; (b) Y1's config diff; (c) the bake manifest's `train_ids_sha1` == A3's and `width` == 512 + 768; (d) the V2-B T2 cache manifest == K's and every T2 source present; (e) protocol B covers the 569 clips for both arms with the same labels; (f) every diag reads the same `global_step` as protocol B. Any failure stops the read-out | the arm must differ from the control only by the encoder |

Sets: DoTA-CAP-dev decides; T2-val is the guardrail. Nothing sealed is opened; DoTA-CAP-eval and T2-test are not read
in SC-2.

**Status (2026-10-15): DEFERRED by the user before any run** (GPU cost). The rule stays frozen as written; if SC-2 is
ever run, it is run under Y1–Y11 unchanged. Until then `B` = A3-ign and no V2-B claim is made either way.

## 26. Amendment 15 (2026-10-21) — TAD zero-shot: does v2 transfer to a fixed camera? (T1–T10)

**Authorized by the user on 2026-10-21 ("Hãy tạo notebook để chạy"), written before any TAD VideoMAE feature exists and
before any v2 arm is scored on TAD; not reviewed by the advisor.** Tool `core.tools.tadz_readout`; runbook
`colab/v2/tad_zeroshot.ipynb`; plan `.project/plans/katvad-v2-tad-zeroshot.md`.

| # | Choice | Why |
|---|---|---|
| T1 | **Scope.** A generalization probe on TAD's official test protocol (`tad_test_anno.json`, 100 videos: 60 `Car Accident` + 40 `Normal`, fixed traffic cameras). It writes one sentence (T5) and one printed comparison (T6); **it never moves `B`, the recipe or any v2 choice**, whatever it reads | every v2 number so far is dashcam (T2, DoTA-CAP); TAD is the first fixed-camera read |
| T2 | **Protocol.** `core.evaluate --set data.dataset=TAD`, stride **8**, whole video, raw scores (`--score-norm auto` → none: TAD has normals), TAD's own definition (the definition-conditioned setting, as Gate T0). Stride 8 because it is (a) T0's protocol, so LaGoVAD is read on the identical rows, and (b) T2's 0.27 s/step if TAD runs at ~30 fps (assumption **A-T1**: TAD's fps is not recorded; E1's rate-matching logic, which picked s3 for 10 fps DoTA, picks s8 here). Stride 3 is **not** read — no second endpoint | one protocol, chosen before a number |
| T3 | **Arms.** A0 (E3, CLIP only), A3 (E3, CRN + V2-S, `span`), **A3-ign** (E4-0, the deployed `B`); seeds 2024–2028, checkpoints at step 1740 (J10), A3-ign's config carries `ignore`. Inputs: A0 = Drive `clip/TAD_ncc` (the cache T0 was measured on); A3 / A3-ign = `build_v2_inputs apply --stride 1` with the stats of `cache/v2/A3_R2_S/DADA2000_orig` over `TAD_ncc` + the TAD V2-S cache (T4). CRN reference = each video's own R2 median, as each DoTA clip is its own reference | the five deployed runs, their control, and the arm that splits factorial from `ignore` |
| T4 | **V2-S on TAD.** Frames from Kaggle `nikanvasei/traffic-anomaly-dataset-tad`, the 100 test videos only, through a VM-local symlink farm whose names are zero-padded by the frames' **numeric** order (TAD_SETUP §4.3). `extract_video_features --encoder vit_s_k710_dl_from_giant --stride 8`; its manifest must equal the T2 V2-S cache's manifest on every key (C2) | the same encoder, geometry and transform the arm trained on |
| T5 | **Endpoint and sentence.** Macro AUC over TAD's two-class videos (expected 60). Δ = A3-ign − A0. (a) paired t95 (t = 2.776) over the 5 per-seed macro Δ; (b) cluster bootstrap (each video its own cluster, B = 10,000, 95 %) of the per-video Δ on seed-averaged scores. Both lower bounds > 0 → **"v2's components transfer to fixed-camera TAD beyond CLIP"**; both upper bounds < 0 → **"v2's components cost detection on fixed-camera TAD"**; otherwise **"no transfer gain of v2's components is shown on TAD at n = 5"**. No tie clause | the project's two-interval reading (§25 Y5 (a)(b)), without a minimum-gain bar: this sentence adopts nothing |
| T6 | **LaGoVAD, printed.** T0 (LaGoVAD's released checkpoint, `ckpts/best.ckpt`) is re-scored in the same run. Per-video bootstrap of A3-ign (seed-averaged) − T0: lower > 0 → "above", upper < 0 → "below", else "not separated". **Not a matched comparison**: T0 was trained on PreVAD (far larger, multi-domain), A3-ign on ~800 T2 windows of one dashcam corpus | the reader will ask; the caveat travels with the number |
| T7 | **Printed, never decided on.** Seed-averaged macro / clip-level AUC / micro per arm; per-seed macros; A3 − A0 and A3-ign − A3; the clip oracle and the length ruler beside every micro; `p_T2`, monotone `t/N` and middle rulers beside every macro; the O1′-shaped collapse print (clip-level up, macro down) vs A0 | position and length are printed beside every number (§14.5.1) |
| T8 | **Banned.** A TAD micro beside the published 89.56 (C8b, C14); tuning or selecting anything on TAD; reading stride 3 or any other protocol after this one (a new amendment, written before its number) | lesson 14 |
| T9 | **Integrity, before any number.** (a) T0 re-scored at micro **0.7912 ± 0.001** (TAD_SETUP §8.1) — else the labels or `TAD_ncc` drifted, stop; (b) 100 videos / 60 abnormal; (c) the Kaggle frames are T0's frames: for every test video `len(TAD_ncc) == ⌈N/8⌉` with `N` the folder's image count, and CLIP s8 re-extracted from the farm on 5 test videos matches `TAD_ncc` row for row at cos ≥ 0.999; (d) every run scored the same videos with identical labels; (e) J10 + the arm's config (`v2.crn`, `v2.motion`, `dvs_anchor_mode`) on every checkpoint; (f) the TAD bake's manifest names the A3 arm, `train_ids_sha1` == the T2 bake's, width 512 + 384, and V2-S rows == CLIP rows on every video. Any failure stops the read-out | the arm must be read on T0's rows and nothing else |
| T10 | **Record.** The read-out goes to `core/docs/v2/RESULTS_TADZ.md` whatever it says. Whether TAD enters the thesis report (which pools DoTA-CAP as the official DoTA) is the user's and the advisor's call, made after reading this table, not after reading the number | report scope is not a statistic |

## 27. Amendment 16 (2026-10-21) — a clip-level gate for TAD: `s = c(v) · minmax_v(y)` (G1–G10)

**Authorized by the user on 2026-10-21 ("làm thử B đi xem sao"), written before any gate is fitted or any gate score
exists; not reviewed by the advisor.** Tool `core.tools.tad_clip_gate`; runbook `colab/v2/tad_clipgate.ipynb`;
plan `.project/plans/katvad-v2-tad-clipgate.md`.

**Disclosure (read before the table).** §26 found v2 blind at clip level on TAD (clip-level AUC A3-ign 0.544, A0 0.578,
LaGoVAD 0.769). The fusion *form* below was then looked at **post-hoc on TAD-test** with LaGoVAD's scores as `c`
(c = T0 mean → micro 0.752, clip 0.800; T0 max → 0.720 / 0.769; A0 max → 0.630 / 0.578). That look fixed the form; it
did not fit anything. What this amendment adds and has not seen: a gate **trained by this project** on a corpus other
than TAD, and its read.

| # | Choice | Why |
|---|---|---|
| G1 | **Scope.** Exploratory repair of §26's clip-level weakness. It writes one sentence (G6); **it never moves `B`**, the recipe or any DoTA number | macro is invariant to the gate by construction (G3), so no DoTA claim can change |
| G2 | **Gate `c(v)`.** Logistic regression on `mean_t x_t` (raw CLIP, `*_ncc`, **no CRN**), standardized, `C = 1.0`, `class_weight = balanced`, `max_iter = 2000`, lbfgs; `c = σ(logit)`. Mean pooling, not max: the max of a longer video is larger by construction (TAD normals ≈ 4× longer), and a length-aware gate would score the length ruler | one fixed, length-blind, parameter-light head; nothing swept |
| G3 | **Fusion.** `s(v, t) = c(v) · minmax_v(y_A3ign(v, t))`. `c > 0`, so every video's ranking is unchanged: **macro(s) == macro(A3-ign) exactly** (hard check, |Δ| < 1e-9 per seed) | the two metrics are decoupled |
| G4 | **Training corpus: MSAD-train** (480 videos, 120 abnormal / 360 normal, video labels from `labels_train.json`), cache `clip/MSAD_ncc`. Chosen because (a) both classes come from the **same** sources and cameras, so a source shortcut cannot predict the label (C38's failure mode), (b) `MSAD_ncc` is the same extractor, transform and stride as `TAD_ncc` (C2). All 11 anomaly classes are used: the gate is "anomalous content present", not "accident". **No TAD video is ever fitted on.** PreVAD (larger, but a third cache with a 4-frame offset, PREVAD_SETUP §2) is not read here | the first corpus in this project that satisfies both C2 and C38 |
| G5 | **Dev gate, before TAD-test is read: TAD-train** (the frame folders `tad_test_anno.json` does not name; labels = folder class, cross-checked on the 100 test ids). Gate passes iff `c`'s clip-level AUC on TAD-train **≥ 0.60**; printed beside the clip-level length ruler on the same videos. Fail → stop: "the MSAD gate does not reach TAD", the test is not read | TAD-train is never fitted on, only read once as a go/no-go |
| G6 | **Endpoint and sentence (TAD-test, §26's 100 videos and score files, unchanged).** Δ = micro(s) − micro(A3-ign). (a) paired t95 (t = 2.776) over seeds 2024–2028, `s_k = c · minmax(y_k)`; (b) cluster bootstrap over videos (B = 10,000, 95 %) of the Δ on seed-averaged `y`. Both lower > 0 → **"a clip-level gate trained on MSAD restores clip-level discrimination on TAD"**; both upper < 0 → **"the MSAD gate costs on TAD"**; otherwise **"no clip-level gain of the MSAD gate is shown on TAD"**. No tie clause | §26's two-interval reading |
| G7 | **Printed, never decided on.** `c`'s clip-level AUC on TAD-test / TAD-train / MSAD-test (in-domain sanity); the clip-level length ruler on both TAD splits; micro of `s` beside §26's clip oracle 0.9226 and length ruler 0.8968; micro(s) − micro(T0) bootstrap ("not matched", T6's caveat); the post-hoc LaGoVAD-gate row (`c = mean T0`), labelled "form seen on this set" | the reader will compare; the caveat travels with it |
| G8 | **Banned.** Changing pooling, `C`, features, corpus or fusion after the dev gate or the test is read; fitting on any TAD video; a TAD micro beside 89.56 | lesson 14, C8b |
| G9 | **Integrity.** (a) §26's T9 still holds (the readout re-runs `tadz_readout`'s label check and T0 micro 0.7912); (b) every MSAD-train id has a feature file, both classes present, counts 120 / 360; (c) TAD-train ∩ TAD-test = ∅, ≥ 300 dev videos, both classes; folder-derived labels reproduce the 100 test labels exactly; (d) gate features are 512-wide; (e) G3's macro identity | the gate must see what G4/G5 name and nothing else |
| G10 | **Record.** The read-out goes to `core/docs/v2/RESULTS_TADZ.md` (a section after §26's) whatever it says. Whether it enters the thesis report is the user's call after reading it | report scope is not a statistic |

## 28. Amendment 17 (2026-10-22) — Nexar trains `B`'s recipe: both arms `dvs_anchor_mode = ignore` (D-N11–D-N15)

**Authorized by the user on 2026-10-22 ("A0 cũng dùng ignore. is_egocentric vẫn giữ false"), while N2 extraction was
running and before any Nexar score, checkpoint or position read exists; not reviewed by the advisor.** It amends §21's
D-N5 and D-N10 only. Runbook `colab/v2/nexar_train.ipynb`; plan `.project/plans/katvad-v2-nexar-feasibility.md`.

**Why now.** §21 was signed on 2026-10-10; E4-0 adopted A3-ign as `B` on 2026-10-11 (§22). §21's "E3 recipe" therefore
named a recipe `v2` no longer deploys. On `whole` the stale choice is also the C29 defect itself: the DVS anchor is a
whole ~40 s positive video, and `span` mode supervises every pre-event frame of it as abnormal.

| # | Choice | Why |
|---|---|---|
| D-N11 | **Both arms train with `loss.dvs_anchor_mode = ignore`** (`constants.NEXAR_DVS_ANCHOR_MODE`). §21's A0 / A3 now mean **A0-ign / A3-ign**: everything else is D-N5 unchanged (kernel 3, top-k 5 %, stage 2, KIP off, AMP, CRN R2 + V2-S fitted on `nexar_train`, step budget D-N6, `whole` DVS cap 3) | `B` = A3-ign (§22). A0 takes the same anchor so the A3 − A0 contrast of D-N9 stays one factor (CRN + motion) and is not confounded with the anchor |
| D-N12 | **`data.is_egocentric = false`, kept** (θ 0.7, δ_m 5 → `min(3, 5) = 3` clips on `whole`) | Every T2-trained v2 arm used `false`; its effect was never measured (`DADA_SETUP.md` §9 scope note). Switching it here would make Nexar-trained vs T2 zero-shot differ in corpus *and* DVS recipe. An ego arm, if ever wanted, is its own amendment beside A3-ign, never folded into this recipe |
| D-N13 | **W2's bar (D-N10) = E4-0's five T2-trained A3-ign checkpoints** (`outputs/v2_e4_0/A3ign/s2024–2028`, J10 at 1,740 steps) scored zero-shot on `nexar_val` with their T2 statistics (`cache/v2/A3_R2_S/DADA2000_orig`, the same bake E3's A3 used). D-N9(a)'s printed zero-shot endpoint is the same five checkpoints | The bar must be the deployed recipe trained on T2, so "Nexar training adds nothing" compares like with like |
| D-N14 | **Outputs are new paths**: runs `outputs/v2_nexar_ign/<c>/<arm>/s<seed>/`, reads `REPORTS/nexar_val_ign/`, `REPORTS/nexar_dota_cap_ign/`, zero-shot `REPORTS/nexar_val_ign/zeroshot_e4_0_A3ign/`. N2 features, N4 corpora, the A3 bake and the KNN caches are anchor-free and reused unchanged | A `span` run (none exists at signing) can never be mistaken for, or resumed into, an `ignore` run |
| D-N15 | **Nothing else changes.** D-N1–D-N4, D-N6–D-N8, D-N9's rules, D-N10's thresholds and the frozen split stand. No `span`-trained Nexar arm is run; a span vs ignore contrast on Nexar is not part of this step | One change, written before the number |

## 29. Amendment 18 (2026-10-25) — Nexar: `window` only, T2's five arms, one read on `nexar_test` + zero-shot (D-N16–D-N22)

**Authorized by the user on 2026-10-25 ("chạy chỉ window … trên các arm từ A0 -> A3-ign", "5 arm như T2",
"Amendment 18 mới", "chỉ eval theo flow bình thường … nexar test, tad, dada2000 (full video), DoTA (cũng full
video)"), before any 5-seed Nexar run and before `nexar_test` is opened; not reviewed by the advisor.** It amends
§21's D-N5, D-N7, D-N9 and §28's D-N11, D-N14, D-N15. Runbooks `colab/v2/nexar_train.ipynb` (train + `nexar_val`)
and `colab/v2/nexar_zeroshot.ipynb` (every read below); tools `core.tools.{nexar_eval, dada_whole_eval,
nexar_transfer_readout}`.

**Disclosure (read before the table).** The s2099 pilots were read on `nexar_val` (committed `0e8c8f7`,
`outputs/v2/v2/REPORTS/nexar_val_ignore/summary_pilot.md`; the Drive copies are deleted by the user so this run's
outputs are clean). D-N10 fired (W1, W2). On that single seed `window`/A3-ign 0.822 vs `window`/A0-ign 0.678 and
`whole`/A3-ign 0.582: the direction of D-N18's contrast was **seen on val, at n = 1, with A0 trained `ignore`**.
`nexar_test` has never been read by any arm. §21 D-N9(b)'s primary (`window` − `whole`) is **withdrawn**, not read:
no 5-seed `whole` arm is trained, so "window is needed" is never written from the test set.

| # | Choice | Why |
|---|---|---|
| D-N16 | **Arms = T2's five, `window` only**: A0 (CLIP), A1 (CRN R2), A2 (V2-S), A3 (CRN R2 + V2-S) with `loss.dvs_anchor_mode = span` (E3's recipe), and A3-ign (A3 + `ignore`, E4-0 = `B`'s recipe). Everything else is D-N5 / D-N6 / D-N12 (kernel 3, top-k 5 %, stage 2, KIP off, AMP, step budget 1,740, `is_egocentric = false`). A1 / A2 / A3 statistics are fitted on `nexar_train` by `build_v2_inputs fit-ids` (one bake per arm input; A3 and A3-ign share one). `constants.NEXAR_FULL_ARMS`, `NEXAR_FULL_ANCHOR`. Supersedes D-N11 and D-N15 | Row-for-row with the T2 ladder, so every Nexar-trained row has a T2-trained twin |
| D-N17 | **Seeds 2024–2028**, 25 runs, J10 on every checkpoint. Runs `outputs/v2_nexar_window/<arm>/s<seed>/`, reads `outputs/v2_nexar_window/{val,eval}/`, read-out `outputs/v2_nexar_window/REPORTS/` (`NEXAR_FULL_RUNS_DIR`). Supersedes D-N7, D-N14 | A new root: no pilot or `ignore`-only run can be resumed into it |
| D-N18 | **The one sentence (read on `nexar_test`, once).** Per-seed D-N2 crop endpoint (`nexar_eval`'s `per_run_crop_endpoint`) of `window`/A3-ign − `window`/A0, paired t95 over the 5 seeds (t = 2.776): **"A3-ign helps on Nexar"** iff the interval lies above 0, **"A3-ign hurts on Nexar"** iff below 0, otherwise **"no difference shown at n = 5"**. The video bootstrap is printed beside. No tie clause | D-N9(a)'s form, on the arms that exist |
| D-N19 | **Printed, never decided on**: every other arm and contrast (A1 − A0, A2 − A0, A3 − A0, A3 − A1, A3 − A2, A3-ign − A3) on every set; whole-video macro / micro / clip AUC; the middle ruler per placement; the same five **T2-trained** arms on `nexar_test` (E3 / E4-0 checkpoints, their own T2 statistics) and Nexar-trained − T2-trained per arm (video bootstrap only — different runs cannot be paired by seed) | Lesson (aw): only the pre-registered contrast decides |
| D-N20 | **Zero-shot sets (Nexar-trained arms, each input baked with the Nexar statistics, `build_v2_inputs apply`), read the ordinary way, no gate:** (a) **TAD** — §26's protocol unchanged (100 test videos, s8, `core.evaluate --set data.dataset=TAD`, raw scores); (b) **DADA-2000 whole videos** — the source videos behind T2-test's windows, every s8 row in one forward, labels = the original annotation's span at s8 (`kill_switch_probe.source_labels`), macro over two-class sources, micro; `core.tools.dada_whole_eval`; (c) **DoTA whole clips** — protocol B (`s1[::3]`, whole clip, native frames) on DoTA-CAP-dev **and** DoTA-CAP-eval, pooled to the 1,129 clips (macro = per-clip mean, bootstrap clusters = source video; per-seed macro = the two-class-count-weighted mean of the splits) | TAD / DoTA as the thesis already reads them; DADA whole because Nexar never saw a DADA window and a window set would re-cut the videos T2 was built from |
| D-N21 | **No clip-level gate** (§27) on any of these reads | The user's choice, 2026-10-25: ordinary scores only |
| D-N22 | **Once.** `nexar_test` is unsealed by `nexar_eval --final` only after all 25 Nexar checkpoints and the 25 T2 checkpoints pass J10 and their config check; `eval/nexar_test/NEXAR_TEST_OPENED.json` is written before the first read and every read skips an output already on disk. DoTA-CAP-eval was opened by §19 (a second look, descriptive). Nothing is re-scored with other settings after any number is seen | §21's "once" kept |

**Signature:** signed by the user, 2026-10-25 (D-N16–D-N22 as written above; no 5-seed Nexar run and no `nexar_test`
score existed).

## 30. Amendment 19 (2026-10-25) — Nexar: is CRN's crop cost the CRN reference unit? (C1–C9)

**Authorized by the user on 2026-10-25 ("làm 0, 1 trước đi": step 1 = the CRN diagnostic), after §29's read-out and
before any number of this section exists; not reviewed by the advisor.** A diagnostic: it decides no arm, never moves
`B`, and reads `nexar_val` only. Tools `core.tools.nexar_eval --crn-reference source`, `core.tools.crn_cost_diag`;
runbook `colab/v2/nexar_crn_ref.ipynb`. Results go to `core/docs/v2/RESULTS_NEXAR.md` §5.

**Disclosure (read before the table).** §29's read-out (`outputs/v2/v2_nexar_window/REPORTS/transfer_readout.md`)
showed the CRN arms losing on every set (A3 − A2 −0.061 … −0.084, A1 − A0 −0.024 … −0.054) and fitting their training
bags ~10× tighter (last-step `mil` A1 / A3 0.02–0.09 vs A0 0.35–0.49, A2 0.18–0.29). The hypothesis tested here (H1)
was written after that. Before this section was written, C7's share table was also run exploratorily on the saved
scores: on `nexar_val`, A3 − A2 is **−0.043 on whole videos** (event share of the reference ≈ 0.07, as in training)
and **−0.066 on crops** (≈ 0.35); on DoTA / DADA / TAD the per-clip cost does **not** grow with the event share
(Spearman +0.02 … +0.13, the cheapest quartile is the highest share). Those reads predict C6's fraction `f` ≤ ≈ ⅓.
They are printed in C7 as seen, not re-chosen.

| # | Choice | Why |
|---|---|---|
| C1 | **Question (H1).** Training bakes each window's CRN reference (R2, per-dimension median) over its **source video** (~40 s, event ≈ 2–5 % of the rows); `nexar_eval` re-references each 8 s **crop** by itself (event ≈ 35 %). Does that train / eval mismatch of the reference unit explain the CRN arms' crop cost? | The one mechanism that §29's protocol creates and T2 does not (T2 sources are short, so a T2 reference already carries the event) |
| C2 | **Set and checkpoints.** `nexar_val` only (`nexar_test` stays read once, D-N22). §29's Nexar-trained checkpoints, 5 seeds each, J10 = 1,749. Primary arm **A3**; A1 and A3-ign printed. Statistics = each arm's §29 `fit-ids` cache, unchanged. No retraining | The test set's one read is spent; val is unsealed and already read with the item reference |
| C3 | **Manipulation.** `--crn-reference source`: an item `[start, end)` is cut from its source baked whole on the item's own stride grid, `bake_rows(x1[start % 3 :: 3], u1[start % 3 :: 3])[start // 3 : …]` — the reference (CLIP and, for A3, the motion stream) is the whole video's R2 at the eval stride; every other row of the input is unchanged. `--crn-reference item` (the default) is §29's path. A no-CRN arm refuses `source` | The training reference at the eval stride; no new formula (one `bake_rows`) |
| C4 | **Gate (before any number is printed).** (a) every checkpoint's `global_step` = the `window` plan's 1,749 (notebook, from `build_report.json`) and both reads name the same checkpoints and steps; (b) whole-video items are phase-0 slices of the whole source, so their per-item AUCs under `source` must equal the saved §29 `val/<arm>/item_aucs.json` within **1e-3** (`CRN_DIAG_PARITY_TOL`) on every item; (c) the crop item set is identical. Any failure stops the read | The `source` path must be the `item` path wherever the two coincide |
| C5 | **Primary.** `Δ_ref` = per-seed crop endpoint (`per_run_crop_endpoint`) under `source` − under `item`, A3, paired t95 over the 5 seeds (t = 2.776). Fraction closed `f = mean Δ_ref / (E(A2) − E(A3, item))`, seed means of the per-seed endpoints, all `nexar_val` | Same endpoint as D-N2 / D-N18 |
| C6 | **Sentences (checked in order).** (1) t95 low > 0 and `f` ≥ **0.5** (`CRN_DIAG_MOST`) → **"the reference unit explains most of CRN's crop cost on Nexar"**; (2) low > 0 and `f` < 0.5 → **"the reference unit explains part of it"**; (3) high < 0 → **"the training reference costs more on crops; the reference unit does not explain the cost"**; (4) otherwise **"the reference unit does not explain CRN's crop cost (no change shown at n = 5)"**. No tie clause | Fixed before the read; C-disclosure predicts (2) or (4) |
| C7 | **Printed only.** A1 (`f` against A0) and A3-ign (`f` against A2) under the same rule's numbers; per placement (0.1 … 0.9) for every arm; `crn_cost_diag share`: per-clip CRN cost (A3 − A2, A1 − A0, A3-ign − A2; Nexar-trained) against the clip's event share on DoTA-CAP dev / eval, DADA whole, TAD, and Nexar whole vs crops (val and the already-read test) — quartile bins and Spearman | The share table is the exploratory read disclosed above |
| C8 | **Reading.** Whatever C6 writes, CRN is not rescued: a deployed model scores a clip without its source video. (1)/(2) explain the cost and say which half of it is protocol; (3)/(4) leave the cost with CRN itself (H2: centring removes absolute scene content the Nexar task needs; untested here) | So the result cannot be read as "fix the eval and keep CRN" |
| C9 | **Nothing moves.** `B` stays A3-ign; §29's sentence stands. A2-ign (proposal for the next amendment) is not authorized by this section | One question per amendment |

**Signature:** authorized by the user, 2026-10-25 (C1–C9 as written above; no `--crn-reference source` score existed).

## 31. Amendment 20 (2026-10-26) — can `B` drop CRN? A2-ign vs A3-ign on T2, non-inferiority (Z1–Z10)

**Authorized by the user on 2026-10-26 ("ghi kết quả crn_ref rồi viết luôn amendment 20 kèm notebook"), after §30's
read-out and before any A2-ign run; not reviewed by the advisor.** Tool `core.tools.crn_drop_readout`; runbook
`colab/v2/crn_drop_a2ign.ipynb`; plan `.project/plans/katvad-v2-crn-drop.md`. Results go to
`core/docs/v2/RESULTS_CRN_DROP.md`.

**Disclosure (read before the table).** This rule is written knowing: (i) E3 on T2 — A3 − A2 ≈ −0.007 by seed / +0.004
by clip on DoTA-CAP-dev macro (E3 §2), but beyond position (w = 2) A3 − A0 **+0.023 [+0.013, +0.032]** vs A2 − A0
**+0.007 ∋ 0** on dev and **+0.004 ∋ 0** on CAP-eval (Final); CRN's one mechanical effect, `V^t` source-shortcut AUC
0.9998 → 0.569; (ii) Nexar §29: CRN costs on every set when trained on Nexar (A3 − A2 −0.06 … −0.08); (iii) §30: the
CRN reference unit explains only `f` = 0.17 of that cost. (i) predicts that Z4 passes and Z5 fails, i.e. **A3-ign
stays**; the rule is not chosen to avoid that. No A2-ign number (any `dvs_anchor_mode = ignore` arm without CRN on
T2) exists.

| # | Choice | Why |
|---|---|---|
| Z1 | **Arm.** A2-ign = E4-0's A3-ign (`loss.dvs_anchor_mode = ignore`) with `v2.crn = none`. Its config differs from E4-0's A3-ign of the same seed in **exactly** `{v2.crn}` (asserted before any GPU time). Inputs = E3's **A2** caches as baked (`A2_S`: T2 and `DoTA_CAP_s3_from_s1`); their manifest must read `crn = none`, `motion = V2-S`, width 512 + 384 and the same `train_ids_sha1` as A3's. No extraction, no new bake | one change: CRN. A2's input is the A3 input without the centring |
| Z2 | **Control.** E4-0's five A3-ign runs (seeds 2024–2028), their protocol-B read and diags, read from `v2_e4_0/`, **never retrained**. Pairing by seed | the deployed `B` |
| Z3 | **Primary set and endpoint.** DoTA-CAP-dev macro, protocol B (`s1[::3]`, whole clip, native frames), 569 clips. Δ = A2-ign − A3-ign | the set every `B` decision was taken on (E3, E4-0, §25) |
| Z4 | **Detection, non-inferiority.** (a) paired t95 over the 5 per-seed macro Δ: lower bound **> −0.01**; (b) cluster bootstrap (source videos, B = 10,000, 95 %) of the per-clip Δ on seed-averaged scores: lower bound **> −0.01** | margin = proposal §7.1's bar (as §25 Y5(c), Y7); a removal must show it costs < the smallest gain this project would adopt |
| Z5 | **Beyond position, non-inferiority.** `f_2 = z(score) + 2 z(p_T2)` per clip (E3's `p_T2`), Δ cluster bootstrap lower bound **> −0.01**. Qualified by DF8: it holds for DoTA-CAP's placement only (pending (bn)) | the one claim CRN carries on T2 (E3 §3, Final); stricter than §25 Y6's point because here the burden is on the removal |
| Z6 | **Guardrails.** Seed-mean T2-val macro(A2-ign) ≥ A3-ign's − 0.01; **O1′** (§17 Q2) on every A2-ign seed against E3's A0 of the same seed. The normal-window peak is printed, not gated | as §25 Y7 |
| Z7 | **Decision, mechanical.** Z4 (a)(b) + Z5 + Z6 all hold → **drop CRN**: A2-ign becomes `B`, the deployed model and the base of any later arm; sentence "Removing CRN does not cost DoTA-CAP-dev detection, in total or beyond position (non-inferiority margin 0.01, n = 5)." Otherwise **A3-ign stays**; sentence "Removing CRN is not shown to be harmless on DoTA-CAP-dev at n = 5." The failing checks are listed. A failure never licenses "CRN helps" (that would need A3-ign − A2-ign > 0, which this rule does not test). No tie clause | lesson 14; a simplification is adopted on proof of no harm, not on no proof of harm |
| Z8 | **Printed, never decided on.** Macros with `p_T2` beside them; clip-level raw Δ; `f_2` macros; per native-fps group (`r1` / `r3`, D7); E3's A2 (`span`) macro beside A2-ign (the `ignore` effect without CRN); T2-val span profile; source-shortcut AUC and position R² of `v^t` (CRN's mechanical effect); final-epoch losses | context |
| Z9 | **Forward.** If Z7 drops CRN: every A3-ign read on an opened set (E4-0 second look, DF8, TAD §26–§27, Nexar) describes A3-ign, not the new `B`; reading A2-ign on CAP-eval, T2-test, TAD or Nexar is a new amendment. If A3-ign stays: nothing moves; Nexar's "CRN costs when trained on Nexar" and §30 stay as recorded | one question per amendment; nothing sealed is opened |
| Z10 | **Integrity, before any number.** (a) J10 on every A2-ign checkpoint (1,740 steps); (b) Z1's config diff on the planned and the written config; (c) Z1's cache manifests; (d) protocol B covers the 569 clips for both arms with identical labels; (e) every diag reads the same `global_step` as protocol B. Any failure stops the read-out | the arm must differ from the control only by CRN |

Sets: DoTA-CAP-dev decides; T2-val is the guardrail. DoTA-CAP-eval, DoTA-eval and T2-test are not read.

**Signature:** authorized by the user, 2026-10-26 (Z1–Z10 as written above; no A2-ign run existed).
