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
