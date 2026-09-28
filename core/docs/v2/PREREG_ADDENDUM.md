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
