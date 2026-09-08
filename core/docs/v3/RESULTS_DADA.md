# DADA-2000 campaign — the in-domain number is a clip-level number

**Measured:** 2026-09-06, from the saved score curves in `outputs/v3/DADA2000/**`
plus the archived MSAD/DoTA arms in `outputs/v3/{MSAD,DoTA}_*`.
**Runbook:** `core/docs/v3/setup/DADA_V3_SETUP.md`; data runbook `core/docs/DADA_SETUP.md`.
**Notebook that ran it:** `collab/DADA/v3/train.py`.
**Answers:** the pre-registered question in `DADA_V3_SETUP.md` §0, and the user's
question "why is my DoTA transfer 0.6x when SimpleTAD reports 0.80?".

> **One line.** The DADA-2000 in-domain **0.86 micro AUC is not a frame-level
> result** — a model emitting one *constant score per clip* scores **0.9086** on
> this test set, a detector reading only the clip's **frame count** scores
> **0.8654** (lesson **C28**), and every arm's `auc_macro` sits at **0.44–0.57, i.e. chance**.
> The cause is structural: DADA's median clip is **9 stride-8 frames** and the
> score head is a single `Conv1d(kernel=9)`, so its receptive field covers the
> **whole clip**. The model is trained and evaluated as a clip classifier.
> Consequently the pre-registered ordering **inverts**: A2 − A0 = **−0.0918** on
> zero-shot DoTA (MSAD gave +0.1025), and A1 − A2 = **+0.0300** (MSAD gave
> −0.0683). The 0.6x-vs-0.80 gap to SimpleTAD is **~82 % a method-class gap**
> (weakly-supervised frozen CLIP vs fully-supervised fine-tuned VideoMAE), not a
> DADA-run defect.

> ### ⚠️ Corrected 2026-09-08 — read this before quoting anything below
>
> Phase 0 of `core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` re-measured this
> campaign from the same score curves and found **two errors and one missing
> baseline** in the document you are reading:
>
> 1. **The clip oracle is 0.9086, not 0.9069.** The original used the 3,880-frame
>    `0_Normal_Driving` *subgroup* count instead of the all-normal-*clip* count
>    (3,896). The 16-frame difference is exactly the four vanished-window clips of
>    §8.3 — abnormal in `meta.json`, all-zero after stride-8 rounding, therefore
>    all-normal *clips* for every metric. Corrected in §3, §4 and
>    `core/tests/test_eda.py`.
> 2. **A missing baseline changes the verdict.** A detector reading **only the
>    clip's frame count** scores **micro AUC 0.8654** here (lesson **C28**). The
>    best arm reaches 0.8739 — **+0.0085 over a ruler**, and four of seven arms
>    are *below* it. §4.1.
> 3. **The vanished-clip exclusion §8.3 asks for has now been computed** (§3.1a).
>    It moves micro by +0.001–0.002 and leaves `auc_macro` untouched.
>
> Nothing about the arm *ordering* changes. What changes is that the in-domain
> column was never a result to begin with.

---

## 1. What ran

Seven arms from `DADA_V3_SETUP.md` §2, trained on the DADA-2000 train split,
scored in-domain on the DADA-2000 test split and zero-shot on DoTA. Seed 2024 for
all; **A2 also has seed 2025**.

| Arm | dir | Gate | Stage 1 | Flow / KIP losses | Seeds |
|---|---|---|---|---|---|
| **A0** KIP-off | `kipoff_s2024` | — | n/a | no | 2024 |
| **A1** rank (the v3 model) | `rank_s2024` | `rank` | yes | yes | 2024 |
| **A2** plain-TSM control | `constant_s2024/5` | `constant`, r = 0.5 | **no** | **no** | **2024, 2025** |
| **A2b** constant + PMG | `constant_pmg_s2024` | `constant`, r = 0.5 | inherited | yes | 2024 |
| **A3** v1 bridge | `mlp_frozen_s2024` | `mlp_frozen` | yes | yes | 2024 |
| **A4** gradient intervention | `mlp_ste_s2024` | `mlp_ste` | inherited | yes | 2024 |
| *gate_d0* LaGoVAD `best.ckpt` | `gate_d0` | — | n/a | no | one-off |

All arms: `data.dataset=DADA2000`, `is_egocentric=true`, `frame_stride=8`,
`lr=5e-5`, `batch_size=64`, `num_epochs=20` — **500 optimizer steps total**
(25 batches × 20 epochs). Config diff across arms is exactly the seven flags the
runbook prescribes; nothing else drifted.

### 1.1 Metric protocol — read this before quoting a number

Per lesson **C12**, pooling is decided by the label distribution:

* **DADA-2000 test** — `score_norm: none` (raw pooling). 383 clips, 5,244
  stride-8 frames, 476 positive (9.1 %). **192 clips are all-normal** and
  contribute 3,880 frames — **74 % of the test set is frames from
  `0_Normal_Driving` clips.**
* **DoTA** — `score_norm: minmax` (per-clip min-max, 1,394/1,397 abnormal),
  the `offline_dota_eval.py` convention. 1,397 clips, 18,369 frames, identical
  across all seven evals.

**Never put a DADA raw-micro number and a DoTA min-max number in one column**,
and see §3 before putting a DADA micro number anywhere at all.

---

## 2. Validity gates

| Gate | Result |
|---|---|
| Protocol parity | **PASS** — DADA 383 clips / 5,244 frames and DoTA 1,397 / 18,369 identical across all seven arms |
| Score-norm resolution | **PASS** — DADA `none`, DoTA `minmax`, on every arm |
| KIP-off carries no gate diagnostics | **PASS** — A0's `.npz` hold `{class_names, gt, score, sim}` and no `kip_*` key |
| Right checkpoint scored | **PASS** — every `results.json` points at that arm's own `stage2*/checkpoint_last.pt` |
| Gate ranges (setup §6.8) | **PASS**, and see §7 — `rank` spans 0–128 (128 distinct values, ratio sd 0.312); `constant` is exactly 64; `mlp_frozen` is **61 on every frame of every clip** (ratio sd 0.001) and `mlp_ste` is **62** — lesson **C24** / H4′ reproduced on a third corpus |
| Warm start actually happened | **PASS by inference** (lesson **C17** still open). A1/A3 have their own `stage1/`; A2b opens stage 2 at `kip_rec` = 43.99, bit-identical to A1's opening, and A4 at 42.98, bit-identical to A3's — they warm-started from the sibling stage-1 checkpoints, as the runbook intends |

---

## 3. Headline table

DADA is raw-pooled micro; DoTA is per-clip min-max micro. `macro` is the mean of
per-clip AUCs (190 two-class clips on DADA, 1,392 on DoTA).

| Arm | seed | DADA micro | **DADA micro, clip-mean removed** | **DADA macro** | DADA AP | DoTA micro | DoTA macro | DoTA AP |
|---|---|---|---|---|---|---|---|---|
| **A0** KIP-off | 2024 | 0.7050 | 0.4912 | 0.5190 | 0.1966 | **0.6069** | **0.6254** | 0.4165 |
| **A1** rank (v3) | 2024 | 0.7136 | 0.4785 | 0.5181 | 0.1904 | 0.5451 | 0.5512 | 0.3461 |
| **A2** plain-TSM | 2024 | 0.8617 | 0.4943 | 0.5292 | 0.3118 | 0.5151 | 0.5211 | 0.3528 |
| **A2** plain-TSM | 2025 | 0.8739 | 0.5419 | 0.5716 | 0.3471 | 0.4980 | 0.5086 | 0.3383 |
| **A2b** const+PMG | 2024 | 0.8474 | 0.4078 | **0.4399** | 0.2795 | 0.4322 | 0.4223 | 0.2852 |
| **A3** `mlp_frozen` | 2024 | 0.8457 | 0.4536 | 0.4927 | 0.2982 | **0.3867** | 0.3732 | 0.2664 |
| **A4** `mlp_ste` | 2024 | 0.8024 | 0.4671 | 0.4849 | 0.2484 | 0.4244 | 0.4170 | 0.2843 |
| *gate_d0* `best.ckpt` | — | 0.6063 | 0.5321 | 0.5272 | 0.1410 | — | — | — |
| **length-only baseline** (§4.1) | — | **0.8654** | 0.5000 | 0.5000 | 0.2630 | — | — | — |
| **clip-level oracle** (§4) | — | **0.9086** | 0.5000 | 0.5000 | 0.3531 | — | — | — |

**How to read this table.** The DADA micro column is bounded above by two
baselines that do no localization at all, and every arm sits between them. The
`clip-mean removed` column subtracts each clip's own mean from its curve, leaving
only the ranking localization could have earned: **six of eight rows are at or
below chance**, and the best is 0.5419. That column and `auc_macro` agree; the
raw micro column disagrees with both, and it is the one that is wrong.

Zero-shot DoTA is the honest column, and its ordering is the **inverse** of the
DADA micro ordering — the arm that fits DADA worst (A0, 0.7050) transfers best
(0.6254 macro). See §4.2.

### 3.1a The vanished-clip exclusion, computed (Phase 0.2)

`DADA_SETUP.md` §5.1 asks for the four clips whose anomaly window rounds away at
stride 8 to be excluded at scoring time. Done, offline, from the same `.npz`
files — no retraining, no re-eval:

```
0_Non_Ego_Fault__type1_vid017   0_Non_Ego_Fault__type1_vid052
0_Non_Ego_Fault__type6_vid112   1_Ego_Fault__type1_vid024
```

| | 383 clips | **379 clips (corrected)** |
|---|---:|---:|
| A0 micro | 0.7050 | 0.7062 |
| A2 s2024 micro | 0.8617 | 0.8638 |
| A2 s2025 micro | 0.8739 | **0.8756** |
| A2b micro | 0.8474 | 0.8497 |
| A3 micro | 0.8457 | 0.8476 |
| A4 micro | 0.8024 | 0.8044 |
| gate_d0 micro | 0.6063 | 0.6075 |
| **clip oracle** | 0.9086 | **0.9082** |
| **length-only baseline** | 0.8654 | **0.8681** |

`auc_macro` is **unchanged to four decimals** on every arm: the four clips are
single-class, so `macro_video_auc` already skipped them. Micro moves +0.001 to
+0.002 — and the length-only baseline moves *up* more than any arm does, so the
corrected margin of the best arm over a ruler is **+0.0075**, not +0.0085.

**Use the 379-clip column from now on**, and say which one a number came from.

---

## 4. The in-domain 0.86 is a clip-level number

74 % of the DADA test frames come from clips that are negative in their entirety.
A model that emits **one constant score per clip**, perfectly ranking accident
clips above normal clips and doing **zero** within-clip localization, scores:

```
micro AUC = 0.9086     macro AUC = 0.5000 (undefined per clip -> chance)
```

*(Corrected from 0.9069 on 2026-09-08 — see the banner. The 0.9086 figure is what
`core.eda.protocol.clip_constant_oracle` measures on the label file and what the
`.npz` files on disk reproduce.)*

A2 at seed 2025 reaches **0.8739 — 96 % of that oracle** — with `auc_macro` at
0.5716. A2b lands at `auc_macro` **0.4399**, i.e. *below* chance. **No arm has
demonstrated frame-level anomaly localization on DADA-2000.**

Direct evidence, from the score curves themselves. Variance decomposition
(between-clip variance of the per-clip mean vs. frame-weighted within-clip
variance):

| Arm | B/W ratio, DADA | median within-clip score **range** | DADA micro | DADA macro |
|---|---:|---:|---:|---:|
| A0 KIP-off | 3.8 | 0.055 | 0.7050 | 0.5190 |
| A1 rank | 6.4 | 0.050 | 0.7136 | 0.5181 |
| A4 `mlp_ste` | 12.8 | 0.100 | 0.8024 | 0.4849 |
| A2 constant s2024 | 18.8 | 0.114 | 0.8617 | 0.5292 |
| A2 constant s2025 | 22.9 | 0.124 | 0.8739 | 0.5716 |
| A2b const+PMG | 38.1 | 0.067 | 0.8474 | 0.4399 |
| A3 `mlp_frozen` | 48.1 | 0.071 | 0.8457 | 0.4927 |

The median clip's whole score curve spans **0.05–0.12 on a [0,1] scale**. It is
flat. Between-clip variance is 4–48× the within-clip variance. **Spearman
(B/W ratio, DADA micro AUC) = +0.68** across the seven arms: the flatter the
curve, the higher the "frame-level AUC".

> **Rule.** On DADA-2000, report `auc_macro` with the 0.9086 clip-oracle row
> beside it. A DADA micro AUC is a video-classification score and must never be
> printed next to a published frame-level number.

### 4.1 A ruler scores 0.8654 — the corpus leaks its label through clip length

**Added 2026-09-08 (lesson C28).** The oracle above assumes a model that ranks
clips *perfectly*. It turns out most of that ranking is free.

The reconstructed DADA-2000 this project trains on — not the original >100 GB
release — trims accident videos around the accident and leaves normal-driving
videos at full length:

| test split | abnormal clips | normal clips |
|---|---:|---:|
| n | 191 | 192 |
| median T (stride-8 frames) | **7** | **19** |
| min / max T | 1 / **17** | 1 / 72 |
| clips with T ≥ 18 | **0** | **107** (3,157 frames) |

A detector whose only input is the frame count, emitting the constant score `−T`
across the clip — no pixels, no model, no training — scores:

```
clip-level AUC = 0.8253      micro AUC = 0.8681      macro AUC = 0.5000
                                     (379-clip corrected split; 0.8105 / 0.8654 on 383)
```

The best arm reaches 0.8756. **The margin over a ruler is +0.0075**, and A0, A1,
A4 and gate_d0 are all *below* it.

**This is DADA-specific, not a dashcam-corpus property.** The identical check on
DoTA returns clip-level AUC **0.5280**, micro **0.4993**, and **zero** clips
separable by length — measured with the same function on the same `.npz` files.

The leak is reachable by the model: attention band masks and RoPE positions are
length-dependent (`core/models/temporal_encoder.py:238`) and `core/inference.py:95`
scores each clip at its true length with no padding.

> **Rule (C28).** Print the length-only baseline beside every micro AUC from this
> corpus, and treat any arm that fails to beat it as unmeasured.
> `python -m core.tools.eda report ...` now computes it for every corpus (§3.3 of
> its output) and raises a CRITICAL verdict above clip-level AUC 0.65.

### 4.2 The arms that fit DADA best transfer worst

| Arm | DADA clip-level AUC | DoTA macro (zero-shot) |
|---|---:|---:|
| A0 KIP-off | 0.7665 | **0.6254** |
| A1 rank | 0.7566 | 0.5512 |
| A4 `mlp_ste` | 0.8338 | 0.4170 |
| A2b const+PMG | 0.8815 | 0.4223 |
| A3 `mlp_frozen` | 0.8868 | 0.3732 |
| A2 const s2024 | 0.9260 | 0.5211 |
| A2 const s2025 | **0.9335** | 0.5086 |

Spearman across the seven arms = **−0.39 (p = 0.38)** — **suggestive only, not
significant** at n = 7 single-seed arms. Report it as a consistent direction with
a mechanism (§4.1 + §5), never as an established correlation.

---

## 5. Why the curves are flat: 9 frames under a 9-tap kernel

`core/models/heads.py:21` — `ConvScoreHead` with `score_head_layers=1` is a
single `Conv1d(512 -> 1, kernel_size=9, padding=4, padding_mode="replicate")`.
Its receptive field is 9 timesteps.

| corpus | median T (stride-8 frames) | fraction of clip inside the kernel | clips with T ≤ 9 |
|---|---:|---:|---:|
| MSAD | **86** | 10 % | 0 % |
| DoTA | 13 | 69 % | 12 % |
| **DADA-2000** | **9** | **100 %** | **55 %** |

On the median DADA clip, **every output timestep sees every input frame**, and
adjacent timesteps differ only by a one-step slide of a fully-overlapping 9-tap
window. The head is structurally a clip-pooling operator on 55 % of the corpus.
`kernel_size=9` was inherited from LaGoVAD, which trains on clips ~10× longer.

The supervision is at the same floor. `core/losses/mil.py:22` computes
`k = max(1, n // topk_pct)`; with `mil_topk_pct=16` and n = 9 this is **k = 1**,
so `L_MIL` on the median DADA clip is a plain max over 9 frames.

And the labels are at the floor too, measured on the test split:

* median **2** positive frames per abnormal clip (max 7); **53 clips have
  exactly 1**;
* 52 clips have ≤ 4 frames total, one clip has 1;
* **4 abnormal clips lost their anomaly window entirely** at stride 8 (192
  all-negative clips vs 188 `0_Normal_Driving` folders) — `build_frame_labels`
  warned, `--strict` was not set.

A per-clip AUC computed from 9 points with 2 positives has a resolution of about
1/14. `auc_macro` on DADA is honest but **coarse**; quote it with the per-clip
counts, never as a bare point estimate.

---

## 6. The gap to SimpleTAD is a method-class gap

[SimpleTAD (ICCVW 2025)](https://arxiv.org/abs/2507.09338),
[tue-mps/simple-tad](https://github.com/tue-mps/simple-tad), reports
**DADA-2000 → DoTA = 80.3** (VideoMAE-B), in-domain DoTA 86.4–88.4, D2K
85.6–88.5.

| | SimpleTAD | KAT-VAD (this campaign) |
|---|---|---|
| Backbone | VideoMAE-B/L, **fully fine-tuned**, MVM pre-train + DAPT on driving video | frozen CLIP ViT-B/16, **image** features |
| Temporal model | Video ViT tubelets, sliding 224×224×**16 frames** | 2-layer transformer over per-frame embeddings |
| Sampling | **10 FPS**, 1.5 s window per prediction | stride 8 ≈ **3.75 FPS** |
| Supervision | **per-frame binary labels** ("each frame … is assigned an anomaly label") | **video-level labels only** — `dada.py` deliberately withholds train windows from `labels_train.json` |
| Optimisation | 50 epochs × 50 K sampled windows ≈ 2.5 M examples | **500 optimizer steps** |

Gap decomposition on DoTA:

| | |
|---|---|
| SimpleTAD DADA→DoTA | **0.803** |
| KAT-VAD best DoTA number ever measured (MSAD-trained A2, `RESULTS_V3_GATE_ATTRIBUTION` §3) | **0.6423** |
| LaGoVAD released `best.ckpt`, DoTA min-max | 0.6142 (published 62.60) |
| KAT-VAD best DADA-trained arm (A0) | **0.6069** |

**0.803 − 0.6423 = 0.161 is the method-class ceiling** — present regardless of
which corpus we train on, and reached by neither LaGoVAD's own checkpoint nor
any arm in this project. **0.6423 − 0.6069 = 0.035 is the DADA-specific cost.**
82 % of the gap the user asked about is the first term.

Note also that DADA-trained A0 (0.6069) *beats* MSAD-trained A0 (0.5606) — the
dashcam domain match does help the KIP-off trunk. The loss is entirely in the
KIP arms.

---

## 7. The pre-registered prediction failed — the ordering inverts

`DADA_V3_SETUP.md` §0 registered, before any arm was trained:
`A2 − A0 > 0` and `A1 − A2 <= 0` on zero-shot DoTA.

| Δ, DoTA micro | MSAD-trained (n=3) | **DADA-trained** | verdict |
|---|---|---|---|
| **A2 − A0** | **+0.1025 ± 0.0350**, t95 [+0.0154, +0.1896] | **−0.0918** (seed 2024); −0.1089 (A2 s2025 vs A0 s2024) | **inverted** |
| **A1 − A2** | −0.0683, CI [−0.0790, −0.0580] | **+0.0300** | **inverted** |
| A1 − A0 | +0.0134 | −0.0618 | — |

Both directions flip. The seed-2025 A2 arm agrees with seed 2024 (0.4980 vs
0.5151), so this is not a seed artifact; more seeds would sharpen the interval,
not the sign. This is the "**Ordering inverts**" row of the runbook's §0 decision
table, and it now has a mechanism.

### 7.1 Mechanism: the smoother's sign depends on clip length

Across all seven arms, on the **DoTA transfer**:

| Arm | B/W ratio, DoTA | DoTA micro | DoTA macro |
|---|---:|---:|---:|
| A0 KIP-off | 1.4 | 0.6069 | 0.6254 |
| A1 rank | 2.6 | 0.5451 | 0.5512 |
| A2 constant s2025 | 5.0 | 0.4980 | 0.5086 |
| A4 `mlp_ste` | 5.1 | 0.4244 | 0.4170 |
| A2 constant s2024 | 5.4 | 0.5151 | 0.5211 |
| A2b const+PMG | 9.5 | 0.4322 | 0.4223 |
| A3 `mlp_frozen` | 14.0 | 0.3867 | 0.3732 |

**Spearman (B/W ratio, DoTA micro AUC) = −0.82**; same for macro. One mechanism
explains both this column and §4's:

1. On MSAD (median T = 86) a 50 % channel shift by one timestep is a **mild
   blur** of a long curve, and it helped DoTA transfer by +0.10.
2. On DADA (median T = 9, under a kernel-9 head) the same shift **collapses the
   curve to a per-clip constant**. Training rewards that — DADA's micro metric is
   74 % clip-classification — so the KIP arms drive `L_MIL` far lower than A0
   (epoch-19 mean `mil`: A2 0.255 vs A0 0.490) by learning clip-level separation.
3. DoTA's per-clip **min-max** protocol then deletes exactly the between-clip
   information those arms learned (lesson C8: flat curves map to zeros), so they
   land at or below chance. A3 at 0.3867 and A2b macro at 0.4223 are *inverted*,
   not merely uninformative.

This weakens the MSAD +0.09 further: a component whose sign flips with training
clip length is a smoothing hyperparameter, not a motion mechanism.

---

## 8. Defects found in this campaign

1. **`mlp_ste` (A4) trained through NaNs.** 33 of 500 logged steps carry
   `NaN`, spread over 17 of 20 epochs, always in `mil` + `mul_mil` (plus `kin` in
   17, `dvs_sup*` in 16). `kip_rec` / `kip_align` are finite in the same steps, so
   the NaN enters on the score path *after* KIP. No other arm has a single NaN at
   the same seed, batch size and data, so it is specific to
   `shift_channels_straight_through` (`core/kip/gate_shift.py:99`) under
   `amp: true`. **A4's DADA row is not trustworthy** and is excluded from every
   claim above except the gate-range gate. Not yet root-caused.
2. **The PMG head is nowhere near converged on DADA.** Stage 1 moves `kip_rec`
   121.7 → 41.6 in 500 steps and is still falling; stage 2 ends at ≈ 22–30.
   Compare MSAD, where stage-2 opened at 7.4–8.5. Any A2b / A3 / A1 reading
   inherits an under-trained flow branch — lesson **C14** applies to those rows,
   but **not** to A2, which uses no flow at all.
3. **4 abnormal test clips lost their window at stride 8.** `--strict` was not
   set, so they entered the test set contributing only negative frames.
4. **`--init-weights` is still absent from `config.yaml`** (lesson C17, open
   since the MSAD campaign) — warm start had to be inferred from `kip_rec`.

---

## 9. Limitations — read before citing

* **n = 1 seed for every arm but A2.** The Δ signs in §7 are single-seed except
  A2, which has two agreeing seeds. Report them as *point estimates with a
  measured mechanism*, not as intervals.
* **No published DADA-2000 number is pinned in this project**, and our split is
  a project-invented seeded 20 % stratified split (`dada.py`). Nothing in §3 is
  comparable to SimpleTAD's D2K 85.6, in either direction.
* **A2b, A3, A1 are measured under defect 8.2**; A4 under 8.1.
* `H_mul` is near-degenerate here (`C = 2`, taxonomy `["Normal",
  "CarAccident"]`); read nothing into `mul_mil`.
* **The in-domain column is bounded by two zero-localization baselines** (§4,
  §4.1) and every arm sits between them. No DADA number in this project may be
  placed beside a published frame-level AUC, in either direction.
* **The corpus is a reconstruction with a measured label leak** (§4.1, lesson
  **C28**). Until it is rebuilt into fixed-length windows, an in-domain DADA
  A/B measures the leak as much as the model — lesson **C14** applies to every
  row of §3's DADA columns.
* Per lesson **14**: nothing may be tuned against any number in this document.

---

## 10. What to run next

| | Cost | Buys |
|---|---|---|
| ~~**A. Reporting fix**~~ — **DONE 2026-09-08.** `auc_macro`, the clip-mean-removed micro, the **0.9086** oracle and the **0.8654** length-only baseline are now the DADA headline (§3, §4.1), and `core.tools.eda` computes the last one for every corpus. | zero GPU | Stopped the mirage reaching a thesis table. |
| ~~**B. Frame-level linear probe**~~ — **RUN.** DADA frame probe `auc_macro` **0.5228**, clip probe 0.6799; DoTA frame probe **0.6708** on the same frozen CLIP. Read `DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` §7.1 before drawing a backbone conclusion: the DADA number was measured under C27 **and** C28 and is not admissible. | done | Isolated the deficit as **supervision**, not representation, on DoTA. |
| **C. Re-extract DADA at stride 2–4 with `score_head_kernel=3`** | full re-extraction; **fires lesson C2** — invalidates every DADA cache *and every number in this document* | Tests whether §5's collapse is resolution-driven. Only worth it if B shows the features carry frame-level signal. |
| **D. Root-cause defect 8.1** (`mlp_ste` NaN under AMP) | small | A4 is currently unreportable on DADA. |

Chasing 0.80 on DoTA is out of scope: it requires a fine-tuned video backbone
with per-frame supervision, i.e. abandoning the frozen-CLIP WS-VAD baseline the
project is built on.

---

## 11. Lessons written from this campaign (2026-09-06)

* **Lesson 27 [HIGH] — a score head whose kernel spans the clip is a clip
  classifier, not a frame detector.** Passed all five gates; promoted to
  `lessons-learned/{index,detailed}.md` and added to the `meta-index.md` trigger
  map ("train on a new corpus, or change `frame_stride` / `score_head_kernel` /
  `temporal_window` / MIL top-k").
* **Lesson 12 extended, not duplicated** (gate 5). C12 already owned the
  all-*abnormal* direction; the mirror case — micro AUC over a test set dominated
  by all-*normal* clips measures clip classification — is now folded into it,
  including the rule to compute and print the constant-score-per-clip oracle.
* **Pending: `mlp_ste` NaNs under AMP** (defect 8.1). Held in
  `lessons-learned/pending.md`; gate 4 fails because the mechanism is not yet
  established. Validation plan is recorded there.

---

## 12. Provenance

* Score curves: `outputs/v3/DADA2000/{kipoff,rank,constant,constant_pmg,mlp_frozen,mlp_ste}_s202{4,5}/eval_{dada,dota}/scores/*.npz` (383 + 1,397 files per arm), plus `outputs/v3/DADA2000/gate_d0/`.
* Metrics: each arm's `eval_*/results.json` and `stage2*/metrics.jsonl`; configs from `stage2*/config.yaml`.
* MSAD/DoTA reference rows: `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` §3–§4.
* Baseline protocol facts: `core/docs/RESULTS_DOTA.md` (lesson C8/C12), `core/docs/RESULTS_PHASE_A.md` §4 (C8b).
* SimpleTAD figures: arXiv 2507.09338v2, Table 2 and Table 8 (cross-dataset).
* **`outputs/` is gitignored** — this document is the durable record.
