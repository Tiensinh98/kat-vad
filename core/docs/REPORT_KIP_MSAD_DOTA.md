# KAT-VAD — Experimental Report: the Kinematic Induction Pathway on MSAD and DoTA

**Compiled:** 2026-08-21 · **Scope:** every training and evaluation run on the
`no_center_crop` pipeline, plus the center-crop pipeline it replaced.
**Source of every number:** `outputs/**/results.json`, `outputs/**/scores/*.npz`,
`outputs/**/metrics.jsonl`, `outputs/**/config.yaml`, `data/{MSAD,DoTA}/*`.
No number in this document was copied from a paper or estimated.

**Primary evidence chain (measurement order):** `RESULTS_MSAD.md` →
`RESULTS_DOTA.md` → `RESULTS_NCC.md` → `RESULTS_PHASE_A.md` →
`RESULTS_ARM4_PROBE.md`. This report consolidates them into one data-driven
account; where an earlier document was superseded, only the surviving number
appears here.

---

## 0. Executive summary

The Kinematic Induction Pathway (KIP) is a train-time motion-induction module
added to a language-definition-conditioned weakly-supervised VAD baseline. It
learns to regress cached RAFT optical-flow evidence from RGB features, and at
inference it runs RGB-only at **+1.82 % parameters**.

Across **three seeds** and **four training arms**, the measurement is a clean
asymmetry:

| Benchmark | Regime | Δ(KIP-on − KIP-off) micro AUC | Verdict |
|---|---|---|---|
| **DoTA** | zero-shot, ego-centric driving | **+0.0915 ± 0.0088** (cold control)<br>**+0.0988 ± 0.0148** (warm control) | **Real.** 18 bootstrap CIs, 18 exclusions of zero |
| **MSAD-full** | in-domain, fixed-camera surveillance | **−0.0019** (cold), **−0.0013** (warm) | **Null.** Every CI includes zero; sign unstable |

**The asymmetry is the finding.** The motion pathway pays where motion *is* the
anomaly and is inert where it is not. That is a conditioned claim, and it is
stronger than a uniform win because it names the condition.

Two rival explanations were tested and **eliminated**:

- **Warm start** — KIP-on warm-starts from a stage-1 checkpoint; a fourth arm
  (`kip_off_warm`) gives KIP-off the same trunk. The delta did not shrink; it
  grew marginally. Trunk pretraining alone moves DoTA by ±0.02 with a
  **seed-dependent sign**.
- **Under-convergence** — KIP-on is 5–6× less fitted to MSAD than KIP-off. A
  10-point trajectory probe shows KIP-off's DoTA AUC is flat to **±0.002 across
  a 33× range of training loss**, and KIP-on *gains* as it fits harder. The
  regularizer story is dead at matched convergence.

The proposed **ego-kinematics mechanism is refuted, six comparisons out of six**:
the gain on third-party (`other:`) motion consistently exceeds the gain on
ego-motion. The effect is real, attributable to the module, and its **mechanism
remains unidentified**. This report says so explicitly rather than papering over it.

---

## 1. Datasets

### 1.1 MSAD (Multi-Scenario Anomaly Detection) — in-domain, training benchmark

Fixed-camera surveillance footage spanning many deployment contexts. Anomalies
are largely appearance-separable; the camera does not move.

| Property | Value | Source |
|---|---|---|
| Videos | **720** | `data/MSAD/meta.json` |
| Raw frames | **446,838** (mean 620.6, min 73, max 6,026) | `meta.json` |
| Categories | **11 anomaly classes + `Normal`** (12-way) | `data/MSAD/defs.json` |
| Scenario tags | **17** (`frontdoor` 100, `shop` 65, `office` 52, `road` 51, `sidewalk` 49, `parkinglot` 34, `mall` 34, …; 120 `unknown`) | `meta.json` |
| Camera | fixed / static | — |
| Supervision | **video-level binary labels only** (weakly supervised) | `labels_train.json` |

**Splits** (MSAD protocol ii ratios, seeded and stratified by (label, scenario)):

| Split | Videos | Normal | Abnormal | Notes |
|---|---:|---:|---:|---|
| Train | **480** | 360 | 120 | video-level labels only; anomaly windows exist in `meta.json` but are **never** used as supervision |
| Test | **240** | 120 | 120 | frame-level labels for evaluation |

**Test set as actually scored** (stride 8):

| Quantity | Value |
|---|---:|
| Sampled frames | **18,350** |
| Positive frames | **4,220** (**23.0 %**) |
| Videos with ≥1 positive sampled frame | **119** |
| Videos with zero positives | **121** — 120 `Normal` + **1 abnormal whose window rounds away at stride 8** |
| Macro-scorable videos (both classes present) | **96** |

Per-class test composition: `People_falling` 22, `Traffic_accident` 20,
`Robbery` 14, `Fire` 12, `Object_falling` 10, `Shooting` 9, `Assault` 8,
`Explosion` 8, `Fighting` 6, `Vandalism` 6, `Water_incident` 5.

### 1.2 DoTA (Detection of Traffic Anomaly) — zero-shot transfer benchmark

Ego-centric dashcam driving clips. The anomaly **is** a kinematic event. This is
the benchmark on which the motion claim lives, and it is evaluated **zero-shot**:
models are trained on MSAD only and never see a DoTA frame at train time.

| Property | Value | Source |
|---|---|---|
| Clips (val split) | **1,402** — **every one abnormal** | `data/DoTA/metadata_val.json` |
| Ego-involved / third-party | **805 / 597** (from the `anomaly_class` prefix) | `metadata_val.json` |
| Raw frames | **142,747** at native **10 fps** (mean 101.8, median 99, range 26–284) | `metadata_val.json` |
| Anomaly window | mean **33.7** frames, median 29; 1 % below 7 frames | `metadata_val.json` |
| Positive base rate | **33.1 %** (stride-invariant) | measured at strides 1/2/4/8 |
| Anomaly classes | **20 fine-grained** | see below |
| Camera | ego / moving | — |

Class distribution (top): `ego: turning` 279, `other: turning` 206,
`ego: lateral` 153, `ego: moving_ahead_or_waiting` 119, `ego: oncoming` 104,
`other: lateral` 83, `other: leave_to_right` 77,
`other: moving_ahead_or_waiting` 75, `other: leave_to_left` 54,
`ego: leave_to_left` 45, `ego: leave_to_right` 41, + 9 smaller classes.

**Sampling.** Stride 8, matching both the baseline's own feature extraction
(`interval=8`) and the stride the MSAD checkpoints were trained at:

| Stride | Sampled frames | Positive rate | Clips losing their window |
|---:|---:|---:|---:|
| 1 | 142,747 | 0.3314 | 0 |
| 2 | 71,822 | 0.3295 | 2 |
| 4 | 36,235 | 0.3306 | 2 |
| **8** | **18,478** | **0.3311** | **3** |

**Coverage as actually run: 1,397 / 1,402 clips (99.6 %).** Five clips were lost
to a FUSE unzip that created the folders but wrote no images — an empty directory
is not evidence of data. All arms are scored on the identical 1,397-clip subset,
so the paired Δ is unaffected. Frames scored: **18,369**; macro-scorable clips
(both classes present): **1,392**.

**Label parity — verified, not assumed.** The spans derived from
`metadata_val.json` match the baseline's shipped `anomaly_span` on **all 1,402**
clips to floating-point equality. The ground truth is identical.

### 1.3 Why this pair of benchmarks

They are opposites on exactly the axis KIP targets:

| | MSAD-full | DoTA |
|---|---|---|
| Camera | fixed | ego / moving |
| Anomaly cue | largely appearance | largely kinematic |
| Normal clips in test | 50.4 % | **0.21 %** (3 of 1,397) |
| Baseline headroom | low — KIP-off already ≈ 0.89 | high — baseline ≈ 0.56 |
| Role here | in-domain cost check | out-of-domain effect test |

A null on MSAD is weak evidence against a motion module: the benchmark is close
to saturated for appearance-driven scoring. A delta on DoTA is the whole claim.

---

## 2. Model and the KIP module under test

```
video → frozen CLIP ViT-B/16 → F (L×512)
      → temporal encoder (2-layer Transformer, RoPE) → v^t (L×512)
      → [ KIP ] → v^k (L×512)                        ← the only splice
      → co-attention fusion U(v^k, z^t) with frozen CLIP text definitions
      → H_bin → y^bin (anomaly curve, main output)
      → H_mul → y^mul (per-frame category)
```

KIP internals, all operating on `v^t`:

| Sub-module | Role | Params | Path |
|---|---|---:|---|
| `pmg` (PMG-flow head) | `v^t → ê_O` (L×256) pseudo-flow | **311,808** | inference |
| `shift` (kinematic gate + adaptive temporal shift) | gate α on `flow_norm`, shifts features | **321** | inference |
| `mhead` (motion-score head) | `ê_O → ŷ_O` motion curve | **33,025** | inference |
| `proj_flow` + `proj_rgb` | alignment projections | 32,896 + 65,664 = **98,560** | **train-time only** |

| Parameter budget | Count | vs. baseline |
|---|---:|---:|
| Baseline trainable (KIP off) | **18,923,524** | — |
| Full model trainable (KIP on) | **19,367,238** | +2.34 % |
| KIP total | **443,714** | +2.34 % |
| **KIP on the inference path** | **345,154** | **+1.82 %** |

RAFT never runs at test time. Flow is extracted **offline, once**, cached, and
used only as a regression target. The efficiency claim is measured, not asserted.

---

## 3. The proposed loss functions

Stage-2 objective as wired (weights from `config.yaml`):

```
total = L_MIL
      + 1.0 · L_MIL-align        (multi-class MIL)
      + 1.0 · L_dvs-sup          (dynamic video synthesis, row-gated)
      + 1.0 · L_dvs-supMIL
      + 1.0 · L_neg              (caption contrastive — inactive on MSAD)
      + λ_rec   = 1.0  · L_KIP_rec       ← proposed
      + λ_align = 0.1  · L_KIP_align     ← proposed
      + γ_kin   = 0.2  · L_kin           ← proposed
```

| Loss | Definition | Weight | Notes |
|---|---|---:|---|
| **`L_KIP_rec`** | masked MSE between pseudo-flow `ê_O` and cached RAFT target `e_O` | **1.0** | left unweighted so pseudo-flow stays bounded to the targets |
| **`L_KIP_align`** | bidirectional snippet-level InfoNCE between `proj_flow(ê_O)` and `proj_rgb(v^t)`, τ = 0.07, negatives = the video's own other valid positions | **0.1** | A11 mitigations `align_subsample` and `align_exclude_window` both **off (0/0)** — spec-as-written |
| **`L_kin`** | top-k MIL on the motion curve `ŷ_O` + β·smooth-L1 consistency against `y^bin.detach()` (or `y^p` on synthesized clips) | **0.2** | β = 0.5, `use_yp_anchor=true` |

**Stage 1** trains **KIP only** (everything else frozen) on
`1.0·L_KIP_rec + 0.1·L_KIP_align`. No text is encoded in stage 1.

Ablation gates available: `kip.enabled=false` (pure baseline — the KIP-off arm),
`kip.pmg_only=true` (rec+align only), `kip.use_lkin=false`,
`kip.use_gate_shift=false`.

---

## 4. Training configuration

Identical across all runs except where a column says otherwise. Verified:
**all nine seed-arm `config.yaml` files are byte-identical once Drive paths and
`train.seed` are normalised.** The KIP-on/off pair differs by exactly one line
(`kip.enabled`). The seed is the only other variable.

### 4.1 Model

| Key | Value |
|---|---|
| `backbone` | `clip_vitb16` (frozen), revision-pinned |
| `hidden_dim` | 512 |
| `temporal_layers` / `temporal_heads` / `temporal_window` | 2 / 4 / 25 |
| `temporal_max_positions` | 1536 |
| `num_soft_prompts` | 32 |
| `fusion_num_layers` / `fusion_heads` | 2 / 8 |
| `score_head_layers` / `score_head_kernel` | 1 / 9 |
| `bin_head_type` | `adaptive` (α₀ = 0.25, scale = 10.0) |
| `multiclass_temp` | 0.2 |

### 4.2 KIP

| Key | Value |
|---|---|
| `d_flow` | 256 |
| `pmg_latent_dim` / `align_proj_dim` | 128 / 128 |
| `folding_factor` | 4 |
| `gate_signal` | `flow_norm` |
| `use_gate_shift` / `use_lkin` / `pmg_only` / `on_raw_features` | true / true / false / false |

### 4.3 Losses and DVS

| Key | Value |
|---|---|
| `lambda_rec` / `lambda_align` / `gamma_kin` / `beta_cons` | 1.0 / 0.1 / 0.2 / 0.5 |
| `tau_align` | 0.07 |
| `mil_topk_pct` / `sup_mil_topk_pct` / `mul_mil_topk_pct` | 16 / 4 / 16 |
| `mul_weight` / `pseudo_sup_weight` / `pseudo_sup_mil_weight` / `cap_contrastive_weight` | 1.0 each |
| `contrastive_neg_mining` / `contrastive_temp` | `n3` / 0.02 |
| `use_yp_anchor` | true |
| `align_subsample` / `align_exclude_window` | **0 / 0 (both mitigations off)** |
| DVS `theta` / `theta_ego` / `delta_m` / `delta_m_ego` | 0.7 / 0.85 / 5 / 2 |
| DVS `syn_max_num_clips` / `knn_filler_ratio` | 5 / 0.5 |

### 4.4 Data and optimization

| Key | Value |
|---|---|
| `dataset` / `is_egocentric` | `MSAD-full` / false |
| `frame_stride` / `crop_size` / `max_vis_len` | **8** / 224 / 512 |
| Feature transform | **`no_center_crop`** — anisotropic `Resize((224,224))`, full field of view |
| `learning_rate` / `batch_size` / `weight_decay` / `warmup_steps` | 5e-5 / 64 / 0.01 / 20 |
| `num_epochs` | 125 → **500 optimizer steps** |
| `grad_accum_steps` | 1 |
| `seed` | **2024 / 2025 / 2026** |
| AMP | stage 1 **off**, stage 2 **on** |
| `checkpoint_every_steps` | stage 1: 0; stage 2: **100** |
| Device | CUDA (Colab). *Local MPS training is known to diverge on this graph.* |

### 4.5 The transform decision (a precondition, not a result)

The pipeline moved from center-crop to `no_center_crop` before the seed
campaign. The reason was **internal consistency**, not convention: RAFT
preprocessing is full-frame, while the CLIP transform center-cropped away
roughly the left and right quarters of a 16:9 frame. KIP was therefore being
trained to regress motion evidence that had been cropped out of its own input —
and **349 of 1,402 DoTA clips are `ego`/`other: lateral`**, where exactly that
evidence lives.

Every artifact from that point is `*_ncc`. A feature cache is bound to the
transform and stride that built it; mixing caches invalidates every metric
measured on them.

---

## 5. Evaluation configuration

| Item | Value |
|---|---|
| Checkpoint scored | **`checkpoint_last.pt`** for every arm — no validation split, no model selection |
| Windowing | full-length videos scored in `max_vis_len = 512` windows, concatenated; definitions re-verbalized per window |
| Score pooling | **`--score-norm auto`** — resolved from the **label distribution**, not the dataset name: per-clip min-max when normal videos are < 5 % of the test set, otherwise raw |
| → MSAD (50.4 % normal) | **raw** |
| → DoTA (0.21 % normal) | **per-clip min-max** |
| Micro AUC / AP | over all concatenated sampled frames |
| Macro AUC | mean per-clip AUC, skipping single-label clips (MSAD 96 videos, DoTA 1,392 clips) |
| Confidence intervals | paired bootstrap resampling **clips** (not frames), 2,000 draws, `default_rng(0)`, both arms scored on the same resample |

### 5.1 Why pooling is decided by labels

This is not a stylistic choice — it is worth **11 AUC points** on the reference
checkpoint, and getting it wrong once put every DoTA arm at chance:

| DoTA arm | raw pooled | **per-clip min-max** |
|---|---:|---:|
| Released baseline `best.ckpt` (center-crop features) | 0.5055 | **0.6142** |
| Released baseline `best.ckpt` (`_ncc` features) | 0.4956 | **0.6012** |
| Ours, KIP-off (`_ncc`, seed 2024) | 0.5204 | **0.5607** |
| Ours, KIP-on (`_ncc`, seed 2024) | 0.5371 | **0.6519** |

On an all-abnormal test set the between-clip score *scale* carries no label
information, so pooling it in lets a confident clip's negatives outrank a
hesitant clip's positives. Having the released checkpoint in the run is what made
this findable: a released checkpoint cannot be at chance on its own benchmark.

---

## 6. Experimental design — four arms and a probe

| Arm | Description | KIP at inference | Trunk init | Runs |
|---|---|:---:|---|---:|
| `stage2_kip_off` | pure baseline, **cold** start | no | random | 3 seeds |
| `stage2_kip_off_warm` | baseline warm-started from the **same** stage-1 checkpoint KIP-on used | no | stage 1 | 3 seeds |
| `stage2_kip_on` | full KAT-VAD | **yes** | stage 1 | 3 seeds |
| `gate_a` / `full_gate_a` | LaGoVAD's released `best.ckpt` through our eval | n/a | — | 1 |

Plus the **trajectory probe**: 10 DoTA evaluations of seed 2024's
`checkpoint_step_{100…500}.pt` for both the on and off arms — no training, cached
features only.

**Why arm 3 exists.** Without `kip_off_warm`, the A/B is "KIP + warm start" vs
"no KIP, cold". With it, the A/B is **"KIP vs no KIP from a shared trunk"** —
the strongest version of the claim the artifacts support.

**Warm start verified from the loss, not assumed** (`--init-weights` is not
recorded in `config.yaml` — a provenance defect noted in §11). Step-1 training
`mil`:

| seed | cold off | **off_warm** | on |
|---|---:|---:|---:|
| 2024 | 0.86818 | **0.70148** | 0.70146 |
| 2025 | 0.68530 | **0.82810** | 0.81630 |
| 2026 | 0.78010 | **0.66500** | 0.66030 |

`off_warm` enters stage 2 next to `on` and away from cold `off` in every seed —
seed 2024 agrees to four decimals. The two warm arms share a trunk; the cold arm
does not.

---

## 7. Results — DoTA (zero-shot, per-clip min-max)

### 7.1 All arms, all seeds

| Seed | Metric | off (cold) | off (warm) | **KIP-on** | released `best.ckpt` |
|---|---|---:|---:|---:|---:|
| 2024 | micro AUC | 0.5607 | 0.5480 | **0.6519** | 0.6012 |
| | AP | 0.3503 | 0.3398 | **0.4303** | 0.3777 |
| | macro AUC | 0.5638 | 0.5470 | **0.6746** | 0.6158 |
| 2025 | micro AUC | 0.5585 | 0.5311 | **0.6416** | — |
| | AP | 0.3443 | 0.3273 | **0.4163** | — |
| | macro AUC | 0.5532 | 0.5289 | **0.6550** | — |
| 2026 | micro AUC | 0.5283 | 0.5464 | **0.6288** | — |
| | AP | 0.3241 | 0.3370 | **0.4033** | — |
| | macro AUC | 0.5190 | 0.5398 | **0.6291** | — |

`best.ckpt` is seed-independent, hence one row.

### 7.2 The primary contrast, Δ(KIP-on − KIP-off cold)

| Seed | Δ micro AUC | 95 % CI | Δ AP | 95 % CI | Δ macro |
|---|---:|---|---:|---|---:|
| 2024 | **+0.0911** | [+0.0797, +0.1021] | +0.0792 | [+0.0685, +0.0896] | +0.1108 |
| 2025 | **+0.0829** | [+0.0725, +0.0945] | +0.0712 | [+0.0622, +0.0810] | +0.1018 |
| 2026 | **+0.1004** | [+0.0886, +0.1124] | +0.0788 | [+0.0690, +0.0885] | +0.1101 |
| **mean** | **+0.0915 ± 0.0088** | — | +0.0764 | — | +0.1076 |

**Nine intervals, nine exclusions of zero.** All three micro CIs mutually
overlap — consistent with one underlying effect, not three lucky draws. The
minimum, +0.0829, is **≈ 2.8× the pre-registered +0.03 bar**.

### 7.3 The controlled contrast, Δ(KIP-on − KIP-off warm)

| Seed | Δ micro AUC | 95 % CI | Δ AP | Δ macro |
|---|---:|---|---:|---:|
| 2024 | **+0.1039** | [+0.0961, +0.1115] | +0.0899 | +0.1277 |
| 2025 | **+0.1104** | [+0.1022, +0.1193] | +0.0887 | +0.1260 |
| 2026 | **+0.0822** | [+0.0731, +0.0920] | +0.0653 | +0.0891 |
| **mean** | **+0.0988 ± 0.0148** | — | +0.0813 | +0.1143 |

Nine more intervals, nine more exclusions of zero. Against the cold Δ
(+0.0915), the warm-corrected Δ **did not shrink — it is marginally larger**,
and the per-seed change moves in both directions (+0.0911→+0.1039,
+0.0829→+0.1104, +0.1004→+0.0822).

**The warm-start confound is closed.** The comparison is now KIP vs no KIP from
a shared trunk.

### 7.4 What the stage-1 pretraining alone buys: nothing stable

Δ(off_warm − off_cold) — the effect of 125 epochs of `L_KIP_rec` / `L_KIP_align`
transmitted through the **shared temporal encoder only**, with no KIP module at
inference:

| Seed | Δ micro AUC | 95 % CI | Δ AP | Δ macro |
|---|---:|---|---:|---:|
| 2024 | **−0.0128** | [−0.0229, −0.0030] | −0.0107 | −0.0169 |
| 2025 | **−0.0273** | [−0.0370, −0.0177] | −0.0174 | −0.0241 |
| 2026 | **+0.0181** | [+0.0079, +0.0274] | +0.0134 | +0.0208 |
| mean | −0.0073 | — | −0.0049 | −0.0067 |

Every CI excludes zero **and the sign flips across seeds**. Stage-1 pretraining
is a ±0.02 seed-dependent *level* shift — an order of magnitude below KIP's
+0.099, and not even reliably positive.

**Conclusion: the transferable thing is the KIP module in the forward pass at
inference, not the pretraining it delivered to the trunk.** That claim is only
available because arm 3 exists.

### 7.5 Against the released baseline checkpoint

KIP-on also beats LaGoVAD's own released checkpoint on `_ncc` features:
0.6519 vs 0.6012 micro, **Δ +0.0508, CI [+0.0355, +0.0659]**.

---

## 8. Results — MSAD-full (in-domain, raw pooling)

### 8.1 All arms, all seeds

| Seed | Metric | off (cold) | off (warm) | KIP-on | released `best.ckpt` |
|---|---|---:|---:|---:|---:|
| 2024 | micro AUC | 0.8922 | 0.8866 | 0.8868 | 0.8949 |
| | AP | 0.6587 | 0.6521 | 0.6574 | 0.6432 |
| | macro AUC | 0.7177 | 0.6954 | 0.7122 | 0.7177 |
| 2025 | micro AUC | 0.8857 | 0.8872 | 0.8862 | — |
| | AP | 0.6531 | 0.6503 | 0.6469 | — |
| | macro AUC | 0.6785 | 0.6995 | 0.7158 | — |
| 2026 | micro AUC | 0.8824 | 0.8844 | 0.8815 | — |
| | AP | 0.6568 | 0.6370 | 0.6454 | — |
| | macro AUC | 0.6773 | 0.7014 | 0.7099 | — |

### 8.2 Δ(KIP-on − KIP-off), both controls

| Seed | Δ vs cold | 95 % CI | Δ vs warm | 95 % CI |
|---|---:|---|---:|---|
| 2024 | −0.0054 | [−0.0154, +0.0036] | +0.0001 | [−0.0047, +0.0045] |
| 2025 | +0.0005 | [−0.0066, +0.0073] | −0.0010 | [−0.0058, +0.0035] |
| 2026 | −0.0008 | [−0.0096, +0.0082] | −0.0030 | [−0.0078, +0.0015] |
| **mean** | **−0.0019** | — | **−0.0013** | — |

**Every CI includes zero, in both controls, in every seed, in micro AUC and AP.
The sign is not even stable.** KIP costs nothing in-domain and buys nothing.

### 8.3 The reproduction gate — measured against the checkpoint, not the paper

The published baseline MSAD number is **0.9041**. Our `_ncc` arms land at
0.8815–0.8922. That looks like a failure until the released checkpoint is run
through the identical protocol:

| Transform | released `best.ckpt` | our KIP-off | published |
|---|---:|---:|---:|
| center-crop | 0.8991 | 0.9052 | 0.9041 |
| **`no_center_crop`** | **0.8949** | 0.8922 | 0.9041 |

**The released checkpoint cannot reach 0.9041 either.** Paired bootstrap of every
arm against it on the identical protocol, 240 videos:

| Comparison | Δ AUC | 95 % CI | Δ AP |
|---|---:|---|---:|
| crop, KIP-off − `best.ckpt` | +0.0061 | [−0.0201, +0.0304] | +0.0437 |
| crop, KIP-on − `best.ckpt` | +0.0073 | [−0.0171, +0.0308] | +0.0523 |
| ncc s2024, KIP-off − `best.ckpt` | −0.0027 | [−0.0246, +0.0197] | +0.0155 |
| ncc s2024, KIP-on − `best.ckpt` | −0.0081 | [−0.0306, +0.0139] | +0.0142 |
| ncc s2025, KIP-off − `best.ckpt` | −0.0092 | [−0.0335, +0.0143] | +0.0099 |
| ncc s2025, KIP-on − `best.ckpt` | −0.0087 | [−0.0312, +0.0133] | +0.0037 |
| ncc s2026, KIP-off − `best.ckpt` | −0.0125 | [−0.0374, +0.0108] | +0.0136 |
| ncc s2026, KIP-on − `best.ckpt` | −0.0134 | [−0.0377, +0.0101] | +0.0022 |

**All eight CIs include zero; AP is consistently higher for our arms.** The
baseline port is statistically indistinguishable from the shipped checkpoint.

**Gate definition used in this report:** *"our KIP-off arm matches the released
`best.ckpt` under one identical protocol."* By that definition the gate
**passes**, on both transforms, in all three seeds. 0.9041 is quoted only as the
paper's printed number. A residual explanation for the gap: our AUC is over
18,350 stride-8 sampled frames, not the paper's full frame count.

### 8.4 A weak signal, stated with its caveat

KIP-on's MSAD **macro** AUC is markedly more seed-stable than KIP-off's:

| | 2024 | 2025 | 2026 | spread |
|---|---:|---:|---:|---:|
| KIP-off macro | 0.7177 | 0.6785 | 0.6773 | 0.040 |
| KIP-on macro | 0.7122 | 0.7158 | 0.7099 | **0.006** |

Δmacro is positive in all three seeds against both controls (vs warm:
+0.0167 / +0.0164 / +0.0085, P>0 = 0.966 / 0.915 / 0.811), but **only one of six
CIs excludes zero, and barely**. At n = 3 this is an observation, not a result.
Do not build an argument on it.

---

## 9. Effect of the proposed loss functions

All figures below are **20-step trailing means** of the logged per-step loss,
first 20 steps → last 20 steps, recomputed from `metrics.jsonl`.

### 9.1 Stage 1 — KIP-only warm-up (`L_KIP_rec` + 0.1 · `L_KIP_align`)

| Seed | `kip_rec` first→last | drop | `kip_align` first→last | drop |
|---|---|---:|---|---:|
| 2024 | 16.541 → **10.520** | 36.4 % | 4.631 → **4.076** | 12.0 % |
| 2025 | 17.151 → **9.634** | 43.8 % | 4.620 → **4.034** | 12.7 % |
| 2026 | 16.090 → **9.290** | 42.3 % | 4.641 → **4.056** | 12.6 % |

`L_KIP_rec` is doing the work; `L_KIP_align` barely moves.

### 9.2 Stage 2 — full objective, KIP-on

| Loss | Seed 2024 | Seed 2025 | Seed 2026 | Trend |
|---|---|---|---|---|
| `kip_rec` | 9.427 → **4.907** (−48 %) | 9.733 → **4.652** (−52 %) | 9.021 → **4.603** (−49 %) | halves again on top of stage 1 |
| `kip_align` | 4.067 → **3.702** (−9.0 %) | 4.020 → **3.612** (−10.1 %) | 4.047 → **3.689** (−8.8 %) | **near-flat, near-chance** |
| `kin` | 0.759 → **0.469** (−38 %) | 0.758 → **0.320** (−58 %) | 0.724 → **0.383** (−47 %) | converges, seed-variable |
| `mil` | 0.648 → **0.0075** | 0.682 → **0.0057** | 0.582 → **0.0073** | fully fits weak labels |
| `mul_mil` | 2.447 → 0.182 | 2.435 → 0.330 | 2.416 → 0.260 | — |
| `dvs_sup` | 0.493 → 0.019 | 0.357 → 0.020 | 0.395 → 0.022 | — |
| `total` | 14.15 → **5.59** | 14.23 → **5.44** | 13.45 → **5.34** | floor set by `kip_rec` + `kip_align` |

### 9.3 Stage 2 — KIP-off arms (for contrast)

| Arm | Seed 2024 `mil` | 2025 | 2026 |
|---|---|---|---|
| off (cold) | 0.747 → **0.0012** | 0.604 → **0.0011** | 0.680 → **0.0012** |
| off (warm) | 0.636 → **0.0011** | 0.639 → **0.0012** | 0.563 → **0.0012** |
| **on** | 0.648 → **0.0075** | 0.682 → **0.0057** | 0.582 → **0.0073** |

**KIP-on ends 5–6× less fitted to MSAD than either KIP-off arm, in every seed**
(6.3× / 5.2× / 6.1×). This is the observation that generated the
under-convergence hypothesis — and §10 kills it.

> *Averaging-window note:* `RESULTS_PHASE_A.md` §5 quotes 0.0100 / 0.0016 for
> seed 2024 using a different trailing window. The ratio (5–6×) is identical
> either way; the numbers here use the 20-step window consistently throughout.

### 9.4 What each proposed loss actually contributed

**`L_KIP_rec` (λ = 1.0) — the load-bearing term.** It falls 36–44 % in stage 1
and a further 48–52 % in stage 2, monotonically, in every seed. It is the only
KIP loss with a large, consistent gradient signal, and it is the term that makes
`ê_O` a faithful reconstruction of cached RAFT evidence. **Caveat:** its 20-step
moving average first came within 5 % of its minimum at step ≈ 446 of 500, in both
stages — **the reconstruction head is still improving when the schedule ends.**
The warm-up is undertrained, and the whole reported effect is achieved *without*
`L_KIP_rec` having converged.

**`L_KIP_align` (λ = 0.1) — running near chance, and it did not matter.** The
loss is a within-video bidirectional InfoNCE over the video's own valid
positions. Mean sampled train length is 78.2 frames, so **chance loss =
E[ln n] ≈ 4.265**. Final values: **3.70 / 3.61 / 3.69** — only **0.56–0.65 nats
(13–15 %) below chance after 1,000 steps across both stages**.

This is the predicted near-duplicate-positive failure: adjacent frames are
near-identical *positives* being scored as *negatives*, so the discrimination
task is close to ill-posed. **Both mitigation knobs are off**
(`align_subsample = 0`, `align_exclude_window = 0`).

The consequence is a genuine finding, not a footnote: **KIP delivers +0.09 DoTA
AUC while one of its three proposed losses is contributing almost no learning
signal.** Whatever the mechanism is, it does not run through cross-modal
alignment as specified. Any future ablation that reports "align doesn't matter"
would be right for the wrong reason.

**`L_kin` (γ = 0.2, β = 0.5) — converges, contribution unresolved.** It drops
38–58 %, but with the widest seed spread of any KIP loss (final 0.320–0.469, a
1.5× range against `kip_rec`'s 1.07× range). No arm isolates it: the
`kip.use_lkin=false` ablation has **not been run**. Its contribution to the +0.09
is currently **unmeasured**.

**The `mil` asymmetry — a side effect worth naming.** Adding KIP raises final
`mil` 5–6× in every seed. The three KIP losses act as a constraint on the shared
temporal encoder that measurably prevents full memorization of 480 weak labels.
That is a *description* of what happened, not an explanation of the transfer
gain — §10 shows the transfer gain does not follow from it.

---

## 10. Trajectory probe — convergence does not explain the gain

**Hypothesis tested:** KIP-on transfers better *because* it is less fitted to
MSAD (§9.3). If true, an equally under-fitted KIP-off model should transfer
equally well.

Seed 2024, both arms, DoTA scored at five training checkpoints. `mil` is the
20-step trailing mean ending at that checkpoint:

| Arm | Step | train `mil` | DoTA micro AUC | AP | macro |
|---|---:|---:|---:|---:|---:|
| off | 100 | 0.0410 | 0.5589 | 0.3517 | 0.5620 |
| off | 200 | 0.0031 | 0.5604 | 0.3511 | 0.5630 |
| off | 300 | 0.0016 | 0.5604 | 0.3502 | 0.5623 |
| off | 400 | 0.0012 | 0.5608 | 0.3505 | 0.5640 |
| off | 500 | 0.0013 | 0.5607 | 0.3503 | 0.5638 |
| on | 100 | 0.0834 | 0.6097 | 0.3947 | 0.6186 |
| on | 200 | 0.0207 | 0.6412 | 0.4214 | 0.6587 |
| on | 300 | 0.0103 | 0.6482 | 0.4273 | 0.6691 |
| on | 400 | 0.0074 | 0.6514 | 0.4298 | 0.6737 |
| on | 500 | **0.0075** | **0.6519** | 0.4303 | 0.6746 |

Four independent readings, all pointing the same way:

1. **At matched convergence the gap is undiminished.** KIP-on's endpoint sits at
   `mil` ≈ 0.0075; KIP-off crosses that value at **step ≈ 161**, bracketed by
   step 100 (`mil` 0.0410, *less* fitted → 0.5589) and step 200 (`mil` 0.0031,
   *more* fitted → 0.5604). Interpolated KIP-off ≈ **0.560** against KIP-on's
   **0.6519** — a gap of **+0.092, the entire headline delta**.
2. **KIP-off's transfer is flat.** Its `mil` falls **33×** (0.0410 → 0.0012)
   while DoTA AUC moves 0.5589 → 0.5607 — a span of **0.0019**. Convergence level
   does not index DoTA transfer for that arm at all.
3. **The direction is backwards for the hypothesis.** KIP-on *gains* +0.042 as it
   fits harder (0.6097 → 0.6519). Less fitting → better transfer predicts the
   opposite.
4. **The curves never overlap.** KIP-on at its least-converged probed point
   (step 100, `mil` 0.0834 — 11× less fitted than KIP-off's endpoint) already
   scores 0.6097, above KIP-off's best-ever 0.5608.

**The under-convergence explanation is rejected.** Because it was unambiguous at
seed 2024, the probe was not extended to the other seeds.

**Integrity checks.** Probe step-500 reproduces the headline arms exactly
(off 0.5607, on 0.6519), confirming `checkpoint_last == checkpoint_step_500` and
that the probe scored the same models. All 10 runs cover 18,369 frames /
1,397 clips / 1,392 macro-scorable clips.

**The one segment the probe cannot see.** `checkpoint_every_steps = 100` over a
500-step run leaves **no checkpoint below step 100**, where `mil` falls from 0.63
to 0.041 — **94 % of the loss range sits inside the first, unprobed 20 % of
steps.** The verdict holds (the matched point is bracketed, and an arm flat to
±0.002 across a 33× range is not hiding a +0.07 jump just below), but
step-uniform checkpointing undersamples what a convergence probe needs.

**Discipline note.** This section reads DoTA *test* scores along a training
trajectory. It is a **diagnostic, not model selection.** Every reported arm in
§7 and §8 is `checkpoint_last`. The probe curve must never become a headline
number, or the result reads as tuned on test.

---

## 11. Mechanism checks — what KIP is *not* doing

### 11.1 Ego vs third-party motion — the proposed mechanism is refuted

The proposal's story is that KIP models the ego-vehicle's own kinematics. DoTA's
`ego:` classes are precisely where the camera's own motion carries the anomaly,
so the gain should concentrate there. Per-clip macro AUC Δ, split on the
`anomaly_class` prefix:

| Seed | Contrast | ego (n = 802) | other (n = 590) | Per-clip win / loss |
|---|---|---:|---:|---|
| 2024 | on − off_cold | +0.0969 | **+0.1296** | — |
| 2025 | on − off_cold | +0.0734 | **+0.1403** | — |
| 2026 | on − off_cold | +0.0873 | **+0.1411** | — |
| 2024 | on − off_warm | +0.1134 | **+0.1469** | 73.8 % / 13.1 % |
| 2025 | on − off_warm | +0.1070 | **+0.1518** | 71.1 % / 15.8 % |
| 2026 | on − off_warm | +0.0577 | **+0.1323** | 64.8 % / 20.6 % |

**`other > ego` in six comparisons out of six.** At n = 1 this was a failed
check; at n = 6 it is a settled negative. **Do not claim KIP works by modelling
ego-vehicle kinematics.**

Group sizes are on the 1,392 macro-scorable clips (802 `ego:` / 590 `other:`),
not the 805 / 597 of the full 1,402-clip split.

Both groups still gain substantially and per-clip wins outnumber losses ≈ 4:1, so
this is a broad effect, not one subgroup carrying the average. The top-gaining
fine-grained classes in the seed-2024 on−off split are
`other: leave_to_right` (+0.187), `other: oncoming` (+0.165) and
`other: turning` (+0.144) — peripheral, third-party events, consistent with
field-of-view restoration rather than ego-kinematics.

### 11.2 The MSAD multi-class thread did not replicate

Under center-crop features, the strongest MSAD signal was multi-class frame
accuracy on anomalous frames (n = 4,220), concentrated on motion-textured
classes. Under `_ncc` it evaporated:

| Transform | KIP-off | KIP-on | Δ |
|---|---:|---:|---:|
| center-crop | 0.4754 | 0.5116 | **+3.6 pp** |
| `no_center_crop` | 0.4758 | 0.4699 | **−0.6 pp** |

It vanished in exactly the run where the DoTA gain appeared. **They are not one
mechanism**, and no unified story should be written across them.

### 11.3 Score saturation — a caveat, not a defect

Both `_ncc` arms are extreme on DoTA: **89.1 %** (off) / **86.6 %** (on) of
frames score above 0.99; within-clip dynamic range median 1.8e-4 / 7.4e-4. The
released checkpoint by contrast sits at mean 0.221 with range 0.038.

AUC is rank-based and per-clip min-max is monotone, so macro AUC is unaffected.
The one way saturation could fake the result is float32 ties, checked directly:

| Run | Tied frames | Clips with any tie |
|---|---:|---:|
| `_ncc` KIP-off | 270 / 18,369 (1.47 %) | 210 / 1,397 |
| `_ncc` KIP-on | 93 / 18,369 (0.51 %) | 84 / 1,397 |
| `_ncc` `best.ckpt` | 1 / 18,369 (0.01 %) | 1 / 1,397 |

≈ 0.2 tied frames per clip — far too few to manufacture a +0.11 macro delta.
**The effect is not a numerical artifact.** What it does mean: these models are
far outside their calibrated range, every threshold-based metric is meaningless
on them, and **no single absolute score from these arms should be quoted alone.**

### 11.4 The transform ablation — the effect moves through KIP, not around it

Seed 2024, same comparison per arm, center-crop → `no_center_crop`:

| Arm | Δ macro AUC (DoTA) | 95 % CI |
|---|---:|---|
| **KIP-on** | **+0.1514** | [+0.1354, +0.1683] |
| KIP-off | +0.0077 | [−0.0065, +0.0225] — includes 0 |
| released `best.ckpt` | **−0.0170** | [−0.0301, −0.0037] — **negative** |

**Only the arm that consumes motion moved.** This is the strongest available
evidence that KIP's pathway is genuinely motion-mediated: restoring the field of
view that the flow targets always had is worth +0.15 macro AUC to KIP-on, nothing
measurable to KIP-off, and is actively harmful to the reference checkpoint.

Full effect of the transform switch, both benchmarks, seed 2024:

| | MSAD-full (raw) | DoTA (min-max) |
|---|---|---|
| KIP-off, crop | 0.9052 / AP 0.7249 | 0.5539 (macro 0.5561) |
| KIP-on, crop | 0.9064 / 0.7334 | 0.5215 (0.5232) |
| Δ(on − off), crop | +0.0012, CI [−0.0060, +0.0090] — null | **−0.0324** |
| KIP-off, ncc | 0.8922 / 0.6587 | 0.5607 (0.5638) |
| KIP-on, ncc | 0.8868 / 0.6574 | **0.6519 (0.6746)** |
| Δ(on − off), ncc | −0.0054, CI [−0.0155, +0.0033] — null | **+0.0911** |

**Methodological consequence.** The crop was a *known-open precondition defect*
at the time the −0.0324 was recorded: KIP was regressing motion evidence cropped
out of its own input. Removing it flipped the sign and tripled the magnitude. A
signed number with a confidence interval outlives its prose caveats — an A/B run
under an open precondition defect measures the defect, and should be reported as
**blocked, with no number**, until the precondition is closed.

The in-domain cost of the switch: MSAD KIP-on −0.0196 AUC / −0.0761 AP, both CIs
excluding zero. The trade is ≈ 2 pp in-domain AUC and ≈ 7 pp AP for ≈ 9 pp
out-of-domain AUC.

---

## 12. Trends

1. **The gap is stable; the level is not.** Across seeds, both DoTA arms drift
   down together (off 0.5607 → 0.5585 → 0.5283; on 0.6519 → 0.6416 → 0.6288),
   while the paired Δ stays within ±0.009 of its mean. **Report the paired Δ,
   never either arm's absolute number.** The same pattern holds for the warm
   start: it is a level effect (±0.02, sign-flipping); KIP is the gap.
2. **The benefit is out-of-domain only, and monotone in domain shift.** Zero
   in-domain (MSAD, 6/6 CIs include zero), large out-of-domain (DoTA, 18/18 CIs
   exclude zero). Nothing in between has been measured.
3. **The benefit grows with training, not shrinks.** KIP-on's DoTA AUC rises
   monotonically 0.6097 → 0.6519 across steps 100–500 while it fits MSAD harder.
   KIP-off's is flat at 0.559–0.561 over the same span.
4. **Reconstruction converges; alignment does not.** `L_KIP_rec` falls ~50 % per
   stage in every seed; `L_KIP_align` sits 13–15 % below chance throughout. The
   two proposed KIP losses behave completely differently, and only one is
   supplying signal.
5. **KIP is a consistent brake on in-domain memorization.** Final `mil` is 5–6×
   higher with KIP in all three seeds — a reliable, reproducible side effect that
   nonetheless does **not** explain the transfer gain (§10).
6. **Field of view is a first-order variable for a motion module.** The crop
   removal moved KIP-on by +0.15 macro AUC and everything else by ≈ 0. A motion
   pathway must see the same frame its flow targets were computed on.

---

## 13. Conclusions

1. **KIP produces a large, reproducible, out-of-domain gain.**
   Δ = **+0.0915 ± 0.0088** micro AUC on DoTA zero-shot against a cold baseline
   and **+0.0988 ± 0.0148** against a trunk-matched warm baseline; 18 bootstrap
   CIs across micro AUC, AP and macro AUC over three seeds, **all excluding
   zero**. KIP-on also beats the released baseline checkpoint by +0.0508
   (CI [+0.0355, +0.0659]).

2. **The gain is attributable to the KIP module itself.** Two rival explanations
   were tested and eliminated: warm-start transfer (arm 3 — the delta grew) and
   under-convergence (the probe — flat over a 33× loss range, and the direction
   is backwards). Stage-1 trunk pretraining alone transfers **nothing stable**.
   What transfers is the module in the forward pass at inference.

3. **KIP costs nothing in-domain.** MSAD Δ is null in every seed, in both
   controls, in micro AUC and AP, with an unstable sign. There is no in-domain
   penalty to trade against the out-of-domain gain.

4. **The efficiency claim holds as designed.** +1.82 % inference parameters
   (345,154 of 18.9 M), **RGB-only at test time**. RAFT runs offline, once, as a
   cached training target and never on the scoring path.

5. **The baseline port is sound.** Our KIP-off arm is statistically
   indistinguishable from the released checkpoint on MSAD under an identical
   protocol — eight paired bootstraps, all CIs including zero, AP consistently
   higher for our arms.

6. **The proposed mechanism is refuted; the actual mechanism is unknown.**
   Ego-kinematics fails six comparisons out of six (`other` > `ego` every time).
   The multi-class categorisation thread did not replicate. `L_KIP_align` runs at
   near-chance throughout, so the effect does not run through cross-modal
   alignment as specified. **Every rival explanation has been eliminated by
   subtraction; none has been confirmed by construction.**

7. **The measurement protocol is part of the result.** Score pooling must be
   decided by the test set's *label distribution*, not its name — on an
   all-abnormal benchmark, raw pooling cost the reference checkpoint 11 AUC
   points and put every arm at chance. Keeping a released checkpoint in every run
   is what made that findable.

**The honest one-line claim this evidence supports:** *a train-time motion
induction pathway, costing +1.8 % inference parameters and no test-time flow,
improves zero-shot transfer from fixed-camera to ego-centric anomaly detection by
≈ 9 AUC points with no in-domain cost — by a mechanism that is not the one it was
designed around.*

---

## 14. Limitations and threats to validity

| # | Limitation | Severity | Status |
|---|---|---|---|
| 1 | **Mechanism unidentified.** The effect is real and attributed to the module, but *what* the module does is unknown. No eval-time diagnostics of ŷ_O, gate α or shift magnitude have been saved. | **High** | Open — next task |
| 2 | **One transfer benchmark.** The gain is DoTA-only. Nothing distinguishes "motion" from "DoTA". A second benchmark (PreVAD / TAD) is needed. | **High** | Open |
| 3 | **`L_KIP_align` near chance** (3.61–3.70 vs 4.265) with both mitigation knobs off. One of three proposed losses is nearly inert. | Medium | Open |
| 4 | **No validation split, no model selection.** Every arm is `checkpoint_last` at train `mil` ≈ 0.001–0.01 — an arbitrary point deep in the overfit regime, chosen for neither arm. | Medium | Deliberately deferred; the probe showed the endpoint is not special for either arm |
| 5 | **`L_kin` never isolated.** The `use_lkin=false` ablation has not been run; its contribution is unmeasured. | Medium | Open |
| 6 | **`L_KIP_rec` undertrained.** Still improving at step ~446 of 500 in both stages. The reported gain is achieved *without* reconstruction converging. | Low–Medium | Open (upside, if anything) |
| 7 | **Zero-shot only.** No DoTA-trained result exists. Do not write "KAT-VAD on DoTA = X" from this evidence. | Medium | By design |
| 8 | **Extreme score saturation** (86–89 % of DoTA frames > 0.99). Rank metrics are safe; every calibration/threshold metric is meaningless on these arms. | Medium | Characterised (§11.3) |
| 9 | **Coverage 1,397 / 1,402 DoTA clips.** Absolute numbers are not strictly comparable to the published 62.60, which is defined on 1,402. | Low | All arms on the identical subset; paired Δ unaffected |
| 10 | **Metric frame basis.** AUC is over 18,350 (MSAD) / 18,369 (DoTA) stride-8 *sampled* frames, not raw frame counts. | Low | Consistent across all arms |
| 11 | **`--init-weights` is not recorded in `config.yaml`.** A warm-started arm's config is byte-identical to the cold arm's — the one fact the arm exists to establish is absent from its own output. Warm start was verified from step-1 loss instead. | Medium | Provenance defect; a run manifest is needed |
| 12 | **Probe blind spot.** No checkpoint below step 100, where 94 % of the loss range occurs. | Low | Bracketed; verdict unaffected |
| 13 | **n = 3 seeds.** Adequate for the +0.09 (spread ±0.009); inadequate for the macro-variance observation in §8.4. | Low | Stated per-claim |

**Standing discipline:** do **not** tune KIP's architecture, losses, or
hyperparameters against the +0.09. The confounds are closed but the mechanism is
unexplained, and tuning against an unexplained delta is how a benchmark gets fit
rather than a method validated.

---

## 15. Provenance

| Section | Artifacts |
|---|---|
| §1 datasets | `data/MSAD/{meta,defs,labels_train,frame_labels_test}.json`, `data/DoTA/{metadata_val.json,val_split.txt}` |
| §2 parameters | live parameter count from the shipped configs (`kip.enabled` true/false) |
| §3–4 configuration | `outputs/MSAD_ncc*/stage{1,2_*}/config.yaml` (9 files, byte-identical modulo path + seed) |
| §7–8 metrics | `outputs/{MSAD,DoTA}_ncc{,_s2025,_s2026}/*/results.json` and `scores/*.npz` |
| §9 loss trends | `outputs/MSAD_ncc*/stage{1,2_*}/metrics.jsonl`, 20-step trailing means |
| §10 probe | `outputs/DoTA_ncc/probe/{on,off}_checkpoint_step_*/results.json` joined to `metrics.jsonl` by `global_step` |
| §11 mechanism | per-clip AUC deltas grouped by `metadata_val.json → anomaly_class`; center-crop arms from `outputs/{MSAD,DoTA}/*` |

**Statistics.** Micro AUC = `roc_auc_score` over concatenated sampled frames.
Macro AUC = mean per-clip `roc_auc_score`, skipping single-label clips.
Confidence intervals = paired bootstrap resampling **clips** (not frames), 2,000
draws, `default_rng(0)`, both arms scored on the same resample. The macro
bootstrap resamples the precomputed per-clip AUC vector — equivalent to
re-fitting, ~200× faster. No re-inference was performed for this report; every
figure is derived from saved score curves and logs.

**Reproducibility caveat.** `outputs/` is gitignored. The `RESULTS_*.md` chain
and this report are the durable record; the underlying score curves live on the
author's Drive.

**Detailed source documents:** `RESULTS_MSAD.md` (center-crop MSAD, per-class and
per-video breakdowns), `RESULTS_DOTA.md` (the pooling defect and its diagnosis),
`RESULTS_NCC.md` (the transform switch), `RESULTS_PHASE_A.md` (seed campaign,
gate redefinition), `RESULTS_ARM4_PROBE.md` (warm-start control and trajectory
probe). Protocol definitions: `DOTA_EVAL.md`, `TRAINING.md`, `DATA_LAYOUT.md`.
