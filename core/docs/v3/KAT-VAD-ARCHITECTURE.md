# KAT-VAD v3 — Architecture

This document describes **only the model architecture**: every phase in order, with its **input**, **output**, and the **intuition** behind it. Nothing about datasets, training schedules, experiments, or evaluation.

**Lineage.** The skeleton is LaGoVAD (frozen CLIP + temporal encoder + language-conditioned co-attention + dual heads). Phase 3 — the **Kinematic Induction Pathway (KIP)** — is the added component: pseudo-flow generation from Pi-VAD's PMG, an ego-compensated motion residual from DSANet's self-guided normality logic, and a motion-gated adaptive temporal shift from RefineVAD's MoTAR. Phase 7 is Holmes-VAU's ATS pattern.

**What is new in v3 versus v2/v1.** The shift gate is no longer a learned MLP on absolute flow magnitude. It is a **parameter-free deterministic rank map over an ego-compensated motion residual** (Phase 3b–3c). The motion-score head is **removed from the inference graph** (Phase 3e).

---

## Global picture

```
                 ┌──────────────────  TRAIN-TIME ONLY  ────────────────────────┐
                 │  RAFT optical flow → cached embeddings e_O  (target for 3a) │
                 └─────────────────────────────────────────────────────────────┘
                                            ▲
video V ──► [P1] Frame encoder ──► [P2] Temporal encoder ──► [P3] KIP ──► v^k ∈ ℝ^{L×512}
 (frames)        F ∈ ℝ^{L×512}          v^t ∈ ℝ^{L×512}                        │
                                                                               │
definition Z ──► [P4] Text encoder ──► z^t ∈ ℝ^{C×512} ──► [P5] Co-attention fusion
 (C strings)                                                  v^u ∈ ℝ^{L×512}, z^u ∈ ℝ^{C×512}
                                                                               │
                                       ┌───────────────────────────────────────┴──────────┐
                                       ▼                                                  ▼
                            [P6a] Detection head H_bin                        [P6b] Classification head H_mul
                            y^bin ∈ ℝ^{L}  (anomaly curve)                    y^mul ∈ ℝ^{L×C}
                                       │
                                       ▼  (asynchronous, optional)
                            [P7] ATS sampler → MLLM → incident report (text)
```

**Notation.** `L` = number of sampled frames (variable per video; one frame kept every 8). `C` = number of categories in the operator's anomaly definition. `D = 512` = hidden width throughout. `d_O = 256` = flow-embedding width. Frozen: the CLIP image and text encoders. Trainable: temporal encoder, KIP's PMG head and motion head, fusion, heads, projections.

---

## Phase 0 — Inputs

**Input 1 — the video `V`.** Raw stream or clip, sampled every 8 frames, each frame resized to `224×224×3`. After sampling: a sequence of `L` frames.

**Input 2 — the anomaly definition `Z = {z₀,…,z_{C−1}}`.** `C` natural-language strings, each either a class name (*"vehicle collision"*) or a full description (*"a vehicle drives against the direction of traffic flow on a highway"*). Supplied per deployment site, editable at any time without retraining.

**Intuition.** The model does not learn a fixed notion of "anomalous." It learns a *function of the video and the definition*, so the same weights can serve a freeway camera and a school-zone camera with different `Z`.

---

## Phase 1 — Frame encoder (frozen)

| | |
|---|---|
| **Input** | `L × 3 × 224 × 224` sampled frames |
| **Output** | `F ∈ ℝ^{L×512}` — one embedding per sampled frame |
| **Module** | Frozen CLIP ViT-B/16 image encoder (see deviation below) |

**Intuition.** Give the detector eyes that already separate "almost a crash" from "a crash." Alert-CLIP is a CLIP checkpoint retuned with semantic hard negatives (*"white car carefully reversed"* against *"car crashes into multiple vehicles"*), so the normal/abnormal distinction would be partly resolved at the representation level before any head sees the features. Frozen, so nothing here trains.

> **DEVIATION — as-built, 2026-08-30.** The implementation uses **stock CLIP ViT-B/16** at the pinned revision (`constants.CLIP_MODEL_NAME` / `CLIP_MODEL_REVISION`), **not Alert-CLIP**: no Alert-CLIP checkpoint is publicly available. The swap is deferred, not rejected — the paper is on disk (`Zhu_Alert-CLIP_..._CVPR_2026_paper.pdf`) and §10.3 ablation 10 still holds a slot for it. **Every v3 number is a stock-CLIP number**, and the "eyes tuned to abnormality" framing in the one-sentence mental model below is aspirational until that swap happens.

**Geometric constraint.** The transform applied here — crop, resize, aspect handling — must be **identical** to the one used when the RAFT targets of Phase 3a were computed. The pathway regresses motion evidence from its own input; if the input has been cropped or anisotropically squeezed relative to the target, the regression is being asked for evidence it cannot see.

---

## Phase 2 — Temporal encoder

| | |
|---|---|
| **Input** | `F ∈ ℝ^{L×512}` |
| **Output** | `v^t ∈ ℝ^{L×512}` |
| **Module** | 2-layer Transformer, 4 heads, rotary positional encoding, max 1536 positions |

**Intuition.** Per-frame CLIP embeddings know nothing about sequence. This layer supplies **long-range** context — *"traffic was flowing freely two hundred frames ago, now everything is stationary"* — which no local operation can reach. It deliberately stays generic: all kinematic work is delegated to Phase 3, so the pathway remains a clean, separable addition.

**Why it stays even though Phase 3 also mixes across time.** They operate at different ranges and do different jobs. The Transformer is global but motion-blind; the Phase 3 shift is local (±1 position), motion-modulated, and parameter-free. One answers *"what happened over this sequence,"* the other *"how sharply should this instant blend with its immediate neighbours."*

---

## Phase 3 — Kinematic Induction Pathway (KIP)

The pathway is shape-preserving: `v^t ∈ ℝ^{L×512}` in, `v^k ∈ ℝ^{L×512}` out. It is a drop-in replacement at the fusion input.

### 3a — Pseudo-flow generation (PMG head)

| | |
|---|---|
| **Input** | `v^t ∈ ℝ^{L×512}` |
| **Output** | `ê_O ∈ ℝ^{L×256}` — pseudo optical-flow embedding |
| **Structure** | 1D-conv encoder → `ℝ^{L×128}` shared latent → linear translator → `ℝ^{L×128}` flow latent → 1D-conv decoder → `ℝ^{L×256}` |
| **Parameters** | 311,808 |
| **Runs at inference** | **Yes** |

**Train-time supervision.** RAFT is run offline, once, on the training videos; each snippet's flow field is pooled into a target `e_O ∈ ℝ^{L×256}`. The head is trained to reproduce it.

**Intuition.** Teach the RGB stream to *hallucinate* the anomaly-relevant slice of the optical-flow field, then throw the flow backbone away. A crash is a velocity discontinuity, not an appearance change, and CLIP frame embeddings barely encode velocity — so the model has to be taught to recover it. At test time RAFT is never loaded; `ê_O` is regenerated from `v^t` alone.

### 3b — Ego-compensated motion residual (ECMR)

| | |
|---|---|
| **Input** | `ê_O ∈ ℝ^{L×256}` |
| **Output** | `m ∈ ℝ^{L}` — per-position residual motion magnitude |
| **Parameters** | 0 |
| **Runs at inference** | **Yes** |

```
μ_t = Σ_{τ ≤ t} (1−λ)·λ^{t−τ} · ê_{O,τ}        causal EMA, λ = 0.9   →  the clip's own dominant-motion prototype
δ_t = ê_{O,t} − μ_t                             motion residual
m_t = ‖δ_t‖₂
```

**Intuition — and the reason this replaces raw magnitude.** In a moving-camera video the camera's own motion is a *floor* under the flow magnitude at every timestep: the scene is always moving. So `‖ê_O‖` is high nearly everywhere and carries almost no temporal contrast, precisely in the situations where the ego vehicle is involved in the event. Subtracting a running prototype of the clip's own dominant motion converts a **global, uninformative magnitude** into a **local, differential one**: what stands out is motion the clip's recent history does not explain — a third-party vehicle cutting across, an abrupt deceleration, a trajectory that breaks from the flow.

This is the same operator as "model normality from the video itself and flag deviation," applied to the motion channel rather than the appearance channel.

**Correction, measured 2026-08-30.** An earlier draft said "on a fixed camera `μ_t ≈ 0` and the residual reduces to the raw magnitude." **That is wrong.** `μ_t` tracks whatever the clip's own recent motion is, including zero motion, so on a *constant* `ê_O` the prototype converges to the signal itself and the residual goes to **exactly zero**, not to the raw magnitude (`test_constant_flow_gives_zero_residual`). The module still degrades gracefully — a clip with no motion *variation* gets a schedule driven by nothing, which is harmless — but the mechanism is "deviation from the clip's own recent motion", never "raw magnitude". A truly flat residual makes the rank map arbitrary; `rank_map` logs a warning when it detects one.

### 3c — Rank gate

| | |
|---|---|
| **Input** | `m ∈ ℝ^{L}` |
| **Output** | `s ∈ ℤ^{L}`, `s_t ∈ [0, 128]` — number of channels to shift at each position |
| **Parameters** | **0** |
| **Runs at inference** | **Yes** |

```
r_t = rank_t(m) / (L − 1)  ∈ [0,1]              within-clip rank of the residual
s_t = ⌊ r_t · D/K ⌋,   D = 512, K = 4           ⇒  s_t ∈ [0, 128]
```

**Intuition.** The gate answers one question: *at this instant, how much temporal context should be borrowed from the neighbours?*

> **Why this replaced a learned MLP — measured, not argued.** v1 gated on `σ(MLP(m̂))` with a 321-parameter MLP that never received gradient (the `floor` below is non-differentiable and its output is a slice index). Random init was therefore the deployed function. Because `m̂` is min-max normalized, `[0, 1]` is the gate's *entire* reachable input domain — and sweeping it moves `s_t` by **0–4 channels out of 128** across 8 seeds (seed 0: exactly 0; the three MSAD arm seeds: 1, 3, 3). v1's gate was a **fixed ~50 % shift at `s ≈ 58–69`**, statistically indistinguishable from `gate_type="constant"` at `r = 0.5`. Hypothesis H4′ ("KIP is a domain-conditioned constant smoother") is **confirmed for v1**. The rank map spans `[0, 128]` on every clip by construction, so it cannot degenerate that way. Ranking within the clip makes that decision **scale-free** — it does not matter whether the footage is a dashcam at highway speed or a static parking-lot camera, because only the *ordering* of residual motion inside that clip is used. It also guarantees the shift schedule spans its full range on every clip, so the module can never collapse into a constant smoother that ignores its input.

`K = 4` is the folding factor: it caps the shift at `D/K = 128` channels so that the past slice, the future slice, and the retained present slice remain non-overlapping.

### 3d — Adaptive bidirectional temporal shift

| | |
|---|---|
| **Input** | `v^t ∈ ℝ^{L×512}`, `s ∈ ℤ^{L}` |
| **Output** | `v^k ∈ ℝ^{L×512}` |
| **Parameters** | 0 |
| **Runs at inference** | **Yes** — this is the *only* KIP operation on the score path |

```
v^k_t = [ v^t_{t−1}(1 : s_t) ,  v^t_{t+1}(s_t : 2s_t) ,  v^t_t(2s_t : 512) ]
```

The first `s_t` channels come from the **past** position, the next `s_t` from the **future** position, and the remaining `512 − 2s_t` are kept from the **present**. Dimensions sum to 512, so the shape is preserved. Boundary positions (`t = 1`, `t = L`) zero-pad the missing neighbour.

**Intuition.** A collision instant should aggressively pull context from before (the approach) and after (the aftermath) — high residual motion, large `s_t`. A vehicle idling at a red light should be left alone — low residual, `s_t ≈ 0`, no contamination of stable content. The amount of temporal mixing is tied to how much *unexplained* motion is present, not to how much motion is present in total.

**Implementation note.** Four gate types are selectable (`kip.gate_type`): `rank` (this design, default), `mlp_frozen` (v1, bit-identical — reproduces every number in `core/docs/RESULTS_*.md`), `mlp_ste` (v1 forward with the backward pass unblocked), and `constant` (fixed ratio, `ê_O` ignored — the plain-TSM control). Only the `mlp_*` types own parameters. **Under a `rank` or `constant` gate `s_t` is a hard integer used as a slice index, so no gradient reaches `ê_O` or the PMG head through the shift — as was already true in v1.** The PMG head is trained by `L_KIP_rec`, `L_KIP_align` and `L_kin`-via-`mhead`, and by nothing else, which makes stage-1 convergence a precondition rather than a nicety.

### 3e — Motion-score head (train-time only)

| | |
|---|---|
| **Input** | `ê_O ∈ ℝ^{L×256}` |
| **Output** | `ŷ_O ∈ ℝ^{L}` — a motion-only anomaly curve |
| **Parameters** | 33,025 |
| **Runs at inference** | **No — not instantiated** |

**Intuition.** During training this head forces the pseudo-flow stream to be *independently* discriminative: the motion evidence alone must be able to tell an abnormal video from a normal one, and must agree with the main detector about *where* in an abnormal video the event sits. That pressure is what keeps `ê_O` anomaly-relevant rather than merely a faithful reconstruction. At test time the curve is not consumed by anything, so the head is left out of the graph entirely.

### 3f — Alignment projections (train-time only)

| | |
|---|---|
| **Input** | `ê_O`, `v^t` |
| **Output** | projected pairs for a contrastive objective |
| **Parameters** | 98,560 (`proj_flow` 32,896 + `proj_rgb` 65,664) |
| **Runs at inference** | **No** |

**Intuition.** Bind the induced flow embedding and the RGB representation into a shared space so the two streams describe the same instant in compatible terms. The contrast is computed over **pooled groups of 4 positions** rather than single positions, because adjacent single-frame CLIP embeddings of a driving clip are near-identical and would otherwise be treated as negatives of one another.

### KIP summary

| Sub-module | Params | Inference | Output |
|---|---:|:---:|---|
| 3a PMG head | 311,808 | ✅ | `ê_O ∈ ℝ^{L×256}` |
| 3b ECMR | 0 | ✅ | `m ∈ ℝ^{L}` |
| 3c Rank gate | **0** | ✅ | `s ∈ ℤ^{L}` |
| 3d Adaptive shift | 0 | ✅ | `v^k ∈ ℝ^{L×512}` |
| 3e Motion head | 33,025 | ❌ | `ŷ_O ∈ ℝ^{L}` |
| 3f Projections | 98,560 | ❌ | contrastive pairs |

---

## Phase 4 — Definition encoder (frozen)

| | |
|---|---|
| **Input** | `Z` — `C` natural-language strings |
| **Output** | `z^t ∈ ℝ^{C×512}` |
| **Module** | Frozen CLIP text encoder, with 32 soft prompts |

**Intuition.** Turn the operator's written rules into vectors the video stream can be compared against. Because the encoder is frozen and the definition changes rarely, `z^t` is computed **once per definition and cached** — changing what counts as anomalous costs one text-encoder pass, not a retraining run.

---

## Phase 5 — Co-attention fusion

| | |
|---|---|
| **Input** | `v^k ∈ ℝ^{L×512}`, `z^t ∈ ℝ^{C×512}` |
| **Output** | `v^u ∈ ℝ^{L×512}`, `z^u ∈ ℝ^{C×512}` |
| **Module** | 2 layers, 8 heads; each modality cross-attends to the other, then a feed-forward block |

**Intuition.** This is where the anomaly score becomes *conditional on what the operator wrote*. Each frame representation is re-expressed in terms of how well it matches the definition, and each definition vector is re-expressed in terms of what the video actually contains. Placing fusion **before** scoring rather than after is what makes the definition able to change the score at all — if it came afterwards, the detector would have already committed to a fixed notion of abnormality.

---

## Phase 6a — Detection head `H_bin`

| | |
|---|---|
| **Input** | pre-fusion `v^k` (language-agnostic path) + post-fusion `v^u` (language-guided path) |
| **Output** | `y^bin ∈ ℝ^{L}` — per-frame anomaly score in [0,1] |
| **Module** | 1D convolution, kernel 9, replicate padding, one layer per path; the two paths combined by a learnable scalar. Adaptive binary head (α₀ = 0.25, scale 10.0) |

**Intuition.** The language-agnostic path catches *"this looks generically wrong"*; the language-guided path catches *"this matches the operator's definition"*; the learnable blend lets the data decide the mix. The kernel-9 convolution is a ~9-position local smoother, matching the seconds-scale span of a collision.

**This curve is the detection deliverable.** Everything downstream consumes it; nothing downstream modifies it.

---

## Phase 6b — Classification head `H_mul`

| | |
|---|---|
| **Input** | `v^u ∈ ℝ^{L×512}`, `z^u ∈ ℝ^{C×512}` |
| **Output** | `y^mul ∈ ℝ^{L×C}` — per-frame probability over the definition's categories |
| **Module** | similarity matrix between linearly projected modalities (temperature 0.2); min-over-time on the normal class, max-over-time on the abnormal class, softmax at video level |

**Intuition.** Incident triage. Distinguishing *"vehicle–vehicle collision"* from *"pedestrian on carriageway"* changes the emergency response. Because the categories come from `Z` rather than from a fixed label set, the head transfers to a new taxonomy by rewriting the definition.

---

## Phase 7 — ATS reasoning layer (asynchronous, optional)

| | |
|---|---|
| **Input** | the frozen curve `y^bin` + raw frames |
| **Output** | a natural-language incident report |
| **Module** | inverse-CDF sampler over `y^bin` treated as a density → MLLM (LoRA adapter) |

**Intuition.** The score curve says *where* to look; the MLLM says *what happened*. Sampling frames in proportion to the anomaly density concentrates a fixed frame budget on the incident window rather than spreading it uniformly over a mostly-normal stream. The layer consumes the curve and never feeds back into it, so it adds **zero detection latency** and detaches cleanly if the deployment wants no LLM at all.

**Precondition.** Inverse-CDF sampling is only meaningful if `y^bin` has real dynamic range. On a saturated curve it degenerates into uniform sampling and the layer is pointless.

---

## Train-time-only auxiliary signals attached to KIP

These define what `ê_O` and `ŷ_O` *are*; without them Phase 3 has no meaning. They exist only during training.

| Signal | Attaches to | Form |
|---|---|---|
| **Flow reconstruction** | 3a | masked MSE between `ê_O` and the cached RAFT target `e_O`; left unweighted so the pseudo-flow stays bounded to the real flow manifold |
| **Cross-modal alignment** | 3f | bidirectional InfoNCE between `proj_flow(ê_O)` and `proj_rgb(v^t)` over pooled groups of 4 positions, with a ±1-group temporal exclusion window and negatives drawn both from the same video and from other videos in the batch |
| **Kinematic consistency** | 3e | top-k MIL on `ŷ_O`, plus a scale-free agreement term — KL divergence between the softmax-over-time of `y^bin` (stop-gradient) and of `ŷ_O` — so agreement depends on *where* the mass sits, not on absolute score level |

Alongside these, the baseline's own objectives operate on Phases 5–6 unchanged: MIL detection, MIL-align classification, duration-diversifying synthesis, and same-scene hard-negative contrastive alignment.

---

## What runs at inference, and what does not

| Component | Params | At inference |
|---|---:|:---:|
| Frozen CLIP image encoder | — (frozen) | ✅ |
| Temporal encoder | — | ✅ |
| KIP 3a PMG head | 311,808 | ✅ |
| KIP 3b ECMR | 0 | ✅ |
| KIP 3c Rank gate | **0** | ✅ |
| KIP 3d Adaptive shift | 0 | ✅ |
| Frozen CLIP text encoder | — (frozen) | ✅ (cached per definition) |
| Co-attention fusion | — | ✅ |
| `H_bin`, `H_mul` | — | ✅ |
| KIP 3e Motion head | 33,025 | ❌ |
| KIP 3f Projections | 98,560 | ❌ |
| RAFT optical-flow backbone | — | ❌ (offline, once, training only) |
| ATS + MLLM | — | ⏸ asynchronous, detachable |

**Cost of the pathway.**

| | Count | vs. baseline (18,923,524) |
|---|---:|---:|
| KIP total, training | 443,393 | +2.34 % |
| KIP total, training, alignment retired | 344,833 | +1.82 % |
| **KIP on the inference path** | **311,808** | **+1.65 %** |
| **KIP on the score path** | **0** | **0.00 %** |

The last row is the point: after the pseudo-flow embedding is produced, everything the pathway does to the anomaly score — the residual, the rank map, the shift — is **deterministic and parameter-free**. Inference is RGB plus a cached text embedding, and the optical-flow backbone does not exist at test time.

---

## One-sentence mental model

*Look at each frame with eyes tuned to abnormality (aspirational — see the Phase 1 deviation; as built, stock CLIP), give them sequence memory, teach them to hallucinate the optical flow they cannot see and to notice the motion that the clip's own recent history fails to explain, let that surprise decide how sharply each instant blends with its neighbours, compare the result against what the operator wrote is anomalous, and read off a per-frame score — then, only where the score is high, ask a language model what happened.*
