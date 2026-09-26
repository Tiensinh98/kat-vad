# Proposed Road-Traffic Video Anomaly Detection Architecture — **KAT-VAD**
### (Kinematics-Aware, definition-conditioned Traffic VAD — LaGoVAD baseline + Kinematic Induction Pathway)

---

## 0. Design summary

KAT-VAD is a **weakly-supervised, language-definition-conditioned traffic anomaly detector** built on **LaGoVAD (ICLR 2026)** — the only venue-eligible baseline natively evaluated on real traffic datasets (DoTA 62.60 / TAD 89.56 AUC) and shipping a traffic-inclusive dataset (PreVAD, with a Vehicle-Accident first-level category and traffic-camera highway feeds). LaGoVAD's one documented weakness — a deliberately simple CLIP-frame backbone with **no motion modeling** (its lowest score is exactly on the ego-centric, motion-dominated DoTA) — is closed by the novel component: a **Kinematic Induction Pathway (KIP)**, derived from Pi-VAD's Pseudo-Modality Generation (train-time-only optical-flow induction, RGB-only inference) and RefineVAD's MoTAR (motion-adaptive temporal recalibration), with a **kinematic-consistency MIL loss** that forces the anomaly location chosen by the detector to be supported by induced motion evidence. Modalities: **RGB (inference) + optical flow (train-time-only, induced) + language definition** — lean and temporal/kinematic-context-motivated. An **LLM reasoning layer sits off the critical path** via Holmes-VAU's Anomaly-focused Temporal Sampler (ATS), generating incident reports only for score-dense windows. Backbone: **frozen stock CLIP ViT-B/16** (`openai/clip-vit-base-patch16`, the same encoder LaGoVAD uses) for both the frames and the definition text. No Alert-CLIP checkpoint is used; its weights are not public (§3.2a). Target data: **PreVAD (train) + MSAD Protocol-ii traffic slice (train/eval, fixed-camera) + DoTA & DADA-2000 (eval, ego-centric) + TAD**, with UCF-Crime RoadAccidents retained only as a comparability bridge. Every noun above is defended below.

---

## 1. Gap analysis (per-paper, w.r.t. road traffic)

A load-bearing fact from the project's evidence base: **only 2 of 14 papers (SimpleTAD, LaGoVAD) ever touch real traffic data.** All others treat "RoadAccidents" as 1 of 13 UCF-Crime classes.

| Paper | One-line traffic gap | Reusable idea for KAT-VAD |
|---|---|---|
| **Pi-VAD** (arXiv 2025) | Never trained/evaluated on a traffic dataset; modalities chosen for human-centric anomalies; arXiv-only venue | **PMG train-time-only modality induction** (RGB-only inference, ~30 FPS); motion = largest AUC gain (Table 6: 87.92); two-step warm-up→task training |
| **DSANet** (AAAI 2026) | No motion modeling (CLIP+GCN only); RoadAccidents 1/13 | SG-NM self-guided normality (traffic flow is highly regular) — optional aux branch |
| **RefineVAD** (AAAI 2026) | Motion captured only as CLIP-feature variance, a cheap proxy; no traffic benchmark; drops at high IoU | **MoTAR motion-adaptive temporal shift** — high-motion segments borrow more temporal context; CORE soft prototypes transfer cross-dataset (77.56 AP zero-shot) |
| **Alert-CLIP** (CVPR 2026) | A backbone, not a detector; image-CLIP core, thin temporal model | **Abnormality-tuned CLIP checkpoint + semantic hard negatives** ("white car carefully reversed" vs "car crashes into multiple vehicles") — drop-in backbone swap |
| **LAVAD** (CVPR 2024) | Per-frame caption+LLM too slow; below trained SOTA; no motion | Cross-modal score refinement; LLM temporal summary as *explanation* only |
| **VERA** (CVPR 2025) | Hard-coded anomaly-in-the-middle Gaussian prior — wrong for dashcam streams where crashes happen anywhere | Learned traffic guiding questions ("is any vehicle's trajectory discontinuous?"); its midpoint prior is a documented flaw the ATS layer replaces |
| **AnomalyRuler** (ECCV 2024) | One-class, not WS; campus benchmarks | Induced language rule set as a normality prior ("vehicles move with the flow, stop at signals") |
| **Holmes-VAU** (CVPR 2025) | General VAU; no traffic benchmark; no motion | **ATS**: cheap scorer decides *where* the expensive MLLM looks — the correct LLM-integration pattern for long, mostly-normal traffic streams |
| **LAVIDA** (CVPR 2026) | Degrades on low-res (roadside CCTV is low-res); no temporal onset labels | Query-conditioned anomaly definition; pixel decoder if spatial localization becomes a deliverable |
| **AnyAnomaly** (WACV 2026) | Training-free ceiling; segment-level only | User-text anomaly definition; temporal-context grid as cheap motion surrogate |
| **PANDA** (NeurIPS 2025) | ~0.8 FPS; no spatial localization; no motion | RAG-grounded traffic-rule planning; "Insufficient" deferral for ambiguous frames |
| **SimpleTAD** (CVPR 2025) | **Fully-supervised** (frame labels) → ineligible as WS baseline; ego-centric only; no LLM/multi-modality | **DAPT on unlabeled normal driving video** (no anomalies needed); motion-aware MVM wins; DoTA/DADA-2000 as eval anchors; **MCC/AUCMCC** for imbalance |
| **LaGoVAD** (ICLR 2026) | Deliberately simple backbone, **no motion pathway** → weak on DoTA (62.60) | **The baseline.** Language definition `Z` kills concept drift; same-scene hard negatives; PreVAD dataset; duration-diversifying synthesis |
| **VideoRAG** (2025) | Not a detector; no anomaly scoring | Optional cross-video retrieval memory ("find past clips similar to this near-miss") — out of scope for v1 |

**Cross-cutting gaps this architecture targets** (from the synthesis):
- **Gap A — Domain data:** must train/evaluate on DoTA / DADA-2000 / PreVAD, not just UCF-Crime RoadAccidents. → §7.
- **Gap B — Motion:** traffic anomalies are *kinematic* (velocity/heading discontinuity), yet the eligible WS baselines model motion weakly or not at all. → the KIP novel component, §4.
- **Gap C — WS + traffic + strong backbone:** no paper is simultaneously WS + traffic-native + multi-modal + explainable. **KAT-VAD fills exactly this hole.**
- **Gap D — Concept drift:** "pedestrian on road" flips normal↔abnormal by location. → LaGoVAD's definition conditioning, kept intact.
- **Gap E — Explanation off the latency path:** → ATS reasoning layer, §6.
- **Gap F — Structured traffic normality:** → optional SG-NM aux branch + DAPT-style self-supervised warm-up, §8.

---

## 2. Baseline selection (compared, not assumed)

Eligibility: weakly-supervised, official working GitHub, CVPR/AAAI/ICLR 2025–26 preferred, motion/traffic-relevant.

| Criterion | Pi-VAD (arXiv 5/2025) | DSANet (AAAI 2026) | RefineVAD (AAAI 2026) | **LaGoVAD (ICLR 2026)** |
|---|---|---|---|---|
| Weakly-supervised | ✅ | ✅ | ✅ | ✅ (open-world WS) |
| Venue 2025–26 | ❌ arXiv | ✅ | ✅ | ✅ |
| Official code | ✅ | ✅ | ✅ | ✅ (LaGoVAD-PreVAD) |
| Explicit motion modeling | ✅✅ (flow induced) | ❌ | ✅ (MoTAR variance proxy) | ❌ |
| Multi-modal | ✅✅✅ (5, train-only) | ⚠️ CLIP text | ⚠️ CLIP+text | ✅ RGB + language definition |
| Real-time RGB-only inference | ✅ ~30 FPS | ✅ | ✅ | ✅ (single RTX 4090) |
| **Traffic/accident evidence** | ✅ RoadAccident-127; ~2× UR-DMU on Explosion | ⚠️ 1/13 class | ⚠️ 1/13 class | ✅✅ **DoTA 62.60 / TAD 89.56; PreVAD Vehicle-Accident + traffic-cam feeds** |
| Concept-drift robustness | ❌ | ❌ | ⚠️ category transfer | ✅✅ (drift@5 protocol is its thesis) |
| Normality prior | ⚠️ | ✅✅ SG-NM | ⚠️ | ⚠️ |
| LLM extensibility | ✅ | ✅ | ✅ | ✅ (natively language-conditioned) |

**Pick: LaGoVAD**, because the user's stated goal is *road-traffic* WS-VAD and LaGoVAD is the **only eligible baseline already tested on real traffic datasets** (DoTA/TAD) with a traffic-inclusive training set (PreVAD). Its concept-drift framing directly handles the defining traffic ambiguity ("pedestrian on road is normal on a footpath, abnormal on a freeway" — the paper's own example). Crucially, its one real weakness (no motion pathway; DoTA is its lowest score at 62.60) is exactly the gap a coherent novel component can close — the strongest thesis narrative pairs a baseline's weakness with a fix derived from another project paper.

**Why the alternatives lost (recorded, and you can override):**
- **RefineVAD** — the strongest *motion-native* venue-eligible alternative (MoTAR). Lost because it has zero traffic evidence beyond the UCF-Crime slice (Gap A), its motion signal is a feature-variance proxy rather than true kinematics, and it documents boundary-precision drops at high IoU. It is the recommended fallback if you prioritize a peer-reviewed *motion* pedigree over traffic-native evaluation; in that case swap the novel component to a normality/hard-negative branch (RefineVAD already has motion).
- **DSANet** — best normality prior (SG-NM) but completely motion-blind and no traffic data; SG-NM is instead imported as an optional auxiliary branch.
- **Pi-VAD** — strongest on motion + multi-modality + traffic-class evidence, but arXiv-only (fails venue strictness). Imported as the *source* of the novel component instead of the baseline — the best of both.

---

## 3. Proposed architecture (with per-decision justification)

### 3.0 Notation & problem formulation

A video `V` of `L` sampled frames (sample every 8 frames, LaGoVAD convention; DoTA at its provided 10 fps) plus a **language anomaly definition** `Z = {z₀,…,z_{C−1}}` (class names or free-text descriptions, e.g. *"vehicle collision", "pedestrian on carriageway", "vehicle driving against traffic flow"*). Training uses **video-level labels only** (weak supervision, MIL). The model learns `Φ : (V, Z) → Y`, per LaGoVAD Eq. 2 — conditioning on `Z` makes `P(Y|V,Z)` domain-invariant (LaGoVAD Prop. 1), which is what lets one trained model serve a school zone and a freeway with different definitions and no retraining.

### 3.1 Data-flow overview

```
video V ──► Frozen CLIP ViT-B/16 image encoder ──► F ∈ ℝ^{L×512}
                    │
                    ▼
        Temporal encoder (2-layer Transformer, RoPE) ──► v^t ∈ ℝ^{L×512}
                    │                                        │
                    ▼                                        │
   ┌──────── KIP (novel) ────────┐                           │
   │ PMG-flow head: v^t → ê_O    │  train-time target:       │
   │ ê_O ∈ ℝ^{L×d_O}             │  RAFT flow embeddings e_O │
   │ Kinematic gate: ê_O → r_t   │                           │
   │ adaptive temporal shift     │                           │
   └──────────► v^k ∈ ℝ^{L×512} ◄┘                           │
                    │                                        │
                    ▼                                        ▼
      definition Z ──► Frozen CLIP text encoder G ──► z^t ∈ ℝ^{C×512}
                    │
                    ▼
        Co-attention fusion U(v^k, z^t) ──► v^u ∈ ℝ^{L×512}, z^u ∈ ℝ^{C×512}
                    │
        ┌───────────┴─────────────┐
        ▼                         ▼
  H_bin (1D-conv, k=9)      H_mul (similarity head)
  y^bin ∈ ℝ^{L}             y^mul ∈ ℝ^{L×C}
  per-frame anomaly score   per-frame class probability
        │
        ▼ (off critical path, optional)
  ATS inverse-CDF sampler ──► MLLM ──► incident report (text)
```

### 3.2 Component-by-component

**(a) Frame encoder — frozen stock CLIP ViT-B/16 (as implemented).**
*Input:* raw frames `L×3×224×224`. *Output:* `F ∈ ℝ^{L×512}` frame embeddings.
*What runs:* **stock OpenAI CLIP ViT-B/16** (`openai/clip-vit-base-patch16`, revision pinned in `core/constants.py:CLIP_MODEL_REVISION`). It is **fully frozen**, applied to every 8th frame (`FRAME_STRIDE = 8`) with the no-center-crop transform. Features are cached offline, once per corpus. The **same CLIP model's text tower** (frozen, with LaGoVAD's learnable soft prompts) encodes the definition `Z` (§3.2d), so frames and definitions share one embedding space. This is **exactly what LaGoVAD uses**, so the baseline reproduction is honest and the DoTA/TAD numbers are comparable. The contribution that closes the traffic gap is KIP (§4), not the backbone.
*Alert-CLIP is **not** used.* Why we originally wanted it: Alert-CLIP diagnoses that normal/abnormal text is *entangled* in stock CLIP space and repairs it with multi-level contrastive tuning whose hard negatives are literally traffic examples ("car crashes into multiple vehicles" vs "white car carefully reversed") — directly attacking traffic's dominant failure mode (false positives on hard braking / aggressive merges). **But Alert-CLIP is a CVPR 2026 paper whose pretrained weights are not publicly downloadable** (only its dataset repo is public). We therefore do **not** make the architecture depend on them. The options below were considered. **Only option 1 is implemented.** Options 2–4 are recorded for future work and are not part of the current architecture:

  1. **Implemented — stock CLIP ViT-B/16 (no external weights beyond OpenAI CLIP).** This is what every result in this project was measured with.
  2. **Not implemented (self-contained) — CLIP + a lightweight *Traffic Hard-Negative Tuning* stage (Stage 0.5, §8).** Reproduces Alert-CLIP's *effect* without its checkpoint by fine-tuning the (otherwise frozen) CLIP text/vision alignment on same-scene + semantic traffic hard-negative pairs ("car brakes hard and stops safely" vs "car collides"), reusing LaGoVAD's `L_neg` machinery + the traffic hard-negative text bank (§4.4). This turns a missing dependency into a reproducible contribution of ours.
  3. **Not implemented (downloadable substitute) — a stronger open CLIP** from `open_clip` (e.g. a LAION-2B ViT-L key via `open_clip.list_pretrained()`), then apply option 2's tuning on top. Use if a bigger generic backbone helps.
  4. **Not implemented (opportunistic) — Alert-CLIP swap, gated behind weight availability.** Keep the encoder behind a single config flag; if the CVPR camera-ready checkpoint is released (watch `github.com/ClarkZhu216/Alert-CLIP_dataset`), flip the flag and re-run as an ablation. Never a blocker.

*Intuition:* keep LaGoVAD's own eyes, so any gain is attributable to what KAT-VAD adds on top of them rather than to a different encoder. A traffic-tuned encoder (options 2–4) remains a separate, later question.

**(b) Temporal encoder.**
*Input:* `F ∈ ℝ^{L×512}`. *Output:* `v^t ∈ ℝ^{L×512}`.
2-layer vanilla Transformer with rotary positional encoding (LaGoVAD App. C.1). Kept as-is: it provides sequence context; the kinematic work is delegated to KIP so the baseline remains recognizable and the ablation clean (baseline vs baseline+KIP).

**(c) Kinematic Induction Pathway (KIP) — the novel component.** Full detail in §4. *Input:* `v^t ∈ ℝ^{L×512}` (+ train-time RAFT embeddings `e_O ∈ ℝ^{L×d_O}`). *Output:* kinematically recalibrated features `v^k ∈ ℝ^{L×512}` and an auxiliary motion score curve `ŷ_O ∈ ℝ^{L}`.
*Intuition:* a crash is a **velocity discontinuity, not an appearance change**. CLIP frame features barely encode it; KIP teaches the model to hallucinate the flow field from RGB (Pi-VAD PMG) and to let *how much things move* control *how much temporal context each instant borrows* (RefineVAD MoTAR, upgraded from a variance proxy to true induced kinematics).

**(d) Definition encoder & fusion.**
*Input:* definition `Z` (C strings). Frozen CLIP text encoder `G` → `z^t ∈ ℝ^{C×512}`. Co-attention fusion `U` (2 layers; each modality cross-attends to the other, then FFN) over `(v^k, z^t)` → fused `v^u ∈ ℝ^{L×512}`, `z^u ∈ ℝ^{C×512}`.
*Source & why:* verbatim LaGoVAD. This is the concept-drift mechanism: the LaGoVAD ablation shows that placing fusion *after* detection (i.e., not conditioning scores on text) collapses cross-domain classification (Cls-avg 52.57 → 46.23) — the definition must gate the score path. For traffic, `Z` is where the operator writes "stopped vehicle on shoulder = abnormal" for a freeway camera and omits it for a parking-lot camera.

**(e) Detection head `H_bin`.**
*Input:* pre-fusion `v^k` (language-agnostic path) + post-fusion `v^u` (language-guided path). *Output:* `y^bin ∈ ℝ^{L}`, per-frame anomaly score in [0,1].
1D-conv, kernel 9, replicate padding, two pathways fused by a learnable scalar (LaGoVAD). Kernel-9 conv = a ~9-frame local temporal smoother, matching the seconds-scale span of a collision.
*Intuition:* the language-agnostic path catches "this looks generically wrong"; the language-guided path catches "this matches the operator's definition"; the learnable blend lets the data decide the mix.

**(f) Classification head `H_mul`.**
*Input:* `(v^u, z^u)`. *Output:* `y^mul ∈ ℝ^{L×C}` per-frame probability over the definition's categories (similarity matrix between linearly projected modalities; min-over-time on normal, max-over-time on abnormal, Softmax at video level).
*Why for traffic:* incident triage — distinguishing "vehicle–vehicle collision" from "pedestrian on carriageway" changes the emergency response, and LaGoVAD's classification transfers zero-shot to new taxonomies.

**(g) ATS reasoning layer (off critical path).** §6. *Input:* frozen `y^bin` curve + raw frames. *Output:* natural-language incident report.

---

## 4. Novel component / loss — the Kinematic Induction Pathway (KIP)

**Name:** Kinematic Induction Pathway + Kinematic-Consistency MIL loss.
**Derived from:** Pi-VAD (PMG train-time-only optical-flow induction + InfoNCE alignment; motion is its single best modality on AUC, Table 6: 87.92 vs 86.97 baseline) and RefineVAD (MoTAR motion-adaptive temporal shift), with SimpleTAD's finding (motion-aware MVM objectives beat pixel/semantic ones on DoTA/DADA) as domain evidence that motion-awareness is *the* winning ingredient in traffic.
**Exact gap closed:** LaGoVAD's authors themselves flag "room for better temporal/multimodal modeling"; its worst benchmark is ego-centric DoTA (62.60), where kinematics dominate — this is Gap B instantiated on the chosen baseline. KIP adds motion **without breaking LaGoVAD's cheap RGB-only inference**, because the flow backbone (RAFT) exists only at training time (Pi-VAD's PMG pattern, which costs only ≈0.25% AUC vs real modality features while dropping ~2,561 → ~20 GFLOPs in Pi-VAD's measurements).

### 4.1 Sub-module 1 — Pseudo-flow generation (PMG-style)

*Structure (mirrors Pi-VAD PMG, single modality):* a 1D-conv encoder projects `v^t` to a low-dim shared latent; one linear layer translates it to a flow-specific latent; a 1D-conv decoder emits the pseudo-flow embedding.

| Step | Input | Output |
|---|---|---|
| 1D-conv encoder | `v^t ∈ ℝ^{L×512}` | shared latent `h ∈ ℝ^{L×128}` |
| linear translator | `h` | flow latent `h_O ∈ ℝ^{L×128}` |
| 1D-conv decoder | `h_O` | pseudo-flow embedding `ê_O ∈ ℝ^{L×d_O}` (d_O ≈ 256) |

*Train-time supervision:* RAFT (the flow extractor Pi-VAD uses) is run on the training videos; a small frozen flow encoder pools each snippet's flow field into `e_O ∈ ℝ^{L×d_O}`. Two auxiliary losses, exactly Pi-VAD's recipe:

```
L_KIP-rec   = (1/d_O) Σ_k ( ê_{O,k} − e_{O,k} )²                       (MSE; keeps ê_O bounded to the real flow manifold)
L_KIP-align = bidirectional snippet-level InfoNCE( ê_O , v^t )          (positives = same snippet index; negatives = other snippets)
```

*Why both:* Pi-VAD's ablation is unambiguous — PMG reconstruction alone *hurts* (84.66 < 86.97 baseline; decoupled, unguided modalities are noise), and alignment+distillation is what makes induced modalities usable. KIP keeps the reconstruction + alignment pair; the "distillation" role is played by the kinematic-consistency loss below, which ties the flow stream to the *task* rather than to a frozen teacher (LaGoVAD has no UR-DMU-style teacher to distill toward).
*Intuition:* teach the RGB stream to hallucinate the anomaly-relevant slice of the optical-flow field, then throw RAFT away.

### 4.2 Sub-module 2 — Kinematic gate + adaptive temporal shift (MoTAR, upgraded)

RefineVAD's MoTAR drives its shift ratio from `Var(x_t − x_{t−1})` — a **CLIP-feature variance proxy** that the gap analysis flags as its weakness ("may miss slow-onset anomalies — a car drifting across lanes"). KIP's twist: drive the gate from the **induced kinematic embedding**, which actually encodes flow magnitude and direction.

```
m_t = ‖ê_{O,t}‖₂  (or a learned scalar MLP(ê_{O,t}))                     m ∈ ℝ^{L}      kinematic intensity
r_t = σ( MLP(m_t) ) ∈ [0,1]                                                              shift ratio (RefineVAD Eq. 1 form)
s_t = ⌊ r_t · 512/K ⌋                                                                    channels to shift (K = folding factor)
v^k_t = [ v^t_{t−1}(1:s_t)  v^t_{t+1}(s_t:2s_t)  v^t_t(2s_t:512) ]                       bidirectional adaptive shift (RefineVAD Eq. 3)
```

*Input:* `v^t ∈ ℝ^{L×512}`, `ê_O ∈ ℝ^{L×d_O}`. *Output:* `v^k ∈ ℝ^{L×512}` (shape-preserving; drop-in for `v^t` at the fusion input).
*Intuition (traffic-specific):* a collision instant should aggressively pull context from the frames before (approach) and after (aftermath) — high `m_t` → large `s_t`; a car idling at a red light should be left alone — near-zero `m_t` → no contamination of stable content. A *slow lane drift* produces sustained moderate flow that a per-step feature-variance proxy underestimates but a flow-norm gate registers — the precise failure mode the upgrade fixes.

### 4.3 Sub-module 3 — Kinematic-Consistency MIL loss (the novel loss)

An auxiliary motion score head reads the pseudo-flow alone:

```
ŷ_O = σ( MLP(ê_O) ) ∈ ℝ^{L}          motion-only anomaly curve
```

```
L_kin = L_MIL(ŷ_O)  +  β · Σ_{abnormal videos}  smoothL1( topk-region(ŷ_O) , sg(topk-region(y^bin)) )
```

- The first term is a standard top-k MIL ranking on the motion-only curve → the flow stream must *itself* discriminate abnormal from normal videos.
- The second term (stop-gradient on the main score) requires the main detector's chosen top-k anomaly window and the motion stream's window to **agree** on abnormal videos.

*Exact gap this loss closes & why it helps traffic:* WS-MIL detectors are notorious for latching onto **appearance shortcuts** — wet-road glare, headlights at night, dynamic advertising screens (the MSAD paper explicitly lists these as false-positive drivers). By demanding that every claimed anomaly window be *kinematically corroborated*, the loss suppresses appearance-only false positives while leaving genuinely kinematic events (crash, sudden stop, wrong-way driver) untouched. It is the traffic translation of Pi-VAD's finding that motion is the primary normal/abnormal discriminator, wired into LaGoVAD's MIL machinery.

### 4.4 Secondary (optional) component — Traffic hard-negative text bank

LaGoVAD already mines **same-scene visual hard negatives** (its `L_neg`: the anomaly video's own normal portion is the hard negative for its description — same scene, lighting, camera). The optional secondary extends the **text side** with Alert-CLIP-style semantic hard negatives: for each traffic definition, curated near-miss phrasings ("vehicle brakes hard and stops safely", "car reverses carefully into a space") added as extra negative rows in the `L_neg` similarity matrix. Kept secondary and off by default: it is an ablation. Because the backbone is stock CLIP (§3.2a), no traffic-specific hard negatives enter at the representation level.

---

## 5. Multi-modality justification

| Modality | Status | Justification (paper-cited) |
|---|---|---|
| **RGB** | inference + training | The only signal guaranteed at every deployment (dashcam, roadside CCTV) |
| **Optical flow / motion** | **train-time only, induced (KIP)** | #1 extra modality: Pi-VAD Table 6 — motion gives the largest single-modality AUC gain; SimpleTAD — motion-aware MVM objectives win on DoTA/DADA; traffic anomalies are definitionally kinematic (Gap B). Train-time-only induction (PMG) keeps inference RGB-cheap — Pi-VAD shows regeneration costs only ≈0.25% AUC vs. running the real backbone |
| **Language (definition + text encoder)** | inference + training | Not a raw perception modality — it buys **concept-drift robustness** (LaGoVAD Prop. 1; drift@5 85.7/85.6 vs Qwen2.5-VL 62.7) and operator customization per deployment site. Already native to the baseline, so free |
| **Depth** | *deferred* (iteration hook) | Pi-VAD shows depth dominates AUC_A (inter-entity spatial interaction — vehicle proximity, pedestrian distance-to-carriageway). Deferred from v1 to keep the contribution focused (one novel pathway, clean ablation); adding a second PMG decoder head supervised by DepthAnythingV2 is a mechanical extension |
| **Pose** | excluded | Human-centric (Pi-VAD: pose+depth drive Fighting/Shoplifting); helps pedestrian-on-highway but not vehicle–vehicle collisions, which dominate traffic anomaly mass |
| **Panoptic masks** | excluded | Scene-oriented and expensive; Pi-VAD shows text+panoptic drive *scene-based* classes (Burglary, Arson), not kinematic ones |
| **Audio** | excluded | Usually absent in traffic CCTV and unreliable in dashcam (cabin noise); XD-Violence-style audio fusion has no traffic substrate |

Net: **RGB + induced flow + language** — "multi-modal but not many modalities", every member motivated by temporal/kinematic context or drift robustness, and inference stays RGB+text-only.

---

## 6. LLM integration — off the critical path (justified)

**Pattern:** Holmes-VAU's **Anomaly-focused Temporal Sampler (ATS)**. The detector's own frame-score curve `y^bin` is treated as a density; inverse-CDF sampling concentrates a fixed frame budget on anomaly-dense windows; only those frames go to an MLLM (LoRA-tuned, Holmes-VAU config r=64/α=128, or a frozen VLM with VERA-style traffic guiding questions) which emits the **incident report**: what happened, which agents, apparent severity.

**Why this pattern and not per-frame LLM scoring:** LAVAD and PANDA put the LLM on the scoring path and pay for it (~0.8 FPS for PANDA) — unusable for a live traffic feed. In ATS the LLM adds **zero detection latency**: the score curve is already the detection output; the report is generated asynchronously for the handful of windows that matter. For long traffic streams that are >99% normal, concentrating MLLM budget on the crash window is exactly right (Gap E).

**Concrete improvement over VERA baked in:** VERA's documented weakness is a hard-coded anomaly-in-the-middle Gaussian temporal prior — wrong for streaming/dashcam traffic where a crash can occur at any offset. The ATS layer replaces it with **content-adaptive weighting centered on the detector's own peak-evidence location** — sampling density follows `y^bin`, not a fixed midpoint.

**Optional drift loop:** because the detector is definition-conditioned, the MLLM's report can propose a refined definition `Z'` ("recurring near-misses at this merge lane") that the operator approves — LLM-assisted concept-drift adaptation without retraining (LaGoVAD's prompt-sensitivity result: better definitions lift UCF AUC 80.44 → 83.03).

**Detachability:** if the deployment demands zero LLM footprint, the layer unplugs with no effect on `y^bin` — it consumes the score curve, never feeds it (except through the optional, human-approved `Z'`).

---

## 7. Datasets & preprocessing

### 7.1 Datasets identified from the attached MSAD paper (Zhu et al., NeurIPS 2024) — applicability to road-traffic VAD

Reading the review + dataset paper you attached surfaces the following traffic-usable resources beyond the skill's core set:

1. **MSAD itself — the headline find, directly usable.** Traffic Accident is the **second-largest anomaly category (16.3%)** with **9 detailed subtypes** (car crash, car crash with people / object / train, motorcycle crash, car falling, speeding, car rushing into building, other), plus explicitly traffic-relevant scenarios among its 14: **highway, road, street highview, parking lot, sidewalk, pedestrian street**. Critically for us: (i) **Protocol ii is weakly-supervised** (360 normal + 120 abnormal training videos, video-level labels only) — a drop-in WS training/eval set; (ii) all videos are **fixed-camera** (moving-camera footage was filtered out), high-resolution (mostly 1920×1080, 30 FPS) — the perfect **roadside/CCTV complement** to ego-centric DoTA/DADA, matching the "match camera geometry to deployment" rule; (iii) the paper's own per-scenario results expose a **hard traffic slice**: the *highway* scenario collapses to AP 1.4–4.1 across RTFM/MGFN/UR-DMU — attributed to limited data and complex multi-directional motion — making MSAD-highway a stress benchmark KAT-VAD's motion pathway should visibly improve; (iv) I3D/SwinT features and a blurred (privacy-preserving) version are released; (v) LaGoVAD already evaluates on MSAD (90.41 AUC) and Pi-VAD reports MSAD numbers → direct comparability.
2. **Street Scene** (Ramachandra & Jones, WACV 2020; MSAD Table 1 lists its domain as literally "Traffic"): 81 videos, 1280×720, single fixed overhead street view, 17 human-related anomaly types around a two-lane street (jaywalking, illegal U-turns, loitering). Comes with **pixel-level annotations and the RBDC/TBDC metrics** (which the MSAD appendix discusses) → the right add-on if **spatial localization** of the incident becomes a deliverable.
3. **UCF-Crime RoadAccidents slice** — reconfirmed as the comparability bridge every baseline reports on; the MSAD paper also documents its weaknesses (monochrome, low-res, moving cameras, redundancy) — reinforcing the guardrail that it must never be the *only* traffic evidence.
4. **DoTA** (Yao et al., TPAMI 2022 — cited in the MSAD references): ego-centric dashcam traffic anomalies — already our primary eval anchor; the MSAD citation confirms its standing.
5. **A3D** (Yao et al., IROS 2019, "Unsupervised traffic accident detection in first-person videos" — MSAD ref [123]): ~1,500 ego-centric dashcam accident clips with temporal anomaly windows. Usable as an **extra zero-shot eval set** (video-level labels derivable), same team/lineage as DoTA.
6. **DAD** (Chan et al., ACCV 2016, "Anticipating accidents in dashcam videos" — MSAD ref [10]): 620 dashcam accident + 1,130 normal clips. Anticipation-oriented, but its positive/negative clip structure supports **WS-style video-level training augmentation** and an anticipation-flavored eval.
7. **CCD / Car-Crash Dataset** (Bao et al., ACM-MM 2020, "Uncertainty-based traffic accident anticipation" — MSAD ref [5]): ~1,500 crash dashcam videos with environmental annotations (weather, ego-involvement). Same role as DAD: auxiliary WS training data / robustness eval across weather.
8. **CUVA** (Du et al., 2024 — discussed at length in the review): 1,000 videos, 42 anomaly types *including traffic accidents*, with **free-text causation descriptions** — not ideal for detector training (news footage, moving cameras, scene cuts, subtitles, per the MSAD paper's own critique), but a strong **evaluation corpus for the ATS reasoning layer** (does the generated incident report match CUVA's cause/effect annotations?).
9. **NWPU Campus** (Cao et al., CVPR 2023): 43 views, **scene-dependent anomalies with vehicle-related classes** and the *anomaly anticipation* task — marginal for pure traffic but relevant if scene-dependent definitions (`Z` per camera) are showcased.
10. **PIE** (Rasouli et al., ICCV 2019 — MSAD ref [69]): pedestrian intention/trajectory — not an anomaly dataset per se; noted as a source of **unlabeled normal urban driving footage** for the self-supervised warm-up (SimpleTAD's DAPT needs no anomalies).

### 7.2 Final dataset plan (justified)

| Role | Dataset | Why |
|---|---|---|
| WS training (primary) | **PreVAD** (full for pre-training; Vehicle-Accident slice emphasized) | Largest description-annotated WS corpus; traffic-camera normal feeds; LaGoVAD's native training set → clean reproduction baseline |
| WS training (fixed-camera traffic) | **MSAD Protocol ii** (traffic scenarios) | Video-level labels; fixed-camera roadside geometry; hard highway slice; found via the attached paper |
| WS training (bridge) | **UCF-Crime** | Apples-to-apples with every baseline; never the sole traffic evidence (Gap A guardrail) |
| Eval — ego-centric | **DoTA** (+ **DADA-2000** cross-dataset, + **A3D** zero-shot) | SimpleTAD/LaGoVAD convention; DoTA is where LaGoVAD is weakest (62.60) → the KIP headline number |
| Eval — fixed-camera | **TAD**, **MSAD traffic slice**, (optional **Street Scene** for spatial metrics) | TAD is LaGoVAD's other traffic eval (89.56); MSAD-highway is the stress slice |
| Reasoning eval | **CUVA** traffic subset | Text causation annotations match the ATS report deliverable |
| Self-supervised warm-up (no labels) | **BDD100K** normal driving + PreVAD traffic-cam normal feeds (+ PIE) | SimpleTAD: DAPT on *normal-only* driving video gives large gains, anomalies add nothing |

### 7.3 Preprocessing

- **Sampling:** every 8 frames (LaGoVAD); DoTA at provided 10 fps. Variable-length `L`; no 5/10-crop or score-smoothing tricks (LaGoVAD reports none — keeps comparisons honest).
- **Features:** frozen stock CLIP ViT-B/16 (`openai/clip-vit-base-patch16`, §3.2a), 512-d per sampled frame. RAFT flow embeddings precomputed **once, offline, train-set only**.
- **Duration bias:** keep LaGoVAD's **dynamic video synthesis** (`L_dvs`, θ=0.7, δ_m=5, same-scene KNN filler) — web traffic clips are duration-biased (anomaly fills >70% of the clip in 38–42% of MSAD/PreVAD/LAD videos), and synthesis is what teaches the normal↔abnormal *boundary*; its ablation shows removing it costs ~4 points (69.98 → 65.73). **Traffic adaptation (ego-centric):** for moving-camera sources (DoTA/DADA), extend the same-scene KNN filler key with a coarse ego-motion descriptor (mean RAFT flow magnitude/direction — already computed for KIP) so spliced neighbors share ego-dynamics and no synthetic motion seam is created; restrict aggressive multi-segment splicing (`m>1`) to fixed-camera sources (MSAD, PreVAD traffic-cam, UCF-Crime), and lean on the `m=1` (no-splice) path more for DoTA/DADA by conditioning `θ` on camera geometry. See §8 Stage-2 note.
- **Class imbalance:** balanced minibatches (32 normal + 32 abnormal, RefineVAD convention); report MCC-family metrics (§10) because traffic streams are overwhelmingly normal and AUC alone flatters.
- **Resolution caution:** keep native resolution in the feature extractor; do **not** route low-res roadside CCTV through MLLM pixel paths (LAVIDA documents degradation on low-res) — the MLLM only ever sees ATS-sampled frames for reasoning, never scores pixels.
- **Definitions `Z`:** prepare per-domain traffic definition sets (freeway / urban intersection / school zone / parking) mixing class names and rich descriptions — LaGoVAD trains with both randomly and richer descriptions test better (80.44 → 83.03).

---

## 8. Training pipeline

**Stage 0 (optional, self-supervised, no anomalies) — Kinematic warm-up on normal driving video.** Following SimpleTAD's DAPT finding (domain-adaptive masked pre-training on *unlabeled normal* driving video gives large gains; anomalies add nothing), pre-train the temporal encoder + KIP on BDD100K/PreVAD-normal: mask 75% of the frame-feature sequence, reconstruct masked `v^t` tokens *and* their RAFT embeddings (a motion-aware objective, echoing SimpleTAD's result that motion-aware MVM beats pixel/semantic targets). Discard the reconstruction decoder, keep the encoder + PMG head. *Why optional:* it is a free-win upgrade, not the contribution; the ablation should report with/without.

**Stage 0.5 (optional, not implemented; would replace the Alert-CLIP dependency) — Traffic Hard-Negative Tuning.** Because Alert-CLIP's weights are not downloadable, reproduce its *effect* here instead of importing its checkpoint. Assemble hard-negative pairs — same-scene (LaGoVAD `L_neg` style: an accident clip's own pre-crash normal portion) plus curated semantic pairs (Alert-CLIP style: "car brakes hard and stops safely" / "car reverses carefully" vs "car collides") from the traffic hard-negative text bank (§4.4). Fine-tune only a thin projection on top of frozen CLIP (or a small number of adapter params) with the contrastive `L_neg` objective so the normal/abnormal traffic manifold separates *before* WS-VAD training. *Why:* this is the weights-free, from-scratch substitute for the backbone swap — it needs no external checkpoint, is fully reproducible, and doubles as a contribution. Skip this stage if you use the stock-CLIP default (option 1); enable it for options 2–3 of the backbone ladder.

**Stage 1 — KIP warm-up (Pi-VAD Eq. 5 pattern).** Freeze CLIP encoders; train KIP only:
```
L_stage1 = L_KIP-rec + L_KIP-align
```
*Why:* Pi-VAD shows unguided pseudo-modalities actively hurt (84.66 < baseline 86.97); the flow head must be faithful and RGB-aligned *before* the task loss can exploit it, and warm-up prevents the motion stream from being dominated or ignored at task-training onset.

**Stage 2 — Task training (full objective).** Train temporal encoder, KIP, fusion, heads (CLIP frozen throughout):
```
L_total = L_MIL + L_MIL-align + L_dvs + L_neg              ← LaGoVAD Eq. 6, unchanged
        + λ_rec·L_KIP-rec + λ_al·L_KIP-align + γ·L_kin     ← the novel pathway
        (+ optional: extended-text L_neg hard negatives)    ← secondary component, ablation
```
- `L_MIL` — top-k MIL ranking on `y^bin`; `L_MIL-align` — MIL-align on `y^mul` (LaGoVAD).
- `L_KIP-rec` kept **unweighted-ish (λ_rec = 1)** during Stage 2, mirroring Pi-VAD's deliberate choice to leave `L_PMG` unweighted so pseudo-modalities stay bounded to the real flow manifold for the whole run.
- Suggested starting weights: `λ_al = 0.1, γ = 0.2, β(inside L_kin) = 0.5`, tuned on the PreVAD val split (the only split with frame labels).
- Optimizer/schedule inherited from LaGoVAD: AdamW, LR 5e-5, batch 64, ~40 epochs, single high-end GPU.
- **DVS interaction with KIP (traffic):** the dynamic-video-synthesis pseudo-label `y^p` marks exactly which frames are anomalous in synthesized clips → use it to *partially supervise* `L_kin` (verify the induced-flow curve `ŷ_O` peaks on the true anchor block), upgrading the kinematic-consistency term from self-consistency to partially-supervised on synthesized samples. Apply the motion-aware KNN + camera-geometry-conditioned `θ` from §7.3 so ego-centric splices don't create spurious motion seams the flow head would wrongly flag.

**Stage 3 (optional) — Reasoning head.** Freeze the entire detector. Train the ATS sampler head and a LoRA adapter (r=64, α=128, Holmes-VAU config) on hierarchical instruction data (HIVAU-70k style + CUVA traffic causation text). The frozen `y^bin` is simultaneously the detection output and the sampler signal — the detector is never perturbed by the LLM.

---

## 9. Inference pipeline (step-by-step, real-time argument)

1. **Input:** video stream `V`; operator's definition `Z` for this deployment (set once, editable anytime — no retraining).
2. **Encode:** frozen stock CLIP ViT-B/16 (§3.2a) per sampled frame → `F ∈ ℝ^{L×512}`; temporal encoder → `v^t`.
3. **KIP forward (RGB-only):** PMG head regenerates `ê_O` **from `v^t` alone — RAFT is not loaded**; kinematic gate computes `r_t, s_t`; adaptive shift → `v^k`. Added cost: two 1D-convs + a linear + an MLP over 512-d sequences — negligible next to the CLIP encoder (Pi-VAD precedent: the whole 5-modality PI adds ~18 GFLOPs and still runs 30 FPS; a single-modality KIP is far lighter).
4. **Fuse:** frozen CLIP text on `Z` → `z^t` (computed **once per definition**, cached); co-attention → `v^u`.
5. **Score:** `H_bin` → **frame-level anomaly curve `y^bin ∈ ℝ^{L}`** — the detection deliverable; `H_mul` → per-frame category probabilities for triage.
6. **Localize:** threshold + light Gaussian smoothing (VERA/AnyAnomaly convention) → incident windows.
7. **(Optional, asynchronous)** ATS inverse-CDF-samples frames from high-score windows → MLLM → **incident report**; optional operator-approved definition refinement `Z'`.

**Real-time claim:** the critical path is CLIP ViT-B/16 + ~6 tiny layers, all RGB; flow and LLM are off-path. Report FPS/GFLOPs against Pi-VAD (30.51 FPS / 19.88 GFLOPs) and SimpleTAD (up to 95 FPS) as reference points, with the reasoning latency reported **separately**.

---

## 10. Evaluation

| Metric | Datasets | Why (traffic-specific) |
|---|---|---|
| Frame-level **AUC** | DoTA, DADA-2000, TAD, MSAD, UCF-Crime | Field convention; the DoTA delta over LaGoVAD's 62.60 is the headline KIP number |
| **AP** | XD-style / PreVAD-val, MSAD | Precision-recall under anomaly rarity |
| **AUC_A / AnoAUC** | all | AUC on the abnormal subset — measures catching *the crash itself*, not normal-vs-anything (Pi-VAD's largest gains were AUC_A: +6.96 over UR-DMU) |
| **MCC, AUCMCC, MCC@0.5** | DoTA, DADA-2000 | SimpleTAD's argument: traffic is extremely imbalanced; MCC accounts for true negatives where AUC flatters |
| **mAP@IoU 0.1–0.5 + AVG** | MSAD, UCF-Crime | Temporal localization precision of the incident window (RefineVAD documents high-IoU drops — show KIP's sharper kinematic boundaries help) |
| **Cross-dataset / drift** | DoTA↔DADA (SimpleTAD protocol); drift@5 (LaGoVAD Protocol 2) on MSAD/TAD | The real deployment test: new city, new weather, new definition |
| **Per-scenario slices** | MSAD highway / road / street-highview | The attached paper shows highway AP collapses to 1.4–4.1 for current methods — a named stress slice |
| **Efficiency** | — | GFLOPs / params / FPS; detector vs (off-path) reasoning latency reported separately |
| **Reasoning quality** (if Stage 3) | CUVA traffic subset, HIVAU-style | BLEU/CIDEr/METEOR/ROUGE vs causation annotations |

**Key ablations:** baseline LaGoVAD reproduction → +KIP (each sub-module: PMG head only / +gate-shift / +L_kin) → +text hard negatives → +Stage 0 warm-up; plus a MoTAR-variance-vs-flow-norm gate head-to-head (isolates the "true kinematics beat the variance proxy" claim).

---

## 11. Justification ledger

| Decision | Rationale | Source paper(s) |
|---|---|---|
| Baseline = LaGoVAD | Only venue-eligible WS baseline evaluated on real traffic data (DoTA 62.60 / TAD 89.56); ships PreVAD; concept-drift framing is the defining traffic problem | LaGoVAD |
| Rejected RefineVAD as baseline | No traffic data; motion = variance proxy; high-IoU drops — kept as fallback & MoTAR donor | RefineVAD, gap analysis |
| Rejected DSANet as baseline | Motion-blind; no traffic data; SG-NM imported as optional branch instead | DSANet |
| Rejected Pi-VAD as baseline | arXiv-only (venue rule); imported as the novel-component source instead | Pi-VAD |
| Rejected SimpleTAD as baseline | Fully-supervised (frame labels) — violates the WS requirement; DAPT + metrics + datasets reused | SimpleTAD |
| Novel component = KIP (motion pathway) | Plugs the chosen baseline's *specific* gap (no motion; weakest on DoTA); motion is Pi-VAD's best single modality (87.92) and SimpleTAD's winning pre-training signal | Pi-VAD Table 6, SimpleTAD, LaGoVAD limitations |
| Train-time-only flow via PMG | Keeps RGB-only real-time inference; regeneration costs ≈0.25% AUC vs real features at ~1/128 the GFLOPs | Pi-VAD Tables 3, 5 |
| Kinematic gate replaces MoTAR variance | Variance proxy misses slow-onset drift; flow-norm gate registers sustained moderate motion | RefineVAD MoTAR + its documented gap |
| Kinematic-consistency loss | WS-MIL latches onto appearance shortcuts (glare, headlights — MSAD lists them); demand kinematic corroboration of every anomaly window | Pi-VAD (motion = discriminator), MSAD paper §A |
| Warm-up before task training | Unguided pseudo-modalities *hurt* (84.66 < 86.97); two-step optimization is Pi-VAD's proven recipe | Pi-VAD Table 3, Eq. 5–6 |
| Keep L_dvs + L_neg | Removing either degrades detection+classification; both removed → large drop (65.73); L_neg drives the low false-alarm rate | LaGoVAD Tables 5/G/E |
| Backbone = frozen stock CLIP ViT-B/16; Alert-CLIP not used | Alert-CLIP (CVPR 2026) weights are **not publicly downloadable**; only its dataset repo is public. Depending on them would block the project. Use stock CLIP, which is LaGoVAD's own choice and gives an honest baseline. Stage-0.5 tuning and an Alert-CLIP swap stay future work, not part of the architecture | Alert-CLIP (effect), LaGoVAD (default), web check (weights unavailable) |
| Stage-0.5 Traffic Hard-Negative Tuning | Weights-free substitute for the Alert-CLIP swap: separates "braking normally" from "colliding" using same-scene + semantic hard negatives; reproducible, doubles as a contribution | Alert-CLIP (semantic HN), LaGoVAD (`L_neg` same-scene HN) |
| Keep DVS (training-only), + motion-aware KNN for ego-centric | Duration-bias is worst in traffic (brief, rare incidents vs anomaly-heavy training clips); DVS teaches the boundary and gives free frame labels that also partially-supervise `L_kin`; ablation worth ~4 pts; motion-aware KNN prevents synthetic seams on moving-camera data | LaGoVAD (DVS, ablation), MSAD paper (duration histogram) |
| Modality set = RGB + flow + language | Motion #1 (kinematic anomalies), language buys drift robustness; depth deferred; pose/panoptic/audio excluded as human-/scene-centric or absent in traffic feeds | Pi-VAD Table 6, LaGoVAD, baseline_decision §2 |
| LLM via ATS, off critical path | Per-frame LLM ≈0.8 FPS (PANDA) is unusable; score-guided sampling concentrates MLLM budget on the crash window at zero detection latency | Holmes-VAU, PANDA, LAVAD |
| Replace midpoint prior with score-adaptive sampling | VERA's Gaussian midpoint prior is wrong for streaming traffic; sampling follows the detector's evidence | VERA (documented weakness), Holmes-VAU |
| MSAD added to the data plan | 16.3% Traffic Accident + 9 subtypes; WS Protocol ii; fixed-camera complement to ego-centric DoTA; highway slice AP 1.4–4.1 = stress benchmark | attached MSAD paper (Figs. 3–4, Tables 5/8) |
| DoTA/DADA required for eval | Gap A guardrail: UCF-Crime RoadAccidents alone is not traffic validity | gap-analysis synthesis, SimpleTAD, LaGoVAD |
| MCC/AUCMCC reported | Extreme normal-class imbalance in traffic; AUC alone flatters | SimpleTAD |
| Stage 0 DAPT-style warm-up (optional) | Normal-only masked pre-training gives large gains, needs no anomalies; motion-aware targets win | SimpleTAD |

---

## 12. Open iteration hooks

- **Swap the baseline to RefineVAD** (motion-native, AAAI 2026) if peer-reviewed motion pedigree outranks traffic-native evaluation for your committee — the novel component then flips to the Self-Guided Traffic Normality branch (DSANet SG-NM) or the hard-negative loss, since motion is already covered.
- **Add depth as a second PMG head** (DepthAnythingV2 targets) — Pi-VAD shows depth dominates AUC_A; mechanical extension of KIP.
- **Enable the SG-NM auxiliary branch** (DSANet) to exploit traffic-flow regularity (Gap F) — one more branch + `L_compact + L_consist`.
- **Pixel-level localization** — bolt on LAVIDA's dual-granularity decoder and evaluate with Street Scene's RBDC/TBDC.
- **Strengthen the text hard-negative bank** into a first-class contribution (Alert-CLIP-style curated traffic near-miss captions).
- **Anticipation extension** — DAD/CCD/A3D support time-to-accident metrics if the thesis pivots from detection to early warning (NWPU-style anticipation).
- **VideoRAG memory** — cross-camera retrieval of similar past incidents as a city-network feature (out of v1 scope).

*Tell me which knob to turn — e.g. "swap to RefineVAD", "add the depth head", "make the hard-negative bank the primary contribution" — and I'll re-run only the affected sections and update the ledger.*
