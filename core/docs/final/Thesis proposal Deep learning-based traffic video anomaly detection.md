# Thesis proposal: Deep learning-based traffic video anomaly detection

Oct 26, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-MQR**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. It starts from LaGoVAD (ICLR 2026) and adds a frozen VideoMAE V2 motion stream as the permitted backbone change. The one novel component is a **Motion Query Readout (MQR)**: learnable queries, following Efficient Probing (ICLR 2026), read the dense VideoMAE tokens and are trained with a spatial top-k MIL loss. The claim under test: reading motion where it happens lifts detection of small, distant and ego-uninvolved traffic agents, and yields evidence maps scored with DoTA's spatio-temporal AUC (STAUC), which our search (to October 2026) found no weakly supervised method reporting.

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event is happening in it. Abnormal events include collisions, loss of control, illegal turns, wrong-way driving, pedestrians on the road and road spills. These are the categories used by TAD and DoTA.

Two camera settings define the problem space:

- **Fixed camera (non-ego).** TAD and the traffic scenes of MSAD. The camera is static; the anomalous agents are often small and far away.
- **Dashcam (ego-centric).** DoTA, DADA-2000, CCD and Nexar. The camera moves with the car. DoTA also marks each clip as ego-involved (the ego car takes part in the anomaly) or ego-uninvolved (it only observes); this proposal uses those two terms for that flag.

The thesis follows LaGoVAD's weakly supervised, language-guided formulation. Training uses video-level annotations only: one label y\_i ∈ {0, 1} per video, plus a category and a hand-written caption template. At test time the model receives a video and a text definition Z of what counts as anomalous, and returns a score per frame. This thesis adds a spatial evidence map per frame as a second output.

```latex
\Phi_\theta : (V, Z) \;\mapsto\; \{ s_t \}_{t=1}^{N},\; \{ E_t \}_{t=1}^{N}, \qquad s_t \in [0,1],\;\; E_t \in [0,1]^{H \times W}
```

Here s\_t is the anomaly score of frame t and E\_t is a coarse map of where the evidence lies, normalised per video. The video-level annotations supervise training through multiple-instance learning (MIL): the mean of the top-k frame scores of a video must match its video label.

Four properties make traffic video harder than generic surveillance:

1. **Anomalies are motion events.** A collision is defined by how trajectories meet, not by what one frame looks like. LaGoVAD encodes single RGB frames with CLIP and has no motion input; it reaches 89.56 AUC on TAD but 62.60 on dashcam DoTA. Part of this gap is protocol: DoTA is abnormal-only and min-max normalised per video, so its AUC measures timing inside a clip. The I3D-based PEL scores 44.97–53.05 on DoTA, so a generic, surveillance-trained motion encoder does not fix this either; the thesis uses the masked-video encoder that transfers best to traffic anomalies and reads it spatially.
2. **The agents that matter are small.** A distant agent of about 40×30 px in a 1280×720 frame shrinks to about 7×9 px at 224×224 input, smaller than one 16-px patch (estimate). One global vector per frame dilutes it.
3. **The camera moves.** In dashcam video, ego-motion dominates frame-to-frame change. In preliminary experiments, hand-crafted motion cues (feature differences, ego-motion-compensated optical flow) did not help.
4. **Labels are scarce.** Frame-level labels are expensive and anomalies are rare, so the model must learn from video-level labels.

The scope is offline detection: the whole video is available and the question is whether an anomaly is happening, not whether one will happen (anticipation, as in Nexar). No large language model or multimodal LLM runs at training or test time; a frozen CLIP text encoder is allowed. All training text is hand-written: class names, category definitions and caption templates. The released LaGoVAD checkpoint, used only as a reference row, was trained on PreVAD, whose captions its authors generated offline with Qwen models.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from cheap video-level labels, sees motion, and shows where in the frame the anomaly is. It must keep LaGoVAD's ability to take the anomaly definition as text, and run on one academic GPU without any LLM.

The thesis builds this in two steps, matching the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** A frozen VideoMAE V2-S encoder (CVPR 2023), the strongest small masked-video encoder for traffic anomalies in Simple-TAD's comparison, feeds motion features into LaGoVAD through a zero-initialised gate. Preliminary experiments show this step helps; their setup and gains will be documented in an appendix.
- **Novel component (MQR).** The motion stream's dense tokens are currently averaged into one vector per clip. MQR replaces that average with a learned multi-query readout, trained with a spatial MIL loss, so the evidence of small and distant agents is kept.

A second purpose is a fair evaluation: published traffic numbers are mostly single-metric AUC on inconsistent splits. The thesis will report at least two metrics against at least two published methods on the same fixed-camera traffic data, plus zero-shot and dashcam results.

### 1.3 Research aims

The thesis has five aims:

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint and retrain LaGoVAD on the thesis training data, with AUC, AP and AUC on abnormal videos (AUC\_A) on TAD, MSAD, DoTA and DADA-2000.
2. **Add the motion stream.** Integrate the frozen VideoMAE V2-S stream and measure its gain over CLIP-only LaGoVAD.
3. **Design and train MQR.** Replace the motion stream's mean pooling with the multi-query readout and spatial MIL loss, using video-level annotations only.
4. **Measure spatial localisation under weak supervision.** Score MQR's maps against DoTA's anomalous-agent boxes with STAUC and a pointing game.
5. **Explain the gains.** Break results down by ego-involved vs ego-uninvolved clips, agent box size and anomaly category, and check that MSAD's non-traffic scenes do not regress.

Aims 1–2 are prerequisites; aims 3–5 map to the hypotheses below. "Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval over test videos excludes zero.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. MQR beats the mean-pooled motion stream, mostly on small and ego-uninvolved agents | DoTA AUC by ego-involved / ego-uninvolved and by clip-level box-size bin; TAD AUC for Accident, PedestrianOnRoad and the other categories pooled | Gains on ego-uninvolved clips and small-box bins exceed those on ego-involved clips and large-box bins |
| H2. MQR's maps point at the anomalous agents | STAUC and pointing game on DoTA's official 1,305-video list | MQR beats a uniform map (the mean-pooled motion stream) and a centre prior |
| H3. The gain holds across camera settings without hurting surveillance | TAD AUC\_O / AUC\_A; MSAD AUC / AP, its traffic slice and its non-traffic scenes; CCD, DoTA and DADA-2000 AUC | Gains on TAD, the MSAD traffic slice and the dashcam sets; no significant drop on MSAD's non-traffic scenes |
| H4. Learned selection matters, not just more tokens | Ablation ladder: max / GeM pooling, single-query, multi-query | Multi-query with the spatial MIL loss beats every simpler readout |

## 2. Literature review

### 2.1 Overview and classification of video anomaly detection method branches

Video anomaly detection (VAD) methods divide by the supervision they train with into five branches, plus the adjacent task of accident anticipation. Traffic methods cut across these branches along a second axis: fixed camera (non-ego) or dashcam (ego-centric). The table places the papers reviewed for this thesis, mostly 2024–2026.

| Branch | Training signal | Typical mechanism | Representative methods (venue) | Traffic evidence | Role in this thesis |
| --- | --- | --- | --- | --- | --- |
| Unsupervised / one-class | Normal videos only | Reconstruction, frame prediction, diffusion, object-centric or scene-graph normality | MA-PDM (AAAI 2025), ADSM (CVPR 2025), TAO (CVPR 2025), Structured Graph Reasoning (IEEE TMM 2026) | DoTA and MM-AU, AUC and STAUC (graph method); DoTA's own baselines reach 73.0 AUC / 48.5 STAUC | Context; ego-motion breaks most predictors |
| Weakly supervised | Video-level labels, MIL | Snippet features (I3D, CLIP) + top-k MIL; CLIP text prompts; prototypes; multi-modal induction | Sultani (CVPR 2018), RTFM (ICCV 2021), WSAL (IEEE TIP 2021), UMIL (CVPR 2023), MultiDomain (NeurIPS 2024, multi-domain I3D MIL), VadCLIP (AAAI 2024), π-VAD (CVPR 2025), GS-MoE (ICCV 2025), DSANet (AAAI 2026), RefineVAD (AAAI 2026), LAS-VAD, Fine-VAD, Road Less Seen, Alert-CLIP (CVPR 2026) | TAD (WSAL, UMIL; zero-shot MultiDomain), MSAD (π-VAD, GS-MoE, Alert-CLIP), dashcam WS-DoTA (ECCV 2024 Workshop) | **The thesis's branch** |
| Open-vocabulary / open-world | Video labels + category text | CLIP text space defines anomaly classes or free-text definitions at test time | Anomize (CVPR 2025), **LaGoVAD (ICLR 2026)** | Zero-shot TAD, DoTA, MSAD, LAD (LaGoVAD) | **Baseline** |
| Fully supervised traffic detection | Frame-level labels (+ boxes) | Video encoder + temporal head; object prompts; agent graphs | MOVAD (ICASSP 2024), TTHF (IEEE TCSVT 2024), PromptTAD (IEEE RA-L 2025), ISCRTAD (IEEE T-ITS 2025), Simple-TAD (ICCV 2025 Workshops) | DoTA and DADA-2000, AUC up to 88.4 | Context rows; evidence for the choice of motion encoder |
| LLM / MLLM-based | None, or few-shot rules, or light tuning | Captioning + LLM reasoning, prompting, probing of MLLM heads | LAVAD (CVPR 2024), AnomalyRuler (ECCV 2024), VERA, Holmes-VAU (CVPR 2025), EventVAD (ACM MM 2025), HeadHunt-VAD (AAAI 2026), LAVIDA (CVPR 2026), AnyAnomaly (WACV 2026), SteerVAD | Almost none on traffic sets | Excluded: LLM cost |
| Accident anticipation (adjacent task) | Positive / negative dashcam clips | Predict a collision before it happens | Nexar dataset (CVPR 2025 Workshops), BADAS, FLaRA | Nexar, CCD, DAD; mean AP at fixed times before the event | Different task; source of evidence for attentive probes |

Three findings from this review shape the proposal. First, CLIP-feature weak supervision dominates surveillance benchmarks but sits near chance on dashcam DoTA. Second, the strongest traffic results come from video foundation models trained with masked video modelling, and Simple-TAD found that this pre-training beats vision-language pre-training for traffic anomalies. Third, the methods that win on rare or distant traffic agents (PromptTAD, ISCRTAD) all add object-level evidence, but through detectors, trackers and frame-level or box supervision.

### 2.2 Emerging trends and unresolved challenges

Six trends run through the 2024–2026 literature:

1. **Vision-language features as the default.** Since VadCLIP (AAAI 2024), most weakly supervised methods start from frozen CLIP frame features and text prompts: DSANet and RefineVAD (AAAI 2026), LAS-VAD, Fine-VAD and Alert-CLIP (CVPR 2026).
2. **Anomalies defined by language.** Open-vocabulary (Anomize, CVPR 2025) and open-world (LaGoVAD, ICLR 2026) models accept the anomaly definition at test time. LaGoVAD's drift@5 protocol changes which categories count as abnormal and measures robustness to that change.
3. **MLLMs for explanation and training-free detection.** LAVAD, VERA, Holmes-VAU, HeadHunt-VAD and LAVIDA rely on LLMs or multimodal LLMs of 2B parameters and more; AnomalyRuler also calls GPT-4. They are not clearly better at detection: on MSAD drift@5, LaGoVAD reaches 85.6 AUC / 40.1 AP against 84.3 / 34.3 for Holmes-VAU and 63.1 / 22.4 for Qwen2.5-VL-7B.
4. **Masked-video foundation models for traffic.** Simple-TAD showed that masked video modelling transfers best to traffic anomalies. After supervised fine-tuning, VideoMAE-B (1600 epochs) reaches AUC-MCC 56.0 on DoTAref (refined DoTA labels) and 52.2 on DADA-2000, against 50.0 and 43.1 for InternVideo2-B. Further masked pre-training on normal driving video (DAPT) adds more without needing any anomalies.
5. **Spatial and object-level grounding.** Alert-CLIP aligns region features with box and caption supervision, PromptTAD prompts with detected boxes, TAO tracks objects, and STPrompt (ACM MM 2024) re-encodes motion-selected image crops with CLIP's global token and localises with an LLM. Two 2026 preprints, "Localizing to Debias" (UCF-Crime, XD-Violence, MSAD) and "Learning Where and When" (datasets not yet known), bring spatial localisation to weakly supervised VAD.
6. **Extra modalities.** π-VAD (CVPR 2025) induces pose, depth, flow and other modalities from RGB features at training time, and found flow the most useful. GS-MoE (ICCV 2025) routes anomaly types to experts.

Five challenges remain open for traffic video:

1. **Motion blindness.** Frame-level CLIP features miss what makes a collision a collision. In LaGoVAD's table every surveillance-trained model is near chance on DoTA's within-clip protocol, whether it uses CLIP features (VadCLIP, LaGoVAD: 49.00–62.60) or I3D (PEL: 44.97–53.05), while the CLIP-feature models reach 74.46–89.56 on TAD's video-pooled protocol.
2. **Spatial dilution.** Global vectors lose small and distant agents. Even frame-supervised Simple-TAD averages 81.6 / 84.1 / 85.2 AUC on ego-uninvolved DoTA clips (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips. In DSANet's analysis, VadCLIP assigns 84.0% of UCF-Crime road-accident videos to "Normal". GS-MoE reports a road-accident video scored near zero "because the anomaly occupies a very small pixel region".
3. **No spatial evidence under weak supervision.** We found no weakly supervised traffic method that outputs or evaluates spatial maps, and no report of DoTA's spatio-temporal AUC (STAUC) under weak supervision (search to October 2026).
4. **Domain shift and ego-motion.** Training data are mostly surveillance or web video, test data are dashcam video, and camera motion masks agent motion.
5. **Inconsistent evaluation.** TAD has two incompatible test lists in use (76 of LaGoVAD's 100 TAD test videos belong to UMIL's training list). DoTA contains only abnormal clips and is scored after per-video min-max normalisation. Most traffic tables report AUC only.

### 2.3 Research gaps

Four gaps follow from the review. The thesis addresses each one, and its novelty claim is deliberately narrow.

1. **No motion-aware foundation features in weakly supervised traffic VAD.** LaGoVAD and VadCLIP read CLIP frames; the masked-video evidence of Simple-TAD comes from frame-level supervision. The one weakly supervised dashcam study, WS-DoTA (ECCV 2024 Workshop), added a temporal-regularity transform of CLIP features and lifted RTFM from 57.9 to 78.2 AUC. *Addressed by the motion stream (backbone change).*
2. **Frozen masked-video features are read by mean pooling, which loses much of their discriminative signal.** Frozen VideoMAE ViT-L scores 52.5 with a linear probe and 77.8 with an attentive probe on Kinetics-400, and 41.3 vs 61.2 on Something-Something v2 (V-JEPA; the pooling behind its linear column is not stated). Masked image models show the same pattern: 67.7 vs 75.6 for MAE ViT-B with Efficient Probing (ICLR 2026), 65.8 vs 71.6 with selective aggregation (Beyond \[cls\], ICCV 2025). These rows come from different experiments. No VAD work reads such tokens with learned queries under weak supervision. *Addressed by MQR.*
3. **Weak supervision gives no spatial evidence on traffic data.** We found no weakly supervised traffic method that produces maps, and no report of STAUC under weak supervision. *Addressed by MQR's per-query maps, scored with STAUC and a pointing game.*
4. **Traffic results are hard to compare.** Most rows are AUC-only and two TAD test lists circulate. *Addressed by four split-consistent, multi-metric protocols (Section 4.4).*

The closest prior work bounds the novelty claim:

| Closest work | What it does | How MQR differs |
| --- | --- | --- |
| BADAS (arXiv 2510.14876) | Attentive probe with 12 learned queries on a fine-tuned V-JEPA2, fully supervised ego-collision anticipation | Weak labels, detection, per-query spatial MIL, frozen encoder, language-guided host model |
| FLaRA (arXiv 2606.14380) | Single-query attentive probe for accident anticipation | Multiple spatial queries, weak labels, maps evaluated |
| STPrompt (ACM MM 2024) | Motion-selected image crops encoded by CLIP's global token for weakly supervised surveillance VAD; LLM-assisted localisation; no code | Motion-foundation tokens, traffic data, no LLM |
| "Localizing to Debias" (arXiv 2608.12045) | Sparse spatio-temporal weakly supervised VAD on UCF-Crime, XD-Violence and MSAD (details unverified) | Traffic focus, dense motion tokens, STAUC on DoTA; **main novelty risk, to be read in full and compared** |
| "Learning Where and When" (arXiv 2606.29498) | Patch-based spatio-temporal localisation under weak supervision (datasets not yet known) | Traffic focus, motion tokens, language-guided host |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with boxes and captions | Needs neither boxes nor captions for the readout |
| PromptTAD, ISCRTAD | Object prompts or agent graphs from detectors, frame-level supervision | Detector-free, video-level labels |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that reads dense motion-foundation tokens with a learned per-query spatial MIL readout, and the first to report STAUC under weak supervision.** Attentive pooling itself is not claimed as new, and both claims will be re-checked against the two 2026 preprints before submission.

## 3. Proposed model

The proposed model, **LaGoVAD-MQR**, is LaGoVAD with two changes: a frozen VideoMAE motion stream (the backbone change) and the Motion Query Readout (the novel component). Every other part of LaGoVAD stays as released, so any gain can be traced to these two changes.

### 3.1 Base architecture

**LaGoVAD, unchanged parts.** The released code fixes each part:

- **Sampling.** One frame in eight is kept (`interval=8`); for 10 fps dashcam sets (DoTA, CCD) every provided frame is used, as LaGoVAD does for DoTA.
- **CLIP stream.** A frozen CLIP ViT-B/16 image encoder gives one 512-d embedding c\_t per sampled frame t, from a 224×224 resize without cropping.
- **Temporal encoder.** A residual RoFormer (rotary position encoding): 2 layers, 4 heads, hidden size 512, local window of 25 steps. Training sequences are resampled or padded to 512 steps.
- **Text branch.** The frozen CLIP ViT-B/16 text encoder with 32 learnable soft-prompt tokens encodes the anomaly definition Z into τ ∈ R^512 (class names and captions during training).
- **Fusion.** A 2-layer co-attention Transformer lets video steps and text tokens attend to each other.
- **Heads.** A 1-D convolution head (kernel 9) scores the pre-fusion (language-agnostic) and post-fusion (language-guided) features. A learned weight w = σ(10·a), with a initialised at 0, blends them; the frame score is the sigmoid of the blend.
- **Losses.** Top-k MIL binary cross-entropy (k = max(⌊L/16⌋, 1) for a video of L steps), a multi-class MIL alignment loss, dynamic video synthesis (an abnormal clip inserted among 1–5 kNN-retrieved normal clips, giving frame pseudo-labels), and a caption contrastive loss with hard negatives (temperature 0.02).

**Backbone change: the motion stream.** For each sampled step t, a 16-frame clip at 10 fps centred on t (1.6 s, the input Simple-TAD used for traffic) passes through a frozen **VideoMAE V2-S** encoder (Wang et al., CVPR 2023): the ViT-S distilled from VideoMAE V2-g on Kinetics-710, with D = 384. In Simple-TAD's comparison of pre-training strategies, VideoMAE V2 was the strongest small encoder for traffic anomalies after fine-tuning: AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000, against 52.1 / 48.1 for VideoMAE-S and 49.7 / 43.7 for InternVideo2-S.

The last layer gives X\_t, 8 temporal × 14 × 14 spatial tokens of dimension D. The preliminary version ("A") averages all 1,568 tokens and adds the result to the CLIP embedding through a gate that starts closed:

```latex
v_t = \frac{1}{1568} \sum_{j=1}^{1568} X_{t,j}, \qquad x_t = c_t + \tanh(\alpha)\, W\, \mathrm{LN}(v_t), \qquad \alpha_{\text{init}} = 0,\; W \in \mathbb{R}^{512 \times D}
```

The fused x\_t replaces c\_t as the RoFormer input. Because α starts at 0, the model starts as CLIP-only LaGoVAD and opens the gate only as far as motion helps. The kNN retrieval of dynamic video synthesis still uses CLIP features.

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips and distilled on Kinetics-710; none of the seven traffic datasets used in this thesis is among these sources. Simple-TAD's driving-adapted DAPT-VideoMAE-S (BDD100K only) is kept as an encoder ablation.

&#91;embedded content: LaGoVAD-MQR architecture · the new component highlighted\]

Only the highlighted readout and its dashed training loss are new. The motion stream is the backbone change, and every plain box is LaGoVAD as released.

### 3.2 Novel component: Motion Query Readout (MQR)

MQR replaces the motion stream's average with M learnable queries, each attending to its own part of the frame's motion tokens. One extra loss, a spatial top-k MIL loss, turns every query into a region-level anomaly detector trained from video labels alone. The readout follows Efficient Probing (Psomas et al., "Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency", ICLR 2026; Apache-2.0 code, `EfficientProbing(dim, num_queries)`), with three changes of ours: the attention logits use layer-normalised tokens, the queries start at zero, and the value projection starts at the identity with zero bias.

**Token grid.** The 8×14×14 tokens of each clip are reduced by a 2×2 spatial average and a mean over the 8 temporal slots to a 7×7 grid P\_t of 49 tokens (D = 384 for the default ViT-S). Because both poolings weight tokens equally, the grid's average equals the average of all 1,568 tokens. Each tile covers 32×32 px of the 224×224 input, so the 7×9 px distant agent of Section 1.1 fills only about 6% of a tile; the Step 0 diagnostic (Section 4.3) decides before main training whether 7×7 keeps enough of its signal or a 14×14 grid is needed.

**Readout.** Queries u\_m score the layer-normalised tokens, and each query collects its own D/M-wide slice of the projected values:

```latex
a_{t,m,j} = \operatorname{softmax}_j\!\left( \frac{u_m^{\top}\, \mathrm{LN}(P_{t,j})}{\sqrt{D}} \right), \qquad z_{t,m} = \Big[ W_V \sum_{j=1}^{49} a_{t,m,j}\, P_{t,j} \Big]_m \in \mathbb{R}^{D/M}, \qquad r_t = \big[ z_{t,1}; \dots ; z_{t,M} \big] \in \mathbb{R}^{D}
```

Pooling before projecting is exactly equal to projecting every token first, because W\_V is linear, and it keeps memory low. The readout r\_t enters the existing gate unchanged: x\_t = c\_t + tanh(α)·W·LN(r\_t). The default is M = 8 queries. The parameters are W\_V (D×D) and the queries (M×D): about 0.15M for ViT-S and 0.6M for ViT-B.

**Same starting function.** With all queries at zero, every attention map is uniform; with W\_V = I and zero bias, r\_t then equals the token average v\_t exactly, for any M. This requires v\_t to be the plain mean of the same last-layer tokens, so the preliminary code is checked for any extra normalisation (such as a final `fc_norm`). At initialisation LaGoVAD-MQR is therefore the same function as the mean-pooled motion-stream model. The spatial MIL loss updates W\_V and the queries from the first step, so the two models diverge during training even where spatial selection does not help; whether MQR ends above the motion-stream model is what the ablation tests.

**Spatial MIL loss (the one extra loss).** Each query also scores its own slice through one linear layer g shared by all queries. A step's spatial score is its highest query score, and a video's score is the mean of its k highest step scores:

```latex
s_{t,m} = \sigma\big( g(z_{t,m}) \big), \qquad S_t = \max_m s_{t,m}, \qquad \hat{s}_V = \frac{1}{k} \sum_{t \in \operatorname{top}k(S)} S_t, \qquad \mathcal{L}_{\text{sMIL}} = \operatorname{BCE}\big( \hat{s}_V,\, y \big)
```

Top-k is taken over valid (unpadded) steps only, with k = max(⌊L/16⌋, 1) from the true length L. On a normal video the loss lowers the highest-scoring query at each of the k highest-scoring steps; repeated over training, this caps every query's score on normal videos and discourages a query from settling on a salient but normal agent. On an abnormal video only the top steps and their best query must be high, so supervision is multiple-instance in time and in space at once. A dense variant that applies BCE to every s\_{t,m} of normal videos is part of the loss-form ablation. The total loss is L\_LaGoVAD + λ·L\_sMIL with λ ∈ {0.1, 0.5, 1}.

**Evidence map.** E\_t(j) = Σ\_m s\_{t,m}·a\_{t,m,j}, which lies in \[0, M\]; it is reshaped to 7×7, upsampled to the frame and normalised per video to \[0, 1\]. The layer g is trained only through the spatial MIL loss; at test time it only weights this map.

**Exploratory arm, outside the main claim.** The queries can be conditioned on the definition embedding LaGoVAD already computes, q\_{t,m} = u\_m + G(τ), with G a zero-initialised linear map from 512 to D, so training starts from static MQR. This is additive conditioning inspired by, but not the same as, Grounding DINO's language-guided query selection (ECCV 2024). Because it borrows from a second paper and makes the pre-fusion head depend on the text, it is reported only as an analysis (ablation row 6) and is never part of LaGoVAD-MQR.

**Why MQR should help on traffic video.** Five pieces of evidence point the same way:

1. **The motion features are the kind mean pooling hurts most.** Masked-modelling attention spreads over many patches: MAE's aggregation entropy is 5.03 against a maximum of 5.27 (Beyond \[cls\], ICCV 2025). Attentive readouts recover much of what averaging or linear probing loses (Section 2.3).
2. **Several queries read several parts.** Efficient Probing's queries attend to distinct regions (complementarity 0.65 against 0.37 for the best MAE heads), and accuracy grows with query count, from 60.6 with 2 queries to 65.1 with 64 (SimMIM ViT-B).
3. **Traffic anomalies are small and local.** The ego-uninvolved gap of Simple-TAD, PromptTAD's "neglect of distant or off-center objects" and GS-MoE's small-region failure all show that global vectors miss them (Section 2.2).
4. **VideoMAE tokens keep local, moving content; CLIP's do not.** Masked image modelling "preserves and amplifies token-level information" (Park et al., ICLR 2023). CLIP's global embedding favours large objects (CLIP Under the Microscope, CVPR 2025), and contrastive embeddings sit in a narrow cone where even unrelated inputs average about 0.5 cosine (Mind the Gap, NeurIPS 2022). The high cosine between CLIP frame embeddings (above 0.9 in pilot measurements) therefore does not rule out local signal. CLIP's last-layer patches share the same peak dimensions (ClearCLIP, ECCV 2024); dense CLIP tokens can carry anomaly evidence once attention is corrected (AnomalyCLIP, ICLR 2024), but they carry no motion. That is why the readout runs on VideoMAE tokens.
5. **Same starting function.** Because MQR starts as the mean-pooled model, any difference after training comes from learned selection and the extra loss, not from a different starting point.

| Property | Mean pooling (preliminary) | MQR |
| --- | --- | --- |
| Extra parameters | — | about 0.15M (ViT-S); 0.6M (ViT-B) |
| Extra extraction compute | — | none: same VideoMAE pass |
| Storage per clip (ViT-S, fp16) | 0.75 KB | about 37.6 KB (7×7 grid) |
| Spatial map | none | one per step |
| Extra loss | — | spatial top-k MIL |

The expected signature is specific: gains concentrated on ego-uninvolved DoTA clips, small agent boxes and the "unknown" categories, and on TAD's small CCTV anomalies. Overall DoTA AUC may move little, because per-video normalisation makes it a within-clip timing measure.

### 3.3 Training strategy

Training uses video-level annotations only and LaGoVAD's own schedule, so the baseline, the motion stream and MQR differ only in architecture and the one extra loss.

**Training data.** All training sets come from the seven traffic datasets used in the thesis. Only TAD, MSAD, CCD and Nexar contain both normal and abnormal videos, so only they are used for training; DoTA, DADA-2000 and MM-AU contain accident videos only and serve as test sets.

| Protocol | Training set | Normal / abnormal videos | Anomaly text (hand-written) |
| --- | --- | --- | --- |
| P1, TAD | TAD train, UMIL split | 210 / 190 | TAD's 7 category names and definitions |
| P2, MSAD | MSAD weak-protocol train | 360 / 120 | MSAD's 11 anomaly types |
| P3, dashcam | Nexar labelled set + CCD official train split | Nexar 750 / 750; CCD's train part of 3,000 normal / 1,500 crash clips | Collision, near-miss and car-crash definitions |
| P4, cross-setting | None: reuses the P2 and P3 models | — | Each test set's own definitions |

**Captions.** LaGoVAD's caption contrastive loss needs one description per video. Every model uses the same hand-written caption template per category (for example, "a dashcam video in which the car collides with another vehicle"), so no LLM-written text enters training. Removing this loss costs LaGoVAD 6.3 AUC on TAD zero-shot (89.56 to 83.26), so a no-caption variant is reported as an ablation.

**Shortcut control.** CCD's crash clips and its normal clips (taken from BDD100K) come from different sources, so a model could learn the source instead of the anomaly. Nexar's positive and negative clips both come from Nexar's dashcam network, so P3 is also reported with Nexar-only training.

**Stages.**

1. **Cache features.** Extract CLIP frame features with LaGoVAD's script and the 7×7 VideoMAE V2-S token grids once, stored as memory-mapped fp16 arrays.
2. **Train all models with one schedule.** Every row (LaGoVAD, motion stream, MQR, ablations) is trained from the same random initialisation of LaGoVAD's trainable parts, with the same schedule and 3 seeds. MQR starts with zero queries and W\_V = I.
3. **Warm start as a speed variant.** MQR can also start from a trained motion-stream checkpoint, since it computes the same function at initialisation. This is reported separately and never mixed into ablation rows.

**Settings.** AdamW, batch 64, learning rate 5e-5 with a cosine schedule, 40 epochs, as in LaGoVAD. The spatial loss weight λ ∈ {0.1, 0.5, 1}, the query count M ∈ {4, 8, 16} and a learning-rate multiplier of 1 or 10 for MQR's parameters are chosen once, on a held-out 15% of the P3 training videos scored by video-level AUC, and then frozen for every protocol. No hyperparameter is chosen on a test list.

**Checkpoint selection.** LaGoVAD's released config selects checkpoints on test scores. The thesis instead reports the last epoch of the fixed schedule for every model, baselines included.

**Compute.** A 512-step training sequence of 7×7 ViT-S grids is about 19 MB, so a batch of 64 is about 1.2 GB of input. Pooling before projecting avoids building W\_V·P for all 49 tokens, and with bf16 autocast the readout's activations should stay within a few GB. One 24 GB GPU should therefore suffice, to be confirmed by a profiling run in the first week. Disk is the main cost (Section 4.2).

### 3.4 Inference

At test time the model takes a video and an anomaly definition and returns a score and a 7×7 evidence map per sampled step. No LLM runs at any point.

1. **Sample** one frame in eight (every frame for 10 fps dashcam sets).
2. **Encode** each sampled frame with frozen CLIP, and the 16-frame clip centred on it with the frozen VideoMAE V2-S, kept as a 7×7 token grid.
3. **Read** the grid with MQR and add the result to the CLIP embedding through the gate.
4. **Score** with LaGoVAD's RoFormer, text co-attention and heads, given the definition text. Each step's score is assigned to the frames it stands for.
5. **Map** each step's evidence E\_t from the per-query scores and attention, upsample it to frame size and normalise it per video.

**Definitions.** For comparability, every main result uses LaGoVAD's released definitions: TAD is prompted as {"Normal", "Car Accident"}, and DoTA's "Abnormal" expands to the same car-accident texts. Yet 37 of the 60 abnormal videos on LaGoVAD's TAD list (36 of 60 on UMIL's list) are not accidents. A separate analysis therefore prompts each of TAD's 7 categories with its own hand-written definition; this is prompt engineering, not part of the component.

**Score post-processing.** DoTA and DADA-2000 contain only abnormal clips, so their scores are min-max normalised per video before pooling, following LaGoVAD's DoTA protocol (`offline_dota_eval.py`), which the thesis extends to DADA-2000. TAD, MSAD and CCD scores, which come with normal videos, are used raw. The spatial head g only weights the evidence map at test time; adding its score to the frame score is tested only as an ablation.

**Cost.** MQR and LaGoVAD's heads are negligible next to the encoders. One VideoMAE clip runs per sampled step, about 57 GFLOPs for ViT-S and 180 for ViT-B. On 30 fps sets that is one clip per 8 frames; on 10 fps DoTA and CCD it is one clip per frame. Simple-TAD measured a ViT-S video encoder of the same size at 95 FPS on half an A100 with one clip per frame. The model is offline: the temporal window looks both ways.

## 4. Experimental plan

The plan trains with video-level labels only, compares against published rows on identical test lists, and adds every second metric by re-scoring released checkpoints.

### 4.1 Dataset preparation

The thesis uses seven traffic datasets. TAD (fixed camera) carries the main comparison and MSAD (multi-scene surveillance with a traffic-accident slice) a second one. CCD and Nexar train the dashcam model, DoTA and DADA-2000 test it zero-shot, and DoTA's boxes test the evidence maps.

| Dataset | View | Size and split | Labels | Role | Source |
| --- | --- | --- | --- | --- | --- |
| TAD | Fixed CCTV, some first-person | 500 videos; UMIL split 400 train (210 normal / 190 abnormal), 100 test (40 / 60); 7 categories | Video labels (train), anomaly interval (test) | P1 train and test; P4 zero-shot test | Frames and split files via the UMIL repository; native fps unknown |
| MSAD | Fixed, 14 scenes | 720 videos at 30 fps; weak protocol 360 normal + 120 abnormal train, 120 + 120 test (20 traffic accidents) | Video labels (train), frame labels (test) | P2 train and test; traffic slice; P4 zero-shot test | MSAD project page (videos gated); I3D features public |
| CCD | Dashcam | 1,500 crash + 3,000 normal clips (normal clips from BDD100K), 50 frames at 10 fps, official split | Video labels; accident onset | P3 train and in-domain test | GitHub (Cogito2012/CarCrashDataset) |
| Nexar | Dashcam | 1,500 labelled clips (750 positive + 750 normal), about 40 s each; the 1,344-clip challenge test set is scored on Kaggle | Video labels; event and alert times | P3 training | Hugging Face |
| DoTA | Dashcam | 4,677 clips (3,275 train / 1,402 test), 10 fps, abnormal only | Anomaly window; 9 categories × ego-involved / ego-uninvolved; boxes with track IDs on 122,447 of 156,728 anomalous frames | P3 zero-shot test; STAUC; Step 0 pilot (train split only) | GitHub (MoonBlvd/Detection-of-Traffic-Anomaly) |
| DADA-2000 | Dashcam | 2,000 accident clips, 658,476 frames at 30 fps (Nexar counts 1,962 clips / 649K frames) | Accident intervals | P3 zero-shot test | Simple-TAD's split files |
| MM-AU | Dashcam | 11,727 accident videos pooled from CCD, A3D, DoTA and DADA-2000 | Accident start, collision, end | Optional zero-shot test, after removing videos shared with CCD, DoTA and DADA-2000 | MM-AU release (CVPR 2024) |

**Weights to download** (all public): CLIP ViT-B/16 and VideoMAE V2-S (`vit_s_k710_dl_from_giant`); LaGoVAD's `best.ckpt` and `pred_results.pkl` (Apache-2.0), trained on PreVAD and used only as a reference row; VadCLIP, PEL4VAD and DSANet checkpoints trained on UCF-Crime and XD-Violence, for zero-shot re-scoring; RTFM and MGFN checkpoints trained on MSAD; MOVAD and Simple-TAD DoTA and DADA-2000 checkpoints as frame-supervised context; and Simple-TAD's DAPT-VideoMAE-S (BDD100K only, CC BY-NC 4.0) for the encoder ablation. The links come from the official READMEs; none has been test-downloaded yet, and that check is the first task. These checkpoints were trained by their authors, so re-scoring them needs only their weights, not their training data.

### 4.2 Dataset processing

Every dataset goes through the same pipeline, so that one feature cache serves the baseline, the motion stream, MQR and all ablations.

1. **Decode and resample.** The CLIP stream keeps LaGoVAD's convention: one frame in eight, or every frame for the 10 fps sets (DoTA, CCD). The VideoMAE stream resamples every video to 10 fps; at 30 fps a raw 16-frame clip would span only 0.53 s instead of 1.6 s. TAD's native frame rate is unknown, so its clips take 16 consecutive released frames and their duration is reported as unknown.
2. **Cut clips.** For each CLIP step at time t₀, take the 16 frames centred on t₀, repeating edge frames at video boundaries. Resize to 224×224 without cropping, as for the CLIP stream.
3. **Extract CLIP features** with LaGoVAD's `extract_feat_clip.py` for all seven datasets, so that features match the released checkpoint.
4. **Extract and reduce VideoMAE V2-S tokens.** Keep the last-layer tokens (8×14×14×384), average 2×2 spatially and over the 8 temporal slots, and store 7×7×384 in fp16 (about 37.6 KB per clip) as one array per video. The token average reproduces the preliminary motion-stream feature from the same cache, provided that feature is a plain mean of the same tokens.
5. **Build labels.** Training uses video-level labels, category indices and the caption templates: TAD from UMIL's `TAD_train.txt`, MSAD from its weak split, CCD from its official split, Nexar from its positive / negative labels (event and alert times are kept for analysis only). Test labels are frame-level: TAD intervals, MSAD frame labels, CCD accident onsets, DoTA anomaly windows and DADA-2000 intervals. Each step's score is copied to the frames it stands for.
6. **Prepare DoTA's spatial labels.** For each clip keep `ego_involve`, the category, and the per-frame boxes of anomalous agents. Bin clips by the median, over the anomaly window, of the largest box's share of the frame: under 0.5%, 0.5–2%, 2–8%, over 8%. AUC per bin uses all frames of those clips, with precursor frames as negatives. The obstacle (VO) and out-of-control (OO) categories have no participant boxes and are left out of STAUC.
7. **Fix test lists.** LaGoVAD's lists (DoTA 1,402 clips; its 100-video TAD list) for zero-shot AUC comparability; DoTA's official 1,305-video list for STAUC; UMIL's TAD test list for P1 and as a second zero-shot column; MSAD's weak-protocol test set; CCD's official test split; Simple-TAD's DADA-2000 test split.
8. **Hold out validation data.** 15% of the P3 training videos, stratified by label and source, for the one-time hyperparameter choice.

**Storage of the token cache** (GB, fp16, VideoMAE V2-S):

| Dataset | 7×7 grid (default) | 4×4 grid | Mean vector only |
| --- | --- | --- | --- |
| TAD (all 500 videos) | 2.5 | 0.83 | 0.05 |
| CCD (one clip per frame) | 8.5 | 2.8 | 0.17 |
| Nexar (1.70M frames) | 8.0 | 2.6 | 0.16 |
| DoTA, all clips (one clip per frame) | 27.5 | 9.0 | 0.56 |
| DADA-2000 | 3.1 | 1.0 | 0.06 |

These are estimates. DoTA and CCD use one clip per 10 fps frame; the other sets one clip per eight frames. Only DoTA's 1,402 test clips and a 500-clip training subset are needed, roughly a third of its figure. TAD's quoted duration ("about 25 h") conflicts with its frame count (537,244 frames), so its figure may be off by up to about 5×. MSAD and MM-AU are measured at extraction.

**Leakage checks before any training.**

- Seventy-six of LaGoVAD's 100 TAD test videos are in UMIL's TAD training list, so a model trained on UMIL's split is never scored on LaGoVAD's TAD list.
- MM-AU pools CCD, A3D, DoTA and DADA-2000 videos. It is never used for training, and as a test set only after removing videos shared with CCD, DoTA and DADA-2000.
- Several datasets were collected from online videos, so a near-duplicate check with CLIP-feature similarity runs between every training set (TAD, MSAD, CCD, Nexar) and every test set.
- LaGoVAD's published zero-shot numbers are reproduced from `pred_results.pkl` before anything is trained.

### 4.3 Baseline running

Baselines run in three groups, preceded by one pilot diagnostic that checks MQR's premise before any full training.

**Step 0: pilot diagnostic (inference only, about one GPU-day, on a random 500-clip subset of DoTA's 3,275-clip training split).** It asks whether local motion tokens carry anomaly evidence that the global vectors lose. The 1,402 test clips are never used for design decisions.

- **D0.** Cosine similarity of CLIP embeddings 8 frames apart within a video, against random pairs across videos, raw and mean-centred. This calibrates the above-0.9 observation.
- **D1.** Mean token change inside the anomalous-agent boxes divided by the change outside, by ego-involved / ego-uninvolved clips and box-size bin. Computed for the CLIP global embedding, raw CLIP patches, ClearCLIP patches and VideoMAE tokens at 14×14 and 7×7.
- **D2.** Training-free AUC of three scores: global change, maximum token change and top-k token change.
- **D3.** A pointing game: how often the token with the largest change falls inside a box, against the box-area chance level.

MQR's premise holds if VideoMAE tokens change much more inside boxes than outside during the anomaly window, and if maximum-token scoring beats global scoring on ego-uninvolved and small-box clips. A ratio near 1 for raw CLIP patches would confirm, with data, that CLIP patches are the wrong place to read.

**Group A: reproduce LaGoVAD.**

| Row | Model | Training | Purpose |
| --- | --- | --- | --- |
| B0 | LaGoVAD, released checkpoint | PreVAD (by its authors) | Reproduce the published zero-shot numbers from `pred_results.pkl` (TAD 89.56, DoTA 62.60, MSAD 90.41), then re-score on UMIL's TAD list, DADA-2000, CCD and the MSAD traffic slice with AUC, AP and AUC\_A |
| B1 | LaGoVAD, retrained | Each protocol's training set (P1, P2, P3), 3 seeds | Fair baseline with exactly the thesis data, features and schedule |
| B2 | LaGoVAD + motion stream (VideoMAE V2-S, mean pooling) | Same, 3 seeds | The preliminary model; MQR's starting point |

**Group B: published methods with released weights, re-scored for a second metric.** VadCLIP and DSANet (UCF-Crime and XD-Violence checkpoints, on the same CLIP ViT-B/16 features); PEL4VAD (UCF-Crime and XD-Violence checkpoints, with I3D features extracted for TAD, DoTA and DADA-2000); RTFM and MGFN (MSAD checkpoints, on MSAD's public I3D features, for the traffic slice); MOVAD and Simple-TAD (DoTA and DADA-2000, frame-supervised context only). Each runs on the same test lists and is scored with AUC, AP and, where defined, AUC\_A.

**Group C: a same-feature competitor retrained in-domain.** VadCLIP's public code is retrained on P1, P2 and P3 with the thesis's CLIP features, giving a second weakly supervised CLIP row in every protocol. Methods without released weights are quoted from their papers only: UMIL and WSAL on TAD; π-VAD, GS-MoE and Alert-CLIP on MSAD; MultiDomain and WS-DoTA; and the frame-supervised TTHF, PromptTAD and ISCRTAD.

### 4.4 Evaluation protocols

Four protocols cover in-domain and zero-shot use on fixed-camera and dashcam video, using only the seven datasets above. P1 meets the comparison rule (at least two published methods on at least two metrics, same fixed-camera traffic data) without any recomputation. P2 adds four-metric rows on MSAD. P3 and P4 meet the rule on DoTA, TAD and MSAD once released checkpoints are re-scored for the second metric.

**Metrics.**

- **AUC\_O / AP:** frame-level ROC AUC and average precision, pooled over all test frames.
- **AUC\_A / AP\_A:** the same, over abnormal test videos only; they measure localisation inside abnormal videos.
- **DoTA and DADA-2000 AUC:** after per-video min-max normalisation (LaGoVAD's DoTA protocol), reported overall and for ego-involved and ego-uninvolved clips.
- **STAUC:** DoTA's spatio-temporal AUC. For each true-positive frame, TARR is the fraction of the top-N% highest-scoring map pixels (N set per frame, as in DoTA) that fall inside the union of anomalous-agent boxes. STAUC is the area under the TARR-weighted true-positive rate against the false-positive rate, so STAUC ≤ AUC. It is computed on DoTA's official 1,305-video list with `stauc.py`, adapted to take the 7×7 maps upsampled to its 256×256 grid; the unknown categories are analysed with AUC only.
- **Pointing game:** the share of anomalous frames whose map peak falls inside a box.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P1 TAD in-domain (fixed camera) | TAD train → TAD test, UMIL split | AUC\_O, AUC\_A, AP | UMIL and WSAL on both metrics; AUC\_O-only rows; VADor flagged; LaGoVAD and VadCLIP retrained |
| P2 MSAD in-domain (multi-scene surveillance) | MSAD weak train → MSAD test; traffic slice and non-traffic scenes scored separately | AUC, AP, AUC\_A, AP\_A | π-VAD and GS-MoE (four metrics) and Alert-CLIP (two), all on full MSAD; RTFM, MGFN, UR-DMU on the traffic slice |
| P3 dashcam | Nexar + CCD train → CCD test (in-domain); DoTA and DADA-2000 (zero-shot) | AUC, AP; DoTA ego-involved / ego-uninvolved AUC, STAUC, pointing game | DoTA: LaGoVAD, VadCLIP, PEL (published AUC, AP re-scored), WS-DoTA flagged; CCD and DADA-2000: released checkpoints re-scored; frame-supervised rows as context |
| P4 cross-setting zero-shot | P3 model → TAD (both lists) and MSAD; P2 model → TAD (both lists) | AUC, AP, AUC\_A | LaGoVAD, VadCLIP, PEL (published AUC on TAD and MSAD, AP re-scored); MultiDomain and MSAD's own cross-dataset rows (AUC) |

Training sources differ across zero-shot rows: LaGoVAD was trained on PreVAD, VadCLIP and PEL on UCF-Crime or XD-Violence, and the thesis models on traffic data only. Every zero-shot table therefore names each row's training set.

**Published in-domain rows (P1, P2).**

| Method | Venue | TAD AUC\_O | TAD AUC\_A | MSAD AUC | MSAD AP | MSAD AUC\_A | MSAD AP\_A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WSAL | IEEE TIP 2021 | 89.64 | 61.66 |  |  |  |  |
| UMIL | CVPR 2023 | 92.93 | 65.82 |  |  |  |  |
| VADor (LLM-partial, arXiv only) | — | 91.77 | 70.78 |  |  |  |  |
| GS-MoE | ICCV 2025 |  |  | 87.72 | 68.26 | 69.54 | 76.68 |
| π-VAD | CVPR 2025 |  |  | 88.68 | 71.26 | 71.25 | 77.86 |
| VadCLIP framework + CLIP | Alert-CLIP's baseline |  |  | 87.45 | 71.09 |  |  |
| VadCLIP framework + Alert-CLIP | CVPR 2026 |  |  | 89.63 | 78.24 |  |  |

TAD AUC\_O-only rows add context: MIST 89.2, RTFM 89.6, CUPL 91.66, MultiDomain (TAD→TAD) 90.71, TEVAD 92.3 (venue unverified) and LAP 94.4 (journal unverified; prompts possibly LLM-generated). UMIL drops to 90.71 AUC\_O when its backbone is frozen, the fairer reference for frozen features. MSAD's own traffic-accident slice gives AUC / AP of 62.2 / 51.8 (RTFM), 68.6 / 54.5 (MGFN) and 62.0 / 55.6 (UR-DMU); because its computation is not documented, it is recomputed with the released RTFM and MGFN checkpoints.

**Published zero-shot rows (P3, P4), AUC.** From LaGoVAD's Table 2; AP is re-scored for every row with released weights.

| Method | Training set | TAD (LaGoVAD list) | DoTA | MSAD |
| --- | --- | --- | --- | --- |
| MultiDomain | Multiple | 79.21 |  |  |
| PEL (I3D) | UCF-Crime | 86.27 | 53.05 | 79.82 |
| PEL (I3D) | XD-Violence | 43.02 | 44.97 | 68.25 |
| VadCLIP | UCF-Crime | 74.46 | 50.93 | 88.09 |
| VadCLIP | XD-Violence | 83.56 | 49.00 | 88.48 |
| VadCLIP | PreVAD | 85.96 | 50.59 | 89.79 |
| LaGoVAD | PreVAD | 89.56 | 62.60 | 90.41 |

Two more cross-dataset results add AUC context. MultiDomain's single-domain I3D models tested on TAD (split not stated) reach 73.75 from UCF-Crime, 71.10 from XD-Violence, 73.80 from LAD, 68.16 from UBIF and 48.57 from ShanghaiTech. MSAD's authors report UCF-Crime → MSAD AUC / AP of 71.9 / 47.4 (RTFM), 61.8 / 31.2 (MGFN) and 74.3 / 53.4 (UR-DMU).

Frame-supervised rows trained on DoTA are context only and are never ranked against weakly supervised rows. On DoTA: MOVAD 82.2, PromptTAD 83.9, Simple-TAD DAPT-VideoMAE-B 87.9 and -L 88.4. On DoTA → DADA-2000: MOVAD 75.2, PromptTAD 74.6, Simple-TAD DAPT-VideoMAE-S 84.3. WS-DoTA's best weakly supervised row, RTFM with its temporal-regularity transform, reaches 78.2 AUC on DoTA when trained in-domain with D²-City normal clips, on its own 1,031-video subset.

**Reporting rules.**

- TAD is reported on two separate columns, LaGoVAD's list and UMIL's list, and the two are never mixed.
- Every thesis number is a mean ± standard deviation over 3 seeds.
- Subsets (ego-involved / ego-uninvolved, box-size bins, categories, traffic slices) carry 95% bootstrap confidence intervals over videos. MQR against the preliminary model is tested with a paired bootstrap.
- All models use the same test lists, frame-to-score mapping and normalisation.

### 4.5 Ablation studies

The ablation ladder adds one ingredient per row on the same features, data and schedule, so each step isolates one design choice. Rows 0–6 report P1 (AUC\_O, AUC\_A), P2 (AUC, AP), P3 (CCD AUC; DoTA AUC by ego-involved / ego-uninvolved clips; DADA-2000 AUC), P4 on TAD's two lists, and STAUC where a map exists.

| Row | Variant | Isolates | Expected if MQR's premise holds |
| --- | --- | --- | --- |
| 0 | LaGoVAD, CLIP only | Baseline | Reference |
| 1 | + motion stream, mean pooling | Backbone change | Gain overall, smallest on ego-uninvolved clips |
| 2 | Max or GeM pooling over the 49 tokens | More tokens, no learning | Small or no gain over row 1 |
| 3 | Single-query attentive pooling | Learned selection | Gain over row 2 |
| 4 | Multi-query readout, M ∈ {4, 8, 16} | Several queries | Further gain on ego-uninvolved clips and small boxes |
| 5 | Row 4 + spatial MIL loss (**full MQR**) | The extra loss | Best detection; STAUC above row 1 (uniform map) and above the unweighted attention maps of rows 3–4 |
| 6 | Row 5 + definition-conditioned queries (exploratory, outside LaGoVAD-MQR) | Language-guided "where" | Higher sensitivity to the definition; detection at least equal |
| 7 | Row 5's readout on raw CLIP patches, and on ClearCLIP patches | Whether CLIP patches could replace motion tokens | Below row 5; raw CLIP patches near row 0 |

**Side ablations within the component.**

- **Grid:** 7×7 (default); two temporal halves of 7×7; 4×4, whose adaptive pooling over 14×14 uses overlapping bins and so does not start exactly at the mean; and 14×14 with tokens PCA-compressed to D/4, at the same storage as 7×7, whose zero-query start is the mean of the compressed tokens.
- **Loss form:** λ ∈ {0.1, 0.5, 1}; query aggregation by max, top-2 mean or log-sum-exp; the dense variant that applies BCE to every query score of normal videos.
- **Query interaction:** one self-attention layer among queries before scoring, a detector-free stand-in for agent interaction.
- **Diversity:** an orthogonality penalty on the queries' attention maps.
- **Initialisation:** zero queries with W\_V = I, against random initialisation.
- **Score fusion:** adding the spatial score to the frame score at inference.
- **Encoder:** VideoMAE V2-S (default) against VideoMAE-S (Kinetics-400) and the driving-adapted DAPT-VideoMAE-S (BDD100K only), to show MQR's gain does not depend on one encoder.

**Analyses.**

1. **Where the gain lands.** DoTA ego-involved vs ego-uninvolved clips crossed with clip-level box-size bins; the "unknown" categories (UK, UK\*); TAD AUC for Accident, PedestrianOnRoad and the remaining categories pooled; the MSAD traffic slice against MSAD's non-traffic scenes; Nexar-only against Nexar + CCD training.
2. **Spatial evidence.** STAUC and pointing game for MQR against a uniform map (row 1) and a centre prior, plus maps of successes and failures.
3. **Query behaviour.** Attention entropy per query, overlap between queries' maps, and which queries fire on which categories.
4. **Language.** For row 6, how much the score of accident frames changes when the definition includes or excludes accidents, and drift@5 on MSAD and XD-Violence.
5. **TAD definitions.** Main results with LaGoVAD's two-class TAD prompt, against each of TAD's 7 categories prompted with its own hand-written definition.
6. **Cost.** Parameters, storage and measured frames per second for rows 0, 1 and 5.

**Budget.** Rows 0–6 run on P1, P2 and P3 with 3 seeds; P4 reuses those models. Row 7 and the side ablations run on P1 and P3 only, first with one seed, then with three for the variants that are kept. Row 7 also needs its own dense CLIP-patch cache, about twice the default column in Section 4.2. A one-epoch timing run in the first week converts this plan into GPU-hours; if the total exceeds the budget, the ladder is cut to rows 0, 1, 3 and 5.

## References

**Baseline, backbone and component**

- Liu et al., [Language-Guided Open-World Video Anomaly Detection under Weak Supervision (LaGoVAD)](https://arxiv.org/abs/2503.13160), ICLR 2026; [code, checkpoint, PreVAD annotations](https://github.com/Kamino666/LaGoVAD-PreVAD)
- Psomas et al., [Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency (Efficient Probing)](https://arxiv.org/abs/2506.10178), [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7cf40d1ede4b494931bea0eeb836c650-Abstract-Conference.html); [code](https://github.com/billpsomas/efficient-probing)
- Orlova et al., [Simplifying Traffic Anomaly Detection with Video Foundation Models (Simple-TAD)](https://arxiv.org/abs/2507.09338), ICCV 2025 Workshops (2COOOL); [supplement](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/supplemental/Orlova_Simplifying_Traffic_Anomaly_ICCVW_2025_supplemental.pdf); [code and DAPT encoders](https://github.com/tue-mps/simple-tad)
- Wang et al., [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727), CVPR 2023 (motion encoder: distilled ViT-S, `vit_s_k710_dl_from_giant`)
- Tong et al., [VideoMAE](https://arxiv.org/abs/2203.12602), NeurIPS 2022
- Liu et al., [Grounding DINO](https://arxiv.org/abs/2303.05499), ECCV 2024

**Evidence for the component**

- Przewięźlikowski et al., [Beyond \[cls\]: Exploring the True Potential of Masked Image Modeling](https://openaccess.thecvf.com/content/ICCV2025/html/Przewiezlikowski_Beyond_cls_Exploring_the_True_Potential_of_Masked_Image_Modeling_ICCV_2025_paper.html), ICCV 2025
- Bardes et al., [V-JEPA](https://arxiv.org/abs/2404.08471), arXiv 2404.08471 (venue not verified)
- Lan et al., [ClearCLIP](https://arxiv.org/abs/2407.12442), ECCV 2024
- [CLIP Under the Microscope](https://arxiv.org/abs/2502.19842), CVPR 2025
- Liang et al., [Mind the Gap](https://arxiv.org/abs/2203.02053), NeurIPS 2022
- Park et al., [What Do Self-Supervised Vision Transformers Learn?](https://arxiv.org/abs/2305.00729), ICLR 2023
- Zhou et al., [AnomalyCLIP](https://arxiv.org/abs/2310.18961), ICLR 2024

**Weakly supervised and open-world VAD**

- Wu et al., VadCLIP, AAAI 2024; [code](https://github.com/nwpu-zxr/VadCLIP)
- Pu et al., Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection (PEL4VAD), IEEE TIP 2024; [code and checkpoints](https://github.com/Aaron-Pu/PEL4VAD)
- [DSANet](https://arxiv.org/abs/2511.10334), AAAI 2026; [code](https://github.com/lessiYin/DSANet)
- Lv et al., [Unbiased Multiple Instance Learning (UMIL)](https://arxiv.org/abs/2303.12369), CVPR 2023; [TAD split files](https://github.com/ktr-hubrt/UMIL)
- Lv et al., Localizing Anomalies from Weakly-Labeled Videos (WSAL, TAD dataset), IEEE TIP 2021; [repository](https://github.com/ktr-hubrt/WSAL)
- [MultiDomain](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf), NeurIPS 2024
- [π-VAD](https://arxiv.org/abs/2505.13123), CVPR 2025
- [GS-MoE](https://arxiv.org/abs/2508.06318), ICCV 2025
- Alert-CLIP, CVPR 2026 (project file); [annotations](https://github.com/ClarkZhu216/Alert-CLIP_dataset)
- [STPrompt](https://arxiv.org/abs/2408.05905), ACM MM 2024
- Localizing to Debias, arXiv 2608.12045 (not yet read in full)
- [Learning Where and When](https://engineers.usf.edu/sislab/publications/learning-where-and-when-patch-based-spatiotemporal-localization-weakly-supervised), arXiv 2606.29498
- Project files: RefineVAD and HeadHunt-VAD (AAAI 2026); LAS-VAD, Fine-VAD, The Road Less Seen and LAVIDA (CVPR 2026); Anomize, VERA, Holmes-VAU, TAO and ADSM (CVPR 2025); FedWSVAD and MA-PDM (AAAI 2025); EventVAD (ACM MM 2025); AnomalyRuler (ECCV 2024); LAVAD (CVPR 2024); AnyAnomaly (WACV 2026); SteerVAD

**Traffic anomaly detection and anticipation**

- Qiu et al., [PromptTAD](https://doi.org/10.1109/LRA.2025.3572428), IEEE RA-L 2025
- Shi et al., [Structured Graph Reasoning for Traffic Anomaly Detection](https://doi.org/10.1109/TMM.2026.3703345), IEEE TMM 2026
- Rossi et al., MOVAD, ICASSP 2024; [code](https://github.com/IMPLabUniPr/movad)
- Liang et al., TTHF, IEEE TCSVT 2024; Liang et al., ISCRTAD, IEEE T-ITS 2025
- [WS-DoTA](https://arxiv.org/abs/2408.05562), ECCV 2024 Workshop; [split files](https://github.com/ut21/WSAD-Driving)
- [BADAS](https://arxiv.org/abs/2510.14876), arXiv 2510.14876; [FLaRA](https://arxiv.org/abs/2606.14380), arXiv 2606.14380

**Datasets**

- Yao et al., [DoTA](https://arxiv.org/abs/2004.03044), IEEE TPAMI 2022; [annotations and STAUC code](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
- Zhu et al., [MSAD](https://msad-dataset.github.io/), NeurIPS 2024 Datasets and Benchmarks; [repository](https://github.com/Tom-roujiang/MSAD)
- Bao et al., [CCD](https://github.com/Cogito2012/CarCrashDataset), ACM MM 2020
- Fang et al., [MM-AU](https://arxiv.org/abs/2403.00436), CVPR 2024; [CAP-DATA](https://arxiv.org/abs/2212.09381)
- Fang et al., DADA-2000, IEEE T-ITS 2021
- Moura et al., [Nexar Dashcam Collision Prediction Dataset and Challenge](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf), CVPR 2025 Workshops

**Other methods, datasets and models cited**

- Sultani et al., Real-World Anomaly Detection in Surveillance Videos (UCF-Crime), CVPR 2018
- Feng et al., MIST, CVPR 2021; Tian et al., RTFM, ICCV 2021
- Zhang et al., CUPL, CVPR 2023; Zhou et al., UR-DMU, AAAI 2023; Chen et al., MGFN, AAAI 2023
- TEVAD (venue unverified); [LAP](https://arxiv.org/abs/2403.01169) (journal unverified); [VADor](https://arxiv.org/abs/2401.05702) (arXiv only)
- Wu et al., XD-Violence, ECCV 2020; LAD (test list from LaGoVAD's repository)
- Yu et al., BDD100K, CVPR 2020; D²-City, arXiv 2019
- Wang et al., InternVideo2, ECCV 2024; Qwen2.5-VL technical report, 2025
