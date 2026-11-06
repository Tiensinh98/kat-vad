# Thesis proposal: Deep learning-based traffic video anomaly detection (LaGoVAD-KIM)

Nov 6, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-KIM**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. The baseline is LaGoVAD (ICLR 2026) with LaGoVAD's released text setup, plus a frozen VideoMAE V2-S motion stream fused into the CLIP branch through a learnable, zero-initialised projection W\_u. Trained on Nexar dashcam video with bag-level labels, this motion-stream model already reaches **0.81 macro / 0.73 micro per-clip min-max AUC zero-shot on DoTA-CAP-dev** (569 clips, one seed), against 0.64 / 0.61 for the released LaGoVAD checkpoint read the same way. The comparison is not matched: the checkpoint was trained on PreVAD.

The one novel component is the **Kinematic Intention Module (KIM)**, a traffic adaptation of the Intention Awareness Module (IAM) of LAS-VAD (CVPR 2026). KIM reads the temporally contextualised features that leave LaGoVAD's RoFormer, derives a position, a velocity and an acceleration feature per step, and returns them to the host before text fusion. It is trained with its own top-k MIL term and a normal-anchored kinematic contrastive loss built on two momentum prototypes. KIM needs no new feature extraction: it reads the cached CLIP and VideoMAE features through the host, and the VideoMAE stream is kept exactly as in the motion-stream model. The claim under test: a collision is an abrupt change in the scene's state, and the *magnitude* of first- and second-order change of the contextualised feature is evidence that LaGoVAD's linear 1-D convolution heads cannot compute from the features themselves. To keep KIM's differences clean, every KIM row and its control train without LaGoVAD's dynamic video synthesis, which splices clips and would put artificial jumps into the sequence KIM differences.

This version drops three parts of the previous version. **SA²** was trained and could not locate agents better than a centre prior, so the CLIP branch returns to one global vector per step and spatial maps leave the scope. The **8-token learnable prefix** is replaced by LaGoVAD's own 32 soft-prompt tokens. LAS-VAD's **anomaly-connected components (ACC)** are not borrowed: they add a T×T graph, a DFS and frame pseudo-labels, and nothing in LAS-VAD shows they help on traffic video.

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event is happening in it. Abnormal events include collisions, loss of control, illegal turns, wrong-way driving, pedestrians on the road and road spills, the categories used by TAD and DoTA.

Two camera settings define the problem space:

- **Fixed camera (non-ego).** TAD and the traffic scenes of MSAD. The camera is static; the anomalous agents are often small and far away.
- **Dashcam (ego-centric).** DoTA, DADA-2000, CCD and Nexar. The camera moves with the car. DoTA marks each clip as ego-involved (the ego car takes part) or ego-uninvolved (it only observes).

The thesis follows LaGoVAD's weakly supervised, language-guided formulation. Training uses one label y\_i ∈ {0, 1} per training bag (a Nexar window labelled from its alert and event times, Section 3.3), plus a hand-written anomaly definition. At test time the model receives a video V and a text definition Z of what counts as anomalous, and returns a score per sampled step:

```latex
\Phi_\theta : (V, Z) \;\mapsto\; \{ s_t \}_{t=1}^{N}, \qquad s_t \in [0,1]
```

The bag labels supervise training through multiple-instance learning (MIL): the top-k step scores of a bag must match its label. This version outputs no spatial map.

Four properties make traffic video harder than generic surveillance:

1. **Anomalies are motion events, and many are abrupt.** A collision is defined by how trajectories meet and by a sudden change of speed or heading, not by what one frame looks like. LaGoVAD encodes single RGB frames with CLIP; it publishes 89.56 AUC on TAD but 62.60 on dashcam DoTA. Part of this gap is protocol: DoTA is abnormal-only and min-max normalised per video, so its AUC measures timing inside a clip. Re-scoring the released checkpoint here gives 0.7912 micro / 0.7579 macro on TAD and 0.6105 micro / 0.6418 macro on DoTA-CAP-dev.
2. **The agents that matter are small.** A distant agent of about 40×30 px in a 1280×720 frame shrinks to about 7×9 px at 224×224 input. The SA² experiment showed that weakly supervised patch selection does not recover them on this data, so the thesis now targets *when* the scene changes rather than *where*.
3. **The camera moves.** In dashcam video, ego-motion dominates frame-to-frame change, so a large change is normal; what is abnormal is a change that is sudden relative to the clip's own motion.
4. **Labels are scarce.** Frame labels are expensive and anomalies are rare, so the model learns from video labels.

The scope is offline detection: the whole video is available, and the question is whether an anomaly is happening, not whether one will happen (anticipation, as in Nexar). No LLM or multimodal LLM runs at training or test time; a frozen CLIP text encoder is allowed. All training text is hand-written.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from cheap video-level labels, sees motion, reacts to *abrupt* change of the scene, and keeps LaGoVAD's ability to take the anomaly definition as text, on one academic GPU without any LLM.

The thesis builds this in two steps, matching the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** A frozen VideoMAE V2-S encoder feeds one 384-d vector per step into LaGoVAD through a zero-initialised projection W\_u. Preliminary result: zero-shot DoTA-CAP-dev **0.81 macro / 0.73 micro** per-clip min-max AUC (one seed). Three attempts to add more (MQR, LSP, SA²) did not help. The stream is used unchanged in this proposal.
- **Novel component (KIM).** On the RoFormer output, KIM computes position, velocity and acceleration features, returns them to the host through a second zero-initialised projection, and trains them with a kinematic MIL term and a contrastive loss anchored on normal training bags.

A second purpose is a fair evaluation: published traffic numbers are mostly single-metric AUC on inconsistent splits. The thesis trains on Nexar only and reports zero-shot results on DoTA-CAP-dev, DADA-2000 and TAD with more than one metric, against the released LaGoVAD checkpoint read the same way, with each row's protocol labelled.

### 1.3 Research aims

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint and retrain it on Nexar, with macro and micro AUC and AP on DoTA-CAP-dev, DADA-2000 and TAD.
2. **Document the motion stream and the negative results.** Report the VideoMAE stream's gain over CLIP-only LaGoVAD, and the MQR, LSP and SA² results that fixed the backbone.
3. **Design and train KIM** with video-level labels only.
4. **Measure temporal precision.** Score how close KIM's evidence lands to the event onset, with DoTA-CAP's anomaly start and DADA-2000's accident interval.
5. **Explain the gains.** Break results down by DoTA category, ego flag and camera setting (dashcam DoTA-CAP and DADA-2000 against fixed-camera TAD); test KIM's placement, its derivative order and its loss; and check that language guidance does not regress.

"Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval over test videos excludes zero.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. KIM improves on the motion-stream model, most on ego-involved collisions | DoTA-CAP-dev macro / micro AUC overall, by category and ego flag; DADA-2000 macro AUC; TAD macro AUC (rows R0, R5) | R5 > R0 overall, with larger gains on ego-involved than on ego-uninvolved clips |
| H2. The gain comes from the magnitude of change, not from extra capacity | Matched-capacity MLP on the RoFormer output (R-cap); KIM with position only (R6a) | R5 > R-cap and R5 > R6a |
| H3. Second-order change matters; contextualised features are the right input | Order ladder p / p+v / p+v+a (R6); placement on c\_t, x\_t or h\_t (R7) | p+v+a best; h\_t ≥ x\_t > c\_t |
| H4. A contrastive loss anchored on normal videos beats LAS-VAD's pseudo-label version | R2 (MIL only), R4 (LAS-VAD contrastive), R5 (normal-anchored) | R5 > R4 ≥ R2 |
| H5. Language guidance is kept | Score change of abnormal steps when the definition includes or excludes the anomaly; TAD per-category definitions; score-level fusion variant (R11) | R5 no less definition-sensitive than R0; R11 less sensitive |
| H6. KIM's evidence is temporally precise | Onset error (s) of KIM's acceleration peak on DoTA-CAP-dev and DADA-2000 | Median onset error below that of R0's score peak |

## 2. Literature review

### 2.1 Overview and classification of video anomaly detection method branches

Video anomaly detection (VAD) methods divide by the supervision they train with into five branches, plus the adjacent task of accident anticipation. Traffic methods cut across these branches along a second axis: fixed camera or dashcam.

| Branch | Training signal | Typical mechanism | Representative methods (venue) | Traffic evidence | Role in this thesis |
| --- | --- | --- | --- | --- | --- |
| Unsupervised / one-class | Normal videos only | Reconstruction, frame or flow prediction, future object location, diffusion, scene graphs | AnoPred (CVPR 2018), FOL / DoTA (IEEE TPAMI 2022), MAMTCF (arXiv 2307.14575), MA-PDM (AAAI 2025), ADSM (CVPR 2025), TAO (CVPR 2025), Structured Graph Reasoning (IEEE TMM 2026) | DoTA AUC 67.5–76.6 | Context; ego-motion breaks most predictors |
| Weakly supervised | Video labels, MIL | Snippet features (I3D, CLIP) + top-k MIL; feature magnitude; memories and prototypes; temporal-change and intention modules | Sultani (CVPR 2018), RTFM (ICCV 2021), WSAL (IEEE TIP 2021), UMIL (CVPR 2023), MGFN, UR-DMU (AAAI 2023), VadCLIP (AAAI 2024), STPrompt (ACM MM 2024), TPWNG (CVPR 2024), π-VAD, GS-MoE (2025), DSANet, RefineVAD (AAAI 2026), **LAS-VAD (CVPR 2026)**, Alert-CLIP (CVPR 2026) | TAD (WSAL, UMIL), MSAD (π-VAD, GS-MoE, Alert-CLIP), WS-DoTA (preprint) | **The thesis's branch; LAS-VAD is the source of the novel component** |
| Open-vocabulary / open-world | Video labels + category text | CLIP text space defines anomaly classes or free-text definitions at test time | Anomize (CVPR 2025), **LaGoVAD (ICLR 2026)** | Zero-shot TAD, DoTA, MSAD (LaGoVAD) | **Baseline** |
| Fully supervised traffic detection | Frame labels (+ boxes) | Video encoder + temporal head; frame differences; object prompts; agent graphs | MOVAD (ICASSP 2024), TTHF (IEEE 2024), PromptTAD (IEEE RA-L 2025), ISCRTAD (IEEE T-ITS 2025), Simple-TAD (ICCV 2025 Workshops) | DoTA AUC 82.2–88.4 in-domain | Upper reference rows; evidence that temporal change helps (TTHF, MOVAD) |
| LLM / MLLM-based | None, few-shot rules or light tuning | Captioning + LLM reasoning, prompting, probing of MLLM heads | LAVAD (CVPR 2024), AnomalyRuler (ECCV 2024), VERA, Holmes-VAU (CVPR 2025), EventVAD (ACM MM 2025), HeadHunt-VAD (AAAI 2026), LAVIDA (CVPR 2026), AnyAnomaly (WACV 2026) | Almost none on traffic sets | Excluded: LLM cost |
| Accident anticipation (adjacent) | Positive / negative dashcam clips | Predict a collision before it happens; forecast future latents | DSTA, UString (ACM MM 2020), BADAS (arXiv 2510.14876), FLaRA (arXiv 2606.14380) | Nexar, CCD, DAD; video-level anticipation AUC | Different task; FLaRA's idea was tested as LSP and dropped |

Three findings from this review shape the proposal. First, CLIP-feature weak supervision dominates surveillance benchmarks but sits near chance on dashcam DoTA, and the strongest traffic results come from video foundation models and explicit temporal-change streams (Simple-TAD, TTHF, MOVAD). Second, local and object-level evidence (PromptTAD, Alert-CLIP) needs detectors or boxes, and this thesis's detector-free attempt (SA²) did not localise agents. Third, LAS-VAD is the one weakly supervised design that models *rate of change* of a contextualised feature as its own module with its own loss, and it has never been tested on traffic.

### 2.2 Emerging trends and unresolved challenges

Six trends run through the 2024–2026 literature:

1. **Vision-language features as the default.** Since VadCLIP, most weakly supervised methods start from frozen CLIP features and CoOp-style learnable text context: STPrompt, DSANet, RefineVAD, LAS-VAD, Fine-VAD and Alert-CLIP.
2. **Anomalies defined by language.** Open-vocabulary (Anomize) and open-world (LaGoVAD) models accept the anomaly definition at test time.
3. **MLLMs for explanation and training-free detection.** LAVAD, VERA, Holmes-VAU, HeadHunt-VAD and LAVIDA rely on models of 2B parameters and more, and are not clearly better at detection: on MSAD drift@5, LaGoVAD reaches 85.6 AUC / 40.1 AP against 84.3 / 34.3 for Holmes-VAU.
4. **Masked-video foundation models for traffic.** Simple-TAD showed that masked video modelling transfers best to traffic anomalies.
5. **Explicit temporal-change modelling.** TTHF's frame-difference branch (+7.9 DoTA AUC), MOVAD's short memory (+7.9), WS-DoTA's CLIP-feature differences and LAS-VAD's velocity and acceleration features all add an explicit measure of change on top of appearance.
6. **Normal-anchored prototypes and contrast.** UR-DMU, MGFN, TPWNG and DSANet build the reliable side of a contrast from normal videos; LAS-VAD uses class prototypes from its own predictions.

Five challenges remain open for traffic video:

1. **Motion blindness.** Frame-level CLIP features miss what makes a collision a collision. In LaGoVAD's table every surveillance-trained model is near chance on DoTA's within-clip protocol (VadCLIP, LaGoVAD: 49.00–62.60).
2. **Abruptness is not modelled under weak labels on traffic.** The traffic methods that model change (TTHF, MOVAD) need frame labels; the only weakly supervised dashcam study (WS-DoTA) uses fixed, raw CLIP differences, and its amplified variant raised false positives.
3. **Spatial dilution.** Global vectors lose small, distant agents (Simple-TAD: 81.6–85.2 AUC on ego-uninvolved DoTA clips against 89.9–90.7 on ego-involved). This thesis's SA² did not fix it; it stays a stated limitation.
4. **Ego-motion and domain shift.** In dashcam video a large change is normal, and training and test data come from different sources.
5. **Inconsistent evaluation.** TAD has two incompatible test lists (76 of LaGoVAD's 100 TAD test videos are in UMIL's training list), and DoTA numbers come from at least four protocols.

### 2.3 Key works behind the proposed model

| Part of LaGoVAD-KIM | Source work (venue) | What is taken | What the thesis changes |
| --- | --- | --- | --- |
| Host detector: CLIP stream, RoFormer, soft prompts, co-attention, heads and losses | LaGoVAD (ICLR 2026), on VadCLIP's frozen-CLIP + MIL design | Architecture, 32 soft-prompt tokens (16 + 16), losses as released | Last-step checkpoints; hand-written definitions; fixed-time sampling; score head kernel 3 and top-k of L/5 for short Nexar bags; no caption loss (Nexar has no captions); dynamic video synthesis off in KIM rows |
| Motion stream (backbone change) | VideoMAE V2 (CVPR 2023), chosen on the evidence of Simple-TAD | Frozen distilled ViT-S and a 16-frame clip input | Causal 1.5 s clip, one pooled 384-d vector per step, standardised with fixed statistics, fused through a zero-initialised W\_u; unchanged by KIM |
| KIM features and placement (novel component) | LAS-VAD's IAM (CVPR 2026) | Position, gated velocity and acceleration after temporal modelling | Padding mask, per-video scale, fixed time step, return path before text fusion |
| KIM score, prototypes and loss | LAS-VAD's IAM; normal-anchored contrast as in UR-DMU, MGFN, TPWNG | Confidence-weighted logits, momentum prototypes, InfoNCE | Binary classes, (1 + cos) / 2 confidence, normal prototype from normal bags, prototype positive, exact hard negatives, own MIL term |
| Evidence that change cues help on traffic | TTHF, MOVAD, WS-DoTA | — | Motivates KIM and the ego-involved analysis |

#### 2.3.1 LaGoVAD (ICLR 2026): the baseline

Liu et al. define open-world VAD as a mapping Φ: (V, Z) → frame scores, with the definition Z given as text at test time. Frozen CLIP ViT-B/16 frame embeddings pass through a 2-layer RoFormer and are fused with the CLIP text encoding of Z by co-attention. Training uses video labels on PreVAD (35,279 videos) with MIL and MIL-alignment losses, dynamic video synthesis and a caption contrastive loss with hard negatives.

- **Strengths.** The best zero-shot traffic results in its comparison: 89.56 AUC on TAD, 62.60 on DoTA, 90.41 on MSAD. It stays robust when the definition changes (MSAD drift@5: 85.6 AUC / 40.1 AP), runs no LLM, trains on one RTX 4090, and its code and checkpoint are public.
- **Weaknesses.** One global CLIP embedding per frame: no motion and no explicit rate of change. Its head is a single linear convolution. Its released configuration selects checkpoints on test scores.
- **Taken into this thesis.** The whole detector, its text setup included.

#### 2.3.2 VideoMAE V2 (CVPR 2023) and Simple-TAD (ICCV 2025 Workshops): the motion stream

Wang et al. scale masked video autoencoding and distil a ViT-g into small ViT-S and ViT-B models. Orlova et al. show that a plain video ViT with one linear head, fine-tuned with frame labels, beats specialised traffic detectors on DoTA (up to 88.4 AUC) and DADA-2000.

- **Strengths.** In Simple-TAD's comparison the distilled ViT-S is the strongest small encoder for traffic anomalies (AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000). In this thesis's pilot its mean vector lifts zero-shot DoTA-CAP-dev to 0.81 macro / 0.73 micro.
- **Weaknesses.** No driving video in pre-training; features are not text-aligned; ego-uninvolved DoTA clips lag ego-involved ones by 5–8 AUC even with frame labels. MQR and LSP added nothing beyond the mean vector.
- **Taken into this thesis.** The frozen ViT-S, read as one standardised vector per step from a causal 1.5 s clip, exactly as in the motion-stream model.

#### 2.3.3 LAS-VAD (CVPR 2026): the source of the novel component

Wang and Zhao add three parts to a CLIP-based weakly supervised detector with a local Transformer and GCN: anomaly-connected components (ACC), which group frames by connected components of a thresholded similarity graph and produce frame pseudo-labels; LLM-written attribute texts per category; and the Intention Awareness Module (IAM).

- **IAM.** From the temporally modelled feature X\_f: X\_p = fc(X\_f); velocity X\_v = σ(Conv(|ΔX\_p|)) ⊙ |ΔX\_p|; acceleration the same on X\_v (kernel 3, zero padding, each D/3). Two FC layers give frame logits, weighted by cosine similarity to the momentum prototype of the predicted class (β = 0.1); an InfoNCE loss takes the least similar same-class frame as positive and the M most similar other-class frames as negatives.

| LAS-VAD ablation, XD-Violence fine-grained mAP | mAP@0.1 | mAP@0.5 | Average |
| --- | --- | --- | --- |
| None of the three parts | 36.86 | 14.11 | 24.24 |
| IAM only | 42.91 | 19.05 | 29.98 |
| ACC + attributes | 46.77 | 21.42 | 33.25 |
| All three | 50.18 | 24.65 | 36.89 |

- **Strengths.** IAM alone adds 5.74 average mAP, as much as ACC alone (+5.54), at small cost and with no extra input. It reports 87.92 AP on XD-Violence and 90.96 AUC on UCF-Crime with CLIP features.
- **Weaknesses.** No code. No traffic data. No internal ablation of IAM (velocity vs acceleration, gate, prototypes, loss), and no IAM ablation on the coarse AUC / AP metrics. Its classes come from its own argmax, its positive is the least similar same-class frame, and its printed InfoNCE leaves the positive out of the denominator. ACC needs a graph threshold and frame pseudo-labels; the attributes need an LLM.
- **Taken into this thesis.** IAM's placement, its three-part kinematic feature with gates, confidence-weighted logits and momentum prototypes. **Changed:** binary classes, normal-anchored contrast with exact negatives, KIM's own MIL term, return to the host before text fusion, and the traffic adaptations of Section 3.2.1. **Not taken:** ACC and the LLM attributes.

#### 2.3.4 TTHF, MOVAD and WS-DoTA: evidence that temporal change helps on traffic

- **TTHF (IEEE 2024).** Encodes consecutive-frame differences with its own ResNet-50 and adds them to a CLIP-aligned visual context; DoTA AUC 76.8 → 84.7, DADA-2000 71.7 zero-shot. Frame-supervised; weakest on ego-uninvolved clips. *Role:* evidence for H1 and its ego-involved prediction.
- **MOVAD (ICASSP 2024).** A short-term memory of 4 frames (0.4 s) adds 7.9 AUC on DoTA; 1 frame overfits and 5 or more hurt. *Role:* evidence for a sub-second time scale.
- **WS-DoTA (arXiv 2408.05562, preprint).** The only weakly supervised dashcam study; absolute shifted CLIP-feature differences with a sigmoid of the features lift RTFM to 78.2 AUC on its own DoTA split; a DCT-amplified difference hurt. *Role:* evidence that CLIP-feature change carries dashcam motion under weak labels, and that amplifying it raises false positives; a flagged P3 context row.

#### 2.3.5 Methods used as comparison rows

| Method (venue) | Approach | Strength | Weakness | Role in this thesis |
| --- | --- | --- | --- | --- |
| WSAL (IEEE TIP 2021) | High-order context encoding; introduced TAD | 89.64 AUC\_O / 61.66 AUC\_A on TAD | Hand-crafted cues | Context row (in-domain TAD) |
| UMIL (CVPR 2023) | Confident / ambiguous snippet split; X-CLIP fine-tuned | 92.93 AUC\_O / 65.82 AUC\_A on TAD | Needs backbone fine-tuning (90.71 frozen) | Context row (in-domain TAD) |
| π-VAD (CVPR 2025) | Induces five modalities from RGB at training time | MSAD 88.68 AUC / 71.26 AP | Five teacher backbones | Context row (in-domain MSAD) |
| GS-MoE (ICCV 2025) | Gaussian pseudo-labels plus one expert per class | MSAD 87.72 / 68.26 / 69.54 / 76.68 | Needs class labels | Context row (in-domain MSAD) |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment with boxes and captions | VadCLIP framework on MSAD 87.45 / 71.09 → 89.63 / 78.24 | Needs boxes and VLM captions | Context row (in-domain MSAD) |
| WS-DoTA (preprint) | Weakly supervised DoTA with CLIP-feature differences | RTFM 57.9 → 78.2 AUC | Own \~1,031-clip subset; AUC only | Flagged context row |
| LAS-VAD (CVPR 2026) | IAM + ACC + attributes | XD-Violence 87.92 AP (CLIP) | No code, no traffic | Group C reimplementation of IAM on the thesis host |

### 2.4 Research gaps

1. **No motion-aware foundation features in weakly supervised traffic VAD.** *Addressed by the motion stream, already shown to lift zero-shot DoTA-CAP-dev to 0.81 macro / 0.73 micro.*
2. **No learned model of abrupt change under weak labels on traffic.** Change-based traffic detectors need frame labels, the weakly supervised dashcam study uses fixed raw CLIP differences, and LAS-VAD's IAM has never been tested on traffic, under ego-motion, next to a motion stream or in a language-guided host. *Addressed by KIM.*
3. **Intention prototypes rely on self-assigned classes.** LAS-VAD's prototypes and contrast take classes from the model's own argmax, which fails silently when the training data are binary. *Addressed by the normal-anchored prototype and contrast, tested against LAS-VAD's form.*
4. **Traffic results are hard to compare.** *Addressed by one training set and three labelled zero-shot test protocols, each with more than one metric (Section 4.4).*

| Closest work | What it does | How LaGoVAD-KIM differs |
| --- | --- | --- |
| LAS-VAD (CVPR 2026) | IAM with pseudo-label prototypes and contrast; XD-Violence, UCF-Crime | Traffic and dashcam data; open-world language-guided host with a motion stream; binary normal-anchored contrast; padding mask; per-video scale; return before text fusion |
| WS-DoTA (preprint) | Fixed absolute CLIP-feature differences for weakly supervised DoTA | Learned position, velocity and acceleration of the contextualised feature, with its own loss |
| TTHF (IEEE 2024) | Pixel frame-difference branch, frame labels | Video-level labels; feature-space kinematics; no extra encoder |
| MGFN (AAAI 2023), RTFM (ICCV 2021) | Feature *magnitude* per snippet | Magnitude of *change* across steps |
| UR-DMU (AAAI 2023) | Normal and abnormal memories with a triplet loss | Prototypes in a kinematic space, with exact hard negatives from normal bags |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that models the velocity and acceleration of contextualised features with a normal-anchored contrastive loss, inside a language-guided open-world detector with a motion-foundation stream, and the first test of LAS-VAD's intention module on traffic video and under ego-motion.** Kinematic features and prototypes are not claimed as new; the contribution is their traffic adaptation, the label-safe loss and the analysis by camera setting, category and onset time.

## 3. Proposed model

The proposed model, **LaGoVAD-KIM**, is LaGoVAD with two changes: a frozen VideoMAE motion stream fused through W\_u (the backbone change) and the Kinematic Intention Module (the novel component), which sits between the RoFormer and the text fusion. Every other part of LaGoVAD stays as released, so any gain can be traced to these changes.

### 3.1 Base architecture

**LaGoVAD, unchanged parts.**

- **CLIP stream.** A frozen CLIP ViT-B/16 image encoder gives one 512-d CLS embedding c\_t per sampled step, from a 224×224 resize without cropping.
- **Temporal encoder.** A residual RoFormer: 2 layers, 4 heads, hidden size 512, local window of 25 steps. Training bags are padded to the longest bag in the batch, and padded steps are masked. Its output is h\_t ∈ R^512 (LaGoVAD's v^t).
- **Text branch.** The frozen CLIP ViT-B/16 text encoder encodes each category of the definition Z (a class name or a description; training picks one at random per batch) into τ\_c ∈ R^512. As released, 32 learnable soft-prompt tokens surround the definition tokens, half before and half after (16 + 16).
- **Fusion.** A 2-layer co-attention Transformer lets video steps and text tokens attend to each other.
- **Heads.** A single 1-D convolution (kernel 3, the short-window recipe of the motion-stream model: LaGoVAD's kernel 9 sees nearly half of a 20-step bag and drifts towards a clip classifier; replicate padding) scores the pre-fusion and post-fusion features; a learned weight w = σ(10·a), a initialised at 0, blends them, and the step score is the sigmoid of the blend. A linear classification head compares projected video and text features.
- **Losses.** Top-k MIL binary cross-entropy (k = max(⌊L/5⌋, 1) on Nexar's 20-step bags; LaGoVAD's L/16 gives k = 1), a multi-class MIL alignment loss, and dynamic video synthesis L\_dvs (an abnormal clip inserted among 1–5 kNN-retrieved normal clips, giving frame pseudo-labels). LaGoVAD's caption contrastive loss with hard negatives L\_neg needs per-video captions; Nexar has none, so L\_neg does not run in any row of this thesis. **Dynamic video synthesis is switched off in every KIM row and in its control row** (Section 3.2.1): a spliced sequence has jumps at its seams that KIM would read as abrupt change. With no splice, L\_dvs keeps only its part on normal bags, where every step carries the exact label 0; that part has no seams and stays on.

**Backbone change: the motion stream (unchanged from the motion-stream model).** For each sampled step t, a causal 16-frame clip that ends at t's frame (every third frame at 30 fps, so 1.5 s; every frame on 10 fps DoTA-CAP), padded by repeating frame 0, is squash-resized to 224×224 and passes through a frozen **VideoMAE V2-S** encoder (ViT-S distilled from VideoMAE V2-g on Kinetics-710, D = 384). Its 8×14×14 last-layer tokens are averaged and passed through the encoder's frozen final normalisation into v̄\_t, standardised with fixed statistics m\_u, σ\_u and a fixed scale c fitted on Nexar's training split, and added through a projection that starts at zero:

```latex
\tilde v_t = \frac{\bar v_t - m_u}{\sigma_u} \in \mathbb{R}^{384}, \qquad x_t = c_t + W_u\, (c\,\tilde v_t), \qquad W_u \in \mathbb{R}^{512 \times 384},\; W_u^{\text{init}} = 0
```

x\_t replaces c\_t as the RoFormer input; the kNN retrieval of dynamic video synthesis, where it runs, still uses c\_t. At initialisation the model is CLIP-only LaGoVAD, and W\_u receives a non-zero gradient from the first step because ṽ\_t is non-zero. Trained on Nexar with dynamic video synthesis, this model reaches **0.81 macro / 0.73 micro** zero-shot on DoTA-CAP-dev (one seed). The c\_t and ṽ\_t features of every training and test set used here are already cached, so this proposal needs no new extraction.

**Sampling at a fixed time step.** KIM differences neighbouring steps, so a step must mean the same time everywhere. The CLIP stream therefore samples by native frame rate: stride 8 at 30 fps (0.27 s), stride 3 at 10 fps (0.30 s), stride 7 at 25 fps (0.28 s). Every row, the released checkpoint's re-score included, reads DoTA-CAP at stride 3 and maps the scores back to native frames by interpolation; LaGoVAD's published 62.60 used every frame of DoTA's own list and is never placed beside these numbers.

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips; none of the seven traffic datasets is among them.

&#91;embedded content: LaGoVAD-KIM · baseline in outline, KIM in colour\]

Only KIM and its two training losses are new; the motion stream is the backbone change, and every other box is LaGoVAD as released. h̃\_t also feeds the pre-fusion head.

### 3.2 Novel component: the Kinematic Intention Module (KIM)

#### 3.2.1 What is taken from LAS-VAD, and why the earlier components were dropped

LAS-VAD's IAM argues that normal and abnormal behaviour can share appearance and differ in *how fast* it unfolds (taking an item vs stealing it). It projects the temporally modelled feature X\_f to a position feature, takes gated absolute first and second differences as velocity and acceleration, classifies the concatenation, weights the logits by cosine similarity to momentum class prototypes, and adds a cross-intention InfoNCE loss. In LAS-VAD's ablation on XD-Violence (fine-grained mAP), IAM alone lifts the average from 24.24 to 29.98, and from 33.25 to 36.89 on top of its other two parts.

Traffic makes the speed argument physical: a collision is an impulsive change of velocity, and a near-miss is a hard brake or swerve. KIM keeps IAM's placement (after temporal modelling) and its three-part feature, and changes every part that depends on fine-grained class labels or on LAS-VAD's host.

Four components were built and tested before this one. **MQR** (query readout of VideoMAE tokens) and **LSP** (latent next-step prediction on the VideoMAE vector) added nothing; both re-processed a global motion vector the host already had. **SA²** (motion-selected CLIP patches) could not point at agents better than a centre prior. The lessons that shape KIM:

1. **Add a quantity the host cannot compute,** not a re-reading of an input it already has: here, the per-channel magnitude of change of the contextualised feature (Section 3.2.6).
2. **Difference after temporal modelling, not on raw frames,** as LAS-VAD does; raw differences are noisy, and on dashcam data amplified CLIP-feature change raises false positives (WS-DoTA).
3. **Give the learned part its own signal.** KIM has its own top-k MIL term, so it learns even while the host ignores it, and it enters the host through a zero-initialised projection, so an unhelpful KIM costs nothing.
4. **Use labels that are certain.** Every step of a normal bag is normal; KIM's contrastive loss is anchored on those steps rather than on pseudo-labelled abnormal frames.
5. **Keep the differenced sequence continuous.** LaGoVAD's dynamic video synthesis splices up to five clips into one sequence and labels the inserted clip; its seams are jumps of the feature that coincide with the pseudo-labels, the exact shortcut a difference module would learn. KIM rows therefore train with synthesis off, and so does their control, so the comparison changes one thing.

| Part of IAM (LAS-VAD) | LAS-VAD | KIM (this thesis) | Reason |
| --- | --- | --- | --- |
| Input | X\_f after local Transformer + GCN, CLIP only | h\_t after the RoFormer (CLIP + VideoMAE) | Same role; differences of a smoothed, contextualised feature |
| Time step | One frame per snippet, any frame rate | Fixed 0.27–0.30 s per step | A velocity needs a fixed Δt |
| Splices and cuts | None in its host | Dynamic video synthesis off; differences masked at padding. Shot cuts are not detected: Nexar dashcam clips are continuous, and fixed-camera sets are test-only | LaGoVAD's dynamic video synthesis creates jumps that coincide with its pseudo-labels |
| Scale | Raw | Divided by the video's median first-difference magnitude | Dashcam ego-motion makes change normal; abrupt means large relative to the clip |
| Classes | C + 1 categories | Binary (normal / abnormal) | Nexar has no categories; categories stay with the text branch |
| Confidence | Cosine to the predicted class prototype, multiplied into the logits | (1 + cos) / 2 | A negative cosine would flip the prediction |
| Prototype update | Frames whose score exceeds α, all classes | Normal prototype from all valid steps of normal bags; abnormal prototype from confident top-k steps of abnormal bags | Normal labels are exact |
| Contrastive positive | The least similar same-class frame | The abnormal prototype | Under pseudo-labels the least similar frame is the likeliest label error |
| Contrastive negatives | Top-M most similar other-class frames | Top-M most similar steps of normal bags, plus the normal prototype | Hard negatives with exact labels |
| Output | Scores averaged into the fine-grained logits | Features returned to h\_t before text fusion; KIM's score used for training and diagnostics only | Keeps detection conditioned on the definition |
| Supervision | Only through the averaged MIL score | Its own top-k MIL term | A learned part without its own signal did not move in MQR |

LAS-VAD's anomaly-connected components (frame grouping by DFS on a thresholded T×T similarity graph) and its LLM-written attribute texts are not used. The first needs frame pseudo-labels and a graph threshold that LAS-VAD tunes on its own benchmarks; the second would bring an LLM into training.

#### 3.2.2 Kinematic features

KIM reads h\_t, the RoFormer output, and builds three 128-d features per step. A binary mask m\_t is 1 when neither step t−1 nor step t is padding, and 0 otherwise (m\_1 = 0). With dynamic video synthesis off, every training bag is one continuous window of one source video.

```latex
p_t = W_p\,\mathrm{LN}(h_t), \qquad \delta^{(1)}_t = m_t \, \frac{\lvert p_t - p_{t-1} \rvert}{\rho_V + \epsilon}, \qquad \rho_V = \operatorname{median}_{t:\,m_t = 1} \tfrac{1}{128}\lVert p_t - p_{t-1} \rVert_1
```

```latex
u_t = \sigma\big(\mathrm{Conv}_u(\delta^{(1)})_t\big) \odot \delta^{(1)}_t, \qquad \delta^{(2)}_t = m_t\, m_{t-1}\, \lvert u_t - u_{t-1} \rvert, \qquad a_t = \sigma\big(\mathrm{Conv}_a(\delta^{(2)})_t\big) \odot \delta^{(2)}_t
```

```latex
k_t = [\, p_t \,;\, u_t \,;\, a_t \,] \in \mathbb{R}^{384}, \qquad W_p \in \mathbb{R}^{128 \times 512}
```

- p\_t is the *position*: where the scene state is in a learned 128-d space. u\_t is the *velocity* and a\_t the *acceleration*, each a per-channel change magnitude passed through a sigmoid gate from a kernel-3 convolution, exactly as in IAM. The gate looks at three neighbouring steps, so an isolated one-step spike can be suppressed while a sustained change passes.
- ρ\_V is computed per video on valid steps and detached from the graph. It removes the clip's baseline rate of change, which on dashcam video is set by ego speed, so a\_t measures change that is abrupt *for this clip*. On static cameras ρ\_V is small and the scaling changes little.
- The mask comes from the padding mask the host already builds; no extra pass over the videos is needed.

#### 3.2.3 Kinematic score, prototypes and return path

Two layers score each step, and two prototypes μ\_0 (normal) and μ\_1 (abnormal) in R^384 weight the score by how typical the step is of its predicted class:

```latex
r_t = W_2\,\mathrm{ReLU}(W_1 k_t) \in \mathbb{R}^{2}, \qquad \hat y_t = \arg\max r_t, \qquad w_t = \tfrac{1}{2}\big(1 + \cos(k_t, \mu_{\hat y_t})\big), \qquad q_t = \mathrm{softmax}(w_t\, r_t)_{1}
```

The prototypes are buffers updated by momentum after each batch, with no gradient (β = 0.1, as in LAS-VAD; α = 0.5):

```latex
\mu_0 \leftarrow (1-\beta)\,\mu_0 + \beta\, \overline{\{k_t : t \in \text{normal videos}\}}, \qquad \mu_1 \leftarrow (1-\beta)\,\mu_1 + \beta\, \overline{\{k_t : t \in \mathcal{A}\}}
```

where 𝒜 holds the top-k steps by q\_t of each abnormal bag whose q\_t exceeds α, and the normal mean is over the valid steps of the normal bags in the batch. A low-confidence step (w\_t near 0) gets a flatter distribution, so it contributes little to the top-k.

KIM returns its features to the host through a second zero-initialised projection:

```latex
\tilde h_t = h_t + W_o\, k_t, \qquad W_o \in \mathbb{R}^{512 \times 384},\; W_o^{\text{init}} = 0
```

h̃\_t replaces h\_t everywhere downstream: the co-attention fusion and the pre-fusion head. Because the return path enters *before* fusion, the text definition still decides whether a kinematic change counts as anomalous. KIM's own score q\_t is not part of the test-time score; LAS-VAD's score-level averaging is ablation R11.

#### 3.2.4 Losses

KIM adds two training terms to LaGoVAD's losses.

**Kinematic MIL.** A top-k MIL term on KIM's own score, with the host's k over the bag's valid steps L. It averages the top-k probabilities q\_t; the host's own MIL term averages top-k logits, and the two are kept as written.

```latex
\mathcal{L}_{\text{kin}} = \mathrm{BCE}\Big( \tfrac{1}{k} \textstyle\sum_{t \in \operatorname{top}k(q)} q_t,\; y \Big), \qquad k = \max(\lfloor L/5 \rfloor, 1)
```

**Normal-anchored kinematic contrast (NKC).** Each anchor t ∈ 𝒜 is pulled to μ\_1 and pushed from its M = 16 most similar valid steps of the normal bags in the batch, 𝒩\_t, and from μ\_0. With cosine similarity s and temperature τ\_c = 0.1, in standard InfoNCE form (the positive is in the denominator, unlike LAS-VAD's printed Eq. 11):

```latex
\mathcal{L}_{\text{nkc}} = -\frac{1}{\lvert \mathcal{A} \rvert} \sum_{t \in \mathcal{A}} \log \frac{e^{s(k_t, \mu_1)/\tau_c}}{e^{s(k_t, \mu_1)/\tau_c} + e^{s(k_t, \mu_0)/\tau_c} + \sum_{n \in \mathcal{N}_t} e^{s(k_t, k_n)/\tau_c}}
```

Every negative is certain: it comes from a bag labelled normal. Only the anchors depend on KIM's own ranking, and only through the top-k of bags known to contain an anomaly.

```latex
\mathcal{L} = \mathcal{L}_{\text{LaGoVAD}} + \lambda_k\, \mathcal{L}_{\text{kin}} + \lambda_c\, \mathcal{L}_{\text{nkc}}
```

Here 𝓛\_LaGoVAD is the host's loss with synthesis off and without L\_neg: top-k MIL, multi-class MIL alignment, and L\_dvs on normal bags only. The first run fixes λ\_k = 1, λ\_c = 0.1, M = 16, α = 0.5 and β = 0.1; λ\_k ∈ {0.1, 0.5, 1} and λ\_c ∈ {0.05, 0.1, 0.5} can later be chosen once on Nexar's validation split. L\_nkc and the prototype updates start after a warm-up of 219 of the 1,749 training steps (the proposal's 5 of 40 epochs), so the prototypes are not built from random features.

#### 3.2.5 Initialisation and monitoring

- **Return path.** W\_o = 0, so at the start LaGoVAD-KIM computes the same scores as the motion-stream model; W\_o's gradient (∂L/∂h̃\_t)·k\_t^⊤ is non-zero because k\_t is non-zero, unlike the zero-query start that froze MQR.
- **KIM layers.** Xavier for W\_p, W\_1, W\_2 and the two convolutions; gate biases 0, so every gate starts at 0.5.
- **Prototypes.** Set to the first post-warm-up batch means, then updated by momentum.
- **Logged from the first step:** ‖W\_o‖, the gate means, the share of anchors in 𝒜 per batch, cos(μ\_0, μ\_1), and KIM's own video-level AUC on the held-out split. A KIM whose W\_o stays near zero or whose prototypes merge is visible at once.

#### 3.2.6 Why KIM should help on traffic video

Six pieces of evidence point the same way; the first two are about traffic, the next two about the host, the last two about the loss.

1. **Crashes are defined by abrupt kinematics.** The SHRP2 naturalistic driving study flags crash events by longitudinal deceleration of −0.65 g or more, lateral acceleration of ±0.75 g held 0.2 s, and jerk below −1.0 g/s; the deceleration trigger alone found 626 crashes and 3,950 near-crashes, the highest yield of any trigger ([SHRP2 report](https://vtechworks.lib.vt.edu/handle/10919/70850)). In real rear-end crashes braking starts 1.1–1.4 s before impact ([SAE 2011-01-0576](https://saemobilus.sae.org/articles/method-estimating-time-collision-braking-real-world-lead-vehicle-stopped-rear-end-crashes-use-pre-crash-system-design-2011-01-0576)). These precursors span 0.2–2 s, so a 0.3 s step with a 3-tap gate (about 0.9 s of support) resolves them with several samples; on Nexar the mean time from first visible danger to the event is 1.6 s.
2. **Change cues help on dashcam data, including under weak labels.** TTHF adds a frame-difference branch to a CLIP-aligned traffic detector and lifts DoTA AUC from 76.8 to 84.7, with its largest gains on ego-involved clips (85.5 vs 75.6 for STFE) ([TTHF](https://arxiv.org/abs/2401.03522)); this is supervised and in pixel space. Under weak supervision, WS-DoTA adds the absolute shifted difference of frozen CLIP features with a sigmoid of the features and lifts RTFM to 78.2 AUC, while a frequency-amplified difference was flat or worse because subtle motion in driving scenes raised false positives ([WS-DoTA](https://arxiv.org/abs/2408.05562), preprint). MOVAD's short multi-frame memory adds 7.9 AUC on DoTA, best at a 4-frame (0.4 s) window ([MOVAD](https://arxiv.org/abs/2302.10719)).
3. **The host cannot form the magnitude of change.** LaGoVAD's detection head is a single linear 1-D convolution followed by a sigmoid, so any temporal filter it learns is signed and linear: a change along an arbitrary direction of the 512-d feature adds up with opposite signs across channels and can cancel. Self-attention outputs a convex combination of value vectors, a smoothing; a difference needs two heads with opposite-signed output weights, which nothing in a top-k MIL loss asks for. KIM supplies the rectified per-channel magnitudes |Δ| and |Δ²| directly, the quantity IAM adds in LAS-VAD.
4. **Differencing belongs after temporal modelling.** LAS-VAD differences the feature after its local Transformer and GCN, and action-recognition modules that gain from differences (TDN: +5.7 top-1 on Something-Something V1; TEA: +2.4) difference trainable, contextualised features and smooth before subtracting ([TDN](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.pdf), [TEA](https://arxiv.org/abs/2004.01398)). WS-DoTA's amplified variant hurt. No paper compares the two placements, so R7 tests it.
5. **Contrast anchored on normal videos is the reliable form.** Weakly supervised methods that gain from prototype or contrastive terms build the reliable side from normal videos and pick abnormal samples only by top-k or score weighting: UR-DMU's two memories give 86.97 UCF-Crime AUC against 85.19 with the normal memory alone ([UR-DMU](https://arxiv.org/abs/2302.05160)); MGFN's magnitude contrast adds 2.13 AUC / 3.98 AP ([MGFN](https://arxiv.org/abs/2211.15098)); TPWNG's normality guidance adds 1.96 / 2.36 ([TPWNG](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.pdf)). LAS-VAD instead takes classes from its own argmax and mines the least similar positive, the pattern that drives confirmation bias in pseudo-labelling ([Arazo et al.](https://arxiv.org/abs/1908.02983)).
6. **Contrast mattered in LaGoVAD, and this host has none.** Removing LaGoVAD's hard-negative caption contrast drops its zero-shot TAD AUC from 89.56 to 83.26 and DoTA from 62.60 to 59.88 in its own paper. That loss needs captions and does not run on Nexar, so the host trains with no contrastive term at all; NKC is the only one, and it lives in the kinematic space.

**What is expected, and the main risks.** LAS-VAD's +5.74 average mAP bundles extra layers, gated differences, prototypes and a loss, with no internal ablation, on a non-traffic fine-grained metric; the gain on binary traffic AUC may be much smaller. Three risks follow from the evidence:

- **Ego-uninvolved anomalies.** KIM differences one global vector per step. A distant collision barely moves it, so gains should concentrate on ego-involved clips, where impacts shake the whole frame; TTHF shows the same pattern (DADA-2000 80.9 ego-involved vs 64.0 ego-uninvolved). H1 is stated accordingly.
- **Normal abrupt manoeuvres.** Hard braking at a red light, sharp turns and potholes are abrupt and normal. The text-conditioned return path and the hard normal negatives of NKC are the planned defence; the per-category and false-alarm analyses check it.
- **Redundancy with VideoMAE.** The VideoMAE vector already summarises 1.6 s of motion. The matched-capacity control (R-cap) and KIM on the VideoMAE sequence alone (R13) separate kinematics of the fused feature from re-reading motion the host has.

| Property | Motion-stream model (baseline) | + KIM |
| --- | --- | --- |
| Extra parameters | — | about 0.41M (W\_p 66K, two convolutions 99K, W\_1 and W\_2 50K, W\_o 197K) |
| Extra feature extraction | — | none; shot-cut detection once per video |
| Storage per step (fp16) | 1 KB (c\_t) + 0.75 KB (ṽ\_t) | unchanged |
| Extra losses | — | L\_kin, L\_nkc (training only) |
| Test-time output | step score | step score; KIM's q\_t and ‖a\_t‖ as a diagnostic curve |

### 3.3 Training strategy

Training uses bag-level labels only and one schedule, so the control and KIM differ only in KIM and its two added terms.

**Training data.** Every model trains on **Nexar only**. Positive bags are 20-step windows labelled from the alert and event times of Nexar's positive clips; negative bags come from its normal clips. Nothing is trained on TAD, MSAD, DoTA or DADA-2000; they are test sets only, so every result below is zero-shot.

| Protocol | Training set | Test sets | Anomaly text (hand-written) |
| --- | --- | --- | --- |
| P3, dashcam → dashcam | Nexar windows | DoTA-CAP-dev, DADA-2000 | Collision and near-miss definitions |
| P4, dashcam → fixed camera | Nexar windows (same models) | TAD test | Each test set's own definitions |

**Shortcut control.** Every model uses the same hand-written definitions. KIM adds a shortcut risk: if splice seams leaked into the differences, KIM could learn "discontinuity = anomaly". Dynamic video synthesis is therefore off in every KIM row and its control, so there are no seams; only padding is masked, and ablation R8 measures what happens without that mask.

**Stages.**

1. **Reuse the cached features.** The CLIP CLS features c\_t at the fixed step and the standardised VideoMAE V2-S vector already exist for Nexar and every test set; KIM computes everything else at run time.
2. **Train all models with one schedule.** Every row starts from the same random initialisation of LaGoVAD's trainable parts, with the same schedule; one seed first, three for the rows that are kept.
3. **Monitor** the quantities of Section 3.2.5.

**Settings.** As the motion-stream model: AdamW, batch 64, learning rate 5e-5 with 20 warm-up steps and a cosine schedule, 33 epochs of 53 steps (1,749 steps) on Nexar, weight decay 0.01, with dynamic video synthesis off. The first run fixes the KIM values of Section 3.2.4; λ\_k, λ\_c, M ∈ {8, 16, 32} and α ∈ {0.3, 0.5} can later be chosen once on Nexar's validation split (225 videos) by macro AUC, then frozen. No hyperparameter is chosen on a test list.

**Checkpoint selection.** Every model, baselines included, reports the last step.

**Compute.** KIM costs O(L·D) per video and no extraction, so training costs about the same as the motion-stream model on one 24 GB GPU.

### 3.4 Inference

1. **Sample** at the fixed step (stride 8 at 30 fps, 3 at 10 fps, 7 at 25 fps).
2. **Encode** each step with frozen CLIP (c\_t) and the causal 16-frame clip that ends on it with frozen VideoMAE V2-S (ṽ\_t), standardised with Nexar's training statistics; fuse x\_t = c\_t + W\_u (c ṽ\_t). Both are read from the existing caches.
3. **Contextualise** with the RoFormer into h\_t.
4. **Kinematics.** Compute p\_t, u\_t, a\_t with the padding mask and the video's ρ\_V; return h̃\_t = h\_t + W\_o k\_t.
5. **Encode the definition** with LaGoVAD's soft prompts and the frozen CLIP text encoder.
6. **Score** with co-attention fusion and the blended heads. Each step's score is assigned to the frames it stands for.
7. **Report** KIM's q\_t and ‖a\_t‖ per step as a diagnostic curve for the onset analysis.

**Score post-processing.** DoTA-CAP contains only abnormal clips, so its scores are min-max normalised per clip before pooling, following LaGoVAD's `offline_dota_eval.py`. DADA-2000 is read on whole source videos that hold normal driving before the accident, and TAD on its test list; both use raw scores. Macro AUC is computed over the videos that contain both classes.

**Cost and causality.** One VideoMAE ViT-S clip (about 57 GFLOPs) runs per step; KIM adds a negligible amount. The model is offline: ρ\_V uses the whole video, and the RoFormer looks both ways; KIM's differences and the VideoMAE clip themselves are backward-looking.

## 4. Experimental plan

The plan trains on Nexar with bag-level labels only, checks KIM's premise with an inference-only pilot on the trained motion-stream model before any KIM training, and reads every model zero-shot on the same test lists as the released LaGoVAD checkpoint.

### 4.1 Dataset preparation

| Dataset | View | Size and split | Labels | Role | Source |
| --- | --- | --- | --- | --- | --- |
| TAD | Fixed CCTV, some first-person | LaGoVAD's 100-video test list (60 with an anomaly); 7 categories | Anomaly interval (test) | P4 zero-shot test only | LaGoVAD's annotation; native fps unknown |
| Nexar | Dashcam | 1,500 labelled clips (750 positive + 750 normal), about 40 s each; train / validation split (225 validation videos) | Video labels; alert and event times, used to cut 20-step positive windows | Training (all rows); validation for any later hyperparameter choice | Hugging Face |
| DoTA | Dashcam | 4,677 clips, 10 fps, abnormal only. DoTA's own pixels are gone: 1,129 of the 1,397 validation clips survive as DoTA-CAP (CAP-DATA copies), split into dev 569 / eval 560 | Anomaly window; 9 categories × ego-involved / ego-uninvolved | P3 zero-shot test (dev); Step 0 pilot; onset analysis | GitHub (MoonBlvd/Detection-of-Traffic-Anomaly) + CAP-DATA frames |
| DADA-2000 | Dashcam | Original release; 370 whole test source videos with normal driving before the accident | Accident intervals; ego / non-ego | P3 zero-shot test; Step 0 pilot; onset analysis | DADA-2000 original annotation |

**Weights.** In use and pinned: CLIP ViT-B/16, VideoMAE V2-S (`vit_s_k710_dl_from_giant`) and LaGoVAD's `best.ckpt`, used only as a reference row. Optional, for context rows only: VadCLIP, PEL4VAD and DSANet checkpoints for zero-shot re-scoring, and MOVAD and Simple-TAD DoTA and DADA-2000 checkpoints as frame-supervised context; none has been test-downloaded. LAS-VAD has no public code, so IAM is reimplemented from the paper.

### 4.2 Dataset processing

**Nothing is extracted for this proposal.** The feature caches built for the motion-stream model serve the control, KIM and every ablation, because KIM computes its features from h\_t at run time. For the record, those caches were built as follows:

1. **Sample at the fixed step.** CLIP stride 8 at 30 fps (Nexar, DADA-2000), 3 at 10 fps (DoTA-CAP); TAD's native fps is unknown, so its effective step is reported as unknown and stride 8 is used.
2. **CLIP CLS features** from a 224×224 resize without centre crop, the transform every cache of this thesis uses.
3. **VideoMAE V2-S vector.** For each step the causal 16 frames that end on it (every third frame at about 30 fps, every frame on 10 fps DoTA-CAP, frame 0 repeated before the start), squash-resized to 224×224; the 8×14×14 last-layer tokens are averaged and passed through the encoder's final normalisation; c, m\_u and σ\_u are fitted on Nexar's training split and applied to every set.
4. **Labels.** Training uses bag labels from Nexar's alert and event times. Test labels are frame-level: DoTA-CAP anomaly windows, DADA-2000 intervals, TAD intervals.
5. **Analysis labels.** For DoTA-CAP keep `ego_involve`, the category and the anomaly start; for DADA-2000 the ego flag and the accident start.
6. **Fixed test lists.** DoTA-CAP-dev (569 clips); DADA-2000's 370 test source videos; LaGoVAD's 100-video TAD list.

**Leakage checks.**

- No test set is trained on; Nexar is the only training source.
- DoTA-CAP-eval (560 clips) stays unread, held for a final read.
- The Step 0 pilot reads the three test sets with the already-trained motion-stream model; it fits nothing, and every KIM read-out states that the pilot saw these sets first.
- The released LaGoVAD checkpoint is re-scored on every test list under the same protocol before anything is compared with it.

### 4.3 Pilot and baseline running

**Step 0: pilot diagnostic (inference only, on the trained motion-stream model B2, on DoTA-CAP-dev, DADA-2000's 370 test sources and TAD's test list, with Nexar's validation split printed beside).** It asks whether the magnitude of change of the RoFormer output already rises at anomalies before anything is trained for it. Each candidate is used directly as a step score, with each set's own post-processing. These are KIM's test sets, so the pass criteria below are frozen before the pilot runs, the pilot fits nothing, and every KIM result states that the pilot read these sets first.

| Check | Question | Pass criterion (proposed) | If it fails |
| --- | --- | --- | --- |
| K1: change signal | Per-video AUC of ‖Δh\_t‖ / ρ\_V and ‖Δ²h\_t‖ / ρ\_V, against the same quantities on c\_t and ṽ\_t; by ego flag | ‖Δ²h‖ clearly above 0.5 (≥ 0.60) at least on ego-involved clips | If c\_t or x\_t pass and h\_t does not, KIM's default input moves there (R7) |
| K2: complementarity | Does a rank average of B2's score with ‖Δ²h‖ beat B2 alone? | Paired gain with CI above zero | KIM's premise is weak; the ladder still runs, but expectations are lowered in writing |
| K3: edges | Share of the top 5% ‖Δh‖ steps that fall on the first or last two valid steps of a video, against the share expected by chance | Not above chance | Mask the edge steps in KIM (a fixed rule, written before training) |
| K4: onset | Median lag between the peak of ‖Δ²h‖ and the annotated onset on DoTA-CAP-dev and DADA-2000 | Within ±0.5 s for most clips | KIM's onset hypothesis (H6) is dropped from the claims |

The pilot costs well under one GPU-day, since all features are cached. A failure does not stop the plan; it decides KIM's input and how strongly the hypotheses are stated before training.

**Group A: reproduce LaGoVAD and the motion stream.**

| Row | Model | Training | Purpose |
| --- | --- | --- | --- |
| B0 | LaGoVAD, released checkpoint | PreVAD (by its authors) | Re-score on DoTA-CAP-dev, DADA-2000 and TAD under this thesis's protocols (DoTA-CAP-dev 0.6418 macro, DADA-2000 0.6202, TAD 0.7579); TAD's published 89.56 was not reproduced (0.7912 micro) |
| B1 | LaGoVAD, retrained | Nexar | Fair baseline with the thesis data, features and schedule; separates domain-matched data from motion |
| B2 | LaGoVAD + motion stream (W\_u), with dynamic video synthesis | Nexar | The motion-stream model already trained; used by the Step 0 pilot |
| R0 | B2 with dynamic video synthesis off | Nexar | KIM's control: identical to every KIM row except KIM; B2 printed beside it |

**Group B (optional): published methods with released weights, re-scored zero-shot.** VadCLIP and DSANet (UCF-Crime and XD-Violence checkpoints, same CLIP features); MOVAD and Simple-TAD (DoTA and DADA-2000, frame-supervised context only).

**Group C: same-feature competitor on the same host.** **LAS-VAD's IAM**, reimplemented as published on the same host and features (row R4 with LAS-VAD's own class assignment, positive choice and score-level averaging), so KIM is compared with the method it adapts rather than only with its own ablations. Methods without released weights are quoted from their papers only, as context: UMIL and WSAL on TAD; WS-DoTA; and the frame-supervised TTHF, PromptTAD and ISCRTAD.

### 4.4 Evaluation protocols

**Metrics.**

- **Macro AUC (endpoint):** frame-level ROC AUC per video, averaged over the videos that contain both classes; it measures timing inside a video.
- **Micro AUC and AP:** pooled over all test frames (DoTA-CAP after per-clip min-max, DADA-2000 and TAD raw), with the positive-frame share printed beside AP.
- **Clip-level AUC on TAD:** TAD's micro AUC mostly measures clip classification, so it is printed beside, never alone.
- **Breakdowns:** DoTA-CAP-dev ego-involved and ego-uninvolved clips and per category; DADA-2000 ego and non-ego.
- **Position ruler:** the macro AUC of a score that sees only the step's position in the video, printed beside every macro number, so a model that has learned "accidents sit mid-clip" is not read as detection.
- **Onset error:** for each abnormal DoTA-CAP-dev and DADA-2000 clip, the time from the annotated onset to the first step whose per-video normalised score reaches 0.5; reported as the median absolute error in seconds and the share of clips within ±0.5 s, for the final score and for KIM's q\_t and ‖a\_t‖.
- **False alarms on normal driving:** on Nexar's normal validation clips, the share of steps above the threshold that gives 90% recall on abnormal validation clips; checks that KIM does not fire on normal hard braking and turns.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P3 dashcam zero-shot | Nexar → DoTA-CAP-dev (stride 3, scores interpolated to native frames, per-clip min-max) and DADA-2000 (stride 8, whole source video, raw) | Macro (endpoint), micro, AP; by ego flag and category; onset error; false alarms on Nexar validation | Released LaGoVAD re-scored the same way; optional VadCLIP / DSANet re-scores; frame-supervised rows as context |
| P4 fixed-camera zero-shot | Nexar → TAD (LaGoVAD's 100-video list, stride 8, raw) | Macro (endpoint), micro, AP, clip-level AUC | Released LaGoVAD re-scored the same way; published in-domain rows as context only |

**Published in-domain rows (context only).** The thesis trains nothing on TAD or MSAD, so these rows are not ranked against it; they show what in-domain training reaches.

| Method | Venue | TAD AUC\_O | TAD AUC\_A | MSAD AUC | MSAD AP | MSAD AUC\_A | MSAD AP\_A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WSAL | IEEE TIP 2021 | 89.64 | 61.66 |  |  |  |  |
| UMIL | CVPR 2023 | 92.93 | 65.82 |  |  |  |  |
| GS-MoE | ICCV 2025 |  |  | 87.72 | 68.26 | 69.54 | 76.68 |
| π-VAD | CVPR 2025 |  |  | 88.68 | 71.26 | 71.25 | 77.86 |
| VadCLIP framework + CLIP | Alert-CLIP's baseline |  |  | 87.45 | 71.09 |  |  |
| VadCLIP framework + Alert-CLIP | CVPR 2026 |  |  | 89.63 | 78.24 |  |  |

TAD AUC\_O-only rows add context: MIST 89.2, RTFM 89.6, CUPL 91.66, MultiDomain (TAD→TAD) 90.71. UMIL drops to 90.71 AUC\_O with a frozen backbone, the fairer reference for frozen features.

**DoTA rows and their protocols.** Only rows under the same per-video normalisation are ranked against the thesis.

| Method | Supervision / training data | DoTA protocol | AUC | Comparable to thesis? |
| --- | --- | --- | --- | --- |
| Thesis motion-stream model (B2) | Bag-level, Nexar, zero-shot | Per-clip min-max, DoTA-CAP-dev 569 clips, stride 3 | 0.81 macro / 0.73 micro | Reference |
| LaGoVAD, released checkpoint re-scored | Video-level, PreVAD, zero-shot | Same as the thesis | 0.64 macro / 0.61 micro | Yes |
| LaGoVAD, as published | Video-level, PreVAD, zero-shot | Per-video normalised, DoTA's 1,402-clip list, every frame | 62.60 | No: different clip list and stride |
| VadCLIP | Weak, UCF-Crime / XD-Violence / PreVAD, zero-shot | As LaGoVAD published | 50.93 / 49.00 / 50.59 | No, unless re-scored |
| PEL (I3D) | Weak, UCF-Crime / XD-Violence, zero-shot | As LaGoVAD published | 53.05 / 44.97 | No |
| WS-DoTA (RTFM + CLIP differences) | Weak, D²-City + DoTA in-domain | \~1,031 clips, normalisation not stated | 78.2 | Context only |
| AnoPred; DoTA ensemble; MAMTCF | Unsupervised, DoTA in-domain | Pooled raw scores | 67.5; 73.0; 76.6 | Context only |
| MOVAD, PromptTAD, TTHF, ISCRTAD | Frame labels, DoTA in-domain | Pooled; TTHF post-processed | 82.2, 83.9, 84.7, 86.2 | Upper reference only |
| Simple-TAD DAPT-VideoMAE-S / -L | Frame labels, fine-tuned | Pooled | 86.4 / 88.4 | Upper reference only |

**Reporting rules.** One seed first, marked preliminary; every kept row then reports a mean ± standard deviation over 3 seeds. Subsets carry 95% bootstrap confidence intervals over videos, and KIM against the motion-stream model is tested with a paired bootstrap. All models use the same test lists, frame-to-score mapping and normalisation, and every table names each row's training set and protocol.

### 4.5 Ablation studies

The ladder adds KIM's parts one at a time on top of the control R0 (the motion-stream model with dynamic video synthesis off), then tests its traffic adaptations. Every row trains on Nexar with bag-level labels and synthesis off, and reports P3 (DoTA-CAP-dev macro / micro by ego flag; DADA-2000 macro; onset error) and P4 (TAD macro). Expected outcomes are priors written down in advance.

| Row | Variant | Isolates | Expected if KIM's premise holds |
| --- | --- | --- | --- |
| R0 | Motion-stream model, dynamic video synthesis off | Reference | At or somewhat below B2's 0.81 on DoTA-CAP-dev (synthesis gave the host its frame pseudo-labels) |
| R1 | + KIM features through W\_o, no KIM losses | The feature path alone | Small or no gain; W\_o may stay near zero, as raw differences did |
| R2 | + L\_kin (q\_t = softmax(r\_t), no prototypes) | KIM's own signal | Gain over R1, mostly on ego-involved clips |
| R3 | + prototype confidence weighting | LAS-VAD's confidence step | About R2; better onset precision |
| R4 | + LAS-VAD's contrast (self-assigned classes, least similar positive, other-class negatives) | IAM as published, on this host | At or slightly above R2; less stable across seeds |
| **R5** | **+ normal-anchored contrast NKC instead (full LaGoVAD-KIM)** | The label-safe loss | **Best overall; lower false alarms than R4** |
| R-cap | R5 with k\_t replaced by an MLP of h\_t with the same width and losses, no differences | Capacity and extra loss without kinematics | Below R5 |
| R6 | Order: p only / p + u / p + u + a | First vs second order | p + u + a best; p only ≈ R-cap |
| R7 | Input: c\_t (CLIP only) / x\_t (before the RoFormer) / h\_t (default) | Placement | h\_t ≥ x\_t > c\_t, unless Step 0 says otherwise |
| R8 | No padding mask | The edge shortcut | Worse zero-shot, KIM evidence piling up at the last valid step |
| R9 | No per-video scale ρ\_V | Ego-speed normalisation | Worse on dashcam sets, about equal on fixed cameras |
| R10 | Signed differences instead of absolute values | IAM's absolute value | About R5; signed may help approach vs recede |
| R11 | KIM's score averaged into the test-time score, as in LAS-VAD | Score-level vs feature-level return | Similar AUC, worse drift@5 and definition sensitivity |
| R12 | Stop-gradient from KIM into the RoFormer and W\_u | Whether KIM's losses reshape the host | Slightly below R5 |
| R13 | KIM on the ṽ\_t sequence only | Kinematics of the motion vector (the MQR / LSP lesson) | Below R5 |
| R14 | KIM on CLIP-only LaGoVAD (no motion stream), synthesis off | Complementarity with VideoMAE | Larger gain over B1 (synthesis off) than R5 over R0 |
| R15 | R5 with dynamic video synthesis on and seams masked | Whether KIM survives the host's own synthesis | About R5 if the seam mask works; recorded, not required |

H1 holds if R5 beats R0 with larger gains on ego-involved clips; H2 if R5 beats R-cap and R6's position-only row; H3 if p + u + a is best in R6 and h\_t is at least as good as x\_t in R7; H4 if R5 beats R4 and R4 is not below R2; H5 if R5 is no less definition-sensitive than R0 and R11 is less; H6 if KIM's onset error beats R0's.

**Side ablations within the component.** M ∈ {8, 16, 32}; α ∈ {0.3, 0.5}; β ∈ {0.05, 0.1, 0.3}; gate kernel 3 vs 5; difference spacing ±1 vs ±2 steps; NKC with and without μ\_0 as a negative; warm-up 0, 219 or 438 steps.

**Analyses.**

1. **Where the gain lands.** DoTA-CAP-dev ego-involved vs ego-uninvolved crossed with the nine categories; DADA-2000 ego vs non-ego; TAD Accident, PedestrianOnRoad and the remaining categories pooled; dashcam sets against fixed-camera TAD.
2. **Kinematic curves.** q\_t, ‖u\_t‖ and ‖a\_t‖ plotted against annotated onsets for successes and failures, including normal hard-braking clips.
3. **Learned parts.** ‖W\_o‖ over training, gate means, the size of 𝒜 per batch and cos(μ\_0, μ\_1); whether the prototypes separate on held-out data.
4. **Language guidance.** Score change of accident frames when the definition includes or excludes accidents; TAD's 7 categories with their own definitions.
5. **Cost.** Parameters and measured frames per second for R0 and R5.

**Budget.** Every row trains once on Nexar and is read on all three test sets. The first pass runs R0, R2 and R5 with one seed; if R5 is not worse than R0, R-cap, R4, R6, R7 and R8 follow with one seed, then three seeds for R0, R2, R5 and R-cap. Rows R1, R3 and R9–R15 and the side ablations follow as budget allows. If R5 costs on both dashcam test sets in the first pass, the line closes with no further variants. Every row shares the existing feature caches and costs about one baseline training run.

## References

**Baseline, backbone and component**

- Liu et al., [Language-Guided Open-World Video Anomaly Detection under Weak Supervision (LaGoVAD)](https://arxiv.org/abs/2503.13160), ICLR 2026; [code, checkpoint, PreVAD annotations](https://github.com/Kamino666/LaGoVAD-PreVAD)
- Wang and Zhao, [Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning (LAS-VAD)](https://arxiv.org/abs/2603.00550), CVPR 2026 (source of KIM; no public code)
- Wu et al., VadCLIP, AAAI 2024; [code](https://github.com/nwpu-zxr/VadCLIP)
- Wang et al., [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727), CVPR 2023 (`vit_s_k710_dl_from_giant`)
- Orlova et al., [Simplifying Traffic Anomaly Detection with Video Foundation Models (Simple-TAD)](https://arxiv.org/abs/2507.09338), ICCV 2025 Workshops; [code and DAPT encoders](https://github.com/tue-mps/simple-tad)
- Radford et al., Learning Transferable Visual Models from Natural Language Supervision (CLIP), ICML 2021
- Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, Neurocomputing 2024

**Temporal change and kinematics**

- [TTHF: Text-Driven Traffic Anomaly Detection with Temporal High-Frequency Modeling in Driving Videos](https://arxiv.org/abs/2401.03522), IEEE 2024 (journal to be confirmed)
- Rossi et al., [MOVAD: Memory-augmented Online Video Anomaly Detection](https://arxiv.org/abs/2302.10719), ICASSP 2024; [code](https://github.com/IMPLabUniPr/movad)
- [What Matters in Autonomous Driving Anomaly Detection: A Weakly Supervised Horizon (WS-DoTA)](https://arxiv.org/abs/2408.05562), arXiv preprint 2024; [split files](https://github.com/ut21/WSAD-Driving)
- Wang et al., [TDN: Temporal Difference Networks for Efficient Action Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.pdf), CVPR 2021
- Li et al., [TEA: Temporal Excitation and Aggregation for Action Recognition](https://arxiv.org/abs/2004.01398), CVPR 2020
- Wu et al., [STPrompt](https://arxiv.org/abs/2408.05905), ACM MM 2024 (source of the dropped SA²)
- Ijjina et al., [Computer Vision-based Accident Detection in Traffic Surveillance](https://arxiv.org/abs/1911.10037), ICCCNT 2019
- Hankey et al., [SHRP 2 Naturalistic Driving Study: Crash and Near-Crash Baseline Report](https://vtechworks.lib.vt.edu/handle/10919/70850), VTTI 2016
- [Method for Estimating Time to Collision at Braking in Real-World, Lead Vehicle Stopped Rear-End Crashes](https://saemobilus.sae.org/articles/method-estimating-time-collision-braking-real-world-lead-vehicle-stopped-rear-end-crashes-use-pre-crash-system-design-2011-01-0576), SAE 2011-01-0576

**Prototypes, memories and contrast under weak supervision**

- Zhou et al., [Dual Memory Units with Uncertainty Regulation (UR-DMU)](https://arxiv.org/abs/2302.05160), AAAI 2023
- Chen et al., [MGFN: Magnitude-Contrastive Glance-and-Focus Network](https://arxiv.org/abs/2211.15098), AAAI 2023
- Yang et al., [Text Prompt with Normality Guidance (TPWNG)](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.pdf), CVPR 2024
- Zhou et al., [BN-WVAD](https://arxiv.org/abs/2311.15367), arXiv preprint
- Zhang et al., [Exploiting Completeness and Uncertainty of Pseudo Labels](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.pdf), CVPR 2023
- Arazo et al., [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://arxiv.org/abs/1908.02983), IJCNN 2020
- [DSANet](https://arxiv.org/abs/2511.10334), AAAI 2026; [code](https://github.com/lessiYin/DSANet)
- Pu et al., PEL4VAD, IEEE TIP 2024; [code and checkpoints](https://github.com/yujiangpu20/PEL4VAD)
- Tian et al., RTFM, ICCV 2021

**Dropped components**

- Psomas et al., [Efficient Probing](https://arxiv.org/abs/2506.10178), ICLR 2026 (basis of MQR)
- Caselli et al., [FLaRA](https://arxiv.org/abs/2606.14380), arXiv 2606.14380 (basis of LSP)

**Weakly supervised and open-world VAD (comparison rows)**

- Lv et al., [UMIL](https://arxiv.org/abs/2303.12369), CVPR 2023; [TAD split files](https://github.com/ktr-hubrt/UMIL)
- Lv et al., Localizing Anomalies from Weakly-Labeled Videos (WSAL, TAD dataset), IEEE TIP 2021
- [π-VAD](https://arxiv.org/abs/2505.13123), CVPR 2025; [GS-MoE](https://arxiv.org/abs/2508.06318), ICCV 2025
- Alert-CLIP, CVPR 2026 (project file)
- [MultiDomain](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf), NeurIPS 2024

**Traffic anomaly detection**

- Qiu et al., [PromptTAD](https://doi.org/10.1109/LRA.2025.3572428), IEEE RA-L 2025
- [MAMTCF](https://arxiv.org/abs/2307.14575), arXiv 2307.14575; Liu et al., AnoPred, CVPR 2018
- Liang et al., ISCRTAD, IEEE T-ITS 2025

**Datasets and tools**

- Yao et al., [DoTA](https://arxiv.org/abs/2004.03044), IEEE TPAMI 2022; [annotations](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
- Zhu et al., [MSAD](https://msad-dataset.github.io/), NeurIPS 2024 Datasets and Benchmarks
- Bao et al., [CCD](https://github.com/Cogito2012/CarCrashDataset), ACM MM 2020
- Moura et al., [Nexar Dashcam Collision Prediction Dataset and Challenge](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf), CVPR 2025 Workshops
- Fang et al., DADA-2000, IEEE T-ITS 2021; Fang et al., [MM-AU](https://arxiv.org/abs/2403.00436), CVPR 2024
