# Thesis proposal: Deep learning-based traffic video anomaly detection (LaGoVAD-KIM)

Nov 6, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-KIM**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. The baseline is LaGoVAD (ICLR 2026) with LaGoVAD's released text setup, plus a frozen VideoMAE V2-S motion stream fused into the CLIP branch through a learnable, zero-initialised projection W\_u. Trained on Nexar and CCD dashcam video with video-level labels, this motion-stream model already reaches **0.81 macro / 0.75 micro per-video min-max AUC zero-shot on DoTA**, against 62.60 for the released LaGoVAD.

The one novel component is the **Kinematic Intention Module (KIM)**, a traffic adaptation of the Intention Awareness Module (IAM) of LAS-VAD (CVPR 2026). KIM reads the temporally contextualised features that leave LaGoVAD's RoFormer, derives a position, a velocity and an acceleration feature per step, and returns them to the host before text fusion. It is trained with its own top-k MIL term and a normal-anchored kinematic contrastive loss built on two momentum prototypes. The claim under test: a collision is an abrupt change in the scene's state, and the *magnitude* of first- and second-order change of the contextualised feature is evidence that LaGoVAD's linear 1-D convolution heads cannot compute from the features themselves.

This version drops three parts of the previous version. **SA²** was trained and could not locate agents better than a centre prior, so the CLIP branch returns to one global vector per step and spatial maps leave the scope. The **8-token learnable prefix** is replaced by LaGoVAD's own 32 soft-prompt tokens. LAS-VAD's **anomaly-connected components (ACC)** are not borrowed: they add a T×T graph, a DFS and frame pseudo-labels, and nothing in LAS-VAD shows they help on traffic video.

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event is happening in it. Abnormal events include collisions, loss of control, illegal turns, wrong-way driving, pedestrians on the road and road spills, the categories used by TAD and DoTA.

Two camera settings define the problem space:

- **Fixed camera (non-ego).** TAD and the traffic scenes of MSAD. The camera is static; the anomalous agents are often small and far away.
- **Dashcam (ego-centric).** DoTA, DADA-2000, CCD and Nexar. The camera moves with the car. DoTA marks each clip as ego-involved (the ego car takes part) or ego-uninvolved (it only observes).

The thesis follows LaGoVAD's weakly supervised, language-guided formulation. Training uses one label y\_i ∈ {0, 1} per video, plus a category and a hand-written caption template. At test time the model receives a video V and a text definition Z of what counts as anomalous, and returns a score per sampled step:

```latex
\Phi_\theta : (V, Z) \;\mapsto\; \{ s_t \}_{t=1}^{N}, \qquad s_t \in [0,1]
```

The video labels supervise training through multiple-instance learning (MIL): the mean of the top-k step scores of a video must match its label. This version outputs no spatial map.

Four properties make traffic video harder than generic surveillance:

1. **Anomalies are motion events, and many are abrupt.** A collision is defined by how trajectories meet and by a sudden change of speed or heading, not by what one frame looks like. LaGoVAD encodes single RGB frames with CLIP; it reaches 89.56 AUC on TAD but 62.60 on dashcam DoTA (part of this gap is DoTA's within-clip protocol).
2. **The agents that matter are small.** A distant agent of about 40×30 px in a 1280×720 frame shrinks to about 7×9 px at 224×224 input. The SA² experiment showed that weakly supervised patch selection does not recover them on this data, so the thesis now targets *when* the scene changes rather than *where*.
3. **The camera moves.** In dashcam video, ego-motion dominates frame-to-frame change, so a large change is normal; what is abnormal is a change that is sudden relative to the clip's own motion.
4. **Labels are scarce.** Frame labels are expensive and anomalies are rare, so the model learns from video labels.

The scope is offline detection: the whole video is available, and the question is whether an anomaly is happening, not whether one will happen (anticipation, as in Nexar). No LLM or multimodal LLM runs at training or test time; a frozen CLIP text encoder is allowed. All training text is hand-written.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from cheap video-level labels, sees motion, reacts to *abrupt* change of the scene, and keeps LaGoVAD's ability to take the anomaly definition as text, on one academic GPU without any LLM.

The thesis builds this in two steps, matching the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** A frozen VideoMAE V2-S encoder feeds one 384-d vector per step into LaGoVAD through a zero-initialised projection W\_u. Preliminary result: zero-shot DoTA **0.81 macro / 0.75 micro**. Three attempts to add more (MQR, LSP, SA²) did not help.
- **Novel component (KIM).** On the RoFormer output, KIM computes position, velocity and acceleration features, returns them to the host through a second zero-initialised projection, and trains them with a kinematic MIL term and a contrastive loss anchored on normal videos.

A second purpose is a fair evaluation: published traffic numbers are mostly single-metric AUC on inconsistent splits. The thesis reports at least two metrics against at least two published methods on the same fixed-camera traffic data, plus zero-shot and dashcam results, with each row's protocol labelled.

### 1.3 Research aims

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint and retrain it on the thesis data, with AUC, AP and AUC on abnormal videos (AUC\_A) on TAD, MSAD, DoTA and DADA-2000.
2. **Document the motion stream and the negative results.** Report the VideoMAE stream's gain over CLIP-only LaGoVAD, and the MQR, LSP and SA² results that fixed the backbone.
3. **Design and train KIM** with video-level labels only.
4. **Measure temporal precision.** Score how close KIM's evidence lands to the event onset, with CCD's accident onset and DoTA's anomaly start.
5. **Explain the gains.** Break results down by DoTA category, ego flag and camera setting; test KIM's placement, its derivative order and its loss; and check that language guidance and MSAD's non-traffic scenes do not regress.

"Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval over test videos excludes zero.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. KIM improves on the motion-stream model, most on ego-involved collisions | DoTA macro / micro AUC overall, by category and ego flag; CCD AUC; TAD AUC\_O / AUC\_A (rows R0, R5) | R5 > R0 overall, with larger gains on ego-involved than on ego-uninvolved clips |
| H2. The gain comes from the magnitude of change, not from extra capacity | Matched-capacity MLP on the RoFormer output (R-cap); KIM with position only (R6a) | R5 > R-cap and R5 > R6a |
| H3. Second-order change matters; contextualised features are the right input | Order ladder p / p+v / p+v+a (R6); placement on c\_t, x\_t or h\_t (R7) | p+v+a best; h\_t ≥ x\_t > c\_t |
| H4. A contrastive loss anchored on normal videos beats LAS-VAD's pseudo-label version | R2 (MIL only), R4 (LAS-VAD contrastive), R5 (normal-anchored) | R5 > R4 ≥ R2 |
| H5. Language guidance and surveillance scenes are kept | MSAD drift@5; TAD per-category definitions; MSAD non-traffic scenes; score-level fusion variant (R11) | R5 no worse than R0 on drift@5 and non-traffic scenes; R11 worse on drift@5 |
| H6. KIM's evidence is temporally precise | Onset error (s) of KIM's acceleration peak on CCD and DoTA; AUC\_A | Median onset error below that of R0's score peak |

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
| Host detector: CLIP stream, RoFormer, soft prompts, co-attention, heads and losses | LaGoVAD (ICLR 2026), on VadCLIP's frozen-CLIP + MIL design | Architecture, 32 soft-prompt tokens (16 + 16), losses and schedule as released | Last-epoch checkpoints; hand-written caption templates; fixed-time sampling |
| Motion stream (backbone change) | VideoMAE V2 (CVPR 2023), chosen on the evidence of Simple-TAD | Frozen distilled ViT-S and a 16-frame clip input | One normalised 384-d vector per step, fused through a zero-initialised W\_u |
| KIM features and placement (novel component) | LAS-VAD's IAM (CVPR 2026) | Position, gated velocity and acceleration after temporal modelling | Seam and cut masks, per-video scale, fixed time step, return path before text fusion |
| KIM score, prototypes and loss | LAS-VAD's IAM; normal-anchored contrast as in UR-DMU, MGFN, TPWNG | Confidence-weighted logits, momentum prototypes, InfoNCE | Binary classes, (1 + cos) / 2 confidence, normal prototype from normal videos, prototype positive, exact hard negatives, own MIL term |
| Evidence that change cues help on traffic | TTHF, MOVAD, WS-DoTA | — | Motivates KIM and the ego-involved analysis |

#### 2.3.1 LaGoVAD (ICLR 2026): the baseline

Liu et al. define open-world VAD as a mapping Φ: (V, Z) → frame scores, with the definition Z given as text at test time. Frozen CLIP ViT-B/16 frame embeddings pass through a 2-layer RoFormer and are fused with the CLIP text encoding of Z by co-attention. Training uses video labels on PreVAD (35,279 videos) with MIL and MIL-alignment losses, dynamic video synthesis and a caption contrastive loss with hard negatives.

- **Strengths.** The best zero-shot traffic results in its comparison: 89.56 AUC on TAD, 62.60 on DoTA, 90.41 on MSAD. It stays robust when the definition changes (MSAD drift@5: 85.6 AUC / 40.1 AP), runs no LLM, trains on one RTX 4090, and its code and checkpoint are public.
- **Weaknesses.** One global CLIP embedding per frame: no motion and no explicit rate of change. Its head is a single linear convolution. Its released configuration selects checkpoints on test scores.
- **Taken into this thesis.** The whole detector, its text setup included.

#### 2.3.2 VideoMAE V2 (CVPR 2023) and Simple-TAD (ICCV 2025 Workshops): the motion stream

Wang et al. scale masked video autoencoding and distil a ViT-g into small ViT-S and ViT-B models. Orlova et al. show that a plain video ViT with one linear head, fine-tuned with frame labels, beats specialised traffic detectors on DoTA (up to 88.4 AUC) and DADA-2000.

- **Strengths.** In Simple-TAD's comparison the distilled ViT-S is the strongest small encoder for traffic anomalies (AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000). In this thesis's pilot its mean vector lifts zero-shot DoTA to 0.81 / 0.75.
- **Weaknesses.** No driving video in pre-training; features are not text-aligned; ego-uninvolved DoTA clips lag ego-involved ones by 5–8 AUC even with frame labels. MQR and LSP added nothing beyond the mean vector.
- **Taken into this thesis.** The frozen ViT-S, read as one normalised vector per step.

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
| WSAL (IEEE TIP 2021) | High-order context encoding; introduced TAD | 89.64 AUC\_O / 61.66 AUC\_A on TAD | Hand-crafted cues | P1 row |
| UMIL (CVPR 2023) | Confident / ambiguous snippet split; X-CLIP fine-tuned | 92.93 AUC\_O / 65.82 AUC\_A on TAD | Needs backbone fine-tuning (90.71 frozen) | P1 row; source of the TAD split |
| π-VAD (CVPR 2025) | Induces five modalities from RGB at training time | MSAD 88.68 AUC / 71.26 AP | Five teacher backbones | P2 row |
| GS-MoE (ICCV 2025) | Gaussian pseudo-labels plus one expert per class | MSAD 87.72 / 68.26 / 69.54 / 76.68 | Needs class labels | P2 row |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment with boxes and captions | VadCLIP framework on MSAD 87.45 / 71.09 → 89.63 / 78.24 | Needs boxes and VLM captions | P2 row |
| WS-DoTA (preprint) | Weakly supervised DoTA with CLIP-feature differences | RTFM 57.9 → 78.2 AUC | Own \~1,031-clip subset; AUC only | Flagged P3 row |
| LAS-VAD (CVPR 2026) | IAM + ACC + attributes | XD-Violence 87.92 AP (CLIP) | No code, no traffic | Group C reimplementation of IAM on the thesis host |

### 2.4 Research gaps

1. **No motion-aware foundation features in weakly supervised traffic VAD.** *Addressed by the motion stream, already shown to lift zero-shot DoTA to 0.81 / 0.75.*
2. **No learned model of abrupt change under weak labels on traffic.** Change-based traffic detectors need frame labels, the weakly supervised dashcam study uses fixed raw CLIP differences, and LAS-VAD's IAM has never been tested on traffic, under ego-motion, next to a motion stream or in a language-guided host. *Addressed by KIM.*
3. **Intention prototypes rely on self-assigned classes.** LAS-VAD's prototypes and contrast take classes from the model's own argmax, which fails silently when the training data are binary. *Addressed by the normal-anchored prototype and contrast, tested against LAS-VAD's form.*
4. **Traffic results are hard to compare.** *Addressed by four split-consistent, multi-metric protocols (Section 4.4).*

| Closest work | What it does | How LaGoVAD-KIM differs |
| --- | --- | --- |
| LAS-VAD (CVPR 2026) | IAM with pseudo-label prototypes and contrast; XD-Violence, UCF-Crime | Traffic and dashcam data; open-world language-guided host with a motion stream; binary normal-anchored contrast; seam and cut masks; per-video scale; return before text fusion |
| WS-DoTA (preprint) | Fixed absolute CLIP-feature differences for weakly supervised DoTA | Learned position, velocity and acceleration of the contextualised feature, with its own loss |
| TTHF (IEEE 2024) | Pixel frame-difference branch, frame labels | Video-level labels; feature-space kinematics; no extra encoder |
| MGFN (AAAI 2023), RTFM (ICCV 2021) | Feature *magnitude* per snippet | Magnitude of *change* across steps |
| UR-DMU (AAAI 2023) | Normal and abnormal memories with a triplet loss | Prototypes in a kinematic space, with exact hard negatives from normal videos |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that models the velocity and acceleration of contextualised features with a normal-anchored contrastive loss, inside a language-guided open-world detector with a motion-foundation stream, and the first test of LAS-VAD's intention module on traffic video and under ego-motion.** Kinematic features and prototypes are not claimed as new; the contribution is their traffic adaptation, the label-safe loss and the analysis by camera setting, category and onset time.

## 3. Proposed model

The proposed model, **LaGoVAD-KIM**, is LaGoVAD with two changes: a frozen VideoMAE motion stream fused through W\_u (the backbone change) and the Kinematic Intention Module (the novel component), which sits between the RoFormer and the text fusion. Every other part of LaGoVAD stays as released, so any gain can be traced to these changes.

### 3.1 Base architecture

**LaGoVAD, unchanged parts.**

- **CLIP stream.** A frozen CLIP ViT-B/16 image encoder gives one 512-d CLS embedding c\_t per sampled step, from a 224×224 resize without cropping.
- **Temporal encoder.** A residual RoFormer: 2 layers, 4 heads, hidden size 512, local window of 25 steps. Training sequences are resampled or padded to 512 steps. Its output is h\_t ∈ R^512 (LaGoVAD's v^t).
- **Text branch.** The frozen CLIP ViT-B/16 text encoder encodes each category of the definition Z (a class name or a description; training picks one at random per batch) into τ\_c ∈ R^512. As released, 32 learnable soft-prompt tokens surround the definition tokens, half before and half after (16 + 16).
- **Fusion.** A 2-layer co-attention Transformer lets video steps and text tokens attend to each other.
- **Heads.** A single 1-D convolution (kernel 9, replicate padding) scores the pre-fusion and post-fusion features; a learned weight w = σ(10·a), a initialised at 0, blends them, and the step score is the sigmoid of the blend. A linear classification head compares projected video and text features.
- **Losses.** Top-k MIL binary cross-entropy (k = max(⌊L/16⌋, 1)), a multi-class MIL alignment loss, dynamic video synthesis L\_dvs (an abnormal clip inserted among 1–5 kNN-retrieved normal clips, giving frame pseudo-labels), and a caption contrastive loss with hard negatives L\_neg (temperature 0.02).

**Backbone change: the motion stream.** For each step t, a 16-frame clip centred on t passes through a frozen **VideoMAE V2-S** (ViT-S distilled from VideoMAE V2-g on Kinetics-710, D = 384). Its 8×14×14 last-layer tokens are averaged and normalised with fixed training statistics (m\_u, σ\_u from the training set) into ṽ\_t, then added through a projection that starts at zero:

```latex
\tilde v_t = \frac{\bar v_t - m_u}{\sigma_u} \in \mathbb{R}^{384}, \qquad x_t = c_t + W_u\, \tilde v_t, \qquad W_u \in \mathbb{R}^{512 \times 384},\; W_u^{\text{init}} = 0
```

x\_t replaces c\_t as the RoFormer input; the kNN retrieval of dynamic video synthesis still uses c\_t. At initialisation the model is CLIP-only LaGoVAD, and W\_u receives a non-zero gradient from the first step because ṽ\_t is non-zero. Trained on Nexar + CCD, this model reaches **0.81 macro / 0.75 micro** zero-shot on DoTA (one configuration; to be repeated with 3 seeds).

**Sampling at a fixed time step.** KIM differences neighbouring steps, so a step must mean the same time everywhere. The CLIP stream therefore samples by native frame rate: stride 8 at 30 fps (0.27 s), stride 3 at 10 fps (0.30 s), stride 7 at 25 fps (0.28 s). The released LaGoVAD's DoTA protocol (every 10 fps frame) is kept only for re-scoring its checkpoint (row B0); every retrained row, B2 included, uses the fixed step, and scores are mapped back to frames.

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips; none of the seven traffic datasets is among them.

&#91;embedded content: LaGoVAD-KIM · baseline in outline, KIM in colour\]

Only KIM and its two training losses are new; the motion stream is the backbone change, and every other box is LaGoVAD as released. h̃\_t also feeds the pre-fusion head and L\_neg.

### 3.2 Novel component: the Kinematic Intention Module (KIM)

#### 3.2.1 What is taken from LAS-VAD, and why the earlier components were dropped

LAS-VAD's IAM argues that normal and abnormal behaviour can share appearance and differ in *how fast* it unfolds (taking an item vs stealing it). It projects the temporally modelled feature X\_f to a position feature, takes gated absolute first and second differences as velocity and acceleration, classifies the concatenation, weights the logits by cosine similarity to momentum class prototypes, and adds a cross-intention InfoNCE loss. In LAS-VAD's ablation on XD-Violence (fine-grained mAP), IAM alone lifts the average from 24.24 to 29.98, and from 33.25 to 36.89 on top of its other two parts.

Traffic makes the speed argument physical: a collision is an impulsive change of velocity, and a near-miss is a hard brake or swerve. KIM keeps IAM's placement (after temporal modelling) and its three-part feature, and changes every part that depends on fine-grained class labels or on LAS-VAD's host.

Four components were built and tested before this one. **MQR** (query readout of VideoMAE tokens) and **LSP** (latent next-step prediction on the VideoMAE vector) added nothing; both re-processed a global motion vector the host already had. **SA²** (motion-selected CLIP patches) could not point at agents better than a centre prior. Raw CLIP feature differences fed as direct inputs also did not help in preliminary tests. The lessons that shape KIM:

1. **Add a quantity the host cannot compute,** not a re-reading of an input it already has: here, the per-channel magnitude of change of the contextualised feature (Section 3.2.6).
2. **Difference after temporal modelling, not on raw frames,** as LAS-VAD does; raw differences are noisy, and on dashcam data amplified CLIP-feature change raises false positives.
3. **Give the learned part its own signal.** KIM has its own top-k MIL term, so it learns even while the host ignores it, and it enters the host through a zero-initialised projection, so an unhelpful KIM costs nothing.
4. **Use labels that are certain.** Every step of a normal video is normal; KIM's contrastive loss is anchored on those steps rather than on pseudo-labelled abnormal frames.

| Part of IAM (LAS-VAD) | LAS-VAD | KIM (this thesis) | Reason |
| --- | --- | --- | --- |
| Input | X\_f after local Transformer + GCN, CLIP only | h\_t after the RoFormer (CLIP + VideoMAE) | Same role; differences of a smoothed, contextualised feature |
| Time step | One frame per snippet, any frame rate | Fixed 0.27–0.30 s per step | A velocity needs a fixed Δt |
| Splices and cuts | None in its host | Differences masked at synthesis seams, padding and detected shot cuts | LaGoVAD's dynamic video synthesis creates jumps that coincide with its pseudo-labels |
| Scale | Raw | Divided by the video's median first-difference magnitude | Dashcam ego-motion makes change normal; abrupt means large relative to the clip |
| Classes | C + 1 categories | Binary (normal / abnormal) | Dashcam training data have no categories; categories stay with the text branch |
| Confidence | Cosine to the predicted class prototype, multiplied into the logits | (1 + cos) / 2 | A negative cosine would flip the prediction |
| Prototype update | Frames whose score exceeds α, all classes | Normal prototype from all steps of normal videos; abnormal prototype from confident top-k steps of abnormal videos | Normal labels are exact |
| Contrastive positive | The least similar same-class frame | The abnormal prototype | Under pseudo-labels the least similar frame is the likeliest label error |
| Contrastive negatives | Top-M most similar other-class frames | Top-M most similar steps of normal videos, plus the normal prototype | Hard negatives with exact labels |
| Output | Scores averaged into the fine-grained logits | Features returned to h\_t before text fusion; KIM's score used for training and diagnostics only | Keeps detection conditioned on the definition |
| Supervision | Only through the averaged MIL score | Its own top-k MIL term | A learned part without its own signal did not move in MQR |

LAS-VAD's anomaly-connected components (frame grouping by DFS on a thresholded T×T similarity graph) and its LLM-written attribute texts are not used. The first needs frame pseudo-labels and a graph threshold that LAS-VAD tunes on its own benchmarks; the second would bring an LLM into training.

#### 3.2.2 Kinematic features

KIM reads h\_t, the RoFormer output, and builds three 128-d features per step. A binary mask m\_t is 1 when steps t−1 and t come from the same source clip with no shot cut and neither is padding, and 0 otherwise.

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
- The masks are built when a video is loaded, before dynamic video synthesis splices clips. Shot cuts are found once at extraction with PySceneDetect for every dataset.

#### 3.2.3 Kinematic score, prototypes and return path

Two layers score each step, and two prototypes μ\_0 (normal) and μ\_1 (abnormal) in R^384 weight the score by how typical the step is of its predicted class:

```latex
r_t = W_2\,\mathrm{ReLU}(W_1 k_t) \in \mathbb{R}^{2}, \qquad \hat y_t = \arg\max r_t, \qquad w_t = \tfrac{1}{2}\big(1 + \cos(k_t, \mu_{\hat y_t})\big), \qquad q_t = \mathrm{softmax}(w_t\, r_t)_{1}
```

The prototypes are buffers updated by momentum after each batch, with no gradient (β = 0.1, as in LAS-VAD; α = 0.5):

```latex
\mu_0 \leftarrow (1-\beta)\,\mu_0 + \beta\, \overline{\{k_t : t \in \text{normal videos}\}}, \qquad \mu_1 \leftarrow (1-\beta)\,\mu_1 + \beta\, \overline{\{k_t : t \in \mathcal{A}\}}
```

where 𝒜 holds the top-k steps by q\_t of each abnormal video whose q\_t exceeds α; on synthesized videos the top-k is taken inside the inserted abnormal clip, as L\_dvs does. A low-confidence step (w\_t near 0) gets a flatter distribution, so it contributes little to the top-k.

KIM returns its features to the host through a second zero-initialised projection:

```latex
\tilde h_t = h_t + W_o\, k_t, \qquad W_o \in \mathbb{R}^{512 \times 384},\; W_o^{\text{init}} = 0
```

h̃\_t replaces h\_t everywhere downstream: the co-attention fusion, the pre-fusion head and L\_neg's score-weighted pooling. Because the return path enters *before* fusion, the text definition still decides whether a kinematic change counts as anomalous. KIM's own score q\_t is not part of the test-time score; LAS-VAD's score-level averaging is ablation R11.

#### 3.2.4 Losses

KIM adds two training terms to LaGoVAD's losses.

**Kinematic MIL.** The host's top-k MIL form on KIM's own score:

```latex
\mathcal{L}_{\text{kin}} = \mathrm{BCE}\Big( \tfrac{1}{k} \textstyle\sum_{t \in \operatorname{top}k(q)} q_t,\; y \Big), \qquad k = \max(\lfloor L/16 \rfloor, 1)
```

**Normal-anchored kinematic contrast (NKC).** Each anchor t ∈ 𝒜 is pulled to μ\_1 and pushed from its M = 16 most similar steps of the normal videos in the batch, 𝒩\_t, and from μ\_0. With cosine similarity s and temperature τ\_c = 0.1, in standard InfoNCE form (the positive is in the denominator, unlike LAS-VAD's printed Eq. 11):

```latex
\mathcal{L}_{\text{nkc}} = -\frac{1}{\lvert \mathcal{A} \rvert} \sum_{t \in \mathcal{A}} \log \frac{e^{s(k_t, \mu_1)/\tau_c}}{e^{s(k_t, \mu_1)/\tau_c} + e^{s(k_t, \mu_0)/\tau_c} + \sum_{n \in \mathcal{N}_t} e^{s(k_t, k_n)/\tau_c}}
```

Every negative is certain: it comes from a video labelled normal. Only the anchors depend on KIM's own ranking, and only through the top-k of videos known to contain an anomaly.

```latex
\mathcal{L} = \mathcal{L}_{\text{LaGoVAD}} + \lambda_k\, \mathcal{L}_{\text{kin}} + \lambda_c\, \mathcal{L}_{\text{nkc}}
```

λ\_k ∈ {0.1, 0.5, 1} and λ\_c ∈ {0.05, 0.1, 0.5} are chosen once on held-out training videos. L\_nkc and the prototype updates start after a 5-epoch warm-up (of 40), so the prototypes are not built from random features.

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
4. **Differencing belongs after temporal modelling.** LAS-VAD differences the feature after its local Transformer and GCN, and action-recognition modules that gain from differences (TDN: +5.7 top-1 on Something-Something V1; TEA: +2.4) difference trainable, contextualised features and smooth before subtracting ([TDN](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.pdf), [TEA](https://arxiv.org/abs/2004.01398)). This thesis's own raw CLIP differences did not help as inputs, and WS-DoTA's amplified variant hurt. No paper compares the two placements, so R7 tests it.
5. **Contrast anchored on normal videos is the reliable form.** Weakly supervised methods that gain from prototype or contrastive terms build the reliable side from normal videos and pick abnormal samples only by top-k or score weighting: UR-DMU's two memories give 86.97 UCF-Crime AUC against 85.19 with the normal memory alone ([UR-DMU](https://arxiv.org/abs/2302.05160)); MGFN's magnitude contrast adds 2.13 AUC / 3.98 AP ([MGFN](https://arxiv.org/abs/2211.15098)); TPWNG's normality guidance adds 1.96 / 2.36 ([TPWNG](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.pdf)). LAS-VAD instead takes classes from its own argmax and mines the least similar positive, the pattern that drives confirmation bias in pseudo-labelling ([Arazo et al.](https://arxiv.org/abs/1908.02983)).
6. **Contrast already matters in this host.** Removing LaGoVAD's own hard-negative contrastive loss drops its zero-shot TAD AUC from 89.56 to 83.26 and DoTA from 62.60 to 59.88. NKC adds a contrast in the kinematic space, which L\_neg (a video–text contrast) does not touch.

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

Training uses video-level annotations only and LaGoVAD's schedule, so the baseline and KIM differ only in architecture and the two added terms.

**Training data.** Only TAD, MSAD, CCD and Nexar contain both normal and abnormal videos, so only they train models; DoTA, DADA-2000 and MM-AU contain accident videos only and serve as test sets.

| Protocol | Training set | Normal / abnormal videos | Anomaly text (hand-written) |
| --- | --- | --- | --- |
| P1, TAD | TAD train, UMIL split | 210 / 190 | TAD's 7 category names and definitions |
| P2, MSAD | MSAD weak-protocol train | 360 / 120 | MSAD's 11 anomaly types |
| P3, dashcam | Nexar labelled set + CCD official train split | Nexar 750 / 750; CCD's train part of 3,000 normal / 1,500 crash | Collision, near-miss and car-crash definitions |
| P4, cross-setting | None: reuses the P2 and P3 models | — | Each test set's own definitions |

**Captions and shortcut control.** Every model uses the same hand-written caption template per category. CCD's crash and normal clips come from different sources, so P3 is also reported with Nexar-only training. KIM adds a second shortcut risk: if shot cuts or splice seams leaked into the differences, KIM could learn "discontinuity = anomaly"; the masks of Section 3.2.2 remove them, and ablation R8 measures what happens without them.

**Stages.**

1. **Cache features.** CLIP CLS features c\_t at the fixed step, the normalised VideoMAE V2-S vector ṽ\_t, and the shot-cut list per video.
2. **Train all models with one schedule.** Every row starts from the same random initialisation of LaGoVAD's trainable parts, with the same schedule and 3 seeds.
3. **Monitor** the quantities of Section 3.2.5.

**Settings.** AdamW, batch 64, learning rate 5e-5 with a cosine schedule, 40 epochs, as in LaGoVAD. λ\_k, λ\_c, M ∈ {8, 16, 32} and α ∈ {0.3, 0.5} are chosen once on a held-out 15% of the P3 training videos by video-level AUC, then frozen for every protocol. No hyperparameter is chosen on a test list.

**Checkpoint selection.** Every model, baselines included, reports the last epoch.

**Compute.** KIM costs O(L·D) per video and no extraction, so training costs about the same as the motion-stream model on one 24 GB GPU.

### 3.4 Inference

1. **Sample** at the fixed step (stride 8 at 30 fps, 3 at 10 fps, 7 at 25 fps).
2. **Encode** each step with frozen CLIP (c\_t) and the 16-frame clip with frozen VideoMAE V2-S (ṽ\_t); fuse x\_t = c\_t + W\_u ṽ\_t.
3. **Contextualise** with the RoFormer into h\_t.
4. **Kinematics.** Compute p\_t, u\_t, a\_t with the shot-cut and padding masks and the video's ρ\_V; return h̃\_t = h\_t + W\_o k\_t.
5. **Encode the definition** with LaGoVAD's soft prompts and the frozen CLIP text encoder.
6. **Score** with co-attention fusion and the blended heads. Each step's score is assigned to the frames it stands for.
7. **Report** KIM's q\_t and ‖a\_t‖ per step as a diagnostic curve for the onset analysis.

**Score post-processing.** DoTA and DADA-2000 contain only abnormal clips, so their scores are min-max normalised per video before pooling, following LaGoVAD's `offline_dota_eval.py`. TAD, MSAD and CCD scores are used raw.

**Cost and causality.** One VideoMAE ViT-S clip (about 57 GFLOPs) runs per step; KIM adds a negligible amount. The model is offline: ρ\_V uses the whole video, and the RoFormer looks both ways; KIM's differences themselves are backward-looking.

## 4. Experimental plan

The plan trains with video-level labels only, checks KIM's premise with an inference-only pilot on the trained baseline before any KIM training, compares against published rows on identical test lists, and adds every second metric by re-scoring released checkpoints.

### 4.1 Dataset preparation

| Dataset | View | Size and split | Labels | Role | Source |
| --- | --- | --- | --- | --- | --- |
| TAD | Fixed CCTV, some first-person | 500 videos; UMIL split 400 train (210 normal / 190 abnormal), 100 test (40 / 60); 7 categories | Video labels (train), anomaly interval (test) | P1 train and test; P4 zero-shot test | Frames and split files via the UMIL repository; native fps unknown |
| MSAD | Fixed, 14 scenes | 720 videos at 30 fps; weak protocol 360 normal + 120 abnormal train, 120 + 120 test (20 traffic accidents) | Video labels (train), frame labels (test) | P2 train and test; traffic slice; P4 zero-shot test | MSAD project page (videos gated) |
| CCD | Dashcam | 1,500 crash + 3,000 normal clips (normal from BDD100K), 50 frames at 10 fps, official split | Video labels; accident onset | P3 train and in-domain test; onset analysis | GitHub (Cogito2012/CarCrashDataset) |
| Nexar | Dashcam | 1,500 labelled clips (750 positive + 750 normal), about 40 s each | Video labels; event and alert times (analysis only) | P3 training | Hugging Face |
| DoTA | Dashcam | 4,677 clips (3,275 train / 1,402 test), 10 fps, abnormal only | Anomaly window; 9 categories × ego-involved / ego-uninvolved | P3 zero-shot test; onset analysis; Step 0 pilot (train split only) | GitHub (MoonBlvd/Detection-of-Traffic-Anomaly) |
| DADA-2000 | Dashcam | 2,000 accident clips at 30 fps | Accident intervals; ego / non-ego | P3 zero-shot test | Simple-TAD's split files |
| MM-AU | Dashcam | 11,727 accident videos pooled from CCD, A3D, DoTA and DADA-2000 | Accident start, collision, end | Optional zero-shot test, after removing shared videos | MM-AU release (CVPR 2024) |

**Weights to download** (all public): CLIP ViT-B/16 and VideoMAE V2-S (`vit_s_k710_dl_from_giant`); LaGoVAD's `best.ckpt` and `pred_results.pkl` (reference row only); VadCLIP, PEL4VAD and DSANet checkpoints for zero-shot re-scoring; RTFM and MGFN MSAD checkpoints for the traffic slice; MOVAD and Simple-TAD DoTA and DADA-2000 checkpoints as frame-supervised context; Simple-TAD's DAPT-VideoMAE-S for the encoder ablation. No link has been test-downloaded yet; that is the first task. LAS-VAD has no public code, so IAM is reimplemented from the paper.

### 4.2 Dataset processing

One feature cache serves the baseline, KIM and every ablation, because KIM needs no new features.

1. **Decode and sample at the fixed step.** CLIP stride 8 at 30 fps, 3 at 10 fps, 7 at 25 fps; TAD's native fps is unknown, so its effective step is reported as unknown and stride 8 is used. The VideoMAE clip for each step takes 16 frames centred on it, resampled to 10 fps (1.6 s), with edge frames repeated at boundaries.
2. **Extract CLIP CLS features** with LaGoVAD's `extract_feat_clip.py`, so features match the released checkpoint.
3. **Extract the VideoMAE V2-S vector**, average the 8×14×14 tokens, and normalise with m\_u, σ\_u from the training set of each protocol.
4. **Detect shot cuts** once per video with PySceneDetect and store the cut steps; they feed KIM's mask m\_t. Edited compilation videos in TAD and MSAD are the expected users of this mask; dashcam clips should have almost none.
5. **Build labels.** Training uses video-level labels, category indices and caption templates (TAD from UMIL's `TAD_train.txt`, MSAD from its weak split, CCD from its official split, Nexar from its labels). Test labels are frame-level: TAD intervals, MSAD frame labels, CCD accident onsets, DoTA anomaly windows, DADA-2000 intervals.
6. **Prepare analysis labels.** For DoTA keep `ego_involve`, the category and the anomaly start; for CCD the accident onset frame; for Nexar the event time (analysis only).
7. **Fix test lists.** LaGoVAD's lists (DoTA 1,402 clips; its 100-video TAD list) for zero-shot comparability; UMIL's TAD test list for P1 and as a second zero-shot column; MSAD's weak-protocol test set; CCD's official test split; Simple-TAD's DADA-2000 test split.
8. **Hold out** 15% of the P3 training videos, stratified by label and source, for the one-time hyperparameter choice.

**Leakage checks before any training.**

- Seventy-six of LaGoVAD's 100 TAD test videos are in UMIL's TAD training list, so a model trained on UMIL's split is never scored on LaGoVAD's TAD list.
- MM-AU is never used for training, and as a test set only after removing shared videos.
- DoTA's training split is used only for the read-only Step 0 pilot, never to fit or tune anything.
- A near-duplicate check with CLIP-feature similarity runs between every training set and every test set.
- LaGoVAD's published zero-shot numbers are reproduced from `pred_results.pkl` before anything is trained.

### 4.3 Pilot and baseline running

**Step 0: pilot diagnostic (inference only, on the trained motion-stream model B2 from P3, a random 500-clip subset of DoTA's training split and CCD's held-out training clips).** It asks whether the magnitude of change of the RoFormer output already rises at anomalies before anything is trained for it. Each candidate is used directly as a step score, min-max normalised per video.

| Check | Question | Pass criterion (proposed) | If it fails |
| --- | --- | --- | --- |
| K1: change signal | Per-video AUC of ‖Δh\_t‖ / ρ\_V and ‖Δ²h\_t‖ / ρ\_V, against the same quantities on c\_t and ṽ\_t; by ego flag | ‖Δ²h‖ clearly above 0.5 (≥ 0.60) at least on ego-involved clips | If c\_t or x\_t pass and h\_t does not, KIM's default input moves there (R7) |
| K2: complementarity | Does a rank average of B2's score with ‖Δ²h‖ beat B2 alone? | Paired gain with CI above zero | KIM's premise is weak; the ladder still runs, but expectations are lowered in writing |
| K3: discontinuities | Share of the top 5% ‖Δh‖ steps on Nexar and CCD normal clips that fall within one step of a shot cut or a synthesis seam, with and without the mask | Near zero with the mask | Fix the cut detector before training |
| K4: onset | Median lag between the peak of ‖Δ²h‖ and the annotated onset on CCD and DoTA | Within ±0.5 s for most clips | KIM's onset hypothesis (H6) is dropped from the claims |

The pilot costs well under one GPU-day, since all features are cached. A failure does not stop the plan; it decides KIM's input and how strongly the hypotheses are stated before training.

**Group A: reproduce LaGoVAD and the motion stream.**

| Row | Model | Training | Purpose |
| --- | --- | --- | --- |
| B0 | LaGoVAD, released checkpoint | PreVAD (by its authors) | Reproduce the published zero-shot numbers (TAD 89.56, DoTA 62.60, MSAD 90.41), then re-score on UMIL's TAD list, DADA-2000, CCD and the MSAD traffic slice with AUC, AP and AUC\_A |
| B1 | LaGoVAD, retrained | Each protocol's training set, 3 seeds | Fair baseline with the thesis data, features and schedule; separates domain-matched data from motion |
| B2 | LaGoVAD + motion stream (W\_u) | Same, 3 seeds | KIM's starting point (row R0) |

**Group B: published methods with released weights, re-scored for a second metric.** VadCLIP and DSANet (UCF-Crime and XD-Violence checkpoints, same CLIP features); PEL4VAD (I3D features for TAD, DoTA and DADA-2000); RTFM and MGFN (MSAD checkpoints, for the traffic slice); MOVAD and Simple-TAD (DoTA and DADA-2000, frame-supervised context only).

**Group C: same-feature competitors retrained in-domain.** VadCLIP's public code retrained on P1–P3 with the thesis's CLIP features. **LAS-VAD's IAM**, reimplemented as published on the same host and features (row R4 with LAS-VAD's own class assignment, positive choice and score-level averaging), so KIM is compared with the method it adapts rather than only with its own ablations. Methods without released weights are quoted from their papers only: UMIL and WSAL on TAD; π-VAD, GS-MoE and Alert-CLIP on MSAD; WS-DoTA; and the frame-supervised TTHF, PromptTAD and ISCRTAD.

### 4.4 Evaluation protocols

**Metrics.**

- **AUC\_O / AP:** frame-level ROC AUC and average precision, pooled over all test frames.
- **AUC\_A / AP\_A:** the same over abnormal test videos only; they measure localisation inside abnormal videos.
- **DoTA and DADA-2000 macro and micro AUC** after per-video min-max normalisation; reported overall, for ego-involved and ego-uninvolved clips and per category.
- **Onset error:** for each abnormal CCD and DoTA clip, the time from the annotated onset to the first step whose per-video normalised score reaches 0.5; reported as the median absolute error in seconds and the share of clips within ±0.5 s, for the final score and for KIM's q\_t and ‖a\_t‖.
- **False alarms on normal driving:** on Nexar's and CCD's normal test clips, the share of steps above the threshold that gives 90% recall on abnormal clips; checks that KIM does not fire on normal hard braking and turns.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P1 TAD in-domain (fixed camera) | TAD train → TAD test, UMIL split | AUC\_O, AUC\_A, AP | UMIL and WSAL on both metrics; AUC\_O-only rows; LaGoVAD, VadCLIP and LAS-VAD IAM retrained |
| P2 MSAD in-domain (multi-scene surveillance) | MSAD weak train → MSAD test; traffic slice and non-traffic scenes scored separately | AUC, AP, AUC\_A, AP\_A; drift@5 | π-VAD and GS-MoE (four metrics), Alert-CLIP (two); RTFM, MGFN, UR-DMU on the traffic slice |
| P3 dashcam | Nexar + CCD train → CCD test (in-domain); DoTA and DADA-2000 (zero-shot) | AUC, AP; DoTA macro / micro by ego flag and category; onset error; false alarms | DoTA: LaGoVAD, VadCLIP, PEL (published AUC, AP re-scored), WS-DoTA flagged; CCD and DADA-2000: released checkpoints re-scored; frame-supervised rows as context |
| P4 cross-setting zero-shot | P3 model → TAD (both lists) and MSAD; P2 model → TAD (both lists) | AUC, AP, AUC\_A | LaGoVAD, VadCLIP, PEL; MultiDomain and MSAD's cross-dataset rows (AUC) |

**Published in-domain rows (P1, P2).**

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
| Thesis motion-stream model (R0) | Video-level, Nexar + CCD, zero-shot | Per-video min-max, 1,402 clips | 0.81 macro / 0.75 micro | Reference |
| LaGoVAD | Video-level, PreVAD, zero-shot | Per-video normalised | 62.60 | Yes (against micro) |
| VadCLIP | Weak, UCF-Crime / XD-Violence / PreVAD, zero-shot | Same | 50.93 / 49.00 / 50.59 | Yes |
| PEL (I3D) | Weak, UCF-Crime / XD-Violence, zero-shot | Same | 53.05 / 44.97 | Yes |
| WS-DoTA (RTFM + CLIP differences) | Weak, D²-City + DoTA in-domain | \~1,031 clips, normalisation not stated | 78.2 | Context only |
| AnoPred; DoTA ensemble; MAMTCF | Unsupervised, DoTA in-domain | Pooled raw scores | 67.5; 73.0; 76.6 | Context only |
| MOVAD, PromptTAD, TTHF, ISCRTAD | Frame labels, DoTA in-domain | Pooled; TTHF post-processed | 82.2, 83.9, 84.7, 86.2 | Upper reference only |
| Simple-TAD DAPT-VideoMAE-S / -L | Frame labels, fine-tuned | Pooled | 86.4 / 88.4 | Upper reference only |

**Reporting rules.** TAD is reported on two separate columns, never mixed. Every thesis number is a mean ± standard deviation over 3 seeds. Subsets carry 95% bootstrap confidence intervals over videos, and KIM against the motion-stream model is tested with a paired bootstrap. All models use the same test lists, frame-to-score mapping and normalisation, and every table names each row's training set and protocol.

### 4.5 Ablation studies

The ladder adds KIM's parts one at a time on top of the motion-stream model, then tests its traffic adaptations. Every row uses video-level labels only and reports P1 (AUC\_O, AUC\_A), P2 (AUC, AP, drift@5), P3 (CCD AUC; DoTA macro / micro by ego flag; DADA-2000 AUC; onset error) and P4 on TAD's two lists. Expected outcomes are priors written down in advance.

| Row | Variant | Isolates | Expected if KIM's premise holds |
| --- | --- | --- | --- |
| R0 | Motion-stream model (B2) | Reference | 0.81 / 0.75 on DoTA |
| R1 | + KIM features through W\_o, no KIM losses | The feature path alone | Small or no gain; W\_o may stay near zero, as raw differences did |
| R2 | + L\_kin (q\_t = softmax(r\_t), no prototypes) | KIM's own signal | Gain over R1, mostly on ego-involved clips |
| R3 | + prototype confidence weighting | LAS-VAD's confidence step | About R2; better onset precision |
| R4 | + LAS-VAD's contrast (self-assigned classes, least similar positive, other-class negatives) | IAM as published, on this host | At or slightly above R2; less stable across seeds |
| **R5** | **+ normal-anchored contrast NKC instead (full LaGoVAD-KIM)** | The label-safe loss | **Best overall; lower false alarms than R4** |
| R-cap | R5 with k\_t replaced by an MLP of h\_t with the same width and losses, no differences | Capacity and extra loss without kinematics | Below R5 |
| R6 | Order: p only / p + u / p + u + a | First vs second order | p + u + a best; p only ≈ R-cap |
| R7 | Input: c\_t (CLIP only) / x\_t (before the RoFormer) / h\_t (default) | Placement | h\_t ≥ x\_t > c\_t, unless Step 0 says otherwise |
| R8 | No seam, padding or cut masks | The discontinuity shortcut | Worse zero-shot, more false alarms at cuts on TAD and MSAD |
| R9 | No per-video scale ρ\_V | Ego-speed normalisation | Worse on dashcam sets, about equal on fixed cameras |
| R10 | Signed differences instead of absolute values | IAM's absolute value | About R5; signed may help approach vs recede |
| R11 | KIM's score averaged into the test-time score, as in LAS-VAD | Score-level vs feature-level return | Similar AUC, worse drift@5 and definition sensitivity |
| R12 | Stop-gradient from KIM into the RoFormer and W\_u | Whether KIM's losses reshape the host | Slightly below R5 |
| R13 | KIM on the ṽ\_t sequence only | Kinematics of the motion vector (the MQR / LSP lesson) | Below R5 |
| R14 | KIM on CLIP-only LaGoVAD (no motion stream) | Complementarity with VideoMAE | Larger gain over B1 than R5 over R0 |

H1 holds if R5 beats R0 with larger gains on ego-involved clips; H2 if R5 beats R-cap and R6's position-only row; H3 if p + u + a is best in R6 and h\_t is at least as good as x\_t in R7; H4 if R5 beats R4 and R4 is not below R2; H5 if R5's drift@5 and non-traffic MSAD scores are no worse than R0's and R11's drift@5 is worse; H6 if KIM's onset error beats R0's.

**Side ablations within the component.** M ∈ {8, 16, 32}; α ∈ {0.3, 0.5}; β ∈ {0.05, 0.1, 0.3}; gate kernel 3 vs 5; difference spacing ±1 vs ±2 steps; class-wise prototypes (C + 1) on P1 and P2, where categories exist; NKC with and without μ\_0 as a negative; warm-up 0, 5 or 10 epochs.

**Analyses.**

1. **Where the gain lands.** DoTA ego-involved vs ego-uninvolved crossed with the nine categories; TAD Accident, PedestrianOnRoad and the remaining categories pooled; the MSAD traffic slice against its non-traffic scenes; Nexar-only against Nexar + CCD training.
2. **Kinematic curves.** q\_t, ‖u\_t‖ and ‖a\_t‖ plotted against annotated onsets for successes and failures, including normal hard-braking clips.
3. **Learned parts.** ‖W\_o‖ over training, gate means, the size of 𝒜 per batch and cos(μ\_0, μ\_1); whether the prototypes separate on held-out data.
4. **Language guidance.** Score change of accident frames when the definition includes or excludes accidents; drift@5 on MSAD; TAD's 7 categories with their own definitions.
5. **Cost.** Parameters and measured frames per second for R0 and R5.

**Budget.** Rows R0–R5 and R-cap run on P1, P2 and P3 with 3 seeds; P4 reuses those models. Rows R6–R14 and the side ablations run on P1 and P3 only, first with one seed, then with three for the variants that are kept. Every row shares one feature cache and costs about one baseline training run. If the total exceeds the budget, the ladder is cut to R0, R2, R4, R5, R-cap and R7, which still test H1–H4.

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
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (shot-cut detection)
