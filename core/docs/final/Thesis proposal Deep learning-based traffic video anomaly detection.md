# Thesis proposal: Deep learning-based traffic video anomaly detection (LaGoVAD-LSP)

Nov 1, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-LSP**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. It starts from LaGoVAD (ICLR 2026) and adds a frozen VideoMAE V2-S motion stream, read as one 384-d global vector per step, as the permitted backbone change. Trained on Nexar and CCD dashcam video with video-level labels, this model already reaches **0.81 macro / 0.75 micro per-video min-max AUC zero-shot on DoTA**, against 62.60 for the released LaGoVAD.

The one novel component is a **Latent Surprise Predictor (LSP)**, adapted from FLaRA (arXiv 2606.14380). A small predictor, trained only on videos labelled normal, forecasts the VideoMAE vector one clip ahead; the gap between forecast and observation enters the detector through a gate that starts closed. The claim under test: a model of normal driving dynamics, learned in a frozen video-foundation latent, adds evidence that video-level MIL cannot learn by itself and that fixed change cues (frame differences, interpolation residuals) do not carry.

This version replaces the earlier Motion Query Readout (MQR). MQR's attention never left its uniform start, and a single global vector gave the same results, so the motion stream stays one vector and the novelty moves from *where* to read to *what is expected next* (Section 3.2.1).

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event is happening in it. Abnormal events include collisions, loss of control, illegal turns, wrong-way driving, pedestrians on the road and road spills. These are the categories used by TAD and DoTA.

Two camera settings define the problem space:

- **Fixed camera (non-ego).** TAD and the traffic scenes of MSAD. The camera is static; the anomalous agents are often small and far away.
- **Dashcam (ego-centric).** DoTA, DADA-2000, CCD and Nexar. The camera moves with the car. DoTA also marks each clip as ego-involved (the ego car takes part in the anomaly) or ego-uninvolved (it only observes); this proposal uses those two terms for that flag.

The thesis follows LaGoVAD's weakly supervised, language-guided formulation. Training uses video-level annotations only: one label y\_i ∈ {0, 1} per video, plus a category and a hand-written caption template. At test time the model receives a video V and a text definition Z of what counts as anomalous, and returns a score per sampled step.

```latex
\Phi_\theta : (V, Z) \;\mapsto\; \{ s_t \}_{t=1}^{N}, \qquad s_t \in [0,1]
```

The video-level annotations supervise training through multiple-instance learning (MIL): the mean of the top-k step scores of a video must match its video label.

Five properties make traffic video harder than generic surveillance, and the fifth is what the novel component exploits:

1. **Anomalies are motion events.** A collision is defined by how trajectories meet, not by what one frame looks like. LaGoVAD encodes single RGB frames with CLIP and has no motion input; it reaches 89.56 AUC on TAD but 62.60 on dashcam DoTA. Part of this gap is protocol: DoTA is abnormal-only and min-max normalised per video, so its AUC measures timing inside a clip.
2. **The agents that matter are small.** A distant agent of about 40×30 px in a 1280×720 frame shrinks to about 7×9 px at 224×224 input, smaller than one 16-px patch (estimate). The main component does not target this; an optional arm does (Section 3.5).
3. **The camera moves.** In dashcam video, ego-motion dominates frame-to-frame change. In preliminary experiments, hand-crafted motion cues (feature differences, ego-motion-compensated optical flow) did not help.
4. **Labels are scarce.** Frame-level labels are expensive and anomalies are rare, so the model must learn from video-level labels.
5. **Normal driving is abundant and regular.** Every training set holds hundreds to thousands of videos labelled normal (Nexar 750, CCD 3,000). Video-level labels identify them exactly, so a model of *normal* motion can be learned with no frame labels at all.

The scope is offline detection: the whole video is available and the question is whether an anomaly is happening, not whether one will happen (anticipation, as in Nexar or FLaRA). No large language model or multimodal LLM runs at training or test time; a frozen CLIP text encoder is allowed. All training text is hand-written: class names, category definitions and caption templates.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from cheap video-level labels, sees motion, and knows what normal motion looks like. It must keep LaGoVAD's ability to take the anomaly definition as text, and run on one academic GPU without any LLM.

The thesis builds this in two steps, matching the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** A frozen VideoMAE V2-S encoder (CVPR 2023), the strongest small masked-video encoder for traffic anomalies in Simple-TAD's comparison, feeds one 384-d vector per step into LaGoVAD through a zero-initialised gate. Preliminary result: zero-shot DoTA **0.81 macro / 0.75 micro** per-video min-max AUC, with the gain coming from this stream. Reading the dense VideoMAE tokens with learned queries (MQR) added nothing, so the stream stays one global vector.
- **Novel component (LSP).** A predictor trained on normal videos only forecasts the next motion vector; the observed-minus-predicted residual, a surprise score and a persistence-normalised surprise enter the fusion through a second zero-initialised gate.

A second purpose is a fair evaluation: published traffic numbers are mostly single-metric AUC on inconsistent splits and protocols (FLaRA's DoTA 90.0, for example, is a video-level anticipation AUC). The thesis will report at least two metrics against at least two published methods on the same fixed-camera traffic data, plus zero-shot and dashcam results, and label each row's protocol.

### 1.3 Research aims

The thesis has five aims:

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint and retrain LaGoVAD on the thesis training data, with AUC, AP and AUC on abnormal videos (AUC\_A) on TAD, MSAD, DoTA and DADA-2000.
2. **Document the motion stream.** Report the frozen VideoMAE V2-S stream's gain over CLIP-only LaGoVAD, and the negative MQR result that fixed it at one global vector.
3. **Design and train LSP.** Learn normal motion dynamics from normal videos only and feed the surprise into the detector, using video-level annotations only.
4. **Separate learned normality from simple change.** Test LSP against fixed change cues (copy-last difference, second-order interpolation residual) and against the same predictor trained without its reconstruction loss.
5. **Explain the gains.** Break results down by ego-involved vs ego-uninvolved clips, anomaly category and position inside the anomaly window, check a same-source test against domain shift, and check that MSAD's non-traffic scenes do not regress.

Aims 1–2 are prerequisites; aims 3–5 map to the hypotheses below. "Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval over test videos excludes zero.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. LSP improves on the motion-stream model and on fixed change cues | DoTA macro and micro AUC; DADA-2000 and CCD AUC (ablation rows R0, R1, R2, R5) | R5 > R0, R1 and R2 on DoTA macro AUC |
| H2. The gain comes from learned normal dynamics, not from capacity | Same predictor without L\_rec (R3); L\_rec on all videos instead of normals (R4) | R5 > R3 and R5 > R4 |
| H3. The gain is not a domain-shift artefact | Same-source Nexar/CCD test; source classifier on surprise statistics (diagnostic D4) | R5 > R1 on the same-source test, and surprise predicts anomaly better than it predicts source |
| H4. The gain holds across camera settings without hurting surveillance | TAD AUC\_O / AUC\_A; MSAD AUC / AP, its traffic slice and non-traffic scenes | Gains on TAD, the MSAD traffic slice and the dashcam sets; no significant drop on MSAD's non-traffic scenes |

One expectation is stated in advance rather than as a hypothesis: a surprise computed from a global vector should favour **ego-involved** anomalies, where the motion of the whole frame changes, and should rise most at anomaly onset. Section 4.5 measures both.

## 2. Literature review

### 2.1 Overview and classification of video anomaly detection method branches

Video anomaly detection (VAD) methods divide by the supervision they train with into five branches, plus two adjacent lines of work that this proposal draws on: accident anticipation and self-supervised latent prediction. Traffic methods cut across these branches along a second axis: fixed camera (non-ego) or dashcam (ego-centric). The table places the papers reviewed for this thesis, mostly 2024–2026.

| Branch | Training signal | Typical mechanism | Representative methods (venue) | Traffic evidence | Role in this thesis |
| --- | --- | --- | --- | --- | --- |
| Unsupervised / one-class | Normal videos only | Reconstruction, frame or flow prediction, future object location, diffusion, scene-graph normality | AnoPred (CVPR 2018), FOL / DoTA (IEEE TPAMI 2022), MAMTCF (arXiv 2307.14575), MA-PDM (AAAI 2025), ADSM (CVPR 2025), TAO (CVPR 2025), Structured Graph Reasoning (IEEE TMM 2026) | DoTA AUC 67.5 (AnoPred), 73.0 (DoTA ensemble), 76.6 (MAMTCF), pooled scores | Shows that normal-only prediction works for traffic, but in pixel, flow or box space, and suffers from ego-motion |
| Weakly supervised | Video-level labels, MIL | Snippet features (I3D, CLIP) + top-k MIL; CLIP text prompts; prototypes; fixed change cues | Sultani (CVPR 2018), RTFM (ICCV 2021), WSAL (IEEE TIP 2021), UMIL (CVPR 2023), MultiDomain (NeurIPS 2024), VadCLIP (AAAI 2024), STPrompt (ACM MM 2024), π-VAD (CVPR 2025), GS-MoE (ICCV 2025), DSANet, RefineVAD (AAAI 2026), Alert-CLIP (CVPR 2026) | TAD (WSAL, UMIL), MSAD (π-VAD, GS-MoE, Alert-CLIP), dashcam WS-DoTA (ECCV 2024 Workshop, 78.2 AUC in-domain) | **The thesis's branch** |
| Open-vocabulary / open-world | Video labels + category text | CLIP text space defines anomaly classes or free-text definitions at test time | Anomize (CVPR 2025), **LaGoVAD (ICLR 2026)** | Zero-shot TAD, DoTA, MSAD (LaGoVAD) | **Baseline** |
| Fully supervised traffic detection | Frame-level labels (+ boxes) | Video encoder + temporal head; object prompts; agent graphs | MOVAD (ICASSP 2024), TTHF (IEEE TCSVT 2024), PromptTAD (IEEE RA-L 2025), ISCRTAD (IEEE T-ITS 2025), Simple-TAD (ICCV 2025 Workshops) | DoTA AUC 82.2–88.4 in-domain | Upper reference rows; evidence for the motion encoder |
| LLM / MLLM-based | None, or few-shot rules, or light tuning | Captioning + LLM reasoning, prompting, probing of MLLM heads | LAVAD (CVPR 2024), AnomalyRuler (ECCV 2024), VERA, Holmes-VAU (CVPR 2025), EventVAD (ACM MM 2025), HeadHunt-VAD (AAAI 2026), LAVIDA (CVPR 2026), AnyAnomaly (WACV 2026) | Almost none on traffic sets | Excluded: LLM cost |
| Accident anticipation (adjacent task) | Positive / negative dashcam clips | Predict a collision before it happens; recently by forecasting future latents | DSTA, UString (ACM MM 2020), BADAS (arXiv 2510.14876), **FLaRA (arXiv 2606.14380)** | Nexar AP 86.6 / AUC 87.3 (FLaRA); video-level anticipation AUC on DAD, DADA-2000, DoTA subsets | **Source of the novel component** |
| Self-supervised latent prediction (adjacent line) | Unlabelled video | Predict masked or future latents; prediction error read as "surprise" | V-JEPA (arXiv 2404.08471), V-JEPA 2 (arXiv 2506.09985), Garrido et al. (arXiv 2502.11831), Asleep at the Wheel (arXiv 2608.01336), Latent Clarity (arXiv 2607.03558) | Surprise from frozen V-JEPA 2 is near chance within one driving dataset (nuScenes AP 0.29 vs 0.24 chance) | Evidence for and against the component; shapes how its output is used |

Three findings from this review shape the proposal. First, CLIP-feature weak supervision dominates surveillance benchmarks but sits near chance on dashcam DoTA, and the strongest traffic results come from masked video foundation models. Second, prediction of the future is an old and working idea for traffic anomalies, but in pixel, flow or box space, where ego-motion and background clutter break it. Third, FLaRA shows that predicting future *latents* with a reconstruction loss transfers across traffic datasets, but only for clip-level supervised anticipation; under video-level labels, the only normality priors in use are fixed change cues.

### 2.2 Emerging trends and unresolved challenges

Seven trends run through the 2024–2026 literature:

1. **Vision-language features as the default.** Since VadCLIP (AAAI 2024), most weakly supervised methods start from frozen CLIP frame features and text prompts: DSANet and RefineVAD (AAAI 2026), LAS-VAD, Fine-VAD and Alert-CLIP (CVPR 2026).
2. **Anomalies defined by language.** Open-vocabulary (Anomize, CVPR 2025) and open-world (LaGoVAD, ICLR 2026) models accept the anomaly definition at test time.
3. **MLLMs for explanation and training-free detection.** LAVAD, VERA, Holmes-VAU, HeadHunt-VAD and LAVIDA rely on LLMs or multimodal LLMs of 2B parameters and more. They are not clearly better at detection: on MSAD drift@5, LaGoVAD reaches 85.6 AUC / 40.1 AP against 84.3 / 34.3 for Holmes-VAU.
4. **Masked-video foundation models for traffic.** Simple-TAD showed that masked video modelling transfers best to traffic anomalies; VideoMAE-B reaches AUC-MCC 56.0 on DoTAref against 50.0 for InternVideo2-B after fine-tuning.
5. **Spatial and object-level grounding.** Alert-CLIP aligns region features with boxes and captions, PromptTAD prompts with detected boxes, STPrompt selects image crops by motion, and two 2026 preprints ("Localizing to Debias", "Learning Where and When") bring patch-level localisation to weak supervision on surveillance data.
6. **Extra modalities.** π-VAD (CVPR 2025) induces pose, depth, flow and other modalities from RGB features at training time and found flow the most useful.
7. **Predictive world models in latent space.** V-JEPA 2 predicts future latents for action anticipation; BADAS and FLaRA build accident anticipators on it; Garrido et al. read latent prediction error as surprise for intuitive physics. Whether such surprise detects real driving anomalies is contested: *Asleep at the Wheel* found it mostly detects which dataset a clip came from.

Five challenges remain open for traffic video:

1. **Motion blindness.** Frame-level CLIP features miss what makes a collision a collision. In LaGoVAD's table every surveillance-trained model is near chance on DoTA's within-clip protocol (VadCLIP, LaGoVAD: 49.00–62.60; PEL with I3D: 44.97–53.05).
2. **No model of normal dynamics under weak supervision.** MIL learns from about L/16 top-scoring steps per video. The normality priors used so far are fixed: WS-DoTA's copy-last CLIP difference (78.2 AUC on DoTA with RTFM) and STPrompt's second-order patch difference. The thesis pilot found such hand-crafted cues unhelpful on dashcam video.
3. **Ego-motion and domain shift.** Camera motion masks agent motion, and frame-prediction baselines flag trees and background noise on DoTA. Training data come from different sources than test data, and prediction-error signals are known to pick up the source.
4. **Spatial dilution.** Global vectors lose small and distant agents: even frame-supervised Simple-TAD averages 81.6 / 84.1 / 85.2 AUC on ego-uninvolved DoTA clips (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips. This thesis addresses it only in an optional arm.
5. **Inconsistent evaluation.** TAD has two incompatible test lists in use (76 of LaGoVAD's 100 TAD test videos belong to UMIL's training list). DoTA numbers come from at least four protocols: pooled raw scores, per-video min-max, post-processed scores, and video-level anticipation AUC on a 367-clip subset.

### 2.3 Key works behind the proposed model

The proposed model is assembled from one baseline, one encoder, one predictor design and the works whose evidence justifies each choice. The table shows where each part of LaGoVAD-LSP comes from; the reviews follow, FLaRA first among the component's sources.

| Part of LaGoVAD-LSP | Source work (venue) | What is taken | What the thesis changes |
| --- | --- | --- | --- |
| Host detector: CLIP stream, RoFormer, text co-attention, heads and losses | LaGoVAD (ICLR 2026), on VadCLIP's frozen-CLIP + MIL design (AAAI 2024) | Architecture, losses and schedule as released | Nothing inside the detector; last-epoch checkpoints; hand-written caption templates |
| Motion stream (backbone change) | VideoMAE V2 (CVPR 2023), chosen on the evidence of Simple-TAD (ICCV 2025 Workshops) | The frozen distilled ViT-S and Simple-TAD's 16-frame, 10 fps clip input | Frozen, one mean-pooled 384-d vector per step, fused through a zero-initialised gate |
| Latent predictor and its loss (novel component) | FLaRA (arXiv 2606.14380), on V-JEPA 2's predictor (arXiv 2506.09985) | A latent predictor, Smooth-L1 to stop-gradient encoder targets, a contiguous non-overlapping future | Offline detection: the observed-minus-predicted residual instead of classifying the prediction; normal videos only; detached from MIL; one 384-d vector instead of 2,048 tokens |
| Surprise readout | Garrido et al. (arXiv 2502.11831); Asleep at the Wheel (arXiv 2608.01336) | Prediction error as surprise; max-over-time pooling | Surprise is an input to a supervised MIL head, not a standalone score |
| Fixed-prior principle and fusion slot | STPrompt (ACM MM 2024); WS-DoTA (ECCV 2024 Workshop) | Priors not learned from video labels; the local signal is added to the frame feature before the temporal encoder | Learned from normal data instead of fixed; their fixed cues become baselines R1 and R2 |
| Add-not-replace fusion and ego diagnostic | PromptTAD (IEEE RA-L 2025) | The enhanced token sits beside the original; ego-involved vs ego-uninvolved analysis | No detector, boxes or frame labels |
| Optional agent token (outside the claim) | ClearCLIP (ECCV 2024); STPrompt's selection + average | Training-free dense CLIP patches; fixed top-K selection with a plain average | Selection by hand-written agent text prompts instead of motion |

#### 2.3.1 LaGoVAD (ICLR 2026): the baseline

Liu et al. define open-world VAD as a mapping Φ: (V, Z) → frame scores, where the anomaly definition Z is given as text at test time. Frozen CLIP ViT-B/16 frame embeddings pass through a 2-layer RoFormer and are fused with the CLIP text encoding of Z by co-attention. Training uses video-level labels on PreVAD (35,279 videos) with MIL and MIL-alignment losses, dynamic video synthesis and a caption contrastive loss with hard negatives.

- **Strengths.** It has the best zero-shot traffic results in its comparison: 89.56 AUC on TAD, 62.60 on DoTA and 90.41 on MSAD, ahead of VadCLIP and PEL. It stays robust when the definition changes (MSAD drift@5: 85.6 AUC / 40.1 AP). The detector runs no LLM, trains on one RTX 4090, and its code, checkpoint and predictions are public.
- **Weaknesses.** Its only visual input is one global CLIP embedding per sampled frame, so it sees neither motion nor what normal motion looks like: DoTA AUC is 62.60 against 89.56 on TAD. Its traffic test sets are prompted with a generic "car accident" definition, its released configuration selects checkpoints on test scores, and PreVAD's captions were written by Qwen models.
- **Taken into this thesis.** The whole detector, unchanged. The thesis adds a motion input and the LSP stream, and replaces test-score checkpoint selection with the last epoch.

#### 2.3.2 VadCLIP (AAAI 2024): the CLIP-based weakly supervised template

Wu et al. adapt frozen CLIP to weakly supervised VAD with a local Transformer and graph convolution (the LGT-Adapter), a binary classification branch and an alignment branch with learnable prompts (MIL-Align).

- **Strengths.** It established the CLIP-plus-text-prompt design that LaGoVAD, DSANet and Alert-CLIP build on: 88.02 AUC on UCF-Crime and 84.51 AP on XD-Violence on one RTX 3090, with public code.
- **Weaknesses.** It reads appearance only and sits near chance zero-shot on DoTA (49.00–50.93 AUC). In DSANet's analysis, its prototypes assign 84.0% of UCF-Crime road-accident videos to "Normal".
- **Taken into this thesis.** Its design reaches the thesis through LaGoVAD, and it is retrained as the same-feature competitor (Group C, Section 4.3).

#### 2.3.3 VideoMAE V2 (CVPR 2023): the motion encoder

Wang et al. scale masked video autoencoding with dual masking and progressive training (self-supervised on 1.35M clips, then supervised on Kinetics-710), and distil the 1B-parameter ViT-g into smaller ViT-S and ViT-B models.

- **Strengths.** The distilled ViT-S is small (about 22M parameters) and public, and in Simple-TAD's comparison it is the strongest small encoder for traffic anomalies after fine-tuning: AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000, against 52.1 / 48.1 for VideoMAE-S and 49.7 / 43.7 for InternVideo2-S. In this thesis's pilot, its mean-pooled vector alone lifts zero-shot DoTA to 0.81 / 0.75.
- **Weaknesses.** Its latents were never trained to be predictable: Garrido et al. found VideoMAEv2's own prediction error only marginally above a randomly initialised network as a surprise signal, where V-JEPA reached 98% on IntPhys. Its pre-training data contain no driving video, and its features are not text-aligned.
- **Taken into this thesis.** The frozen distilled ViT-S, read as one mean-pooled 384-d vector per step. LSP learns a predictor on these vectors; the weakness above is tested directly (diagnostic D3) with a V-JEPA 2 vector as an outside control.

#### 2.3.4 Simple-TAD (ICCV 2025 Workshops): evidence for the encoder choice

Orlova et al. show that a plain video ViT with one linear head, fine-tuned with frame-level labels, outperforms specialised traffic detectors on DoTA and DADA-2000, and add domain-adaptive pre-training (DAPT) on normal driving video.

- **Strengths.** Simple and strong (DoTA AUC up to 88.4; DoTA → DADA-2000 up to 86.6). It shows that masked video modelling transfers best to traffic anomalies and vision-language pre-training worst. DAPT on BDD100K raises ViT-S AUC-MCC on DoTAref from 52.1 to 54.9 without anomalous data.
- **Weaknesses.** It needs frame-level labels, covers dashcam data only and takes no text. Ego-uninvolved clips lag: 81.6 / 84.1 / 85.2 AUC (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips.
- **Taken into this thesis.** The choice of encoder and clip input, and the DAPT encoder as an ablation.

#### 2.3.5 FLaRA (arXiv 2606.14380): the source of LSP

Caselli et al. forecast future latent representations for accident anticipation. A V-JEPA2 ViT-L encoder, frozen except its last block, encodes a 16-frame, 8 fps context clip; the V-JEPA2 predictor forecasts the latent tokens of the next, non-overlapping 16-frame clip; and a 4-block attentive probe classifies **only the predicted future**. A Smooth-L1 reconstruction loss pulls the predicted tokens toward stop-gradient encoder tokens of the real future, weighted λ = 10 against cross-entropy (swept over {1, 2, 10, 100}). It trains 84M parameters on Nexar on one RTX 4090.

| FLaRA ablation (AP / AUC) | Nexar | DAD | DADA-2000 | DoTA |
| --- | --- | --- | --- | --- |
| Frozen backbone | 80.4 / 81.8 | 22.5 / 77.6 | 90.2 / 81.9 | 97.9 / 85.0 |
| Fine-tuned without L\_rec | 85.5 / 87.0 | 25.6 / 88.0 | 95.6 / 90.1 | 98.7 / 89.4 |
| Encoder only | 86.2 / 87.3 | 35.8 / 80.7 | 95.3 / 89.7 | 98.4 / 87.4 |
| FLaRA + present features | 85.1 / 86.0 | 58.4 / 93.7 | 97.0 / 92.8 | 98.8 / 90.0 |
| **FLaRA** | **86.6 / 87.3** | **67.3 / 93.6** | **97.3 / 93.2** | **98.8 / 90.0** |

- **Strengths.** It beats BADAS on Nexar (AP 86.6 vs 85.2) with about 4× fewer trainable parameters, and transfers to unseen traffic datasets without fine-tuning. The ablation isolates the reconstruction loss as the active ingredient: without it, DAD AP falls from 67.3 to 25.6, because the predictor is then guided only by the classification gradient. Code is public.
- **Weaknesses.** It is supervised, clip-level anticipation, not frame-level detection. Its reconstruction loss is computed on positive clips whose future window often contains the crash, so its predictor learns to forecast anomalies. The loss's effect on DoTA is small (89.4 → 90.0 AUC), and its DoTA metric is a video-level anticipation AUC on a 327-positive / 40-synthetic-negative subset, not comparable with per-video frame AUC.
- **Taken into this thesis.** The mechanism: a latent predictor with a Smooth-L1 loss to fixed targets and a contiguous, non-overlapping horizon, and the lesson that the reconstruction loss, not the extra capacity, carries the gain. **Changed:** offline the future is observed, so LSP reads the gap between observation and prediction instead of classifying the prediction; the loss sees normal videos only; the predictor is detached from the detection loss; and the target is one 384-d vector per step. FLaRA's own rows become controls: "without L\_rec" is R3 and "classify the prediction" is R6.

#### 2.3.6 V-JEPA surprise (Garrido et al., 2025) and Asleep at the Wheel (2026): evidence on latent surprise

Garrido et al. compute surprise as the distance between a frozen V-JEPA predictor's forecast and the encoded future, with no adaptation. *Asleep at the Wheel* trains a surprise predictor on frozen V-JEPA 2 features of real driving video.

- **For the idea.** V-JEPA picks the physically impossible video in a pair 98% of the time on IntPhys (66% GRASP, 62% InfLevel), and the maximum surprise in a video works better than the mean, which matches top-k MIL's max-like pooling.
- **Against it.** Garrido et al. found no significant gain on collision events and only marginal results for VideoMAEv2. On nuScenes, prediction-error novelty reached AP 0.29 against 0.24 chance, the same as an untrained predictor (0.301), while it separated datasets at AUC 0.965; a supervised logistic probe on the same features reached 0.50.
- **Taken into this thesis.** Max-style surprise pooling, and the decision to use surprise as an *input* to a supervised MIL head rather than as the anomaly score. The negative results define three of the pre-training diagnostics (D2–D4) and the same-source test of H3.

#### 2.3.7 STPrompt (ACM MM 2024): fixed priors under weak labels

STPrompt adds a spatial branch to a VadCLIP-style detector. Each of 7×7 sliding-window crops is encoded by CLIP's global token; a patch's motion magnitude is the second-order difference ‖2x\_i − x\_{i−1} − x\_{i+1}‖ across neighbouring frames; the top-12 patches are aggregated and added to the frame feature before a distance-based temporal adapter. Localisation is training-free, with LLM-written descriptions.

- **Strengths.** It localises from video labels and improves temporal-IoU over VadCLIP by about 1.9, 5.7 and 4.5 points on UCF-Crime, ShanghaiTech and UBnormal. Its ablation shows that fixed priors carry weak supervision: on UCF-Crime / UBnormal, no spatial branch 86.13 / 63.00, plain average 86.91 / 62.93, **learned attention only 86.67 / 62.64**, motion selection + average 87.79 / 63.69, full SA² 88.08 / 63.98; a distance-only temporal adapter (88.08) also beat a standard Transformer (86.32).
- **Weaknesses.** Its motion cue assumes a still camera; under forward ego-motion second-order differences are largest on near-field background. Localisation needs an LLM, it covers surveillance data only, and no public code was found.
- **Taken into this thesis.** The principle that the new signal must come from a prior not learned from video labels, and its fusion slot (added to the frame feature before the temporal encoder). Its second-order difference is the residual of a *fixed* interpolation predictor; LSP replaces it with a *learned* one, and the fixed version is baseline R2.

#### 2.3.8 PromptTAD (IEEE RA-L 2025): object-level evidence from a detector

Qiu et al. extend MOVAD with YOLOv9 box prompts, instance-wise and relation-wise aggregation (12 learnable query tokens) and an instance-level loss on Hungarian-matched ground-truth boxes, trained with frame labels.

- **Strengths.** It reaches 83.9 AUC on DoTA, reports STAUC, and gains most on distant, off-centre and ego-uninvolved anomalies (DADA-2000 ego-uninvolved 70.4 vs 64.0 for TTHF).
- **Weaknesses.** It needs frame labels, anomalous-object boxes and a detector; when detection fails in dark scenes it "degenerate\[s\] into the baseline". No public code.
- **Taken into this thesis.** Two ideas that survive the constraints: the enhanced token is concatenated beside the original rather than replacing it (LSP adds e\_t beside v\_t), and results are split by ego-involved vs ego-uninvolved anomalies. Its detector-free remnant, learnable queries over a dense map, is what MQR tried and is not reused.

#### 2.3.9 Methods used as comparison rows

These methods contribute no part of the architecture; they supply published numbers (Section 4.4).

| Method (venue) | Approach | Strength | Weakness | Role in this thesis |
| --- | --- | --- | --- | --- |
| WSAL (IEEE TIP 2021) | High-order context encoding of TSN features; introduced the TAD dataset | 89.64 AUC\_O / 61.66 AUC\_A on TAD | Hand-crafted cues; fixed 32 segments | P1 row on both metrics |
| UMIL (CVPR 2023) | Confident / ambiguous snippet split and clustering against context shortcuts; X-CLIP fine-tuned | 92.93 AUC\_O / 65.82 AUC\_A on TAD | Needs backbone fine-tuning (90.71 frozen) | P1 row; source of the TAD split |
| π-VAD (CVPR 2025) | Induces five modalities from RGB at training time | MSAD 88.68 AUC / 71.26 AP; flow most useful | Five teacher backbones; no public code | P2 row |
| GS-MoE (ICCV 2025) | Gaussian pseudo-labels plus one expert per class | MSAD 87.72 AUC / 68.26 AP / 69.54 AUC\_A / 76.68 AP\_A | Needs class labels; misses very small accidents | P2 row |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with boxes and region captions | Lifts the VadCLIP framework on MSAD from 87.45 / 71.09 to 89.63 / 78.24 | Needs boxes and VLM-written captions | P2 row |
| WS-DoTA (ECCV 2024 Workshop) | Weakly supervised DoTA with D²-City normals; copy-last CLIP change, sigmoid(F) + \|ΔF\| | The only weakly supervised dashcam study; lifts RTFM from 57.9 to 78.2 AUC | Own \~1,031-clip subset; AUC only | Flagged P3 row; its cue is baseline R1 |
| MAMTCF (arXiv 2307.14575) | Normal-only flow reconstruction + future object localisation | DoTA 76.6 AUC; non-ego 78.6 | Pooled raw scores; detector and flow | Context row for prediction-based detection |

### 2.4 Research gaps

Four gaps follow from the review. The thesis addresses each one, and its novelty claim is deliberately narrow.

1. **No motion-aware foundation features in weakly supervised traffic VAD.** LaGoVAD and VadCLIP read CLIP frames; the masked-video evidence of Simple-TAD comes from frame-level supervision. *Addressed by the motion stream (backbone change), already shown to lift zero-shot DoTA to 0.81 / 0.75.*
2. **Weak supervision learns no model of normal dynamics.** The normality priors used under video-level labels are fixed change cues: WS-DoTA's copy-last CLIP difference and STPrompt's second-order interpolation residual. Latent future prediction with a reconstruction loss (FLaRA) has been used only for supervised, clip-level anticipation, trained on futures that contain the crash. *Addressed by LSP: a predictor learned from normal videos only, whose residual generalises both fixed cues.*
3. **Latent surprise is untested inside a supervised detector on traffic data.** As a standalone score it is near chance within one driving dataset and detects dataset source instead (*Asleep at the Wheel*), while a supervised probe on the same features does better. No work found feeds it into a MIL detector. *Addressed by feeding the LSP residual to LaGoVAD through a zero-initialised gate, with a same-source test and a source classifier against the domain-shift artefact.*
4. **Traffic results are hard to compare.** Most rows are AUC-only, two TAD test lists circulate, and DoTA numbers mix four protocols. *Addressed by four split-consistent, multi-metric protocols with each row's protocol labelled (Section 4.4).*

The closest prior work bounds the novelty claim:

| Closest work | What it does | How LSP differs |
| --- | --- | --- |
| FLaRA (arXiv 2606.14380) | Predicts V-JEPA2 future tokens and classifies the prediction; clip-level supervised anticipation | Offline detection; observed-minus-predicted residual; normal-only and detached; one VideoMAE vector; MIL under video labels |
| BADAS (arXiv 2510.14876) | Fine-tuned V-JEPA2 with an attentive probe on observed context; supervised ego-collision anticipation | No predictor; supervised anticipation |
| Garrido et al. (arXiv 2502.11831) | Frozen V-JEPA surprise, zero-shot, synthetic intuitive physics | Trained predictor on driving video; surprise is an input to a supervised head |
| Asleep at the Wheel (arXiv 2608.01336) | Trained predictor on frozen V-JEPA 2 driving features as a standalone novelty score; at chance within one dataset | Uses its finding by embedding the residual in MIL; tests for the source confound |
| Latent Clarity (arXiv 2607.03558) | V-JEPA 2 prediction error as a gate (z > 2), text prototypes in a VLM embedding space, surveillance data | Residual is a learned evidence stream; no LLM or VLM; MIL; traffic |
| WS-DoTA (arXiv 2408.05562) | Copy-last change of CLIP features in weak MIL, DoTA in-domain | Learned normal-trained predictor; copy-last is baseline R1 |
| STPrompt SA² (ACM MM 2024) | Fixed interpolation residual on patch crops for spatial selection | Temporal, learned, global vector; interpolation is baseline R2 |
| AnoPred, FOL, MAMTCF | Pixel, flow or box prediction error as an unsupervised score | Latent target from a frozen video foundation model; supervised fusion |
| RiskProp (arXiv 2603.27165) | Future risk-score regularisation under weak labels, anticipation | Predicts features, not scores; detection |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that injects the residual of a predictor trained on normal driving in a frozen video-foundation latent into a language-guided MIL detector, and the first test of whether learning that predictor beats the fixed change cues used in prior weak supervision.** Latent prediction and surprise are not claimed as new. Search coverage of 2026 arXiv is incomplete, so the claim will be re-checked before submission; the two 2026 spatial-localisation preprints ("Localizing to Debias", arXiv 2608.12045; "Learning Where and When", arXiv 2606.29498) use neither future prediction nor DoTA and bear only on the optional arm.

## 3. Proposed model

The proposed model, **LaGoVAD-LSP**, is LaGoVAD with two changes: a frozen VideoMAE motion stream read as one vector per step (the backbone change) and the Latent Surprise Predictor (the novel component). Every other part of LaGoVAD stays as released, so any gain can be traced to these two changes.

### 3.1 Base architecture

**LaGoVAD, unchanged parts.** The released code fixes each part:

- **Sampling.** One frame in eight is kept (`interval=8`); for 10 fps dashcam sets (DoTA, CCD) every provided frame is used, as LaGoVAD does for DoTA.
- **CLIP stream.** A frozen CLIP ViT-B/16 image encoder gives one 512-d embedding c\_t per sampled frame t, from a 224×224 resize without cropping.
- **Temporal encoder.** A residual RoFormer: 2 layers, 4 heads, hidden size 512, local window of 25 steps. Training sequences are resampled or padded to 512 steps.
- **Text branch.** The frozen CLIP text encoder with 32 learnable soft-prompt tokens encodes the anomaly definition Z into τ ∈ R^512.
- **Fusion.** A 2-layer co-attention Transformer lets video steps and text tokens attend to each other.
- **Heads.** A 1-D convolution head (kernel 9) scores the pre-fusion and post-fusion features; a learned weight w = σ(10·a), a initialised at 0, blends them, and the step score is the sigmoid of the blend.
- **Losses.** Top-k MIL binary cross-entropy (k = max(⌊L/16⌋, 1) for L steps), a multi-class MIL alignment loss, dynamic video synthesis (an abnormal clip inserted among 1–5 kNN-retrieved normal clips, giving frame pseudo-labels), and a caption contrastive loss with hard negatives (temperature 0.02).

**Backbone change: the motion stream.** For each sampled step t, a 16-frame clip at 10 fps centred on t (1.6 s, Simple-TAD's traffic input) passes through a frozen **VideoMAE V2-S** encoder: the ViT-S distilled from VideoMAE V2-g on Kinetics-710, D = 384. Its 8×14×14 last-layer tokens are averaged into one vector and added to the CLIP embedding through a gate that starts closed:

```latex
v_t = \frac{1}{1568} \sum_{j=1}^{1568} X_{t,j} \in \mathbb{R}^{384}, \qquad x_t = c_t + \tanh(\alpha)\, W\, \mathrm{LN}(v_t), \qquad \alpha_{\text{init}} = 0,\; W \in \mathbb{R}^{512 \times 384}
```

The fused x\_t replaces c\_t as the RoFormer input; the kNN retrieval of dynamic video synthesis still uses CLIP features. Trained on Nexar + CCD, this model reaches **0.81 macro / 0.75 micro per-video min-max AUC zero-shot on DoTA** (preliminary, one configuration; to be repeated with 3 seeds). Reading the token grid with learned queries instead of averaging it gave no gain (Section 3.2.1), so the stream is fixed at this one vector.

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips and distilled on Kinetics-710; none of the seven traffic datasets used here is among them. Simple-TAD's driving-adapted DAPT-VideoMAE-S (BDD100K only) is kept as an encoder ablation.

&#91;embedded content: LaGoVAD-LSP architecture · the new component highlighted\]

Only the highlighted predictor and its stage-1 loss are new; the motion stream is the backbone change, and every other box is LaGoVAD as released.

### 3.2 Novel component: Latent Surprise Predictor (LSP)

LSP is a small causal transformer that sees past motion vectors and predicts the vector one non-overlapping clip ahead. It is trained with FLaRA's Smooth-L1 latent loss on normal videos only and frozen before the detector is trained. What reaches the detector is how much the observed motion departs from the predicted normal motion.

#### 3.2.1 Why MQR was dropped, and three design rules

The previous component, the Motion Query Readout, read the 7×7 VideoMAE token grid with eight zero-initialised, key-free queries (Efficient Probing, ICLR 2026) and added a spatial top-k MIL loss. Its attention entropy stayed at its uniform start for all of training, and the model without it reached the same DoTA result. The initialisation explains this rather than bad luck:

- **It started at a solution the host already had.** Zero queries with an identity value projection give exactly the mean of the tokens, which already delivered the full gain, and nothing in the loss pushed away from it.
- **Its learning signal was too weak.** At uniform attention the query gradient is a covariance across 49 near-identical tokens divided by roughly 49·√384 ≈ 970; masked-model tokens are close to maximal attention entropy (MAE: 5.03 against a maximum of 5.27, Beyond \[cls\], ICCV 2025). Softmax scores that start uniform receive gradients 0.5–1 orders of magnitude smaller than value weights (Hoffmann et al., arXiv 2310.12956). Efficient Probing trains its queries under full ImageNet supervision with LARS at learning rates of 0.1–5.0; MQR had video labels and AdamW at about 1e-4.
- **The host diluted it further.** Every gradient passed through the gate tanh(α); top-k MIL updates only about L/16 steps per video; and the max over eight identical queries sent the spatial loss to one query.
- **The metric could not see it.** Per-video min-max AUC on abnormal-only DoTA rewards changes in *when* the score rises inside a clip, not uniform sharpening.

These points are an analytical diagnosis, to be confirmed from the logged query-gradient norms and tanh(α). They agree with STPrompt's ablation, where learned attention scored below a plain average. They give three rules for the replacement: (1) it must have a **learning signal that does not depend on the MIL gradient**; (2) it must **not be able to collapse back to a solution the host already has** unless its own loss pushes it away; and (3) it must change the **temporal shape** of the score within a clip. LSP is built to satisfy all three.

#### 3.2.2 Definition

Let τ\_s be the step spacing in seconds. Normalise the motion vector without affine parameters, as FLaRA normalises its targets with the encoder's final LayerNorm, and choose the horizon Δ (in steps) so that the target clip does not overlap the last context clip:

```latex
u_t = \mathrm{LN}_{\text{no-affine}}(v_t), \qquad \Delta = \left\lceil 1.6\,\mathrm{s} \,/\, \tau_s \right\rceil
```

The predictor g\_φ reads a context of K steps ending Δ steps before the target, and predicts the target **as a correction to copy-last**:

```latex
\hat u_t = u_{t-\Delta} + g_\phi\big(u_{t-\Delta-K+1}, \ldots, u_{t-\Delta}\big), \qquad K \in \{8, 16\}
```

g\_φ has two pre-LN causal transformer layers of width 384 with 6 heads and a 4× MLP, learned positional embeddings over the K positions, and a linear readout from the last position: about 3.5M parameters, roughly 25× fewer than FLaRA's 84M trainable parameters. From the prediction come the residual, the surprise, the copy-last surprise and their log ratio, with ℓ the Smooth-L1 function (β = 1, as in FLaRA):

```latex
r_t = u_t - \hat u_t, \qquad s_t = \tfrac{1}{384}\textstyle\sum_d \ell(r_{t,d}), \qquad p_t = \tfrac{1}{384}\textstyle\sum_d \ell(u_{t,d} - u_{t-\Delta,d}), \qquad \rho_t = \log\frac{s_t + \epsilon}{p_t + \epsilon}
```

p\_t is the surprise of the trivial copy-last predictor, and ρ\_t asks whether a step is less predictable than its own raw change suggests; ρ\_t is zero whenever the predictor is no better than copy-last. s\_t is standardised to ŝ\_t with the mean and standard deviation of s on held-out normal training videos. The evidence vector enters the existing fusion as a second gated term:

```latex
e_t = m_t \cdot \big[\,\mathrm{LN}(\mathrm{sg}(r_t))\,;\; \hat s_t\,;\; \rho_t\,\big] \in \mathbb{R}^{386}, \qquad x_t = c_t + \tanh(\alpha)\,W\,\mathrm{LN}(v_t) + \tanh(\beta)\,W_e\,e_t
```

Here sg is stop-gradient, and the mask m\_t is 0 for the first Δ + K − 1 steps of each source video, where no full context exists. Everything downstream of x\_t (the RoFormer, text co-attention, heads and all LaGoVAD losses) is unchanged.

#### 3.2.3 Initialisation

- **Predictor.** The readout layer of g\_φ is zero-initialised, so at the start û\_t = u\_{t−Δ}, s\_t = p\_t and ρ\_t = 0: the predictor *starts* as copy-last, and its own loss must move it away.
- **Gate.** β = 0, so the detector starts as exactly the current 0.81 / 0.75 model. W\_e ∈ R^{512×386} is Xavier-random, so ∂L/∂β = gᵀW\_e e\_t is non-zero from the first step; β opens only if the stream lowers the detection loss. This is the same mechanism as the host's α gate, which demonstrably opened for v\_t in this codebase.
- **Fallback.** β gets no weight decay and a 10× learning rate. If tanh(β) is still below 0.01 after the first epoch, training restarts with tanh(β₀) = 0.1, the remedy the MQR diagnosis prescribes for a gate that never opens.

#### 3.2.4 Why LSP should help on traffic video

Five pieces of evidence point the same way:

1. **The reconstruction loss is what transfers.** In FLaRA, removing L\_rec drops DAD AP from 67.3 to 25.6 and DADA-2000 AUC from 93.2 to 90.1, although the predictor's capacity is unchanged. LSP keeps that loss as its only training signal.
2. **Fixed priors beat priors learned from weak labels.** STPrompt's learned attention scored below a plain average and its fixed motion prior did most of the work. LSP's prior is learned, but from normal data and a dense loss, then frozen, so the detector sees it as fixed.
3. **The fixed change cues that work are special cases.** WS-DoTA's best transform (copy-last CLIP change) lifted RTFM to 78.2 on DoTA, and STPrompt's motion magnitude is an interpolation residual. A predictor trained on normal driving should flag steps that merely *change* less often than steps that change *unexpectedly*, which matters most under ego-motion, where everything changes.
4. **Surprise helps most inside a supervised head.** *Asleep at the Wheel* found standalone latent surprise at chance (AP 0.29 vs 0.24) while a supervised probe on the same features reached 0.50; LSP is an input to MIL, not the score.
5. **It addresses what the metric measures.** Surprise is temporal by construction and should peak at the transition into an anomaly, which is what per-video min-max AUC rewards, whereas MQR's spatial readout could not move it.

The obvious objection is that û\_t is a function of past vectors the RoFormer already sees. In principle the RoFormer could learn it, but it sees only MIL gradients from about L/16 steps per video and has no reason to learn a calibrated model of *normal* dynamics. Ablation R3 (the same predictor without L\_rec, trained through MIL) tests this objection directly.

| Property | Motion stream (preliminary) | + LSP |
| --- | --- | --- |
| Extra parameters | — | about 3.5M (predictor) + 0.2M (gate and W\_e) |
| Extra extraction compute | — | none: same VideoMAE vector |
| Storage per step (fp16) | 0.75 KB | 0.75 KB (e\_t is recomputed from cached v\_t) |
| Extra training stage | — | predictor pre-training on normal videos, minutes on cached vectors |
| Extra loss | — | Smooth-L1 latent reconstruction, normal videos only |
| VideoMAE input | one 384-d vector per step | unchanged |

### 3.3 Training strategy

Training uses video-level annotations only. The predictor learns from videos labelled normal; the detector then trains with LaGoVAD's own schedule, so the baseline, the motion stream and LSP differ only in the new stream.

**Training data.** Only TAD, MSAD, CCD and Nexar contain both normal and abnormal videos, so only they are used for training; DoTA, DADA-2000 and MM-AU contain accident videos only and serve as test sets. Each protocol's predictor is trained on that protocol's own normal videos.

| Protocol | Training set | Normal / abnormal videos | LSP predictor data | Anomaly text (hand-written) |
| --- | --- | --- | --- | --- |
| P1, TAD | TAD train, UMIL split | 210 / 190 | 210 normal TAD videos | TAD's 7 category names and definitions |
| P2, MSAD | MSAD weak-protocol train | 360 / 120 | 360 normal MSAD videos | MSAD's 11 anomaly types |
| P3, dashcam | Nexar labelled set + CCD official train split | Nexar 750 / 750; CCD train part of 3,000 normal / 1,500 crash | Nexar negatives + CCD normal clips | Collision, near-miss and car-crash definitions |
| P4, cross-setting | None: reuses the P2 and P3 models | — | — | Each test set's own definitions |

**Captions.** Every model uses the same hand-written caption template per category, so no LLM-written text enters training. A no-caption variant is reported as an ablation (removing this loss costs LaGoVAD 6.3 AUC on TAD zero-shot).

**Shortcut control.** CCD's crash clips and normal clips (from BDD100K) come from different sources, so a model could learn the source instead of the anomaly; a surprise predictor trained on CCD normals is especially exposed to this. P3 is therefore also reported with Nexar-only training, where positives and negatives share one dashcam network.

**Stages.**

1. **Cache features.** Extract CLIP frame features with LaGoVAD's script and the VideoMAE V2-S mean vector per step once, stored as memory-mapped fp16 arrays.
2. **Pre-train the predictor (stage 1).** Train g\_φ on the protocol's normal videos only, over random windows, with the loss below. Hold out 10% of normal videos for early stopping and for the statistics that standardise s\_t. Positive videos are excluded because their normal lead-in can only be isolated with timing labels.
3. **Train the detector (stage 2).** Freeze g\_φ, compute e\_t for every original video, and train LaGoVAD with its original losses; the only new parameters are β and W\_e. All rows of the ablation ladder start from the same random initialisation of LaGoVAD's trainable parts, with 3 seeds.
4. **Optional joint fine-tuning (stage 3).** Unfreeze g\_φ and add λ·L\_rec on normal videos to the LaGoVAD loss. Because r\_t is stop-gradiented, the two losses touch disjoint parameters and λ has essentially no effect; FLaRA's λ sweep {1, 2, 10, 100} matters only in ablation R8, where MIL gradients reach the predictor.

```latex
\mathcal{L}_{\text{rec}} = \frac{1}{|\mathcal{T}_{\text{neg}}|} \sum_{t \in \mathcal{T}_{\text{neg}}} \frac{1}{384} \sum_{d} \ell\big(\hat u_{t,d} - u_{t,d}\big), \qquad \mathcal{L}_{\text{stage 2}} = \mathcal{L}_{\text{LaGoVAD}}
```

**Dynamic video synthesis.** LaGoVAD splices clips from different videos. A predictor run on a spliced sequence would produce a large surprise at every seam, and MIL would learn "seam = anomaly". e\_t is therefore computed per original video and spliced together with c\_t and v\_t, with the mask m\_t restarting at each seam.

**No EMA target.** JEPA-style training needs an EMA target encoder to avoid collapse when the encoder is trainable (without it, a 2026 study saw the surprise spread collapse from 0.602 to 0.014). Here VideoMAE is frozen, so the targets are constants and collapse is impossible.

**Settings.**

| Setting | Value | Source |
| --- | --- | --- |
| Detector optimiser | AdamW, batch 64, learning rate 5e-5, cosine, 40 epochs | LaGoVAD |
| Predictor loss | Smooth-L1, β = 1, mean-reduced, to stop-gradient normalised targets | FLaRA code |
| Horizon Δ | non-overlapping, at least one 1.6 s clip ahead; swept Δ and 2Δ | FLaRA's contiguous future; rule is this proposal's |
| Context K | 8 or 16 steps | this proposal |
| Predictor optimiser | AdamW 3e-4, weight decay 0.05, cosine, 50 epochs | this proposal (FLaRA's 2.5e-5 is for fine-tuning a pretrained predictor) |
| Gate β | 0, no weight decay, 10× learning rate | mirrors the host's α |

Δ, K and the β learning-rate multiplier are chosen once from diagnostics D1–D2 and a held-out 15% of the P3 training videos scored by video-level AUC, then frozen for every protocol. No hyperparameter is chosen on a test list.

**Checkpoint selection.** Every model, baselines included, reports the last epoch of the fixed schedule.

**Compute.** The predictor trains on cached 384-d vectors (a few hundred MB for all P3 normals) in minutes to an hour on one GPU. Stage 2 costs the same as the motion-stream model; one 24 GB GPU suffices.

### 3.4 Inference

At test time the model takes a video and an anomaly definition and returns a score per sampled step. No LLM runs at any point.

1. **Sample** one frame in eight (every frame for 10 fps dashcam sets).
2. **Encode** each sampled frame with frozen CLIP, and the 16-frame clip centred on it with the frozen VideoMAE V2-S, averaged to one 384-d vector.
3. **Predict** each step's vector with the frozen LSP from the K steps ending Δ steps earlier, and form e\_t.
4. **Fuse** c\_t, the gated v\_t and the gated e\_t.
5. **Score** with LaGoVAD's RoFormer, text co-attention and heads, given the definition text. Each step's score is assigned to the frames it stands for.

**Definitions.** Every main result uses LaGoVAD's released definitions (TAD as {"Normal", "Car Accident"}; DoTA's "Abnormal" expands to the same car-accident texts). A separate analysis prompts each of TAD's 7 categories with its own hand-written definition.

**Score post-processing.** DoTA and DADA-2000 contain only abnormal clips, so their scores are min-max normalised per video before pooling, following LaGoVAD's `offline_dota_eval.py`. TAD, MSAD and CCD scores are used raw.

**Cost.** LSP and LaGoVAD's heads are negligible next to the encoders: one VideoMAE ViT-S clip (about 57 GFLOPs) runs per sampled step. LSP itself is causal, but the RoFormer looks both ways, so the model is offline.

### 3.5 Optional arm outside the main claim: a text-selected ClearCLIP agent token

This arm changes the CLIP branch, is not part of LaGoVAD-LSP or the novelty claim, and runs only after the main ablation ladder. It targets the small-agent problem (property 2) with a fixed, language-defined selection instead of learned attention, to avoid MQR's failure.

Raw last-layer CLIP patches localise poorly; ClearCLIP's training-free change to the last block lifts Cityscapes mIoU from 5.5 to 30.0 in a single forward pass (SCLIP: 32.2). Let P\_{t,j} be the 14×14 ClearCLIP patch features of frame t, 𝒜 a hand-written list of agent prompts ("a car", "a truck", "a motorcycle", "a pedestrian", "a cyclist") and ℬ background prompts ("road", "sky", "building", "tree"), encoded once by the frozen text encoder:

```latex
a_{t,j} = \max_{q \in \mathcal{A}} \cos(P_{t,j}, \tau_q) - \max_{b \in \mathcal{B}} \cos(P_{t,j}, \tau_b), \qquad c^{\text{loc}}_t = \frac{1}{K_s} \sum_{j \in \text{top-}K_s(a_t)} P_{t,j}, \qquad x_t \mathrel{+}= \tanh(\gamma)\, W_c\, \mathrm{LN}(c^{\text{loc}}_t)
```

with γ = 0, W\_c Xavier-random and K\_s ≈ 16 of 196 patches. No parameter in the selection is learned, and the plain average follows STPrompt, where selection + average came within 0.3 AUC of its best variant. Three limits: at a 224 resize a distant car falls under one 16-px patch, so native-resolution tiles (about 6 extra CLIP passes per frame) may be needed; raw CLIP patches cannot tell normal from anomalous (similarities overlap near 0.5 in AnomalyCLIP), which is why the prior selects agents rather than anomalies; and the arm should be judged on DoTA's small-box and ego-uninvolved clips and with STAUC, not on overall AUC. It is reported as ablation row R12.

## 4. Experimental plan

The plan trains with video-level labels only, runs cheap go/no-go diagnostics on cached features before any full training, compares against published rows on identical test lists, and adds every second metric by re-scoring released checkpoints.

### 4.1 Dataset preparation

The thesis uses seven traffic datasets. TAD (fixed camera) carries the main comparison and MSAD (multi-scene surveillance with a traffic-accident slice) a second one. CCD and Nexar train the dashcam model and its predictor, DoTA and DADA-2000 test it zero-shot, and DoTA's ego flag and categories drive the analyses.

| Dataset | View | Size and split | Labels | Role | Source |
| --- | --- | --- | --- | --- | --- |
| TAD | Fixed CCTV, some first-person | 500 videos; UMIL split 400 train (210 normal / 190 abnormal), 100 test (40 / 60); 7 categories | Video labels (train), anomaly interval (test) | P1 train, predictor and test; P4 zero-shot test | Frames and split files via the UMIL repository; native fps unknown |
| MSAD | Fixed, 14 scenes | 720 videos at 30 fps; weak protocol 360 normal + 120 abnormal train, 120 + 120 test (20 traffic accidents) | Video labels (train), frame labels (test) | P2 train, predictor and test; traffic slice; P4 zero-shot test | MSAD project page (videos gated) |
| CCD | Dashcam | 1,500 crash + 3,000 normal clips (normal from BDD100K), 50 frames at 10 fps, official split | Video labels; accident onset | P3 train and predictor; in-domain and same-source test | GitHub (Cogito2012/CarCrashDataset) |
| Nexar | Dashcam | 1,500 labelled clips (750 positive + 750 normal), about 40 s each | Video labels; event and alert times (analysis only) | P3 train and predictor; same-source held-out test | Hugging Face |
| DoTA | Dashcam | 4,677 clips (3,275 train / 1,402 test), 10 fps, abnormal only | Anomaly window; 9 categories × ego-involved / ego-uninvolved; agent boxes | P3 zero-shot test; diagnostics on the train split only; boxes only for the optional arm | GitHub (MoonBlvd/Detection-of-Traffic-Anomaly) |
| DADA-2000 | Dashcam | 2,000 accident clips at 30 fps | Accident intervals; ego / non-ego | P3 zero-shot test | Simple-TAD's split files |
| MM-AU | Dashcam | 11,727 accident videos pooled from CCD, A3D, DoTA and DADA-2000 | Accident start, collision, end | Optional zero-shot test, after removing videos shared with CCD, DoTA and DADA-2000 | MM-AU release (CVPR 2024) |

**Weights to download** (all public): CLIP ViT-B/16 and VideoMAE V2-S (`vit_s_k710_dl_from_giant`); LaGoVAD's `best.ckpt` and `pred_results.pkl`, used only as a reference row; VadCLIP, PEL4VAD and DSANet checkpoints for zero-shot re-scoring; RTFM and MGFN MSAD checkpoints for the traffic slice; MOVAD, Simple-TAD and FLaRA checkpoints as frame-supervised or anticipation context; Simple-TAD's DAPT-VideoMAE-S (BDD100K only, CC BY-NC 4.0) for the encoder ablation; and a V-JEPA 2 encoder, used only as an outside diagnostic control (D3) and never inside the model. ClearCLIP is a training-free change to the CLIP code, needed only for the optional arm. None of these links has been test-downloaded yet; that check is the first task.

### 4.2 Dataset processing

Every dataset goes through the same pipeline, so one feature cache serves the baseline, the motion stream, LSP and all ablations.

1. **Decode and resample.** The CLIP stream keeps LaGoVAD's convention: one frame in eight, or every frame for the 10 fps sets. The VideoMAE stream resamples every video to 10 fps, so a 16-frame clip spans 1.6 s; TAD's clips take 16 consecutive released frames and their duration is reported as unknown.
2. **Cut clips.** For each CLIP step at time t₀, take the 16 frames centred on t₀, repeating edge frames at video boundaries. Resize to 224×224 without cropping.
3. **Extract CLIP features** with LaGoVAD's `extract_feat_clip.py` for all seven datasets, so features match the released checkpoint.
4. **Extract the VideoMAE V2-S vector.** Average the 8×14×14 last-layer tokens into one 384-d vector per step and store it in fp16 (0.75 KB per step), as one array per video. Check whether the preliminary code applied the final `fc_norm`, and keep that choice fixed.
5. **Record the step spacing.** Store τ\_s per dataset (0.1 s for the 10 fps dashcam sets; 8 frames at the native rate for the others), which sets the horizon Δ = ⌈1.6 s / τ\_s⌉.
6. **Build labels.** Training uses video-level labels, category indices and caption templates: TAD from UMIL's `TAD_train.txt`, MSAD from its weak split, CCD from its official split, Nexar from its positive / negative labels. The same video labels select the predictor's normal videos. Test labels are frame-level: TAD intervals, MSAD frame labels, CCD accident onsets, DoTA anomaly windows and DADA-2000 intervals.
7. **Prepare DoTA's analysis labels.** For each clip keep `ego_involve`, the category, the anomaly window (split into early, middle and late thirds) and the precursor frames. Agent boxes are kept only for the optional arm's STAUC and pointing game.
8. **Fix test lists.** LaGoVAD's lists (DoTA 1,402 clips; its 100-video TAD list) for zero-shot comparability; UMIL's TAD test list for P1 and as a second zero-shot column; MSAD's weak-protocol test set; CCD's official test split; Simple-TAD's DADA-2000 test split; DoTA's official 1,305-video list for the optional arm's STAUC.
9. **Hold out validation data.** 15% of the P3 training videos, stratified by label and source, for the one-time hyperparameter choice, and 10% of each protocol's normal videos for predictor early stopping and surprise statistics.

**Storage of the motion cache** (GB, fp16, one 384-d VideoMAE V2-S vector per step): TAD 0.05, CCD 0.17, Nexar 0.16, DoTA (all clips) 0.56, DADA-2000 0.06. These are estimates; TAD's quoted duration conflicts with its frame count, so its figure may be off by up to about 5×. Dropping the token grid that MQR needed (27.5 GB for DoTA alone) removes disk as a cost.

**Leakage checks before any training.**

- Seventy-six of LaGoVAD's 100 TAD test videos are in UMIL's TAD training list, so a model trained on UMIL's split is never scored on LaGoVAD's TAD list.
- MM-AU pools CCD, A3D, DoTA and DADA-2000 videos. It is never used for training, and as a test set only after removing shared videos.
- The predictor never sees DoTA, DADA-2000, MM-AU or any test video; DoTA's training split is used only for read-only diagnostics, never to fit or tune anything.
- A near-duplicate check with CLIP-feature similarity runs between every training set (TAD, MSAD, CCD, Nexar) and every test set.
- LaGoVAD's published zero-shot numbers are reproduced from `pred_results.pkl` before anything is trained.

### 4.3 Diagnostics and baseline running

Baselines run in three groups, preceded by a set of diagnostics that check LSP's premise on cached features before the detector is trained.

**Step 0: go/no-go diagnostics (hours, cached vectors and the 3.5M-parameter predictor only).** The literature makes clear that latent surprise is not a safe bet on dashcam video, so these checks decide whether stage 2 is worth running. Decisions use same-source held-out splits of the training data (Nexar and CCD test videos); DoTA's training split is a secondary, disclosed check, and the 1,402 test clips are never used for design decisions.

| Diagnostic | Risk it tests | When | Pass criterion (proposed) | If it fails |
| --- | --- | --- | --- | --- |
| D1: cos(u\_t, u\_{t+Δ}) as a function of Δ on Nexar normals | Copy-last trivial solution: overlapping clips make the identity the best predictor | Before stage 1 | Chosen Δ lies past the overlap plateau | Increase Δ |
| D2: held-out normal L\_rec of LSP vs copy-last, constant-velocity (2u\_{t−Δ} − u\_{t−2Δ}) and context-mean predictors | The predictor learns nothing a fixed rule does not | After stage 1 | At least 10% below the best fixed predictor | Increase K or normal data; reconsider |
| D3: standalone video-level AUC of max\_t ŝ\_t and of ρ\_t, against copy-last p\_t; per-video AUC on the DoTA train split | VideoMAE latents carry too little predictable structure (Garrido et al.) | After stage 1 | ŝ\_t beats p\_t on the Nexar / CCD test and the DoTA train split | Run a V-JEPA 2 vector as an outside control; stop if both fail |
| D4: logistic classifier predicting source dataset (Nexar, CCD, DoTA lead-ins) from per-video surprise statistics | Surprise detects domain shift (Asleep at the Wheel: source AUC 0.965) | After stage 1 | Source AUC not far above anomaly AUC | Add a video-median-subtracted ŝ\_t; stop if the confound dominates |
| D5: onset-aligned surprise curves; share of DoTA clips whose surprise maximum falls in the lead-in; ego-involved vs ego-uninvolved; correlation of ŝ\_t with p\_t on normals | Ego-motion false surprise compresses the true peak under min-max normalisation | After stage 1 | Fewer lead-in maxima than p\_t, and lower correlation with raw change | Heteroscedastic head (R9) |
| D6: trajectory of tanh(β), gradient norm of W\_e, AUC in early, middle and late thirds of each anomaly window | The gate never opens; surprise only fires at onset | During and after stage 2 | β opens; no late-window collapse | tanh(β₀) = 0.1; bidirectional surprise (R11) |

D3 uses the maximum surprise per video because Garrido et al. found the maximum works better than the mean for single-video decisions. The V-JEPA 2 control separates "the method fails" from "VideoMAE latents are not predictable enough" and stays outside the model, which keeps the one-VideoMAE-vector constraint.

**Group A: reproduce LaGoVAD and the motion stream.**

| Row | Model | Training | Purpose |
| --- | --- | --- | --- |
| B0 | LaGoVAD, released checkpoint | PreVAD (by its authors) | Reproduce the published zero-shot numbers from `pred_results.pkl` (TAD 89.56, DoTA 62.60, MSAD 90.41), then re-score on UMIL's TAD list, DADA-2000, CCD and the MSAD traffic slice with AUC, AP and AUC\_A |
| B1 | LaGoVAD, retrained | Each protocol's training set, 3 seeds | Fair baseline with exactly the thesis data, features and schedule; separates the gain of domain-matched data from the gain of motion |
| B2 | LaGoVAD + motion stream (one VideoMAE V2-S vector) | Same, 3 seeds | The preliminary 0.81 / 0.75 model; LSP's starting point (ablation row R0) |

**Group B: published methods with released weights, re-scored for a second metric.** VadCLIP and DSANet (UCF-Crime and XD-Violence checkpoints, on the same CLIP ViT-B/16 features); PEL4VAD (with I3D features extracted for TAD, DoTA and DADA-2000); RTFM and MGFN (MSAD checkpoints, for the traffic slice); MOVAD and Simple-TAD (DoTA and DADA-2000, frame-supervised context only); and FLaRA's public checkpoint, re-scored frame by frame under the thesis protocol so that it can appear in the same table. Each runs on the same test lists and is scored with AUC, AP and, where defined, AUC\_A.

**Group C: a same-feature competitor retrained in-domain.** VadCLIP's public code is retrained on P1, P2 and P3 with the thesis's CLIP features. A VadCLIP + WS-DoTA-style variant (copy-last CLIP change appended) is also retrained on P3, as the fixed-cue competitor from the literature. Methods without released weights are quoted from their papers only: UMIL and WSAL on TAD; π-VAD, GS-MoE and Alert-CLIP on MSAD; MultiDomain and WS-DoTA; and the frame-supervised TTHF, PromptTAD and ISCRTAD.

### 4.4 Evaluation protocols

Four protocols cover in-domain and zero-shot use on fixed-camera and dashcam video. P1 meets the comparison rule (at least two published methods on at least two metrics, same fixed-camera traffic data) without recomputation. P2 adds four-metric rows on MSAD. P3 and P4 meet the rule on DoTA, TAD and MSAD once released checkpoints are re-scored for the second metric.

**Metrics.**

- **AUC\_O / AP:** frame-level ROC AUC and average precision, pooled over all test frames.
- **AUC\_A / AP\_A:** the same, over abnormal test videos only; they measure localisation inside abnormal videos.
- **DoTA and DADA-2000 macro and micro AUC:** after per-video min-max normalisation (LaGoVAD's DoTA protocol). Micro pools all frames after normalisation; macro averages per-video AUC and measures pure within-clip timing. Both are reported overall, for ego-involved and ego-uninvolved clips, and for the early, middle and late thirds of the anomaly window.
- **Same-source test:** video-level and frame-level AUC on held-out Nexar and the CCD test split, which share sources with training and so cannot reward domain shift (hypothesis H3).
- **STAUC and pointing game:** only for the optional arm (R12), on DoTA's official 1,305-video list with `stauc.py`.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P1 TAD in-domain (fixed camera) | TAD train → TAD test, UMIL split | AUC\_O, AUC\_A, AP | UMIL and WSAL on both metrics; AUC\_O-only rows; LaGoVAD and VadCLIP retrained |
| P2 MSAD in-domain (multi-scene surveillance) | MSAD weak train → MSAD test; traffic slice and non-traffic scenes scored separately | AUC, AP, AUC\_A, AP\_A | π-VAD and GS-MoE (four metrics) and Alert-CLIP (two); RTFM, MGFN, UR-DMU on the traffic slice |
| P3 dashcam | Nexar + CCD train → CCD test and Nexar held-out (same source); DoTA and DADA-2000 (zero-shot) | AUC, AP; DoTA macro / micro by ego flag and window third | DoTA: LaGoVAD, VadCLIP, PEL (published AUC, AP re-scored), WS-DoTA flagged; CCD and DADA-2000: released checkpoints re-scored; FLaRA re-scored; frame-supervised rows as context |
| P4 cross-setting zero-shot | P3 model → TAD (both lists) and MSAD; P2 model → TAD (both lists) | AUC, AP, AUC\_A | LaGoVAD, VadCLIP, PEL; MultiDomain and MSAD's own cross-dataset rows (AUC) |

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

**DoTA rows and their protocols.** Only rows under the same per-video normalisation are ranked against the thesis; the rest are context or kept out of the table.

| Method | Supervision / training data | DoTA protocol | AUC | Comparable to thesis? |
| --- | --- | --- | --- | --- |
| Thesis motion-stream model (R0) | Video-level, Nexar + CCD, zero-shot | Per-video min-max, 1,402 clips | 0.81 macro / 0.75 micro | Reference |
| LaGoVAD | Video-level, PreVAD, zero-shot | Per-video normalised | 62.60 | Yes (against micro) |
| VadCLIP | Weak, UCF-Crime / XD-Violence / PreVAD, zero-shot | Same | 50.93 / 49.00 / 50.59 | Yes |
| PEL (I3D) | Weak, UCF-Crime / XD-Violence, zero-shot | Same | 53.05 / 44.97 | Yes |
| AnoPred; DoTA ensemble; MAMTCF | Unsupervised, DoTA in-domain | Pooled raw scores | 67.5; 73.0; 76.6 | Context only |
| WS-DoTA (RTFM + transform) | Weak, D²-City + DoTA in-domain | \~1,031 clips, normalisation not stated | 78.2 | Context only |
| MOVAD, PromptTAD, TTHF, ISCRTAD | Frame labels, DoTA in-domain | Pooled; TTHF post-processed | 82.2, 83.9, 84.7, 86.2 | Upper reference only |
| Simple-TAD DAPT-VideoMAE-S / -L | Frame labels, fine-tuned, DoTA in-domain | Pooled | 86.4 / 88.4 | Upper reference only |
| FLaRA; BADAS | Clip-level, Nexar | Video-level anticipation AUC, 327 positive / 40 synthetic negative clips | 90.0; 90.6 as re-scored by FLaRA | No: re-scored under the thesis protocol instead |

The 0.75 micro sits 12.4 points above LaGoVAD's 62.60 under the same normalisation; row B1 separates how much of this comes from dashcam training data and how much from motion. TIME-VAD (IEEE T-ITS 2026) reports 94.44 on DoTA with an unknown protocol and is not cited until that is checked.

**Reporting rules.**

- TAD is reported on two separate columns, LaGoVAD's list and UMIL's list, never mixed.
- Every thesis number is a mean ± standard deviation over 3 seeds.
- Subsets (ego flag, categories, window thirds, traffic slices) carry 95% bootstrap confidence intervals over videos. LSP against the motion-stream model is tested with a paired bootstrap.
- All models use the same test lists, frame-to-score mapping and normalisation, and every table names each row's training set and protocol.

### 4.5 Ablation studies

The ablation ladder asks one question in order: does the gain come from *knowledge of normal motion dynamics*, or merely from adding a change signal, capacity, or FLaRA's future-space readout? Every row keeps VideoMAE at one global vector per step, uses video-level labels only, and reports DoTA macro / micro AUC (overall, by ego flag and by window third), the same-source Nexar / CCD test, DADA-2000, and P1 / P2 where marked. The predicted outcomes are priors written down in advance so the experiments can falsify them.

| Row | Variant | Isolates | Expected if LSP's premise holds |
| --- | --- | --- | --- |
| R0 | Motion stream only (gated v\_t) | Reference | 0.81 / 0.75 |
| R1 | + copy-last change e\_t = \[LN(u\_t − u\_{t−Δ}); p̂\_t\] | Raw change, the WS-DoTA cue on VideoMAE | About R0 (within ±0.01), as in the thesis pilot on feature differences |
| R2 | + fixed interpolation residual 2u\_t − u\_{t−Δ} − u\_{t+Δ} | STPrompt's second-order prior on a global vector | About R0 |
| R3 | + LSP architecture, no L\_rec, trained through MIL | Capacity only (FLaRA's "without L\_rec") | About R0 or slightly below; β stays small. **Decisive control** |
| R4 | + LSP, L\_rec on all training videos, frozen | Whether normal-only training matters | Above R0, below R5 |
| **R5** | **+ LSP, L\_rec on normal videos only, frozen, detached (full LaGoVAD-LSP)** | Learned normal-dynamics surprise | **+0.01 to +0.03 macro, at least as much on micro; larger on ego-involved clips and early window thirds** |
| R6 | FLaRA-literal: replace LN(v\_t) by the prediction û\_t | Whether a future-space readout helps offline | Below R0: it discards the observation |
| R7 | Add û\_t instead of r\_t as the stream | Whether the residual is the active ingredient | About R3 |
| R8 | R5 without detach, joint, λ ∈ {1, 2, 10, 100} | Whether MIL turns the predictor into a hidden classifier | Similar to R5 on the same-source test, no better zero-shot on DoTA |
| R9 | R5 + heteroscedastic head predicting (μ, log σ²), surprise by Gaussian NLL | Whether modelling uncertainty suppresses ego-motion false peaks | Fewer lead-in maxima (D5); small further gain |
| R10 | R5 with Δ ∈ {1, 2 clips}, K ∈ {8, 16} | Horizon and context sensitivity | 1 clip ≥ 2 clips; K = 16 ≥ 8 |
| R11 | R5 + a backward predictor (bidirectional surprise) | Late-window decay of a forward surprise | Gains concentrated in late window thirds |
| R12 | R5 + text-selected ClearCLIP agent token (optional arm) | Whether small-agent evidence adds to surprise | Gains on ego-uninvolved and small-box clips and on STAUC; overall AUC about R5 |

H1 holds if R5 beats R0, R1 and R2; H2 if R5 beats R3 and R4; H3 if R5 beats R1 on the same-source test and D4 passes. If R5 only matches R3, the honest conclusion is that VideoMAE V2-S global latents carry no exploitable normal-dynamics structure at this scale; with D3 and the V-JEPA 2 control, that is still a reportable negative result extending *Asleep at the Wheel* to weakly supervised detection.

**Side ablations within the component.**

- **Evidence vector:** residual only, ŝ\_t only, ρ\_t only, and all three (default).
- **Domain-shift guard:** ŝ\_t against video-median-subtracted ŝ\_t − med(ŝ).
- **Initialisation:** zero-initialised readout (start at copy-last) against random initialisation; tanh(β₀) = 0 against 0.1.
- **Predictor size:** 1, 2 and 4 layers.
- **Predictor data:** Nexar normals only, CCD normals only, both (default), and both plus unlabelled normal BDD100K driving video.
- **Encoder:** VideoMAE V2-S (default) against VideoMAE-S (Kinetics-400) and DAPT-VideoMAE-S (BDD100K), each still one 384-d vector; DAPT's normal-driving pre-training may make latents more predictable.
- **Captions:** with and without LaGoVAD's caption contrastive loss.

**Analyses.**

1. **Where the gain lands.** DoTA ego-involved vs ego-uninvolved clips; the nine categories; early, middle and late thirds of the anomaly window; TAD Accident, PedestrianOnRoad and the remaining categories pooled; the MSAD traffic slice against its non-traffic scenes; Nexar-only against Nexar + CCD training.
2. **Surprise behaviour.** Onset-aligned curves of ŝ\_t and p\_t with success and failure examples; the share of lead-in maxima; surprise on normal manoeuvres (turns, braking) in Nexar negatives.
3. **Gate behaviour.** tanh(α) and tanh(β) over training, as a check that the new stream is used.
4. **Language.** Whether surprise changes LaGoVAD's sensitivity to the definition: score change of accident frames when the definition includes or excludes accidents, and drift@5 on MSAD.
5. **TAD definitions.** Main results with LaGoVAD's two-class TAD prompt against each of TAD's 7 categories with its own definition.
6. **Cost.** Parameters, storage and measured frames per second for rows R0 and R5.

**Budget.** Diagnostics D1–D5 take hours on cached vectors. Rows R0–R5 run on P1, P2 and P3 with 3 seeds; P4 reuses those models. Rows R6–R12 and the side ablations run on P3 (and P1 where listed), first with one seed, then three for variants that are kept. Because every row reuses the same cached 384-d vectors and LaGoVAD's schedule, a one-epoch timing run in the first week converts this plan into GPU-hours; if the total exceeds the budget, the ladder is cut to R0, R1, R3 and R5, which still test H1–H3.

## References

**Baseline, backbone and component**

- Liu et al., [Language-Guided Open-World Video Anomaly Detection under Weak Supervision (LaGoVAD)](https://arxiv.org/abs/2503.13160), ICLR 2026; [code, checkpoint, PreVAD annotations](https://github.com/Kamino666/LaGoVAD-PreVAD)
- Caselli et al., [FLaRA: Predicting Future Latent Representations for Accident Anticipation](https://arxiv.org/abs/2606.14380), arXiv 2606.14380; [code](https://github.com/LoreCase073/FLaRA)
- Assran et al., [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985), arXiv 2506.09985
- Bardes et al., [V-JEPA](https://arxiv.org/abs/2404.08471), arXiv 2404.08471 (venue not verified)
- Wang et al., [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727), CVPR 2023 (motion encoder: `vit_s_k710_dl_from_giant`)
- Tong et al., [VideoMAE](https://arxiv.org/abs/2203.12602), NeurIPS 2022
- Orlova et al., [Simplifying Traffic Anomaly Detection with Video Foundation Models (Simple-TAD)](https://arxiv.org/abs/2507.09338), ICCV 2025 Workshops; [code and DAPT encoders](https://github.com/tue-mps/simple-tad)

**Evidence on latent surprise**

- Garrido et al., [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831), arXiv 2502.11831
- [Asleep at the Wheel](https://arxiv.org/abs/2608.01336), arXiv 2608.01336 (preprint)
- [Latent Clarity](https://arxiv.org/abs/2607.03558), arXiv 2607.03558 (preprint, single author)
- [JEPA surprise on driving data without EMA](https://arxiv.org/abs/2606.28383), arXiv 2606.28383 (preprint)
- Vondrick et al., Anticipating Visual Representations from Unlabeled Video, CVPR 2016

**Evidence behind the MQR diagnosis**

- Psomas et al., [Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency (Efficient Probing)](https://arxiv.org/abs/2506.10178), ICLR 2026
- Przewięźlikowski et al., [Beyond \[cls\]: Exploring the True Potential of Masked Image Modeling](https://arxiv.org/abs/2412.03215), ICCV 2025
- Hoffmann et al., [Eureka-Moments in Transformers](https://arxiv.org/abs/2310.12956), arXiv 2310.12956

**Weakly supervised and open-world VAD**

- Wu et al., [Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts (STPrompt)](https://arxiv.org/abs/2408.05905), ACM MM 2024
- Wu et al., VadCLIP, AAAI 2024; [code](https://github.com/nwpu-zxr/VadCLIP)
- Pu et al., PEL4VAD, IEEE TIP 2024; [code and checkpoints](https://github.com/Aaron-Pu/PEL4VAD)
- [DSANet](https://arxiv.org/abs/2511.10334), AAAI 2026; [code](https://github.com/lessiYin/DSANet)
- Lv et al., [Unbiased Multiple Instance Learning (UMIL)](https://arxiv.org/abs/2303.12369), CVPR 2023; [TAD split files](https://github.com/ktr-hubrt/UMIL)
- Lv et al., Localizing Anomalies from Weakly-Labeled Videos (WSAL, TAD dataset), IEEE TIP 2021; [repository](https://github.com/ktr-hubrt/WSAL)
- [MultiDomain](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf), NeurIPS 2024
- [π-VAD](https://arxiv.org/abs/2505.13123), CVPR 2025; [GS-MoE](https://arxiv.org/abs/2508.06318), ICCV 2025
- Alert-CLIP, CVPR 2026 (project file); [annotations](https://github.com/ClarkZhu216/Alert-CLIP_dataset)
- [WS-DoTA](https://arxiv.org/abs/2408.05562), ECCV 2024 Workshop; [split files](https://github.com/ut21/WSAD-Driving)
- [RiskProp](https://arxiv.org/abs/2603.27165), arXiv 2603.27165
- [Localizing to Debias](https://arxiv.org/abs/2608.12045), arXiv 2608.12045; [Learning Where and When](https://engineers.usf.edu/sislab/publications/learning-where-and-when-patch-based-spatiotemporal-localization-weakly-supervised), arXiv 2606.29498

**Traffic anomaly detection and anticipation**

- Qiu et al., [PromptTAD: Object-Prompt Enhanced Traffic Anomaly Detection](https://doi.org/10.1109/LRA.2025.3572428), IEEE RA-L 2025
- Goldshmidt et al., [BADAS](https://arxiv.org/abs/2510.14876), arXiv 2510.14876
- [MAMTCF](https://arxiv.org/abs/2307.14575), arXiv 2307.14575
- Liu et al., Future Frame Prediction for Anomaly Detection (AnoPred), CVPR 2018
- Rossi et al., MOVAD, ICASSP 2024; [code](https://github.com/IMPLabUniPr/movad)
- Liang et al., TTHF, IEEE TCSVT 2024; Liang et al., ISCRTAD, IEEE T-ITS 2025
- Shi et al., [Structured Graph Reasoning for Traffic Anomaly Detection](https://doi.org/10.1109/TMM.2026.3703345), IEEE TMM 2026
- TIME-VAD, IEEE T-ITS 2026 ([record](https://trid.trb.org/View/2685868); protocol unverified)

**Optional CLIP arm**

- Lan et al., [ClearCLIP](https://arxiv.org/abs/2407.12442), ECCV 2024; Wang et al., [SCLIP](https://arxiv.org/abs/2312.01597), ECCV 2024
- Zhou et al., [AnomalyCLIP](https://arxiv.org/abs/2310.18961), ICLR 2024

**Datasets**

- Yao et al., [DoTA](https://arxiv.org/abs/2004.03044), IEEE TPAMI 2022; [annotations and STAUC code](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
- Zhu et al., [MSAD](https://msad-dataset.github.io/), NeurIPS 2024 Datasets and Benchmarks
- Bao et al., [CCD](https://github.com/Cogito2012/CarCrashDataset), ACM MM 2020
- Moura et al., [Nexar Dashcam Collision Prediction Dataset and Challenge](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf), CVPR 2025 Workshops
- Fang et al., DADA-2000, IEEE T-ITS 2021; Fang et al., [MM-AU](https://arxiv.org/abs/2403.00436), CVPR 2024
- Yu et al., BDD100K, CVPR 2020; D²-City, arXiv 2019
