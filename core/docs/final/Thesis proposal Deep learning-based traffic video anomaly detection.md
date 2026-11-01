# Thesis proposal: Deep learning-based traffic video anomaly detection (LaGoVAD-STP)

Nov 1, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-STP**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. It starts from LaGoVAD (ICLR 2026) and adds a frozen VideoMAE V2-S motion stream, read as one 384-d global vector per step, as the permitted backbone change. Trained on Nexar and CCD dashcam video with video-level labels, this model already reaches **0.81 macro / 0.75 micro per-video min-max AUC zero-shot on DoTA**, against 62.60 for the released LaGoVAD.

The one novel component is **traffic spatio-temporal prompting (STP)**, adapted from STPrompt (ACM MM 2024) for the CLIP and text branches. On the visual side, a Spatial Attention Aggregation (SA²) step picks the CLIP patch features that change most over time, aggregates them by attention, and adds the result element-wise to the frame's CLIP feature. On the text side, a learnable prefix of context tokens is prepended to every anomaly-class text. The claim under test: local, motion-selected appearance evidence and learned label context recover small and distant traffic agents that one global CLIP vector dilutes, and the SA² weights double as spatial evidence maps scored with DoTA's STAUC.

This version replaces two components that were built and tested on the VideoMAE stream and did not help: the Motion Query Readout (MQR) and the Latent Surprise Predictor (LSP). Both tried to extract more from a motion vector that already seems saturated, so the novelty moves to the CLIP branch, which still reads each frame as one global vector (Section 3.2.1).

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event is happening in it. Abnormal events include collisions, loss of control, illegal turns, wrong-way driving, pedestrians on the road and road spills. These are the categories used by TAD and DoTA.

Two camera settings define the problem space:

- **Fixed camera (non-ego).** TAD and the traffic scenes of MSAD. The camera is static; the anomalous agents are often small and far away.
- **Dashcam (ego-centric).** DoTA, DADA-2000, CCD and Nexar. The camera moves with the car. DoTA also marks each clip as ego-involved (the ego car takes part in the anomaly) or ego-uninvolved (it only observes); this proposal uses those two terms for that flag.

The thesis follows LaGoVAD's weakly supervised, language-guided formulation. Training uses video-level annotations only: one label y\_i ∈ {0, 1} per video, plus a category and a hand-written caption template. At test time the model receives a video V and a text definition Z of what counts as anomalous, and returns a score per sampled step. This thesis adds a coarse spatial evidence map per step as a second output.

```latex
\Phi_\theta : (V, Z) \;\mapsto\; \{ s_t \}_{t=1}^{N},\; \{ E_t \}_{t=1}^{N}, \qquad s_t \in [0,1],\;\; E_t \in [0,1]^{7 \times 7}
```

The video-level annotations supervise training through multiple-instance learning (MIL): the mean of the top-k step scores of a video must match its video label.

Four properties make traffic video harder than generic surveillance:

1. **Anomalies are motion events.** A collision is defined by how trajectories meet, not by what one frame looks like. LaGoVAD encodes single RGB frames with CLIP and has no motion input; it reaches 89.56 AUC on TAD but 62.60 on dashcam DoTA. Part of this gap is protocol: DoTA is abnormal-only and min-max normalised per video, so its AUC measures timing inside a clip.
2. **The agents that matter are small.** A distant agent of about 40×30 px in a 1280×720 frame shrinks to about 7×9 px at 224×224 input, smaller than one 16-px patch (estimate). One global CLIP vector per frame dilutes it; this is what the novel component targets.
3. **The camera moves.** In dashcam video, ego-motion dominates frame-to-frame change. In preliminary experiments, hand-crafted motion cues (feature differences, ego-motion-compensated optical flow) did not help as direct inputs; here a motion cue is used only to *select* where to look, and a variant compensates for the change shared by the whole frame.
4. **Labels are scarce.** Frame-level labels are expensive and anomalies are rare, so the model must learn from video-level labels.

The scope is offline detection: the whole video is available and the question is whether an anomaly is happening, not whether one will happen (anticipation, as in Nexar). No large language model or multimodal LLM runs at training or test time; a frozen CLIP text encoder is allowed. All training text is hand-written: class names, category definitions and caption templates. STPrompt's LLM-written localisation prompts are therefore not used.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from cheap video-level labels, sees motion, attends to the part of the frame where things change, and shows where that is. It must keep LaGoVAD's ability to take the anomaly definition as text, and run on one academic GPU without any LLM.

The thesis builds this in two steps, matching the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** A frozen VideoMAE V2-S encoder (CVPR 2023), the strongest small masked-video encoder for traffic anomalies in Simple-TAD's comparison, feeds one 384-d vector per step into LaGoVAD through a zero-initialised gate. Preliminary result: zero-shot DoTA **0.81 macro / 0.75 micro** per-video min-max AUC. Two attempts to extract more from this stream (MQR, LSP) added nothing, so it stays one global vector.
- **Novel component (STP).** In the CLIP branch, SA² adds a motion-selected, attention-aggregated patch feature to each frame's CLIP feature; in the text branch, a learnable prefix gives every anomaly-class text a learned context.

A second purpose is a fair evaluation: published traffic numbers are mostly single-metric AUC on inconsistent splits and protocols. The thesis will report at least two metrics against at least two published methods on the same fixed-camera traffic data, plus zero-shot and dashcam results, and label each row's protocol.

### 1.3 Research aims

The thesis has five aims:

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint and retrain LaGoVAD on the thesis training data, with AUC, AP and AUC on abnormal videos (AUC\_A) on TAD, MSAD, DoTA and DADA-2000.
2. **Document the motion stream.** Report the frozen VideoMAE V2-S stream's gain over CLIP-only LaGoVAD, and the negative MQR and LSP results that fixed it at one global vector.
3. **Design and train STP.** Add SA² to the CLIP branch and a learnable prefix to the text branch, using video-level annotations only.
4. **Measure spatial localisation under weak supervision.** Score SA²'s maps against DoTA's anomalous-agent boxes with STAUC and a pointing game.
5. **Explain the gains.** Break results down by ego-involved vs ego-uninvolved clips, agent box size and anomaly category; test whether the motion prior survives ego-motion; and check that MSAD's non-traffic scenes do not regress.

Aims 1–2 are prerequisites; aims 3–5 map to the hypotheses below. "Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval over test videos excludes zero.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. SA² improves on the motion-stream model, mostly on small and ego-uninvolved agents | DoTA macro / micro AUC by ego flag and box-size bin; TAD AUC by category (ablation rows R0, R4) | R4 > R0 overall, with larger gains on ego-uninvolved clips and small-box bins |
| H2. Motion-prior selection matters on traffic video, as on surveillance | STPrompt's ladder: average only, attention only, selection + average, SA² (R1–R4); raw vs ego-compensated motion (R7) | R3 and R4 beat R1 and R2; on dashcam sets, R7 ≥ R6 |
| H3. The learnable prefix improves transfer without weakening language guidance | Zero-shot DoTA, DADA-2000 and TAD AUC; MSAD drift@5 (R5, R6) | R6 > R4 on zero-shot AUC, and drift@5 no worse |
| H4. SA²'s maps point at the anomalous agents | STAUC and pointing game on DoTA's official 1,305-video list | SA² beats a uniform map and a centre prior |
| H5. The gain holds across camera settings without hurting surveillance | TAD AUC\_O / AUC\_A; MSAD AUC / AP, its traffic slice and non-traffic scenes | Gains on TAD, the MSAD traffic slice and the dashcam sets; no significant drop on MSAD's non-traffic scenes |

## 2. Literature review

### 2.1 Overview and classification of video anomaly detection method branches

Video anomaly detection (VAD) methods divide by the supervision they train with into five branches, plus the adjacent task of accident anticipation. Traffic methods cut across these branches along a second axis: fixed camera (non-ego) or dashcam (ego-centric). The table places the papers reviewed for this thesis, mostly 2024–2026.

| Branch | Training signal | Typical mechanism | Representative methods (venue) | Traffic evidence | Role in this thesis |
| --- | --- | --- | --- | --- | --- |
| Unsupervised / one-class | Normal videos only | Reconstruction, frame or flow prediction, future object location, diffusion, scene-graph normality | AnoPred (CVPR 2018), FOL / DoTA (IEEE TPAMI 2022), MAMTCF (arXiv 2307.14575), MA-PDM (AAAI 2025), ADSM (CVPR 2025), TAO (CVPR 2025), Structured Graph Reasoning (IEEE TMM 2026) | DoTA AUC 67.5–76.6 and STAUC (DoTA baselines up to 48.5) | Context; ego-motion breaks most predictors |
| Weakly supervised | Video-level labels, MIL | Snippet features (I3D, CLIP) + top-k MIL; CLIP text prompts with learnable context; prototypes; spatial aggregation of patches | Sultani (CVPR 2018), RTFM (ICCV 2021), WSAL (IEEE TIP 2021), UMIL (CVPR 2023), MultiDomain (NeurIPS 2024), VadCLIP (AAAI 2024), **STPrompt (ACM MM 2024)**, π-VAD (CVPR 2025), GS-MoE (ICCV 2025), DSANet, RefineVAD (AAAI 2026), Alert-CLIP (CVPR 2026) | TAD (WSAL, UMIL), MSAD (π-VAD, GS-MoE, Alert-CLIP), dashcam WS-DoTA (ECCV 2024 Workshop) | **The thesis's branch; STPrompt is the source of the novel component** |
| Open-vocabulary / open-world | Video labels + category text | CLIP text space defines anomaly classes or free-text definitions at test time | Anomize (CVPR 2025), **LaGoVAD (ICLR 2026)** | Zero-shot TAD, DoTA, MSAD (LaGoVAD) | **Baseline** |
| Fully supervised traffic detection | Frame-level labels (+ boxes) | Video encoder + temporal head; object prompts; agent graphs | MOVAD (ICASSP 2024), TTHF (IEEE TCSVT 2024), PromptTAD (IEEE RA-L 2025), ISCRTAD (IEEE T-ITS 2025), Simple-TAD (ICCV 2025 Workshops) | DoTA AUC 82.2–88.4 in-domain | Upper reference rows; evidence that local, object-level evidence helps |
| LLM / MLLM-based | None, or few-shot rules, or light tuning | Captioning + LLM reasoning, prompting, probing of MLLM heads | LAVAD (CVPR 2024), AnomalyRuler (ECCV 2024), VERA, Holmes-VAU (CVPR 2025), EventVAD (ACM MM 2025), HeadHunt-VAD (AAAI 2026), LAVIDA (CVPR 2026), AnyAnomaly (WACV 2026) | Almost none on traffic sets | Excluded: LLM cost |
| Accident anticipation (adjacent task) | Positive / negative dashcam clips | Predict a collision before it happens, recently by forecasting future latents | DSTA, UString (ACM MM 2020), BADAS (arXiv 2510.14876), FLaRA (arXiv 2606.14380) | Nexar, CCD, DAD; video-level anticipation AUC | Different task; FLaRA's latent prediction was tested as LSP and dropped |

Three findings from this review shape the proposal. First, CLIP-feature weak supervision dominates surveillance benchmarks but sits near chance on dashcam DoTA, and the strongest traffic results come from masked video foundation models. Second, the methods that win on small, distant or off-centre traffic agents (PromptTAD, ISCRTAD, Alert-CLIP) add local evidence, but through detectors, boxes or region captions. Third, STPrompt is the one weakly supervised design that adds local evidence with no detector and no boxes, and its own ablation shows that a fixed motion prior does most of the work; it has never been tested on traffic or under a moving camera.

### 2.2 Emerging trends and unresolved challenges

Six trends run through the 2024–2026 literature:

1. **Vision-language features as the default.** Since VadCLIP (AAAI 2024), most weakly supervised methods start from frozen CLIP frame features and text prompts, usually with CoOp-style learnable context tokens: STPrompt (ACM MM 2024), DSANet and RefineVAD (AAAI 2026), LAS-VAD, Fine-VAD and Alert-CLIP (CVPR 2026).
2. **Anomalies defined by language.** Open-vocabulary (Anomize, CVPR 2025) and open-world (LaGoVAD, ICLR 2026) models accept the anomaly definition at test time.
3. **MLLMs for explanation and training-free detection.** LAVAD, VERA, Holmes-VAU, HeadHunt-VAD and LAVIDA rely on LLMs or multimodal LLMs of 2B parameters and more. They are not clearly better at detection: on MSAD drift@5, LaGoVAD reaches 85.6 AUC / 40.1 AP against 84.3 / 34.3 for Holmes-VAU.
4. **Masked-video foundation models for traffic.** Simple-TAD showed that masked video modelling transfers best to traffic anomalies; VideoMAE-B reaches AUC-MCC 56.0 on DoTAref against 50.0 for InternVideo2-B after fine-tuning.
5. **Spatial and object-level grounding.** Alert-CLIP aligns region features with boxes and captions, PromptTAD prompts with detected boxes, TAO tracks objects, and STPrompt selects image patches by motion. Two 2026 preprints ("Localizing to Debias", "Learning Where and When") bring patch-level localisation to weak supervision on surveillance data.
6. **Extra modalities.** π-VAD (CVPR 2025) induces pose, depth, flow and other modalities from RGB features at training time and found flow the most useful.

Five challenges remain open for traffic video:

1. **Motion blindness.** Frame-level CLIP features miss what makes a collision a collision. In LaGoVAD's table every surveillance-trained model is near chance on DoTA's within-clip protocol (VadCLIP, LaGoVAD: 49.00–62.60; PEL with I3D: 44.97–53.05).
2. **Spatial dilution.** Global vectors lose small and distant agents. Even frame-supervised Simple-TAD averages 81.6 / 84.1 / 85.2 AUC on ego-uninvolved DoTA clips (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips. GS-MoE reports a road-accident video scored near zero because the anomaly occupies a very small pixel region.
3. **No spatial evidence under weak supervision on traffic.** We found no weakly supervised traffic method that outputs or evaluates spatial maps, and no report of DoTA's STAUC under weak supervision (search to October 2026).
4. **Ego-motion and domain shift.** Camera motion masks agent motion: frame-prediction baselines on DoTA flag trees and background, and motion priors designed for static cameras, like STPrompt's, may select near-field background. Training and test data also come from different sources.
5. **Inconsistent evaluation.** TAD has two incompatible test lists in use (76 of LaGoVAD's 100 TAD test videos belong to UMIL's training list). DoTA numbers come from at least four protocols: pooled raw scores, per-video min-max, post-processed scores, and video-level anticipation AUC on a 367-clip subset.

### 2.3 Key works behind the proposed model

The proposed model is assembled from one baseline, one encoder, one source paper for the novel component and the works whose evidence justifies each choice. The table shows where each part of LaGoVAD-STP comes from; the reviews follow.

| Part of LaGoVAD-STP | Source work (venue) | What is taken | What the thesis changes |
| --- | --- | --- | --- |
| Host detector: CLIP stream, RoFormer, text co-attention, heads and losses | LaGoVAD (ICLR 2026), on VadCLIP's frozen-CLIP + MIL design (AAAI 2024) | Architecture, losses and schedule as released | Last-epoch checkpoints; hand-written caption templates |
| Motion stream (backbone change) | VideoMAE V2 (CVPR 2023), chosen on the evidence of Simple-TAD (ICCV 2025 Workshops) | The frozen distilled ViT-S and Simple-TAD's 16-frame, 10 fps clip input | Frozen, one mean-pooled 384-d vector per step, fused through a zero-initialised gate |
| SA² in the CLIP branch (novel component, visual part) | STPrompt (ACM MM 2024) | Window-crop CLIP patch features, second-order motion magnitude, top-K selection, attention aggregation, element-wise sum with the frame feature | Placed before LaGoVAD's RoFormer alongside a motion stream; attention starts as a plain average; a global-change-compensated motion variant for dashcams; weights used as evidence maps |
| Learnable prefix in the text branch (novel component, text part) | STPrompt and VadCLIP, after CoOp (IJCV 2022) | l learnable context tokens prepended to each class text before the frozen CLIP text encoder | Applied to LaGoVAD's open-world definitions (class names and descriptions) in place of its released soft prompts |
| Evidence-map evaluation | DoTA (IEEE TPAMI 2022) | STAUC code, anomalous-agent boxes and the ego-involved flag | Maps from SA²'s weights, scored under weak supervision |
| Ego-motion diagnosis | DoTA's baselines; PromptTAD (IEEE RA-L 2025) | Frame-level methods are distracted by dynamic backgrounds and miss off-centre agents | Motivates the compensated-motion variant and the ego / box-size analyses |

#### 2.3.1 LaGoVAD (ICLR 2026): the baseline

Liu et al. define open-world VAD as a mapping Φ: (V, Z) → frame scores, where the anomaly definition Z is given as text at test time. Frozen CLIP ViT-B/16 frame embeddings pass through a 2-layer RoFormer and are fused with the CLIP text encoding of Z by co-attention. The definition is a set of categories, each given by a class name or a description; training randomly uses one or the other. Training uses video-level labels on PreVAD (35,279 videos) with MIL and MIL-alignment losses, dynamic video synthesis and a caption contrastive loss with hard negatives.

- **Strengths.** It has the best zero-shot traffic results in its comparison: 89.56 AUC on TAD, 62.60 on DoTA and 90.41 on MSAD, ahead of VadCLIP and PEL. It stays robust when the definition changes (MSAD drift@5: 85.6 AUC / 40.1 AP), and better definitions help (UCF-Crime AUC 80.44 with class names, 81.12 with manual descriptions). The detector runs no LLM, trains on one RTX 4090, and its code, checkpoint and predictions are public.
- **Weaknesses.** Its only visual input is one global CLIP embedding per sampled frame, so it sees neither motion nor where in the frame something happens: DoTA AUC is 62.60 against 89.56 on TAD. It outputs no spatial evidence. Its released configuration selects checkpoints on test scores, and PreVAD's captions were written by Qwen models.
- **Taken into this thesis.** The whole detector. The thesis adds a motion input, SA² in the CLIP branch and a learnable prefix in the text branch, and replaces test-score checkpoint selection with the last epoch.

#### 2.3.2 VadCLIP (AAAI 2024): the CLIP-based weakly supervised template and its learnable prompt

Wu et al. adapt frozen CLIP to weakly supervised VAD with a local Transformer and graph convolution (the LGT-Adapter), a binary classification branch and an alignment branch that matches frames to class embeddings. Following CoOp, the class tokens are concatenated with l learnable context tokens before the frozen text encoder, because short labels are too succinct to describe abnormal events.

- **Strengths.** It established the CLIP-plus-learnable-prompt design that STPrompt, LaGoVAD, DSANet and Alert-CLIP build on: 88.02 AUC on UCF-Crime and 84.51 AP on XD-Violence on one RTX 3090, with public code.
- **Weaknesses.** It reads appearance only and one vector per frame, and sits near chance zero-shot on DoTA (49.00–50.93 AUC). In DSANet's analysis, its prototypes assign 84.0% of UCF-Crime road-accident videos to "Normal".
- **Taken into this thesis.** Its learnable-prompt form, as passed on by STPrompt, and a retrained same-feature competitor (Group C, Section 4.3).

#### 2.3.3 VideoMAE V2 (CVPR 2023): the motion encoder

Wang et al. scale masked video autoencoding with dual masking and progressive training (self-supervised on 1.35M clips, then supervised on Kinetics-710), and distil the 1B-parameter ViT-g into smaller ViT-S and ViT-B models.

- **Strengths.** The distilled ViT-S is small (about 22M parameters) and public, and in Simple-TAD's comparison it is the strongest small encoder for traffic anomalies after fine-tuning: AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000, against 52.1 / 48.1 for VideoMAE-S and 49.7 / 43.7 for InternVideo2-S. In this thesis's pilot, its mean-pooled vector lifts zero-shot DoTA to 0.81 / 0.75.
- **Weaknesses.** Its pre-training data contain no driving video, and its features are not text-aligned. In this thesis, learned readouts of its tokens (MQR) and learned predictors over its vectors (LSP) added nothing beyond the mean vector.
- **Taken into this thesis.** The frozen distilled ViT-S, read as one mean-pooled 384-d vector per step.

#### 2.3.4 Simple-TAD (ICCV 2025 Workshops): evidence for the encoder choice

Orlova et al. show that a plain video ViT with one linear head, fine-tuned with frame-level labels, outperforms specialised traffic detectors on DoTA and DADA-2000, and add domain-adaptive pre-training (DAPT) on normal driving video.

- **Strengths.** Simple and strong (DoTA AUC up to 88.4; DoTA → DADA-2000 up to 86.6). It shows that masked video modelling transfers best to traffic anomalies and vision-language pre-training worst.
- **Weaknesses.** It needs frame-level labels, covers dashcam data only and takes no text. Ego-uninvolved clips lag: 81.6 / 84.1 / 85.2 AUC (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips, and the authors note that object-aware methods do better on rare categories.
- **Taken into this thesis.** The choice of encoder and clip input, the DAPT encoder as an ablation, and the ego-uninvolved gap, which motivates hypothesis H1.

#### 2.3.5 STPrompt (ACM MM 2024): the source of the novel component

Wu et al. learn spatio-temporal prompts on frozen CLIP for weakly supervised detection and localisation. The detection side has two parts that this thesis adopts.

- **Spatial Attention Aggregation (SA²).** Each frame, resized to 224×224, is cut into a 7×7 grid of 32×32 windows (stride 32); each window passes through CLIP and its CLS embedding becomes a patch feature. A patch's motion magnitude is the second-order difference of its feature across neighbouring frames, ‖2x\_i − x\_{i−1} − x\_{i+1}‖. The K = 12 patches with the largest motion are aggregated by attention into x\_AS, which is summed element-wise with the frame's CLIP feature before the temporal adapter. STPrompt argues that the two are complementary: x\_CLIP gives every pixel nearly equal influence, while x\_AS focuses on likely anomaly locations.
- **Learnable prompt prefix.** Following CoOp, l = 8 learnable context tokens are prepended to each category's tokens, {e\_1, …, e\_l, t\_c}, before the frozen text encoder; a contrastive loss keeps the label embeddings apart. Localisation at test time is training-free and uses LLM-written descriptions, which this thesis excludes.

| STPrompt ablation (AUC %) | UCF-Crime | UBnormal |
| --- | --- | --- |
| Without SA² | 86.13 | 63.00 |
| Average of all patches | 86.91 | 62.93 |
| Attention only | 86.67 | 62.64 |
| Motion selection + average | 87.79 | 63.69 |
| **Motion selection + attention (SA²)** | **88.08** | **63.98** |

- **Strengths.** It adds local evidence with no detector, boxes or region captions, at 31.5M parameters and 44.8 GFLOPs. The ablation shows that the fixed motion prior does most of the work: selection adds about 0.9 AUC over a plain average on UCF-Crime and is the only variant that helps on both datasets, while learned attention alone scores below averaging. It improves temporal-IoU localisation over VadCLIP by about 1.9, 5.7 and 4.5 points on UCF-Crime, ShanghaiTech and UBnormal.
- **Weaknesses.** Its motion cue assumes a still camera: under forward ego-motion, second-order differences are largest on near-field road and frame edges. Patch features cost 49 extra CLIP passes per frame. The exact attention formula (Eq. 2) is not legible in the project copy and must be checked against the published PDF. It covers surveillance data only, and no public code was found.
- **Taken into this thesis.** SA² and the learnable prefix, in STPrompt's pattern: patch CLIP features → SA² → element-wise sum with the CLS CLIP feature, and prefix + class text → frozen text encoder. **Changed:** the attention starts as a plain average so the model begins at STPrompt's strong "selection + average" variant; a variant subtracts the change shared by the whole frame before ranking patches; SA²'s weights become evidence maps scored with STAUC instead of LLM-based localisation; and both parts sit in LaGoVAD's open-world host next to a VideoMAE motion stream.

#### 2.3.6 PromptTAD (IEEE RA-L 2025): object-level evidence from a detector

Qiu et al. extend MOVAD with YOLOv9 box prompts, instance-wise and relation-wise aggregation (12 learnable query tokens) and an instance-level loss on Hungarian-matched ground-truth boxes, trained with frame labels.

- **Strengths.** It names the failure STP targets: the global view emphasises prominent regions and neglects distant or off-centre objects. It reaches 83.9 AUC on DoTA, reports STAUC, evaluates a small-object subset (334 test videos with anomalous boxes under 2,267 pixels), and gains most on ego-uninvolved anomalies (DADA-2000 70.4 vs 64.0 for TTHF).
- **Weaknesses.** It needs frame labels, anomalous-object boxes and a detector; when detection fails in dark scenes it degenerates into the baseline. No public code.
- **Taken into this thesis.** The diagnosis, the small-object subset and the ego-involved vs ego-uninvolved analysis. STP replaces detector boxes with motion-selected CLIP patches, and box supervision with video-level MIL.

#### 2.3.7 Methods used as comparison rows

These methods contribute no part of the architecture; they supply published numbers (Section 4.4).

| Method (venue) | Approach | Strength | Weakness | Role in this thesis |
| --- | --- | --- | --- | --- |
| WSAL (IEEE TIP 2021) | High-order context encoding of TSN features; introduced the TAD dataset | 89.64 AUC\_O / 61.66 AUC\_A on TAD | Hand-crafted cues; fixed 32 segments | P1 row on both metrics |
| UMIL (CVPR 2023) | Confident / ambiguous snippet split and clustering against context shortcuts; X-CLIP fine-tuned | 92.93 AUC\_O / 65.82 AUC\_A on TAD | Needs backbone fine-tuning (90.71 frozen) | P1 row; source of the TAD split |
| π-VAD (CVPR 2025) | Induces five modalities from RGB at training time | MSAD 88.68 AUC / 71.26 AP; flow most useful | Five teacher backbones; no public code | P2 row |
| GS-MoE (ICCV 2025) | Gaussian pseudo-labels plus one expert per class | MSAD 87.72 AUC / 68.26 AP / 69.54 AUC\_A / 76.68 AP\_A | Needs class labels; misses very small accidents | P2 row |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with boxes and region captions | Lifts the VadCLIP framework on MSAD from 87.45 / 71.09 to 89.63 / 78.24 | Needs boxes and VLM-written captions | P2 row; evidence that region-level CLIP evidence helps |
| WS-DoTA (ECCV 2024 Workshop) | Weakly supervised DoTA with D²-City normals; copy-last CLIP change | The only weakly supervised dashcam study; lifts RTFM from 57.9 to 78.2 AUC | Own \~1,031-clip subset; AUC only | Flagged P3 row |
| FLaRA (arXiv 2606.14380) | Forecasts V-JEPA2 future latents and classifies them; supervised anticipation | Nexar AP 86.6 / AUC 87.3 | Anticipation metric, not comparable with frame AUC | Context only; its idea was tested as LSP and dropped |

### 2.4 Research gaps

Four gaps follow from the review. The thesis addresses each one, and its novelty claim is deliberately narrow.

1. **No motion-aware foundation features in weakly supervised traffic VAD.** LaGoVAD and VadCLIP read CLIP frames; the masked-video evidence of Simple-TAD comes from frame-level supervision. *Addressed by the motion stream (backbone change), already shown to lift zero-shot DoTA to 0.81 / 0.75.*
2. **The appearance branch still reads one global vector per frame.** Small, distant and ego-uninvolved agents are diluted, and the detector-free fix that works under weak labels, STPrompt's SA², has been tested only on static surveillance cameras, never on traffic, under ego-motion, or next to a motion stream. *Addressed by SA² in the CLIP branch, with an ego-compensated motion variant.*
3. **Weak supervision gives no spatial evidence on traffic data.** We found no weakly supervised traffic method that produces maps, and no report of STAUC under weak supervision. *Addressed by SA²'s selection and attention weights, scored with STAUC and a pointing game, with no LLM.*
4. **Traffic results are hard to compare.** Most rows are AUC-only, two TAD test lists circulate, and DoTA numbers mix four protocols. *Addressed by four split-consistent, multi-metric protocols with each row's protocol labelled (Section 4.4).*

The learnable prefix fills a smaller gap: LaGoVAD's definitions are hand-written texts, and how a learned context shared by all definitions affects zero-shot traffic transfer and definition sensitivity has not been measured.

The closest prior work bounds the novelty claim:

| Closest work | What it does | How LaGoVAD-STP differs |
| --- | --- | --- |
| STPrompt (ACM MM 2024) | SA² and learnable prefixes on a VadCLIP-style detector; LLM-assisted localisation; surveillance data | Traffic and dashcam data; open-world language-guided host with a motion stream; ego-compensated motion; maps scored with STAUC; no LLM |
| VadCLIP (AAAI 2024) | Learnable prompt context on a CLIP MIL detector, one vector per frame | Adds motion-selected local evidence and a motion stream |
| PromptTAD (IEEE RA-L 2025) | Detector box prompts with frame labels and box supervision | Detector-free, video-level labels |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with boxes and captions | Needs neither boxes nor captions |
| WS-DoTA (ECCV 2024 Workshop) | Global CLIP change features for weakly supervised DoTA | Change used per patch, to select where to look |
| "Localizing to Debias" (arXiv 2608.12045) | Sparse spatio-temporal weak localisation with motion-aware regularisation; UCF-Crime, XD-Violence, MSAD | Traffic focus, STAUC on DoTA, language-guided host; **novelty risk, to be read in full** |
| "Learning Where and When" (arXiv 2606.29498) | Patch-level MIL with proximity-aware top-k selection; surveillance | Traffic focus, motion prior, language-guided host |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that adds motion-selected CLIP patch evidence (SA²) and learned label context to a language-guided open-world detector with a motion-foundation stream, the first test of STPrompt's motion prior under ego-motion, and the first report of STAUC under weak supervision.** SA² and learnable prompts are not claimed as new; the contribution is their transfer to traffic, the ego-motion analysis and the spatial evaluation. Both claims will be re-checked against the two 2026 preprints before submission.

## 3. Proposed model

The proposed model, **LaGoVAD-STP**, is LaGoVAD with two changes: a frozen VideoMAE motion stream read as one vector per step (the backbone change) and traffic spatio-temporal prompting (the novel component), which adds SA² to the CLIP branch and a learnable prefix to the text branch. Every other part of LaGoVAD stays as released, so any gain can be traced to these changes.

### 3.1 Base architecture

**LaGoVAD, unchanged parts.** The released code fixes each part:

- **Sampling.** One frame in eight is kept (`interval=8`); for 10 fps dashcam sets (DoTA, CCD) every provided frame is used, as LaGoVAD does for DoTA.
- **CLIP stream.** A frozen CLIP ViT-B/16 image encoder gives one 512-d CLS embedding c\_t per sampled frame t, from a 224×224 resize without cropping.
- **Temporal encoder.** A residual RoFormer: 2 layers, 4 heads, hidden size 512, local window of 25 steps. Training sequences are resampled or padded to 512 steps.
- **Text branch.** The frozen CLIP ViT-B/16 text encoder encodes each category of the anomaly definition Z (a class name or a description) into τ\_c ∈ R^512; the released model carries 32 learnable soft-prompt tokens, which STP's prefix replaces (Section 3.2.3).
- **Fusion.** A 2-layer co-attention Transformer lets video steps and text tokens attend to each other.
- **Heads.** A 1-D convolution head (kernel 9) scores the pre-fusion and post-fusion features; a learned weight w = σ(10·a), a initialised at 0, blends them, and the step score is the sigmoid of the blend.
- **Losses.** Top-k MIL binary cross-entropy (k = max(⌊L/16⌋, 1) for L steps), a multi-class MIL alignment loss, dynamic video synthesis (an abnormal clip inserted among 1–5 kNN-retrieved normal clips, giving frame pseudo-labels), and a caption contrastive loss with hard negatives (temperature 0.02).

**Backbone change: the motion stream.** For each sampled step t, a 16-frame clip at 10 fps centred on t (1.6 s, Simple-TAD's traffic input) passes through a frozen **VideoMAE V2-S** encoder: the ViT-S distilled from VideoMAE V2-g on Kinetics-710, D = 384. Its 8×14×14 last-layer tokens are averaged into one vector and added through a gate that starts closed:

```latex
v_t = \frac{1}{1568} \sum_{j=1}^{1568} X_{t,j} \in \mathbb{R}^{384}, \qquad x_t = \tilde c_t + \tanh(\alpha)\, W\, \mathrm{LN}(v_t), \qquad \alpha_{\text{init}} = 0,\; W \in \mathbb{R}^{512 \times 384}
```

Here c̃\_t is the CLIP-branch feature: c\_t in the preliminary model, and c\_t plus the SA² feature in LaGoVAD-STP (Section 3.2.2). The fused x\_t replaces c\_t as the RoFormer input; the kNN retrieval of dynamic video synthesis still uses c\_t. Trained on Nexar + CCD, the preliminary model reaches **0.81 macro / 0.75 micro per-video min-max AUC zero-shot on DoTA** (one configuration; to be repeated with 3 seeds).

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips and distilled on Kinetics-710; none of the seven traffic datasets used here is among them. Simple-TAD's driving-adapted DAPT-VideoMAE-S (BDD100K only) is kept as an encoder ablation.

&#91;embedded content: LaGoVAD-STP architecture · the two parts of the new component highlighted\]

Only the highlighted SA² step and the text prefix are new; the motion stream is the backbone change, and every other box is LaGoVAD as released.

### 3.2 Novel component: traffic spatio-temporal prompting (STP)

STP follows STPrompt's pattern on both branches. In the CLIP branch: patch CLIP features → SA² → element-wise sum with the CLS CLIP feature. In the text branch: learnable prefix + class text → frozen CLIP text encoder.

#### 3.2.1 Why the earlier components were dropped

Two components were built on the VideoMAE stream and tested before this one.

- **Motion Query Readout (MQR).** Eight zero-initialised, key-free queries (Efficient Probing, ICLR 2026) read the 7×7 VideoMAE token grid, with a spatial top-k MIL loss. Its attention entropy never left its uniform start, and the model without it reached the same DoTA result. It started at mean pooling, a solution the host already had; its queries received a gradient divided by roughly 49·√384 over near-identical masked-model tokens; and every gradient passed through the motion gate and the sparse top-k MIL loss.
- **Latent Surprise Predictor (LSP).** Following FLaRA, a small predictor trained on normal videos forecast the next VideoMAE vector, and the residual entered the fusion through a second gate. It did not improve on the motion-stream model in our tests.

Both tried to extract more from the same global motion vector. The lessons that shape STP are: (1) add evidence the host does **not** already have, here local appearance from a different encoder pass, rather than re-processing the VideoMAE vector; (2) let a **fixed prior** do the selection, as STPrompt's ablation shows, rather than attention learned from video labels alone; and (3) start from a configuration that already works, so a learned part that does not move costs nothing. Details of both negative results will go in an appendix.

#### 3.2.2 Visual part: Spatial Attention Aggregation (SA²) in the CLIP branch

**Patch features.** As in STPrompt, each sampled frame, resized to 224×224, is cut into a 7×7 grid of 32×32 windows (stride 32). Each window is resized to CLIP's input size and encoded by the frozen CLIP image encoder; its CLS embedding is the patch feature P\_{t,j} ∈ R^512, j = 1, …, 49. STPrompt uses crops instead of CLIP's dense tokens because the dense tokens are not supervised with language and already mix global context.

**Motion magnitude and selection** (STPrompt Eq. 1). With P̂ the channel-wise L2-normalised patch features of the neighbouring sampled steps:

```latex
D_{t,j} = 2\hat P_{t,j} - \hat P_{t-1,j} - \hat P_{t+1,j}, \qquad Mo_{t,j} = \lVert D_{t,j} \rVert_2, \qquad \mathcal{S}_t = \operatorname{TopK}_j\big(Mo_{t,j}\big), \quad K = 12
```

**Attention aggregation and element-wise sum** (STPrompt Eqs. 2–3, reconstructed from the text because the equations are not legible in the project copy; to be checked against the published PDF):

```latex
a_{t,k} = \operatorname{softmax}_{k \in \mathcal{S}_t}\!\left( \frac{(W_q c_t)^{\top} (W_k P_{t,k})}{\sqrt{d_a}} \right), \qquad x^{AS}_t = \sum_{k \in \mathcal{S}_t} a_{t,k}\, P_{t,k}, \qquad \tilde c_t = c_t + x^{AS}_t
```

with W\_q, W\_k ∈ R^{128×512}. The frame's CLS feature acts as the query, so the aggregation asks which of the moving patches matter for this frame; x^AS\_t stays in CLIP's embedding space, so the element-wise sum keeps c̃\_t text-aligned. c̃\_t then enters the motion fusion of Section 3.1.

**Ego-compensated variant (ablation, not the default).** Forward ego-motion changes every patch, mostly on near-field road and frame edges. Subtracting the change shared by the whole frame before ranking keeps patches that move *differently* from the scene:

```latex
Mo^{\text{ego}}_{t,j} = \Big\lVert D_{t,j} - \tfrac{1}{49} \textstyle\sum_{i=1}^{49} D_{t,i} \Big\rVert_2
```

**Boundaries and synthesis.** The first and last steps of a video repeat their neighbour. Because Mo uses neighbouring steps, it is computed per original video before LaGoVAD's dynamic video synthesis splices clips, so seams create no false motion.

**Evidence map.** E\_t(j) = a\_{t,j} for j ∈ S\_t and 0 otherwise, reshaped to 7×7, upsampled to the frame and normalised per video. It costs nothing extra and replaces STPrompt's LLM-based localisation.

#### 3.2.3 Text part: learnable prefix

For each category c of the definition Z, with token embeddings t\_c of its class name or description, l learnable context vectors shared by all categories are placed in front, as in STPrompt and CoOp:

```latex
\tau_c = \mathrm{TextEnc}\big( [\, e_1, e_2, \ldots, e_l,\; t_c \,] \big), \qquad e_i \in \mathbb{R}^{512}, \quad l = 8
```

The prefix replaces LaGoVAD's released soft prompts; their exact placement in the released code is checked first, and both versions are compared in the ablation. The prefix is applied to definition texts only, not to the training captions of the caption contrastive loss, so it cannot fit the caption templates. Long descriptions are truncated to the 77-token context minus l and the start and end tokens.

#### 3.2.4 Initialisation

- **Attention.** W\_q is zero-initialised and W\_k Xavier-random, so at the start every selected patch gets weight 1/K and SA² equals STPrompt's "selection + average" variant (87.79 vs 88.08 for full SA² on UCF-Crime). If the attention never learns, as MQR's did not, the model still keeps the part that did most of the work. Attention entropy and the gradient norm of W\_q are logged from the first step.
- **Sum.** x^AS\_t is added at full weight, as in STPrompt; every model is trained from scratch, so there is no pretrained starting point to protect. A zero-initialised gate tanh(γ)·x^AS\_t is a side ablation.
- **Prefix.** e\_i \~ N(0, 0.02²), as in CoOp's generic initialisation; initialising from the word embeddings of "a dashcam video of" is a side ablation.

#### 3.2.5 Why STP should help on traffic video

Five pieces of evidence point the same way:

1. **SA² helps under weak labels, through its fixed prior.** In STPrompt it adds 1.95 AUC on UCF-Crime and 0.98 on UBnormal, and motion-based selection is the only variant that helps on both datasets.
2. **Traffic anomalies are small and local.** The ego-uninvolved gap of Simple-TAD (81.6–85.2 vs 89.9–90.7 AUC), PromptTAD's neglect of distant or off-centre objects and GS-MoE's small-region failure all show that global vectors miss them.
3. **Region-level CLIP evidence helps.** Alert-CLIP moves CLIP's attention from context to the collided vehicles and lifts the VadCLIP framework on MSAD from 87.45 / 71.09 to 89.63 / 78.24, though with boxes; SA² seeks a similar focus without them.
4. **It adds what the host lacks.** The VideoMAE vector carries global motion and c\_t global appearance; SA² adds local appearance where things change, from separate CLIP passes. MQR and LSP instead re-processed the same motion vector.
5. **Text quality matters in LaGoVAD.** Better definitions raise its UCF-Crime AUC from 80.44 (class names) to 81.12 (manual descriptions) and 83.03 (video-specific descriptions). A learned prefix adapts hand-written traffic definitions without an LLM.

The main risk is ego-motion: on dashcam video the top-K moving patches may be road surface and frame edges rather than agents. Step 0 (Section 4.3) measures this before training, and the ego-compensated variant is the planned remedy.

| Property | Motion-stream model (preliminary) | + STP |
| --- | --- | --- |
| Extra parameters | — | about 0.13M (W\_q, W\_k) + 4K (prefix, l = 8) |
| Extra extraction compute | — | 49 CLIP ViT-B/16 passes per sampled frame (window crops) |
| Storage per sampled frame (fp16) | 1 KB (c\_t) + 0.75 KB (v\_t) | + 12 KB (top-12 patch features and indices); 49 KB for the full grid, ablations only |
| Spatial map | none | 7×7 per step |
| Extra loss | — | none |
| VideoMAE input | one 384-d vector per step | unchanged |

### 3.3 Training strategy

Training uses video-level annotations only and LaGoVAD's own schedule, so the baseline, the motion stream and STP differ only in architecture; STP adds no loss.

**Training data.** Only TAD, MSAD, CCD and Nexar contain both normal and abnormal videos, so only they are used for training; DoTA, DADA-2000 and MM-AU contain accident videos only and serve as test sets.

| Protocol | Training set | Normal / abnormal videos | Anomaly text (hand-written) |
| --- | --- | --- | --- |
| P1, TAD | TAD train, UMIL split | 210 / 190 | TAD's 7 category names and definitions |
| P2, MSAD | MSAD weak-protocol train | 360 / 120 | MSAD's 11 anomaly types |
| P3, dashcam | Nexar labelled set + CCD official train split | Nexar 750 / 750; CCD's train part of 3,000 normal / 1,500 crash | Collision, near-miss and car-crash definitions |
| P4, cross-setting | None: reuses the P2 and P3 models | — | Each test set's own definitions |

**Captions.** Every model uses the same hand-written caption template per category, so no LLM-written text enters training. A no-caption variant is reported as an ablation (removing this loss costs LaGoVAD 6.3 AUC on TAD zero-shot).

**Shortcut control.** CCD's crash clips and normal clips (from BDD100K) come from different sources, so a model could learn the source instead of the anomaly. P3 is also reported with Nexar-only training, where positives and negatives share one dashcam network.

**Stages.**

1. **Cache features.** Extract, once: CLIP CLS features c\_t with LaGoVAD's script; the 49 window-crop CLIP features per sampled frame, from which the motion magnitudes, the top-12 indices and the selected patch features are computed and stored; and the VideoMAE V2-S mean vector per step. Because selection is fixed, only the selected patches need storing; the full 7×7 grid is kept for the P3 ablation subset only.
2. **Train all models with one schedule.** Every row (LaGoVAD, motion stream, STP, ablations) starts from the same random initialisation of LaGoVAD's trainable parts, with the same schedule and 3 seeds. W\_q starts at zero, so SA² starts as selection + average.
3. **Monitor the learned parts.** Log SA²'s attention entropy, the gradient norm of W\_q and the norm of the prefix vectors from the first step, so a learned part that does not move, as MQR's did not, is visible at once.

**Settings.** AdamW, batch 64, learning rate 5e-5 with a cosine schedule, 40 epochs, as in LaGoVAD. SA² keeps STPrompt's K = 12 and the prefix STPrompt's l = 8; K ∈ {6, 12, 24}, l ∈ {4, 8, 16} and a learning-rate multiplier of 1 or 10 for the new parameters are chosen once on a held-out 15% of the P3 training videos scored by video-level AUC, then frozen for every protocol. No hyperparameter is chosen on a test list.

**Checkpoint selection.** Every model, baselines included, reports the last epoch of the fixed schedule.

**Compute.** The new parameters are tiny, so training costs about the same as the motion-stream model on one 24 GB GPU. The cost is in extraction: 49 CLIP ViT-B/16 passes per sampled frame. By our estimate (not yet measured) that is roughly 10–11M crops each for Nexar (about 212K sampled frames) and CCD (about 225K frames), and about 7M for DoTA's 1,402 test clips, which is hours, not days, on one GPU. A one-day profiling run in the first week fixes the real numbers.

### 3.4 Inference

At test time the model takes a video and an anomaly definition and returns a score and a 7×7 evidence map per sampled step. No LLM runs at any point.

1. **Sample** one frame in eight (every frame for 10 fps dashcam sets).
2. **Encode** each sampled frame with frozen CLIP (CLS feature c\_t), its 49 window crops with the same CLIP, and the 16-frame clip centred on it with the frozen VideoMAE V2-S, averaged to one 384-d vector.
3. **Aggregate** with SA²: motion magnitudes from neighbouring steps, top-12 selection, attention with c\_t as the query, and the element-wise sum c̃\_t = c\_t + x^AS\_t.
4. **Fuse** c̃\_t with the gated VideoMAE vector.
5. **Encode the definition** with the learned prefix and the frozen CLIP text encoder.
6. **Score** with LaGoVAD's RoFormer, text co-attention and heads. Each step's score is assigned to the frames it stands for.
7. **Map** each step's evidence from SA²'s weights, upsample it to frame size and normalise it per video.

**Definitions.** Every main result uses LaGoVAD's released definitions (TAD as {"Normal", "Car Accident"}; DoTA's "Abnormal" expands to the same car-accident texts). A separate analysis prompts each of TAD's 7 categories with its own hand-written definition.

**Score post-processing.** DoTA and DADA-2000 contain only abnormal clips, so their scores are min-max normalised per video before pooling, following LaGoVAD's `offline_dota_eval.py`. TAD, MSAD and CCD scores are used raw.

**Cost.** The window crops multiply CLIP's cost per frame by about 50, as in STPrompt; one VideoMAE ViT-S clip (about 57 GFLOPs) still runs per step. A cheaper variant that reads ClearCLIP's dense patch tokens from the single full-frame pass is an ablation (R9). The model is offline: the motion magnitude uses the next step and the RoFormer looks both ways.

## 4. Experimental plan

The plan trains with video-level labels only, checks STP's premise with an inference-only pilot before full training, compares against published rows on identical test lists, and adds every second metric by re-scoring released checkpoints.

### 4.1 Dataset preparation

The thesis uses seven traffic datasets. TAD (fixed camera) carries the main comparison and MSAD (multi-scene surveillance with a traffic-accident slice) a second one. CCD and Nexar train the dashcam model, DoTA and DADA-2000 test it zero-shot, and DoTA's boxes test the evidence maps.

| Dataset | View | Size and split | Labels | Role | Source |
| --- | --- | --- | --- | --- | --- |
| TAD | Fixed CCTV, some first-person | 500 videos; UMIL split 400 train (210 normal / 190 abnormal), 100 test (40 / 60); 7 categories | Video labels (train), anomaly interval (test) | P1 train and test; P4 zero-shot test | Frames and split files via the UMIL repository; native fps unknown |
| MSAD | Fixed, 14 scenes | 720 videos at 30 fps; weak protocol 360 normal + 120 abnormal train, 120 + 120 test (20 traffic accidents) | Video labels (train), frame labels (test) | P2 train and test; traffic slice; P4 zero-shot test | MSAD project page (videos gated) |
| CCD | Dashcam | 1,500 crash + 3,000 normal clips (normal from BDD100K), 50 frames at 10 fps, official split | Video labels; accident onset | P3 train and in-domain test | GitHub (Cogito2012/CarCrashDataset) |
| Nexar | Dashcam | 1,500 labelled clips (750 positive + 750 normal), about 40 s each | Video labels; event and alert times (analysis only) | P3 training | Hugging Face |
| DoTA | Dashcam | 4,677 clips (3,275 train / 1,402 test), 10 fps, abnormal only | Anomaly window; 9 categories × ego-involved / ego-uninvolved; boxes with track IDs on 122,447 of 156,728 anomalous frames | P3 zero-shot test; STAUC and pointing game; Step 0 pilot (train split only) | GitHub (MoonBlvd/Detection-of-Traffic-Anomaly) |
| DADA-2000 | Dashcam | 2,000 accident clips at 30 fps | Accident intervals; ego / non-ego | P3 zero-shot test | Simple-TAD's split files |
| MM-AU | Dashcam | 11,727 accident videos pooled from CCD, A3D, DoTA and DADA-2000 | Accident start, collision, end | Optional zero-shot test, after removing videos shared with CCD, DoTA and DADA-2000 | MM-AU release (CVPR 2024) |

**Weights to download** (all public): CLIP ViT-B/16 and VideoMAE V2-S (`vit_s_k710_dl_from_giant`); LaGoVAD's `best.ckpt` and `pred_results.pkl`, used only as a reference row; VadCLIP, PEL4VAD and DSANet checkpoints for zero-shot re-scoring; RTFM and MGFN MSAD checkpoints for the traffic slice; MOVAD and Simple-TAD DoTA and DADA-2000 checkpoints as frame-supervised context; and Simple-TAD's DAPT-VideoMAE-S (BDD100K only, CC BY-NC 4.0) for the encoder ablation. ClearCLIP is a training-free change to the CLIP code, needed only for ablation R9. None of these links has been test-downloaded yet; that check is the first task.

### 4.2 Dataset processing

Every dataset goes through the same pipeline, so one feature cache serves the baseline, the motion stream, STP and all ablations.

1. **Decode and resample.** The CLIP stream keeps LaGoVAD's convention: one frame in eight, or every frame for the 10 fps sets. The VideoMAE stream resamples every video to 10 fps, so a 16-frame clip spans 1.6 s; TAD's clips take 16 consecutive released frames and their duration is reported as unknown.
2. **Cut clips.** For each CLIP step at time t₀, take the 16 frames centred on t₀, repeating edge frames at video boundaries. Resize to 224×224 without cropping.
3. **Extract CLIP CLS features** with LaGoVAD's `extract_feat_clip.py` for all seven datasets, so features match the released checkpoint.
4. **Extract window-crop patch features.** Cut each 224×224 frame into the 7×7 grid of 32×32 windows, encode each window with the same CLIP, compute the motion magnitude per video (raw and ego-compensated), and store the top-12 patch features with their indices and magnitudes in fp16. Keep the full 7×7 grid for the P3 training videos used in ablations and for DoTA's test clips.
5. **Extract the VideoMAE V2-S vector.** Average the 8×14×14 last-layer tokens into one 384-d vector per step (0.75 KB fp16), with the same `fc_norm` choice as the preliminary model.
6. **Build labels.** Training uses video-level labels, category indices and caption templates: TAD from UMIL's `TAD_train.txt`, MSAD from its weak split, CCD from its official split, Nexar from its positive / negative labels. Test labels are frame-level: TAD intervals, MSAD frame labels, CCD accident onsets, DoTA anomaly windows and DADA-2000 intervals.
7. **Prepare DoTA's spatial labels.** For each clip keep `ego_involve`, the category and the per-frame boxes of anomalous agents. Bin clips by the median, over the anomaly window, of the largest box's share of the frame: under 0.5%, 0.5–2%, 2–8%, over 8%. Also mark PromptTAD's small-object subset (anomalous boxes under 2,267 pixels). The obstacle (VO) and out-of-control (OO) categories have no participant boxes and are left out of STAUC.
8. **Fix test lists.** LaGoVAD's lists (DoTA 1,402 clips; its 100-video TAD list) for zero-shot AUC comparability; DoTA's official 1,305-video list for STAUC; UMIL's TAD test list for P1 and as a second zero-shot column; MSAD's weak-protocol test set; CCD's official test split; Simple-TAD's DADA-2000 test split.
9. **Hold out validation data.** 15% of the P3 training videos, stratified by label and source, for the one-time hyperparameter choice.

**Storage** (GB, fp16, estimates):

| Dataset | VideoMAE vector | Top-12 patches (default) | Full 7×7 patch grid |
| --- | --- | --- | --- |
| TAD (all 500 videos) | 0.05 | 0.8 | 3.3 |
| CCD (one step per frame) | 0.17 | 2.7 | 11.3 |
| Nexar | 0.16 | 2.6 | 10.7 |
| DoTA, all clips | 0.56 | 9.0 | 36.7 |
| DADA-2000 | 0.06 | 1.0 | 4.1 |

Only DoTA's 1,402 test clips and a 500-clip training subset are needed, roughly a third of its figures. TAD's quoted duration conflicts with its frame count, so its figures may be off by up to about 5×. MSAD and MM-AU are measured at extraction.

**Leakage checks before any training.**

- Seventy-six of LaGoVAD's 100 TAD test videos are in UMIL's TAD training list, so a model trained on UMIL's split is never scored on LaGoVAD's TAD list.
- MM-AU pools CCD, A3D, DoTA and DADA-2000 videos. It is never used for training, and as a test set only after removing shared videos.
- DoTA's training split is used only for the read-only Step 0 pilot, never to fit or tune anything.
- A near-duplicate check with CLIP-feature similarity runs between every training set (TAD, MSAD, CCD, Nexar) and every test set.
- LaGoVAD's published zero-shot numbers are reproduced from `pred_results.pkl` before anything is trained.

### 4.3 Pilot and baseline running

Baselines run in three groups, preceded by one pilot that checks STP's premise before any full training.

**Step 0: pilot diagnostic (inference only, cached patch features, a random 500-clip subset of DoTA's 3,275-clip training split).** It asks whether STPrompt's motion prior still selects the agents that matter when the camera moves. The 1,402 test clips are never used for design decisions.

| Check | Question | Pass criterion (proposed) | If it fails |
| --- | --- | --- | --- |
| S1: hit rate | In what share of anomalous frames does at least one of the top-12 patches overlap an anomalous-agent box, against the share expected from box area alone? By ego flag and box-size bin, for raw and ego-compensated motion | Clearly above chance, at least on ego-uninvolved clips | Make the ego-compensated motion the default |
| S2: in-box ratio | Mean motion magnitude of patches inside boxes divided by that of patches outside, during the anomaly window | Ratio above 1 | As above; if both fail, switch to text-agent selection (R8) |
| S3: where selection lands | Share of top-12 patches in the bottom row and the left and right edge columns of the grid (near-field road and frame edges) | Below the share these cells hold in the grid (about 39%) | Ego-compensated motion, or exclude the bottom row (side ablation) |

The pilot costs about one GPU-day including patch extraction for the 500 clips. A failure does not stop the plan: it decides which selection prior becomes the default before any training.

**Group A: reproduce LaGoVAD and the motion stream.**

| Row | Model | Training | Purpose |
| --- | --- | --- | --- |
| B0 | LaGoVAD, released checkpoint | PreVAD (by its authors) | Reproduce the published zero-shot numbers from `pred_results.pkl` (TAD 89.56, DoTA 62.60, MSAD 90.41), then re-score on UMIL's TAD list, DADA-2000, CCD and the MSAD traffic slice with AUC, AP and AUC\_A |
| B1 | LaGoVAD, retrained | Each protocol's training set, 3 seeds | Fair baseline with exactly the thesis data, features and schedule; separates the gain of domain-matched data from the gain of motion |
| B2 | LaGoVAD + motion stream (one VideoMAE V2-S vector) | Same, 3 seeds | The preliminary 0.81 / 0.75 model; STP's starting point (ablation row R0) |

**Group B: published methods with released weights, re-scored for a second metric.** VadCLIP and DSANet (UCF-Crime and XD-Violence checkpoints, on the same CLIP ViT-B/16 features); PEL4VAD (with I3D features extracted for TAD, DoTA and DADA-2000); RTFM and MGFN (MSAD checkpoints, for the traffic slice); MOVAD and Simple-TAD (DoTA and DADA-2000, frame-supervised context only). Each runs on the same test lists and is scored with AUC, AP and, where defined, AUC\_A.

**Group C: same-feature competitors retrained in-domain.** VadCLIP's public code is retrained on P1, P2 and P3 with the thesis's CLIP features. STPrompt has no public code, so its detection branch (VadCLIP + SA² + learnable prefix + distance-based temporal adapter, without LLM localisation) is reimplemented on the same cached features and retrained on P1–P3. This row shows whether LaGoVAD's host and the motion stream add anything over STPrompt itself. Methods without released weights are quoted from their papers only: UMIL and WSAL on TAD; π-VAD, GS-MoE and Alert-CLIP on MSAD; MultiDomain and WS-DoTA; and the frame-supervised TTHF, PromptTAD and ISCRTAD.

### 4.4 Evaluation protocols

Four protocols cover in-domain and zero-shot use on fixed-camera and dashcam video. P1 meets the comparison rule (at least two published methods on at least two metrics, same fixed-camera traffic data) without recomputation. P2 adds four-metric rows on MSAD. P3 and P4 meet the rule on DoTA, TAD and MSAD once released checkpoints are re-scored for the second metric.

**Metrics.**

- **AUC\_O / AP:** frame-level ROC AUC and average precision, pooled over all test frames.
- **AUC\_A / AP\_A:** the same, over abnormal test videos only; they measure localisation inside abnormal videos.
- **DoTA and DADA-2000 macro and micro AUC:** after per-video min-max normalisation (LaGoVAD's DoTA protocol). Micro pools all frames after normalisation; macro averages per-video AUC. Both are reported overall, for ego-involved and ego-uninvolved clips, per box-size bin and on PromptTAD's small-object subset.
- **STAUC:** DoTA's spatio-temporal AUC. For each true-positive frame, TARR is the fraction of the top-N% highest-scoring map pixels that fall inside the union of anomalous-agent boxes; STAUC is the area under the TARR-weighted true-positive rate against the false-positive rate, so STAUC ≤ AUC. It is computed on DoTA's official 1,305-video list with `stauc.py`, adapted to take the 7×7 maps upsampled to its 256×256 grid.
- **Pointing game:** the share of anomalous frames whose map peak falls inside a box.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P1 TAD in-domain (fixed camera) | TAD train → TAD test, UMIL split | AUC\_O, AUC\_A, AP | UMIL and WSAL on both metrics; AUC\_O-only rows; LaGoVAD, VadCLIP and STPrompt retrained |
| P2 MSAD in-domain (multi-scene surveillance) | MSAD weak train → MSAD test; traffic slice and non-traffic scenes scored separately | AUC, AP, AUC\_A, AP\_A | π-VAD and GS-MoE (four metrics) and Alert-CLIP (two); RTFM, MGFN, UR-DMU on the traffic slice |
| P3 dashcam | Nexar + CCD train → CCD test (in-domain); DoTA and DADA-2000 (zero-shot) | AUC, AP; DoTA macro / micro by ego flag and box size; STAUC, pointing game | DoTA: LaGoVAD, VadCLIP, PEL (published AUC, AP re-scored), WS-DoTA flagged; CCD and DADA-2000: released checkpoints re-scored; frame-supervised rows as context |
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

**DoTA rows and their protocols.** Only rows under the same per-video normalisation are ranked against the thesis; the rest are context.

| Method | Supervision / training data | DoTA protocol | AUC | Comparable to thesis? |
| --- | --- | --- | --- | --- |
| Thesis motion-stream model (R0) | Video-level, Nexar + CCD, zero-shot | Per-video min-max, 1,402 clips | 0.81 macro / 0.75 micro | Reference |
| LaGoVAD | Video-level, PreVAD, zero-shot | Per-video normalised | 62.60 | Yes (against micro) |
| VadCLIP | Weak, UCF-Crime / XD-Violence / PreVAD, zero-shot | Same | 50.93 / 49.00 / 50.59 | Yes |
| PEL (I3D) | Weak, UCF-Crime / XD-Violence, zero-shot | Same | 53.05 / 44.97 | Yes |
| AnoPred; DoTA ensemble; MAMTCF | Unsupervised, DoTA in-domain | Pooled raw scores | 67.5; 73.0 (STAUC 48.5); 76.6 | Context only |
| WS-DoTA (RTFM + transform) | Weak, D²-City + DoTA in-domain | \~1,031 clips, normalisation not stated | 78.2 | Context only |
| MOVAD, PromptTAD, TTHF, ISCRTAD | Frame labels, DoTA in-domain | Pooled; TTHF post-processed | 82.2, 83.9, 84.7, 86.2 | Upper reference only |
| Simple-TAD DAPT-VideoMAE-S / -L | Frame labels, fine-tuned, DoTA in-domain | Pooled | 86.4 / 88.4 | Upper reference only |
| FLaRA; BADAS | Clip-level, Nexar | Video-level anticipation AUC, 327 positive / 40 synthetic negative clips | 90.0; 90.6 | No |

The 0.75 micro sits 12.4 points above LaGoVAD's 62.60 under the same normalisation; row B1 separates how much of this comes from dashcam training data and how much from motion.

**Reporting rules.**

- TAD is reported on two separate columns, LaGoVAD's list and UMIL's list, never mixed.
- Every thesis number is a mean ± standard deviation over 3 seeds.
- Subsets (ego flag, box-size bins, categories, traffic slices) carry 95% bootstrap confidence intervals over videos. STP against the motion-stream model is tested with a paired bootstrap.
- All models use the same test lists, frame-to-score mapping and normalisation, and every table names each row's training set and protocol.

### 4.5 Ablation studies

The ablation ladder repeats STPrompt's own spatial ladder on traffic data, then adds the text prefix and the traffic-specific variants. Every row keeps VideoMAE at one global vector per step, uses video-level labels only, and reports P1 (AUC\_O, AUC\_A), P2 (AUC, AP), P3 (CCD AUC; DoTA macro / micro by ego flag and box size; DADA-2000 AUC), P4 on TAD's two lists, and STAUC where a map exists. The expected outcomes are priors written down in advance.

| Row | Variant | Isolates | Expected if STP's premise holds |
| --- | --- | --- | --- |
| R0 | Motion stream only (c\_t + gated v\_t) | Reference | 0.81 / 0.75 on DoTA |
| R1 | + average of all 49 patch features | More local features, no selection | Small or no gain (STPrompt: +0.8 UCF, −0.1 UBnormal) |
| R2 | + attention over all 49 patches (no selection) | Learned selection alone | About R1 or below, as in STPrompt |
| R3 | + motion top-12 selection + plain average | The fixed motion prior | Gain over R1 and R2, largest on ego-uninvolved and small-box clips |
| R4 | + SA² (selection + attention), the visual part of STP | Learned weighting on top of the prior | Small further gain over R3; STAUC above R3's uniform weights |
| R5 | R0 + learnable prefix only | The text part alone | Small zero-shot gain; drift@5 no worse |
| **R6** | **R4 + learnable prefix (full LaGoVAD-STP)** | Both parts together | **Best overall; gains concentrated on ego-uninvolved and small-box clips** |
| R7 | R6 with ego-compensated motion | Whether removing the frame-wide change helps under ego-motion | ≥ R6 on dashcam sets, about equal on fixed-camera TAD and MSAD |
| R8 | R6 with selection by hand-written agent text ("a car", "a pedestrian", …) instead of motion | A language prior instead of a motion prior | Better on dashcams if S1–S3 fail; otherwise below R6 |
| R9 | R6 with dense ClearCLIP patch tokens from one pass, pooled to 7×7, instead of 49 crops | Cost against quality | Somewhat below R6 at about 1/50 of the CLIP cost |
| R10 | R6 with K ∈ {6, 12, 24} | Selection size | K = 12 near best; small K favours small agents |
| R11 | R6 with LaGoVAD's released 32 soft-prompt tokens instead of the prefix | Prefix form against the released prompts | At or below R6 |

H1 holds if R4 beats R0 with larger gains on ego-uninvolved and small-box clips; H2 if R3 and R4 beat R1 and R2, and R7 ≥ R6 on dashcam sets; H3 if R6 beats R4 on zero-shot AUC without a worse drift@5; H4 if R4 or R6 beats a uniform map and a centre prior on STAUC and the pointing game.

**Side ablations within the component.**

- **Fusion:** plain element-wise sum (default, as in STPrompt) against a zero-initialised gate tanh(γ)·x^AS\_t.
- **Attention query:** CLS-guided attention (default) against weights from a softmax over the selected patches' motion magnitudes, which has no learned parameters.
- **Motion spacing:** neighbours at ±1 step (default) against ±2 steps, since 10 fps dashcam steps are only 0.1 s apart.
- **Crop resolution:** windows cut from the 224×224 resize (default, as in STPrompt) against the same 7×7 grid cut from the native-resolution frame, which keeps more pixels of distant agents.
- **Prefix:** length l ∈ {4, 8, 16}; random against "a dashcam video of" initialisation; with and without STPrompt's contrastive loss that keeps label embeddings apart.
- **Encoder:** VideoMAE V2-S (default) against VideoMAE-S (Kinetics-400) and DAPT-VideoMAE-S (BDD100K), each still one 384-d vector.
- **Captions:** with and without LaGoVAD's caption contrastive loss.

**Analyses.**

1. **Where the gain lands.** DoTA ego-involved vs ego-uninvolved clips crossed with box-size bins and PromptTAD's small-object subset; the nine categories; TAD Accident, PedestrianOnRoad and the remaining categories pooled; the MSAD traffic slice against its non-traffic scenes; Nexar-only against Nexar + CCD training.
2. **Spatial evidence.** STAUC and pointing game for SA²'s maps against a uniform map (R0), the unweighted selection map (R3) and a centre prior, with examples of successes and failures.
3. **Selection behaviour.** Where the top-12 patches land on dashcam vs fixed-camera video (the S3 statistic on test data), raw against ego-compensated motion.
4. **Learned parts.** Attention entropy over training (does SA²'s attention leave the uniform start, unlike MQR's?), and the prefix's effect on definition sensitivity: score change of accident frames when the definition includes or excludes accidents, and drift@5 on MSAD.
5. **TAD definitions.** Main results with LaGoVAD's two-class TAD prompt against each of TAD's 7 categories with its own definition.
6. **Cost.** Parameters, storage and measured frames per second for rows R0, R6 and R9.

**Budget.** Rows R0–R6 run on P1, P2 and P3 with 3 seeds; P4 reuses those models. Rows R7–R11 and the side ablations run on P1 and P3 only, first with one seed, then with three for the variants that are kept. Patch extraction is done once and shared by every row. A one-epoch timing run in the first week converts this plan into GPU-hours; if the total exceeds the budget, the ladder is cut to R0, R1, R3, R4 and R6, which still test H1–H4.

## References

**Baseline, backbone and component**

- Liu et al., [Language-Guided Open-World Video Anomaly Detection under Weak Supervision (LaGoVAD)](https://arxiv.org/abs/2503.13160), ICLR 2026; [code, checkpoint, PreVAD annotations](https://github.com/Kamino666/LaGoVAD-PreVAD)
- Wu et al., [Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts (STPrompt)](https://arxiv.org/abs/2408.05905), ACM MM 2024 (source of SA² and the learnable prefix)
- Wu et al., VadCLIP, AAAI 2024; [code](https://github.com/nwpu-zxr/VadCLIP)
- Zhou et al., Learning to Prompt for Vision-Language Models (CoOp), IJCV 2022
- Wang et al., [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727), CVPR 2023 (motion encoder: `vit_s_k710_dl_from_giant`)
- Tong et al., [VideoMAE](https://arxiv.org/abs/2203.12602), NeurIPS 2022
- Orlova et al., [Simplifying Traffic Anomaly Detection with Video Foundation Models (Simple-TAD)](https://arxiv.org/abs/2507.09338), ICCV 2025 Workshops; [code and DAPT encoders](https://github.com/tue-mps/simple-tad)
- Radford et al., Learning Transferable Visual Models from Natural Language Supervision (CLIP), ICML 2021

**Dropped components (Section 3.2.1)**

- Psomas et al., [Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency (Efficient Probing)](https://arxiv.org/abs/2506.10178), ICLR 2026 (basis of MQR)
- Przewięźlikowski et al., [Beyond \[cls\]: Exploring the True Potential of Masked Image Modeling](https://arxiv.org/abs/2412.03215), ICCV 2025
- Caselli et al., [FLaRA: Predicting Future Latent Representations for Accident Anticipation](https://arxiv.org/abs/2606.14380), arXiv 2606.14380; [code](https://github.com/LoreCase073/FLaRA) (basis of LSP)
- Goldshmidt et al., [BADAS](https://arxiv.org/abs/2510.14876), arXiv 2510.14876

**Weakly supervised and open-world VAD**

- Pu et al., PEL4VAD, IEEE TIP 2024; [code and checkpoints](https://github.com/Aaron-Pu/PEL4VAD)
- [DSANet](https://arxiv.org/abs/2511.10334), AAAI 2026; [code](https://github.com/lessiYin/DSANet)
- Lv et al., [Unbiased Multiple Instance Learning (UMIL)](https://arxiv.org/abs/2303.12369), CVPR 2023; [TAD split files](https://github.com/ktr-hubrt/UMIL)
- Lv et al., Localizing Anomalies from Weakly-Labeled Videos (WSAL, TAD dataset), IEEE TIP 2021; [repository](https://github.com/ktr-hubrt/WSAL)
- [MultiDomain](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf), NeurIPS 2024
- [π-VAD](https://arxiv.org/abs/2505.13123), CVPR 2025; [GS-MoE](https://arxiv.org/abs/2508.06318), ICCV 2025
- Alert-CLIP, CVPR 2026 (project file); [annotations](https://github.com/ClarkZhu216/Alert-CLIP_dataset)
- [WS-DoTA](https://arxiv.org/abs/2408.05562), ECCV 2024 Workshop; [split files](https://github.com/ut21/WSAD-Driving)
- [Localizing to Debias](https://arxiv.org/abs/2608.12045), arXiv 2608.12045; [Learning Where and When](https://engineers.usf.edu/sislab/publications/learning-where-and-when-patch-based-spatiotemporal-localization-weakly-supervised), arXiv 2606.29498

**Traffic anomaly detection**

- Qiu et al., [PromptTAD: Object-Prompt Enhanced Traffic Anomaly Detection](https://doi.org/10.1109/LRA.2025.3572428), IEEE RA-L 2025
- [MAMTCF](https://arxiv.org/abs/2307.14575), arXiv 2307.14575
- Liu et al., Future Frame Prediction for Anomaly Detection (AnoPred), CVPR 2018
- Rossi et al., MOVAD, ICASSP 2024; [code](https://github.com/IMPLabUniPr/movad)
- Liang et al., TTHF, IEEE TCSVT 2024; Liang et al., ISCRTAD, IEEE T-ITS 2025
- Shi et al., [Structured Graph Reasoning for Traffic Anomaly Detection](https://doi.org/10.1109/TMM.2026.3703345), IEEE TMM 2026

**Dense CLIP features (ablations R8–R9)**

- Lan et al., [ClearCLIP](https://arxiv.org/abs/2407.12442), ECCV 2024; Wang et al., [SCLIP](https://arxiv.org/abs/2312.01597), ECCV 2024
- Zhou et al., [AnomalyCLIP](https://arxiv.org/abs/2310.18961), ICLR 2024

**Datasets**

- Yao et al., [DoTA](https://arxiv.org/abs/2004.03044), IEEE TPAMI 2022; [annotations and STAUC code](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
- Zhu et al., [MSAD](https://msad-dataset.github.io/), NeurIPS 2024 Datasets and Benchmarks
- Bao et al., [CCD](https://github.com/Cogito2012/CarCrashDataset), ACM MM 2020
- Moura et al., [Nexar Dashcam Collision Prediction Dataset and Challenge](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf), CVPR 2025 Workshops
- Fang et al., DADA-2000, IEEE T-ITS 2021; Fang et al., [MM-AU](https://arxiv.org/abs/2403.00436), CVPR 2024
- Yu et al., BDD100K, CVPR 2020; D²-City, arXiv 2019
