# Thesis proposal: Deep learning-based traffic video anomaly detection (LaGoVAD-STP v2)

Nov 4, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-STP v2**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. It starts from LaGoVAD (ICLR 2026) and adds a frozen VideoMAE V2-S motion stream, read as one 384-d global vector per step, as the permitted backbone change. Trained on Nexar dashcam video with bag-level labels (20-step windows, Section 3.3), this model already reaches **0.81 macro / 0.73 micro per-clip min-max AUC zero-shot on DoTA-CAP-dev** (569 clips, one seed), against 0.64 / 0.61 for the released LaGoVAD checkpoint read the same way. The comparison is not matched: the checkpoint was trained on PreVAD.

The one novel component is **traffic spatio-temporal prompting (STP)**, adapted from STPrompt (ACM MM 2024). This version redesigns its visual part. In the first version, the SA² attention barely learned (its entropy fell only from 1.0 to about 0.91), and the first SA² added nothing measurable on top of the VideoMAE stream (+0.0001 macro, one seed). **SA²-v2** therefore scores all 49 CLIP crops of a frame with a learned scorer built from a fixed motion and outlier prior, a text-conditioned term and a learned residual. It pools the crops by softmax, and it is trained with patch-level multiple-instance learning (MIL), a branch-only head and motion dropout. The text side keeps the learnable prefix.

The claim under test: local, prior-guided and patch-supervised appearance evidence recovers small and distant traffic agents that one global CLIP vector dilutes, and the patch scores double as spatial evidence maps scored with DoTA's STAUC. Expect a modest change in overall AUC (STPrompt's own learned attention was worth about 0.3 AUC); the main tests are STAUC, the pointing game and the small-box and ego-uninvolved subsets of DoTA-CAP.

## What changed in this version

Only the visual part of STP changes. The LaGoVAD host, the VideoMAE motion stream and the text prefix stay as before.

**Kept unchanged**

- LaGoVAD host: 2-layer RoFormer, text co-attention, conv heads, MIL and alignment losses, dynamic video synthesis. The caption loss is not used on Nexar.
- Frozen CLIP ViT-B/16: frame feature c\_t (512) and the 49 window-crop features P (49×512, a 7×7 grid of 32×32 windows).
- Frozen VideoMAE V2-S: one 384-d vector per step, added through the zero-initialised projection W\_u.
- Learnable text prefix (l = 8) before the frozen CLIP text encoder.
- The host still receives one 512-d vector per step, so its input size does not change.

**Old SA² against SA²-v2**

|  | Old SA² (version 1) | SA²-v2 |
| --- | --- | --- |
| Crops used | 12, chosen by a hard motion top-K | all 49, weighted softly |
| Weights | CLS-queried attention, W\_q zero-initialised (STPrompt's own attention appears to be parameter-free) | logits = fixed prior + text term + learned residual, z-scored per frame |
| Motion | selector only | selector and a learned input (ΔP) |
| Supervision | host loss only | host loss + patch-level MIL + branch-only head + motion dropout |
| Extra parameters | about 0.13M | about 0.9M |
| Output to the host | x\_AS: 512, summed with c\_t | x\_AS: 512, summed with c\_t (same) |
| Spatial output | 7×7 attention map (weights) | 7×7 patch probability map |

**Input and output of each new part (one sampled step t)**

| Part | Input | Output | Learned |
| --- | --- | --- | --- |
| Cached inputs | P: 49×512; c\_t: 512; ΔP: 49×512; priors Mo, ρ: 49 each; text vector τ\_Δ: 512 | none | no |
| A. Patch encoder | \[P\_j, P\_j − c\_t, ΔP\_j\] (1536 per crop) + grid position + priors | H: 49×256 | yes, about 0.9M parameters in A and B together |
| B. Patch scorer | H, priors, cosine of each crop to τ\_Δ | logits ℓ: 49 | yes (three scalars + a linear residual head) |
| C. Outputs | ℓ | map q: 7×7; pooled feature x\_AS: 512 | no extra parameters |
| Patch-level MIL (training only) | q for all crops, bag label, host frame scores | patch loss | no |
| Branch-only head (training only) | c̃\_t with the VideoMAE term off | one frame score per step | 513 parameters |
| Motion dropout (training only) | VideoMAE term W\_u(c·ṽ\_t) | zeroed in 30% of training bags | no |
| Definition flip (training only) | definition set Z and bag label | Z without the genuine class, label set to normal | no |

**Open points to check before running**

- STPrompt's Eq. 2 is not legible in the project copy. The shapes it gives (Mo\_TOP of size T×K×1) fit a parameter-free softmax over motion magnitudes; the published PDF decides this.
- Whether LaGoVAD's training already removes the genuine class from a definition. The definition flip may then be redundant.
- Sizes (256, one layer, two heads), the 30% dropout, m = 3 crops and k = 5 neighbours in ρ are my design choices, not taken from a paper; they are tuned once on Nexar's validation split (Section 3.3).

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event is happening in it. Abnormal events include collisions, loss of control, illegal turns, wrong-way driving, pedestrians on the road and road spills. These are the categories used by TAD and DoTA.

Two camera settings define the problem space:

- **Fixed camera (non-ego).** TAD and the traffic scenes of MSAD. The camera is static; the anomalous agents are often small and far away.
- **Dashcam (ego-centric).** DoTA, DADA-2000, CCD and Nexar. The camera moves with the car. DoTA also marks each clip as ego-involved (the ego car takes part in the anomaly) or ego-uninvolved (it only observes); this proposal uses those two terms for that flag.

The thesis follows LaGoVAD's weakly supervised, language-guided formulation. Training uses one label y\_i ∈ {0, 1} per bag, plus a category. On TAD and MSAD a bag is a whole video. On Nexar, the set behind every preliminary number, a bag is a 20-step window whose label comes from the annotated alert and event times, so this supervision is weaker than frame labels but stronger than one label per video. At test time the model receives a video V and a text definition Z of what counts as anomalous, and returns a score per sampled step. This thesis adds a coarse spatial evidence map per step as a second output.

```latex
\Phi_\theta : (V, Z) \;\mapsto\; \{ s_t \}_{t=1}^{N},\; \{ E_t \}_{t=1}^{N}, \qquad s_t \in [0,1],\;\; E_t \in [0,1]^{7 \times 7}
```

The bag labels supervise training through multiple-instance learning (MIL): the mean of the top-k step scores of a bag must match its label.

Four properties make traffic video harder than generic surveillance:

1. **Anomalies are motion events.** A collision is defined by how trajectories meet, not by what one frame looks like. LaGoVAD encodes single RGB frames with CLIP and has no motion input; it publishes 89.56 AUC on TAD but 62.60 on dashcam DoTA. Part of this gap is protocol: DoTA is abnormal-only and min-max normalised per video, so its AUC measures timing inside a clip. Re-scoring the released checkpoint here gives 0.7912 micro / 0.7579 macro on TAD and 0.6105 micro / 0.6418 macro on DoTA-CAP-dev; TAD's published number could not be reproduced and its micro AUC mostly measures clip classification (Section 4.4).
2. **The agents that matter are small.** A distant agent of about 40×30 px in a 1280×720 frame shrinks to about 7×9 px at 224×224 input, smaller than one 16-px patch (estimate). One global CLIP vector per frame dilutes it; this is what the novel component targets.
3. **The camera moves.** In dashcam video, ego-motion dominates frame-to-frame change. In preliminary experiments, hand-crafted motion cues (feature differences, ego-motion-compensated optical flow) did not help as direct inputs; here a motion cue is used as a prior and as one input among several, and a variant compensates for the change shared by the whole frame (on DoTA-CAP-dev the variant pointed at agents no better than the raw cue, Section 3.2.1).
4. **Labels are scarce.** Frame-level labels are expensive and anomalies are rare, so the model must learn from bag-level labels.

The scope is offline detection: the whole video is available and the question is whether an anomaly is happening, not whether one will happen (anticipation, as in Nexar). No large language model or multimodal LLM runs at training or test time; a frozen CLIP text encoder is allowed. All training text is hand-written: class names and category definitions (Nexar uses {Normal, CarAccident}, and the caption path does not run on it). STPrompt's LLM-written localisation prompts are therefore not used.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from cheap bag-level labels, sees motion, attends to the part of the frame where things change, and shows where that is. It must keep LaGoVAD's ability to take the anomaly definition as text, and run on one academic GPU without any LLM.

The thesis builds this in two steps, matching the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** A frozen VideoMAE V2-S encoder (CVPR 2023), the strongest small masked-video encoder for traffic anomalies in Simple-TAD's comparison, feeds one 384-d vector per step into LaGoVAD through the zero-initialised projection W\_u. Preliminary result: zero-shot DoTA-CAP-dev **0.81 macro / 0.73 micro** per-clip min-max AUC (one seed); the Nexar-trained CLIP-only model reads 0.71 macro and the same model with VideoMAE 0.82 (5 seeds). Two attempts to extract more from this stream (MQR, LSP) added nothing, so it stays one global vector.
- **Novel component (STP).** In the CLIP branch, **SA²-v2** weights all 49 crop features with a learned, text-conditioned, prior-biased scorer, trained with patch-level MIL so that the branch carries signal of its own; in the text branch, a learnable prefix gives every anomaly-class text a learned context.

Why SA² is redesigned. In the first version, SA²'s attention entropy fell only from 1.0 to about 0.91, and the gain over CLIP-only LaGoVAD came from the VideoMAE stream. Three causes are likely: re-weighting 12 near-identical crop features has almost no leverage on the pooled vector; a zero-initialised W\_q gives its key matrix zero gradient at the start; and the host loss reaches the module only through the VideoMAE-fused features and a sparse top-k MIL loss. SA²-v2 addresses these with a prior-biased scorer over all 49 crops and with losses that do not pass through the VideoMAE shortcut (Section 3.2).

A second purpose is a fair evaluation: published traffic numbers are mostly single-metric AUC on inconsistent splits and protocols. The thesis will report at least two metrics against at least two published methods on the same fixed-camera traffic data, plus zero-shot and dashcam results, and label each row's protocol.

### 1.3 Research aims

The thesis has five aims:

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint (done on DoTA-CAP-dev, TAD, DADA-2000 and DAD; MSAD on the earlier 18,350-frame cache) and retrain LaGoVAD on the thesis training data, with AUC, AP and AUC on abnormal videos (AUC\_A).
2. **Document the motion stream.** Report the frozen VideoMAE V2-S stream's gain over CLIP-only LaGoVAD, and the negative MQR and LSP results that fixed it at one global vector.
3. **Diagnose and redesign the CLIP-branch component.** Measure whether the first SA² contributes (uniform-replacement test, branch-only run), then design and train SA²-v2 and the learnable prefix using bag-level labels only.
4. **Measure spatial localisation under weak supervision.** Score the patch maps against DoTA's anomalous-agent boxes, on the DoTA-CAP clips that have them, with STAUC and a pointing game.
5. **Explain the gains.** Break results down by ego-involved vs ego-uninvolved clips, agent box size and anomaly category; test whether the motion prior survives ego-motion; and check that MSAD's non-traffic scenes do not regress.

Aims 1–2 are prerequisites; aims 3–5 map to the hypotheses below. "Supported" means the 3-seed mean difference is at least +0.01 and the low bound of its 95% paired bootstrap confidence interval, clustered by source video, is above zero. Ablation rows are defined in Section 4.5.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. SA²-v2 improves on the motion-stream model, mostly on small and ego-uninvolved agents | DoTA macro / micro AUC by ego flag and box-size bin; TAD AUC by category (rows R0, R8) | R8 > R0 overall, with larger gains on ego-uninvolved clips and small-box bins |
| H2. A fixed prior helps, and patch-level supervision adds to it | STPrompt's ladder (R1–R3), then the SA²-v2 steps (R4–R8); raw vs ego-compensated motion (R11) | R3 beats R1; R8 beats R3 on DoTA subsets and STAUC; on dashcam sets R11 ≥ R10 |
| H3. The learnable prefix improves transfer without weakening language guidance | Zero-shot DoTA, DADA-2000 and TAD AUC; MSAD drift@5 (R9, R10) | R10 > R8 on zero-shot AUC, and drift@5 no worse |
| H4. The patch maps point at the anomalous agents | STAUC and pointing game on the boxed DoTA-CAP clips (528 of the 569 dev clips) | SA²-v2's map beats a uniform map, a centre prior and the unweighted selection map (R3) |
| H5. The gain holds across camera settings without hurting surveillance | TAD AUC\_O / AUC\_A; MSAD AUC / AP, its traffic slice and non-traffic scenes | Gains on TAD, the MSAD traffic slice and the dashcam sets; no significant drop on MSAD's non-traffic scenes |
| H6. The patch branch carries signal of its own, beyond the VideoMAE stream | Branch-only run without the VideoMAE term (R8 branch-only against CLIP-only LaGoVAD, B1); uniform-replacement ΔAUC | Branch-only AUC above B1, and replacing SA²-v2's weights by uniform weights lowers AUC |

**Preliminary results so far (first SA², one seed).** On H1, the split runs against the hypothesis: first SA² minus host is +0.012 on ego-involved clips and −0.013 on ego-uninvolved clips. On H2, the ego-compensated prior points no better than the raw one. On H3, the prefix did not demonstrably train, so nothing is known yet. On H6, a patch-only local branch (without text) reached 0.61 macro, and 0.70 with its motion magnitudes in the token, and it underfit Nexar. Section 3.2.1 gives the numbers.

## 2. Literature review

### 2.1 Overview and classification of video anomaly detection method branches

Video anomaly detection (VAD) methods divide by the supervision they train with into five branches, plus the adjacent task of accident anticipation. Traffic methods cut across these branches along a second axis: fixed camera (non-ego) or dashcam (ego-centric). The table places the papers reviewed for this thesis, mostly 2024–2026.

| Branch | Training signal | Typical mechanism | Representative methods (venue) | Traffic evidence | Role in this thesis |
| --- | --- | --- | --- | --- | --- |
| Unsupervised / one-class | Normal videos only | Reconstruction, frame or flow prediction, future object location, diffusion, scene-graph normality | AnoPred (CVPR 2018), FOL / DoTA (IEEE TPAMI 2022), MAMTCF (arXiv 2307.14575), MA-PDM (AAAI 2025), ADSM (CVPR 2025), TAO (CVPR 2025), Structured Graph Reasoning (IEEE TMM 2026) | DoTA AUC 67.5–76.6 and STAUC (DoTA baselines up to 48.5) | Context; ego-motion breaks most predictors |
| Weakly supervised | Video-level labels, MIL | Snippet features (I3D, CLIP) + top-k MIL; CLIP text prompts with learnable context; prototypes; spatial aggregation of patches | Sultani (CVPR 2018), RTFM (ICCV 2021), WSAL (IEEE TIP 2021), UMIL (CVPR 2023), MultiDomain (NeurIPS 2024), VadCLIP (AAAI 2024), **STPrompt (ACM MM 2024)**, π-VAD (CVPR 2025), GS-MoE (ICCV 2025), DSANet, RefineVAD (AAAI 2026), Alert-CLIP (CVPR 2026) | TAD (WSAL, UMIL), MSAD (π-VAD, GS-MoE, Alert-CLIP), dashcam WS-DoTA (ECCV 2024 Workshop) | **The thesis's branch; STPrompt is the source of the novel component** |
| Open-vocabulary / open-world | Video labels + category text | CLIP text space defines anomaly classes or free-text definitions at test time | Anomize (CVPR 2025), **LaGoVAD (ICLR 2026)** | Zero-shot TAD, DoTA, MSAD (LaGoVAD) | **Baseline** |
| Fully supervised traffic detection | Frame-level labels (+ boxes) | Video encoder + temporal head; object prompts; agent graphs | MOVAD (ICASSP 2024), TTHF (IEEE TCSVT 2024), PromptTAD (IEEE RA-L 2025), ISCRTAD (IEEE T-ITS 2025), Simple-TAD (ICCV 2025 Workshops) | DoTA AUC 82.2–88.4 in-domain | Upper reference rows; evidence that local, object-level evidence helps |
| LLM / MLLM-based | None, or few-shot rules, or light tuning, or pseudo-anomaly pixel masks | Captioning + LLM reasoning, prompting, probing of MLLM heads, MLLM + SAM2 mask decoder | LAVAD (CVPR 2024), AnomalyRuler (ECCV 2024), VERA, Holmes-VAU (CVPR 2025), EventVAD (ACM MM 2025), HeadHunt-VAD (AAAI 2026), **LAVIDA (CVPR 2026)**, AnyAnomaly (WACV 2026) | Almost none on traffic sets | Excluded for cost; LAVIDA supplies three ideas (Section 2.3.8) |
| Accident anticipation (adjacent task) | Positive / negative dashcam clips | Predict a collision before it happens, recently by forecasting future latents | DSTA, UString (ACM MM 2020), BADAS (arXiv 2510.14876), FLaRA (arXiv 2606.14380) | Nexar, CCD, DAD; video-level anticipation AUC | Different task; FLaRA's latent prediction was tested as LSP and dropped |

Three findings from this review shape the proposal. First, CLIP-feature weak supervision dominates surveillance benchmarks but sits near chance on dashcam DoTA, and the strongest traffic results come from masked video foundation models. Second, the methods that win on small, distant or off-centre traffic agents (PromptTAD, ISCRTAD, Alert-CLIP) add local evidence, but through detectors, boxes or region captions. Third, STPrompt is the one weakly supervised design that adds local evidence with no detector and no boxes, and its own ablation shows that a fixed motion prior does most of the work; it has never been tested on traffic or under a moving camera.

### 2.2 Emerging trends and unresolved challenges

Seven trends run through the 2024–2026 literature:

1. **Vision-language features as the default.** Since VadCLIP (AAAI 2024), most weakly supervised methods start from frozen CLIP frame features and text prompts, usually with CoOp-style learnable context tokens: STPrompt (ACM MM 2024), DSANet and RefineVAD (AAAI 2026), LAS-VAD, Fine-VAD and Alert-CLIP (CVPR 2026).
2. **Anomalies defined by language.** Open-vocabulary (Anomize, CVPR 2025) and open-world (LaGoVAD, ICLR 2026) models accept the anomaly definition at test time.
3. **MLLMs for explanation and training-free detection.** LAVAD, VERA, Holmes-VAU, HeadHunt-VAD and LAVIDA rely on LLMs or multimodal LLMs of 2B parameters and more. They are not clearly better at detection: on MSAD drift@5, LaGoVAD reaches 85.6 AUC / 40.1 AP against 84.3 / 34.3 for Holmes-VAU.
4. **Masked-video foundation models for traffic.** Simple-TAD showed that masked video modelling transfers best to traffic anomalies; VideoMAE-B reaches AUC-MCC 56.0 on DoTAref against 50.0 for InternVideo2-B after fine-tuning.
5. **Spatial and object-level grounding.** Alert-CLIP aligns region features with boxes and captions, PromptTAD prompts with detected boxes, TAO tracks objects, and STPrompt selects image patches by motion. Two 2026 preprints ("Localizing to Debias", "Learning Where and When") bring patch-level localisation to weak supervision on surveillance data.
6. **Extra modalities.** π-VAD (CVPR 2025) induces pose, depth, flow and other modalities from RGB features at training time and found flow the most useful.
7. **Patch-level supervision as the source of localisation.** The two 2026 preprints train per-patch scoring heads with patch-level MIL and treat all patches of a normal video as negatives; LAVIDA gets its localisation from pixel masks. The common point is that a spatial module learns when it receives its own spatial signal, not only a video-level loss.

Five challenges remain open for traffic video:

1. **Motion blindness.** Frame-level CLIP features miss what makes a collision a collision. In LaGoVAD's table every surveillance-trained model is near chance on DoTA's within-clip protocol (VadCLIP, LaGoVAD: 49.00–62.60; PEL with I3D: 44.97–53.05).
2. **Spatial dilution.** Global vectors lose small and distant agents. Even frame-supervised Simple-TAD averages 81.6 / 84.1 / 85.2 AUC on ego-uninvolved DoTA clips (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips. GS-MoE reports a road-accident video scored near zero because the anomaly occupies a very small pixel region.
3. **No spatial evidence under weak supervision on traffic.** We found no weakly supervised traffic method that outputs or evaluates spatial maps, and no report of DoTA's STAUC under weak supervision (search to October 2026).
4. **Ego-motion and domain shift.** Camera motion masks agent motion: frame-prediction baselines on DoTA flag trees and background, and motion priors designed for static cameras, like STPrompt's, may select near-field background. Training and test data also come from different sources.
5. **Inconsistent evaluation.** TAD has two incompatible test lists in use (76 of LaGoVAD's 100 TAD test videos belong to UMIL's training list). DoTA numbers come from at least four protocols: pooled raw scores, per-video min-max, post-processed scores, and video-level anticipation AUC on a 367-clip subset.

### 2.3 Key works behind the proposed model

The proposed model is assembled from one baseline, one encoder, two source papers for the novel component and the works whose evidence justifies each choice. The table shows where each part of LaGoVAD-STP v2 comes from; the reviews follow.

| Part of LaGoVAD-STP v2 | Source work (venue) | What is taken | What the thesis changes |
| --- | --- | --- | --- |
| Host detector: CLIP stream, RoFormer, text co-attention, heads and losses | LaGoVAD (ICLR 2026), on VadCLIP's frozen-CLIP + MIL design (AAAI 2024) | Architecture and losses as released | Last-epoch checkpoints; Nexar setup (window bags, head kernel 3, top-k 5%); caption loss not used on Nexar |
| Motion stream (backbone change) | VideoMAE V2 (CVPR 2023), chosen on the evidence of Simple-TAD (ICCV 2025 Workshops) | The frozen distilled ViT-S and a 16-frame clip at about 10 fps | Frozen, causal 1.5 s clip, one pooled 384-d vector per step, standardised with fixed statistics, fused through zero-initialised W\_u |
| SA²-v2 in the CLIP branch (novel component, visual part) | STPrompt (ACM MM 2024) for window crops and the motion prior; LAVIDA (CVPR 2026) for the text-queried scoring, the outlier prior and the distractor idea | Window-crop CLIP patch features, second-order motion magnitude, element-wise sum with the frame feature; text-conditioned patch relevance; a within-frame outlier score; removing the genuine class from the definition | All 49 crops kept; prior-biased scorer with a learned residual; patch-level MIL with dense negatives; branch-only head; motion dropout; ego-compensated motion variant; maps scored with STAUC |
| Learnable prefix in the text branch (novel component, text part) | STPrompt and VadCLIP, after CoOp (IJCV 2022) | l learnable context tokens prepended to each class text before the frozen CLIP text encoder | Applied to LaGoVAD's open-world definitions (class names and descriptions) in place of its released soft prompts |
| Evidence-map evaluation | DoTA (IEEE TPAMI 2022) | STAUC code, anomalous-agent boxes and the ego-involved flag | Maps from SA²-v2's patch probabilities, scored under weak supervision |
| Ego-motion diagnosis | DoTA's baselines; PromptTAD (IEEE RA-L 2025) | Frame-level methods are distracted by dynamic backgrounds and miss off-centre agents | Motivates the compensated-motion variant and the ego / box-size analyses |

#### 2.3.1 LaGoVAD (ICLR 2026): the baseline

Liu et al. define open-world VAD as a mapping Φ: (V, Z) → frame scores, where the anomaly definition Z is given as text at test time. Frozen CLIP ViT-B/16 frame embeddings pass through a 2-layer RoFormer and are fused with the CLIP text encoding of Z by co-attention. The definition is a set of categories, each given by a class name or a description; training randomly uses one or the other. Training uses video-level labels on PreVAD (35,279 videos) with MIL and MIL-alignment losses, dynamic video synthesis and a caption contrastive loss with hard negatives.

- **Strengths.** It has the best zero-shot traffic results in its comparison: 89.56 AUC on TAD, 62.60 on DoTA and 90.41 on MSAD, ahead of VadCLIP and PEL. It stays robust when the definition changes (MSAD drift@5: 85.6 AUC / 40.1 AP), and better definitions help (UCF-Crime AUC 80.44 with class names, 81.12 with manual descriptions). The detector runs no LLM, trains on one RTX 4090, and its code, checkpoint and predictions are public.
- **Weaknesses.** Its only visual input is one global CLIP embedding per sampled frame, so it sees neither motion nor where in the frame something happens: DoTA AUC is 62.60 against 89.56 on TAD. It outputs no spatial evidence. Its released configuration selects checkpoints on test scores, and PreVAD's captions were written by Qwen models.
- **Taken into this thesis.** The whole detector. The thesis adds a motion input, SA²-v2 in the CLIP branch and a learnable prefix in the text branch, and replaces test-score checkpoint selection with the last epoch.

#### 2.3.2 VadCLIP (AAAI 2024): the CLIP-based weakly supervised template and its learnable prompt

Wu et al. adapt frozen CLIP to weakly supervised VAD with a local Transformer and graph convolution (the LGT-Adapter), a binary classification branch and an alignment branch that matches frames to class embeddings. Following CoOp, the class tokens are concatenated with l learnable context tokens before the frozen text encoder, because short labels are too succinct to describe abnormal events.

- **Strengths.** It established the CLIP-plus-learnable-prompt design that STPrompt, LaGoVAD, DSANet and Alert-CLIP build on: 88.02 AUC on UCF-Crime and 84.51 AP on XD-Violence on one RTX 3090, with public code.
- **Weaknesses.** It reads appearance only and one vector per frame, and sits near chance zero-shot on DoTA (49.00–50.93 AUC). In DSANet's analysis, its prototypes assign 84.0% of UCF-Crime road-accident videos to "Normal".
- **Taken into this thesis.** Its learnable-prompt form, as passed on by STPrompt, and a retrained same-feature competitor (Group C, Section 4.3).

#### 2.3.3 VideoMAE V2 (CVPR 2023): the motion encoder

Wang et al. scale masked video autoencoding with dual masking and progressive training (self-supervised on 1.35M clips, then supervised on Kinetics-710), and distil the 1B-parameter ViT-g into smaller ViT-S and ViT-B models.

- **Strengths.** The distilled ViT-S is small (about 22M parameters) and public, and in Simple-TAD's comparison it is the strongest small encoder for traffic anomalies after fine-tuning: AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000, against 52.1 / 48.1 for VideoMAE-S and 49.7 / 43.7 for InternVideo2-S. In this thesis's pilot, its pooled vector lifts zero-shot DoTA-CAP-dev to 0.81 macro / 0.73 micro (one seed).
- **Weaknesses.** Its pre-training data contain no driving video, and its features are not text-aligned. In this thesis, learned readouts of its tokens (MQR) and learned predictors over its vectors (LSP) added nothing beyond the mean vector.
- **Taken into this thesis.** The frozen distilled ViT-S, read as one mean-pooled 384-d vector per step.

#### 2.3.4 Simple-TAD (ICCV 2025 Workshops): evidence for the encoder choice

Orlova et al. show that a plain video ViT with one linear head, fine-tuned with frame-level labels, outperforms specialised traffic detectors on DoTA and DADA-2000, and add domain-adaptive pre-training (DAPT) on normal driving video.

- **Strengths.** Simple and strong (DoTA AUC up to 88.4; DoTA → DADA-2000 up to 86.6). It shows that masked video modelling transfers best to traffic anomalies and vision-language pre-training worst.
- **Weaknesses.** It needs frame-level labels, covers dashcam data only and takes no text. Ego-uninvolved clips lag: 81.6 / 84.1 / 85.2 AUC (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips, and the authors note that object-aware methods do better on rare categories.
- **Taken into this thesis.** The choice of encoder and clip input, the DAPT encoder as an ablation, and the ego-uninvolved gap, which motivates hypothesis H1.

#### 2.3.5 STPrompt (ACM MM 2024): the source of the novel component

Wu et al. learn spatio-temporal prompts on frozen CLIP for weakly supervised detection and localisation. The detection side has two parts that this thesis adopts and, for the visual part, redesigns.

- **Spatial Attention Aggregation (SA²).** Each frame, resized to 224×224, is cut into a 7×7 grid of 32×32 windows (stride 32); each window passes through CLIP and its CLS embedding becomes a patch feature. A patch's motion magnitude is the second-order difference of its feature across neighbouring frames, ‖2x\_i − x\_{i−1} − x\_{i+1}‖ (their Eq. 1). The K = 12 patches with the largest motion are aggregated into x\_AS, which is summed element-wise with the frame's CLIP feature before the temporal adapter. STPrompt argues that the two are complementary: x\_CLIP gives every pixel nearly equal influence, while x\_AS focuses on likely anomaly locations.
- **Learnable prompt prefix.** Following CoOp, l = 8 learnable context tokens are prepended to each category's tokens, {e\_1, …, e\_l, t\_c}, before the frozen text encoder; a contrastive loss keeps the label embeddings apart. Localisation at test time is training-free and uses LLM-written descriptions, which this thesis excludes.

| STPrompt ablation (AUC %) | UCF-Crime | UBnormal |
| --- | --- | --- |
| Without SA² | 86.13 | 63.00 |
| Average of all patches | 86.91 | 62.93 |
| Attention only | 86.67 | 62.64 |
| Motion selection + average | 87.79 | 63.69 |
| **Motion selection + attention (SA²)** | **88.08** | **63.98** |

- **Strengths.** It adds local evidence with no detector, boxes or region captions, at 31.5M parameters and 44.8 GFLOPs. The ablation shows that the fixed motion prior does most of the work: selection adds about 0.9 AUC over a plain average on UCF-Crime and is the only variant that helps on both datasets, while learned attention alone scores below averaging. The attention on top of selection adds only 0.29 AUC on both datasets.
- **Weaknesses.** Its motion cue assumes a still camera, and forward ego-motion was expected to put the largest second-order differences on near-field road and frame edges; on DoTA-CAP-dev the top-12 crops in fact avoided the border (Section 3.2.1). Patch features cost 49 extra CLIP passes per frame. The attention in its Eq. 2 is not legible in the project copy; the shapes it gives (motion scores of size T×K×1) suggest a parameter-free softmax over the motion magnitudes, not a learned query-key attention, and the published PDF decides this. It covers surveillance data only, and no public code was found.
- **Taken into this thesis.** The window-crop patch features, the second-order motion magnitude as a prior, the element-wise sum with the frame feature, and the learnable prefix. **Changed:** crops are not pre-selected but scored by a learned, prior-biased scorer; the scorer is trained with patch-level supervision; a variant subtracts the change shared by the whole frame; the weights become evidence maps scored with STAUC instead of LLM-based localisation; and both parts sit in LaGoVAD's open-world host next to a VideoMAE motion stream.

#### 2.3.6 PromptTAD (IEEE RA-L 2025): object-level evidence from a detector

Qiu et al. extend MOVAD with YOLOv9 box prompts, instance-wise and relation-wise aggregation (12 learnable query tokens) and an instance-level loss on Hungarian-matched ground-truth boxes, trained with frame labels.

- **Strengths.** It names the failure STP targets: the global view emphasises prominent regions and neglects distant or off-centre objects. It reaches 83.9 AUC on DoTA, reports STAUC, evaluates a small-object subset (334 test videos with anomalous boxes under 2,267 pixels), and gains most on ego-uninvolved anomalies (DADA-2000 70.4 vs 64.0 for TTHF).
- **Weaknesses.** It needs frame labels, anomalous-object boxes and a detector; when detection fails in dark scenes it degenerates into the baseline. No public code.
- **Taken into this thesis.** The diagnosis, the small-object subset and the ego-involved vs ego-uninvolved analysis. STP replaces detector boxes with prior-guided CLIP crops, and box supervision with bag-level MIL.

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

#### 2.3.8 LAVIDA (CVPR 2026): three ideas for the patch component

Dai et al. build an end-to-end zero-shot detector from a multimodal LLM (Qwen2-VL), a CLIP text encoder, a vision backbone and a SAM2 mask decoder. It is trained only on pseudo-anomalies made from segmentation datasets. Its parts that matter here are the Anomaly Exposure Sampler, a training-free token compression with local density and reverse attention, and a Multi-Scale Semantic Projector with learnable queries and text-conditioned cross-attention. A reading of its released code shows details the paper leaves out: the compression runs before the LLM layers, the density is a local-outlier style score, and the 48-query projector has about 55M parameters.

- **Strengths.** It reports 76.45 AUC on UBnormal, 85.28 on ShanghaiTech, 82.18 on UCF-Crime and 90.62 AP on XD-Violence zero-shot, and 87.68 pixel-level AUC on UCSD Ped2, all without any VAD training data. Its ablation shows that most background tokens can be dropped (up to about 80%) without losing accuracy.
- **Weaknesses.** Its localisation comes from dense pixel supervision and an LLM, neither of which exists under video-level labels. The reverse-attention step on unit-norm 512-d features has logits of at most 1/√512 ≈ 0.044, so its weights are almost uniform; some of its tables disagree (ShanghaiTech 85.28 against 80.64 for the same setting). It has no moving-camera evidence.
- **Taken into this thesis.** Three ideas, not weights: text-queried attention over patches (the text term of SA²-v2), a within-frame outlier score as a second prior (ρ), and the distractor-category sampler as a definition flip. The projector, the SAM2 decoder, the `<SEG>` path and the mask losses are not used.

### 2.4 Research gaps

Five gaps follow from the review. The thesis addresses each one, and its novelty claim is deliberately narrow.

1. **No motion-aware foundation features in weakly supervised traffic VAD.** LaGoVAD and VadCLIP read CLIP frames; the masked-video evidence of Simple-TAD comes from frame-level supervision. *Addressed by the motion stream (backbone change), which lifts Nexar-trained zero-shot DoTA-CAP-dev from 0.71 to 0.82 macro (5 seeds).*
2. **The appearance branch still reads one global vector per frame.** Small, distant and ego-uninvolved agents are diluted, and the detector-free fix that works under weak labels, STPrompt's SA², has been tested only on static surveillance cameras, never on traffic, under ego-motion, or next to a motion stream. *Addressed by SA²-v2, with an ego-compensated motion variant.*
3. **Learned patch attention under weak labels barely moves.** STPrompt's own ablation gives learned attention alone no gain over averaging, and in this thesis's first version the entropy fell only from 1.0 to about 0.91. Methods that localise well add patch-level supervision (the two 2026 preprints, LAVIDA's pixel masks). *Addressed by a prior-biased scorer, patch-level MIL with dense negatives from normal-labelled bags, a branch-only head and motion dropout; whether this changes the outcome is tested by hypotheses H2 and H6.*
4. **Weak supervision gives no spatial evidence on traffic data.** We found no weakly supervised traffic method that produces maps, and no report of STAUC under weak supervision. *Addressed by SA²-v2's patch probabilities, scored with STAUC and a pointing game, with no LLM.*
5. **Traffic results are hard to compare.** Most rows are AUC-only, two TAD test lists circulate, and DoTA numbers mix four protocols. *Addressed by four split-consistent, multi-metric protocols with each row's protocol labelled (Section 4.4).*

The learnable prefix fills a smaller gap: LaGoVAD's definitions are hand-written texts, and how a learned context shared by all definitions affects zero-shot traffic transfer and definition sensitivity has not been measured.

The closest prior work bounds the novelty claim:

| Closest work | What it does | How LaGoVAD-STP v2 differs |
| --- | --- | --- |
| STPrompt (ACM MM 2024) | SA² and learnable prefixes on a VadCLIP-style detector; LLM-assisted localisation; surveillance data | Traffic and dashcam data; open-world language-guided host with a motion stream; prior-biased learned scorer with patch-level supervision; ego-compensated motion; maps scored with STAUC; no LLM |
| VadCLIP (AAAI 2024) | Learnable prompt context on a CLIP MIL detector, one vector per frame | Adds prior-guided local evidence and a motion stream |
| PromptTAD (IEEE RA-L 2025) | Detector box prompts with frame labels and box supervision | Detector-free, bag-level labels |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with boxes and captions | Needs neither boxes nor captions |
| LAVIDA (CVPR 2026) | MLLM + SAM2 mask decoder trained on pseudo-anomalies with pixel masks | No LLM, no mask decoder, no pixel masks; keeps three ideas (text-queried scoring, outlier prior, distractor flip) in a CLIP detector |
| WS-DoTA (ECCV 2024 Workshop) | Global CLIP change features for weakly supervised DoTA | Change used per patch, as a prior and an input |
| "Localizing to Debias" (arXiv 2608.12045) | Sparse spatio-temporal weak localisation with motion-aware regularisation; UCF-Crime, XD-Violence, MSAD | Traffic focus, STAUC on DoTA, language-guided host; **novelty risk, to be read in full** |
| "Learning Where and When" (arXiv 2606.29498) | Patch-level MIL with proximity-aware top-k selection; surveillance | Traffic focus, motion and outlier priors, language-guided host; **novelty risk, to be read in full** |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that adds prior-guided, patch-supervised CLIP crop evidence (SA²-v2) and learned label context to a language-guided open-world detector with a motion-foundation stream, the first test of STPrompt's motion prior under ego-motion, and the first report of STAUC under weak supervision.** SA² and learnable prompts are not claimed as new, and patch-level MIL is not new in itself; the contribution is their combination in this host, their transfer to traffic, the ego-motion analysis and the spatial evaluation. The claim will be re-checked against the two 2026 preprints before submission, and the two preprints were read here only through summaries.

## 3. Proposed model

The proposed model, **LaGoVAD-STP v2**, is LaGoVAD with two changes: a frozen VideoMAE motion stream read as one vector per step (the backbone change) and traffic spatio-temporal prompting (the novel component), which adds SA²-v2 to the CLIP branch and a learnable prefix to the text branch. Every other part of LaGoVAD stays as released, so any gain can be traced to these changes. Values below are those of the Nexar setup that produced the preliminary numbers.

&#91;embedded content: Overview · released LaGoVAD against LaGoVAD-STP v2\]

The top panel is the released model: one CLIP vector per step and 32 soft prompts. In the bottom panel the host box is unchanged; SA²-v2 and the VideoMAE term change the visual input it receives, and the prefix replaces the soft prompts. Section 3.1 opens the host and Section 3.2.2 opens SA²-v2.

### 3.1 Base architecture

**LaGoVAD, unchanged parts.** The released code fixes each part:

- **Sampling.** Training uses one frame in eight (about 0.27 s at Nexar's 30 fps). Test sampling follows Section 3.4; DoTA-CAP is sampled every third frame (0.30 s at 10 fps), with scores interpolated to native frames.
- **CLIP stream.** A frozen CLIP ViT-B/16 image encoder gives one 512-d CLS embedding c\_t per sampled frame t, from a 224×224 resize without cropping.
- **Temporal encoder.** A residual RoFormer: 2 layers, 4 heads, hidden size 512, local window of 25 steps. Sequences are capped at 512 steps; a Nexar training item is a 20-step window spliced by dynamic video synthesis.
- **Text branch.** The frozen CLIP ViT-B/16 text encoder encodes each category of the anomaly definition Z (a class name or a description) into τ\_c ∈ R^512; the released model carries 32 learnable soft-prompt tokens, which STP's prefix replaces (Section 3.2.4).
- **Fusion.** A 2-layer co-attention Transformer lets video steps and text tokens attend to each other.
- **Heads.** A 1-D convolution head (kernel 3 on Nexar; LaGoVAD's released value is 9, which spans too much of a short window) scores the pre-fusion and post-fusion features; a learned weight w = σ(10·a), a initialised at 0.25, blends them, and the step score is the sigmoid of the blend.
- **Losses.** Top-k MIL binary cross-entropy (k = max(⌊L/5⌋, 1) for L steps on Nexar; LaGoVAD's released divisor is 16), a multi-class MIL alignment loss (divisor 16) and dynamic video synthesis (an abnormal clip inserted among 1–5 kNN-retrieved normal clips, giving frame pseudo-labels, with the whole abnormal clip marked positive). LaGoVAD's caption contrastive loss with hard negatives is not used on Nexar.

&#91;embedded content: LaGoVAD host · components and training losses\]

The figure shows the same parts in order of data flow. Blue marks the only two places STP v2 touches: the input x\_t and the text encoder's prompts.

**Backbone change: the motion stream.** For each sampled step t, a causal 16-frame clip that ends at t's frame (every third frame at 30 fps, so 1.5 s; every frame on 10 fps DoTA-CAP), padded by repeating frame 0, is squash-resized to 224×224 and passes through a frozen **VideoMAE V2-S** encoder: the ViT-S distilled from VideoMAE V2-g on Kinetics-710, D = 384. Its 8×14×14 last-layer tokens are averaged and passed through the encoder's frozen final normalisation into one vector, standardised with fixed statistics, and added through a projection that starts at zero:

```latex
v_t = \mathrm{Norm}\Big(\frac{1}{1568} \sum_{j=1}^{1568} X_{t,j}\Big) \in \mathbb{R}^{384}, \qquad \tilde v_t = \frac{v_t - m_u}{\sigma_u}, \qquad x_t = \tilde c_t + W_u\,(c\,\tilde v_t), \qquad W_u \in \mathbb{R}^{512 \times 384},\; W_{u,\text{init}} = 0
```

Here c̃\_t is the CLIP-branch feature: c\_t in the preliminary model, and c\_t plus the SA²-v2 feature in LaGoVAD-STP v2 (Section 3.2.2). m\_u, σ\_u and the scale c = mean‖c\_t‖/√512 are fitted once on Nexar's training split and reused on every test set. W\_u has a bias, also zero at the start, and there is deliberately no gate and no LayerNorm on the term. The fused x\_t replaces c\_t as the RoFormer input; the kNN retrieval of dynamic video synthesis still uses c\_t. Trained on Nexar, the preliminary model reaches **0.81 macro / 0.73 micro per-clip min-max AUC zero-shot on DoTA-CAP-dev** (one seed; the 5-seed mean of the same model is 0.82 macro; to be repeated with 3 seeds for every row).

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips and distilled on Kinetics-710; none of the eight traffic datasets used here is among them. Simple-TAD's driving-adapted DAPT-VideoMAE-S (BDD100K only) is kept as an encoder ablation.

### 3.2 Novel component: traffic spatio-temporal prompting (STP)

STP follows STPrompt's pattern on both branches. In the CLIP branch: crop CLIP features → SA²-v2 → element-wise sum with the CLS CLIP feature. In the text branch: learnable prefix + class text → frozen CLIP text encoder. Only the visual part differs from the first version of this proposal.

#### 3.2.1 Why the earlier components were dropped, and what the first SA² showed

Three components were built and tested before this one.

- **Motion Query Readout (MQR).** Eight zero-initialised, key-free queries (Efficient Probing, ICLR 2026) read the 7×7 VideoMAE token grid, with a spatial top-k MIL loss. Its attention entropy never left its uniform start, and the model without it reached the same DoTA result. It started at mean pooling, a solution the host already had; its queries received a gradient divided by roughly 49·√384 over near-identical masked-model tokens; and every gradient passed through the motion gate and the sparse top-k MIL loss.
- **Latent Surprise Predictor (LSP).** Following FLaRA, a small predictor trained on normal videos forecast the next VideoMAE vector, and the residual entered the fusion through a second gate. It did not improve on the motion-stream model in our tests.
- **First SA² (version 1 of this proposal).** Top-12 crops by motion, CLS-queried attention with W\_q zero-initialised, summed with c\_t. The zero-shot gain over CLIP-only LaGoVAD came almost entirely from the VideoMAE stream, and the attention entropy fell only from 1.0 to about 0.91 (0.90 with the prefix).

The literature check behind this redesign points to five causes for the first SA², ranked by how well the evidence supports them:

1. **No leverage.** An entropy of 0.91 for 12 crops equals a logit spread of about 0.6–0.7 (a simulation with Gaussian logits). The attention did move, but re-weighting 12 near-identical CLIP crop features changes the pooled vector very little, so the loss barely reacts.
2. **A saddle at W\_q = 0.** In a bilinear query-key form, W\_k gets exactly zero gradient until W\_q moves.
3. **A weak path for the gradient.** The module is reached only through the VideoMAE-fused features and a top-k MIL loss that touches about 6% of a video's steps.
4. **A faster shortcut.** The VideoMAE stream is strong and learns quickly; evidence from other multimodal settings says the slower branch then stays under-trained.
5. **The fixed prior was already doing the work.** In STPrompt's own ablation, learned attention alone scores below plain averaging (86.67 vs 86.91 on UCF-Crime) and adds only 0.29 AUC on top of motion selection; its attention may even be parameter-free (Section 2.3.5).

**What the first SA² measured (one seed, Nexar training, DoTA-CAP-dev, 569 clips; preliminary).**

| Read | Result | Reading |
| --- | --- | --- |
| Macro: motion-stream model / first SA² / first SA² + prefix | 0.8110 / 0.8111 / 0.8124 | No difference shown |
| First SA² − motion-stream model | +0.0001 \[−0.0100, +0.0098\] | No difference shown |
| The same difference, by ego flag | ego-involved +0.012 [+0.001, +0.020]; ego-uninvolved −0.013 [−0.031, +0.004] | Against the direction of H1 |
| Top-12 crops on the agent: lift minus pre-anomaly lift | +0.132 [+0.122, +0.142] | The selection points at the agent |
| Top-12 crops against the grid: border / bottom row / centre 3×3 | −0.080 / −0.082 / +0.052 | It avoids the border and the near-field road |
| Same lift on ego-uninvolved clips, raw / ego-compensated Mo | +0.1007 / +0.1004 | Compensation changes nothing here |
| Prefix: gradient signal-to-noise; movement | 0.136 (zero-mean noise gives 0.184); 3.6% of its norm | It did not demonstrably train, so first SA² + prefix does not measure the prefix |
| W\_u trained with learning rate ×10 | −0.042 [−0.054, −0.030] | A larger learning rate is costly on the motion term |
| Patch-only branch, no text, no VideoMAE: content / with motion magnitudes | 0.612 / 0.704 (Mo alone, untrained: 0.713) | It underfit Nexar and did not beat the untrained prior |

The selection is right and yet the pooled vector adds nothing, so a weak attention cannot be the whole explanation: the content of the crop features is a competing cause, and SA²-v2 must be read against it. For scale, VideoMAE token maps from the earlier MQR work scored STAUC 0.32–0.33 on 528 boxed clips.

The lessons that shape STP are:

#### 3.2.2 Visual part: SA²-v2 in the CLIP branch

SA²-v2 scores each crop of a frame, turns the scores into weights, and pools the crops into one 512-d vector that is added to the frame feature. The figure below shows one step and the sizes of every tensor.

&#91;embedded content: SA²-v2, one sampled step · tensor sizes\]

Cached crop features, priors and the definition vector feed a learned encoder and scorer; the logits give a 7×7 evidence map and a 512-d pooled vector that replaces the old SA² output, and the three dashed boxes act only in training.

**Cached inputs (per sampled step t).**

- Crop features P\_{t,j} ∈ R^512, j = 1, …, 49: as in STPrompt, each frame resized to 224×224 is cut into a 7×7 grid of 32×32 windows (stride 32), each window is resized to CLIP's input size and encoded by the frozen CLIP image encoder, and its CLS embedding is the crop feature. Crops are used instead of CLIP's dense tokens because the dense tokens are not supervised with language and already mix global context. All 49 are kept (the first version kept 12, and its Nexar cache holds only those).
- Frame feature c\_t ∈ R^512 (LaGoVAD's CLS feature).
- Two fixed priors per crop, computed from L2-normalised crop features P̂:

```latex
D_{t,j} = 2\hat P_{t,j} - \hat P_{t-1,j} - \hat P_{t+1,j}, \qquad Mo_{t,j} = \lVert D_{t,j} \rVert_2, \qquad \rho_{t,j} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(j)} \lVert \hat P_{t,j} - \hat P_{t,i} \rVert_2, \quad k = 5
```

Mo is STPrompt's second-order motion magnitude (their Eq. 1); D\_{t,j} ∈ R^512 is kept as a learned input. ρ is a within-frame outlier score: the mean distance of a crop to its k nearest other crops in the same frame, high for crops unlike the rest. It is a simplification of the density idea in LAVIDA (Section 2.3.8). Both are z-scored per frame across the 49 crops: z(x)\_j = (x\_j − mean\_i x\_i) / (std\_i x\_i + ε).

- Text vector τ\_Δ ∈ R^512: the frozen CLIP text embedding of the anomaly definition (mean over its categories, with the learnable prefix) minus the embedding of "normal", both L2-normalised.

**Block A: patch encoder (learned).** Each crop becomes a 256-d token and one Transformer layer lets the 49 tokens see each other:

```latex
u_{t,j} = W_u\,[\,P_{t,j};\; P_{t,j} - c_t;\; D_{t,j}\,] + e^{pos}_j + W_\pi\,[\,z(Mo)_{t,j};\; z(\rho)_{t,j}\,], \qquad H_t = \mathrm{TransformerLayer}(u_t) \in \mathbb{R}^{49 \times 256}
```

with W\_u ∈ R^{256×1536}, a learned position table e^{pos} ∈ R^{49×256}, W\_π ∈ R^{256×2}, and a pre-LN layer with 2 heads and FFN width 512. The term P − c\_t says what this crop has that the whole frame does not; ΔP lets the module learn its own use of motion instead of relying on a hard selection.

**Block B: patch scorer (learned).** One logit per crop, the sum of a prior term, a text term and a learned residual:

```latex
\ell_{t,j} = \beta_1\, z(Mo)_{t,j} + \beta_2\, z(\rho)_{t,j} + \gamma\, z\big(\cos(P_{t,j}, \tau_\Delta)\big)_j + w^{\top} H_{t,j}
```

The prior term starts as STPrompt's motion prior (β\_1 = 1, β\_2 = 0); the text term asks whether a crop looks more like the anomaly definition than like "normal" (γ starts at 0 and is learned); the residual is a linear read-out of H\_{t,j} (w starts at 0). Because every term is z-scored per frame, the logits have a usable spread from the first step, unlike the near-uniform weights that unit-norm cosine logits give (at most 1/√512 ≈ 0.044 apart).

**Block C: two outputs from the logits (no extra parameters).**

```latex
q_{t,j} = \sigma(\ell_{t,j} + b), \qquad \alpha_{t,j} = \operatorname{softmax}_j\!\big(\ell_{t,j}/T\big), \qquad x^{AS}_t = \sum_{j=1}^{49} \alpha_{t,j}\, P_{t,j}, \qquad \tilde c_t = c_t + x^{AS}_t
```

q is the patch anomaly probability; reshaped to 7×7 it is the evidence map E\_t. α pools the raw CLIP crop features, so x^AS stays in CLIP's embedding space and the sum c̃\_t stays text-aligned. T = exp(s) is a learnable temperature (s starts at 0) and b a learnable bias (starts at −2). c̃\_t (512) goes to the motion fusion of Section 3.1 exactly as c\_t did before.

**Ego-compensated variant (ablation, not the default).** Forward ego-motion changes every crop, mostly on near-field road and frame edges. Subtracting the change shared by the whole frame before the motion prior keeps crops that move differently from the scene:

```latex
Mo^{\text{ego}}_{t,j} = \Big\lVert D_{t,j} - \tfrac{1}{49} \textstyle\sum_{i=1}^{49} D_{t,i} \Big\rVert_2
```

**Boundaries and synthesis.** The first and last steps of a video repeat their neighbour. Because Mo and D use neighbouring steps, they are computed per source video before windows are cut and before dynamic video synthesis splices clips, so seams create no false motion.

#### 3.2.3 Training signals for SA²-v2

SA²-v2 adds no loss at test time. During training it adds three signals and one data change, all from bag-level labels:

```latex
\mathcal{L} = \mathcal{L}_{host} + \lambda_p\, \mathcal{L}_{patch} + \lambda_b\, \mathcal{L}_{branch}
```

- **Patch-level MIL (L\_patch).** Input: q for all crops of a bag, the bag label, and the host's frame scores. In a normal-labelled bag every crop of every step is pushed to 0, binary cross-entropy averaged over the T·49 crops: these are dense negatives, which SA²'s softmax over 12 never had. On Nexar such a bag is a pre- or post-event window of a positive video, because training uses positive videos only. In an abnormal bag, take the top-k steps by host frame score (the same k as the host's MIL, max(⌊L/5⌋, 1) on Nexar); in each, take the top-m = 3 crops by q and push them to 1. Default λ\_p = 0.2.
- **Branch-only head (L\_branch).** Input: c̃\_t alone, without the VideoMAE term. A linear layer (513 parameters) gives one frame score per step, trained with the same top-k MIL loss. It is meant to force x^AS to carry signal without help from VideoMAE. Default λ\_b = 0.3.
- **Motion dropout.** In 30% of training bags the VideoMAE term W\_u(c·ṽ\_t) is set to zero, so the host cannot rely on it alone. Not used at test time.
- **Definition flip (data change).** With probability 0.2, for an abnormal bag whose definition set Z holds other categories, remove the bag's own category from Z and set the label to normal. Input: Z and the label; output: the changed pair. It makes the text term matter, in the spirit of LAVIDA's distractor categories. For P3, where Z is {Normal, CarAccident}, distractor texts are borrowed from TAD's and MSAD's category lists. This must be checked against LaGoVAD's code first (it may already do this) and is its own ablation row.

#### 3.2.4 Text part: learnable prefix

For each category c of the definition Z, with token embeddings t\_c of its class name or description, l learnable context vectors shared by all categories are placed in front, as in STPrompt and CoOp:

```latex
\tau_c = \mathrm{TextEnc}\big( [\, e_1, e_2, \ldots, e_l,\; t_c \,] \big), \qquad e_i \in \mathbb{R}^{512}, \quad l = 8
```

The prefix replaces LaGoVAD's released soft prompts, which sit half before and half after the text (32 tokens); both versions are compared in the ablation. The prefix is applied to definition texts only; captions are not used on Nexar. Long descriptions are truncated to the 77-token context minus l and the start and end tokens. The same prefixed embeddings supply τ\_Δ to SA²-v2.

#### 3.2.5 Initialisation

- **Scorer.** β = (1, 0), γ = 0, w = 0, b = −2, s = 0. At the start the weights are a soft version of STPrompt's motion prior over all 49 crops. The zero start of w and γ is safe because their gradients depend on H and on z(cos), which are not zero; this differs from the zero W\_q of the first version, where the gradient to W\_k vanished.
- **Encoder.** Xavier for W\_u, W\_π and the Transformer layer; e^pos \~ N(0, 0.02²).
- **Sum.** x^AS is added at full weight, as in STPrompt; every model is trained from scratch, so there is no pretrained starting point to protect. A zero-initialised gate tanh(γ\_s)·x^AS is a side ablation.
- **Prefix.** e\_i \~ N(0, 0.02²), as in CoOp's generic initialisation; initialising from the word embeddings of "a dashcam video of" is a side ablation.

#### 3.2.6 Why STP should help on traffic video, and what to expect

Six pieces of evidence point the same way:

1. **A fixed prior does work under weak labels.** In STPrompt, motion selection adds 0.88 AUC on UCF-Crime and 0.76 on UBnormal over averaging, and is the only variant that helps on both datasets.
2. **Traffic anomalies are small and local.** The ego-uninvolved gap of Simple-TAD (81.6–85.2 vs 89.9–90.7 AUC), PromptTAD's neglect of distant or off-centre objects and GS-MoE's small-region failure all show that global vectors miss them.
3. **Region-level CLIP evidence helps.** Alert-CLIP moves CLIP's attention from context to the collided vehicles and lifts the VadCLIP framework on MSAD from 87.45 / 71.09 to 89.63 / 78.24, though with boxes (its box-supervised region loss alone adds only 0.32 AUC).
4. **Patch-level supervision is where localisation gains come from.** The two 2026 preprints and LAVIDA all train a spatial module with its own spatial signal; SA²-v2 gets the same kind of signal from normal-labelled bags, where every crop is treated as a negative.
5. **It adds what the host lacks.** The VideoMAE vector carries global motion and c\_t global appearance; SA²-v2 adds local appearance from separate CLIP passes, and a branch-only loss keeps it from being bypassed. MQR and LSP instead re-processed the same motion vector.
6. **Text quality matters in LaGoVAD.** Better definitions raise its UCF-Crime AUC from 80.44 (class names) to 81.12 (manual descriptions) and 83.03 (video-specific descriptions). A learned prefix adapts hand-written traffic definitions without an LLM.

**Expectation.** STPrompt's learned attention was worth only about 0.3 AUC, so the overall AUC change may be small. The planned tests are STAUC, the pointing game, the small-box and ego-uninvolved subsets of DoTA, and the branch-only run (H4, H6).

**Risks.**

- Ego-motion: on dashcam video ρ may select road surface, hood or frame edges. For the motion prior this was measured and did not appear on DoTA-CAP-dev, and the ego-compensated variant did not change the pointing (Section 3.2.1).
- ρ ranks objectness, not anomaly: near-field vehicles can look like dense background and the hood is always an outlier.
- The patch branch may learn a source shortcut if CCD is added, since its crash clips and BDD100K normals come from different sources. Imported normal pools have already failed in this project (D2City and BDD-A failed the transfer probe with shortcut AUC 1.000), so Nexar stays the only training set until CCD passes the same probe.
- Memory: 64 items × about 120 steps (six 20-step windows after synthesis, estimate; the limit is 512) × 49 crops × 512 features is about 0.4 GB in fp16 for the inputs alone; activations are not yet profiled, and chunking over steps or gradient checkpointing is the fallback.

| Property | Motion-stream model (preliminary) | + STP v2 |
| --- | --- | --- |
| Extra parameters | — | about 0.93M (SA²-v2) + 4K (prefix, l = 8) + 513 (branch-only head, training only) |
| Extra extraction compute | — | 49 CLIP ViT-B/16 passes per sampled frame (window crops), as in version 1 |
| Storage per sampled frame (fp16) | 1 KB (c\_t) + 0.75 KB (v\_t) | + 49 KB (all 49 crop features); priors and ΔP are computed on the fly |
| Spatial map | none | 7×7 per step |
| Extra loss | — | patch-level MIL and branch-only loss (training only) |
| VideoMAE input | one 384-d vector per step | unchanged |
| Extra cost at test time | — | the encoder layer on 49 tokens per step; no extra backbone passes beyond the crops |

### 3.3 Training strategy

Training uses bag-level labels only and one fixed schedule per protocol, so the baseline, the motion stream and STP differ only in architecture and in the training signals of Section 3.2.3.

**Training data.** Only TAD, MSAD, CCD and Nexar contain both normal and abnormal videos, so only they can train; DoTA, DADA-2000 and MM-AU contain accident videos only and serve as test sets. Only Nexar has been trained so far, and P1 and P2 are planned.

| Protocol | Training set | Normal / abnormal videos | Anomaly text (hand-written) |
| --- | --- | --- | --- |
| P1, TAD | TAD train, UMIL split | 210 / 190 | TAD's 7 category names and definitions |
| P2, MSAD | MSAD weak-protocol train | 360 / 120 | MSAD's 11 anomaly types |
| P3, dashcam | Nexar windows: 20-step windows, hop 8, from the 451 positive videos of the training split | Windows labelled by their own frames; the 449 negative videos are not used | {Normal, CarAccident} definitions |
| P4, cross-setting | None: reuses the P2 and P3 models | — | Each test set's own definitions |

**Captions.** The caption contrastive loss is not used on Nexar, so no caption enters any preliminary number and no LLM-written text enters training. Hand-written templates are an ablation (removing this loss costs LaGoVAD 6.3 AUC on TAD zero-shot in its paper).

**Why windows.** A frame of a positive Nexar video is abnormal from the alert time to 1.4 s after the event, and a window is labelled by its own frames. In 730 of the 750 positive videos the event sits at 0.4–0.6 of the video, so labelling whole videos invites a position shortcut; windows close it.

**Shortcut control.** CCD is not in the default setup: its crash clips and normal clips (from BDD100K) come from different sources, so a model could learn the source instead of the anomaly, through the host or through the patch branch. Nexar + CCD is an optional row that first needs a transfer probe (CCD as the only negatives, within 0.03 of the in-video reference).

**Stages.**

1. **Finish the diagnosis of the first SA² (cached features).** The signal, pointing and branch-only checks are done (Section 3.2.1). What remains from Section 4.3 (Step D) is the uniform-replacement ΔAUC, the leverage of the pooled feature and the candidate similarity.
2. **Cache features.** The CLIP CLS features and the VideoMAE V2-S vector exist for the read sets. New: all 49 window-crop CLIP features per step for the 451 Nexar positive videos (the first SA²'s cache keeps 12). The motion difference D, the priors Mo and ρ and their z-scores are computed from the cached crops at load time, per source video.
3. **Train all models with one schedule.** Every row (LaGoVAD, motion stream, STP, ablations) uses the same seeds and schedule; the first run is one seed and its result is preliminary, and conclusions need 3 seeds. The scorer's zero-initialised terms mean SA²-v2 starts as a soft motion-prior pooling.
4. **Monitor the learned parts.** Log, from the first step: the attention entropy of α; the gradient norms of w, γ and the encoder; the spread of the logits ℓ; the mean of q on normal and abnormal bags; the norm of the prefix vectors; and the uniform-replacement ΔAUC on the validation split every few epochs, so a learned part that does not move, as MQR's did not, is visible at once.

**Settings.** AdamW, batch 64, learning rate 5e-5 with 20 warm-up steps and a cosine schedule: 33 epochs of 53 steps (1,749 steps) on Nexar, weight decay 0.01. The first run fixes K = 12, l = 8 and a learning-rate multiplier of 1. The values λ\_p ∈ {0.05, 0.2, 0.5}, λ\_b ∈ {0.1, 0.3}, m ∈ {1, 3, 5}, k in ρ ∈ {3, 5, 8}, dropout p ∈ {0, 0.3} and l ∈ {4, 8, 16} can be chosen once on Nexar's validation split (225 videos) scored by macro AUC, then frozen for every protocol. No hyperparameter is chosen on a test list.

**Checkpoint selection.** Every model reports the last epoch of the fixed schedule, and each checkpoint's step is checked against its training log. The released LaGoVAD checkpoint is scored as published.

**Compute.** The new parameters are few (about 0.93M), but the encoder layer runs on 49 tokens per step, so one batch of 64 items × about 120 steps holds about 0.4M tokens (estimate; the limit is 512 steps); activation memory is not yet measured, and chunking over steps or gradient checkpointing is the fallback. The extraction cost is the same as in version 1, 49 CLIP ViT-B/16 passes per step, but the Nexar pass must be repeated because only the top-12 crops were kept. By our estimate (not yet measured) that is roughly 3.3M crops for the Nexar positives (451 videos × about 150 steps) and 1.9M for the 1,129 DoTA-CAP clips. A one-day profiling run in the first week fixes the real numbers.

### 3.4 Inference

At test time the model takes a video and an anomaly definition and returns a score and a 7×7 evidence map per sampled step. No LLM runs at any point, and none of the training-only parts (patch-level MIL, branch-only head, motion dropout, definition flip) is active.

1. **Sample** frames per set: DoTA-CAP every third frame over the whole clip; DADA-2000 and TAD one frame in eight; DAD one in seven (0.28 s at 25 fps).
2. **Encode** each sampled frame with frozen CLIP (CLS feature c\_t, no centre crop) and its 49 window crops with the same CLIP, and the causal 16-frame clip that ends on it with the frozen VideoMAE V2-S, pooled to one 384-d vector and standardised with the statistics of Nexar's training split.
3. **Compute priors** from neighbouring steps: ΔP, Mo and ρ, z-scored per frame.
4. **Encode the definition** with the learned prefix and the frozen CLIP text encoder, giving τ\_Δ.
5. **Score the crops** with SA²-v2: patch encoder (49×256), logits ℓ (49), evidence map q (7×7), weights α, pooled feature x^AS (512), and c̃\_t = c\_t + x^AS\_t.
6. **Fuse** c̃\_t with the VideoMAE term through W\_u.
7. **Score the steps** with LaGoVAD's RoFormer, text co-attention and heads. On DoTA-CAP step scores are linearly interpolated to native frames; elsewhere each step's score is assigned to the frames it stands for.
8. **Map** each step's evidence from q, upsample it to frame size and normalise it per video.

**Definitions.** Every main result uses LaGoVAD's released definitions (TAD as {"Normal", "Car Accident"}; Nexar, DoTA-CAP, DADA-2000 and DAD as {"Normal", "CarAccident"} with the same car-accident texts). A separate analysis prompts each of TAD's 7 categories with its own hand-written definition.

**Score post-processing.** DoTA-CAP clips are almost all abnormal, so their scores are min-max normalised per clip before pooling, following LaGoVAD's `offline_dota_eval.py`. TAD, DADA-2000 (whole source videos), DAD and MSAD scores are used raw, and macro AUC is averaged over two-class videos.

**Cost.** The window crops multiply CLIP's cost per frame by about 50, as in STPrompt; one VideoMAE ViT-S clip (about 57 GFLOPs) still runs per step. A cheaper variant that reads ClearCLIP's dense patch tokens from the single full-frame pass is an ablation (R14). The model is offline: the motion magnitude uses the next step and the RoFormer looks both ways, although the VideoMAE clip itself is causal.

## 4. Experimental plan

The plan trains with bag-level labels only, finishes the diagnosis of the first SA² and checks the premise of STP with inference-only pilots before full training, compares against published rows on identical test lists, and adds every second metric by re-scoring released checkpoints.

### 4.1 Dataset preparation

The thesis uses eight traffic datasets. Nexar trains the dashcam model (CCD only if it passes the probe of Section 3.3). DoTA-CAP, DADA-2000, DAD and TAD test it zero-shot, and DoTA's boxes test the evidence maps. TAD (fixed camera) and MSAD (multi-scene surveillance with a traffic-accident slice) carry the planned in-domain comparisons.

| Dataset | View | Size and split | Labels | Role | Source |
| --- | --- | --- | --- | --- | --- |
| TAD | Fixed CCTV, some first-person | 500 videos; UMIL split 400 train (210 normal / 190 abnormal), 100 test (40 / 60); 7 categories | Video labels (train), anomaly interval (test) | P1 train and test; P4 zero-shot test | Frames and split files via the UMIL repository; native fps unknown |
| MSAD | Fixed, 14 scenes | 720 videos at 30 fps; weak protocol 360 normal + 120 abnormal train, 120 + 120 test (20 traffic accidents) | Video labels (train), frame labels (test) | P2 train and test; traffic slice; P4 zero-shot test | MSAD project page (videos gated) |
| CCD | Dashcam | 1,500 crash + 3,000 normal clips (normal from BDD100K), 50 frames at 10 fps, official split | Video labels; accident onset | Optional P3 training, after the transfer probe | GitHub (Cogito2012/CarCrashDataset) |
| Nexar | Dashcam | 1,500 labelled clips, about 40 s each; fixed split 900 train (451 positive) / 225 validation / 375 test | Video labels; alert and event times, which build the window labels | P3 training | Hugging Face |
| DoTA | Dashcam | 4,677 clips, 10 fps, abnormal only. DoTA's own pixels are gone: 1,129 of the 1,397 validation clips survive as DoTA-CAP (CAP-DATA copies), split into dev 569 / eval 560 | Anomaly window; 9 categories × ego-involved / ego-uninvolved; boxes with track IDs | P3 zero-shot test (dev read repeatedly, descriptive; eval kept for SA²'s final read); STAUC and pointing game | GitHub (MoonBlvd/Detection-of-Traffic-Anomaly) + CAP-DATA frames |
| DADA-2000 | Dashcam | Original release; 370 test videos, whole video, one frame in eight | Accident intervals; ego / non-ego | P3 zero-shot test | Project split files |
| DAD | Dashcam | 466 test videos (165 positive), 100 frames at 25 fps | Accident in the last 10 frames (toa = 90) | P3 zero-shot test; its macro AUC is not localisation (Section 4.4) | DAD release |
| MM-AU | Dashcam | 11,727 accident videos pooled from CCD, A3D, DoTA and DADA-2000 | Accident start, collision, end | Optional zero-shot test, after removing videos shared with CCD, DoTA and DADA-2000; needs the CAP-DATA archives, the source of DoTA-CAP's frames | MM-AU release (CVPR 2024) |

**Weights** (all public). In use and pinned: CLIP ViT-B/16, VideoMAE V2-S (`vit_s_k710_dl_from_giant`) and LaGoVAD's `best.ckpt`, used only as a reference row. Still to download: `pred_results.pkl`; VadCLIP, PEL4VAD and DSANet checkpoints for zero-shot re-scoring; RTFM and MGFN MSAD checkpoints for the traffic slice; MOVAD and Simple-TAD DoTA and DADA-2000 checkpoints as frame-supervised context; and Simple-TAD's DAPT-VideoMAE-S (BDD100K only, CC BY-NC 4.0) for the encoder ablation. ClearCLIP is a training-free change to the CLIP code, needed only for ablation R14. None of these remaining links has been test-downloaded yet; that check is the first task.

### 4.2 Dataset processing

Every dataset goes through the same pipeline, so one feature cache serves the baseline, the motion stream, STP and all ablations.

1. **Decode and sample.** CLIP features are extracted for every frame, and each set takes its own sampling: Nexar training one frame in eight, DoTA-CAP every third frame, TAD and DADA-2000 one in eight, DAD one in seven. Frame rates are not resampled.
2. **Cut clips.** For each step take the causal 16 frames that end on it, every third frame at about 30 fps (Nexar, DADA-2000), every frame on 10 fps DoTA-CAP, every third frame on DAD (1.8 s at 25 fps) and on TAD (native fps unknown, so its duration is reported as unknown). Repeat frame 0 before the start. Squash-resize to 224×224 without cropping, with ImageNet normalisation.
3. **Extract CLIP CLS features** without a centre crop, the transform of LaGoVAD's `extract_feat_clip.py`, so the released checkpoint is scored on matching features. A feature cache stays bound to its transform and stride.
4. **Extract window-crop patch features.** Cut each 224×224 frame into the 7×7 grid of 32×32 windows, encode each window with the same CLIP, check the full-frame window against the frame's cached feature, and store all 49 crop features per step in fp16. The motion difference, the priors Mo and ρ (raw and ego-compensated) and their z-scores are computed from these at load time, per source video, so the cache holds no derived tensors.
5. **Extract the VideoMAE V2-S vector.** Average the 8×14×14 last-layer tokens and apply the encoder's final normalisation, one 384-d vector per step (0.75 KB fp16). Fit c, m\_u and σ\_u on Nexar's training split and apply them to every set.
6. **Build labels.** Nexar: frames from the alert time to 1.4 s after the event, cut into window bags. TAD from UMIL's `TAD_train.txt` (P1), MSAD from its weak split (P2). Test labels are frame-level: TAD intervals, MSAD frame labels, DoTA-CAP anomaly windows, DADA-2000 intervals, DAD's toa = 90.
7. **Prepare DoTA's spatial labels.** For each DoTA-CAP clip keep `ego_involve`, the category and the per-frame boxes of anomalous agents. Bin clips by the median, over the anomaly window, of the largest box's share of the frame: under 0.5%, 0.5–2%, 2–8%, over 8%. Also mark PromptTAD's small-object subset (anomalous boxes under 2,267 pixels). The obstacle (VO) and out-of-control (OO) categories have no participant boxes and are left out of STAUC.
8. **Fix test lists.** DoTA-CAP dev (569 clips, macro over two-class clips) for development and eval (560) for the final result, both reported as DoTA-CAP (1,129 of 1,397 clips have pixels); the boxed DoTA-CAP clips for STAUC (528 on dev; DoTA's official 1,305-video list cannot be scored in full); TAD's 100-video list (60 two-class) for zero-shot; UMIL's TAD test list for P1; MSAD's weak-protocol test set; the 370 DADA-2000 videos; DAD's 466 test videos.
9. **Hold out validation data.** Nexar's validation split (225 videos), for sanity checks and any one-time hyperparameter choice; its test split (375) is opened once.

**Storage** (GB, fp16, estimates; the full 7×7 grid of 49 crop features is the default for training data, 49 KB per step):

| Dataset | VideoMAE vector | All 49 crop features |
| --- | --- | --- |
| TAD (all 500 videos) | 0.05 | 3.3 |
| CCD, if added (one step per frame) | 0.17 | 11.3 |
| Nexar, positive videos (about 68K steps) | 0.05 | 3.3 |
| DoTA-CAP, 1,129 clips, every third frame (about 38K steps) | 0.03 | 1.9 |
| DADA-2000 | 0.06 | 4.1 |
| DAD, 466 videos (about 7K steps) | 0.01 | 0.3 |

TAD's quoted duration conflicts with its frame count, so its figures may be off by up to about 5×. MSAD and MM-AU are measured at extraction. The window caches behind the first SA²'s reads on DoTA-CAP, TAD, DADA-2000 and DAD should be reused if they are still on disk.

**Leakage checks before any training.**

- Seventy-six of LaGoVAD's 100 TAD test videos are in UMIL's TAD training list, so a model trained on UMIL's split is never scored on LaGoVAD's TAD list.
- MM-AU pools CCD, A3D, DoTA and DADA-2000 videos. It is never used for training, and as a test set only after removing shared videos.
- DoTA-CAP-dev has been used repeatedly during development, so results on it are descriptive. DoTA-CAP-eval was opened once as a descriptive check of the motion-stream model (0.81 macro, 5 seeds); no SA² model has been scored on it, and it is kept for the final result.
- A near-duplicate check with CLIP-feature similarity runs between every training set (TAD, MSAD, CCD, Nexar) and every test set. Until then, and because PreVAD was not checked either, zero-shot status on these sets is unverified.
- The released checkpoint is re-scored before any comparison, and the re-scored number is the reference, not the printed one. Done: DoTA-CAP-dev, TAD (0.7912 micro, not 89.56), DADA-2000, DAD. `pred_results.pkl` is still to be checked against the printed numbers.

### 4.3 Pilot, diagnostics and baseline running

Baselines run in three groups, preceded by two inference-only studies that check the premise of STP before any full training of SA²-v2. Both are partly done.

**Step D: diagnose the first SA² (cached features, one GPU-day).** It asks whether the first SA² contributed at all and which of the causes in Section 3.2.1 apply. All checks use the trained version-1 model. The state column is as of this revision.

| Check | What is measured | Reading | State |
| --- | --- | --- | --- |
| D1: uniform replacement | AUC change when SA²'s weights are replaced by uniform weights over the same 12 crops | About zero: the attention does nothing and entropy 0.91 is irrelevant | To run |
| D2: leverage | Norm of x\_AS − mean of the 12 selected crops, against the norm of c\_t | Tiny relative to c\_t: the pooled vector cannot differ by reweighting | To run |
| D3: candidate similarity | Mean cosine similarity among the 12 selected crops (and among all 49) | High: near-identical candidates, no room to select | To run |
| D4: gradient signal | Gradient norm and signal-to-noise ratio of W\_q and W\_k over the training | Near zero for W\_k while W\_q is small: the saddle | Partly done: SNR at the end of training 0.20–0.22, about W\_u's 0.22, and ‖W\_q‖ reached 1.4 |
| D5: branch-only run | Train without the VideoMAE term and compare with CLIP-only LaGoVAD (B1) | If the branch gains nothing alone, the shortcut explanation is supported | Closest read: patch-only local branch 0.612 / 0.704 (Section 3.2.1); SA²'s own branch-only run is still to do |

If D1 shows that uniform replacement costs nothing, the diagnosis is confirmed and SA²-v2 proceeds; if it costs something, the redesign is still run, but its target changes from "make it learn" to "make it learn more".

**Step 0: pilot diagnostic of the priors (inference only, cached crop features).** It asks whether the motion prior and the outlier prior still select the agents that matter when the camera moves. DoTA's training pixels are gone, so the check runs on DoTA-CAP-dev, which is a development read and never decides a test result. Raw and ego-compensated Mo are done; ρ is not.

| Check | Question | Pass criterion (proposed) | If it fails |
| --- | --- | --- | --- |
| S1: hit rate | In what share of anomalous frames does at least one of the 12 highest-prior crops overlap an anomalous-agent box, against the share expected from box area alone? By ego flag and box-size bin, for ρ (Mo: lift +0.196, top-1 hit − chance +0.314) | Clearly above chance, at least on ego-uninvolved clips | Drop ρ (β\_2 stays 0) |
| S2: in-box ratio | Lift of ρ's top crops on anomaly rows minus the same read before the anomaly (Mo: +0.132) | Above zero | As above; if all priors fail, drop the prior term (β = 0) and rely on the text term and patch-level MIL |
| S3: where selection lands | Top-12 share in the border ring, bottom row and centre 3×3 against each region's share of the grid, for ρ (Mo: −0.080 / −0.082 / +0.052) | No excess on the border or bottom row | Exclude the bottom row (side ablation) |

The remaining pilot needs no GPU if the DoTA-CAP-dev window cache is still on disk. A failure does not stop the plan: it decides which prior becomes the default before any training.

**Group A: reproduce LaGoVAD and the motion stream.**

| Row | Model | Training | Purpose |
| --- | --- | --- | --- |
| B0 | LaGoVAD, released checkpoint | PreVAD (by its authors) | Re-score with AUC, AP and AUC\_A. Done on DoTA-CAP-dev, TAD, DADA-2000, DAD (Section 4.4); earlier reads on the full cached DoTA 0.6142 (published 62.60) and MSAD 0.8991 / 0.8949 (published 90.41). To do: `pred_results.pkl`, UMIL's TAD list, the MSAD traffic slice |
| B1 | LaGoVAD, retrained | Each protocol's training set, 3 seeds | Fair baseline with exactly the thesis data, features and schedule; separates the gain of domain-matched data from the gain of motion. P3 done (5 seeds): DoTA-CAP-dev macro 0.71 |
| B2 | LaGoVAD + motion stream (one VideoMAE V2-S vector) | Same, 3 seeds | The preliminary model; STP's starting point (ablation row R0). P3 done: 0.81 macro with one seed, 0.82 over 5 seeds |

**Group B: published methods with released weights, re-scored for a second metric.** VadCLIP and DSANet (UCF-Crime and XD-Violence checkpoints, on the same CLIP ViT-B/16 features); PEL4VAD (with I3D features extracted for TAD, DoTA and DADA-2000); RTFM and MGFN (MSAD checkpoints, for the traffic slice); MOVAD and Simple-TAD (DoTA and DADA-2000, frame-supervised context only). Each runs on the same test lists and is scored with AUC, AP and, where defined, AUC\_A.

**Group C: same-feature competitors retrained in-domain.** VadCLIP's public code is retrained on P1, P2 and P3 with the thesis's CLIP features. STPrompt has no public code, so its detection branch (VadCLIP + SA² + learnable prefix + distance-based temporal adapter, without LLM localisation) is reimplemented on the same cached features and retrained on P1–P3. This row shows whether LaGoVAD's host and the motion stream add anything over STPrompt itself. Methods without released weights are quoted from their papers only: UMIL and WSAL on TAD; π-VAD, GS-MoE and Alert-CLIP on MSAD; MultiDomain and WS-DoTA; and the frame-supervised TTHF, PromptTAD and ISCRTAD.

### 4.4 Evaluation protocols

Four protocols cover in-domain and zero-shot use on fixed-camera and dashcam video. P1 meets the comparison rule (at least two published methods on at least two metrics, same fixed-camera traffic data) without recomputation. P2 adds four-metric rows on MSAD. P3 already reports macro, micro and AP on four zero-shot sets; P4 meets the rule on TAD and MSAD once released checkpoints are re-scored for the second metric.

**Metrics.**

- **AUC\_O / AP:** frame-level ROC AUC and average precision, pooled over all test frames.
- **AUC\_A / AP\_A:** the same, over abnormal test videos only; they measure localisation inside abnormal videos.
- **DoTA-CAP macro and micro AUC:** every third frame, whole clip, scores interpolated to native frames. Micro pools all frames after per-clip min-max normalisation; macro averages per-clip AUC over two-class clips. Both are reported overall, for ego-involved and ego-uninvolved clips, per box-size bin and on PromptTAD's small-object subset. DADA-2000 and DAD use raw scores.
- **STAUC:** DoTA's spatio-temporal AUC. For each true-positive frame, TARR is the fraction of the top-N% highest-scoring map pixels that fall inside the union of anomalous-agent boxes; STAUC is the area under the TARR-weighted true-positive rate against the false-positive rate, so STAUC ≤ AUC. It is computed on the boxed DoTA-CAP clips with `stauc.py`, with the 7×7 maps upsampled first. DoTA's official 1,305-video list cannot be scored in full, so these numbers are labelled as a subset. Only the ranking of crops inside a frame matters for TARR.
- **Pointing game:** the share of anomalous frames whose map peak falls inside a box.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P1 TAD in-domain (fixed camera) | TAD train → TAD test, UMIL split | AUC\_O, AUC\_A, AP | UMIL and WSAL on both metrics; AUC\_O-only rows; LaGoVAD, VadCLIP and STPrompt retrained |
| P2 MSAD in-domain (multi-scene surveillance) | MSAD weak train → MSAD test; traffic slice and non-traffic scenes scored separately | AUC, AP, AUC\_A, AP\_A | π-VAD and GS-MoE (four metrics) and Alert-CLIP (two); RTFM, MGFN, UR-DMU on the traffic slice |
| P3 dashcam | Nexar windows train → Nexar test (in-domain); DoTA-CAP, DADA-2000, DAD and TAD (zero-shot) | Macro, micro, AP; DoTA-CAP by ego flag and box size; STAUC, pointing game | Released LaGoVAD re-scored on the same frames; VadCLIP and PEL re-scored; published rows as context only |
| P4 cross-setting zero-shot | P3 model → MSAD and UMIL's TAD list; P2 model → TAD (both lists) | AUC, AP, AUC\_A | LaGoVAD, VadCLIP, PEL; MultiDomain and MSAD's own cross-dataset rows (AUC) |

**Published in-domain rows (P1, P2).**

| Method | Venue | TAD AUC\_O | TAD AUC\_A | MSAD AUC | MSAD AP | MSAD AUC\_A | MSAD AP\_A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WSAL | IEEE TIP 2021 | 89.64 | 61.66 |  |  |  |  |
| UMIL | CVPR 2023 | 92.93 | 65.82 |  |  |  |  |
| GS-MoE | ICCV 2025 |  |  | 87.72 | 68.26 | 69.54 | 76.68 |
| π-VAD | CVPR 2025 |  |  | 88.68 | 71.26 | 71.25 | 77.86 |
| VadCLIP framework + CLIP | Alert-CLIP's baseline |  |  | 87.45 | 71.09 |  |  |
| VadCLIP framework + Alert-CLIP | CVPR 2026 |  |  | 89.63 | 78.24 |  |  |

TAD AUC\_O-only rows add context: MIST 89.2, RTFM 89.6, CUPL 91.66, MultiDomain (TAD→TAD) 90.71. UMIL drops to 90.71 AUC\_O with a frozen backbone, the fairer reference for frozen features. Two warnings for P1 come from this project's runs. A constant score per clip reaches 0.9226 micro on the TAD list used here, so AUC\_O largely measures clip classification. In-domain TAD training with a CLIP-only host collapsed into a clip classifier (clip AUC 0.9975, macro down from 0.7578 to 0.6174).

**DoTA rows and their protocols.** Only rows read on the same clips and the same protocol are ranked against the thesis; everything else is context, including published numbers under per-video normalisation, because they use the full DoTA list and another stride.

| Method | Supervision / training data | DoTA protocol | AUC | Comparable to thesis? |
| --- | --- | --- | --- | --- |
| Thesis motion-stream model (R0) | Bag-level, Nexar, zero-shot | DoTA-CAP-dev, per-clip min-max, 569 clips | 0.81 macro / 0.73 micro | Reference |
| LaGoVAD, released checkpoint re-scored | Video-level, PreVAD, zero-shot | Same clips, same evaluation | 0.64 macro / 0.61 micro | Yes, but not matched (other training set) |
| LaGoVAD, published | Video-level, PreVAD, zero-shot | Full DoTA, stride 8, per-video normalised | 62.60 | Context only |
| VadCLIP | Weak, UCF-Crime / XD-Violence / PreVAD, zero-shot | Same as published LaGoVAD | 50.93 / 49.00 / 50.59 | Context only, until re-scored |
| PEL (I3D) | Weak, UCF-Crime / XD-Violence, zero-shot | Same as published LaGoVAD | 53.05 / 44.97 | Context only, until re-scored |
| AnoPred; DoTA ensemble; MAMTCF | Unsupervised, DoTA in-domain | Pooled raw scores | 67.5; 73.0 (STAUC 48.5); 76.6 | Context only |
| WS-DoTA (RTFM + transform) | Weak, D²-City + DoTA in-domain | \~1,031 clips, normalisation not stated | 78.2 | Context only |
| MOVAD, PromptTAD, TTHF, ISCRTAD | Frame labels, DoTA in-domain | Pooled; TTHF post-processed | 82.2, 83.9, 84.7, 86.2 | Upper reference only |
| Simple-TAD DAPT-VideoMAE-S / -L | Frame labels, fine-tuned, DoTA in-domain | Pooled | 86.4 / 88.4 | Upper reference only |
| FLaRA; BADAS | Clip-level, Nexar | Video-level anticipation AUC, 327 positive / 40 synthetic negative clips | 90.0; 90.6 | No |

On the same clips and evaluation the 0.73 micro sits 11.5 points above the re-scored LaGoVAD's 0.61 (macro +0.169 [+0.144, +0.196]). Row B1 separates how much of this comes from dashcam training data and how much from motion: the Nexar-trained CLIP-only model already reads 0.71 macro, and adding VideoMAE gives 0.82 (5 seeds).

**Zero-shot results so far (one seed, preliminary).** LaGoVAD is the released checkpoint trained on PreVAD. The motion-stream model (R0) and the first SA² (R2) are trained on Nexar. Brackets are 95% cluster bootstrap intervals of the macro.

| Set | Metric | LaGoVAD | Motion stream (R0) | First SA² (R2) | Base rate |
| --- | --- | --- | --- | --- | --- |
| DoTA-CAP-dev (569 clips, min-max) | Macro AUC | 0.642 \[0.617, 0.667\] | 0.811 \[0.795, 0.827\] | 0.811 \[0.793, 0.829\] |  |
|  | Micro AUC | 0.611 | 0.726 | 0.729 |  |
|  | AP | 0.384 | 0.556 | 0.578 | 0.327 |
| DADA-2000 (370 videos, raw) | Macro AUC | 0.620 \[0.593, 0.648\] | 0.762 \[0.737, 0.786\] | 0.758 \[0.733, 0.782\] |  |
|  | Micro AUC | 0.539 | 0.698 | 0.705 |  |
|  | AP | 0.314 | 0.527 | 0.536 | 0.289 |
| TAD (100 videos, 60 two-class, raw) | Macro AUC | 0.758 \[0.696, 0.815\] | 0.698 \[0.622, 0.770\] | 0.722 \[0.652, 0.789\] |  |
|  | Micro AUC | 0.791 | 0.603 | 0.717 |  |
|  | AP | 0.257 | 0.212 | 0.275 | 0.072 |
| DAD (466 videos, 165 positive, raw) | Macro AUC | 0.646 \[0.588, 0.702\] | 0.732 \[0.690, 0.772\] | 0.757 \[0.718, 0.796\] |  |
|  | Micro AUC | 0.568 | 0.669 | 0.702 |  |
|  | AP | 0.064 | 0.110 | 0.139 | 0.047 |

The first SA² shows no difference from the motion-stream model on DoTA-CAP-dev, DADA-2000 and DAD, and a fragile gain on TAD (+0.024 \[+0.0005, +0.050\], mostly clip-level). On TAD both models are not separated from LaGoVAD on macro and sit below it on micro. DAD's macro is not localisation: every positive is abnormal on the same last two rows, so a no-pixel time ruler scores 1.000. Micro on TAD and DAD measures clip classification (constant-score-per-clip oracle 0.9226 and 0.8390). These rows must never be written as "beats LaGoVAD".

**Reporting rules.**

- TAD is reported on two separate columns, LaGoVAD's list and UMIL's list, never mixed.
- Every final thesis number is a mean ± standard deviation over 3 seeds; a one-seed result is labelled preliminary.
- Subsets (ego flag, box-size bins, categories, traffic slices) carry 95% bootstrap confidence intervals clustered by source video. STP against the motion-stream model is tested with a paired bootstrap.
- All models use the same test lists, frame-to-score mapping and normalisation, and every table names each row's training set and protocol.

### 4.5 Ablation studies

The ablation ladder repeats STPrompt's own spatial ladder on traffic data (R1–R3), adds the first SA² for reference (R2), then adds the parts of SA²-v2 one at a time (R4–R8), then the text prefix and the traffic-specific variants. Every row keeps VideoMAE at one global vector per step, uses bag-level labels only, and reports P3 (DoTA-CAP macro / micro by ego flag and box size; DADA-2000, DAD and TAD) plus P1 (AUC\_O, AUC\_A) and P2 (AUC, AP) once those protocols are built, and STAUC where a map exists. The expected outcomes are priors written down in advance; R0 and R2 already have one-seed results.

| Row | Variant | Isolates | Expected if STP's premise holds |
| --- | --- | --- | --- |
| R0 | Motion stream only (c\_t + W\_u term) | Reference | Measured: 0.81 / 0.73 on DoTA-CAP-dev |
| R1 | + average of all 49 crop features | More local features, no selection | Small or no gain (STPrompt: +0.8 UCF, −0.1 UBnormal) |
| R2 | + first SA²: top-12 by motion, CLS-queried attention, W\_q zero-initialised (version 1) | The module as first built | Measured: about R0 (+0.0001); uniform-replacement ΔAUC still to measure (Step D) |
| R3 | + motion top-12 selection + plain average | The fixed motion prior | Gain over R1, largest on ego-uninvolved and small-box clips |
| R4 | + soft pooling over all 49 crops with logits ℓ = prior + learned residual (no text term, no new loss) | Scorer and leverage, without new supervision | About R3 or slightly above; entropy leaves its start |
| R5 | R4 + patch-level MIL (dense negatives from normal-labelled bags) | Patch-level supervision | Gain over R4 on STAUC and small-box clips |
| R6 | R5 + text term (γ·z(cos(P, τ\_Δ))) | Text-conditioned relevance | Small gain; larger on TAD categories and open-world definitions |
| R7 | R6 + branch-only head + motion dropout | Shortcut control | Branch-only AUC (VideoMAE term off) above B1; full model about equal |
| **R8** | **R7 + definition flip (full SA²-v2 visual part)** | Distractor signal | **Best visual variant; gains concentrated on ego-uninvolved and small-box clips** |
| R9 | R0 + learnable prefix only | The text part alone | Small zero-shot gain; drift@5 no worse |
| **R10** | **R8 + learnable prefix (full LaGoVAD-STP v2)** | Both parts together | **Best overall** |
| R11 | R10 with ego-compensated motion prior | Whether removing the frame-wide change helps under ego-motion | About R10: on DoTA-CAP-dev it did not change the pointing, so this row is low priority |
| R12 | R10 without ρ (motion prior only) | The outlier prior | About R10 on fixed cameras; ρ may hurt on dashcams (hood, near-field vehicles) |
| R13 | R10 with ρ only (β\_1 = 0) | The motion prior | Below R10 on fixed cameras; may be above on dashcams |
| R14 | R10 with dense ClearCLIP patch tokens from one pass, pooled to 7×7, instead of 49 crops | Cost against quality | Somewhat below R10 at about 1/50 of the CLIP cost |
| R15 | R10 with LaGoVAD's released 32 soft-prompt tokens instead of the prefix | Prefix form against the released prompts | At or below R10 |

H1 holds if R8 beats R0 with larger gains on ego-uninvolved and small-box clips; H2 if R3 beats R1, R8 beats R3 and R11 ≥ R10 on dashcam sets; H3 if R10 beats R8 on zero-shot AUC without a worse drift@5; H4 if R8 or R10 beats a uniform map, a centre prior and R3's unweighted map on STAUC and the pointing game; H6 if the branch-only run of R8 beats B1 and uniform replacement lowers R8's AUC.

**Side ablations within the component.**

- **Encoder depth:** the patch encoder with one Transformer layer (default), zero layers (scorer on projected tokens only), and two layers.
- **Temperature:** learnable T (default) against T fixed at 1.
- **Patch-level MIL:** top-m crops m ∈ {1, 3, 5} in abnormal bags; weight λ\_p ∈ {0.05, 0.2, 0.5}; normal-only negatives against both terms.
- **Motion dropout:** p ∈ {0, 0.3}; and the branch-only head with and without the VideoMAE term.
- **Fusion:** plain element-wise sum (default, as in STPrompt) against a zero-initialised gate tanh(γ\_s)·x^AS\_t.
- **Motion spacing:** neighbours at ±1 row (default: 0.27 s on Nexar, 0.30 s on DoTA-CAP) against ±2 rows.
- **Crop resolution:** windows cut from the 224×224 resize only, as in STPrompt and the existing caches. Native-resolution crops are not planned: the line that asked for them stopped on the patch-only read of Section 3.2.1.
- **Neighbours in ρ:** k ∈ {3, 5, 8}.
- **Prefix:** length l ∈ {4, 8, 16}; random against "a dashcam video of" initialisation; with and without STPrompt's contrastive loss that keeps label embeddings apart.
- **Encoder:** VideoMAE V2-S (default) against VideoMAE-S (Kinetics-400) and DAPT-VideoMAE-S (BDD100K), each still one 384-d vector.
- **Captions:** with and without LaGoVAD's caption contrastive loss.

**Analyses.**

1. **Where the gain lands.** DoTA ego-involved vs ego-uninvolved clips crossed with box-size bins and PromptTAD's small-object subset; the nine categories; TAD Accident, PedestrianOnRoad and the remaining categories pooled; the MSAD traffic slice against its non-traffic scenes; Nexar-only against Nexar + CCD training, if CCD passes its probe.
2. **Spatial evidence.** STAUC and pointing game for SA²-v2's patch map q against a uniform map (R0), the unweighted selection map (R3), the weights α and a centre prior, with examples of successes and failures.
3. **Selection behaviour.** Where the highest-logit crops land on dashcam vs fixed-camera video (the S3 statistic on test data), raw against ego-compensated motion, and how much of the logit comes from each of the prior, text and residual terms.
4. **Learned parts.** Attention entropy over training (does α leave its start, unlike MQR's and the first SA²'s?); uniform-replacement ΔAUC; the branch-only run; and the prefix's effect on definition sensitivity: score change of accident frames when the definition includes or excludes accidents, and drift@5 on MSAD.
5. **TAD definitions.** Main results with LaGoVAD's two-class TAD prompt against each of TAD's 7 categories with its own definition.
6. **Cost.** Parameters, storage, peak training memory and measured frames per second for rows R0, R10 and R14.

**Budget.** Rows R0–R10 run first on P3 with 3 seeds, one seed at a time, with 1,749 steps per run; P1 and P2 need their features and in-domain settings first. P4 reuses those models. Rows R11–R15 and the side ablations run on P3 only, first with one seed, then with three for the variants that are kept. Crop extraction is done once and shared by every row. A one-epoch timing run in the first week converts this plan into GPU-hours; if the total exceeds the budget, the ladder is cut to R0, R1, R3, R5, R8, R9 and R10 plus the diagnostics of Step D, which still test H1–H4 and H6.

## References

**Baseline, backbone and component**

- Liu et al., [Language-Guided Open-World Video Anomaly Detection under Weak Supervision (LaGoVAD)](https://arxiv.org/abs/2503.13160), ICLR 2026; [code, checkpoint, PreVAD annotations](https://github.com/Kamino666/LaGoVAD-PreVAD)
- Wu et al., [Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts (STPrompt)](https://arxiv.org/abs/2408.05905), ACM MM 2024 (source of SA² and the learnable prefix)
- Dai et al., No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection (LAVIDA), CVPR 2026; [code](https://github.com/VitaminCreed/LAVIDA) (source of the text-queried scoring, outlier-prior and distractor ideas)
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
- [Localizing to Debias](https://arxiv.org/abs/2608.12045), arXiv 2608.12045; [Learning Where and When](https://engineers.usf.edu/sislab/publications/learning-where-and-when-patch-based-spatiotemporal-localization-weakly-supervised), arXiv 2606.29498 (both read through summaries; verify in the PDFs)

**Traffic anomaly detection**

- Qiu et al., [PromptTAD: Object-Prompt Enhanced Traffic Anomaly Detection](https://doi.org/10.1109/LRA.2025.3572428), IEEE RA-L 2025
- [MAMTCF](https://arxiv.org/abs/2307.14575), arXiv 2307.14575
- Liu et al., Future Frame Prediction for Anomaly Detection (AnoPred), CVPR 2018
- Rossi et al., MOVAD, ICASSP 2024; [code](https://github.com/IMPLabUniPr/movad)
- Liang et al., TTHF, IEEE TCSVT 2024; Liang et al., ISCRTAD, IEEE T-ITS 2025
- Shi et al., [Structured Graph Reasoning for Traffic Anomaly Detection](https://doi.org/10.1109/TMM.2026.3703345), IEEE TMM 2026

**Dense CLIP features (ablation R14)**

- Lan et al., [ClearCLIP](https://arxiv.org/abs/2407.12442), ECCV 2024; Wang et al., [SCLIP](https://arxiv.org/abs/2312.01597), ECCV 2024
- Zhou et al., [AnomalyCLIP](https://arxiv.org/abs/2310.18961), ICLR 2024

**Datasets**

- Yao et al., [DoTA](https://arxiv.org/abs/2004.03044), IEEE TPAMI 2022; [annotations and STAUC code](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
- Zhu et al., [MSAD](https://msad-dataset.github.io/), NeurIPS 2024 Datasets and Benchmarks
- Bao et al., [CCD](https://github.com/Cogito2012/CarCrashDataset), ACM MM 2020
- Moura et al., [Nexar Dashcam Collision Prediction Dataset and Challenge](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf), CVPR 2025 Workshops
- Fang et al., DADA-2000, IEEE T-ITS 2021; Fang et al., [MM-AU](https://arxiv.org/abs/2403.00436), CVPR 2024
- Yu et al., BDD100K, CVPR 2020; D²-City, arXiv 2019
