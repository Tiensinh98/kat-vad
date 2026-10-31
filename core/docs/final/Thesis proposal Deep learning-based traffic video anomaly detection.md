# Thesis proposal: Deep learning-based traffic video anomaly detection

Oct 31, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-AP**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. It starts from LaGoVAD (ICLR 2026) and keeps a frozen VideoMAE V2-S motion stream as the permitted backbone change; in the student's single-seed preliminary runs this stream lifts zero-shot DoTA to 0.81 macro / 0.75 micro min-max AUC. The one novel component is a weakly supervised **Agent Prompt (AP) branch**. An off-the-shelf detector and tracker propose road agents; each agent becomes a token built from a PromptTAD-style box encoding, a frozen-CLIP embedding of its crop and VideoMAE motion pooled along its track; a small relation module lets agents, the scene and an ego token interact; and an agent-level MIL loss, which treats every agent of every normal video as a negative, trains a per-agent anomaly score from video labels alone. The branch enters LaGoVAD through a second zero-initialised gate. The evaluation is rebuilt around a finding computed from DoTA's released annotations: a score that uses only each frame's relative position in its clip reaches about 0.85 macro AUC on DoTA's test list, above the preliminary model. The claim under test is therefore position-controlled: scoring detected agents lifts position-stratified AUC on ego-uninvolved DoTA clips and points at the anomalous agent more often than centre and largest-box priors, with per-agent evidence also scored by DoTA's spatio-temporal AUC (STAUC), which our search (to October 2026) found no weakly supervised method reporting.

**What changed from the October 29 proposal.** The earlier novel component, the Motion Query Readout (MQR), learned a tile-level readout over the frozen VideoMAE tokens. In the student's single-seed experiments it did not learn under video labels. Free queries never left the uniform state (with a 100× learning rate they saturated over about 40 of 49 tiles) and attention entropy stayed flat. The tile scorer with dense normal-video negatives pointed at the anomalous agent below chance (−0.04) and 0.35 below a centre prior; run on CLIP patches, the same scorer put 93% of its peaks on the top and bottom rows (sky and hood), with mean tile probabilities of 0.116 on normal and 0.117 on abnormal videos. Position-controlled gains of every spatial readout lay between −0.01 and +0.002, and a second "tile query" design also failed. The pivot keeps what worked (the motion stream) and the MQR loss logic (a direct score per instance, dense negatives on normal videos), and changes the instance set: from 49 tiles, most of them background, to the few agents a detector proposes. MQR is written up as a documented negative result in an appendix chapter and reappears as one controlled row of the ablation ladder (Section 4.5).

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event is happening in it. Abnormal events include collisions, loss of control, illegal turns, wrong-way driving, pedestrians on the road and road spills. These are the categories used by TAD and DoTA.

Two camera settings define the problem space:

- **Fixed camera (non-ego).** TAD and the traffic scenes of MSAD. The camera is static; the anomalous agents are often small and far away.
- **Dashcam (ego-centric).** DoTA, DADA-2000, CCD and Nexar. The camera moves with the car. DoTA also marks each clip as ego-involved (the ego car takes part in the anomaly) or ego-uninvolved (it only observes); this proposal uses those two terms for that flag.

The thesis follows LaGoVAD's weakly supervised, language-guided formulation. Training uses video-level annotations only: one label y\_i ∈ {0, 1} per video, plus a category and a hand-written caption template. At test time the model receives a video and a text definition Z of what counts as anomalous, and returns a score per frame. This thesis adds a second output: an anomaly score for every detected road agent, which places the evidence on a box.

```latex
\Phi_\theta : (V, Z) \;\mapsto\; \{ s_t \}_{t=1}^{N},\; \{ (b_{t,i},\, p_{t,i}) \}_{i \in \mathcal{A}_t}, \qquad s_t \in [0,1],\;\; p_{t,i} \in [0,1], \qquad E_t = \sum_{i \in \mathcal{A}_t} p_{t,i}\, G(b_{t,i})
```

Here s\_t is the anomaly score of frame t, 𝒜\_t is the set of agents a detector and tracker propose at t, b\_{t,i} is the box of agent i and p\_{t,i} its anomaly score. The evidence map E\_t paints each box with a Gaussian G scaled by its score, which is DoTA's own convention for object-based methods. The video-level annotations supervise training through multiple-instance learning (MIL): the mean of the top-k frame scores of a video must match its video label, and, for the new branch, the top agents of the top steps must too.

Five properties make traffic video harder than generic surveillance:

1. **Anomalies are motion events.** A collision is defined by how trajectories meet, not by what one frame looks like. LaGoVAD encodes single RGB frames with CLIP and has no motion input; it reaches 89.56 AUC on TAD but 62.60 on dashcam DoTA. The I3D-based PEL scores 44.97–53.05 on DoTA, so a generic, surveillance-trained motion encoder does not fix this. In the student's single-seed preliminary runs, a frozen VideoMAE V2-S stream, mean-pooled and added through a zero-initialised gate, lifts zero-shot DoTA to 0.81 macro / 0.75 micro min-max AUC, and most of the gain over CLIP-only LaGoVAD comes from it.
2. **The agents that matter are small.** In DoTA's released annotations (computed, to be re-verified), the median anomalous-agent box covers 2.3% of a 1280×720 frame, and after LaGoVAD's 224×224 resize it is about 31×38 px, roughly 2×2.4 CLIP patches; 12.6% of these boxes are smaller than one 16-px patch on both sides and 26.2% on at least one side. One global vector per frame, or one 32-px tile, dilutes them.
3. **The camera moves.** In dashcam video, ego-motion dominates frame-to-frame change. In the student's preliminary probes, hand-crafted cues (feature differences, ego-motion-compensated optical flow) did not help, and VideoMAE token change between overlapping windows pointed at the anomalous agent only at chance.
4. **Labels are scarce.** Frame-level labels are expensive and anomalies are rare, so the model must learn from video-level labels.
5. **The main dashcam benchmark rewards timing.** DoTA contains only abnormal clips and is scored after per-video min-max normalisation, so its AUC depends only on the order of frames inside each clip. DoTA's clips were trimmed so that the anomaly sits in the middle: computed from DoTA's metadata, the anomaly starts at a median 36% and ends at a median 71% of the clip on the 1,402-clip test list. A score that uses only relative position reaches about 0.847 macro / 0.819 micro AUC on that list, and 0.853 macro on ego-uninvolved clips. The preliminary model's 0.81 macro is below this video-blind score; on ego-uninvolved clips (0.68–0.74) it is 0.11–0.17 below it, while on ego-involved clips (about 0.86–0.88 against 0.842) it is slightly above it. On the student's 569-clip DoTA dev split, the position-only predictor beat every learned model. These figures are computed from DoTA's released annotations and are to be re-verified before they are quoted.

The scope is offline detection: the whole video is available and the question is whether an anomaly is happening, not whether one will happen (anticipation, as in Nexar). No large language model or multimodal LLM runs at training or test time; a frozen CLIP text encoder is allowed. The object detector and tracker are fixed, off-the-shelf preprocessing tools trained on COCO and MOT data; their boxes are inputs, never supervision, and no ground-truth box or frame label enters training. All training text is hand-written: class names, category definitions and caption templates. The released LaGoVAD checkpoint, used only as a reference row, was trained on PreVAD, whose captions its authors generated offline with Qwen models.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from cheap video-level labels, sees motion, and says which road agent is anomalous. It must keep LaGoVAD's ability to take the anomaly definition as text, and run on one academic GPU without any LLM.

The thesis builds this in two steps, matching the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** A frozen VideoMAE V2-S encoder (CVPR 2023), the strongest small masked-video encoder for traffic anomalies in Simple-TAD's comparison, feeds mean-pooled motion features into LaGoVAD through a zero-initialised gate. The student's preliminary experiments show this step helps; their setup and gains will be documented in an appendix.
- **Novel component (Agent Prompt branch).** Detected and tracked road agents become tokens that carry box geometry and kinematics, a CLIP crop embedding in the text space, and VideoMAE motion pooled inside the track. A small relation module adds agent, scene and ego interaction, and an agent-level MIL loss with dense normal-agent negatives trains one anomaly score per agent from video labels. The branch adds an agent-enhanced frame token to LaGoVAD's input through a second zero-initialised gate, and its agent scores form the evidence maps.

A second purpose is a fair evaluation. Published traffic numbers are mostly single-metric AUC on inconsistent splits, and DoTA's min-max AUC is matched by a position-only score. The thesis will report at least two metrics against at least two published methods on the same fixed-camera traffic data, a position-only reference row next to every DoTA number, position-stratified AUC as the primary dashcam detection metric, box-level spatial metrics against centre and box-size priors, and the recall of the detector on DoTA's anomalous agents, which no paper we found reports.

### 1.3 Research aims

The thesis has six aims:

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint and retrain LaGoVAD on the thesis training data, with AUC, AP and AUC on abnormal videos (AUC\_A) on TAD, MSAD, DoTA and DADA-2000.
2. **Add the motion stream.** Integrate the frozen VideoMAE V2-S stream and measure its gain over CLIP-only LaGoVAD, under both the standard and the position-controlled protocols.
3. **Audit DoTA's position bias.** Recompute and release the position-only reference, define position-stratified and onset-centred AUC, and apply time-reversal, shuffle and padding controls to every model.
4. **Design and train the Agent Prompt branch.** Detector and tracker proposals, agent tokens, relation module, agent text alignment and agent-level MIL loss, using video-level annotations only, after a pre-registered go/no-go check of detector recall (Step 0).
5. **Measure spatial localisation under weak supervision.** Score the agent maps against DoTA's anomalous-agent boxes with box-level hit rate, pointing game and STAUC, against centre, largest-box and lowest-box priors.
6. **Explain the gains.** Break results down by ego-involved vs ego-uninvolved clips, agent box size, night, anomaly category and detector recall, and check that MSAD's non-traffic scenes do not regress.

Aims 1–3 are prerequisites; aims 4–6 map to the hypotheses below. "Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval excludes zero; on DoTA the bootstrap resamples source videos, not clips (Section 4.4).

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1 (primary). The agent branch improves detection beyond clip position, mostly on ego-uninvolved clips | 50-bin position-stratified AUC on DoTA's ego-uninvolved test clips (597 of 1,402), LaGoVAD-AP against the same model without the branch; secondary: ego-involved clips, box-size bins, onset-centred AUC | Gain ≥ 0.01 with a 95% source-clustered paired CI that excludes zero; gain on ego-uninvolved clips at least as large as on ego-involved clips |
| H2 (co-primary). The agent scores point at the anomalous agents | Box-level hit rate on anomalous frames of DoTA's official 1,305-video list (the top-scored detected agent overlaps an anomalous-agent box at IoU ≥ 0.3); pointing game and mass-in-box on the painted map; STAUC in both conventions (Section 4.4) | Beats the centre-nearest-box, largest-box and lowest-box priors and the training-free change ranking, with paired CIs that exclude zero |
| H3. The gain holds across camera settings without hurting surveillance | TAD AUC\_O / AUC\_A; MSAD AUC / AP, its traffic slice and its non-traffic scenes; CCD raw-score AUC; DADA-2000 position-stratified AUC | Gains on TAD, the MSAD traffic slice and the dashcam sets; MSAD's non-traffic scenes equivalent to the motion-stream model within ±0.01 AUC (two one-sided tests) |
| H4. Learned agent scoring matters, not just boxes or a fixed ranking | Ablation ladder: training-free change-map pooling, fixed top-K agents by change, geometry-only tokens, random boxes, bag-only normal negatives | Full LaGoVAD-AP beats every simpler row on the H1 and H2 endpoints |

H1 and H2 are co-primary and are tested with a Holm correction. Raw DoTA min-max AUC is still reported for comparability with published rows, but never as evidence of a gain without the position-only row beside it: a model can improve it by producing a better mid-clip bump for reasons unrelated to the anomaly.

## 2. Literature review

### 2.1 Overview and classification of video anomaly detection method branches

Video anomaly detection (VAD) methods divide by the supervision they train with into five branches, plus the adjacent task of accident anticipation. Traffic methods cut across these branches along a second axis: fixed camera (non-ego) or dashcam (ego-centric). The table places the papers reviewed for this thesis, mostly 2024–2026.

| Branch | Training signal | Typical mechanism | Representative methods (venue) | Traffic evidence | Role in this thesis |
| --- | --- | --- | --- | --- | --- |
| Unsupervised / one-class | Normal videos only | Reconstruction, frame prediction, diffusion, object-centric or scene-graph normality | MA-PDM (AAAI 2025), ADSM (CVPR 2025), TAO (CVPR 2025), Structured Graph Reasoning (IEEE TMM 2026), DoTA's FOL baselines | DoTA and MM-AU, AUC and STAUC; DoTA's own baselines reach 73.0 AUC / 48.5 STAUC | Context; source of the box-painted STAUC map and of the ego node |
| Weakly supervised | Video-level labels, MIL | Snippet features (I3D, CLIP) + top-k MIL; CLIP text prompts; object tubes or grid patches as instances | Sultani (CVPR 2018), WSSTAD (2021, object tubes), RTFM (ICCV 2021), WSAL (IEEE TIP 2021), UMIL (CVPR 2023), MultiDomain (NeurIPS 2024), VadCLIP (AAAI 2024), STPrompt (ACM MM 2024), π-VAD (CVPR 2025), GS-MoE (ICCV 2025), DSANet and RefineVAD (AAAI 2026), LAS-VAD, Fine-VAD, Road Less Seen, Alert-CLIP (CVPR 2026) | TAD (WSAL, UMIL; zero-shot MultiDomain), MSAD (π-VAD, GS-MoE, Alert-CLIP), CCTV road accidents (WSSTAD's STRA), dashcam WS-DoTA (ECCV 2024 Workshop) | **The thesis's branch** |
| Open-vocabulary / open-world | Video labels + category text | CLIP text space defines anomaly classes or free-text definitions at test time | Anomize (CVPR 2025), **LaGoVAD (ICLR 2026)** | Zero-shot TAD, DoTA, MSAD, LAD (LaGoVAD) | **Baseline** |
| Fully supervised traffic detection | Frame-level labels (+ boxes) | Video encoder + temporal head; object prompts; agent graphs | MOVAD (ICASSP 2024), TTHF (IEEE TCSVT 2024), **PromptTAD (IEEE RA-L 2025)**, ISCRTAD (IEEE T-ITS 2025), Simple-TAD (ICCV 2025 Workshops) | DoTA and DADA-2000, AUC up to 88.4 | **Source of the component's architecture (PromptTAD)**; context rows; evidence for the motion encoder |
| LLM / MLLM-based | None, or few-shot rules, or light tuning | Captioning + LLM reasoning, prompting, probing of MLLM heads | LAVAD (CVPR 2024), AnomalyRuler (ECCV 2024), VERA, Holmes-VAU (CVPR 2025), EventVAD (ACM MM 2025), HeadHunt-VAD (AAAI 2026), LAVIDA (CVPR 2026), AnyAnomaly (WACV 2026), SteerVAD, GridVAD | Almost none on traffic sets | Excluded: LLM cost |
| Accident anticipation (adjacent task) | Positive / negative dashcam clips | Predict a collision before it happens; object attention | Nexar dataset (CVPR 2025 Workshops), DSTA, BADAS, FLaRA | Nexar, CCD, DAD; mean AP at fixed times before the event | Different task; evidence that per-object features help on dashcam video |

Three findings from this review shape the proposal. First, CLIP-feature weak supervision dominates surveillance benchmarks but sits near chance on dashcam DoTA. Second, the strongest traffic results come from video foundation models trained with masked video modelling, and Simple-TAD found that this pre-training beats vision-language pre-training for traffic anomalies. Third, the methods that win on rare, distant and ego-uninvolved traffic agents (PromptTAD, ISCRTAD) all add object-level evidence from a detector, but train it with frame labels and, in PromptTAD's case, ground-truth anomalous boxes. Under video labels, the only method with detector-proposed instances is WSSTAD (2021); since 2024 the weakly supervised spatial methods have all moved to grid patches or tubelets, and none of them runs on dashcam data.

### 2.2 Emerging trends and unresolved challenges

Six trends run through the 2024–2026 literature:

1. **Vision-language features as the default.** Since VadCLIP (AAAI 2024), most weakly supervised methods start from frozen CLIP frame features and text prompts: DSANet and RefineVAD (AAAI 2026), LAS-VAD, Fine-VAD and Alert-CLIP (CVPR 2026).
2. **Anomalies defined by language.** Open-vocabulary (Anomize, CVPR 2025) and open-world (LaGoVAD, ICLR 2026) models accept the anomaly definition at test time. LaGoVAD's drift@5 protocol changes which categories count as abnormal and measures robustness to that change.
3. **MLLMs for explanation and training-free detection.** LAVAD, VERA, Holmes-VAU, HeadHunt-VAD and LAVIDA rely on LLMs or multimodal LLMs of 2B parameters and more; AnomalyRuler also calls GPT-4, and GridVAD runs a 30B-parameter Qwen3-VL model at inference. They are not clearly better at detection: on MSAD drift@5, LaGoVAD reaches 85.6 AUC / 40.1 AP against 84.3 / 34.3 for Holmes-VAU and 63.1 / 22.4 for Qwen2.5-VL-7B.
4. **Masked-video foundation models for traffic.** Simple-TAD showed that masked video modelling transfers best to traffic anomalies. After supervised fine-tuning, VideoMAE-B (1600 epochs) reaches AUC-MCC 56.0 on DoTAref (refined DoTA labels) and 52.2 on DADA-2000, against 50.0 and 43.1 for InternVideo2-B. Further masked pre-training on normal driving video (DAPT) adds more without needing any anomalies.
5. **Spatial and object-level grounding.** PromptTAD prompts a frame-supervised detector with YOLOv9 boxes; ISCRTAD builds agent graphs from a detector, a tracker, odometry and depth; Alert-CLIP aligns region features with box and caption supervision. Under weak labels the field has gone detector-free: STPrompt (ACM MM 2024) pools motion-selected CLIP patches, "Learning Where and When" (arXiv 2606.29498) scores 7×7 grid patches with every patch of a normal video as a negative, and "Localizing to Debias" (arXiv 2608.12045) prunes VideoMAEv2 tubelets. All three report surveillance data only.
6. **Extra modalities.** π-VAD (CVPR 2025) induces pose, depth, flow and other modalities from RGB features at training time, and found flow the most useful. GS-MoE (ICCV 2025) routes anomaly types to experts.

Six challenges remain open for traffic video:

1. **Motion blindness.** Frame-level CLIP features miss what makes a collision a collision. In LaGoVAD's table every surveillance-trained model is near chance on DoTA's within-clip protocol, whether it uses CLIP features (VadCLIP, LaGoVAD: 49.00–62.60) or I3D (PEL: 44.97–53.05), while the CLIP-feature models reach 74.46–89.56 on TAD's video-pooled protocol.
2. **Spatial dilution and the ego-uninvolved gap.** Global vectors lose small and distant agents. Even frame-supervised Simple-TAD averages 81.6 / 84.1 / 85.2 AUC on ego-uninvolved DoTA clips (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips. In DSANet's analysis, VadCLIP assigns 84.0% of UCF-Crime road-accident videos to "Normal". GS-MoE reports a road-accident video scored near zero "because the anomaly occupies a very small pixel region". The student's preliminary model shows the same gap more sharply (0.68–0.74 ego-uninvolved against 0.86–0.88 ego-involved).
3. **No spatial evidence under weak supervision.** We found no weakly supervised traffic method that outputs or evaluates spatial maps, and no report of DoTA's STAUC or box-level localisation on DoTA or DADA-2000 under weak supervision (search to October 2026).
4. **Domain shift and ego-motion.** Training data are mostly surveillance or web video, test data are dashcam video, and camera motion masks agent motion.
5. **Timing bias in trimmed benchmarks.** DoTA's anomalies sit in the middle third of almost every clip, so per-video min-max AUC rewards any single mid-clip bump; computed from DoTA's metadata, a position-only score reaches about 0.85 macro AUC, level with frame-supervised MOVAD (82.2). No traffic paper we found reports a position-only reference.
6. **Inconsistent evaluation.** TAD has two incompatible test lists in use (76 of LaGoVAD's 100 TAD test videos belong to UMIL's training list). Most traffic tables report AUC only, and DoTA's 1,402 test clips come from 179 source videos, 177 of which also contribute training clips (computed), so clip-level confidence intervals are too narrow.

### 2.3 Key works behind the proposed model

The proposed model is assembled from a small set of prior works: one baseline, one encoder, one object-prompt architecture, the weak-label instance methods that bound and inform its loss, and the detector and tracker that supply its agents. This subsection reviews each of them: what it does, its strengths, its weaknesses, and exactly what the thesis takes from it and changes. The table shows where each part of LaGoVAD-AP comes from; the reviews follow, and the last table covers the methods used only as comparison rows.

| Part of LaGoVAD-AP | Source work (venue) | What is taken | What the thesis changes |
| --- | --- | --- | --- |
| Host detector: CLIP stream, RoFormer, text co-attention, heads and losses | LaGoVAD (ICLR 2026), which inherits the frozen-CLIP + MIL design of VadCLIP (AAAI 2024) | Architecture, losses and schedule as released | Nothing inside the detector; checkpoints are taken at the last epoch, captions are hand-written templates, and training clips get position-robust crops |
| Motion stream (backbone change) | VideoMAE V2 (CVPR 2023), chosen on the evidence of Simple-TAD (ICCV 2025 Workshops) | The frozen distilled ViT-S, and Simple-TAD's 16-frame, 10 fps clip input | Used frozen under video-level labels, mean-pooled and fused through a zero-initialised gate, instead of fine-tuned with frame labels |
| Agent proposals | YOLOv9 (the detector PromptTAD uses); BoT-SORT and ByteTrack | A COCO-trained detector restricted to traffic classes; tracking with camera-motion compensation for dashcam video | Run at native resolution with a low confidence floor; tracks, not single last-frame boxes; boxes are fixed inputs, never supervision |
| Agent tokens (novel component) | PromptTAD (IEEE RA-L 2025) for the prompt encoder; ORViT (CVPR 2022) for RoI object tokens; STPrompt (ACM MM 2024) for CLIP crop re-encoding; TrajVAD (arXiv 2605.21957) for box kinematics | Fourier-encoded box corners with a learnable token and a "no prompt" token; RoIAlign of backbone tokens per box; a crop encoded by CLIP's global embedding; ego-compensated kinematics | Content comes from a native-resolution CLIP crop and VideoMAE tokens pooled along the track, instead of cross-attention to a fine-tuned frame map |
| Agent interaction | PromptTAD's relation-wise module; the ego node of Structured Graph Reasoning (IEEE TMM 2026) and ISCRTAD | 12 learnable relation queries; an ego pseudo-agent; pairwise geometric relations | Two blocks instead of four, plus one instance self-attention layer with a pairwise box bias; the queries build the enhanced frame token only, never the map |
| Agent text alignment | LaGoVAD's and VadCLIP's MIL alignment loss | Class-text alignment pooled by top-k over time | Pooled over agents as well as time, on CLIP crop embeddings that already live in the text space; no captions or region text |
| Agent-level MIL loss | WSSTAD (2021) for detector instances under video labels; LaGoVAD's top-k MIL; the normal-patch negatives of "Learning Where and When" (arXiv 2606.29498) | Top instances of top steps on abnormal videos; every instance of a normal video labelled 0 | Replaces PromptTAD's Hungarian-matched ground-truth box loss; instances are tracked agents with frozen foundation features |
| Position-robust training and evaluation | LaGoVAD's dynamic video synthesis; DoTA's metadata (computed audit); Nexar's fixed-offset test design | Random placement of the anomaly in synthetic sequences | Random crops before synthesis, boundary anchors, position-only reference rows, position-stratified AUC, time-reversal and shuffle controls |
| Evidence-map evaluation | DoTA (IEEE TPAMI 2022); TAO (CVPR 2025) | `stauc.py`, box-painted Gaussian maps, anomalous-agent boxes, the ego-involved flag; TAO's box-persistence filter | Scored under weak supervision against centre, largest-box and lowest-box priors, in two STAUC conventions |

#### 2.3.1 LaGoVAD (ICLR 2026): the baseline

Liu et al. define open-world VAD as a mapping Φ: (V, Z) → frame scores, where the anomaly definition Z is given as text at test time. Frozen CLIP ViT-B/16 frame embeddings pass through a 2-layer RoFormer and are fused with the CLIP text encoding of Z by co-attention. Training uses video-level labels on PreVAD (35,279 videos, 11,979 abnormal) with MIL and MIL-alignment losses, dynamic video synthesis and a caption contrastive loss with hard negatives.

- **Strengths.** It has the best zero-shot traffic results in its comparison: 89.56 AUC on TAD, 62.60 on DoTA and 90.41 on MSAD, ahead of VadCLIP and PEL. It stays robust when the definition changes (MSAD drift@5: 85.6 AUC / 40.1 AP, against 84.3 / 34.3 for Holmes-VAU). Its dynamic video synthesis, which inserts an abnormal clip at a random slot among 1–5 retrieved normal clips, already varies where an anomaly falls in a training sequence; removing it drops DoTA from 62.60 to 60.02. The detector runs no LLM, trains on one RTX 4090, and its code, checkpoint and predictions are public.
- **Weaknesses.** Its only visual input is one global CLIP embedding per sampled frame, so it sees neither motion nor where in the frame something happens: DoTA AUC is 62.60 against 89.56 on TAD, and language guidance adds only 1.20 AUC on DoTA against 5.60 on TAD. It outputs no spatial evidence. Its DoTA protocol min-max normalises each clip with no position control. Its traffic test sets are prompted with a generic "car accident" definition, although 37 of the 60 abnormal TAD test videos on its list are not accidents. Its released configuration selects checkpoints on test scores, PreVAD's captions were written by Qwen models, and the authors call the design "simplified".
- **Taken into this thesis.** The whole detector, unchanged. The thesis adds a motion input and an agent branch, replaces test-score checkpoint selection with the last epoch, and adds position-robust crops before dynamic video synthesis.

#### 2.3.2 VadCLIP (AAAI 2024): the CLIP-based weakly supervised template

Wu et al. adapt frozen CLIP to weakly supervised VAD. Frame embeddings pass through a local Transformer and a graph convolution (the LGT-Adapter). A classification branch gives binary frame scores, and an alignment branch matches frames to class-name embeddings built with learnable prompts (MIL-Align).

- **Strengths.** It established the CLIP-plus-text-prompt design that LaGoVAD, DSANet and Alert-CLIP build on, reaching 88.02 AUC on UCF-Crime and 84.51 AP on XD-Violence on a single RTX 3090, with public code and features.
- **Weaknesses.** It reads appearance only, and most of its gain comes from the temporal adapter (XD-Violence AP 72.22 without it, 84.51 with it). Zero-shot, it sits near chance on DoTA (49.00–50.93 AUC). In DSANet's analysis, its prototypes assign 84.0% of UCF-Crime road-accident videos to "Normal". Its classes are fixed at training time.
- **Taken into this thesis.** Its design reaches the thesis through LaGoVAD; its MIL-Align loss is the template for the agent text alignment; and it is retrained as the same-feature weakly supervised competitor (Group C, Section 4.3).

#### 2.3.3 VideoMAE V2 (CVPR 2023): the motion encoder

Wang et al. scale masked video autoencoding with dual masking: the encoder sees only about 10% of the video tokens (tube masking), and the decoder reconstructs only part of the masked cubes (running-cell masking at 50%). Training is progressive: self-supervised pre-training on 1.35M unlabelled clips, then supervised post-pre-training on Kinetics-710 (0.66M clips, 710 classes). The resulting billion-parameter ViT-g is distilled into smaller ViT-S and ViT-B models.

- **Strengths.** Dual masking made a 1B-parameter video ViT trainable, reaching 90.0% top-1 on Kinetics-400 and 77.0% on Something-Something V2. The distilled ViT-S is small (about 22M parameters) and public. In Simple-TAD's comparison it is the strongest small encoder for traffic anomalies after fine-tuning: AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000, against 52.1 / 48.1 for VideoMAE-S and 49.7 / 43.7 for InternVideo2-S. In the student's preliminary runs it carries most of the gain over CLIP-only LaGoVAD, even frozen and mean-pooled.
- **Weaknesses.** It is designed to be fine-tuned, and masked-model features read frozen with a linear probe lose much of their signal (V-JEPA's frozen evaluation: VideoMAE ViT-L 52.5 on Kinetics-400 with a linear probe, 77.8 with an attentive probe). Its tokens are 2×16×16 tubelets of a 224×224 input, so the 12.6% of DoTA anomalous boxes smaller than one token on both sides (computed) are mixed with background in any token. In the student's probes, token change between overlapping windows pointed at the anomalous agent only at chance, and between non-overlapping windows about 0.6 s apart only 0.08–0.11 above chance; and a learned tile readout over these tokens did not learn (the MQR result). Its pre-training data contain no driving video, and its tokens are not aligned with text.
- **Taken into this thesis.** The frozen distilled ViT-S as the motion encoder. Its mean token stays the motion stream; for the agent branch its last-layer 8×14×14 tokens are pooled with RoIAlign inside each agent's track box, slot by slot. Appearance and text alignment come from CLIP crops instead.

#### 2.3.4 Simple-TAD (ICCV 2025 Workshops): evidence for the encoder choice

Orlova et al. show that a plain video ViT with one linear head, fine-tuned with frame-level labels, outperforms specialised traffic detectors on DoTA and DADA-2000. They compare pre-training paradigms and add domain-adaptive pre-training (DAPT): further masked pre-training on normal driving video.

- **Strengths.** It is simple and fast (DAPT-VideoMAE-S: 22M parameters, 95 FPS on an A100 partition) and strong (DoTA AUC up to 88.4; DoTA → DADA-2000 up to 86.6). It shows systematically that masked video modelling transfers best to traffic anomalies and vision-language pre-training (InternVideo2, UMT) worst. DAPT on BDD100K raises ViT-S AUC-MCC on DoTAref from 52.1 to 54.9 without any anomalous data. Code, encoders and refined DoTA labels are public, and its Tables 2, 7 and 8 are the only recoverable source of PromptTAD's per-category and DADA-2000 numbers.
- **Weaknesses.** It needs frame-level labels, covers dashcam data only and takes no text. It averages all tokens before the linear head, and ego-uninvolved clips lag: 81.6 / 84.1 / 85.2 AUC (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips. It is a workshop paper, so the thesis uses it as evidence, not as the source of the component.
- **Taken into this thesis.** The choice of a masked video encoder and its clip input; the DAPT encoder as an ablation; the ego-uninvolved gap, which motivates H1; and its observation that object-aware methods (ISCRTAD, PromptTAD) do better on the rare and ambiguous categories UK and UK\*.

#### 2.3.5 PromptTAD (IEEE RA-L 2025): the source of the Agent Prompt branch

Qiu, Yang and Gong extend MOVAD (a Video Swin Transformer per short clip plus a stacked LSTM) with object prompts. YOLOv9 detects traffic objects on the last frame of each 4-frame clip. Each box becomes a SAM-style prompt token: the two corners are Fourier-encoded into D\_E = 256 dimensions and a learnable token is appended, with a learnable "no prompt" token when nothing is detected. The prompt carries geometry only; it gets visual content by cross-attending, in four SAM-like two-way blocks, to an FPN map of the Video Swin features at stride 8. A relation-wise module of four blocks with 12 learnable query tokens then aggregates the object tokens and the averaged instance-enhanced frame embedding into an enhanced frame token, which is concatenated with the original frame token before the LSTM. Training uses MOVAD's weighted cross-entropy on frame labels plus an instance loss whose targets come from Hungarian matching of YOLOv9 boxes to DoTA's ground-truth anomalous boxes.

- **Strengths.** It names the failure the thesis targets: a global view emphasises prominent regions and neglects distant or off-centre objects. Its gains land where the student's model is weakest. In Simple-TAD's reproduction of its per-category table, it beats MOVAD on ego-uninvolved categories such as LA\* (81.8 against 74.8), TC\* (85.1 against 79.6) and UK\* (79.1 against 72.9, the best of all methods listed, including DAPT-VideoMAE-L at 75.6). Transferred to DADA-2000 without fine-tuning, it holds up better than TTHF on ego-uninvolved clips (70.4 against 64.0). It reaches 83.9 AUC on DoTA, reports gains in STAUC, and evaluates a small-object subset (334 of 1,402 test videos whose anomalous boxes are under 2,267 pixels, the smallest 10%). It uses no LLM.
- **Weaknesses.** It needs frame labels and ground-truth anomalous-object boxes, neither of which exists under weak supervision. When detection fails, as in dark scenes, it "degenerate[s] into the baseline". Its object stream sees one frame per clip, with no track. It does not help ego collisions (DADA-2000 ego-involved 79.7, below TTHF's 80.9 and VideoMAE-S's 90.2), and a plain fine-tuned VideoMAE-S beats it on DADA-2000 overall (83.7 against 74.6) and on ego-uninvolved clips (78.3 against 70.4). It costs 106M parameters, 1.88 GB and 18 FPS in Simple-TAD's measurement, trains on four A6000 GPUs for 200 epochs, has no public code we could find, and is a journal paper outside the venue list. Every table in its paper, including the module ablation, the relation-module ablation and its STAUC values, was an image that did not survive text extraction; those numbers were not recoverable and are not quoted here. Its YOLOv9 variant, confidence threshold and box limit are not stated, and how it builds its STAUC map is not described.
- **Taken into this thesis.** The architecture family: box-corner Fourier prompts with a learnable type token and a no-agent token, a relation module with 12 learnable queries, and an enhanced frame token that joins the frame stream. The thesis changes the three things that need labels or a fine-tuned backbone. Supervision becomes agent-level MIL with dense normal-agent negatives instead of frame cross-entropy and Hungarian-matched box targets. Content comes from a CLIP crop embedding and VideoMAE RoI pooling along a track, instead of cross-attention to a fine-tuned feature map (kept as a side ablation). The enhanced frame token enters through a zero-initialised gate instead of concatenation. Its object-score maps are made explicit as DoTA's box-painted Gaussians.

#### 2.3.6 WSSTAD (2021): detector instances under video labels

Wu et al. (arXiv 2108.03825; commonly cited as IJCAI 2021, venue unverified) link Faster R-CNN boxes into single-object tubes and union-box tubes of interacting objects, encode them with C3D, relate them with an 8-head self-attention module, and train with max-instance MIL, a mutually guided ranking loss and cross-entropy from video labels. They introduced STRA, 787 road-accident videos from traffic police and video platforms (CCTV and online footage, not dashcam).

- **Strengths.** It is the only weakly supervised VAD method we found whose instances are detector proposals. On STRA it reaches 92.88 video AUC, and its tube branch beats its temporal branch (92.40 against 90.04). It reports localisation against random and upper-bound references (STRA MIoU 7.23 against 1.33 random and 18.62 with perfect selection among its proposals). It shows that a cross-entropy term on the max instance is needed: without it, video AUC collapses to 76.83 because all scores go to zero.
- **Weaknesses.** C3D features, no text, no foundation model, no dashcam data and no STAUC. Its upper bound shows how much proposal quality caps localisation. Whether STRA is public could not be confirmed.
- **Taken into this thesis.** It bounds the novelty claim (object instances under video labels are not new), and supplies two design rules: a cross-entropy term on the top instances, and upper-bound reporting with perfect selection among proposals (the GT-box oracle row, Section 4.5).

#### 2.3.7 STPrompt (ACM MM 2024): fixed top-K selection and CLIP crops

STPrompt adds a spatial branch to a VadCLIP-style detector on frozen CLIP. Each of 7×7 patches gets a motion magnitude, a second-order difference of its features across neighbouring frames; the top K = 12 patches are softmax-weighted by that magnitude (no learned weights) and added to the frame feature. Localisation at test time is separate and training-free: sliding-window crops are encoded by CLIP's global token and matched against hand-written normal captions and LLM-augmented abnormal phrases.

- **Strengths.** It localises in space and time from video labels, improving temporal-IoU over VadCLIP on UCF-Crime (23.90 against 22.05), ShanghaiTech and UBnormal. Its ablation separates the ingredients: learned attention alone does no better than averaging (86.67 against 86.91 UCF-Crime AUC), while selection before pooling helps (87.79 with averaging, 88.08 with attention), so most of the gain comes from shrinking the instance set.
- **Weaknesses.** Its detection gain over VadCLIP is small (88.08 against 88.02 on UCF-Crime). Its selection is a fixed motion ranking, and in dashcam video ego-motion dominates such feature differences; the student's preliminary probes found feature-difference cues unhelpful and the learned tile scorer collapsed to sky and hood rows. Its localisation never uses the trained model's spatial scores and needs an LLM. It covers surveillance data only, has no public code, and the authors found treating every patch as an MIL instance hard to optimise.
- **Taken into this thesis.** Two things. Its lesson that a prior which shrinks the instance set matters more than learned attention: a detector proposes a few agent-shaped instances instead of 12 high-motion tiles. And its use of CLIP's global embedding on crops, which the thesis applies to detected agents at native resolution so that each agent has an embedding in the text space. A fixed top-K ranking of agents by ego-compensated change, in the spirit of STPrompt, is an ablation row that LaGoVAD-AP must beat (row 3, Section 4.5).

#### 2.3.8 "Learning Where and When" and "Localizing to Debias" (2026 preprints): detector-free weak spatial VAD

"Learning Where and When" (Karim et al., arXiv 2606.29498) scores each of 7×7 grid patches of UniFormer segment features, lifts the best 2×2 block inside the top segments of abnormal videos, and labels every patch of every normal video 0; it reaches 40.51 temporal-IoU on UCF-Crime against STPrompt's 23.90 and releases test boxes for XD-Violence and UBnormal. "Localizing to Debias" (Abdulaziz et al., arXiv 2608.12045) runs on VideoMAEv2 tubelets with hard token pruning and a time-reversal motion term, adds per-frame boxes to UCF-Crime, XD-Violence and MSAD, and finds hard selection grounds better than soft (patch AP 25.80 against 18.69).

- **Strengths.** They show that weak labels can localise with modern features, far beyond the older tube method (UCF-Crime MIoU on the box subset 27.92 against WSSTAD's 8.98, across different features and eras). Dense normal-instance negatives and measured context bias (Bias-AUC 0.113 → 0.100, which the authors call modest) are their defences against context shortcuts. Neither uses an LLM.
- **Weaknesses.** Both are preprints on surveillance data with no detector baseline and no dashcam or ego-motion; neither reports STAUC. The student's own MQR experiments, which used the same dense-negative scheme on VideoMAE V2-S tiles of dashcam video, did not learn to point at the agent.
- **Taken into this thesis.** The dense negative term (every candidate of every normal video labelled 0) and the expectation that hard selection among few instances beats soft attention. They also bound the claim: weak spatial VAD with normal-instance negatives is not new; doing it over detector-proposed agents on dashcam video, scored with STAUC, is what remains open.

#### 2.3.9 ORViT (CVPR 2022): object tokens and the random-box control

ORViT inserts object tokens into a video transformer: for each tracked box, RoIAlign on the patch tokens, an MLP and pooling give one token per object per frame, a box-coordinate embedding is added, and a separate stream embeds box trajectories. Boxes come from a COCO Faster R-CNN tracked with SORT.

- **Strengths.** Object tokens lift SomethingElse from 60.2 to 69.7 and Diving48 from 80.0 to 88.0, and the box-only dynamics stream adds +2.1 on top (66.7 to 68.8). Its control is the cleanest test of whether boxes matter: replacing real boxes with full-image, grid or random boxes returns accuracy to about 60.4–60.9, the baseline. Tracked boxes beat per-frame detections (69.7 against 68.2).
- **Weaknesses.** Fully supervised action recognition, fine-tuned backbone, no anomalies and no weak labels.
- **Taken into this thesis.** RoIAlign of backbone tokens per box (here on frozen VideoMAE tokens, slot by slot along the track), a box-trajectory embedding, and the random-box control (row 10, Section 4.5).

#### 2.3.10 Alert-CLIP (CVPR 2026): region-level alignment learned from boxes

Alert-CLIP tunes a CLIP ViT-L backbone with a temporal Transformer through three contrastive alignments: video to caption, region (ROI-aligned from annotated boxes) to region caption, and region against hard-negative captions. It is trained on VAGTA, a re-annotation of UCF-Crime and MSAD with boxes and region captions, and then used as a drop-in backbone.

- **Strengths.** Region-level training moves CLIP's attention from the surrounding context to the collided vehicles (its Fig. 5), and lifts the VadCLIP framework on MSAD from 87.45 AUC / 71.09 AP to 89.63 / 78.24. Boxes are needed only for training; inference is CLIP-only.
- **Weaknesses.** It needs manual boxes, segmentation and region captions (its hard-negative captions are written offline by Qwen-VL), and heavy tuning on an 80 GB GPU. It reports no quantitative traffic result, does not state that its checkpoint is released, and still reads frame-level appearance without motion.
- **Taken into this thesis.** Evidence that region-level evidence improves detection, and a comparison row on MSAD. LaGoVAD-AP aligns agent crops with the anomaly definition through the MIL alignment loss, with neither boxes nor captions as supervision.

#### 2.3.11 YOLOv9, BoT-SORT and ByteTrack: the agent proposals

These are tools rather than VAD methods, but their settings decide what the branch can see, so they are reviewed like the others. YOLOv9-C reaches 53.0 COCO AP (70.2 AP50, AP\_small 36.2) with 25.5M parameters and 102.8 GFLOPs at 640; YOLO11-m and -l reach 51.5 and 53.4 AP at 68.1 and 87.2 GFLOPs. BoT-SORT adds global camera-motion compensation (an affine model from sparse keypoints, with detections masked out) to ByteTrack's two-stage association; on MOT17 validation, compensation adds 0.94 HOTA and 1.62 IDF1. ByteTrack keeps low-score boxes for a second matching step and runs at about 30 FPS end to end.

- **Strengths.** Off-the-shelf, trained on data disjoint from the thesis's datasets, and cheap next to the video encoders. BoT-SORT's affine transform is a free ego-motion estimate (6 numbers per frame) that also yields ego-compensated agent velocities. YOLOv9 is PromptTAD's detector, which keeps the comparison clean.
- **Weaknesses.** COCO has no "rider" class (DoTA has one), no obstacle or debris classes, and no night-specific training; DoTA's obstacle (VO) and out-of-control (OO) anomalies have no participant boxes at all. No paper we found reports any detector's recall on DoTA's anomalous agents. By analogy, in weakly supervised action localisation with COCO person tubelets as MIL instances (Arnab et al.), the detector recalled only 46.9% of actors on UCF101-24 and this capped what MIL could recover. Compensation can fail with few background keypoints, and on static cameras it can only hurt.
- **Taken into this thesis.** YOLOv9-C at native resolution as the default detector (YOLO11 and YOLO-World as ablations), BoT-SORT with compensation and no ReID on dashcam sets, ByteTrack on fixed-camera sets, and a pre-registered recall check before any training (Step 0, Section 4.3).

#### 2.3.12 Methods used as comparison rows

These methods contribute no part of the architecture; they supply the published numbers the thesis is compared against (Section 4.4).

| Method (venue) | Approach | Strength | Weakness | Role in this thesis |
| --- | --- | --- | --- | --- |
| WSAL (IEEE TIP 2021) | High-order context encoding of TSN snippet features plus segment-to-segment variation cues; introduced the TAD dataset | First weakly supervised traffic benchmark; 89.64 AUC\_O / 61.66 AUC\_A on TAD | Hand-crafted augmentations and cues; a fixed 32 segments per video; no spatial output | P1 row on both metrics |
| UMIL (CVPR 2023) | Splits snippets into confident and ambiguous sets by prediction variance and clusters the ambiguous ones, so the classifier stops relying on context shortcuts; X-CLIP-B/32 fine-tuned end to end | Best two-metric TAD result: 92.93 AUC\_O / 65.82 AUC\_A | Needs backbone fine-tuning (90.71 AUC\_O when frozen); clustering can go wrong; sensitive to its trade-off weight on TAD | P1 row; source of the official TAD split |
| π-VAD (CVPR 2025) | Induces five extra modalities (including pose, depth and optical flow) from RGB features during training; RGB-only inference | MSAD 88.68 AUC / 71.26 AP; flow is the most useful modality | Needs five teacher backbones for training; no public code found | P2 row; evidence that motion matters |
| GS-MoE (ICCV 2025) | Gaussian pseudo-labels around score peaks plus one expert per anomaly class, on I3D and UR-DMU features | MSAD 87.72 AUC / 68.26 AP / 69.54 AUC\_A / 76.68 AP\_A | Needs class labels for its experts; 16.02M parameters at about 9.6 FPS; misses an accident that occupies "a very small pixel-region" | P2 row; evidence of spatial dilution |
| WS-DoTA (ECCV 2024 Workshop) | Weakly supervised DoTA with D²-City normal clips; a spatial-aware temporal-regularity transform of CLIP features | The only weakly supervised dashcam study; lifts RTFM from 57.9 to 78.2 AUC | Its own 1,031-video subset; AUC only; no box or STAUC evaluation | Flagged P3 row; evidence that CLIP frames lack motion |
| ISCRTAD (IEEE T-ITS 2025) | YOLOv6 + Deep OC-Sort + visual odometry + depth; agent graphs with an ego node and a scene super-node | DoTA 86.2 AUC; ego-uninvolved average 82.1 | Frame-supervised; about 359M parameters; DADA-2000 overall 74.2, no better than PromptTAD; paper not accessible, numbers via Simple-TAD | Frame-supervised context row; source of the ego-node idea |
| DoTA baselines and Structured Graph Reasoning (IEEE TPAMI 2022; IEEE TMM 2026) | Unsupervised future-object-localisation and scene-graph prediction from tracked boxes; box-painted maps for STAUC | The only published DoTA STAUC rows we recovered (Ensemble 73.0 AUC / 48.5 STAUC); SGR finds interaction scores localise better than per-object scores | Unsupervised; SGR's table values were lost in extraction | STAUC context rows |

### 2.4 Research gaps

Five gaps follow from the review. The thesis addresses each one, and its novelty claim is deliberately narrow.

1. **No motion-aware foundation features in weakly supervised traffic VAD.** LaGoVAD and VadCLIP read CLIP frames; the masked-video evidence of Simple-TAD comes from frame-level supervision. The one weakly supervised dashcam study, WS-DoTA (ECCV 2024 Workshop), added a temporal-regularity transform of CLIP features and lifted RTFM from 57.9 to 78.2 AUC. *Addressed by the motion stream (backbone change).*
2. **No agent-level evidence under weak labels on traffic data.** Object-level traffic methods (PromptTAD, ISCRTAD) train with frame labels, and PromptTAD also with ground-truth anomalous boxes. The only weak-label method with detector instances, WSSTAD (2021), uses C3D on CCTV footage; since 2024, weak spatial methods use grid patches or tubelets on surveillance data. On dashcam video, the student's grid readouts over frozen VideoMAE tokens did not learn (position-controlled gains −0.01 to +0.002). No controlled comparison of agent instances against grid instances under the same weak labels exists. *Addressed by the Agent Prompt branch and the controlled tile row (row 12, Section 4.5).*
3. **Weak supervision gives no spatial evidence on traffic data.** We found no weakly supervised traffic method that produces maps, and no report of STAUC or box-level localisation on DoTA under weak supervision. *Addressed by the painted agent maps, scored with box-level hit rate, pointing game and STAUC.*
4. **DoTA's AUC mostly measures timing.** Computed from DoTA's metadata, a position-only score reaches about 0.847 macro / 0.819 micro AUC on the test list and 0.853 macro on ego-uninvolved clips, and falls to 0.510 under 50-bin position stratification. No traffic paper we found reports this reference. *Addressed by position-only rows, position-stratified AUC, onset-centred AUC and shortcut controls (Section 4.4).*
5. **Traffic results are hard to compare.** Most rows are AUC-only, two TAD test lists circulate, and DoTA test clips share source videos. *Addressed by four split-consistent, multi-metric protocols with source-clustered bootstrap intervals (Section 4.4).*

The closest prior work bounds the novelty claim:

| Closest work | What it does | How LaGoVAD-AP differs |
| --- | --- | --- |
| PromptTAD (IEEE RA-L 2025) | YOLOv9 box prompts, instance and relation modules, 12 relation queries; frame labels and Hungarian-matched ground-truth anomalous boxes; AUC and STAUC on DoTA | Video labels only, with agent-level MIL and dense normal-agent negatives; tracked agents with frozen CLIP-crop and VideoMAE features; language-guided host with agent text alignment; zero-initialised gate. **Main novelty risk: it can be read as "PromptTAD trained with MIL"**, so the weak-label training scheme and the position-controlled evaluation must carry the contribution |
| WSSTAD (2021) | Faster R-CNN tubes as MIL instances under video labels, including CCTV road accidents (STRA) | Frozen video-foundation and CLIP features, text, dashcam video with ego-motion compensation, STAUC and position-controlled evaluation |
| STPrompt (ACM MM 2024) | Fixed motion top-K CLIP patches for detection; training-free, LLM-assisted crop localisation; surveillance | Learned scores over detected agents, maps from the trained scores, traffic data, no LLM; a fixed top-K agent ranking is an ablation row |
| "Learning Where and When" (arXiv 2606.29498) | Per-patch scores on a 7×7 grid, top 2×2 positives, every patch of normal videos labelled 0; surveillance | Agents instead of grid patches, dashcam data, text, STAUC; **second novelty risk: the closest training scheme** |
| "Localizing to Debias" (arXiv 2608.12045) | Weak spatial VAD on VideoMAEv2 tubelets with hard pruning and a time-reversal motion term; UCF-Crime, XD-Violence, MSAD | Detector-proposed agent tokens instead of pruned tubelets, dashcam data, STAUC on DoTA |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with manual boxes and captions | Needs neither boxes nor captions; agent-text alignment from video labels |
| WS-DoTA (ECCV 2024 Workshop) | Weakly supervised dashcam VAD with frame-level CLIP features | Agent-level evidence, spatial evaluation, position control |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that scores detector-proposed agent tokens built from frozen video-foundation and CLIP features inside a language-guided detector, and the first to report STAUC and position-controlled AUC on DoTA under video-level supervision.** Object instances under video labels (WSSTAD), object prompt tokens and relation queries (PromptTAD) and dense normal-instance negatives ("Learning Where and When") are credited as prior art and are not claimed as new; the claim is their combination on frozen foundation features, with tracked, ego-compensated agents, inside a language-guided detector on dashcam video, evaluated against position and box-size priors. Both parts will be re-checked against the 2026 preprints and the ACM MM 2026 and NeurIPS 2026 accepted lists before submission. If the branch fails its gates (Sections 4.3 and 4.4), the thesis reports it as a documented negative result alongside MQR, and the contribution rests on the motion stream, the DoTA position audit with its position-stratified protocol, and the detector-recall table, the first published for DoTA's anomalous agents as far as we found.

## 3. Proposed model

The proposed model, **LaGoVAD-AP**, is LaGoVAD with two changes: a frozen VideoMAE V2-S motion stream (the backbone change) and the Agent Prompt branch (the novel component). Every other part of LaGoVAD stays as released, so any gain can be traced to these two changes.

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

**Clip alignment.** "Centred" means frames t−8 … t+7 of the 10 fps stream, with edge frames repeated at video boundaries, so the clip's temporal centre falls on the frame it scores. A causal window (t−15 … t) would centre the motion evidence about 0.75 s before that frame; under video-level MIL nothing forces the detector to undo such a lag, and the frame pseudo-labels of dynamic video synthesis would be misaligned by the same amount, so the window is fixed to the centred form for every model. The same centred window defines which track boxes an agent's motion feature is pooled over (Section 3.2). A shift test checks the alignment: frame AUC as a function of a test-time offset applied to the score sequence, computed on the DoTA-train pilot subset (Section 4.3), should peak at an offset of zero.

The last layer gives X\_t, 8 temporal × 14 × 14 spatial tokens of dimension D. The motion stream averages all 1,568 tokens and adds the result to the CLIP embedding through a gate that starts closed:

```latex
v_t = \frac{1}{1568} \sum_{j=1}^{1568} X_{t,j}, \qquad x_t = c_t + \tanh(\alpha_m)\, W_m\, \mathrm{LN}(v_t), \qquad \alpha_{m,\text{init}} = 0,\; W_m \in \mathbb{R}^{512 \times D}
```

The fused x\_t replaces c\_t as the RoFormer input. Because α\_m starts at 0, the model starts as CLIP-only LaGoVAD and opens the gate only as far as motion helps. The kNN retrieval of dynamic video synthesis still uses CLIP features. In the student's single-seed preliminary runs this model reaches 0.81 macro / 0.75 micro zero-shot DoTA AUC (ego-involved about 0.86–0.88, ego-uninvolved about 0.68–0.74), below the position-only reference of Section 1.1 overall and on ego-uninvolved clips.

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips and distilled on Kinetics-710; none of the seven traffic datasets used in this thesis is among these sources. The detectors are trained on COCO (YOLO-World on its own grounding corpora), which are not known to include any of the seven datasets; this is checked before use. Simple-TAD's driving-adapted DAPT-VideoMAE-S (BDD100K only) is kept as an encoder ablation.

*Figure (to be drawn): LaGoVAD-AP architecture. The CLIP stream, RoFormer, text co-attention and heads are LaGoVAD as released; the motion stream is the backbone change; the agent branch (detector and tracker, agent tokens, relation module, agent scorer) and its dashed training losses are the new component.*

### 3.2 Novel component: Agent Prompt (AP) branch

The Agent Prompt branch turns each detected and tracked road agent into a token, lets the tokens interact with each other, the scene and an ego pseudo-agent, and scores each agent with its own anomaly probability. One extra loss, an agent-level MIL loss, trains these scores from video labels alone: on abnormal videos, the top agent of a few steps must score high; on normal videos, every agent at every step must score low. A second, light loss aligns agent tokens with the class texts. The branch then feeds an agent-enhanced frame token into LaGoVAD's input through a zero-initialised gate, and its per-agent scores are painted onto the boxes as the evidence map. The name follows PromptTAD: boxes act as prompts, and the agents are what they prompt.

**Why agents instead of tiles.** MQR, the previous component, failed in five observed ways. Each design choice below answers one of them:

1. **Background candidates won.** On CLIP patches the tile scorer put 93% of its peaks on the top and bottom rows, sky and hood, which are present in every dashcam frame. *Answer:* the only candidates are detected traffic agents (plus an ego and a no-agent token, which are never painted into maps), so sky, hood and road texture cannot win.
2. **The evidence was smaller than a tile.** A 7×7 tile covers 32×32 px of the 224×224 input, and 26.2% of DoTA's anomalous boxes are under 16 px on at least one side there (computed). *Answer:* appearance comes from re-encoding each agent's crop at native resolution with CLIP. A 40×30 px agent keeps only about 63 of its 1,200 native pixels after the 224×224 frame resize, but fills about 149×112 px of a 60×60 context crop resized to 224, roughly 19× more image information (arithmetic from these sizes).
3. **Ego-motion dominated change.** VideoMAE token change between overlapping windows pointed at the agent only at chance. *Answer:* motion is pooled inside the agent's own track box, slot by slot, so the pooling window moves with the agent; velocities are compensated with the tracker's camera-motion transform.
4. **The softmax never left the uniform state.** Free queries stayed uniform and attention entropy stayed flat. *Answer:* no attention selects candidates. Each agent has its own sigmoid score with a direct BCE gradient from the first step, and the relation queries only build the frame token, never the map.
5. **Position was the easiest signal.** A position-only predictor beat every learned model on the dev split. *Answer:* the agent scorer has no time index and sees only agent content; training clips are cropped and the anomaly is placed at sequence boundaries (Section 3.3); evaluation is position-stratified (Section 4.4).

The branch cannot fix detector misses or agent-less anomalies; these are tested before training (Step 0) and handled by the no-agent fallback below.

**Agents.** For every frame at 10 fps, a COCO-trained **YOLOv9-C** runs at native resolution (capped at 1280 px on the long side), keeping the six traffic classes person, bicycle, car, motorcycle, bus and truck (COCO ids 0, 1, 2, 3, 5, 7) at a confidence floor of 0.10 after NMS. A person box overlapping a two-wheeler box (IoU or containment above 0.5) is merged into a "rider" agent, matching DoTA's class set. Dashcam sets are tracked with **BoT-SORT** with global camera-motion compensation and no ReID; fixed-camera sets with **ByteTrack**; both with a track buffer of 10–15 frames (1–1.5 s, matching DoTA's own `max_age: 10`). Tracks shorter than 3 frames are dropped. At each LaGoVAD step t, the agent set 𝒜\_t holds the tracked agents present at t, at most K\_max = 16 by track-smoothed confidence (DoTA has at most 10 anomalous boxes per annotated frame and 1.39 on average, computed); padding is masked. Kalman-smoothed boxes are used for kinematics and raw boxes for features. The detector and tracker are frozen tools; their boxes are inputs, never labels.

**Agent tokens.** Each agent i at step t gets four feature groups:

- **Geometry.** The two box corners, normalised by frame size, are Fourier-encoded (random Gaussian frequencies, as in SAM's prompt encoder) to d = 256, summed with a learnable agent-type token, a class embedding and an MLP of log-area, aspect ratio, detector confidence and the distance from the box to the bottom centre of the frame.
- **Appearance.** The box, enlarged 1.5× per side (minimum 32 px in the source frame) and padded to a square, is cropped from the native frame, resized to 224×224 and encoded by the frozen CLIP ViT-B/16 image encoder, giving a\_{t,i} ∈ R^512 in the same space as c\_t and the text embeddings. The crop scale is a token feature.
- **Motion.** For each of VideoMAE's 8 temporal slots τ of the centred clip, the agent's track box at the slot's two frames (their union), enlarged to at least one 16-px token, is pooled from the slot's 14×14 tokens with a 2×2 RoIAlign; slots where the track is absent are masked:

```latex
m_{t,i} = \frac{1}{|\mathcal{T}_{t,i}|} \sum_{\tau \in \mathcal{T}_{t,i}} \frac{1}{4} \sum_{q=1}^{4} \mathrm{RoIAlign}_{2\times 2}\!\big(X_{t,\tau},\, \hat{b}_{t,i,\tau}\big)_q \;\in\; \mathbb{R}^{D}
```

- **Kinematics.** From the track, with A\_{t−1→t} the camera-motion affine estimated by BoT-SORT (identity for fixed cameras) applied to the previous box centre, and h the box height:

```latex
\mathbf{u}_{t,i} = \frac{\mathbf{c}_{t,i} - A_{t-1\to t}\,\mathbf{c}_{t-1,i}}{h_{t,i}\,\Delta t}, \qquad \rho_{t,i} = \frac{d \log h_{t,i}}{dt}, \qquad \kappa_{t,i} = \big[\,\mathbf{u}_{t,i};\ \dot{\mathbf{u}}_{t,i};\ \rho_{t,i};\ \text{track age};\ \min_{j} \mathrm{gap}_{ij};\ \min_j \dot{\mathrm{gap}}_{ij};\ \mathrm{gap}_{i,\text{ego}}\,\big]
```

The velocity u is ego-compensated and measured in box heights per second (perspective-normalised, as TrajVAD found useful); ρ is the scale-change rate, a looming cue; the gaps are normalised centre distances and approach rates to the nearest other agent and to the ego region at the bottom centre. The four groups are summed into one token:

```latex
o_{t,i} = \gamma\big(\tilde{b}^{\,tl}_{t,i}\big) + \gamma\big(\tilde{b}^{\,br}_{t,i}\big) + e_{\text{agent}} + E_{\text{cls}}[c_{t,i}] + \mathrm{MLP}_g(\text{area, aspect, conf, ego-dist}) + W_a\,\mathrm{LN}(a_{t,i}) + W_m\,\mathrm{LN}(m_{t,i}) + \mathrm{MLP}_k(\kappa_{t,i}) \;\in\; \mathbb{R}^{256}
```

**Scene, ego and no-agent tokens.** A scene token o^scene\_t = W\_s[c\_t; v\_t] carries the frame's CLIP and mean motion features. An ego token has a fixed box at the bottom centre of the frame, as Structured Graph Reasoning and ISCRTAD give the ego car a fixed node; its content is the camera-motion affine sequence over the centred clip (the tracker's ego-motion estimate), not the appearance of the hood region, which is where MQR's peaks collapsed. It gives ego-involved collisions, where the ego car has no box, a candidate. When no agent is detected at a step, a learnable no-agent token takes the agent slot, as in PromptTAD. The ego and no-agent tokens are scored and enter the MIL loss, but are never painted into evidence maps, and the share of top-k picks they take is logged.

**Interaction.** One instance self-attention layer runs over the agent, ego and no-agent tokens of a step, with an additive pairwise bias B\_{t,ij} = MLP(r\_{t,ij}) computed from the box difference, IoU and normalised gap of each pair (the geometric relations of Structured Graph Reasoning). A relation module then follows PromptTAD with N\_K = 12 learnable queries but only L\_R = 2 blocks. In each block the queries first gather from the agent tokens, then from the scene token, and the agent tokens update from the refined queries. The enhanced frame token z\_t is the mean of the 12 query outputs (PromptTAD does not state how it collapses its queries; the mean is this thesis's choice):

```latex
\hat{O}_t = \mathrm{SA}\big(O_t;\, B_t\big), \qquad \big(Q_t,\, \hat{O}_t\big) \leftarrow \mathrm{RA}^{(L_R = 2)}\big(Q^{(0)},\, \hat{O}_t,\, o^{\text{scene}}_t\big), \qquad z_t = \frac{1}{N_K} \sum_{n=1}^{N_K} Q_{t,n}
```

All branch layers use d = 256, 4 heads and an FFN ratio of 2, about 4M parameters in all (estimate).

**Fusion.** The enhanced frame token enters LaGoVAD's input through a second gate that starts closed:

```latex
x_t = c_t + \tanh(\alpha_m)\, W_m\, \mathrm{LN}(v_t) + \tanh(\alpha_o)\, W_o\, \mathrm{LN}(z_t), \qquad \alpha_{o,\text{init}} = 0,\; W_o \in \mathbb{R}^{512 \times 256}
```

**Agent scorer.** One small head, shared by all candidates, scores each enhanced agent token:

```latex
p_{t,i} = \sigma\big(h(\hat{o}_{t,i})\big), \qquad h(o) = w_2^{\top}\, \mathrm{GELU}(W_1 o + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{128 \times 256}, \quad (w_2, b_2)_{\text{init}} = 0
```

Every p\_{t,i} starts at 0.5 and receives a direct gradient from the agent-level MIL loss at every step, so nothing has to leave a uniform softmax before anything is learned. Each p\_{t,i} is a signed, anomaly-specific probability for one agent at one step.

**Same starting function.** With α\_o = 0, x\_t is exactly the motion-stream input, so at initialisation LaGoVAD-AP computes the same frame scores as the mean-pooled motion-stream model. The agent-level losses depend only on frozen features and branch parameters, and the alignment loss stops gradients into the text branch, so the branch's own losses never move the host; the host is reached only through the gate, whose gradient at α\_o = 0 is non-zero (tanh′(0) = 1). Whether LaGoVAD-AP ends above the motion-stream model is what the ablation tests.

**Agent-level MIL loss (the one extra loss).** Let 𝒞\_t be the candidates at step t: the detected agents, the ego token, and the no-agent token when no agent is detected. A step's agent score is its highest candidate score. On an abnormal video, the loss lifts the top K candidates of each of the k highest-scoring steps; on a normal video, it lowers every candidate of every valid step:

```latex
S_t = \max_{i \in \mathcal{C}_t} p_{t,i}, \qquad \hat{s}_V = \frac{1}{kK} \sum_{t \in \operatorname{top}k(S)}\; \sum_{i \in \operatorname{top}K(p_t)} p_{t,i}, \qquad \mathcal{L}_{\text{neg}} = \frac{1}{\sum_{t} |\mathcal{C}_t|} \sum_{t=1}^{L} \sum_{i \in \mathcal{C}_t} \operatorname{BCE}(p_{t,i}, 0)
```

```latex
\mathcal{L}_{\text{agent}} = y \cdot \operatorname{BCE}(\hat{s}_V, 1) + (1-y) \cdot \mathcal{L}_{\text{neg}} + \beta\, y\, \overline{p}
```

Top-k is taken over valid steps with k = max(⌊L/16⌋, 1), K = 1 by default (DoTA averages 1.39 anomalous agents per annotated frame, computed), and the last term is a light L1 sparsity penalty with β = 0.01 over all candidates of abnormal videos. The BCE on the top instances is the cross-entropy term WSSTAD found necessary against score collapse. The dense negative term turns every agent of every normal video into a labelled negative, including the BDD100K and Nexar normals full of moving cars, pedestrians and camera shake; it is what teaches the scorer that a car is not anomalous by being a car, near, large or central. Synthetic videos from dynamic video synthesis count as abnormal for the MIL term, and their candidates at steps whose pseudo-label is normal also enter the dense negative term, except within 8 frames of an inserted boundary, where the centred clip spans both sides. Ground-truth frame labels and boxes are never used; PromptTAD's Hungarian-matched instance loss has no counterpart here. A track-level variant, which pools each track's scores over its top steps before the max over tracks, and a label-uncertainty variant for abnormal bags whose positive may not have been detected (after Arnab et al.) are side ablations.

**Agent text alignment.** Each agent's CLIP crop embedding already lives in the text space. The alignment term scores agents against the class texts τ\_c (normal plus the training anomaly classes) through a zero-initialised residual, so it starts as zero-shot CLIP crop–text similarity:

```latex
\ell_{t,i,c} = \frac{\cos\!\big(a_{t,i} + W_\tau\, \hat{o}_{t,i},\; \operatorname{sg}[\tau_c]\big)}{T_a}, \qquad L_{V,c} = \frac{1}{k} \sum_{t \in \operatorname{top}k_c} \max_{i \in \mathcal{A}_t} \ell_{t,i,c}, \qquad \mathcal{L}^{\text{agent}}_{\text{align}} = \operatorname{CE}\big(\operatorname{softmax}_c L_{V,\cdot},\; y^{\text{cls}}_V\big)
```

W\_τ starts at zero, T\_a is learnable and starts at CLIP's 0.01, sg[·] stops the gradient into the text branch, and top-k\_c is taken over steps for class c, as in LaGoVAD's MIL alignment. Only detected agents with crops enter, and videos with no detected agent skip the term. At test time the definition Z replaces the training classes, and the language-guided agent score g\_{t,i} = 1 − softmax\_c(ℓ\_{t,i,·})\_normal is reported for the language analysis. No captions or region texts are needed. The total loss is L\_LaGoVAD + λ·L\_agent + λ\_a·L^agent\_align, with λ ∈ {0.1, 0.5, 1} and λ\_a ∈ {0, 0.1}.

**Evidence map.** Following DoTA's convention for object-based methods and its `bbox_to_score_map`, each detected agent's box at native 1280×720 coordinates is painted with a Gaussian scaled by its score, on DoTA's 256×256 grid:

```latex
E_t(x, y) = \sum_{i \in \mathcal{A}_t} p_{t,i}\, \exp\!\Big( -\tfrac{1}{2}\Big[ \tfrac{(x - x_{0,i})^2}{w_i} + \tfrac{(y - y_{0,i})^2}{h_i} \Big] \Big)\, \mathbb{1}\big[(x, y) \in b_{t,i}\big]
```

The denominators are w and h, not their squares, exactly as in DoTA's code, so painted mass concentrates at the centre of large boxes and is nearly flat on small ones. The ego and no-agent tokens are not painted. The map is the trained agent classifier itself, not an attention proxy. A TAO-style persistence filter (an agent's score is kept only if its box persists at IoU > 0.2 in at least 3 of 5 neighbouring frames) is a label-free post-process tested as a side ablation.

**Why the branch should help on traffic video.** Five pieces of evidence point the same way:

1. **Object evidence helps where the student's model is weakest.** PromptTAD's gains concentrate on ego-uninvolved categories, UK\* and DADA-2000's ego-uninvolved clips (Section 2.3.5); the unsupervised MAMTCF, which fuses box tracks with flow pooled inside each box, reaches 78.6 average AUC on ego-uninvolved DoTA categories against 71.5 for FOL-Ensemble, while ego-involved clips stay level (74.3 against 74.5). These are frame-supervised or unsupervised results, so they set the direction of the effect, not its size.
2. **A small instance set beats learned attention under weak labels.** In STPrompt's ablation, selection supplies most of the gain while attention alone does nothing (Section 2.3.7); in "Localizing to Debias", hard selection grounds better than soft. A detector is a stronger, label-free instance prior than a motion ranking.
3. **Real boxes carry the signal, not the extra tokens.** ORViT's random-box control returns to baseline (Section 2.3.9); box dynamics alone add +2.1 on top of appearance tokens.
4. **Per-agent content features exist that the tile readout could not use.** In the student's training-free probes, CLIP patch change pointed at the anomalous agent 0.29 above chance, nearly as well as a centre prior (+0.30), while VideoMAE change did so only between non-overlapping windows (+0.08 to +0.11, stronger on ego-uninvolved clips). Change maps say where, not when; the agent branch uses where-evidence to choose agents and leaves when to the host's temporal model and the MIL top-k.
5. **Same starting function.** Because LaGoVAD-AP starts as the motion-stream model, any difference after training comes from the agent branch and its losses.

**Known risks.** Detector misses: in dark scenes PromptTAD degrades to its baseline, 12.6% of DoTA clips carry the night flag (computed), and Arnab et al.'s COCO detector recalled only 46.9% of actors on UCF101-24. Agent-less anomalies: the obstacle and out-of-control categories have no participant boxes, and 102 ego-involved test clips (obstacle 16, leave-to-left 45, leave-to-right 41, computed) have no other agent at fault. Salient but normal agents: the large lead car can win the top instance, the analogue of part domination in weakly supervised object detection, which would make the agent map a new largest-box or nearest-box prior; the dense negatives and the box-prior baselines target this. Source shortcut: dense negatives on BDD100K normals can teach the scorer the source, so the Nexar-only control matters more here (Section 3.3).

| Property | Mean pooling (motion stream) | LaGoVAD-AP |
| --- | --- | --- |
| Extra parameters | — | about 4M (estimate; d = 256, FFN ratio 2) |
| Extra extraction compute | — | detector and tracker at 10 fps; up to 16 CLIP crop passes per step; RoIAlign inside the same VideoMAE pass |
| Storage per step (fp16) | 0.75 KB | about 29 KB at K\_max = 16 (1.8 KB per agent: box, 384-d motion, 512-d crop, confidence, class, track id) |
| Spatial output | none | one score per detected agent per step, painted as box Gaussians |
| Extra losses | — | agent-level MIL (top-k steps × top-K agents on abnormal videos, dense negatives on every agent of normal videos); agent text alignment |

The expected signature is specific: gains concentrated on ego-uninvolved DoTA clips, small agent boxes, the "unknown" and turning categories, and TAD's small CCTV anomalies, with no gain where no agent is detected. Raw DoTA AUC may move little or even fall, as DoTA's own AnoPred+Mask lowered AUC (67.5 to 64.8) while raising STAUC (24.4 to 42.1).

### 3.3 Training strategy

Training uses video-level annotations only and LaGoVAD's own schedule, so the baseline, the motion stream and LaGoVAD-AP differ only in architecture, the extra losses and, identically for every row, the position-robust crops.

**Training data.** All training sets come from the seven traffic datasets used in the thesis. Only TAD, MSAD, CCD and Nexar contain both normal and abnormal videos, so only they are used for training; DoTA, DADA-2000 and MM-AU contain accident videos only and serve as test sets.

| Protocol | Training set | Normal / abnormal videos | Anomaly text (hand-written) |
| --- | --- | --- | --- |
| P1, TAD | TAD train, UMIL split | 210 / 190 | TAD's 7 category names and definitions |
| P2, MSAD | MSAD weak-protocol train | 360 / 120 | MSAD's 11 anomaly types |
| P3, dashcam | Nexar labelled set + CCD official train split | Nexar 750 / 750; CCD's train part of 3,000 normal / 1,500 crash clips | Collision, near-miss and car-crash definitions |
| P4, cross-setting | None: reuses the P2 and P3 models | — | Each test set's own definitions |

**Captions.** LaGoVAD's caption contrastive loss needs one description per video. Every model uses the same hand-written caption template per category (for example, "a dashcam video in which the car collides with another vehicle"), so no LLM-written text enters training. Removing this loss costs LaGoVAD 6.3 AUC on TAD zero-shot (89.56 to 83.26), so a no-caption variant is reported as an ablation.

**Shortcut control.** CCD's crash clips and its normal clips (taken from BDD100K) come from different sources, so a model could learn the source instead of the anomaly. Nexar's positive and negative clips both come from Nexar's dashcam network, so P3 is also reported with Nexar-only training. The dense negative term of the agent-level MIL loss makes this control more important: if normal agents are easy to reject because of their source, the scorer can learn the source.

**Position-robust training.** LaGoVAD's dynamic video synthesis already places the abnormal anchor at a random slot, but with probability 0.3 no synthesis happens, the anchor keeps its own precursor–anomaly–aftermath structure, and the RoFormer's rotary encoding keeps relative position learnable. Three cheap changes apply to every model row alike:

1. **Random crops before synthesis.** Normal videos are cropped with probability 0.5 to a random 30–100% of their length at a random offset. Abnormal videos are cropped with probability 0.5 by removing at most 25% of their length from one randomly chosen end. Under video labels a crop of an abnormal video can cut out the anomaly; the 25% bound limits this label noise, which cannot be measured on the training sets, and crop probability 0 is an ablation.
2. **Boundary anchors.** With probability 0.5 the synthetic anchor is placed in the first or last slot instead of a uniform slot, so the anomaly sometimes touches the sequence boundary and the quiet–bump–quiet template is broken.
3. **No time index in the branch.** The agent scorer sees only agent content; temporal position reaches the model only through the host's RoFormer.

**Stages.**

1. **Detect and track.** Run the detector at 10 fps and the tracker on every video once; store all raw tracks (about 2.4 GB for 3M frames, estimate) so that K\_max and feature choices can change without re-detecting.
2. **Cache features.** Extract CLIP frame features with LaGoVAD's script; run the VideoMAE V2-S pass once per centred clip, storing the mean token and, inside the same pass, the slot-wise RoIAlign of every kept track box; encode the agent crops with CLIP at each step; compute kinematics from the tracks. All are stored as memory-mapped fp16 arrays, one cache for every row.
3. **Train all models with one schedule.** Every row (LaGoVAD, motion stream, LaGoVAD-AP, ablations) is trained from the same random initialisation of LaGoVAD's trainable parts, with the same schedule and 3 seeds. LaGoVAD-AP starts with α\_o = 0 and the last layer of the agent scorer and W\_τ at zero.
4. **Warm start as a speed variant.** LaGoVAD-AP can also start from a trained motion-stream checkpoint, since it computes the same function at initialisation. This is reported separately and never mixed into ablation rows.

**Settings.** AdamW, batch 64, learning rate 5e-5 with a cosine schedule, 40 epochs, as in LaGoVAD. The gate α\_o, the temperature T\_a and the relation queries sit in their own parameter group without weight decay. The loss weights λ and λ\_a are chosen once on a held-out 15% of the P3 training videos scored by video-level AUC. The spatial settings, the top-K agents per step (K ∈ {1, 2, 3}), the sparsity weight β ∈ {0, 0.01} and the crop enlargement (1.5× or 2×), cannot be judged by video-level AUC, so they are chosen once by box-level hit rate on the DoTA-train pilot subset of Step 0 (Section 4.3). All choices are then frozen for every protocol. No hyperparameter is chosen on a test list.

**Checkpoint selection.** LaGoVAD's released config selects checkpoints on test scores. The thesis instead reports the last epoch of the fixed schedule for every model, baselines included.

**Compute.** A batch of 64 sequences of 512 steps with 16 agents each is about 0.95 GB of cached agent input in fp16, and each 256-d activation over the 18 tokens per step is about 0.3 GB in bf16 (estimates from these sizes). With activation checkpointing over the relation module, one 24 GB GPU should suffice; if the profiling run in the first week shows otherwise, batch 64 is kept through two accumulated steps of 32. Extraction is the main cost: detection at native resolution takes roughly 14–21 GPU-hours for about 3M frames, BoT-SORT's camera-motion step under 2 hours on 8 CPU processes, and crop re-encoding about 1.7 hours at 375k steps or about 13 hours if every 10 fps frame is a step (all estimates, to be benchmarked on 1,000 DoTA frames before committing).

### 3.4 Inference

At test time the model takes a video and an anomaly definition and returns a score per sampled step and a score per detected agent. No LLM runs at any point.

1. **Detect and track** agents at 10 fps with the fixed detector and tracker.
2. **Sample** one frame in eight (every frame for 10 fps dashcam sets).
3. **Encode** each sampled frame with frozen CLIP, the 16-frame clip centred on it with the frozen VideoMAE V2-S (mean token and per-agent RoIAlign), and each kept agent's crop with frozen CLIP; compute kinematics.
4. **Score agents** with the agent tokens, the instance and relation layers and the agent scorer.
5. **Score frames** with LaGoVAD's RoFormer, text co-attention and heads on x\_t, given the definition text. Each step's score is assigned to the frames it stands for.
6. **Map** each step's evidence by painting the agents' scores onto their boxes on DoTA's 256×256 grid.

**Definitions.** For comparability, every main result uses LaGoVAD's released definitions: TAD is prompted as {"Normal", "Car Accident"}, and DoTA's "Abnormal" expands to the same car-accident texts. Yet 37 of the 60 abnormal videos on LaGoVAD's TAD list (36 of 60 on UMIL's list) are not accidents. A separate analysis therefore prompts each of TAD's 7 categories with its own hand-written definition; this is prompt engineering, not part of the component.

**Score post-processing.** DoTA and DADA-2000 contain only abnormal clips, so their scores are min-max normalised per video before pooling, following LaGoVAD's DoTA protocol (`offline_dota_eval.py`), which the thesis extends to DADA-2000. TAD, MSAD and CCD scores, which come with normal videos, are used raw. The agent scores build the evidence map only; adding the step's top agent score S\_t to the frame score is tested only as an ablation.

**Cost.** The branch's layers are negligible next to the encoders. One VideoMAE clip runs per sampled step, about 57 GFLOPs for ViT-S. Up to 16 CLIP crop passes per step add more than the VideoMAE pass (about 17.6 GFLOPs each, a standard figure not taken from the notes), and the detector at native resolution runs on every 10 fps frame, so LaGoVAD-AP is several times costlier per step than the motion-stream model (estimate; measured FPS is reported in Section 4.5). The model is offline: the temporal window looks both ways.

## 4. Experimental plan

The plan trains with video-level labels only, compares against published rows on identical test lists, adds every second metric by re-scoring released checkpoints, and places a position-only reference next to every DoTA number.

### 4.1 Dataset preparation

The thesis uses seven traffic datasets. TAD (fixed camera) carries the main comparison and MSAD (multi-scene surveillance with a traffic-accident slice) a second one. CCD and Nexar train the dashcam model, DoTA and DADA-2000 test it zero-shot, and DoTA's boxes test the agent scores.

| Dataset | View | Size and split | Labels | Role | Source |
| --- | --- | --- | --- | --- | --- |
| TAD | Fixed CCTV, some first-person | 500 videos; UMIL split 400 train (210 normal / 190 abnormal), 100 test (40 / 60); 7 categories | Video labels (train), anomaly interval (test) | P1 train and test; P4 zero-shot test | Frames and split files via the UMIL repository; native fps unknown |
| MSAD | Fixed, 14 scenes | 720 videos at 30 fps; weak protocol 360 normal + 120 abnormal train, 120 + 120 test (20 traffic accidents) | Video labels (train), frame labels (test) | P2 train and test; traffic slice; P4 zero-shot test | MSAD project page (videos gated); I3D features public |
| CCD | Dashcam | 1,500 crash + 3,000 normal clips (normal clips from BDD100K), 50 frames at 10 fps, official split; accident onset randomly placed in the last 2 s | Video labels; accident onset | P3 train and in-domain test | GitHub (Cogito2012/CarCrashDataset) |
| Nexar | Dashcam | 1,500 labelled clips (750 positive + 750 normal), about 40 s each; the 1,344-clip challenge test set is scored on Kaggle | Video labels; event and alert times | P3 training | Hugging Face |
| DoTA | Dashcam | 4,677 clips (3,275 train / 1,402 test), 10 fps, abnormal only; test clips from 179 source videos (computed) | Anomaly window; 9 categories × ego-involved / ego-uninvolved; anomalous-agent boxes with track IDs on 122,447 of 156,728 anomalous frames (170,739 boxes in 7 classes, computed); night flag on 587 clips (computed) | P3 zero-shot test; STAUC and box-level hit; Step 0 pilot (train split only) | GitHub (MoonBlvd/Detection-of-Traffic-Anomaly) |
| DADA-2000 | Dashcam | 2,000 accident clips, 658,476 frames at 30 fps (Nexar counts 1,962 clips / 649K frames) | Accident intervals; driver gaze, no boxes | P3 zero-shot test | Simple-TAD's split files |
| MM-AU | Dashcam | 11,727 accident videos pooled from CCD, A3D, DoTA and DADA-2000 | Accident start, collision, end | Optional zero-shot test, after removing videos shared with CCD, DoTA and DADA-2000 | MM-AU release (CVPR 2024) |

DADA-2000 has gaze maps, not boxes, so STAUC and box-level metrics run on DoTA only; Structured Graph Reasoning's re-annotation of anomalous-object boxes on 1,905 DADA videos is added as a spatial test if it is released.

**Weights to download** (all public): CLIP ViT-B/16 and VideoMAE V2-S (`vit_s_k710_dl_from_giant`); YOLOv9-C (official repository), YOLO11-m/l and YOLO-World-L for the detector ablation; BoT-SORT and ByteTrack implementations; LaGoVAD's `best.ckpt` and `pred_results.pkl` (Apache-2.0), trained on PreVAD and used only as a reference row; VadCLIP, PEL4VAD and DSANet checkpoints trained on UCF-Crime and XD-Violence, for zero-shot re-scoring; RTFM and MGFN checkpoints trained on MSAD; MOVAD and Simple-TAD DoTA and DADA-2000 checkpoints as frame-supervised context; and Simple-TAD's DAPT-VideoMAE-S (BDD100K only, CC BY-NC 4.0) for the encoder ablation. The YOLO and YOLO-World licences (GPL-3.0 / AGPL-3.0) allow academic use. The links come from the official READMEs; none has been test-downloaded yet, and that check is the first task. These checkpoints were trained by their authors, so re-scoring them needs only their weights, not their training data.

### 4.2 Dataset processing

Every dataset goes through the same pipeline, so that one feature cache serves the baseline, the motion stream, LaGoVAD-AP and all ablations.

1. **Decode and resample.** The CLIP stream keeps LaGoVAD's convention: one frame in eight, or every frame for the 10 fps sets (DoTA, CCD). The VideoMAE stream, the detector and the tracker use every video at 10 fps; at 30 fps a raw 16-frame clip would span only 0.53 s instead of 1.6 s. TAD's native frame rate is unknown, so its clips take 16 consecutive released frames and their duration is reported as unknown.
2. **Detect and track.** Run YOLOv9-C at native resolution (long side capped at 1280) on every 10 fps frame with the six traffic classes and a 0.10 confidence floor; merge riders; track with BoT-SORT plus camera-motion compensation (dashcam) or ByteTrack (fixed camera); drop tracks under 3 frames; store raw tracks and the per-frame camera affine.
3. **Cut clips.** For each CLIP step at time t₀, take the 16 frames centred on t₀ (frames t₀−8 … t₀+7 of the 10 fps stream), repeating edge frames at video boundaries. The shift test confirms the alignment before training. Resize to 224×224 without cropping, as for the CLIP stream.
4. **Extract CLIP features** with LaGoVAD's `extract_feat_clip.py` for all seven datasets, so that features match the released checkpoint.
5. **Extract VideoMAE V2-S features.** Keep the mean of the last-layer tokens per clip (the motion stream) and, in the same pass, the slot-wise 2×2 RoIAlign of each kept agent's track box, averaged over the slots where the track exists (384-d per agent). The full 8×14×14 tokens are not stored. The existing 4×7×7 grid cache from the MQR experiments is kept for the DoTA pilot subset, the DoTA test clips and P1, where it serves the training-free change-map row and the controlled tile row.
6. **Encode agent crops** at each step with frozen CLIP (1.5× enlarged, minimum 32 px, padded square, 224×224), and compute kinematics from the smoothed tracks and camera affines.
7. **Build labels.** Training uses video-level labels, category indices and the caption templates: TAD from UMIL's `TAD_train.txt`, MSAD from its weak split, CCD from its official split, Nexar from its positive / negative labels (event and alert times are kept for analysis only). Test labels are frame-level: TAD intervals, MSAD frame labels, CCD accident onsets, DoTA anomaly windows and DADA-2000 intervals. Each step's score is copied to the frames it stands for.
8. **Prepare DoTA's spatial labels.** For each clip keep `ego_involve`, `night`, the category, and the per-frame boxes and track IDs of anomalous agents. Bin clips by the median, over the anomaly window, of the largest box's share of the frame: under 0.5%, 0.5–2%, 2–8%, over 8%; also keep PromptTAD's small-object subset (anomalous boxes under 2,267 pixels). Map DoTA's classes to the detector's (rider ↔ merged rider or person). The obstacle (VO) and out-of-control (OO) categories have no participant boxes and are handled by the two STAUC conventions (Section 4.4).
9. **Fix test lists.** LaGoVAD's lists (DoTA 1,402 clips; its 100-video TAD list) for zero-shot AUC comparability; DoTA's official 1,305-video list for STAUC and box-level metrics; UMIL's TAD test list for P1 and as a second zero-shot column; MSAD's weak-protocol test set; CCD's official test split; Simple-TAD's DADA-2000 test split.
10. **Hold out validation data.** 15% of the P3 training videos, stratified by label and source, for the one-time hyperparameter choice; and a 500-clip DoTA-train pilot subset drawn by source video, for Step 0 and the spatial settings.

**Storage of the agent cache** (GB, fp16; estimates scaled from the step counts behind the previous proposal's token-cache table):

| Dataset | Agent cache (K\_max = 16, about 29 KB per step) | Mean motion vector only | 4×7×7 grid (kept from MQR, selected sets only) |
| --- | --- | --- | --- |
| TAD (all 500 videos) | 1.9 | 0.05 | 10.0 |
| CCD (one step per frame) | 6.6 | 0.17 | — |
| Nexar (1.70M frames) | 6.2 | 0.16 | — |
| DoTA, all clips (one step per frame) | 21.2 | 0.56 | about 37 for the test list and pilot subset |
| DADA-2000 | 2.4 | 0.06 | — |

DoTA and CCD use one step per 10 fps frame; the other sets one step per eight frames. Only DoTA's 1,402 test clips and the 500-clip pilot subset are needed, roughly a third of its figure. TAD's quoted duration ("about 25 h") conflicts with its frame count (537,244 frames), so its figure may be off by up to about 5×. MSAD and MM-AU are measured at extraction.

**Leakage checks before any training.**

- Seventy-six of LaGoVAD's 100 TAD test videos are in UMIL's TAD training list, so a model trained on UMIL's split is never scored on LaGoVAD's TAD list.
- MM-AU pools CCD, A3D, DoTA and DADA-2000 videos. It is never used for training, and as a test set only after removing videos shared with CCD, DoTA and DADA-2000.
- Several datasets were collected from online videos, so a near-duplicate check with CLIP-feature similarity runs between every training set (TAD, MSAD, CCD, Nexar) and every test set.
- DoTA's test clips come from 179 source videos, 177 of which also contribute training clips (computed). DoTA is never trained on, so this does not affect zero-shot results, but the Step 0 pilot subset is drawn by source video and every DoTA confidence interval resamples source videos.
- LaGoVAD's published zero-shot numbers are reproduced from `pred_results.pkl` before anything is trained, and the position-only numbers of Section 1.1 are recomputed from the student's own copy of DoTA's metadata.

### 4.3 Baseline running

Baselines run in three groups, preceded by one pre-registered go/no-go diagnostic that checks the branch's premise before any training.

**Step 0: go/no-go diagnostic (inference only, under one GPU-day, on the 500-clip DoTA-train pilot subset drawn by source video).** It asks whether the detector finds the anomalous agents and whether agent-level content ranks them above simple priors. The 1,402 test clips are never used for design decisions; the full test-list recall table is computed later and reported only as a description.

- **D0. Detector and tracker recall.** On every frame of each anomaly window, the share of DoTA's anomalous-agent boxes matched by a kept detection at IoU ≥ 0.5 and ≥ 0.3 (DoTA's boxes for partly visible agents can be loose), per frame and per track (a track counts as found if matched in at least 50% of its frames). Broken down by ego-involved / ego-uninvolved, box-size bin (including the 2,267-pixel cut), night flag and category; VO, OO and ego clips without another agent at fault are reported separately.
- **D1. Candidate load and selection ceiling.** Mean kept agents per step, the share of found anomalous agents that survive the K\_max = 16 cut, and the chance hit rate 1/K per frame. A perfect selector among kept agents gives the ceiling of the box-level hit rate.
- **D2. Training-free per-agent ranking.** The rank of the anomalous agent among kept agents under five label-free scores: ego-compensated VideoMAE RoI change between non-overlapping windows; CLIP crop change; zero-shot CLIP crop–definition similarity (the alignment term's starting point); the looming and approach-rate kinematics; and, as priors, the agent nearest the frame centre, the largest agent and the lowest agent (closest to the ego car). For context, the student's map-level probes found change maps 0.08–0.11 above chance, CLIP patch change +0.29 and a centre prior +0.30.
- **D3. Training-free change-map pooling.** The student's fallback idea: VideoMAE token change between non-overlapping windows on the 4×7×7 grid, warped by the camera affine to compensate ego-motion, with the top and bottom rows masked (where 93% of the CLIP scorer's peaks fell). Its pointing game and its pooled readout become ladder row 2.
- **D4. Alignment and position.** The shift test of Section 3.1 on the motion-stream model; the position-only score, raw and 50-bin stratified, on the pilot subset; and the reversal control on the motion-stream model.

**Pre-registered decision rule** (thresholds fixed in this proposal, before any look at the data):

- **GO** if track-level recall at IoU ≥ 0.3 (VO and OO excluded) is at least 0.80 overall and at least 0.70 on ego-uninvolved clips and on the smallest box bin, and at least one training-free per-agent score from D2 ranks the anomalous agent above the centre-nearest-agent prior (paired bootstrap over source videos, 95% CI excluding zero).
- **NARROW** if overall recall is between 0.60 and 0.80, or a stratum misses its 0.70 bar: training proceeds, and every stratum whose recall is below 0.50 (for example small boxes or night) is dropped from H1 and H2 before training. If recall passes but no training-free score beats the centre-nearest prior, training proceeds with H2 flagged as resting entirely on what the scorer learns.
- **NO-GO** if overall track-level recall is below 0.60: the agent branch is not trained as the main component, and the thesis falls back to the motion stream, the position audit and the documented negative results.

**Group A: reproduce LaGoVAD and set the references.**

| Row | Model | Training | Purpose |
| --- | --- | --- | --- |
| B0 | LaGoVAD, released checkpoint | PreVAD (by its authors) | Reproduce the published zero-shot numbers from `pred_results.pkl` (TAD 89.56, DoTA 62.60, MSAD 90.41), then re-score on UMIL's TAD list, DADA-2000, CCD and the MSAD traffic slice with AUC, AP and AUC\_A, plus position-stratified DoTA AUC |
| B1 | LaGoVAD, retrained | Each protocol's training set (P1, P2, P3), 3 seeds | Fair baseline with exactly the thesis data, features and schedule |
| B2 | LaGoVAD + motion stream (VideoMAE V2-S, mean pooling) | Same, 3 seeds | The preliminary model; LaGoVAD-AP's starting point and primary contrast |
| B3 | Position-only score | None (fixed curve peaked at 55% of the clip, and a 20-bin prior from DoTA train) | Reference row next to every DoTA and DADA-2000 number; a "later is more anomalous" score for CCD and onset-centred AUC |

**Group B: published methods with released weights, re-scored for a second metric.** VadCLIP and DSANet (UCF-Crime and XD-Violence checkpoints, on the same CLIP ViT-B/16 features); PEL4VAD (UCF-Crime and XD-Violence checkpoints, with I3D features extracted for TAD, DoTA and DADA-2000); RTFM and MGFN (MSAD checkpoints, on MSAD's public I3D features, for the traffic slice); MOVAD and Simple-TAD (DoTA and DADA-2000, frame-supervised context only). Each runs on the same test lists and is scored with AUC, AP and, where defined, AUC\_A, and on DoTA also with position-stratified AUC.

**Group C: a same-feature competitor retrained in-domain.** VadCLIP's public code is retrained on P1, P2 and P3 with the thesis's CLIP features, giving a second weakly supervised CLIP row in every protocol. Methods without released weights are quoted from their papers only: UMIL and WSAL on TAD; π-VAD, GS-MoE and Alert-CLIP on MSAD; MultiDomain and WS-DoTA; and the frame-supervised TTHF, PromptTAD and ISCRTAD. PromptTAD has no public code we could find, so no weakly supervised re-run of it is possible; ladder rows 4–8 stand in for "PromptTAD-style agent tokens under MIL" on the thesis's features.

### 4.4 Evaluation protocols

Four protocols cover in-domain and zero-shot use on fixed-camera and dashcam video, using only the seven datasets above. P1 meets the comparison rule (at least two published methods on at least two metrics, same fixed-camera traffic data) without any recomputation. P2 adds four-metric rows on MSAD. P3 and P4 meet the rule on DoTA, TAD and MSAD once released checkpoints are re-scored for the second metric.

**Metrics.**

- **AUC\_O / AP:** frame-level ROC AUC and average precision, pooled over all test frames.
- **AUC\_A / AP\_A:** the same, over abnormal test videos only; they measure localisation inside abnormal videos.
- **DoTA and DADA-2000 AUC (secondary):** after per-video min-max normalisation (LaGoVAD's DoTA protocol), reported macro (clip-averaged) and micro (pooled), overall and for ego-involved and ego-uninvolved clips, always next to the position-only row B3.
- **Position-stratified AUC (primary for detection):** each frame is assigned to one of 50 bins of relative position in its clip; AUC is computed within each bin over pairs of a normal and an anomalous frame (from different clips) and averaged over bins, with scores min-max normalised per video as in the standard protocol. Computed from DoTA's metadata, the position-only score falls from about 0.85 to 0.585, 0.545 and 0.510 at 10, 20 and 50 bins, so 50 bins removes almost all position signal while keeping about 2,850 frames per bin on the full test list (about 1,200 on its ego-uninvolved clips, estimate).
- **Onset-centred AUC (secondary):** the last W precursor frames against the first W anomalous frames of each clip (W = 16; 1,311 test clips have at least 16 precursor frames, computed), reported against a monotone "later is more anomalous" reference.
- **Shortcut controls:** time reversal (score the reversed video and un-reverse the scores), shuffling of 1–2 s feature blocks, and padding with normal footage from the same source. A content-driven model should lose AUC when shuffled and keep its peak on the anomaly when padded; a position-driven model keeps raw AUC under reversal and moves its peak when padded.
- **Raw-score AUC on mixed sets:** on CCD and Nexar-style data with normal videos, frame AUC on raw scores, never min-max normalised, reported against the late-in-clip reference for CCD, whose onsets all fall in the last 2 s.
- **Box-level hit rate (primary for space):** on anomalous frames of DoTA's 1,305-video list, whether the top-scored detected agent overlaps an anomalous-agent box at IoU ≥ 0.3 (and ≥ 0.5), reported unconditionally and conditional on the anomalous agent being detected (D0). Because it is computed on positive frames only, the temporal confound drops out.
- **Pointing game and mass-in-box:** the share of anomalous frames whose painted-map peak falls inside an anomalous box, and the share of map mass inside the boxes.
- **STAUC:** DoTA's spatio-temporal AUC. For each true-positive frame, TARR is the share of the top-K map pixels inside the union of anomalous-agent boxes, with K the number of box pixels on the 256×256 grid; STAUC is the area under the TARR-weighted true-positive rate against the false-positive rate, so STAUC ≤ AUC. The frame score always comes from the full model, not from the mean of agent scores as the code's object mode does, and the painted map is passed to `get_tarr` as a map. Two conventions are reported: **as released**, which keeps the code's rules (positive frames without ground-truth boxes, such as VO, OO and agents that have left the view, receive TARR = 1, and so, as in object mode, do frames where the model detects no agent), comparable with published rows; and **strict**, where frames without ground-truth boxes are excluded and frames with no painted agent receive TARR = 0. STAUC/AUC is reported as a localisation-quality ratio (0.66 for DoTA's Ensemble). Because STAUC's temporal part inherits the position bias, the box-level metrics carry the spatial claim and STAUC is secondary.
- **Reference rows for every spatial metric:** a map uniform over detected agents, the centre-nearest, largest and lowest agent, the training-free change ranking and change map (D2, D3), random boxes, and a pixel centre prior for the pointing game. Attention entropy of the relation queries is logged as a diagnostic, not used as a criterion.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P1 TAD in-domain (fixed camera) | TAD train → TAD test, UMIL split | AUC\_O, AUC\_A, AP | UMIL and WSAL on both metrics; AUC\_O-only rows; VADor flagged; LaGoVAD and VadCLIP retrained |
| P2 MSAD in-domain (multi-scene surveillance) | MSAD weak train → MSAD test; traffic slice and non-traffic scenes scored separately | AUC, AP, AUC\_A, AP\_A | π-VAD and GS-MoE (four metrics) and Alert-CLIP (two), all on full MSAD; RTFM, MGFN, UR-DMU on the traffic slice |
| P3 dashcam | Nexar + CCD train → CCD test (in-domain); DoTA and DADA-2000 (zero-shot) | CCD raw-score AUC, AP; DoTA position-stratified AUC (primary), min-max AUC macro / micro, onset-centred AUC, ego-involved / ego-uninvolved; box-level hit, pointing game, STAUC (both conventions); DADA-2000 position-stratified and min-max AUC | Position-only row B3 everywhere; DoTA: LaGoVAD, VadCLIP, PEL (published AUC, AP and stratified AUC re-scored), WS-DoTA flagged; DoTA STAUC: DoTA's baselines; CCD and DADA-2000: released checkpoints re-scored; frame-supervised rows as context |
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

**Published zero-shot rows (P3, P4), AUC.** From LaGoVAD's Table 2; AP and position-stratified AUC are re-scored for every row with released weights. The position-only row is computed from DoTA's metadata on the same 1,402-clip list after per-video min-max (to be re-verified); it uses no video.

| Method | Training set | TAD (LaGoVAD list) | DoTA | MSAD |
| --- | --- | --- | --- | --- |
| Position-only (B3, computed) | None (DoTA train position prior, or a fixed curve) | — | 84.7 macro / 81.9 micro | — |
| MultiDomain | Multiple | 79.21 |  |  |
| PEL (I3D) | UCF-Crime | 86.27 | 53.05 | 79.82 |
| PEL (I3D) | XD-Violence | 43.02 | 44.97 | 68.25 |
| VadCLIP | UCF-Crime | 74.46 | 50.93 | 88.09 |
| VadCLIP | XD-Violence | 83.56 | 49.00 | 88.48 |
| VadCLIP | PreVAD | 85.96 | 50.59 | 89.79 |
| LaGoVAD | PreVAD | 89.56 | 62.60 | 90.41 |

Two more cross-dataset results add AUC context. MultiDomain's single-domain I3D models tested on TAD (split not stated) reach 73.75 from UCF-Crime, 71.10 from XD-Violence, 73.80 from LAD, 68.16 from UBIF and 48.57 from ShanghaiTech. MSAD's authors report UCF-Crime → MSAD AUC / AP of 71.9 / 47.4 (RTFM), 61.8 / 31.2 (MGFN) and 74.3 / 53.4 (UR-DMU).

**Published DoTA STAUC rows (unsupervised, DoTA's Table 3; AUC / STAUC).** ConvAE gray 64.3 / 7.4 and flow 66.3 / 7.9; ConvLSTMAE gray 53.8 / 12.7 and flow 62.5 / 12.2; AnoPred 67.5 / 24.4; AnoPred+Mask 64.8 / 42.1; FOL (TAD) 69.2 / 43.3; TAD+ML 69.7 / 43.7; Ensemble 73.0 / 48.5. PromptTAD and Structured Graph Reasoning report STAUC, but their table values were not recoverable from the available text and are not quoted. These rows use the as-released convention and their own frame-score protocol, so the thesis's rows are compared with them under that convention only.

Frame-supervised rows trained on DoTA are context only and are never ranked against weakly supervised rows. On DoTA: MOVAD 82.2, PromptTAD 83.9, TTHF 84.7, ISCRTAD 86.2, Simple-TAD DAPT-VideoMAE-B 87.9 and -L 88.4. On DoTA → DADA-2000: MOVAD 75.2, PromptTAD 74.6, Simple-TAD DAPT-VideoMAE-S 84.3. WS-DoTA's best weakly supervised row, RTFM with its temporal-regularity transform, reaches 78.2 AUC on DoTA when trained in-domain with D²-City normal clips, on its own 1,031-video subset. These published DoTA numbers use pooled frame AUC under varying normalisation, which the position-only row (81.9 micro) puts in perspective.

**Reporting rules.**

- TAD is reported on two separate columns, LaGoVAD's list and UMIL's list, and the two are never mixed.
- Every thesis number is a mean ± standard deviation over 3 seeds.
- Subsets (ego-involved / ego-uninvolved, box-size bins, night, categories, traffic slices) carry 95% bootstrap confidence intervals over videos; on DoTA the bootstrap resamples the 179 source videos and keeps all their clips, because clips of one source are correlated.
- LaGoVAD-AP against the motion-stream model is tested with a paired bootstrap of the per-draw difference, with each model's frame scores averaged over seeds.
- The two primary endpoints (H1 and H2) are Holm-corrected; every other comparison is labelled secondary or exploratory.
- LaGoVAD-AP against the motion-stream model is also tested for equivalence (two one-sided tests, margin ±0.01 AUC, the size of the MQR effects), so that a null result is reported as "equivalent" or "inconclusive" rather than as no difference.
- All models use the same test lists, centred clips, frame-to-score mapping, normalisation and detector outputs.

**Decision gates for the agent branch.** The gates are fixed before any test-list run. Step 0 decides whether the branch is trained (Section 4.3). Pilots then run with one seed and are judged only on the held-out P3 videos and the DoTA-train pilot subset; a variant moves to 3-seed confirmation only if it beats the centre-nearest and largest-agent priors on box-level hit rate, shows no validation drop beyond the seed spread of the motion-stream model, and is not below it on position-stratified AUC. LaGoVAD-AP counts as supported if, over 3 seeds on the test lists, H1 or H2 holds after Holm correction with no in-domain regression. Otherwise it is reported as a negative result with its diagnostics, next to MQR.

### 4.5 Ablation studies

The ablation ladder adds one ingredient per row on the same features, data, schedule and position-robust crops, so each step isolates one design choice. Rows report P1 (AUC\_O, AUC\_A), P2 (AUC, AP), P3 (CCD raw AUC; DoTA position-stratified AUC by ego-involved / ego-uninvolved clips, min-max AUC, box-level hit and STAUC where agent scores exist; DADA-2000 AUC), and P4 on TAD's two lists.

| Row | Variant | Isolates | Expected if the premise holds |
| --- | --- | --- | --- |
| 0 | LaGoVAD, CLIP only | Baseline | Reference |
| 1 | + motion stream, mean pooling | Backbone change | Gain overall, mostly ego-involved; ego-uninvolved still below the position-only row |
| 2 | Row 1 + training-free change-map pooling: ego-compensated, border-masked VideoMAE change between non-overlapping windows weights the grid readout | The student's fallback: untrained "where" from motion | Map above uniform but at or below the centre prior; little detection gain |
| 3 | Row 1 + fixed top-K agents: kept agents ranked by ego-compensated change, top 4 weighted by a softmax of the change and fed through the gate; no agent scorer or agent loss | A fixed ranking over agents, in the spirit of STPrompt | Small gain over row 2; hit rate above row 2's map |
| 4 | Row 1 + agent tokens with geometry and kinematics only; agent MIL with dense negatives; z\_t as the score-weighted mean of agent tokens | Learned scoring of box tracks alone | Above row 3 on ego-uninvolved clips; hit rate near the largest-agent prior |
| 5 | Row 4 + CLIP crop appearance | Native-resolution agent content | Largest step on small-box bins |
| 6 | Row 5 + VideoMAE RoI motion along the track | Agent motion | Gain on motion-defined categories (turning, lateral, leaving the road) |
| 7 | Row 6 + instance and relation module (pairwise bias, 12 queries, scene and ego tokens) | Agent interaction | Gain on multi-agent categories and on STAUC |
| 8 | Row 7 + agent text alignment (**full LaGoVAD-AP**) | Language at agent level | Detection at least equal; higher sensitivity to the definition |
| 9 | Row 8 with normal videos as bag negatives only (top agents of top steps pushed to 0) | Dense normal-agent negatives | Below row 8 on hit rate; top agent drifts towards the largest or nearest agent |
| 10 | Row 8 with random boxes matched in count and size distribution to the detector's | Whether real agents matter (ORViT control) | Back to row 1 |
| 11 | Row 8 with DoTA's ground-truth anomalous boxes added to the candidates at test (oracle, analysis only) | Cost of detector misses | Upper bound; never ranked as a method |
| 12 | MQR tile scorer on the same schedule (documented negative result) | Tiles against agents, controlled | Below row 8; reproduces the MQR result |

Rows 2 and 3 test whether untrained motion evidence, pooled over tiles or over agents, already gives the gain; rows 4–8 build the branch one feature group at a time; rows 9–12 test the loss, the boxes, the detector ceiling and the instance type. Row 12 is, to our knowledge, the first controlled comparison of agent and grid instances under the same weak labels and frozen features.

**Side ablations within the component.**

- **Detector:** YOLOv9-C (default) against YOLO11-m and -l, and YOLO-World-L with a fixed offline traffic vocabulary that adds "debris", "fallen object" and "animal" for agent-less categories; never conditioned on the test-time definition. Native resolution against 640 px input.
- **Tracker:** BoT-SORT with camera-motion compensation (default, dashcam) against ByteTrack on dashcam; tracker off, with PromptTAD's last-frame boxes and no kinematics or per-slot motion.
- **Agent count:** K\_max ∈ {8, 16}.
- **Content path:** slot-wise RoIAlign (default) against PromptTAD-style cross-attention from agent tokens to the 14×14 VideoMAE grid with Fourier position encodings added to the keys; CLIP crop re-encoding against RoIAlign on ClearCLIP-corrected CLIP patches; crop enlargement 1.5× against 2×.
- **Special tokens:** with and without the ego token; with and without the no-agent token.
- **Loss form:** λ ∈ {0.1, 0.5, 1}; K ∈ {1, 2, 3}; β ∈ {0, 0.01}; track-level against step-level bags; with and without the label-uncertainty term; with and without synthetic-video negatives; λ\_a ∈ {0, 0.1}.
- **Position-robust training:** crops and boundary anchors on (default) against off, for row 1 and row 8.
- **Gate:** α\_o,init = 0 (default) against 0.25.
- **Maps:** with and without the persistence filter; an ensemble map averaging the painted agents with the row-2 change map, as DoTA's Ensemble averages two maps, to cover frames without agents.
- **Score fusion:** adding the step's top agent score S\_t to the frame score at inference.
- **Encoder:** VideoMAE V2-S (default) against VideoMAE-S (Kinetics-400) and DAPT-VideoMAE-S (BDD100K only), to show the branch's gain does not depend on one encoder.

**Analyses.**

1. **Where the gain lands.** DoTA ego-involved vs ego-uninvolved clips crossed with clip-level box-size bins, including PromptTAD's small-object subset (334 test clips); night against day; the categories VO, OO, UK and UK\*; TAD AUC for Accident, PedestrianOnRoad and the remaining categories pooled; the MSAD traffic slice against MSAD's non-traffic scenes; Nexar-only against Nexar + CCD training.
2. **Spatial evidence.** Box-level hit rate, pointing game, mass-in-box and STAUC in both conventions for rows 2–12 against the box and pixel priors, the full test-list detector-recall table, and maps of successes and failures.
3. **Selection behaviour.** Which agents win on which categories; how often the top agent is the largest, the nearest to the centre or the lowest; the share of top-k picks taken by the ego and no-agent tokens; the correlation of agent scores with box size and confidence.
4. **Shortcut controls.** Reversal, shuffle and padding results for rows 0, 1 and 8, and the raw-minus-stratified AUC gap of every row.
5. **Language.** For row 8, how much the language-guided agent score g changes when the definition includes or excludes accidents, and drift@5 on MSAD.
6. **TAD definitions.** Main results with LaGoVAD's two-class TAD prompt, against each of TAD's 7 categories prompted with its own hand-written definition.
7. **Cost.** Parameters, storage, extraction time and measured frames per second for rows 0, 1 and 8, with and without the detector.

**Budget.** Rows 0, 1 and 4–8 run on P1, P2 and P3 with 3 seeds; P4 reuses those models. Rows 2, 3 and 9–12 and the side ablations run on P1 and P3 only, first with one seed, then with three for the variants that are kept. Detection, tracking and crop encoding are one-time extraction costs (Section 3.3); a one-epoch timing run in the first week converts this plan into GPU-hours, and if the total exceeds the budget, the ladder is cut to rows 0, 1, 3, 4, 8, 9 and 10.

## References

**Baseline, backbone and component**

- Liu et al., [Language-Guided Open-World Video Anomaly Detection under Weak Supervision (LaGoVAD)](https://arxiv.org/abs/2503.13160), ICLR 2026; [code, checkpoint, PreVAD annotations](https://github.com/Kamino666/LaGoVAD-PreVAD)
- Qiu, Yang and Gong, [PromptTAD: Object-Prompt Enhanced Traffic Anomaly Detection](https://doi.org/10.1109/LRA.2025.3572428), IEEE RA-L 10(7):7174–7181, 2025 (tables not recoverable from the available text)
- Orlova et al., [Simplifying Traffic Anomaly Detection with Video Foundation Models (Simple-TAD)](https://arxiv.org/abs/2507.09338), ICCV 2025 Workshops (2COOOL); [supplement](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/supplemental/Orlova_Simplifying_Traffic_Anomaly_ICCVW_2025_supplemental.pdf); [code and DAPT encoders](https://github.com/tue-mps/simple-tad)
- Wang et al., [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727), CVPR 2023 (motion encoder: distilled ViT-S, `vit_s_k710_dl_from_giant`)
- Tong et al., [VideoMAE](https://arxiv.org/abs/2203.12602), NeurIPS 2022
- Herzig et al., [Object-Region Video Transformers (ORViT)](https://ar5iv.arxiv.org/html/2110.06915), CVPR 2022
- Kirillov et al., Segment Anything, ICCV 2023 (prompt encoder); Tancik et al., Fourier Features Let Networks Learn High Frequency Functions, NeurIPS 2020

**Detectors and trackers**

- Wang et al., YOLOv9, 2024; [official repository](https://github.com/WongKinYiu/yolov9) ([README mirror with COCO numbers](https://github.com/frankool/yolov9))
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11); [YOLOv10](https://github.com/THU-MIG/yolov10); [RT-DETR](https://huggingface.co/PekingU/rtdetr_r50vd/blob/refs%2Fpr%2F3/README.md)
- Cheng et al., [YOLO-World](https://arxiv.org/html/2401.17270v3), CVPR 2024; [repository](https://www.github.com/AILab-CVC/YOLO-World)
- Liu et al., [Grounding DINO](https://arxiv.org/pdf/2303.05499v3), ECCV 2024
- Aharon et al., [BoT-SORT](https://ar5iv.labs.arxiv.org/html/2206.14651), arXiv 2206.14651
- Zhang et al., [ByteTrack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf), ECCV 2022
- Cao et al., [OC-SORT](https://arxiv.org/abs/2203.14360v1), arXiv 2203.14360

**Evidence for the component**

- Wu et al., [Weakly-Supervised Spatio-Temporal Anomaly Detection in Surveillance Video (WSSTAD)](https://arxiv.org/pdf/2108.03825), arXiv 2108.03825 (commonly cited as IJCAI 2021; venue unverified)
- Arnab et al., [Uncertainty-Aware Weakly Supervised Action Detection from Untrimmed Videos](https://arxiv.org/pdf/2007.10703), arXiv 2007.10703
- [Bounding-Box Trajectories Matter for VAD (TrajVAD)](https://arxiv.org/pdf/2605.21957), arXiv 2605.21957
- [MAMTCF](https://arxiv.org/pdf/2307.14575), arXiv 2307.14575
- Karim et al., [DSTA](https://arxiv.org/pdf/2106.10197), arXiv 2106.10197
- Lan et al., [ClearCLIP](https://ar5iv.labs.arxiv.org/html/2407.12442), ECCV 2024
- Kuo et al., [F-VLM](https://arxiv.org/pdf/2209.15639), ICLR 2023
- [Learning from Counting](https://arxiv.org/pdf/2103.04009), arXiv 2103.04009 (failure modes of proposal-based weakly supervised detection)
- Huang et al., [Track Any Anomalous Object (TAO)](https://tao-25.github.io/), CVPR 2025

**Previous component (documented negative result)**

- Psomas et al., [Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency (Efficient Probing)](https://arxiv.org/abs/2506.10178), ICLR 2026 (source of MQR's readout form)
- Przewięźlikowski et al., [Beyond \[cls\]](https://openaccess.thecvf.com/content/ICCV2025/html/Przewiezlikowski_Beyond_cls_Exploring_the_True_Potential_of_Masked_Image_Modeling_ICCV_2025_paper.html), ICCV 2025; Bardes et al., [V-JEPA](https://arxiv.org/abs/2404.08471), arXiv 2404.08471

**Weakly supervised and open-world VAD**

- Wu et al., VadCLIP, AAAI 2024; [code](https://github.com/nwpu-zxr/VadCLIP)
- Pu et al., Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection (PEL4VAD), IEEE TIP 2024; [code and checkpoints](https://github.com/Aaron-Pu/PEL4VAD)
- [DSANet](https://arxiv.org/abs/2511.10334), AAAI 2026; [code](https://github.com/lessiYin/DSANet)
- Lv et al., [Unbiased Multiple Instance Learning (UMIL)](https://arxiv.org/abs/2303.12369), CVPR 2023; [TAD split files](https://github.com/ktr-hubrt/UMIL)
- Lv et al., Localizing Anomalies from Weakly-Labeled Videos (WSAL, TAD dataset), IEEE TIP 2021; [repository](https://github.com/ktr-hubrt/WSAL)
- [MultiDomain](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf), NeurIPS 2024
- [π-VAD](https://arxiv.org/abs/2505.13123), CVPR 2025
- [GS-MoE](https://arxiv.org/abs/2508.06318), ICCV 2025
- Alert-CLIP, [CVPR 2026](https://cvpr.thecvf.com/virtual/2026/poster/36334); [annotations](https://github.com/ClarkZhu216/Alert-CLIP_dataset)
- [STPrompt](https://export.arxiv.org/pdf/2408.05905), ACM MM 2024
- Abdulaziz et al., [Localizing to Debias](https://arxiv.org/pdf/2608.12045), arXiv 2608.12045
- Karim et al., [Learning Where and When](https://arxiv.org/pdf/2606.29498), arXiv 2606.29498; [project page](https://engineers.usf.edu/sislab/publications/learning-where-and-when-patch-based-spatiotemporal-localization-weakly-supervised)
- Gianchandani et al., [Weakly-Supervised Spatiotemporal Anomaly Detection](https://arxiv.org/abs/2605.13746), arXiv 2605.13746
- [GridVAD](https://arxiv.org/html/2603.25467v1), arXiv 2603.25467
- Project files: RefineVAD and HeadHunt-VAD (AAAI 2026); LAS-VAD, Fine-VAD, The Road Less Seen and LAVIDA (CVPR 2026); Anomize, VERA, Holmes-VAU and ADSM (CVPR 2025); FedWSVAD and MA-PDM (AAAI 2025); EventVAD (ACM MM 2025); AnomalyRuler (ECCV 2024); LAVAD (CVPR 2024); AnyAnomaly (WACV 2026); SteerVAD

**Traffic anomaly detection and anticipation**

- Shi et al., [Structured Graph Reasoning for Traffic Anomaly Detection](https://doi.org/10.1109/TMM.2026.3703345), IEEE TMM 2026
- Rossi et al., [MOVAD](https://arxiv.org/pdf/2302.10719), ICASSP 2024; [code](https://github.com/IMPLabUniPr/movad)
- Liang et al., [TTHF](https://arxiv.org/pdf/2401.03522), IEEE TCSVT 2024; Liang et al., ISCRTAD, IEEE T-ITS 2025 ([summary mirror](https://zhaozw-szu.github.io/papers/24ci6ou9); paper not accessed)
- [WS-DoTA](https://arxiv.org/html/2408.05562v1), ECCV 2024 Workshop; [split files](https://github.com/ut21/WSAD-Driving)
- [BADAS](https://arxiv.org/abs/2510.14876), arXiv 2510.14876; [FLaRA](https://arxiv.org/abs/2606.14380), arXiv 2606.14380

**Evaluation and statistics**

- Bouthillier et al., [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/pdf/2103.03098), MLSys 2021
- Lakens, Equivalence Tests, Social Psychological and Personality Science 2017 (DOI 10.1177/1948550617697177); [TOSTER](https://cran.r-project.org/web/packages/TOSTER/vignettes/IntroductionToTOSTER.html)
- Abdulaziz and Bondarev, audit of frame-level AUC in weakly supervised VAD, [arXiv 2608.11985](https://tldr.takara.ai/p/2608.11985) (abstract summary only)
- Position statistics, position-only scores and source-video counts computed from [DoTA's metadata](https://raw.githubusercontent.com/MoonBlvd/Detection-of-Traffic-Anomaly/master/dataset/metadata_val.json) and annotations (to be re-run by the student before quoting)

**Datasets**

- Yao et al., [DoTA](https://arxiv.org/abs/2004.03044), IEEE TPAMI 2022; [annotations and STAUC code](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
- Zhu et al., [MSAD](https://msad-dataset.github.io/), NeurIPS 2024 Datasets and Benchmarks; [repository](https://github.com/Tom-roujiang/MSAD)
- Bao et al., [CCD](https://github.com/Cogito2012/CarCrashDataset), ACM MM 2020 ([paper](https://arxiv.org/pdf/2008.00334))
- Fang et al., [MM-AU](https://arxiv.org/abs/2403.00436), CVPR 2024; [CAP-DATA](https://arxiv.org/abs/2212.09381)
- Fang et al., [DADA-2000](https://arxiv.org/pdf/1912.12148), IEEE T-ITS 2021
- Moura et al., [Nexar Dashcam Collision Prediction Dataset and Challenge](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf), CVPR 2025 Workshops

**Other methods, datasets and models cited**

- Sultani et al., Real-World Anomaly Detection in Surveillance Videos (UCF-Crime), CVPR 2018
- Feng et al., MIST, CVPR 2021; Tian et al., RTFM, ICCV 2021
- Zhang et al., CUPL, CVPR 2023; Zhou et al., UR-DMU, AAAI 2023; Chen et al., MGFN, AAAI 2023
- TEVAD (venue unverified); [LAP](https://arxiv.org/abs/2403.01169) (journal unverified); [VADor](https://arxiv.org/abs/2401.05702) (arXiv only)
- Wu et al., XD-Violence, ECCV 2020; LAD (test list from LaGoVAD's repository)
- Yu et al., BDD100K, CVPR 2020; D²-City, arXiv 2019; Lin et al., COCO, ECCV 2014
- Wang et al., InternVideo2, ECCV 2024; Qwen2.5-VL technical report, 2025
