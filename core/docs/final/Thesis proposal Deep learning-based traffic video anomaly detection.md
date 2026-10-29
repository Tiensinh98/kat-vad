# Thesis proposal: Deep learning-based traffic video anomaly detection

Oct 29, 2026 · @Sinh Pham

This proposal builds **LaGoVAD-MQR**, a weakly supervised detector for road-traffic video anomalies that runs no LLM at training or inference. It starts from LaGoVAD (ICLR 2026) and adds a frozen VideoMAE V2 motion stream as the permitted backbone change. The one novel component is a **Motion Query Readout (MQR)**: a learned scorer rates every space-time tile of the dense VideoMAE tokens by how its motion departs from the rest of the frame, is trained with a tile-level MIL loss that treats every tile of a normal video as a negative, and steers the readout that replaces mean pooling. The claim under test: reading motion where it happens lifts detection of small, distant and ego-uninvolved traffic agents, and yields evidence maps scored with DoTA's spatio-temporal AUC (STAUC), which our search (to October 2026) found no weakly supervised method reporting.

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
- **Novel component (MQR).** The motion stream's dense tokens are currently averaged into one vector per clip. MQR replaces that average with a readout steered by a tile scorer on residual and change features of the tokens, trained with a tile-level MIL loss, so the evidence of small and distant agents is kept.

A second purpose is a fair evaluation: published traffic numbers are mostly single-metric AUC on inconsistent splits. The thesis will report at least two metrics against at least two published methods on the same fixed-camera traffic data, plus zero-shot and dashcam results.

### 1.3 Research aims

The thesis has five aims:

1. **Reproduce the baseline.** Re-score the released LaGoVAD checkpoint and retrain LaGoVAD on the thesis training data, with AUC, AP and AUC on abnormal videos (AUC\_A) on TAD, MSAD, DoTA and DADA-2000.
2. **Add the motion stream.** Integrate the frozen VideoMAE V2-S stream and measure its gain over CLIP-only LaGoVAD.
3. **Design and train MQR.** Replace the motion stream's mean pooling with the tile-scored readout and tile-level MIL loss, using video-level annotations only.
4. **Measure spatial localisation under weak supervision.** Score MQR's maps against DoTA's anomalous-agent boxes with STAUC and a pointing game.
5. **Explain the gains.** Break results down by ego-involved vs ego-uninvolved clips, agent box size and anomaly category, and check that MSAD's non-traffic scenes do not regress.

Aims 1–2 are prerequisites; aims 3–5 map to the hypotheses below. "Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval over test videos excludes zero.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. MQR beats the mean-pooled motion stream, mostly on small and ego-uninvolved agents | Position-controlled DoTA AUC (Section 4.4) by ego-involved / ego-uninvolved and by clip-level box-size bin; TAD AUC for Accident, PedestrianOnRoad and the other categories pooled | Gains on ego-uninvolved clips and small-box bins exceed those on ego-involved clips and large-box bins |
| H2. MQR's maps point at the anomalous agents | STAUC and pointing game on DoTA's official 1,305-video list | MQR beats a uniform map (the mean-pooled motion stream), a centre prior and an untrained change-magnitude map |
| H3. The gain holds across camera settings without hurting surveillance | TAD AUC\_O / AUC\_A; MSAD AUC / AP, its traffic slice and its non-traffic scenes; CCD, DoTA and DADA-2000 AUC | Gains on TAD, the MSAD traffic slice and the dashcam sets; no significant drop on MSAD's non-traffic scenes |
| H4. Learned tile selection matters, not just more tokens or a fixed motion ranking | Ablation ladder: max / GeM pooling, untrained change top-K, free-query readout, tile scorer without and with normal-video tile negatives | Full MQR beats every simpler readout |

H2 is the primary hypothesis, because it tests the component's purpose directly. DoTA AUC alone cannot show region selection: clip-averaged AUC on abnormal-only clips is a within-clip timing measure, which a score based only on each frame's position in the clip can already do well on. Detection gains on DoTA are therefore always read against a position-only reference (Section 4.4).

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

### 2.3 Key works behind the proposed model

The proposed model is assembled from a small set of prior works: one baseline, one encoder, one readout, and the methods whose evidence justifies each choice. This subsection reviews each of them: what it does, its strengths, its weaknesses, and exactly what the thesis takes from it and changes. The table shows where each part of LaGoVAD-MQR comes from; the reviews follow, and the last table covers the methods used only as comparison rows.

| Part of LaGoVAD-MQR | Source work (venue) | What is taken | What the thesis changes |
| --- | --- | --- | --- |
| Host detector: CLIP stream, RoFormer, text co-attention, heads and losses | LaGoVAD (ICLR 2026), which inherits the frozen-CLIP + MIL design of VadCLIP (AAAI 2024) | Architecture, losses and schedule as released | Nothing inside the detector; checkpoints are taken at the last epoch, and captions are hand-written templates |
| Motion stream (backbone change) | VideoMAE V2 (CVPR 2023), chosen on the evidence of Simple-TAD (ICCV 2025 Workshops) | The frozen distilled ViT-S, and Simple-TAD's 16-frame, 10 fps clip input | Used frozen under video-level labels and fused through a zero-initialised gate, instead of fine-tuned with frame labels |
| Motion Query Readout (novel component) | Efficient Probing (ICLR 2026) for the readout form; per-patch scoring from "Learning Where and When" (arXiv 2606.29498); rationale from Beyond \[cls\] (ICCV 2025) and V-JEPA's frozen-probe tables | Attention weights over tokens pooling a single value projection; one score per patch | Weights come from a tile scorer on frame-wide residual and change features of a 4×7×7 grid of centred clips; the scorer's last layer starts at zero and the value projection at the identity, so it starts as mean pooling; run at every step inside a weakly supervised detector |
| Tile-level MIL loss | The top-k MIL loss of LaGoVAD and VadCLIP; the normal-patch negatives of "Learning Where and When"; motivated by the local-evidence results of PromptTAD (IEEE RA-L 2025), Alert-CLIP (CVPR 2026) and STPrompt (ACM MM 2024) | Top-k pooling of scores over time; every patch of a normal video labelled 0 | Top-k over steps and cells on abnormal videos and BCE on every cell of normal videos, on frozen motion tokens, with no detector, boxes, region captions or LLM |
| Evidence-map evaluation | DoTA (IEEE TPAMI 2022) | STAUC code, anomalous-agent boxes and the ego-involved flag | Maps taken from the tile scores, scored under weak supervision against uniform, centre and untrained change maps |
| Exploratory definition-conditioned scoring | Grounding DINO (ECCV 2024) | The idea that text decides where the readout looks | Additive, zero-initialised conditioning of the tile scorer; outside the main claim |

#### 2.3.1 LaGoVAD (ICLR 2026): the baseline

Liu et al. define open-world VAD as a mapping Φ: (V, Z) → frame scores, where the anomaly definition Z is given as text at test time. Frozen CLIP ViT-B/16 frame embeddings pass through a 2-layer RoFormer and are fused with the CLIP text encoding of Z by co-attention. Training uses video-level labels on PreVAD (35,279 videos, 11,979 abnormal) with MIL and MIL-alignment losses, dynamic video synthesis and a caption contrastive loss with hard negatives.

- **Strengths.** It has the best zero-shot traffic results in its comparison: 89.56 AUC on TAD, 62.60 on DoTA and 90.41 on MSAD, ahead of VadCLIP and PEL. It stays robust when the definition changes (MSAD drift@5: 85.6 AUC / 40.1 AP, against 84.3 / 34.3 for Holmes-VAU). The detector runs no LLM, trains on one RTX 4090, and its code, checkpoint and predictions are public.
- **Weaknesses.** Its only visual input is one global CLIP embedding per sampled frame, so it sees neither motion nor where in the frame something happens: DoTA AUC is 62.60 against 89.56 on TAD, and language guidance adds only 1.20 AUC on DoTA against 5.60 on TAD. It outputs no spatial evidence. Its traffic test sets are prompted with a generic "car accident" definition, although 37 of the 60 abnormal TAD test videos on its list are not accidents. Its released configuration selects checkpoints on test scores, PreVAD's captions were written by Qwen models, and the authors call the design "simplified".
- **Taken into this thesis.** The whole detector, unchanged. The thesis adds a motion input and a spatial readout, and replaces test-score checkpoint selection with the last epoch.

#### 2.3.2 VadCLIP (AAAI 2024): the CLIP-based weakly supervised template

Wu et al. adapt frozen CLIP to weakly supervised VAD. Frame embeddings pass through a local Transformer and a graph convolution (the LGT-Adapter). A classification branch gives binary frame scores, and an alignment branch matches frames to class-name embeddings built with learnable prompts (MIL-Align).

- **Strengths.** It established the CLIP-plus-text-prompt design that LaGoVAD, DSANet and Alert-CLIP build on, reaching 88.02 AUC on UCF-Crime and 84.51 AP on XD-Violence on a single RTX 3090, with public code and features.
- **Weaknesses.** It reads appearance only, and most of its gain comes from the temporal adapter (XD-Violence AP 72.22 without it, 84.51 with it). Zero-shot, it sits near chance on DoTA (49.00–50.93 AUC). In DSANet's analysis, its prototypes assign 84.0% of UCF-Crime road-accident videos to "Normal". Its classes are fixed at training time.
- **Taken into this thesis.** Its design reaches the thesis through LaGoVAD, and it is retrained as the same-feature weakly supervised competitor (Group C, Section 4.3).

#### 2.3.3 VideoMAE V2 (CVPR 2023): the motion encoder

Wang et al. scale masked video autoencoding with dual masking: the encoder sees only about 10% of the video tokens (tube masking), and the decoder reconstructs only part of the masked cubes (running-cell masking at 50%). Training is progressive: self-supervised pre-training on 1.35M unlabelled clips, then supervised post-pre-training on Kinetics-710 (0.66M clips, 710 classes). The resulting billion-parameter ViT-g is distilled into smaller ViT-S and ViT-B models.

- **Strengths.** Dual masking made a 1B-parameter video ViT trainable, reaching 90.0% top-1 on Kinetics-400 and 77.0% on Something-Something V2. The distilled ViT-S is small (about 22M parameters) and public. In Simple-TAD's comparison it is the strongest small encoder for traffic anomalies after fine-tuning: AUC-MCC 55.2 on DoTAref and 50.3 on DADA-2000, against 52.1 / 48.1 for VideoMAE-S and 49.7 / 43.7 for InternVideo2-S. Its tokens are 2×16×16 space-time tubelets of a 16-frame clip, so they keep local, moving content.
- **Weaknesses.** It is designed to be fine-tuned, and masked-model features read frozen with a linear probe lose much of their signal: in V-JEPA's frozen evaluation, VideoMAE ViT-L scores 52.5 on Kinetics-400 with a linear probe and 77.8 with an attentive probe. Its pre-training data contain no driving video, its post-pre-training labels are human actions, and its tokens are not aligned with text, so LaGoVAD's definitions cannot address them directly.
- **Taken into this thesis.** The frozen distilled ViT-S as the motion encoder; its 8×14×14 last-layer tokens, reduced to a 4×7×7 grid, are what MQR reads. Text alignment stays with the CLIP stream.

#### 2.3.4 Simple-TAD (ICCV 2025 Workshops): evidence for the encoder choice

Orlova et al. show that a plain video ViT with one linear head, fine-tuned with frame-level labels, outperforms specialised traffic detectors on DoTA and DADA-2000. They compare pre-training paradigms and add domain-adaptive pre-training (DAPT): further masked pre-training on normal driving video.

- **Strengths.** It is simple and fast (DAPT-VideoMAE-S: 22M parameters, 95 FPS on an A100 partition) and strong (DoTA AUC up to 88.4; DoTA → DADA-2000 up to 86.6). It shows systematically that masked video modelling transfers best to traffic anomalies and vision-language pre-training (InternVideo2, UMT) worst. DAPT on BDD100K raises ViT-S AUC-MCC on DoTAref from 52.1 to 54.9 without any anomalous data. Code, encoders and refined DoTA labels are public.
- **Weaknesses.** It needs frame-level labels, covers dashcam data only and takes no text. It averages all tokens before the linear head, and ego-uninvolved clips lag: 81.6 / 84.1 / 85.2 AUC (ViT-S/B/L) against 89.9 / 90.7 / 90.7 on ego-involved clips. The authors note that object-aware methods (PromptTAD, ISCRTAD) do better on rare and ambiguous categories. It is a workshop paper, so the thesis uses it as evidence, not as the source of the component.
- **Taken into this thesis.** The choice of a masked video encoder and its clip input; the DAPT encoder as an ablation; and the ego-uninvolved gap, which motivates hypothesis H1.

#### 2.3.5 Efficient Probing (ICLR 2026): the source of MQR

Psomas et al. revisit attentive probing of frozen encoders. Efficient Probing (EP) uses M learnable queries that score the tokens directly, with no query or key projections. Each query takes a softmax over the tokens and pools its own slice of a single value projection, and the slices are concatenated. The authors describe it as a single-iteration form of slot attention.

- **Strengths.** It gives large gains where linear probing is weakest, on masked models: MAE ViT-B on ImageNet-1K rises from 67.7 with a linear probe to 75.6. It matches or beats heavier attentive probes (AIM, the V-JEPA probe) with fewer parameters, and a full Transformer block with over 10× less compute. Its queries attend to distinct regions (complementarity 0.65, against 0.37 for the best internal MAE heads), and accuracy grows with the number of queries (SimMIM ViT-B: 60.6 with 2 queries, 65.1 with 64). The code is Apache-2.0.
- **Weaknesses.** It is evaluated on fully labelled image classification only: no video encoder, no weak labels, no anomalies. Nothing in its training ties a query to a region of interest; the distinct attention maps are a by-product of classification, and no loss encourages them.
- **Taken into this thesis.** The readout form: attention weights over the tokens pool a single value projection, started at the identity so that the readout begins as mean pooling. Because nothing in EP's training ties a query to a region, and video-level labels give a free query only an indirect signal through a pooled score, the thesis derives the attention weights from a tile scorer trained directly by the tile-level MIL loss (Section 3.2). EP's free queries, with cosine logits, a learnable scale and random initialisation, are kept as an ablation row (Section 4.5).

#### 2.3.6 Beyond \[cls\] (ICCV 2025): why masked-model tokens need selective pooling

Przewięźlikowski et al. analyse why the \[cls\] token and average pooling under-use the patch tokens of masked image models, and propose selective aggregation through attention-based MIL pooling (AbMILP).

- **Strengths.** It explains the gap: attention in masked models is spread almost uniformly (MAE aggregation entropy 5.03 against a maximum of 5.27), so neither \[cls\] nor averaging isolates what is specific. Selective aggregation lifts MAE ViT-B from 65.8 (average of patches) to 71.6, while for the joint-embedding DINO it gives no gain over \[cls\] (75.2 against 76.6). The effect is therefore specific to masked models, the family VideoMAE V2 belongs to.
- **Weaknesses.** Images only, classification only, and a single attention query; its code carries a non-commercial licence.
- **Taken into this thesis.** The rationale for replacing mean pooling over VideoMAE tokens. Its single-query attentive pooling corresponds to the free-query row of the ablation ladder with M = 1 (row 4, Section 4.5). V-JEPA's frozen-probe tables show the same pattern for video encoders (Section 2.3.3).

#### 2.3.7 PromptTAD (IEEE RA-L 2025): object-level evidence from a detector

Qiu et al. extend MOVAD (Video Swin Transformer with an LSTM) with object prompts. YOLOv9 boxes are encoded as prompt tokens; an instance-wise module lets object tokens and frame features attend to each other, and a relation-wise module with 12 learnable query tokens aggregates them into an enhanced frame token. Training uses frame labels and ground-truth boxes of the anomalous objects.

- **Strengths.** It names the failure MQR targets: "the global perspective often emphasizes prominent regions within the frames, resulting in the neglect of distant or off-center objects". It reports gains in both AUC and STAUC, evaluates a small-object subset (334 DoTA test videos whose anomalous boxes are under 2,267 pixels), and reaches 83.9 AUC on DoTA. Transferred to DADA-2000, it holds up better than TTHF on ego-uninvolved clips (70.4 against 64.0).
- **Weaknesses.** It needs frame labels, anomalous-object boxes and an external detector; when detection fails, as in dark scenes, the method "degenerate\[s\] into the baseline". It trains on four A6000 GPUs, has no public code, covers dashcam data only, and is a journal paper outside the venue list.
- **Taken into this thesis.** The diagnosis, and the idea of learnable queries that gather local evidence. MQR replaces detector boxes with a tile scorer over motion tokens, and box supervision with the tile-level MIL loss.

#### 2.3.8 Alert-CLIP (CVPR 2026): region-level alignment learned from boxes

Alert-CLIP tunes a CLIP ViT-L backbone with a temporal Transformer through three contrastive alignments: video to caption, region (ROI-aligned from annotated boxes) to region caption, and region against hard-negative captions. It is trained on VAGTA, a re-annotation of UCF-Crime and MSAD with boxes and region captions, and then used as a drop-in backbone.

- **Strengths.** Region-level training moves CLIP's attention from the surrounding context to the collided vehicles (its Fig. 5), and lifts the VadCLIP framework on MSAD from 87.45 AUC / 71.09 AP to 89.63 / 78.24. Boxes are needed only for training; inference is CLIP-only.
- **Weaknesses.** It needs manual boxes, segmentation and region captions (its hard-negative captions are written offline by Qwen-VL), and heavy tuning on an 80 GB GPU. It reports no quantitative traffic result, does not state that its checkpoint is released, and still reads frame-level appearance without motion.
- **Taken into this thesis.** Evidence that region-level evidence improves detection, and a comparison row on MSAD. MQR seeks the same region focus with neither boxes nor captions.

#### 2.3.9 STPrompt (ACM MM 2024): the closest weakly supervised spatial design

STPrompt adds a spatial branch to a VadCLIP-style detector on frozen CLIP. Each patch's motion magnitude is a second-order difference of its features across neighbouring frames; the top-K moving patches are softmax-weighted and added to the frame feature for temporal detection. At test time, localisation is training-free: sliding-window crops are encoded by CLIP's global token and matched against normal and abnormal descriptions generated by an LLM.

- **Strengths.** It is one of the few methods that localise anomalies in space and time from video labels, and it improves temporal-IoU localisation over VadCLIP by about 1.9, 5.7 and 4.5 points on UCF-Crime, ShanghaiTech and UBnormal.
- **Weaknesses.** Its patch selection is a fixed motion ranking, and in dashcam video ego-motion dominates such feature differences; the thesis's own preliminary experiments found feature-difference cues unhelpful. Localisation needs an LLM and is not trained. It covers surveillance data only, and no public code was found.
- **Taken into this thesis.** It bounds the novelty claim, and its ablation supplies a design lesson: learned attention alone did no better than average pooling (86.67 against 86.91 UCF-Crime AUC), while motion top-K selection followed by attention reached 88.08. MQR learns the selection from video labels on motion-foundation tokens instead of using a fixed motion ranking, trains its maps through the tile-level MIL loss, and uses no LLM. A fixed change ranking in the spirit of STPrompt is an ablation row that MQR must beat (row 3, Section 4.5).

#### 2.3.10 Methods used as comparison rows

These methods contribute no part of the architecture; they supply the published numbers the thesis is compared against (Section 4.4).

| Method (venue) | Approach | Strength | Weakness | Role in this thesis |
| --- | --- | --- | --- | --- |
| WSAL (IEEE TIP 2021) | High-order context encoding of TSN snippet features plus segment-to-segment variation cues; introduced the TAD dataset | First weakly supervised traffic benchmark; 89.64 AUC\_O / 61.66 AUC\_A on TAD | Hand-crafted augmentations and cues; a fixed 32 segments per video; no spatial output | P1 row on both metrics |
| UMIL (CVPR 2023) | Splits snippets into confident and ambiguous sets by prediction variance and clusters the ambiguous ones, so the classifier stops relying on context shortcuts; X-CLIP-B/32 fine-tuned end to end | Best two-metric TAD result: 92.93 AUC\_O / 65.82 AUC\_A | Needs backbone fine-tuning (90.71 AUC\_O when frozen); clustering can go wrong; sensitive to its trade-off weight on TAD | P1 row; source of the official TAD split |
| π-VAD (CVPR 2025) | Induces five extra modalities (including pose, depth and optical flow) from RGB features during training; RGB-only inference | MSAD 88.68 AUC / 71.26 AP; flow is the most useful modality | Needs five teacher backbones for training; no public code found | P2 row; evidence that motion matters |
| GS-MoE (ICCV 2025) | Gaussian pseudo-labels around score peaks plus one expert per anomaly class, on I3D and UR-DMU features | MSAD 87.72 AUC / 68.26 AP / 69.54 AUC\_A / 76.68 AP\_A | Needs class labels for its experts; 16.02M parameters at about 9.6 FPS; misses an accident that occupies "a very small pixel-region" | P2 row; evidence of spatial dilution |
| WS-DoTA (ECCV 2024 Workshop) | Weakly supervised DoTA with D²-City normal clips; a spatial-aware temporal-regularity transform of CLIP features | The only weakly supervised dashcam study; lifts RTFM from 57.9 to 78.2 AUC | Its own 1,031-video subset; AUC only | Flagged P3 row; evidence that CLIP frames lack motion |

### 2.4 Research gaps

Four gaps follow from the review. The thesis addresses each one, and its novelty claim is deliberately narrow.

1. **No motion-aware foundation features in weakly supervised traffic VAD.** LaGoVAD and VadCLIP read CLIP frames; the masked-video evidence of Simple-TAD comes from frame-level supervision. The one weakly supervised dashcam study, WS-DoTA (ECCV 2024 Workshop), added a temporal-regularity transform of CLIP features and lifted RTFM from 57.9 to 78.2 AUC. *Addressed by the motion stream (backbone change).*
2. **Frozen masked-video features are read by mean pooling, which loses much of their discriminative signal.** Frozen VideoMAE ViT-L scores 52.5 with a linear probe and 77.8 with an attentive probe on Kinetics-400, and 41.3 vs 61.2 on Something-Something v2 (V-JEPA; the pooling behind its linear column is not stated). Masked image models show the same pattern: 67.7 vs 75.6 for MAE ViT-B with Efficient Probing (ICLR 2026), 65.8 vs 71.6 with selective aggregation (Beyond \[cls\], ICCV 2025). These rows come from different experiments. No VAD work reads such tokens with a learned, tile-level selection under weak supervision on traffic data. *Addressed by MQR.*
3. **Weak supervision gives no spatial evidence on traffic data.** We found no weakly supervised traffic method that produces maps, and no report of STAUC under weak supervision. *Addressed by MQR's tile-score maps, scored with STAUC and a pointing game.*
4. **Traffic results are hard to compare.** Most rows are AUC-only and two TAD test lists circulate. *Addressed by four split-consistent, multi-metric protocols (Section 4.4).*

The closest prior work bounds the novelty claim:

| Closest work | What it does | How MQR differs |
| --- | --- | --- |
| BADAS (arXiv 2510.14876) | Attentive probe with 12 learned queries on a fine-tuned V-JEPA2, fully supervised ego-collision anticipation; no attention maps reported | Weak labels, detection, tile-level MIL with normal-tile negatives, frozen encoder, language-guided host model |
| FLaRA (arXiv 2606.14380) | Single-query attentive probe for accident anticipation | Tile-level selection, weak labels, maps evaluated |
| STPrompt (ACM MM 2024) | Motion-selected image crops encoded by CLIP's global token for weakly supervised surveillance VAD; LLM-assisted localisation; no code | Learned instead of fixed motion selection, motion-foundation tokens, traffic data, no LLM |
| "Localizing to Debias" (arXiv 2608.12045) | Spatio-temporal weakly supervised VAD on VideoMAEv2 tubelet tokens with hard token pruning and a time-reversal motion term; UCF-Crime, XD-Violence and MSAD (scoring details in an unread supplement) | Dense tile scoring with normal-video tile negatives instead of pruning, frame-wide residual and change features, dashcam data, STAUC on DoTA; **main novelty risk, to be read in full and compared** |
| "Learning Where and When" (arXiv 2606.29498) | One score per patch of a 7×7 grid of segment features, a hard 2×2 positive window and every patch of normal videos labelled 0; surveillance data | Frozen motion tokens with residual and change features, score-steered readout inside a language-guided host, dashcam data, STAUC; **second novelty risk: the closest training scheme** |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with boxes and captions | Needs neither boxes nor captions for the readout |
| PromptTAD, ISCRTAD | Object prompts or agent graphs from detectors, frame-level supervision | Detector-free, video-level labels |

The resulting claim, to our knowledge: **the first weakly supervised traffic anomaly detector that reads dense motion-foundation tokens through a learned tile-level change-evidence readout, and the first to report STAUC under weak supervision.** Neither attentive pooling nor patch-level MIL is claimed as new; the claim is their combination on frozen motion tokens, with frame-wide residual and change features against ego-motion, inside a language-guided detector on dashcam video. Both claims will be re-checked against the two 2026 preprints before submission. If MQR fails its gates (Section 4.4), the thesis reports it as a documented negative result, with the mechanism and the position-prior audit, and the contribution rests on the motion stream and the evaluation.

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

**Clip alignment.** "Centred" means frames t−8 … t+7 of the 10 fps stream, with edge frames repeated at video boundaries, so the clip's temporal centre falls on the frame it scores. A causal window (t−15 … t) would centre the motion evidence about 0.75 s before that frame. Under video-level MIL nothing forces the detector to undo such a lag, and the frame pseudo-labels of dynamic video synthesis would be misaligned by the same amount, so the window is fixed to the centred form for every model. For the 10 fps sets, whose cache holds one clip per frame, a causal cache converts exactly by re-indexing (centred\[t\] = causal\[t+7\], repeating the last clip at the end); 30 fps sets are re-extracted. A shift test checks the alignment: frame AUC as a function of a test-time offset applied to the score sequence, computed on the DoTA-train pilot subset (Section 4.3), should peak at an offset of zero.

The last layer gives X\_t, 8 temporal × 14 × 14 spatial tokens of dimension D. The preliminary version ("A") averages all 1,568 tokens and adds the result to the CLIP embedding through a gate that starts closed:

```latex
v_t = \frac{1}{1568} \sum_{j=1}^{1568} X_{t,j}, \qquad x_t = c_t + \tanh(\alpha)\, W\, \mathrm{LN}(v_t), \qquad \alpha_{\text{init}} = 0,\; W \in \mathbb{R}^{512 \times D}
```

The fused x\_t replaces c\_t as the RoFormer input. Because α starts at 0, the model starts as CLIP-only LaGoVAD and opens the gate only as far as motion helps. The kNN retrieval of dynamic video synthesis still uses CLIP features.

**No overlap with the evaluation data.** VideoMAE V2 was pre-trained on Kinetics-710, Something-Something V2, AVA, WebVid2M and Instagram clips and distilled on Kinetics-710; none of the seven traffic datasets used in this thesis is among these sources. Simple-TAD's driving-adapted DAPT-VideoMAE-S (BDD100K only) is kept as an encoder ablation.

&#91;embedded content: LaGoVAD-MQR architecture · the new component highlighted\]

Only the highlighted readout and its dashed training loss are new. The motion stream is the backbone change, and every plain box is LaGoVAD as released.

### 3.2 Novel component: Motion Query Readout (MQR)

MQR replaces the motion stream's average with a readout steered by a learned scorer on every space-time tile of the motion tokens. The scorer reads residual and change features, so a tile scores high when its motion departs from the rest of the frame. One extra loss, a tile-level MIL loss, trains the scorer from video labels alone: on abnormal videos a few tiles in a few steps must score high, and on normal videos every tile must score low. The readout then pools the motion tokens with the scorer's own logits as attention weights. The name is kept: the readout still queries the tokens, but its query is the tile scorer's output rather than a free vector. The free-query form of Efficient Probing (Psomas et al., ICLR 2026) is kept as an ablation.

**Why a tile scorer instead of free queries.** A direct transfer of Efficient Probing to this setting, with free queries over a time-averaged grid trained only through a pooled video score, faces four obstacles, and each design choice below answers one of them:

1. **Weak gradient at a uniform start.** At zero or near-zero queries, the gradient of each query is a covariance over tiles between the layer-normalised keys and the per-tile value gradients, so it shrinks roughly quadratically as tiles become alike. Masked-model tokens are dominated by a shared component, and a 1/√D scale shrinks the logits further. *Answer:* each tile gets its own score with a direct gradient, and the frame-wide component is removed before LayerNorm.
2. **Dilution of transients.** Averaging VideoMAE's 8 temporal slots and 2×2 patches divides an event that fills one slot and one sub-patch by up to 32, which removes the transient a collision produces. *Answer:* the grid keeps time inside the clip, and change between slots is a feature.
3. **Starved routing.** With a closed gate, the main loss sends no gradient into a readout at the start; a max over M queries with top-k over time then trains only about 1/M × 1/16 of each video's query–step pairs. *Answer:* the scorer is trained by its own loss on every cell, and the loss uses top-K cells instead of one query.
4. **No tile-level negatives.** A bag loss on normal videos only pushes down soft averages of tiles, so no tile is ever told that it is normal. *Answer:* every tile of every normal video is a labelled negative.

Attention entropy is therefore not used as a success criterion: near a uniform start it barely moves even when the ranking of tiles is already right. Success is judged by pointing game, STAUC and position-controlled detection (Section 4.4), with entropy logged only as a diagnostic.

**Token grid.** The 8×14×14 tokens of each clip are reduced by a 2×2 spatial average and by averaging consecutive pairs of VideoMAE's 8 temporal slots, to S = 4 slots of a 7×7 grid, Q\_t ∈ R^{4×7×7×D}: 196 cells per step, each covering 32×32 px of the 224×224 input and about 0.4 s of the clip (D = 384 for the default ViT-S). Both poolings weight tokens equally, so the average of the 196 cells still equals the average of all 1,568 tokens. The full 8 slots and a 14×14 grid with tokens PCA-compressed to D/4 are side ablations; the 7×9 px distant agent of Section 1.1 fills only about 6% of a 7×7 tile, so the Step 0 diagnostic (Section 4.3) decides before main training whether 14×14 is needed. Each cell is indexed by i = (s, j), with s the slot and j the tile.

**Residual and change features.** Each cell is described by how it differs from the rest of its frame, and by how it changes between slots. The median is taken over the 49 tiles of the same slot, coordinate by coordinate:

```latex
R_{t,s,j} = Q_{t,s,j} - \operatorname{med}_{j'} Q_{t,s,j'}, \qquad \tilde{\Delta}_{t,s,j} = Q_{t,s+1,j} - Q_{t,s,j}, \qquad \Delta_{t,s,j} = \tilde{\Delta}_{t,s,j} - \operatorname{med}_{j'} \tilde{\Delta}_{t,s,j'}, \qquad F_{t,s,j} = \big[\, \mathrm{LN}(R_{t,s,j}) ;\, \mathrm{LN}(\Delta_{t,s,j}) \,\big] \in \mathbb{R}^{2D}
```

Δ is defined for s < S and repeated for the last slot. Subtracting the frame-wide median removes what all tiles share: the scene, the lighting and most of the ego-motion, which in dashcam video dominates every tile's change. The median is used instead of the mean so that the tile holding the agent does not pull the reference towards itself. The subtraction comes before LayerNorm: after LN it would add the same constant to every logit and cancel in the softmax, while before LN it lets LN rescale the remaining per-tile deviation. The change feature is computed in VideoMAE's feature space, not from optical flow, because hand-crafted flow and feature-difference cues did not help in preliminary experiments (Section 1.1), and a learned scorer on frozen-feature change can weigh the camera's own motion instead of being dominated by it.

**Tile scorer.** One small head h, shared by all cells, scores each cell's features:

```latex
e_{t,i} = h(F_{t,i}), \qquad p_{t,i} = \sigma(e_{t,i}), \qquad h(F) = w_2^{\top}\, \mathrm{GELU}(W_1 F + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{128 \times 2D}
```

The last layer (w\_2, b\_2) starts at zero, so every e\_{t,i} is 0 at initialisation. Every cell receives gradient from the tile-level MIL loss at every step, so no softmax has to leave a uniform plateau before anything is learned. Each p\_{t,i} is a signed, anomaly-specific probability for one tile in one slot, which an attention weight is not.

**Readout.** The readout pools the motion tokens with the scorer's logits as attention weights, through a single value projection:

```latex
a_{t,i} = \operatorname{softmax}_i\!\left( \frac{e_{t,i}}{T} \right), \qquad r_t = W_V \sum_{i=1}^{196} a_{t,i}\, Q_{t,i} \in \mathbb{R}^{D}, \qquad x_t = c_t + \tanh(\alpha)\, W\, \mathrm{LN}(r_t)
```

T = exp(θ) is a learnable temperature with θ initialised at 0 and T clamped to at least 0.05. Pooling before projecting is exactly equal to projecting every token first, because W\_V is linear, and it keeps memory low. The readout r\_t enters the existing gate unchanged, with α\_init = 0 as in the motion-stream baseline. A closed gate does not starve the component: the scorer is trained through its own loss, and the main loss reaches h, T and W\_V as soon as α moves. A stop-gradient variant, in which the main loss does not reach h, is a side ablation.

**Same starting function.** With the last layer of h at zero, every attention map is uniform over the 196 cells; with W\_V = I and zero bias, r\_t then equals the token average v\_t exactly. This requires v\_t to be the plain mean of the same last-layer tokens, so the preliminary code is checked for any extra normalisation (such as a final `fc_norm`). At initialisation LaGoVAD-MQR is therefore the same function as the mean-pooled motion-stream model. The tile-level MIL loss updates h from the first step, and the readout sharpens only as far as the scorer learns to separate tiles; whether MQR ends above the motion-stream model is what the ablation tests.

**Tile-level MIL loss (the one extra loss).** A step's spatial score is its highest cell score. On an abnormal video, the loss lifts the K highest cells of each of the k highest-scoring steps; on a normal video, it lowers every cell of every valid step:

```latex
S_t = \max_i p_{t,i}, \qquad \hat{s}_V = \frac{1}{kK} \sum_{t \in \operatorname{top}k(S)}\; \sum_{i \in \operatorname{top}K(p_t)} p_{t,i}, \qquad \mathcal{L}_{\text{neg}} = \frac{1}{196\,L} \sum_{t=1}^{L} \sum_{i=1}^{196} \operatorname{BCE}(p_{t,i}, 0)
```

```latex
\mathcal{L}_{\text{tile}} = y \cdot \operatorname{BCE}(\hat{s}_V, 1) + (1-y) \cdot \mathcal{L}_{\text{neg}} + \beta \Big( y \cdot \overline{p} + (1-y) \cdot \overline{p^{2}} \Big)
```

Top-k is taken over valid (unpadded) steps only, with k = max(⌊L/16⌋, 1) from the true length L, and K = 4 cells by default. The last term is a light sparsity penalty, L1 on abnormal videos and L2 on normal videos, with β = 0.01 and the bars denoting means over all valid cells. On an abnormal video, supervision is multiple-instance in time and in space at once: only a few cells of a few steps must be high. The dense negative term is the signal a bag-level loss lacks. It turns every tile of every normal video, including the BDD100K and Nexar normals with moving cars, pedestrians and camera shake, into a labelled negative, and it is the component's main lever against the context shortcut that makes weakly supervised maps light up the scene instead of the agent. Synthetic videos from dynamic video synthesis count as abnormal for the MIL term, and their steps whose pseudo-label is normal also enter the dense negative term; steps within 8 frames of an inserted boundary are left out of it, because their centred clip spans both sides. A hard-negative variant, which lowers only the top-K cells of each step of a normal video, is part of the loss-form ablation. The total loss is L\_LaGoVAD + λ·L\_tile with λ ∈ {0.1, 0.5, 1}.

**Evidence map.** E\_t(j) = max\_s p\_{t,s,j}, the highest score of tile j over the clip's slots, so a tile lights up if it is anomalous at any moment of the clip. The map lies in \[0, 1\]; it is reshaped to 7×7, upsampled to the frame and normalised per video to \[0, 1\]. The map is the trained tile classifier itself, not an attention proxy; the attention weights a\_{t,i} are reported only as a diagnostic.

**Free-query ablation.** The Efficient Probing form is kept for the ablation ladder (Section 4.5, row 4), repaired so that it can learn: M query vectors u\_m score the centred cells through cosine logits with a learnable scale, ℓ\_{t,m,i} = κ\_m·cos(u\_m, R\_{t,i}) + e\_{t,i}, with κ\_m initialised at log₂(N² − N) ≈ 15 for N = 196 cells (the QK-norm rule) and clamped. The queries start at random (randn·0.02) or at k-means centres of centred cells from the training set, sit in their own parameter group without weight decay, and are aggregated in the loss by log-sum-exp over queries instead of a hard max, with an overlap penalty Σ\_{m≠m′}⟨a\_m, a\_{m′}⟩ against collapse. Each query pools its own D/M-wide slice of W\_V·Q, as in Efficient Probing.

**Exploratory arm, outside the main claim.** The tile scorer can be conditioned on the definition embedding LaGoVAD already computes, e\_{t,i} = h(F\_{t,i}) + G(τ)ᵀ F\_{t,i}, with G a zero-initialised linear map from 512 to 2D, so training starts from static MQR. This is additive conditioning inspired by, but not the same as, Grounding DINO's language-guided query selection (ECCV 2024). Because it borrows from a second paper and makes the pre-fusion head depend on the text, it is reported only as an analysis (ablation row 7) and is never part of LaGoVAD-MQR.

**Why MQR should help on traffic video.** Six pieces of evidence point the same way:

1. **The motion features are the kind mean pooling hurts most.** Masked-modelling attention spreads over many patches: MAE's aggregation entropy is 5.03 against a maximum of 5.27 (Beyond \[cls\], ICCV 2025). Attentive readouts recover much of what averaging or linear probing loses (Sections 2.3.5, 2.3.6 and 2.4).
2. **Weak labels localise through tile scores, not free queries.** The weakly supervised methods that report spatial results all score tiles or patches directly. "Learning Where and When" (arXiv 2606.29498) gives each of 7×7 patches its own score, lifts a 2×2 window inside the top segments and labels every patch of normal videos 0. In "Localizing to Debias" (arXiv 2608.12045), hard token selection grounds better than soft (patch AP 25.80 against 18.69), even though soft selection gives higher frame AP. In STPrompt, learned attention alone does no better than average pooling (86.67 against 86.91 UCF-Crime AUC), while motion top-K selection followed by attention reaches 88.08. In histopathology MIL, instance-level max pooling localises far better than attention pooling (patch AUCPR 0.690 against 0.388; FocusMIL, arXiv 2408.09449). These are preprints or other domains, and none uses dashcam data.
3. **Traffic anomalies are small and local.** The ego-uninvolved gap of Simple-TAD, PromptTAD's "neglect of distant or off-center objects" and GS-MoE's small-region failure all show that global vectors miss them (Section 2.2).
4. **VideoMAE tokens keep local, moving content; CLIP's do not.** Masked image modelling "preserves and amplifies token-level information" (Park et al., ICLR 2023). CLIP's global embedding favours large objects (CLIP Under the Microscope, CVPR 2025), and contrastive embeddings sit in a narrow cone where even unrelated inputs average about 0.5 cosine (Mind the Gap, NeurIPS 2022). The high cosine between CLIP frame embeddings (above 0.9 in pilot measurements) therefore does not rule out local signal. CLIP's last-layer patches share the same peak dimensions (ClearCLIP, ECCV 2024); dense CLIP tokens can carry anomaly evidence once attention is corrected (AnomalyCLIP, ICLR 2024), but they carry no motion. That is why the readout runs on VideoMAE tokens.
5. **Change inside the clip is where the evidence is.** A collision is a short transient, and the methods that localise under weak labels lean on motion: STPrompt selects patches by feature change over time, and "Localizing to Debias" rewards tokens whose motion is time-asymmetric. Keeping slots and removing the frame-wide component keep exactly that signal, which a time average over the clip dilutes.
6. **Same starting function.** Because MQR starts as the mean-pooled model, any difference after training comes from learned selection and the extra loss, not from a different starting point.

| Property | Mean pooling (preliminary) | MQR |
| --- | --- | --- |
| Extra parameters | — | about 0.25M (ViT-S: tile scorer 0.1M, W\_V 0.15M); 0.8M (ViT-B) |
| Extra extraction compute | — | none: same VideoMAE pass |
| Storage per clip (ViT-S, fp16) | 0.75 KB | about 150 KB (4 slots × 7×7 grid); 301 KB with all 8 slots |
| Spatial map | none | one per step, from the tile scores |
| Extra loss | — | tile-level MIL: space-time top-k on abnormal videos, dense negatives on normal videos |

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

**Shortcut control.** CCD's crash clips and its normal clips (taken from BDD100K) come from different sources, so a model could learn the source instead of the anomaly. Nexar's positive and negative clips both come from Nexar's dashcam network, so P3 is also reported with Nexar-only training. The dense negative term of the tile-level MIL loss makes this control more important, not less: if normal tiles are easy to reject because of their source, the scorer can learn the source.

**Stages.**

1. **Cache features.** Extract CLIP frame features with LaGoVAD's script and the 4×7×7 VideoMAE V2-S token grids of centred clips once, stored as memory-mapped fp16 arrays. Residual and change features are computed on the GPU from the cached grid, so one cache serves mean pooling, MQR and all its ablations.
2. **Train all models with one schedule.** Every row (LaGoVAD, motion stream, MQR, ablations) is trained from the same random initialisation of LaGoVAD's trainable parts, with the same schedule and 3 seeds. MQR starts with the last layer of the tile scorer at zero and W\_V = I.
3. **Warm start as a speed variant.** MQR can also start from a trained motion-stream checkpoint, since it computes the same function at initialisation. This is reported separately and never mixed into ablation rows.

**Settings.** AdamW, batch 64, learning rate 5e-5 with a cosine schedule, 40 epochs, as in LaGoVAD. The temperature parameter θ and, in the free-query ablation, the queries and their scales sit in their own parameter group without weight decay. The loss weight λ ∈ {0.1, 0.5, 1} is chosen once on a held-out 15% of the P3 training videos scored by video-level AUC. The spatial settings, the cell count K ∈ {2, 4, 8} and the sparsity weight β ∈ {0, 0.01}, cannot be judged by video-level AUC, so they are chosen once by the pointing game on the 500-clip DoTA-train pilot subset of Step 0 (Section 4.3). All choices are then frozen for every protocol. No hyperparameter is chosen on a test list.

**Checkpoint selection.** LaGoVAD's released config selects checkpoints on test scores. The thesis instead reports the last epoch of the fixed schedule for every model, baselines included.

**Compute.** A 512-step training sequence of 4×7×7 ViT-S grids is about 77 MB, so a batch of 64 is about 4.9 GB of input, four times a time-averaged 7×7 grid. The residual and change features are formed in bf16 chunk by chunk over steps, with activation checkpointing, and projected to 128 dimensions by the first layer of the scorer, so only the 128-d activations are kept, about a third of the input size; pooling before projecting avoids building W\_V·Q for all 196 cells. One 24 GB GPU should therefore suffice; if the profiling run in the first week shows otherwise, batch 64 is kept through two accumulated steps of 32. Disk is the main cost (Section 4.2).

### 3.4 Inference

At test time the model takes a video and an anomaly definition and returns a score and a 7×7 evidence map per sampled step. No LLM runs at any point.

1. **Sample** one frame in eight (every frame for 10 fps dashcam sets).
2. **Encode** each sampled frame with frozen CLIP, and the 16-frame clip centred on it (frames t−8 … t+7 at 10 fps) with the frozen VideoMAE V2-S, kept as a 4×7×7 token grid.
3. **Score tiles** with MQR's tile scorer on the residual and change features of the grid.
4. **Read** the grid with the score-weighted readout and add the result to the CLIP embedding through the gate.
5. **Score** with LaGoVAD's RoFormer, text co-attention and heads, given the definition text. Each step's score is assigned to the frames it stands for.
6. **Map** each step's evidence E\_t as the per-tile maximum of the cell scores over slots, upsample it to frame size and normalise it per video.

**Definitions.** For comparability, every main result uses LaGoVAD's released definitions: TAD is prompted as {"Normal", "Car Accident"}, and DoTA's "Abnormal" expands to the same car-accident texts. Yet 37 of the 60 abnormal videos on LaGoVAD's TAD list (36 of 60 on UMIL's list) are not accidents. A separate analysis therefore prompts each of TAD's 7 categories with its own hand-written definition; this is prompt engineering, not part of the component.

**Score post-processing.** DoTA and DADA-2000 contain only abnormal clips, so their scores are min-max normalised per video before pooling, following LaGoVAD's DoTA protocol (`offline_dota_eval.py`), which the thesis extends to DADA-2000. TAD, MSAD and CCD scores, which come with normal videos, are used raw. The cell scores p only build the evidence map at test time; adding the spatial score S\_t to the frame score is tested only as an ablation.

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
2. **Cut clips.** For each CLIP step at time t₀, take the 16 frames centred on t₀ (frames t₀−8 … t₀+7 of the 10 fps stream), repeating edge frames at video boundaries. A cache built with causal windows is converted as described in Section 3.1, and the shift test confirms the alignment before training. Resize to 224×224 without cropping, as for the CLIP stream.
3. **Extract CLIP features** with LaGoVAD's `extract_feat_clip.py` for all seven datasets, so that features match the released checkpoint.
4. **Extract and reduce VideoMAE V2-S tokens.** Keep the last-layer tokens (8×14×14×384), average 2×2 spatially and over consecutive pairs of the 8 temporal slots, and store 4×7×7×384 in fp16 (about 150 KB per clip) as one array per video. Residual and change features are computed from this grid at training time and are not stored. The token average reproduces the preliminary motion-stream feature from the same cache, provided that feature is a plain mean of the same tokens, and the time-averaged 7×7 grid of the grid ablation is its mean over slots.
5. **Build labels.** Training uses video-level labels, category indices and the caption templates: TAD from UMIL's `TAD_train.txt`, MSAD from its weak split, CCD from its official split, Nexar from its positive / negative labels (event and alert times are kept for analysis only). Test labels are frame-level: TAD intervals, MSAD frame labels, CCD accident onsets, DoTA anomaly windows and DADA-2000 intervals. Each step's score is copied to the frames it stands for.
6. **Prepare DoTA's spatial labels.** For each clip keep `ego_involve`, the category, and the per-frame boxes of anomalous agents. Bin clips by the median, over the anomaly window, of the largest box's share of the frame: under 0.5%, 0.5–2%, 2–8%, over 8%. AUC per bin uses all frames of those clips, with precursor frames as negatives. The obstacle (VO) and out-of-control (OO) categories have no participant boxes and are left out of STAUC.
7. **Fix test lists.** LaGoVAD's lists (DoTA 1,402 clips; its 100-video TAD list) for zero-shot AUC comparability; DoTA's official 1,305-video list for STAUC; UMIL's TAD test list for P1 and as a second zero-shot column; MSAD's weak-protocol test set; CCD's official test split; Simple-TAD's DADA-2000 test split.
8. **Hold out validation data.** 15% of the P3 training videos, stratified by label and source, for the one-time hyperparameter choice.

**Storage of the token cache** (GB, fp16, VideoMAE V2-S):

| Dataset | 4×7×7 grid (default) | 7×7 grid, time-averaged | Mean vector only |
| --- | --- | --- | --- |
| TAD (all 500 videos) | 10.0 | 2.5 | 0.05 |
| CCD (one clip per frame) | 34.0 | 8.5 | 0.17 |
| Nexar (1.70M frames) | 32.0 | 8.0 | 0.16 |
| DoTA, all clips (one clip per frame) | 110.0 | 27.5 | 0.56 |
| DADA-2000 | 12.4 | 3.1 | 0.06 |

These are estimates. DoTA and CCD use one clip per 10 fps frame; the other sets one clip per eight frames. Only DoTA's 1,402 test clips and a 500-clip training subset are needed, roughly a third of its figure (about 37 GB at the default grid). The 8-slot side ablation doubles the first column, so it is cached only for P1 and the DoTA pilot subset. TAD's quoted duration ("about 25 h") conflicts with its frame count (537,244 frames), so its figure may be off by up to about 5×. MSAD and MM-AU are measured at extraction.

**Leakage checks before any training.**

- Seventy-six of LaGoVAD's 100 TAD test videos are in UMIL's TAD training list, so a model trained on UMIL's split is never scored on LaGoVAD's TAD list.
- MM-AU pools CCD, A3D, DoTA and DADA-2000 videos. It is never used for training, and as a test set only after removing videos shared with CCD, DoTA and DADA-2000.
- Several datasets were collected from online videos, so a near-duplicate check with CLIP-feature similarity runs between every training set (TAD, MSAD, CCD, Nexar) and every test set.
- LaGoVAD's published zero-shot numbers are reproduced from `pred_results.pkl` before anything is trained.

### 4.3 Baseline running

Baselines run in three groups, preceded by one pilot diagnostic that checks MQR's premise before any full training.

**Step 0: pilot diagnostic (inference only, about one GPU-day, on a random 500-clip subset of DoTA's 3,275-clip training split).** It asks whether local motion tokens carry anomaly evidence that the global vectors lose. The 1,402 test clips are never used for design decisions.

- **D0.** Cosine similarity of CLIP embeddings 8 frames apart within a video, against random pairs across videos, raw and mean-centred. This calibrates the above-0.9 observation.
- **D1.** Mean token change inside the anomalous-agent boxes divided by the change outside, by ego-involved / ego-uninvolved clips and box-size bin. Computed for the CLIP global embedding, raw CLIP patches, ClearCLIP patches, VideoMAE tokens at 14×14 and 7×7, and the residual and change features of MQR's 4×7×7 grid.
- **D2.** Training-free AUC of three scores: global change, maximum token change and top-k token change.
- **D3.** A pointing game: how often the token with the largest change falls inside a box, against the box-area chance level. The map of residual-and-change magnitude used here becomes the untrained change map that MQR must beat (H2).
- **D4.** Tile similarity: mean pairwise cosine of the layer-normalised tiles, raw and after the frame-wide median is subtracted. This measures the near-identity of tiles that makes attention over raw tiles hard to train (Section 3.2), and shows how much the residual step removes.
- **D5.** Alignment and position: the shift test of Section 3.1 on the motion-stream model, and a position-only score (each frame's relative position in its clip) fitted on the pilot subset, which becomes a reference row in Section 4.4.

MQR's premise holds if VideoMAE residual and change features differ much more inside boxes than outside during the anomaly window, and if maximum-tile scoring beats global scoring on ego-uninvolved and small-box clips. A ratio near 1 for raw CLIP patches would confirm, with data, that CLIP patches are the wrong place to read.

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
- **DoTA and DADA-2000 AUC:** after per-video min-max normalisation (LaGoVAD's DoTA protocol), reported overall and for ego-involved and ego-uninvolved clips. Clip-averaged AUC depends only on the order of frames within each clip, so it is a within-clip timing measure.
- **Position-controlled AUC:** within a clip, normal and anomalous frames always sit at different positions, so position is controlled across clips. It is reported two ways: the AUC over pairs of a normal and an anomalous frame drawn from different clips at the same relative position (scores min-max normalised per video), and the gain in clip-averaged AUC of the model's score combined with position over the position-only score of Step 0 (D5), the combination fitted on the pilot subset. A time-reversed-clip control checks that a model does not score "later" rather than "abnormal".
- **STAUC:** DoTA's spatio-temporal AUC. For each true-positive frame, TARR is the fraction of the top-N% highest-scoring map pixels (N set per frame, as in DoTA) that fall inside the union of anomalous-agent boxes. STAUC is the area under the TARR-weighted true-positive rate against the false-positive rate, so STAUC ≤ AUC. It is computed on DoTA's official 1,305-video list with `stauc.py`, adapted to take the 7×7 maps upsampled to its 256×256 grid; the unknown categories are analysed with AUC only.
- **Pointing game:** the share of anomalous frames whose map peak falls inside a box. Mass-in-box, the share of the map's total mass inside the boxes, is reported with it. STAUC, pointing game and mass-in-box are always compared with a uniform map, a centre prior and the untrained change map of Step 0 (D3). Attention entropy is logged as a diagnostic, not used as a criterion.

| Protocol | Train → test | Metrics | Comparison rows |
| --- | --- | --- | --- |
| P1 TAD in-domain (fixed camera) | TAD train → TAD test, UMIL split | AUC\_O, AUC\_A, AP | UMIL and WSAL on both metrics; AUC\_O-only rows; VADor flagged; LaGoVAD and VadCLIP retrained |
| P2 MSAD in-domain (multi-scene surveillance) | MSAD weak train → MSAD test; traffic slice and non-traffic scenes scored separately | AUC, AP, AUC\_A, AP\_A | π-VAD and GS-MoE (four metrics) and Alert-CLIP (two), all on full MSAD; RTFM, MGFN, UR-DMU on the traffic slice |
| P3 dashcam | Nexar + CCD train → CCD test (in-domain); DoTA and DADA-2000 (zero-shot) | AUC, AP; position-controlled AUC; DoTA ego-involved / ego-uninvolved AUC, STAUC, pointing game | DoTA: LaGoVAD, VadCLIP, PEL (published AUC, AP re-scored), WS-DoTA flagged; CCD and DADA-2000: released checkpoints re-scored; frame-supervised rows as context |
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
- Subsets (ego-involved / ego-uninvolved, box-size bins, categories, traffic slices) carry 95% bootstrap confidence intervals over videos. MQR against the preliminary model is tested with a paired bootstrap over clips.
- MQR against the mean-pooled model is also tested for equivalence (two one-sided tests, margin ±0.01 AUC), so that a null result is reported as "equivalent" or "inconclusive" rather than as no difference.
- All models use the same test lists, centred clips, frame-to-score mapping and normalisation.

**Decision gates for MQR.** The gates are fixed before any test-list run. Pilots run with one seed and are judged only on the held-out P3 videos and the DoTA-train pilot subset; a variant moves to 3-seed confirmation only if it beats the centre prior and the untrained change map on pointing game, shows no validation drop beyond the seed spread of the mean-pooled model, and is not below mean pooling on position-controlled AUC. MQR counts as supported if, over 3 seeds on the test lists, its gain on position-controlled AUC or on STAUC and pointing game has a 95% paired confidence interval that excludes zero, with no in-domain regression. Otherwise it is reported as a negative result with its diagnostics.

### 4.5 Ablation studies

The ablation ladder adds one ingredient per row on the same features, data and schedule, so each step isolates one design choice. Rows 0–7 report P1 (AUC\_O, AUC\_A), P2 (AUC, AP), P3 (CCD AUC; DoTA AUC and position-controlled AUC by ego-involved / ego-uninvolved clips; DADA-2000 AUC), P4 on TAD's two lists, and STAUC and pointing game where a map exists.

| Row | Variant | Isolates | Expected if MQR's premise holds |
| --- | --- | --- | --- |
| 0 | LaGoVAD, CLIP only | Baseline | Reference |
| 1 | + motion stream, mean pooling | Backbone change | Gain overall, smallest on ego-uninvolved clips |
| 2 | Max or GeM pooling over the 196 cells | More tokens, no learning | Small or no gain over row 1 |
| 3 | Untrained change top-K: readout weights from the residual-and-change magnitude, top-K cells kept | A fixed motion ranking, in the spirit of STPrompt | Some gain over row 2; map above the centre prior |
| 4 | Free-query readout (Efficient Probing, repaired), M ∈ {1, 4, 8}, with a per-query spatial MIL loss | Learned queries | No better than row 3, as in STPrompt's ablation |
| 5 | Tile scorer with score-steered readout; MIL on abnormal videos, normal videos as bag negatives only | Tile-level scoring | Gain over rows 3–4; map above row 3 |
| 6 | Row 5 + dense negatives on every tile of normal videos (**full MQR**) | Tile-level negatives | Best detection and best maps; largest gain on ego-uninvolved and small-box clips |
| 7 | Row 6 + definition-conditioned tile scoring (exploratory, outside LaGoVAD-MQR) | Language-guided "where" | Higher sensitivity to the definition; detection at least equal |
| 8 | Row 6's component on raw CLIP patches, and on ClearCLIP patches | Whether CLIP patches could replace motion tokens | Below row 6; raw CLIP patches near row 0 |

Row 4 tests, on the same data and schedule, whether free queries, the readout of Efficient Probing, learn region selection from video labels at all; rows 5 and 6 test what tile-level scoring and tile-level negatives add on top of it.

**Side ablations within the component.**

- **Grid:** 4×7×7 (default); all 8 slots; a 7×7 grid averaged over all slots; and 14×14 with tokens PCA-compressed to D/4, whose zero-score start is the mean of the compressed tokens.
- **Features:** residual only, change only, both (default); median against mean as the frame-wide reference; no frame-wide subtraction.
- **Loss form:** λ ∈ {0.1, 0.5, 1}; K ∈ {2, 4, 8} cells against the best 2×2 window; dense against hard-negative (top-K) normal-video terms; β ∈ {0, 0.01}; with and without the synthetic-video negatives.
- **Readout:** score-steered readout (default) against a stop-gradient from the main loss into the scorer; learnable against fixed temperature.
- **Gate:** α\_init = 0 (default) against 0.25.
- **Tile interaction:** one self-attention layer over the 196 cell features before scoring, a detector-free stand-in for agent interaction; Structured Graph Reasoning (IEEE TMM 2026) found interaction scores localise better than per-object scores.
- **Free-query row:** zero, random and k-means initialisation; hard max against log-sum-exp over queries; with and without the overlap penalty.
- **Clip window:** centred (default) against causal, to measure what the alignment fix is worth.
- **Score fusion:** adding the spatial score S\_t to the frame score at inference.
- **Encoder:** VideoMAE V2-S (default) against VideoMAE-S (Kinetics-400) and the driving-adapted DAPT-VideoMAE-S (BDD100K only), to show MQR's gain does not depend on one encoder.

**Analyses.**

1. **Where the gain lands.** DoTA ego-involved vs ego-uninvolved clips crossed with clip-level box-size bins, including PromptTAD's small-object subset (334 test clips with anomalous boxes under 2,267 pixels); the "unknown" categories (UK, UK\*); TAD AUC for Accident, PedestrianOnRoad and the remaining categories pooled; the MSAD traffic slice against MSAD's non-traffic scenes; Nexar-only against Nexar + CCD training.
2. **Spatial evidence.** STAUC, pointing game and mass-in-box for MQR against a uniform map (row 1), a centre prior and the untrained change map (row 3), plus maps of successes and failures.
3. **Selection behaviour.** Readout attention entropy and tile-score sparsity over training, which tiles and slots fire on which categories, and how often the highest tile is a salient but normal agent; for row 4, per-query entropy and overlap between queries' maps.
4. **Language.** For row 7, how much the score of accident frames changes when the definition includes or excludes accidents, and drift@5 on MSAD.
5. **TAD definitions.** Main results with LaGoVAD's two-class TAD prompt, against each of TAD's 7 categories prompted with its own hand-written definition.
6. **Cost.** Parameters, storage and measured frames per second for rows 0, 1 and 6.

**Budget.** Rows 0–7 run on P1, P2 and P3 with 3 seeds; P4 reuses those models. Row 8 and the side ablations run on P1 and P3 only, first with one seed, then with three for the variants that are kept. Row 8 also needs its own dense CLIP-patch cache, about twice the time-averaged 7×7 column in Section 4.2. A one-epoch timing run in the first week converts this plan into GPU-hours; if the total exceeds the budget, the ladder is cut to rows 0, 1, 3, 4 and 6.

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
- [FocusMIL](https://arxiv.org/abs/2408.09449), arXiv 2408.09449 (instance-level max pooling against attention pooling under weak labels)
- Henry et al., [Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245), Findings of EMNLP 2020 (cosine logits with a learnable scale)

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
