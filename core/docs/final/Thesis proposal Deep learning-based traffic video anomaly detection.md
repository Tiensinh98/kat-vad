# Thesis proposal: Deep learning-based traffic video anomaly detection (LaGoVAD-OSTP)

Nov 2, 2026 · @Sinh Pham

This revision adds a **YOLO object branch** to LaGoVAD-STP and fuses it with the **7×7 crop-window CLIP patch features** of SA², never with the global CLIP frame feature. The new model, **LaGoVAD-OSTP** (object-prompted spatio-temporal prompting), keeps the whole weakly supervised, LLM-free design of the previous proposal: LaGoVAD as host, a frozen VideoMAE V2-S motion vector (zero-shot DoTA **0.81 macro / 0.75 micro** already measured), SA² and the learnable text prefix.

The fusion is adapted from **PromptTAD** (IEEE RA-L 2025), which turns YOLOv9 boxes into SAM-style prompt tokens and fuses them with a dense frame embedding through instance-wise and relation-wise cross-attention. Three things change so the idea fits this thesis:

- **Fusion partner.** PromptTAD fuses objects with a Video Swin + FPN feature map. Here the partner is the 7×7 grid of window-crop CLIP features, a dense, text-aligned map whose cells can be matched to boxes exactly. The global CLIP vector c\_t is left untouched.
- **An explicit object–patch fit.** Boxes and patches share one Fourier positional encoding; a box-to-window overlap matrix gives each object a CLIP appearance pooled from the patches under it and a geometric bias on its attention logits; and an agreement loss ties object scores to SA²'s patch weights. The model starts at a geometry-correct pooling and learns from there.
- **Weak supervision instead of boxes.** PromptTAD's instance loss needs ground-truth anomalous boxes and Hungarian matching. It is replaced by an object-level MIL loss (every object in a normal video is a negative; the most anomalous object of the top-k steps of an abnormal video is a positive) plus the object–patch agreement loss.

**Claim under test.** Detected objects tell the patch branch *which* cells hold agents, the patch branch tells each object *what it looks like* in CLIP space and what surrounds it, and their fusion recovers small, distant and ego-uninvolved anomalies better than motion-selected patches alone, while turning SA²'s coarse 7×7 maps into box-level evidence scored with DoTA's STAUC.

| Part | LaGoVAD-STP (Nov 1) | LaGoVAD-OSTP (this proposal) |
| --- | --- | --- |
| Host, losses, schedule | LaGoVAD | Unchanged |
| Motion stream | VideoMAE V2-S, one 384-d vector, gated | Unchanged |
| CLIP global feature c\_t | + SA² output | + SA² output over **object-enhanced** patches |
| Object branch | none | YOLO boxes → prompt tokens → two-way fusion with the 7×7 patch grid → relation token |
| Text branch | Learnable prefix (l = 8) | Unchanged; also scores object tokens against the definition |
| Extra losses | none | Object MIL + object–patch agreement |
| Spatial output | 7×7 SA² map | Per-object anomaly scores (box-level map) + 7×7 map |

## 1. Introduction

### 1.1 Problem definition

Traffic video anomaly detection scores every frame of a road video by how likely an abnormal event (collision, loss of control, illegal turn, wrong-way driving, pedestrian on the road, road spill) is happening in it. Two camera settings define the problem: **fixed camera** (TAD, MSAD traffic scenes), where agents are often small and far away, and **dashcam** (DoTA, DADA-2000, CCD, Nexar), where the camera moves and DoTA marks each clip as ego-involved or ego-uninvolved.

The thesis keeps LaGoVAD's weakly supervised, language-guided formulation: one video-level label y\_i ∈ {0, 1} per training video, a category and a hand-written caption template. At test time the model receives a video V and a text definition Z and returns a score per sampled step. This version adds a second spatial output: a set of scored object boxes per step, beside the 7×7 SA² map.

```latex
\Phi_\theta : (V, Z, \{B_t\}) \;\mapsto\; \{ s_t \}_{t=1}^{N},\; \{ E_t \}_{t=1}^{N},\; \{ (b_{t,i}, s^{O}_{t,i}) \}_{t,i}, \qquad s_t, s^{O}_{t,i} \in [0,1],\; E_t \in [0,1]^{7 \times 7}
```

Here B\_t is the set of boxes a frozen YOLO detector finds in the sampled frame t; it is computed from V, so the model still needs nothing but the video and the definition. Training uses multiple-instance learning (MIL): the mean of the top-k step scores of a video must match its label, and, new here, the top object scores must too.

Four properties make traffic video hard, and each motivates a part of the model:

1. **Anomalies are motion events** → the VideoMAE motion stream (kept).
2. **The agents that matter are small.** A distant agent of about 40×30 px in a 1280×720 frame becomes about 7×9 px at 224×224, smaller than one 16-px CLIP patch (estimate) → SA² windows now cut from the native frame, so the agent keeps about 1,200 real pixels instead of about 63, and a detector run at native resolution that finds the agent before any resizing.
3. **The camera moves.** Under ego-motion, SA²'s second-order patch change is largest on near-field road and frame edges, which hold no agents → object boxes tell the patch branch which cells contain agents.
4. **Labels are scarce** → every new part trains from video-level labels; detector boxes are inputs, never targets.

Scope is unchanged: offline detection, no LLM or MLLM at training or test time, frozen CLIP text encoder allowed, all text hand-written. A frozen, COCO-pretrained YOLO is allowed as a perception tool, like the frozen CLIP and VideoMAE encoders.

### 1.2 Research purpose

The purpose is a traffic anomaly detector that learns from video-level labels, sees motion, attends to the agents in the frame, and shows which agent is anomalous. The thesis builds it in three steps, keeping the rule of one novel component on a possibly changed backbone:

- **Backbone change (motion stream).** Frozen VideoMAE V2-S, one 384-d vector per step through a zero-initialised gate. Zero-shot DoTA **0.81 / 0.75**. MQR and LSP, two attempts to read more from it, added nothing.
- **Novel component, visual part (STP).** SA² over 7×7 window-crop CLIP patches plus a learnable text prefix.
- **Novel component, object part (new, OPF).** Object–Patch Fusion: YOLO boxes become prompt tokens fused two-way with the 7×7 patch grid, with an explicit geometric fit and two weak losses. OPF and STP form one component: OPF changes the patches SA² reads and the cells it selects.

### 1.3 Research aims

1. **Reproduce the baseline** (LaGoVAD released and retrained) with AUC, AP and AUC\_A on TAD, MSAD, DoTA and DADA-2000.
2. **Document the motion stream** and the negative MQR and LSP results.
3. **Design and train OSTP**: SA², the learnable prefix and Object–Patch Fusion, with video-level labels only.
4. **Measure spatial localisation under weak supervision**: STAUC and pointing game for the 7×7 map and for the per-object scores, against DoTA's anomalous-agent boxes.
5. **Explain the gains** by ego flag, box size, category, detector recall and illumination.

"Supported" means the 3-seed mean difference is positive and its 95% paired bootstrap confidence interval over test videos excludes zero.

| Hypothesis | Measured by | Supported if |
| --- | --- | --- |
| H1. SA² improves on the motion-stream model, mostly on small and ego-uninvolved agents | DoTA macro / micro AUC by ego flag and box size; TAD AUC (rows R0, R4) | R4 > R0, larger gains on ego-uninvolved and small-box clips |
| H2. Motion-prior selection matters on traffic video | STPrompt's ladder R1–R4; raw vs ego-compensated motion (R7) | R3 and R4 beat R1 and R2; on dashcams R7 ≥ R6 |
| H3. The learnable prefix improves transfer without weakening language guidance | Zero-shot DoTA, DADA-2000, TAD; MSAD drift@5 (R5, R6) | R6 > R4 on zero-shot AUC, drift@5 no worse |
| H4. The maps point at the anomalous agents | STAUC and pointing game on DoTA's 1,305-video list | Beats a uniform map and a centre prior |
| **H5. Objects fused with the patch grid beat objects fused with the global CLIP vector** | O-ladder: object token added to c\_t (O1) vs OPF on patches (O4–O6) | O6 > O1 overall and on the small-object subset |
| **H6. The object–patch fit is what makes the object branch learn** | O-ladder with and without shared PE, box bias, RoI appearance and agreement loss; attention-entropy and gate logs | Each fit part adds AUC or STAUC; OPF's gate and attention move away from their start, unlike MQR |
| H7. Gains hold across camera settings without hurting surveillance | TAD AUC\_O / AUC\_A; MSAD AUC / AP, traffic slice and non-traffic scenes | Gains on TAD, MSAD traffic and dashcam sets; no significant drop on MSAD non-traffic |

## 2. Literature review

The review of the Nov 1 proposal (VAD branches, 2024–2026 trends, open challenges, LaGoVAD, VadCLIP, VideoMAE V2, Simple-TAD, STPrompt and the comparison rows) stands unchanged and is not repeated here. This revision expands one work, PromptTAD, because its object–frame fusion is the template for the new branch.

### 2.1 PromptTAD (IEEE RA-L 2025): how it fuses objects and frames

Qiu, Yang and Gong add detected objects to MOVAD, a frame-level dashcam detector (Video Swin Transformer + LSTM). Their diagnosis: frame-level features are distracted by dynamic backgrounds and, because the global view favours prominent regions, they miss distant or off-centre objects. Object-centric methods avoid this but lose scene context. PromptTAD takes both, using boxes as prompts in the way SAM uses box prompts.

**Pipeline.**

1. **Dense frame embedding.** A Video Swin Transformer with an FPN gives a multi-scale map averaged over time, f\_t ∈ R^{H×W×D\_E} at stride 8 (D\_E = 256). The FPN was added because patch merging loses small objects.
2. **Object prompt encoder (SAM style).** YOLOv9 detects traffic objects in the last frame of each 4-frame clip. Each box's top-left and bottom-right corners are encoded with Fourier positional encodings and a learnable token v\_i is added: o\_i = PE(tl\_i) + PE(br\_i) + v\_i. A learnable "no prompt" token stands in when nothing is detected.
3. **Instance-wise aggregation (L\_I = 4 blocks, SAM two-way style).** Each object token passes through self-attention, then cross-attends to the frame embedding (object reads scene), an MLP, then the frame embedding cross-attends to the object tokens (scene reads objects). Output: refined object tokens and an instance-enhanced frame embedding.
4. **Relation-wise aggregation (L\_R = 4 blocks, N\_K = 12 learnable queries).** The queries read the object tokens, then the average of the instance-enhanced frame embedding, then the object tokens are updated from the queries. Output: an enhanced frame token (from the queries) and enhanced object tokens.
5. **Injection.** The enhanced frame token is concatenated with the original pooled frame token before the LSTM and the frame regressor; the enhanced object tokens go to an instance regressor.
6. **Losses.** A weighted frame cross-entropy (frame labels) plus an instance loss: YOLO boxes are Hungarian-matched to ground-truth anomalous boxes, matched boxes are positives and unmatched ones negatives, combined with weight α = 0.7.

**Evidence.** 83.9 AUC on DoTA with the best STAUC among compared methods; each part helps, more on the 334-video small-object subset (anomalous boxes under 2,267 px); relation-wise aggregation clearly beats a variant that only concatenates instance-enhanced frame tokens; the largest gains are on ego-uninvolved anomalies, and DoTA → DADA-2000 transfer without fine-tuning beats TTHF, again most on the ego-uninvolved set.

**Limits.** It needs frame labels and anomalous-object boxes. Its fusion partner is a learned, non-text-aligned feature map, and the only link between a box and the map is positional encoding learned end to end, with frame and box supervision to guide it. When the detector misses (dark scenes, slow reversing) it degenerates to the baseline. No public code.

### 2.2 What this thesis takes from PromptTAD, and what it must change

| PromptTAD element | Taken | Changed for weak supervision and the CLIP patch grid |
| --- | --- | --- |
| YOLO boxes as SAM-style prompts (corner PE + learnable token) | Yes | Add class embedding, detector confidence, box dynamics across steps and a CLIP appearance pooled from the patches under the box |
| Fusion partner: dense frame embedding (VST + FPN, stride 8) | Idea of a dense partner | The 7×7 window-crop CLIP grid of SA², text-aligned and box-addressable; the global c\_t is not a partner |
| Positional link between boxes and map | Shared PE | One Fourier PE shared by box corners and patch centres (as in SAM), plus an explicit box-to-window overlap bias on attention logits |
| Instance-wise two-way aggregation | Yes, 2 blocks | Zero-initialised output projections so patches start unchanged and SA² starts as in STP |
| Relation-wise aggregation with 12 queries | Yes, 2 blocks | Output enters the step feature through a zero-initialised gate, in CLIP space |
| Concatenate enhanced and original frame tokens | No | Enhanced patches feed SA²; the relation token is added to the CLIP-branch feature |
| Instance loss with Hungarian matching to GT boxes | No | Object-level MIL from video labels + object–patch agreement loss |
| Frame-level weighted cross-entropy | No | LaGoVAD's top-k MIL and alignment losses, unchanged |

The key lesson: PromptTAD's objects learn because two dense signals (frame labels and box labels) reach them directly. Under video labels the thesis must replace these with (a) a structural fit that is correct before any training and (b) a loss that reaches the object tokens without passing through a closed gate. The previous proposal's MQR failed for exactly this reason: its queries started at a solution the host already had and got almost no gradient.

### 2.3 Research gaps and novelty

The four gaps of the Nov 1 proposal stand (no motion-aware foundation features in weakly supervised traffic VAD; one global appearance vector per frame; no spatial evidence under weak supervision on traffic; inconsistent evaluation). One gap is added:

5. **Object evidence needs frame or box labels.** Object-prompt methods for traffic (PromptTAD, and the object-centric line before it) are fully supervised; Alert-CLIP needs boxes and region captions. No weakly supervised, language-guided traffic detector fuses detector boxes with dense CLIP evidence. *Addressed by OPF with object-level MIL.*

| Closest work | What it does | How LaGoVAD-OSTP differs |
| --- | --- | --- |
| PromptTAD (IEEE RA-L 2025) | YOLO box prompts fused with a VST + FPN map; frame and box labels | Video labels only; fused with text-aligned CLIP window patches; explicit geometric fit; open-world text host; motion stream |
| STPrompt (ACM MM 2024) | SA² over motion-selected CLIP patches; surveillance | Objects refine the patches and the selection; traffic and dashcam data; STAUC |
| Alert-CLIP (CVPR 2026) | Region-level CLIP alignment trained with boxes and captions | Boxes only as frozen-detector inputs, no box or caption supervision |
| VadCLIP, LaGoVAD | One global CLIP vector per frame | Local, object-addressed evidence |

The claim, to our knowledge: **the first weakly supervised, language-guided traffic anomaly detector that fuses detector object prompts with motion-selected CLIP patch evidence through an explicit object–patch fit, and the first report of object-level STAUC under weak supervision.** Box prompts, SA² and learnable prompts are not claimed as new; the contribution is their fit under video labels, on traffic data, with a motion stream. The two 2026 preprints flagged before ("Localizing to Debias", "Learning Where and When") still need a full read before submission.

## 3. Proposed model

LaGoVAD-OSTP is LaGoVAD with three additions: the VideoMAE motion stream (backbone change), STP (SA² and the learnable prefix) and Object–Patch Fusion (OPF). OPF sits between the patch features and SA², so the order inside the CLIP branch is: window-crop patches → OPF (objects ⇄ patches) → SA² → sum with c\_t. Everything else is LaGoVAD as released.

### 3.1 Base architecture (unchanged)

- **Sampling.** One frame in eight (`interval=8`); every frame for 10 fps sets (DoTA, CCD).
- **CLIP stream.** Frozen CLIP ViT-B/16, one 512-d CLS embedding c\_t per sampled frame (224×224 resize, no crop).
- **Temporal encoder.** Residual RoFormer, 2 layers, 4 heads, hidden 512, local window 25; sequences resampled or padded to 512 steps.
- **Text branch.** Frozen CLIP text encoder; each category of Z → τ\_c ∈ R^512.
- **Fusion and heads.** 2-layer video–text co-attention; 1-D conv head (kernel 9) on pre- and post-fusion features, blended by w = σ(10·a), a₀ = 0.
- **Losses.** Top-k MIL BCE (k = max(⌊L/16⌋, 1)), multi-class MIL alignment, dynamic video synthesis, caption contrastive loss (temperature 0.02).
- **Motion stream.** 16 frames at 10 fps centred on t → frozen VideoMAE V2-S (`vit_s_k710_dl_from_giant`) → mean of 1,568 tokens v\_t ∈ R^384, fused as x\_t = c̃\_t + tanh(α)·W·LN(v\_t), α₀ = 0.

### 3.2 STP (unchanged from the Nov 1 proposal)

**Patch grid (native-resolution windows, default).** Each sampled frame is cut into a 7×7 grid at its **native resolution** (cells of about 183×103 px for a 1280×720 frame); each cell is resized to 224×224 and encoded by the frozen CLIP, and its CLS embedding is P\_{t,j} ∈ R^512, j = 1…49. Window j covers the normalised rectangle g\_j = \[u\_j, u\_j + 1/7\] × \[v\_j, v\_j + 1/7\] of the frame. STPrompt instead cuts 32×32 windows from the 224×224 resize and upsamples each 7×: a 40×30 px agent then reaches CLIP as about 7×9 real pixels blown up to about 49×63. In a native cell the same agent also spans about 49×65 px of the CLIP input, but carries about 1,200 real pixels instead of about 63. The cost is the same 49 CLIP passes, and the normalised cell geometry is identical, so the overlap matrix, motion magnitude and storage do not change. Cells are squashed to square, as CLIP's own full-frame resize already does. STPrompt's 224-resize mode is kept as a side ablation for fidelity. PromptTAD's FPN upsampling is not a substitute: it restores detail lost inside a backbone that saw a 640×480 input, while here the detail is lost by the resize before CLIP, and no feature upsampling can bring it back.

**Motion and selection** (STPrompt Eq. 1), on L2-normalised patches:

```latex
Mo_{t,j} = \lVert 2\hat P_{t,j} - \hat P_{t-1,j} - \hat P_{t+1,j} \rVert_2, \qquad \mathcal{S}_t = \operatorname{TopK}_j(Mo_{t,j}),\; K = 12
```

**Attention aggregation and sum**, with the CLS feature as query, W\_q zero-initialised so SA² starts as "selection + average":

```latex
a_{t,k} = \operatorname{softmax}_{k \in \mathcal{S}_t}\!\left( \tfrac{(W_q c_t)^{\top} (W_k P'_{t,k})}{\sqrt{d_a}} \right), \qquad x^{AS}_t = \sum_{k \in \mathcal{S}_t} a_{t,k}\, P'_{t,k}, \qquad \tilde c_t = c_t + x^{AS}_t + \tanh(10\gamma)\, W_r\, r_t
```

The only differences from the Nov 1 version are the native-resolution windows and three marks of OPF: SA² reads the object-enhanced patches P′ (Section 3.3.4), may use an object-aware selection score (Section 3.3.5), and the relation token r\_t enters through the gate tanh(10γ), γ₀ = 0. The factor 10 follows LaGoVAD's own blend weight σ(10·a) and is needed for the gate to open within Adam's step budget (Section 3.5.1). With OPF switched off, P′ = P, r\_t is absent, and the model is LaGoVAD-STP with native-resolution windows.

**Learnable prefix.** l = 8 shared context vectors prepended to each category's tokens before the frozen text encoder: τ\_c = TextEnc(\[e\_1 … e\_l, t\_c\]). Applied to definitions only, not to training captions.

**Ego-compensated motion** (ablation): Mo^ego\_{t,j} = ‖D\_{t,j} − mean\_i D\_{t,i}‖₂.

### 3.3 New: Object–Patch Fusion (OPF)

OPF turns YOLO boxes into object tokens and fuses them two-way with the 49 window-crop CLIP patches before SA² reads them. It follows PromptTAD's three stages (prompt encoder, instance-wise aggregation, relation-wise aggregation) but replaces PromptTAD's learned-only link between boxes and the feature map with an explicit **object–patch fit** that is correct before training: shared positional encoding, a box-to-window overlap matrix, RoI appearance pooled from the patches, and a geometric attention bias. Every OPF output is added through a zero-initialised projection or gate, so an untrained OPF leaves LaGoVAD-STP unchanged.

**Why the patch grid and not the global CLIP vector.** A box and a global vector share no coordinates: fusing them (as an object token added to c\_t) can only say "there are cars somewhere". The 7×7 grid has coordinates: window j covers a known square g\_j of the frame, so a box can be matched to the exact cells it overlaps. This gives each object a CLIP appearance (from its cells), gives each cell an object identity (from the boxes covering it), and lets SA² prefer cells that hold agents. The patches are CLIP CLS embeddings, so the fused result stays in the space of the definition texts. Row O1 of the ablation fuses objects with c\_t instead, to test this claim (H5).

&#91;embedded content: LaGoVAD-OSTP CLIP branch · object–patch fusion feeding SA²\]

Boxes and window crops meet only through the parameter-free fit and the two-way fusion; the global c\_t reaches the sum unchanged, and the dashed link is the agreement loss between SA²'s attention and the object scores.

#### 3.3.1 Detection

- **Detector.** Frozen YOLOv9-C (COCO), as in PromptTAD, run once per sampled frame at native resolution (input size 1280 by default, 640 as an ablation), so a 40×30 px agent is still about 40×30 px for the detector while it is about 7×9 px for CLIP.
- **Classes.** Person, bicycle, car, motorcycle, bus, truck. Confidence ≥ 0.25 after NMS; at most N\_max = 16 boxes per frame, highest confidence first.
- **Tracking.** ByteTrack links boxes across sampled steps to give track IDs, used only for box dynamics (3.3.3).
- **Coordinates.** Boxes are normalised to \[0, 1\] in the frame. CLIP's 224×224 resize has no crop, so normalised box and window coordinates live in the same frame.
- **Empty frames.** A learnable "no object" token, as in PromptTAD.

#### 3.3.2 The object–patch fit: overlap matrix and RoI appearance

For box b\_i and window g\_j, two overlap weights are computed once, with no parameters:

```latex
M_{t,ij} = \frac{|b_i \cap g_j|}{|g_j|} \;\;(\text{share of window } j \text{ covered}), \qquad \bar A_{t,ij} = \frac{|b_i \cap g_j|}{|b_i|} \;\;(\text{share of box } i \text{ in window } j), \qquad C_{t,j} = \min\big(1, \textstyle\sum_i M_{t,ij}\big)
```

C\_t is the object coverage map of the grid. The **RoI appearance** of object i is the CLIP feature of the cells it falls in, weighted by how much of the box each cell holds:

```latex
u_{t,i} = \sum_{j=1}^{49} \bar A_{t,ij}\, P_{t,j} \in \mathbb{R}^{512}
```

A small box inside one window takes that window's feature; a large box averages its cells. With native-resolution windows the agent keeps its real pixels, but a 40×30 px agent still fills only about 6% of its 183×103 px cell, so u\_{t,i} also carries road and background. The box coordinates therefore enter in two other places: the geometric bias of 3.3.4 and the object-aware selection of 3.3.5. A dedicated box crop (each box enlarged 1.5×, at least 64 px on its short side, cut from the native frame and encoded by CLIP; at most 16 extra passes per frame) is ablation O9, and becomes the default if Step 0 check S6 shows it clearly better than u.

#### 3.3.3 Tokens and the shared positional encoding

One random-Fourier positional encoding G: \[0,1\]² → R^256 (SAM's encoder, a fixed Gaussian matrix with sin and cos) encodes both box corners and window centres, so a box and the windows it covers have similar codes before training. Internal width d = 256, as in PromptTAD.

```latex
o^{(0)}_{t,i} = G(\mathrm{tl}_i) + G(\mathrm{br}_i) + E_{\mathrm{cls}}[k_i] + W_s\,[\kappa_i, \log|b_i|, \log\tfrac{w_i}{h_i}] + W_a\,\mathrm{LN}(u_{t,i}) + W_d\,\delta_{t,i}
```

```latex
p^{(0)}_{t,j} = W_p\,\mathrm{LN}(P_{t,j}) + G(\mathrm{centre}_j) + w_m\, Mo_{t,j}
```

k\_i is the YOLO class, κ\_i the confidence, and δ\_{t,i} ∈ R^6 the track's centre shift and log-scale change to steps t−1 and t+1, minus the median shift of all tracks in the frame (a cheap ego-motion compensation; zero for untracked boxes). The patch token carries its CLIP feature, its position and its SA² motion magnitude, so the fusion sees what the selection sees.

#### 3.3.4 Instance-wise two-way fusion (L\_I = 2 blocks)

Each block, following PromptTAD and SAM's two-way transformer:

1. **Object self-attention** across the objects of the frame (who is near whom).
2. **Object → patch cross-attention** with a geometric bias. Object i reads the patches, and the logits are biased toward the cells its box covers, using Ā′ computed for the box enlarged 2× so nearby context cells are reachable:

```latex
\mathrm{logit}_{ij} = \frac{(W^{Q} o_i)^{\top}(W^{K} p_j)}{\sqrt{d}} + \beta \log\!\big(\bar A'_{t,ij} + \epsilon\big), \qquad \beta_{\mathrm{init}} = 1,\; \epsilon = 10^{-2},\; W^{Q}_{\mathrm{init}} = 0
```

With W^Q zero at the start, the attention equals (Ā′ + ε) normalised: plain RoI pooling over the box and its surroundings. Training can then sharpen it (β up), widen it (β down) or move it to informative cells (W^Q).

3. **MLP** on the object tokens.
4. **Patch → object cross-attention** with the transposed bias β′ log(M\_{t,ji} + ε) and one learnable null key, so a cell reads the objects covering it and an empty cell reads the null key.

The patch update returns to CLIP space through a zero-initialised projection:

```latex
P'_{t,j} = P_{t,j} + \tanh(10\, g_P)\, W_{\uparrow}\,\big(p^{(L_I)}_{t,j} - p^{(0)}_{t,j}\big), \qquad W_{\uparrow} \in \mathbb{R}^{512 \times 256} \text{ Xavier},\; g_{P,\mathrm{init}} = 0,\; \beta = \exp(4 b_\beta),\; b_{\beta,\mathrm{init}} = 0
```

At step 0 the gate is closed, so P′ = P and SA² behaves as in LaGoVAD-STP. Only one value per path starts at zero: W↑ is random and the scalar gate g\_P carries the closed start. A zero W↑ behind a zero W^Q would give a second-order-slow start, the shape that kept MQR at its initial values. The gate's gradient at zero is the full signal of the projected update, and with the ×10 scaling and ×10 learning rate of Section 3.5.1 it can reach tanh(10 × 0.44) ≈ 1 within one cosine schedule, instead of at most about 0.09. β is parameterised as exp(4 b\_β) for the same reason: it starts at 1 and can move to roughly 0.2–5.8 rather than 0.91–1.09.

#### 3.3.5 Object-aware selection (fixed prior)

SA² keeps its fixed motion prior, but cells that hold detected agents are favoured:

```latex
Mo^{\mathrm{obj}}_{t,j} = Mo_{t,j} \cdot \big(\lambda_0 + (1 - \lambda_0)\, C_{t,j}\big), \qquad \lambda_0 = 0.3, \qquad \mathcal{S}_t = \operatorname{TopK}_j\big(Mo^{\mathrm{obj}}_{t,j}\big)
```

A moving cell with no agent keeps 30% of its motion score, so road spills, debris and missed detections can still be selected. This is the main remedy for ego-motion: near-field road changes most under forward motion but rarely holds a box. λ₀ is fixed, not learned, following STPrompt's lesson that a fixed prior does most of the work. Whether it is the default is decided by the Step 0 pilot (S5).

#### 3.3.6 Relation-wise aggregation and object scores

**Relation aggregation (L\_R = 2 blocks, N\_K = 12 queries)**, as in PromptTAD: the queries read the object tokens, then the mean of the enhanced patch tokens, then the object tokens read the queries. The mean query, projected to 512-d, is the relation token r\_t, added to c̃\_t through the gate tanh(γ), γ₀ = 0 (Section 3.2).

**Object scores.** Each final object token gives an anomaly score from a learned head and from the definition text, so objects stay language-guided like the frames:

```latex
s^{O}_{t,i} = \sigma\Big( h(o'_{t,i}) + \tfrac{1}{T_o}\big[\max_{c \in Z_{\mathrm{abn}}} \cos(W_t o'_{t,i}, \tau_c) - \cos(W_t o'_{t,i}, \tau_{\mathrm{normal}})\big] \Big), \qquad T_o = 0.07
```

h is a 2-layer MLP (as PromptTAD's instance regressor) and W\_t ∈ R^{512×256} is initialised so that W\_t o′ ≈ u at the start (least-squares fit on a cached batch), so the text term begins as plain CLIP zero-shot scoring of each object's RoI appearance.

#### 3.3.7 Size and cost

| Item | Value (estimate) |
| --- | --- |
| OPF parameters | about 8M (2 two-way blocks ≈ 2.6M, 2 relation blocks ≈ 4.7M, projections ≈ 0.6M) |
| Detection compute | one YOLOv9-C pass per sampled frame (about 650K frames for Nexar + CCD + DoTA test), once, cached |
| Storage per sampled frame | ≤ 16 boxes × (4 coords, class, confidence, track ID) ≈ 0.2 KB |
| Extra CLIP passes | none (RoI appearance reuses the 49 window crops); ablation O9 adds ≤ 16 box crops |
| Training cost | small next to extraction: N\_O ≤ 16 tokens × 49 patches per step, cut to the 12 selected patches + covered cells when memory is tight |

### 3.4 Losses: making the object–patch fit learnable under video labels

The object branch gets its own gradient from two weak losses, so it does not depend on a closed gate the way MQR did. PromptTAD's instance loss (Hungarian matching to ground-truth anomalous boxes) is not used, because the training sets have no such boxes and the thesis uses video labels only.

```latex
\mathcal{L} = \mathcal{L}_{\mathrm{LaGoVAD}} + \lambda_O\, \mathcal{L}_{\mathrm{obj}} + \lambda_N\, \mathcal{L}_{\mathrm{neg}} + \lambda_A(e)\, \mathcal{L}_{\mathrm{agree}}
```

**1. Object-level MIL (L\_obj).** The step's object score is its most anomalous object, m\_t = max\_i s^O\_{t,i}. As in the host's MIL, the mean of the top-k step scores of a video must match its label:

```latex
\mathcal{L}_{\mathrm{obj}} = \mathrm{BCE}\Big( \tfrac{1}{k} \textstyle\sum_{t \in \mathrm{top}k(m)} m_t,\; y \Big), \qquad k = \max(\lfloor L_O/16 \rfloor, 1)
```

L\_O counts steps with at least one detection; steps without detections are left out of this loss.

**2. Dense object negatives (L\_neg).** Every object in a normal video, and every object in the normal segments that dynamic video synthesis splices around an abnormal clip, is normal. This gives thousands of reliable object-level negatives per epoch, the weak-label counterpart of PromptTAD's unmatched boxes:

```latex
\mathcal{L}_{\mathrm{neg}} = -\frac{1}{|\mathcal{N}|} \sum_{(t,i) \in \mathcal{N}} \log\big(1 - s^{O}_{t,i}\big)
```

The positives stay weak (one object per top-k step), but the negatives teach the branch what parked cars, crossing pedestrians and ordinary traffic look like, which is most of what it sees.

**3. Object–patch agreement (L\_agree), the learned part of the fit.** The two branches give two estimates of *where* the anomaly is. The object branch spreads its scores over the cells its boxes cover; SA²'s attention, computed over all 49 cells for this loss, gives a distribution over cells:

```latex
q_t(j) = \frac{\sum_i s^{O}_{t,i} M_{t,ij}}{\sum_{j'} \sum_i s^{O}_{t,i} M_{t,ij'}}, \qquad A_t(j) = \operatorname{softmax}_{j \le 49}\!\left( \tfrac{(W_q c_t)^{\top} (W_k P'_{t,j})}{\sqrt{d_a}} \right)
```

```latex
\pi_t(i) = \frac{\sum_j A_t(j)\, \bar A_{t,ij}}{\sum_{i'} \sum_j A_t(j)\, \bar A_{t,i'j}}, \qquad \mathcal{L}_{\mathrm{agree}} = \mathrm{KL}\big(\mathrm{sg}[q_t] \,\Vert\, A_t\big) + \mathrm{KL}\big(\mathrm{sg}[\pi_t] \,\Vert\, \operatorname{softmax}_i(s^{O}_{t,i}/T_a)\big)
```

The first term teaches SA²'s attention to look at cells whose objects score high; the second teaches the object head to rank highest the objects whose cells SA² attends to. sg is stop-gradient, so each side is a teacher for the other but neither can collapse onto it. The loss applies only to the top-k steps of abnormal videos with at least two detections, where *which* agent is anomalous is a real question. It is a co-training signal: an agent that both moves unusually (patch view) and looks or behaves unusually (object view) is the most reliable evidence weak labels can give.

**Warm-up.** λ\_A(e) is 0 for the first 5 epochs, so both branches first learn from MIL, then rises linearly to λ\_A over epochs 5–10. Without the warm-up both distributions are near uniform and the loss teaches nothing.

**Weights.** λ\_O = 0.5, λ\_N = 0.5, λ\_A = 0.1, T\_a = 0.5 as starting values; λ\_O ∈ {0.25, 0.5, 1} and λ\_A ∈ {0.05, 0.1, 0.2} are chosen once on the held-out 15% of P3 training videos by video-level AUC, then frozen for every protocol.

**Why this fit should learn where MQR did not.**

| MQR failure (Section 3.2.1 of the Nov 1 proposal) | OPF design choice |
| --- | --- |
| Started at mean pooling, which the host already had | Starts at geometry-correct RoI pooling and object-aware selection, which the host does not have |
| Gradient divided by ≈ 49·√384 over near-identical tokens | Attention restricted by a box bias to a few distinct CLIP windows; object tokens differ by position, class and dynamics |
| Every gradient passed through a closed gate and sparse top-k MIL | Object tokens get direct gradient from L\_obj and dense L\_neg; only the injection into c̃\_t is gated |
| No signal said which tokens matter | L\_agree ties the object and patch views of "where" together |
| Adam moved each new parameter by at most about 0.09 in 1,749 steps at 5e-5 | ×10 learning rate for new parameters by default, ×10-scaled gates, one zero per path (Section 3.5.1) |

The learning-rate budget, the gate scaling and the logged quantities that decide whether OPF has learned are fixed in advance in Section 3.5.1.

### 3.5 Training strategy

Training uses video-level annotations only and LaGoVAD's schedule; OSTP adds two loss terms and the warm-up, nothing else.

| Protocol | Training set | Normal / abnormal videos | Anomaly text (hand-written) |
| --- | --- | --- | --- |
| P1, TAD | TAD train, UMIL split | 210 / 190 | TAD's 7 category names and definitions |
| P2, MSAD | MSAD weak-protocol train | 360 / 120 | MSAD's 11 anomaly types |
| P3, dashcam | Nexar labelled set + CCD official train split | Nexar 750 / 750; CCD train part of 3,000 normal / 1,500 crash | Collision, near-miss and car-crash definitions |
| P4, cross-setting | None: reuses the P2 and P3 models | — | Each test set's own definitions |

**Stages.**

1. **Cache features once.** CLIP CLS c\_t; the 49 window-crop CLIP features per sampled frame (full grid, since OPF reads all cells; about 49 KB fp16 per frame); motion magnitudes; the VideoMAE V2-S vector; and YOLO boxes with track IDs. The overlap matrices M and Ā are computed on the fly from the boxes.
2. **Train every row from the same random initialisation** with one schedule and 3 seeds. At step 0, the gates tanh(10γ) and tanh(10 g\_P) are 0 and W\_q = 0, so OSTP starts as LaGoVAD-STP with "selection + average" SA² (plus object-aware selection when it is the default).
3. **Warm up the agreement loss** over epochs 5–10.
4. **Monitor** the attention entropies, β, ‖W↑‖, tanh(γ), the prefix norm, and the share of top-k positive objects that fall on vehicles vs persons.

**Settings.** AdamW, batch 64, cosine schedule, 40 epochs (LaGoVAD). Host parameters train at 5e-5; **every new parameter (SA², prefix, OPF) trains at 5e-4 by default** (Section 3.5.1). K, l, λ\_O, λ\_A and the multiplier (×10 default; ×3 and ×30 checked) are chosen once on a held-out 15% of the P3 training videos by video-level AUC, then frozen for every protocol. Last-epoch checkpoints for every model. No hyperparameter is chosen on a test list.

**Captions and shortcuts.** Hand-written caption templates only; P3 is also reported with Nexar-only training because CCD's normals come from a different source (BDD100K).

**Compute.** The new cost is extraction. YOLOv9-C runs once per sampled frame (about 650K frames for Nexar, CCD and the DoTA test set; hours on one GPU, to be measured in the first week). Keeping the full 7×7 grid raises patch storage from about 12 KB to 49 KB per frame: about 11 GB each for Nexar and CCD and 12 GB for DoTA's test clips (estimates from the Nov 1 table). Training fits one 24 GB GPU.

#### 3.5.1 Optimisation budget and learning checks

Adam moves each parameter by at most about the learning rate per step, whatever the gradient's size, so the total distance a parameter can travel is the sum of the learning rates over training. In v5 Phase 2 this was measured: 5e-5 over 1,749 steps moved each parameter by only about 0.04–0.09. A gate tanh(γ) starting at 0 can then open to at most about 0.09, and a scalar β starting at 1 stays in 0.91–1.09. This is the MQR trap, and OPF would fall into it with the default settings.

Four rules apply to every new component:

1. **×10 learning rate by default.** New parameters train at 5e-4, a budget of about 0.44 (cosine mean) to 0.87 (peak) per parameter over the same steps.
2. **Scaled scalars.** Every scalar that decides whether OPF is used is scaled, as LaGoVAD scales its blend weight σ(10·a): tanh(10γ) for the relation token, tanh(10 g\_P) for the patch update, β = exp(4 b\_β) for the box bias. With rule 1 the gates can reach 1 and β can span about 0.2–5.8.
3. **One zero per path.** A path starts closed through one scalar gate, never through two zero-initialised matrices in series (W↑ is random; only g\_P is 0). W^Q of the object → patch attention stays zero-initialised because the bias already gives a useful start and W^K p is non-zero, so W^Q's gradient is non-zero from step 1.
4. **Enough steps.** The agreement warm-up spans epochs 5–10 (about 220–440 of 1,749 steps for P3). If the checks below fail at epoch 10, the run is repeated once with 80 epochs for the new parameters only, and the result is reported either way.

**Learning checks, logged every 50 steps and judged at epoch 10 and at the end:**

| Quantity | Start | Counts as learned if |
| --- | --- | --- |
| Cumulative learning rate per group (budget) | 0 | Logged; ≥ 0.4 for new parameters |
| Relation gate tanh(10γ) | 0 | > 0.1 |
| Patch-update gate tanh(10 g\_P) | 0 | > 0.1 |
| Box-bias strength β | 1 | \|log β\| > 0.2 |
| Object → patch attention entropy | Entropy of RoI pooling | ≥ 5% away from its start |
| SA² attention entropy | log 12 (uniform over selected) | ≥ 5% below log 12 |
| ‖W↑ − W↑₀‖ / ‖W↑₀‖ | 0 | > 0.1 |
| Prefix norm change | 0 | Logged |

A run where both gates stay under 0.1 is flagged "OPF not learning" and reported as such, rather than read as "objects do not help". The ×1 vs ×10 multiplier and scaled vs unscaled gates are a side ablation (4.5).

### 3.6 Inference

No LLM runs at any point. For each video and anomaly definition:

1. **Sample** one frame in eight (every frame for 10 fps sets).
2. **Encode** each frame with CLIP (c\_t), its 49 native-resolution window crops with the same CLIP, and its 16-frame clip with VideoMAE V2-S.
3. **Detect and track** agents with YOLOv9-C + ByteTrack at native resolution.
4. **Fuse** objects and patches with OPF (two-way blocks with the geometric bias), giving P′, refined object tokens and r\_t.
5. **Aggregate** with SA² over P′ using object-aware motion selection; c̃\_t = c\_t + x^AS\_t + tanh(10γ)·W\_r·r\_t.
6. **Add motion** through the VideoMAE gate, then score with LaGoVAD's RoFormer, text co-attention and heads (definition encoded with the learned prefix).
7. **Output** per step: the frame score s\_t; the 7×7 SA² map E\_t; and each box with its score s^O\_{t,i}. The box-level map paints each box with its score (maximum where boxes overlap) and, where no box exists, falls back to E\_t; this map is scored with STAUC.

The main reported score is s\_t. Blending it with the object score, ŝ\_t = (1 − ω)s\_t + ω·max\_i s^O\_{t,i}, is ablation O8. Score post-processing is unchanged: per-video min-max for DoTA and DADA-2000, raw scores elsewhere.

## 4. Experimental plan

The datasets, test lists, leakage checks, metrics and published comparison rows of the Nov 1 proposal are kept. This section lists only what the object branch adds.

### 4.1 Datasets (unchanged)

| Dataset | View | Role | Object-branch note |
| --- | --- | --- | --- |
| TAD | Fixed CCTV | P1 train and test; P4 zero-shot | Small, distant agents; detector recall at CCTV scale to be measured |
| MSAD | Fixed, 14 scenes | P2 train and test; traffic slice | Non-traffic scenes have few COCO traffic classes: checks that OPF does no harm when boxes are few or irrelevant |
| CCD | Dashcam | P3 train and in-domain test | — |
| Nexar | Dashcam | P3 train | — |
| DoTA | Dashcam | P3 zero-shot test; STAUC; Step 0 pilot (train split) | Anomalous-agent boxes with track IDs: used only for evaluation and the pilot, never for training |
| DADA-2000 | Dashcam | P3 zero-shot test | Ego / non-ego split |
| MM-AU | Dashcam | Optional zero-shot test after de-duplication | — |
|  |  |  |  |

**New weights to download:** YOLOv9-C COCO weights (Ultralytics) and ByteTrack (bundled with Ultralytics' tracker). COCO contains no frames from the seven traffic datasets, so the detector adds no test leakage.

### 4.2 Processing additions

1. **Detect and track** every sampled frame of every dataset at native resolution; store boxes, class, confidence and track ID.
2. **Keep the full 7×7 window-crop grid** for all training and test videos (OPF reads every cell, not only the top-12).
3. **Detector quality on DoTA.** Match YOLO boxes to DoTA's anomalous-agent boxes (IoU ≥ 0.5) to record, per test clip, whether the anomalous agent was detected. This gives the "detected / missed" split used in the analyses; it is never used in training.
4. **Illumination tag.** Mark night clips (DoTA metadata where available, else mean frame brightness under a fixed threshold) to test PromptTAD's dark-scene failure.

### 4.3 Step 0 pilot (inference only, 500 random DoTA training clips)

The Nov 1 checks S1–S3 (does SA²'s motion top-12 hit the anomalous agents under ego-motion?) are kept. Three checks are added for the detector and the fit. None needs training.

| Check | Question | Pass criterion (proposed) | If it fails |
| --- | --- | --- | --- |
| S4: detector recall | Share of anomalous-agent boxes matched by a YOLO box (IoU ≥ 0.5), by box-size bin, ego flag and day / night, at input 640 vs 1280 | Clearly higher than SA² top-12 hit rate (S1) on small boxes | Lower confidence threshold to 0.1; if still low on small boxes, the object branch cannot help them and H5 is tested on medium and large boxes only |
| S5: object-aware selection | S1's hit rate with Mo vs Mo^obj (λ₀ = 0.3), raw and ego-compensated | Mo^obj hit rate above Mo's, mostly on dashcam clips | Keep plain Mo as default; Mo^obj becomes an ablation |
| S6: RoI appearance zero-shot | AUC of ranking anomalous vs other detected objects by CLIP similarity to the accident definition of u\_{t,i} from native windows, u from 224-resize windows, and a native box crop, vs "normal" | Above 0.5 for box crops; record u's value | If u is near chance but box crops are not, make box crops (O9) the default |

The pilot costs about one GPU-day including detection for the 500 clips. It decides defaults before any training and touches no test clip.

### 4.4 Baselines and protocols

**Baselines.** Group A (LaGoVAD released B0, retrained B1, + motion stream B2), Group B (released checkpoints re-scored: VadCLIP, DSANet, PEL4VAD, RTFM, MGFN, MOVAD, Simple-TAD) and Group C (VadCLIP and a reimplemented STPrompt detection branch, retrained in-domain) are kept. One row is added to Group C:

- **C3: PromptTAD-style fusion under weak labels.** A reimplementation of PromptTAD's prompt scheme (corner-PE object tokens, two-way and relation aggregation, no FPN) attached to LaGoVAD-STP, fused with the patch grid but **without** the fit (no shared-PE alignment, no overlap bias, no RoI appearance) and trained with the same object MIL. It shows what PromptTAD's mechanism gives under video labels, and how much the fit adds. PromptTAD's published frame-supervised DoTA numbers (83.9 AUC) stay an upper reference only.

**Protocols P1–P4** and their metrics (AUC\_O, AP, AUC\_A, AP\_A; DoTA and DADA-2000 macro / micro AUC by ego flag and box size; STAUC; pointing game) are unchanged. Additions for the object branch:

- **Object-level STAUC.** STAUC computed with the box-level map of Section 3.6, beside the 7×7 SA² map, on DoTA's official 1,305-video list.
- **Anomalous-object ranking.** In anomalous DoTA frames where the anomalous agent was detected, the share of frames whose top-scored box is that agent (object pointing game), against random and largest-box baselines.
- **Detected vs missed.** AUC on clips where the anomalous agent was or was not detected (from 4.2 step 3), to show the model falls back to STP, not below it, when detection fails.
- **PromptTAD's small-object subset** (334 test clips, boxes under 2,267 px) reported for every row.

All thesis numbers are 3-seed means ± standard deviation, subsets carry 95% bootstrap intervals, and OSTP vs STP is tested with a paired bootstrap.

### 4.5 Ablation studies

Two ladders. The **STP ladder** (R0–R11) is kept from the Nov 1 proposal and tests H1–H4 with OPF off. The **object ladder** (O1–O10) starts from the best STP row, R6, and adds the object branch one part at a time to test H5 (patch-level beats global fusion) and H6 (the fit is what makes it learn). Every row uses video-level labels only and reports P1, P2, P3 (with DoTA by ego flag, box size and the small-object subset) and STAUC where a map exists.

**STP ladder (kept).** R0 motion stream only · R1 + average of 49 patches · R2 + attention over 49 · R3 + motion top-12 + average · R4 + SA² · R5 R0 + prefix · **R6 full LaGoVAD-STP** · R7 ego-compensated motion · R8 text-agent selection · R9 dense ClearCLIP tokens · R10 K ∈ {6, 12, 24} · R11 released soft prompts.

**Object ladder (new).**

| Row | Variant (on top of R6) | Isolates | Expected if the design holds |
| --- | --- | --- | --- |
| O1 | Object tokens (PromptTAD prompt encoder + relation aggregation) fused with the **global** c\_t only; r\_t added to c̃\_t | Global-level object fusion | Small gain at most; little on small boxes |
| O2 | PromptTAD-style two-way fusion with the patch grid, corner PE only, separate learned PE for patches, no bias, no RoI appearance (= baseline C3) | Patch-level fusion without the fit | Above O1, below O4; attention near uniform |
| O3 | O2 + shared Fourier PE for boxes and patches | Positional alignment | Small gain over O2 |
| O4 | O3 + overlap bias β log(Ā′ + ε) and RoI appearance u | The structural fit | Clear gain over O2; attention starts at RoI pooling |
| O5 | O4 + object MIL + dense negatives | Direct object supervision | Gain over O4; W↑ and γ move away from 0 |
| **O6** | **O5 + agreement loss (full LaGoVAD-OSTP)** | Learned object–patch fit | **Best AUC and best object-level STAUC; largest gains on ego-uninvolved and small-box clips** |
| O7 | O6 + object-aware selection Mo^obj | Objects guiding SA²'s fixed prior | ≥ O6 on dashcam sets, about equal on TAD (default if S5 passes) |
| O8 | O6 + score blend ω·max\_i s^O | Using object scores directly in the frame score | Higher AUC\_A; risk of false alarms on normal videos with many objects |
| O9 | O6 with a native-resolution box-crop CLIP feature added to pooled u (default if S6 passes) | Patch pooling vs dedicated crops | Better on small boxes at ≤ 16 extra CLIP passes per frame |
| O10 | O6 without the text term in s^O | Language guidance of objects | Lower zero-shot transfer and drift@5 robustness |

H5 holds if O6 > O1 overall and on PromptTAD's small-object subset. H6 holds if O4 > O2 and O6 > O5, and if the logged W↑, γ and attention entropy move away from their initial values in O6 but not in O2.

**Side ablations.** Window source: native frame (default) vs STPrompt's 224×224 resize, for R4, R6 and O6; learning-rate multiplier ×1 / ×3 / ×10 / ×30 and scaled vs unscaled gates; detector input 640 vs 1280; confidence 0.1 / 0.25 / 0.4; N\_max 8 / 16 / 32; YOLOv9-C vs a newer Ultralytics YOLO model of similar size; with and without tracking (δ = 0); L\_I and L\_R ∈ {1, 2, 4}; λ₀ ∈ {0.1, 0.3, 0.5}; agreement loss one-directional (patch → object only, object → patch only).

**Oracle diagnostic (analysis only).** On DoTA test clips, replace YOLO boxes with DoTA's ground-truth boxes for all agents. This is not a comparable result (it leaks which objects exist, not which are anomalous) and is reported only to bound how much better detection could help.

**Analyses.**

1. **Where the gain lands:** ego flag × box-size bin; small-object subset; categories; detected vs missed anomalous agent; day vs night; MSAD traffic slice vs non-traffic scenes.
2. **Spatial evidence:** STAUC and pointing game for the 7×7 map (R6, O6) and the box-level map (O6), against uniform and centre priors; object pointing game against random and largest-box choices.
3. **What the fusion learned:** β and attention entropy over training; how far object → patch attention spreads beyond the box; which classes the top-k positive objects belong to.
4. **Failure cases:** dark scenes and missed detections (PromptTAD's failure), crowded parked-car scenes (false object positives), anomalies with no agent (road spills, obstacles).
5. **Cost:** parameters, storage and measured frames per second for R0, R6, O6 and O9.

**Budget.** O1, O2, O4, O5 and O6 run on P1, P2 and P3 with 3 seeds; O3 and O7–O10 and the side ablations on P1 and P3, one seed first, three for the variants kept. If time runs short, the object ladder is cut to O1, O2, O4 and O6, which still test H5 and H6.

## 5. Risks and fallbacks

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Detector misses small or night-time agents (PromptTAD's own failure case) | S4 recall by size and day / night | Zero-init design falls back to STP when no box exists; lower confidence threshold; box-crop appearance (O9) |
| Object MIL picks the wrong positive (e.g. a large nearby car) every time | Class and size of top-k positive objects during training | Agreement loss with patch motion; dense negatives; size-normalised score as side ablation |
| OPF does not move from its start, like MQR | ‖W↑‖, tanh(γ) and attention entropy flat after warm-up | Budget rules of 3.5.1 already applied; if the checks still fail, rerun once with 80 epochs for new parameters and report the negative result; keep the fixed parts (RoI appearance, object-aware selection, O7), which help even without learning |
| Ego-motion still dominates selection | S1 and S5 on dashcam clips | Object-aware selection plus ego-compensated motion (R7 + O7) |
| Parked and queued vehicles create false object positives on normal videos | AP and false-alarm rate on CCD and MSAD normals | Box dynamics δ and the dense negatives; no score blend (O8 off) |
| Agents not in COCO's classes (debris, animals, spills) | Per-category DoTA results (obstacle and out-of-control) | Patch branch still covers them through λ₀ > 0; reported as a limitation |
| Storage of the full 7×7 grid | Disk use during extraction | Store the full grid for training sets and DoTA test only; top-12 + covered cells elsewhere |
| Novelty overlap with 2026 preprints | Full read of "Localizing to Debias" and "Learning Where and When" | Narrow the claim to object–patch fusion under weak labels on traffic data |

## References

All references of the Nov 1 proposal are kept. Added or newly central for the object branch:

- Qiu, Yang and Gong, [PromptTAD: Object-Prompt Enhanced Traffic Anomaly Detection](https://doi.org/10.1109/LRA.2025.3572428), IEEE RA-L 10(7), 2025 (source of the object-prompt scheme; project copy)
- Wang, Yeh and Liao, YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information, ECCV 2024 (detector)
- Kirillov et al., Segment Anything (SAM), ICCV 2023 (box prompt encoder and two-way transformer, via PromptTAD)
- Tancik et al., Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains, NeurIPS 2020 (positional encoding shared by boxes and patches)
- Zhang et al., ByteTrack: Multi-Object Tracking by Associating Every Detection Box, ECCV 2022 (tracking for box dynamics)
- Carion et al., End-to-End Object Detection with Transformers (DETR), ECCV 2020 (Hungarian matching in PromptTAD; not used here)
- Rossi et al., Memory-Augmented Online Video Anomaly Detection (MOVAD), ICASSP 2024 (PromptTAD's host)
- Wu et al., [STPrompt](https://arxiv.org/abs/2408.05905), ACM MM 2024 (SA² and the learnable prefix)
- Liu et al., [LaGoVAD](https://arxiv.org/abs/2503.13160), ICLR 2026 (host detector)
- Wang et al., [VideoMAE V2](https://arxiv.org/abs/2303.16727), CVPR 2023 (motion encoder)
- Yao et al., [DoTA](https://arxiv.org/abs/2004.03044), IEEE TPAMI 2022 (anomalous-agent boxes, STAUC)
