STEERING AND RECTIFYING LATENT REPRESENTATION MANIFOLDS IN FROZEN MULTI-MODAL LLMS FOR VIDEO ANOMALY DETECTION 

**Zhaolin Cai**<sup>1</sup> **, Fan Li**<sup>1</sup><sup>_,_2</sup><sup>_∗_</sup> **, Huiyu Duan**<sup>3</sup><sup>_∗_</sup> **, Lijun He**<sup>2</sup> **, Guangtao Zhai**<sup>3</sup><sup>_,_4</sup><sup>_∗_</sup> 1Xinjiang University 2Xi’an Jiao Tong University 3Shanghai Jiao Tong University 4Shanghai AI Laboratory 

# ABSTRACT 

Video anomaly detection (VAD) aims to identify abnormal events in videos. Traditional VAD methods generally suffer from the high costs of labeled data and full training, thus some recent works have explored leveraging frozen multi-modal large language models (MLLMs) in a tuning-free manner to perform VAD. However, their performance is limited as they directly inherit pre-training biases and cannot adapt internal representations to specific video contexts, leading to difficulties in handling subtle or ambiguous anomalies. To address these limitations, we propose a novel intervention framework, termed **SteerVAD** , which advances MLLM-based VAD by shifting from passively reading to actively steering and rectifying internal representations. Our approach first leverages the gradient-free representational separability analysis (RSA) to identify top attention heads as latent anomaly experts (LAEs) which are most discriminative for VAD. Then a hierarchical meta-controller (HMC) generates dynamic rectification signals by jointly conditioning on global context and these LAE outputs. The signals execute targeted, anisotropic scaling directly upon the LAE representation manifolds, amplifying anomaly-relevant dimensions while suppressing inherent biases. Extensive experiments on mainstream benchmarks demonstrate our method achieves stateof-the-art performance among tuning-free approaches requiring only 1% of training data, establishing it as a powerful new direction for video anomaly detection. The code will be released upon the publication. 

# 1 INTRODUCTION 

Video anomaly detection (VAD) aims to identify events that deviate from expected patterns, which plays a critical role in intelligent surveillance (Sultani et al., 2018), industrial quality control (Roth et al., 2022; Liu et al., 2024b), and autonomous systems (Yao et al., 2023; Bogdoll et al., 2022). While traditional paradigms, encompassing supervised (Liu et al., 2018; Landi et al., 2019), weakly supervised (Wu et al., 2024b) and unsupervised approaches (Lv et al., 2021; Liu et al., 2021), have demonstrated notable success but they are constrained by the reliance on large-scale training data. This dependency not only incurs substantial computational and annotation costs but also limits their ability to generalize to unseen scenarios, hindering real-world deployment (Noghre et al., 2025). 

The advent of multi-modal large language models (MLLMs) (Zhu et al., 2023; Liu et al., 2023) has opened a new frontier for VAD by leveraging their powerful capabilities. While fine-tuning these models for VAD is effective (Zhang et al., 2024a;b), these methods reintroduce the burden of computational costs and data requirements, undermining the advantages of pre-trained models. Recent studies have shifted towards tuning-free paradigms (Shao et al., 2025), which utilize frozen MLLMs to detect anomalies by their generated interpreting text or features (Zanella et al., 2024; Ye et al., 2025). Despite their effectiveness, these approaches are constrained by their passive nature. They overlook two critical deficiencies deeply rooted within the models. (1) The first is inherent representational bias of MLLMs, since they are pre-trained on web-scale corpora, developing a feature space optimized for frequent and prototypical concepts. As a result, their representations exhibit reduced sensitivity to subtle and rare patterns typical in anomalous events, leading to missed or biased detection. (2) The second is contextual ambiguity. Since the semantic meaning of a local 

> _∗_ Corresponding authors. 

1 



<!-- Start of picture text -->
(a) Previous Full-Training based Methods (b) Existing MLLM based Methods  (c) Our SteerVAD Method<br>Video Frames Compute  Video Frames Extra  Video Frames LAE<br>Backbone  Resources Foundation  LLM/VLM Foundation  Identification<br>Network Training Models Assisting Models Steering<br>Costly  Labeled  Lossy  Prompt  Accurate Manifold<br>Burden Datasets Biased Learning Unbiased Rectification<br>Yes. This video<br>... ... No. This video does not show  ... shows car accident.<br>any abnormal events  … The video  ……<br>Anomaly Frames  Description Anomaly Frames  Description<br><!-- End of picture text -->

**Figure 1:** Comparison of traditional full-training methods, existing tuning-free methods and our proposed SteerVAD. Our method overcomes the issue of costly training resources and inherent biases with minimal data required from pre-trained foundation models compared to previous methods. 

action is determined by its global context, passively relying on isolated features can lead the model to produce confounding representations for visually similar but contextually distinct events. These issues are not mere surface-level classification errors, but stem from structural flaws in the internal representations of MLLMs, highlighting the inherent limitations of passive interpretation. 

To address these limitations, we advocate a perspective shift from passive feature reading to active geometric intervention. Our approach builds on the well-established manifold hypothesis (Whiteley et al., 2022), which posits that high-dimensional natural data concentrates on low-dimensional structures. We extend this hypothesis to the latent geometry of MLLMs. Within their high-dimensional feature space, the representations for a semantic class coalesce into a coherent, low-dimensional structure, which we term representation manifold. From this geometric perspective, the identified deficiencies can be re-contextualized. _Inherent representational bias_ manifests as a systemic proximity between the manifolds of normal and anomalous events. The feature space optimized for frequent normal patterns fails to allocate sufficient geometric separation for their low-frequency abnormal counterparts. This proximity is compounded by _contextual ambiguity_ , manifesting as the local geometric entanglement of these manifolds. In these entangled regions, representations of visually similar but semantically distinct events become indistinguishable, making robust separation extremely challenging. This reframing highlights that the core challenge is not merely the classification of static features, but the dynamic steering and rectification of the underlying latent manifolds. 

In this paper, we introduce SteerVAD, a novel tuning-free framework that implements this geometric intervention by steering latent representation manifolds within a frozen MLLM. SteerVAD operationalizes our geometric insight through two synergistic innovations. First, we develop a gradientfree representational separability analysis (RSA) to identify internal attention heads termed latent anomaly experts (LAEs), whose representations are effective for separating normal and anomalous events, exhibiting strong potential for rectification. This identification allows our intervention to be focused and efficient, avoiding unnecessary manipulation of the entire model. Second, we design a lightweight hierarchical meta-controller (HMC) that learns to execute dynamic, context-dependent geometric transformations. Conditioned on a global understanding of the scene, the HMC executes targeted anisotropic scaling to the features produced by identified LAEs, thereby actively reshaping their corresponding manifolds by amplifying anomaly-relevant dimensions while suppressing biased ones. By calibrating on only 1% of the training set, this mechanism effectively disentangling the representations of normal and anomalous events without any fine-tuning on the pre-trained model. 

Our main contributions are as follows: 

- We introduce a novel intervention paradigm for tuning-free video anomaly detection that moves beyond passive feature interpretation to active geometric intervention. Our framework is the first to operationalize this paradigm by directly steering and reshaping latent representation manifolds within completely frozen MLLM. 

- We introduce representational separability analysis (RSA), a novel gradient-free geometric method to precisely identify latent anomaly experts whose internal feature spaces are most aligned for VAD, ensuring the data-efficiency of our framework. 

- We design a hierarchical meta-controller (HMC) that generates context-aware signals to perform anisotropic manifold scaling, dynamically disentangling class representations to overcome pre-training biases and contextual ambiguities. 

2 

- We establish new state-of-the-art performance on UCF-Crime and XD-Violence datasets among tuning-free methods using a frozen MLLM, demonstrating that targeted, dataefficient intervention method as a promising way against expensive fine-tuning approaches. 

# 2 RELATED WORK 

**Traditional Video Anomaly Detection.** Research in video anomaly detection (VAD) has traditionally focused on models trained specifically for the task, categorized by the level of supervision. Supervised methods (Liu et al., 2018; Landi et al., 2019) have shown strong performance by training on extensive datasets with frame-level labels, but the high cost of annotation limits their practical application. To mitigate this, weakly-supervised approaches (Li et al., 2022a; Wang et al., 2024) leverage more accessible video-level labels, often using multiple instance learning (MIL) frameworks. Unsupervised methods (Tur et al., 2023; Zaheer et al., 2022; Thakare et al., 2023) are trained exclusively on normal data. These models learn to identify anomalies as deviations from a learned normality model, typically based on reconstruction error (Xu et al., 2017) or future frame prediction (Luo et al., 2022). While effective, these traditional paradigms require heavily training and struggle to generalize to unseen anomalies, motivating a shift towards new paradigms. 

**Video Anomaly Detection with LLMs and MLLMs.** The advent of large language models (Vaswani et al., 2017; Touvron et al., 2023) and multi-modal large language models (MLLMs) (Zhu et al., 2023; Li et al., 2023; Wu et al., 2024d), has created a new way for VAD. Initial efforts involves adapting these models through supervised fine-tuning (Zhang et al., 2023; Yuan et al., 2024). While effective, these methods reintroduce the need for large labeled datasets and high computational costs. A more recent and promising direction is the tuning-free paradigm, which uses MLLMs in a zero-shot or few-shot capacity (Shao et al., 2025). These methods involve querying the MLLM with modified inputs to generate enhanced textual descriptions or decisions about video, which are then parsed to identify anomalies (Zanella et al., 2024; Ye et al., 2025). Although these approaches effectively leverage the MLLMs without expensive training, they treat the representations of the model as static and immutable, which makes them susceptible to pre-training biases and ambiguity, leaving a crucial gap in how to actively utilize the powerful internals of these models. 

**Internal Analysis of LLMs and MLLMs.** Recent research in mechanistic interpretability of LLMs and MLLMs provides the foundation for intervention (Luo & Specia, 2024). Studies have shown that complex behaviors in foundation models can be localized to specific modules, such as individual attention heads or neuron groups (Zheng et al., 2025), which act as functional circuits (Zhou et al., 2024; Bi et al., 2024). Building on this insight, the field of model editing has the ability to make targeted modifications to model weights to alter specific behaviors (Wang & Veitch, 2025; Zweiger et al., 2025). These techniques have been applied to language tasks or involve static, inputindependent interventions (Nguyen et al., 2025; Mahmoud et al., 2025). The application of dynamic, context-dependent rectification of internal geometric structures for a complex, spatio-temporal reasoning task remains an unexplored frontier. Our work is the first to bridge this gap by introducing actively geometric rectifications on the representation manifolds with frozen MLLM for VAD. 

# 3 METHODOLOGY 

## 3.1 PRELIMINARY: REPRESENTATION MANIFOLDS IN MLLMS 

Our theoretical approach is based on the manifold hypothesis (Whiteley et al., 2022; Genovese et al., 2010; Boissonnat & Ghosh, 2010), which posits that high-dimensional data concentrates on or near a low-dimensional manifold. We extend this to the latent space of MLLMs, where an attention head _h_ : _V →_ R<sup>_d_head</sup> acts as a feature extractor that maps semantically similar video inputs into geometrically proximal representations, causing the varied manifestations of a class to coalesce into a coherent structure. We model these emergent structures as representation manifolds. Consequently, for a given head _h_ , all normal event representations form a manifold _M_ norm, while anomalous events form a manifold _M_ anom, both embedded within the head’s high-dimensional feature space. A rigorous topological analysis of these structures is detailed in Appendix A. 

Visualizations using UMAP, Figure 2 provides an intuitive illustration consistent with this geometric perspective. The features appear to form two distinct yet intertwined manifolds, visually confirming the problems of representational bias and contextual ambiguity. This geometric entanglement 

3 



<!-- Start of picture text -->
Manifold Structures (Viewport1) Manifold Structures (Viewport2) Manifold Structures (Viewport1) Manifold Structures (Viewport2)<br>35 13<br>30 12<br>25 11<br>20 10<br>15 9<br>6 8 10 12 14 16 -4-3-2-101231005 35302520151050 -4 -3 -2 -1 0 1 2 3 1614121086 6 8 10 12 14 16 -4-3-2-10123678 131211109876 -4 -3 -2 -1 0 1 2 3 1614121086<br>(a)  Cubic interpolation surface (b)  Triangulated surface<br>Component 1 Component 2 Component 1 Component 2<br>Component 2 Component 1 Component 2 Component 1<br>Component 3 Component 3<br>Component 3 Component 3<br><!-- End of picture text -->

**Figure 2:** 3D UMAP visualization of representation manifolds of normal (blue) and anomalous (red) events from InternVL, illustrating their geometric structure. Each manifold is rendered from two perspectives using (a) cubic interpolation and (b) triangulation. 

makes reliable separation by a simple classifier difficult. Our approach moves beyond passively accepting this topology, we aim to learn a minimal, context-dependent geometric transformation _T_ **c** that actively rectifies these manifolds to achieve clear differentiation. 

## 3.2 FRAMEWORK OVERVIEW 

As shown in Figure 3, SteerVAD enables geometric intervention within a frozen MLLM for VAD. The framework rests on two principles: identification of relevant representational subspaces and rectification of them. To accomplish the first, a representational separability analysis (RSA) identifies few internal attention heads that inherently aligned with VAD, which we term latent anomaly experts (LAEs). Then the hierarchical meta-controller (HMC) drives rectification by integrating the global context with LAE outputs to generate steering signals. These signals perform geometric transformations on LAE feature manifolds, amplifying anomaly cues and suppressing biases. The rectified representations are aggregated and scored by anomaly scorer to produce final decision and curves. Detailed component implementation and analysis of SteerVAD are provided in Appendix B. 

**Problem Formulation** During the inference, given an input video stream _V_ , the stream is first partitioned into a sequence of non-overlapping temporal segments, _V_ = _{S_ 1 _, S_ 2 _, . . . , ST }_ . For each segment _St_ , our objective is to compute an anomaly probability _pt ∈_ [0 _,_ 1] in a real-time, singlepass manner. The identification of LAEs and the training of the HMC and scorer are performed in a preceding offline calibration phase, utilizing a small, representative labeled dataset _D_ calib. 

## 3.3 REPRESENTATIONAL SEPARABILITY ANALYSIS 

To identify sub-modules within frozen MLLM that are inherently aligned with the video anomaly detection (VAD) task, we propose representational separability analysis (RSA) to discover latent anomaly experts (LAEs), defined as specific attention heads whose feature representations exhibit a high degree of geometric separability between normal and anomalous patterns. 

**Separability Metric.** We compute the Inter-to-Intra Scatter Ratio as the RSA score to quantify geometric suitability of each head. This metric measures the ratio of between-class separation to within-class compactness. For head _hl,k_ (layer _l_ , index _k_ ), the RSA score is computed as: 



**Feature Extraction and LAE Selection.** The LAE selection is performed on the calibration set _D_ calib via a single-forward pass for each video through the frozen MLLM. We extract and store features from all attention heads to build feature banks, compute the _S_ RSA score and select the top- _K_ heads with the highest scores as LAEs. This process efficiently pinpoints the most informative internal subspaces of the MLLM, providing optimal targets for dynamic rectification. 

## 3.4 HIERARCHICAL META-CONTROLLER FOR DYNAMIC MANIFOLD RECTIFICATION 

The hierarchical meta-controller (HMC) dynamically orchestrates manifold rectification via a single MLLM forward pass. The HMC receives two concurrent information streams extracted from the 

4 



<!-- Start of picture text -->
 Single Forward Multi-Modal LLM<br>Text Inputs<br>If there are any of the abnormal  MHA MHA MHA MHA MHA Yes. The video shows a<br>events occurred in the video that  person holding a gun, pointing<br>deviate from the normal scene？ Final Context Vector it forward, which indicates a<br>potential armed conflict or<br>Hidden  shooting situation ……<br>States<br>Full Autoregression<br>Multi-Modal LLM<br>Representational Separability Analysis Module<br>Videos<br>…… …… …… …… Latent Anomaly Expert Heads<br>Local Gating Module Manifolds in LAEs<br>Adapter<br>…… Anisotropic  Scaling<br>Adapter<br>Abnormal Local Signals Anomaly Frames<br>Global Scrutiny Gate  Anomaly<br>Global Signal Scorer<br>Manipulation<br>Context  Signals<br>Vector<br>Normal Hierarchical Meta-Controller Anomaly Distribution<br>Layer<br>……<br>Encoder Vision<br>Norm Relu Layer<br><!-- End of picture text -->

**Figure 3:** Framework overview of SteerVAD. We first apply the Representational Separability Analysis to find top _K_ Latent Anomaly Experts inside frozen MLLM. During the single pass, The global context vector **c** and LAE features _{_ **h** _i}_ are extracted. The Hierarchical Meta-Controller ingests these signals, using Global Scrutiny Gate and Local Gating Module to generate manipulation signals ( _s_ global, _{_ **g** _i}_ ). These signals perform Anisotropic Manifold Scaling to rectify LAE features. A lightweight Anomaly Scorer receives the rectified features and outputs the final anomaly curve. Detected anomalous frames can be passed to the full MLLM to produce a textual explanation. 

frozen MLLM: the fine-grained feature vectors _{_ **h** _i}_<sup>_K_</sup> _i_ =1<sup>fromthe</sup><sup>_K_LAEs,andaglobalcontext</sup> vector **c** _∈_ R<sup>_d_model</sup> . The context vector **c** is obtained by extracting the final hidden state corresponding to the first generated token, which serves as a holistic semantic summary of the scene by the MLLM. The hierarchical design of the HMC decouples the rectification task into two synergistic levels. 

**Global Scrutiny Gate.** The global scrutiny gate (GSG) computes a holistic suspicion score _s_ global as a global signal to quantify the overall possibility of an anomaly. It is implemented as a small multi-layer perceptron (MLP) that takes the global context vector **c** as input: 



where Linear1 _∈_ R<sup>_d_model</sup><sup>_×d_hidden</sup> , Linear2 _∈_ R<sup>_d_hidden</sup><sup>_×_1</sup> , BN denotes BatchNorm1d, and _σ_ is the sigmoid function. A value near 0 indicates a benign scene, prompting the HMC to remain quiescent, while a value near 1 signals a high-suspicion event that requires significant representational steering. This scalar acts as a master switch, determining the overall intensity of the subsequent rectification. 

**Local Gating Module.** While the global gate provides a coarse-level signal, the local gating module generates a fine-grained, per-feature control signal for each LAE. It consists of _K_ parallel, lightweight adapter networks. Each adapter is a low-rank network that takes the global context **c** as input and outputs a unique, dense steering vector **g** _i ∈_ R<sup>_d_head</sup> : 



where Lineardown _∈_ R<sup>_d_model</sup><sup>_×r_</sup> projects the context vector into a low-rank bottleneck space, and Linearup _∈_ R<sup>_r×d_head</sup> projects it back to the feature dimension of the expert head. The rank _r_ is a hyperparameter set to a small value to minimize trainable parameters. The hyperbolic tangent tanh activation constrains the output local signals _gi_ to [ _−_ 1 _,_ 1], enabling the controller to learn not only to amplify feature dimensions in positive values but also to actively suppress them. 

**Anisotropic Manifold Scaling.** The control signals generated by the GSG and LGM converge at the core rectification mechanism, anisotropic manifold scaling. Our goal is to apply a targeted, 

5 

context-dependent geometric transformation to the LAE feature space. We operationalize this transformation through a simple yet powerful element-wise operation, ensuring it incurs negligible computational overhead. A detailed geometric interpretation of this operation is provided in the Appendix A. For each LAE feature **h** _i_ , the rectified feature **h**<sup>_′_</sup> _i_<sup>is computed as:</sup> 

This formulation instantiates the geometric **h**<sup>_′_</sup> _i_<sup>=</sup><sup>**h**</sup><sup>_i⊙_</sup> transformation<sup>(1 +</sup><sup>_s_global</sup><sup>_·_</sup><sup>**g**</sup> as<sup>_i_)</sup> a stable, residual modulation. The(4) global signal _s_ global governs the intensity of transformation, while the local vector **g** _i_ provides its critical anisotropy, enabling the targeted stretching and compressing of the manifold along specific semantic axes potentially linked to pre-training biases. 

## 3.5 ANOMALY AGGREGATION AND SCORING 

The anomaly scorer processes the rectified features from selected LAEs by estimating frame level anomaly probabilities and then applying temporal smoothing to yield final coherent anomaly curves. 

**Probabilistic Anomaly Scoring.** For each segment _St_ , the rectified feature vectors _{_ **h**<sup>_′_</sup> _i ∈_ R<sup>_d_head</sup> _}_<sup>_K_</sup> _i_ =1<sup>fromselectedLAEsareaggregatedviaconcatenationintoasinglevector</sup><sup>_F_(</sup> final<sup>_t_)=</sup> Concat( **h** 1<sup>_′, . . . ,_</sup><sup>**h**</sup><sup>_′_</sup> _K_<sup>)</sup><sup>_∈_R</sup><sup>_K·d_head.Theaggregatedvectoristheninputtoanomalyscorer,alogis-</sup> tic regression classifier that maps the feature to a frame-level anomaly probability _pt_ : 



where **w** and _b_ are the learnable weights and bias, and _σ_ is the sigmoid function. This scorer ensures minimal computational overhead and prevents overfitting on the small calibration dataset, thereby focusing the learning complexity within the hierarchical meta-controller. 

**Anomaly Curve Generation.** To mitigate noise from transient visual fluctuations in the raw anomaly probability sequence _{p_ 1 _, p_ 2 _, . . . , pT }_ , we employ temporal smoothing via 1D Gaussian convolution. This process, predicated on the temporal locality of anomalous events, yields a stable anomaly curve _At_ by evaluating each point within its temporal neighborhood. The final score _At_ is computed as a weighted average over a temporal window: 



where _G_ ( _x_ ; _σg_ ) = exp( _−x_<sup>2</sup> _/_ (2 _σg_<sup>2)) is the Gaussian kernel with standard deviation</sup><sup>_σg_.and normal-</sup> ization ensures unit weight sum. This filtering suppresses transient noise while preserving sustained anomalies, yielding a smoother, more reliable detection signal for practical deployment. 

**Post-Hoc Explainability.** When segments _St_ are flagged as anomalous, the corresponding video frames can be concatenated and re-submitted to the frozen auto-regressive MLLM. The model then generates a textual description, offering explanations for the anomaly alert and enhancing the transparency and trustworthiness of our framework. 

## 3.6 TRAINING OBJECTIVE 

The trainable parameters of our framework consist of the HMC including GSG, LGM and anomaly scorer. We train the framework with a composite objective that jointly maximizes classification accuracy and enforces an inductive bias for anomaly detection. The main component of this objective is the binary cross-entropy (BCE) loss between the predicted anomaly probability _pt_ and the ground-truth _y ∈{_ 0 _,_ 1 _}_ : 



To mitigate false positives by preventing the model from overreacting to benign activity, we introduce a sparsity-inducing regularization on the global signal _s_ global for normal samples. Let _B_ norm be the set of normal samples in a training batch. The regularization loss is: 



where _s_<sup>(</sup><sup>_j_)</sup> global<sup>istheglobalsignalforthe</sup><sup>_j_-thnormalsample.ThisL2penaltyencouragesthecon-</sup> troller to remain dormant ( _s_ global _≈_ 0) for normal inputs. 

6 

(9) 

The final objective is a weighted combination of these two losses: 

_L_ total = _L_ BCE + _λ_ reg _L_ reg 

where _λ_ reg controls the regularization strength. This composite objective trains the HMC to activate its feature-modulating capacity only when necessary. Promoting sparsity in the activation of the global gate enhances the robustness of the model and reduces the potential for false positives. 

# 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL SETTINGS 

- **Datasets.** We evaluate our framework on two widely adopted, large-scale benchmarks for VAD. 

- **UCF-Crime (Sultani et al., 2018)** contains 1900 untrimmed real-world surveillance videos consist of 13 distinct anomaly types, totaling over 128 hours. The official split contains 1610 videos (810 abnormal and 800 normal) for training and 290 (140 abnormal and 150 normal) for testing. 

- **XD-Violence (Wu et al., 2020)** comprises 4754 untrimmed videos from movies, sports, and online sources. It includes 6 different anomaly categories. The dataset is divided into 3954 training videos and 800 test videos (500 abnormal and 300 normal). 

We use a small, randomly sampled 1% subset of the training sets from each dataset for our LAE discovery and the training of the HMC. The full official test sets are used for evaluation. 

**Evaluation Metric.** We adopt the standard evaluation metrics for each dataset to ensure a fair comparison with prior works. For UCF-Crime, we report the frame-level Area Under the Receiver Operating Characteristic Curve (AUC). This metric provides a comprehensive measure of the ability of a model to distinguish between normal and anomalous frames across all decision thresholds. For XD-Violence, we report the Average Precision (AP), which evaluates the precision-recall trade-off and is sensitive to the correct classification of positive (anomalous) instances. For both metrics, higher values indicate superior performance. 

**Implementation Details.** SteerVAD is built upon the InternVL3 (Zhu et al., 2025) as the frozen MLLM backbone. For video processing, each video is divided into non-overlapping segments at an interval of 48 frames and uniformly sample _F_ = 16 frames from each segment as the input to MLLM. For the offline calibration (LAE discovery and HMC training), we select the top _K_ = 4 attention heads as the LAEs based on the RSA scores. The HMC and anomaly scorer are trained for 1000 epochs using the Adam optimizer with a learning rate of 1 _×_ 10<sup>_−_3</sup> and a batch size of 64, the regularization weight _λ_ reg = 0 _._ 1, and the standard deviation for the Gaussian smoothing kernel _σg_ = 6. All experiments were conducted on a single NVIDIA RTX A6000 GPU. 

## 4.2 COMPARISON WITH STATE-OF-THE-ART METHODS 

As detailed in Table 1, our proposed SteerVAD establishes a new state-of-the-art (SOTA) in the tuning-free video anomaly detection approaches, demonstrating consistent superiority on both the UCF-Crime and XD-Violence benchmarks. This strong performance is attributed to our active intervention mechanism, where a hierarchical meta-controller executes targeted geometric steering within the feature space. Unlike passive methods that rely on fixed representations from a pretrained MLLM, our approach dynamically rectifies the feature space to counteract inherent model biases. This process selectively amplifies anomalous signals that would otherwise be obscured. 

Furthermore, SteerVAD narrows the performance gap with fully-trained fine-tuned methods while operating with exceptional data efficiency. On UCF-Crime, our model achieves an AUC of 87.15%, which is highly competitive with the 89.51% AUC of the Holmes-VAD. Crucially, SteerVAD utilizing a minuscule fraction of the data demanded by its counterparts. This validates that leveraging the global understanding of an MLLM through targeted geometric steering presents a compelling and efficient alternative to expensive fine-tuning for adapting MLLMs to specialized downstream tasks. 

## 4.3 VISUALIZATION OF LATENT SPACE RECTIFICATION 

To provide a qualitative assessment of our manifold rectification strategy, we employ t-SNE to visualize the latent feature distributions from the XD-Violence and UCF-Crime datasets in Figure 4. The 

7 

**Table 1:** Comparison with existing methods on the UCF-Crime and XD-Violence datasets. 

|**Mode**|**Methods**|**Backbone**|**Param**|**s**<br>**UCF**|**XD**|
|---|---|---|---|---|---|
|||||**AUC (%)**|**AP (%)**|
||Wu et al. (Wu et al.,2020)|I3D|-|82.44|73.20|
||MIST (Feng et al.,2021)|I3D|-|82.30|-|
||RTFM (Tian et al.,2021)|I3D|-|84.30|77.81|
||<br>S3R (Wu et al.,2022)|I3D|-|85.99|80.26|
|Weakl|MSL (Li et al.,2022b)|I3D|-|85.30|78.28|
|y<br>|UR-DMU (Zhou et al.,2023)|I3D|-|86.97|81.66|
|Supervised|<br>MFGN (Chen et al.,2023)|I3D|-|86.98|79.19|
||<br>Wu et al. (Wu et al.,2024a)|ViT|-|86.40|66.53|
||CLIP-TSA (Joo et al.,2023)|ViT|-|87.58|82.19|
||<br>Yang et al. (Yang et al.,2024)|ViT|-|87.79|83.68|
||VadCLIP (Wu et al.,2024c)|ViT|-|88.02|84.51|
|Slf|TUR et al. (Tur et al.,2023)|Resnet|-|66.85|-|
|e<br>Suervised|BODS (Wang & Cherian,2019)|I3D|-|68.26|-|
|p|GODS (Wang & Cherian,2019)|I3D|-|70.46|-|
|i|GCL (Zaheer et al.,2022)|ResNext|-|71.04|-|
|Unsupervsed|DYANNET (Thakare et al.,2023)|I3D|-|84.50|-|
||Holmes-VAU (Zhang et al.,2024b)|ViT+LLM|_∼_2B|88.96|87.68|
|Fine-Tuned|<br>Holmes-VAD (Zhang et al.,2024a)|ViT+LLM|_∼_7B|89.51|90.67|
||**_Full-Training Free_**|||||
||<br>Zero-Shot CLIP (Radford et al.,2021|)<br>ViT|_∼_0.4B|53.16|17.83|
||<br>ZS ImageBind (Girdhar et al.,2023)|ViT|_∼_1.2B|55.78|25.36|
||<br>LLAVA-1.5 (Liu et al.,2024a)|ViT+LLM|_∼_7B|72.84|50.26|
|Tuning-Free|<br>LAVAD (Zanella et al.,2024)|BLIP-2+LLM|_∼_14B|80.28|62.01|
|Multimodal<br>|EventVAD (Shao et al.,2025)|ViT+LLM|_∼_7B|82.03|64.04|
|VAD|**_Few-Parameter Learning_**|||||
||<br>VERA (Ye et al.,2025)|ViT+LLM|_∼_8B|86.55|70.54|
||HiProbeVAD (Cai et al.,2025)|ViT+LLM|_∼_8B|86.72|82.15|
||**SteerVAD (Ours)**|**ViT+LLM**|_∼_**8B**|**87.15**|**83.02**|
|Origin|al<br>Rectified|Original||Re|ctified|
|50<br>2<br>Normal<br>Abnormal|60<br>2<br>~~Normal~~<br>Abnormal|40<br>2<br>Normal<br>||2|Normal<br>|
|ent<br>|40<br>ent<br>|ent<br>Abnormal||20<br>ent|Abnormal|
|25<br>pon|pon|20<br>on||on||
|0<br>Com|20<br>Com|0<br>Comp||0<br>Comp||
|E|0<br>E|||||
|25<br>N|N|20<br>NE||20<br>NE||
|50<br>t-S|20<br>t-S|t-S||t-S||
|40<br>20<br>t-SNE Comp|0<br>onent 1<br>50<br>25<br>0<br>25<br>50<br>t-SNE Component 1|30<br>20<br>10<br>t-SNE Compon|0<br>10<br>ent 1|20<br>0<br>t-SNE C|20<br>40<br>60<br>omponent 1|



**Figure 4:** Visualization of manifold rectification via t-SNE on aggregated features from the LAEs for the XD-Violence and UCF-Crime datasets. 

original features extracted by the frozen MLLM exhibit a high degree of overlap, where the representations of normal (blue) and anomalous (red) samples are largely indistinguishable. This observation is consistent across both datasets, highlights the representational bias that hinders downstream separability. In contrast, after the application of our hierarchical meta-controller, the rectified feature space is fundamentally restructured. The same samples now form two distinct, compact clusters with a large inter-class margin. This enhancement in separability qualitatively demonstrates that our method effectively remaps the latent manifold to mitigate the inherent bias, yielding representations more conducive to classification. Further visualizations are available in Appendix G. 

## 4.4 ABLATION STUDIES 

**Effectiveness of the Hierarchical Meta-Controller.** We validate the design of our hierarchical meta-controller (HMC) through the ablation studies presented in Table 2 (Part A). The strategy that trains a linear classifier on frozen raw LAE features, achieves only 81.33% AUC, confirming the need for feature rectification. We then explore context-agnostic modulation strategies. A static steer- 

8 

**Table 2:** Ablation studies on the UCF-Crime dataset, analyzing the impact of key components and expert selection strategies. 

|**Configuration / Strategy**|**GSG**|**LGM**|**Steering**|**AUC (%)**|
|---|---|---|---|---|
|**_Part A: HMC Component_**|**_Ablation_**||||
|**Full Model**|✓|✓|**Anisotropic**|**87.15**|
|– w/o Global Gate|✗|✓|Anisotropic|85.94|
|– w/ Additive Steering|✓|✓|Additive|85.02|
|– w/o LGM (Static)|✓|✗|Static Scaling|84.21|
|– Linear Probing|✗|✗|NA|81.33|
|**_Part B: Expert Selection S_**|**_trategy A_**|**_blation_**|||
|**RSA (Ours)**||–||**87.15**|
|Mid-to-Last Layer||–||84.68|
|Random Selection||–||69.57|
|First-to-Mid Layer||–||63.16|



**Table 3:** Stability analysis of LAE selection and downstream performance across 10 independent runs with different random seeds on the 1% calibration subset. 

|**Seed ID**|7270|860|5390|5191|5734|6265|466|4426|5578|8322|**Statistics**|
|---|---|---|---|---|---|---|---|---|---|---|---|
|**Selected LAEs**|||Ident|ical (L1|8H4, L23|H24, L2|1H21,|L22H7)|||**100% Match**|
|**AUC (%)**|87.05|87.12|87.18|87.14|87.09|87.21|87.16|87.10|87.13|87.19|**87.15**_±_**0.04**|



ing variant, which applies a single, learnable channel-wise scaling vector to LAE features, improves performance by +2.88% AUC. Employing a more expressive additive steering mechanism, which instead modulates features with a learnable additive bias vector, further boosts the AUC to 85.02%. However, these limited gains underscore the inherent constraints of context-unaware modulation. Relying solely on the local gating module (LGM) to perform its context-dependent anisotropic scaling, without the global scrutiny gate (GSG), results in a 1.21% AUC degradation from our full model. This highlights the GSG’s critical function as a high-level gate that determines if modulation is necessary, thereby preventing the LGM from altering features of benign frames indiscriminately. This result demonstrates that the synergistic interplay between the global gating mechanism and the local, fine-grained modulator is crucial for optimal performance. 

**Effectiveness of Representational Separability Analysis.** We validate our RSA against several heuristic selection strategies, with results shown in Part B from Table 2. Choosing heads at random or first-to-mid layers yield poor performance due to their failure to capture task-relevant semantics. Positional heuristics selecting heads from mid-to-last layers provide better results. The late-layer heads perform better due to their encoding of higher-level semantic features, but the selection lacks precision. Our RSA method, which directly optimizes for geometric separability, significantly outperforms all baselines. This performance advantage stems from RSA’s ability to measure the manifold disentanglement of each head rather than relying on qualitative architectural assumptions. By filtering for subspaces where normal and anomalous patterns naturally diverge, RSA ensures that the subsequent steering mechanism operates on the most geometrically compliant representations. The selected experts represent stable, task-specific functional circuits intrinsic to the frozen backbone, distinguishing them from transient or noisy features often selected by heuristic methods. **Calibration Stability and Robustness.** To empirically verify that these identified circuits rely on intrinsic model properties rather than artifacts of the limited 1% calibration data, we conducted a rigorous stability analysis. As detailed in Table 3, we performed 10 independent calibration runs using different random seeds. The results demonstrate exceptional robustness: the RSA algorithm identified the strictly identical set of Latent Anomaly Experts (specifically heads L18H4, L23H24, L21H21, and L22H7) across all 10 runs, resulting in negligible AUC fluctuation ( _σ ≈_ 0 _._ 04%). Furthermore, comparative analysis reveals that increasing data coverage from 1% to 100% yields marginal gains, confirming that the geometric signatures of these experts are sufficiently distinct to be robustly captured even with minimal supervision. This definitive consistency proves that our expert selection mechanism is invariant to specific data splits, establishing a solid and reproducible foundation for the subsequent geometric rectification. 

9 

**Data Efficiency and Geometric Saturation.** To rigorously quantify the calibration efficiency, we systematically investigate the performance trajectory as the calibration set size scales from the minimal 1% subset ( _N ≈_ 16 videos) up to the full 100% training set. As detailed in Table 4, the model exhibits a distinct phenomenon of asymptotic saturation. While increasing the data coverage from 1% to 100% causes the computational cost to escalate linearly from less than one minute to approximately 49 minutes, the resulting performance gain is statistically negligible, yielding only a +0.27% improvement in AUC. This sharp plateau indicates that the HMC does not rely on massive data density to memorize patterns; instead, it rapidly converges to an optimal steering policy by capturing the intrinsic functional circuits of the backbone. This confirms that the geometric signature of anomalies is a low-rank property that can be robustly estimated with minimal supervision. Visual evidence in Appendix further corroborates this stability, showing that the spatial distribution of identified experts remains invariant across all data scales. 

**Table 4:** Data efficiency analysis. The rapid performance saturation indicates that the geometric steering policy converges to an optimal solution with minimal supervision (1%). 

|**Data Ratio**|**AUC (%)**|**Improvement**|**Training Time**|
|---|---|---|---|
|**1% (Ours)**|**87.15**|-|_<_**1 min**|
|10%|87.29|+0.14|_∼_5 min|
|25%|87.35|+0.20|_∼_13 min|
|50%|87.39|+0.24|_∼_25 min|
|100%|87.42|+0.27|_∼_49 min|



**Table 5:** Hyperparameter analysis of selected LAEs ( _K_ ) and input frames ( _F_ ) on UCF-Crime. 

|**Number of Experts (**_K_**)**|**Number of Frames (**_F_**)**|
|---|---|
|**Value**<br>**AUC (%)**|**Value**<br>**AUC (%)**|
|2<br>85.91|4<br>82.54|
|**4**<br>**87.15**|8<br>85.03|
|8<br>87.09|**16**<br>**87.15**|
|16<br>86.88|24<br>87.23|



**Hyperparameter Analysis** We evaluate two critical hyperparameters in SteerVAD: the number of selected latent anomaly experts ( _K_ ) and sampled frames ( _F_ ) from each segment. As detailed in Table 5, the performance exhibits a clear parabolic relationship with _K_ ; the AUC peaks at 87.15% when _K_ = 4 by striking an optimal balance between incorporating diverse, anomaly-centric features and mitigating the noise introduced by less discriminative experts at higher values. The choice of _F_ reveals a classic pattern of diminishing returns. Extending the input frames to 16 yields a substantial performance gain by integrating richer contextual information, this improvement saturates thereafter offering negligible further gains that do not justify the considerable increase in computational overhead. Therefore, we adopt _K_ = 4 and _F_ = 16 as our final configuration, representing the most compelling trade-off between model efficacy and computational efficiency. 

# 5 CONCLUSION 

In this paper, we introduce a novel intervention framework for tuning-free video anomaly detection, moving from passive feature interpretation to active geometric rectification within frozen MLLMs. Our SteerVAD operationalizes this by using representational separability analysis (RSA) to pinpoint task-aligned latent anomaly experts and employing a hierarchical meta-controller (HMC) to dynamically rectify their representation manifolds through context-aware anisotropic scaling. This active steering mechanism effectively overcomes pre-training biases and resolves contextual ambiguities, leading to new state-of-the-art performance on the UCF-Crime and XD-Violence benchmarks among tuning-free methods. Our work demonstrates that targeted, dynamic intervention is a powerful and data-efficient alternative to costly fine-tuning, paving the way for more adaptable applications of frozen foundation models for video anomaly detection. 

10 

# ACKNOWLEDGEMENTS 

This work was supported in part by the National Natural Science Foundation of China under Grant 62225112, 62595732, 62401365, 62271312, 62132006, U24A20220, in part by the Sichuan Science and Technology Program under Grant 2026YFHZ0205, and in part by the XJTU Research Fund for Al Science, and in part by the China Postdoctoral Science Foundation under Grants BX20250411, 2025M773473. 

# ETHICS STATEMENT 

Our work strictly adheres to the ICLR Code of Ethics. The research is conducted exclusively on publicly available, standard academic benchmarks (UCF-Crime and XD-Violence), which are widely used for evaluating video anomaly detection algorithms. We acknowledge the dual-use nature of video anomaly detection technology. However, our primary motivation is to advance the field for societal benefit, such as enhancing public safety through intelligent surveillance and improving industrial quality control. We believe that our framework represents a step towards more responsible AI by promoting transparency and intervention accountability. Unlike opaque fine-tuning methods that alter millions of parameters, our approach isolates intervention to a small, identifiable set of latent anomaly experts (LAEs). This architectural choice makes the model’s decision-making process more scrutable. Furthermore, our post-hoc explanation capability is not merely a descriptive feature; it provides a user-facing check on the system’s outputs, allowing human operators to verify alerts against generated textual rationales. By focusing on rectifying pre-training biases at a geometric level, our work also contributes to mitigating one of the core ethical challenges in deploying large models. We advocate for the responsible development and deployment of this technology, ensuring its application aligns with ethical principles and contributes positively to human well-being. 

# REPRODUCIBILITY STATEMENT 

We are fully committed to ensuring the complete reproducibility of our research. To this end, we provide an exhaustive description of our methodology in Section 3, supplemented by a rigorous theoretical analysis in Appendix A and detailed implementation specifics, including architectural layouts and tensor transformations, in Appendix B. All crucial implementation details, including the specific frozen MLLM backbone used (InternVL3), dataset preprocessing steps, and a complete list of hyperparameters, are clearly documented in Section 4 and Appendix C. Our experiments are built on standard public benchmarks and common libraries (PyTorch, Hugging Face Transformers) to ensure broad accessibility. Furthermore, to significantly enhance the verifiability of our results, we will provide comprehensive JSON file samples of our experimental run outputs in the supplementary materials. We also pledge to release our source code upon the publication of this paper, which we believe will allow the community to fully replicate our findings and extend our work. 

# REFERENCES 

- Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_ , February 2025. doi: 10.48550/arXiv.2502.13923. 

- Jing Bi, Junjia Guo, Yunlong Tang, Lianggong Bruce Wen, Zhang Liu, and Chenliang Xu. Unveiling visual perception in language models: an attention head analysis approach. _arXiv preprint arXiv:2412.18108_ , December 2024. doi: 10.48550/arXiv.2412.18108. 

- Daniel Bogdoll, Maximilian Nitsche, and J. Marius Zollner. Anomaly detection in autonomous driving: A survey. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)_ , pp. 4487–4498. IEEE, June 2022. doi: 10.1109/ CVPRW56347.2022.00495. 

11 

- Jean-Daniel Boissonnat and Arijit Ghosh. Manifold reconstruction using tangential delaunay complexes. In _Proceedings of the Twenty-Sixth Annual Symposium on Computational Geometry_ , SoCG ’10, pp. 324–333, New York, NY, USA, 2010. Association for Computing Machinery. ISBN 9781450300162. doi: 10.1145/1810959.1811013. URL https://doi.org/10. 1145/1810959.1811013. 

- Zhaolin Cai, Fan Li, Ziwei Zheng, and Yanjun Qin. Hiprobe-vad: Video anomaly detection via hidden states probing in tuning-free multimodal llms. _arXiv preprint arXiv:2507.17394_ , 2025. 

- Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection. In _Proceedings of the AAAI conference on artificial intelligence_ , pp. 387–395, 2023. 

- Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 14004–14013. IEEE, June 2021. doi: 10.1109/CVPR46437.2021.01379. 

- Christopher R. Genovese, Marco Perone-Pacifico, Isabella Verdinelli, and Larry A. Wasserman. Minimax manifold estimation. _ArXiv_ , abs/1007.0549, 2010. URL https://api. semanticscholar.org/CorpusID:8833773. 

- Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind one embedding space to bind them all. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 15180–15190. IEEE, June 2023. doi: 10.1109/CVPR52729.2023.01457. 

- Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In _Proceedings of the IEEE International Conference on Image Processing (ICIP)_ , pp. 3230–3234. IEEE, 2023. doi: 10.1109/ICIP49359.2023.10222289. 

- Federico Landi, Cees G. M. Snoek, and Rita Cucchiara. Anomaly locality in video surveillance. _arXiv preprint arXiv:1901.10364_ , January 2019. doi: 10.48550/arXiv.1901.10364. 

- Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: easy visual task transfer. _arXiv preprint arXiv:2408.03326_ , October 2024. doi: 10.48550/arXiv.2408.03326. 

- Guoqiu Li, Guanxiong Cai, Xingyu Zeng, and Rui Zhao. Scale-aware spatio-temporal relation learning for video anomaly detection. In _ECCV 2022_ , volume 13664, pp. 333–350. Springer Nature Switzerland, 2022a. doi: 10.1007/978-3-031-19772-7 ~~2~~ 0. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pp. 19730–19742. PMLR, 2023. 

- Shuo Li, Fang Liu, and Licheng Jiao. Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pp. 1395–1403, 2022b. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36:34892–34916, 2023. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 26286–26296. IEEE, June 2024a. doi: 10.1109/CVPR52733.2024.02484. 

- Jiaqi Liu, Guoyang Xie, Jinbao Wang, Shangnian Li, Chengjie Wang, Feng Zheng, and Yaochu Jin. Deep industrial image anomaly detection: A survey. _Machine Intelligence Research_ , 21(1): 104–135, January 2024b. ISSN 2731-5398. doi: 10.1007/s11633-023-1459-z. URL http: //dx.doi.org/10.1007/s11633-023-1459-z. 

12 

- Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection - a new baseline. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 6536–6545. IEEE, June 2018. doi: 10.1109/CVPR.2018.00684. 

- Zhian Liu, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pp. 13568–13577. IEEE, October 2021. doi: 10.1109/ICCV48922.2021.01333. 

- Haoyan Luo and Lucia Specia. From understanding to utilization: a survey on explainability for large language models. _arXiv preprint arXiv:2401.12874_ , 2024. 

- Weixin Luo, Wen Liu, Dongze Lian, and Shenghua Gao. Future frame prediction network for video anomaly detection. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 44(11): 7505–7520, 2022. doi: 10.1109/TPAMI.2021.3129349. 

- Hui Lv, Chen Chen, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang. Learning normal dynamics in videos with meta prototype network. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 15420–15429. IEEE, June 2021. doi: 10.1109/ CVPR46437.2021.01517. 

- Omar Mahmoud, Buddhika Laknath Semage, Thommen George Karimpanal, and Santu Rana. Improving multilingual language models by aligning representations through steering. _arXiv preprint arXiv:2505.12584_ , August 2025. doi: 10.48550/arXiv.2505.12584. 

- Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. Multi-attribute steering of language models via targeted intervention. _arXiv preprint arXiv:2502.12446_ , July 2025. doi: 10.48550/arXiv.2502.12446. 

- Ghazal Alinezhad Noghre, Armin Danesh Pazho, and Hamed Tabkhi. A survey on video anomaly detection via deep learning: Human, vehicle, and environment. _arXiv preprint arXiv:2508.14203_ , 2025. 

- Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. _arXiv preprint arXiv:2103.00020_ , February 2021. doi: 10.48550/arXiv.2103.00020. 

- Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Scholkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 14298–14308. IEEE, June 2022. doi: 10.1109/CVPR52688.2022.01392. 

- Yihua Shao, Haojin He, Sijie Li, Siyu Chen, Xinwei Long, Fanhu Zeng, Yuxuan Fan, Muyang Zhang, Ziyang Yan, Ao Ma, Xiaochen Wang, Hao Tang, Yan Wang, and Shuyan Li. Eventvad: training-free event-aware video anomaly detection. _arXiv preprint arXiv:2504.13092_ , July 2025. doi: 10.48550/arXiv.2504.13092. 

- Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 6479–6488. IEEE, June 2018. doi: 10.1109/CVPR.2018.00678. 

- Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. Dyannet: A scene dynamicity guided self-trained video anomaly detection network. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pp. 5530– 5539. IEEE, January 2023. doi: 10.1109/WACV56688.2023.00550. 

13 

- Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W. Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pp. 4955–4966. IEEE, October 2021. doi: 10.1109/ICCV48922.2021.00493. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Anil Osman Tur, Nicola Dall’Asen, Cigdem Beyan, and Elisa Ricci. Unsupervised video anomaly detection with diffusion models conditioned on compact motion representations. In _International Conference on Image Analysis and Processing_ , pp. 49–62. Springer, 2023. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. _arXiv preprint arXiv:1706.03762_ , August 2017. doi: 10.48550/arXiv.1706.03762. 

- Haibo Wang, Chenghang Lai, Yixuan Sun, and Weifeng Ge. Weakly supervised gaussian contrastive grounding with large multimodal models for video question answering. In _Proceedings of the ACM International Conference on Multimedia_ , pp. 5289–5298. ACM, October 2024. doi: 10. 1145/3664647.3680826. 

- Jue Wang and Anoop Cherian. Gods: Generalized one-class discriminative subspaces for anomaly detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pp. 8200–8210. IEEE, October 2019. doi: 10.1109/ICCV.2019.00829. 

- Zihao Wang and Victor Veitch. Does editing provide evidence for localization? _arXiv preprint arXiv:2502.11447_ , 2025. 

- Nick Whiteley, Annie Gray, and Patrick Rubin-Delanchy. Statistical exploration of the manifold hypothesis. _arXiv preprint arXiv:2208.11665_ , 2022. 

- Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Selfsupervised sparse representation for video anomaly detection. In _ECCV 2022_ , volume 13673, pp. 729–745. Springer Nature Switzerland, 2022. doi: 10.1007/978-3-031-19778-9 ~~4~~ 2. 

- Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: learning multimodal violence detection under weak supervision. In _ECCV 2020_ , volume 12375, pp. 322–339. Springer International Publishing, 2020. doi: 10.1007/ 978-3-030-58577-8 ~~2~~ 0. 

- Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. _arXiv preprint arXiv:2311.07042_ , 2024a. 

- Peng Wu, Xuerong Zhou, Guansong Pang, Zhiwei Yang, Qingsen Yan, Peng Wang, and Yanning Zhang. Weakly supervised video anomaly detection and localization with spatio-temporal prompts. In _Proceedings of the ACM International Conference on Multimedia_ , pp. 9301–9310. ACM, October 2024b. doi: 10.1145/3664647.3681442. 

- Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pp. 6074–6082, 2024c. 

- Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: mixtureof-experts vision-language models for advanced multimodal understanding. _arXiv preprint arXiv:2412.10302_ , 2024d. 

- Dan Xu, Yan Yan, Elisa Ricci, and Nicu Sebe. Detecting anomalous events in videos by learning deep representations of appearance and motion. _Computer Vision and Image Understanding_ , 156: 117–127, March 2017. doi: 10.1016/j.cviu.2016.10.010. 

14 

- Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection. _arXiv preprint arXiv:2404.08531_ , 2024. 

- Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David J. Crandall. Dota: Unsupervised detection of traffic anomaly in driving videos. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 45(1):444–459, 2023. doi: 10.1109/TPAMI.2022.3150763. 

- Muchao Ye, Weiyang Liu, and Pan He. Vera: Explainable video anomaly detection via verbalized learning of vision-language models. In _Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)_ , pp. 8679–8688, June 2025. 

- Tongtong Yuan, Xuange Zhang, Kun Liu, Bo Liu, Chen Chen, Jian Jin, and Zhenzhen Jiao. Towards surveillance video-and-language understanding: New dataset, baselines, and challenges. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 22052–22061. IEEE, June 2024. doi: 10.1109/CVPR52733.2024.02082. 

- M. Zaigham Zaheer, Arif Mahmood, M. Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 14724–14734. IEEE, June 2022. doi: 10.1109/CVPR52688.2022.01433. 

- Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 18527–18536. IEEE, June 2024. doi: 10.1109/CVPR52733.2024.01753. 

- Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ , 2023. 

- Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Chuchu Han, Xiaonan Huang, Changxin Gao, Yuehuan Wang, and Nong Sang. Holmes-vad: towards unbiased and explainable video anomaly detection via multi-modal llm. _arXiv preprint arXiv:2406.12235_ , June 2024a. 

- Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Xiaonan Huang, Changxin Gao, Shanjun Zhang, Li Yu, and Nong Sang. Holmes-vau: towards long-term video anomaly understanding at any granularity. _arXiv preprint arXiv:2412.06171_ , December 2024b. 

- Ziwei Zheng, Junyao Zhao, Le Yang, Lijun He, and Fan Li. Spot risks before speaking! unraveling safety attention heads in large vision-language models. _arXiv preprint arXiv:2501.02029_ , 2025. 

- Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. _arXiv preprint arXiv:2302.05160_ , 2023. 

- Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Kun Wang, Yang Liu, Junfeng Fang, and Yongbin Li. On the role of attention heads in large language model safety. _arXiv preprint arXiv:2410.13708_ , 2024. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , October 2023. 

- Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. _arXiv preprint arXiv:2504.10479_ , 2025. 

- Adam Zweiger, Jyothish Pari, Han Guo, Ekin Aky¨urek, Yoon Kim, and Pulkit Agrawal. Selfadapting language models. _arXiv preprint arXiv:2506.10943_ , 2025. 

15 

# A THEORETICAL ANALYSIS OF THE REPRESENTATION MANIFOLD 

This section formalizes the theoretical underpinnings of our geometric steering methodology. We argue that the emergence of structured, low-dimensional representational geometries is a natural consequence of applying continuous deep networks to high-dimensional, semantically coherent data. Our argument proceeds by first establishing the topological properties of the input data space, then analyzing how these properties are preserved under the MLLM’s feature mapping. 

## A.1 TOPOLOGICAL PROPERTIES OF THE INPUT DATA SPACE 

We begin by characterizing the topological structure of the data space _V ⊂_ R<sup>_N_</sup> from which video segments are drawn. A semantic class _C_ (e.g., ”normal” or ”anomaly”) corresponds to a subset of this space, denoted _VC ⊂V_ . 

**Compactness.** We model the data subset _VC_ for any given class _C_ as a compact set. This model is grounded in two properties. First, video data is inherently bounded; pixel intensities are confined to a finite range (e.g., [0 _,_ 255]) and video dimensions are fixed, making _VC_ a bounded subset of R<sup>_N_</sup> . Second, while the semantic boundary between classes can be ambiguous in practice, we posit that the underlying generative process for a class occupies a topologically closed region in the data space. This idealized assumption is a necessary abstraction for analytical tractability, allowing us to model the core data distribution as a well-defined set. Under the Heine-Borel theorem, a subset of R<sup>_N_</sup> that is both closed and bounded is compact. This compactness provides a stable topological foundation, ensuring that continuous maps such as the attention heads in an MLLM exhibit wellbehaved properties when applied to the data. 

**Piecewise Path-Connectedness.** The data structure for complex semantic categories is inherently piecewise, not monolithic. A high-level concept such as ”anomaly” encompasses a diverse family of distinct scenarios; for instance, ”arson” in a building and a ”vehicle collision” on a street are both anomalies, yet no continuous transformation path of semantically valid videos exists in the high-dimensional data space to connect them. 

This intrinsic heterogeneity dictates that the data space for a class _C_ is most accurately modeled as a union of path-connected subsets. Formally, we express this structure as: 



where each component _VC_<sup>(</sup><sup>_i_)</sup> is a path-connected and compact set, representing a semantically coherent subcluster (e.g., ”theft” and ”vandalism” as components of ”anomaly”). Crucially, these components are not necessarily disjoint; a single event, such as breaking a car window to steal an item, can naturally belong to multiple subclusters. This potential for overlap in the input space is a critical feature, as it implies that the resulting representation manifolds in the latent space may also intersect, creating a complex, entangled geometry that is difficult to separate. 

This topological structure has a direct implication for the learned representation space. As the feature extractor _h_ is a continuous map, it preserves path-connectedness. Consequently, the image of _VC_ under _h_ , which we term the representation manifold _MC_ = _h_ ( _VC_ ), is also a collection of pathconnected manifold components: 



Here we formally define the representation manifold _MC_ as this complete, set-theoretic union. Crucially, the MLLM’s pre-training on vast semantic data induces a critical property in this structure: the manifold components _{M_<sup>(</sup> _C_<sup>_i_)</sup><sup>_}_are not scattered arbitrarily across the latent space but are mapped into</sup> a bounded and largely contiguous region. This geometric coherence ensures that the global properties of the parent manifold _MC_ specifically its centroid (center of mass) are not just well-defined, but serve as a geometrically meaningful representation of the class’s central tendency. Consequently, our 

16 

RSA metric is not a naive simplification but a robust measure of the separation between the centers of these high-level structures. 

This leads to a more precise geometric formulation of the VAD challenge: the task is not to separate two single, cleanly-defined manifolds, but to distinguish between two families of potentially overlapping and entangled manifold components _{M_<sup>(</sup> norm<sup>_i_)</sup><sup>_}_and</sup><sup>_{M_(</sup> anom<sup>_j_)</sup><sup>_}_.</sup> 

## A.2 FEATURE EXTRACTORS AS CONTINUOUS MAPS 

An MLLM feature extractor, such as an attention head, is modeled as a function _h_ : _V →_ R<sup>_d_head</sup> . This function is a composition of a finite number of layers _h_ = _fL ◦ fL−_ 1 _◦· · · ◦ f_ 1, each consisting of linear transformations and continuous, non-linear activation functions (e.g., GELU, Softmax). Since the composition of continuous functions is itself continuous, the entire feature extractor _h_ constitutes a continuous map from the input space _V_ to the representation space R<sup>_d_head</sup> . 

A fundamental theorem in topology states that the continuous image of a path-connected set is pathconnected. Given our model of the input data space where each component _VC_<sup>(</sup><sup>_i_)</sup> is path-connected and the map _h_ is continuous, it follows that the image of each component, _M_<sup>(</sup> _C_<sup>_i_)</sup> = _h_ ( _VC_<sup>(</sup><sup>_i_)),must</sup> also be a path-connected subset of R<sup>_d_head</sup> . This formalizes the observation that feature representations of semantically coherent subclusters coalesce into distinct geometric structures within the representation space, collectively forming the full representation manifold _MC_ . 

## A.3 LOCAL EUCLIDEAN STRUCTURE AND INTRINSIC DIMENSIONALITY 

For _M_ to qualify as a differentiable manifold, it must also be locally homeomorphic to a Euclidean space. This property emerges from the smooth mapping of local degrees of freedom in the input data to the representation space. 

Under the central tenet of the manifold hypothesis, we assume that local variations within a small neighborhood of a video _v ∈C_ (e.g., an object’s position, orientation) can be parameterized by a low-dimensional vector **u** _∈ U ⊂_ R<sup>_dM_</sup> , where _U_ is an open set and _dM ≪ N_ is the number of local generative factors. This defines video in this neighborhood as a local parameterization _v_ ( **u** ). 

The feature extractor _h_ acts on this parameterization, defining a local coordinate chart _ϕ_ : _U →_ R<sup>_d_head</sup> via the composition: 



Since _h_ is composed of differentiable (or piecewise differentiable) functions, the map _ϕ_ is also differentiable (or piecewise differentiable). This map embeds a patch from the low-dimensional parameter space R<sup>_dM_</sup> as a surface within the high-dimensional representation space R<sup>_d_head</sup> . 

The local geometry of this embedded surface is characterized by the Jacobian of the chart: 



The _dM_ column vectors of the Jacobian span the tangent space _TpM_ at a point _p_ = _ϕ_ ( **u** ). We operate under the reasonable assumption that for non-degenerate natural variations, this map is an immersion, meaning the Jacobian has full column rank, i.e., rank( _Jϕ_ ) = _dM_ . This non-collapsing condition ensures that the local dimensionality is preserved. The existence of such a differentiable atlas (a collection of charts covering _M_ ) establishes _M_ as a differentiable manifold. 

## A.4 IMPLICATIONS AND RATIONALE FOR GEOMETRIC STEERING 

**Geometric Problem Formulation.** The preceding analysis provides a rigorous geometric framework for understanding the VAD task and justifying our intervention strategy. From this perspective, the VAD challenge arises when the representation manifolds for normal ( _M_ norm) and anomalous ( _M_ anom) data are proximal or intersect in R<sup>_d_head</sup> . Such proximity, where inf **p** _∈M_ norm _,_ **q** _∈M_ anom _∥_ **p** _−_ **q** _∥_ 2 _≈_ 0, is often due to shared local features (e.g., ”a person running” is common to both jogging and fleeing). This entanglement makes reliable separation by a static classifier difficult. 

17 

**Intervention as a Controlled Duality of Transformation.** As introduced in the methodology, our hierarchical meta-controller (HMC) learns a context-dependent transformation _T_ **c** : R<sup>_d_head</sup> _→_ R<sup>_d_head</sup> via anisotropic manifold scaling. This transformation exhibits a powerful duality: 

- **Diffeomorphic Remodeling.** When all scaling factors are non-zero, _T_ **c** acts as a local diffeomorphism, smoothly remodeling the geometric structure by anisotropically stretching and compressing the manifold along discriminative axes while preserving local topology. This alters the local metric tensor, effectively increasing the geometric distance between manifolds without tearing their topology. 

- **Singular Projection.** The true expressive power of our method is unleashed when the framework drives one or more scaling factors to zero. This induces a singularity, rendering the transformation locally non-invertible and fundamentally changing its nature from remodeling to structural simplification. Geometrically, this singular transformation acts as a projection, mapping the feature space R<sup>_d_head</sup> onto a lower-dimensional effective subspace spanned by the dimensions with non-zero scaling. Information within the projection’s null space, the dimensions corresponding to zero scaling, is actively discarded. This mechanism serves as a form of potent, contextaware feature selection, allowing the model to completely nullify dimensions associated with pre-training biases or task-irrelevant variance. This projection effectively creates what can be conceptualized as a stable anchor space; representations are anchored to this simpler subspace, and their final structure becomes robust to perturbations in the suppressed dimensions, whose outputs are mapped to a stable fixed point (e.g., the origin) within the projected subspace. 

This capability justifies our choice of anisotropic manifold scaling. It is not merely a heuristic but a highly expressive yet efficient transformation capable of learning both smooth, geometry-preserving diffeomorphisms and radical, context-aware projections for dimensional reduction. This provides a minimal yet powerful mechanism to achieve the desired manifold separation. 

**Theoretical Objective and Practical Realization.** The ultimate objective of our adaptive transformation is to maximize the geometric separation between transformed manifolds, thereby resolving structural ambiguity in their latent spaces. This objective can be formally expressed as maximizing a metric like the Hausdorff distance, _dH_ ( _T_ **c** ( _M_ norm) _, T_ **c** ( _M_ anom)). While direct optimization of such a metric is often intractable, our practical training objective, binary cross-entropy (BCE) loss serves as an effective and stable surrogate. By penalizing misclassifications, the BCE loss implicitly compels the HMC to learn a transformation _T_ **c** that renders the manifolds maximally separable to a linear classifier. This aligns the tractable optimization process with our fundamental geometric goal, training an expressive transformation to achieve a well-defined structural objective. 

This concludes the theoretical derivation, which demonstrates that representation manifolds are a predictable consequence of applying continuous deep networks to structured data and provides a rigorous foundation for our geometric steering approach. 

## A.5 TOPOLOGICAL GENERALIZATION TO OPEN-SET SCENARIOS 

A fundamental consideration in applying the manifold hypothesis to real-world surveillance is the unbounded, open-set nature of video data. While the input space of raw pixels _V_ is theoretically infinite, we posit that the latent representation space induced by a pre-trained MLLM imposes a structured topology that facilitates generalization to unseen events. Here, we extend our theoretical framework to formally account for open-set robustness. 

**Open-Set as Modular Component Expansion.** Recall from Eq. 11 that the anomaly manifold _Manom_ is modeled not as a monolithic shape, but as a union of semantically coherent pathconnected components: _Manom_ =<sup>�</sup> _i_<sup>_M_</sup> _anom_<sup>(</sup><sup>_i_).Underopen-setconditions,anunseenanomaly</sup> type does not violate this manifold structure; rather, it manifests as the emergence of a novel topological component, _M_<sup>(</sup> _new_<sup>_k_).Critically, due to the generalized discriminative capacity of the foundation</sup> model, this new component _M_<sup>(</sup> _new_<sup>_k_)is mapped to a region topologically distinct from the normality</sup> manifold _Mnorm_ . Our framework relies on the principle of geometric invariance, while the intrinsic coordinates of _M_<sup>(</sup> _new_<sup>_k_)are unknown during calibration, its geometric divergence from the normality</sup> baseline remains a consistent, detectable property. 

18 

**RSA as Geometric Consistency Filter.** The Representational Separability Analysis (RSA) operationalizes this principle by functioning as a geometric filter. Rather than searching for attention heads that classify specific known anomalies, RSA identifies functional subspaces where the topological separation between the support of _Mnorm_ and any divergent signal is naturally maximized. Subspaces exhibiting chaotic mapping where open-set samples might collapse into the normality manifold are assigned low separability scores and are strictly excluded from the intervention scope. 

**Entropy-Driven Steering Policy.** Consequently, the intervention policy learned by the hierarchical meta-controller (HMC) is not one of closed-set classification, but of context-relative divergence correction. Conditioned on the global context _c_ , the HMC identifies directions in the tangent space _TpM_ that exhibit high deviation from the expected context coherence (analogous to high semantic entropy). This formulation explains the model’s ability to handle unseen data, novel anomalies act as high-entropy perturbations that trigger the pre-learned steering mechanism to project them away from _Mnorm_ , regardless of their specific semantic content. 

# B DETAILS OF STEERVAD 

This section provides a detailed description of the implementation of our proposed SteerVAD framework, complementing the methodology presented in Section 3. We elaborate on the precise mathematical formulations, the architectural specifications of our trainable modules, and the algorithmic procedures for latent anomaly expert (LAE) discovery, model training, and real-time inference. 

## B.1 PREPARATION OF STEERVAD 

## B.1.1 REPRESENTATIONAL SEPARABILITY ANALYSIS (RSA) 

The representational separability analysis (RSA) is a systematic, gradient-free procedure designed to identify a select set of latent anomaly experts (LAEs). The objective is to pinpoint attention heads whose internal feature spaces are most naturally aligned with the VAD task, meaning they maximize a criterion of class separability. This section provides the detailed mathematical formulation and algorithmic implementation, complementing the conceptual overview in Section 3. The complete procedure is formally outlined in Algorithm 1. 

At its core, RSA leverages a geometric separability criterion inspired by the Fisher Discriminant Ratio to evaluate and rank the representational efficacy of attention heads. This metric is deliberately chosen for its computational tractability, as it operates solely on first and second order statistics (mean and variance), and circumvents the need to train auxiliary classifiers per head, which would incur prohibitive computational and optimization overhead. More fundamentally, its mathematical objective of maximizing between-class scatter while minimizing within-class scatter directly aligns with our architectural goal: to steer the manifold and rectify them for unbiased detection. An attention head that scores highly under this criterion induces a feature geometry in which intra-class compactness and inter-class margin are simultaneously optimized, rendering the representation ideally suited for subsequent classification by a simple and parameter-efficient linear scorer. 

For each attention head _hl,k_ (at layer _l_ , index _k_ ), its RSA score is computed as the ratio of the between-class scatter to the total within-class scatter: 



where _ϵ_ is a small constant for numerical stability. 

The numerator, _SB_ ( _l, k_ ), represents the between-class scatter. It is defined as the squared Euclidean distance between the centroids of the anomalous and normal feature distributions, rewarding heads that map the two classes to distant locations in the embedding space. It is calculated as: 



The denominator, _SW_ ( _l, k_ ), represents the total within-class scatter. It quantifies the compactness of the clusters by summing the squared Euclidean distances of each feature vector to its respective class centroid. A lower value indicates more tightly formed, less diffuse clusters. It is defined as: 

19 



where _D_ norm/anom<sup>(</sup><sup>_l,k_)denotes the set of feature vectors</sup><sup>_{_</sup><sup>**x**</sup><sup>_}_produced by head (</sup><sup>_l, k_) for all normal and</sup> anomalous samples in the calibration set, where each **x** _∈_ R<sup>_d_head</sup> . The term **_µ_** represents the centroid (mean vector) of the corresponding feature set. 

**Algorithm 1** Representational Separability Analysis (RSA) for LAE Discovery 

- 1: **Input:** Frozen MLLM _M_ , calibration dataset _D_ calib. 

- 2: **Parameter:** Number of experts to select _K_ . 

- 3: **Output:** A set of _K_ Latent Anomaly Experts (LAEs). 

- 4: _▷ Phase 1: Prepare a class-balanced calibration set_ 

- 5: Construct the calibration set _D_ calib to balanced by randomly undersampling the majority class. 

6: _▷ Phase 2: Feature Extraction_ 7: Initialize empty feature banks for all heads ( _l, k_ ): 8: _B_ norm<sup>(</sup><sup>_l,k_)</sup><sup>_←∅_,</sup><sup>_B_</sup> anom<sup>(</sup><sup>_l,k_)</sup><sup>_←∅_</sup> 

9: **for** each video segment _St_ with label _yt_ in _D_ calib **do** 10: Perform a single forward pass of _St_ through _M_ . 11: For each attention head ( _l, k_ ), extract its output feature vector **h**<sup>(</sup> _l,k_<sup>_t_).</sup> 12: **if** _yt_ is normal **then** 13: Add **h**<sup>(</sup> _l,k_<sup>_t_)to</sup><sup>_B_</sup> norm<sup>(</sup><sup>_l,k_).</sup> 14: **else** 15: Add **h**<sup>(</sup> _l,k_<sup>_t_)to</sup><sup>_B_</sup> anom<sup>(</sup><sup>_l,k_).</sup> 16: **end if** 

17: **end for** 

18: _▷ Phase 3: Score Calculation_ 19: Initialize an empty list of scores _S ←_ []. 20: **for** each attention head ( _l, k_ ) **do** 21: Compute centroids: **_µ_**<sup>(</sup> norm<sup>_l,k_)= mean(</sup><sup>_B_</sup> norm<sup>(</sup><sup>_l,k_)),</sup><sup>**_µ_**(</sup> anom<sup>_l,k_)= mean(</sup><sup>_B_</sup> anom<sup>(</sup><sup>_l,k_)).</sup> 22: Compute between-class scatter _SB_ ( _l, k_ ) = _∥_ **_µ_** anom<sup>(</sup><sup>_l,k_)</sup><sup>_−_</sup><sup>**_µ_**(</sup> norm<sup>_l,k_)</sup><sup>_∥_</sup> 2<sup>2.</sup> 23: Compute within-class scatters: 24: _SW,_ norm =<sup>�</sup> **h** _∈B_ norm<sup>(</sup><sup>_l,k_)</sup><sup>_∥_</sup><sup>**h**</sup><sup>_−_</sup><sup>**_µ_**</sup> norm<sup>(</sup><sup>_l,k_)</sup><sup>_∥_</sup> 2<sup>2.</sup> 25: _SW,_ anom =<sup>�</sup> **h** _∈B_ anom<sup>(</sup><sup>_l,k_)</sup><sup>_∥_</sup><sup>**h**</sup><sup>_−_</sup><sup>**_µ_**</sup> anom<sup>(</sup><sup>_l,k_)</sup><sup>_∥_</sup> 2<sup>2.</sup> 26: Compute RSA score _S_ RSA( _l, k_ ) = _SB_ ( _l, k_ ) _/_ ( _SW,_ norm + _SW,_ anom + _ϵ_ ). 27: Add ( _l, k, S_ RSA( _l, k_ )) to _S_ . 

28: **end for** 

29: _▷ Phase 4: LAE Selection_ 

30: Sort _S_ in descending order based on _S_ RSA. 

- 31: **return** the top _K_ heads from the sorted list _S_ . 

**Equivalence to the Variance-Based Metric.** To ensure robust and unbiased estimation, we compute the RSA score on a class-balanced calibration set, constructed by randomly undersampling the majority class to match the size of the minority class. This procedural detail is critical as it establishes a direct mathematical link between the scatter-based formula used for implementation (Eq. 14) and the more intuitive variance-based metric presented in Methods (Eq. 1). Specifically, the variance of a cluster is defined as _σ_<sup>2</sup> = _N_<sup><u>1</u></sup> � **x**<sup>_∥_</sup><sup>**x**</sup><sup>_−_</sup><sup>**_µ_**</sup><sup>_∥_</sup> 2<sup>2.Consequently, the within-</sup> class scatter for a class is _SW_<sup>class</sup> = _N · σ_<sup>2</sup> . On a balanced dataset, the total within-class scatter becomes: _SW_ ( _l, k_ ) = _N · σ_ anom<sup>2(</sup><sup>_l, k_) +</sup><sup>_N· σ_</sup> norm<sup>2(</sup><sup>_l, k_) =</sup><sup>_N_</sup> � _σ_ anom<sup>2(</sup><sup>_l, k_) +</sup><sup>_σ_</sup> norm<sup>2(</sup><sup>_l, k_)</sup> � (17) 

20 

Substituting this into our RSA score gives: 



Since our goal is to rank the attention heads, maximizing this score is equivalent to maximizing the metric from Eq. 1, as _N_ is a positive constant that does not affect the ranking order. This balancing act prevents the statistics of a majority class from dominating the score and justifies our use of the more computationally direct scatter formulation. 

This principled, quantitative analysis allows us to survey all attention heads within the frozen MLLM and rigorously select the top- _K_ performers. These selected heads, designated as latent anomaly experts, are not just randomly chosen sub-modules; they are empirically validated as the most informative and geometrically aligned subspaces for the VAD task, making them the optimal targets for our subsequent manifold rectification. 

## B.1.2 HIERARCHICAL META-CONTROLLER (HMC) AND ANOMALY SCORER 

This section provides detailed implementations of trainable components in SteerVAD. We move beyond the conceptual overview in Section 3 to detail the precise tensor transformations, architectural choices, and design rationales essential for reproducibility. The dimensions specified here correspond to our experiments using the InternVL model, where the hidden dimension is _d_ model = 3584 and the attention head dimension is _d_ head = 128. 

**Global Context Vector Rationale** The efficacy of our context-aware rectification hinges upon the extraction of the global context vector, **c** _∈_ R<sup>_d_model</sup> . As introduced in Section 3.4, this vector is derived from the final hidden state of the frozen MLLM at the first generated token. This choice is motivated by the auto-regressive nature of the underlying language model. The model generates the initial token by constructing a holistic semantic summary through comprehensive encoding of the entire input sequence, yielding a hidden state that encapsulates global contextual understanding. Further from a structural standpoint, the causal attention mechanism ensures that this representation is purely conditioned on the inputs, free from the influence of any generated text. This yields an unbiased and information-rich summary of the scene’s semantics, providing an ideal input for the HMC while remaining computationally efficient as a natural byproduct of the single forward pass. 

**Global Scrutiny Gate (GSG)** The global scrutiny gate is used to distill the high-dimensional global context vector **c** into a single suspicion score _s_ global. It processes a batch of context vectors of shape [ _B, d_ model] to produce a batch of scores of shape [ _B,_ 1] 

**Table 6:** Architecture of the global scrutiny gate (GSG). The input batch size is denoted by _B_ . 

|**Layer**|**Input Shape**|**Output Shape**|**Activation / Notes**|
|---|---|---|---|
|Input**c**|[_B,_3584]|-|Global context vector|
|Linear-1|[_B,_3584]|[_B,_128]|-|
|BatchNorm1d|[_B,_128]|[_B,_128]|Stabilizes training|
|ReLU|[_B,_128]|[_B,_128]|Non-linearity|
|Linear-2|[_B,_128]|[_B,_1]|Output logit|
|Sigmoid|[_B,_1]|[_B,_1]|Produces_s_global|



**Architecture.** The module is implemented as a lightweight MLP, whose architecture is detailed in Table 6. The inclusion of a BatchNorm1d layer is a deliberate choice to stabilize the training dynamics on the small calibration set. It normalizes the feature activations from the first linear layer, preventing the model from becoming reliant on specific dimensions of the context vector whose statistics might be inconsistent across small batches. 

**Local Gating Module (LGM)** The local gating module is designed for parameter-efficient, context-dependent feature modulation. It consists of _K_ parallel low-rank adapters, where _K_ is the number of selected LAEs. Each adapter takes the same global context vector **c** and generates a unique steering vector **g** _i_ for its corresponding LAE. 

21 

**Table 7:** Architecture of a single low-rank adapter within the LGM. 

|**Layer**|**Input Shape**|**Output Shape**|**Activation / Notes**|
|---|---|---|---|
|Input**c**|[_B,_3584]|-|Shared across_K_ adapters|
|Linear-down (_r_ = 4)|[_B,_3584]|[_B,_4]|bias=False|
|Linear-up|[_B,_4]|[_B,_128]|bias=False|
|Tanh|[_B,_128]|[_B,_128]|Produces**g**_i ∈_[_−_1_,_1]|



**Architecture.** Each of the _K_ adapters is identical in structure, as detailed in Table 7. The design choice to set bias=False in both linear layers is intentional. It ensures that the adapter’s operation is purely modulatory (scaling existing features) rather than additive (shifting features), preserving the original manifold’s position while only altering its geometry. This aligns with our core objective of rectifying, not replacing the learned representations of MLLM. Numerical computations are performed using bfloat16 precision, consistent with the native format of the InternVL backbone. 

**Algorithm 2** Training Procedure for HMC and Anomaly Scorer 

- 1: **Input:** A batch of pre-extracted LAE features, context vectors, and labels _{_ ( _{_ **h** _i}t,_ **c** _t, yt_ ) _}_<sup>_B_</sup> _t_ =1<sup>.</sup> 

- 2: **Parameters:** HMC parameters _θ_ HMC, Scorer parameters _θ_ Scorer, regularization weight _λ_ reg. 

- 3: Initialize batch loss _L_ batch _←_ 0. 

- 4: **for** each sample ( _{_ **h** _i},_ **c** _, y_ ) in the batch **do** 

- 5: _▷ Generate Rectification Signals_ 

- 6: Compute global suspicion score _s_ global = GSG( **c** ; _θ_ HMC). 

- 7: Compute local steering vectors _{_ **g** _i}_<sup>_K_</sup> _i_ =1<sup>= LGM(</sup><sup>**c**;</sup><sup>_θ_HMC).</sup> 

- 8: _▷ Perform Anisotropic Manifold Scaling_ 9: Compute rectified features: **h**<sup>_′_</sup> _i_<sup>=</sup><sup>**h**</sup><sup>_i ⊙_(1 +</sup><sup>_s_global</sup><sup>_·_</sup><sup>**g**</sup><sup>_i_) for</sup><sup>_i_= 1</sup><sup>_, . . . , K_.</sup> 

- 10: _▷ Score and Compute Loss_ 11: Concatenate features: _F_ final = Concat( **h** 1<sup>_′, . . . ,_</sup><sup>**h**</sup><sup>_′_</sup> _K_<sup>).</sup> 

- 12: Compute logit _z_ = AnomalyScorer( _F_ final; _θ_ Scorer). 

- 13: Compute classification loss _L_ BCE = BCEWithLogitsLoss( _z, y_ ). 

- 14: _▷ Compute Regularization Loss for Normal Samples_ 

- 15: Initialize _L_ reg _←_ 0. 

- 16: **if** _y_ is normal **then** 

- 17: _L_ reg = ( _s_ global)<sup>2</sup> . 

- 18: **end if** 

- 19: Accumulate total loss: _L_ batch _←L_ batch + _L_ BCE + _λ_ reg _L_ reg. 

- 20: **end for** 

21: Perform backpropagation on _L_ batch to compute gradients _∇θ_ HMC _,θ_ Scorer _L_ batch. 

22: Update parameters _θ_ HMC and _θ_ Scorer using an optimizer (e.g., Adam). 

**Anisotropic Manifold Scaling** The core rectification mechanism, anisotropic scaling, is an element-wise tensor operation that integrates the outputs of the GSG and LGM to modulate the raw LAE features. The data flow for a batch of inputs during inference is as follows: 1. **Inputs** : The process begins with two tensors: the raw LAE features **H** _∈_ R<sup>_B×K×d_head</sup> and the global context vectors **c** _∈_ R<sup>_B×d_model</sup> . 

2. **Signal Generation** : The context **c** is passed through the GSG to produce the global scores **s** global _∈_ R<sup>_B×_1</sup> , and concurrently through the _K_ adapters of the LGM to produce the stacked local steering vectors **G** _∈_ R<sup>_B×K×d_head</sup> . 

3. **Rectification** : The global score **s** global is unsqueezed to a shape of [ _B,_ 1 _,_ 1] to enable broadcasting across the _K_ experts and _d_ head dimensions. The final rectified features **H**<sup>_′_</sup> are computed via the element-wise operation: 

**H**<sup>_′_</sup> = **H** _⊙_ (1 + unsqueeze( **s** global) _·_ **G** ) (19) 

where _⊙_ denotes the Hadamard product. This mechanism provides dynamic, fine-grained control over the feature geometry. For instance, a high global score ( _s_ global _≈_ 1) combined with positive steering signals in **G** amplifies the norm of corresponding feature vectors, while negative signals attenuate them, thus actively steering the representation on its manifold. 

22 

**Algorithm 3** Real-time Inference Pipeline of SteerVAD 1: **Input:** Untrimmed video stream _V_ ; Pre-trained HMC and Anomaly Scorer; LAE configuration. 2: **Parameters:** Frames per segment _L_ ; Sampled frames per segment _F_ ; Gaussian kernel std. dev. _σg_ ; Anomaly threshold _τ_ anomaly. 3: **Output:** Smoothed anomaly curve **A** smooth; Set of textual explanations _E_ (optional). 4: _▷ Phase 1: Frame-level Probability Scoring_ 5: Partition _V_ into _T_ non-overlapping segments _{S_ 1 _, . . . , ST }_ , each of length _L_ . 6: Initialize an empty list of probabilities _P ←_ []. 7: **for** each segment _St_ from _t_ = 1 to _T_ **do** 8: Uniformly sample _F_ frames from _St_ . 9: Perform a single forward pass through the frozen MLLM to extract raw LAE features _{_ **h** _i}t_ and context **c** _t_ . 10: With gradients disabled: 11: _s_<sup>(</sup> global<sup>_t_)</sup><sup>_←_GSG(</sup><sup>**c**</sup><sup>_t_),</sup> _{_ **g** _i}t ←_ LGM( **c** _t_ ). 12: **h**<sup>_′_</sup> _i_<sup>=</sup><sup>**h**</sup><sup>_i ⊙_(1 +</sup><sup>_s_(</sup> global<sup>_t_)</sup><sup>_·_</sup><sup>**g**</sup><sup>_i_) for</sup><sup>_i_= 1</sup><sup>_, . . . , K_.</sup> 13: _F_ final<sup>(</sup><sup>_t_)= Concat(</sup><sup>**h**</sup> 1<sup>_′, . . . ,_</sup><sup>**h**</sup><sup>_′_</sup> _K_<sup>).</sup> 14: _zt_ = AnomalyScorer( _F_ final<sup>(</sup><sup>_t_)).</sup> 15: _pt_ = _σ_ ( _zt_ ), where _σ_ is the Sigmoid function. 16: Append _pt_ to _P_ . 17: **end for** 18: _▷ Phase 2: Anomaly Curve Generation_ 19: Let _Nframes_ be the total number of frames in _V_ . Initialize frame-level scores **A** raw _∈_ R<sup>_Nframes_</sup> with zeros. 20: **for** _t_ = 1 to _T_ **do** 21: Let [ _start, end_ ) be the frame indices for segment _St_ . 22: **A** raw[ _start_ : _end_ ] = _pt_ . _▷_ Assign segment probability to corresponding frames 23: **end for** 24: Compute smoothed curve **A** smooth = GaussianConvolution1D( **A** raw _, σg_ ). 25: _▷ Phase 3: Post-Hoc Explanation (Optional)_ 26: Initialize an empty set of explanations _E ←∅_ . 27: **for** _t_ = 1 to _T_ **do** 28: Compute representative score for segment _St_ , e.g., ¯ _at_ = mean( **A** smooth[ _start_ : _end_ ]). 29: **if** ¯ _at > τ_ anomaly **then** 30: Let _Vt_ be the original frames of segment _St_ . 31: Construct prompt: _Qt_ . 32: Generate explanation: _Et_ = MLLM.generate( _Qt, Vt_ ). 33: Add ( _t, Et_ ) to _E_ . 34: **end if** 35: **end for** 36: **return A** smooth _, E_ . 

**Anomaly Scorer** The design of anomaly scorer prioritizes robustness over complexity. Its primary role is to map the rectified feature space to a final anomaly probability. 

**Architecture and Rationale.** The scorer first reshapes the batch of rectified features **H**<sup>_′_</sup> _∈_ R<sup>_B×K×d_head</sup> into a flattened feature matrix **F** final _∈_ R<sup>_B×_(</sup><sup>_K·d_head)</sup> . For our configuration with _K_ = 4, this results in an input shape of [ _B,_ 512]. This matrix is then processed by a single linear layer (nn.Linear(512, 1)) to produce a batch of logits **z** _∈_ R<sup>_B×_1</sup> . 

This minimalist design, equivalent to a logistic regression classifier, is a deliberate methodological choice that acts as a form of regularization. By constraining the classifier’s capacity, we ensure that performance gains are attributable to the quality of the rectified representations produced by the HMC, rather than the power of the final classifier. It compels the HMC to learn a feature space that is as linearly separable as possible. 

23 

**Training Procedure** The HMC and anomaly scorer are trained end-to-end using the composite loss function from Equation 9. The complete training process is detailed in Algorithm 2. 

## B.2 REAL-TIME INFERENCE OF STEERVAD 

The inference pipeline of SteerVAD is designed for efficiency, enabling the processing of long, untrimmed videos in a single pass and culminating in a final, interpretable output. The entire process, formally detailed in Algorithm 3, integrates segment-wise scoring, temporal aggregation for curve generation, and an optional mechanism for generating textual explanations for detected anomalies. 

The procedure begins by partitioning the input video into a sequence of non-overlapping temporal segments. Each segment is processed independently through a single forward pass of the frozen MLLM to extract raw LAE features and a global context vector. The pre-trained HMC and Scorer modules then compute a segment-level anomaly probability. This raw sequence of probabilities, while informative, can exhibit high-frequency noise due to transient visual fluctuations. 

To address this and produce a temporally coherent result, we introduce an aggregation step. The segment-level scores are first mapped to a frame-level representation. Subsequently, we apply a temporal smoothing filter, as implemented in our visualization code via a standard 1D Gaussian convolution. This operation, which efficiently realizes the weighted averaging formula in Equation 6, smooths the raw scores to produce the final anomaly curve, effectively suppressing spurious spikes while preserving sustained anomalous events. 

Finally, to enhance the framework’s utility and trustworthiness, an optional post-hoc explanation module can be activated. For any segment where the smoothed anomaly score surpasses a predefined threshold, the corresponding video frames are re-submitted to the frozen MLLM’s generative interface. By prompting the MLLM to describe the events in these frames, we obtain a natural language explanation for the detected anomaly, bridging the gap between a numerical score and a human-understandable reason. 

# C FURTHER EXPERIMENT DETAILS 

This section provides additional details regarding our experimental setup to ensure full reproducibility. We outline the specific implementation environment, the prompts used to query the MLLM, and a comprehensive summary of all hyperparameters used in our framework. 

## C.1 IMPLEMENTATION DETAILS 

**Model and Architecture.** Our experiments are conducted using the InternVL3-8B (Zhu et al., 2025) model as the frozen MLLM backbone. This model is based on the InternViT-300M-448pxV2.5 vision encoder and a Qwen2.5-7B-based language model (Qwen et al., 2025). The language model comprises 28 transformer layers, with each layer containing 28 attention heads. This results in a total of _N_ total = 28 _×_ 28 = 784 attention heads available for the representational separability analysis (RSA). The hidden dimension of each attention head is _d_ head = 128. 

**Hardware and Software.** All experiments were performed on a single server running Ubuntu 22.04.4 LTS. The hardware configuration features dual AMD EPYC 75F3 CPUs (128 logical cores), 755 GB of system RAM, and a single NVIDIA RTX A6000 GPU with 48 GB of VRAM. Our implementation is built upon PyTorch (v2.3.0) and the Hugging Face Transformers library (v4.51.3), utilizing CUDA 12.1 for GPU acceleration. 

## C.2 PROMPT DETAILS 

During the development of our framework, we explored several prompts to elicit the most effective global context vector from the MLLM. Below are the main variants we considered. For all experiments reported in this paper, we utilized the structured system prompt (Prompt 3), which we found to yield the most consistent and task-aligned representations for anomaly detection. 

24 

## **Prompt 1 (General Query).** 

“If there are any abnormal events that occurred in the video that deviate from the normal scene, please identify them and describe what happened.” 

## **Prompt 2 (Structured Response Query).** 

“Please carefully watch the video and identify if any abnormal events occur. Your task is to first provide a clear and concise summary of the video content, then state whether any mentioned abnormal events are present. If yes, specify which events occurred and provide a detailed description of what happened, including the timing and context. If no, explain why you believe the video is normal. Please structure your response as follows: 

- Yes or No answer first with anomaly detection. 

- **Video Summary:** A brief summary of the video content. 

- **Details:** If Yes, describe the abnormal events in detail. If No, explain your reasoning. 

Take your time to analyze the video thoroughly and ensure your assessment is accurate. Take a deep breath and work on this problem step-by-step.” 

## **Prompt 3 (Final System Prompt).** 

## **System Identity** 

You are a professional AI assistant specialized in video analysis, focusing on abnormal behavior detection in real-time surveillance. 

## **Task Context** 

Analyse if there are abnormal events happened in video frames. 

## **Core Objectives** 

1. Analyze the following video frames. 

2. Detect high-risk abnormal events which should not happen in this video. 

3. Perform spatiotemporal analysis across consecutive frames. 

## **Output Requirements** 

Implement a two-level response system with Yes or No first, then subsequent paragraphs must include: Detected event type(s), Detailed description of the behavior, and Risk level assessment (low/medium/high). 

## C.3 HYPERPARAMETER CONFIGURATION 

Table 8 provides a comprehensive summary of all hyperparameters used throughout our framework, including those for LAE discovery, HMC training, and real-time inference. 

# D FURTHER EXPERIMENTS 

## D.1 PERFORMANCE EVALUATION 

To provide a comprehensive understanding of the practicality of our framework, this section details the computational performance of SteerVAD. We evaluate the model size, training resource requirements, and the inference costs associated with the trainable components, demonstrating that our active intervention approach is not only effective but also remarkably efficient. 

**Model Size and Complexity.** The primary advantage of our tuning-free framework is the minimal size of the trainable modules. The hierarchical meta-controller (HMC) and the anomaly scorer operate on top of a frozen, multi-billion parameter MLLM, yet their own footprint is negligible. Table 9 summarizes the architectural and size statistics of these components, based on the final configuration used in our experiments ( _K_ = 4 experts). The total number of trainable parameters is approximately 0.52 million, which occupies about 1 MB of disk space. This is several orders 

25 

**Table 8:** Consolidated list of hyperparameters for the SteerVAD framework. 

|**Hyperparameter**|**Value**|**Description**|
|---|---|---|
|**_Data Processing_**|**_& Model Arc_**|**_hitecture_**|
|_L_|48 frames|The duration of each non-overlapping video segment.|
|_F_|16 frames|The number of frames sampled from each segment as input to the MLLM.|
|_K_|4|The number of top-scoring latent anomaly experts (LAEs) selected by RSA.|
|**_HMC Training_**|||
|Epochs|1000|The total number of training epochs for the HMC and anomaly scorer.|
|Batch Size|64<br>|The number of samples per batch during training.|
|Learning Rate|1_×_10<sup>_−_3</sup>|The initial learning rate for the Adam optimizer.|
|_λ_reg|0.1|The weight for the sparsity regularization loss (_L_reg).|
|_d_hidden|128|The hidden dimension of the MLP in the Global Scrutiny Gate (GSG).|
|_r_|4|The low-rank dimension of the adapters in the Local Gating Module (LGM).|
|**_Inference & Anal_**|**_ysis_**||
|_σg_|6.0|The standard deviation of the 1D Gaussian kernel for temporal smoothing.|
|_τ_anomaly|0.5<br>|The score threshold to trigger the optional post-hoc explanation module.|
|_ϵ_|1_×_10<sup>_−_8</sup>|A small constant for numerical stability in the RSA score calculation.|



of magnitude smaller than the frozen backbone, highlighting the parameter-efficient nature of our method. The computational overhead is similarly minimal, requiring just over 1 million FLOPs per segment inference, ensuring that the bottleneck remains the single forward pass through the MLLM. 

**Table 9:** Architectural size and computational complexity of SteerVAD’s trainable modules. 

|**Component**<br>**Parameters**|**Size / Complexity**|
|---|---|
|**_Parameter Count & Disk Size_**||
|Global Scrutiny Gate<br>459,265|—|
|Local Gating Module<br>59,392|—|
|**HMC (Total)**<br>**518,657**|**0.99 MB**|
|Anomaly Scorer<br>513|_<_0.01 MB|
|**Total Trainable**<br>**519,170**|**˜0.99 MB**|
|**_Computational Complexity (per infer_**|**_ence)_**|
|HMC FLOPs|1,037,568|
|Scorer FLOPs|1,024|
|**Total Added FLOPs**|**1,038,592**|



**Table 10:** Training cost statistics for calibrating the HMC and Anomaly Scorer. 

|**Metric**|**Value**|
|---|---|
|Total Training Epochs|1000|
|Batch Size|64|
|Learning Rate|0.001|
|**Total Training Time**|**27.83 seconds (0.46 minutes)**|
|Average Time per Epoch|0.03 seconds|
|Peak GPU Memory Usage|203.88 MB|
|Training Throughput|4168.81 samples/second|



**Training Cost.** The calibration of the HMC and anomaly scorer is extremely efficient. The training is performed on a small, representative calibration set randomly sampled 1% videos from the training set from UCF-Crime and Xd-Violence. As detailed in Table 10, the entire training process completes in approximately one minute on a single commodity GPU. The memory footprint is minimal, and the training throughput is exceptionally high. This low-cost calibration phase makes SteerVAD easy to adapt to new domains or datasets without incurring the prohibitive expenses associated with fine-tuning large foundation models. 

26 

**Table 11:** Inference cost as a function of the number of sampled frames ( _F_ ) per segment. 

|**Sampled Frames (**_F_**)**|**Avg. Inference Time (s)**|**Peak VRAM (GB)**|
|---|---|---|
|4|0.43|17.2|
|8|0.78|18.3|
|16|1.52|20.4|
|24|2.29|22.3|



**Inference Cost Analysis.** The primary factor influencing inference cost in our framework is the number of frames, _F_ , sampled per video segment to be processed by the MLLM. Table 11 presents the trade-off between the temporal density of this sampling and the resulting computational demands in terms of latency and GPU memory. As expected, both inference time and peak VRAM usage scale near-linearly with the number of sampled frames. This analysis demonstrates a clear and predictable trade-off, allowing practitioners to balance detection performance with their available computational budget. The configuration used in our main experiments ( _F_ = 16) was chosen to provide a robust temporal context while remaining well within the capacity of common research hardware. 

**Table 12:** Cross-dataset generalization comparison. To ensure fairness, we reproduce HiProbeVAD under identical settings with InternVL3-8B backbone with same prompt1 setting in App.C.2. 

|**Source (Calibration)**|**Target (Evaluation)**|**Metric**|**HiProbeVAD (Rep)**|**SteerVAD (Ours)**|**Gap**|
|---|---|---|---|---|---|
|UCF-Crime (1%)|XD-Violence (Unseen)|AP (%)|70.66|**71.31**|**+0.65**|
|XD-Violence (1%)|UCF-Crime (Unseen)|AUC (%)|80.82|**81.04**|**+0.22**|



**Zero-Shot Generalization.** We further evaluate the robustness of SteerVAD by testing its zeroshot transfer capability across distinct domains. In this setting, the HMC module is calibrated on a source dataset and directly applied to a completely unseen target dataset. To provide a rigorous benchmark, we compare our method against the state-of-the-art probing baseline, HiProbe-VAD, implemented with the same backbone and settings. As presented in Table 12, SteerVAD demonstrates consistent superiority over the baseline in both transfer directions, retaining high performance despite significant shifts in anomaly definitions and environmental contexts. This cross-dataset stability indicates that the hierarchical meta-controller does not overfit to superficial domain-specific cues within the small calibration set. Instead, it effectively learns a transferable geometric rectification policy, identifying and amplifying high-entropy divergences that remains valid across diverse video distributions. 

**Table 13:** Cross-model generalization performance. The framework is applied to different most widely used and advanced frozen MLLM backbones. 

|**Backbone**|**Params**|**UCF-Crime (AUC %)**|**XD-Violence (AP %)**|
|---|---|---|---|
|LLaVA-OV (Li et al.,2024)|_∼_7B|81.52|74.65|
|Qwen2.5-VL (Bai et al.,2025)|_∼_7B|84.11|79.82|
|**InternVL3 (Zhu et al., 2025)**|_∼_**8B**|**87.15**|**83.02**|



**Cross-Model Generalization.** Our active manifold steering strategy is a fundamental principle rather than a technique coupled to a specific architecture. We applied the SteerVAD framework to other prominent MLLMs, as presented in Table 13, different models including LLaVA-OV and Qwen2.5VL with our steering mechanism unlocks strong anomaly detection capabilities. On both the UCF-Crime and XD-Violence datasets, these models achieve a level of performance that is highly competitive. This outcome powerfully suggests that the existence of latent, geometrically steerable expert subspaces is a general property of MLLMs. SteerVAD framework should be understood as a model-agnostic paradigm for adapting a wide range of frozen foundation models to complex perception tasks, rather than a solution tied to a single model. 

27 

# E EXTENDED ROBUSTNESS AND GENERALIZATION ANALYSIS 

In this section, we present a comprehensive empirical evaluation of the SteerVAD framework’s adaptability. Building on the data efficiency analysis in the main text, we focus on the model’s sensitivity to class distribution imbalances, the structural stability of the expert selection process under random initialization, and its capability to generalize to strictly unseen open-set scenarios. Furthermore, we substantiate these quantitative results with extensive visual evidence of the underlying latent geometric structures. 

## E.1 VISUALIZATION OF GEOMETRIC SATURATION 

While the quantitative saturation is discussed in the main paper, we provide further intuitive visualization of the RSA score distribution across all attention heads ( _L × H_ ). To demonstrate the universality of our findings, we generate these heatmaps for all 14 categories within the UCF-Crime dataset under varying calibration set sizes. As illustrated in Figure 5 and 6, the heatmaps of anomaly sensitivity exhibit remarkable invariance. The high-scoring regions, which represent the latent anomaly experts, emerge distinctly at the 1% scale and maintain their spatial positions and relative intensities as data scales to 100%. This visual evidence, consistent across diverse anomaly semantics, powerfully corroborates that the geometric signature of anomalies is a stable, low-rank property, confirming that massive calibration data is redundant for identifying the core functional circuits. 



<!-- Start of picture text -->
Attention Head Heatmap for: Abuse 0.225 Attention Head Heatmap for: Abuse 0.250 Attention Head Heatmap for: Abuse 0.225 Attention Head Heatmap for: Abuse 0.30 Attention Head Heatmap for: Abuse<br>0.2000.175 0.2250.200 0.2000.175 0.25 0.300.25<br>0.175<br>0.150 0.150 0.150 0.20 0.20<br>0.1250.100 0.1250.100 0.1250.100 0.15 0.15<br>0.075 0.075 0.075 0.10 0.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627<br>Attention Head Heatmap for: Arrest 0.225 Attention Head Heatmap for: Arrest 0.20 Attention Head Heatmap for: Arrest Attention Head Heatmap for: Arrest 0.30 Attention Head Heatmap for: Arrest<br>0.200 0.18 0.225 0.25<br>0.1750.150 0.160.140.12 0.2000.1750.150 0.250.20 0.20<br>0.125 0.10 0.125 0.15 0.15<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.1000.0750.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.080.060.04 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.1000.0750.0500.025 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.100.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.100.05<br>Attention Head Heatmap for: Arson 0.225 Attention Head Heatmap for: Arson Attention Head Heatmap for: Arson 0.250 Attention Head Heatmap for: Arson Attention Head Heatmap for: Arson<br>0.2000.175 0.2250.2000.175 0.2250.2000.175 0.250.20 0.300.25<br>0.150 0.150 0.150 0.20<br>0.1250.100 0.1250.100 0.1250.100 0.15 0.15<br>0.075 0.075 0.075 0.10 0.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627<br>Attention Head Heatmap for: Assault 0.225 Attention Head Heatmap for: Assault 0.25 Attention Head Heatmap for: Assault 0.225 Attention Head Heatmap for: Assault Attention Head Heatmap for: Assault 0.35<br>0.2000.175 0.20 0.2000.1750.150 0.300.25 0.300.25<br>0.150 0.15 0.125 0.20 0.20<br>0.1250.100 0.10 0.1000.075 0.15 0.15<br>0.075 0.05 0.050 0.10 0.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627<br>Attention Head Heatmap for: Burglary 0.250.20 Attention Head Heatmap for: Burglary 0.2250.2000.175 Attention Head Heatmap for: Burglary 0.2000.1750.150 Attention Head Heatmap for: Burglary 0.250.20 Attention Head Heatmap for: Burglary 0.300.25<br>0.150.10 0.1500.1250.1000.075 0.1250.1000.075 0.150.10 0.200.150.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br><!-- End of picture text -->

**Figure 5:** Evolution of anomaly expert activations across different calibration data scales. The consistent distribution of hotspots from 1% to 100% visually confirms that the identification of anomaly experts saturates rapidly. 

28 



<!-- Start of picture text -->
Attention Head Heatmap for: Explosion 0.2000.1750.150 Attention Head Heatmap for: Explosion 0.2250.2000.1750.150 Attention Head Heatmap for: Explosion 0.2250.2000.1750.150 Attention Head Heatmap for: Explosion 0.300.250.20 Attention Head Heatmap for: Explosion 0.250.20<br>0.125 0.125 0.125 0.15 0.15<br>0.100 0.100 0.100<br>0.075 0.075 0.075 0.10 0.10<br>0.050 0.050 0.050 0.05 0.05<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627<br>Attention Head Heatmap for: Fighting 0.2250.2000.1750.1500.125 Attention Head Heatmap for: Fighting 0.2000.1750.1500.125 Attention Head Heatmap for: Fighting 0.180.160.140.120.10 Attention Head Heatmap for: Fighting 0.300.250.20 Attention Head Heatmap for: Fighting 0.300.250.20<br>0.100 0.100 0.08 0.15 0.15<br>0.0750.050 0.0750.050 0.060.04 0.10 0.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627<br>Attention Head Heatmap for: RoadAccidents Attention Head Heatmap for: RoadAccidents 0.20 Attention Head Heatmap for: RoadAccidents 0.22 Attention Head Heatmap for: RoadAccidents Attention Head Heatmap for: RoadAccidents<br>0.225 0.18 0.20 0.30 0.30<br>0.2000.175 0.160.14 0.180.16 0.25 0.25<br>0.150 0.12 0.14 0.20 0.20<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.1250.1000.075 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.100.080.06 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.120.100.080.06 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.150.10 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.150.10<br>Attention Head Heatmap for: Robbery 0.25 Attention Head Heatmap for: Robbery 0.25 Attention Head Heatmap for: Robbery 0.2250.200 Attention Head Heatmap for: Robbery 0.25 Attention Head Heatmap for: Robbery 0.25<br>0.20 0.20 0.175 0.20 0.20<br>0.150<br>0.15 0.15 0.125 0.15 0.15<br>0.100<br>0.10 0.10 0.075 0.10 0.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.0500.025 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05<br>Attention Head Heatmap for: Shooting Attention Head Heatmap for: Shooting Attention Head Heatmap for: Shooting Attention Head Heatmap for: Shooting Attention Head Heatmap for: Shooting<br>0.200.18 0.25 0.2250.200 0.30 0.30<br>0.16 0.20 0.175 0.25 0.25<br>0.140.120.10 0.150.10 0.1500.1250.100 0.200.15 0.200.15<br>0.08 0.075 0.10 0.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.06 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.0500.025 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05<br>Attention Head Heatmap for: Shoplifting 0.250.200.150.10 Attention Head Heatmap for: Shoplifting 0.2250.2000.1750.1500.1250.1000.075 Attention Head Heatmap for: Shoplifting 0.200.180.160.140.120.100.08 Attention Head Heatmap for: Shoplifting 0.2500.2250.2000.1750.1500.125 Attention Head Heatmap for: Shoplifting 0.2750.2500.2250.2000.1750.1500.125<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.060.04 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.1000.075 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.1000.075<br>Attention Head Heatmap for: Stealing Attention Head Heatmap for: Stealing Attention Head Heatmap for: Stealing Attention Head Heatmap for: Stealing Attention Head Heatmap for: Stealing<br>0.225 0.25 0.225 0.30 0.35<br>0.2000.175 0.20 0.2000.175 0.25 0.30<br>0.150 0.15 0.150 0.20 0.25<br>0.125 0.125 0.20<br>0.100 0.10 0.100 0.15<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.0750.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.0750.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.10 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.150.10<br>Attention Head Heatmap for: Vandalism 0.225 Attention Head Heatmap for: Vandalism 0.225 Attention Head Heatmap for: Vandalism 0.225 Attention Head Heatmap for: Vandalism 0.30 Attention Head Heatmap for: Vandalism 0.30<br>0.2000.175 0.2000.175 0.2000.175 0.25 0.25<br>0.150 0.150 0.150 0.20 0.20<br>0.1250.100 0.1250.100 0.1250.100 0.15 0.15<br>0.075 0.075 0.075 0.10 0.10<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.050 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05<br>Attention Head Heatmap for: Normal 0.30 Attention Head Heatmap for: Normal 0.30 Attention Head Heatmap for: Normal 0.45 Attention Head Heatmap for: Normal 0.50 Attention Head Heatmap for: Normal 0.50<br>0.25 0.25 0.40 0.45 0.45<br>0.20 0.20 0.35 0.40 0.40<br>0.35 0.35<br>0.15 0.15 0.30 0.30 0.30<br>0.10 0.10 0.25 0.25 0.25<br>0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.05 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.200.15 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.20 0 1 2 3 4 5 6 7 8 9 10 11Head Index12131415161718192021222324252627 0.200.15<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>024 024 024 024 024<br>6 6 6 6 6<br>810 810 810 810 810<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618 Layer Index 1618<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br>0 0 0 0 0<br>2 2 2 2 2<br>4 4 4 4 4<br>6 6 6 6 6<br>8 8 8 8 8<br>10 10 10 10 10<br>12 12 12 12 12<br>14 14 14 14 14<br>Layer Index 16 Layer Index 16 Layer Index 16 Layer Index 16 Layer Index 16<br>18 18 18 18 18<br>20 20 20 20 20<br>22 22 22 22 22<br>24 24 24 24 24<br>26 26 26 26 26<br><!-- End of picture text -->

**Figure 6:** Evolution of anomaly expert activations across different calibration data scales. The consistent distribution of hotspots from 1% to 100% visually confirms that the identification of anomaly experts saturates rapidly. 

29 

**Table 14:** Impact of normal/abnormal ratio in the calibration set on UCF-Crime performance. The model maintains robust detection capabilities across a wide range of class distributions. 

|**Ratio (N:A)**|1:9|2:8|3:7|4:6|**5:5 (Default)**|6:4|7:3|8:2|9:1|
|---|---|---|---|---|---|---|---|---|---|
|**AUC (%)**|83.67|85.69|85.75|86.34|**87.15**|87.00|85.70|85.62|83.72|



## E.2 SENSITIVITY TO CLASS DISTRIBUTION IMBALANCE 

The robustness of the hierarchical meta-controller (HMC) against label distribution shifts is a critical factor for real-world deployment. We systematically analyzed the sensitivity of our framework to varying ratios of normal versus abnormal videos in the calibration set. Theoretically, to learn a discriminative steering vector _gi_ that effectively projects anomalous features away from the normality manifold, the HMC requires a contrastive set of examples to model the direction of geometric divergence. As detailed in Table 14, empirical results demonstrate significant resilience. While a balanced distribution (5:5) yields the peak performance, the method maintains robust AUC scores above 86% across a wide effective range (from 4:6 to 6:4). Performance degradation becomes notable only at extreme imbalances. In such regimes, the scarcity of counter-examples prevents the HMC from establishing a stable normality baseline, leading to a less precise rectification policy. 

Crucially, this finding highlights a distinct advantage of our few-shot calibration paradigm in handling the inherent sparsity of real-world anomalies. Unlike fully supervised methods that require massive, balanced datasets for convergence a condition rarely met in practice due to the long-tail nature of abnormal events, our framework requires only a minute, representative subset for calibration. Consequently, even in scenarios where anomalies are extremely rare in the wild, the ability to construct a small, locally balanced calibration set from limited samples allows SteerVAD to mitigate the data sparsity problem effectively. This capability provides a robust foundation for learning in extremely sparse scenarios, where traditional approaches would fail to acquire sufficient signal. 

## E.3 ROBUSTNESS TO SUPERVISION GRANULARITY 

To evaluate the robustness of SteerVAD under coarser supervision signals, we conducted ablation study using video-level labels. In this setting, all segments belonging to an anomalous video are labeled as positive during the calibration phase. As shown in Table 15, the model achieves 87.01% AUC with coarse video-level labels, which is statistically comparable to the 87.15% AUC obtained with standard segment-level labels. This minimal performance drop (-0.14%) demonstrates that the RSA-based expert selection and HMC modulation are highly robust to label noise, effectively distilling the anomaly signature even from coarse annotations. 

**Table 15:** Impact of supervision granularity on UCF-Crime performance. SteerVAD maintains high performance even with coarse video-level labels. 

|**Supervision Type**|**Granularity**|**AUC (%)**|
|---|---|---|
|**Segment-Level (Default)**|**Fine**|**87.15**|
|Video-Level|Coarse|87.01|



## E.4 STRUCTURAL STABILITY OF EXPERT SELECTION 

While the main text confirmed the stability of final AUC performance across random seeds, we further scrutinized the structural consistency of the selected feature subspaces. Figure 7 presents the jaccard similarity matrix of the Top-20 ranked attention heads across 10 independent random seeds. The heatmap reveals a high degree of overlap between any pair of random seeds. This indicates that the set of identified functional circuits is not sensitive to specific data samples. The algorithm consistently converges to the same subspace of expert heads, reinforcing that RSA captures stable, intrinsic functional components of the frozen backbone rather than fitting to transient data noise. 

30 



<!-- Start of picture text -->
Jaccard Similarity Between SeedsSeed 1 1 0.38 0.54 0.6 0.54 0.54 0.43 0.67 0.6 0.6 1.0<br>Seed 2 0.38 1 0.54 0.48 0.54 0.43 0.38 0.48 0.54 0.43<br>Seed 3 0.54 0.54 1 0.6 0.67 0.54 0.43 0.67 0.67 0.54 0.8<br>Seed 4 0.6 0.48 0.6 1 0.67 0.67 0.54 0.54 0.67 0.67<br>Seed 5 0.54 0.54 0.67 0.67 1 0.6 0.48 0.67 0.9 0.6 0.6<br>Seed 6 0.54 0.43 0.54 0.67 0.6 1 0.54 0.54 0.6 0.54<br>Seed 7 0.43 0.38 0.43 0.54 0.48 0.54 1 0.48 0.48 0.54 0.4<br>Seed 8 0.67 0.48 0.67 0.54 0.67 0.54 0.48 1 0.74 0.54<br>Seed 9 0.6 0.54 0.67 0.67 0.9 0.6 0.48 0.74 1 0.6 0.2<br>Seed 10 0.6 0.43 0.54 0.67 0.6 0.54 0.54 0.54 0.6 1<br>0.0<br>Seeds<br>Seeds<br>Seed 1 Seed 2 Seed 3 Seed 4 Seed 5 Seed 6 Seed 7 Seed 8 Seed 9 Seed 10<br><!-- End of picture text -->

**Figure 7:** Jaccard similarity analysis of the Top-20 identified anomaly experts across 10 random seeds. The high similarity scores confirm that the expert selection is structurally robust to sampling variations. 

**Table 16:** Detailed per-class AUC (%) performance on UCF-Crime. The active steering mechanism demonstrates consistent robustness across diverse anomaly types, with particularly high performance in dynamic events. 

|**Class**|**AUC (%)**|**Class**|**AUC (%)**|
|---|---|---|---|
|Abuse|68.84|Robbery|84.02|
|Arrest|82.58|RoadAccident|89.64|
|Arson|59.02|Shooting|85.74|
|Assault|95.17|Shoplifting|81.57|
|Burglary|75.47|Stealing|80.18|
|Explosion|79.72|Vandalism|88.16|
|Fighting|83.03|**Average**|**87.15**|



## E.5 DETAILED ANALYSIS OF ANOMALY SEMANTICS AND FUNCTIONAL CIRCUITS 

To investigate the operational boundaries and internal stability of our framework, we synthesize a granular performance analysis across all 13 anomaly classes (Table 16) with a visualization of the structural consistency of attention head selection (Figure 8). 

**Geometric Divergence with Contextual Mimicry.** As detailed in Table 16, the framework exhibits performance heterogeneity that correlates with the semantic ambiguity of the event. Categories characterized by high-velocity motion and explicit visual conflict, such as Assault (95.17%), Road Accidents (89.64%), and Vandalism (88.16%), achieve peak AUC scores. These events generate representations with large displacement vectors relative to the normality manifold, which are easily amplified by our steering mechanism. 

In contrast, categories such as Abuse (68.84%) and Burglary (75.47%) present a distinct challenge rooted in contextual mimicry. Visually, a burglary closely resembles a normal entry, and abuse often mimics normal physical interaction. These anomalies possess a smaller geometric footprint and high semantic overlap with normal activities. The difficulty here lies not merely in geometric separation, but in resolving the intent behind visually ambiguous actions, which suggests that future improvements for these classes may require incorporating longer-range temporal reasoning into the steering logic. 

**Latent Structural Invariance.** To verify that the identified anomaly experts represent stable functional circuits rather than artifacts of random initialization, Figure 8 visualizes the selection frequency of top attention heads across 10 independent random seeds. The resulting heatmap reveals a striking structural invariance. A specific subset of heads is consistently selected with nearly identical across all runs, while the majority of the background heads remain inactive. This consistent convergence pattern demonstrates that RSA robustly identifies intrinsic functional anchors within the frozen MLLM that are uniquely sensitive to anomaly semantics, independent of the stochasticity in data sampling. 

31 



<!-- Start of picture text -->
Head Stable Analysis: Abuse 1.0 Head Stable Analysis: Arrest 1.0 Head Stable Analysis: Arson 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0<br>Head Stable Analysis: Assault 1.0 Head Stable Analysis: Burglary 1.0 Head Stable Analysis: Explosion 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0<br>Head Stable Analysis: Fighting 1.0 Head Stable Analysis: RoadAccidents 1.0 Head Stable Analysis: Robbery 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0<br>Head Stable Analysis: Shooting 1.0 Head Stable Analysis: Shoplifting 1.0 Head Stable Analysis: Stealing 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0<br>Head Stable Analysis: Vandalism 1.0 Head Stable Analysis: Normal 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0<br>Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency<br>Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency<br>Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency<br>Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency<br>Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency Layer Index 0123689101112131415161718192021222324252627475 Selection Frequency<br><!-- End of picture text -->

**Figure 8:** Visualization of top attention head selection frequency across 10 independent random seeds. The emergence of consistent high-frequency heads confirms that the model reliably identifies the same intrinsic functional circuits for anomaly detection, demonstrating robustness against random calibration splits. 

32 

## E.6 GENERALIZATION TO OPEN-SET AND UNSEEN ANOMALIES 

To empirically validate the topological generalization framework proposed in Appendix A.5, we conducted a stress test on the UBnormal dataset. SteerVAD achieves a compelling **72.21% AUC** on this benchmark without any re-calibration. This result provides strong empirical evidence for our semantic entropy hypothesis, even for strictly unseen anomaly types, the anomalous events manifest as high-entropy geometric perturbations in the latent space, which are effectively rectified by our entropy-driven steering policy. 

# F MECHANISM DESIGN VALIDATION 

This section provides a deeper investigation into the architectural design choices of our framework, specifically validating the rationale behind using a linear metric for RSA and the architectural constraints of the HMC. 

F.1 VALIDITY OF LINEAR RSA VS. NON-LINEAR METRICS 

To justify the use of a linear metric (RSA) for analyzing complex manifolds, we compared it against two computationally intensive metrics, Silhouette Coefficient and k-Nearest Neighbor (k-NN) Purity. We visualize the trade-off between computational cost and selection consistency in Figure 9. 

The top panel highlights the dramatic efficiency gap that RSA completes calibration in milliseconds (0.14s), whereas k-NN incurs computational costs that are orders of magnitude higher (6.87s). Despite the speed difference, the head stability heatmaps for RSA, Silhouette, and k-NN are visually indistinguishable. All three metrics highlight the exact same expert regions. This visual evidence, combined with the strictly identical expert overlap shown in Table 17, conclusively proves that linearity is not a limitation but a feature, RSA serves as a lossless, high-speed proxy for identifying the manifold structure. 

**Table 17:** Comparison of RSA with non-linear selection metrics regarding expert overlap and computational efficiency. RSA identifies the identical set of experts as expensive non-linear metrics but with significantly higher speed. 

|**Metric**|**Linearity**|**Expert Overlap**|**Calibration Time**|**Speedup**|
|---|---|---|---|---|
|**RSA (Ours)**|**Linear**|**100% (Reference)**|**0.14 s**|**49**_×_|
|Silhouette|Non-Linear|100% Identical|0.86 s|8_×_|
|k-NN Purity|Non-Linear|100% Identical|6.87 s|1_×_|



## F.2 ABLATION OF HMC INPUT DESIGN 

We explicitly designed the HMC to condition purely on the global context vector _c_ . To verify this design, we implemented a local-aware variant where the input to the LGM adapter is the concatenation of the global context and the local feature ([ _c, hi_ ]). 

As shown in Table 18, the local-aware variant results in a performance decrease of 0.80% AUC. This confirms that including the high-dimensional local feature causes the lightweight HMC to overfit to local noise. By restricting the input to the global context, we enforce a top-down modulation mechanism, ensuring that rectification signals are robust and context-driven. 

# G QUALITATIVE ANALYSIS 

**Visualization of Attention** The improved geometric organization of our SteerVAD corresponds to a refined semantic attention mechanism, as evidenced by the activation heatmaps in Figure 10. Initially, feature activations from the base model are scattered, indicating a failure to prioritize the semantically critical regions and an apparent distraction by biased objects within the video frames. Upon rectification by our framework, the focus of SteerVAD is unequivocally sharpened. The resulting feature activations are highly concentrated and precisely highlight the key anomalous entities, 

33 





<!-- Start of picture text -->
Head Selection Frequency (Metric: k-NN) 1.0 Head Selection Frequency (Metric: Silhouette) 1.0 Head Selection Frequency (Metric: RSA) 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0 0 1 2 3 4 5 6 7 8 9 10 11 12Head Index13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 0.0<br>0 0 0<br>1 1 1<br>2 2 2<br>3 3 3<br>4 4 4<br>5 5 5<br>6 6 6<br>7 7 7<br>8 8 8<br>910 910 910<br>11 11 11<br>12 12 12<br>13 13 13<br>14 14 14<br>Layer Index 15 Layer Index 15 Layer Index 15<br>1617 Selection Frequency 1617 Selection Frequency 1617 Selection Frequency<br>18 18 18<br>19 19 19<br>20 20 20<br>21 21 21<br>22 22 22<br>23 23 23<br>24 24 24<br>25 25 25<br>26 26 26<br>27 27 27<br><!-- End of picture text -->

**Figure 9:** Validation of RSA efficiency and consistency. (Top) Average computation time per method, showing RSA’s superior speed advantage. (Bottom) Heatmaps of head importance scores generated by RSA, Silhouette, and k-NN. The nearly identical visual patterns confirm that RSA captures the same underlying geometric structure as computationally expensive non-linear metrics. 

**Table 18:** Design ablation for HMC input signals on UCF-Crime. Conditioning solely on global context prevents overfitting to local feature noise and ensures robust top-down modulation. 

|**Design Variant**|**Input Signal**|**AUC (%)**|**Observation**|
|---|---|---|---|
|Local-Aware|[_c, hi_](Concat)|86.35|Overfits to local noise & Self-Bias|
|**SteerVAD (Ours)**|**Global Context**_c_**Only**|**87.15**|**Robust Top-Down Modulation**|



such as the central figures in the video. This demonstrates a core capability of our hierarchical metacontroller: it dynamically modulates the feature channels to amplify anomaly-relevant signals and attenuate noise, thereby correcting the semantic gaze of the model for improved task performance. 

**Case Studies on Anomaly Detection.** Figure 11 demonstrates the end to end effectiveness and robustness of SteerVAD across normal and abnormal examples from both benchmark datasets. For an abnormal event in UCF-Crime, the anomaly score curve accurately rises at the moment of action, a conclusion supported by the generated explanation. Crucially, the model maintains a low score in a busy but normal surveillance scene, demonstrating resilience to false positives. This robustness extends to the challenging XD-Violence dataset. The model precisely captures abnormal event while correctly discerning a high motion sports clip as normal, a key challenge that confounds many methods. This ability to distinguish chaotic yet benign activity from genuine threats, coupled with the semantic coherence of the generated explanations, confirms that our framework delivers not only quantitatively superior results but also practically reliable and interpretable detections. 

34 



<!-- Start of picture text -->
Original Rectified<br><!-- End of picture text -->

**Figure 10:** Semantic impact of feature rectification on an example from XD-Violence. 



<!-- Start of picture text -->
No. The video shows a view from above an atrium or indoor space, likely  Yes. The video shows significant damage to a gas station in broad daylight.<br>within a mall or a similar environment. People can be seen walking, sitting,  There appears to have been an explosion, as indicated by the bright light flash<br>and interacting in various parts of the space. There are no visible signs of  in one of the frames. Following the explosion, there's visible debris and<br>damage, disturbance, or unusual behavior that would indicate abnormal  destruction around the gas station, with metal sheets and other materials<br>activities. scattered across the pavement.<br>(a)  Qualitative results on UCF-Crime dataset.<br>No. The video shows two individuals inside a car at night. The scene  Yes. The video depicts a physical altercation between hockey players on the<br>alternates between close-ups of the individuals as they converse while  ice. Players from opposing teams grapple and push each other, leading to a<br>driving. They are evading police pursuit, as indicated by the flashing  group scuffle near the goal. Referees attempt to intervene and separate the<br>lights of several police vehicles in the background. players, but the physical conflict persists.<br>(b)  Qualitative results on XD-Violence dataset.<br><!-- End of picture text -->

**Figure 11:** Qualitative results of SteerVAD on UCF-Crime and XD-Violence datasets. Each panel shows a representative video snippet and the anomaly curve with one normal and one abnormal video. The shaded regions denote the ground truth. Further generated descriptions are also provided. 

## G.1 ADDITIONAL VISUALIZATIONS 

To further illustrate the effectiveness and interpretability of our framework, we provide additional qualitative results and attention visualizations. 

Figures 12 and 13 present a comprehensive selection of qualitative results on the UCF-Crime and XD-Violence datasets, respectively showcasing representative examples from all anomaly categories within each dataset. These examples reinforce the findings from our main case studies and demonstrate the consistent ability of SteerVAD to generate precise anomaly scores and coherent textual explanations across a wide variety of complex scenes. The inclusion of all categories highlights the model’s robustness in correctly identifying diverse anomalies while avoiding false positives in dynamic but normal environments, further validating the practical reliability of our method. 

Furthermore, Figures 14 and 15 offer more visualizations the semantic impact of our manifold rectification process. We visualize the model’s effective gaze both before and after intervention. The original focus heatmaps frequently reveal the MLLM’s inherent biases, showing attention scattered across irrelevant background elements or non-discriminative objects. In stark contrast, the rectified focus heatmaps demonstrate a clear and decisive shift. Our HMC successfully steers the model’s attention to the core anomalous action, providing compelling visual evidence that our method not only improves geometric separability but also enforces a more task-relevant semantic understanding. 

35 

# H LLM USAGE STATEMENT 

In the preparation of this manuscript, the large language model (LLM) is solely utilized for the purpose of language polishing and grammar correction to improve clarity and readability. We confirm that all core research ideas, the conceptual framework, experimental design, data analysis are entirely the original work of the authors. The authors take full and sole responsibility for the accuracy, integrity, and all claims made within this work. 

36 



<!-- Start of picture text -->
No. The video shows a busy road at night with multiple vehicles, including taxis and private cars,  Yes. In the video frames, a person approaches a dog on the street. The person engages with the dog,<br>moving in both directions. The traffic appears to follow the flow smoothly. The scene is consistent  appearing to restrain or control it. The sequence suggests the person is potentially trying to hit the dog<br>with typical nighttime traffic, and the vehicles are seen behaving normally on the road.  in the area, which could indicate an attempt to abuse or attack to the dog for some reason.<br>Yes. The video shows a quiet street with cars parked on both sides. Suddenly, there is a bright flash of  Yes. The video shows a group of people engaging in what appears to be a physical altercation or brawl<br>light followed by visible smoke and flames emerging from a dark-colored car. The light and smoke  in what looks like a retail store or a small shop. The situation intensifies as more individuals join in.<br>suggest the car has caught fire. The incident involves arson, which poses a significant fire risk to  The people are grabbing, pushing, and throwing each other, indicating a violent conflict. The setting<br>nearby properties and individuals.  suggests a chaotic scenario, possibly involving an assault or a fight.<br>Yes. The video captures a busy urban road with multiple cars. Suddenly, a white vehicle comes to an abrupt stop, and individuals appear to exit the vehicle hurriedly. Law enforcement officers in tactical gear respond swiftly to the scene. Armed personnel are seen surrounding the vehicle, indicating a violent context such as a robbery or an abduction scenario.  Yes. The individual is attempting to break into a home by prying open a door using their foot and hand. This action indicates unauthorized entry, which is part of burglary. The behavior demonstrates a clear intent to unlawfully gain access to the premises, posing a significant security risk.<br>Yes. An explosion occurs in the background, creating a burst of light and smoke. The event is sudden  Yes. The individuals in the video appear to be involved in a physical altercation. In the frames, we can<br>and unexpected, with visible effects spreading across the room. The risk level is high due to the  see pushing, falling, and what seems like attempts to restrain each other. The confrontation escalates to<br>potential danger and immediate disruption to the environment. multiple people on the ground, indicating a high level of physical conflict.<br>Yes. In the video, a motorcycle rider collides with a white van, causing the rider to fall to the ground.  Yes. In the video, a person is seen walking towards a silver car, then talking to another individual who<br>Shortly after, an explosion occurs near the fallen rider, likely due to a firework or malfunctioning  approaches on a scooter. The person in the scooter seems to be involved in a tense interaction, leading<br>equipment. The rider appears to be on fire and is lying on the road as the situation unfolds. Other  to the person in the white shirt opening the car door and handing over something. They then follow the<br>motorcyclists in the vicinity are seen observing from a distance, while vehicles pass by on the road. scooter away quickly, indicating a hurried departure, which suggests a possible robbery scenario.<br>Yes. In the video, a group of individuals are interacting around a counter. One person is sitting while  Yes. Two individuals appear to be engaged in a physical altercation, with one person in a blue shirt<br>others are standing. Money appears to be exchanged hands, suggesting a transaction. The interactions  reaching out towards the other person who is seated. The interaction suggests a potential conflict,<br>seem confrontational, with people pointing and gesturing towards the seated person, indicating a  indicating a fighting event. The physical nature of the interaction could escalate, potentially leading to<br>coercive situation typical of a shoplifting. more severe consequences if not resolved quickly."<br>unauthorized access. The person on the left is leaning into the vehicle, possibly searching for or taking Yes. Two individuals are seen interacting with the open back of a parked vehicle, which appears to be items, while the person on the right stands nearby, observing or assisting. This behavior suggests an intent to commit theft, fitting the category of burglary. behavior constitutes intentional destruction of property. The act of vandalism can escalate into more Yes.  The individual is seen hitting the glass door with a hammer, causing visible damage. This serious criminal activities and poses a risk to both property and potential bystanders.<br><!-- End of picture text -->

**Figure 12:** Further qualitative results on UCF-Crime dataset. Covered all 13 abnormal category and normal examples. 

37 



<!-- Start of picture text -->
interaction between the two people. The lighting and atmosphere contribute to an intimate and ordinary No. The video depicts a man and a woman conversing in a warmly lit room, possibly a restaurant, with chandeliers and soft background lighting. The setting is a cozy environment, and the focus is on the scene. No. The video captures a military setting where soldiers in uniform are walking and interacting with vehicles. The scene shifts to a bathroom where soldiers are brushing their teeth, washing their faces, and performing their morning routines.<br>coaches. There is pushing, shoving, and possibly attempts at restraint by referees and other individuals confronting each other. The encounter escalates into a full-scale fight involving multiple players and Yes. The video shows two players, one in a white jersey and one in a black jersey, physically trying to separate the conflicting parties. Initially, a confrontation occurs near the goal, escalating into a fight. Players from both teams grapple Yes. The video shows a physical altercation between ice hockey players from opposing teams. and push each other aggressively, with referees attempting to separate them.<br>what appears to be a remote-controlled vehicle equipped with a gun. The individual fires the weapon,   Yes. The video initially shows a person marking a tree before transitioning to them standing next to resulting in a noticeable explosion and smoke. This is repeated in subsequent frames, where more smoke and sparks are visible, indicating continued firing and potential damage. others wearing high-visibility vests appear to be engaged in physical conflicts. The presence of both Yes. There are visible confrontations between groups of people. Individuals in protective gear and civilian groups and uniformed personnel suggests a volatile situation with possible altercations.<br>Yes. The video shows police officers interacting with civilians in what appears to be a tense situation, movement across the street, and barricades being carried, which indicates organized control measures involving several officers and possibly individuals obstructing their path. There is evidence of potentially targeting a protest or conflict area. Yes. The video shows several instances of law enforcement officers, identifiable by their uniform and equipment, apprehending individuals. The most notable scenes include police officers escorting individuals through a building and outdoors, where the individuals are restrained and held.<br>aggressively and confronting each other, suggesting a potential altercation. As the scene transitions, the room appears abandoned and cluttered, showing signs of chaos. Later, two individuals in a domestic Yes. In the initial frames, a group of men is seen in a room, causing a disruption by standing setting are seen physically struggling, indicating a fight.  Yes. The video shows a vehicle traveling behind a dark-colored car on a snowy road. As the car moves forward, it appears to gradually lose control and begins to slide across the road. It eventually crosses the road's centerline and seems to collide with a trailer that appears in the opposite lane.<br>Yes. The video captures a vehicle moving on a road flanked by trees. As the vehicle approaches, an explosion occurs in front of it, causing a burst of flames and smoke. The impact is visible directly ahead of the vehicle. Yes. There is a visible explosion near a military vehicle in the field, producing a large cloud of smoke. The presence of soldiers and military vehicles suggests the use of firearms in a combat scenario. Explosions and active combat situations pose significant risks to individuals and surroundings. Immediate danger to people and property is likely.<br><!-- End of picture text -->

**Figure 13:** Further qualitative results on XD-Violence dataset. Covered all 6 abnormal category and normal examples. 

38 















**Figure 14:** Further visualizations of SteerVAD on XD-Violence datasets. Our steering method effectively rectify the biased or ambiguous focus on the original videos. 

39 













**Figure 15:** Further visualizations of SteerVAD on UCF-Crime datasets. Our steering method effectively rectify the biased or ambiguous focus on the original videos. 

40 

