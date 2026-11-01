# FLaRA: Predicting Future Latent Representations for Accident Anticipation 

Lorenzo Caselli<sup>1</sup> , Tomaso Trinci<sup>2</sup> , Tommaso Bianconcini<sup>2</sup> , Simone Magistri<sup>1</sup> , Leonardo Taccari<sup>2</sup> , Francesco Sambo<sup>2</sup> , Andrew D. Bagdanov<sup>1</sup> 

**_Abstract_ — Anticipating traffic accidents from dashcam videos is a critical challenge in intelligent transportation systems. Existing methods typically map visual context directly to a collision probability without explicitly modeling the future evolution of the driving scene. In this paper we propose FLaRA (Predicting Future Latent Representations for Accident Anticipation), a novel predictive architecture that shifts this paradigm by forecasting future latent representations for accident anticipation. Building upon the Video Joint-Embedding Predictive Architecture (V-JEPA2), our model conditions a predictor network on observed context frames to predict the forthcoming latent features of the scene. A classifier then operates on these predicted future representations rather than only on past observations. To ensure these forecasts remain grounded in realistic future dynamics, we introduce a joint training objective that simultaneously optimizes an auxiliary feature-level reconstruction loss and a cross-entropy classification loss. Extensive evaluations on the Nexar dataset, alongside cross-domain validations on the DAD, DADA-2000, and DoTA benchmarks, demonstrate that our approach achieves state-ofthe-art performance while maintaining realistic early warning capabilities. The code is publicly available at: https:// github.com/LoreCase073/FLaRA.** 

## I. INTRODUCTION 

Dashcam videos are becoming increasingly common, with millions of vehicles worldwide equipped with forwardfacing cameras continuously recording driving scenes. This widespread availability makes them a natural and scalable data source for developing proactive safety systems applied for tasks such as viewpoint estimation [1], [2], anomaly detection [3] and accident detection [4]. However, while modern driving safety systems excel at detecting collisions as they happen [4], [5], the ability to predict accidents before they actually occur, relying just on video data, remains an open challenge. It is crucial that the prediction be made in advance so that there is a large enough buffer to actually perform an evasive maneuver and avoid the dangerous event. Existing approaches for accident anticipation follow a common paradigm: processing observed frames and directly mapping their representations to a collision probability. Most methods operate frame-by-frame, using recurrent architectures with spatio-temporal attention to accumulate temporal evidence [6]–[8], while others apply graph neural networks 

> 1Department of Information Engineering, University of Florence, Italy. Email: name.surname@unifi.it 

> 2Verizon Connect, Florence, Italy. 

> Email: name.surname@verizonconnect.com 

©2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media. 

alongside LSTM architectures [9] or incorporate monocular depth cues for 3D scene modeling [10]. 

To advance beyond this paradigm, we propose Predicting Future Latent Representations for Accident Anticipation (FLaRA), a novel predictive architecture built on top of the Video Joint-Embedding Predictive Architecture (VJEPA [11]) that leverages video foundation models for accident anticipation. Rather than mapping observed context directly to a classification output, as seen in a recent work by Goldshmidt et al. [12], our approach conditions a predictor network on context frames to forecast the _future representations_ of the scene within a latent space. By shifting the classification task to operate on these predicted future features rather than past observations, the model effectively simulates the forthcoming progression of the scene. To ensure these forecasts accurately reflect actual future events, we introduce a joint-training paradigm. Inspired by the original self-supervised objective used in the pre-training stage of V-JEPA, our approach optimizes an auxiliary reconstruction loss between the predicted features and the ground-truth latent representations of actual _future_ frames, in conjunction with a standard classification loss, allowing the model to make informed inferences regarding the development of the driving scene. 

Figure 1 shows an overview of FLaRA. During training, FLaRA observes a context window (purple) and forecasts future latent representations (green) to produce the predictions. At inference, it uses only this context window to predict the future latent representations used for classification. 

We train our proposed solution on the Nexar dataset [13] and evaluate it under a clearly defined protocol using standard metrics (AP, AUC, mTTA). To demonstrate the robustness and generalization capabilities we further validate ours solution in out-of-domain scenarios across several established benchmarks, including DAD [14], DADA-2000 [15], and DoTA [16]. 

Our primary contributions are summarized as follows: 

- We introduce a novel accident anticipation framework that leverages V-JEPA2’s capacity to forecast future latent representations from historical context and utilizes these predicted features to classify impending collisions. 

- We introduce a multi-task optimization strategy that combines a standard classification loss with an auxiliary latent reconstruction loss. Inspired by V-JEPA’s self-supervised pre-training objective, this ensures our predicted features accurately reflect ground-truth future states. 



<!-- Start of picture text -->
Shared weights<br>stop grad<br>(a) Training phase<br>(b) Inference phase<br><!-- End of picture text -->

**Fig. 1:** Overview of FLaRA. **(a)** During training, the encoder _fθ_ processes context frames (purple segment), and the predictor forecasts future latent representations. These predictions are supervised by a reconstruction loss against target representations obtained by encoding context and future frames (green segment) jointly. Only the predicted future features are used for the classification objective. **(b)** At inference, only context frames are available, thus requiring the model to classify exclusively from predicted future representations. 

- We conduct extensive experiments across multiple datasets, demonstrating that our proposed method achieves highly competitive results. It matches or surpasses existing state-of-the-art baselines in predictive accuracy (AP, AUC) while maintaining an early warning capability. 

## II. RELATED WORK 

Early works in accident anticipation, such as the Dynamic Spatial Attention (DSA) model [6], utilized LSTM networks to evaluate candidate objects and generate frame-by-frame future accident probabilities. Subsequent developments expanded upon these foundations by incorporating novel modules to model object interactions. For instance, UString [8] leverages Recurrent Neural Network and Graph Convolutional Network architectures to capture complex spatiotemporal relationships. A key contribution of UString is its adaptive loss function, designed to facilitate early hazard prediction by dynamically adjusting loss weights based on temporal proximity to the collision. Following this, DSTA [7] employed Transformer-based architectures equipped with dynamic spatial-temporal attention mechanisms. This design enables the model to effectively isolate highly relevant spatial regions while simultaneously tracking their temporal evolution across video sequences. More recently, Pjetri et al. [17] proposed a self-supervised approach that exploits a non-decreasing danger assumption to design a loss function that does not require hand-crafted temporal annotations. 

Shifting toward video foundation models, recent work has explored V-JEPA2 [11], a self-supervised architecture that learns by predicting latent representations of masked spatio-temporal regions. Adapting this powerful encoder for accident anticipation, Goldshmidt et al. [12] proposed 

BADAS. This method fine-tunes V-JEPA2 alongside a dedicated prediction head, yet it continues to base its collision predictions strictly on the representations of the observed context window. 

An alternative line of work in action anticipation instead proposes to first forecast future visual representations and classify on those. Vondrick et al. [18] is one of the first methods to do this. They trained a network to predict future CNN representations from the current frame, then training a classifier on the predicted output. Gammulle et al. [19] further extended this approach through a recurrent GAN framework that jointly optimizes future visual and temporal feature synthesis alongside a classification objective. Assran et al. [11] also build on the pretrained V-JEPA2, applying it to action anticipation. Given currently observed frames, the frozen V-JEPA2 predictor produces future latent representations. These predicted features are concatenated with the encoded present features and passed to an attentive probe, which is the only component trained for the downstream task, using only classification loss. Although this predict-thenclassify paradigm has been explored in action anticipation, to our knowledge, it has not been applied to the accident anticipation domain. 

## III. METHODOLOGY 

In this section we formalize our approach. We first define the problem setup (Section III-A), then describe the architecture and training procedure, including the joint reconstruction-classification objective (Section III-B), and finally detail the inference scheme based on predicted future representations (Section III-C). 

## _A. Problem Setup_ 

Given a video sequence, the goal of accident anticipation is to predict whether a safety-critical event will occur using only the visual information that precedes it. 

Let _V_ = ( _v_ 1 _, v_ 2 _, . . . , vT_ ) denote a video sequence of _T_ frames. Let _t_ event represent the timestamp of a safety-critical event, and let _k_ event be the corresponding discrete frame index. The objective is to learn a model _M_ that predicts the label _y ∈{_ 0 _,_ 1 _}_ using an observed context window up to frame _kc_ , where _kc < k_ event. The prediction is given by: 



This anticipates the event with a lead time of _k_ event _− kc_ frames (or _t_ event _− tc_ seconds, where _tc_ is the timestamp of frame _kc_ ). 

To guide the predictor to produce meaningful future representation, we compute an auxiliary reconstruction loss: 



where _ℓ_ denotes the Smooth-L1 loss, sg( _·_ ) is the stopgradient operator, and _z_ ˆ _i_<sup>_f_and</sup><sup>_z_˜</sup> _i_<sup>_f_denotethe</sup><sup>_i_-thtokenof</sup> the predicted and target future representations, respectively. Simultaneously, we compute the classification loss using the cross-entropy objective: 



where _y_ ˆ = _pψ_ (ˆ _z_<sup>_f_</sup> ) is the prediction derived from passing the forecasted future representations through the attentive probe. The overall training objective is formulated as: 



## _B. Architecture and Training_ 

Our approach is based on the idea of leveraging predicted future representations for classification [18]. Instead of using the features of the observed frames directly, a predictor network conditioned only on the context window produces feature-level forecasts. These predicted features are then used as inputs for the classification task. To ensure the predictor learns meaningful future representations and adapts to the new distribution, we supervise it during training with an auxiliary reconstruction loss where the targets are the encoder’s representations of the ground-truth future frames. 

In particular, our model _M_ builds on V-JEPA2 and comprises three main components: a video encoder _fθ_ , a future predictor _gϕ_ and an attentive probe _pψ_ [11], [20]. The probe is composed of four transformer blocks. The first three are self-attention layers that refine the input token representations. The final block replaces self-attention with a crossattention layer where a single learnable query token attends to the refined features, producing a pooled representation. This is then passed to a linear layer for classification. 

During training (see Figure 1(a)) the model receives both context frames and future frames. Let _x_<sup>_c_</sup> = ( _v_ 1 _, . . . , vkc_ ) = denote the sequence of context frames, and let _x_<sup>_f_</sup> ( _vkc_ +1 _, . . . , vkc_ + _L_ ) denote the sequence of future frames, which spans a fixed length of _L_ frames immediately following the context window. The context frames are sampled such that the window ends at _t_ event _− tc >_ 0 seconds prior to the potential event, matching our anticipation time window. 

The encoder _fθ_ first processes both context and fu= ture frames jointly to establish targets, producing _z_ ˜ _fθ_ ([ _x_<sup>_c_</sup> _, x_<sup>_f_</sup> ]) = [˜ _z_<sup>_c_</sup> _,_ ˜ _z_<sup>_f_</sup> ] _∈_ R<sup>(</sup><sup>_Nc_+</sup><sup>_Nf_)</sup><sup>_×D_</sup> , where _Nc_ and _Nf_ are the number of context and future tokens respectively, _D_ is the encoder embedding dimension, _z_ ˜<sup>_c_</sup> is the feature representation of the context, and _z_ ˜<sup>_f_</sup> is the representation of the future. The encoder also processes the context frames _x_<sup>_c_</sup> without access to the future frames, producing _z_<sup>_c_</sup> = _fθ_ ( _x_<sup>_c_</sup> ) _∈_ R<sup>_Nc×D_</sup> . Then, the predictor _gϕ_ maps the context features _z_<sup>_c_</sup> to the predicted future representations _z_ ˆ<sup>_f_</sup> = _gϕ_ ( _z_<sup>_c_</sup> ) _∈_ R<sup>_Nf ×D_</sup> . 

where _λ ∈_ R is a hyperparameter modulating the influence of the reconstruction loss. 

The reconstruction loss in Eq. 2 adopts the V-JEPA2 self-supervised pretraining objective but differs in application. While V-JEPA2 reconstructs randomly masked spatiotemporal patches within a single clip, FLaRA uses as targets the latent representations of subsequent temporal frames. Furthermore, _L_ rec serves as an auxiliary signal alongside the classification loss, while the original V-JEPA2 pretraining has only the reconstruction objective. Mapping targets to future frame features aligns the model with the requirements of accident anticipation by anchoring outputs to future scene dynamics. 

FLaRA represents the first attempt to adapt the V-JEPA2 predictor for accident anticipation by utilizing a reconstruction signal and performing classification exclusively on predicted future latent representations. 

## _C. Inference_ 

At inference time (see Figure 1(b)), only the context frames _x_<sup>_c_</sup> are available. The encoder extracts the context features _z_<sup>_c_</sup> = _fθ_ ( _x_<sup>_c_</sup> ), which the predictor then uses to forecast the future representations _z_ ˆ<sup>_f_</sup> = _gϕ_ ( _z_<sup>_c_</sup> ). The final classification is performed by computing the prediction _y_ ˆ = _pψ_ (ˆ _z_<sup>_f_</sup> ). In this way the real future frames _x_<sup>_f_</sup> are _never_ accessed. 

## IV. EXPERIMENTAL RESULTS 

We begin by describing the datasets and implementation details, followed by the evaluation protocol and metrics. We then compare our method against prior work and conclude with ablation studies and analysis. 

## _A. Datasets_ 

We train our model on the training split of the Nexar dataset [13], which consists of 1,500 samples evenly balanced between positive and negative clips. 

Each positive clip is annotated with two timestamps: the earliest moment of recognizable danger _t_ alert and the precise 

**TABLE I:** Benchmark statistics. For DAD, DADA-2000, and DoTA, we use the specific subsets defined by [12]. 

|**Dataset**|**Positives**|**Negatives**|**Mean** _t_**event**<br>**Mean**|_t_**event**_−t_**alert**|
|---|---|---|---|---|
|Nexar (test)|672|672|10.9|1.9|
|DAD|13|301|3.0|1.2|
|DADA-2000|75|38|6.8|1.8|
|DoTA|327|40|5.1|2.0|



time of the safety critical event _t_ event, which can be either a collision or a near-collision. We use _t_ event to ensure that the sampled context window never includes frames of the actual event. We evaluate our approach on the Nexar test split, comprising 1,344 clips (672 positive and 672 negative). Each positive clip is temporally cropped prior to the safetycritical event using a variable offset ∆ _t ∈{_ 0 _._ 5 _,_ 1 _._ 0 _,_ 1 _._ 5 _}_ seconds. 

To evaluate cross-dataset generalization, we further evaluate our model on three additional benchmarks: DAD [14], DADA-2000 [15] and DoTA [16] datasets, using the subsets and re-annotations defined by [12]. These subsets retain only ego-vehicle collisions and include manually annotated _t_ alert timestamps, which are absent or inconsistent from the original datasets. As summarized in Table I, these subsets are heavily imbalanced. We also report mean _t_ event and mean _t_ event _− t_ alert computed for each dataset by averaging across positive samples. Furthermore, because the original DADA-2000 and DoTA datasets lack negative samples, following [12] we construct negatives by extracting the first 4.0 seconds from positive clips in which the alert timestamp ( _t_ alert) occurs at least 4.5 seconds after the start of the video, guaranteeing that the sampled segment captures only normal driving dynamics and no temporal overlap with the safetycritical event. 

## _B. Implementation Details_ 

We initialize our model using a V-JEPA2 ViT-L backbone pretrained on the Something-Something v2 dataset [21], which provides the initial weights for the encoder, predictor, and attentive probe. During fine-tuning, we freeze the encoder except for its final block. Only this final encoder block, the predictor, the attentive probe and the newly instantiated classifier are updated, resulting in 84M trainable parameters (see Table II). 

Both the context clip _x_<sup>_c_</sup> and future clip _x_<sup>_f_</sup> consist of 16 frames sampled at 8 fps with a spatial resolution of 256 _×_ 256. The future frames are sampled contiguously, starting immediately after the end of the context window (see Figure 1). For positive samples, the context clip is extracted such that its last frame precedes the collision by a random offset sampled uniformly from [0 _, γ_ ], where _γ_ = _t_ event _−t_ alert. This ensures the model is exposed to clips at varying temporal distances from the event, spanning from the moment the alert was triggered up to the collision boundary. 

We train the network for 30 epochs using a batch size of 4, using AdamW with a weight decay of 1 _×_ 10<sup>_−_2</sup> and using a peak learning rate of 2 _._ 5 _×_ 10<sup>_−_5</sup> . We also adopt a cosine annealing schedule warmup. We sweep the reconstruction 

loss weight over _λ ∈{_ 1 _,_ 2 _,_ 10 _,_ 100 _}_ and use _λ_ = 10 in all experiments. Reported results are averaged over three runs and all experiments are run on a single RTX 4090 GPU. 

## _C. Evaluation Protocol_ 

At inference, we use our model in a sliding window fashion over the input video. Each window consists of 16 frames sampled at 8 fps, mirroring the training configuration. The window has a stride of 8 video frames, resulting in overlapping windows. For positive videos, only windows whose end time falls before the annotated time of event _t_ event are considered valid, ensuring the model cannot observe the safety-critical event itself. For negative videos, the full video is evaluated. 

Following standard evaluation protocol metrics [7], [8], we report Area Under the ROC Curve (AUC), Average Precision (AP) and mean Time-to-Accident (mTTA). For AUC, the score is aggregated per video by selecting the maximum score among all the sliding windows. For AP and mTTA, we sweep detection thresholds _σ_ by considering a positive video detected if any valid window exceeds the threshold. For positive samples, the Time-to-Accident (TTA) at a given threshold _σ_ is defined as: 



where _t_ det( _σ_ ) denotes the _end_ timestamp of the earliest window exceeding the threshold. We use the final frame of the temporal window, as the model requires the complete 16-frame sequence to generate an inference. We then compute the mean Time-to-Accident (mTTA) by averaging TTA values over true positive videos detected at each threshold. The final metric is thus obtained by averaging these results across all recall operating points. 

We evaluate our method against UString [8], DSTA [7], both trained on the CCD dataset [8] and BADAS [12], trained on Nexar, using the publicly available checkpoints. All clip-based methods (Ours, BADAS) are evaluated within this same sliding window framework. UString and DSTA are per-frame recurrent models that produce a prediction at each timestep using all previously observed frames. For these models, we extract features at native video FPS and run the temporal model over the full sequence, yielding one prediction per frame. These per-frame scores are then fed into the same metric computation pipeline without any windowing or temporal resampling. 

## _D. Comparative Performance Evaluation_ 

Table II presents the performance of our method on the Nexar test set compared to prior approaches. Our model achieves the highest AP (86 _._ 6) and AUC (87 _._ 3), outperforming the previous best performing model BADAS while using roughly 4 _×_ fewer trainable parameters (84M vs 332M). In terms of mTTA values, we are comparable to BADAS, indicating that the improvement on AP and AUC did not come at the cost of later warnings. 

Table III reports cross-dataset generalization, where all methods are evaluated on DAD, DADA-2000 and DoTA. 

**TABLE II:** Accident anticipation performance on Nexar. 

|**Method**|**N. Params**|**AP** _↑_|**AUC** _↑_|**mTTA**|
|---|---|---|---|---|
|DSTA|4.8M|50.0|45.5|10.216|
|UString|1.9M|49.6|41.0|10.263|
|BADAS|332M|85.2|85.1|3.712|
|**FLaRA**|84M|**86.6**|**87.3**|3.744|



**TABLE III:** Generalization performance on unseen accident anticipation benchmarks. 

|**Dataset**|**Method**|**AP** _↑_|**AUC** _↑_|**mTTA**|
|---|---|---|---|---|
||DSTA|4.7|61.2|2.832|
||UString|5.5|65.6|2.694|
|**DAD**|BADAS|48.9|92.5|0.782|
||**FLaRA**|**67.3**|**93.6**|0.663|
||DSTA|66.4|52.5|6.552|
||UString|66.4|58.1|6.598|
|**DADA**|BADAS|**98.0**|**95.0**|1.601|
||**FLaRA**|97.3|93.2|1.595|
||DSTA|89.1|48.1|4.759|
||UString|88.5|50.8|4.732|
|**DoTA**|BADAS|98.7|**90.6**|1.562|
||**FLaRA**|**98.8**|90.0|1.649|



On DAD, FLaRA achieves a substantial improvement over BADAS (+18 _._ 4 AP), indicating a stronger generalization to unseen driving scenarios, while on DADA-2000 and DoTA our model still performs competitively. These results are achieved without any domain-specific fine-tuning, demonstrating that the predicted future representations learned on Nexar transfer to other accident anticipation benchmarks. 

Across datasets, mTTA values should be interpreted in light of the video characteristics we reported in Table I. Both FLaRA and BADAS require a minimum window of 2 seconds (16 frames at 8 fps), which constrains how early detection can occur on short clips. On datasets like DAD, where mean _t_ event is just 3 _._ 0 seconds, there is very little margin, explaining the low mTTA values across all clipbased methods. Conversely, on the Nexar dataset, _t_ event is 10.9 seconds, allowing the model to start observing context windows further away from the actual event and potentially achieve earlier detections. Moreover, the difference _t_ event _− t_ alert averages 1.9 seconds, meaning that our mTTA of 3.744 seconds indicates detection well before visual cues of safetycritical events are recognizable by a human observer. 

## _E. Ablation Study_ 

Table IV presents an ablation study isolating the contribution of each component. _Frozen Backbone_ keeps the encoder _fθ_ and predictor _gϕ_ frozen, training only the attentive probe _pψ_ on the classification objective. Since _gϕ_ receives no training signal, it relies entirely on V-JEPA2’s pretrained representations. The _Fine-tuned w/o Lrec_ matches our full training setup, but removes _L_ rec, meaning _gϕ_ is guided only by the classification objective. Comparing these variants demonstrates that fine-tuning _gϕ_ and last layer of _fθ_ results in a substantial improvement, especially on Nexar (+5 _._ 1 AP) and DADA-2000 (+5 _._ 4 AP). However, without the reconstruction objective, _gϕ_ is guided solely by the classification 

**TABLE IV:** Ablation study on the effect of fine-tuning, reconstruction loss and features used for classification. 

|||**Nexar**|||**Dad**||
|---|---|---|---|---|---|---|
|**Method**|**AP** _↑_|**AUC** _↑_|**mTTA**|**AP** _↑_|**AUC** _↑_|**mTTA**|
|Frozen Backbone|80.4|81.8|4.688|22.5|77.6|0.651|
|Fine-tuned w/o _L_rec|85.5|87.0|3.882|25.6|88.0|0.788|
|Encoder Only|86.2|87.3|3.867|35.8|80.7|0.713|
|FLaRA + _z_<sup>_c_</sup>|85.1|86.0|3.570|58.4|**93.7**|0.654|
|**FLaRA**|**86.6**|**87.3**|3.744|**67.3**|93.6|0.663|
|||**DADA-200**|**0**||**Dota**||
|**Method**|**AP** _↑_|**AUC** _↑_|**mTTA**|**AP** _↑_|**AUC** _↑_|**mTTA**|
|Frozen Backbone|90.2|81.9|1.774|97.9|85.0|1.568|
|Fine-tuned w/o _L_rec|95.6|90.1|2.056|98.7|89.4|1.783|
|Encoder Only|95.3|89.7|1.748|98.4|87.4|1.619|
|FLaRA + _z_<sup>_c_</sup>|97.0|92.8|1.551|**98.8**|**90.0**|1.511|
|**FLaRA**|**97.3**|**93.2**|1.595|**98.8**|**90.0**|1.649|



gradient, without any guarantee that its outputs correspond to meaningful future representations. 

Introducing _L_ rec addresses this by guiding the predicted features closer to the latent dynamics of the actual future scene. This drives consistent gains across all benchmarks, most notably on DAD, where AP grows from 25 _._ 6 to 67 _._ 3. Because the DAD subset exhibits a severe class imbalance (only 13 accidents in 314 clips) that closely mirrors the real-world rarity of collisions, this confirms that training on semantically coherent future predictions is crucial for generalizing to _realistic_ , rare-event domains. 

In the _Encoder Only_ ablation we remove _gϕ_ entirely and classify directly from the fine-tuned encoder features, effectively using only current frames information. This ablation mirrors BADAS’s training regime, fine-tuning the encoder directly, while retaining our reduced parameter count. While on Nexar this variant has closer performance to our full model, cross-dataset performance drops considerably, confirming that the predictor, guided by _L_ rec, learns future representations that generalize well to other unseen distributions, improving with respect to the _Encoder Only_ training regime. 

Finally, the _FLaRA + z_<sup>_c_</sup> variant retains our full training objective but classifies using the concatenation of context ( _z_<sup>_c_</sup> ) and predicted future ( _z_ ˆ<sup>_f_</sup> ) features. This degrades performance on Nexar (-1.5 AP) and DAD, suggesting that present features hinder classification by introducing redundant information that interferes with the model’s discriminative capabilities. While performance on DADA-2000 and DoTA remains comparable, classifying exclusively from predicted future features proves to be the most effective design for accident anticipation. 

## _F. Anticipation Horizon Analysis_ 

The Nexar test set contains positive videos cropped at different temporal distances from the safety-critical event (∆ _t ∈{_ 0 _._ 5 _,_ 1 _._ 0 _,_ 1 _._ 5 _}_ seconds). Figure 2 reports the AUC on positive videos filtered by offset, comparing our method to BADAS. As expected, performance degrades for both methods as temporal distance increases, since earlier crops contain fewer visual cues of the impending collision. However, FLaRA consistently outperforms BADAS across all offsets, with the performance gap widening at larger distances (+1.3 



<!-- Start of picture text -->
95<br>BADAS<br>FLaRA<br>90<br>85<br>80<br>75<br>70<br>0.5 s 1.0 s 1.5 s<br>Temporal offset Δ t  before collision (s)<br>AUC<br><!-- End of picture text -->

**Fig. 2:** AUC on the Nexar test set as a function of the temporal offset before the collision event. Videos are grouped by their annotated prediction horizons of ∆ _t ∈{_ 0 _._ 5 _,_ 1 _._ 0 _,_ 1 _._ 5 _}._ 

AUC at ∆ _t_ = 0 _._ 5 compared to +5.4 AUC at ∆ _t_ = 1 _._ 5). This demonstrates that predicting future representations is particularly _beneficial_ with larger horizons, where the observed context alone provides limited discriminative information and the model must rely on forecasting the scene’s evolution. 

## V. CONCLUSIONS AND LIMITATIONS 

In this work we showed that forecasting future latent representations and using them for classification, rather than relying on observed features, consistently improves accident anticipation performance. The joint reconstructionclassification objective we introduced anchors the role of the V-JEPA2 predictor to output actual future scene latent representations and allows effective adaptation of the backbone and predictor to the target domain. Our framework achieves competitive or superior results across four benchmarks with four times fewer trainable parameters than the previous stateof-the-art: the largest gains are achieved on DAD, whose class distribution most closely reflects real-world distributions, and on the improvement of detection capabilities at higher temporal offsets, as shown in Figure 2. However, our approach comes with some limitations. While the detection quality of the model improves, the anticipation time remains comparable to BADAS, our strongest competitor, and improving mTTA without sacrificing precision remains an open challenge. Also, the predictor operates with a fixed temporal horizon. As future work, introducing an adaptive prediction range or multiple predictions could better reflect and improve performance in real-world applications, by allowing the model to predict further in the future when far from the event. In addition, architectural scaling and hardwarespecific optimizations to reduce the inference cost represents an interesting direction for real-time edge deployment. 

## REFERENCES 

- [1] S. Magistri, F. Sambo, F. Schoen, D. C. d. Andrade, M. Simoncini, S. Caprasecca, L. Kubin, L. Bravi, and L. Taccari, “A lightweight deep learning model for vehicle viewpoint estimation from dashcam images,” in _2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)_ , 2020, pp. 1–6. 

- [2] S. Magistri, M. Boschi, F. Sambo, D. C. de Andrade, M. Simoncini, L. Kubin, L. Taccari, L. de Luigi, and S. Salti, “Lightweight and effective convolutional neural networks for vehicle viewpoint estimation from monocular images,” _IEEE Transactions on Intelligent Transportation Systems_ , vol. 24, no. 1, pp. 191–200, 2023. 

- [3] S. Haresh, S. Kumar, M. Z. Zia, and Q.-H. Tran, “Towards anomaly detection in dashcam videos,” in _2020 IEEE Intelligent Vehicles Symposium (IV)_ , 2020, pp. 1407–1414. 

- [4] Z. Zhou, X. Dong, Z. Li, K. Yu, C. Ding, and Y. Yang, “Spatiotemporal feature encoding for traffic accident detection in vanet environment,” _IEEE Transactions on Intelligent Transportation Systems_ , vol. 23, no. 10, pp. 19 772–19 781, 2022. 

- [5] L. Taccari, F. Sambo, L. Bravi, S. Salti, L. Sarti, M. Simoncini, and A. Lori, “Classification of crash and near-crash events from dashcam videos and telematics,” in _2018 21st International Conference on Intelligent Transportation Systems (ITSC)_ , 2018, pp. 2460–2465. 

- [6] F.-H. Chan, Y.-T. Chen, Y. Xiang, and M. Sun, “Anticipating accidents in dashcam videos,” in _Asian Conference on Computer Vision_ . Springer, 2016, pp. 136–153. 

- [7] M. M. Karim, Y. Li, R. Qin, and Z. Yin, “A dynamic spatial-temporal attention network for early anticipation of traffic accidents,” _CoRR_ , vol. abs/2106.10197, 2021. 

- [8] W. Bao, Q. Yu, and Y. Kong, “Uncertainty-based traffic accident anticipation with spatio-temporal relational learning,” in _Proceedings of the 28th ACM International Conference on Multimedia_ . ACM, Oct. 2020, p. 2682–2690. 

- [9] A. V. Malawade, S.-Y. Yu, B. Hsu, D. Muthirayan, P. P. Khargonekar, and M. A. A. Faruque, “Spatiotemporal scene-graph embedding for autonomous vehicle collision prediction,” _IEEE Internet of Things Journal_ , vol. 9, no. 12, pp. 9379–9388, 2022. 

- [10] H. Liao, Y. Li, Z. Li, Z. Bian, J. Lee, Z. Cui, G. Zhang, and C. Xu, “Real-time accident anticipation for autonomous driving through monocular depth-enhanced 3d modeling,” _Accident Analysis & Prevention_ , vol. 207, p. 107760, 2024. 

- [11] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus _et al._ , “V-jepa 2: Selfsupervised video models enable understanding, prediction and planning,” _arXiv preprint arXiv:2506.09985_ , 2025. 

- [12] R. Goldshmidt, H. Scott, L. Niccolini, S. Zhu, D. Moura, and O. Zvitia, “Badas: Context aware collision prediction using real-world dashcam data,” 2025. 

- [13] D. Moura, S. Zhu, and O. Zvitia, “Nexar dashcam collision prediction dataset and challenge,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops_ , June 2025, pp. 2608–2616. 

- [14] F.-H. Chan, Y.-T. Chen, Y. Xiang, and M. Sun, “Anticipating accidents in dashcam videos,” in _Computer Vision – ACCV 2016_ , S.-H. Lai, V. Lepetit, K. Nishino, and Y. Sato, Eds. Cham: Springer International Publishing, 2017, pp. 136–153. 

- [15] J. Fang, D. Yan, J. Qiao, J. Xue, H. Wang, and S. Li, “Dada2000: Can driving accident be predicted by driver attentionƒ analyzed by a benchmark,” in _2019 IEEE Intelligent Transportation Systems Conference (ITSC)_ , 2019, pp. 4303–4309. 

- [16] Y. Yao, X. Wang, M. Xu, Z. Pu, Y. Wang, E. Atkins, and D. J. Crandall, “Dota: Unsupervised detection of traffic anomaly in driving videos,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , vol. 45, no. 1, pp. 444–459, 2023. 

- [17] A. Pjetri, D. Abbondandolo, D. C. de Andrade, S. Caprasecca, F. Sambo, and A. D. Bagdanov, “Self-supervised road accident anticipation with non-decreasing danger,” in _Computer Vision – ECCV 2024 Workshops_ . Springer Nature Switzerland, 2025, pp. 65–79. 

- [18] C. Vondrick, H. Pirsiavash, and A. Torralba, “Anticipating visual representations from unlabeled video,” in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 

- [19] H. Gammulle, S. Denman, S. Sridharan, and C. Fookes, “Predicting the future: A jointly learnt model for action anticipation,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , October 2019. 

- [20] J. Lee, Y. Lee, J. Kim, A. Kosiorek, S. Choi, and Y. W. Teh, “Set transformer: A framework for attention-based permutation-invariant neural networks,” in _Proceedings of the 36th International Conference on Machine Learning_ , ser. Proceedings of Machine Learning Research, 2019, pp. 3744–3753. 

- [21] R. Goyal _et al._ , “The” something something” video database for learning and evaluating visual common sense,” in _Proceedings of the IEEE international conference on computer vision_ , 2017. 

