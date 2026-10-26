This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **Anomize: Better Open Vocabulary Video Anomaly Detection** 

Fei Li<sup>1</sup><sup>_,_2#</sup> Wenxuan Liu<sup>3</sup><sup>_,_4#</sup> Jingjing Chen<sup>2</sup> Ruixu Zhang<sup>1</sup> Yuran Wang<sup>1</sup> Xian Zhong<sup>4</sup> Zheng Wang<sup>1*</sup> 

1National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University 

- 2Shanghai Key Lab of Intelligent Information Processing, School of Computer Science, Fudan University 

3State Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University 

- 4Hubei Key Laboratory of Transportation Internet of Things, Wuhan University of Technology lifeiwhu@whu.edu.cn, liuwx66@pku.edu.cn 

## **Abstract** 

_Open Vocabulary Video Anomaly Detection (OVVAD) seeks to detect and classify both base and novel anomalies. However, existing methods face two specific challenges related to novel anomalies. The first challenge is detection ambiguity, where the model struggles to assign accurate anomaly scores to unfamiliar anomalies. The second challenge is categorization confusion, where novel anomalies are often misclassified as visually similar base instances. To address these challenges, we explore supplementary information from multiple sources to mitigate detection ambiguity by leveraging multiple levels of visual data alongside matching textual information. Furthermore, we propose incorporating label relations to guide the encoding of new labels, thereby improving alignment between novel videos and their corresponding labels, which helps reduce categorization confusion. The resulting Anomize framework effectively tackles these issues, achieving superior performance on_ UCF-CRIME _and_ XD-VIOLENCE _datasets, demonstrating its effectiveness in OVVAD._ 

## **1. Introduction** 

Video Anomaly Detection (VAD) identifies anomaly in videos and is widely used in public safety systems. Traditional VAD methods can be categorized based on the type of training data used. Semi-supervised VAD [3, 22, 49] is trained exclusively on normal samples, detecting anomalies as deviations from learned normal patterns. In contrast, weakly supervised VAD [13, 33, 50] is trained on both normal and anomalous samples but lacks precise temporal labels, treating VAD as a binary classification problem. Both methods focus on detecting specific anomaly within a closed set and exhibit limitations in open-world scenarios. 

> * Corresponding author. # Contributed equally to this work. 



<!-- Start of picture text -->
··· ··· visually similar<br>familiar unfamiliar familiar Novel Base<br>GT. Fighting Assault<br>misclassified<br>as base one<br>ambiguous  Pred.    Assault  Assault<br>score<br>(a) Detection Ambiguity (b) Categorization Confusion<br><!-- End of picture text -->

Figure 1. **Challenges Related to Novel Anomalies.** (a) Detection ambiguity: The model struggles to assign accurate anomaly scores to unfamiliar frames containing novel anomalies. (b) Categorization confusion: Novel anomalies are misclassified as visually similar base instances from the training set. 

Recent research has explored open-set VAD [1, 9], where anomalies seen in training are considered base cases, while others are treated as novel cases. It trains on normal and base anomalies to detect all anomalies, overcoming the limitations of closed-set detection. However, it struggles with understanding anomaly categories, leading to unclear outputs [42]. Consequently, a leading study [42] has further investigated open vocabulary (OV) VAD, which aims to detect and categorize all anomalies using the same training data as open-set VAD, offering more informative results. 

Novel anomalies in OVVAD introduce two challenges that remain unexplored by existing methods: (1) _Detection ambiguity_ , where the model often lacks sufficient information to accurately assign anomaly scores to unfamiliar data, as shown in Fig. 1(a). Current methods rely on training or fine-tuning the model, which is inherently limited and cannot adapt to the variability of samples in an open setting. (2) _Categorization confusion_ , where novel cases visually similar to base cases are misclassified, as shown in Fig. 1(b). OV tasks generally rely on multimodal alignment for categorization. Since the model tends to extract visual features for novel videos similar to base videos, these features are more likely to align with base label encodings, leading to miscategorization. Traditional OV methods use pre-trained encoders to encode text, where the input contains labels with 

29203 



<!-- Start of picture text -->
normal visual feature<br>anomaly textual feature<br>anomaly visual feature<br>guided<br>textual feature space<br>augment mis-aligned well-aligned<br>(before guided) (after guided)<br>augmented similar<br>visual feature space<br>(a) augmented visual features  (b) guided textual features<br>of novel anomalies in detection of novel labels in categorization<br><!-- End of picture text -->

Figure 2. **Feature Visualization of Our Design.** (a) Text augmentation shifts ambiguous frames to the anomalous feature space. In the static stream, text represents anomaly-related nouns ( _e.g.,_ “abandoned fire starter”), while in the dynamic stream, it denotes label descriptions. (b) Group-guided text encoding improves the alignment of novel anomalies with novel labels, especially for those resembling base samples. 

unified templates [16, 29, 44, 45] or embeddings [2, 4, 53]. These methods rely solely on pre-trained encoders without spatial guidance in label encoding, limiting multimodal alignment for novel cases. 

To address detection ambiguity, we introduce a **TextAugmented Dual Stream** mechanism with dynamic and static streams, each focusing on different visual features augmented by corresponding textual information. The dynamic stream captures sequential information through temporal visual encoding, augmented by label descriptions related to dynamic characteristics. The static stream captures scene information through original contrastive languageimage pre-training (CLIP)-encoded visual features, augmented by a concept library related to static characteristics. The complementarity between dynamic and static data is crucial: certain anomalies rely on temporal information, such as tailing, while others depend on contextual cues, such as running on a highway. Synergistic training of the streams ensures mutual supplementation and constraints, delivering comprehensive temporal and contextual information, minimizing overfitting to specific anomaly categories, and improving overall performance. Additionally, the augmentation follows common-sense reasoning. To detect anomalies in real-world scenarios, we first define the anomaly and establish correlations between visual data and anomaly texts, providing a reference for detection within the overall context. Similarly, we augment visual features with relevant anomaly text, providing additional information for detection. As shown in Fig. 2(a), novel visual features that cause ambiguous detections are shifted into the anomalous feature space with support from text, helping the model better assess unfamiliar anomalies. 

To address categorization confusion, we introduce a **Group-Guided Text Encoding** mechanism, encoding labels using group-based descriptions, with labels sharing similar visual characteristics grouped together. As shown in Fig. 2(b), this mechanism establishes connections be- 

tween base and novel data through grouping, positioning the encodings of novel labels close to those of base labels, where videos associated with both base and novel labels are visually similar, thereby enhancing multimodal alignment for novel data. For novel labels not grouped with base labels, the descriptions provide contextual support to pretrained encoders for text encoding, thus enhancing alignment. Compared to previous methods mainly relying on pre-trained models, our approach strengthens the guidance of the feature space for novel labels, achieving more effective alignment for categorization. 

With the **Anomize** framework, we achieve notable results across both XD-VIOLENCE [41] and UCFCRIME [33] datasets. For anomaly detection, we obtain a 2.78% overall improvement on XD-VIOLENCE and 8.21% on novel cases, with UCF-CRIME results comparable to a more complex state-of-the-art model. For categorization, we achieve a 25.61% overall increase in Top-1 accuracy on XD-VIOLENCE and 5.71% on UCF-CRIME, with improvements of 56.53% and 4.49% on novel cases, respectively. In summary, our contributions are threefold: 

- To address detection ambiguity, we discover the importance of providing sufficient informational support. We combine dynamic and static streams to effectively constrain and complement each other. Operating at different levels of visual features, each stream is augmented with corresponding textual information, offering comprehensive support for detection. 

- To tackle categorization confusion, we emphasize the importance of establishing connections between labels to guide their encodings. We propose a text encoding mechanism that groups labels based on visual characteristics and generates corresponding descriptions for encodings. 

- Our **Anomize** framework targets the challenges of novel anomalies that remain unexplored, offering new insights for OVVAD and demonstrating superior performance on two widely-used datasets, particularly for novel cases. 

## **2. Related Work** 

### **2.1. Video Anomaly Detection** 

**Semi-Supervised VAD.** Existing semi-supervised video anomaly detection (VAD) methods are typically categorized into three groups: one-class classification (OCC), reconstruction-based models, and prediction-based models, all of which are trained exclusively on normal data. OCC models [31, 32, 35, 40, 49] classify anomalies by identifying data points that fall outside a learned hypersphere of normal data. However, defining normality can be ambiguous, which often reduces their effectiveness [23, 46]. Reconstruction-based methods [3, 25, 30, 48, 52] use deep auto-encoders (DAEs) to learn normal patterns, detecting anomalies through high reconstruction errors. However, 

29204 

DAEs may still reconstruct anomalous frames with low error, weakening detection performance. Prediction models [5, 8, 18, 22, 24, 26], often utilizing GANs, forecast future frames and identify anomalies by comparing predicted frames with actual frames. 

**Weakly-Supervised VAD.** Weakly-supervised video anomaly detection (WSVAD) identifies anomalies using only video-level labels without precise temporal or spatial information. WSVAD methods typically frame the task as a multiple instance learning (MIL) problem [19, 33, 34, 39, 50], where videos are divided into segments, and predictions are aggregated into video-level anomaly scores. Sultani _et al_ . [33] first define the WSVAD paradigm using a deep multiple-instance ranking framework. Recent methods focus on optimizing models. Tian _et al_ . [34] introduce RTFM, which combines dilated convolutions and self-attention to detect subtle anomalies, while Zaheer _et al_ . [50] add a clustering-based normalcy suppression mechanism. Other approaches [13, 27] leverage pre-trained models to gain task-agnostic knowledge. Wu _et al_ . [43] propose VadCLIP, which uses CLIP [29] for dual-branch outputs of anomaly scores and labels. 

**Open-Set VAD.** Open-set VAD models are trained on normal behaviors and base anomalies to detect all anomalies, addressing the challenges of open-world environments. Acsintoae _et al_ . [1] first introduce open-set VAD, along with a benchmark dataset and evaluation framework. Zhu _et al_ . [54] combine evidential deep learning and normalizing flows within a multiple instance learning framework. Hirschorn _et al_ . [9] propose a lightweight normalizing flows framework that utilizes human pose graph structures. 

Our method provides both detection and categorization results in an open setting, focusing on addressing the challenges related to novel anomalies. 

### **2.2. Open Vocabulary Learning** 

Recent advancements in pre-trained vision-language models [11, 29] have spurred significant interest in open vocabulary tasks, including object detection [6, 15, 51], semantic segmentation [7, 20, 47], and action recognition [12, 14, 21, 36]. These studies leverage the pre-trained knowledge of multimodal models, demonstrating strong generalization. Wu _et al_ . [42] first introduce open vocabulary video anomaly detection (OVVAD) using the pre-trained model CLIP. However, most methods emphasize the visual encoder while neglecting the text encoder, limiting zero-shot capabilities. Our method explores the text encoder and incorporates a guided encoding mechanism to enhance multimodal alignment in OVVAD. 

## **3. Proposed Anomize Method** 

### **3.1. Overview** 

Following Wu _et al_ . [42], we define the training sample set as _D_ = _{_ ( _vi, yi_ ) _}_<sup>_N_</sup> _i_ =1<sup>+</sup><sup>_A_,whichconsistsof</sup><sup>_N_normalsam-</sup> ples _Dn_ and _A_ abnormal samples _Da_ . Here, _vi_ represents video samples, and _yi ∈_ Cbase denotes the corresponding anomaly labels. Each _vi ∈ Da_ contains at least one anomalous frame, while _vi ∈ Dn_ consists entirely of normal frames. The complete label set C includes both base and novel anomaly labels. The objective of OVVAD is to train a model on _D_ to predict frame-level anomaly scores and video-level anomaly labels from C. 

Fig. 3 illustrates the overview of framework. We leverage the encoder of the pre-trained CLIP model for its strong generalization capabilities. Video frames are processed by the CLIP image encoder Φvisual to extract original visual features _xf ∈_ R<sup>_n×d_</sup> , where _n_ is the number of frames and _d_ is the feature dimension. These features are temporally modeled by a lightweight temporal encoder. The original features, augmented by a concept library ConceptLib, pass through the static stream, while temporal features, augmented by label descriptions, pass through the dynamic stream. The prediction from each stream is obtained and aggregated to generate the final frame-level anomaly score _s ∈_ R<sup>_n×_1</sup> . For categorization, a multimodal alignment method is used. A fused visual feature is first generated, and the CLIP text encoder Φtext extracts textual features via the group-guided text encoding mechanism. Frame-level predictions are then obtained through alignment and aggregated for the final video-level result _p_ video. 

### **3.2. Lightweight Temporal Encoder** 

We utilize the frozen Φvisual for visual features to leverage its zero-shot capabilities. However, since CLIP is pretrained on image-text pairs, it lacks temporal modeling for video. Recent methods [14, 37, 38] commonly introduce a temporal encoder. However, this often leads to performance degradation on novel cases, as the additional parameters in the encoder may become specialized for the training set, leading to overfitting. Therefore, we adopt a lightweight long short-term memory (LSTM) [10] for temporal modeling, resulting in the temporal visual feature _x_ tem _∈_ R<sup>_n×d_</sup> : 



Other parameter-efficient models may also be suitable, as discussed in the supplementary material. 

### **3.3. Group-Guided Text Encoding** 

Previous methods mainly rely on the generalization capabilities of pre-trained models without task-specific guidance, often leading to categorization confusion. We introduce a group-guided text encoding mechanism to address this. 

29205 



<!-- Start of picture text -->
(a) Group-Guided Text Encoding (b) Concept Library Creation<br>Group  labels  fighting Give some  broken window<br>and describe anomaly in {label}… stealingassault ... keywords about {label}... blood on the groundthick black smoke ...<br>…<br>Text label feats.  Text ConceptLib<br>Programmer LLM Encoder Programmer LLM Encoder<br>(c) Overall Framework … label feats.  Detection Branch<br>ℒ��� (Eq.20)<br>dynamic stream<br>Temporal Encoder Lightweight … Augmenter Detector (Eq.24)ℒ�−���<br>temporal<br>visual feats.<br>Image … Augmenter Detector  (Eq.24)ℒ�−��� anomaly score<br>frames Encoder original visual feats.  static stream<br>ConceptLib<br>Stage 1:       Categorization Branch  (Lightweight Temporal Encoder) fuse alignment average …<br>Stage 2:       Detection Branch  video label pred.<br>(Augmenter, Detector) label feats.  … Categorization Branch<br><!-- End of picture text -->

Figure 3. **Overview of Our Anomize Framework.** (a) Process for obtaining label features via the Group-Guided Text Encoding mechanism. (b) Creation of the concept library ConceptLib for anomaly detection. (c) The framework processes anomaly labels and video frames to generate frame-level anomaly scores and detected labels. Scoring is performed using a Text-Augmented Dual Stream mechanism, where each stream receives corresponding text and visual features, and the fused scores are produced as output. For labeling, the model aligns label features from the Group-Guided Text Encoding mechanism with the fused original and temporal visual encodings. Both the text and image encoders, pre-trained on CLIP, remain frozen without further optimization. 

We leverage large language models (LLM), specifically GPT-4 [28], for textual encoding. We first use the prompt promptgroup to group labels, ensuring that corresponding videos in each group exhibit high visual similarity. Then, we apply the prompt promptdesc to generate text descriptions for each label based on the grouping. These descriptions capture shared elements while emphasizing unique characteristics within each group, ensuring the encodings remain similar yet distinguishable: 





where resultgroup and resultdesc represent the label groups and their descriptions. The descriptions are then passed into the frozen CLIP text encoder Φtext to obtain encodings _t_ desc _∈_ R<sup>_c×d_</sup> , where _c_ is the number of anomaly labels: _t_ desc = Φtext (resultdesc) _._ (4) These encodings are used for multimodal alignment. For the visual features, we combine the temporal and original encodings, preserving the knowledge captured by the pretrained model: 



where _α_ is a scalar weight. The prediction probabilities for video frames are expressed as: 



where _p_ frame _∈_ R<sup>_n×c_</sup> represents the probability distribution over _c_ anomaly labels for each frame. To obtain the videolevel prediction, we select the top _M_ probabilities for each label and average these top values, where _M_ is the total number of frames divided by 16: 



where _p_ avg _∈_ R<sup>_c_</sup> is the average probabilities for each label after applying the softmax function _σ_ ( _·_ ). Finally, the videolevel prediction _p_ video is determined as the label with the highest average probability: 



### **3.4. Augmenter** 

In our method, both streams utilize text to enhance visual encodings via a unified augmenter. The augmenter takes visual encodings _e_ visual and textual encodings _e_ text as input. These are processed by a multi-head attention layer MHA( _·_ ), where _e_ visual acts as the query and _e_ text acts as the key and value. This operation extracts the most relevant textual features for supplementation, denoted as _e_ refine: 



A fully connected layer FC( _·_ ) linearly projects the visual encoding, which is concatenated with the refined textual features and passed through a multi-layer perceptron 

29206 

MLP( _·_ ) for dimensionality reduction. This results in the augmented output _e_ aug: 



### **3.5. Text-Augmented Dual Stream** 

In open settings, models may struggle to assess unfamiliar anomalies due to limited information. We propose a TextAugmented Dual Stream mechanism with complementary dynamic and static streams, each augmented by relevant text to provide sufficient support for detection. 

Since video anomaly detection (VAD) relies on temporal cues, we employ a dynamic stream to predict anomaly scores _s_ dyn _∈_ R<sup>_n×_1</sup> based on refined visual features _f_ aug _∈_ R<sup>_n×d_</sup> , derived from temporal visual features and augmented by label descriptions via the augmenter in Sec. 3.4: 





where Sigmoid( _·_ ) converts predictions to [0, 1]. 

Since the dynamic stream is limited in scene context, we employ a static stream with original visual features augmented by anomaly-relevant concept data. 

Specifically, we create a concept library ConceptLib containing key features related to anomalies. These features are generated by Φtext from various nouns describing significant characteristics of the anomalies: 



where promptconc is a prompt for relevant nouns. We then compute the cosine similarity between the visual feature of frame _i_ , _x_<sup>(</sup><sup>_i_)</sup> _f_<sup>, and concept features</sup><sup>_h ∈_ConceptLib:</sup> 



The top _K_ relevant concept features _hf_<sup>(</sup><sup>_i_)</sup> _∈_ R<sup>_K×d_</sup> and their scores _s_<sup>(</sup><sup>_i_)</sup> _∈_ R<sup>_K_</sup> are selected: _f_ 



The selected features are then weighted by their scores and concatenated to form refined textual features for the video: 



Next, the refined features _h_<sup>new</sup> _f ∈_ R<sup>_n×k×d_</sup> and the original visual features _xf_ are passed through the augmenter to generate the augmented encoding _x_ aug _∈_ R<sup>_n×d_</sup> : 



Similar to the dynamic stream, _x_ aug is fed into a detector for anomaly score prediction _s_ sta in the static stream: 



Finally, the outputs of the two streams are aggregated for the overall anomaly score prediction _s_ : 



where _β_ is a tunable parameter balancing the contributions of the dynamic and static streams. 

### **3.6. Objective Functions** 

**First Training Stage.** In the first stage, we focus on video anomaly categorization to train the LSTM while freezing other modules to prevent optimization conflicts. We use cross-entropy loss _L_ ce for categorization. To prevent overfitting to normal data due to class imbalance, we add a separation loss _L_ sep to enhance the distinction between normal and anomalous predictions. The loss for the first stage is: 







where _p_ avg _,i_ denotes the predicted probabilities for the _i_ -th video, and the normal label is at the first index. _gi_ is the one-hot ground truth, and _N_ denotes the batch size. 

**Second Training Stage.** In the second stage, we focus on anomaly detection to train the static and dynamic streams while freezing other modules. Following Wu _et al_ . [43], we apply the MIL loss. Specifically, we first average the top _M_ frame anomaly scores to obtain the video-level prediction _q_ ˆ _i_ , then compute the loss _L_ X _−_ MIL for each stream using binary cross-entropy to quantify the difference between predictions and binary labels _qi_ , where _qi_ = 1 denotes an anomaly and _X ∈{D, S}_ denotes the type of stream. Additionally, we apply a weight _wi_ in this phase to tackle data imbalance by increasing the penalty for incorrect scores related to anomalous videos. The loss is defined as follows: 





29207 

|Method|AP|AP_b_|AP_n_|AUC|AUC_b_|AUC_n_|Method|ACC|ACC_b_|ACC_n_|ACC|ACC_b_|ACC_n_|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Zhu et al._∗_[54]|64.40|-|-|78.82|-|-|Wu et al. [42]|64.68|89.31|30.90|41.43|49.02|37.08|
|Sultani et al. [33]|52.26|51.25|54.64|78.25|86.31|80.12|Ours|**90.29**|**92.37**|**87.43**|**47.14**|**56.86**|**41.57**|
|Wu et al. [41]|55.43|52.94|64.10|82.24|90.62|84.13||||||||
|RTFM [34]|58.99|55.72|65.97|84.47|92.54|85.87|Table 2. **Top-1**|**Accura**|**cy (%)**|**Compar**|**isons o**|**n XD-V**|**IOLEN**|
|Wu et al. [42]|66.53|57.10|76.03|**86.40**|**93.80**|**88.20**|**(left) and UC**|**F-CRIM**|**E (right**|**).**||||
|Ours|**69.31**|**57.37**|**84.24**|84.49|93.00|87.05||||||||



Table 2. **Top-1 Accuracy (%) Comparisons on XD-VIOLENCE (left) and UCF-CRIME (right).** 

Table 1. **Detection Metrics (%) Comparisons for XDVIOLENCE (left) and UCF-CRIME (right).** The best results are highlighted in bold, our method is shaded in gray, the symbol _∗_ indicates different category divisions, and underlined values represent the second-best results. 

## **4. Experimental Results** 

### **4.1. Datasets and Implementation Details** 

**Datasets.** We evaluate the performance of Anomize on two widely-used benchmark datasets. XD-VIOLENCE [41] is the largest dataset focused on violent events in videos, containing 3,954 training videos and 800 testing videos. The videos are collected from movies and YouTube, capturing six types of anomalous events across diverse scenarios. UCF-CRIME [33] is a large-scale dataset with 1,610 untrimmed surveillance videos for training and 290 for testing, totaling 128 hours. This dataset includes 13 types of anomalous events, spanning both indoor and outdoor settings and providing broad coverage of real-world scenarios. 

**Evaluation Metrics.** For anomaly detection, we employ standard metrics from previous works [33, 41]. Specifically, for UCF-CRIME, we compute AUC, which captures the trade-off between true positive and false positive rates. For XD-VIOLENCE, we report AP, reflecting the balance between precision and recall. For anomaly categorization, we report Top-1 accuracy on anomalous test videos from both datasets. These metrics are provided for all categories combined, as well as separately for base and novel categories, denoted by the subscripts b and n, respectively. 

**Implementation Details.** We implement our model in PyTorch and train it on an RTX 4090 with a 256-frame limit. Using the AdamW optimizer [17] with a learning rate of 2 _×_ 10<sup>_−_5</sup> and a batch size of 32, we train for 16 and 64 epochs in two phases. We use the pre-trained CLIP (ViT-B/16) model. The MLP module contains 2 fully connected layers with GeLU activation. The fusion weight _α_ is 1 during training and 2 during testing. _K_ is 25 for XDVIOLENCE and 5 for UCF-CRIME. Score weight _β_ is 1 on XD-VIOLENCE (0 for base categories) and 0.5 on UCFCRIME (0 for novel categories). Loss weight _wi_ follows the normal-to-anomaly ratio per iteration. 

### **4.2. Comparison with State-of-the-Art Methods** 

In Tab. 1, we compare the performance of our method with prior VAD methods, ensuring that all methods use the same 

visual features from CLIP and adopt an open-set setting. On XD-VIOLENCE, we achieve the best performance, with an increase of 2.78% overall and 8.21% on novel cases, demonstrating the effectiveness of our method in reducing detection ambiguity. On UCF-CRIME, our method achieves competitive results, likely due to the lightweight temporal encoder and segmented training, which limits optimization for the detection branch. Since other studies focus solely on traditional VAD without categorization, we compare our method’s categorization performance with a leading study [42] in Tab. 2. Our method shows significant improvements, with a 25.61% gain on XD-VIOLENCE and 5.71% on UCF-CRIME, as well as further improvements of 56.53% and 4.49% on novel cases, highlighting its effectiveness in reducing categorization confusion. 

### **4.3. Ablation Studies** 

**Effectiveness of Lightweight Temporal Encoder.** The experiments confirm the importance of temporal information in video-level tasks. Tab. 3 shows that the dynamic stream, with temporal encoding, provides useful temporal cues and complements the static stream effectively. While relying solely on temporal information performs poorly due to noise, the dynamic stream becomes more effective with loss weighting and text augmentation. Tab. 4 shows that adding the temporal encoder improves performance on base cases, but without further guidance, the lightweight encoder still introduces confusion for novel anomalies. 

**Effectiveness of Group-Guided Text Encoding Mechanism.** Comparisons in Tab. 4 between the second and third rows, or the fourth and fifth rows, on both datasets show that textual encodings based on group descriptions outperform the baseline, demonstrating the importance of our text encoding mechanism. 

**Effectiveness of Text Augmentation.** As shown in Tab. 3, text augmentation in both dynamic and static streams generally reduces detection ambiguity by compensating for the limitations of visual features. The dynamic stream with only text augmentation shows a slight drop on UCFCRIME, suggesting noise in the temporal encodings. However, when combined with the loss weight, it demonstrates the importance of text, as shown in Rows 6 and 7. 

**Effectiveness of Integrating Dynamic and Static Streams.** Tab. 3 shows that integrating the two streams is generally more effective than using them independently, as 

29208 

||_M_sta|||_M_dyn||ST||XD-VIOLENC<br>|E<br>||UCF-CRIME<br>||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|visual|text|_wi_|visual|text|_wi_||AP (%)|AP_b_ (%)|AP_n_ (%)|AUC (%)|AUC_b_ (%)|AUC_n_ (%)|
|_~~√~~_|_×_|_×_|_×_|_×_|_×_|_~~√~~_|54.75|46.30|76.61|48.47|47.72|50.05|
|_√_|_√_|_×_|_×_|_×_|_×_|_√_|59.65|56.41|75.29|52.21|59.18|48.45|
|_√_|_√_|_√_|_×_|_×_|_×_|_√_|59.26|57.20|74.28|84.26|92.06|86.86|
|_×_|_×_|_×_|_√_|_×_|_×_|_√_|47.19|36.87|69.79|26.93|17.39|25.73|
|_×_|_×_|_×_|_√_|_√_|_×_|_√_|54.98|57.79|69.40|23.06|14.68|19.74|
|_×_|_×_|_×_|_√_|_√_|_√_|_√_|54.16|56.03|68.19|83.21|91.94|85.43|
|_×_|_×_|_×_|_√_|_×_|_√_|_√_|48.55|51.15|59.93|80.97|90.71|83.63|
|_√_|_√_|_×_|_√_|_√_|_×_|_√_|64.92|52.93|79.91|52.37|59.44|48.61|
|_√_|_√_|_√_|_√_|_√_|_√_|_×_|58.22|**58.58**|72.31|**84.52**|92.62|86.52|
|_√_|_√_|_√_|_√_|_√_|_√_|_√_|**69.31**|57.37|**84.24**|84.49|**93.00**|**87.05**|



Table 3. **Effectiveness of Dynamic** _M_ dyn **and Static** _M_ sta **Streams with Text-Augmented Visual Data, Additional Loss Weight** _wi_ **, and Segmented Training (ST) on XD-VIOLENCE (left) and UCF-CRIME (right).** In _M_ sta, the visual data is the original visual feature output by CLIP, while in _M_ dyn, it is derived from a lightweight temporal encoder. 

|_E_tem|_M_group|_F_fus|_L_sep|ST||XD-VIOLENCE|||UCF-CRIME||
|---|---|---|---|---|---|---|---|---|---|---|
||||||ACC (%)|ACC_b_ (%)|ACC_n_ (%)|ACC (%)|ACC_b_ (%)|ACC_n_ (%)|
|_×_|_×_|_×_|_×_|_~~√~~_|45.92|74.81|6.28|35.71|56.86|23.60|
|_√_|_×_|_×_|_×_|_√_|53.42|92.37|0|25.71|70.59|0|
|_√_|_√_|_×_|_×_|_√_|56.07|92.75|5.76|27.86|68.63|4.49|
|_√_|_×_|_√_|_×_|_√_|56.51|**94.66**|4.19|27.14|**74.51**|0|
|_√_|_√_|_√_|_×_|_√_|90.07|91.98|87.43|46.43|56.86|40.45|
|_√_|_√_|_√_|_√_|_×_|89.95|91.98|86.91|46.43|52.94|**42.70**|
|_√_|_√_|_√_|_√_|_√_|**90.29**|92.37|**87.43**|**47.14**|56.86|41.57|



Table 4. **Effectiveness of the Lightweight Temporal Encoder** _E_ tem **, Group-Guided Text Encoding Mechanism** _M_ group **, Fusion Function** _F_ fus **, Additional Separation Loss** _L_ sep **, and Segmented Training (ST) on XD-VIOLENCE (left) and UCF-CRIME (right).** 

they complement and constrain each other, which is evident from the comparisons in the third, sixth, and last rows. 

**Effectiveness of Additional Loss Design.** Tab. 3 emphasizes the importance of loss weight _wi_ , especially when training dynamic and static streams together. On XD-VIOLENCE, adding _wi_ slightly degrades single-stream training but significantly benefits integrated streams. Besides, Tab. 4 shows that adding separation loss _L_ sep improves performance, effectively addressing class imbalance. 

**Effectiveness of Segmented Training.** Results in Tab. 3 and Tab. 4 show that segmented training generally outperforms single-phase training, especially for novel cases, confirming that single-phase training may cause optimization conflicts and increase overfitting risks. On UCF-CRIME, single-phase training shows slightly better categorization performance on novel categories, likely due to randomness caused by optimization conflicts, as reflected in the poor performance on base categories. 

### **4.4. Qualitative results** 

Fig. 4 presents qualitative results for anomaly detection, featuring two base and two novel categories from each dataset to cover all label groups. The comparison between the blue predicted score lines and the pink rectangles for ground truth demonstrates the effectiveness of our model in detecting anomalies. Notably, our method demonstrates 

|Test|XD-V|IOLENCE|UCF-|CRIME|
|---|---|---|---|---|
|Train|AP (%)|ACC (%)|AUC (%)|ACC (%)|
|XD-VIOLENCE|**69.31**|**90.29**|81.69|45.71|
|UCF-CRIME|66.16|85.87|**84.68**|**47.14**|



Table 5. **Cross-Dataset Detection and Categorization Results.** 

its capability to handle novel cases with minimal detection ambiguity, highlighting the strong support of the textaugmented dual stream. 

Fig. 5 shows similarity matrices of textual encodings, where labels indicated by the same dot are grouped together. Fig. 5(a) and (c) show results from encoding text using only label text, where visually similar anomalies lack corresponding textual similarity, revealing the limitations of relying solely on pre-trained models. Fig. 5(b) and (d) present results after being guided, where textual similarities within groups are enhanced to better align with visual similarities. This demonstrates the benefit of establishing label connections to guide encoding. 

### **4.5. Adaptability and Generalization** 

**Analysis of Cross-Dataset Ability.** Tab. 5 shows that training on one dataset in an open-set manner and testing on another achieves results comparable to direct training on the target dataset, highlighting the adaptability of our method. 

29209 



<!-- Start of picture text -->
base / fighting base / car accident novel / riot novel / explosion<br>base / assault base / burglary novel / arrest novel / arson<br><!-- End of picture text -->

Figure 4. **Qualitative Results for Anomaly Detection.** The first and second rows present results on XD-VIOLENCE and UCF-CRIME respectively. Red boxes and rectangles highlight the ground-truth anomalous frames, while blue lines represent predicted anomaly scores. 



<!-- Start of picture text -->
-- a group<br>(a) XD-Violence origin (b) XD-Violence refine (c) UCF-Crime origin (d) UCF-Crime refine<br><!-- End of picture text -->

Figure 5. **Similarity Matrices of Textual Encoding.** (a) and (c) depict results using encodings from the original label data, while (b) and (d) show improvements achieved with the group-guided text encoding mechanism. 

|Setting|ACC (%)|ACC_b_ (%)|ACC_n_ (%)|
|---|---|---|---|
|orig.|90.29|**92.37**|87.43|
|+ shoplifting|**90.51**|92.37|**88.21**|
|+ arrest|89.96|92.37|86.73|
|+ arson|88.77|92.37|84.08|
|+ assault|87.28|87.40|87.11|
|+ labels|85.21|86.64|83.25|
|orig.|47.14|**56.86**|41.57|
|+ riot|**65.69**|56.86|**68.09**|
|+ labels|45.00|54.90|39.33|



Table 6. **Impact of Data Addition on Top-1 Accuracy for XDVIOLENCE (top) and UCF-CRIME (bottom).** “+ shoplifting” denotes the addition of both videos and labels, while “+ labels” indicates adding new labels only. 

**Analysis of Open Vocabulary Ability.** The open vocabulary ability is demonstrated by stable categorization performance after adding novel anomaly data, which we further validate by evaluating top-1 accuracy with the added data, as shown in Tab. 6. The additional data for one dataset is sourced from another. On XD-VIOLENCE, incorporating both same-group ( _e.g._ , arson) and differentgroup data ( _e.g._ , shoplifting) leads to stable performance, though assault shows the clearest decline due to confusion 

with similar labels like fighting. On UCF-CRIME, incorporating riot leads to notable improvement, with most instances accurately categorized despite being grouped with four other labels. Although the addition of new labels introduces some confusion, the performance remains robust across both datasets. 

## **5. Conclusion** 

We propose the Anomize framework to address detection ambiguity and categorization confusion in open vocabulary video anomaly detection (OVVAD). By augmenting visual encodings with anomaly-related text through a dualstream mechanism, Anomize improves the detection of unfamiliar samples and resolves ambiguity. Additionally, a group-guided text encoding mechanism enhances multimodal alignment and reduces categorization confusion. Experiments on XD-VIOLENCE and UCF-CRIME demonstrate the effectiveness of our method. 

**Acknowledgments.** This work was supported by the National Natural Science Foundation of China (Grants 62171325, 62271361), the Hubei Provincial Key Research and Development Program (Grant 2024BAB039), and the supercomputing system at Wuhan University. 

29210 

## **References** 

- [1] Andra Acsintoae, Andrei Florescu, Mariana-Iuliana Georgescu, Tudor Mare, Paul Sumedrea, Radu Tudor Ionescu, Fahad Shahbaz Khan, and Mubarak Shah. Ubnormal: New benchmark for supervised open-set video anomaly detection. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 20111–20121, 2022. 1, 3 

- [2] Dibyadip Chatterjee, Fadime Sener, Shugao Ma, and Angela Yao. Opening the vocabulary of egocentric actions. In _Adv. Neural Inf. Process. Syst._ , 2023. 2 

- [3] Yang Cong, Junsong Yuan, and Ji Liu. Sparse reconstruction cost for abnormal event detection. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 3449–3456, 2011. 1, 2 

- [4] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 14064–14073, 2022. 2 

- [5] Xinyang Feng, Dongjin Song, Yuncong Chen, Zhengzhang Chen, Jingchao Ni, and Haifeng Chen. Convolutional transformer based dual discriminator generative adversarial networks for video anomaly detection. In _Proc. ACM Int. Conf. Multimedia_ , pages 5546–5554, 2021. 3 

- [6] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. In _Proc. Int. Conf. Learn. Represent._ , 2022. 3 

- [7] Kunyang Han, Yong Liu, Jun Hao Liew, Henghui Ding, Jiajun Liu, Yitong Wang, Yansong Tang, Yujiu Yang, Jiashi Feng, Yao Zhao, and Yunchao Wei. Global knowledge calibration for fast open-vocabulary segmentation. In _Proc. IEEE/CVF Int. Conf. Comput. Vis._ , pages 797–807, 2023. 3 

- [8] Yi Hao, Jie Li, Nannan Wang, Xiaoyu Wang, and Xinbo Gao. Spatiotemporal consistency-enhanced network for video anomaly detection. _Pattern Recognit._ , 121:108232, 2022. 3 

- [9] Or Hirschorn and Shai Avidan. Normalizing flows for human pose anomaly detection. In _Proc. IEEE/CVF Int. Conf. Comput. Vis._ , pages 13499–13508, 2023. 1, 3 

- [10] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural Comput._ , 9(8):1735–1780, 1997. 3 

- [11] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _Proc. Int. Conf. Mach. Learn._ , pages 4904–4916, 2021. 3 

- [12] Chengyou Jia, Minnan Luo, Xiaojun Chang, Zhuohang Dang, Mingfei Han, Mengmeng Wang, Guang Dai, Sizhe Dang, and Jingdong Wang. Generating action-conditioned prompts for open-vocabulary video action recognition. In _Proc. ACM Int. Conf. Multimedia_ , pages 4640–4649, 2024. 

   - 3 

- [13] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. CLIP-TSA: clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In _Proc. IEEE Int. Conf. Image Process._ , pages 3230–3234, 2023. 1, 3 

- [14] Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, and Weidi Xie. Prompting visual-language models for efficient video understanding. In _Proc. Eur. Conf. Comput. Vis._ , pages 105– 124, 2022. 3 

- [15] Dahun Kim, Anelia Angelova, and Weicheng Kuo. Regionaware pretraining for open-vocabulary object detection with vision transformers. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 11144–11154, 2023. 3 

- [16] Jooyeon Kim, Eulrang Cho, Sehyung Kim, and Hyunwoo J. Kim. Retrieval-augmented open-vocabulary object detection. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 17427–17436, 2024. 2 

- [17] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _Proc. Int. Conf. Learn. Represent._ , 2015. 6 

- [18] Sangmin Lee, Hak Gu Kim, and Yong Man Ro. BMAN: bidirectional multi-scale aggregation networks for abnormal event detection. _IEEE Trans. Image Process._ , 29:2395–2408, 2020. 3 

- [19] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. In _Proc. AAAI Conf. Artif. Intell._ , pages 1395–1403, 2022. 3 

- [20] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted CLIP. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 7061–7070, 2023. 3 

- [21] Kun-Yu Lin, Henghui Ding, Jiaming Zhou, Yi-Xing Peng, Zhilin Zhao, Chen Change Loy, and Wei-Shi Zheng. Rethinking clip-based video learners in cross-domain openvocabulary action recognition. _arXiv:2403.01560_ , 2024. 3 

- [22] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection - A new baseline. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 6536–6545, 2018. 1, 3 

- [23] Wenxuan Liu, Shilei Zhao, Xiyu Han, Aoyu Yi, Kui Jiang, Zheng Wang, and Xian Zhong. Pixel-refocused navigated trimargin for semi-supervised action detection. In _Proc. ACM Int. Conf. Multimedia Workshop_ , pages 23–31, 2024. 2 

- [24] Zhian Liu, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In _Proc. IEEE/CVF Int. Conf. Comput. Vis._ , pages 13568–13577, 2021. 3 

- [25] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 FPS in MATLAB. In _Proc. IEEE/CVF Int. Conf. Comput. Vis._ , pages 2720–2727, 2013. 2 

- [26] Yiwei Lu, K. Mahesh Kumar, Seyed Shahabeddin Nabavi, and Yang Wang. Future frame prediction using convolutional VRNN for anomaly detection. In _Proc. IEEE Int. Conf. Adv. Video Signal Based Surveill._ , pages 1–8, 2019. 3 

- [27] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 8022–8031, 2023. 3 

29211 

- [28] OpenAI. GPT-4 technical report. _arXiv:2303.08774_ , 2023. 

   - 4 

- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _Proc. Int. Conf. Mach. Learn._ , pages 8748–8763, 2021. 2, 3 

- [30] Nicolae-Catalin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, and Mubarak Shah. Self-distilled masked auto-encoders are efficient video anomaly detectors. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 15984–15995, 2024. 

   - 2 

- [31] Mohammad Sabokrou, Mohammad Khalooei, Mahmood Fathy, and Ehsan Adeli. Adversarially learned one-class classifier for novelty detection. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 3379–3388, 2018. 2 

- [32] Bernhard Sch¨olkopf, Robert C. Williamson, Alexander J. Smola, John Shawe-Taylor, and John C. Platt. Support vector method for novelty detection. In _Adv. Neural Inf. Process. Syst._ , pages 582–588, 1999. 2 

- [33] Waqas Sultani, Chen Chen, and Mubarak Shah. Realworld anomaly detection in surveillance videos. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 6479–6488, 2018. 1, 2, 3, 6 

- [34] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W. Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In _Proc. IEEE/CVF Int. Conf. Comput. Vis._ , pages 4955–4966, 2021. 3, 6 

- [35] Jue Wang and Anoop Cherian. GODS: generalized one-class discriminative subspaces for anomaly detection. In _Proc. IEEE/CVF Int. Conf. Comput. Vis._ , pages 8200–8210, 2019. 2 

- [36] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip: A new paradigm for video action recognition. _arXiv:2109.08472_ , 2021. 3 

- [37] Syed Talal Wasim, Muzammal Naseer, Salman H. Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Videogrounding-dino: Towards open-vocabulary spatiotemporal video grounding. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 18909–18918, 2024. 3 

- [38] Zejia Weng, Xitong Yang, Ang Li, Zuxuan Wu, and YuGang Jiang. Open-vclip: Transforming CLIP to an openvocabulary video model via interpolated weight optimization. In _Proc. Int. Conf. Mach. Learn._ , pages 36978–36989, 2023. 3 

- [39] Peng Wu and Jing Liu. Learning causal temporal relation and feature discrimination for anomaly detection. _IEEE Trans. Image Process._ , 30:3513–3527, 2021. 3 

- [40] Peng Wu, Jing Liu, and Fang Shen. A deep one-class neural network for anomalous event detection in complex scenes. _IEEE Trans. Neural Networks Learn. Syst._ , 31(7): 2609–2622, 2020. 2 

- [41] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also 

listen: Learning multimodal violence detection under weak supervision. In _Proc. Eur. Conf. Comput. Vis._ , pages 322– 339, 2020. 2, 6 

- [42] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 18297–18307, 2024. 1, 3, 6 

- [43] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In _Proc. AAAI Conf. Artif. Intell._ , pages 6074–6082, 2024. 3, 5 

- [44] Tao Wu, Shuqiu Ge, Jie Qin, Gangshan Wu, and Limin Wang. Open-vocabulary spatio-temporal action detection. _arXiv:2405.10832_ , 2024. 2 

- [45] Zuxuan Wu, Zejia Weng, Wujian Peng, Xitong Yang, Ang Li, Larry S. Davis, and Yu-Gang Jiang. Building an openvocabulary video CLIP model with better architectures, optimization and data. _IEEE Trans. Pattern Anal. Mach. Intell._ , 46(7):4747–4762, 2024. 2 

- [46] Haiyang Xie, Zhengwei Yang, Huilin Zhu, and Zheng Wang. Striking a balance: Unsupervised cross-domain crowd counting via knowledge diffusion. In _Proceedings of the 31st ACM international conference on multimedia_ , pages 6520–6529, 2023. 2 

- [47] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. SAN: side adapter network for open-vocabulary semantic segmentation. _IEEE Trans. Pattern Anal. Mach. Intell._ , 45(12):15546–15561, 2023. 3 

- [48] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on keyframes for video anomaly detection. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 14592–14601, 2023. 2 

- [49] Muhammad Zaigham Zaheer, Jin-Ha Lee, Marcella Astrid, and Seung-Ik Lee. Old is gold: Redefining the adversarially learned one-class classifier training paradigm. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 14171–14181, 2020. 1, 2 

- [50] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. CLAWS: clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection. In _Proc. Eur. Conf. Comput. Vis._ , pages 358–376, 2020. 1, 3 

- [51] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and ShihFu Chang. Open-vocabulary object detection using captions. In _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , pages 14393–14402, 2021. 3 

- [52] Yuanhong Zhong, Xia Chen, Jinyang Jiang, and Fan Ren. A cascade reconstruction model with generalization ability evaluation for anomaly detection in videos. _Pattern Recognit._ , 122:108336, 2022. 2 

- [53] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. _Int. J. Comput. Vis._ , 130(9):2337–2348, 2022. 2 

- [54] Yuansheng Zhu, Wentao Bao, and Qi Yu. Towards open set video anomaly detection. In _Proc. Eur. Conf. Comput. Vis._ , pages 395–412, 2022. 3, 6 

29212 

