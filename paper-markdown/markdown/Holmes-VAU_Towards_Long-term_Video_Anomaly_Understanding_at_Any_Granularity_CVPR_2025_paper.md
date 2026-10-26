This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity** 

- Huaxin Zhang<sup>1</sup> Xiaohao Xu<sup>2</sup> Xiang Wang<sup>1</sup> Jialong Zuo<sup>1</sup> Xiaonan Huang<sup>2</sup> Changxin Gao<sup>1</sup> Shanjun Zhang<sup>3</sup> Li Yu<sup>1</sup> Nong Sang<sup>1*</sup> 

1Key Laboratory of Image Processing and Intelligent Control, 

School of Artificial Intelligence and Automation, Huazhong University of Science and Technology 

   - 2University of Michigan, Ann Arbor 3Kanagawa University 

- _{_ zhanghuaxin,wxiang,cgao,hustlyu,nsang _}_ @hust.edu.cn, _{_ xiaohaox,xiaonanh _}_ @umich.edu, _{_ chiyoz01 _}_ @kanagawa-u.ac.jp 



<!-- Start of picture text -->
Q<br>5s 10s 15s 35s 79s 84s 89s 117s<br>Q :Describe this video Q :Provide a caption … Q … Q Q … Q Anomaly-focused<br>A :The public gathers  A :A group of guards  Temporal Sampler<br>together, creating a  stands in front of a  A A A A<br>scene of disorder damaged glass door<br>5s 35s 79s 117s<br>Q :Summarize the anomaly-related event in the video … Q :Provide a summary of the video<br>A :Individuals on the street are marching and  A :The video begins with a<br>vandalizing the surrounding buildings, while a  burning car, smoke and flames  Multi-modal LLM<br>group of uniformed guards is maintaining order. filling the air, creating a<br>chaotic atmosphere.<br>0s 130s A A A<br>Q :Summarize the anomaly events in the video<br>A :The anomaly events depict chaotic and violent scenes, including individuals holding firearms,  Caption Judgement Judgement<br>people throwing objects at a car, a person being restrained by police officers, a burning  Description Description<br>object, a car engulfed in flames, and a person smashing a glass door. Analysis Analysis<br>Temporal Anomaly Granularity: Clip-level Event-level Video-level<br><!-- End of picture text -->

Figure 1. **Motivation** . **Left** : Existing datasets lack the hierarchical structure to capture transient and sustained anomalies across varying temporal scales. Our **_HIVAU-70k_** dataset addresses this by providing multi-granularity annotations—clip, event, and video levels—that enable detailed anomaly analysis in complex real-world scenarios. **Right** : Inspired by Sherlock Holmes’s knack for zeroing in on critical details, our **_Holmes-VAU_** method integrates an Anomaly-focused Temporal Sampler with a multi-modal LLM, directing model attention to anomaly-rich segments, which enables models to decode complex, long-term video anomalies efficiently. 

## **Abstract** 

_How can we enable models to comprehend video anomalies occurring over varying temporal scales and contexts? Traditional Video Anomaly Understanding (VAU) methods focus on frame-level anomaly prediction, often missing the interpretability of complex and diverse real-world anomalies. Recent multimodal approaches leverage visual and textual data but lack hierarchical annotations that capture both short-term and long-term anomalies. To address this challenge, we introduce HIVAU-70k, a large-scale benchmark for hierarchical video anomaly understanding across any granularity. We develop a semi-automated annotation engine that efficiently scales high-quality annotations by combining manual video segmentation with recursive free-text_ 

_annotation using large language models (LLMs). This results in over 70,000 multi-granular annotations organized at clip-level, event-level, and video-level segments. For efficient anomaly detection in long videos, we propose the Anomaly-focused Temporal Sampler (ATS). ATS integrates an anomaly scorer with a density-aware sampler to adaptively select frames based on anomaly scores, ensuring that the multimodal LLM concentrates on anomaly-rich regions, which significantly enhances both efficiency and accuracy. Extensive experiments demonstrate that our hierarchical instruction data markedly improves anomaly comprehension. The integrated ATS and visual-language model outperform traditional methods in processing long videos. Our benchmark and model are publicly available at https: //github.com/pipixin321/HolmesVAU ._ 

*Corresponding author 

13843 

## **1. Introduction** 

Video Anomaly Understanding (VAU) is crucial for applications such as video surveillance [46], violent content analysis [56], and autonomous driving [63]. Detecting deviations from normal patterns aids in hazard prevention and real-time decision-making. Traditional methods [14, 49, 74] mainly focus on frame-level predefined closed-set anomaly prediction, assigning an anomaly score to each frame. However, these approaches often fail to describe and understand complex anomalies in the real world. 

To address this gap, open-world anomaly understanding [57] embraces the diversity and unpredictability of real-world anomalies. Recent work integrates multimodal approaches, combining visual data with textual descriptions [42, 58, 62, 65], while advances in multimodal visuallanguage models (VLMs) [6, 21, 27, 29, 68] have enabled more nuanced understanding through anomaly-related instruction tuning and text generation [9, 36, 47, 67]. 

Despite these advancements, **a significant gap remains in models’ ability to comprehend anomalies across multiple temporal scales.** For instance, while anomalies such as explosions or fights may be captured in a single frame, more complex events like theft or arson require understanding long-term contextual patterns. Existing VAU datasets [46, 56] typically provide annotations at a single level of granularity, limiting models to understanding either immediate perceptual anomalies or those requiring extended contextual reasoning. The lack of datasets with hierarchical annotations—encompassing both shortterm and long-term anomalies—hinders models’ capacity to reason about anomalies with diverse temporal characteristics. Moreover, constructing datasets that encapsulate this hierarchical complexity poses significant challenges in scalability and annotation quality. 

To address these issues, we develop a semi-automated annotation engine that efficiently scales high-quality annotation by combining manual video segmentation with recursive free-text annotation using large language models (LLMs). The process involves three key stages: **1** ) _hierarchical video decoupling_ , where we manually identify anomaly events and segment them into shorter clips; **2** ) _hierarchical free-text annotation_ , where captions for each clip are generated through human effort or video captioning models, then summarized at the event and video levels via LLMs; and **3** ) _hierarchical instruction construction_ , where the textual data is transformed into question-answer instruction prompts by combining captions and summaries with designed prompts, creating a dataset with rich annotations for training and evaluating models. 

Utilizing the annotation engine, we introduce **_HIVAU70k_** , a large-scale video anomaly understanding benchmark with hierarchical instructions. Our dataset comprises over 70,000 multi-granular instruction data organized 

across clip-level, event-level, and video-level segments as shown in Fig. 1. This hierarchical structure empowers models to detect immediate anomalies, _e.g._ , sudden explosions or fighting, as well as complex events that require an understanding of long-term context, like theft or arson. By annotating at multiple temporal levels, HIVAU-70k provides diverse anomalies in open-world scenarios. 

Towards long-term VAU, efficiency remains a critical challenge. Previous methods [9, 47, 67] often rely on _uniform frame sampling_ , which can either miss crucial anomaly frames or incur large computational costs [25, 47, 67]. To address this, we propose the **_Holmes-VAU_** method, which combines the proposed Anomaly-focused Temporal Sampler (ATS) with the multimodal visual-language model for efficient long-term video anomaly understanding (See Fig. 1). The ATS combines a _anomaly scorer_ with a _density-aware sampler_ that adaptively selects frames by their anomaly scores. This integration ensures that the visual-language model concentrates on anomaly-rich regions, enhancing both efficiency and accuracy. 

Our contributions are threefold: **1)** We introduce **_HIVAU-70k_** , a large-scale, multi-granular benchmark for hierarchical video anomaly understanding. **2)** We propose the **_Holmes-VAU_** method, which combines the proposed Anomaly-focused Temporal Sampler (ATS) to boost the efficiency and accuracy of inference on long-term videos. **3)** We conduct extensive experiments demonstrating the effectiveness of hierarchical instruction data in enhancing anomaly comprehension and validate the performance gains provided by the integrated ATS and visual-language model in processing long videos. 

## **2. Related Works** 

**Video Anomaly Detection.** This task aims to temporally detect abnormal frames in a long untrimmed video [1, 14, 23, 34, 40, 52]. Early attempts are based on handcrafted features [1, 19, 23, 34, 40, 72]. Recently, deep learning approaches [14, 37, 61] have become dominant, broadly classified into unsupervised, weakly-supervised, and fully-supervised methods. Unsupervised methods train only on normal videos to learn normal patterns and are often designed as reconstruction-based [13, 14, 59, 61], prediction-based [31], or a combination [33]. Some methods [48, 50, 66] also explore a fully unsupervised setting, including both normal and abnormal videos in training set without real labels. Weakly-supervised methods [11, 22, 37, 46, 49, 55, 56, 70, 73, 74] use both normal and abnormal videos with video-level annotations. Fully-supervised methods [20, 30] are less studied due to the high cost of precise frame-level annotations. 

**Multi-modal Video Anomaly Understanding.** Largescale visual-language pre-trained models such as CLIP [44] 

13844 

serve as a bridge between visual and textual modalities. Some recent works [17, 42, 58, 62] in the realm of video anomaly detection have leveraged textual information as prompts to enhance the model’s anomaly representation. Based on this, [57] firstly proposed the open vocabulary VAD task, [65] introduced a multimodal video anomaly dataset composed of dense clip captions. Furthermore, [67] extracted captions from video frames and designed prompts for LLMs to provide anomaly scores, [9] and [47] construct diverse and interactive instruction data at the video level. However, these datasets consider only a single temporal level of anomaly understanding data construction, _i.e_ . clip-level [65] or video-level [9, 47]. Unlike these methods, we focus on building large-scale _hierarchical_ video anomaly understanding data for multimodal instruction tuning. 

**Hierarchical Video Understanding.** Video understanding is a challenging task due to its temporal-scale diversity. To better comprehend videos, many previous works have focused on both datasets and models in hierarchical video understanding. For example, [3, 24, 32, 35, 45, 54] proposed fine-grained action recognition and localization datasets, [16] provided free-form hierarchical captions for hour-long videos at multiple temporal scales, and [8, 18, 38, 43, 60, 69] trained models on hierarchical levels to obtain better video feature representation. Recently, to assess the capability of video vision-language models (VLMs) in handling challenges in real-world scenarios, several video benchmarks [10, 12] have been built, incorporating data at multiple temporal scales for evaluation. Unlike these works, we dive into the field of Video Anomaly Understanding, thus filling the gap in multi-scale annotations in this area. 

## **3. HIVAU-70k Benchmark** 

We first define the video anomaly understanding task in Sec.3.1. Then, the construction process of the HIVAU-70k benchmark will be elaborated in Sec.3.2. Finally, we will present HIVAU-70k’s statistical information in Sec. 3.3. 

### **3.1. Task Description** 

This work focuses on video anomaly understanding, involving temporal anomaly detection and anomaly explainability. Temporal anomaly detection aims to predict an anomaly score for each frame in a video _V_ , represented as _S ∈_ R<sup>_T_</sup> , where _T_ is the total number of frames. Building on this, we explore the model’s ability to generate explanatory text outputs related to anomalies based on video input and user queries. Specifically, given a video _V_ and a text query _Q_ , the model generates an anomaly-related response _A_ . We consider two key abilities: **1** ) **visual perception** , which involves recognizing main entities in the video, and **2** ) **anomaly reasoning** , which encompasses the 



<!-- Start of picture text -->
Temporal<br>Granularity …<br>video<br>event ……<br>clip<br>CAP CAP CAP CAP<br>Clip Caption Clip Caption Clip Caption Clip Caption<br>Prompt+Label LLM P+L LLM<br>Event Summary Event Summary<br>Prompt+Label LLM<br>Video Summary<br>Clip Caption Event Summary Video Summary<br>Manual Checking<br>{Q:Prompt_C, {Q:Prompt_J, A:Judgement, {Q:Prompt_J, A:Judgement,<br> A:Caption}  Q:Prompt_D, A:Description,  Q:Prompt_D, A:Description,<br> Q:Prompt_A, A:Analysis}  Q:Prompt_A, A:Analysis}<br><!-- End of picture text -->

Figure 2. **Data Engine** . We present a structured workflow for generating hierarchical annotations across video, event, and clip levels. Clips are first captioned, then processed through a large language model (LLM) with prompts for event summarization. The outputs include clip captions, event summaries, and video summaries, followed by manual checking and refinement. This multi-step approach enriches the dataset with detailed judgments, descriptions, and analyses of anomalies, enabling robust contextual understanding at varying granularities. 

model’s judgment and analysis of the anomaly content. 

### **3.2. LLM-Empowered Data Engine** 

As shown in Fig.2, we develop a semi-automated annotation engine that efficiently scales high-quality annotations, which consists of three main steps: 1) hierarchical video decoupling, 2) hierarchical free-text annotation, and 3) hierarchical instruction construction. 

**Hierarchical Video Decoupling.** Our video sources include the training set of the UCF-Crime [46] dataset and the XD-Violence [56] dataset, which contains videos of varying durations and diverse real-world anomalies. For abnormal videos, we first manually obtain the temporal boundaries of each anomaly event in the video. Non-continuous anomalous states are considered separate events. Then, we divide each event into clips of random lengths. For normal videos, we apply random sampling to obtain corresponding segments of varying granularities. Ultimately, we obtained 5,443 videos, 11,076 events, and 55,806 clips. This process took 5 annotators approximately 20 hours to complete. More details can be found in the appendix. 

**Hierarchical Free-text Annotation.** To fully extract semantic information from the clip-level videos, we utilize a powerful off-the-shell video perception model LLaVANext-Video [28] to generate detailed captions for each clip. We also include the UCA dataset [64], which provides manually annotated captions for video clips in the UCFCrime [46] dataset. Then, we use an LLM [2] to consoli- 

13845 



<!-- Start of picture text -->
(a) 54k Perception (b) (c)<br>11k Reasoning<br>… 5k<br>… …<br>Clip-level Event-level Video-level<br>Granularity<br>Q:  video clip.Give a short and clear caption of the  Caption (d) Description Q:  Describe the anomaly events in the video.<br>A:  The anomaly event involves the  three<br>A:  A group of policemen dragged a man in  policemen turning the man in red over and<br>red out of the gate. moving him to the wall, indicating a<br>possible arrest or detention.<br>Q:  Are there any anomaly events detected  Q:  Why is this event as an anomaly?<br>in the video?<br>A:  The unusual and potentially  forceful<br>A : An anomaly exists, specifically an  interaction between the policemen and the<br>Arrest anomaly. man  in  red,  deviating  from  normal<br>HIVAU-70k pedestrian behavior in the scene<br>Judgement Analysis<br>Count<br><!-- End of picture text -->

Figure 3. **HIVAU-70k dataset** . ( **a** ) Duration distributions for clips, events, and full videos, showing dominance of short clips. ( **b** ) Hierarchical data organization from clip-level to video-level, enabling perception-to-reasoning insights. ( **c** ) Word count variations across annotation levels, with more detailed descriptions at the video level. ( **d** ) Sample annotations capturing captioning, judgment, description, and anomaly analysis, highlighting nuanced understanding of anomaly events in complex scenes. 

date all clip captions within an event, generating an eventlevel video summary. Specifically, we design prompts<sup>1</sup> to guide the LLM to produce three parts of content for each anomalous event summary: **1** ) _Judgment_ : A determination of whether an anomaly exists and its specific category, **2** ) _Description_ : A detailed description of the anomalous or normal event, **3** ) _Analysis_ : The reasoning behind the anomaly judgment, including causal analysis. To guide the LLM in generating reliable responses, we also inject the event’s category label ( _e.g_ ., ”Shooting”, ”Explosion”) into the LLM prompt. We consolidate all event-level summaries to obtain the video-level summary. This results in free-text annotations across multiple temporal scales, including short-term visual perception (clip-level) and longterm anomaly reasoning (event-level, video-level). **Hierarchical Instruction Data Construction.** The ability of VLMs to follow user instructions and generate responses is achieved through instruction-tuning [21, 25, 27, 29]. The training data format typically consists of the following: 

‘ _{Q_ : _user instruction_ , _A_ : _model response.}_ ’ 

To build instruction-tuning data for VLMs in the domain of Video Anomaly Understanding, we matched free-text annotations with pre-designed anomaly-related user instructions. Specifically, for clip-level segments, we only construct instructions related to captions, as it is challenging to obtain anomaly-related analysis for short videos. For event-level and video-level segments, we construct instruction data from the perspectives of Judgment, Description, and Analysis. Typical examples are shown in the Fig. 3 (d). **Manual Checking for Data Quality Control.** To ensure dataset quality, we implemented several human inspec- 

> 1For detailed prompts, please refer to the appendix. 

tion and curation strategies. First, we labeled the temporal boundaries of abnormal event segments and reviewed them at the second level. Next, anomaly labels were incorporated during summary generation using LLMs, directing focus on relevant entities. Finally, manual reviews were performed to correct low-quality instruction data. 

### **3.3. Data Statistic of HIVAU-70k** 

Utilizing the proposed annotation engine, we introduce HIVAU-70k, as shown in Fig. 3, a large-scale benchmark designed for hierarchical instruction-based video anomaly understanding. As shown in Fig. 3(b), HIVAU-70k contains over 70,000 multi-granular annotations organized at clip-level, event-level, and video-level segments, achieving a progression from perception to reasoning. As shown in Fig. 3(a) and (c), the durations of segments and the word numbers of text annotations at different granularities exhibit significant distributional differences. As shown in Fig. 3(d), HIVAU-70k’s instruction data covers _Caption_ , _Judgment_ , _Description_ , and _Analysis_ for real-world anomalies, which guide the model to develop both short-term and long-term video anomaly understanding capabilities. 

## **4. Method: Holmes-VAU** 

Long-term video anomaly understanding with LLMs/VLMs has traditionally been hindered by frame redundancy, complicating accurate anomaly detection. Previous VAU approaches struggle with focus: methods like dense window sampling [67] add redundancy, and uniform frame sampling [9, 47] often misses key anomalies, limiting application to short videos. We introduce the Anomaly-focused Temporal Sampler (ATS) to address this, 

13846 



<!-- Start of picture text -->
Video Frames Visual Token Anomaly-focused Temporal Sampler<br>[patch] [cls]<br>Anomaly Scorer<br>Anomaly Distribution 𝑺<br>T<br>𝑺𝒄𝒖𝒎𝒔𝒖𝒎 cumulation<br>T<br>Token Selection  Sampled Frame Indices<br>& Flattening<br>Prompt Anomaly Understanding Output<br>LoRA<br>Describe the anomaly  The  video  shows  a<br>events in the video. collision between a car<br>and a motorcycle, with<br>scattered debris on the<br>ground. People are seen<br>coming to inspect the<br>scene, indicating that<br>a  traffic accident  may<br>have occurred in the<br>Text  video.<br>Token<br>Encoder Visual<br>… …<br> Sampler<br>…<br>Projector<br>… Large<br>Text  Language Model<br>Encoder<br>Density-aware<br><!-- End of picture text -->

Figure 4. **Holmes-VAU: a multi-modal-LLM-based video anomaly detection framework with adaptive anomaly focus.** 

integrating it into the VLM, and fine-tuning it via instruction on HIVAU-70k to form our Holmes-VAU model. 

### **4.1. Pipeline Overview** 

The overall pipeline of our Holmes-VAU model is shown in Fig. 4. Video frames are processed by a visual encoder, creating visual tokens. These tokens are analyzed by an Anomaly-focused Temporal Sampler using an anomaly scorer and cumulative sum ( _Scumsum_ ) to select keyframes. A text encoder processes a prompt (e.g., ’Describe the abnormal events in the video’). Visual and textual representations are combined in a pre-trained language model, fine-tuned with LoRA, to generate a description of detected anomalies, such as a ’collision between a car and a motorcycle’ and ’traffic accident’ indicators. 

### **4.2. Model Architecture** 

**Visual and Text embedding.** We utilize the frozen visual encoder in InternVL2 [6], which inherits the ViT structure from CLIP [44], we refer to it as _ϕv_ . Following previous VAU works [55, 58, 67, 74], we sample dense video frames at an interval of 16 frames from the input video. Each video frame is then processed through the visual encoder to obtain the corresponding visual tokens. Given the input video frame sequence _V ∈_ R<sup>_T ×H×W ×C_</sup> , the output features of _i_ -th frame can be denotes as: 



where _vi_<sup>_cls_</sup> indicates the class token, _vi_<sup>_j_(</sup><sup>_j∈{_1</sup><sup>_,_2</sup><sup>_, ..., Np}_)</sup> denotes the patch tokens, and _Np_ reperesents the number of patches. The text encoder _ϕt_ is also initialized from [6], 

which includes a tokenizer and an embedding layer. The prompt text _Q_ is converted into text tokens through the text encoder: _Xq_ = _ϕt_ ( _Q_ ). 

**Anomaly-focused Temporal Sampler (ATS).** ATS consists of two components: the _anomaly scorer_ and the _density-aware sampler_ . The _anomaly scorer ϕs_ is a featurebased VAD network which estimates the anomaly score for each frame. We follow the network architecture in [74] due to its simplicity and good performance. Given the class token of the video frames _{v_ 1<sup>_cls, v_</sup> 2<sup>_cls, ..., v_</sup> _T_<sup>_cls}_,theanomaly</sup> scores can be obtained: _si_ = _ϕs_ ( _vi_<sup>_cls_), where</sup><sup>_si_denotes the</sup> anomaly score of the _i_ -th frame. 

Anomalous frames typically contain more information and exhibit greater variation than normal frames [49]. This observation motivates us to sample more frames in regions with higher anomaly scores while reducing the sampling in areas with lower anomaly scores. As shown in Fig. 4, to achieve non-uniform sampling, we propose _density-aware sampler_ to selectively choose _N_ frames from a total of _T_ input frames. Specifically, we treat the anomaly scores _S ∈_ R<sup>_T_</sup> as a probability mass function and first accumulate them along the temporal dimension to obtain the cumulative distribution function, denoted as _Scumsum_ : 



We uniformly sample _N_ points along the cumulative axis, then map these points to the cumulative distribution _Scumsum_ , the corresponding _N_ timestamps on the time axis are mapped to the closest frame index and finally form the sampled frame indices, denoted as _G_ . _τ_ is used to control the uniformity of the sampling. 

**Projector and LLM.** We select the tokens corresponding to the sampled frame, _i.e_ ., _G_ , as the visual embedding. A projector _ϕp_ is then used to map the visual embedding to the language feature space. Finally, we concatenate these embeddings with the text embeddings, input them into the pre-trained large language model, and compute the probability of the target answers _Xa_ . To obtain an initial visuallanguage alignment space, we initialize the projector and LLM parameters from [6], with the parameters kept frozen. The above process can be expressed as follows: 





where cat[ _·_ ] represents the concatenation operation, _θ_ is the trainable parameters, _Xins,<i, Xa,<i_ are the instruction and answer tokens in all turns before the current prediction token _xi_ , _L_ is the length of sequence, respectively. 

13847 

### **4.3. Training and Testing** 

**Training.** We train the model following two steps. In the first step, we use the video data and annotated framelevel label ( _y_ ˆ _∈_ R<sup>_T_</sup> ) from HIVAU-70k to train the _anomaly scorer_ , which provides more accurate anomaly supervision compared to previous unsupervised and weakly-supervised methods [14, 46, 55, 74]. 



In the second step, we keep the anomaly scorer fixed, and use all the instruction data from HIVAU-70k to train the model. To achieve more efficient fine-tuning without disrupting the original capabilities of the LLM, we employ LoRA [15] for fine-tuning, optimizing the cross entropy loss between the predicted and the ground truth tokens. **Testing.** During testing, the user inputs a video and text prompts; the model will generate the corresponding text response following the user’s instruction. 

Table 1. **Comparison of detection performance with state-ofthe-art Video Anomaly Detection approaches** . We include the results of explainable and non-explainable methods. ” _∗_ ” represents the result reported in [67]. 

|**Methods**|**Backbone**|XD-Violence<br>|UCF-Crime<br>|
|---|---|---|---|
|||AP/%|AUC/%|
|No|n-explainable|VAD||
|Conv-AE [14] (CVPR’16)|-|27.25|50.60|
|GODS [52] (ICCV’19)|I3D|N/A|70.46|
|GCL [66] (CVPR’22)|ResNext|N/A|71.04|
|DYANNET [48] (WACV’23)|I3D|N/A|84.50|
|MIST [11] (CVPR’21)|I3D|N/A|82.30|
|Wu_et al_. [56] (ECCV’20)|I3D|78.64|82.44|
|RTFM [49] (ICCV’21)|I3D|77.81|84.30|
|MSL [22] (AAAI’22)|I3D|78.28|85.30|
|S3R [55] (ECCV’22)|I3D|80.26|85.99|
|MGFN [5] (AAAI’23)|I3D|79.19|86.98|
|UR-DMU [74] (AAAI’23)|I3D|81.66|86.97|
|CLIP-TSA [17] (ICIP’23)|ViT|82.19|87.58|
|VadCLIP [58] (AAAI’24)|ViT|84.51|88.02|
|Yang_et al_. [62] (CVPR’24)|ViT|83.68|87.79|
|Wu_et al_.[57] (CVPR’24)|ViT|66.53|86.40|
|Explain|able Multi-m|odal VAD||
|Zero-Shot CLIP [44]<sup>_∗_</sup>|ViT|17.83|53.16|
|LLAVA-1.5 [27]<sup>_∗_</sup>|ViT|50.26|72.84|
|LAVAD [67] (CVPR’24)|ViT|62.01|80.28|
|**_Holmes_-VAU(Ours)**|ViT|**87.68**|**88.96**|



## **5. Experiments** 

### **5.1. Experiment Setup** 

**Dataset.** Our HIVAU-70k is built upon two large-scale realworld datasets, _i.e_ ., UCF-Crime [46] and XD-Violence [56], they provide a diverse range of videos with anomalous events. UCF-Crime [46] comprises 1,900 untrimmed videos totaling 128 hours from outdoor and indoor surveillance cameras. It encompasses 13 classes of real-world anomalies, including _Abuse_ , _Explosion_ , _Fighting_ , and _Shooting_ . XD-Violence [56] is the largest VAD benchmark, comprising 4,754 videos totaling 217 hours sourced from surveillance, movies, car cameras, and games. It encompasses 6 anomaly classes: _Abuse_ , _Car Accidents_ , _Explosions_ , _Fighting_ , _Riots_ , and _Shooting_ . 

**Metric.** We assess the anomaly understanding ability from two aspects: **_anomaly detection_** and **_reasoning_** . **1** ) For **_anomaly detection_** , we use the anomaly scores output by the Anomaly Scorer as the prediction and perform the evaluation. Following [14, 46, 67, 74], we use AUC and AP to quantify detection performance, which is evaluated only on the video level. **2** ) For **_anomaly reasoning_** , we annotate instruction data from the UCF-Crime and XD-Violence test sets, which have been carefully reviewed and filtered by annotators. We finally collected 3,300 test samples at multiple granularities. The test set contains 2200/732/398 samples at clip/event/video levels. We calculate metrics including BLEU [41], CIDEr [51], METEOR [4] and ROUGE [26] to measure the quality of the reasoning text output by the model, comparing with the annotated ground truth text. **Implementation Details.** For the proposed Holmes-VAU 

method, we initialize the Multimodal LLM with InternVL22B [6]. To optimize the Anomaly-focused Temporal Sampler, we adopt the Adam optimizer with a learning rate of 1e-4. Note that when evaluating detection performance on XD-Violence and UCF-Crime, only videos in the corresponding training sets are used to train our model for fair comparisons. For instruction tuning, we train with a batch size of 512 for 1 epoch, using the AdamW optimizer with cosine learning rate decay and a warm-up period. The LoRA [15] parameters are set as: _r_ =64, _α_ =128, and learning rate=4e-5. During testing, _τ_ in Eq. 2 is set to 0.1. Experiments are conducted on 2 NVIDIA A100 GPUs. 

### **5.2. Main Results** 

**Anomaly Detection Results.** We compare our method with state-of-the-art methods, including semi-supervised methods [14, 52], unsupervised methods [48, 66], weaklysupervised methods [17, 22, 49, 55, 58, 74] and recently training-free method [67]. We have indicated their backbones and performance on the UCF-Crime and XDViolence datasets, as shown in Table 1. Our method has an AP of 87.68% on XD-Violence and an AUC of 88.96% on UCF-Crime, significantly outperforming the prior state-ofthe-art methods, which demonstrates that our method can generate less biased anomaly scores. It is worth noting that while achieving precise localization of anomalies, HolmesVAU is also capable of providing explanations and analysis for the detected anomalies by the model, a feature unavailable in existing non-explainable VAD methods. Although LAVAD [67] has explainability, the training-free large lan- 

13848 

Table 2. **Comparison of reasoning performance with state-of-the-art Multimodal Large Language Models (MLLMs).** ’BLEU’ refers to the cumulative values from BLEU-1 to BLEU-4. We evaluate the quality of the generated text at different granularities, including cliplevel (C), event-level (E), and video-level (V). 

|**Method**|**Params**|**BL**<br>|**EU** [41]<br>|(_↑_)<br>|**CI**<br>|**DEr** [51]<br>|(_↑_)<br>|**ME**<br>|**TEOR** [<br>|4](_↑_)<br>|**RO**<br>|**UGE** [26<br>|](_↑_)<br>|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||C|E|V|C|E|V|C|E|V|C|E|V|
|Video-ChatGPT [39]|7B|0.152|0.068|0.066|0.033|0.011|0.013|0.102|0.069|0.044|0.153|0.048|0.079|
|Video-LLaMA [68]|7B|0.151|0.079|0.104|0.024|0.014|0.017|0.112|0.076|0.057|0.156|0.067|0.090|
|Video-LLaVA [25]|7B|0.164|0.046|0.055|0.032|0.009|0.013|0.097|0.022|0.014|0.132|0.023|0.045|
|LLaVA-Next-Video [71]|7B|0.435|0.091|0.120|0.102|0.015|0.031|0.117|0.085|0.096|0.198|0.080|0.106|
|QwenVL2 [53]|7B|0.312|0.082|0.155|0.044|0.020|0.044|0.133|0.092|0.112|0.163|0.081|0.137|
|InternVL2[6]|8B|0.331|0.101|0.145|0.052|0.022|0.035|0.141|0.095|0.101|0.182|0.102|0.122|
|**Holmes-VAU(Ours)**|2B|**0.913**|**0.804**|**0.566**|**0.467**|**1.519**|**1.437**|**0.190**|**0.165**|**0.121**|**0.329**|**0.370**|**0.355**|



Table 3. **Ablation of hierarchical instruction data.** During the instruction tuning phase, we combined training data of different granularities, including clip (C), event (E), and video (V) levels. 

|**Tra**|**ining **|**Data**||**BLEU**(_↑_)||**C**|**IDEr**(_↑_|)|
|---|---|---|---|---|---|---|---|---|
|C|E|V|C|E|V|C|E|V|
|✓|||**0.984**|0.261|0.351|0.459|0.120|0.106|
||✓||0.508|0.576|0.292|0.097|1.183|0.872|
|||✓|0.280|0.222|0.279|0.039|0.708|0.884|
|✓|✓||0.889|0.741|0.349|0.470|1.285|0.889|
|✓||✓|0.906|0.341|0.522|**0.472**|0.962|1.093|
||✓|✓|0.394|0.797|0.505|0.081|1.472|1.074|
|✓|✓|✓|0.913|**0.804**|**0.566**|0.467|**1.519**|**1.437**|



guage model lacks an understanding of anomaly knowledge due to the limitation of insufficient supervised data. **Anomaly Reasoning Results.** We compare the anomalyrelated text quality generated by Holmes-VAU with that produced by state-of-the-art general Multimodal Large Language Models (MLLMs), and presented the results at different temporal granularities, including clip-level, event-level, and video-level, as shown in Table. 2. Earlier MLLMs such as Video-ChatGPT [21] and Video-LLaMA [68], struggled with basic visual perception and instruction-following capabilities. Recent MLLMs [6, 7, 71] trained on larger and higher-quality video instruction data have made significant progress in general video understanding, with noticeable improvements at the clip-level perception task. However, due to the absence of learning from complex, real-world anomaly data, their reasoning abilities at the event-level and video-level are still lacking. Our Holmes-VAU, however, shows significant improvements in video understanding across all temporal granularities compared to existing general MLLMs, highlighting the importance of injecting anomaly-related knowledge through instruction tuning on high-quality Video Anomaly Understanding benchmarks. 

### **5.3. Analytic Results** 

**Influence of Hierarchical Instruction.** To explore the impact of different granularity video training data on the model’s anomaly reasoning ability, we designed various training data combinations during the instruction tuning phase and evaluated the model’s performance, as shown 

Table 4. **Ablation study of sampling methods and the number of sampled frames.** We compare the proposed Anomaly-focused Temporal Sampler (ATS) with other sampling methods under different frame sampling numbers, including _Uniform_ and _Top-K_ . Latency is the time delay in generating the first token. 

|**Frames (N)**|**Sampler**|**Latency (ms)**|**Vide**<br>BLEU(_↑_)|**o-level**<br>CIDEr(_↑_)|
|---|---|---|---|---|
||_Top-K_||0.462|1.229|
|8|_Uniform_|244|0.491|1.276|
||**ATS(Ours)**||**0.514**|**1.324**|
||_Top-K_||0.476|1.302|
|16|_Uniform_|566|0.511|1.345|
||**ATS(Ours)**||**0.566**|**1.437**|
||_Top-K_||0.481|1.332|
|32|_Uniform_|1402|0.558|1.357|
||**ATS(Ours)**||**0.576**|**1.460**|



in Table 3. The inclusion of clip-level data primarily enhanced the model’s basic visual perception abilities regarding actions and scenes within the video. Adding eventlevel data improved the model’s ability to judge and understand complete anomaly events. Furthermore, the involvement of video-level data further enhanced the model’s ability to analyze and summarize anomaly-related information across longer-span videos. The hierarchical instruction data structure facilitated a comprehensive and complementary improvement in the model’s anomaly-related perception-toreasoning capabilities. 

**Influence of different sampling methods and the number of sampled frames.** Our ATS (Anomaly-focused Temporal Sampler) is designed to adaptively sample frames input to the LLM based on the frame-level anomaly scores. To validate its advantages, we compared ATS with other sampling methods at various sample frame counts, including _Uniform_ and _Top-K_ sampling. In _Uniform_ sampling, N frames are uniformly sampled from all frames, while Top-K sampling selects the frames with the top _N_ highest anomaly scores. As shown in Table 4, ATS consistently outperforms other sampling methods, regardless of the sample count. We believe that _Uniform_ sampling tends to overlook key anomaly frames, though this issue lessens as more frames are sampled. Besides, _Top-K_ sampling tends to overly focus on local anomaly frames, missing contextual frame informa- 

13849 



<!-- Start of picture text -->
… … …<br>Prompt:  Determine whether there are  Prompt: How would you describe the particular anomaly events<br>anomaly events in the video and the  InternVL2-8B [Incorrect Judgement] in the video? What evidence do you have to support your<br>specific name of the anomaly. Yes, there are anomaly events in judgment of this event as an anomaly?<br>the video. The specific anomaly is<br>Ground Truth that the man in the video is seen Ground Truth<br>throwing a punch at the other man. The video anomaly event depicts chaotic street scenes where people are<br>The anomaly exists and is specifically  running, police officers are clad in riot gear, and confrontations or<br>identified as a Burglary event.  protests are underway, with scenes of apprehension, restraint, and property<br>QwenVL2-7B [Incorrect Judgement] damage unfolding from start to end. The presence of violent behavior, law<br>enforcement intervention, and disruption of public order in the urban<br>Holmes-VAU (Ours) Yes, there is an anomaly in the Therevideo.are no anomaly events in the environmentdeviation fromdepictednormal,inpeacefulthe videocivic, whichactivity.collectively indicate a significant<br>video,burglary. specifically a break-in Burglary037_x264 Holmes-VAU (Ours)<br>The anomaly events involve a series of chaotic and violent scenes, including<br>police officers in riot gear, people running and pushing against barriers,<br>and a helicopter flying over the scene, suggesting a high level of tension<br>… andpropertyunrest.destruction,The presenceandofphysicaldestructivealtercationsbehavior,, whichincludingdeviatespropertyfrom damage,normal<br>and peaceful urban activity.<br>Prompt:  Identify any abnormal events  InternVL2-8B [Visual Hallucination,  Incomplete Analysis]<br>depicted in the video. InternVL2-8B [Incorrect Judgement] The video shows a group of people walking down a street, and a man is seen<br>The video shows a white truck walking down the street with a skateboard. The man is wearing a black jacket<br>parked on the side of the road, and a black hat. The event depicted in the video is classified as an anomaly<br>Ground Truth which is an abnormal behavior as due to the presence of a large group of people engaging in violent behavior.<br>it is not supposed to be parked on<br>No anomaly exists in the video the road. QwenVL2-7B [Incomplete Description,  Incomplete Analysis]<br>The video shows a group of people walking down the street, and suddenly, a<br>QwenVL2-7B car crashes into a crowd of people. The car is on fire, and the people are<br>Holmes-VAU (Ours) Theabnormalvideoevents.does not show any runningevent, butaway.it Theis clearvideothatdoesitnotisprovidean anomaly.any contextThe carorcrashexplanationis a violentfor thisand<br>unexpected event that is not related to the rest of the video.<br>There is no anomaly in the video.<br>Normal_Videos_902_x264 v=3WmMhircZOc__#1_label_B4-0-0<br><!-- End of picture text -->

Figure 5. **Qualitative comparison of anomaly understanding explanation.** Compared with state-of-the-art general MLLMs, _i.e_ ., InternVL2 [6] and QwenVL2 [53], our proposed Holmes-VAU demonstrates more accurate anomaly judgment, along with more detailed and comprehensive anomaly-related descriptions and analysis. Correct and wrong explanations are highlighted in green and red, respectively. 



<!-- Start of picture text -->
(a) (b)<br>r=32 r=64 r=128<br>iter<br>loss<br><!-- End of picture text -->

Figure 6. **Ablation study of trainable parameters. (a)** Loss curve during instruction-tuning. **(b)** We tuned the LoRA [15] parameter _r_ to control trainable parameters, evaluating its impact on VAU capability, and _General_ performance on Video-MME [12]. 

tion. Our proposed ATS mitigates both issues. To balance inference efficiency and performance, we set _N_ =16 as the default sample frame number. 

**Instruction Tuning Parameters.** We conducted an ablation study on the parameter _r_ in LoRA [15] to explore how the trainable parameters affects both the model’s VAU performance and its general capability. We use VideoMME [12] to evaluate the model’s general capability. The results are shown in Fig. 6, as _r_ increases, the model gradually adapts to the VAU task. However, when _r_ becomes too large, the model’s general capability decreases. To retain the original general video understanding capability of the MLLM, we set _r_ =64 as the default value. 

### **5.4. Qualitative Comparision** 

We provide qualitative comparisons between Holmes-VAU and existing MLLMs in Fig. 5. The results demonstrate that Holmes-VAU can accurately identify anomalies in videos and provide accurate and complete explanations, highlight the effectiveness and advantage of Holmes-VAU in perceiving video events and analyzing anomalies. 

## **6. Conclusion** 

In conclusion, this work pushes the boundaries of video anomaly understanding by introducing hierarchical anomaly detection across diverse temporal scales, from momentary clips to extended events. The HIVAU-70k benchmark, with over 70,000 multi-level annotations, addresses a critical gap in the field, enabling comprehensive anomaly analysis in real-world scenarios. Our Anomaly-focused Temporal Sampler (ATS) strategically enhances focus on anomaly-dense segments, optimizing both efficiency and accuracy in long-term anomaly detection. Extensive experiments demonstrate that our hierarchical dataset and ATS-enhanced VLM achieve significant performance gains over conventional methods, proving robust for open-world anomaly understanding. This work sets a new standard for multi-granular anomaly comprehension, paving the way for more fine-grained video anomaly understanding. 

13850 

**Acknowledgement** This work is supported by the National Natural Science Foundation of China under grants U22B2053 and 623B2039, and in part by the Interdisciplinary Research Program of HUST (2024JCYJ034). 

## **References** 

- [1] Amit Adam, Ehud Rivlin, Ilan Shimshoni, and Daviv Reinitz. Robust real-time unusual event detection using multiple fixed-location monitors. _IEEE transactions on pattern analysis and machine intelligence_ , 30(3):555–560, 2008. 2 

- [2] AI@Meta. Llama 3 model card. 2024. 3 

- [3] Kumar Ashutosh, Rohit Girdhar, Lorenzo Torresani, and Kristen Grauman. Hiervl: Learning hierarchical videolanguage embeddings. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 23066–23078, 2023. 3 

- [4] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 6, 7 

- [5] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 387–395, 2023. 6 

- [6] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 2, 5, 6, 7, 8 

- [7] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. _arXiv preprint arXiv:2406.07476_ , 2024. 7 

- [8] Jisheng Dang, Huicheng Zheng, Xiaohao Xu, Longguang Wang, Qingyong Hu, and Yulan Guo. Adaptive sparse memory networks for efficient and robust video object segmentation. _IEEE Transactions on Neural Networks and Learning Systems_ , 2024. 3 

- [9] Hang Du, Sicheng Zhang, Binzhu Xie, Guoshun Nan, Jiayang Zhang, Junrui Xu, Hangyu Liu, Sicong Leng, Jiangming Liu, Hehe Fan, et al. Uncovering what why and how: A comprehensive benchmark for causation understanding of video anomaly. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18793– 18803, 2024. 2, 3, 4 

- [10] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. _arXiv preprint arXiv:2406.14515_ , 2024. 3 

- [11] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14009– 14018, 2021. 2, 6 

- [12] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. _arXiv preprint arXiv:2405.21075_ , 2024. 3, 8 

- [13] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1705–1714, 2019. 2 

- [14] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 733–742, 2016. 2, 6 

- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 6, 8 

- [16] Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and Gedas Bertasius. Video recap: Recursive captioning of hour-long videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18198–18208, 2024. 3 

- [17] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In _2023 IEEE International Conference on Image Processing (ICIP)_ , pages 3230–3234. IEEE, 2023. 3, 6 

- [18] Kumara Kahatapitiya and Michael S Ryoo. Coarse-fine networks for temporal activity detection in videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8385–8394, 2021. 3 

- [19] Jaechul Kim and Kristen Grauman. Observe locally, infer globally: a space-time mrf for detecting abnormal activities with incremental updates. In _2009 IEEE conference on computer vision and pattern recognition_ , pages 2921–2928. IEEE, 2009. 2 

- [20] Federico Landi, Cees GM Snoek, and Rita Cucchiara. Anomaly locality in video surveillance. _arXiv preprint arXiv:1901.10364_ , 2019. 2 

- [21] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. _arXiv preprint arXiv:2305.06355_ , 2023. 2, 4, 7 

- [22] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 1395–1403, 2022. 2, 6 

- [23] Weixin Li, Vijay Mahadevan, and Nuno Vasconcelos. Anomaly detection and localization in crowded scenes. _IEEE transactions on pattern analysis and machine intelligence_ , 36(1):18–32, 2013. 2 

- [24] Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In _Pro-_ 

13851 

_ceedings of the European Conference on Computer Vision (ECCV)_ , pages 513–528, 2018. 3 

- [25] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_ , 2023. 2, 4, 7 

- [26] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 6, 7 

- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023. 2, 4, 6 

- [28] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3 

- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 2, 4 

- [30] Kun Liu and Huadong Ma. Exploring background-bias for anomaly detection in surveillance videos. In _Proceedings of the 27th ACM International Conference on Multimedia_ , pages 1490–1499, 2019. 2 

- [31] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6536–6545, 2018. 2 

- [32] Yi Liu, Limin Wang, Yali Wang, Xiao Ma, and Yu Qiao. Fineaction: A fine-grained video dataset for temporal action localization. _IEEE transactions on image processing_ , 31: 6937–6950, 2022. 3 

- [33] Zhian Liu, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 13588–13597, 2021. 2 

- [34] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In _Proceedings of the IEEE international conference on computer vision_ , pages 2720–2727, 2013. 2 

- [35] Jian Lu, Xuanfeng Li, Bo Zhao, and Jian Zhou. A review of skeleton-based human action recognition. _Journal of Image and Graphics_ , 28(12):3651–3669, 2023. 3 

- [36] Hui Lv and Qianru Sun. Video anomaly detection and explanation via large language models. _arXiv preprint arXiv:2401.05702_ , 2024. 2 

- [37] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8022–8031, 2023. 2 

- [38] Baiteng Ma, Shiwei Zhang, Changxin Gao, and Nong Sang. Temporal global correlation network for end-to-end action proposal generation. _Acta Electronica Sinica_ , 50(10):2452– 2461, 2022. 3 

- [39] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video 

understanding via large vision and language models. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)_ , 2024. 7 

- [40] Ramin Mehran, Alexis Oyama, and Mubarak Shah. Abnormal crowd behavior detection using social force model. In _2009 IEEE conference on computer vision and pattern recognition_ , pages 935–942. IEEE, 2009. 2 

- [41] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 6, 7 

- [42] Yujiang Pu, Xiaoyu Wu, Lulu Yang, and Shengjin Wang. Learning prompt-enhanced context features for weaklysupervised video anomaly detection. _IEEE Transactions on Image Processing_ , 2024. 2, 3 

- [43] Zhiwu Qing, Shiwei Zhang, Ziyuan Huang, Yi Xu, Xiang Wang, Mingqian Tang, Changxin Gao, Rong Jin, and Nong Sang. Learning from untrimmed videos: Self-supervised video representation learning with hierarchical consistency. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13821–13831, 2022. 3 

- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 2, 5, 6 

- [45] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2616–2625, 2020. 3 

- [46] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6479–6488, 2018. 2, 3, 6 

- [47] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and YingCong Chen. Hawk: Learning to understand open-world video anomalies. _arXiv preprint arXiv:2405.16886_ , 2024. 2, 3, 4 

- [48] Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. Dyannet: A scene dynamicity guided self-trained video anomaly detection network. In _Proceedings of the IEEE/CVF Winter conference on applications of computer vision_ , pages 5541–5550, 2023. 2, 6 

- [49] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 4975–4986, 2021. 2, 5, 6 

- [50] Anil Osman Tur, Nicola Dall’Asen, Cigdem Beyan, and Elisa Ricci. Exploring diffusion models for unsupervised 

13852 

video anomaly detection. In _2023 IEEE International Conference on Image Processing (ICIP)_ , pages 2540–2544. IEEE, 2023. 2 

- [51] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 6, 7 

- [52] Jue Wang and Anoop Cherian. Gods: Generalized one-class discriminative subspaces for anomaly detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 8201–8211, 2019. 2, 6 

- [53] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 7, 8 

- [54] SC Wang, Q Huang, YF Zhang, X Li, YQ Nie, and GC Luo. Review of action recognition based on multimodal data. _Image Graph_ , 27(11):3139–3159, 2022. 3 

- [55] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Self-supervised sparse representation for video anomaly detection. In _European Conference on Computer Vision_ , pages 729–745. Springer, 2022. 2, 5, 6 

- [56] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16_ , pages 322–339. Springer, 2020. 2, 3, 

   - 6 

- [57] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18297–18307, 2024. 2, 3, 6 

- [58] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 6074–6082, 2024. 2, 3, 5, 6 

- [59] Dan Xu, Yan Yan, Elisa Ricci, and Nicu Sebe. Detecting anomalous events in videos by learning deep representations of appearance and motion. _Computer Vision and Image Understanding_ , 156:117–127, 2017. 2 

- [60] Xiaohao Xu, Jinglu Wang, Xiang Ming, and Yan Lu. Towards robust video object segmentation with adaptive object calibration. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 2709–2718, 2022. 3 

- [61] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on keyframes for video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14592–14601, 2023. 2 

- [62] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detec- 

tion. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18899–18908, 2024. 2, 3, 6 

- [63] Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David J Crandall. Dota: Unsupervised detection of traffic anomaly in driving videos. _IEEE transactions on pattern analysis and machine intelligence_ , 45(1): 444–459, 2022. 2 

- [64] Tongtong Yuan, Xuange Zhang, Kun Liu, Bo Liu, Chen Chen, Jian Jin, and Zhenzhen Jiao. Towards surveillance video-and-language understanding: New dataset, baselines, and challenges, 2023. 3 

- [65] Tongtong Yuan, Xuange Zhang, Kun Liu, Bo Liu, Chen Chen, Jian Jin, and Zhenzhen Jiao. Towards surveillance video-and-language understanding: New dataset baselines and challenges. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 22052– 22061, 2024. 2, 3 

- [66] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14744–14754, 2022. 2, 6 

- [67] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18527–18536, 2024. 2, 3, 4, 5, 6 

- [68] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ , 2023. 2, 7 

- [69] Huaxin Zhang, Xiang Wang, Xiaohao Xu, Zhiwu Qing, Changxin Gao, and Nong Sang. Hr-pro: Point-supervised temporal action localization via hierarchical reliability propagation. _arXiv preprint arXiv:2308.12608_ , 2023. 3 

- [70] Huaxin Zhang, Xiang Wang, Xiaohao Xu, Xiaonan Huang, Chuchu Han, Yuehuan Wang, Changxin Gao, Shanjun Zhang, and Nong Sang. Glancevad: Exploring glance supervision for label-efficient video anomaly detection. _arXiv preprint arXiv:2403.06154_ , 2024. 2 

- [71] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. _arXiv preprint arXiv:2410.02713_ , 2024. 7 

- [72] Bin Zhao, Li Fei-Fei, and Eric P Xing. Online detection of unusual events in videos via dynamic sparse coding. In _CVPR 2011_ , pages 3313–3320. IEEE, 2011. 2 

- [73] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1237–1246, 2019. 2 

- [74] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 3769–3777, 2023. 2, 5, 6 

13853 

