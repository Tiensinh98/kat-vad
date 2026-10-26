# **EventVAD: Training-Free Event-Aware Video Anomaly Detection** 

Yihua Shao Haojin He<sup>∗</sup> Peking University Guangdong University of Technology Beijing, China Guangzhou, China yihuajerry@gmail.com 3122009330@mail2.gdut.edu.cn 

Siyu Chen Xinwei Long<sup>†</sup> University of Science and Technology Tsinghua University Beijing Beijing, China Beijing, China longxw22@mails.tsinghua.edu.cn siyuchen200311@163.com 

Yuxuan Fan Muyang Zhang The Hong Kong University of Science Nanjing University and Technology (Guangzhou) Nanjing, China Guangzhou, China zmy417419@163.com orionisfan@outlook.com 

Ao Ma<sup>‡</sup> 

Ao Ma<sup>‡</sup> Xiaochen Wang JD.com University of Science and Technology Beijing, China Beijing maaoaoma@126.com Beijing, China wangxiaochen@ustb.edu.cn 

Sijie Li The University of Sheffield Sheffield, United Kingdom sli256@sheffield.ac.uk 

Fanhu Zeng Tsinghua University Beijing, China zengfanhu2022@ia.ac.cn 

Ziyang Yan University of Trento Trento, Italy ziyang.yan@unitn.it 

Hao Tang Peking University Beijing, China haotang@pku.edu.cn 

Yan Wang Tsinghua University Beijing, China wangyan@air.tsinghua.edu.cn 

### **Abstract** 

Video Anomaly Detection (VAD) focuses on identifying anomalies within videos. Supervised methods require an amount of in-domain training data and often struggle to generalize to unseen anomalies. In contrast, training-free methods leverage the intrinsic world knowledge of large language models (LLMs) to detect anomalies but face challenges in localizing fine-grained visual transitions and diverse events. Therefore, we propose **EventVAD** , an eventaware video anomaly detection framework that combines tailored dynamic graph architectures and multimodal LLMs to perform fine-grained temporal-event reasoning. Specifically, EventVAD first employs dynamic spatiotemporal graph modeling with time-decay constraints to capture event-aware video features. Then, it performs 

∗Equal contribution with Yihua Shao. 

†Minor corresponding author. 

‡Project leader. §Major corresponding author. 

## Shuyan Li<sup>§</sup> 

Queen’s University Belfast Belfast, United Kingdom li-sy16@tsinghua.org.cn 

adaptive noise filtering and uses signal ratio thresholding to detect event boundaries via unsupervised statistical features. Finally, it utilizes a hierarchical prompting strategy to guide MLLMs in performing reasoning and making final decisions. We conducted extensive experiments on the UCF-Crime and XD-Violence datasets. The results demonstrate that EventVAD with a 7B MLLM achieves _stateof-the-art (SOTA)_ in training-free settings, outperforming strong baselines that use 7B or larger MLLMs. The code is available at https://github.com/YihuaJerry/EventVAD. 

### **CCS Concepts** 

• **Computing methodologies** → Scene anomaly detection. 

### **Keywords** 

Multimodal Large Language Models, Vision-Language Model, Video Understanding, Video Anomaly Detection 

#### **ACM Reference Format:** 

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. _MM ’25, Dublin, Ireland_ 

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3754500 

Yihua Shao, Haojin He, Sijie Li, Siyu Chen, Xinwei Long, Fanhu Zeng, Yuxuan Fan, Muyang Zhang, Ziyang Yan, Ao Ma, Xiaochen Wang, Hao Tang, Yan Wang, and Shuyan Li. 2025. EventVAD: Training-Free Event-Aware Video Anomaly Detection. In _Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland._ ACM, Dublin, Ireland, 10 pages. https://doi.org/10.1145/3746027.3754500 

2586 

MM ’25, October 27–31, 2025, Dublin, Ireland 

Yihua Shao et al. 



<!-- Start of picture text -->
Frozen Parameter Normal Frame Abnormal Frame<br>Are there any<br>anomalies ?<br>Frame-to-Text  Caption Scoring<br>Captioner<br>Frame-to-Text  Caption Scoring<br>Captioner<br>Frame-to-Text  Caption Scoring<br>MLLM<br>Captioner<br>(a) Previous Training-Free Video Anomaly Detection<br>Are there any   Hierarchical Prompting<br>anomalies ?<br>RAFT  Event1<br>Scoring<br>Graph<br>Attention Event2<br>MLLM<br>(b) Event-Aware Training-Free Video Anomaly Detection(ours)<br><!-- End of picture text -->

**Figure 1: Existing methods struggle to localize fine-grained abnormal frames. EventVAD segments events to enhance event consistency and fine-grained abnormal detection.** 

### **1 Introduction** 

Recent years have witnessed a growing trend in employing large language models (LLMs) for multi-modal understanding across various multimedia applications. [24, 28, 29, 61, 70]. Among these applications, video anomaly detection (VAD) [34] has emerged as an important research direction to automatically identify and localize anomalous events or segments within video sequences [68, 69]. Before the advent of LLMs [5, 8, 23, 49, 73], supervised training-based methods [39, 43, 58, 59, 65] employ neural networks to achieve high precision (exceeding 90%) on specific datasets. However, these methods struggle to generalize in unseen settings [40] and typically require in-domain fine-tuning when applied to new scenarios. Weakly-supervised[43, 54] or unsupervised methods[66] utilize single-class learning or auto-encoding strategies to train their models, but cannot achieve comparable results due to the shortage of direct and high-quality supervision signals. 

With the advancement of MLLMs [5, 23, 52, 72, 73], recent approaches have employed image-based models (e.g., LLaVA) to detect anomalous video segments via frame-by-frame analysis. Among these, LAVAD [67] represents the first attempt to propose a trainingfree framework, leveraging MLLMs to score anomalous frames without requiring any task-specific training. As shown in Figure 1, despite their success, existing approaches still face two key challenges: (1) Image-based MLLMs often struggle to model temporal dynamics and diverse event patterns in videos, resulting in prediction inconsistency across frames. (2) These methods (e.g., LAVAD) 

rely on a redundant multi-stage VQA pipeline [8, 17, 26], which is inefficient in both storage and inference [37, 38, 71]. 

Inspired by LAVAD [67], we propose a training-free framework **EventVAD** , which segments long videos into short videos to enhance event consistency during MLLM understanding. EventVAD first extracts features from each frame using the ViT-B/16 of EVACLIP [44], and then augments inter-frame features using RAFT optical flow [45] to capture fine-grained features. For temporal features, EventVAD employs the Graph Attention mechanism to aggregate and refine temporal features across frames. This design improves event consistency over time, helping detect boundaries and segment events more accurately. Finally, EventVAD inputs the segmented events into an MLLM [5] for scoring, to determine the anomaly level of each frame. 

We evaluate EventVAD on UCF-Crime [43] and XD-Violence [59] datasets. Experimental results demonstrate that EventVAD achieves _state-of-the-art (SOTA)_ compared to training-free, self-supervised, and one-class supervised methods, and even outperforms some weak-supervised methods. Moreover, our 7B-parameter model is smaller than 13B-parameter baselines (e.g., LAVAD), leading to more efficient storage and lower inference costs. 

Our contributions can be summarized as follows: 

- We propose a training-free event-aware video anomaly detection framework, **EventVAD** , which reduces errors in MLLMs scoring by segmenting events into short events to enhance videos’ temporal consistency. 

- We design a Graph Attention mechanism to model features across multiple frames. By incorporating RAFT optical flow to enhance frame features, Graph Attention enables more accurate event boundary detection and event segmentation. 

- EventVAD achieves _SOTA_ performance among training-free, unsupervised, and one-class methods on both UCF-Crime and XD-Violence datasets, and even surpasses some weakly supervised methods. Additionally, EventVAD compresses the model’s parameters from 13 to 7 billion, significantly reducing computational costs. 

### **2 Related Work** 

### **2.1 Video Anomaly Detection** 

Video Anomaly Detection (VAD) has gained significant attention in recent years due to its wide range of applications aimed at identifying anomalous events in video sequences [21, 32, 34, 43, 55]. Most research in VAD has focused on weakly supervised training paradigms, unsupervised training paradigms and training-free paradigms. Weakly-supervised VAD leverages video-level labels to localize frame-level anomalies, avoiding the prohibitive cost of manual frame annotation [14, 16, 20, 43, 48, 58]. Most of these methods utilize 3D convolutional neural networks for feature learning and are trained with multi-instance learning (MIL) loss [35, 60, 62, 63]. Unsupervised VAD methods do not require any labeled data and instead rely on the inherent structure of normal video data to detect anomalies [46, 47, 50, 51, 66]. Among them, using generative models to capture normal data patterns in videos performs best in unsupervised video anomaly detection, especially in complex scenes and time series data processing. However, these trainingbased approaches require data collection and model retraining in 

2587 

MM ’25, October 27–31, 2025, Dublin, Ireland 

EventVAD: Training-Free Event-Aware Video Anomaly Detection 

practical deployment scenarios. As the first language-based method for training-free VAD using LLMs, LAVAD [67] leverages modalityaligned MLLMs to query and enhance the anomaly scores generated by large language models (LLMs). Based on this, MCANet [6] is a training-free video anomaly detection framework that dynamically generates multimodal text descriptions and analyzes them for efficient anomaly detection by fusing visual language models, audio language models and large language models. Building upon these innovations, through temporal segmentation into discrete event units, EventVAD significantly mitigates error propagation in long-form video analysis, reducing false detection rates. 

### **2.2 Video Understanding** 

Video understanding tasks have been driven by the integration of MLLMs with temporal modeling. Early approaches in Video Large Language Models (Video-LLMs) [2, 12, 19, 22, 25, 31, 74] primarily adopted vision-language adapters from Image-LLMs, such as crossattention [1, 27], Q-former [17, 64, 75], linear projection [3, 24, 56] and dynamic visual tokenizer [13]. These adapters bridge visual features with language models by compressing frame-level representations into fixed-length tokens. For instance, Video-LLaMA [74] employs a video-specific Q-Former to aggregate temporal features, while Video-ChatGPT [31] incorporates spatial and temporal pooling layers to enhance frame-level reasoning. Recent works like VTimeLLM [11] and CogVLM2 [10] propose boundary-aware training and timestamp-aware encoding, respectively, to address the problem of the limited effective context length of LLM for long video processing and redundancy in processing similar frames. The integration of multimodal alignment techniques has further expanded video understanding capabilities. Models like ImageBind [7] unify embeddings across six modalities into a shared latent space, enabling zero-shot cross-modal retrieval. This approach has been widely adopted in frameworks such as PandaGPT [42], where audio streams are aligned with visual features via pre-trained encoders. However, these methods prioritize modality fusion over temporalaudio synchronization, limiting their ability to correlate dynamic events with acoustic cues. EventVAD leverages video LLMs to analyze segmented events, generating frame-level anomaly scores for precise temporal localization of abnormal content. 

### **3 Methodology** 

As shown in Figure 2, given an input video sequence V = {I _𝑡_ }<sup>_𝑇_</sup> _𝑡_ =1 with _𝑇_ frames, our objective is to detect anomaly frames. { _𝑎𝑡_ }<sup>_𝑇_</sup> _𝑡_ =1 are anomaly indicators, where _𝑎𝑡_ ∈{0 _,_ 1} represents the binary label for frame I _𝑡_ . The overall pipeline is shown as Fig 2, which consists of four components. Specifically, our method first constructs a dynamic graph with multimodal temporal decay, where nodes represent fused frame features and edges denote spatiotemporal connections. Then, graph attention propagation refines node features through orthogonally constrained messaging. Finally, event-centric scoring uses semantic cues to assess anomalies within detected event units. 

### **3.1 Event-Aware Dynamic Graph Construction** 

**Node Feature Construction.** To address the inherent limitations of static graphs in capturing dynamic video events and the slow 

training speed of graph neural networks [36], we propose a dynamic spatiotemporal graph model based on video frames. Our model jointly optimizes multimodal node representations and timesensitive edge connections. The proposed dynamic spatiotemporal graph model _𝐺_ = ( _𝐹, 𝐸_ ) introduces adaptive node representations through multimodal feature interaction. The vertex set F corresponds to video frame units with enhanced feature encoding, while edges in _𝐸_ are dynamically established using cross-modal similarity metrics. 

The nodal feature representation integrates a dual-stream architecture that combines semantic and motion patterns. As shown in Eq. (1), the semantic branch employs CLIP [33] to generate discriminative embeddings via L2-normalization. 



where _𝜙_ CLIP generates 512-dimensional semantic embeddings. This explicit normalization amplifies feature discontinuities between frames, a critical component for detecting semantic-level event boundaries. 

The motion branch captures temporal dynamics using optical flow processing. This branch is formulated as Eq. (2): 



where o _𝑖_ → _𝑖_ −1 ∈ R<sup>_𝐻_×</sup><sup>_𝑊_×2</sup> represents the optical flow fields from RAFT [45]. The projection matrix P ∈ R<sup>2×128</sup> maps high-dimensional statistical features to 128-dimensional compact vectors. 

The fused node representation for the graph vertex _𝑣𝑖_ ∈ _𝐹_ is calculated using Eq. (3): 



where the semantic-motion fusion coefficient _𝛼_ = 0 _._ 75 and the eigenvector f _𝑖_ integrate semantic and motion constraints to precisely define event transformation boundaries. 

**Temporal Decay Adjacency Matrix.** To capture relationships between nodes, we construct a dynamic graph with time decay. The time decay adjacency matrix _𝐸_ is defined as Eq. (4): 



where the semantic-motion fusion coefficient _𝛼_ = 0 _._ 75 is shared with the node usage coefficient. To emphasize short-term correlations and suppress long-term ones, our model combines multimodal similarity between frames _𝑖_ and _𝑗_ in the numerator. In the denominator, we introduce a _𝛾_ time decay factor to penalize remote associations, which diminish with temporal distance. 

### **3.2 Graph Attention Networks Propagation** 

To improve frame-by-frame representation while preserving temporal consistency, we propose a training-free attention mechanism based on orthogonal feature projection in spacetime. This mechanism amplifies segment contrast through graph-guided message passing, enhancing event boundary distinguishability. Our approach builds upon graph attention networks [53] with orthogonal constraints. 

2588 

MM ’25, October 27–31, 2025, Dublin, Ireland 

Yihua Shao et al. 



<!-- Start of picture text -->
RGB   Frame Optical   Flow<br>Visual  Optical<br>Frame i  Encoder Encoder<br>� 1 −�<br>�<br>… …<br>Frame i-2  Frame i -1  Frame i +1  Frame n  �<br>Event-Aware Dynamic Graph Construction  Graph Attention Propagation<br>#Question : Are there any obvious or potential anomalies<br>in the video?<br>#Instruction:  Let’s think step by step to judge the  The K-th Event  Softmax<br>anomaly of the video. Finally,  output the anomaly score  +<br>based on the thinking process. ReLU<br>MLLM l-th layer<br>#Answer:  A man is seen with a gun, pointing it at the  �<br>employees. He takes money from the register and …  �<br>Therefore, the final score is 0.8. �<br>�<br>Event-Centric Anomaly Scoring Statistical Boundary Detection<br>A. Overall Pipeline of EventVAD B. Graph Attention Module<br>Input<br>Output<br><!-- End of picture text -->

**Figure 2: EventVAD comprises four core modules: (1) Dynamic Graph Construction for establishing spatiotemporal correlations, (2) Graph Attention Propagation for refining node features, (3) Statistical Boundary Detection for identifying event transitions and boundaries, and (4) Event-Centric Scoring for evaluating anomalies via MLLMs within semantic event units.** 

Given the propagated node features F<sup>(0)</sup> = [f1 _, ...,_ f _𝑛_ ]<sup>⊤</sup> ∈ R<sup>_𝑛_×</sup><sup>_𝑑_</sup> from Section 3.1, we project the features into orthogonal subspaces to prevent dimension collapse as Eq (5), 



where QR(·) denotes orthonormal columns via QR decomposition, _𝑑_ = 640 is the fused feature dimension, and _𝑘_ = 64 is the projected dimension. These fixed orthogonal matrices maximize feature retention. 

For node _𝑓𝑖_ the attention weight for neighbor _𝑓𝑗_ in the dynamic graph is calculated as: 



The indicator function E( _𝑖,𝑗_ ) corresponds to temporal connectivity in Section 3.1, ensuring attention respects event-induced topology. This constraint prevents attention dispersion to irrelevant frames. Features update iteratively via: 



After each iteration, features are centered as: 



Through iterative graph attention updates, feature propagation maintains global divergence and local consistency, preserving relative differences critical for anomaly detection. 

### **3.3 Statistical Boundary Detection** 

For enhanced graph attention network features, we propose a statistical boundary detection mechanism to identify event transitions via temporal discontinuity analysis. 

For consecutive frames ( _𝑖,𝑖_ + 1), we compute a composite divergence metric: 



where the squared L2 term amplifies abrupt feature-space jumps, and the cosine term detects directional changes in the manifold. This dual strategy captures amplitude and directional discontinuities caused by anomalies. 

The raw dissimilarity score _𝑠𝑖_ includes high-frequency noise from camera shake or minor motion. To suppress noise while preserving true boundaries, we apply a Savitzky-Golay filter with a window width _𝑤_ = 60, defined as: 



2589 

MM ’25, October 27–31, 2025, Dublin, Ireland 

EventVAD: Training-Free Event-Aware Video Anomaly Detection 

where _𝑎𝑘_ derives from a quadratic polynomial fit, retaining true discontinuities. 

For the smoothed signal _𝑠_ ˜ _𝑖_ , we compute a moving average: 



The signal ratio _𝑟𝑖_ is then calculated as: 



where equal window weights suppress transient fluctuations, enhancing noise robustness. 

To mitigate outlier sensitivity from occlusions or jitter, we use median absolute deviation for adaptive thresholding: 





Adjusting _𝑘_ = 3 aligns with the 3 _𝜎_ principle [15], capturing 99.7% of normal variation. Boundaries are identified when _𝑟𝑖 > 𝑀_ . Finally, adjacent boundary points are merged to prevent over-segmentation, refining event boundary accuracy. 

### **3.4 Event-Centric Anomaly Scoring** 

Existing video anomaly detection methods [67] often rely on discrete frame-level analysis or global video processing. However, their non-adaptive segmentation mechanisms face a critical tradeoff: overly short temporal fragments lose contextual information, while excessively long units weaken spatial feature representation. This limitation hinders visual language models’ ability to interpret videos effectively. 

To address this, we construct event semantic units as spatiotemporal primitives, creating an optimized feature representation framework for visual-language models [24]. For event unit analysis, we propose a semantic-driven hierarchical prompting framework. This framework directs multimodal large language models to produce structured outputs: first generating video content descriptions and then deriving anomaly scores. When processing a video, the multimodal large language models initially generate descriptive text by identifying surface and latent semantic features. It subsequently outputs an anomaly score based on this description, enabling crossmodal evaluation against predefined criteria. 

By implementing a two-stage reasoning framework, we establish a self-correction mechanism. This architecture allows systematic score derivation through video content analysis, ensuring contextual relevance and reducing scoring inaccuracies. 

### **4 Experiment** 

### **4.1 Experiment Setting** 

**Setting Up.** We adopt the CLIP ViT-B/16 model [33] to generate 512-dimensional semantic embeddings, and the pre-trained RAFT model to obtain 128-dimensional optical flow features, with _𝐹𝑃𝑆_ = 30. We also use bilinear interpolation for spatial alignment to maintain resolution consistency. We employ the VideoLLaMA2.17B-16F model [5] as the backbone model. Additionally, the time decay factor is set to _𝛾_ = 0 _._ 6, the semantic-motion fusion coefficient 

|**Method**|**Backbone**|**Supervised**|**AUC (%)**|
|---|---|---|---|
|Wu_et al._[59]|I3D-RGB|Weakly Supervised|82.44|
|RTFM [48]|I3D-RGB|Weakly Supervised|84.03|
|Wu & Liu_et al._[58]|I3D-RGB|Weakly Supervised|84.89|
|MSL [20]|VideoSwin-RGB|Weakly Supervised|85.62|
|S3R [57]|I3D-RGB|Weakly Supervised|85.99|
|MGFN [4]|I3D-RGB|Weakly Supervised|86.98|
|SSRL [16]|I3D-RGB|Weakly Supervised|87.43|
|CLIP-TSA [14]|ViT|Weakly Supervised|87.58|
|SVM [43]|-|One Class|50.00|
|SSV [41]|-|One Class|58.50|
|BODS [54]|I3D-RGB|One Class|68.26|
|GODS [54]|I3D-RGB|One Class|70.46|
|GCL [66]|ResNext|Unsupervised|74.20|
|Tur [50]|ResNet|Unsupervised|65.22|
|Tur [51]|ResNet|Unsupervised|66.85|
|DyAnNet [47]|I3D|Unsupervised|79.76|
|Blip2 [17]|ViT|Training Free|46.42|
|ZS CLIP [33]|ViT|Training Free|53.16|
|ZS ImageBind (Image) [7]|ViT|Training Free|53.65|
|ZS ImageBind (Video) [7]|ViT|Training Free|55.78|
|LLaVA-1.5 [23]|ViT|Training Free|72.84|
|Video-Llama2 [74]|ViT|Training Free|74.42|
|LAVAD [67]|ViT|Training Free|78.33|
|**EventVAD (Ours)**|**ViT**|**Training Free**|**82.03**|



**Table 1: Results on the UCF-Crime dataset.** 

is set to _𝛼_ = 0 _._ 75, and the graph attention propagation is only a single iteration. All experiments were conducted on a single NVIDIA A800 (80GB) GPU. 

**Datasets.** We evaluate EventVAD on two widely adopted video anomaly detection benchmarks, UCF-Crime [43] and XD-Violence [59]. **Evaluation Metrics.** For the UCF-Crime dataset, we employ framelevel **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve). For the XD-Violence dataset, we also employ **AP** (Average Precision) as the primary metric to align with standard VAD settings. 

### **4.2 Main Results** 

**Results on UCF-Crime.** Since the UCF-Crime data comes from the real shooting scene of the camera, the similarity between events in the same video is high. It is more challenging when the model divides the events. As shown in Tab. 1, **EventVAD surpasses all training-free methods to** **_state-of-the-art (SOTA)_ and outperforms LAVAD by nearly 4%.** compared to LAVAD. EventVAD’s foundational model is Video-Llama2 [5] with a parameter count of 7b, which has a significant decrease in the overall parameter of the model. Also, EventVAD outperforms all unsupervised and one-class methods, as well as some weakly supervised methods on UCF-Crime. 

By analyzing the results, we found that since the other methods ignore dividing the video, different model parameter sizes cannot achieve accurate localization when dealing with long videos. This suggests that EventVAD’s approach of dividing events and then scoring event anomalies can effectively mitigate the model’s inability to achieve anomaly detection for long videos. Even in noisy scenarios, EventVAD demonstrates robust detection capability under challenging conditions with both rare abnormal samples (low occurrence rate <5%) and significant visual noise. **Results on XD-Violence.** Since the XD-Violence datasets are all derived from video recordings, the videos are clearer, and there is 

2590 

MM ’25, October 27–31, 2025, Dublin, Ireland 

Yihua Shao et al. 

|**Method**|**Backbone**|**Supervised**|**AP (%) **|**AUC (%)**|
|---|---|---|---|---|
|Wu_et al._[59]|I3D-RGB|Weakly Supervised|73.20|-|
|Wu and Liu [58]|I3D-RGB|Weakly Supervised|75.90|-|
|RTFM [48]|I3D-RGB|Weakly Supervised|77.81|-|
|MSL [20]|VideoSwin-RGB|Weakly Supervised|78.58|-|
|S3R [57]|I3D-RGB|Weakly Supervised|80.26|-|
|MGFN [4]|VideoSwin-RGB|Weakly Supervised|80.11|-|
|Hasan_et al._[9]|AE<sup>RGB</sup>|One Class|-|50.32|
|Lu_et al._[30]|Dictionary|One Class|-|53.56|
|BODS [54]|I3D-RGB|One Class|-|57.32|
|GODS [54]|I3D-RGB|One Class|-|61.56|
|RareAnom [46]|I3D-RGB|Unsupervised|-|68.33|
|Blip2 [17]|ViT|Training Free|10.89|29.43|
|ZS CLIP [33]|ViT|Training Free|17.83|38.21|
|ZS ImageBind (Image) [7]|ViT|Training Free|27.25|58.81|
|ZS ImageBind (Video) [7]|ViT|Training Free|25.36|55.06|
|LLaVA-1.5 [23]|ViT|Training Free|50.26|79.62|
|Video-Llama2 [74]|ViT|Training Free|53.57|80.21|
|LAVAD [67]|ViT|Training Free|60.02|82.89|
|**EventVAD (Ours)**|**ViT**|**Training Free**|**64.04**|**87.51**|



**Table 2: Results on the XD-Violence dataset.** 

less noise in the picture. As shown in Tab. 2, **EventVAD’s AP and AUC on this dataset are almost 5% higher than the current training-free SOTA LAVAD [67].** Also, EventVAD anomaly detection capability surpasses all one-class and unsupervised methods. This indicates that EventVAD is not only suitable for low-resolution scenarios but also applicable to high-resolution scenarios. Even though EventVAD has significantly fewer model parameters than LAVAD, its performance shows a substantial improvement compared to LAVAD. This can significantly reduce computational and deployment costs for the model. 

After analyzing the samples that LAVAD failed to detect, we found that the current model performs poorly in long-video detection. Additionally, many abnormal events only occupy a small portion of the frame, making it difficult for the model to achieve precise localization. Moreover, compared to LAVAD, EventVAD’s prompt setup is very straightforward. We achieve multi-stage reasoning in MLLM through hierarchical prompting to enhance its scene-understanding capability. 

**Visual Analysis.** As shown in Fig. 3, using a UCF-Crime video sample as an example, we demonstrate a comprehensive visualization of abnormal and normal sequences by the graph attention propagation process. 

Fig. 3 reveals fundamental differences in the modeling of the temporal relationship between normal and abnormal sequences. Both the Subgraph 3a and Subgraph 3c show chaotic inter-frame connections, with low discriminative long-distance dependence. After the attention propagation of the applied graph, subgraphs (c) and (d) clearly show strengthened connections between adjacent frames, and the pseudo-correlation with the distant frame is suppressed. In particular, Subgraph 3d as a whole presents a relatively uniform inter-frame weight, but Subgraph 3c has a unique cluster of low weights around frames 965-975, which corresponds to the actual boundary division. 

This actually verifies how the propagation process of our graph attention acts as a feature contrast amplifier, with the help of which the Statistical Boundary Detection module can convert the amplified signal difference into an accurate event boundary. 

**Boundary Detection Visualization.** As shown in Fig. 4, we visualize the boundary detection of all frames of these video samples. The light blue curve represents the original signal used for segmentation, the green curve represents the signal after being smoothed by Savitzky-Golay filtering, the red curve represents the change in the ratio of the smoothed signal within the window to the average line of the corresponding window, the black horizontal line represents the adaptive threshold, and the purple horizontal line represents the detected boundary. 

For different types of videos, there are usually different event boundary patterns. Normal video sequences have fewer events due to their good consistency, while abnormal videos have the opposite. The difference between Subgraph 4a and Subgraph 4b validates that our framework can take advantage of this feature to obtain clear boundary points at different boundary densities when the score threshold is set to 2, thus achieving a robust distinction between normal time coherence and anomalous discontinuity. 

In the Subgraph 4a, for frames 965-975 of the video, our framework can effectively capture the key boundary detection patterns after the graph attention propagation, which also quantitatively verifies the subsequent boundary detection process of the Subgraph 3b. In addition, the framework exhibits noise elasticity when dealing with high variance clips, such as 1200 to 1300 frames of video, and maintains some sensitivity to interframe variations in overall signal-stable clips, such as 1750 to 1900 frames of video. **Qualitative Results** To intuitively demonstrate EventVAD’s capabilities, we visually compare its performance against LAVAD by selecting failure cases from both the UCF-Crime and XD-Violence datasets for qualitative analysis. 

For anomalous samples, as shown in Fig. 5a, the video is a long sequence with over 2,500 frames, and the anomaly (robbery) is primarily concentrated in the lower-right corner of the footage. Therefore, most methods like LAVAD [67] struggle to detect such subtle anomalies. Although EventVAD’s segmentation of event boundaries (start/end points) lacks precision, it effectively identifies the core segments of events. As shown in Fig. 5c, despite the visual prominence of the main subjects in the video, environmental interference can occasionally cause them to disappear from view. EventVAD segments videos into distinct events and leverages event consistency to enable precise anomaly localization within each event, effectively reducing false detections by the model. 

For normal samples, as shown in Fig. 5b and Fig. 5d, Fig. 5b exhibits significant blurriness with poorly defined subjects, which poses substantial challenges for QA models like Blip [17, 18] in performing accurate detection. Fig. 5d exceeds 1,400 frames in length, presenting significant detection challenges for LLMs [24, 49]. Using an event-aware approach, EventVAD optimizes video length for LLM input through event consistency. Even for lengthy videos, consistent events enable the LLM to achieve comprehensive anomaly detection by analyzing only the initial frames, thereby avoiding length-induced misjudgments. 

### **4.3 Ablation Study** 

**Efficiency-Accuracy Trade-off.** Since the model parameters of the original LAVAD is 13 billion, we will explore the impact of model size on the framework’s performance. As shown in Tab. 3, 

2591 

MM ’25, October 27–31, 2025, Dublin, Ireland 

EventVAD: Training-Free Event-Aware Video Anomaly Detection 





**(a) Abnormal video before graph at(b) Abnormal video after graph attentention networks propagation tion networks propagation** 





**(c) Normal video before graph atten(d) Normal video after graph attention networks propagation tion networks propagation** 

**Figure 3: Graph attention propagation visualization analysis. We take the abnormal and normal video samples in the UCFCrime dataset to visualize the inter-frame relationship before and after the application of Graph Attention Propagation. The heat maps show the change of the weight relationship between frames in the corresponding frame interval.** 

|**Method**<br>**F**|**oundational Model**|**Parameters**<br>**AUC (%)**|
|---|---|---|
|LAVAD [67]<br>**EventVAD (Ours)**|Llama2-7b|7B<br>68.58<br>**78.39**|
|LAVAD [67]<br>**EventVAD (Ours)**|Llama2-13b|13B<br>78.33<br>**80.12**|
|LAVAD [67]<br>**EventVAD (Ours)**|Video-Llama2|7B<br>73.28<br>**82.03**|
|**Table 3: Re**|**sults of scaling u**|**p the MLLMs**|



we first aligned the base model parameters of LAVAD and EventVAD. When both base models used Llama2 [49], we found that increasing LAVAD’s model parameters could slightly enhance its anomaly detection capability. Although EventVAD also showed some improvement with larger model parameters, the gain was less pronounced compared to LAVAD. When using the 7B-parameter Video-Llama2 [5], EventVAD achieved an AUC of 82.03%, which is 3.69% higher than the 13B-parameter LAVAD baseline.This validates the effectiveness of our event-centric architecture, which divides long videos into coherent segments to improve LLM comprehension, thus enabling reduced reliance on parametric scaling. **Impact of Hyperparameters on Model Performance.** EventVAD’s core hyperparameters - the temporal decay coefficient _𝛾_ and semantic-motion fusion coefficient _𝛼_ - critically govern model behavior. We present a systematic analysis of their impacts. 





<!-- Start of picture text -->
(a) Abnormal video statistical boundary detection<br><!-- End of picture text -->



**(b) Normal video statistical boundary detection** 

**Figure 4: Statistical boundary detection visual analysis. We use samples from the UCF-Crime dataset as an example to visualize the boundary detection process of abnormal and normal videos.** 

||_𝛼_=0|_𝛼_=0_._25|_𝛼_=0_._5|_𝛼_=0_._75|_𝛼_=1|
|---|---|---|---|---|---|
|_𝛾_=0_._0|77.92|77.13|78.47|78.91|77.34|
|_𝛾_=0_._2|78.22|78.85|79.48|80.56|77.93|
|_𝛾_=0_._4|78.95|79.12|79.95|81.78|79.46|
|_𝛾_=0_._6|79.89|80.67|81.57|**82.03**|79.69|
|_𝛾_=0_._8|78.92|79.54|80.38|81.34|79.62|
|_𝛾_=1_._0|77.12|78.81|78.75|80.97|78.97|



**Table 4: Ablation Studies of Hyperparameters.** 

As shown in Table 4, the temporal decay factor and the semanticmotion fusion coefficient have a serious impact on model behavior through their coordinated control of spatiotemporal feature integration. With the increase of and , it is initially possible for the framework to improve the quality of segmentation by enhancing the semantic correlation between frames while maintaining appropriate temporal continuity constraints. However, too high parameter values will lead to performance degradation, which will make the model overly sensitive to semantic changes, generate fragmented fragments, and thus disrupt event continuity. These ultra-short fragments prevent MLLMs from establishing a full semantic context, resulting in hallucinogenic anomalous scoring. 

To maintain a good semantic correlation between frames while having a certain time continuity constraint, we choose _𝛼_ = 0 _._ 75 and _𝛾_ = 0 _._ 6 for segmentation. 

**Key Module Ablation.** For EventVAD, the Optical Flow Module, the Graph Attention propagation module, and the video content description before MLLMs scoring provide the main contribution 

2592 

MM ’25, October 27–31, 2025, Dublin, Ireland 

Yihua Shao et al. 



<!-- Start of picture text -->
The supermarket cashier was  One robber took a bag and took a lot  The video shows a restaurant with a room full of tables and chairs. Some people sat at some tables while<br>checking out normally when suddenly two men came in and  of stuff from the supermarket, and another robber forced the cashier at  others walked around the dining room, with a woman walking quickly through the dining room towards the kitchen area, and then she walked briskly back to the dining area. Throughout the video, different<br>looked ill-intentioned. gunpoint to hand over the money at  people can be seen sitting at different tables eating or talking.<br>the cash register.<br>Event1 Event2 Event3 Event4 Event1<br>(a) Abnormal Examples in UCF-Crime Dataset (b) Normal Examples in UCF-Crime Dataset<br>The video depicts a controlled  The video shows an aerial view of  In the video, we see a man driving a car with two children seated beside him. Suddenly, a police officer<br>demolition process where a large  the site, showing the extent of the  appears and asks to see their license and registration. The man hands them over reluctantly and the<br>concrete structure is being demolished  damage caused by the blast and  officer begins to write them a ticket while speaking to them. After the interaction is over, the man and<br>using explosives. the surrounding landscape. his passengers get out of the car and gather their groceries before walking away together.<br>Event1 Event2 Event3 Event4 Event1<br>(c) Abnormal Examples in XD-Violence Dataset (d) Normal Examples in XD-Violence Dataset<br><!-- End of picture text -->

**Figure 5: Visualization of normal and abnormal samples in the UCF-Crime and XD-Violence datasets. We visualized the samples where LAVAD failed to detect anomalies correctly. The figure presents event segmentation results along with the MLLM’s anomaly frame scores, with comparisons to the ground-truth.** 

|**RAFT**|**GANP**|**Thinking**|**AUC (%)**|Δ**(%)**|
|---|---|---|---|---|
|�|�<br>|�|73.93|-|
|�|�|�|77.81|+3.88|
|�|�|�|75.35|+1.42|
|�|�|�|80.14|+6.21|
|�|�|�|75.64|+1.71|
|�|�|�|79.97|+6.04|
|�|�|�|76.86|+2.93|
|�|�|�|82.03|+8.10|



**Table 5: Ablation Studies of Key Modules on EventVAD. Through ablation studies, we validate their effectiveness and identify the most critical ones.** 

neural network training times. Although RAFT optical flow cannot directly enhance the capabilities of the Dynamic Graph or Graph Attention Propagation modules, it can enhance the modeling capabilities of the Dynamic Graph through temporal feature matching, thereby improving its boundary detection performance. In addition, the quantitative experiments show the structured MLLM output can be specified, so that it can summarize the video content before outputting the abnormal score, and this process of letting the model think can better help him understand these small fragments and give a more reasonable score. 

to the model’s performance. To systematically evaluate each component’s contribution, we conducted modular ablation studies focusing on three key aspects: (1) the inclusion of RAFT-based optical flow features during encoding, (2) the use of graph attention network propagation (GANP) following dynamic graph construction, and (3) the implementation of deliberative reasoning in MLLM outputs before anomaly scoring. We further analyzed cross-component interactions to explore potential synergies. 

As shown in Table 5, when the original model is combined with the RAFT optical flow, the detection AUC increases by 2%, while with the help of graph attention propagation, the detection AUC of the model is further increased by about 4%. This is a significant improvement over static graphs and solves the problem of long graph 

### **5 Conclusion** 

In this paper, we propose a novel training-free event-aware video anomaly detection framework, EventVAD. EventVAD addresses the challenge of localizing fine-grained visual transitions and diverse events in long videos by segmenting events in videos. EventVAD achieves more accurate event boundary detection and events division by introducing optical flow matching and dynamic graph architectures so that MLLMs can score anomalies accurately within events. We evaluated EventVAD on both UCF-Crime and XD-Violence datasets. Experimental results demonstrate that EventVAD can segment events accurately in videos and achieves SOTA in video anomaly detection on both datasets in the training-free setting with 7B MLLMs compared to baseline 13B or larger MLLMs. 

2593 

MM ’25, October 27–31, 2025, Dublin, Ireland 

EventVAD: Training-Free Event-Aware Video Anomaly Detection 

### **Acknowledgements** 

This research was supported by funding from the Start-up Package, School of EEECS, Queen’s University Belfast (D8203EEC) 

### **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ 35 (2022), 23716–23736. 

- [2] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. 2024. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. _arXiv preprint arXiv:2404.03413_ (2024). 

- [3] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023. Shikra: Unleashing multimodal llm’s referential dialogue magic. _arXiv preprint arXiv:2306.15195_ (2023). 

- [4] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. 2023. Mgfn: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection. In _AAAI_ . 

- [5] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. 2024. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. _arXiv preprint arXiv:2406.07476_ (2024). 

- [6] Prabhu Prasad Dev, Raju Hazari, and Pranesh Das. 2025. MCANet: Multimodal Caption Aware Training-Free Video Anomaly Detection via Large Language Model. In _International Conference on Pattern Recognition_ . Springer, 362–379. 

- [7] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. Imagebind: One embedding space to bind them all. In _CVPR_ . 

- [8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ (2024). 

- [9] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. 2016. Learning temporal regularity in video sequences. In _CVPR_ . 

- [10] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. 2024. Cogvlm2: Visual language models for image and video understanding. _arXiv preprint arXiv:2408.16500_ (2024). 

- [11] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. 2024. Vtimellm: Empower llm to grasp video moments. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 14271–14280. 

- [12] Yang Jin, Zhicheng Sun, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, et al. 2024. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. _arXiv preprint arXiv:2402.03161_ (2024). 

- [13] Yang Jin, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Quzhe Huang, Bin Chen, Chenyi Lei, An Liu, Chengru Song, et al. 2023. Unified language-vision pretraining in llm with dynamic discrete visual tokenization. _arXiv preprint arXiv:2309.04669_ (2023). 

- [14] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. 2023. CLIP-TSA: CLIP-Assisted Temporal Self-Attention for Weakly-Supervised Video Anomaly Detection. In _ICIP_ . 

- [15] Christophe Leys, Christophe Ley, Olivier Klein, Philippe Bernard, and Laurent Licata. 2013. Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. _Journal of experimental social psychology_ 49, 4 (2013), 764–766. 

- [16] Guoqiu Li, Guanxiong Cai, Xingyu Zeng, and Rui Zhao. 2022. Scale-Aware Spatio-Temporal Relation Learning for Video Anomaly Detection. In _ECCV_ . 

- [17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _ICML_ . 

- [18] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _International conference on machine learning_ . PMLR, 12888–12900. 

- [19] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. 2024. Mvbench: A comprehensive multi-modal video understanding benchmark. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 22195–22206. 

- [20] Shuo Li, Fang Liu, and Licheng Jiao. 2022. Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection. In _AAAI_ . 

- [21] Minwen Liao, Hao Bo Dong, Xinyi Wang, Ziyang Yan, and Yihua Shao. 2025. GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts. _arXiv preprint arXiv:2503.07417_ (2025). 

- [22] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-llava: Learning united visual representation by alignment before projection. 

_arXiv preprint arXiv:2311.10122_ (2023). 

- [23] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023. Improved baselines with visual instruction tuning. _arXiv_ (2023). 

- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. _Advances in neural information processing systems_ 36 (2023), 34892–34916. 

- [25] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. 2024. St-llm: Large language models are effective temporal learners. In _European Conference on Computer Vision_ . Springer, 1–18. 

- [26] Xinwei Long, Zhiyuan Ma, Ermo Hua, Kaiyan Zhang, Biqing Qi, and Bowen Zhou. 2025. Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 39. 24723–24731. 

- [27] Xinwei Long, Shuzi Niu, and Yucheng Li. 2021. Position enhanced mention graph attention network for dialogue relation extraction. In _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval_ . 1985–1989. 

- [28] Xinwei Long, Jiali Zeng, Fandong Meng, Zhiyuan Ma, Kaiyan Zhang, Bowen Zhou, and Jie Zhou. 2024. Generative multi-modal knowledge retrieval with large language models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 38. 18733–18741. 

- [29] Xinwei Long, Jiali Zeng, Fandong Meng, Jie Zhou, and Bowen Zhou. 2024. Trust in internal or external knowledge? generative multi-modal entity linking with knowledge retriever. In _Findings of the Association for Computational Linguistics ACL 2024_ . 7559–7569. 

- [30] Cewu Lu, Jianping Shi, and Jiaya Jia. 2013. Abnormal event detection at 150 fps in matlab. In _ICCV_ . 

- [31] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. 2023. Video-chatgpt: Towards detailed video understanding via large vision and language models. _arXiv preprint arXiv:2306.05424_ (2023). 

- [32] Rashmiranjan Nayak, Umesh Chandra Pati, and Santos Kumar Das. 2021. A comprehensive review on deep learning-based methods for video anomaly detection. _Image and Vision Computing_ (2021). 

- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ . PmLR, 8748–8763. 

- [34] Bharathkumar Ramachandra, Michael J Jones, and Ranga Raju Vatsavai. 2020. A survey of single-scene video anomaly detection. _IEEE transactions on pattern analysis and machine intelligence_ 44, 5 (2020), 2293–2312. 

- [35] Fabio Remondino, Ali Karami, Ziyang Yan, Gabriele Mazzacca, Simone Rigon, and Rongjun Qin. 2023. A critical analysis of NeRF-based 3D reconstruction. _Remote Sensing_ 15, 14 (2023), 3585. 

- [36] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. 2008. The graph neural network model. _IEEE transactions on neural networks_ 20, 1 (2008), 61–80. 

- [37] Yihua Shao, Siyu Liang, Zijian Ling, Minxi Yan, Haiyang Liu, Siyu Chen, Ziyang Yan, Chenyu Zhang, Haotong Qin, Michele Magno, et al. 2024. GWQ: Gradient-Aware Weight Quantization for Large Language Models. _arXiv preprint arXiv:2411.00850_ (2024). 

- [38] Yihua Shao, Deyang Lin, Fanhu Zeng, Minxi Yan, Muyang Zhang, Siyu Chen, Yuxuan Fan, Ziyang Yan, Haozhe Wang, Jingcai Guo, et al. 2025. TR-DQ: TimeRotation Diffusion Quantization. _arXiv preprint arXiv:2503.06564_ (2025). 

- [39] Yihua Shao, Yeling Xu, Xinwei Long, Siyu Chen, Ziyang Yan, Yang Yang, Haoting Liu, Yan Wang, Hao Tang, and Zhen Lei. 2024. AccidentBlip: Agent of Accident Warning based on MA-former. _arXiv preprint arXiv:2404.12149_ (2024). 

- [40] Yihua Shao, Minxi Yan, Yang Liu, Siyu Chen, Wenjie Chen, Xinwei Long, Ziyang Yan, Lei Li, Chenyu Zhang, Nicu Sebe, et al. 2025. In-Context Meta LoRA Generation. _arXiv preprint arXiv:2501.17635_ (2025). 

- [41] Fahad Sohrab, Jenni Raitoharju, Moncef Gabbouj, and Alexandros Iosifidis. 2018. Subspace support vector data description. In _ICPR_ . 

- [42] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. 2023. Pandagpt: One model to instruction-follow them all. _arXiv preprint arXiv:2305.16355_ (2023). 

- [43] Waqas Sultani, Chen Chen, and Mubarak Shah. 2018. Real-world anomaly detection in surveillance videos. In _CVPR_ . 

- [44] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. 2023. Eva-clip: Improved training techniques for clip at scale. _arXiv preprint arXiv:2303.15389_ (2023). 

- [45] Zachary Teed and Jia Deng. 2020. Raft: Recurrent all-pairs field transforms for optical flow. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16_ . Springer, 402–419. 

- [46] Kamalakar Vijay Thakare, Debi Prosad Dogra, Heeseung Choi, Haksub Kim, and Ig-Jae Kim. 2023. RareAnom: A Benchmark Video Dataset for Rare Type Anomalies. _Pattern Recognition_ (2023). 

- [47] Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. 2023. DyAnNet: A Scene Dynamicity Guided Self-Trained Video Anomaly Detection Network. In _WACV_ . 

2594 

MM ’25, October 27–31, 2025, Dublin, Ireland 

Yihua Shao et al. 

- [48] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. 2021. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In _ICCV_ . 

- [49] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. _arXiv_ (2023). 

- [50] Anil Osman Tur, Nicola Dall’Asen, Cigdem Beyan, and Elisa Ricci. 2023. Exploring diffusion models for unsupervised video anomaly detection. In _ICIP_ . 

- [51] Anil Osman Tur, Nicola Dall’Asen, Cigdem Beyan, and Elisa Ricci. 2023. Unsupervised Video Anomaly Detection with Diffusion Models Conditioned on Compact Motion Representations. In _ICIAP_ . 

- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _NeurIPS_ (2017). 

- [53] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In _International Conference on Learning Representations_ . 

- [54] Jue Wang and Anoop Cherian. 2019. Gods: Generalized one-class discriminative subspaces for anomaly detection. In _ICCV_ . 

- [55] Nan Wang, Yuantao Chen, Lixing Xiao, Weiqing Xiao, Bohan Li, Zhaoxi Chen, Chongjie Ye, Shaocong Xu, Saining Zhang, Ziyang Yan, et al. 2025. Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting. _arXiv preprint arXiv:2506.05280_ (2025). 

- [56] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. 2024. Cogvlm: Visual expert for pretrained language models. _Advances in Neural Information Processing Systems_ 37 (2024), 121475–121499. 

- [57] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. 2022. Self-supervised Sparse Representation for Video Anomaly Detection. In _ECCV_ . 

- [58] Peng Wu and Jing Liu. 2021. Learning causal temporal relation and feature discrimination for anomaly detection. _IEEE TIP_ (2021). 

- [59] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. 2020. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In _ECCV_ . 

- [60] Ziyang Yan, Wenzhen Dong, Yihua Shao, Yuhang Lu, Liu Haiyang, Jingwen Liu, Haozhe Wang, Zhe Wang, Yan Wang, Fabio Remondino, et al. 2024. Renderworld: World model with self-supervised 3d label. _arXiv preprint arXiv:2409.11356_ (2024). 

- [61] Ziyang Yan, Lei Li, Yihua Shao, Siyu Chen, Zongkai Wu, Jenq-Neng Hwang, Hao Zhao, and Fabio Remondino. 2024. 3dsceneeditor: Controllable 3d scene editing with gaussian splatting. _arXiv preprint arXiv:2412.01583_ (2024). 

- [62] Ziyang Yan, Gabriele Mazzacca, Simone Rigon, Elisa Mariarosaria Farella, Pawel Trybala, Fabio Remondino, et al. 2023. NeRFBK: a holistic dataset for benchmarking NeRF-based 3D reconstruction. _International Archives of the Photogrammetry,_ 

_Remote Sensing and Spatial Information Sciences_ 48, 1 (2023), 219–226. 

- [63] Ziyang Yan, Nazanin Padkan, Paweł Trybała, Elisa Mariarosaria Farella, and Fabio Remondino. 2025. Learning-Based 3D Reconstruction Methods for NonCollaborative Surfaces—A Metrological Evaluation. _Metrology_ 5, 2 (2025), 20. 

- [64] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. 2024. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In _Proceedings of the ieee/cvf conference on computer vision and pattern recognition_ . 13040–13051. 

- [65] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. 2020. Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection. In _ECCV_ . 

- [66] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. 2022. Generative cooperative learning for unsupervised video anomaly detection. In _CVPR_ . 

- [67] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. 2024. Harnessing large language models for training-free video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 18527–18536. 

- [68] Fanhu Zeng, Zhen Cheng, Fei Zhu, Hongxin Wei, and Xu-Yao Zhang. 2025. LocalPrompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection. In _The Thirteenth International Conference on Learning Representations_ . 

- [69] Fanhu Zeng, Zhen Cheng, Fei Zhu, and Xu-Yao Zhang. 2025. Towards Efficient and General-Purpose Few-Shot Misclassification Detection for Vision-Language Models. _arXiv preprint arXiv:2503.20492_ (2025). 

- [70] Fanhu Zeng, Haiyang Guo, Fei Zhu, Li Shen, and Hao Tang. 2025. Parameter Efficient Merging for Multimodal Large Language Models with Complementary Parameter Adaptation. _arXiv preprint arXiv:2502.17159_ (2025). 

- [71] Fanhu Zeng, Hao Tang, Yihua Shao, Siyu Chen, Ling Shao, and Yan Wang. 2025. MambaIC: State Space Models for High-Performance Learned Image Compression. _arXiv preprint arXiv:2503.12461_ (2025). 

- [72] Fanhu Zeng and Deli Yu. 2024. M2M-TAG: Training-Free Many-to-Many Token Aggregation for Vision Transformer Acceleration. In _Workshop on Machine Learning and Compression, NeurIPS 2024_ . 

- [73] Fanhu Zeng, Fei Zhu, Haiyang Guo, Xu-Yao Zhang, and Cheng-Lin Liu. 2024. Modalprompt: Dual-modality guided prompt for continual learning of large multimodal models. _arXiv preprint arXiv:2410.05849_ (2024). 

- [74] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-llama: An instructiontuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ (2023). 

- [75] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. 2023. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. _arXiv preprint arXiv:2310.01852_ (2023). 

2595 

