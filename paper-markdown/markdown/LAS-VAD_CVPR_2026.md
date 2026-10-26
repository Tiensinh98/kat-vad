This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning** 

Yu Wang<sup>*</sup> Shengjie Zhao 

School of Computer Science and Technology, Tongji University, Shanghai, China 

## **Abstract** 

_Weakly supervised video anomaly detection (WS-VAD) involves identifying the temporal intervals that contain anomalous events in untrimmed videos, where only videolevel annotations are provided as supervisory signals. However, a key limitation persists in WS-VAD, as dense framelevel annotations are absent, which often leaves existing methods struggling to learn anomaly semantics effectively. To address this issue, we propose a novel framework named LAS-VAD, short for_ **_L_** _earning_ **_A_** _nomaly_ **_S_** _emantics for WS-_ **_VAD_** _, which integrates anomaly-connected component mechanism and intention awareness mechanism. The former is designed to assign video frames into distinct semantic groups within a video, and frame segments within the same group are deemed to share identical semantic information. The latter leverages an intention-aware strategy to distinguish between similar normal and abnormal behaviors (e.g., taking items and stealing). To further model the semantic information of anomalies, as anomaly occurrence is accompanied by distinct characteristic attributes (i.e., explosions are characterized by flames and thick smoke), we additionally incorporate anomaly attribute information to guide accurate detection. Extensive experiments on two benchmark datasets, XD-Violence and UCF-Crime, demonstrate that our LAS-VAD outperforms current state-of-theart methods with remarkable gains._ 

## **1. Introduction** 

In recent years, video anomaly detection (VAD) has attracted growing attention owing to its broad practical applications, including intelligent surveillance [6, 46], multimedia content understanding [44, 47], and smart manufacturing [9]. VAD seeks to identify temporal intervals containing anomalous events in arbitrarily long, untrimmed videos. A key challenge for VAD, however, lies in its dependence on expensive, dense annotations that precisely delineate the start and end of each anomalous event. To alleviate this 

> *Corresponding author: Yu Wang (email:yuwangtj@yeah.net) 

reliance, weakly supervised VAD (WS-VAD) has been proposed. This paradigm enables training using only videolevel annotations, which in turn streamlines the annotation workflow and improves the practical feasibility of VAD in real-world scenarios. 

Most current WS-VAD approaches [5, 8, 22, 25, 35, 44, 45, 55, 57, 67, 70] follow a standardized pipeline. First, the process involves extracting frame-level features using pre-trained models [33]. Next, these extracted features are fed into classifiers while incorporating a multiple instance learning (MIL) strategy [29] to identify anomaly. Despite significant advancements, there are still two limitations. On the one hand, the lack of frame-level supervisory signals results in difficulty in capturing anomaly semantic information. On the other hand, the distinction between abnormal and normal behaviors is ambiguous. For instance, actions like taking items and stealing exhibit little difference, with the only discrepancy lying in the speed of grasping. 

To address the above limitations, we propose a novel framework named LAS-VAD, short for **L** earning **A** nomaly **S** emantics for WS- **VAD** . Specifically, to alleviate the problem of difficulty in capturing semantic information caused by the absence of frame-level supervisory signals, we develop a novel anomaly-connected component (ACC) approach. By calculating pairwise similarity between frames, ACC can divide frames into non-overlapping groups through the identification of connected components. Frame segments within the same group are deemed to share identical semantic information, which is utilized to guide frame-level learning. Besides, as anomaly occurrence is accompanied by distinct characteristic attributes, _i.e._ , explosions are characterized by flames and thick smoke, we further incorporate anomaly attribute information to guide accurate detection. 

To enhance the discriminability between normal and abnormal behaviors, we devise a novel intention-aware strategy. Specifically, normal and abnormal behaviors usually share similar appearance features, for example, taking items and stealing, but their intentions differ. To capture intentions, we first extract position features, velocity features and acceleration features, which are intended to better reason 

28545 

intentions. We then develop a novel intention prototype to store the semantic information of different intentions, along with a cross-intention contrastive learning mechanism to distinguish between distinct intentions. 

The main contributions of this work are five-fold: 1) We propose LAS-VAD, a novel framework for weakly supervised video anomaly detection with anomaly-connected components and intention reasoning mechanisms. 2) To alleviate insufficient semantic supervisory signals caused by missing frame-level labels, we propose a novel anomalyconnected component approach to cluster video frames into distinct semantic groups, where frames in the same group share identical semantics. 3) To enhance the discriminability between normal and abnormal behaviors, we propose a novel intention awareness mechanism, utilizing intention prototypes and cross-intention contrastive learning, to distinguish between distinct intentions. 4) Since anomaly occurrence is accompanied by distinct characteristic attributes, we propose to incorporate anomaly attribute information to guide accurate detection. 5) Extensive evaluations on XD-Violence and UCF-Crime datasets have shown that our LAS-VAD achieves state-of-the-art performance. 

## **2. Related Work** 

### **2.1. Vision-Language Pre-training** 

Cross-modal vision-language comprehension [13, 14, 30, 48, 71] is a core task requiring accurate alignment of representations between the visual and linguistic modalities. Mainstream approaches fall into two categories, namely joint and dual encoder architectures. Joint encoder-based methods [18, 24, 66] leverage a multi-modal encoder to enable fine-grained interactions between vision and language. Nevertheless, despite their powerful performance, these methods have a key limitation, _i.e._ , they require processing each text-image pair separately during inference, resulting in substantial inefficiencies. In contrast, dual encoder-based approaches [33, 44, 46] adopt two separate encoders to extract visual and linguistic features. Recent years have seen significant progress driven by large-scale contrastive pretraining under this paradigm—progress that has greatly improved the performance of various multi-modal tasks, such as text-image retrieval [15, 50], visual question answering [21, 38], and video grounding [20, 58]. In this work, we adopt CLIP [33] and transfer it to the task of weakly supervised video anomaly detection. 

### **2.2. Weakly Supervised Video Anomaly Detection** 

Weakly supervised video anomaly detection (WS-VAD) has garnered significant attention in recent years, due to its broad applicability [9, 44, 46, 47, 73] and affordable manpower costs. Sultani _et al._ [39], recognized as pioneers in this field, have compiled a large-scale video anomaly de- 

tection dataset with video-level annotations, and they have utilized a multiple instance ranking strategy for the localization of anomalous events. Subsequent efforts have focused on multiple aspects of improving WS-VAD performance. One research direction [11, 19, 40, 68–70] has explored capturing temporal relationships among video segments, achieved via graph structures [68], self-attention strategies [40, 49, 70], and transformers [11, 19, 26]. These methods offer a deeper understanding of how distinct video parts interact, a factor crucial for accurate anomaly detection. The other research direction has been to explore selftraining schemes [8, 12, 34, 59, 64] that generate snippetlevel pseudo-labels, which are then utilized to iteratively refine anomaly scores. This process aids in enhancing anomaly detection accuracy by offering more granular feedback. Additionally, Sapkota _et al._ [36] and Zaheer _et al._ [63] have sought solutions to mitigate the impact of false positives. 

Recently, research efforts to utilize multi-modal knowledge for enhancing WS-VAD performance have surged. Notable examples include PEL4VAD [32], VadCLIP [57], and ITC [22], which inject text clues corresponding to anomaly-event categories into WS-VAD, whereas PEMIL [2] and MACILSD [60] introduce audio features to strengthen the capability of WS-VAD. Such innovations have delivered promising outcomes. Wu _et al._ [54] have taken a step further to develop a framework enabling the retrieval of video anomalies using language queries or synchronous audio. Wang _et al._ [44] first propose to learn event completeness with Gaussian priors for WS-VAD. _π_ -VAD [27] leverages five additional modalities to augment RGB features of images. 

Besides, pre-trained vision-language models have also emerged as a powerful tool to extract robust representations [7, 57, 59, 72]. For instance, Wu _et al._ propose an open-vocabulary video anomaly detection paradigm [55], designed to identify both known and novel anomalies in open-world scenarios. Huang _et al._ [12] further devise a multi-scale evidential vision-language model to achieve open-world video anomaly detection. Yang _et al._ [59] employ pre-trained CLIP for generating reliable pseudo-labels. Lv _et al._ [26] have designed an unbiased multiple-instance strategy for learning invariant representations. Wang _el al._ [42] design a prompt generator driven by both global and local contexts, which achieves global generalization and local personalization. STPrompt [56] acquires spatio-temporal prompt embeddings by leveraging pre-trained vision-language models. 

Despite significant progress, previous methods struggle to learn anomaly semantics due to the absence of framelevel annotations. In contrast, we propose LAS-VAD, a novel framework for WS-VAD with anomaly-connected components and intention reasoning mechanisms. 

28546 



<!-- Start of picture text -->
Abnormal  Connected<br>T Similarity Matrix Adjacency Matrix Components<br>Visual<br>Correlation<br>T …<br>T T<br>C<br>Cross-modal  T T<br>Correlation<br>T<br>Video Features Coarse-grained   𝓚<br>Local Classifier<br>Transformer Top-K<br>Global GCN Fine-ground   𝓜<br>Classifier<br>MIL<br>T<br>Intention-aware<br>Features Intention Prototype<br>Position Features<br>T T<br>Acceleration Features Confidence<br>Velocity Features Cross-Intention<br>Contrastive Learning<br>T<br>Frame-wise Logits<br>[Anomaly Category]  normal<br>[Attributes] regular surroundings,<br>no chaotic elements, peaceful<br>scene<br>[Anomaly Category]  explosion<br>[Attributes] flames, thick smoke, T<br>flying debris, damaged structures,<br>bright flash at blast site<br>Module<br>Correlation Refinement Abnormal Grouping<br>…<br>Encoder<br>CLIP Image<br>Visual Module<br>…<br>…<br>… …<br>Module<br>ReLU Intention Awareness<br>FC Layer FC Layer<br>…<br>…<br>CLIP Text Encoder Cross-Modal  Enhancement Module<br>Cross-modal<br>…<br>…<br>…<br><!-- End of picture text -->

Figure 1. Overview of the proposed LAS-VAD. We integrate an intention awareness mechanism and anomaly-connected component mechanism into WS-VAD to learn object intentions and more discriminative semantics. Additionally, attribute information of anomaly categories is introduced to facilitate accurate anomaly detection. 

## **3. Methodology** 

### **3.1. Overview** 

**Problem Definition.** Consider a dataset of _n_ videos _V_ = _{vi}_<sup>_n_</sup> _i_ =1<sup>with different annotation granularities.Each video</sup> _vi_ has both coarse-grained and fine-grained video-level labels, with _yi ∈{_ 0 _,_ 1 _}_ and _gi ∈{_ 0 _, ..., C}_ corresponding to these two types respectively, and _C_ here denotes the number of anomaly categories. If _vi_ is considered normal, then _yi_ = 0 and _gi_ = 0, otherwise _yi_ = 1 and _gi_ is assigned a specific category. For coarse-grained WS-VAD, the task is to estimate the likelihood of an anomaly being present in each snippet. Fine-grained WS-VAD, however, calls for predicting specific anomalous event instances that are denoted as _{sj, ej, aj, uj}_ . Here, _sj_ and _ej_ mark the start and end timestamps of the _j_ -th anomaly instance, _aj_ stands for its corresponding anomaly category, and _uj_ indicates the prediction confidence. 

**Framework.** This paper proposes an innovative framework, LAS-VAD, for WS-VAD, as illustrated in the Fig. 1. The core idea is to strengthen the semantic awareness capability. To this end, we develop a novel anomaly-connected component approach for partitioning video frames into distinct semantic groups, where frames within the same group hold consistent semantics. This allows us to guide the frame-level learning using the semantics of the corresponding group. To better capture anomaly semantics, we note that anomalies are accompanied by distinct characteristic attributes, _e.g._ , flames and thick smoke define explosions, and thus integrate anomaly attribute features to enhance detection accuracy. In addition, to boost the discriminability between normal and abnormal behaviors, we devise a novel intention awareness mechanism that employs intention prototypes and cross-intention contrastive learning to distinguish distinct intentions. In detail, we adopt a pretrained image encoder of CLIP to extract visual features 

28547 

_Xvideo ∈R_<sup>_T ×D_</sup> , in which _T_ is the total number of frames and _D_ denotes the dimension of features. To encode temporal dependencies, frame-level features are partitioned into overlapping windows of uniform length. A local transformer layer with constrained attention ranges [23] is subsequently applied, thereby precluding information exchange across these windows. The enhanced feature _Xh ∈R_<sup>_T ×D_</sup> is generated through this procedure. Next, to further model global temporal dependencies, we deploy a GCN module on _Xh_ . Specifically, in line with the configuration outlined in [57], the global temporal correlations among features are modeled according to their similarity. This process can be summarized as follows: 



where _A_ refers to the adjacency matrix, _W_ denotes the learnable weight, and _Xf_ corresponds to the enhanced video feature. The Softmax operation is employed to guarantee that the sum of elements in each row equals unity. Subsequently, a pre-trained text encoder is employed to extract linguistic features for all anomaly event categories, denoted as _Xlang ∈R_<sup>(</sup><sup>_C_+1)</sup><sup>_×D_</sup> . To better capture the semantic information of anomalies, we note that anomalies exhibit distinct attributes ( _e.g._ , explosions are marked by flames and thick smoke) and further integrate anomaly attribute information to enhance detection accuracy. To further model the semantic information of anomalies, as anomaly occurrence is accompanied by distinct characteristic attributes ( _i.e._ , explosions are characterized by flames and thick smoke), we additionally incorporate anomaly at- 



_{_ 0 _, ..., C}_ . Subsequently, we employ the pre-trained text encoder to extract features from these descriptions, which are denoted as _Xaux ∈R_<sup>(</sup><sup>_C_+1)</sup><sup>_×D_</sup> . We concatenate _Xlang_ and _Xaux_ to acquire the final textual features _Xtext_ for each anomaly category. We further compute the cross-modal cosine similarity between _Xtext_ and _Xf_ to obtain frame-level fine-grained classification scores _q_<sup>_l_</sup> _∈R_<sup>_T ×_(</sup><sup>_C_+1)</sup> . 

Meanwhile, we apply a fully-connected layer to _Xf_ , followed sequentially by a Sigmoid operation for binary classification in category-agnostic anomaly detection, and a Softmax operation for multi-class classification in categoryaware anomaly detection, generating frame-level predictions _q_<sup>_b_</sup> _∈R_<sup>_T ×_2</sup> and _q_<sup>_m_</sup> _∈R_<sup>_T ×_(</sup><sup>_C_+1)</sup> , respectively. 

For the _c_ -th category in _q_<sup>_b_</sup> , we calculate the average of the top- _K_ scores along the temporal dimension, as follows: 



where _U_ is the index set of top- _K_ frames, and _p_<sup>_b_</sup> _c_<sup>indicates</sup> the category-agnostic prediction. Then, a binary crossentropy loss is applied between _p_<sup>_b_</sup> and the ground truth _y_ : 



Although _q_<sup>_m_</sup> encodes category-aware anomaly clues, it suffers from insufficient discriminative ability and limited intention awareness due to the lack of frame-wise annotations. To this end, we propose an intention awareness module (IAM) as in Section 3.2 and a novel anomalyconnected component module (ACC) as in Section 3.3. In general, to enhance the discriminability between normal and abnormal behaviors, IAM attempts to establish intention prototypes and adopt a cross-intention contrastive learning mechanism to distinguish between distinct intentions, yielding a contrastive loss _Lcst_ and frame-level predictions _q_<sup>_a_</sup> _∈R_<sup>_T ×_(</sup><sup>_C_+1)</sup> . Finally, we define frame-level fine-grained classification logits as _p_<sup>_f_</sup> = 3<sup><u>1</u>(</sup><sup>_qm_+</sup><sup>_qa_+</sup><sup>_ql_).</sup> _p_<sup>_f_</sup> is then integrated into video-level scores _p_<sup>_v_</sup> via a MIL mechanism. Specifically, for the _c_ -th category, the _p_<sup>_v_</sup> _c_<sup>is</sup> computed as follows: 



Next, a cross-entropy constraint is employed to supervise fine-grained anomaly classification, as follows: 



Meanwhile, ACC module partitions video frames into distinct semantic groups, where frames within the same group are deemed to share identical semantics, yielding frame-level pseudo labels _g ∈R_<sup>_T ×_(</sup><sup>_C_+1)</sup> . These pseudo labels are further used to aid the learning of _q_<sup>_m_</sup> by: 



Therefore, the base optimization loss of LAS-VAD is: 



Besides, to ensure the consistency between categoryagnostic and category-aware predictions, _i.e._ , _p_<sup>_b_</sup> [ _t,_ 1] and 1 _− p_<sup>_f_</sup> [ _t,_ 0], _ℓ_ 1 regularization is introduced, formulated as: 



Overall, the optimization loss of LAS-VAD is: 



where _λ_ is used to balance different terms. 

28548 

### **3.2. Intention Awareness Mechanism** 

To enhance the discriminability between normal and abnormal behaviors, we propose a novel intention-aware strategy in this section. Specifically, normal and abnormal behaviors frequently share similar appearance features. For instance, the behaviors of “taking objects” and “stealing” exhibit similar appearance but distinct intentions, a difference reflected in their object-taking speed, _i.e._ , the speed in theft cases is significantly faster than that in normal situations. For this purpose, we seek to reason about the behavioral intention to better capture anomalies. 

Based on the video feature _Xf_ , we first generate position features _Xp ∈R_<sup>_T ×_(</sup><sup>_D/_3)</sup> , velocity features _Xv ∈ R_<sup>_T ×_(</sup><sup>_D/_3)</sup> using the difference between adjacent positions, and acceleration features _Xa ∈R_<sup>_T ×_(</sup><sup>_D/_3)</sup> using the difference between adjacent velocities. The implementation of this procedure is detailed below: 



where _fc_ denotes the fully-connected layer, _Conv_ is a convolution operation with a kernel size of 3, and zero elements are padded at the start and end positions of _Xv_ and _Xa_ to ensure that their temporal dimension remains _T_ . Then, we concatenate _Xp_ , _Xv_ , and _Xa_ along the feature dimension, forming intention-aware features _Xint ∈R_<sup>_T ×D_</sup> . 

Utilizing intention-aware features _Xint_ , we acquire frame-wise logits _qint ∈ R_<sup>_T ×C_</sup> through two fullyconnected layers with a ReLU activation. Meanwhile, we establish intention prototypes _Z ∈R_<sup>(</sup><sup>_C_+1)</sup><sup>_×D_</sup> to retain semantic prototypes of different categories. On the one hand, for the label corresponding to the dimension where the maximum value lies at each time-step in _qint_ , we calculate the cosine similarity between the prototype of this label from _Z_ and its feature from _Xint_ , yielding the confidence _wint ∈ R_<sup>_T ×_1</sup> for the predicted logits _qint_ . We then obtain the weighted frame-wise scores _qa_ = _Softmax_ ( _qint × wint_ ). On the other hand, we also update the intention prototypes in a momentum manner. In detail, for the anomaly category _c_ , the corresponding features with classification scores in _qa_ greater than a threshold _α_ are selected, and their mean value _Xcenter_<sup>_c_is used to update the corresponding class prototype</sup> as _Z_ [ _c,_ :] = (1 _−β_ ) _×Z_ [ _c,_ :]+ _β ×Xcenter_<sup>_c_, where</sup><sup>_β_denotes</sup> the momentum coefficient. 

Furthermore, we propose a cross-intention contrastive learning strategy aimed at explicitly mitigating the issue of intention confusion. The core principle resides in mining hard-to-distinguish frame pairs with different intentions and easily confusable frame pairs with the same intentions. 

Specifically, for each frame feature _Xint_<sup>_t_, we select frames</sup> as positive samples _Spos_<sup>_t_withthelowestsimilarityamong</sup> samples of the same category, and as negative samples _Sneg_<sup>_t_</sup> with the top- _M_ highest similarity among different intentions. Subsequently, the infoNCE loss is adopted to constrain the intention distribution: 



### **3.3. Anomaly-Connected Component Mechanism** 

To alleviate insufficient semantic supervisory signals arising from the lack of frame-level annotations, we propose a novel anomaly-connected component approach to partition video frames into distinct semantic groups, with frames within the same group sharing consistent semantics. Specifically, given video features _Xf ∈R_<sup>_T ×D_</sup> , we compute the pairwise similarity between each frame: 



However, _Av_ only captures the visual similarity of frames, which tends to be biased. To address this issue, we further leverage cross-modal similarity _q_<sup>_l_</sup> _∈R_<sup>_T ×_(</sup><sup>_C_+1)</sup> to rectify such bias. The core principle is that for any _i_ -th and _j_ -th frames, if both exhibit high semantic similarity to the same text category, their similarity is enhanced; otherwise, it is weakened. This process is formulated as: 



where _η_ denotes the degree of rectification, max _c_ min( _q_<sup>_l_</sup> [ _i, c_ ] _, q_<sup>_l_</sup> [ _j, c_ ]) measures the textual semantic consistency between the two frames, and _A_<sup>ˆ</sup> is the enhanced similarity map. We then convert it into a binary map _A_<sup>ˆ</sup> based on a threshold _τ_ : _A_ = ( _A_<sup>ˆ</sup> _> τ_ ) _∈R_<sup>_T ×T_</sup> _._ Here, each frame is regarded as a vertex, and _A_ is treated as an adjacency matrix that characterizes inter-vertex connectivity. On this basis, we construct a graph, and the problem of identifying frames with consistent semantics is converted into the task of detecting all connected components in the graph via the adjacency matrix _A_ . To this end, we employ the Depth-First Search algorithm to obtain _r_ distinct connected components _B_ 1, _B_ 2,..., and _Br_ from _A_ . 

For any _Bi_ , which contains _u_ video frames, we get the feature prototype _Xac_<sup>_i_by averaging their corresponding fea-</sup> tures within _Xf_ , and semantic category _yproto_<sup>_i_by averaging</sup> their logits within _p_<sup>_f_</sup> . Subsequently, we compute the similarity between each frame’s feature and all connected components’ feature prototypes, and assign the semantic category of the most similar one as the frame’s category label, yielding frame-level pseudo-labels _g ∈R_<sup>_T ×_(</sup><sup>_C_+1)</sup> . 

28549 

|Supervision|Modality|Methods|Feature|AP(%)|
|---|---|---|---|---|
|Unsupervised|RGB+Audio|LTR [10]|I3D+VGGish|30.77|
|||CTRFD [51]<br>|I3D+VGGish<br>|75.90<br>|
|||WS-AVVD [53]|I3D+VGGish|78.64|
||RGB+Audio|ECU [64]|I3D+VGGish|81.43|
|||DMU [70]|I3D+VGGish|81.77|
|||MACIL-SD [60]|I3D+VGGish|83.40|
|||PE-MIL[2]|I3D+VGGish|88.21|
|||RAD [39]|C3D|73.20|
|||RTFML [40]|I3D|77.81|
|||ST-MSL [19]|I3D|78.28|
|||LA-Net [31]<br>|I3D<br>|80.72<br>|
|||CoMo [4]|I3D|81.30|
|Weakly||DMU [70]<br>|I3D<br>|82.41<br>|
|Suervised||PEL4VAD [32]|I3D|85.59|
|p||PE-MIL [2]|I3D|88.05|
|||_π_-VAD [27]|I3D|85.37|
||RGB|LEC-VAD [44]|I3D|88.47|
|||LAS-VAD(Ours)|I3D|**89.96**|
|||OVVAD [55]|CLIP|66.53|
|||CLIP-TSA [16]|CLIP|82.19|
|||IFS-VAD [69]|CLIP|83.14|
|||TPWNG [59]|CLIP|83.68|
|||VadCLIP [57]|CLIP|84.51|
|||ITC [22]|CLIP|85.45|
|||ReFLIP [7]|CLIP|85.81|
|||LEC-VAD [44]|CLIP|86.56|
|||LAS-VAD (Ours)|CLIP|**87.92**|



Table 1. Coarse-grained comparisons on XD-Violence. 

### **3.4. Inference** 

During inference, coarse-grained WS-VAD calculates the average value of 1 _− p_<sup>_f_</sup> [ _t,_ 0] and _p_<sup>_b_</sup> [ _t,_ 1] as the anomaly confidence for the _t_ -th frame. For fine-grained WS-VAD, a two-step thresholding strategy is used to generate anomaly instances. First, anomaly categories with video-level activations exceeding the predefined threshold _θv_ are retained. Then, for each retained category, snippets with fine-grained matching scores in _p_<sup>_f_</sup> above _θs_ are selected as candidates. These temporally consecutive candidates are merged into anomaly instances. Following AutoLoc [37], each instance’s outer-inner score derived from _p_<sup>_f_</sup> serves as its confidence score, based on which non-maximal suppression (NMS) is applied to eliminate redundant proposals. 

## **4. Experiments** 

### **4.1. Experimental Settings** 

**Datasets. UCF-Crime** [39] comprises 1900 untrimmed videos, which span 13 distinct categories of anomalous events. These videos are derived from surveillance recordings captured in a range of indoor environments and streetbased scenarios. For experimental consistency, we adopt a standard data split: the training set includes 1610 videos, and the testing set contains 290 videos. As a larger-scale counterpart, **XD-Violence** [52], consists of 4754 untrimmed videos sourced from movies and YouTube, covering 6 categories of violent events. Of these, 3954 videos are utilized for training, and 800 videos for testing. **Evaluation Protocol.** We assess the algorithm perfor- 

We assess the algorithm perfor- 

|Supervision|Methods|Feature|AUC(%)|
|---|---|---|---|
||LTR [10]|I3D+VGGish|50.60|
|Unsupervised|GODS [43]|I3D|70.46|
||GCL [62]|I3D|71.04|
||TCN-CIBL [65]|C3D|78.66|
||GCN-Anomaly [68]|C3D|81.08|
||GLAWS [61]|C3D|83.03|
||LEC-VAD [44]|C3D|84.75|
||LAS-VAD(Ours)|C3D|**86.04**|
||CTRFD [51]|I3D|84.89|
||LA-Net [31]|I3D|85.12|
||ST-MSL [19]|I3D|85.30|
||IFS-VAD [69]|I3D|85.47|
||NG-MIL [28]|I3D|85.63|
||CLAV [5]|I3D|86.10|
||ECU [64]|I3D|86.22|
||DMU [70]|I3D|86.75|
|Weakl|PE-MIL [2]|I3D|86.83|
|y<br>Sid|MGFN [3]|I3D|86.98|
|upervse|_π_-VAD [27]|I3D|90.33|
||LEC-VAD [44]|I3D|88.21|
||LAS-VAD(Ours)|I3D|**91.05**|
||Ju_et al._ [17]|CLIP|84.72|
||OVVAD [55]|CLIP|86.40|
||IFS-VAD [69]|CLIP|86.57|
||UMIL [26]|CLIP|86.75|
||CLIP-TSA [16]|CLIP|87.58|
||TPWNG [59]|CLIP|87.79|
||VadCLIP [57]|CLIP|88.02|
||STPrompt [56]|CLIP|88.08|
||ReFLIP [7]|CLIP|88.57|
||ITC [22]|CLIP|89.04|
||LEC-VAD [44]|CLIP|89.97|
||LAS-VAD (Ours)|CLIP|**90.86**|



#### Table 2. Coarse-grained comparisons on UCF-Crime. 

|Methods|||mAP|@IoU|||
|---|---|---|---|---|---|---|
||0.1|0.2|0.3|0.4|0.5|AVG|
|RAD [39]|22.72|15.57|9.98|6.20|3.78|11.65|
|AVVD [53]|30.51|25.75|20.18|14.83|9.79|20.21|
|VadCLIP [57]|37.03|30.84|23.38|17.90|14.31|24.70|
|ITC [22]|40.83|32.80|25.42|19.65|15.47|26.83|
|ReFLIP [7]|39.24|33.45|27.71|20.86|17.22|27.36|
|LEC-VAD [44]|48.78|40.94|34.28|28.02|23.68|35.14|
|LAS-VAD (Ours)|**50.18**|**44.09**|**36.00**|**29.51**|**24.65**|**36.89**|



Table 3. Fine-grained comparisons on XD-Violence. 

mance across both coarse-grained and fine-grained metrics. Specifically, for coarse-grained WS-VAD, we employ frame-level Average Precision (AP) as the metric for XD-Violence, and frame-level AUC for UCF-Crime. For fine-grained WS-VAD, consistent with prior works [22, 57], we compute the mean Average Precision (mAP) over IoU thresholds ranging from 0.1 to 0.5 with an increment of 0.1. Besides, the average of these mAP values (denoted as AVG) is reported to provide a more comprehensive assessment. **Implementation Details.** We employ the pre-trained text encoder of CLIP with ViT-B/16, while multiple vision en- 

28550 

|Methods|||mAP@|IoU|||
|---|---|---|---|---|---|---|
||0.1|0.2|0.3|0.4|0.5|AVG|
|RAD [39]|5.73|4.41|2.69|1.93|1.44|3.24|
|AVVD [53]|10.27|7.01|6.25|3.42|3.29|6.05|
|VadCLIP [57]|11.72|7.83|6.40|4.53|2.93|6.68|
|ITC [22]|13.54|9.24|7.45|5.46|3.79|7.90|
|ReFLIP [7]|14.23|10.34|9.32|7.54|6.81|9.62|
|LEC-VAD [44]|19.65|17.17|14.37|9.45|7.18|13.56|
|LAS-VAD (Ours)|**22.07**|**19.96**|**16.18**|**11.24**|**8.64**|**15.62**|



Table 4. Fine-grained comparisons on UCF-Crime. 

|ATT|ACC|IAM|||mAP|@IoU|||
|---|---|---|---|---|---|---|---|---|
||||0.1|0.2|0.3|0.4|0.5|AVG|
||||36.86|29.75|23.00|17.49|14.11|24.24|
||||39.15|32.36|25.47|19.02|16.50|26.50|
||||42.53|35.15|28.48|23.91|18.83|29.78|
||||42.91|35.64|28.46|23.86|19.05|29.98|
||||46.77|39.53|32.15|26.39|21.42|33.25|
||||46.65|39.28|32.06|26.51|21.39|33.18|
||||48.13|41.96|33.70|27.95|22.69|34.89|
||||**50.18**|**44.09**|**36.00**|**29.51**|**24.65**|**36.89**|



Table 5. Ablation Studies on XD-Violence. “IAM” denotes the intention awareness module, “ACC” denotes the anomaly-connected components, and “AAT” denotes anomaly attribute clues. 

coders, including I3D [1], C3D [41], and the CLIP visual encoder with ViT-B/16, are explored for frame feature extraction. GPT4 is adopted to extract anomaly attribute information. During training, the AdamW optimizer is used with a batch size of 64, a learning rate of 2e-5, and a total of 10 epochs. The thresholds _θv_ and _θs_ are set to 0.1 and 0.2. The value of _K_ is determined as _K_ = _max_ ( _⌊T/_ 16 _⌋,_ 1), the momentum coefficient _β_ is 0.1, and _τ_ is 0.9. 

### **4.2. Main Results** 

We conduct a thorough evaluation of anomaly detection performance via benchmarking against prevalent methods on two datasets. Experimental results demonstrate that our LAS-VAD achieves state-of-the-art performance across all evaluation metrics and granularity levels. 

First, we evaluate coarse-grained anomaly detection across the XD-Violence and UCF-Crime datasets, with corresponding results reported in Tables 1 and Table 2. For XD-Violence, I3D and CLIP features are adopted, and both of them achieve the best performance. In detail, our LASVAD achieves 89.96 AP with I3D features and 87.92 AP with CLIP features, yielding respective absolute gains of 1.49 and 1.36 over LEC-VAD. Notably, our method even outperforms PE-MIL, which uses additional audio features alongside I3D features. For UCF-Crime, we employ C3D, I3D, and CLIP features, all of which achieve remarkable performance. In detail, our LAS-VAD achieves 86.04 AP with C3D features, yielding an absolute gain of 1.29 over LEC-VAD. Similarly, LAS-VAD attains 91.05 AP and 90.96 AP with I3D and CLIP features respectively, outperforming the SOTA _π_ -VAD by 0.72 and LEC-VAD by 0.89 



<!-- Start of picture text -->
Model Performance Comparison<br>55<br>k-means<br>50.18 ACC<br>50<br>46.71<br>45 44.09<br>40 38.96<br>36.0 36.89<br>35 32.51 33.12<br>30 29.51<br>26.79<br>25 24.65<br>20.64<br>20<br>15<br>mAP@IoU 0.1 mAP@IoU 0.2 mAP@IoU 0.3 mAP@IoU 0.4 mAP@IoU 0.5 AVG<br>Value<br><!-- End of picture text -->

Figure 2. Ablation study on the utility of ACC. 

in absolute gain. These experimental results reveal that our model’s superiority when using diverse features. 

Moreover, our LAS-VAD delivers strong performance in fine-grained WS-VAD. As displayed in Table 3, it achieves substantial improvements across all metrics on XD-Violence, yielding an amazing AVG of 36.89, a 5% improvement over LEC-VAD. For the more challenging UCFCrime dataset, corresponding results are presented in Table 4. Our LAS-VAD maintains consistent and remarkable gains across all evaluation metrics, delivering a notable AVG of 15.62 and outperforming LEC-VAD by 15.2%. These results further highlight our algorithm’s strength in distinguishing subtle differences among diverse anomalies. 

### **4.3. Ablation Study** 

In this section, we perform comprehensive ablation studies on XD-Violence, systematically presenting more challenging fine-grained detection results to elucidate the contributions of individual factors to overall performance. 

**Analysis on model components.** We analyze the impact of our model’s components on detection performance, with results presented in Table 5. In detail, we investigate the utilities of the proposed intention awareness mechanism (IAM), anomaly-connected components (ACC), and anomaly attribute clues (AAT) to clarify their individual contributions to the model’s overall performance. From Table 5, it can be observed that integrating each component brings performance gains across all metrics, and the full model achieves the optimal performance over its castrated counterparts. **Analysis on ACC.** To further validate ACC’s utility, we adopt the k-means algorithm as an alternative to partition video frames into distinct groups. Notably, for the choice of _k_ in k-means, we distill the knowledge from ACC and set the number of components _r_ to _k_ . Results are presented in Fig. 2. Despite access to prior information of _k_ from ACC, k-means is observed to perform significantly worse than our ACC. We attribute this to the fact that ACC can model structural information among video frames via their adjacency matrix, whereas k-means only relies on feature similarity for partitioning. Besides, we investigate their ca- 

28551 



<!-- Start of picture text -->
GT GT<br>(a) Fighting (b) Explosion<br>w/o IAM<br>w/ IAM<br><!-- End of picture text -->

Figure 3. Fine-grained predictions w/ and w/o the proposed IAM. 

|LLM|||mAP|@IoU|||
|---|---|---|---|---|---|---|
||0.1|0.2|0.3|0.4|0.5|AVG|
|Lamma3|50.03|44.04|35.87|**29.53**|24.58|36.81|
|Qwen2|50.09|44.04|35.91|29.46|24.57|36.81|
|Gemini2.5|50.14|**44.11**|35.94|29.49|24.59|36.85|
|DeepSeek V3|50.15|44.08|35.97|29.47|24.61|36.86|
|GPT4|**50.18**|44.09|**36.00**|29.51|**24.65**|36.89|



Table 6. Different LLMs are tested on XD-Violence. 

pability in learning discriminative visual features, visualizing feature distributions from the original CLIP, the version with k-means, and our ACC-equipped model, and the results are presented in Fig. 4. Features from our model form tighter, more sharply bounded clusters per category, indicating its capacity to learn discriminative features even without frame-level supervision. 

Besides, we analyze the utility of rectification on the visual adjacent matrix _A_ . To validate this, we calculate recognition accuracy for normal and abnormal frames with and without the correction mechanism separately on test set. As shown in Fig. 4 (d), the correction mechanism improves performance for both normal and abnormal frames. 

**Analysis on IAM.** To investigate LAS-VAD’s ablity in recognizing anomalous intentions, we compare some examples that adopt the version with IAM and without IAM, and the results presented in Fig. 3. Compared with the model without the IAM module, the model integrated with the IAM module clearly perceives target intentions and critical events, such as rapid illegal physical contact (as shown in (a)) and rapid fire spread (as illustrated in (b)), thereby enabling more reasonable predictions. 

**Analysis on anomaly attribute generation.** To generate attribute information of anomaly categories, we prompt LLMs to produce corresponding attribute descriptions. We compare performance across different LLMs, with results shown in Table 6. Notably, different LLMs yield similar results, indicating that our model is insensitive to LLM types. GPT-4 is used as the default for attribute clue generation. **Analysis on hyper-parameters.** We conduct an extensive 



<!-- Start of picture text -->
normalfightingshootingriotabusecar accidentexplosion normalfightingshootingriotabusecar accidentexplosion<br>(a) CLIP (b) k-means<br>normalfightingshootingriotabusecar accidentexplosion 10095 Model Performance Comparison w/o rectificationw/ rectification91.54<br>90<br>86.35 86.62<br>85<br>81.03<br>80<br>75<br>70<br>65<br>60 normal abnormal<br>(c) ACC (d) Rectification mechanism<br>Figure 4. Visualization of representations for different categories<br>in (a), (b), and (c). (d) shows the utility of rectification mechanism.<br>38.0 38.0<br>37.5 37.5<br>37.0 37.0<br>36.5 36.5<br>36.0 36.0<br>35.5 35.5<br>35.00.01 0.05 0.1 0.3 0.5 1.0 35.00.1 0.3 0.5 0.7 0.9 1.0<br>(a)  λ (b)  η<br>Value<br>AVG AVG<br><!-- End of picture text -->

Figure 4. Visualization of representations for different categories in (a), (b), and (c). (d) shows the utility of rectification mechanism. 

Figure 5. Effects of hyper-parameters. 

exploration of the hyper-parameter configuration for _λ_ and _η_ , with results presented in Fig. 5. For _λ_ , we explore different values and get the optimal value at _λ_ = 0 _._ 3. For _η_ , we incrementally adjust it from 0.1 to 1, and get the optimal value at _η_ = 0 _._ 5, which indicates that appropriately rectifying the visual adjacency matrix _A_ via cross-modal information brings performance gains, while excessive rectification is counterproductive. 

## **5. Conclusion** 

This paper proposed a novel LAS-VAD for WS-VAD to effectively learn anomaly semantics. LAS-VAD integrated an anomaly-connected component mechanism and an intention awareness mechanism. The former assigned video frames into distinct semantic groups where intra-group frames share consistent semantics, while the latter used an intention-aware strategy to distinguish similar normal and abnormal behaviors. Besides, we incorporated anomaly attribute clues to guide more accurate detection. 

**Acknowledge.** This work was supported in part by the National Natural Science Foundation of China under Grant 62406226, in part sponsored by Shanghai Sailing Program under Grant 24YF2748700, in part by New-Generation Information Technology under the Shanghai Key Technology R&D Program under Grant 25511103500. 

28552 

## **References** 

- [1] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 6299–6308, 2017. 7 

- [2] Junxi Chen, Liang Li, Li Su, Zheng-jun Zha, and Qingming Huang. Prompt-enhanced multiple instance learning for weakly supervised video anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 18319–18329, 2024. 2, 6 

- [3] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. In _Proceedings of the AAAI conference on artificial intelligence_ , pages 387–395, 2023. 6 

- [4] MyeongAh Cho, Minjung Kim, Sangwon Hwang, Chaewon Park, Kyungjae Lee, and Sangyoun Lee. Look around for anomalies: Weakly-supervised anomaly detection via context-motion relational learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 12137–12146, 2023. 6 

- [5] MyeongAh Cho, Minjung Kim, Sangwon Hwang, Chaewon Park, Kyungjae Lee, and Sangyoun Lee. Look around for anomalies: Weakly-supervised anomaly detection via context-motion relational learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 12137–12146, 2023. 1, 6 

- [6] Khaled Dawoud, Zaigham Zaheer, Mustaqeem Khan, Karthik Nandakumar, Abdulmotaleb Elsaddik, and Muhammad Haris Khan. Fusedvision: A knowledge-infusing approach for practical anomaly detection in real-world surveillance videos. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 4036–4046, 2025. 1 

- [7] Prabhu Prasad Dev, Raju Hazari, and Pranesh Das. Reflipvad: Towards weakly supervised video anomaly detection via vision-language model. _IEEE Transactions on Circuits and Systems for Video Technology_ , 2024. 2, 6, 7 

- [8] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 14009–14018, 2021. 1, 2 

- [9] Huma Gupta, Sanjeev Sharma, and Sanjay Agrawal. Artificial intelligence-based anomalies detection scheme for identifying cyber threat on iot-based transport network. _IEEE Transactions on Consumer Electronics_ , 70(1):1716–1724, 2024. 1, 2 

- [10] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 733–742, 2016. 6 

- [11] Chao Huang, Chengliang Liu, Jie Wen, Lian Wu, Yong Xu, Qiuping Jiang, and Yaowei Wang. Weakly supervised video anomaly detection via self-guided temporal discriminative transformer. _IEEE Transactions on Cybernetics_ , 54 (5):3197–3210, 2022. 2 

- [12] Chao Huang, Weiliang Huang, Qiuping Jiang, Wei Wang, Jie Wen, and Bob Zhang. Multimodal evidential learning for open-world weakly-supervised video anomaly detection. _IEEE Transactions on Multimedia_ , 27:3132–3143, 2025. 2 

- [13] Jiangyong Huang, Baoxiong Jia, Yan Wang, Ziyu Zhu, Xiongkun Linghu, Qing Li, Song-Chun Zhu, and Siyuan Huang. Unveiling the mist over 3d vision-language understanding: Object-centric evaluation with chain-of-analysis. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 24570–24581, 2025. 2 

- [14] Yiren Jian, Chongyang Gao, and Soroush Vosoughi. Bootstrapping vision-language learning with decoupled language pre-training. In _Proceedings of the Neural Information Processing Systems_ , pages 57–72, 2024. 2 

- [15] Ding Jiang and Mang Ye. Cross-modal implicit relation reasoning and aligning for text-to-image person retrieval. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 2787–2797, 2023. 2 

- [16] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In _Proceedings of the IEEE International Conference on Image Processing_ , pages 3230–3234, 2023. 6 

- [17] Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, and Weidi Xie. Prompting visual-language models for efficient video understanding. In _Proceedings of the European Conference on Computer Vision_ , pages 105–124, 2022. 6 

- [18] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In _Proceedings of the Neural Information Processing Systems_ , pages 9694–9705, 2021. 2 

- [19] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 1395–1403, 2022. 2, 6 

- [20] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 2794– 2804, 2023. 2 

- [21] Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. Fine-grained late-interaction multi-modal retrieval for retrieval augmented visual question answering. In _Proceedings of the Neural Information Processing Systems_ , pages 22820–22840, 2023. 2 

- [22] Tianshan Liu, Kin-Man Lam, and Bing-Kun Bao. Injecting text clues for improving anomalous event detection from weakly labeled videos. _IEEE Transactions on Image Processing_ , 33:5907–5920, 2024. 1, 2, 6, 7 

- [23] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 10012–10022, 2021. 4 

28553 

- [24] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In _Proceedings of the Neural Information Processing Systems_ , pages 13–23, 2019. 2 

- [25] Weixin Luo, Wen Liu, Dongze Lian, and Shenghua Gao. Future frame prediction network for video anomaly detection. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 44(11):7505–7520, 2021. 1 

- [26] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 8022–8031, 2023. 2, 6 

- [27] Snehashis Majhi, Giacomo D’Amicantonio, Antitza Dantcheva, Quan Kong, Lorenzo Garattoni, Gianpiero Francesca, Egor Bondarev, and Franc¸ois Br´emond. Just dance with pi! a poly-modal inductor for weakly-supervised video anomaly detection. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 24265–24274, 2025. 2, 6 

- [28] Seongheon Park, Hanjae Kim, Minsu Kim, Dahye Kim, and Kwanghoon Sohn. Normality guided multiple instance learning for weakly supervised video anomaly detection. In _Proceedings of the IEEE Winter Conference on Applications of Computer Vision_ , pages 2665–2674, 2023. 6 

- [29] Sujoy Paul, Sourya Roy, and Amit K Roy-Chowdhury. W- talc: Weakly-supervised temporal activity localization and classification. In _Proceedings of the European Conference on Computer Vision_ , pages 563–579, 2018. 1 

- [30] Wujian Peng, Sicheng Xie, Zuyao You, Shiyi Lan, and Zuxuan Wu. Synthesize diagnose and optimize: Towards fine-grained vision-language understanding. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 13279–13288, 2024. 2 

- [31] Yujiang Pu and Xiaoyu Wu. Locality-aware attention network with discriminative dynamics learning for weakly supervised anomaly detection. In _Proceedings of the IEEE International Conference on Multimedia and Expo_ , pages 1–6, 2022. 6 

- [32] Yujiang Pu, Xiaoyu Wu, Lulu Yang, and Shengjin Wang. Learning prompt-enhanced context features for weaklysupervised video anomaly detection. _IEEE Transactions on Image Processing_ , 33:4923–4936, 2024. 2, 6 

- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _Proceedings of the International Conference on Machine Learning_ , pages 8748–8763, 2021. 1, 2 

- [34] Ayush K Rai, Tarun Krishna, Feiyan Hu, Alexandru Drimbarean, Kevin McGuinness, Alan F Smeaton, and Noel E O’connor. Video anomaly detection via spatio-temporal pseudo-anomaly generation: A unified approach. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 3887–3899, 2024. 2 

- [35] Bharathkumar Ramachandra, Michael J Jones, and Ranga Raju Vatsavai. A survey of single-scene video 

   - anomaly detection. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 44(5):2293–2312, 2020. 1 

- [36] Hitesh Sapkota and Qi Yu. Bayesian nonparametric submodular video partition for robust anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 3212–3221, 2022. 2 

- [37] Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, and Shih-Fu Chang. Autoloc: Weakly-supervised temporal action localization in untrimmed videos. In _Proceedings of the European Conference on Computer Vision_ , pages 154– 171, 2018. 6 

- [38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In _Proceedings of the European Conference on Computer Vision_ , pages 256–274, 2025. 2 

- [39] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 6479–6488, 2018. 2, 6, 7 

- [40] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 4975–4986, 2021. 2, 6 

- [41] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 4489–4497, 2015. 7 

- [42] Benfeng Wang, Chao Huang, Jie Wen, Wei Wang, Yabo Liu, and Yong Xu. Federated weakly supervised video anomaly detection with multimodal prompt. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 21017– 21025, 2025. 2 

- [43] Jue Wang and Anoop Cherian. Gods: Generalized one-class discriminative subspaces for anomaly detection. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 8201–8211, 2019. 6 

- [44] Yu Wang and Shiwei Chen. Learning event completeness for weakly supervised video anomaly detection. _arXiv preprint arXiv:2506.13095_ , 2025. 1, 2, 6, 7 

- [45] Yu Wang, Yadong Li, and Hongbin Wang. Two-stream networks for weakly-supervised temporal action localization with semantic-aware mechanisms. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 18878–18887, 2023. 1 

- [46] Yu Wang, Shengjie Zhao, and Shiwei Chen. Action-semantic consistent knowledge for weakly-supervised action localization. _IEEE Transactions on Multimedia_ , 26:10279–10289, 2024. 1, 2 

- [47] Yu Wang, Shengjie Zhao, and Shiwei Chen. Sql-net: Semantic query learning for point-supervised temporal action localization. _IEEE Transactions on Multimedia_ , 27:84–94, 2025. 1, 2 

28554 

- [48] Yu Wang, Fengxia Han, and Jianyu Wang. Boosting noisy correspondence discrimination via dynamic neighborhood semantic verification. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 26706–26714, 2026. 2 

- [49] Yu Wang, Shengjie Zhao, Jianyu Wang, and Xutao Chu. Learning local semantic signals and inter-class discrepancy for weakly supervised video anomaly detection. _IEEE Transactions on Multimedia_ , 2026. 2 

- [50] Zheng Wang, Zhenwei Gao, Kangshuai Guo, Yang Yang, Xiaoming Wang, and Heng Tao Shen. Multilateral semantic relations modeling for image text retrieval. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 2830–2839, 2023. 2 

- [51] Peng Wu and Jing Liu. Learning causal temporal relation and feature discrimination for anomaly detection. _IEEE Transactions on Image Processing_ , 30:3513–3527, 2021. 6 

- [52] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In _Proceedings of the European Conference on Computer Vision_ , pages 322–339, 2020. 6 

- [53] Peng Wu, Xiaotao Liu, and Jing Liu. Weakly supervised audio-visual violence detection. _IEEE Transactions on Multimedia_ , 25:1674–1685, 2022. 6, 7 

- [54] Peng Wu, Jing Liu, Xiangteng He, Yuxin Peng, Peng Wang, and Yanning Zhang. Toward video anomaly retrieval from video anomaly detection: New benchmarks and model. _IEEE Transactions on Image Processing_ , 33:2213–2225, 2024. 2 

- [55] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 18297–18307, 2024. 1, 2, 6 

- [56] Peng Wu, Xuerong Zhou, Guansong Pang, Zhiwei Yang, Qingsen Yan, Peng Wang, and Yanning Zhang. Weakly supervised video anomaly detection and localization with spatio-temporal prompts. In _Proceedings of the ACM International Conference on Multimedia_ , pages 9301–9310, 2024. 2, 6 

- [57] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 6074–6082, 2024. 1, 2, 4, 6, 7 

- [58] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 13204–13214, 2024. 2 

- [59] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 18899–18908, 2024. 2, 6 

- [60] Jiashuo Yu, Jinyu Liu, Ying Cheng, Rui Feng, and Yuejie Zhang. Modality-aware contrastive instance learning with 

   - self-distillation for weakly-supervised audio-visual violence detection. In _Proceedings of the ACM International Conference on Multimedia_ , pages 6278–6287, 2022. 2, 6 

- [61] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection. In _Proceedings of the European Conference on Computer Vision_ , pages 358–376, 2020. 6 

- [62] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 14744–14754, 2022. 6 

- [63] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Clustering aided weakly supervised training to detect anomalous events in surveillance videos. _IEEE Transactions on Neural Networks and Learning Systems_ , 35(10):14085–14098, 2024. 2 

- [64] Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, and Ming-Hsuan Yang. Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 16271–16280, 2023. 2, 6 

- [65] Jiangong Zhang, Laiyun Qing, and Jun Miao. Temporal convolutional network with complementary inner bag loss for weakly supervised anomaly detection. In _Proceedings of the IEEE International Conference on Image Processing_ , pages 4030–4034, 2019. 6 

- [66] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 5579–5588, 2021. 2 

- [67] Sijia Zhang, Maoguo Gong, Yiming Lin, Hao Li, Yuan Gao, and Yihong Zhang. Dual distillation fusion for weakly supervised anomaly detection in surveillance videos. _IEEE TCSVT_ , 35(8):7512–7525, 2025. 1 

- [68] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1237–1246, 2019. 2, 6 

- [69] Yuanhong Zhong, Ruyue Zhu, Ge Yan, Ping Gan, Xuerui Shen, and Dong Zhu. Inter-clip feature similarity based weakly supervised video anomaly detection via multi-scale temporal mlp. _IEEE Transactions on Circuits and Systems for Video Technology_ , 2024. 6 

- [70] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 3769–3777, 2023. 1, 2, 6 

- [71] Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, and Achuta Kadambi. Vlm4d: 

28555 

Towards spatiotemporal awareness in vision language models. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 8600–8612, 2025. 2 

- [72] Jiawen Zhu and Guansong Pang. Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 17826– 17836, 2024. 2 

- [73] Lei Zhu, Chaoqun Zheng, Weili Guan, Jingjing Li, Yang Yang, and Heng Tao Shen. Multi-modal hashing for efficient multimedia retrieval: A survey. _IEEE Transactions on Knowledge and Data Engineering_ , 36(1):239–260, 2023. 2 

28556 

