IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 28, 2026 

10384 

# Structured Graph Reasoning for Traffic Anomaly Detection 

Haoyue Shi , Le Wang _, Senior Member, IEEE_ , Sanping Zhou _, Member, IEEE_ , Fei Hui , Zhanheng Gao, Gang Hua _, Fellow, IEEE_ , and Wei Tang _, Member, IEEE_ 

**_Abstract_ —Unsupervised traffic anomaly detection aims to detect abnormal frames and objects in dashcam videos given only normal videos during training. Most prior works focus on reconstruction or prediction pipelines, while recent graph-based extensions still rely on graph feature learning or limited statistical interactions. Consequently, reasoning over structured relations and abnormal interactions remains largely unexplored. This paper introduces a structured graph reasoning method that bridges normality learning, anomaly detection, and structured reasoning in a unified framework. Unlike prior works, our approach interprets the graph as a semantic abstraction tailored for comprehensive traffic scene understanding. It includes three innovative components: (1) A domain-specific graph representation that models semantic relationships for road scenarios, providing a holistic description of traffic activities. (2) A graph reasoning detector that detects traffic anomalies by reasoning over structured traffic scene graphs. It integrates node anomaly, i.e., unusual object movements, and edge anomaly, i.e., abnormal object interactions, within a unified graph optimization approach, enabling the effective detection of complex, interaction-driven anomalies. (3) An Anomaly Transformer that predicts future traffic scene graphs by attending node and edge queries to object locations and relation histories, thereby effectively learning individual and interaction-level normal traffic behaviors. Experiments on two public datasets, i.e., DoTA and MM-AU, demonstrate the superior performance of our method and its effectiveness in detecting complex abnormal traffic events.** 

**_Index Terms_ —Action and event understanding, traffic accident detection, video anomaly detection, unsupervised learning.** 

Received 30 October 2025; revised 1 January 2026, 15 January 2026, and 2 February 2026; accepted 10 February 2026. Date of publication 12 June 2026; date of current version 13 October 2026. This work was supported in part by the National Science and Technology Major Project under Grant 2024YFB4708100, in part by the National Natural Science Foundation of China under Grant 62088102 and Grant U24A20325, and in part by the Key Research and Development Plan of Shaanxi Province under Grant 2024PT-ZCK-80. The associate editor coordinating the review of this article and approving it for publication was Dr. Jongyoo Kim. _(Corresponding author: Le Wang.)_ 

Haoyue Shi is with the School of Electronics and Control Engineering, Chang’an University, Xi’an 710018, China, and also with the Department of Computer Science, University of Illinois Chicago, Chicago, IL 60607 USA (e-mail: haoyueshi@chd.edu.cn). 

Le Wang and Sanping Zhou are with the National Key Laboratory of HumanMachine Hybrid Augmented Intelligence, and Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, Xi’an, Shaanxi 710049, China (e-mail: lewang@mail.xjtu.edu.cn; spzhou@mail.xjtu.edu.cn). 

Fei Hui and Zhanheng Gao are with the School of Electronics and Control Engineering, Chang’an University, Xi’an 710018, China (e-mail: feihui@chd. edu.cn; snage_gao@stu.xjtu.edu.cn). 

Gang Hua is with Amazon Alexa AI, Bellevue, WA 98004 USA (e-mail: ganghua@gmail.com). 

Wei Tang is with the Department of Computer Science, University of Illinois Chicago, Chicago, IL 60607 USA (e-mail: tangw@uic.edu). This article has supplementary downloadable material available at https://doi.org/10.1109/TMM.2026.3703345, provided by the authors. Digital Object Identifier 10.1109/TMM.2026.3703345 

## I. INTRODUCTION 

RAFFIC accident detection has gained significant atten- **T** tion in recent years, driving the development of various applications in autonomous driving [1] and driver assistance systems [2]. In contrast to these approaches that often rely on supervised learning or handcrafted rules, this work investigates the unsupervised traffic anomaly detection [3], [4], where the model is trained solely on normal driving videos. Our objective is to develop models that deeply understand regular traffic patterns to better identify abnormal frames and objects in dashcam videos. The unsupervised setting is practically valuable because traffic accidents are rare and highly variable, making it challenging to obtain a sufficiently diverse and comprehensive labeled dataset for supervised learning. 

In recent years, numerous video anomaly detection methods [4], [5], [6], [7] have been proposed. Reconstruction and prediction-based methods [6], [8] apply autoencoders or generative adversarial networks (GANs) to memorize normal patterns, and identify anomalies with high reconstruction errors. Density-based methods [9], [10] model normal behavior with parametric distributions, identifying anomalies in low-density regions. To address rapid background changes in driving videos, some methods [3], [7] detect anomalies based on the consistency between detected and predicted object locations. More recent hybrid methods [11], [12] further enhance robustness by incorporating broader contextual cues such as RGB images and optical flow. 

Beyond feature-level modeling, several methods [13], [14], [15] have begun to explore graph-based representations to explicitly model object interactions for video anomaly detection. For example, Sun et al. [13] model generic spatio-temporal context consistency via scene graphs and message passing, while ComplexVAD [14] detects anomalies through exemplar matching on proximity-based interaction graphs. These methods demonstrate the potential of graph representations for modeling interactions. However, they typically rely on generic spatial adjacency or distance-induced relations, together with static context reasoning or exemplar matching, which limits their ability to capture rich inter-agent dynamics and causally related behaviors in complex traffic scenarios. As shown in Fig. 1, incorporating both individual actions and domain-specific interactions is essential for modeling normal traffic patterns and understanding abnormal events. In particular, without explicit reasoning over relational dependencies, models may fail to recognize passive participants in accidents, such as the normally driving car ‘B’ 

10385 

SHI et al.: STRUCTURED GRAPH REASONING FOR TRAFFIC ANOMALY DETECTION 



Fig. 1. Traffic anomalies often arise from complex interactions, making it essential to understand the overall scene context. Our method unifies normality learning, anomaly detection, and structured reasoning in a structured framework. By modeling interactive variations in traffic scenes, our method facilitates a more comprehensive understanding of normal traffic patterns. Through the joint consideration of object states and pairwise relationships, it enhances the detection of both isolated anomalies and those arising from inter-agent interactions. Best viewed in color. 

struck by the speeding car ‘A’ in Fig. 1. This motivates us to explore a structured graph reasoning paradigm that jointly models normal relational patterns and detects deviations from them in both object behaviors and their interactions. 

In this paper, we propose a structured graph reasoning method for unsupervised detection of complex traffic anomalies. Our method interprets the graph as a semantic abstraction of the traffic scene and unifies normality learning, anomaly detection, and structured reasoning in a framework. It consists of three main components. 

_1) Traffic Scene Graph Representation:_ As shown in Fig. 1, we model each traffic scene as a dynamic graph, where nodes correspond to traffic participants (e.g., vehicles, pedestrians) and edges encode their geometry and semantic relationships (e.g., distances, “very near”, “in front of”). In contrast to prior works using limited scene graphs, we design a traffic-centric graph representation that encodes both positional and semantic relationships tailored to road scenarios. This structured representation provides a holistic and interpretable abstraction of agent interactions, serving as a unified foundation for both normality learning and anomaly detection. 

_2) Graph Reasoning for Anomaly Detection:_ Our method identifies anomalies by reasoning over structured traffic scene graphs. It compares observed graphs with those predicted by a future prediction model to capture deviations in both individual behaviors and inter-agent interactions. Specifically, we explicitly model two types of abnormal patterns: node-level anomalies reflecting unusual individual behaviors, and edgelevel anomalies capturing relational deviations. A novel graph optimization scheme is proposed to fuse these heterogeneous anomaly cues with temporal priors, enabling more robust and interpretable detection. This multi-faceted approach enhances robustness against noisy detections and effectively handles complex, interaction-driven anomalies. 

_3) Anomaly Transformer for Graph Prediction:_ We propose an Anomaly Transformer architecture that formulates traffic scene graph prediction as a query-driven prediction formulation. Unlike DETR’s learnable queries, we use detected objects or relations in the current frame as semantic queries attending to 

temporally aggregated historical features. This design enables the model to selectively reason about future locations and interactions based on contextual dynamics while preserving structural alignment with the current scene. Trained auto-regressively on normal traffic videos, the architecture captures structured behavior and interaction patterns. It exhibits strong traffic scene graph prediction capabilities, thereby facilitating our structured framework for anomaly detection. 

Our main contributions are summarized below: 

- r We propose a structured graph reasoning method that models both object behaviors and their domain-specific interactions for unsupervised traffic anomaly detection. It differs from existing approaches in three aspects: traffic-specific scene graph construction, structured graph anomaly reasoning for anomaly detection, and explicit future graph prediction via the Anomaly Transformer. 

- r We introduce a structured traffic scene graph representation, where nodes encode traffic participants, and edges encode domain-specific relations. It serves as a foundation for capturing interpretable agent interactions essential to anomaly detection. 

- r We develop a graph reasoning detector that detects traffic anomalies by comparing observed and predicted scene graphs. By integrating node- and edge-level deviations through a unified graph optimization approach, our approach can effectively capture complex, interaction-driven anomalies. 

- r We propose a novel Anomaly Transformer for traffic scene graph prediction, which attends node and edge queries to object locations and relation histories. Trained on normal driving videos in an auto-regressive manner, it learns both individual and interaction-level traffic patterns, enabling accurate detection of abnormal behaviors. 

- r Experimental results demonstrate state-of-the-art performance of our method and its effectiveness in detecting complex abnormal traffic events. 

## II. RELATED WORK 

## _A. Unsupervised Traffic Anomaly Detection_ 

Video anomaly detection has attracted increasing attention in recent years, with researchers exploring different learning paradigms to detect abnormal events in surveillance videos, including one-class, unsupervised, and weakly supervised settings [16], [17]. Anomaly detection in traffic scenes is considerablymorecomplexandchallengingduetodiverseobjectinteractions, dynamic camera motions, and varying environmental conditions. The growing focus on traffic anomaly detection [2], [18], [19], [20] has led to the development of diverse accident warning systems. Despite these advancements, many existing methods [18], [21], [22] depend on supervised learning paradigms that require extensive labeled datasets comprising both normal and anomalous traffic scenarios. In contrast, unsupervised traffic anomaly detection [7], where only normal data are provided, is more adaptable to the unpredictable nature of real-world traffic and plays a dominant role in past studies. 

10386 

IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 28, 2026 

In recent years, a series of methods [5], [6], [8], [23], [24], [25] for unsupervised video anomaly detection have been proposed. Reconstruction and prediction-based methods [4], [6], [8], [23] build models to reconstruct video frames. They try to memorize normal patterns using generative models like autoencoders [5], [6], [8] and GANs [26], and detect anomalies by identifying instances with high reconstruction errors. Specifically, Wu et al. [4] propose a dynamic self-supervised network to model normal spatio-temporal patterns. Liu et al. [23] propose a hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. Some subsequent studies [24], [25] attempt to incorporate memory modules and model optical flows to overcome overfitting. Moreover, density-based methods [9], [10] take an object-centric perspective, modeling the distribution of normal patterns using parametric density functions, and identifying anomalies in low-density regions. For example, Reiss et al. [9] extract highlevel attributes first, and then use parametric density functions (e.g., GMM or KNN) to model the data distribution. As rapid background changes in driving videos can confuse frame prediction models, FOL-based methods [3], [7] model normal patterns by employing recurrent neural networks (RNNs) to forecast the future locations of objects and detect anomalies based on location consistency. Recent hybrid approaches [11], [12] further integrate broader contextual cues, e.g., image RGB or image flow, to enhance robustness in traffic anomaly detection. For instance, Fang et al. [11] combine frame prediction and future object location to learn normal scene patterns and normal trajectory patterns. Liang et al. [12] incorporate the reconstruction of the optical flow of the whole frame [23] for future object location. Unlike prior works that often employ reconstruction or prediction pipelines, we propose a structured reasoning method that jointly considers object motion and relational dependencies. This formulation improves interpretability and robustness against dynamic scene variations, offering a unified framework for structured traffic understanding. 

reasoning, and do not explicitly capture domain-specific interaction semantics or dynamic structures. In contrast, our method differs in three aspects. (1) We construct traffic-centric semantic scene graphs with explicitly defined interaction relations. (2) We introduce explicit future graph prediction to model interaction dynamics beyond static reasoning. (3) We jointly aggregate multi-source abnormal evidence under an optimization framework, rather than using static scoring rules. 

## _C. Future Prediction Methods_ 

Future prediction models have demonstrated notable success across various domains, including weather forecasting [30], action forecasting [31], and trajectory prediction [32], [33]. These models utilize advanced machine learning techniques such as RNNs (e.g., LSTMs, GRUs), GANs [26], and the conditional variational autoencoder (CVAE) [34] to predict future outcomes. Krishnan et al. [30] propose a global weather prediction model that leverages Adaptive Fourier Neural Operators (AFNOs) for efficient and accurate forecasting by learning from historical weather data. Li et al. [31] propose to explore relational spatio-temporal learning for multi-person action forecasting. Many RNN-based methods, such as [33], have been used to model pedestrian trajectory patterns, enabling motion forecasting in crowded or dynamic environments. Recent works such as AgentFormer [32] employ Transformer architectures for multi-agent future trajectory prediction. While AgentFormer models multi-agent trajectory forecasting via latent social interactions, our approach explicitly predicts structured traffic scene graphs by attending current-frame object queries to temporally compressed historical contexts. This design not only enables interpretable interaction reasoning but also supports downstream anomaly detection by comparing predicted and observed graph structures. 

## III. METHOD 

## _A. Overview_ 

## _B. Graph-Based Video Anomaly Detection Methods_ 

Scene graphs have been explored in many previous works [27], [27], [28] to capture relationships between objects. In the field of traffic risk forecasting, Yu et al. [27] use a graph-based approach to capture complex spatial dependencies in traffic networks, enabling traffic flow map forecasting in urban environments. UApredictor [29] forecasts urban traffic anomalies by modeling complex spatial–temporal dependencies across road networks. As our work mainly targets unsupervised video anomaly detection, Chen et al. [28] propose a Bag-of-Triplets representation derived from scene graphs for interpretable anomaly classification. Doshi et al. [15] use object-predicate-object triplets with semantic embeddings for metric-based anomaly detection. Sun et al. [13] propose sceneaware context reasoning that models generic spatio-temporal context consistency via scene graphs and message passing. ComplexVAD [14] detects interaction anomalies by constructing proximity-based object graphs and identifying deviations from memorized normal exemplars. However, these methods primarily rely on generic spatial interactions or static context 

Unsupervised traffic anomaly detection aims to detect both object and frame-level anomalies in dashcam videos using only normal driving data for training. We propose a structured graph reasoning framework that jointly integrates learning, detection, and reasoning about traffic scenes. As illustrated in Fig. 2, our method comprises three key components: (1) traffic scene graph representation, which encodes geometry and semantic relations among participants; (2) graph reasoning for anomaly detection, which compares predicted and observed graphs to identify nodeand edge-level anomalies through graph optimization; and (3) Anomaly Transformer for graph prediction, which leverages a query-driven transformer to forecast future graphs from historical context. Detailed descriptions of each component are provided in Sections III-B–D. 

## _B. Traffic Scene Graph Representation_ 

We represent the traffic scene in a video as a dynamic graph _Gt_ = _{Bt, Rt}_ , which is specified by a set of nodes _Bt_ and a set of edges _Rt_ . 

SHI et al.: STRUCTURED GRAPH REASONING FOR TRAFFIC ANOMALY DETECTION 

10387 



Fig. 2. Framework Overview. (a) We model the traffic scene as a domain-specific graph, where nodes represent traffic participants and edges encode traffic-specific geometry and semantic relations. (b) Anomalies are detected by reasoning over structured scene graphs through comparison between observed and predicted graphs. We integrate node anomaly and edge anomaly within a graph optimization framework to derive abnormal scores, capturing traffic anomalies with both unusual object positions and irregular interactions. (c) We propose an Anomaly Transformer to predict future traffic scene graphs by attending node and edge queries to object locations and relation histories. It enhances anomaly detection by effectively modeling normal traffic behaviors. 

_Nodes:_ Each node corresponds to a traffic participant, such as a vehicle or the ego car. At time _t_ , we represent each node _i_ as a 4-dimensional bounding box **b** _t,i_ = ( _x, y, w, h_ ), where ( _x, y_ ) denotes the center position of the participant and ( _w, h_ ) indicates its width and height in the image plane. The collection of all nodes is denoted as _Bt_ = _{_ **b** _t,i}_<sup>_N_</sup> _i_ =1<sup>_t_, where</sup><sup>_Nt_is the number of</sup> participants at time _t_ . The node corresponding to the ego car is represented as a fixed position. 

_Edges:_ Each edge **r** _t,i,j_ between two nodes _i_ and _j_ at time _t_ captures both geometric and semantic relationships. Specifically, we define a 4-dimensional relative geometric feature **r**<sup>d</sup> _t,i,j_<sup>=</sup><sup>**b**</sup><sup>_t,i −_</sup><sup>**b**</sup><sup>_t,j_,where</sup><sup>**b**</sup><sup>_t,i_= (</sup><sup>_x, y, w, h_)representsthe</sup> center position and size of participant _i_ ’s bounding box. This geometric difference provides a coarse representation of spatial offset in position and scale. To model high-level traffic-specific interactions, we further incorporate semantic relations **r**<sup>s</sup> _t,i,j_<sup>de-</sup> rived from the roadscene2vec tool [35]. It first constructs a bird’s-eye-view representation and then predicts 11 discrete relation categories, including proximity relations (e.g., Near Collision, Super Near) and directional relations (e.g., to Left of, at Direct Rear of). These relations are not used to define anomaly types, but instead serve as interpretable interaction primitives that enable structured modeling of multi-agent dynamics. Each relation is encoded as a one-hot vector over the 11 categories. We consider edges between every pair of different nodes, and therefore have _Rt_ = _{_ **r** _t,i,j}_<sup>_N_</sup> _i_ =1<sup>_t,N_</sup> _,j_<sup>_t_</sup> =1<sup>.</sup> 

_Discussion:_ For graph construction, previous graph-based anomaly detection methods typically construct graphs using generic visual relationships, such as spatial adjacency [13] or proximity-based interactions [14]. In contrast, we construct a traffic-centric scene graph, which provides a structured and interpretable abstraction of multi-agent dynamics by explicitly modeling both geometric and semantic relations among traffic participants. This design enables fine-grained reasoning about spatial proximity and directional interactions that are crucial for understanding complex traffic behaviors. This structured formulation also provides a unified foundation for downstream tasks such as anomaly detection and scene prediction. 



Fig. 3. Multiple prediction generation by the sliding-window forecasting. This case takes _T_ = _H_ = 3 as an example. 

## _C. Graph Reasoning for Anomaly Detection_ 

At time _t_ , we first generate _H_ predicted traffic scene graphs _{G_<sup>�</sup> _t}_<sup>_H_</sup> _t_ =1<sup>using the Anomaly Transformer (Section III-D), where</sup> eachgraphispredictedfromadistincthistoricalsequenceending between frames _t − H_ and _t −_ 1. As shown in Fig. 2(b), we then detect anomalies by comparing the observed graph _Gt_ with these predicted graphs in a unified graph reasoning and optimization framework. Specifically, we develop a Bayesian-based Graph Anomaly Optimization (GAO) scheme that combines node- and edge-level anomaly likelihoods with temporal priors, yielding a robust and interpretable abnormality score for each frame. 

_Multiple Prediction Generation:_ The Anomaly Transformer predicts _H_ future graphs based on observations over _T_ frames using a sliding-window strategy, exactly as stated in Section III-D. During inference, the observation window is slid forward by one frame to generate predictions over the entire sequence. Fig. 3 illustrates this sliding-window forecasting. As a result of overlapping prediction horizons, a target frame _t_ may appear in multiple prediction windows and is therefore predicted _H_ times, corresponding to _H_ different observation windows whose end frames range from _t − H_ to _t −_ 1. These multiple predictions for the same frame are then used differently for node- and edge-level anomaly scoring, i.e., the node-level score is computed from a representative horizon, whereas the edge-level score is derived from cross-horizon inconsistency across all _H_ predictions. 

_Node Anomaly:_ It calculates abnormal scores based on the displacements between predicted and detected node states i.e., bounding boxes of participants. This approach helps detect 

IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 28, 2026 

10388 

abnormal objects that deviate from normal traffic patterns due to their irregular actions, such as sudden acceleration or abrupt lane changes. Formally, the node-based abnormal score is defined as: 



where **b** _t,i_ and **b**<sup>�</sup> _t,i_ are the detected state and the predicted state for the _i_ -th node in the _t_ -th frame, respectively. We take **b**<sup>�</sup> _t,i_ from the prediction with the longest forecasting horizon, i.e., generated from the observation window ending at _t − H_ . Such longer-horizon predictions tend to amplify abnormal object motion and displacement. 

_Edge Anomaly:_ To identify abnormal interactions between traffic participants, we compute edge anomaly scores based on both predicted pairwise distances and semantic relations. These signals are particularly effective in detecting passive abnormal agents whose individual motion may appear normal but whose interactions deviate from regular patterns. We first construct an edge-induced state for each node _i_ at time _t_ by aggregating predicted information from all its connected edges. Specifically, each predicted bounding box **b**<sup>�</sup> _t,i ∈_ R<sup>4</sup> is combined with its predicted pairwise distance vectors � **r**<sup>d</sup> _t,i,j_<sup>_∈_R4 (defined in the same</sup> coordinate space) to compute an interaction-aware box feature: **b** _t,i_ = _N_ <u>1</u> _t_ � _Nj_ =1 _t_<sup>(</sup><sup>**b**�</sup><sup>_t,i_+ �</sup><sup>**r**</sup> _t,i,j_<sup>d)</sup><sup>_._It can be interpreted as the con-</sup> textually adjusted location and size of node _i_ after considering its spatial relations with surrounding objects. In parallel, we compute the average predicted semantic relation vector � **r**<sup>s</sup> _t,i,j_<sup>_∈_R11</sup> and concatenate it with **b** _t,i_ to obtain the final edge-aware state: **~~r~~** _t,i_ = [ _N_<sup><u>1</u></sup> _t_ � _Nj_ =1 _t_<sup>�</sup><sup>**r**</sup> _t,i,j_<sup>s</sup><sup>_,_</sup> **b** _t,i_ ] _._ We measure the uncertainty of edge-aware features across _H_ predicted graphs by computing the feature-wise standard deviation followed by mean pooling: 



where the standard deviation operator _σ_ ( _·_ ) is computed featurewise along the prediction dimension _H_ . Then the mean operator _μ_ ( _·_ ) averages the resulting values over all feature dimensions to produce a scalar edge anomaly score _s_<sup>edge</sup> _t,i_<sup>.Ahigherscore</sup> indicates greater inconsistency or uncertainty in the predicted interactions involving the node, suggesting that the object may be involved in an abnormal relational pattern. 

_Graph Anomaly Optimization:_ We propose a graph anomaly optimization method that integrates node and edge anomaly likelihoods with temporal priors from a principled Bayesian perspective [36]. Formally, let **s**<sup>node</sup> _∈_ R<sup>_T_tr</sup> and **s**<sup>edge</sup> _∈_ R<sup>_T_tr</sup> be the sequences of node-based and edge-based abnormal scores of an object tracklet across _T_<sup>tr</sup> frames, we aim to solve abnormal scores **x** _∈_ R<sup>_T_tr</sup> . 

Since anomaly scores are strictly positive and often heavytailed, we model them using a log-normal distribution. Its probability density function is: 



Each point _xt_ should be simultaneously close to its corresponding node anomaly _s_<sup>node</sup> _t_ and edge anomaly _s_<sup>edge</sup> _t_ , while also ensuring that the anomaly score is temporally smooth. Therefore, we formulate the likelihood distributions corresponding to node anomaly and edge anomaly as _s_<sup>node</sup> _t |xt ∼ f_ ( _s_<sup>node</sup> _t_ ; _xt, α_<sup><u>1</u>)and</sup> _s_<sup>edge</sup> _t |xt ∼ f_ ( _s_<sup>edge</sup> _t_ ; _xt, β_<sup><u>1</u>), respectively. The prior distribution of</sup> _xt_ is _xt ∼ f_ ( _xt_ ; _xt−_ 1 _, γ_<sup><u>1</u>), where</sup><sup>_xt_depends on the result of the</sup> previous time step. 

Given the observed and predicted graphs, node- and edgelevel anomalies are derived from two complementary cues, i.e., individual motion and relational consistency. We adopt a conditional independence approximation between the node- and edge-level anomalies to enable tractable inference. Such approximations are widely used in Bayesian fusion frameworks [37], [38] and are empirically effective, as further demonstrated in our experiments. Therefore, we can obtain the posterior distribution as: 



By taking the expectation of the distribution function (See supplementary material for detailed derivations), we obtain: 



Intuitively, (5) adaptively fuses node, edge, and temporal priors by weighting their log-domain contributions with their respective coefficients. _α_ and _β_ control the relative importance of the node anomaly and edge anomaly, respectively, while _γ_ regulates the impact of neighboring frames. The log-normal modeling naturally supports multiplicative evidence accumulation and avoids the linear compensation effect inherent in the weighted-sum fusion. Specifically, GAO is executed once for each object tracklet in each video, producing frame-wise fused anomaly scores. At each frame, it takes the current node and edge scores together with the previously fused state to update the anomaly score. We derive the anomaly score for a frame by averaging the anomaly scores of all objects within it. 

_Discussion:_ For anomaly score generation, rather than relying on a single context-inconsistency score or feature-level deviation, we define node-level and edge-level anomaly cues from the discrepancy between predicted and observed graphs, and further aggregate them with temporal priors under a unified graph optimization framework. This formulation improves interpretability and robustness against noisy detections and dynamic scene variations. Rare or complex traffic anomalies often emerge as higher-order patterns characterized by concurrent multiple edge or node deviations and temporal inconsistency, which can be effectively captured by our graph reasoning detector. In addition, we establish one-to-one correspondences between the observed graph _Gt_ and the predicted graphs _{G_<sup>�</sup> _t}_<sup>_H_</sup> _t_ =1<sup>. For nodes that newly</sup> appear and have no counterpart in the predicted graphs, we treat them as unmatched and normal. This design reduces false positives caused by normal agent appearances while preserving the ability to detect truly abnormal appearances. 

SHI et al.: STRUCTURED GRAPH REASONING FOR TRAFFIC ANOMALY DETECTION 

10389 



Fig. 4. Comparison of traffic anomaly detection methods in terms of object location prediction. (a) Existing methods use RNNs [3], [7] to predict individual object locations based on their historical observations. (b) The Anomaly Transformer directly predicts a set of object bounding boxes by leveraging a sequence representation of objects. Besides, it predicts object locations and interactions together. Here, we use object location prediction as an example. 



Fig. 5. Anomaly Transformer for graph prediction. The Anomaly Transformer predicts future traffic scene graphs by attending node and edge queries to object locations and relation histories, thereby effectively learning individual and interaction-level normal traffic behaviors. 

## _D. Anomaly Transformer for Graph Prediction_ 

To establish a robust model for predicting future traffic scene graphs, we introduce the Anomaly Transformer. It leverages a sequence of historical observation graphs over _T_ frames, i.e., _{Gt}_<sup>_T_</sup> _t_ =1<sup>, to forecast future graphs over the next</sup><sup>_H_frames, i.e.,</sup> _{Gt}_<sup>_T_</sup> _t_ =<sup>+</sup> _T_<sup>_H_</sup> +1<sup>.Fig.4presentsdifferenttrafficanomalydetection</sup> approaches in terms of object location prediction. Compared with existing TAD methods [3], [7], our traffic transformer directly predicts (in parallel) a set of object bounding boxes in the future frames by leveraging a sequence representation of objects in each frame as input tokens. Our graph prediction module, as shown in Fig. 5, includes the feature encoder, the Anomaly Transformer, and prediction heads. 

_Feature Encoder:_ To obtain node and edge embeddings, we firstencodeanobjectlocationorapairwiserelationusingalinear layer. Then, encoded features are combined with the cropped optical flow feature [3], [7] to generate the node feature **e**<sup>node</sup> _∈_ R<sup>_D_</sup> and edge feature **e**<sup>edge</sup> _∈_ R<sup>_D_</sup> , where _D_ is the feature dimension. 

_Anomaly Transformer:_ It includes a history encoder and a Transformer decoder. The history encoder aims to encode node or edge features from history frames, i.e., the 1st to ( _T −_ 1)-th frames, as the history embedding **E**<sup>kv</sup> _∈_ R<sup>_N ∗_</sup> _T −_ 1<sup>_×D_</sup> . The superscript ‘*’ indicates the type, which can be either node with 

_NT_<sup>node</sup> _−_ 1<sup>=</sup><sup>_NT −_1oredgewith</sup><sup>_N_edge</sup> _T −_ 1<sup>=</sup><sup>_N_2</sup> _T −_ 1<sup>.Wetreatnodeor</sup> edge features in the ( _T −_ 1)-th frame as anchors and build _NT_<sup>_∗_</sup> _−_ 1 tracklets based on the tracking results. Each tracklet is then padded to a uniform length of _T −_ 1, yielding history feature representations **K** _∈_ R<sup>_N ∗_</sup> _T −_ 1<sup>_×_(</sup><sup>_T −_1)</sup><sup>_×D_</sup> . The features are first reshaped into R<sup>_N ∗_</sup> _T −_ 1<sup>_×_(</sup><sup>_T −_1)</sup><sup>_D_</sup> and then embedded by a learnable linear transformation to obtain the history embeddings. 

The Transformer decoder treats node or edge features in the _T_ -th frame **E**<sup>q</sup> _∈_ R<sup>_N ∗_</sup> _T_<sup>_×D_</sup> as query tokens, node or edge features in **E**<sup>kv</sup> as key and value tokens to predict the future node or edge states. These tokens are fed into the Transformer decoder including three multi-head attention layers to obtain the output feature embeddings **E**<sup>o</sup> _∈_ R<sup>_N ∗_</sup> _T_<sup>_×D_</sup> as: 



where Θ denotes the model parameters. Each query token characterizes a node or edge feature in the _T_ -th frame, and then predicts its representation in the future frames. Through alternating self-attention and cross-attention layers, the Anomaly Transformer reasons about the relationships of items in both the temporal and social dimensions to directly output a set of predictions in parallel. Unlike the traditional Transformer decoder [42], we first apply cross-attention and then apply self-attention. We do not use positional encoding, as the input embeddings inherently include positional information. 

_Prediction Head:_ We apply a linear layer to the output feature embeddings, followed by reshaping to obtain the final predictions over the next _H_ frames _{G_<sup>�</sup> _t}_<sup>_T_</sup> _t_ =<sup>+</sup> _T_<sup>_H_</sup> +1<sup>. Instead of directly pre-</sup> dicting the absolute coordinates of object locations, the prediction head generates a displacement sequence _{_ Δ **b**<sup>�</sup> _t,i}_<sup>_T_</sup> _t_ =<sup>+</sup> _T_<sup>_H,N_</sup> +1 _,i_<sup>_T_</sup> =1 and pairwise relations _{_ � **r** _t,i}Tt_ = + _TH,N_ +1 _,iT_<sup>2</sup> =1<sup>, where</sup><sup>_i_is the object or</sup> relation tracked in the _T_ -th frame. The predicted location is defined as **b**<sup>�</sup> _t,i_ = **b** _T,i_ + Δ **b**<sup>�</sup> _t,i_ . The predicted relations are further reshaped as _{_ � **r** _t,i,j}_<sup>_T_</sup> _t_ =<sup>+</sup> _T_<sup>_H,N_</sup> +1 _,i_<sup>_T_</sup> =1<sup>_,N_</sup> _,j_<sup>_T_</sup> =1<sup>.</sup> 

_Loss Function:_ We compute the training loss based on the consistency between the predicted and observed graphs, includingnodepredictionconsistencyandedgepredictionconsistency. Formally, the node prediction loss _L_<sup>node</sup> is: 



where Δ **b** _t,i_ = **b** _t,i −_ **b** _T,i_ is the detected displacement of the _i_ -th object between the _T_ -th frame and the _t_ -th frame. The edge prediction loss _L_<sup>edge</sup> is calculated as follows: 



The summations in (7) and (8) are performed only over objects present in the anchor _T_ -th frame, and the loss terms corresponding to disappearing objects are set to zero. Finally, we train the model to predict the future traffic scene graph with the total loss _L_ , which is defined as: 



IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 28, 2026 

10390 

We perform a sliding-window forecasting process with a stride of one frame. This prediction paradigm is consistently used during both training and inference, while supervision is applied only during training. 

_Discussion:_ For the prediction mechanism, we introduce the Anomaly Transformer for explicit future traffic scene graph prediction to model interaction dynamics, instead of static context reasoning or exemplar matching within observed frames. In dynamic traffic scenes, the number and identity of participants continuously change as new vehicles enter and others leave. Using a fixed number of learnable queries, as in DETR, lacks explicit correspondence to real agents, leading to unstable alignment and limited interpretability. In contrast, our Anomaly Transformer uses detected objects and relations in the current frame as semantic queries, maintaining one-to-one correspondence with physical entities and ensuring structural alignment between predicted and observed graphs. This design allows the model to reason about agent-specific temporal evolution while mitigating the error accumulation inherent in cascaded relation prediction. Unlike prior forecasting models such as AgentFormer that focus primarily on trajectory prediction, our approach directly predicts both future positions and interactions within a structured scene graph. This enables structurally and semantically aligned prediction for downstream reasoning. 

## IV. EXPERIMENTS 

Inthissection,wefirstprovideexperimentaldetails,thendraw comparisons with existing methods, and finally study different components of our method. 

## _A. Experimental Details_ 

_Evaluation Datasets:_ DoTA [7] is the first dataset designed specifically for traffic anomaly detection. It contains 4677 abnormal videos captured from dashcams in driving scenarios. These data includes videos from various scenes under different weather and lighting conditions. Recent methods [3], [7], [12] followed the data organization of [7], where the training set contains 3275 videos (using only the normal video segments preceding the anomaly), and the test set contains 1402 videos. 

MM-AU [43] is a recently released dataset. It contains 11727 in-the-wild ego-view accident videos sourced from LOTVSCap [44] and LOTVS-DADA [45]. Following [11], we only use LOTVS-DADA for training and testing. We reorganize this dataset and annotate the bounding boxes of abnormal objects, resulting in 1905 videos. Each video is annotated by two annotators, and the final annotations are obtained by averaging to reduce individual bias (see supplementary material for annotation details). It is split into training and test sets to balance the number of accident types, with 977 videos in the training set (using only the normal video segments preceding the anomaly), and 928 videos in the test set. We will release these annotations and splits upon acceptance. 

_Evaluation Metrics:_ Following prior work [3], [5], [7], [9], [12], we use the area under the ROC curve (AUC) metric for evaluating the frame-level anomaly detection performance. We use the spatialtemporal area under the curve (STAUC) metric 

to evaluate abnormal object detection performance. STAUC [7] adopts the overlap between the predicted score map and the annotated anomalous region as a weighting factor for true positive predictions. It captures how well a model detects both temporal and spatial locations of anomalies. Larger AUC (or STAUC) values indicate better performance. 

_Implementation Details:_ The Anomaly Transformer consists of three attention blocks, each with a hidden dimension of 512 and 4 attention heads. Each mini-batch consists of 32 videos for DoTA and 64 videos for MM-AU. We train the model for 100 epochs with the RMSprop optimizer using a learning rate of 0.0001 for both datasets. Following standard practice in prior traffic anomaly detection studies [3], [7], [12], we set the number of predicted frames _H_ = 16 and observed frames _T_ = 16 for two datasets in both training and inference stages. This configuration balances temporal context with computational cost. We directly use the tracking results provided by [3], [7], [40] to obtain consistent object trajectories across frames. The hyperparameters _α_ = 6, _β_ = 1, and _γ_ = 0 _._ 2 are consistent across all datasets. 

## _B. Comparison With State-of-The-Art Methods_ 

Table I compares the performance of our method with existing traffic anomaly detection methods on the DoTA and MM-AU datasets. Following [23], existing methods are categorized into two groups: Hybrid methods like SSC-TAD [11] and MAMTCF [12] adopt both frame-level and object-centric feature combination as inputs. Object/Frame-only methods rely on single-modality inputs. We reimplement Interpretable VAD [9] underthesameevaluationprotocol.WealsoreportAUC/STAUC together with 95% confidence intervals estimated via bootstrap resampling on the test set to reflect result variability. Among the object/frame-only methods, our approach achieves the best performance in terms of both AUC and STAUC on DoTA, surpassing all other methods that use either frame-level or object-centric features alone. Reconstruction-based methods [8], [39] struggle due to rapid background changes in driving videos, making it difficult for frame reconstruction models to distinguish normal variations from anomalies. Prediction-based methods [3], [5], [7] perform poorly as they simplify interactions through generic visual predicates or even neglect them entirely. In contrast, our method achieves promising results by providing a structured anomaly interpretation. Moreover, our method achieves competitive performance compared with the hybrid method MAMTCF. While hybrid approaches benefit from dense RGB appearance information, they typically incur higher computational cost and reduced interpretability. In contrast, our method emphasizes object-centric interactions to better capture interaction-driven anomalies with improved interpretability. 

Table I also presents results of different methods on MM-AU. Reported results (marked with<sup>_†_</sup> ) of some methods [11], [12] are evaluated under a different protocol with a subset (212 videos) of DADA. Despite the more challenging and realistic setting, our approach achieves the best performance among all object/frame-only methods in terms of AUC and STAUC. While slightly lower than the hybrid methods, our results demonstrate 

SHI et al.: STRUCTURED GRAPH REASONING FOR TRAFFIC ANOMALY DETECTION 

10391 

TABLE I 

COMPARISON WITH STATE-OF-THE-ART METHODS IN AUC (%) AND STAUC (%) ON DOTA AND MM-AU DATASETS 



### TABLE II 

ABLATION STUDY OF EACH COMPONENT OF OUR METHOD ON DOTA 

### TABLE III 

ABLATION STUDY OF GRAPH REASONING DETECTOR ON DOTA 





strong robustness and adaptability under practical deployment conditions, confirming the effectiveness of our framework. Our method explicitly models both individual behaviors and relational dynamics between traffic participants. This enables more accurate anomaly detection, capturing the subtle interactions that are essential in traffic scenarios. 

## _C. Ablation Studies and Analysis_ 

To better understand how the proposed method works, we conduct a series of ablation studies considering the following aspects: effectiveness of the proposed components, ablation study of the graph reasoning detector, semantic and geometric relation analysis, interpretability analysis, complex traffic abnormal events detection, ablation study of hyperparameters, generalizability study, robustness analysis, and inference time. 

_Effectiveness of Proposed Components:_ In Table II, we conduct ablation studies to investigate the contribution of each component in our method. Starting with the RNN baseline that models individual trajectories without structural reasoning, the performance is limited. It adopts a GRU-based recurrent architecture with a single-layer GRU of hidden size 128 to predict future per-object bounding box trajectories, using the same per-object observations and MSE loss as our method. Objects are modeled independently, and predictions are aggregated at the frame level without explicit interaction modeling. Replacing it with the Anomaly Transformer notably improves results, 

showing that the Anomaly Transformer benefits normality learning and normality detection. Introducing the graph anomaly detector further enhances detection, where relations are derived from predicted object locations. When the graph representations are used without graph anomaly detection, i.e., only detecting by node anomalies, the performances are comparable with the Transformer baseline. By further incorporating graph anomaly detection, the performance is consistently improved. This is because our graph anomaly detector employs a multi-faceted approach that simultaneously captures node and edge anomalies, which are important for detecting interaction-driven anomalies but have been neglected by previous methods. By normality learning and anomaly detection in a graph-based manner, our graph reasoning method significantly improves unsupervised traffic anomaly detection, yielding more robust anomaly inference under complex multi-agent interactions. 

_Ablation Study of the Graph Reasoning Detector:_ Table III presents the ablation studies for our graph reasoning detector. It compares GAO with node/edge-only scores and several representative fusion strategies, including a normalized weighted sum, a geometric mean without the temporal prior, an Exponential Moving Average (EMA) applied to weighted or geometric scores, and a Gaussian-based fusion method. The results show that node-based anomaly detection performs better in terms of AUC, while edge-based anomaly detection excels in STAUC. This difference can be attributed to the fact that node anomaly 

IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 28, 2026 

10392 

TABLE IV 

ABLATION STUDY OF GEOMETRIC AND SEMANTIC RELATIONS ON DOTA 





Fig. 6. Distribution of semantic relations on DoTA normal training data. 

focuses on unusual object movements, such as abnormal vehicles, while edge anomaly focuses on abnormal relationships between objects, like unusual distances. When combining these two types of scores, consistent values across both metrics are achieved. This suggests that the two scores are complementary. Note that GAO consistently outperforms these linear fusion strategies, highlighting the benefit of multiplicative fusion for positive, heavy-tailed anomaly scores. The weighted sum is more sensitive to noisy scores, the geometric mean is overly sensitive to weak anomaly evidence, and EMA introduces a noticeable detection delay to abrupt anomalies. While the Gaussian-based fusion method improves scale alignment, it still relies on fixed linear fusion. In contrast, GAO achieves a better balance between robustness and responsiveness by explicitly incorporating temporal priors within the optimization framework. These results indicate that the performance gain of GAO does not stem merely from increased complexity, but from its principled integration of multi-level anomaly evidence over time. 

_Semantic and Geometric Relation Analysis:_ In Table IV, we conduct ablation studies to investigate the contribution and robustness of geometric and semantic relations in anomaly detection. It explicitly compares different types of relations under the same backbone, including geometry-only, semantics-only, and geometry+semantics, together with additional robustness evaluations by randomly corrupting semantic relation labels with different probabilities. The results show that geometric relations are sensitive to global structural changes that facilitate higher AUC scores, while semantic relations encode agent-centric interaction roles that enable better STAUC scores. Combining geometry and semantics yields the best performance, indicating that the two types of relations are complementary rather than redundant. Moreover, when a portion of semantic relation labels is corrupted, the performance remains stable under mild substantial label corruption. In addition, we present Fig. 6 to 



Fig. 7. Interpretability analysis of our method. Shaded regions indicate ground-truth anomalous intervals. Blue and orange curves represent fused anomaly scores of our method and FOL, respectively. Red and blue bounding boxes denote anomalous and normal objects. Scores above each box correspond to node (v), edge (e), and fused (s) scores of our method (blue), as well as predicted scores of FOL (orange). White and black arrows denote locations of normal and abnormal frames. 

report the distribution of semantic relation categories in the normal training data. The distribution reveals that no single relation type dominates the normal data, and risk-related relations such as near-collision are common in normal traffic scenarios, e.g., close car-following or dense driving conditions. This suggests that the semantic relations provide diverse and structured interaction cues rather than acting as shortcut signals for anomaly detection. 

_Interpretability Analysis:_ We provide a representative traffic anomaly case in Fig. 7 to further validate the interpretability of our method. Specifically, Fig. 7(a) provides an overview of the video sequence together with object- and frame-level anomaly score curves. It shows that node-level and edge-level anomaly cuesjointlycontributetothefinalanomalyscore,andourmethod yields more accurate temporal localization compared to FOL. Fig. 7(b) visualizes a key frame by explicitly comparing the observed and predicted traffic scene graphs. The visualization includes both the bounding boxes and semantic relations, such as near_coll and super_near, making the semantic discrepancies directly observable. By contrasting predicted and observed nodes and edges, the graph reveals how interaction-driven anomalies are localized through edge-level reasoning, while individual motion irregularities are captured by node-level cues. Moreover, the node-, edge-, and fused-score curves of the object car demonstrate that the proposed GAO consistently amplifies coherent anomaly evidence, providing an intuitive explanation of its effectiveness for interpretable anomaly detection. 

_Complex Traffic Abnormal Events Detection:_ To verify the effectiveness of our method in detecting complex traffic 

SHI et al.: STRUCTURED GRAPH REASONING FOR TRAFFIC ANOMALY DETECTION 

10393 

TABLE V 

EFFECTIVENESS IN DETECTING COMPLEX TRAFFIC ANOMALIES ACROSS SIX TYPES OF ABNORMAL EVENTS ON DOTA 



Fig. 8. Robustness analysis on DoTA under controlled perturbations. 

abnormal events, we evaluate its performance on six distinct types of abnormal events from the DoTA dataset, each involving more than two agents. These event types include ST (collision with another vehicle that starts, stops, or is stationary), AH (ahead collision), LA (lateral collision), OC (oncoming collision), TC (turning or crossing collision), and VP (vehiclepedestrian collision). Table V displays the detection results for our method as well as other approaches [5], [7]. We include object/frame-only methods to ensure a fair comparison. We note that class ST is particularly challenging for our method, as the involved vehicles tend to exhibit small, hard-to-detect changes in speed or direction, making it difficult to distinguish from normal traffic patterns. Nevertheless, our method demonstrates superior performance compared to previous approaches across a variety of complex abnormal events. In classes AH and VP, our method consistently achieves higher AUC and STAUC scores when detecting abnormal frames and objects. These results demonstrate the strength of our approach in modeling multi-agent interactions, enabling more accurate detection of complex abnormal events than previous methods. 

_Ablation Study of Hyperparameters:_ The sensitivity analysis of three key hyperparameters _α_ , _β_ , and _γ_ is shown in Fig. 9. Specifically, _α_ and _β_ control the relative importance of the node and edge anomalies, while _γ_ regulates the impact of neighboring frames. The optimal performance is achieved when _α_ = 6 and _β_ = 1 for both DoTA and MM-AU. Increasing _γ_ brings in information from more distant frames, which broadens the temporal context but can also introduce irrelevant data, potentially 



Fig. 9. Hyperparameter sensitivity analysis on key fusion parameters. _α_ and _β_ control the relative importance of the node and edge anomaly, while _γ_ regulates the impact of neighboring frames. 

TABLE VI COMPARISON IN AUC (%) ON SHANGHAITECH 

TABLE VII COMPARISON OF MODEL SIZE AND INFERENCE TIME ON DOTA 

lowering the accuracy of anomaly detection. Nevertheless, our method consistently achieves higher scores when _γ_ is set to a value smaller than 1. This observation validates the effectiveness of our graph anomaly optimization strategy, which integrates both node and edge anomalies with temporal priors from a principled Bayesian perspective. 

_Generalizability Study:_ To assess generalizability beyond traffic scenarios, we further evaluate our method on the ShanghaiTech dataset. Although our framework is designed for structured traffic anomaly detection with traffic-specific semantic relations, it can be naturally adapted to this dataset by operating on object tracking representations and geometric relation reasoning. As shown in Table VI, our method achieves competitive performance compared with representative baselines, indicating a certain degree of generalization capability in more general surveillance settings. 

_Robustness Analysis:_ To evaluate the robustness of our method, we conduct experiments introducing bbox&relation noise and dropping during inference. For noise corruption, we randomly select bounding boxes and their relations, and then perturb them to simulate tracking jitter. For the missed detection simulation, we randomly drop object bounding boxes and their relations to mimic occlusions and detection failures. As shown in Fig. 8, our method maintains strong performance under all test corruption conditions. This demonstrates the robustness of our framework to imperfect detection and tracking inputs. 

_Inference Time:_ We present the runtime evaluation comparison on the DoTA test set in Table VII. For a fair comparison, all experiments were conducted on an NVIDIA RTX 1080 GPU. Tracking, relation extraction, and optical-flow feature extraction are excluded from the runtime evaluation for both methods and are implemented using the same preprocessing pipeline. Our method achieves an average inference speed of 61.40 FPS with 10.30 M model parameters. In comparison, the FOL baseline runs at 23.98 FPS with 5.02 M parameters. Despite having a 

IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 28, 2026 

10394 



Fig. 10. Visualization of the testing results of our method and FOL [7] on DoTA and MM-AU datasets. Shaded regions indicate ground-truth anomalous intervals. Blue and orange curves represent fused anomaly scores of our method and FOL, respectively. Red and blue bounding boxes denote ground-truth and predicted anomalous objects. Scores above each box correspond to fused object abnormal scores of our method (blue) and predicted scores of FOL (orange). 

larger model size, our method runs nearly three times faster owing to its transformer-based prediction mechanism. These results demonstrate that the proposed structured framework is not only more accurate but also more computationally efficient for traffic anomaly detection. 

## _D. Qualitative Analysis_ 

Fig. 10 presents visualizations of the testing results of our method and FOL [7] on DoTA and MM-AU datasets. As we can see, our method tends to classify objects involved in accidents as abnormal based on their predicted higher abnormal scores. We can detect abnormal events accurately and predict abnormal scores of normal frames that are very close to zero. Specifically, in Fig. 10(b), we can clearly observe that our reasoning method successfully identifies all the objects involved in the accident. Whenaspeedingcar crashes intoanother car, theanomalyscores of both vehicles and the impacted frames increase significantly. Furthermore, as shown in Fig. 10(c), our method predicts higher abnormal scores for abnormal objects and frames in long-term videos of the MM-AU dataset. 

_Limitations and Future Work:_ While our method effectively captures interaction-driven anomalies, its performance may be affected by imperfect perception results, such as inaccurate egovehicle dynamics and tracking failures under partial visibility. As shown in Fig. 11, the anomaly score of the ego car tends to be lower than expected. In future work, we plan to incorporate sensor data, appearance inputs, and environmental factors to enhance scene understanding and improve robustness in these challenging scenarios. 



Fig. 11. Failure case of our method. 

## V. CONCLUSION 

This paper introduces a structured graph reasoning framework forunsupervisedtrafficanomalydetection.Bymodelingthetraffic scene as a domain-specific graph, our approach provides a comprehensive understanding of traffic dynamics. We propose a graph reasoning detector that detects anomalies by comparing observed and predicted graphs. It integrates both node anomaly, i.e., unusual object movements, and edge anomaly, i.e., irregular object interactions, within a principled Bayesian formulation. We introduce the Anomaly Transformer to predict future graphs by attending node and edge queries to object locations and relation histories. Experiments on two public datasets demonstrate the effectiveness of our method, showcasing its superior performance in detecting traffic accidents and abnormal objects involved. While our method effectively detects complex traffic anomalies, its ability to handle partial visibility remains limited, and we plan to integrate sensor, context, and appearance cues in the future. 

10395 

SHI et al.: STRUCTURED GRAPH REASONING FOR TRAFFIC ANOMALY DETECTION 

## REFERENCES 

- [1] Y.-L. Jin, Z.-Y. Ji, D. Zeng, and X.-P. Zhang, “VWP: An efficient DRLbased autonomous driving model,” _IEEE Trans. Multimedia_ , vol. 26, pp. 2096–2108, 2024. 

- [2] K. Lv, H. Sheng, Z. Xiong, W. Li, and L. Zheng, “Improving driver gaze prediction with reinforced attention,” _IEEE Trans. Multimedia_ , vol. 23, pp. 4198–4207, 2021. 

- [3] Y. Yao, M. Xu, Y. Wang, D. J. Crandall, and E. M. Atkins, “Unsupervised traffic accident detection in first-person videos,” in _Proc. 2019 Int. Conf. Intell. Robots Syst._ , 2019, pp. 273–280. 

- [4] P. Wu, W. Wang, F. Chang, C. Liu, and B. Wang, “DSS-Net: Dynamic self-supervised network for video anomaly detection,” _IEEE Trans. Multimedia_ , vol. 26, pp. 2124–2136, 2024. 

- [5] W. Liu, W. Luo, D. Lian, and S. Gao, “Future frame prediction for anomaly detection–A new baseline,” in _Proc. 2018 IEEE Conf. Comput. Vis. Pattern Recognit._ , 2018, pp. 6536–6545. 

- [6] N. Li, F. Chang, and C. Liu, “Spatial-temporal cascade autoencoder for video anomaly detection in crowded scenes,” _IEEE Trans. Multimedia_ , vol. 23, pp. 203–215, 2021. 

- [7] Y. Yao et al., “DoTA: Unsupervised detection of traffic anomaly in driving videos,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 45, no. 1, pp. 444–459, Jan. 2023. 

- [8] Y. S. Chong and Y. H. Tay, “Abnormal event detection in videos using spatiotemporal autoencoder,” in _Proc. 2017 Int. Symp. Neural Netw._ , 2017, pp. 189–196. 

- [9] T. Reiss and Y. Hoshen, “An attribute-based method for video anomaly detection,” _Trans. Mach. Learn. Res._ , vol. 2025, Jan. 2025. 

- [10] A. Singh, M. J. Jones, and E. G. Learned-Miller, “EVAL: Explainable video anomaly localization,” in _Proc. 2023 IEEE Conf. Comput. Vis. Pattern Recognit._ , 2023, pp. 18717–18726. 

- [11] J. Fang, J. Qiao, J. Bai, H. Yu, and J. Xue, “Traffic accident detection via self-supervised consistency learning in driving scenarios,” _IEEE Trans. Intell. Transp. Syst._ , vol. 23, no. 7, pp. 9601–9614, Jul. 2022. 

- [12] R. Liang, Y. Li, Y. Yi, J. Zhou, and X. Li, “A memory-augmented multitask collaborative framework for unsupervised traffic anomaly detection in driving videos,” _Pattern Recognit._ , vol. 168, 2025, Art. no. 111789. 

- [13] C. Sun, Y. Jia, Y. Hu, and Y. Wu, “Scene-aware context reasoning for unsupervised abnormal event detection in videos,” in _Proc. 2020 ACM Int. Conf. Multimedia_ , 2020, pp. 184–192. 

- [14] F.Mumcu,M.Jones,Y.Yilmaz,andA.Cherian,“ComplexVAD:Detecting interaction anomalies in video,” 2025, pp. 1093–1102. 

- [15] K. Doshi and Y. Yilmaz, “Towards interpretable video anomaly detection,” 2023, pp. 2655–2664. 

- [16] C. Huang et al., “Multimodal evidential learning for open-world weaklysupervised video anomaly detection,” _IEEE Trans. Multimedia_ , vol. 27, pp. 3132–3143, 2025. 

- [17] S. Chang, Y. Li, S. Shen, J. Feng, and Z. Zhou, “Contrastive attention for video anomaly detection,” _IEEE Trans. Multimedia_ , vol. 24, pp. 4067–4076, 2022. 

- [18] R. Liang, Y. Li, J. Zhou, and X. Li, “Text-driven traffic anomaly detection with temporal high-frequency modeling in driving videos,” _IEEE Trans. Circuits Syst. Video Technol._ , vol. 34, no. 9, pp. 8684–8697, Sep. 2024. 

- [19] K. Saleh, A. Grigorev, and A.-S. Mihaita, “Traffic accident risk forecasting using contextual vision transformers,” in _Proc. 2022 IEEE 25th Int. Conf. Intell. Transp. Syst._ , 2022, pp. 2086–2092. 

- [20] M. Naphade et al., “The 4th AI city challenge,” in _Proc. 2020 IEEE Conf. Comput. Vis. Pattern Recognit. Workshops_ , Jun. 2020, pp. 2665–2674. 

- [21] H. Shi, L. Wang, S. Zhou, G. Hua, and W. Tang, “Abnormal ratios guided multi-phase self-training for weakly-supervised video anomaly detection,” _IEEE Trans. Multimedia_ , vol. 26, pp. 5575–5587, 2023. 

- [22] J. Meng, H. Tian, G. Lin, J.-F. Hu, and W.-S. Zheng, “Audiovisual collaborative learning for weakly supervised video anomaly detection,” _IEEE Trans. Multimedia_ , early access, Jan. 27, 2025, doi: 10.1109/TMM.2025.3535377. 

- [23] Z. Liu, Y. Nie, C. Long, Q. Zhang, and G. Li, “A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flowguided frame prediction,” in _Proc. 2021 Int. Conf. Comput. Vis_ ., 2021, pp. 13588–13597. 

- [24] H. Park, J. Noh, and B. Ham, “Learning memory-guided normality for anomaly detection,” in _Proc. 2020 IEEE Conf. Comput. Vis. Pattern Recognit._ , 2020, pp. 14372–14381. 

- [25] L. Wang, J. Tian, S. Zhou, H. Shi, and G. Hua, “Memory-augmented appearance-motion network for video anomaly detection,” _Pattern Recognit._ , vol. 138, 2023, Art. no. 109335. 

- [26] I. Goodfellow et al., “Generative adversarial networks,” _Commun. ACM_ , vol. 63, no. 11, pp. 139–144, 2020. 

- [27] B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,” in _Proc. 27th Int. Joint Conf. Artif. Intell._ , 2018, pp. 3634–3640. 

- [28] N. Chen, Z. Du, and K. H. Ng, “Scene graphs for interpretable video anomaly classification,” in _Proc. 2018 NeurIPS Workshop_ , 2018. 

- [29] D. Das et al., “UApredictor: Urban anomaly prediction from spatialtemporal data using graph transformer neural network,” in _Proc. 2022 Int. Joint Conf. Neural Netw._ , 2022, pp. 1–8. 

- [30] S. Krishnan et al., “FourCastNet: A global data-driven high-resolution weather model using adaptive fourier neural operators,” in _Proc. 2022 Int. Conf. Learn. Representations_ , 2022, pp. 1–11. 

- [31] Y. Li, P. Wang, and C.-Y. Chan, “RESTEP into the future: Relational spatio-temporal learning for multi-person action forecasting,” _IEEE Trans. Multimedia_ , vol. 25, pp. 1954–1963, 2023. 

- [32] Y. Yuan, X. Weng, Y. Ou, and K. M. Kitani, “AgentFormer: Agent-aware transformers for socio-temporal multi-agent forecasting,” in _Proc. 2021 Int. Conf. Comput. Vis_ ., 2021, pp. 9813–9823. 

- [33] Y. Dong, L. Wang, S. Zhou, G. Hua, and C. Sun, “Recurrent aligned network for generalized pedestrian trajectory prediction,” _IEEE Trans. Circuits Syst. Video Technol._ , vol. 35, no. 3, pp. 2188–2201, Mar. 2025. 

- [34] P. Xu, J.-B. Hayet, and I. Karamouzas, “SocialVAE: Human trajectory prediction using timewise latents,” in _Proc. 2022 Eur. Conf. Comput. Vis_ ., 2022, pp. 511–528. 

- [35] A. V. Malawade et al., “roadscene2vec: A tool for extracting and embedding road scene-graphs,” _Knowl.-Based Syst._ , vol. 242, 2022, Art. no. 108245. 

- [36] R. M. Neal, _Bayesian Learning for Neural Networks_ , vol. 118. Berlin, Germany: Springer, 2012. 

- [37] Y.-T. Chen et al., “Multimodal object detection via probabilistic ensembling,” in _Proc. 2022 Eur. Conf. Comput. Vis_ ., 2022, pp. 139–158. 

- [38] O. Dagan and N. R. Ahmed, “Exact and approximate heterogeneous Bayesian decentralized data fusion,” _IEEE Trans. Robot._ , vol. 39, no. 2, pp. 1136–1150, Feb. 2022. 

- [39] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, “Learning temporal regularity in video sequences,” in _Proc. 2016 IEEE Conf. Comput. Vis. Pattern Recognit._ , 2016, pp. 733–742. 

- [40] Y. Yao et al., “Egocentric vision-based future vehicle localization for intelligent driving assistance systems,” in _Proc. 2019 Int. Conf. Robot. Automat._ , 2019, pp. 9711–9717. 

- [41] R. T. Ionescu, F. S. Khan, M.-I. Georgescu, and L. Shao, “Object-centric auto-encoders and dummy anomalies for abnormal event detection in video,” in _Proc. 2019 IEEE Conf. Comput. Vis. Pattern Recognit._ , 2019, pp. 7842–7851. 

- [42] N. Carion et al., “End-to-end object detection with transformers,” in _Proc. 2020 Eur. Conf. Comput. Vis_ ., 2020, pp. 213–229. 

- [43] J. Fang et al., “Abductive ego-view accident video understanding for safe driving perception,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit._ , 2024, pp. 22030–22040. 

- [44] L.-L. Li, J. Fang, and J. Xue, “Cognitive traffic accident anticipation,” _IEEE Intell. Transp. Syst. Mag._ , vol. 16, no. 5, pp. 17–32, 2024. 

- [45] J. Fang, D. Yan, J. Qiao, J. Xue, and H. Yu, “DADA: Driver attention prediction in driving accident scenarios,” _IEEE Trans. Intell. Transp. Syst._ , vol. 23, no. 6, pp. 4959–4971, Jun. 2021. 

- [46] D. Gong et al., “Memorizing normality to detect anomaly: Memoryaugmented deep autoencoder for unsupervised anomaly detection,” in _Proc. 2019 IEEE Conf. Comput. Vis. Pattern Recognit._ , 2019, pp. 1705–1714. 

**Haoyue Shi** received the Ph.D. degree in control science and engineering from the Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, Xi’an, China, in 2025. From 2023 to 2024, she was a Visiting Ph.D. degree with the University of Illinois at Chicago, Chicago, Illinois, USA. She is currently an Assistant Professor with the School of Electronics and Control Engineering, Chang’an University. Her research interestsinclude anomaly detection, computer vision, and machine learning. 



10396 

IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 28, 2026 



**Le Wang** (Senior Member, IEEE) received the B.S. and Ph.D. degrees in control science and engineering from Xi’an Jiaotong University, Xi’an, China, in 2008 and 2014, respectively. From 2013 to 2014, he was a Visiting Ph.D. degree with the Stevens Institute of Technology, Hoboken, NJ, USA. From 2016 to 2017, he was a Visiting Scholar with Northwestern University, Evanston, IL, USA. He is currently a Professor with the Institute of Artificial IntelligenceandRobotics,Xi’anJiaotongUniversity,Xi’an, China.Hisresearchinterestsincludecomputervision, 



**Zhanheng Gao** received the B.S. degree in information engineering from the Huazhong University of Science and Technology, Wuhan, China, in 2021. He is currently working toward the M.S. degree with the Institute of Artificial Intelligence and Robotics of Xi’an Jiaotong University, Xi’an, China. His research interests include computer vision, machine learning, and pattern recognition. 

pattern recognition, and machine learning. 

**Sanping Zhou** (Member, IEEE) received the Ph.D. degree in control science and engineering from Xi’an Jiaotong University, Xi’an, China, in 2020. From 2018 to 2019, he was a Visiting Ph.D. degree with the Robotics Institute, Carnegie Mellon University. He is currently an Assistant Professor with the Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University. His research interests include machine learning, deep learning, and computer vision, with a focus on person re-identification, salient object detection, medical image segmentation, image clas- 



sification, and visual tracking. 

**Fei Hui** received the Ph.D. degree from the Department of Computer System Architecture, Xi’an Institute of Microelectronics Technology, Xi’an, China, in 2009. He is currently a Professor with the School of Electronics and Control Engineering, Chang’an University, Xi’an. His research interests include connected vehicles, intelligent transportation systems, and autonomous driving virtual simulation testing. He has been involved with China “863” Project, the Information Technology Major Project of the Transportation Ministry, National Natural Science Foundation, and the National K & R Programmer of China. 

**Gang Hua** (Fellow, IEEE) received the B.S. and M.S. degrees in automatic control engineering from Xi’an Jiaotong University (XJTU), Xi’an, China, in 1999 and 2002, respectively, and the Ph.D. degree in electrical engineering and computer science from Northwestern University, Evanston, IL, USA, in 2006. He is currently the Vice President and Chief Scientist of Wormpex AI Research. He has authored of more than 200 peer reviewed publications in prestigious international journals and conferences. He also holds 19 U.S. Patents and has 15 more U.S. patents pending. 



He is also an Associate Editor for IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE and MVA. He is a General Chair of ICCV’2025, Program Chair of CVPR’2019&2022, and Area Chair of CVPR’2015&2017 and ICCV’2011&2017. He was the recipient of the 2015 IAPR Young Biometrics Investigator Award. He is an IAPR Fellow and an ACM Distinguished Scientist. 

**Wei Tang** (Member, IEEE) received the B.E. and M.E. degrees from Beihang University, Beijing, China, in 2012 and 2015, respectively, and the Ph.D. degree in electrical engineering from Northwestern University, Evanston, IL, USA, in 2019. He is currently an Assistant Professor with the Department of Computer Science, University of Illinois at Chicago, Chicago, IL. His research interests include computer vision, pattern recognition, and machine learning. 



