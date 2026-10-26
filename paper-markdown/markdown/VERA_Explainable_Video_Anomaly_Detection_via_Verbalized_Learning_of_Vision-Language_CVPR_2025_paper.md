This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models** 

Muchao Ye<sup>1*</sup> Weiyang Liu<sup>2</sup> Pan He<sup>3</sup> 

1The University of Iowa 2Max Planck Institute for Intelligent Systems, T¨ubingen 3Auburn University 1muye@uiowa.edu 2weiyang.liu@tuebingen.mpg.de 3pan.he@auburn.edu _⇤_ Corresponding Author https://vera-framework.github.io 

## **Abstract** 

_The rapid advancement of vision-language models (VLMs) has established a new paradigm in video anomaly detection (VAD): leveraging VLMs to simultaneously detect anomalies and provide comprehendible explanations for the decisions. Existing work in this direction often assumes the complex reasoning required for VAD exceeds the capabilities of pretrained VLMs. Consequently, these approaches either incorporate specialized reasoning modules during inference or rely on instruction tuning datasets through additional training to adapt VLMs for VAD. However, such strategies often incur substantial computational costs or data annotation overhead. To address these challenges in explainable VAD, we introduce a verbalized learning framework named VERA that enables VLMs to perform VAD without model parameter modifications. Specifically, VERA automatically decomposes the complex reasoning required for VAD into reflections on simpler, more focused guiding questions capturing distinct abnormal patterns. It treats these reflective questions as learnable parameters and optimizes them through data-driven verbal interactions between learner and optimizer VLMs, using coarsely labeled training data. During inference, VERA embeds the learned questions into model prompts to guide VLMs in generating segment-level anomaly scores, which are then refined into frame-level scores via the fusion of scene and temporal contexts. Experimental results on challenging benchmarks demonstrate that the learned questions of VERA are highly adaptable, significantly improving both detection performance and explainability of VLMs for VAD._ 

## **1. Introduction** 

Video anomaly detection (VAD) aims to automatically identify unexpected and abnormal events in video sequences, with broad applications ranging from autonomous driving [2] to industrial manufacturing [31]. While achieving good performance in VAD is essential, providing clear explanations for detected anomalies is even more crucial. 

To this end, our work primarily focuses on explain- 



<!-- Start of picture text -->
Existing pipeline 1 uses external LLMs to assist frozen VLMs in reasoning<br>…<br>… DescribeVLM  ❄ ReasonLLM ❄ or " …Abnormal Video… …<br>Test Video Training Sets with Coarse Labels<br>Existing pipeline 2 utilizes instruction tuning to make VLMs integrated VAD systems<br>…<br>VLM  " " … … …<br>… Describe & Reason Normal Abnormal Abnormal Normal<br>Test Video # Instruction Tuning Datasets with Frame-Level Annotations<br>VERA employs learnable guiding questions to elicit the reasoning of frozen VLMs<br>VAD Guiding Questions "<br>… VLM  ❄ … … …<br>… Abnormal Video<br>Describe & Reason<br>Test Video Training Sets with Coarse Labels<br>Inference Training ❄ Frozen Module " Learnable Module # Extra Annotation<br><!-- End of picture text -->

Figure 1. VERA renders frozen VLMs to describe and reason with learnable guiding questions learned from coarsely labeled data. 

able VAD, which requires both comprehensive visual understanding and the ability to generate human-interpretable predictions. The rapid advancement of vision-language models (VLMs) [7, 18, 21, 58] enables us to address both requirements through their strong visual reasoning and language interaction capabilities. As multi-modal architectures that effectively combine the reasoning capabilities from large language models (LLMs) [4] and the visual understanding capabilities from pretrained vision encoders [8], VLMs are particularly well-suited for VAD for they can offer explainable predictions that clearly illustrate the rationale behind specific anomalies, making the results more interpretable to users. Recent research on VAD has consequently focused on how to effectively leverage the power of pretrained VLM. As shown in Fig. 1, existing approaches aim to address the misalignment problem between VLMs’ pretraining tasks and the VAD requirements through either additional reasoning modules or instruction tuning (IT): 

- One line of research _introduces external LLMs to assist frozen VLMs to reason in VAD_ [46, 52]. It uses VLMs to caption what they see given a video, and the descriptions are then passed to an external LLM, _e.g_ ., GPT-4 [1], to reason whether an anomaly occurs. 

- Another line of research, instead, _expands VLMs to generate explainable prediction via IT_ [26, 55]. This research line creates additional VAD datasets with frame- 

8679 

level annotations and leverages exemplary instructions to fine-tune the VLM, enabling it to detect anomalies and generate human-interpretable explanations. 

**Key Observations and Research Question** . While prior research demonstrates the potential of applying VLMs to VAD, we identify that this new paradigm is hindered by a shared critical issue: the use of additional reasoning modules or fine-grained labeled datasets incurs significant computational cost either in the inference or training phases. First, decoupling a VAD system into a frozen VLM and an extra LLM introduces more overhead in inference, because it separates the description generation and reasoning processes. Secondly, although IT-based methods enable VLMs to effectively integrate description and reasoning for VAD, they require additional manpower and computational resources for annotating and finetuning on fine-grained labeled instruction datasets, which is time-consuming and not scalable for large-scale datasets. In light of this, we investigate the following unexplored yet important question: 

_Can we enable a frozen VLM to integrate description and reasoning for VAD without instruction tuning?_ 

**Our Approach** . This research question is nontrivial because the reasoning ability of a frozen VLM is limited in general visual tasks, and it struggles to handle complex reasoning tasks like VAD, which requires the understanding of subtle, context-dependent outliers. To illustrate, Table 1 shows that prompting frozen VLMs with simple VAD questions used in existing works leads to unsatisfactory results. Thus, instruction-tuning a VLM seems necessary to make it responsive to specific instructional cues and capture delicate visual variations. In this paper, we question the necessity of such an operation and propose a principled approach to tailor frozen VLMs for VAD. 

Specifically, our solution is guided by the intuition that the reasoning ability of VLMs for VAD will improve if we find questions with suitable and concrete description of abnormal patterns rather than with abstract and general words like “anomaly” to prompt them. Our idea is to iteratively refine anomaly descriptions from abstract ones ( _e.g_ ., “is there any anomaly?”) to detailed, specific characterizations. 

Driven by such insight, we propose a framework, termed VERA, to explore verbalized learning (VL) for VAD. This framework considers the practical constraint that it is suboptimal to manually write down VAD guiding questions across VLMs, so it introduces a data-driven learning task to identify suitable anomaly-characterization questions containing concrete abnormal patterns for the frozen VLM using coarsely labeled datasets, eliminating the need for IT. Specifically, in the training <u>phase,</u> VERA treats the questions guiding the reasoning of VLMs in VAD as learnable parameters, improving them based on the verbal feedback from an optimizer VLM on the performance of a learner 

VAD Question for InternVL2-8B AUC (%) “Describe the video and is there any anomaly?” [26] 53.05 “Are there any abnormal events in the video?” [55] 65.03 Table 1. Instructing a frozen VLM (InternVL2-8B [7]) with simple questions to perform VAD yields poor AUC on UCF-Crime [32] dataset. 

VLM on an intermediate VAD subtask—binary video classification for each video in the VAD training set. This design is both efficient and appropriate for VAD, as it accounts for video-specific properties like temporality while relying solely on provided coarse video-level labels. After that, considering the large scale of video frames, VERA assigns a fine-grained anomaly score for each frame in a coarse-tofine manner in the inference phase. First, VERA generates segment-level anomaly scores by querying VLMs with the learned guiding questions. Next, VERA improves the initial score by incorporating scene context into each segment score via ensembling. Finally, VERA outputs frame-level scores by fusing temporal context via Gaussian smoothing and frame-level position weighting. 

**Contributions** . To sum up, our contributions are: 

- To our knowledge, we present the first approach, that is, VERA, to adapt frozen VLMs as an integrated system for VAD by learning detailed anomaly-characterization questions in prompts that decompose anomalies into concrete and recognizable patterns. VERA learns them directly from coarsely labeled datasets, eliminating the need for IT or external reasoning modules. 

- We introduce an effective VL-based algorithm for VLMs in VAD, allowing direct adaptation without modifying model parameters. With coarse labeled VAD datasets only, our approach obtains good guiding questions in VAD by relying on the verbal interaction between learner and optimizer VLMs in verbalized training. Additionally, we design a coarse-to-fine strategy to derive frame-level anomaly scores from verbally learned guiding questions in VAD, integrating both scene and temporal contexts for better VAD performance and reasoning. 

- The learned guiding questions from VERA are expressed in natural languages, providing a unified method to encode and transfer prior VAD knowledge seamlessly to other datasets or VLMs. In challenging VAD datasets like UCF-Crime [32] and XD-Violence [42], VERA achieves state-of-the-art explainable VAD performance and enjoys good generalization ability across models and datasets. 

## **2. Related Work** 

**Video Anomaly Detection** . VAD is the task of localizing frames that contain abnormal events in a given video. This task is challenging for anomalies cover a broad scope of events like accidents and criminal activities while training sets only offer coarse annotations. Modern VAD methods are based on deep neural networks (DNNs) for their superi- 

8680 

ority and are going through a paradigm shift in using VLMs: (1) Early DNNs for VAD are task-specific, which often employ unsupervised (including one-class) or weakly supervised (WS) learning techniques for training. Most unsupervised learning methods [23, 25, 37, 38, 48, 56] train DNNs on frame reconstruction/prediction tasks to establish representation spaces for normal/abnormal videos. WS learning methods [5, 27, 32, 44, 47, 53] leverage both normal and abnormal videos to train a feature extractor that distinguishes anomalies from normalcy, typically using multiple instance learning [32] objectives. (2) Recent VAD methods adopt VLMs due to their remarkable success across core vision tasks [12, 21, 28, 33]. Early research [26, 46, 52, 55] has leveraged VLMs to generate textual descriptions of detected anomalies to enhance prediction explainability for VAD. However, current approaches incur high processing demands from external LLMs or require substantial effort and cost for fine-tuning on additional datasets, which are computationally inefficient in training or inference. Our work reduces the processing overhead by adapting frozen VLMs for VAD without model parameter modification or extra reasoning modules via learnable guiding questions, which elicit superior reasoning from frozen VLMs and significantly boost their performance in VAD. 

**Verbalized Learning for VLMs** . The designed VL framework is inspired by a recent technique called verbalized machine learning (VML) [45]. The main idea of VML is to use LLMs to approximate functions and learn the verbal rules and descriptions of performing specific tasks, which casts traditional machine learning tasks as language-based learning tasks. This approach regards the language expressions that define classification rules and other task-specific criteria as learned parameters, and optimize them in a datadriven fashion through interactions between a learner and an optimizer modeled by LLMs or VLMs. However, the VML framework is limited to tasks involving regression on scalar values or classification for static images. A similar idea has also been explored in a concurrent method, TextGrad [49], which integrates the process of incorporating textual feedback from LLMs for improving prompts in PyTorch and further proves its effectiveness in coding, question answering, and optimization in chemistry and medicine. Compared to existing works, our work pioneers VL for the VAD task and video data, which remains unsolved for previous VL frameworks focus on static-data tasks and cannot handle the challenges of temporality and scene dynamics in videos. Specifically, VERA introduces a new learning paradigm for VAD: generating effective questions that encapsulate key abnormal patterns in videos to elicit the reasoning ability from VLMs for explainable VAD. Additionally, VERA works for any VAD dataset and supports WS learning. Unlike previous WS methods, VERA only needs to learn concise text but not millions of parameters, so the training is lightweight. 

## **3. The VERA Framework** 

Our approach adapts VLMs to detect video anomalies without additional reasoning modules or IT. We now formulate the VAD task and detail the design of VERA. 

### **3.1. Problem Formulation** 

**Video Anomaly Detection** . Let _V_ be a video with _F_ frames, represented as _V_ = _{Ii}_<sup>_F_</sup> _i_ =1<sup>,where</sup><sup>_Ii_isthe</sup><sup>_i_-th</sup> frame (1 _ i  F_ ). Our objective is to locate and detect the start and end of anomalous events within _V_ . In standard labeling, any frame associated with an anomaly is labeled as 1, and normal frames are labeled as 0. Therefore, the ground truth label sequence for _V_ is _Y_ = [ _y_ 1 _, . . . , yF_ ], where _yi 2 {_ 0 _,_ 1 _}_ represents the fine-grained label for _Ii_ . We aim to use a frozen VLM, _f_ VLM, to generate anomaly score predictions across all frames, _Y_<sup>ˆ</sup> = [ˆ _y_ 1 _, . . . ,_ ˆ _yF_ ], where _y_ ˆ _i 2_ [0 _,_ 1] is a continuous anomaly score for _Ii_ . 

**Available Training Data for VAD** . Typically, VAD datasets only provide coarsely labeled training sets [23, 25, 32, 42]. We denote a VAD training set as _D_ = _{_ ( _V_<sup>(</sup><sup>_j_)</sup> _, Y_<sup>(</sup><sup>_j_)</sup> ) _}_<sup>_N_</sup> _j_ =1<sup>,where</sup><sup>_N_isthetotalnumberoftrain-</sup> ing videos, _V_<sup>(</sup><sup>_j_)</sup> represents the _j_ -th video (1 _ j  N_ ) and _Y_<sup>(</sup><sup>_j_)</sup> is the corresponding video-level label. _Y_<sup>(</sup><sup>_j_)</sup> = 1 if _V_<sup>(</sup><sup>_j_)</sup> contains any anomaly defined by the dataset annotators, _e.g_ ., abuse or arson activities, and _Y_<sup>(</sup><sup>_j_)</sup> = 0 if _V_<sup>(</sup><sup>_j_)</sup> has no anomalies. For _V_<sup>(</sup><sup>_j_)</sup> , we suppose it contains _Fj_ frames and denote the frames sequence as _V_<sup>(</sup><sup>_j_)</sup> = _{Ii_<sup>(</sup><sup>_j_)</sup> _}Fi_ =1 _j_<sup>, where</sup> _Ii_<sup>(</sup><sup>_j_)</sup> is the _i_ -th frame (1 _ i  Fj_ ) in _V_<sup>(</sup><sup>_j_)</sup> . 

### **3.2. Training in VERA** 

**Training Objective** . We aim to learn guiding questions that break down a complex and ambiguous concept ( _i.e_ ., what is an “anomaly”) into a set of identifiable anomalous patterns to unlock reasoning capabilities within frozen VLMs for VAD tasks. Those patterns vary among datasets, making manually designed descriptions ineffective for generalization. To address this, we propose a general VL framework shown in Fig. 2 to generate the desired guiding questions. We denote the guiding question set as **Q** = _{q_ 1 _, . . . , qm}_ , where _qi_ is the _i_ -th question (1 _ i  m_ ) and _m_ is the number of questions. The training framework considers **Q** as the learnable parameters, which are optimized through verbal interaction between a learner and an optimizer, modeled by VLMs through leveraging their ability to follow instructions with given prompts. 

**Training Data** . The training data for learning **Q** consist of paired sampled video frames and video-level labels. Sampling is necessary because the amount of video frames is so huge that we cannot compute with every frame. We explore three types of sampling strategies and find that uniform sampling [54] yields the best results. That is, with any video _V_<sup>(</sup><sup>_j_)</sup> _2 D_ , we first calculate the interval between 

8681 



<!-- Start of picture text -->
Input Video Frames  " ! (") Guiding Questions  (+  (Learnable Parameters) Forward Pass for Prediction<br>1. Are there any unusual movements of the vehicles in the video frames? Backward Pass for Optimization<br>2. Are there any 3. Are there any 4. Are there any unusual behaviors of the peopleunusual objects or items that are not in their usual positionsobjects or people that appear out of placein the video frames? in the scene? ? Update Optimizer  '&'#())<br>Ground Truth :  % (") 5. Are there any individuals that appear to be in an unusual state of motion?<br>Batched Inputs, Predictions,<br>! Learner  ',-"./-.()) Prediction  & % (0) and Ground Truths  "<br>#!"#$%, & % !"#$%, &!"#$%<br>Learner Prompt Template  " Optimizer Prompt Template  !<br>You are the model. You are the optimizer for a model. Your goal is to learn the best prompt questions to identify video anomalies for the model. The model used<br>Model Description:  You are designed to do binary  the Current Prompt Questions below to predict the class labels for the given inputs. You are given the target labels. Please optimize the<br>classification. The input is a sequence of video frames for  Current Prompt Questions for better prediction.<br>identifying whether there is an anomaly in the video; you  Inputs: [A batch of input video frames  "%&'() ]<br>need to output the class label, i.e., an integer in the set {0,  Model Description:  You are designed to do binary classification. The input is a sequence of video frames for identifying whether there is an<br>1}. 0 represents normal video, and 1 represents abnormal  anomaly in the video; you need to output the class label, i.e., an integer in the set {0, 1}. 0 represents normal video, and 1 represents abnormal<br>video. Please answer the prompt questions. video. Please answer the prompt questions.<br>Prompt Questions:  Answer the following questions based  Current Prompt Questions:  Answer the following questions based on what you see from the video frames and provide an explanation in one<br>on what you see from the video frames and provide an  sentence.<br>explanation in one sentence. [Guiding Questions  #$]<br>[Guiding Questions  #$] Based on the analysis above, please conclude your answer to “Is there any anomaly in the video?” in “Yes, there is an anomaly” or “No, there<br>Based on the analysis above, please conclude your answer  is no anomaly”.<br>to “Is there any anomaly in the video?” in “Yes, there is an  The Model Predictions: [A batch of predictions  % & %&'()  made by the learner]<br>anomaly” or “No, there is no anomaly”. The Targets:  [A batch of video-level ground truths  %%&'()  for the input video frames]<br>Input:  [Video Frames  " ! (") ] If the model is doing well, you can keep using the current prompt questions. However, if the model is not performing well, please update the<br>Please give your output strictly in the following format: model by improving upon the “Current Prompt Questions”, which should result in lower classification error both on the current and the next<br>• Answers to Prompt Questions : [Provide your analysis  batch of i.i.d. data. Limit your “New Prompt Questions” to be no more than five questions! Please think step by step and give your outputs<br>by answering the questions listed in Prompt Questions.] strictly in the following format:<br>• Output : [ONLY the integer class label; make necessary  • Reasoning : [Be explicit and verbose, improve the Current Prompt Questions by yourself; Please show your work and use the features in<br>assumptions if needed] the videos; note that you don't have access to computers]<br>Please ONLY reply according to this format. Don't give  • New Prompt Questions : [Put your new prompt questions here. The questions MUST be based on the features in the input videos. Please<br>me any other words. limit the number of questions to be at most five!]. Please ONLY reply according to this format. Don't give me any other words.<br><!-- End of picture text -->

Figure 2. The overall training pipeline in VERA aims to optimize VAD guiding questions iteratively. In each iteration _t_ , the optimization is verbalized by providing verbal instructions for the learner and optimizer to follow. They will generate predictions and new guiding questions, respectively. 

sampled frames as _l_ = floor( _Fj/S_ ), where _S_ is the number of sampled frames, and floor denotes rounding down to the nearest integer. Given _l_ , the uniformly sampled frames from _V_<sup>(</sup><sup>_j_)</sup> are represented by _V_<sup>˜(</sup><sup>_j_)</sup> = [ _I_ 1<sup>(</sup><sup>_j_)</sup><sup>_, I_</sup> _l_<sup>(</sup> +1<sup>_j_)</sup><sup>_, . . . , I_</sup> (<sup>(</sup> _S_<sup>_j_)</sup> _−_ 1) _·l_ +1<sup>].</sup> The label used for training is _Y_<sup>(</sup><sup>_j_)</sup> only, resulting in training data pairs _{_ ( _V_<sup>˜(</sup><sup>_j_)</sup> _, Y_<sup>(</sup><sup>_j_)</sup> ) _}_<sup>_N_</sup> _j_ =1<sup>for VERA.</sup> 

**Updating Q via Learner and Optimizer** . Since **Q** are verbal expressions for specific anomaly patterns, VERA inherits the idea of VML [45] in training: optimizing language-based parameters by verbal communication between a learner agent _f_ learner and an optimizer agent _f_ opt, rather than by numerical optimization algorithms like Adam [16]. W.l.o.g., we take an arbitrary iteration _t_ when implementing the complete algorithm (detailed in Supplementary Material) for illustration. We denote any LLMbased model as _f_ ( _x_ ; _φ_ ) where _x_ represents the input data, and _φ_ denotes the natural language instructions for _f_ to follow, which is considered as learnable parameters in our VL framework. Specifically, **Q** contains parameters to be learned in VERA. As depicted in Fig. 2, in each iteration _t_ , the learner agent _f_ learner<sup>(</sup><sup>_t_)ismodeledbythefrozenVLM</sup> _f_ VLM( _·_ ) used for VAD with a specific prompt template _✓_ that guide _f_ VLM( _·_ ) to conduct a learning task by pondering on current guiding questions **Q** _t_ . We denote the learner agent as _f_ learner<sup>(</sup><sup>_t_)(</sup><sup>_x_) =</sup><sup>_f_VLM(</sup><sup>_x_; (</sup><sup>_✓,_</sup><sup>**Q**</sup><sup>_t_)), where</sup><sup>_x_is the input</sup> in a learning task, and **Q** _t_ , the learnable guiding questions applied in each iteration _t_ , constitutes the core parameters that distinguish the learner between iterations. Meanwhile, we introduce an optimizer _f_ opt<sup>(</sup><sup>_t_)toassessthequalityofthe</sup> 

predictions of the learner and to optimize **Q** _t_ . W.l.o.g., we use the same frozen VLM _f_ VLM to model the optimizer. As demonstrated in Fig. 2, we provide another specific prompt template for the learner to follow to optimize **Q** _t_ , so we denote the optimizer agent as _f_ opt<sup>(</sup><sup>_t_)(</sup><sup>_z_)=</sup><sup>_f_VLM(</sup><sup>_z_; (</sup><sup>_,_</sup><sup>**Q**</sup><sup>_t_)),</sup> where _z_ is its input and is the instruction to improve **Q** _t_ . It is important to note that _f_ learner<sup>(</sup><sup>_t_)</sup><sup>_6_=</sup><sup>_f_(</sup> opt<sup>_t_)because</sup><sup>_f_(</sup> learner<sup>_t_)fol-</sup> lows ( _✓,_ **Q** _t_ ) to conduct a learning task, while _f_ opt<sup>(</sup><sup>_t_)follows</sup> ( _,_ **Q** _t_ ) to refine **Q** _t_ . 

_Learning Task for_ _<u>f</u>_ learner. The learner executes the “forward pass” and outputs a prediction. Recall that we only use the original coarsely labeled information for training. Thus, we design a binary classification task for _f_ learner, which accounts for the temporal nature of video data, the sparsity of anomalies, and the weak supervision in VAD datasets. In this task, the job of the learner _f_ learner is to produce a binary classification prediction _Y_<sup>ˆ(</sup><sup>_j_)</sup> to determine whether there is an anomaly in the video based on the sampled frames _V_<sup>˜(</sup><sup>_j_)</sup> . As shown in Fig. 2, we explain the task in natural language in the “Model Description” section in _✓_ . Guiding questions **Q** _t_ are inserted in the “Prompt Questions” section in _✓_ to elicit reasoning of the VLM. This template design is based on the prompt structures used in VML, with targeted modifications to help the learner effectively address this WS learning task. Given _✓_ and a sampled frame set _V_<sup>˜(</sup><sup>_j_)</sup> , the learner will output a prediction as 



where _Y_<sup>ˆ(</sup><sup>_j_)</sup> = 1 if the learner thinks there is an anomaly af- 

8682 

ter skimming across the sampled frames _V_<sup>˜(</sup><sup>_j_)</sup> and reasoning through the guiding questions **Q** _t_ , and otherwise, _Y_<sup>ˆ</sup> _i_ = 0. _Optimization Step in_ _<u>f</u>_ opt. The optimizer executes the “backward pass” to update the questions **Q** _t_ via a mini-batch (batch size is _n_ ). Suppose the visual = input in a batch is _V_ batch [ _V_<sup>˜</sup> batch<sup>(1)</sup><sup>_, · · ·,V_˜</sup> batch<sup>(</sup><sup>_n_)]</sup> = and the corresponding ground truths are _Y_ batch [ _Y_ batch<sup>(1)</sup><sup>_, · · ·, Y_</sup> batch<sup>(</sup><sup>_n_)].Thelearnergeneratespredictionas</sup> _Y_ ˆbatch = [ ˆ _Y_ batch<sup>(1)</sup><sup>_, · · ·,Y_ˆ</sup> batch<sup>(</sup><sup>_n_)]withthecurrentquestions</sup> **Q** _t_ by Eq. (1). The optimizer will output a new set of questions **Q** _t_ +1 by following the prompt with batched data. We denote the optimization step as 



where **Q** _t_ +1 is a new set of guiding questions constructed from _f_ opt<sup>(</sup><sup>_t_)owingtoitstextgenerationandinstructionfol-</sup> lowing abilities after reading . 

### **3.3. Inference in VERA** 

During training, we denote the one with the largest validation accuracy as **Q**<sup>_⇤_</sup> . In inference, given **Q**<sup>_⇤_</sup> , VERA yields fine-grained anomaly score _Y_<sup>ˆ</sup> for a test video _V_ via a coarse-to-fine process shown in Fig. 3. 

**Step 1: Initial Anomaly Scores via Learned Guiding Questions** . We divide the video into segments and analyze each segment independently first. Following [52], we perform equidistant frame sampling within _V_ to obtain the set of segment centers _C_ = _{I_ 1 _, Id_ +1 _, · · · , I_ ( _h−_ 1) _·d_ +1 _}_ , where _d_ is the interval between centers and _h_ = floor( _F/d_ ) is the total number of segments. For each center frame _I_ ( _u−_ 1) _·d_ +1 (1 _ u  h_ ), we define a 10-second window around it as the _u_ -th segment, within which we uniformly sample 8 frames. We denote the sampled frame set in the _u_ -th segment as _Vu_ . Next, we input _Vu_ in _f_ VLM with the prompt ( _✓,_ **Q**<sup>_⇤_</sup> ) to get the initial score 



where _y_ ˜ _u_ = 1 if _f_ VLM thinks the segment contains an anomaly after reasoning via **Q**<sup>_⇤_</sup> with _Vu_ , and otherwise, _y_ ˜ _u_ = 0. By repeating Eq. (3) for each segment, we have a segment-level initial anomaly score set _Y_<sup>˜</sup> = [˜ _y_ 1 _, · · · ,_ ˜ _yh_ ]. **Step 2: Ensemble Segment-Level Anomaly Scores with Scene Context** . Note that the scores derived above only examine a short moment in a long video without considering any context. To resolve it, we refine the initial segment-level score by incorporating scene context—defined as preceding and following segments that contain similar elements, such as actors and background, to those in the current segment. 

We measure the relevance between different video segments by the cosine similarity of their feature representations [22], extracted by a pretrained vision feature extractor 



<!-- Start of picture text -->
Step 1: Initial Anomaly Scores via Learned Guiding Questions<br>Frozen VLM  !!"# "#%<br>((  in the  " -th Segment Initial Anomaly Score<br>for the  " -th Segment<br>{ ... } Prompt with Guiding<br>Questions  ($, & ∗ )<br>All Segments in  (<br>Step 2: Ensemble Segment-Level Anomaly Scores with Scene Context<br>Retrieve By Eq. (4)<br>{ ... } {"#!%& , ⋯, "#!%' } (#%<br>(( All Segments in  ( Relevant Segment Ensemble<br>Anomaly Scores Score<br>Step 3: Frame-Level Anomaly Scoring with Temporal Context<br>Gaussian  Flattening &<br>Smoothing Position Weighting<br>[(#", ⋯, (##] [̅+",⋯,̅+#] [,#",⋯, ,#$]<br>Segment-Level  Smoothed Segment-Level Frame-Level<br>Anomaly Scores Anomaly Scores Anomaly Scores<br><!-- End of picture text -->

Figure 3. VERA computes anomaly scores with **Q**<sup>_⇤_</sup> in three steps. 

_g_ , _e.g_ ., ImageBind [10]. For the _u_ -th segment _Vu_ , its similarity with any segment _Vw_ (1 _ w  h_ ) is sim( _u, w_ ) = cos _<u>eu·ew</u>_ , where cos denotes the cosine function, ⇣ _||eu||·||ew||_ ⌘ and _eu_ = _g_ ( _Vu_ ) and _ew_ = _g_ ( _Vw_ ) represent their features. Let _u_ = [ _u_<sup>(1)</sup><sup>_, . . . , _</sup> _u_<sup>(</sup><sup>_K_)</sup> ] denote the indices of the top- _K_ segments similar to _Vu_ . We refine the anomaly score by _K_ exp(sim( _u, _<sup>(</sup> _u_<sup>_i_))</sup><sup>_<u>/⌧</u>_</sup><sup><u>)</u></sup> _y_ ¯ _u_ = X _i_ =1 _y_ ˜ __ ( _ui_ )<sup>_·_</sup> ~~P~~ _Kj_ =1<sup>exp(sim(</sup><sup>_u, _</sup> _u_<sup>(</sup><sup>_j_))</sup><sup>_/⌧_)</sup> _,_ (4) 

where _y_ ¯ _u_ is an ensemble of initial scores of top- _K_ video segments relevant to _Vu_ . Here, the initial score of each retrieved segment is weighted by a factor derived from the cosine similarity and normalized by the Softmax function (with _⌧_ as the temperature hyperparameter). Accordingly, scenes with greater similarity are assigned higher weights, making the ensemble score a more comprehensive reflection of anomalies with the video context. By applying Eq. (4) for all segments, we obtain _Y_<sup>¯</sup> = [¯ _y_ 1 _, . . . ,_ ¯ _yh_ ]. 

**Step 3: Frame-level Anomaly Scoring with Temporal Context** . Given _Y_<sup>¯</sup> , we aim to incorporate temporal context to capture how events evolve over time when computing frame-level anomaly scores, for the abnormality of an event often depends on the timing and progression of observed activities. To detail, we first apply Gaussian smoothing [11] to aggregate local temporal context into the segment-level anomaly scores. We denote the Gaussian kernel (suppose the filter size is _!_ ) as _G_ ( _p_ ) = exp(<sup>_−_</sup> 2 _σ_<sup>_<u>p</u>_</sup> 1<sup>22)where</sup><sup>_p_isthe</sup> distance from the kernel center and _σ_ 1 is the variance. We update segment-level scores as Γ =<sup>¯</sup> _Y_<sup>¯</sup> _⇤ G_ = [¯ _γ_ 1 _, · · · ,_ ¯ _γh_ ], where _⇤_ is the convolution operation. Next, we integrate global temporal context by position weighting. With Γ<sup>¯</sup> , we flatten it into frame-level scores by assigning the score ¯ _γu_ to each frame in the _u_ -th segment, _i.e_ ., [ _I_ ( _u−_ 1) _·d_ +1 _, · · · , Iu·d_ ]. We denote the frame-level score sequence after flattening as [ _⇢_ 1 _, · · · , ⇢F_ ]. We then apply the Gaussian function to _−_ <u>(</u> _i−c_ <u>)</u><sup>2</sup> encode position weights as _w_ ( _i_ ) = exp ⇣ 2 _σ_ 2<sup>2</sup> ⌘, where 

8683 

_i_ (1 _ i  F_ ) is any frame index, _c_ = floor( _F/_ 2) is the center frame index, and _σ_ 2 is the variance. The anomaly score for the _i_ -th frame is: 



This operation scales the score _⇢i_ , diminishing the anomaly score for frames near the beginning and end of the event. This helps better capture the temporal progression of anomalies: the score gradually increases as the anomaly reaches its peak and decreases afterward. The final scores is denoted as _Y_<sup>ˆ</sup> = [ˆ _y_ 1 _, . . . ,_ ˆ _yF_ ] after applying Eq. (5). **Explainable VAD by VERA** . When using template _✓_ embedded with **Q**<sup>_⇤_</sup> to compute _Y_<sup>ˆ</sup> , we ask the VLM to “provide an explanation in one sentence” when reasoning, and VLM will explain the anomaly score it assigns based on **Q**<sup>_⇤_</sup> . 

## **4. Experiments and Results** 

In this section, we present an evaluation of VERA as follows, addressing key questions of interest including: (Q1) Does it enhance the effectiveness of frozen VLMs in VAD? (Q2) Is its design reasonable and well-structured? (Q3) How well does it generalize across different scenarios? 

### **4.1. Experimental Settings** 

**Datasets** . We conduct experiments on two large-scale VAD datasets: (1) UCF-Crime [32] collected from surveillance videos with 13 types of anomalies and 290 (140 abnormal) test videos (2.13 minutes long on average). (2) XDViolence [42] with 6 anomaly categories and 800 (500 abnormal) test videos (1.62 minutes long on average). **Metrics** . Following approaches in [52, 55], we mainly evaluate VAD performance using the Area Under the Curve (AUC) of the frame-level Receiver Operating Characteristic (ROC) curve, as it provides a comprehensive measure of model performance across all thresholds. **Baselines** . We categorize baselines into non-explainable approaches and explainable ones as [55] does. Nonexplainable ones are obtained by WS learning [6, 9, 15, 17, 19, 32, 36, 41–43, 50, 51, 57] and unsupervised learning [13, 25, 34, 35, 37, 38]. These non-explainable approaches cannot provide language-based explanations for VAD. For explainable approaches, we use LAVAD [52], Holmes-VAD [55], and VADor [26] as representatives of Pipeline 1 and Pipeline 2 shown in Fig. 1. It should be noted that [46] does not report performance on UCF-Crime and XD-Violence. Additionally, we include zero-shot (ZS) VAD by frozen VLMs designed by [52] as baselines. 

**Implementation of VERA** . In our experiments, we choose a small VLM, InternVL2-8B [7], as the backbone _f_ VLM for building VERA by default, if not otherwise specified. We also explore other backbones, such as Qwen2-VL-7B [40] and InternVL2-40B [7] for ablation. We train **Q** for no more 

than 10 epochs, with a validation accuracy calculated every 100 iterations to determine **Q**<sup>_⇤_</sup> . We set _n_ as 2, _S_ as 8, and _m_ as 5 for training. The initial questions **Q** 0 is “ _1. Is there any suspicious person or object that looks unusual in this scene? 2. Is there any behavior that looks unusual in this scene?_ ”, inspired by previous VAD methods [13, 43], which assume anomalies appear with unusual appearance or motions. 

### **4.2. Comparison to State-of-the-art Methods** 

|We address Q1 by empiri-|Method|AUC|
|---|---|---|
|cally comparing VERA to|_Non-explainable VAD M_|_ethods_|
|existing VAD methods. First,|Wu et al. [42]<br>|82.44<br>|
|in Table 2, VERA achieves<br><br><br>|OVVAD [43]<br>S3R [41]|86.40<br>85.99|
|the<br>highest<br>AUC<br>among|<br>RTFM [36]|84.30|
|explainable VAD methods on|MSL [19]|85.62|
|UCF-Crime,<br>outperforming|MGFN [6]|86.98|
|Holmes-VAD<br>and<br>VADor|SSRL [17]|87.43|
|(without IT, as reported in|CLIP-TSA [15]<br>|**87.58**<br>|
|their papers) in a fair com-<br><br>|Sultani et al. [32]<br>GCL [51]|77.92<br>79.84|
|parison.<br>Importantly, unlike|<br>GCN [57]|82.12|
|these methods, VERA does|<br>MIST [9]|82.30|
|not need to modify the model|CLAWS [50]|83.03|
|parameters,<br>demonstrating|DYANNET [35]|84.50|
|its<br>suitability<br>to<br>directly|Tur el al. [37]|66.85<br>|
|adapt<br>VLM<br>to<br>the<br>VAD|GODS [38]|70.46|
|task with minimal training|_Explainable VAD Met_<br>LAVAD52|_hods_<br>8028|
|requirements.<br>Moreover,|[]<br>Holmes-VAD [55]|.<br>84.61|
|VERA surpasses LAVAD by|<br>VADor [26]|85.90|
|6% in AUC on UCF-Crime,|ZS CLIP [52]|53.16|
|uniquely<br>integrating<br>both|ZS IMAGEBIND-I[52]|53.65|
|description and reasoning ca-|ZS IMAGEBIND-V[52]|55.78|
|pabilities in VAD. Compared<br>to non-explainable methods,|LLAVA-1.5 [20]<br>VERA|72.84<br>**86.55**|
|VERA achieves AUC per-|Table 2. AUC (%) on UC<br>NoITisusedforHol|F-Crime.<br>es-VAD|
|formance that is comparable|<br>and VADor.||
|to one of the top-performing|||
|methods, CLIP-TSA, on UCF<br>additional advantage of explain|-Crime, while offeri<br>able predictions.|ng the|



Similar advantages are also observed in Table 3 for XD-Violence. Considering multiple factors, including performance, training efficiency, system integration, and explainability, VERA stands out as a promising pipeline for VLMs in VAD. 

|Method<br>AUC|
|---|
|_Non-Explainable VAD Methods_|
|Hasan et al. [13]<br>50.32|
|Lu et al. [25]<br>53.56|
|BODS [38]<br>57.32|
|GODS [38]<br>61.56|
|RareAnom [34]<br>**68.33**|
|_Explainable VAD Methods_|
|LAVAD [52]<br>85.36|
|ZS CLIP [52]<br>38.21|
|ZS IMAGEBIND-I[52]<br>58.81|
|ZS IMAGEBIND-V[52]<br>55.06|
|LLAVA-1.5 [20]<br>79.62|
|VERA<br>**88.26**|



### **4.3. Ablation Studies** 

We perform necessary abla- <mark>VERA</mark> tion studies on UCF-Crime Table 3. AUC (%) on XD-Violence. to answer both Q2 and Q3 for a comprehensive evaluation on our design choices. 

Table 3. AUC (%) on XD-Violence. 

8684 

**Frame Sampling Strategy in Training** . We compare three frame sampling strategies for obtaining each _V_<sup>˜(</sup><sup>_j_)</sup> in training: uniform sampling, random sampling, and TSN sampling (random sampling from equally divided segments). 

|Table4shows that uniform sam-<br>|Strategy<br>AUC (%)|
|---|---|
|pling performs the best (with<br><br>|Random [3]<br>83.63|
|batch size _n_<br>= 2 and _S_<br>=|TSN [39]<br>82.63|
|8).<br>This is because uniform<br>|Uniform [54]<br>**86.55**|
|sampling preserves the temporal<br>structure and maintains consis-<br>|Table 4.<br>Sampling strategies<br>explored in VERA training.|
|tent motion patterns throughout||
|the long video, making it easier|for VLMs to understand|
|the video and update**Q**.||
|Question Type|AUC (%)|
|No questions|78.81|
|Manually written questions by human|81.15|
|Learned questions w/o iteratively inputti|ng_V_batch in Eq. (2)<br>78.06|
|Iteratively learned questions (used in VE|RA)<br>**86.55**|



Table 5. The way we obtain guiding questions affects AUC substantially. 

**How to Obtain Guiding Questions Q for VLM** . As seen in Table 5, if the guiding questions are not incorporated into the VLM prompt, the AUC will drop largely to 78.81%, confirming the need to use simpler and more focused questions to provoke reasoning in the VLMs for VAD. Meanwhile, if we use manually written questions ( **Q** 0), the performance is suboptimal with an 81.15% AUC, which shows the need to use VL to find guiding questions. Lastly, if we only input batched predictions _Y_<sup>ˆ</sup> _batch_ and ground truths _Ybatch_ without inputting _V_ batch in the optimizer, the **Q** updated in this way will dumb the VLMs and make it have a low AUC. Thus, inputting video frames as Eq. (2) does is necessary to learn good **Q** . 



<!-- Start of picture text -->
Number of Questions  m . 86.55<br>As shown in Fig. 4, when 86 0 question 85.79<br>m is set to 1, the rea- 84 83.76<br>soning is limited to a sin- 82<br>gle perspective, resulting 80 78.81<br>in a lower AUC. As  m  in- 78 77.31<br>creases up to 5, the model 76<br>captures more compre- 1 3 5 7<br>Number of Questions<br>hensive anomaly patterns,<br>Figure 4. Effect of the number of guid-<br>leading to improved AUC. ing questions on AUC.<br>However, increasing m<br>AUC (%)<br><!-- End of picture text -->

beyond 5 yields no significant gains. Therefore, we set _m_ to 5 by default in VERA, if not otherwise specified. 

|Operation|AUC (%)|
|---|---|
|Initial (Step 1)|76.10|
|Initial + Retrieval (Step 2)|84.53 (+8.43)|
|Initial + Retrieval + Smoothing (Step 3)|85.48 (+0.95)|
|Initial + Retrieval + Smoothing + Weighting (Step 3)|**86.55**(+1.07)|



Table 6. Ablation study of each step in VERA’s inference. 

**Coarse-to-Fine Anomaly Score Computation** . We also validate the anomaly score computation by VERA. Table 6 

shows the AUC is 76.10% when using the flattened initial score obtained in Step 1, and leveraging retrieved segments in Step 2 significantly boosts the AUC to 84.53%, highlighting the effectiveness of incorporating ensemble scores based on scene context. Meanwhile, smoothing and weighting in Step 3 further improves the AUC by around 1% each, verifying the benefit of integrating temporal context. 

**Generalizability Test** . We further examine the generalizability of VERA across different model sizes, VLM architectures, and datasets to address Q3. First, we apply VERA Source of **Q** _f_ VLM to InternVL2-40B, a InternVL2-8B InternVL2-40B larger model in the InternVL2-8B **86.55** 80.43 InternVL2 family comInternVL2-40B 85.24 **86.72** pared to InternVL2-8B. Table 7. AUC (%) across model sizes . As shown in TaSource of **Q** _f_ VLM ble 7, InternVL2-40B InternVL2-8B Qwen2-VL-7B achieves effective InternVL2-8B **86.55** 81.37 Qwen2-VL-7B 79.60 **82.64** AUC performance, Table 8. AUC (%) across architectures. slightly exceeding Source of **Q** that of InternVL2-8B, Dataset UCF-Crime XD-Violence indicating that VL in UCF-Crime **86.55** 80.42 VERA enables models XD-Violence 86.26 **88.26** of various scales to Table 9. AUC (%) across datasets. identify a **Q** suitable for their reasoning capabilities. Additionally, We also evaluate the transferability of **Q** across different scales and and observe an interesting phenomenon: the **Q** learned by InternVL2-8B remains effective for InternVL2-40B, but not vice versa. This is likely because the **Q** learned by the smaller model is readily interpretable by the larger model, whereas the **Q** derived from the larger model is more complex in syntactic structure and does not align well with the reasoning framework of the smaller model. Secondly, we select a different VLM, Qwen2-VL-7B [40], as the backbone for VERA. As shown in Table 8, while the AUC achieved with Qwen2-VL-7B is lower than that with InternVL2-8B, the VL in VERA remains effective, allowing it to outperform notable baselines such as LAVAD [52]. However, a notable gap exists when transferring **Q** across different model architectures in Table 8. Developing a universal **Q** that can effectively elicit reasoning capabilities across various VLM structures would be an promising direction for future research. Lastly, we observe that the transferability of **Q** depends on the training dataset. From Table 9, we observe that transferring **Q** learned from UCFCrime to XD-Violence results in a smaller performance drop compared to the reverse case. This suggests the source dataset is crucial to the transferability of **Q** across datasets. 

### **4.4. Qualitative Results and Case Studies** 

W.l.o.g., we take one video on UCF-Crime to illustrate the explainability brought by the learned **Q**<sup>_⇤_</sup> qualitatively (on 

8685 



<!-- Start of picture text -->
① ... ② ... ③<br>!: The video shows a person  !: The man is  not in his  !: The person in the video is<br>walking in a hallway and  typical position or engaging  not in a typical position or<br>entering a room . There are  no in activities  consistent with his  engaging in activities  that are<br>people in unusual positions or  usual behavior. He is  bending  not consistent with their usual<br>engaging in activities  that are  over and interacting with the  behavior. They are  bending<br>not consistent with their usual  machines  in a way that is not  over and reaching into the<br>behavior. typical. machine , which is not a<br>common action for them.<br>④ ... ⑤ ... ⑥<br>!: The person in the video is  !: The video shows a person  !: The video shows  a person<br>not in their typical position  or  lying on the floor and being  being  restrained  by  two<br>engaging in activities consistent  attended  to  by  two  individuals  in a room. This<br>with their usual behavior. They  individuals ,  which  is  not  activity is  not consistent with<br>are  lying on the floor , which is  consistent with their usual  typical  behavior  in  the<br>not  typical  behavior  for  behavior . environment.<br>someone in this setting.<br><!-- End of picture text -->

Figure 5. Given **Q**<sup>_⇤_</sup> by VERA, the frozen VLM (InternVL2-8B) will reason and explain the scene based on it. For illustration, we take as an example the video “Arrest007 ~~x~~ 264” from UCF-Crime and include 6 scenes here. The complete anomaly scores are shown in Fig. 8. 

UCF-Crime **Q**<sup>_⇤_</sup> is “ _1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior? 2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function? 3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function? 4. Is there any visible damage or unusual movement in the video that indicates an anomaly? 5. Are there any unusual sounds or noises in the video that suggest an anomaly?_ ”). As shown in Fig. 5, the main anomaly in this video is that a man tries to steal money from the washing machines in a laundromat and is arrested after being found by the police. In the selected 6 main video segments, the frozen VLM with VERA’s learned questions is able to explain the scene by closely following the detailed anomaly characterization of the five learned guiding questions. W.l.o.g., we take the first 3 segments in Fig. 5 for instance to closely compare the caption quality with LAVAD, a representative baseline. As shown in Fig. 6, VERA’s captions include both precise descriptions ( **bold** text) and reasoning (text in purple) about anomalies, while LAVAD’s captions contain only plain descriptions. This difference owes to VERA’s learned guiding questions, which transform VLM’s thinking and phrasing. 

A more interesting advantage of VERA is that it allows humans to further interact with VLMs because it retains the general question-answering ability of pretrained VLMs. This is because VERA does not require finetuning of the VLM backbone weights. Although finetuning VLMs with parameter-efficient methods like [14, 24, 29] is easy and computationally tractable, instruction-tuned models still inevitably lose the flexibility to handle general questions (due to catastrophic forgetting), as they are trained to respond to certain queries with fixed answer styles. In contrast, as shown in Fig. 7, the learned **Q**<sup>_⇤_</sup> can steer reasoning in a frozen VLM while allowing it to flexibly answer openended (like follow-up or counterfactual) questions, which is 



<!-- Start of picture text -->
① ② ③ ...<br>VERA:  The  video  VERA: The man is not  VERA: The person in<br>shows  a  person  in his typical position  the video is  not in a<br>walking in a hallway  or  engaging  in  typical  position  or<br>and entering a room .  activities  consistent  engaging  in  activities<br>There are  no people in  with  his  usual  that are not consistent<br>unusual  positions  or  behavior.  He  is  with their usual behavior.<br>engaging in activities  bending  over  and  They are  bending over<br>that are not consistent  interacting with the  and reaching into the<br>with  their  usual  machines  in a way  machine , which is not a<br>behavior. that is not typical. common action for them.<br>LAVAD:  A  man  is  LAVAD: A man is  LAVAD: A man is shown<br>shown in a laundry  shown in a laundry  in a laundry room for 8<br>room by a surveillance  room by a surveillance  consecutive frames by a<br>camera. camera. surveillance camera.<br>Figure 6. Qualitative comparison between VERA and LAVAD.<br>!: The video shows a person walking in a<br>hallway and entering a room. There are no<br>people in unusual positions or engaging in …<br>activities that are not consistent with their usual<br>behavior. Based on the analysis, there is no<br>anomaly in the video. Segment  ① in the example video<br>": (Follow-up Question) This is a laundromat with washing machines. Why this<br>behavior is normal?<br>!: The behavior in the video is normal because it depicts a person walking in a<br>hallway and entering a room that appears to be a laundromat with washing machines.<br>The person's actions, such as walking and entering the room, are consistent with<br>typical behavior in a laundromat setting .<br>": (Counterfactual Question) Please infer with this hypothetical scenario and<br>answer in a few sentences: what if the man is stealing from the machine?<br>!: If the man in the video is stealing from the washing machines, this behavior would<br>be considered an anomaly. Stealing is an illegal and unethical act that is not<br>consistent with typical behavior in a laundromat setting . The presence of such<br>behavior would indicate a violation of the law and a breach of trust, making it an<br>anomaly in the video.<br><!-- End of picture text -->

Figure 7. VERA can take open-ended questions and interact with humans. 

an important ability lost in IT-based models. Moreover, as shown in Fig. 8, owing to the proposed coarse-to-fine anomaly 1.0 <u>①② ③④⑤⑥</u> **This part contains** scoring, the anomaly 0.8 **a scene where the** **~~suspect is taken~~** score dynamics from **away by the police** 0.6 VERA well represent the actual real-time anomaly 0.4 level in this video and 0.2 gradually increases to 0 0 500 1000 1500 2000 2500 3000 nearly 1 when the man Frame Index is being arrested. This Figure 8. Anomaly scores generated result verifies that VERA byrest007VERAx264” from UCF-Crime.(with InternVL2-8B) in “Arallows VLMs to effectively identify anomalies with a holistic model, reducing the manpower and computational overhead for VAD. 

## **5. Concluding Remarks** 

We propose a novel pipeline, VERA, which can effectively elicit the reasoning ability from VLMs to perform explainable VAD without additional computation overhead. This is done through an effective and novel application of verbalized machine learning [45] to VLM. In training, VERA obtains the guiding questions detailing anomaly patterns through the verbal interaction between the learner and the optimizer agents. In inference, VERA uses them to enhance VLMs for identifying anomalies and compute frame-level anomaly scores in a coarse-to-fine process. Experimental results validate the effectiveness of the VERA framework in achieving state-of-the-art explainable VAD performance. 

8686 

## **References** 

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 1 

- [2] Daniel Bogdoll, Maximilian Nitsche, and J Marius Z¨ollner. Anomaly detection in autonomous driving: A survey. In _CVPR Workshops_ , 2022. 1 

- [3] Meinardus Boris, Batra Anil, Rohrbach Anna, and Rohrbach Marcus. The surprising effectiveness of multimodal large language models for video moment retrieval. _arXiv preprint arXiv:2406.18113_ , 2024. 7 

- [4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _NeurIPS_ , 2020. 1 

- [5] Junxi Chen, Liang Li, Li Su, Zheng-jun Zha, and Qingming Huang. Prompt-enhanced multiple instance learning for weakly supervised video anomaly detection. In _CVPR_ , 2024. 3 

- [6] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. In _AAAI_ , 2023. 6, 14 

- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In _CVPR_ , 2024. 1, 2, 6 

- [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 1 

- [9] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In _CVPR_ , 2021. 6 

- [10] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In _CVPR_ , 2023. 5, 15 

- [11] Rafael C Gonzalez. _Digital image processing_ . Pearson education india, 2009. 5 

- [12] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In _CVPR_ , 2024. 3 

- [13] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K 

Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In _CVPR_ , 2016. 6, 11 

- [14] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _ICLR_ , 2021. 8 

- [15] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In _ICIP_ , 2023. 6, 14 

- [16] Diederik P Kingma. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 4 

- [17] Guoqiu Li, Guanxiong Cai, Xingyu Zeng, and Rui Zhao. Scale-aware spatio-temporal relation learning for video anomaly detection. In _ECCV_ , 2022. 6 

- [18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _ICML_ , 2023. 1 

- [19] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. In _AAAI_ , 2022. 6, 14 

- [20] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _CVPR_ , 2024. 6, 14 

- [21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _NeurIPS_ , 2024. 1, 3 

- [22] Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, and Le Song. Deep hyperspherical learning. In _NeurIPS_ , 2017. 5 

- [23] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In _CVPR_ , 2018. 3 

- [24] Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, et al. Parameter-efficient orthogonal finetuning via butterfly factorization. In _ICLR_ , 2024. 8 

- [25] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In _ICCV_ , 2013. 3, 6 

- [26] Hui Lv and Qianru Sun. Video anomaly detection and explanation via large language models. _arXiv preprint arXiv:2401.05702_ , 2024. 1, 2, 3, 6 

- [27] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In _CVPR_ , 2023. 3 

- [28] Sarah Pratt, Ian Covert, Rosanne Liu, and Ali Farhadi. What does a platypus look like? generating customized prompts for zero-shot image classification. In _ICCV_ , 2023. 3 

- [29] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Sch¨olkopf. Controlling text-to-image diffusion by orthogonal finetuning. In _NeurIPS_ , 2023. 8 

- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 18 

8687 

- [31] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Sch¨olkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In _CVPR_ , 2022. 1 

- [32] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In _CVPR_ , 2018. 2, 3, 6, 14 

- [33] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and Yingcong Chen. Hawk: Learning to understand open-world video anomalies. _Advances in Neural Information Processing Systems_ , 37:139751–139785, 2024. 3 

- [34] Kamalakar Vijay Thakare, Debi Prosad Dogra, Heeseung Choi, Haksub Kim, and Ig-Jae Kim. Rareanom: A benchmark video dataset for rare type anomalies. _Pattern Recognition_ , 140:109567, 2023. 6 

- [35] Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. Dyannet: A scene dynamicity guided self-trained video anomaly detection network. In _WACV_ , 2023. 6 

- [36] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In _ICCV_ , 2021. 6, 14 

- [37] Anil Osman Tur, Nicola Dall’Asen, Cigdem Beyan, and Elisa Ricci. Unsupervised video anomaly detection with diffusion models conditioned on compact motion representations. In _International Conference on Image Analysis and Processing_ , 2023. 3, 6 

- [38] Jue Wang and Anoop Cherian. Gods: Generalized one-class discriminative subspaces for anomaly detection. In _ICCV_ , 2019. 3, 6 

- [39] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In _ECCV_ , 2016. 7 

- [40] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 6, 7 

- [41] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Self-supervised sparse representation for video anomaly detection. In _ECCV_ , 2022. 6, 14 

- [42] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In _ECCV_ , 2020. 2, 3, 6, 14 

- [43] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. In _CVPR_ , pages 18297–18307, 2024. 6, 11, 14 

- [44] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 6074–6082, 2024. 3 

- [45] Tim Z Xiao, Robert Bamler, Bernhard Sch¨olkopf, and Weiyang Liu. Verbalized machine learning: Revisiting machine learning with language models. _arXiv preprint arXiv:2406.04344_ , 2024. 3, 4, 8 

- [46] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. Follow the rules: reasoning for video anomaly detection with large language models. _arXiv preprint arXiv:2407.10299_ , 2024. 1, 3, 6 

- [47] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection. In _CVPR_ , 2024. 3 

- [48] Muchao Ye, Xiaojiang Peng, Weihao Gan, Wei Wu, and Yu Qiao. Anopcn: Video anomaly detection via deep predictive coding network. In _ACM international conference on multimedia_ , 2019. 3 

- [49] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic “differentiation” via text. _arXiv preprint arXiv:2406.07496_ , 2024. 3 

- [50] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection. In _ECCV_ , 2020. 6 

- [51] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In _CVPR_ , 2022. 6 

- [52] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. In _CVPR_ , 2024. 1, 3, 5, 6, 7, 14, 15 

- [53] Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, and Ming-Hsuan Yang. Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection. In _CVPR_ , 2023. 3 

- [54] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In _EMNLP_ , 2023. 3, 7 

- [55] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Chuchu Han, Xiaonan Huang, Changxin Gao, Yuehuan Wang, and Nong Sang. Holmes-vad: Towards unbiased and explainable video anomaly detection via multi-modal llm. _arXiv preprint arXiv:2406.12235_ , 2024. 1, 2, 3, 6, 14 

- [56] Menghao Zhang, Jingyu Wang, Qi Qi, Haifeng Sun, Zirui Zhuang, Pengfei Ren, Ruilong Ma, and Jianxin Liao. Multiscale video anomaly detection by multi-grained spatiotemporal representation learning. In _CVPR_ , 2024. 3 

- [57] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In _CVPR_ , 2019. 6 

- [58] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In _ICLR_ , 2024. 1 

8688 

