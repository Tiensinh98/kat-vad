# **Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models** 

Yuchen Yang<sup>1</sup><sup>_⋆_</sup> , Kwonjoon Lee<sup>2</sup> , Behzad Dariush<sup>2</sup> , Yinzhi Cao<sup>1</sup> , and Shao-Yuan Lo<sup>2</sup> 

> 1 Johns Hopkins University 

```
{yc.yang,yinzhi.cao}@jhu.edu
```

> 2 Honda Research Institute USA 

```
{kwonjoon_lee,bdariush,shao-yuan_lo}@honda-ri.com
```

**Abstract.** Video Anomaly Detection (VAD) is crucial for applications such as security surveillance and autonomous driving. However, existing VAD methods provide little rationale behind detection, hindering public trust in real-world deployments. In this paper, we approach VAD with a reasoning framework. Although Large Language Models (LLMs) have shown revolutionary reasoning ability, we find that their direct use falls short of VAD. Specifically, the implicit knowledge pre-trained in LLMs focuses on general context and thus may not apply to every specific real-world VAD scenario, leading to inflexibility and inaccuracy. To address this, we propose AnomalyRuler, a novel rule-based reasoning framework for VAD with LLMs. AnomalyRuler comprises two main stages: induction and deduction. In the induction stage, the LLM is fed with few-shot normal reference samples and then summarizes these normal patterns to induce a set of rules for detecting anomalies. The deduction stage follows the induced rules to spot anomalous frames in test videos. Additionally, we design rule aggregation, perception smoothing, and robust reasoning strategies to further enhance AnomalyRuler’s robustness. AnomalyRuler is the first reasoning approach for the one-class VAD task, which requires only few-normal-shot prompting without the need for full-shot training, thereby enabling fast adaption to various VAD scenarios. Comprehensive experiments across four VAD benchmarks demonstrate AnomalyRuler’s state-of-the-art detection performance and reasoning ability. AnomalyRuler is open-source and available at: `https://github.com/Yuchen413/AnomalyRuler` 

## **1 Introduction** 

Video Anomaly Detection (VAD) aims to identify anomalous activities, which are infrequent or unexpected in surveillance videos. It has a wide range of practical applications, including security (e.g., violence), autonomous driving (e.g., traffic accidents), etc. VAD is a challenging problem since anomalies are rare and longtailed in real life, leading to a lack of large-scale representative anomaly data. 

> _⋆_ This work was mostly done when Y. Yang was an intern at HRI-USA. 

2 Y. Yang et al. 



<!-- Start of picture text -->
Train: Learn model Induction: Derive rules Deduction: Use rules<br>Test: Use model Full-shotTrainingFramesNormal Is this framenormal oranomaly? Test Frame  ReferenceFew-shotFramesNorm al Environmental ObjectsRules for NormalRules for NormalHuman Activities Environmental ObjectsRules for AnormalyRules for AnomalyHuman Activities Is thisframenormal oranomalyBased onthe  rules ? Test Frame<br>Normal based on implicit Anomaly based on learned rules<br>knowledge in LLM<br>Test Frame  output 0.8340.834? It appears that unusual or anomalousdepicts a typical outdoor setting wherethree individuals are seen engaging inseparate walking, another is standing and lookingat their phone, and the third isskateboarding. The scene seems to be aregular day at a park or outdoor areadesigned for pedestrians.activities: there is nothing overtly in this image. Itone  person  is There are four people in this frame. Starting from the left, the first person appears to be is people, there are people are walking, which matches: skateboards, which matches: Manhole covers: Two manhole covers are visible, which matches: the information provided, the normal activities are people walking, and the anomaly activities are people skateboarding;The normal objects are manhole covers, and there are no anomaly objects mentioned in the description. Therefore, frame is an anomalyriding a skateboardtwo manhole covers due to the presence of anomaly human activities., the third person is Anomaly Human ActivitiesNormal Human Activities visible on the ground. First, let's check for human activities: (1) Walking: Twowalking but further away, and the fourth person is on a , Rule number 1. Second, let's check for environmental objects: (1), Rule number 1. (2) Skateboarding: Two people are onNormal environmental objectswalking, Rule number: 2. Based onskateboard, the second person. Other thanthis<br>a) Traditional VAD b) Ask LLM directly c) Ask LLM with the rules learned in induction stage (Ours)<br><!-- End of picture text -->

**Fig. 1:** Comparison of one-class VAD approaches. In this specific safety application example, only “walking” is normal. The test frame contains “skateboarding”, so it is abnormal. (a) Traditional methods require full-shot training and only output anomaly scores, lacking reasoning. (b) Direct LLM use may not align with specific VAD needs. Here GPT-4V mistakenly treats “skateboarding” as normal. (c) Our AnomalyRuler has induction and deduction stages. It derives rules from few-shot normal reference frames to detect anomalies, correctly identifying “skateboarding” as an anomaly. 

Hence, the one-class VAD (a.k.a. unsupervised VAD) paradigm [16, 36, 43, 45, 53] is preferred, as it assumes that only the more accessible normal data are available for training. Most existing one-class VAD methods learn to model normal patterns via self-supervised pretext tasks, such as frame reconstruction [16, 25, 27, 36, 45, 53, 54] and frame order classification [14, 43, 49]. Despite good performance, these traditional methods can only output anomaly scores, providing little rationale behind their detection results (see Fig. 1a). This hinders them from earning public trust when deployed in real-world products. 

We approach the VAD task with a reasoning framework toward a trustworthy system, which is less explored in the literature. An intuitive way is to incorporate the emergent Large Language Models (LLMs) [1,7,19,38,39,47], which have shown revolutionary capability in various reasoning tasks. Still, we find that their direct use falls short of performing VAD. Specifically, the implicit knowledge pre-trained in LLMs focuses on general context, meaning that it may not always align with specific real-world VAD applications. In other words, there is a mismatch between an LLM’s understanding of anomalies and the anomaly definitions required for certain scenarios. For example, the GPT-4V [1] typically treats “skateboarding” as a normal activity, whereas certain safety applications need to define it as an anomaly, such as within a restricted campus (see Fig. 1b). However, injecting such specific knowledge by fine-tuning LLMs for each application is costly. This highlights the necessity for a flexible prompting approach that steers LLMs’ reasoning strengths to different uses of VAD. 

To arrive at such a solution, we revisit the fundamental process of the scientific method [4] emphasizing reasoning, which involves drawing conclusions in a rigorous manner [41]. Our motivation stems from two types of reasoning: inductive reasoning, which infers generic principles from given observations, and deductive reasoning, which derives conclusions based on given premises. In this 

Follow the Rules: Reasoning for VAD with LLMs 3 

paper, we propose AnomalyRuler, a new VAD framework based on reasoning with LLMs. AnomalyRuler consists of an induction stage and a deduction stage as shown in Fig. 1c. In the induction stage, the LLM is fed with visual descriptions of few-shot normal samples as references to derive a set of rules for determining normality. Here we employ a Vision-Language Model (VLM) [23, 51] to generate the description for each input video frame. Next, the LLM derives a set of rules for detecting anomalies by contrasting the rules for normality. The deduction, which is also an inference stage, follows the induced rules to identify anomalous frames in test video sequences. Additionally, in response to potential perception and reasoning errors by the VLM and LLM, we design strategies including rule aggregation via the randomized smoothing [10] for rule induction error mitigation, perception smoothing via the proposed Exponential Majority Smoothing for perception error reduction together with temporal consistency enhancement, and robust reasoning via a recheck mechanism for reliable reasoning output. These strategies are integrated into the AnomalyRuler pipeline to further enhance its detection robustness. 

Apart from equipping VAD with reasoning ability, AnomalyRuler offers several advantages. First, AnomalyRuler is a novel _few-normal-shot prompting_ approach that utilizes only a few normal samples from a training set as references to derive the rules for VAD. This avoids the need for expensive full-shot training or fine-tuning of the entire training set, as required by traditional one-class VAD methods. Importantly, it enables efficient adaption by redirecting LLM’s implicit knowledge to different specific VAD applications through just a few normal reference samples. Second, AnomalyRuler shows strong domain adaptability across datasets, as the language provides consistent descriptions across different visual domains, e.g., “walking” over visual data variance. This allows the application of induced rules to datasets with similar scenarios but distinct visual appearances. Furthermore, AnomalyRuler is a generic framework that is complementary to VLM and LLM backbones. It accommodates both closed-source models such as the GPT family [1, 39] and open-source alternatives such as Mistral [19]. To the best of our knowledge, the proposed AnomalyRuler is the first reasoning approach for the one-class VAD problem. Extensive experiments on four VAD datasets demonstrate AnomalyRuler’s state-of-the-art performance, reasoning ability, and domain adaptability. 

In summary, this paper has three main contributions. (1) We propose a novel rule-based reasoning framework for VAD with LLMs, namely AnomalyRuler. To the best of our knowledge, it is the first reasoning approach for one-class VAD. (2) The proposed AnomalyRuler is a novel few-normal-shot prompting approach that eliminates the need for expensive full-shot tuning and enables fast adaption to various VAD scenarios. (3) We propose rule aggregation, perception smoothing, and robust reasoning strategies for AnomalyRuler to enhance its robustness, leading to state-of-the-art detection performance, reasoning ability, and domain adaptability. 

4 Y. Yang et al. 

## **2 Related Work** 

**Video Anomaly Detection.** VAD is a challenging task since anomaly data are scarce and long-tailed. Therefore, researchers often focus on the one-class VAD (a.k.a. unsupervised VAD) paradigm [14, 16, 18, 25, 27, 36, 43, 45, 49, 53, 54], which uses only normal data during training. Most one-class methods learn to model normal patterns via self-supervised pretext tasks, based on the assumption that the model would obtain poor pretext task performance on anomaly data. Reconstruction-based methods [16, 25, 27, 36, 45, 53, 54] employ generative models such as auto-encoders and diffusion models to perform frame reconstruction or frame prediction as pretext tasks. Distance-based [14, 43, 49] methods use classifiers to perform pretext tasks such as frame order classification. These traditional methods can only output anomaly scores, providing little rationale behind their detection. Several recent studies explore utilizing VLMs or LLMs in anomaly detection. Elhafsi et al. [12] analyze semantic anomalies with an object detector [33] an LLM [7] in driving scenes. However, it relies on predefined concepts of normality and anomaly, which limits its adaption to different scenarios and cannot handle long-tailed undefined anomalies. Moreover, this method has not been evaluated on standard VAD benchmarks [2, 22, 24, 28]. Cao et al. [8] explore the use of GPT-4V for anomaly detection, but their direct use may fall into the misalignment between GPT-4V’s implicit knowledge and specific VAD needs, as discussed. Gu et al. [15] adopt a large VLM for anomaly detection, but it focuses on industrial images. Despite supporting dialogues, this method can only describe anomalies rather than explain the rationales behind its detection. Lv et al. [31] equip video-based LLMs in the VAD framework to provide detection explanations. It involves three-phase training to fine-tune the heavy video-based LLMs. Besides, it focuses on weakly-supervised VAD, a relaxed paradigm that requires training with anomaly data and labels. Different from these works, our AnomalyRuler provides rule-based reasoning via efficient few-normal-shot prompting and enables fast adaption to different VAD scenarios. 

**Large Language Models.** LLMs [1, 7, 19, 38, 39, 46, 47] have achieved significant success in natural language processing and are recently being explored for computer vision problems. Recent advances, such as the GPT family [1, 7, 38, 39], the LLaMA family [46, 47], and Mistral [19], have shown remarkable capabilities in understanding and generating human language. On the other hand, large VLMs [1,21,23,34,44,51,57,58,60] have shown promise in bridging the vision and language domains. BLIP-2 [21] leverages Q-Former to integrate visual features into a language model. LLaVA [23] introduces a visual instruction tuning method for visual and language understanding. CogVLM [51] trains a visual expert module to improve large VLM’s vision ability. Video-LLaMA [57] extends LLMs to understand video data. These models’ parametric knowledge is trained for general purposes and thus may not apply to every VAD application. Recent studies explore prompting methods to exploit LLMs’ reasoning ability. Chain-of-Thought (CoT) [11, 52] guides LLMs to solve complex problems via multiple smaller and manageable intermediate steps. Least-to-Most (LtM) [20, 59] decomposes a complex problem into multiple simpler sub-problems and solves them in sequence. 

Follow the Rules: Reasoning for VAD with LLMs 5 



<!-- Start of picture text -->
Visual Perception Rule Generation Rule Aggregation<br>Normal Frame Description 0  LLM Rules for Normal 0 Normal Human Activities1. Walking with common objects Rules For Video Anomaly Detection<br>Rules for Anomaly 0 Aggregation:     such as a backpack, bag, umbrella<br>Vote for 2. ...<br>Normal  VLM Frame Description 1  LLM Rules for Normal 1 Robust Rules LLM Normal Environmental Objectives1. Architectural structures such as   building, bridges<br>2. ...<br>Normal Reference Frames...... Normal What are doing? the images than peopleWhat are in peopleother ? Frame Description n......  LLM Rules for Normal nRules for Anomaly 1...... Anomaly Human Activities1. Non-walking movement such as riding   a bicycle, scooting, skateboarding2. ...Anomaly Environmental Objectives<br>Human/EnvironmentDecoupling Rules for Anomaly n 1. Ground transportation such as    vehicles, motorcycles<br>2. ...<br>Visual Perception Perception Smoothing Dummy Robust Reasoning<br>Continuous Frames in a Video Continuous Frames' Descriptions Anomaly keyword matches? 1 1 AnomalyAnswer  LLM Answer: Reason: One person ridingAnomaly.<br>1 1 Anomaly      a bicycle matches<br>Test  VLM "..."..."..."..."...walkingwalkingwalkingwalkingwalking, walking, , , , ridingridingridingriding..."..."..."..."..." Rules  If   ≠  yesno:  Modify 10 th  frame's description  ...... 0 11 ......111 AnomalyAnomalyAnomaly...... QueryMy dummy answer isfor based on the let me know what is your answer.{Modified Frame Description}{Rules}.     rule 1 of 'Anomaly     Human Activities' Double-check and{Dummy Answer}<br>...... ...... following its neighbors' majority within a window size Modified Frame Description<br>I. Induction<br>Smooth<br>II. Deduction Dummy Answer Rules<br><!-- End of picture text -->

**Fig. 2:** The AnomalyRuler pipeline consists of two main stages: induction and deduction. The induction stage involves: i) visual perception transfers normal reference frames to text descriptions; ii) rule generation derives rules based on these descriptions to determine normality and anomaly; iii) rule aggregation employs a voting mechanism to mitigate errors in rules. The deduction stage involves: i) visual perception transfers continuous frames to descriptions; ii) perception smoothing adjusts these descriptions considering temporal consistency to ensure neighboring frames share similar characteristics; iii) robust reasoning rechecks the previous dummy answers and outputs reasoning. 

Hypotheses-to-Theories (HtT) [61] learns a rule library for reasoning from labeled training data in a supervised manner. However, a reasoning approach for the VAD task in the one-class paradigm is not well-explored. 

## **3 Induction** 

The induction stage aims to derive a set of rules from a few normal reference frames for performing VAD. The top part of Fig. 2 shows the three modules in the induction pipeline. The visual perception module utilizes a VLM which takes a few normal reference frames as inputs and outputs frame descriptions. The rule generation module uses an LLM to generate rules based on these descriptions. The rule aggregation module employs a voting mechanism to mitigate the errors from rule generation. In the following sections, we discuss each module and the strategies applied in detail. 

### **3.1 Visual Perception** 

We design the visual perception module as the initial step in our pipeline. This module utilizes a VLM to convert video frames into text descriptions. We define _F_ normal = _{f_ normal0 _, . . . , f_ normal _n}_ as the few-normal-shot reference frames, with each frame _f_ normal _i ∈ F_ normal randomly chosen from the training set. This module outputs the text description of each normal reference frame: 

#### 6 Y. Yang et al. 

_D_ normal = _{_ VLM( _f_ normal _i, pv_ ) _| f_ normal _i ∈ F_ normal _}_ , with _pv_ as the prompt “What are people doing? What are in the images other than people?”. Instead of directly asking “What are in the image?”, we design _pv_ to separate humans and the environment with the following advantages. First, it enhances perception precision by directing the model’s attention to specific aspects of the scene, ensuring that no details are overlooked. Second, it simplifies the following rule generation module by dividing the task into two subproblems [20], i.e., rules for human activities and rules for environmental objects. We denote this strategy as Human and Environment. 

### **3.2 Rule Generation** 

With the text descriptions from normal reference frames _D_ normal, we design a Rule Generation module that uses a frozen LLM to generate rules (denoted as _R_ ). In formal terms, _R_ = _{_ LLM( _d_ normal _i, pg_ ) _| d_ normal _i ∈ D_ normal _}_ , where _pg_ is the prompt detailed in Appendix A.2. We craft _pg_ with three strategies to guide the LLM in gradually deriving rules from the observed normal patterns: 

**Normal and Anomaly.** The prompt _pg_ guides the LLM to perform _contrast_ , which first induces rules for normal based on _D_ normal, which are assumed to be ground-truth normal. Then, it generates rules for anomalies by contrasting them with the rules for normal. For instance, if “walking” is a common pattern in _D_ normal, it becomes a normal rule, and then “non-walking movement” will be included in the rules for anomaly. This strategy sets a clear boundary between normal and anomaly without access to anomaly frames. 

**Abstract and Concrete.** The prompt _pg_ helps the LLM to perform _analogy_ , which starts from an abstract concept and then effectively generalizes to more concrete examples. Taking the same “walking” example, the definition of a normal rule is now expanded to “walking, whether alone or with others.” Consequently, the anomaly rule evolves to include specific non-walking movements, i.e., “nonwalking movement, such as riding a bicycle, scooting, or skateboarding.” This strategy clarifies the rules with detailed examples and enables the LLM to use analogy for reasoning without exhaustively covering every potential scenario. **Human and Environment.** This strategy is inherited from the Visual Perception module. The prompt _pg_ leads the LLM to pay attention separately to environmental elements (e.g., vehicles or scene factors) and human activities, separately. This enriches the rule set for VAD tasks, where anomalies often arise from interactions between humans and their environment. 

These strategies align with the spirit of CoT [52] yet are further refined for the VAD task. The ablation study in Section 5.4 demonstrates their effectiveness. 

### **3.3 Rule Aggregation** 

The rule aggregation module uses a LLM as an aggregator with a voting mechanism to combine _n_ sets of rules (i.e., _R_ ) generated independently from _n_ randomly chosen normal reference frames into one set of robust rules, _R_ robust = LLM( _R, pa_ ). 

Follow the Rules: Reasoning for VAD with LLMs 7 

This module aims to mitigate errors from previous stages, such as the visual perception module’s potential misinterpretation of “walking” as “skateboarding”, leading to incorrect rules. The aggregation process filters out uncommon elements by retaining rule elements consistently present across the _n_ sets. The prompt _pa_ for the LLM to achieve this is detailed in Appendix A.2. This strategy is based on the assumption of _randomize smoothing_ [10], where errors may occur on a single input but are less likely to consistently occur across multiple randomly sampled inputs. Therefore, by aggregating these outputs, AnomalyRuler generates rules more resilient to individual errors. The hyperparameter _n_ can be treated as the number of batches. For simplicity, previous discussions assume that each batch has only one frame, i.e., _m_ = 1. Here we define _m_ as the number of normal reference frames per batch, i.e., batch size. We show the effectiveness of the rule aggregation and provide an ablation on different _n_ and _m_ values in Section 5.4. 

## **4 Deduction** 

After the induction stage derives a set of robust rules, the deduction stage follows these rules to perform VAD. The bottom part of Fig. 2 illustrates the deduction stage, which aims to precisely perceive each frame of videos and then use the LLM to reason if they are normal or abnormal based on the rules. To achieve this goal, we design three modules. First, the visual perception module works similarly as described in the induction stage. However, instead of taking the few-normalshot reference frames, the deduction processes continuous frames from each test video and outputs a series of frame descriptions _D_ = _{d_ 0 _, d_ 1 _, . . . , dt}_ . Second, the perception smoothing module reduces errors with the proposed Exponential Majority Smoothing. This step alone can provide preliminary detection results, referred to as AnomalyRuler-base. Third, the robust reasoning module utilizes an LLM to recheck the preliminary detection results against the rules and perform reasoning. The perception smoothing and robust reasoning modules are elaborated in the following sections. 

### **4.1 Perception Smoothing** 

As we discussed in Section 3.3, visual perception errors would happen in the induction stage, and this concern extends to the deduction stage as well. To address this challenge, we propose a novel mechanism named Exponential Majority Smoothing. This mechanism mitigates the errors by considering temporal consistency in videos, i.e., movements are continuous and should exhibit consistent patterns over time. We utilize the results of this smoothing to guide the correction of frame descriptions, enhancing AnomalyRuler’s robustness to errors. There are four key steps: 

**Initial Anomaly Matching.** For the continuous frame descriptions _D_ = _{d_ 0 _, d_ 1 _, . . . , dt}_ , AnomalyRuler first match anomaly keywords _K_ found within the anomaly rules from the induction stage (see details in Appendix A.2), and assigns _di_ with label _yi_ where _i ∈_ [0 _, t_ ], represents the predicted label. Formally, 

#### 8 Y. Yang et al. 

we have _yi_ = 1 if _∃k ∈ K ⊆ di_ , indicating an anomaly triggered by keywords such as ing-verb “riding” or “running”. Otherwise, _yi_ = 0 indicates the normal. We denote the initial matching predictions as _Y_ = _{y_ 0 _, y_ 1 _, . . . , yt}_ . **Exponential Majority Smoothing.** We propose an approach that combines Exponential Moving Average (EMA) and Majority Vote. This approach is designed to enhance the continuity in human or object movements by adjusting the predictions to reflect the most common state within a specified window. The final smoothed predictions are denoted as _Y_<sup>ˆ</sup> = _{y_ ˆ0 _,_ ˆ _y_ 1 _, . . . ,_ ˆ _yt}_ , where each _y_ ˆ _i_ is either 1 or 0. Formally, we have: 

- Step I: EMA. For original prediction _yt_ , the EMA value _st_ is computed as <u>�</u> _ti_ <u>=0</u><sup><u>(1</u></sup><sup>_−α_</sup><sup><u>)</u></sup><sup>_t−iyi_</sup> 

- _st_ = . We denote _α_ as the parameter that influences the weighting <u>�</u> _<u>ti</u>_ =0<sup>(1</sup><sup>_−α_)</sup><sup>_i_</sup> 

- of data points in the EMA calculation. 

- Step II: Majority Vote. The idea is to apply a majority vote to smooth the prediction within a window centered at each EMA value _si_ with a padding size _p_ . This means that for each _si_ , we consider its neighboring EMA values within the window and determine the smoothed prediction _y_ ˆ _i_ based on the majority of these values being above or below a threshold _τ_ . We define this threshold as the mean of all EMA values: _τ_ =<sup><u>1</u></sup> _t_ � _ti_ =1<sup>_si_.Formally,thesmoothedprediction</sup> _y_ ˆ _i_ is determined as: 



where 1 ( _·_ ) denotes the indicator function and the window size is adaptively defined as min( _i_ + _p, t_ ) _−_ max(1 _, i − p_ ) + 1 ensuring that the window does not extend beyond the boundaries determined by the range from max(1 _, i − p_ ) to min( _i_ + _p, t_ ). 

**Anomaly Score.** Given that _Y_<sup>ˆ</sup> represents the initial detection results of AnomalyRuler, we can further assess these by calculating an anomaly score through a secondary EMA. Specifically, the _anomaly scores_ , denoted as _A_ = _{a_ 0 _, a_ 1 _, . . . , at}_ , where _at_ is: 



We denote the above procedure AnomalyRuler-base as a baseline of our method, which provides a _dummy answer_ , i.e., “Anomaly” if ˆ _yi_ = 1 otherwise “Normal”, with an anomaly score that is comparable with the state-of-the-art VAD methods [3, 25, 35, 43]. Subsequently, AnomalyRuler utilizes the dummy answer in the robust reasoning module for further analysis. 

**Description Modification.** In this step, AnomalyRuler modifies the description _D_ comparing _Y_ and _Y_<sup>ˆ</sup> and outputs the modified _D_<sup>ˆ</sup> . If _yi_ = 0 while _y_ ˆ _i_ = 1, indicating a false negative in the perception module, AnomalyRuler corrects _di_ by adding “There is a person _{k}_ .”, where _k ∈ K_ is the most frequent anomaly keyword within the window size _w_ . Conversely, if _yi_ = 1 while _y_ ˆ _i_ = 0, indicating a false positive in the perception module, so AnomalyRuler modifies _di_ by removing parts of the description that contain the anomaly keyword _k_ . 

Follow the Rules: Reasoning for VAD with LLMs 9 

### **4.2 Robust Reasoning** 

In the robust reasoning module, AnomalyRuler utilizes an LLM to achieve the reasoning task for VAD, with the robust rule _R_ robust derived from the induction stage as the context. The LLM is fed with each frame’s modified description _d_<sup>ˆ</sup> _i_ with its dummy answer, i.e., either “Anomaly” or “Normal” generated from AnomalyRuler-base. We denote the output of robust reasoning as _Y_<sup>_∗_</sup> = _{_ LLM( _d_<sup>ˆ</sup> _i,_ ˆ _yi, R_ robust _, pr_ ) _| d_<sup>ˆ</sup> _i ∈ D,_<sup>ˆ</sup> ˆ _yi ∈ Y_<sup>ˆ</sup> _}_ . To ensure reliable results, the prompt _pr_ , detailed in Appendix A.2, guides the LLM to recheck whether the dummy answer _y_ ˆ _i_ matches the description _di_<sup>ˆ</sup> according to _R_ robust. This validation step, instead of directly asking the LLM to analyze _d_<sup>ˆ</sup> _i_ , improves decision-making by using the dummy answer as a hint. This approach helps AnomalyRuler reduce missed anomalies (false negatives) and ensures that its reasoning is more closely aligned with the rules. Additionally, to compare AnomalyRuler with the state-ofthe-art approaches based on thresholding anomaly scores, we apply Equation (2) with replacing _y_ ˆ _i_ by _yi_<sup>_∗∈Y∗_tooutputanomalyscores.</sup> 

## **5 Experiments** 

This section compares AnomalyRuler with LLM-based baselines and state-ofthe-art methods in terms of both detection and reasoning abilities. We also conduct an ablation study on each module within AnomalyRuler to evaluate their contributions. Examples of complete prompts, derived rules, and outputs are illustrated in Appendix A.2. 

### **5.1 Experimental Setup** 

**Datasets.** We evaluate our method on four VAD benchmark datasets. (1) UCSD Ped2 (Ped2) [22]: A single-scene dataset captured in pedestrian walkways with over 4,500 frames of videos, including anomalies such as skating and biking. (2) CUHK Avenue (Ave) [28]: A single-scene dataset captured in the CUHK campus avenue with over 30,000 frames of videos, including anomalies such as running and biking. (3) ShanghaiTech (ShT) [24]: A challenging dataset that contains 13 campus scenes with over 317,000 frames of videos, containing anomalies such as biking, fighting, and vehicles in pedestrian areas. (4) UBnormal (UB) [2]: An open-set virtual dataset generated by the Cinema4D software, which contains 29 scenes with over 236,000 frames of videos. For each dataset, we use the default training and test sets that adhere to the one-class setting. The normal reference frames used by AnomalyRuler are randomly sampled from the normal training set. The methods are evaluated on the entire test set if not otherwise specified. **Evaluation Metrics.** Following the common practice, we use the Area Under the receiver operating characteristic Curve (AUC) as the main detection performance metric. To compare with LLM-based methods that cannot output anomaly scores, we use the accuracy, precision, and recall metrics. Besides, we adopt the DoublyRight metric [32] to evaluate reasoning ability. All the metrics are calculated with frame-level ground truth labels. 

10 Y. Yang et al. 

**Table 1:** Detection performance with accuracy, precision, and recall (%) compared with different VAD with LLM methods on the ShT dataset. 

|**Method**|**Accuracy**|**Precision**|**Recall**|
|---|---|---|---|
|Ask LLM Directly|52.1|97.1|6.2|
|Ask LLM with Elhafsi et al. [12]|58.4|**97.9**|15.2|
|Ask Video-based LLM Directly|54.7|85.4|8.5|
|AnomalyRuler|**81.8**|90.2|**64.3**|



**Implementation Details.** We implement our method, AnomalyRuler, using PyTorch [37]. If not otherwise specified, we employ CogVLM-17B [51] as the VLM for visual perception, GPT-4-1106-Preview [1] as the LLM for induction, and the open-source Mistral-7B-Instruct-v0.2 [19] as the LLM for deduction (i.e., inference) due to using GPTs on entire test sets is too costly. We discuss other VLMs/LLMs choices in Appendix A.4. The default hyperparameters of AnomalyRuler are set as follows: The number of batches in rule aggregation _n_ = 10, the number of normal reference frames per batch _m_ = 1, the padding size _p_ = 5 in majority vote, and the weighting parameter _α_ = 0 _._ 33 in EMA. 

### **5.2 Comparison with LLM-based Baselines** 

Reasoning for one-class VAD using LLMs is not well-explored. To demonstrate AnomalyRuler’s superiority over the direct LLM use, we build asking LLM/Videobased LLM directly as baselines and also adapt related works [8, 12] to our target problem as baselines. At test time, let us denote test video frames as _F_ = _{f_ 1 _, f_ 2 _, . . . , ft}_ . We elaborate on our four baselines as follows. (1) Ask LLM Directly: _{_ LLM( _di, p_ ) _| di ∈ D}_ , where the LLM is Mistral-7B, _D_ is _F_ ’s frame descriptions generated by CogVLM, and _p_ is “Is this frame description anomaly or normal?” (2) Ask LLM with Elhafsi et al. [12]: _{_ LLM( _di, p_ ) _| di ∈ D}_ , where the LLM is Mistral-7B, _D_ is _F_ ’s frame descriptions generated by CogVLM, and _p_ is [12]’s prompts and predefined concepts of normality/anomaly. (3) Ask Video-based LLMs Directly: _{_ Video-based LLM( _ci, p_ ) _| ci ∈ C}_ , where _p_ is “Is this clip anomaly or normal?” We use Video-LLaMA [57] as the Videobased LLM, which performs clip-wise inference. Each video clip _ci_ consists of consecutive frames in _F_ with the same label. (4) Ask GPT-4V with Cao et al. [8]: _{_ GPT-4V( _fi, p_ ) _| fi ∈ F }_ , where _p_ is [8]’s prompts. As a large VLM, GPT-4V directly takes frames as inputs. 

**Detection Performance.** Table 1 compares the accuracy, precision, and recall on the ShT dataset. Overall, AnomalyRuler achieves significant improvements with an average increase of 26.2% in accuracy and 54.3% in recall. Such improvements are attributed to the reasoning based on the rules generated in the induction stage. In contrast, the baselines tend to predict most samples as normal based on the implicit knowledge pre-trained in LLMs, resulting in very low recall and accuracy close to a random guess. Their relatively high precision is due to that they rarely predict anomalies, leading to fewer false positives. 

Follow the Rules: Reasoning for VAD with LLMs 11 

**Table 2:** Reasoning performance with the Doubly-Right metric: {RR, RW, WR, WW} (%) on 100 (limited by GPT-4’s query capacity) randomly selected frames from the ShT test set. We evaluate cases with visual perception errors (w. Perception Errors) and with manually corrected visual perception (w/o. Perception Errors). 

|**Method**|**w. **|**Percep**|**tion Er**|**rors**|**w/o. **|**Perce**|**ption **|**Errors**|
|---|---|---|---|---|---|---|---|---|
||RR|RW|WR|WW|RR|RW|WR|WW|
|Ask GPT-4 Directly|57|4|15|24|73|3|0|24|
|Ask GPT-4 with Elhafsi et al. [12]|60|3|15|22|76|2|0|22|
|Ask GPT-4V with Cao et al. [8]|74|2|7|17|81|2|0|17|
|AnomalyRuler|**83**|1|15|1|**99**|0|0|1|



**Reasoning Performance.** The reasoning performance is evaluated using the Doubly-Right metric [32]: {RR, RW, WR, WW} (%), where RR denotes Right detection with Right reasoning, RW denotes Right detection with Wrong reasoning, WR denotes Right detection with Wrong reasoning, and WW denotes Wrong detection with Wrong reasoning. We desire a high accuracy of RR (the best is 100%) and low percentages of RW, WR and WW (the best is 0%). Since {RW, WR, WW} may be caused by visual perception errors rather than reasoning errors, we also consider the case with manually corrected visual perception to exclusively evaluate each method’s reasoning ability, i.e., w. Perception Errors vs. w/o. Perception Errors in Table 2. 

Due to the lack of benchmarks for evaluating reasoning for VAD, we create a dataset consisting of 100 randomly selected frames from the ShT test set, with an equal split of 50 normal and 50 abnormal frames. For each frame, we offer four choices: one normal and three anomalies, where only one choice with the matched rules is labeled as RR, while the other choices correspond to RW, WR or WW. Details and examples of this dataset are illustrated in Appendix A.3. Since the 100 randomly selected frames are not consecutive, here AnomalyRuler’s perception smoothing is not used. 

Table 2 shows the evaluation results. With perception errors, AnomalyRuler outperforms the baselines by 10% to 27% RR, and it achieves a very low WW of 1% compared to the 17% WW of the second best Ask GPT-4V with Cao et al. [8]. Without perception errors, AnomalyRuler’s RR jumps to 99%. These results demonstrate AnomalyRuler’s superiority over the GPT-4(V) baselines and its great ability to make correct detection along with correct reasoning. 

### **5.3 Comparison with State-of-the-Art Methods** 

This section compares AnomalyRuler with 15 state-of-the-art one-class VAD methods across four datasets, evaluating their detection performance and domain adaptability. The performance values of these methods are sourced from their respective original papers. 

**Detection Performance.** Table 3 shows the effectiveness of AnomalyRuler. There are three main observations. First, AnomalyRuler, even with its basic version AnomalyRuler-base, outperforms all the Image-Only competitors, which 

12 Y. Yang et al. 

**Table 3:** AUC (%) compared with different one-class VAD methods. “Image Only” methods only rely on image features. In contrast, others employ additional features such as bounding boxes from object detectors or 3D features from action recognition networks. “Training” indicates the methods that need a full-shot training process. 

|**Method**|**Venue**|**Image Only**|**Training**|**Ped2**|**Ave**|**ShT**|**UB**|
|---|---|---|---|---|---|---|---|
|MNAD [36]<br>|CVPR-20|✓|✓|97.0|88.5|70.5|-|
|rGAN [29]<br>|ECCV-20|✓|✓|96.2|85.8|77.9|-|
|CDAE [9]|ECCV-20|✓|✓|96.5|86.0|73.3|-|
|MPN [30]|CVPR-21|✓|✓|96.9|89.5|73.8|-|
|NGOF [50]<br>|CVPR-21|✗|✓|94.2|88.4|75.3|-|
|HF2 [25]|ICCV-21|✗|✓|**99.2**|91.1|76.2|-|
|BAF [14]|TPAMI-21|✗|✓|98.7|92.3|82.7|59.3|
|GCL [56]<br>|CVPR-22|✗|✓|-|-|79.6|-|
|S3R [53]|ECCV-22|✗|✓|-|-|80.5|-|
|SSL [49]<br>|ECCV-22|✗|✓|99.0|**92.2**|84.3|-|
|zxVAD [3]|WACV-23|✗|✓|96.9|-|71.6|-|
|HSC [45]|CVPR-23|✗|✓|98.1|93.7|83.4|-|
|FPDM [54]|ICCV-23|✓|✓|-|90.1|78.6|62.7|
|SLM [43]|ICCV-23|✓|✓|97.6|90.9|78.8|-|
|STG-NF [18]|ICCV-23|✗|✓|-|-|**85.9**|71.8|
|AnomalyRuler-base|-|✓|✗|96.5|82.2|84.6|69.8|
|AnomalyRuler|-|✓|✗|97.9|89.7|85.2|**71.9**|



**Table 4:** AUC (%) compared with different cross-domain VAD methods. We follow the compared works to use ShT as the source domain dataset for other target datasets. 

|**Method**|**Venue**|**Image Only**|**Training**|**Ped2**|**Ave**|**ShT**<sup>**1**</sup>|**UB**|
|---|---|---|---|---|---|---|---|
|rGAN [29]|ECCV-20|✓|✓|81.9|71.4|77.9|-|
|MPN [30]|CVPR-21|✓|✓|84.7|74.1|73.8|-|
|zxVAD [3]|WACV-23|✗|✓|95.7|**82.2**|71.6|-|
|AnomalyRuler-base|-|✓|✗|**97.4**|81.6|**83.5**|**65.4**|



> 1 AnomalyRuler employs UB as the source domain when ShT serves as the target domain. The competitors have no cross-domain evaluation on ShT, so we report their same-domain results. 

also do not use any additional features (e.g., bounding boxes from object detectors or 3D features from action recognition networks), on the challenging ShT and UB datasets. This suggests that our rule-based reasoning benefits the challenging oneclass VAD task. Second, for Ped2 and Ave, AnomalyRuler performs on par with the Image-Only methods. This is achieved without any tuning, meaning that our few-normal-shot prompting approach is as effective as the costly full-shot training on these benchmarks. Third, AnomalyRuler outperforms AnomalyRuler-base, indicating that the robust reasoning module improves performance further. 

**Domain Adaptability.** Domain adaptation considers the scenario that the source domain (i.e., training/induction) dataset differs from the target domain (i.e., testing/deduction) dataset [13, 26, 48]. We compare AnomalyRuler with three state-of-the-art VAD methods that claim their domain adaptation ability [3, 29, 30]. We follow the compared works to use ShT as the source domain dataset for other target datasets. As shown in Table 4, AnomalyRuler achieves the highest AUC on Ped2, ShT and UB, outperforming with an average of 9.88%. While AnomalyRuler trails zxVAD [3] by 0.6%, it is still higher than the others with an average of 8.85%. The results indicate that AnomalyRuler has better 

Follow the Rules: Reasoning for VAD with LLMs 13 

domain adaptability across different datasets. This advantage is due to that the language provides consistent descriptions across different visual domains, which allows the application of induced rules to datasets with similar anomaly scenarios but distinct visual appearances. In contrast, traditional methods extract high-dimensional visual features that are sensitive to visual appearances, thereby struggling to transfer their knowledge across datasets. 

### **5.4 Ablation Study** 

In this section, we look into how the proposed strategies affect AnomalyRuler. We investigate two aspects: rule quantity (i.e., the number of induced rules) and rule quality (i.e., their resulting performance). Regarding this, we evaluate variants of AnomalyRuler-base on the ShT dataset. 

**Ablation on Strategies.** Table 5 shows the effects of removing individual strategies compared to using all strategies. In terms of rule quantity, removing Human and Environment or Normal and Anomaly significantly reduces rules by 47.6% and 82.4%, respectively. This reduction is due to not separating the rules for humans and the environment halves the number of rules. Moreover, without deriving anomaly rules from normal rules, we only have a limited set of normal rules. Removing Abstract and Concrete or Rule Aggregation slightly increases the number of rules, as the former merges rules within the same categories and the latter removes incorrect rules. Perception Smoothing does not affect rule quantity since it is used in the deduction stage. In terms of rule quality, removing Normal and Anomaly or Rule Aggregation has the most negative impact. The former happens because when only normal rules are present, the LLM overreacts to slightly different actions such as “walking with an umbrella” compared to the rule for “walking”, leading to false positives. Furthermore, without rules for anomalies as a reference, the LLM easily misses anomalies. The latter is due to that perception errors in the induction stage would lead to incorrect rules for normal. Besides, removing other strategies also decreases AUC, underscoring their significance. In summary, the proposed strategies effectively improve AnomalyRuler’s performance. There is no direct positive/negative correlation between rule quantity and quality, i.e., having too few rules leads to inadequate coverage of normality and anomaly concepts while having too many rules would cause redundancy and errors. **Ablation on Hyperparameters.** Fig. 3 illustrates the effects of the hyperparameters in the rule aggregation and perception smoothing modules. For rule aggregation, we conduct cross-validation on the number of batches _n_ = [1, 5, 10, 20] and the number of normal reference frames per batch _m_ = [1, 2, 5, 10]. We observe that both the number of rules and AUC increase with the increases of _n_ and _m_ , but they start to fluctuate when _n × m_ becomes large. For example, when _n_ = 20, AUC drops from 85.9% to 72.2% as _m_ increases because having too many reference frames (e.g., over 100) results in redundant information in a long context. For perception smoothing, we test the padding size in majority vote _p_ = [1, 5, 10, 20] and the weighting parameter in EMA _α_ = [0.09, 0.18, 0.33, 1]. We 

14 Y. Yang et al. 

**Table 5:** Ablation on strategies. We assess the effects of removing individual strategies in AnomalyRuler. We conduct the experiments five times with different randomly selected normal reference frames for induction and report their mean and standard deviation on the ShT dataset. 

|**Strategy**|**Stage**|**# R**<br>|**ules**<br>|**Accu**|**racy**<br>|**Prec**<br>|**ision**<br>|**Re**|**call**<br>|**AU**<br>|**C**<br>|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||mean|std|mean|std|mean|std|mean|std|mean|std|
|w. All Below (default)|Both|42.2|4.2|81.6|1.3|90.9|0.8|63.9|2.7|84.5|1.1|
|w/o. Human and Environment|Both|-20.1|+1.1|-3.3|+0.8|-3.9|+0.8|-1.9|+1.6|-2.4|+2.0|
|w/o. Normal and Anomaly|Induction|-34.8|-1.3|-20.5|+4.3|-41.2|+7.0|-14.4|+11.6|-18.8|+1.2|
|w/o. Abstract and Concrete|Induction|+2.3|+2.7|-0.6|-0.2|-0.9|-0.2|-0.3|-0.4|-0.9|+0.1|
|w/o. Rule Aggregation|Induction|+8.5|+6.1|-9.6|+ 14.7|+1.1|+2.9|-10.7|+14.1|-15.8|+0.8|
|w/o. Perception Smoothing<br>|Deduction|NA|NA|-1.7|-0.9|-1.9|+0.1|-3.8|-0.3|-3.3|+0.8|





<!-- Start of picture text -->
50 1.0 1.0<br>4540 3040 0.840.82 0.80.9 0.840.82 0.80.9<br>35 20 0.80 0.7 0.80 0.7<br>30 10 0.78 0.6 0.78 0.6<br>25 0 0.76 0.5 0.76 0.5<br>20 10 0.74 10 0.74 1.0<br>15 0.8<br>1 5 10 5 1 5 10 5 1 5 10 0.4 0.6<br>20 1 2 20 1 2 20 0.2<br>(a) n & m vs. # Rules (b) n & m vs. AUC (c) p & α vs. AUC<br># Batch (n) # Batch (n) Padding size (p)<br>Batch Size (m) Batch Size (m)<br># Rules<br>AUC AUC<br><!-- End of picture text -->

**Fig. 3:** Ablation on hyperparameters of the (a) (b) rule aggregation and (c) perception smoothing modules on the ShT dataset. 

found _p_ = 5 to be optimal for capturing the motion continuity in a video while avoiding the excessive noise that can occur with more neighborhoods. _α_ adjusts the weight of the most recent frames compared to previous frames. A smaller _α_ emphasizes previous frames, resulting in more smoothing but less responsiveness to recent changes. In general, increasing _α_ from 0.09 to 0.33 improves AUC, suggesting that moderate EMA smoothing is beneficial. 

## **6 Conclusion** 

In this paper, we propose AnomalyRuler, a novel rule-based reasoning framework for VAD with LLMs. With the induction and deduction stages, AnomalyRuler requires only few-normal-shot prompting without the need for expensive full-shot tuning, thereby fast steering LLMs’ reasoning strengths to various specific VAD applications. To the best of our knowledge, AnomalyRuler is the first reasoning approach for one-class VAD. Extensive experiments demonstrate AnomalyRuler’s state-of-the-art performance, reasoning ability, and domain adaptability. Limitations and potential negative social impact of this work are discussed in the Appendix A.1. In future research, we expect this work to advance broader oneclass problems and related tasks, such as industrial anomaly detection [6, 55], open-set recognition [5, 40], and out-of-distribution detection [17, 42]. 

Follow the Rules: Reasoning for VAD with LLMs 

15 

## **Acknowledgments** 

This work was supported in part by National Science Foundation (NSF) under grants OAC-23-19742 and Johns Hopkins University Institute for Assured Autonomy (IAA) with grants 80052272 and 80052273. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of NSF or JHU-IAA. 

## **References** 

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023) 

2. Acsintoae, A., Florescu, A., Georgescu, M.I., Mare, T., Sumedrea, P., Ionescu, R.T., Khan, F.S., Shah, M.: Ubnormal: New benchmark for supervised open-set video anomaly detection. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022) 

3. Aich, A., Peng, K.C., Roy-Chowdhury, A.K.: Cross-domain video anomaly detection without target domain adaptation. In: IEEE/CVF Winter Conference on Applications of Computer Vision (2023) 

4. Bacon, F.: Novum organum (1620) 

5. Bendale, A., Boult, T.E.: Towards open set deep networks. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2016) 

6. Bergmann, P., Fauser, M., Sattlegger, D., Steger, C.: Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019) 

7. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: Conference on Neural Information Processing Systems (2020) 

8. Cao, Y., Xu, X., Sun, C., Huang, X., Shen, W.: Towards generic anomaly detection and understanding: Large-scale visual-linguistic model (gpt-4v) takes the lead. arXiv preprint arXiv:2311.02782 (2023) 

9. Chang, Y., Tu, Z., Xie, W., Yuan, J.: Clustering driven deep autoencoder for video anomaly detection. In: European Conference on Computer Vision (2020) 

10. Cohen, J., Rosenfeld, E., Kolter, Z.: Certified adversarial robustness via randomized smoothing. In: International Conference on Machine Learning (2019) 

11. Diao, S., Wang, P., Lin, Y., Zhang, T.: Active prompting with chain-of-thought for large language models. arXiv preprint arXiv:2302.12246 (2023) 

12. Elhafsi, A., Sinha, R., Agia, C., Schmerling, E., Nesnas, I.A., Pavone, M.: Semantic anomaly detection with large language models. In: Autonomous Robots (2023) 

13. Ganin, Y., Lempitsky, V.: Unsupervised domain adaptation by backpropagation. In: International Conference on Machine Learning (2015) 

14. Georgescu, M.I., Ionescu, R.T., Khan, F.S., Popescu, M., Shah, M.: A backgroundagnostic framework with adversarial training for abnormal event detection in video. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2021) 

15. Gu, Z., Zhu, B., Zhu, G., Chen, Y., Tang, M., Wang, J.: Anomalygpt: Detecting industrial anomalies using large vision-language models. In: AAAI Conference on Artificial Intelligence (2024) 

16 Y. Yang et al. 

16. Hasan, M., Choi, J., Neumann, J., Roy-Chowdhury, A.K., Davis, L.S.: Learning temporal regularity in video sequences. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2016) 

17. Hendrycks, D., Gimpel, K.: A baseline for detecting misclassified and out-ofdistribution examples in neural networks. In: International Conference on Learning Representations (2017) 

18. Hirschorn, O., Avidan, S.: Normalizing flows for human pose anomaly detection. In: IEEE/CVF International Conference on Computer Vision (2023) 

19. Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., Casas, D.d.l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al.: Mistral 7b. arXiv preprint arXiv:2310.06825 (2023) 

20. Lee, S., Kim, G.: Recursion of thought: A divide-and-conquer approach to multicontext reasoning with language models. In: Annual Meeting of the Association for Computational Linguistics (2023) 

21. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International Conference on Machine Learning (2023) 

22. Li, W., Mahadevan, V., Vasconcelos, N.: Anomaly detection and localization in crowded scenes. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2013) 

23. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Conference on Neural Information Processing Systems (2023) 

24. Liu, W., Luo, W., Lian, D., Gao, S.: Future frame prediction for anomaly detection– a new baseline. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018) 

25. Liu, Z., Nie, Y., Long, C., Zhang, Q., Li, G.: A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In: IEEE/CVF International Conference on Computer Vision (2021) 

26. Lo, S.Y., Oza, P., Chennupati, S., Galindo, A., Patel, V.M.: Spatio-temporal pixel-level contrastive learning-based source-free domain adaptation for video semantic segmentation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023) 

27. Lo, S.Y., Oza, P., Patel, V.M.: Adversarially robust one-class novelty detection. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2022) 

28. Lu, C., Shi, J., Jia, J.: Abnormal event detection at 150 fps in matlab. In: IEEE/CVF International Conference on Computer Vision (2013) 

29. Lu, Y., Yu, F., Reddy, M.K.K., Wang, Y.: Few-shot scene-adaptive anomaly detection. In: European Conference on Computer Vision (2020) 

30. Lv, H., Chen, C., Cui, Z., Xu, C., Li, Y., Yang, J.: Learning normal dynamics in videos with meta prototype network. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021) 

31. Lv, H., Sun, Q.: Video anomaly detection and explanation via large language models. arXiv preprint arXiv:2401.05702 (2024) 

32. Mao, C., Teotia, R., Sundar, A., Menon, S., Yang, J., Wang, X., Vondrick, C.: Doubly right object recognition: A why prompt for visual rationales. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023) 

33. Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D., Dosovitskiy, A., Mahendran, A., Arnab, A., Dehghani, M., Shen, Z., et al.: Simple open-vocabulary object detection. In: European Conference on Computer Vision (2022) 

Follow the Rules: Reasoning for VAD with LLMs 17 

34. Mittal, H., Agarwal, N., Lo, S.Y., Lee, K.: Can’t make an omelette without breaking some eggs: Plausible action anticipation using large video-language models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024) 

35. Morais, R., Le, V., Tran, T., Saha, B., Mansour, M., Venkatesh, S.: Learning regularity in skeleton trajectories for anomaly detection in videos. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019) 

36. Park, H., Noh, J., Ham, B.: Learning memory-guided normality for anomaly detection. In: IEEE/CVF Conference Computer Vision and Pattern Recognition (2020) 

37. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. In: Conference on Neural Information Processing Systems (2019) 

38. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al.: Improving language understanding by generative pre-training. OpenAI Blog (2018) 

39. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI Blog (2019) 

40. Safaei, B., Vibashan, V., de Melo, C.M., Hu, S., Patel, V.M.: Open-set automatic target recognition. In: IEEE International Conference on Acoustics, Speech and Signal Processing (2023) 

41. Seel, N.M.: Encyclopedia of the sciences of learning (2011) 

42. Sharifi, S., Entesari, T., Safaei, B., Patel, V.M., Fazlyab, M.: Gradient-regularized out-of-distribution detection. In: European Conference on Computer Vision (2024) 

43. Shi, C., Sun, C., Wu, Y., Jia, Y.: Video anomaly detection via sequentially learning multiple pretext tasks. In: IEEE/CVF International Conference on Computer Vision (2023) 

44. Su, Y., Lan, T., Li, H., Xu, J., Wang, Y., Cai, D.: Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355 (2023) 

45. Sun, S., Gong, X.: Hierarchical semantic contrast for scene-aware video anomaly detection. In: IEEE/CVF Computer Vision and Pattern Recognition Conference (2023) 

46. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023) 

47. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023) 

48. Tsai, Y.H., Hung, W.C., Schulter, S., Sohn, K., Yang, M.H., Chandraker, M.: Learning to adapt structured output space for semantic segmentation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018) 

49. Wang, G., Wang, Y., Qin, J., Zhang, D., Bao, X., Huang, D.: Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles. In: European Conference on Computer Vision (2022) 

50. Wang, H., Zhang, X., Yang, S., Zhang, W.: Video anomaly detection by the duality of normality-granted optical flow. arXiv preprint arXiv:2105.04302 (2021) 

51. Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., et al.: Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079 (2023) 

52. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. In: Conference on Neural Information Processing Systems (2022) 

18 Y. Yang et al. 

53. Wu, J.C., Hsieh, H.Y., Chen, D.J., Fuh, C.S., Liu, T.L.: Self-supervised sparse representation for video anomaly detection. In: European Conference on Computer Vision (2022) 

54. Yan, C., Zhang, S., Liu, Y., Pang, G., Wang, W.: Feature prediction diffusion model for video anomaly detection. In: IEEE/CVF International Conference on Computer Vision (2023) 

55. You, Z., Cui, L., Shen, Y., Yang, K., Lu, X., Zheng, Y., Le, X.: A unified model for multi-class anomaly detection. Conference on Neural Information Processing Systems (2022) 

56. Zaheer, M.Z., Mahmood, A., Khan, M.H., Segu, M., Yu, F., Lee, S.I.: Generative cooperative learning for unsupervised video anomaly detection. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022) 

57. Zhang, H., Li, X., Bing, L.: Video-llama: An instruction-tuned audio-visual language model for video understanding. In: Conference on Empirical Methods in Natural Language Processing (2023) 

58. Zhang, Y., Huang, X., Ma, J., Li, Z., Luo, Z., Xie, Y., Qin, Y., Luo, T., Li, Y., Liu, S., et al.: Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514 (2023) 

59. Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., et al.: Least-to-most prompting enables complex reasoning in large language models. In: International Conference on Learning Representations (2023) 

60. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. In: International Conference on Learning Representations (2024) 

61. Zhu, Z., Xue, Y., Chen, X., Zhou, D., Tang, J., Schuurmans, D., Dai, H.: Large language models can learn rules. arXiv preprint arXiv:2310.07064 (2023) 

Follow the Rules: Reasoning for VAD with LLMs 

19 

## **A Appendix** 

### **A.1 Limitations and Potential Negative Social Impact** 

**Limitations.** Similar to most existing LLM-based studies, AnomalyRuler assumes that the employed LLM backbones have decent capabilities. Sub-optimal LLMs may hinder the effectiveness of the methods. Exploring this limitation further could be an interesting future investigation. 

**Potential Negative Social Impact.** The proposed method may enable malicious actors to more easily adapt VLMs/LLMs for illegal surveillance. To mitigate this risk, computer security mechanisms could be integrated. 

### **A.2 Examples of Input Prompts and Outputs Results** 

**Induction.** This stage starts from _n_ randomly chosen normal reference frames _F_ normal = _{f_ normal1 _, . . . , f_ normal _n}_ and outputs a set of robust rules _R_ robust. To simplify the illustration, we show one frame _f_ normal _i ∈ F_ normal as an example in the visual perception and rule generation steps. 

- Visual Perception 

   - Input _f_ normal _i_ and prompt _pv_ : 



_pv_ `= How many people are in the image and what is each of them doing? What are in the image other than people? Think step by step.` 

- Outputs: Frame description _d_ normal _i_ = _V LM_ ( _f_ normal _i, pv_ ) 

_d_ `normal` _i_ `= There are four people in the image. Starting from the left , the first person is walking on the path. The second person is walking on the bridge. The third person is also walking on the bridge. The fourth person is also walking on the bridge. Other than people , there are trees , a railing , a path , and a bridge visible in the image.` 

- Rule Generation 

   - Input prompt _pg_ : 

_pg_ `= [ {‘‘role ’’: ‘‘system ’’, ‘‘content ’’: As a surveillance monitor for urban safety using the ShanghaiTech dataset , my job is to derive rules for detecting abnormal human activities or environmental objects .}, {‘‘role ’’: ‘‘user ’’, ‘‘content ’’: Based on the assumption that the given frame descriptions are normal , Please derive rules for normal , start from an abstract concept , and then generalize to concrete activities or objects .}, {‘‘role ’’: ‘‘assistant ’’, ‘‘content ’’: ** Rules for Normal Human Activities: 1. ** Rules for Normal Environmental Objects: 1. }, {‘‘role ’’: ‘‘user ’’,` 

20 Y. Yang et al. 

`‘‘content ’’: Compared with the above rules for normal , can you provide potential rules for anomaly? Please start from an abstract concept then generalize to concrete activities or objects , compared with normal ones.}, {‘‘role ’’: ‘‘assistant ’’, ‘‘content ’’: ** Rules for Anomaly Human Activities : 1. ** Rules for Anomaly Environmental Objects: 1. }, {‘‘role ’’: ‘‘user ’’, ‘‘content ’’: Now you are given frame description {` _d_ `normal` _i_ `}. What are the Normal and Anomaly rules you have? Think step by step. Reply following the above format , start from an abstract concept and then generalize to concrete activities or objects . List them using short terms , not an entire sentence .}, ]` 

   - Outputs: For each normal reference frame _d_ normal _i_ , we will get one set of rules _ri_ = _LLM_ ( _d_ normal _i, pg_ ). Since the structure of the rules is identical to the robust rules, we only present the robust rules in the following step as an illustration of our final induction output. 

- Rule Aggregation 

   - Input prompt _pa_ : 

_pa_ `= [ {‘‘role ’’: ‘‘system ’’, ‘‘content ’’: As a surveillance monitor for urban safety using the ShanghaiTech dataset , my job is to organize rules for detecting abnormal activities and objects .}, {‘‘role ’’: ‘‘user ’’, ‘‘content ’’: You are given {` _n_ `} independent sets of rules for Normal and Anomaly. For the organized normal Rules , list the given normal rules with high -frequency elements For the organized anomaly Rules , list all the given anomaly rules}, {‘‘role ’’: ‘‘assistant ’’, ‘‘content ’’: ** Rules for Anomaly Human Activities : 1. ** Rules for Anomaly Environmental Objects: 1. ** Rules for Normal Human Activities: 1. ** Rules for Normal Environmental Objects: 1. }, {‘‘role ’’: ‘‘user ’’, ‘‘content ’’: Now you are given {` _n_ `} independent sets of rules as the sublists of {` _R_ `}. What rules for Anomaly and Normal do you get? Think step by step , and reply following the above format .}, ]` 

**–** Outputs: Robust rules _R_ robust = _LLM_ ( _R_ = _{r_ 1 _, . . . , rn}, pa_ ) 

```
** RulesforAnomalyHumanActivities :**
1.Usinganynon -walkingmovementsuchasridingabicycleorscooter ,
skateboarding
```

`2. Running or jumping 3. Moving with the usual speed such as very fast or in a hurry 4. Person lying or bending down on the ground 5. Aggressive or unusual movements such as fighting or pushing` 

`6. Loitering around the trash bin` 

Follow the Rules: Reasoning for VAD with LLMs 21 

|`7.`<br>`8.`<br>`9.`<br>`10`<br>`11`|`Climbing`<br>`on trees or into`<br>`manhole/utility`<br>`access`<br>`points`<br>`Tampering`<br>`with`<br>`manhole`<br>`covers`<br>`Engaging`<br>`in`<br>`physical`<br>`altercation`<br>`.`<br>`Lingering`<br>`without a clear`<br>`purpose`<br>`near a sensitive`<br>`area`<br>`.`<br>`Falling`<br>`unexpectedly`|
|---|---|
|`**`<br>`1. `<br>`2. `<br>`3.`<br>`4.`<br>`5.`<br>`6.`<br>`7.`<br>`8. `<br>`9.`<br>`10`<br>`11`<br>`12`<br>`13`<br>`14`|`Rules`<br>`for`<br>`Anomaly`<br>`Environmental`<br>`Objects :**`<br> `Missing , removed , or ajar`<br>`manhole`<br>`covers`<br> `Ground`<br>`transportation`<br>`such as vehicles , vans , bicycles.`<br>`Unattended`<br>`bags or`<br>`objects (security`<br>`threat)`<br>`Unusual`<br>`objects (vehicles , large`<br>`unattended`<br>`luggage , etc .)`<br>`Unauthorized`<br>`posters`<br>`or signs`<br>`Damaged`<br>`manhole`<br>`cover or trees`<br>`Abandoned`<br>`items in a public`<br>`area`<br> `Yellow`<br>`objects (when`<br>`not`<br>`commonly`<br>`seen)`<br>`Graffiti`<br>`on`<br>`building`<br>`windows`<br>`or`<br>`vandalism`<br>`to a fence`<br>`.`<br>`Obstructions`<br>`on`<br>`pedestrian`<br>`crossing`<br>`.`<br>`Fire`<br>`hydrant`<br>`spraying`<br>`water`<br>`without`<br>`presence`<br>`of`<br>`emergency`<br>`services`<br>`.`<br>`Unidentified`<br>`objects`<br>`near`<br>`the`<br>`tall`<br>`structure`<br>`.`<br>`Smoke or fire`<br>`coming`<br>`from a utility`<br>`access`<br>`point`<br>`.`<br>`Objects`<br>`thrown or`<br>`falling`<br>`from a height`|
|`**`|`Rules`<br>`for`<br>`Normal`<br>`Human`<br>`Activities :**`|
|`1.`<br>`2.`<br>`3.`<br>`4.`|`Walking`<br>`alone or with`<br>`another`<br>`person`<br>`Standing`<br>`and`<br>`using a mobile`<br>`device`<br>`Walking`<br>`on the`<br>`sidewalk`<br>`Walking`<br>`towards`<br>`or away`<br>`from a building`|
|`5.`<br>`6.`|`Entering a building`<br>`Standing`<br>`near a trash`<br>`bin`|
|`7.`|`Waiting`<br>`at`<br>`traffic`<br>`lights`|
|`8.`|`Walking`<br>`on`<br>`pedestrian`<br>`crossing`|
|`9.`|`Sitting`<br>`on bench or`<br>`ground`|
|`**`<br>`1.`<br>`2. `<br>`3. `<br>`4.`<br>`5. `<br>`6. `<br>`7.`<br>`8.`|`Rules`<br>`for`<br>`Normal`<br>`Environmental`<br>`Objects :**`<br>`Manhole`<br>`covers on the`<br>`ground`<br>`or street`<br> `Trees`<br>`along`<br>`the`<br>`street or Plants`<br>`present`<br> `Trash`<br>`bin in`<br>`vicinity`<br>`or placed`<br>`beside`<br>`the`<br>`street`<br>`Posters`<br>`on glass`<br>`door`<br> `Static`<br>`building`<br>`with`<br>`glass`<br>`windows`<br> `Fence`<br>`along`<br>`the`<br>`water`<br>`Pedestrian`<br>`crossing`<br>`markings`<br>`visible`<br>`Standing`<br>`fire`<br>`hydrant`|
|`9. `|`Static`<br>`tall`<br>`structure`<br>`in the`<br>`background`|
|`10`|`.`<br>`Utility`<br>`access`<br>`points on the`<br>`ground`|



**Deduction.** This stage starts from a test video that contains _t_ continuous frames _F_ = _{f_ 1 _, . . . , ft}_ and outputs the reasoning results _Y_<sup>ˆ</sup><sup>_∗_</sup> = _{y_ ˆ1<sup>_∗,. . . ,y_ˆ</sup> _t_<sup>_∗}_.To</sup> simplify the illustration, we show two frames of this test video, _fi, fj ∈ F_ as examples that represent one anomaly frame and one normal frame, respectively. 

- Visual Perception: 

   - Input test frames _fi_ , _fj_ and prompt _pv_ : 





_pv_ `= How many people are in the image and what is each of them doing? What are in the image other than people? Think step by step.` _pv_ `= How many people are in the image and what is each of them doing? What are in the image other than people? Think step by step.` 

- Outputs: Frame descriptions _di_ = _V LM_ ( _fi, pv_ ), _dj_ = _V LM_ ( _fj, pv_ ) 

- 22 Y. Yang et al. 

   - _di_ `= There are four people in the image. One person is walking , another is also walking , the third person is riding a bicycle , and the fourth person is walking near the bicycle. Other than people , there are trees , a pathway , a trash bin , a bicycle , and two manhole covers visible in the image.` 

_dj_ `= There are two people in the image. One person appears to be walking , the other seems to be walking together. Other than people , there are two manhole covers on the ground , a trash bin , and some trees and plants.` 

- Perception Smoothing: 

   - _R_ robust _→ K_ (generate anomaly keywords from anomaly rules, see Section 4). 

      - Input prompt _pk_ : 

_pk_ `= You will be given a set of rules for detecting abnormal activities and objects; please extract the anomaly keywords , activities using ‘‘ing ’’ verbs , and anomaly objects using nouns , and provide a combined Python list with each represented by a single word. The output should be in the format: [" object1", ..., "activity1", "activity2", ...]. Now you are given` _{R_ `robust` _}_ : 

- Output _K_ : 

```
anomaly_from_rule=[" trolley","cart","luggage","bicycle","
skateboard","scooter","vehicles","vans","accident","
running","jumping","riding"," skateboarding ","scooting","
lying","falling","bending","fighting","pushing","
loitering","climbing","tampering","lingering "]
```

   - _di → d_<sup>ˆ</sup> _i_ & ˆ _yi_ , _dj → d_<sup>ˆ</sup> _j_ & ˆ _yj_ (generate modified descriptions and smoothed predictions, see Section 4). 

- Robust Reasoning: 

   - Input prompt _pr_ : 

_pr_ `= You will be given an description of frame , your task is to double check my initial anomaly detection result based on the rules. The rules are: {` _R_ `robust } My initial result is {` _y_ ˆ _i_ `} First , if human activity is present , which rule is matching? List the rule category , e.g., normal or anomaly , with the rule number. Second , if an environmental object is present , which rule is matching? List the rule category , e.g., normal or anomaly , with the rule number. Third , are the human activities or environmental objects anomaly? Answer: anomaly , if ANY anomaly rule (even if only one , no matter human activities or environmental objects) matches , otherwise answer: normal. Now you are given the frame {` _d_<sup>ˆ</sup> _i_ `}, think step by step.` 

- Final reasoning output: _yi_<sup>_∗_=</sup><sup>_LLM_( ˆ</sup><sup>_di,_ˆ</sup><sup>_yi, R_robust</sup><sup>_, pr_)</sup> 

_y_ ˆ _i_<sup>_∗_</sup> `= The third person is riding a bicycle , which matches rule number 1 under Anomaly Human Activities. The bicycle matches rule number 2 under Anomaly Environmental Objects. Answer: anomaly.` 

Follow the Rules: Reasoning for VAD with LLMs 23 

_yj_<sup>_∗_=</sup><sup>_LLM_( ˆ</sup><sup>_dj,_ˆ</sup><sup>_yj, R_robust</sup><sup>_, pr_)</sup> 

- _y_ ˆ _j_<sup>_∗_</sup> `= The human activities present are two people walking. The rules for normal human activities include walking alone or with another` 

- `person. So , the rule matching is:` 

- `1. Walking alone or with another person (normal) The environmental objects present are two manhole covers on the ground , a trash bin , and some trees and plants. So , the rules matching are:` 

- `1. Manhole covers on the ground or street (normal) 2. Trees along the street or Plants present (normal) 3. Trash bin in vicinity or placed beside the street (normal) Answer: normal.` 

### **A.3 Examples of Doubly-Right Reasoning Evaluation** 

In Section 5.2, we evaluate reasoning ability using the Doubly-Right metric [32]. We create a benchmark dataset with multiple choices for Doubly-Right reasoning evaluation. The evaluation is conducted in the deduction stage, where we input the visual perception description and the induced rules to an LLM. The goal is to demonstrate that the induced rules enable LLMs to perform correct reasoning. 

We list the prompt for reasoning evaluation below and one example of the description and its four choices as Table 6. The content in normal choice is fixed, while the anomaly choices include one correct reasoning with a matched rule and two randomly chosen non-matched rules from our generated anomaly rules. In this example, Choices A, B, C and D correspond to RW, WW, RR and RW, respectively. 

`[ {‘‘role ’’: ‘‘system ’’, ‘‘content ’’: You will be given a description of the frame and four choices. Your task is to make the correct choice based on the rules. The rules are: {` _R_ `robust }}, {‘‘role ’’: ‘‘user ’’, ‘‘content ’’: Description: {` _d_<sup>ˆ</sup> _i_ `} Choices: {Four Choices} Choose just one correct answer from the options (A, B, C, or D) and output without any explanation . Please Answer :}, ]` 

### **A.4 Different VLMs/LLMs as Backbones** 

Table 7 shows the results of using various VLMs/LLMs as backbones in the deduction stage, compared to the default setting (the first row). All the results are based on the same rules derived in the induction stage with the default setting. 

We categorize the comparisons into three types: (1) VLMs only: AnomalyRuler uses the same VLMs as an end-to-end solution, combining visual perception and robust reasoning. It inputs the test frame and outputs the reasoning result. This category includes GPT-4V [1], LLaVA [23], and PandaGPT [44]. (2) VLMs + Mistrial [19]: We keep Mistrial as the default LLM for robust reasoning and test different VLMs (e.g., OWLViT [33], LLaVA, BLIP-2 [21], RAM [58]) for visual 

#### 24 Y. Yang et al. 

**Table 6:** An example of reasoning performance evaluation with multiple reasoning choices. In this example, Choices A, B, C and D correspond to RW, WW, RR and RW of the Doubly-Right metric, respectively. The RR choice is highlighted in <mark>yellow.</mark> 

**Frame Description Multiple Choices for Reasoning Evaluation** There are four people in the imA. Anomaly, since “climbing on a tree” matches anomaly human age. One person is walking with activities “Climbing on trees or into manhole/utility access points”. a backpack, another person is B. Normal, since no rules for anomaly human activities or nonriding a bicycle, a third person human objects match. is standing and looking at the <mark>C. Anomaly, since “riding a bicycle” matches</mark> anomaly human acbicyclist, and the fourth person tivities “Using any non-walking movement such as riding a bicycle is sitting on a bench. Other than or scooter, skateboarding”. people, there are trees, a trash D. Anomaly, since “a vehicle parked blocking a pedestrian crossing” bin, and two manhole covers vismatches anomaly non-human objects “Obstructions on pedestrian ible in the image. crossing”. 

**Table 7:** Detection performance with accuracy, precision, and recall (%) using different VLMs/LLMs as backbones in the deduction stage on 100 (limited by GPT-4’s query capacity) randomly selected frames from the ShT test set. 

|**Visual Perception**|**Robust Reasoning**|**Accuracy**|**Precision**|**Recall**|**Open Source**|
|---|---|---|---|---|---|
|CogVLM [51] (default)|Mistral [19] (default)|82.0|88.1|74.0|✓|
|GPT-4V [1]|GPT-4V|83.0|88.4|76.0|✗|
|LLaVA [23]|LLaVA|40.0|40.4|42.0|✓|
|PandaGPT [44]|PandaGPT|37.0|31.4|22.0|✓|
|OWLViT [33]||71.0|82.0|54.0|✓|
|LLaVA|Mitl|76.0|79.5|70.0|✓|
|BLIP-2 [21]|sra|50.0|50.0|94.0|✓|
|RAM [58]||45.0|47.2|84.0|✓|
|CoVLM|GPT-3.5 [7]|81.0|86.0|74.0|✗|
|g|LLaMA-2 [47]|60.0|70.8|34.0|✓|



perception. (3) CogVLM [51] + LLMs: We use CogVLM as the fixed VLM for visual perception and test different LLMs for robust reasoning (e.g., GPT-3.5 [7], LLaMA-2 [47]). We have the following observations. 

For the VLMs-only category, GPT-4V performs well, but it has limitations on the number of queries and a high cost per query, making it expensive for largescale testing. LLaVA and PandaGPT, on the other hand, show poor reasoning ability. They cannot follow the provided robust rules, and generate irrelevant content or hallucinations. An example frame with their outputs is shown below: 

_LLaVA:_ “The scene is set in a city park with a brick walkway, benches, and trees. However, the person on the skateboard is not wearing any pants, which is an unusual and unexpected sight.” 



_PandaGPT:_ “The couple should be the main focus of the image, and they should be actively engaged in skateboarding. Additionally, the surrounding environment should be typical of a public space, with no unusual or out-of-place objects or activities. If any of these conditions are not met, the image would be considered an anomaly.” 

For the VLMs + Mistrial category, using OWLViT and LLaVA as visual perception modules yields usable results, though they are still 6 to 10% lower than using CogVLM. However, the results with BLIP-2 and RAM are not usable due to serious hallucinations. For example, in a normal frame featuring only 

Follow the Rules: Reasoning for VAD with LLMs 25 

people walking, BLIP-2 outputs “A sidewalk with trees, two people are walking down a sidewalk, a man is riding a skateboard on a sidewalk, a woman walking down a sidewalk in a park.”, while RAM (recognize anything) outputs “Image Tags: path | person | skate | park | pavement | plaza | skateboarder | walk”. 

For the CogVLM + LLMs category, GPT-3.5 performs well but is expensive for large-scale testing. LLaMA-2, on the other hand, struggles with reasoning and fails to follow the given rules as context effectively. 

In summary, the propose AnomalyRuler is a generic plug-and-play framework that can improve VAD performance upon both the closed-source GPTs and the open-source VLMs/LLMs such as CogVLM and Mistral. AnomalyRuler applies to various VLMs/LLMs backbones as long as they have decent visual perception and rule-following capabilities. 

### **A.5 Further Discussions on Perception Smoothing and Robust Reasoning** 

Sections 5.3 and 5.4 demonstrate the effectiveness of the proposed perception smoothing and robust reasoning strategies. In this section, we provide a deeper investigation into them. Specifically, we aim to examine the extent to which the smoothing step may incorrectly smooth out anomalies from a sequence of video frames, and the extent to which the robust reasoning step can rectify these errors. 

Table 8 shows that less than 0.7% of anomalies are incorrectly smoothed out by the perception smoothing step (before the robust reasoning step), indicating very low false negative rates. The subsequent robust reasoning step successfully rechecks and corrects inaccuracies in the smoothed results, further reducing the false negative rates to below 0.15%. 

**Table 8:** The percentage (%) of incorrectly smoothed-out anomalies by the perception smoothing strategy on each dataset. 

|Dataset|ShT|Ave|Ped2|UB|
|---|---|---|---|---|
|Before Robust Reasoning|0.7%|0.4%|0.6%|0.3%|
|After Robust Reasoning|0.08%|0.15%|0.08%|0.01%|



The low false negative rates are due to that the smoothing step only smooths out the brief, isolated frames within a sequence of continuous frames. Table 9 shows that brief anomalies are rare in VAD datasets, as they typically persist for 97.9 to 441.3 continuous frames due to the time required for an anomaly to enter and exit the camera’s view. We also calculated the percentage of brief frames, i.e., _≤_ 10 frames, among all continuous anomaly frames. The ShT dataset has the highest percentage at 17.5% and an average length of 5.5 frames. In Section 5.4, we find that a padding size _p_ = 5 in our majority vote step is the optimal window size for ShT for capturing the predominant motion continuity in a video. This aligns with the average length of brief continuous anomalies (5.5 frames) and may explain the reason behind this optimal value. 

26 Y. Yang et al. 

**Table 9:** Statistics for the number of continuous anomaly frames per video clip of each dataset. 

|Dataset|ShT|Ave|Ped2|UB|
|---|---|---|---|---|
|# Average continuous anomaly frames|111.3|97.9|137.3|441.3|
|% Brief continuous anomalies (_≤_10 frames)|17.5%|2.1%|0.0%|0.0%|
|# Average brief continuous anomaly frames|5.5|10.0|0.0|0.0|



### **A.6 Normal Reference Frame Sampling** 

The proposed few-normal-shot prompting method is particularly beneficial when only a few normal data points are available in real-world scenarios. In our experiments, we simulate this scenario by randomly sampling normal reference frames from a training set, assuming only the randomly sampled frames are available. 

However, even when a set of normal data (e.g., a training set) has already been collected, our few-normal-shot prompting method is still useful for fast adaptation. In this scenario, different normal reference frame sampling strategies beyond random sampling can be considered, such as sampling by GPT-4V [1]. Table 10 compares the random sampling and GPT-4V sampling (sampling ten frames) on the ShT dataset. The results of five trials show similar performance. The reason is that normal patterns in existing VAD datasets are not very diverse. Hence, randomly sampled normal frames are efficient as references for rule induction. Requiring only a few randomly sampled reference frames is one of our contributions, but GPT-4V sampling could be a promising extension for more complicated VAD scenarios. 

**Table 10:** Random sampling vs. GPT-4V sampling on the ShT dataset. Results of five trials are reported. 

|Method|# Rules|AUC (%)|
|---|---|---|
|Random sampling (ten frames)|42.2 _±_ 4.2|84.5 _±_ 1.1|
|GPT-4V sampling (ten frames)|39.9 _±_ 6.9|84.8 _±_ 1.6|



Follow the Rules: Reasoning for VAD with LLMs 27 

### **A.7 Unified Anomaly Detection** 

Unified anomaly detection [55] considers image anomaly detection that trains a single model across different object classes. We extend this setting to VAD by considering a single model across different datasets. Specifically, the proposed AnomalyRuler can perform as a unified anomaly detection approach by using normal reference frames randomly sampled from various datasets and deriving a set of unified rules for all datasets. Table 11 shows the results, which are on par with the main evaluation in Table 3. This demonstrates that AnomalyRuler performs well under the unified anomaly detection setting by inducing effective unified rules across datasets with similar anomaly scenarios but distinct visual appearances. 

**Table 11:** AUC (%) of AnomalyRuler under the unified anomaly detection setting. AnomalyRuler induces unified rules from a few normal reference frames across all four datasets and is evaluated on these datasets. 

|**Ped2**|**Ave**|**ShT**|**UB**|
|---|---|---|---|
|97.6|85.6|84.7|68.8|



