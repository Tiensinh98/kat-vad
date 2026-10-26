This ICCV Workshop paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **VRU-Accident: A Vision-Language Benchmark for Video Question Answering and Dense Captioning for Accident Scene Understanding** 

Younggun Kim Ahmed S. Abdelrahman<sup>*</sup> Mohamed Abdel-Aty University of Central Florida, United States 

_{_ younggun.kim, ahmed.abdelrahman, m.aty _}_ @ucf.edu 

**https://vru-accident.github.io/** 

## **Abstract** 

_Ensuring the safety of vulnerable road users (VRUs), such as pedestrians and cyclists, is a critical challenge for autonomous driving systems, as crashes involving VRUs often result in severe or fatal consequences. While multimodal large language models (MLLMs) have shown promise in enhancing scene understanding and decisionmaking in autonomous vehicles, there is currently no standardized benchmark to quantitatively evaluate their reasoning abilities in complex, safety-critical scenarios involving VRUs. To address this gap, we present_ **_VRU-Accident_** _, a large-scale vision-language benchmark designed to evaluate MLLMs in high-risk traffic scenarios involving VRUs. VRU-Accident comprises_ **_1K_** _real-world dashcam accident videos, annotated with_ **_6K_** _multiple-choice question-answer pairs across six safety-critical categories (with_ **_24K_** _candidate options and_ **_3.4K_** _unique answer choices), as well as_ **_1K_** _dense scene descriptions. Unlike prior works, our benchmark focuses explicitly on VRU-vehicle accidents, providing rich, fine-grained annotations that capture both spatial-temporal dynamics and causal semantics of accidents. To assess the current landscape of MLLMs, we conduct a comprehensive evaluation of_ **_17_** _state-of-the-art models on the multiple-choice VQA task and on the dense captioning task. Our findings reveal that while MLLMs perform reasonably well on visually grounded attributes, they face significant challenges in reasoning and describing accident causes, types, and preventability._ 

## **1. Introduction** 

Multimodal Large Language Models (MLLMs) have become an important research area, showing strong performance in video understanding tasks such as video captioning, visual question answering, and visual grounding [2, 27, 51]. MLLMs have exhibited robust generalization across diverse downstream tasks, attributed to pretraining on large- 

> *Corresponding Author. 



Figure 1. VQA distribution in the VRU-Accident benchmark. 

scale, heterogeneous multimodal corpora [14, 23, 24]. This generalizability has sparked increasing interest in applying MLLMs to many scene understanding tasks which is critical to deepen our understanding of machine understanding and reasoning. To facilitate such applications, researchers have developed benchmarks and datasets focused on traffic event-related question-answering and general scene understanding [3, 29, 33]. However, these datasets overlook accident-centric reasoning, which is critical for safe driving perception [1, 15]. 

Recently, several accident datasets [10–12, 52] provide short casual captions per video that describe accident types, causes, and possible prevention strategies, providing context for accident interpretation. Despite their utility, these video-based vision-language accident datasets rely on human annotators to classify accidents into predefined categories, which restricts their ability to evaluate the gen- 

772 

eralization capabilities of MLLMs on accident scenarios, unlike video question-answer (VQA) settings that require models to reason over dynamically constructed and semantically diverse answer spaces. Moreover, these datasets lack detailed accident scene descriptions, making it difficult to assess an MLLM’s ability to generate context-rich and semantically accurate narratives about accident events. Finally, these datasets are heavily biased toward vehicle-tovehicle collisions, resulting in under-representation of incidents involving VRUs, such as pedestrians and cyclists, where they VRUs accidents are more dangerous and lead to fatalities [4, 5, 18]. 

VRUs are among the most at-risk road users in traffic environments, where in 2020 alone, 6,516 pedestrians and 938 bicyclists lost their lives in traffic accidentes with 3.9% and 9% increase from 2019, respectively, and over 54,000 were injured [4]. Alarmingly, pedestrian deaths accounted for 17% of all traffic fatalities, reflecting a 53% increase since 2009 [13]. Despite the severity and frequency of such incidents, we lack extensive vision-language benchmark specifically designed to evaluate the capabilities of MLLMs in answering questions or generating descriptions related to VRU-involved accidents with vehicles. This absence hinders the development and assessment of MLLMs in safety-critical applications such as autonomous vehicles, where evaluating a model’s ability to understand and reason about VRU-related accident scenarios is essential and ensure their driving perception for safe navigation even under adverse weather conditions. 

To address this gap, we propose **VRU-Accident** , a largescale benchmark designed to evaluate the reasoning and description capabilities of MLLMs in VRU traffic accidents. The benchmark includes 1,000 real-world accident and near-accident videos annotated with diverse VQA pairs and dense scene descriptions, covering various contextual and causal aspects of each incident. It serves as the first benchmark specifically curated to assess both VQA and captioning performance for safety-critical scenarios involving VRUs, where Figure 1 shows the distribution of the VQA task of each category in VRU-Accident benchmark. Our key contributions are summarized as follows: 

- We introduce VRU-Accident, a large-scale benchmark comprising **1K** VRU-related accident videos, **6K** VQA questions with **24K** candidate options, and **1K** dense scene-level captions. 

- We propose a semi-automatic benchmark curation pipeline that generates diverse and semantically rich VQA candidate sets. Each question includes one correct answer and three contextually plausible counterfactual, resulting in over **3.4K** unique answer options and making the task challenging for MLLMs. 

- We conduct extensive evaluations of **17** MLLMs, with 15 open-source and 2 closed-source, on the VQA task and 

Table 1. Comparison of accident video datasets. Surv.: Surveillance View, VQA: Video Question Answering, CC: Casual Captioning, DC: Dense Captioning, and R/S: Real/Synthetic videos. 

|**Datasets**|**View**|**#Clips**|**#VRU Accident**|**VQA**|**CC**|**DC**|**R/S**|
|---|---|---|---|---|---|---|---|
|ROL [16]|Dashcam|1,000|-||||R|
|DeepAccident [46]|Dashcam|-|-||||S|
|CTA [52]|Dashcam|1,935|-||✓||R|
|CTAD [26]|Surv.|1,100|-||||S|
|SUTD-TrafficQA [47]|Surv.|10,080|-|✓|||R|
|TUMTraffic-VideoQA [54]|Surv.|1,000|_<_100|✓|✓|✓|R|
|TUMTraf-A [56]|Surv.|48|-||||R|
|A3D [49]|Dashcam|3,757|_<_100||||R|
|DoTA [50]|Dashcam|5,586|100||||R|
|DADA-2000 [10]|Dashcam|2,000|223||✓||R|
|MM-AU [11,12]|Dashcam|**11,727**|510||✓||R|
|**VRU-Accident (Ours)**|Dashcam|1,000|**1,000**|✓|✓|✓|R|



the dense captioning task, providing a unified benchmark for assessing their performance in safety-critical scene understanding task while offering key insights into their limitations. 

## **2. Related Work** 

**Video QA Datasets and Benchmarks in Transportation Domain** Recent advances in vision-language models have spurred the development of various VQA datasets and benchmarks within the traffic domain. Several recent works [29, 32, 33, 38] have proposed multi-modal VQA benchmarks in the context of autonomous driving, aiming to evaluate high-level reasoning about road scenes. Another research [17, 31, 48] investigates language grounding and explanation in complex traffic scenes. These datasets focus on providing textual rationales or identifying key objects and behaviors, thereby enhancing interpretability and supporting downstream decision making. The other works [28, 35] introduce task-specific benchmarks that target action recognition, risk localization, and importance-based reasoning. These efforts promote temporally grounded event understanding and knowledge extraction via joint captioning and QA. Despite their success in advancing scene understanding and reasoning in traffic environments, these benchmarks do not specifically address accident scenarios, particularly those involving vulnerable road users (VRUs) such as pedestrians and cyclists. As a result, they offer limited utility for evaluating a model’s capacity to reason about accident dynamics, causality, and preventability in safety-critical contexts where accurate accident comprehension is essential. 

**Accident Understanding Datasets and Benchmarks** To support research on accident understanding in driving scenarios, several video datasets have been proposed in recent years. A summary of representative accident-centric datasets is presented in Table 1, while Table 2 provides detailed distributions for the subset of ego-view datasets that contain VRU involved accident videos. These datasets vary 

773 

Table 2. Comparison of VRU-related accident distributions across accident datasets, DoTA, DADA-2000, MM-AU, and our proposed VRU-Accident benchmark. VRU-Accident provides broader coverage across all attributes. 

|**Datasets**|**VRU Ty**<br>pedestrian|**pes**<br>cyclist|**Ligh**<br>day|**tning**<br>night|<br>sunny|**Weather**<br>rainy|**Condition**<br>snowy<br>cloudy|arterials|**Road Coi**<br>intersection|**nfiguratio**<br>t-junction|**n**<br>curve|others|highway|**Trai**<br>urban<br>su|**ffic**<br>burb|**i Env**<br>an|**i ironm**<br>rural|**i ent**<br><br>mountain|tunnel|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|DoTA [50]|89|11|79|21|86|2|12<br>0|75|14|10|1|0|2|89|0||9|0|0|
|DADA-2000 [10]|114|109|197|26|213|10|0<br>0|84|87|43|7|2|1|186|0||33|2|**1**|
|MM-AU [11,12]|320|190|452|58|418|29|46<br>17|248|163|71|16|12|3|430|**6**||68|2|**1**|
|**VRU-Accident (Ours)**|**730**|**270**|**873**|**127**|**808**|**61**|**108**<br>**23**|**584**|**241**|**133**|**24**|**18**|**6**|**873**|**6**||**111**|**3**|**1**|





<!-- Start of picture text -->
VRU-Accident Benchmark<br>General and Adversarial<br>VQA Generation Rules<br>𝑮𝑽𝑸𝑨<br>Searched Videos Traffic related- (𝒒𝒋 ,  𝒚𝒋) 𝑼𝑽𝑸𝑨𝒊<br>Questions<br>𝑽𝒊 VQA  Human Expert Q1. What are weather and lighting conditions?VQA Task<br>Generator A.         B.          C.           D<br>𝑽𝒊 𝑼𝒊 Q6. How could this accident be prevented?<br>𝑮𝑫𝑪<br>Manual Filter Accident Videos 𝑼𝑫𝑪𝒊 A.         B.          C.           D<br>Web Video Search<br>Dense Captioning Task<br>Dense Caption<br>Generation Rules Dense Caption A male cyclist, wearing a white jacket with…<br>Revised Query Generator<br>A male cyclist, wearing a white jacket with a blue hood, dark pants, and a backpack, is riding a bicycle in a wide, dry, urban asphalt intersection<br>under clear skies. There are no visible crosswalk markings. Initially, he begins crossing from the right side of the road, traversing multiple lanes. The<br>dashcam vehicle is making a left turn at approximately 30 km/h when the cyclist becomes momentarily obscured from view by a black Volkswagen<br>located to the vehicle's right. As the turn progresses, the cyclist's path intersects with the right-front quadrant of the vehicle, resulting in a collision<br>between the vehicle's front-right corner and the bicycle. The impact violently propels the cyclist onto the road surface, where he lands prone just<br>ahead and slightly to the right of the vehicle’s front bumper. The bicycle falls sideways and comes to rest partially underneath the front end of the<br>vehicle, clearly indicating the point of collision. No evasive braking or steering is observed from the vehicle prior to impact, and the cyclist also does<br>not appear to alter speed or direction.<br><!-- End of picture text -->

Figure 2. Overview of the VRU-Accident benchmark curation pipeline. Accident videos are retrieved from web sources based on trafficrelated queries and manually filtered by human experts. For each video _Vi_ , a set of ground truth answers _yj_ is first annotated by human experts for each question _qj_ . Then, the VQA generator _G_ VQA generates three counterfactual answers _yj_<sup>_∗_based on (</sup><sup>_qj, yj_).Separately,a</sup> dense accident description _Ci_ is generated for _Vi_ using the dense captioning generator _G_ DC. All annotations are verified by human experts before being finalized. Colors: vehicle or pedestrian appearances, environmental factors, weather conditions, pedestrian’s kinematic features, vehicle’s kinematic features, spatial relationship between road users, and detailed collision descriptions. 

in terms of view type, volume, annotation modalities, and coverage of accident content. ROL [16], A3D [49], and DoTA [50] focus on anomaly detection in dashcam videos, while CTAD [26] and DeepAccident [46] leverage synthetic or simulation-based video generation to model urban traffic accidents. SUTD-TrafficQA [47] is a benchmark designed for VQA tasks in traffic events but relies on surveillance footage. In contrast, DADA-2000 [10] includes driver attention data to study accident prediction, and MM-AU [12], which among all their videos only 510 are accidents involving VRUs, with 223 videos already imported from the DADA-2000 dataset, present large-scale collections of realworld dashcam incidents for visual reasoning and forecasting. Among them, a few works [10–12, 52] include short captions describing accident types, causes, or prevention strategies. However, these captions are entirely dependent on human annotators and constrained to predefined categories, which not only limits their expressiveness in capturing the diverse semantics of real-world accident scenar- 

ios but also restricts the ability to evaluate the generalization capabilities of modern MLLMs such as LLaVA family [20, 21, 53], InternVL family [7, 8, 55], Qwen family [40, 45], and the others [25, 30, 36, 37, 39]. One the other hand, SUTD-TrafficQA [47] addresses this gap by introducing a structured VQA benchmark for traffic scenarios, enabling more principled evaluation of VQA performance in MLLMs. However, its reliance on surveillance camera footage limits its ability to capture direct interactions between vehicles and VRUs, such as pedestrians and cyclists, especially during moments of impact or collision. Our VRU-Accident benchmark is set to target VRU-related accidents in a vision-language framework. It combines the MM-AU and DoTA samples while adding more samples reaching 1,000 videos. It includes VQA annotations across six accident-related categories and detailed dense captions for 1,000 real-world ego-view accident videos. This enables a comprehensive and fine-grained evaluation of MLLMs’ reasoning and description capabilities in realistic, high-risk 

774 

interaction scenarios between vehicles and VRUs. 

Table 3. Summary of annotation statistics in the VRU-Accident benchmark. 

## **3. VRU-Accident Benchmark** 

In this section, we describe the overall design of the VRUAccident benchmark, focusing on two key components: the benchmark curation process and statistics. Our goal is to establish a standardized and comprehensive framework for analyzing accident scenes involving VRUs through both VQA and Dense Captioning tasks. We aim to capture a wide range of contextual, environmental, and causal elements from real-world traffic accident videos to facilitate multimodal reasoning in safety-critical scenarios. We focus on collecting real-world traffic accident videos involving VRUs and curating a benchmark tailored for two visionlanguage tasks: VQA and Dense Captioning. The overall curation pipeline is illustrated in Figure 2. We define each accident video as _Vi_ , where _i_ = 1 _,_ 2 _, . . . , N_ . 

### **3.1. VQA Category Selection** 

We define six core VQA categories as _j ∈{_ 1. _Weather & Light_ , 2. _Traffic Environment_ , 3. _Road Configuration_ , 4. _Accident Type_ , 5. _Accident Cause_ , 6. _Accident Prevention Measure}_ . Among these, _Accident Type_ , _Accident Cause_ , and _Accident Prevention Measure_ are essential for understanding the core dynamics and outcomes of VRU-involved accidents, and have been widely used in prior research on traffic incident analysis [10, 12, 47, 52]. In addition, we include contextual attributes such as _Weather & Light_ , _Traffic Environment_ , and _Road Configuration_ , which are known to significantly influence accident circumstances and situational awareness [41–44]. 

### **3.2. VQA Curation Pipeline** 

Our task is framed as a multi-category VQA problem. For each video _Vi_ , we define a set of questions and corresponding answers as _Qi_ and _Yi_ , respectively. Each _Qi_ and _Yi_ contain a question-answer pair ( _qj, yj_ ) for each of the six VQA categories, where _j_ = 1 _, . . . ,_ 6. To enable discriminative evaluation, we additionally define a set of three counterfactual answers _yj_<sup>_∗_per category, which belong to the same do-</sup> main as _yj_ but are incorrect. These are generated using a VQA counterfactual generation function _G_ VQA, such that _yj_<sup>_∗_=</sup><sup>_G_VQA(</sup><sup>_qj, yj_).We then define the final candidate an-</sup> swer set for each question as _Y_<sup>ˆ</sup> _j_ = _{yj, yj_<sup>_∗_(1)</sup> _, yj_<sup>_∗_(2)</sup> _, yj_<sup>_∗_(3)</sup> _}_ . = Each video’s VQA annotation is summarized as _Ui_<sup>_V QA_</sup> _{Vi, Qi, Y_<sup>ˆ</sup> _i}_ , where _Y_<sup>ˆ</sup> _i_ = _{Y_<sup>ˆ</sup> _j}_<sup>6</sup> _j_ =1<sup>denotesthecandidate</sup> answer sets for all six categories. 

### **3.3. Dense Caption Curation Pipeline** 

The dense captioning task aims to generate a comprehensive and temporally grounded textual description of the traffic accident scenario depicted in each video _Vi_ . Rather than 

|Task|Categories|Options/GT|# Samples|# Unique Samples|
|---|---|---|---|---|
||Weather & Liht|Options|4K|38|
||g|Ground Truth|1K|13|
||Trffi Enirnmnt|Options|4K|27|
||aic voe|Ground Truth|1K|6|
||Rd Cfiiti|Options|4K|42|
|VQA|oa onigguraon|Ground Truth|1K|10|
||Aidt T|Options|4K|405|
||ccen ype|Ground Truth|1K|33|
||Aidnt C|Options|4K|782|
||cce ause|Ground Truth|1K|135|
||Pti M|Options|4K|2143|
||revenon easure|Ground Truth|1K|211|
|Dense<br>Captioning|Detailed Description of|Accident Video|1K|1K|



enumerating every visual element, we focus on capturing the essential dynamics of the incident, such as weather and environmental factors, road users’ appearance and posture, road users’ kinematic features, and the sequential interactions between the vehicle and the VRU leading up to the accident. The descriptions are required to be concise yet information-dense, suitable for use as high-quality ground truth data in accident analysis. To achieve this, we employ a Dense Captioning generator _G_ DC that produces a caption _Ci_ for each video _Vi_ , such that _Ci_ = _G_ DC( _Vi_ ). The final annotation for the dense captioning task is represented as _Ui_<sup>DC</sup> = _{Vi, Ci}_ . Finally, the complete annotation for each video of VRU-Accident is defined as _Ui_ = _Ui_<sup>VQA</sup> _∪Ui_<sup>DC</sup> = _{Vi, Qi, Y_<sup>ˆ</sup> _i, Ci}_ , and Figure 3 illustrates an example of both VQA and Dense Captioning annotations in our VRU-Accident benchmark. 

### **3.4. Benchmark Statistics** 

The VRU-Accident benchmark contains a total of 1,000 real-world traffic accident videos of VRUs. Each video is annotated with both VQA and dense captioning labels. As shown in Table 3, the VQA task comprises 6,000 questionanswer (QA) pairs and 1,000 dense captioning. For each QA pair, we construct a set of four candidate answers (one ground truth and three counterfactual options), resulting in a total of 24,000 multiple-choice options across the dataset. In addition to the volume of annotations, a key characteristic of VRU-Accident is the large number of unique answer samples per category. For example, the _Prevention Measure_ and _Accident Cause_ categories contain 2,143 and 782 unique textual answers, respectively. Unlikely accident classification tasks [10, 12, 50] which require predefined number of classes in each category, high variability of our benchmark requires models not only to classify from a fixed label space but to reason about fine-grained and context-dependent semantics, underscoring the importance of VLMs for this benchmark. The dense captioning task also contributes to the benchmark’s complexity and richness, providing 1,000 detailed accident descriptions, resulting in 1,000 unique captions aligned with real-world acci- 

775 



Figure 3. Visualization of annotations in VRU-Accident benchmark illustrating both the VQA and dense captioning tasks. 

dent scenarios. For more details about benchmark statistics, please refer to Supplementary Section B. 

## **4. Experiments** 

### **4.1. Implementation Details** 

**Baseline MLLMs.** We evaluate a total of 17 SOTA MLLMs on the VRU-Accident benchmark. This includes 15 open-source models that are assessed on both the VQA and dense captioning tasks, and 2 closed-source models. All models are tested in a _zero-shot_ setting without any fine-tuning, ensuring a fair comparison of their generalization abilities in unseen VRU-related accident scenarios. All evaluations were conducted on a workstation equipped with 8 NVIDIA TITAN RTX GPUs, each with 24 GB memory. 

**VQA Task.** We construct a total of **6K** VQA pairs based on **1K** VRU-related accident videos, with each video annotated with six question-answer pairs corresponding to distinct reasoning categories. Each question is accompanied by four candidate options (A, B, C, D), and models are prompted to select the correct choice. The generated responses are processed with model-specific postprocessing strategies to extract the predicted option, and _category-wise accuracy_ is computed to quantify performance. 

**Dense Captioning Task.** We evaluate **1K** video-prompt pairs, where models are asked to generate a comprehensive description of each VRU-related accident video, including visual scene elements, road user behavior, vehicle dynamics, and accident events. Generated captions are eval- 

776 

Table 4. Comparison of VQA performance of state-of-the-art models on VRU-Accident Benchmark. WL: Weather&Light, TE: Traffic Environment, RC: Road Configuration, AT: Accident Type, AC: Accident Cause, and PM: Prevention Measure. Rows with gray color background represent closed-source models, while rows with white background denote open-source models. **Black** , **blue** , and **red** colors indicate the best, second-best, and worst performance, respectively. **<u>Underline</u>** denotes the best performance among open-source models. 

|Model(Param.)|Year|_AccW L_|_AccT E_|_AccRC_|_AccAT_|_AccAC_|_AccP M_|_AccAV G._|
|---|---|---|---|---|---|---|---|---|
|LLaVA-OneVision(0.5B) [20]|2024|**72.1**|79.9|40.1|**35.9**|34.8|40.8|50.6|
|InternVL2.5(1B) [8]|2025|69.2|67.7|39.0|45.8|**21.3**|51.0|49.0|
|Mobile-VideoGPT(1.5B) [36]|2025|**78.4**|64.5|44.4|49.3|27.2|47.4|51.9|
|InternVL2.5(2B) [8]|2025|68.3|79.6|46.8|**70.7**|38.0|50.3|59.0|
|InternVL3(2B) [55]|2025|70.7|78.3|53.5|41.2|**50.4**|53.2|57.9|
|Qwen2.5-VL(3B) [40]|2025|63.8|72.1|39.4|47.3|25.6|44.7|**48.8**|
|Video-XL-Pro(3B) [25]|2025|69.0|**87.5**|35.4|43.9|47.5|48.9|55.4|
|InternVL2.5(4B) [8]|2025|69.9|82.7|42.9|56.7|50.1|51.0|58.9|
|Video-XL2(7B) [37]|2024|71.0|82.7|59.3|58.1|45.4|53.5|61.7|
|LLaVA-NeXT-Video(7B) [21]|2024|71.6|**84.1**|54.0|38.8|27.7|**33.1**|51.6|
|LLaVA-Video (7B) [53]|2024|68.2|80.7|64.7|63.7|**50.4**|58.6|**64.4**|
|Qwen2-VL(7B) [45]|2024|70.3|**59.6**|**34.1**|60.1|41.4|46.8|52.1|
|InternVL2(8B) [7]|2024|61.9|78.1|42.8|58.3|43.6|**62.0**|57.8|
|InternVL2.5(8B) [8]|2025|67.6|81.9|51.4|48.0|48.1|**61.9**|59.8|
|InternVL3(8B) [55]|2025|70.0|81.8|**67.8**|64.8|43.7|58.4|**64.4**|
|GPT-4o-mini [30]|2024|**60.2**|78.6|42.1|46.3|34.8|50.3|52.1|
|Gemini 1.5-flash[39]|2024|65.7|78.5|**71.4**|**77.9**|**54.6**|53.0|**66.9**|
|**Human Expert**|2025|**95.1**|**94.7**|**93.8**|**94.5**|**95.1**|**94.8**|**94.7**|



uated using SPICE [6], METEOR [19], COMET [34], and ROUGE [22] to assess linguistic quality and event coverage. BLEU score is not included, as it yields values below 0.1, due to its emphasis on exact n-gram precision without incorporating recall or contextual semantics [9]. We also present detailed evaluation metrics, full prompt templates, and implementation details for both VQA and Dense Captioning tasks in Supplementary Section B. 

### **4.2. Result Analysis** 

**VQA Task.** Table 4 presents the performance comparison of 17 state-of-the-art MLLMs, including 15 open-source and 2 closed-source models, on the VQA task of the VRUAccident benchmark. Overall, Gemini [39] achieves the best average performance (66.9%), followed by LLaVAVideo [53] and InternVL3(8B) [55], which show strong performance across most categories. In contrast, smaller models such as Qwen2.5-VL(3B) [40] and InternVL2.5(1B) [8] struggle particularly on causal and preventive reasoning. We observe that MLLMs generally perform better on visually grounded attributes (e.g., Weather & Light or Traffic Environment), while achieving lower accuracy on highlevel reasoning tasks such as Accident Cause and Prevention Measure. Additionally, models often struggle with categories like ‘Road Configuration’ and ‘Accident Type’, but for distinct reasons. In the case of ‘Road Configuration’, poor performance is largely due to the models’ limited ability to understand complex and dynamic properties 

#### **Weather&Light** 



<!-- Start of picture text -->
100<br>90<br>80<br>70<br>60 Traffic<br>Prevention<br>Measure 50 Environment<br>40<br>30<br>20<br>10<br>0<br>Accident  Road<br>Cause Configuration<br>Accident Type<br>LLaVA-OneVision(0.5B) Mobile-VideoGPT(0.5B) InternVL2.5(1B)<br>Mobile-VideoGPT(1.5B) InternVL2.5(2B) InternVL3(2B)<br>Qwen2.5-VL (3B) Video-XL-Pro (3B) InternVL2.5(4B)<br>Video-XL2(7B) LLaVA-NeXT-Video (7B) LLaVA-Video (7B)<br>Qwen2-VL(7B) InternVL2(8B) InternVL2.5(8B)<br><!-- End of picture text -->

Figure 4. Summary of MLLMs performance on VRU-Accident for the VQA task along with the Human Expert scores. 

of the surrounding environment, such as curves, intersections, or multi-lane structures. For ‘Accident Type’, the 

777 

Table 5. Quantitative comparisons on the dense caption task. We report ROUGE precision (P), recall (R), and F-measure (F), where P and R indicate the overlap of 4-grams with candidate and reference summaries, respectively, and F is their harmonic mean. Higher values indicate better performance. **Black** , **blue** , and **red** colors indicate the best, second-best, and worst performance, respectively. **<u>Underline</u>** denotes the best performance among open-source models. 

|Model (Param.)|SPICE_↑_|METEOR_↑_|COMET_↑_|R<br>|OUGE-1<br>|_↑_<br>|R<br>|OUGE-2 <br>|_↑_<br>|R<br>|OUGE-L<br>|_↑_<br>|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||P|R|F|P|R|F|P|R|F|
|LLaVA-OneVision(0.5B) [20]|0.126|0.224|0.647|0.388|0.384|0.380|0.092|0.091|0.090|0.228|0.229|0.225|
|InternVL2.5(1B) [8]|0.132|0.236|0.676|0.409|0.403|0.400|0.089|0.088|0.087|0.212|0.209|0.207|
|Mobile-VideoGPT(1.5B) [36]|0.128|0.25|0.665|**0.363**|0.420|0.386|**0.069**|0.080|0.073|**0.180**|0.209|**0.192**|
|InternVL2.5(2B) [8]|0.139|0.249|0.682|0.404|0.428|0.411|0.094|0.099|0.095|0.207|0.220|0.211|
|InternVL3(2B) [55]|**0.100**|**0.188**|**0.584**|0.429|**0.309**|**0.345**|0.098|**0.070**|**0.078**|0.249|**0.177**|0.199|
|Qwen2.5-VL(3B) [40]|0.149|0.261|0.682|0.396|0.446|0.416|0.106|0.120|0.111|0.207|0.234|0.218|
|Video-XL-Pro(3B) [25]|0.141|0.265|0.695|0.396|0.450|0.418|0.098|0.111|0.103|0.206|0.236|0.218|
|InternVL2.5(4B) [8]|0.146|0.260|0.696|0.408|0.449|0.423|0.098|0.108|0.102|0.207|0.228|0.215|
|Video-XL2(7B) [37]|0.148|0.233|0.693|**0.491**|0.378|0.421|0.121|0.092|0.103|**0.266**|0.203|0.227|
|LLaVA-NeXT-Video(7B) [21]|0.155|0.259|0.708|0.443|0.438|0.437|0.124|0.123|0.123|0.248|0.246|0.245|
|LLaVA-Video(7B) [53]|0.157|**0.270**|0.686|0.422|0.460|0.436|0.113|0.124|0.117|0.219|0.240|0.227|
|Qwen2-VL(7B) [45]|**0.170**|**0.285**|**0.721**|0.430|**0.494**|**0.456**|**0.126**|**0.145**|**0.134**|0.233|**0.270**|**0.248**|
|InternVL2(8B) [7]|0.139|0.249|0.694|0.398|0.431|0.410|0.091|0.098|0.093|0.201|0.220|0.208|
|InternVL2.5(8B) [8]|0.152|0.259|0.698|0.420|0.442|0.426|0.102|0.108|0.104|0.216|0.228|0.219|
|InternVL3(8B) [55]|0.159|0.267|0.694|0.434|0.454|0.437|0.113|0.118|0.114|0.225|0.234|0.225|
|Gemini 1.5-flash[39]|**0.194**|**0.285**|**0.741**|**0.492**|**0.479**|**0.481**|**0.152**|**0.147**|**0.148**|**0.272**|**0.266**|**0.266**|













**Ground Truth The incident occurs on a wet, two-way urban road under overcast skies, with parked vehicles and patches of greenery lining both sides. The road surface is slick and reflective due to recent rainfall, and the dark-colored vehicle gradually accelerates from approximately 33 km/h to 42 km/h while following queued traffic. Just before the collision, a pedestrian appears partially in the left edge of the frame and begins to cross the road unexpectedly from the left curb to the right. The pedestrian does not appear to check for oncoming traffic, while the vehicle continues straight in the leftmost lane, closing the gap rapidly. Without significant deceleration, the vehicle strikes the pedestrian head-on at around 42 km/h. The impact results in severe damage to the windshield, including both radial and concentric cracks, indicating a high-energy collision. This sequence highlights how the pedestrian’s crossing attempt transitions into a collision within a very brief time window, with no evident evasive action by the driver. Qwen2-VL (7B) The video depicts a traffic crash on a wet road, likely during or after a rainstorm. The weather conditions appear to be rainy, as evidenced by the wet road surface and the presence of rain streaks on the windshield. The road is a multi-lane street with vehicles parked on both sides, indicating it might be a suburban or urban area. A pedestrian is seen crossing the street from the right side to the left side. The pedestrian is wearing casual clothing, including a light-colored shirt and dark pants. The pedestrian's posture suggests they are in motion, likely walking or running across the street. A vehicle, possibly a car or a bus, is approaching the pedestrian from the right side of the street. The vehicle's speed appears to be moderate, as it is not overtaking other vehicles. The vehicle's trajectory is straight, and it is moving towards the pedestrian. As the vehicle approaches the pedestrian, the pedestrian suddenly changes direction and runs towards the left side of the street. Vehicle or Pedestrian Appearances Vehicle’s Kinematic Features Environmental Factors Weather Conditions Spatial Relationship b/w Road Users Pedestrian’s Kinematic Features Detailed Collision Descriptions** 

Figure 5. Accident description from both Ground Truth in VRU-Accident and Qwen2-VL(7B) which achieved the best performance in the dense captioning task. Please refer to Supplementary Section C for descriptions generated from all MLLMs. 

778 

difficulty stems from the models’ inability to reliably distinguish between different types of accident agents. For instance, models frequently misclassify options when both a pedestrian and a cyclist are mentioned, or when they must differentiate between ego-vehicle and surrounding vehicles, indicating limitations in fine-grained object-level reasoning within accident contexts. Notably, performance in the ‘Accident Cause’ category is consistently poor across most models, with several falling below 30%, indicating that current MLLMs lack robust causal reasoning capabilities in safety-critical scenarios. Compared to human experts, who achieve over 94% accuracy in all categories, the gap highlights significant room for improvement in accident-centric understanding. These findings emphasize the importance of including causal and preventive questions in benchmarks for evaluating deeper vision-language reasoning, particularly in scenarios involving VRUs. The proposed VRUAccident benchmark thus provides a challenging and diagnostic testbed for probing the limitations and capabilities of modern MLLMs in accident-related understanding. 

To provide a holistic overview of model performance of VQA task, we visualize the results in Figure 4. This summary highlights the overall limitations of current MLLMs when applied to complex, safety-critical accident scenarios. Despite strong performance on certain visual attributes, most models struggle with causal reasoning and fine-grained collision understanding. These findings underscore the challenging nature of this benchmark and indicate substantial room for improvement in developing more robust and context-aware multimodal models. 

**Dense Captioning Task.** Table 5 benchmarks 16 SOTA VLMs on SPICE, METEOR, COMET, and ROUGE. GPT-4 is excluded from this comparison, as it was used to generate the initial reference captions for the videos in the VRUAccident benchmark. Gemini 1.5-flash leads the leaderboard, recording 0.194 SPICE, 0.285 METEOR, and 0.741 COMET, together with the highest ROUGE-F values of 0.481 (R-1), 0.148 (R-2) and 0.266 (R-L). Compared with the best open-source VLM, Qwen2-VL (7B), Gemini has absolute gains of +0.024 SPICE, +0.020 COMET, +0.025 ROUGE-1 F, +0.014 ROUGE-2 F and +0.018 ROUGE-L F, while matching Qwen2-VL on METEOR. These margins show that Gemini extracts finer scene details and maintains better temporal coverage in complex accident videos. Among open-source VLMs, Qwen2-VL (7B) is the clear front-runner. It couples the highest open-source scores in METEOR (0.285) and COMET (0.721) with strong balance between precision and recall, delivering 0.456/0.134/0.248 ROUGE-F. When precision alone is considered, Video-XL2 (7B) edges ahead with 0.491 (R-1 P) and 0.266 (R-L P), yet its low recall (0.378 R-1) indicates it produces short captions that omit salient context. In contrast, InternVL3 (2B) delivers the weakest semantic quality (0.100 SPICE, 0.584 

COMET) and the lowest ROUGE-F (0.345 R-1), suggesting fragmented, short descriptions. Mobile-VideoGPT (1.5B) also under-performs, posting the smallest ROUGE-1 precision (0.363) and ROUGE-L precision (0.180), consistent with captions that miss core entities and actions. Overall, Gemini 1.5-flash retains the lead across both the VQA and dense-caption tasks on VRU-Accident. For open-source workflows, Qwen2-VL (7B) is the best choice for producing rich, spatiotemporal dynamics aware accident descriptions. _Overall_ , although Gemini 1.5-flash and Qwen2-VL (7B) achieve high COMET score, their low SPICE, METEOR, and ROUGE-L scores reveal clear gaps in causal reasoning, spatial relation understanding and temporal coherence. 

We present a qualitative comparison between the ground truth caption in VRU-Accident and the caption generated by Qwen2-VL (7B), the best-performing open-source model in the dense captioning task, as shown in Figure 5. Qwen2-VL (7B) captures key contextual elements, such as weather conditions, environmental layout, pedestrian appearances, and vehicle trajectories, which contributes to its strong overall performance across the evaluation metrics. However, despite these strengths, the model fails to describe critical details about the collision itself, an essential component in safety-critical scenarios, and even hallucinates the pedestrian’s motion, misrepresenting their crossing direction. This highlights the inherent challenge of the VRUAccident benchmark, which requires not only general scene understanding but also fine-grained reasoning about spatiotemporal dynamics in accident scenarios. 

## **5. Conclusion** 

This work presents VRU-Accident, the first large-scale vision-language benchmark specifically designed to evaluate multimodal large language models (MLLMs) in safetycritical scenarios involving VRUs. Our benchmark comprises 1K real-world traffic accident videos annotated with over 6K question-answer pairs and 1K dense descriptions, enabling fine-grained assessments of both causal reasoning and narrative generation in accident contexts. Through extensive evaluations of 17 state-of-the-art MLLMs, we found that while models perform well on visually grounded attributes, they struggle with high-level reasoning tasks such as accident causes, prevention measures, road configurations, and accident type identification. These limitations highlight the current models’ difficulty in interpreting dynamic and complex environmental cues, as well as distinguishing between fine-grained object roles in collision events. By offering both diagnostic and generative evaluation settings, VRU-Accident provides a unified and challenging testbed to advance the capabilities of future multimodal models for the safe navigation of autonomous driving systems. 

779 

## **References** 

- [1] Ahmed S Abdelrahman, Mohamed Abdel-Aty, and Quoc Dai Tran. Vru-cipi: Crossing intention prediction at intersections for improving vulnerable road users safety. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 5623–5632, 2025. 1 

- [2] Ahmed S Abdelrahman, Mohamed Abdel-Aty, and Dongdong Wang. Video-to-text pedestrian monitoring (vtpm): Leveraging large language models for privacy-preserve pedestrian activity monitoring at intersections. In _Proceedings of the Winter Conference on Applications of Computer Vision_ , pages 366–375, 2025. 1 

- [3] Ahmed S Abdelrahman, Mohamed Abdel-Aty, Samgyu Yang, and Abdulrahman Faden. Advanced crash causation analysis for freeway safety: A large language model approach to identifying key contributing factors. _arXiv preprint arXiv:2505.09949_ , 2025. 1 

- [4] Ahmed S Abdelrahman, Zubayer Islam, and Mohamed Abdel-Aty. Vrucrosssafe for crossing intention prediction of vulnerable road users for improving safe crossing at intersections. _npj Sustainable Mobility and Transport_ , 2(1):20, 2025. 2 

- [5] Md Jamil Ahsan, Mohamed Abdel-Aty, and Ahmed S Abdelrahman. Evaluating the safety impact of mid-block pedestrian signals (mps). _Accident Analysis & Prevention_ , 210: 107847, 2025. 2 

- [6] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14_ , pages 382–398. Springer, 2016. 6 

- [7] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 3, 6, 7 

- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. 3, 6, 7 

- [9] Goutam Datta, Nisheeth Joshi, and Kusum Gupta. Analysis of automatic evaluation metric on low-resourced language: Bertscore vs bleu score. In _International Conference on Speech and Computer_ , pages 155–162. Springer, 2022. 6 

- [10] Jianwu Fang, Dingxin Yan, Jiahuan Qiao, Jianru Xue, He Wang, and Sen Li. Dada-2000: Can driving accident be predicted by driver attention? analyzed by a benchmark, 2019. 1, 2, 3, 4 

- [11] Jianwu Fang, Lei-Lei Li, Kuan Yang, Zhedong Zheng, Jianru Xue, and Tat-Seng Chua. Cognitive accident prediction in driving scenes: A multimodality benchmark, 2023. 2, 3 

- [12] Jianwu Fang, Lei lei Li, Junfei Zhou, Junbin Xiao, Hongkai Yu, Chen Lv, Jianru Xue, and Tat-Seng Chua. Abductive ego-view accident video understanding for safe driving perception, 2024. 1, 2, 3, 4 

- [13] Governors Highway Safety Association (GHSA). Pedestrian traffic fatalities by state: 2019 preliminary data, 2020. 2 

- [14] Akshay Gopalkrishnan, Ross Greer, and Mohan Trivedi. Multi-frame, lightweight & efficient vision-language models for question answering in autonomous driving, 2024. 1 

- [15] Je-Seok Ham, Dae Hoe Kim, NamKyo Jung, and Jinyoung Moon. Cipf: Crossing intention prediction network based on feature fusion modules for improving pedestrian safety. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3666–3675, 2023. 1 

- [16] Muhammad Monjurul Karim, Zhaozheng Yin, and Ruwen Qin. An attention-guided multistream feature fusion network for early localization of risky traffic agents in driving videos. _IEEE Transactions on Intelligent Vehicles_ , 9(1):1792–1803, 2024. 2, 3 

- [17] Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, and Zeynep Akata. Textual explanations for self-driving vehicles. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 563–578, 2018. 2 

- [18] Younggun Kim, Mohamed Abdel-Aty, Keechoo Choi, Zubayer Islam, Dongdong Wang, and Shaoyan Zhai. Pedestrian crossing direction prediction at intersections for pedestrian safety. _IEEE Open Journal of Intelligent Transportation Systems_ , 6:692–707, 2025. 2 

- [19] Alon Lavie and Abhaya Agarwal. Meteor: an automatic metric for mt evaluation with high levels of correlation with human judgments. In _Proceedings of the Second Workshop on Statistical Machine Translation_ , page 228–231, USA, 2007. Association for Computational Linguistics. 6 

- [20] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer, 2024. 3, 6, 7 

- [21] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models, 2024. 3, 6, 7 

- [22] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pages 74–81, Barcelona, Spain, 2004. Association for Computational Linguistics. 6 

- [23] Tianming Liu, Manzi Li, and Yafeng Yin. Aligning llm with human travel choices: a persona-based embedding learning approach, 2025. 1 

- [24] Tianming Liu, Jirong Yang, and Yafeng Yin. Toward llmagent-based modeling of transportation systems: A conceptual framework. _Artificial Intelligence for Transportation_ , 1: 100001, 2025. 1 

- [25] Xiangrui Liu, Yan Shu, Zheng Liu, Ao Li, Yang Tian, and Bo Zhao. Video-xl-pro: Reconstructive token compression for extremely long video understanding, 2025. 3, 6, 7 

780 

- [26] Haohan Luo and Feng Wang. A simulation-based framework for urban traffic accident detection. In _ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 1–5, 2023. 2, 3 

- [27] Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving. In _European Conference on Computer Vision_ , pages 403–420. Springer, 2024. 1 

- [28] Srikanth Malla, Chiho Choi, Isht Dwivedi, Joon Hee Choi, and Jiachen Li. Drama: Joint risk localization and captioning in driving. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 1043–1052, 2023. 2 

- [29] Ana-Maria Marcu, Long Chen, Jan H¨unermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, and Oleg Sinavski. Lingoqa: Visual question answering for autonomous driving. _arXiv preprint arXiv:2312.14115_ , 2023. 1, 2 

- [30] OpenAI. Gpt-4 technical report, 2024. 3, 6 

- [31] Chirag Parikh, Rohit Saluja, CV Jawahar, and Ravi Kiran Sarvadevabhatla. Idd-x: A multi-view dataset for ego-relative important object localization and explanation in dense and unstructured traffic. In _2024 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 14815–14821. IEEE, 2024. 2 

- [32] Ehsan Qasemi, Jonathan M Francis, and Alessandro Oltramari. Traffic-domain video question answering with automatic captioning. _arXiv preprint arXiv:2307.09636_ , 2023. 2 

- [33] Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. _arXiv preprint arXiv:2305.14836_ , 2023. 1, 2 

- [34] Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. Comet: A neural framework for mt evaluation, 2020. 6 

- [35] Enna Sachdeva, Nakul Agarwal, Suhas Chundi, Sean Roelofs, Jiachen Li, Mykel Kochenderfer, Chiho Choi, and Behzad Dariush. Rank2tell: A multimodal driving dataset for joint importance ranking and reasoning. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 7513–7522, 2024. 2 

- [36] Abdelrahman Shaker, Muhammad Maaz, Hamid Rezatofighi, Salman Khan, and Fahad Shahbaz Khan. Mobile-videogpt: Fast and accurate video understanding language model. _arxiv_ , 2025. 3, 6, 7 

- [37] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. _arXiv preprint arXiv:2409.14485_ , 2024. 3, 6, 7 

- [38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. _arXiv preprint arXiv:2312.14150_ , 2023. 2 

- [39] Gemini Team. Gemini: A family of highly capable multimodal models, 2025. 3, 6, 7 

- [40] Qwen Team. Qwen2.5-vl, 2025. 3, 6, 7 

- [41] Chenzhu Wang, Fei Chen, Yunlong Zhang, Shuyi Wang, Bin Yu, and Jianchuan Cheng. Temporal stability of factors affecting injury severity in rear-end and non-rear-end crashes: A random parameter approach with heterogeneity in means and variances. _Analytic Methods in Accident Research_ , 35: 100219, 2022. 4 

- [42] Chenzhu Wang, Mohamed Abdel-Aty, and Lei Han. Effects of speed difference on injury severity of freeway rear-end crashes: Insights from correlated joint random parameters bivariate probit models and temporal instability. _Analytic Methods in Accident Research_ , 42:100320, 2024. 

- [43] Chenzhu Wang, Mohamed Abdel-Aty, Lei Han, and Said M. Easa. Analyzing speed-difference impact on freeway joint injury severities of leading-following vehicles using statistical and data-driven models. _Accident Analysis & Prevention_ , 206:107695, 2024. 

- [44] Chenzhu Wang, Mohamed Abdel-Aty, and Lei Han. Tunnel crash severity and congestion duration joint evaluation based on cross-stitch networks. _Accident Analysis & Prevention_ , 213:107942, 2025. 4 

- [45] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 3, 6, 7 

- [46] Tianqi Wang, Sukmin Kim, Wenxuan Ji, Enze Xie, Chongjian Ge, Junsong Chen, Zhenguo Li, and Ping Luo. Deepaccident: A motion and accident prediction benchmark for v2x autonomous driving, 2023. 2, 3 

- [47] Li Xu, He Huang, and Jun Liu. Sutd-trafficqa: A question answering benchmark and an efficient network for video reasoning over traffic events, 2021. 2, 3, 4 

- [48] Yiran Xu, Xiaoyin Yang, Lihang Gong, Hsuan-Chu Lin, TzYing Wu, Yunsheng Li, and Nuno Vasconcelos. Explainable object-induced action decision for autonomous vehicles. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9523–9532, 2020. 2 

- [49] Yu Yao, Mingze Xu, Yuchen Wang, David J. Crandall, and Ella M. Atkins. Unsupervised traffic accident detection in first-person videos. In _2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pages 273– 280, 2019. 2, 3 

- [50] Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David J. Crandall. Dota: Unsupervised detection of traffic anomaly in driving videos. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 45(1): 444–459, 2023. 2, 3, 4 

- [51] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. _arXiv preprint arXiv:2310.07704_ , 2023. 1 

- [52] Tackgeun You and Bohyung Han. Traffic Accident Benchmark for Causality Recognition. In _ECCV_ , 2020. 1, 2, 3, 4 

781 

- [53] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. 3, 6, 7 

- [54] Xingcheng Zhou, Konstantinos Larintzakis, Hao Guo, Walter Zimmer, Mingyu Liu, Hu Cao, Jiajie Zhang, Venkatnarayanan Lakshminarasimhan, Leah Strand, and Alois C. Knoll. TUMTraffic-VideoQA: A benchmark for unified spatio-temporal video understanding in traffic scenes. _IEEE/CVF Int. Conf. on Machine Learning (ICML)_ , 2025. 

2 

- [55] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. 3, 6, 7 

- [56] Walter Zimmer, Ross Greer, Xingcheng Zhou, Rui Song, Hu Cao, Daniel Lehmberg, Marc Pavel, Ahmed Alaaeldin Ghita, Akshay Gopalkrishnan, Holger Caesar, Mohan M. Trivedi, and Alois C. Knoll. Towards vision zero: The accid3nd dataset. page 10, 2025. 2 

782 

