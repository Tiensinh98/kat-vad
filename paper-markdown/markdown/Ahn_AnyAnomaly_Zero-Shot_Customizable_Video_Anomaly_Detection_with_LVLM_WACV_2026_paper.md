This WACV paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM** 

Sunghyun Ahn<sup>*</sup> , Youngwan Jo<sup>*</sup> , Kijung Lee, Sein Kwon, Inpyo Hong, Sanghyun Park<sup>†</sup> Yonsei University, Seoul, Republic of Korea 

_{_ skd, jyy1551, rlwjd4177, seinkwon97, hip9863, sanghyun _}_ @yonsei.ac.kr 

## **Abstract** 

_Video anomaly detection (VAD) is crucial for video analysis and surveillance in computer vision. However, existing VAD models rely on learned normal patterns, which makes them difficult to apply to diverse environments. Consequently, users should retrain models or develop separate AI models for new environments, which requires expertise in machine learning, high-performance hardware, and extensive data collection, limiting the practical usability of VAD. To address these challenges, this study proposes customizable video anomaly detection (C-VAD) technique and the AnyAnomaly model. C-VAD considers userdefined text as an abnormal event and detects frames containing a specified event in a video. We effectively implemented AnyAnomaly using a context-aware visual question answering without fine-tuning the large vision language model. To validate the effectiveness of the proposed model, we constructed C-VAD datasets and demonstrated the superiority of AnyAnomaly. Furthermore, our approach showed competitive results on VAD benchmarks, achieving state-of-the-art performance on UBnormal and UCF-Crime and surpassing other methods in generalization across all datasets. Our code is available online at github.com/SkiddieAhn/Paper-AnyAnomaly._ 

## **1. Introduction** 

Video anomaly detection (VAD) aims to detect abnormal events in video streams. Abnormal events include the actions of objects that are inappropriate for the environment (e.g., climbing over a fence) or objects with unusual appearances (e.g., a bicycle on a walkway). However, abnormal events are rare and diverse, making it difficult to construct large-scale datasets for VAD. Therefore, VAD is recognized as a highly challenging problem. 

To overcome these limitations, previous studies primarily used one-class classification (OCC) methods that learn only from normal data. In the OCC approach, the model 

> *Equal contribution 

> †Corresponding author: Sanghyun Park (sanghyun@yonsei.ac.kr) 



Figure 1. Comparison of traditional video Anomaly Detection (VAD) and customizable video anomaly detection (C-VAD). Traditional VAD models struggle with generalization, making them hard to apply in diverse environments, while C-VAD can handle various video environments. 

learns normal patterns and classifies the cases that deviate from them as abnormal. Representative OCC methods include classification-[14, 29, 39], distance-[2, 26, 28], and prediction-based methods [10, 15, 17, 37], all of which have demonstrated excellent performance in VAD tasks. 

However, because normal and abnormal classes can be defined differently depending on the environment, OCC methods cannot always guarantee a generalized performance. For example, as shown on the left side of Figure 1, a model trained in a campus environment learns the characteristics of a ‘person’ as a normal pattern and classifies a ‘car’ as abnormal. However, when this model is applied to a road environment, a ‘car’ is still detected as abnormal, which can increase the number of false positives. Therefore, OCC methods require the retraining of normal patterns for each new environment, which entails additional costs, such as data collection, expert intervention, and high-performance equipment. Because of these limitations, the application of VAD models to real-world scenarios is challenging. 

To address this issue, we propose a novel technique called customizable video anomaly detection (C-VAD). C- VAD considers user-defined text as abnormal events and detects the frames containing these events in the video. For instance, in campus videos, ‘car’ can be set as an abnormal event, while in road videos, ‘person’ can be set as an abnor- 

3026 

mal event. In contrast to existing VAD models, which judge abnormalities based on learned normal patterns, C-VAD dynamically detects abnormal patterns based on the text provided. This implies that as the generalizability of visual text analysis improves, anomaly detection becomes more effective in various environments. Consequently, we introduce a zero-shot capable C-VAD approach, as shown on the right side of Figure 1, and propose the AnyAnomaly model, which allows for VAD in various environments without the need for additional training. 

An effective method to implement zero-shot C-VAD is to leverage large vision language models (LVLMs). Recently, LVLMs have demonstrated outstanding generalization performance in visual text analysis. By leveraging this capability, C-VAD can be performed effectively across various environments. The most intuitive method involves performing visual question answering (VQA) [4] on each frame to estimate the anomaly score. For instance, one could provide the model with the prompt: _“Return a value between 0 (no) and 1 (yes) indicating how well the input image represents the text provided by the user”_ . This was used as the baseline model. However, through experiments, we observed the following limitations of the baseline model: 1 Due to the large computational cost of LVLMs, the latency is high. 2 Difficulty in analyzing specific objects due to the characteristics of surveillance videos, such as foreground-background imbalance and object congestion. 3 Difficulty in detecting action-related anomalies because of the inability to utilize temporal information. 

To overcome these limitations, we designed an AnyAnomaly model with the structure shown in Figure 2. First, to reduce the latency, we adopted a segment-level approach that groups consecutive frames into a single segment for processing. For this purpose, we introduced a key frames selection module (KSM) that selects key frames representing the segment and performed VQA per segment. Second, instead of performing simple image-text matching, we introduced a context-aware VQA approach to enable a deeper understanding of the scene. To this end, we additionally utilized two types of information: position context, _PC_ and temporal context, _TC_ . _PC_ is a context that emphasizes important locations within a frame, enhancing the object analysis capability of the LVLM. _TC_ is a context that structures scene changes over time into a grid format, improving the action analysis capability of the LVLM. Notably, the proposed KSM and context generation modules operate in a training-free manner, allowing for easy application of C-VAD without additional training. 

To evaluate the performance of C-VAD, we classified existing VAD benchmark datasets based on anomaly types to create the C-VAD datasets. Through this process, we demonstrated the superiority of AnyAnomaly. Furthermore, despite being a zero-shot approach, AnyAnomaly achieved 



Figure 2. The architecture of AnyAnomaly 

competitive performance on VAD datasets compared to traditional OCC-based VAD models. Remarkably, It achieved state-of-the-art (SOTA) results on UBnormal [1] and UCFCrime [31], and showed superior generalization across all datasets. The proposed approach is expected to be an effective solution for deploying VAD technology in real-world applications. The contributions of this study are as follows: 

- We propose the C-VAD technique for anomaly detection in diverse environments. To the best of our knowledge, it is the first to perform VAD based on user-defined anomalies. 

- We develop the AnyAnomaly model, which applies context-aware VQA to perform C-VAD effectively. 

- To evaluate the performance of C-VAD, we construct new C-VAD datasets and experimentally verify the superiority of AnyAnomaly. 

- AnyAnomaly achieves SOTA performance on UBnormal and UCF-Crime and outperforms other methods in generalization across all datasets. 

## **2. Related Work** 

**Video Anomaly Detection.** Most VAD models adopt the OCC approach to detect anomalies by learning normal patterns. Among them, prediction-based methods train models to predict future or past frames based on normal frames, assuming that abnormal frames exhibit larger prediction errors. Liu [17] proposed a method that utilizes FlowNet [11] and GANs [23] to predict the _t_ + 1-th frame 

3027 

given _t_ input frames. Yang [37] introduced an approach that selects key frames from _t_ input frames to predict the entire sequence. However, because the definitions of normal and abnormal patterns may vary depending on the environment, the OCC approach, which relies on learned normal patterns, has a limited generalization performance. To mitigate this limitation, recent studies have explored cross-domain VAD (xVAD). Both rGAN [19] and MPN [21] adopted metalearning approaches, enabling adaptation to a target domain from only a few training samples. zxVAD [3] further improved adaptability to new environments by synthesizing abnormal patterns through the cut-mix technique applied to images from auxiliary datasets without explicit adaptation to the target domain. However, this approach depends on fixed data transformations, making it difficult to fully capture the diverse abnormal patterns that may occur in real-world scenarios. Therefore, we propose a novel VAD method that uses textual information to dynamically detect abnormal patterns that vary depending on the environment. 

**Large Vision Language Models.** Large language models have primarily been used in natural language processing; however, they have recently been applied to multimodal tasks such as image captioning and VQA. For example, MiniGPT-4 [41] processes multimodal inputs by connecting a pre-trained vision encoder to the Vicuna [7] model through a linear layer. Recent LVLMs have employed novel visual encoding techniques to better understand images. Chat-UniVi [13] generates dynamic tokens for images, thereby reducing unnecessary information and effectively extracting key visual features. This model enables flexible analysis by applying dynamic tokens across various resolutions. MiniCPM-V [38] applies the best partition technique according to the image resolution and generates tokens optimized for each segment, thereby improving the memory efficiency. However, despite the advancements in LVLMs, they are trained for general purposes, making their direct application in VAD challenging. Therefore, we propose a training-free approach to minimize the domain gap between the LVLMs and VAD tasks. 

## **3. Method** 

### **3.1. Overview** 

Figure 2 illustrates the structure of the AnyAnomaly model, which performs context-aware VQA. The input is a video segment _S_ = _{s_ 0 _, . . . , sN −_ 1 _}_ comprising _N_ frames, where _N_ is a multiple of 4. The KSM selects key frames _K_ = _{k_ 0 _, . . . , k_ 3 _}_ from _S_ . Among the selected key frames, the representative frame _k_<sup>ˆ</sup> is used to generate PC, whereas _K_ is used to create _TC_ . Subsequently, _k_<sup>ˆ</sup> , _PC_ , and _TC_ are utilized as image inputs for the LVLM, whereas the userprovided text _X_ is combined with a prompt and used as the 

text input. Finally, the LVLM’s response results are integrated to compute an anomaly score. 

The user-defined text _X_ refers to a natural language description of the anomaly the user wishes to detect. It can be a single word (e.g., “bicycle”), diverse events (e.g., “jumping-falling”), or a complex behavior (e.g., “driving outside the lane”). In the case of diverse events, each event keyword is processed individually as a single word. 

### **3.2. Key frames Selection Module** 

Figure 3a shows KSM, a key component of the segmentlevel approach. For this purpose, we selected four frames representing the segment as _K_ and utilized the CLIP [25] model, which was trained to match the images and text. 

Specifically, _S_ and _X_ are inputs to the image encoder _EI_ and text encoder _ET_ , respectively, and the similarity is calculated using the dot product of _N_ image and text embeddings. The frame with the highest similarity is selected as the representative frame _k_<sup>ˆ</sup> . 



The index of the representative frame _k_<sup>ˆ</sup> , denoted as<sup>ˆ</sup> _i_ , is used to select the other key frames. We divide the segment into four groups of equal size and select the<sup>ˆ</sup> _i_ mod<sup>_<u>N</u>_</sup> 4<sup>-th</sup> frame from each group. For example, when _N_ = 8 and ˆ _i_ = 4, the 0-th frame from each group is selected and the final set is _K_ = _{s_ 0 _, s_ 2 _, s_ 4 _, s_ 6 _}_ . This process is defined as follows: 



Using the KSM, _K_ is generated by considering both text alignment and temporal uniformity, thereby enabling effective context generation. A comparative analysis of the key frames selection method is presented in Section 4.5. 

### **3.3. Context Generation** 

_PC_ and _TC_ are key elements of context-aware VQA, serving as additional information that complements the input image. _PC_ enhances the object analysis capability of the LVLM and is generated through WinCLIP-based attention (WA). _TC_ strengthens the action analysis ability of the LVLM and is created through grid image generation (GIG). 

**WinCLIP-based Attention.** Figure 3b illustrates the WA method. We emphasize the regions related to _X_ at _k_<sup>ˆ</sup> based on WinCLIP, as proposed by Jeong [12]. First, _k_<sup>ˆ</sup> is divided into multiple windows, and individual embeddings are generated from each window using _EI_ . For example, when the image size is 240 _×_ 240, it is divided into 25 windows of size 48 _×_ 48, and the embeddings of each window are 

3028 



<!-- Start of picture text -->
-<br>Segment level approach Position Context Temporal Context<br>(a) Key frames Selection Module (KSM) (b) WinCLIP-based Attention (WA) (c) Grid Image Generation (GIG)<br><!-- End of picture text -->

Figure 3. Architecture of the proposed modules. KSM is essential for the segment-level approach, and WA and GIG are crucial for context generation. 

collected to create a small-scale window embedding map _W_<sup>_s_</sup> _∈_ R<sup>25</sup><sup>_×D_</sup> . By adjusting the window size, a middlescale window embedding map _W_<sup>_m_</sup> and large-scale window embedding map _W_<sup>_l_</sup> are also generated, and the similarity between these embedding maps and the text embedding _z ∈_ R<sup>_D_</sup> is calculated. The final similarity map _M_ is generated by averaging the similarities calculated on three scales: 



We combined the template proposed by Jeong [12] with _X_ and passed it through _ET_ to generate _z_ . Finally, we multiplied _M_ and _k_<sup>ˆ</sup> to create _PC_ : 



Here, _f_ norm represents min-max normalization, and _⊙_ denotes element-wise multiplication. _M_ was used after interpolation and reshaping to match the resolution of _k_<sup>ˆ</sup> . Because _PC_ is created by integrating similarities from multiple scales, it is robust to object size and location, and operates effectively even in situations with multiple objects. 

**Grid Image Generation.** Figure 3c illustrates the GIG method, which comprises two stages. In the multiscale grid generation stage, _K_ is used to create grid images at different scales. Similar to the process described in WA, each frame of _K_ is divided into multiple windows, and the windows at the same position are connected in a 2 _×_ 2 grid format to create a single grid image. This process is defined as follows: 



Here, _u_<sup>_i_</sup> _j_<sup>refers to the</sup><sup>_i_-th window created from</sup><sup>_kj_,and</sup><sup>_gi_</sup> refers to the _i_ -th grid image. We defined the sets of grid images generated using small-, middle-, and large-scale windows as _G_<sup>_s_</sup> , _G_<sup>_m_</sup> , and _G_<sup>_l_</sup> , respectively. 

In the grid image selection stage, the previously created sets are aggregated to generate _G_<sup>_all_</sup> . Then, using the same method as in KSM to select _k_<sup>ˆ</sup> , the grid image with the highest similarity to the text is chosen to generate _TC_ : 





The _TC_ generated through this process represents the object movement over time within the same background, making it advantageous for action analysis and robust to various object sizes. An analysis of the window sizes used in the WA and GIG is presented in Section 4.5. 

### **3.4. Anomaly Detection** 

Instead of tuning the LVLM, we propose a new prompt and context for performing context-aware VQA. The VQA results were used as anomaly scores to enable training-free zero-shot anomaly detection. 

**Prompt Design.** Figure 4 illustrates the proposed prompt _P_ . The prompt comprises three main components: ‘task’, ‘consideration’, and ‘output’. First, ‘task’ defines the operation the LVLM should perform, specifically evaluating whether _X_ is present in the image. Next, ‘consideration’ specifies factors to be taken into account during evaluation, while ‘output’ defines the format for presenting the evaluation results. To leverage the chain-of-thought [33] effect, 

3029 



<!-- Start of picture text -->
Task:  Evaluate whether the given image includes {text} on a scale from 0 to 1.<br>A score of 1 means {text} is clearly present in the image, while a score of 0<br>means {text} is not present at all. For intermediate cases, assign a value<br>between 0 and 1 based on the degree to which {text} is visible.<br>Consideration:  The key is whether {text} is present in the image, not its<br>focus. Thus, if {text} is present, even if it is not the main focus, assign a<br>higher score like 1.0.<br>Output:  Provide the score as a float, rounded to one decimal place, including<br>a brief reason for the score in one short sentence.<br>Prompt for temporal context<br>Context:  The given image represents a sequence (row 1 column 1 → row<br>1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal<br>progression.<br>Detailed prompt<br><!-- End of picture text -->



Figure 5. Comparison between the VAD and C-VAD datasets 

Figure 4. Proposed prompt for VQA 

we instructed the model to provide brief reasoning along with the anomaly score, rounded to one decimal place. When conducting VQA using _TC_ , an additional element, ‘context’, is inserted between ‘task’ and ‘consideration’ in the prompt. This context element conveys the meaning of the rows and columns of _TC_ to the LVLM. We define the modified prompt as _P_<sup>_∗_</sup> . 

**Anomaly Scoring.** The context serves as supplementary information to the image. However, because the LVLM accepts only a single image as input, it is challenging to utilize both the original and additional information simultaneously. To address this issue, we adopt a late fusion approach. Specifically, _k_<sup>ˆ</sup> , _PC_ , and _TC_ were used as the image inputs for the LVLM. The LVLM returns an anomaly score for each input, and these three scores are combined to compute the final _ascore_ : 



Here, _γ_ is a hyperparameter that adjusts the proportion of context reflected in _ascore_ . A performance comparison experiment based on the hyperparameter tuning is presented in supplementary materials. 

Consequently, even if the abnormal frame _k_<sup>ˆ</sup> receives a low score, the final anomaly score will be high if the additional information, _PC_ or _TC_ , is assigned a high score. This enabled accurate anomaly detection. Finally, to create frame-level anomaly scores, we duplicated the _ascore_ for the length of each segment and then applied a temporal 1-D Gaussian filter for smoothing, following prior works [26, 32]. 

## **4. Experiments** 

### **4.1. Datasets** 

Figure 5 illustrates the composition of the VAD and proposed C-VAD datasets. In conventional VAD datasets, 

videos are not categorized by an abnormal class type. In contrast, the proposed C-VAD datasets are organized by abnormal event type, with videos classified as positive or negative based on the presence of each abnormality. This categorization enables a precise evaluation of detection performance for specific types of abnormalities (e.g., bicycle). In this study, we validated the effectiveness of the proposed method on four VAD datasets: CUHK Avenue (Ave) [18], ShanghaiTech Campus (ShT) [20], UBnormal (UB) [1], and UCF-Crime (UCF) [31] as well as two C-VAD datasets: Customizable-ShT (C-ShT) and CustomizableAve (C-Ave). Further details of the datasets are provided in supplementary materials. 

### **4.2. Evaluation Criteria** 

To ensure consistency with previous VAD studies, the performance of the proposed model was evaluated using the micro-averaged area under the receiver operating characteristic curve (micro AUROC) metric. Specifically, the anomaly scores of all the frames in the dataset were aggregated, and the threshold of the anomaly score was progressively adjusted to compute the final evaluation. 

### **4.3. Results** 

Tables 1 and 2 present the evaluation results on the C-VAD datasets. The baseline, as described in Section 1, performs VQA at the frame level to compute anomaly scores. The proposed model achieved performance improvements of 9.88% and 13.65% compared to the baseline on the C-ShT and C-Ave datasets, respectively. Specifically, it showed improvements of 14.34% and 8.2% in the action class, and 3.25% and 21.98% in the appearance class, respectively. 

When only KSM was applied to the baseline, the execution time decreased in proportion to the segment length, whereas the average performance remained similar to that of the baseline. This is because the CLIP effectively selects representative frames for each segment, thereby compensating for the loss of temporal information. However, because it does not fully capture fine-grained spatio-temporal 

3030 

Table 1. Performance comparison on C-ShT dataset. The best results are **bolded** . The second-best results are underlined. 

|**Category**|**Class**|**Baseline**|**+KSM**|**+KSM/PC**|**+KSM/TC**|**Proposed**|**Improvement (%)**|
|---|---|---|---|---|---|---|---|
||Skateboarding|61.30|57.06|57.79|**73.66**|**73.66**|+20.16|
||Throwing|**91.41**|72.82|88.74|82.53|90.67|-0.81|
||Running|53.13|51.93|53.68|59.77|**60.11**|+13.14|
|**Action**|Loitering|61.98|51.96|**81.27**|76.94|**81.27**|+31.12|
||Jumping|82.84|92.89|92.91|**95.31**|**95.31**|+15.05|
||Falling|78.31|78.95|79.24|**88.01**|**88.01**|+12.39|
||Fighting|84.48|91.18|91.18|**98.06**|**98.06**|+16.07|
||**Average**|73.35|72.00|77.83|82.04|**83.87**|+14.34|
||Car|88.72|90.96|**91.46**|90.96|**91.46**|+3.09|
||Hand truck|95.50|98.20|98.91|**99.20**|**99.20**|+3.87|
|**Appearance**|Bicycle|72.36|72.46|**78.47**|72.46|**78.47**|+8.44|
||Motorcycle|**88.04**|86.72|86.72|86.72|86.72|-1.50|
||**Average**|86.16|87.09|88.89|87.34|**88.95**|+3.25|
|**Overall Average**||78.01|77.48|81.85|83.97|**85.72**|+9.88|



Table 2. Performance comparison on C-Ave dataset 

|**Category**|**Class**|**Baseline**|**+KSM**|**+KSM/PC**|**+KSM/TC**|**Proposed**|**Improvement (%)**|
|---|---|---|---|---|---|---|---|
||Throwing|78.44|80.13|**89.77**|82.40|**89.77**|+14.44|
|**Ati**|Running|75.82|77.67|77.67|**77.90**|**77.90**|+2.74|
|**con**|Dancing|85.65|72.28|76.64|**91.92**|**91.92**|+7.32|
||**Average**|79.97|76.69|81.36|84.07|**86.53**|+8.2|
||Too close|57.23|61.48|61.48|**91.78**|**91.78**|+60.37|
|**Appearance**|Bicycle|99.99|99.84|99.99|99.93|**100.00**|+0.01|
||**Average**|78.61|80.66|80.74|95.86|**95.89**|+21.98|
|**Overall Average**||79.43|78.28|81.11|88.79|**90.27**|+13.65|



details, its performance significantly decreases for certain classes. Therefore, we address these issues using the proposed contextual information. First, using _PC_ resulted in performance improvements of 5.64% and 3.62% compared to the KSM, as the LVLM focused on analyzing objects related to _X_ . Additionally, applying _TC_ led to performance improvements of 8.38% and 14.43% over the KSM, respectively, with particularly notable enhancements observed in the action class. This indicates that utilizing the temporal information provided in the grid image is essential for action analysis. Additional validations, such as FPS comparisons by segment length and performance evaluations across various LVLMs, are presented in supplementary materials. 

### **4.4. Qualitative Analysis** 

To analyze the effect of context-aware VQA, we present the visualization results for the anomaly scores and input 

frames in Figure 6. When _PC_ was not applied, the bicycle object appeared smaller then the other objects, leading to a lower detection performance. Once _PC_ is applied, the bicycle region is emphasized, thereby enhancing the object recognition capability of the LVLM. Similarly, without _TC_ , the model misinterpreted fighting as standing, resulting in lower detection performance. Incorporating temporal information through _TC_ improves the action recognition capability of the LVLM. These results demonstrate that contextaware VQA is more effective than the conventional VQA. 

### **4.5. Ablation Study** 

**Key frames Selection.** We conducted an ablation study on key frames selection from two perspectives: temporal uniformity and text alignment. The random method considers neither of these aspects, whereas the CLIP-based approach considers only text alignment. Selecting key 

3031 



<!-- Start of picture text -->
The image does not show a clear  The image includes a group of people<br>bicycle , so the score is  0.4 . riding  bicycles , with at least one<br>clearly visible bicycle. The score<br>would be  0.9 .<br>The image shows a group of young  The image shows a group of people<br>men standing on a paved area, but  fighting , with at least two individuals<br>there is no evidence of   fighting .  engaged in physical altercations. The<br>The score is  0.5 . score is  0.9 .<br><!-- End of picture text -->

Figure 6. Anomaly score comparison with context visualization 

Table 3. Comparison on key frames selection method. RD, CP and Gr. indicate random, CLIP and grouping, respectively. * indicates testing without context. Act. and App. indicate action and appearance class, respectively. 

|**Key**<br>||**C-ShT**|||**C-Ave**||
|---|---|---|---|---|---|---|
|**frames**|Act.|App.|Total|Act.|App.|Total|
|RD*|69.9|84.0|75.0|66.4|78.8|71.3|
|CP*|72.0|87.1|77.5|76.7|80.7|78.3|
|RD|80.0|**89.1**|83.3|79.1|92.3|84.4|
|CP|81.2|88.9|84.0|84.3|81.2|83.1|
|Gr. →CP|82.2|88.8|84.7|83.9|92.2|87.2|
|CP→Gr.|**83.9**|89.0|**85.7**|**86.5**|**95.9**|**90.3**|



frames using CLIP after grouping ensures text alignment but does not guarantee temporal uniformity. Applying grouping after CLIP resulted in evenly distributed key frames, thereby considering both temporal uniformity and text alignment. As shown in Table 3, incorporating both factors yielded the best performance for C-VAD, highlighting the critical role of temporal uniformity in action recognition. Furthermore, RD* and CP*, which do not utilize the contextual information, perform worse than the random method, which disregards both temporal uniformity and text alignment. This demonstrates the importance of leveraging the contextual information. 

**Window Size.** Table 4 presents the experimental results based on the window sizes used in _PC_ and _TC_ . For the action classes, the best performance was achieved with the large window size in C-ShT and the middle window size in C-Ave. This indicates that middle or large window sizes are more effective in capturing temporal movements and interactions between multiple objects. For appearance classes, the optimal performance was observed with the small window size in C-ShT and the middle window size in C-Ave, suggesting that the appropriate window size varies depending on the dataset owing to differences in object sizes. To enhance the generalization performance of the model, we adopted an approach that utilized all three window sizes and found that incorporating them yielded the best overall performance. Further details on the window size are provided in the implementation details section of the supplementary material. 

### **4.6. Comparison with SOTA** 

To assess the effectiveness of AnyAnomaly in handling multiple text inputs, we conducted experiments on the VAD benchmark datasets. For performance evaluation, each anomaly class in the dataset was treated as _X_ , and the maximum anomaly score among all computed scores was assigned to the corresponding segment. Table 5 presents a performance comparison with frame-centric VAD methods. 

3032 

Table 4. Comparison on window size. 

|**Window**<br>||**C-ShT**|||**C-Ave**||
|---|---|---|---|---|---|---|
|**Size**|Act.|App.|Total|Act.|App.|Total|
|small|78.8|**90.6**|83.1|84.7|87.1|85.7|
|middle|81.2|89.0|84.1|**87.5**|92.0|89.3|
|large|82.1|89.7|84.9|86.8|86.4|86.6|
|all|**83.9**|89.0|**85.7**|86.5|**95.9**|**90.3**|



UB and UCF datasets, which cover diverse scenes and abnormal events, thereby demonstrating the effectiveness of the proposed model across various environments. Furthermore, while LLM-based methods (e.g., AnomalyRuler) require rule generation and aggregation using a few normal samples, the proposed method achieves competitive performance solely through zero-shot inference, highlighting its practical applicability. 

### **4.7. Generalization Performance Comparison** 

Table 5. Comparison with state-of-the-art VAD methods. * indicates testing without context. 

|**Method**|**Zero-shot**|**Ave**|**ShT**|**UB**|**UCF**|
|---|---|---|---|---|---|
|AMMC-Net[6]|✗|86.6|73.7|-|-|
|STEAL-Net[5]|✗|87.1|73.7|-|-|
|MPN[21]|✗|89.5|73.8|-|-|
|DLAN-AC[36]|✗|89.9|74.7|-|-|
|UBnormal[1]|✗|-|-|68.5|-|
|FPDM[34]|✗|90.1|78.6|62.7|74.7|
|SLM[30]|✗|90.9|78.8|-|-|
|USTN-DSC[37]|✗|89.9|73.8|-|-|
|AnomalyRuler[35]|✗|89.7|**85.2**|71.9|-|
|MULDE[24]|✗|-|81.3|72.8|78.5|
|AED-MAE[27]|✗|**91.3**|79.1|58.5|-|
|MA-PDM[40]|✗|**91.3**|79.2|63.4|-|
|AccI-VAD[26]|✗|-|76.2|66.8|60.3|
|AnyAnomaly*|✓|81.4|77.2|73.1|77.8|
|AnyAnomaly|✓|87.3|79.7|**74.5**|**80.7**|



Table 6. Generalization performance comparison. Tr.: crossdomain training where models trained on one VAD dataset are evaluated on another. Few.: methods that adapt to the target domain using only a few training samples, Aux.: methods that utilize auxiliary datasets, *: since competitors did not perform crossdomain evaluations on ShT, we present their same-domain results instead. 

|**Method**|**Tr.**|**Few.**|**Aux.**|**Ave**|**ShT**|
|---|---|---|---|---|---|
|STEAL-Net[5]|✓|✗|✗|54.3|51.7|
|Jigsaw[32]|✓|✗|✗|62.9|59.3|
|rGAN[19]|✓|✓|✗|76.6|77.9*|
|MPN[21]|✓|✓|✗|78.9|73.8*|
|zxVAD[3]|✓|✗|✓|82.2|71.6*|
|Shibao et al.[8]|✓|✗|✓|86.2|78.7|
|ZS CLIP[25]|✗|✗|✗|62.3|60.9|
|ZS ImageBind[9]|✗|✗|✗|64.5|61.3|
|LLaVA-1.5[16]|✗|✗|✗|67.4|59.6|
|Video-ChatGPT[22]|✗|✗|✗|76.9|69.1|
|AnyAnomaly|✗|✗|✗|**87.3**|**79.7**|



Despite not being trained on VAD datasets, AnyAnomaly demonstrated a performance comparable to that of SOTA methods. Notably, it achieved the best AUC results on the 

Table 6 presents a comparison of the generalization performance of AnyAnomaly. Although STEAL-Net [5] and Jigsaw [32] achieved high accuracy in same-domain testing, their performance was significantly degraded in crossdomain settings. Specifically, on the Ave dataset, the performances of STEAL-Net and Jigsaw decreased as 87.1% → 54.3% and 92.2% → 62.9%, respectively. Similarly, on the ShT dataset, their performance decreased as 73.7% → 51.7% and 84.3% → 59.3%, respectively. This suggests that the existing OCC-based VAD models tend to overfit the training data, making them less effective when applied to new environments. For instance, ‘Too close’ where an object is in close proximity to the camera is considered anomalous in the Ave dataset but normal in the ShT dataset. Consequently, OCC-based models trained on ShT struggle to detect such anomalies. 

The zero- and few-shot VAD models designed for xVAD exhibited better generalization performance than the OCCbased models. However, few-shot models depend heavily on the number of K-shot samples, whereas zero-shot models require auxiliary datasets. Training-free methods using VLMs, such as ZS-CLIP and Video-ChatGPT, leverage strong image and video understanding capabilities, outperforming some VAD models. Nevertheless, their performance is limited by domain gaps. In contrast, AnyAnomaly effectively overcomes these gaps by incorporating contextual information, achieving superior performance. 

## **5. Conclusion** 

We propose AnyAnomaly, a novel approach that leverages the LVLM for universal VAD. AnyAnomaly effectively performs the C-VAD by incorporating a segmentlevel approach and context-aware VQA. This design reduces latency when processing large videos and minimizes the domain gap between the LVLM and VAD task. Despite being a zero-shot method, AnyAnomaly demonstrates competitive performance on benchmark datasets and holds promise for real-world VAD. Furthermore, because it operates without any training and enables anomaly detection in any video, it significantly improves accessibility in the VAD domain. We anticipate that AnyAnomaly will contribute substantially to VAD research and practical deployment. 

3033 

## **Acknowledgements** 

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT).(No. RS-2025-02312833) 

## **References** 

- [1] Andra Acsintoae, Andrei Florescu, Mariana-Iuliana Georgescu, Tudor Mare, Paul Sumedrea, Radu Tudor Ionescu, Fahad Shahbaz Khan, and Mubarak Shah. Ubnormal: New benchmark for supervised open-set video anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 20143–20153, 2022. 2, 5, 8 

- [2] Sunghyun Ahn, Youngwan Jo, Kijung Lee, and Sanghyun Park. Videopatchcore: An effective method to memorize normality for video anomaly detection. In _Proceedings of the Asian Conference on Computer Vision_ , pages 2179–2195, 2024. 1 

- [3] Abhishek Aich, Kuan-Chuan Peng, and Amit K RoyChowdhury. Cross-domain video anomaly detection without target domain adaptation. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 2579–2591, 2023. 3, 8 

- [4] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pages 2425– 2433, 2015. 2 

- [5] Marcella Astrid, Muhammad Zaigham Zaheer, and Seung-Ik Lee. Synthetic temporal anomaly guided end-to-end video anomaly detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 207–214, 2021. 8 

- [6] Ruichu Cai, Hao Zhang, Wen Liu, Shenghua Gao, and Zhifeng Hao. Appearance-motion memory consistency network for video anomaly detection. In _Proceedings of the AAAI conference on artificial intelligence_ , pages 938–946, 2021. 8 

- [7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2(3):6, 2023. 3 

- [8] Shibo Gao, Peipei Yang, and Linlin Huang. Scene-adaptive svad based on multi-modal action-based feature extraction. In _Proceedings of the Asian Conference on Computer Vision_ , pages 2471–2488, 2024. 8 

- [9] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15180–15190, 2023. 8 

- [10] Seungkyun Hong, Sunghyun Ahn, Youngwan Jo, and Sanghyun Park. Making anomalies more anomalous: Video 

   - anomaly detection using a novel generator and destroyer. _IEEE Access_ , 2024. 1 

- [11] Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox. Flownet 2.0: Evolution of optical flow estimation with deep networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2462–2470, 2017. 2 

- [12] Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer. Winclip: Zero/few-shot anomaly classification and segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19606–19616, 2023. 3, 4 

- [13] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13700– 13710, 2024. 3 

- [14] Dongha Lee, Sehun Yu, and Hwanjo Yu. Multi-class data description for out-of-distribution detection. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , pages 1362–1370, 2020. 1 

- [15] Kijung Lee, Youngwan Jo, Sunghyun Ahn, and Sanghyun Park. Mdvad: Multimodal diffusion for video anomaly detection. In _Pacific-Asia Conference on Knowledge Discovery and Data Mining_ , pages 121–133. Springer, 2025. 1 

- [16] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 26296–26306, 2024. 8 

- [17] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6536–6545, 2018. 1, 2 

- [18] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In _Proceedings of the IEEE international conference on computer vision_ , pages 2720–2727, 2013. 5 

- [19] Yiwei Lu, Frank Yu, Mahesh Kumar Krishna Reddy, and Yang Wang. Few-shot scene-adaptive anomaly detection. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16_ , pages 125–141. Springer, 2020. 3, 8 

- [20] Weixin Luo, Wen Liu, and Shenghua Gao. A revisit of sparse coding based anomaly detection in stacked rnn framework. In _Proceedings of the IEEE international conference on computer vision_ , pages 341–349, 2017. 5 

- [21] Hui Lv, Chen Chen, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang. Learning normal dynamics in videos with meta prototype network. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 15425–15434, 2021. 3, 8 

- [22] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 12585–12602, 2024. 8 

3034 

- [23] Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In _Proceedings of the IEEE international conference on computer vision_ , pages 2794–2802, 2017. 2 

- [24] Jakub Micorek, Horst Possegger, Dominik Narnhofer, Horst Bischof, and Mateusz Kozinski. Mulde: Multiscale logdensity estimation via denoising score matching for video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18868–18877, 2024. 8 

- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 3, 8 

- [26] Tal Reiss and Yedid Hoshen. An attribute-based method for video anomaly detection. _Transactions on Machine Learning Research_ . 1, 5, 8 

- [27] Nicolae-C Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah, et al. Self-distilled masked auto-encoders are efficient video anomaly detectors. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15984–15995, 2024. 8 

- [28] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Sch¨olkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14318–14328, 2022. 1 

_national Conference on Computer Vision_ , pages 5527–5537, 2023. 8 

   - [35] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. Follow the rules: Reasoning for video anomaly detection with large language models. _ArXiv_ , abs/2407.10299, 2024. 8 

   - [36] Zhiwei Yang, Peng Wu, Jing Liu, and Xiaotao Liu. Dynamic local aggregation network with adaptive clusterer for anomaly detection. In _European Conference on Computer Vision_ , pages 404–421. Springer, 2022. 8 

   - [37] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on keyframes for video anomaly detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14592–14601, 2023. 1, 3, 8 

   - [38] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. _arXiv preprint arXiv:2408.01800_ , 2024. 3 

   - [39] Jihun Yi and Sungroh Yoon. Patch svdd: Patch-level svdd for anomaly detection and segmentation. In _Proceedings of the Asian conference on computer vision_ , 2020. 1 

   - [40] Hang Zhou, Jiale Cai, Yuteng Ye, Yonghui Feng, Chenxing Gao, Junqing Yu, Zikai Song, and Wei Yang. Video anomaly detection with motion and appearance guided patch diffusion model. _arXiv preprint arXiv:2412.09026_ , 2024. 8 

   - [41] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 3 

- [29] Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel M¨uller, and Marius Kloft. Deep one-class classification. In _International conference on machine learning_ , pages 4393–4402. PMLR, 2018. 1 

- [30] Chenrui Shi, Che Sun, Yuwei Wu, and Yunde Jia. Video anomaly detection via sequentially learning multiple pretext tasks. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 10330–10340, 2023. 8 

- [31] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6479–6488, 2018. 2, 5 

- [32] Guodong Wang, Yunhong Wang, Jie Qin, Dongming Zhang, Xiuguo Bao, and Di Huang. Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles. In _European Conference on Computer Vision_ , pages 494–511. Springer, 2022. 5, 8 

- [33] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 4 

- [34] Cheng Yan, Shiyu Zhang, Yang Liu, Guansong Pang, and Wenjun Wang. Feature prediction diffusion model for video anomaly detection. In _Proceedings of the IEEE/CVF Inter-_ 

3035 

