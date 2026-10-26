# **MEMORY-AUGMENTED ONLINE VIDEO ANOMALY DETECTION** 

_Leonardo Rossi Vittorio Bernuzzi Tomaso Fontanini Massimo Bertozzi Andrea Prati_ 

University of Parma, Department of Engineering and Architecture 

## **ABSTRACT** 

The ability to understand the surrounding scene is of paramount importance for Autonomous Vehicles (AVs). This paper presents a system capable to work in an online fashion, giving an immediate response to the arise of anomalies surrounding the AV, exploiting only the videos captured by a dashmounted camera. Our architecture, called _MOVAD_ , relies on two main modules: a Short-Term Memory Module to extract information related to the ongoing action, implemented by a Video Swin Transformer (VST), and a Long-Term Memory Module injected inside the classifier that considers also remote past information and action context thanks to the use of a Long-Short Term Memory (LSTM) network. The strengths of MOVAD are not only linked to its excellent performance, but also to its straightforward and modular architecture, trained in a end-to-end fashion with only RGB frames with as less assumptions as possible, which makes it easy to implement and play with. We evaluated the performance of our method on Detection of Traffic Anomaly (DoTA) dataset, a challenging collection of dash-mounted camera videos of accidents. After an extensive ablation study, MOVAD is able to reach an AUC score of 82.17%, surpassing the current state-of-the-art by +2 _._ 87 AUC. 

**_Index Terms_ —** Video Transformer, LSTM, Video Anomaly Detection, Online VAD, Memory-Augmented Networks 

## **1. INTRODUCTION AND RELATED WORKS** 

Autonomous Vehicles (AVs) are becoming every day a reality thanks to the recent enormous scientific and technical advances. Nevertheless, safety of AVs is still a relevant issue which can jeopardize their world-wide diffusion. Increasing safety of AVs can be reached by providing vehicles with the ability of detecting anomalous situations in a prompt way. Their detection provides information to avoid collisions, protect pedestrians, or re-route the current travel [4]. Among the different sensors exploited in AVs, cameras offer rich realtime information about the scene, but, despite all, anomalous situation detection in real traffic scenarios is still very challenging. In particular, we still miss a formal shared model of what an anomaly should be, because many times the riskiness is highly subjective. In addition, there are plenty of possible accident classes with a very exiguous number of ex- 

amples to take into account compared to normal traffic situations. Lastly, the definition of time boundaries of an anomaly is even more subjective and doubtful. Nevertheless, some attempts have been made to propose a deterministic method to define an anomaly. Fang _et al._ [3] labels the anomaly start from the moment in which half part of the object involved in the accident appears in the view. Yao _et al._ [15] proposed the Detection of Traffic Anomaly (DoTA) dataset, where the anomaly is starting the instant after which the accident is unavoidable. We take this last dataset as the benchmark for our work because it is the most complete and used dataset for this type of task in the road traffic safety context. 

There have been several previous works addressing the problem of video anomaly detection. Authors in [6] proposed a Convolutional AutoEncoder (ConvAE) trained only on normal frames with the objective of frame reconstruction. In [9, 13], authors used Convolutional LSTM Auto-Encoder as framework to encode appearance and motion. As noted by [11], auto-encoder-type reconstruction methods are sensitive to the amount of anomalies that occurs in the scene and many times they require additional ad-hoc post-processing techniques. Authors in [7] proposed AnoPred, which uses a multi-task loss, including image intensity, optical flow, gradient and adversarial losses for Video Anomaly Detection (VAD) by predicting a future frame. As stated in [15], AnoPred was thought in a video surveillance context, while videos acquired from inside a moving vehicle are more dynamic and difficult to predict. In [16], authors of STFE model make a two-stage detector: a coarse detection to encode temporal features with Histogram of Optical Flow (HOF) [12] and ordinal features of frames by a CNN, and then, a fine detection, by encoding the CNN features and spatial relationships of the objects. In FOL [15], authors try to avoid the future prediction for the entire frame, focusing instead on tracking actors’ positions and predicting their future locations. Conversely to us, the applicability of both last methods are limited by the presence of actors in the Field of View (FOV). Moreover, AnoPred [7], STFE [16] and FOL [15] rely on supplementary input information other than RGB frames, such as optical flow, bounding boxes and ego-motion information of the moving camera. Our model outperforms them by exclusively processing RGB frames, thereby highlighting its capability to comprehend anomalies in video traffic scenes without the need for auxiliary information. In [14], the TRN 



<!-- Start of picture text -->
s[t] s[t] RNN State<br>(next)<br>... 0/1 ...<br>Linear<br>Dropout<br>Head Linear<br>Reducer Long-Term Memory Module<br>(LSTM)<br>Short-Term Memory Module<br>(Video Swin Transformer - VST) Layer Norm<br>Dropout<br>Time Linear<br>Layer Norm<br>... ...<br>f[t-3] f[t-2] f[t-1] f[t]<br>RNN State<br>Video Clip Length NF X (previous)<br><!-- End of picture text -->

**Fig. 1** : The online frame-level VAD architecture. _f_ [ _t_ ] is the frame at time _t_ , _x_ the output of the Reducer, _NF_ the number of frames in input to the VST, _s_ [ _t_ ] the anomaly classification score of the frame _f_ [ _t_ ]. 

model couples the action detection task with both the temporal dependencies modeled by a RNN and the anticipation of the future via a temporal decoder. Like them, we exploit an RNN to obtain temporal information, but not limiting ourselves to this. We formulated an advanced version of memory, differentiating between short- and long-memory through two distinct and trained-at-same-time modules, obtaining a more informative description of the ongoing situation, without speculating on the future. Summarizing, the main contributions of this paper are: 

- A straightforward end-to-end architecture for Online Video Anomaly Detection (VAD) called _MOVAD_ , which processes ongoing RGB frames, with as less assumptions as possible about the ongoing action and without any extra auxiliary information. 

- A plugged-in Transformer [8] model used as backbone for Short-Term Memory information extraction in an online scenario. This allows the system to incorporate the most recent temporal and spatial correlations by relying only on the current and few past frames. 

- The injection of a LSTM module inside the classification head to model the Long-Term Memory and to exploit contextual information spread across the entire history. 

- An exhaustive ablation study over the dataset DoTA [15]. Compared w.r.t. the state of the art, MOVAD shows superior performance in terms of AUC. 

## **2. MEMORY-AUGMENTED ONLINE VAD** 

In our system, recently observed frames and past frames are taken into account as a source of information related to the ongoing action and the context, respectively. In order to process current and past frames for the Online VAD task, we 

equipped MOVAD of two main components: a Short-Term Memory Module (STMM) and a Long-Term Memory Module (LTMM) inside the classification head (see Fig. 1). 

**Short-Term Memory Module.** In order to pre-elaborate the new coming frame _f_ [ _t_ ] and, in conjunction, to retain nearpast spatio-temporal information _f_ [ _t −_ 1] _..f_ [ _t −_ ( _NF −_ 1)], we chosen a transformer architecture over an RNN, to be able to process them in a parallel fashion. We lean towards transformer over 3D convolutional models because of the well demonstrated abilities to intercept long-distance interactions in space and time [10]. We selected VST [8] as our backbone network for the STMM over models like ViViT [1], due to its superior performance and more efficient computation of self-attention mechanism. Despite this, the selfattention mechanism remains computationally intensive, particularly for long videos. Therefore, we limited the input to a small temporal window of _NF_ = 4 frames of the video, going from the current frame at time _t_ to the previous frame at time _t−_ ( _NF −_ 1). In this way, we obtain a compact representation of the short history that will be subsequently forwarded to the LTMM. VST takes as input a video with size _NF ×H ×W ×_ 3, where _NF_ , _H_ , _W_ and 3 correspond to the number of frames, height, width and RGB channels, respectively. The model internally splits the frames in non-overlapping 3D patches, partitioning the video in<sup>_<u>NF</u>_</sup> 2 _×_<sup>_<u>H</u>_</sup> 4<sup>_×_</sup><sup>_<u>W</u>_</sup> 4<sup>3Dtokens,andus-</sup> ing a 3D shifted window mechanism to obtain cross-window connections and exploit spatio-temporal information. 

**Long-Term Memory Module.** As will be highlighted in ablation in Section 3.1, too much information about the past ( _NF >_ 4) tends to mislead the STMM, resulting in a loss of performance. According to our hypothesis, this occurs because all frames are placed at the same level, without a possibility of weighing them based on how much they belong to the past. For this reason, we engineered a different way to integrate and enrich the output latent space from the STMM with contextual information extrapolated from the distant past. Every time a new frame is available, we transfer the compact representation obtained from the STMM to our LTMM, in order to update its representation of the past. In this way, we can maintain the focus on what is happening now, and at same time accumulating a richer understanding of the scene, enabling more precise classifications. In details, the output of VST goes through Adaptive Average Pool 3D layer (Reducer in Fig. 1) and, finally, enters inside the classification head. As shown in Fig. 1, the head is composed by a series of normalization layers, linear layers and dropout, alternating. A stacked three-cell LSTM module is inserted after the last normalization layer to model the long-term memory. The state, composed by three hidden _h_ [ _t_ ] and cell _c_ [ _t_ ] states, is updated whenever a new frame is available. The LSTM receives in input a features block of [ _B,_ 1024], where _B_ is the batch size, and returns a block of same size together with the state of the cells. Since the state is relatively small, the module is very efficient and leads to a fixed and limited additional com- 



**Fig. 2** : Performance comparison, changing the number of frames ( _NF_ ) in input to the STMM (from 1 to 6). 



**Fig. 3** : Performance comparison, changing the number of LSTM cells (from 0 to 4). “LSTM (2 cells)” corresponds to “ _NF_ = 3” in Fig. 2. 

putational cost. For each frame _f_ [ _t_ ], the classification head outputs the anomaly classification score _s_ [ _t_ ] _∈_ [0 _,_ 1], where 0 means no anomaly and 1 means the frame is anomalous. A weighted cross-entropy loss was chosen to train the model, giving higher weight to the anomaly class reflecting the distribution of the data. We used the formula _wi_ = _e/ei_ , where _wi_ is the weight associated to the class, _e_ the total amount of examples in the dataset and _ei_ the amount of examples for the class _i_ . 

## **3. EXPERIMENTAL RESULTS** 

**Dataset** . We perform our training and test on Task 1, the frame-level VAD, of DoTA dataset [15], using only the anomaly class and its temporal boundaries, strictly in the online scenario. 

**Evaluation Metrics** . We use the well-known Area Under the Curve (AUC) metric at frame-level, to evaluate how well the model is able to temporally locate the anomaly in the videos. **Training details.** We perform the training on a single ma- 

chine with an A100 GPU, with Stochastic Gradient Descent (SGD) optimization algorithm, learning rate of 0.0001, momentum of 0.9. We use SGD instead of Adam because in our experiments the latter led the training to be more unstable, resulting in model diverging after few epochs. Unless otherwise specified, batch size is 8, input video size is 320 _×_ 240, Video Clip Length (VCL), which is the number of frames inside the batch for each video, is 8, LSTM cell number is 2, _NF_ is 3, linear weights are initialized using a uniform distribution, LSTM cells with a (semi-)orthogonal matrix, bias parameters are set to zero and the VST is initialized with a model pretrained on Something-Something v2 [5]. We train using a weighted Cross-Entropy loss to address the issue of imbalanced data within the DoTA dataset, assigning _wn_ = 0 _._ 3 and _wa_ = 0 _._ 7 to the normal and anomaly class, respectively following equation reported at the end of Section 2. 



**Fig. 4** : Performance comparison, changing the VCL. The “8 frames” configuration corresponds to “ _NF_ = 3” in Fig. 2. 

|#|Short-term<br>Long-term|_AUC_|
|---|---|---|
|1||66.53|
|2|✓|74.46|
|3|✓|68.76|
|**4**|✓<br>✓|**79.21**|



**Table 1** : Performance comparison with and w/out short and long-term memory. Short-term: with ( _NF_ = 3) and w/out ( _NF_ = 1). Long-term: with (2 cells) and w/out (0 cells). 

## **3.1. Ablation study** 

In this section different configurations will be individually explored and, finally, an optimal setup of MOVAD will be presented in Section 3.2. 

**Memory modules effectiveness.** As shown in Table 1, we first tested the effect of memories. STMM and LTMM both contribute to enhance the general performance, obtaining the best AUC when both are active, highlighting their importance. **Short-Term Memory Module.** We decided to design our STMM based on a Video Swin-B architecture, characterized 

|#|Method|Input|_AUC_|
|---|---|---|---|
|1|ConvAE [6] (*)|Gray|64.3|
|2|ConvAE [6] (*)|Flow|66.3|
|3|ConvLSTMAE [2] (*)|Gray|53.8|
|4|ConvLSTMAE [2] (*)|Flow|62.5|
|5|AnoPred [7] (*)|RGB|67.5|
|6|AnoPred [7] (*)|Masked RGB|64.8|
|7|FOL-Ensemble [15] (*)|RGB + Box + Flow + Ego|73.0|
|8|STFE[16]|RGB + Flow|79.3|
|**9**|**Our (MOVAD)**|**RGB (**320_×_240**)**|**80.09**|
|**10**|**Our (MOVAD)**|**RGB (**640_×_480**)**|**82.17**|



**Table 2** : Benchmarks of VAD methods on the DoTA dataset. Both MOVAD models are trained with the best configuration of: VCL of 8, 3 LSTM cells and _NF_ = 4. (Note: Results with (*) are taken from DoTA [7] paper. 

|Model|_ST_|_AH_|_LA_|_OC_|_TC_|_V P_|_V O_|_OO_|_UK_|
|---|---|---|---|---|---|---|---|---|---|
|AnoPred [7]|69.9|73.6|75.2|69.7|73.5|66.3|N/A|N/A|N/A|
|AnoPred [7] + Mask|66.3|72.2|64.2|65.4|65.6|66.6|N/A|N/A|N/A|
|FOL-STD [15]|67.3|77.4|71.1|68.6|69.2|65.1|N/A|N/A|N/A|
|FOL-Ensemble [15]|73.3|81.2|74.0|73.4|75.1|70.1|N/A|N/A|N/A|
|STFE [16]|75.2|84.5|72.1|77.3|72.8|71.9|N/A|N/A|N/A|
|**Our (MOVAD)**|85.6|85.1|83.9|82.2|85.3|**86.2**|**79.3**|86.7|**77.1**|
|**Our(MOVAD) †**|**86.6**|**86.3**|**84.9**|**83.7**|**85.5**|81.6|77.4|**87.9**|73.8|
|Model|_ST∗_|_AH∗_|_LA∗_|_OC∗_|_TC∗_|_V P∗_|_V O∗_|_OO∗_|_UK∗_|
|AnoPred [7]|70.9|62.6|60.1|65.6|65.4|64.9|64.2|57.8|N/A|
|AnoPred [7] + Mask|72.9|63.7|60.6|66.9|65.7|64.0|58.8|59.9|N/A|
|FOL-STD [15]|75.1|66.2|66.8|74.1|72.0|69.7|63.8|69.2|N/A|
|FOL-Ensemble [15]|77.5|69.8|68.1|76.7|73.9|71.2|65.2|69.6|N/A|
|STFE [16]|**80.6**|65.6|69.9|76.5|74.2|N/A|75.6|70.5|N/A|
|**Our (MOVAD)**|72.1|71.6|72.3|76.5|75.7|74.1|77.9|71.7|69.1|
|**Our(MOVAD) †**|72.2|**74.0**|**74.8**|**80.2**|**79.6**|**76.8**|**82.2**|**78.3**|**72.9**|



**Table 3** : Detection accuracy (AUC) for individual accident categories. “*” non-ego anomaly categories. “†” if input resolution is 640 _×_ 480 instead of 320 _×_ 240. N/A=Not Available. Bold and red values are the best and second-best results. 

by an embedding dimension of _C_ = 128 after the linear projection of the patches. In Fig. 2, results when varying the number of frames _NF_ processed by the STMM at each step are displayed. As expected, taking into account only the current frame is the worst situation, loosing any temporal information and making the training unstable. This is reasonable: by not having any knowledge of the near past, the STMM leads the LTMM to overfitting since consecutive frames are very similar. On the contrary, increasing the number of frames generally increase the performance. This is true until processing 5 or more frames, where the effect becomes counterproductive. As mentioned in Section 2, according to our hypotheses, overloading the transformer becomes harmful, as it does not have a mechanism to weight the remote and recent past differently. Overall, highest AUC is obtained with 4 frames. **Long-Term Memory Module.** In Fig. 3 the LTMM capabilities are evaluated, varying the number of cells from zero (no LSTM at all) to four. Indeed, having no cells makes training slower in saturating performance and the lowest AUC is reached. In general, by increasing the number of cells, the maximum AUC is reached slower but it is higher than w/out 

LSTM. With 1 cell the performance saturates very quickly, with a slow degradation during the rest of the epochs. On the other side, 4 cells lead to a very slow saturation w/out reaching the best performance. The global maximum AUC is reached with 2 cells, even if with just +0 _._ 02 compared to 3 cells. Despite this, we prefer the latter configuration, because we think its ability to increase the quality of training in a slower and more continuous way has a better general benefit. We verified this hypothesis and, in conjunction with _NF_ = 4 (best configuration in the previous experiment), it permits to obtain higher AUC than with 2 cells. We speculate this happens because both (3 cells in Fig. 3 and _NF_ = 4 in Fig. 2) reach the best AUC at same time (around 400 epochs). 

**Video clip length (VCL).** In Fig. 4, different values of VCL are explored. The worst and most unstable training is obtained with 4 frames, probably because they are too few to exploit the long-term memory effect of LSTM cells. Increasing VCL permits to saturate performance quicker, but a value too high (like 12 or 16) tends to produce a lower AUC overall. The highest AUC is obtained with 8 frames as VCL, which represents a good trade-off between enlarging clip size and exploiting LSTM cells, and reducing it to avoid overfitting, due to consecutive frames being too similar to each other. 

## **3.2. Comparison with the state of the art** 

Finally, in Table 2 we compare MOVAD (with 320 _×_ 240 and 640 _×_ 480 input videos size) with state-of-the-art (SOTA) models. The training of MOVAD followed the best configuration found using information obtained through ablation studies: VCL of 8, 3 LSTM cells and _NF_ = 4. Both MOVAD configurations surpass the SOTA models. Our best MOVAD surpasses SOTA by +2 _._ 87 AUC, reaching the highest value of 82 _._ 17% AUC. Table 3 shows results per class, grouped by ego and non-ego anomaly categories (see [15] for explanation of classes). It is worth to note that, unlike previous SOTA models which incorporate e.g. object detectors, our model also processes without any problem videos in which traffic participants are not present or visible. For this reason, the table shows also the ego-involved VO (vehicle-obstacle collision) and OO (oncoming out-of-control) classes. 

## **4. CONCLUSIONS** 

In this paper, we propose MOVAD, a new architecture for the frame-level VAD task, capable to work in an online fashion, handling the most restrictive VAD scenario with an end-toend training, requiring only RGB frames. It is composed by STMM, which extracts information related to the ongoing action, implemented by a VST, and LTMM that considers remote past, thanks to LSTM injected inside the classifier. We evaluated its performance on DoTA dataset, a collection of dash-mounted camera videos of accidents, reaching 82 _._ 17% AUC, surpassing SOTA by +2 _._ 87 AUC. 

## **5. REFERENCES** 

- [1] Arnab, A., Dehghani, M., Heigold, G., Sun, C., Luˇci´c, M., Schmid, C.: Vivit: A video vision transformer. In: Procs. of the IEEE/CVF Intl. Conf. on Computer Vision (ICCV). pp. 6836–6846 (October 2021) 2 

- [2] Chong, Y.S., Tay, Y.H.: Abnormal event detection in videos using spatiotemporal autoencoder. In: Advances in Neural Networks-ISNN 2017: 14th Intl. Symposium, ISNN 2017, Sapporo, Hakodate, and Muroran, Hokkaido, Japan, June 21–26, 2017, Procs., Part II 14. pp. 189–196. Springer (2017) 4 

- [3] Fang, J., Yan, D., Qiao, J., Xue, J.: Dada: A large-scale benchmark and model for driver attention prediction in accidental scenarios (2019) 1 

- [4] Gandhi, T., Trivedi, M.M.: Pedestrian protection systems: Issues, survey, and challenges. IEEE Tran. on Intelligent Transportation Systems **8** (3), 413–430 (2007) 1 

- [5] Goyal, R., Ebrahimi Kahou, S., Michalski, V., Materzynska, J., Westphal, S., Kim, H., Haenel, V., Fruend, I., Yianilos, P., Mueller-Freitag, M., et al.: The” something something” video database for learning and evaluating visual common sense. In: Procs. of the IEEE international Conf. on computer vision. pp. 5842–5850 (2017) 3 

   - [11] Ramachandra, B., Jones, M.J., Vatsavai, R.R.: A survey of single-scene video anomaly detection. IEEE transactions on pattern analysis and machine intelligence **44** (5), 2293–2312 (2020) 1 

   - [12] Wang, H., Schmid, C.: Action recognition with improved trajectories. In: Procs. of the IEEE international Conf. on computer vision. pp. 3551–3558 (2013) 1 

   - [13] Wang, L., Zhou, F., Li, Z., Zuo, W., Tan, H.: Abnormal event detection in videos using hybrid spatio-temporal autoencoder. In: 2018 25th IEEE Intl. Conf. on Image Processing (ICIP). pp. 2276–2280. IEEE (2018) 1 

   - [14] Xu, M., Gao, M., Chen, Y.T., Davis, L.S., Crandall, D.J.: Temporal recurrent networks for online action detection. In: Procs. of the IEEE/CVF Intl. Conf. on Computer Vision. pp. 5532–5541 (2019) 1 

   - [15] Yao, Y., Wang, X., Xu, M., Pu, Z., Wang, Y., Atkins, E., Crandall, D.: Dota: Unsupervised detection of traffic anomaly in driving videos. IEEE Tran. on Pattern Analysis and Machine Intelligence pp. 1–1 (2022) 1, 2, 3, 4 

   - [16] Zhou, Z., Dong, X., Li, Z., Yu, K., Ding, C., Yang, Y.: Spatio-Temporal Feature Encoding for Traffic Accident Detection in VANET Environment. IEEE Tran. on Intelligent Transportation Systems **23** (10), 19772–19781 (Oct 2022) 1, 4 

- [6] Hasan, M., Choi, J., Neumann, J., Roy-Chowdhury, A.K., Davis, L.S.: Learning temporal regularity in video sequences. In: Procs. of the IEEE Conf. on computer vision and pattern recognition. pp. 733–742 (2016) 1, 4 

- [7] Liu, W., Luo, W., Lian, D., Gao, S.: Future frame prediction for anomaly detection–a new baseline. In: Proc. of Conf. on Computer Vision and Pattern Recognition. pp. 6536–6545 (2018) 1, 4 

- [8] Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., Hu, H.: Video Swin Transformer. In: 2022 IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR). pp. 3192–3201. IEEE, New Orleans, LA, USA (Jun 2022) 2 

- [9] Luo, W., Liu, W., Gao, S.: Remembering history with convolutional lstm for anomaly detection. In: 2017 IEEE Intl. Conf. on Multimedia and Expo. pp. 439–444 (2017) 1 

- [10] Moutik, O., Sekkat, H., Tigani, S., Chehri, A., Saadane, R., Tchakoucht, T.A., Paul, A.: Convolutional neural networks or vision transformers: Who will win the race for action recognitions in visual data? Sensors **23** (2), 734 (2023) 2 

