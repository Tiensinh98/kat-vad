# **Autoregressive Denoising Score Matching is a Good Video Anomaly Detector** 

**Hanwen Zhang**<sup>*</sup> **, Congqi Cao**<sup>*†</sup> **, Qinyi Lv, Lingtong Min, Yanning Zhang** Northwestern Polytechnical University, Xi’an Shaanxi, 710129, China 

zhwwww@mail.nwpu.edu.cn, congqi.cao@mail.nwpu.edu.cn, _{_ lvqinyi, minlingtong, ynzhang _}_ @nwpu.edu.cn 

## **Abstract** 

_Video anomaly detection (VAD) is an important computer vision problem. Thanks to the mode coverage capabilities of generative models, the likelihood-based paradigm is catching growing interest, as it can model normal distribution and detect out-of-distribution anomalies. However, these likelihood-based methods are blind to the anomalies located in local modes near the learned distribution. To handle these “unseen” anomalies, we dive into three gaps uniquely existing in VAD regarding scene, motion and appearance. Specifically, we first build a noise-conditioned score transformer for denoising score matching. Then, we introduce a scene-dependent and motion-aware score function by embedding the scene condition of input sequences into our model and assigning motion weights based on the difference between key frames of input sequences. Next, to solve the problem of blindness in principle, we integrate unaffected visual information via a novel autoregressive denoising score matching mechanism for inference. Through autoregressively injecting intensifying Gaussian noise into the denoised data and estimating the corresponding score function, we compare the denoised data with the original data to get a difference and aggregate it with the score function for an enhanced appearance perception and accumulate the abnormal context. With all three gaps considered, we can compute a more comprehensive anomaly indicator. Experiments on three popular VAD benchmarks demonstrate the state-of-the-art performance of our method. Code is available at https://github.com/Bbeholder/ADSM._ 

## **1. Introduction** 

Video anomaly detection (VAD) is an important public safety application, which aims to accurately detect abnormal events in videos in time. Since anomalies are too scattered and rare to be sufficiently and comprehensively collected, the setting of learning from abundant normal 

> *These authors contributed equally to this work. 

> †Corresponding author. 



Figure 1. An illustration of the local modes of anomalies. We explore two regions in which the anomalies are residing. The likelihood-based method is highly sensitive to regions of anomalies with inherently low probability likelihood while blind to anomalies located in local modes near the learned distribution. 

data is more practical, necessitating the semi-supervised approach [9, 44]. 

Under this semi-supervised framework, reconstructionbased [7, 17, 18, 22, 40, 55, 60] and prediction-based [5, 6, 8, 15, 24, 28, 29, 34, 41, 46, 47] approaches are two of the most representative paradigms. They are based on the premise that anomalies are inherently harder to predict or reconstruct when compared to normal data [18, 28]. However, the underlying assumption of these paradigms can not avoid undesired generalizations on anomalies because it only considers low-level vision information in videos. 

Recent years have witnessed the great progress of generative models [20, 50, 51]. The powerful pattern coverage capability of generative models lays a new solid foundation for the VAD task: an out-of-distribution anomaly can be detected by its low likelihood under the statistical model of normal data. However, these likelihood-based methods build generative models for boosting visual pretext tasks [14, 57] or assisting density estimation models [19, 36, 38]. Directly using a likelihood-based indicator for VAD remains challenging because it has some intrinsic limitations. We consider the Stein score [27] here, which represents the gradient of log density. As shown in Fig. 1, the Stein score tends to be small at maxima, minima, and saddle points, which makes it difficult to detect anomalies that reside in local modes [36]. As a result, solely relying on the likelihood is insufficient to make a distinction. In the 

task of VAD, these local models can be attributed to three gaps between likelihood and vision: scene gap regarding scene details [6, 8], motion gap regarding motion details, and appearance gap regarding visual details. 

To fill these gaps, in this paper, we propose autoregressive denoising score matching (ADSM), which contains a novel model, a novel score function, and a novel autoregressive mechanism to take both the efficiency of likelihood estimation and the characteristic of video anomalies into account. Consider an input video sequence perturbed with various levels of Gaussian noises. We first integrate the denoising score matching into transformer-based architecture [53] to construct a noise-conditioned score transformer (NCST) for approximating the Stein score function of noisy data. Then, we embed the scene information of the input sequence into our NCST to estimate a scene-dependent score. Next, we assign motion weights to the score function based on the difference between the first and last key frames [58] of the input sequence, guiding our method to focus on the motion consistency inherent in videos. The scene gap and motion gap are filled with a scene-dependent and motionaware score function. 

Our main goal is to decrease the probability likelihood of these local modes while maintaining a high probability likelihood for normal data. Therefore, integrating the unaffected visual information to fill the appearance gap is effective. Finally, a novel autoregressive denoising score matching mechanism is developed for inference, in which we autoregressively inject intensifying Gaussian noise into the denoised data and estimate the corresponding score function. Along with each denoising process, we compare the denoised data with the original data to get a difference and aggregate it with the score function. In this way, not only is the abnormal context accumulated through the autoregressive mechanism, but the local mode is also suppressed by the visual information via an enhanced appearance perception. With all three gaps considered, we can compute a more suitable and comprehensive anomaly indicator. 

Our ADSM is the first score-based transformer for VAD. We modify the objective function to combine the denoising score matching with the characteristics of video anomalies for denoising diffusion transformers (DiTs) [43]. We are inspired to utilize the DiTs for their improved mode coverage capabilities shown in a series of generative tasks [2, 35, 39]. Extensive experiments on Avenue [30], ShanghaiTech [32], and NWPU Campus [6] datasets illustrate the superiority of our solution over other methods. To summarize, the main contributions of this paper are as follows: 

- A novel autoregressive denoising score matching mechanism for VAD, which autoregressively approximates the score function of noisy data and combines visual details based on the corresponding denoised data, building a good video anomaly detector. 

- A novel transformer-based model for denoising score matching, which is the first score-based transformer for VAD, exploiting the enhanced mode coverage capabilities of diffusion probabilistic transformers. 

- Two useful adjustment strategies making the score function scene-dependent and motion-aware, polishing a more suitable and comprehensive likelihood-based indicator for a range of noise levels. 

- A thorough validation on Avenue, ShanghaiTech, and NWPU Campus datasets, where our method achieves the state-of-the-art performance in all three datasets, fully demonstrating the effectiveness. 

## **2. Related Work** 

### **2.1. Video Anomaly Detection Methods** 

Prevalent VAD methods can be grouped into the semisupervised category, weakly-supervised category [62, 64], and fully supervised category [63]. We focus on the semisupervised methods in this paper, which mainly contains distance-based [10, 31, 56], reconstruction-based [7, 17, 18, 22, 40, 55, 60], and prediction-based [5, 6, 8, 13, 15, 24, 28, 29, 33, 34, 41, 46, 47, 49] methods. They train deep networks through pretext tasks, such as auto-encoding frames for reconstruction [17, 18, 22, 40, 60], predicting future frames [5, 6, 13, 28, 33, 34, 41, 47, 49], predicting future features [8, 24, 29], learning a dictionary [10, 31, 56], and solving masked jigsaw puzzles [15, 46, 55]. Recently, the powerful mode coverage capability of the generative model enables a novel likelihood-based approach [14, 19, 38, 57] for VAD. They build generative models to learn the distribution of normal data for boosting visual pretext tasks [14, 57] or assisting density estimation models [19, 38]. 

Despite the impressive performance, directly using a likelihood-based indicator is suffering from the “unseen” anomalies lying in the local mode. Our method is tailored to these regions through an autoregressive mechanism. 

### **2.2. Score-Based Generative Model** 

Generative models have made great advances in recent years, in which diffusion models [20] integrated with Langevin dynamics for score matching [50, 51] demonstrate remarkable advantages. The score function is defined as the gradient of the log density with respect to the input, which can be seen as a vector field pointing in the direction where the log data density grows the most. The good properties of the score function naturally serve the task of anomaly detection [8, 38, 47]. Cao _et al_ . [8] use the scorebased diffusion model to build a scene-conditioned autoencoder in latent space for scene-dependent anomaly detection and anticipation. Rodrigues _et al_ . [47] approximate the score function for a range of noise levels for capturing multiscale gradient signals. Micorek _et al_ . [38] find the log 



Figure 2. An overview of the proposed method. We implement autoregressive denoising score matching via a novel noise-conditioned score transformer (NCST), which consists of MLP, embedding layer, NCST blocks, and linear layer. When a video sequence is given, we first perturb it with noises of various levels and tokenize the noisy data into patches. Then, we feed them into the embedding layer to get patch embeddings. Next, the proposed NCST takes the patch embeddings (with position) and the overall condition _z_ (generated based on diffusion time step _i_ and scene class label _y_ ) as input to predict noise patches with the linear layer. Finally, we get anomaly scores through autoregressive denoising score matching. 

density performs better with video features at the cost of sacrificing the nonlinearity of the model. However, these methods suffer form the local modes of the score function, unable to make a distinction solely with the likelihood. 

We dive into the denoising score matching mechanism and propose the autoregressive denoising score matching in VAD, which helps accumulate abnormal contexts and combine vision and likelihood information for complementarity. 

## **3. Proposed Method** 

Fig. 2 presents an overview of the proposed method. We perform anomaly detection directly on the video clip since the extracted features with diverse distribution from pretrained models highly account for the unstable performance of VAD methods. The space of the original video data enables us to focus on the characteristics of video anomalies. 

### **3.1. Denoising Score Matching** 

### **2.3. Transformers in Diffusion Generation** 

Diffusion transformers [2, 3, 43] have recently exhibited remarkable capability in generative tasks. Notably, the diffusion transformer (DiT) [43] introduces a straightforward architecture for learning the denoising diffusion process in the latent space of Stable Diffusion [48]. U-DiT [2] leverages a Vision Transformer (ViT) [12] framework with extensive skip connections between shallow and deep layers for denoising the noisy image patches. UniDiffuser [3] devises a unified transformer tailored to diffusion models, enabling the simultaneous learning of all distributions to handle diverse input modalities effectively. 

Motivated by the promising performance of diffusion transformers, we develop a novel diffusion transformer based on the denoising score matching for VAD. Without the complex generation process, we show that the score output itself estimated by the powerful DiT is a good video anomaly detector. 

Score matching [21] is introduced to learn non-normalized statistical models, where the score is defined as the gradient of the log density _∇xlogp_ ( _x_ ) with respect to the data _x_ . Following work [54], we can perturb the data _x_ with _iid_ Gaussian noise and use score matching to directly train a score network _sθ_ (˜ _x_ ) for estimating the score of the perturbed data distribution _qσ_ (˜ _x_ ) = � _qσ_ (˜ _x|x_ ) _p_ ( _x_ ), where _σ_ is the noise level. The objective function is proved equivalent to matching the score of a non-parametric Parzen density estimator of the data: 



This approximation can be further used for generative models [50, 51], which perturb data with a sequence of intensifying Gaussian noise and jointly estimate the score functions for all noisy data distributions by training a score network _sθ_ ( _x, σ_ ) conditioned on noise levels _σi_ ( _i.e., ∀σ ∈ {σi}_<sup>_L_</sup> _i_ =1<sup>:</sup><sup>_sθ_(</sup><sup>_x, σ_)</sup><sup>_≈∇xlogqσ_(</sup><sup>_x_)).If we choose the noise</sup> 

distribution to be _N_ (˜ _x|x, σ_<sup>2</sup> _I_ ), the score function is equivalent to _∇x_ ˜ _logqσ_ (˜ _x|x_ ) = _−_<sup>_<u>x</u>_˜</sup> _σ_<sup>_−_2</sup><sup>_<u>x</u>_,whichcanbeseenasa</sup> vector field that points in the direction of denoising [50, 54]. Thus the objective function can be written as: 



where _λ_ ( _σi_ ) _>_ 0 is a coefficient function and we set _λ_ ( _σi_ ) = _σi_<sup>2following work[50].In the following subsec-</sup> tions, we build a novel model, a novel score function, and a novel autoregressive mechanism to show that the score itself is a good video anomaly detector rather than utilizing it for generative models. 

### **3.2. Noise-Conditioned Score Transformer** 

In order to implement denoising score matching in the task of VAD, we need a model that can not only coordinate the complex spatiotemporal information of videos but also consider various conditions like noise levels. Attention-based architectures [53] present an intuitive option for capturing long-range contextual relationships in videos. Moreover, DiT [43] has demonstrated the possibility of incorporating the denoising process into the attention mechanisms. Therefore, we modify the denoising score matching formulation to train a novel noise-conditioned score transformer. 

As shown in Fig. 2, let _xt_ = _{ft, · · · , ft_ + _n−_ 1 _}_ be an input sequence of _n_ frames at the time step _t_ , _{σi}_<sup>_L_</sup> _i_ =1<sup>be a</sup> geometric sequence that satisfies _σσLL−_ 1<sup>=</sup><sup>_· · ·_=</sup> _σ_<sup>_<u>σ</u>_</sup> 1<sup><u>2</u></sup><sup>_>_1.At</sup> noise level _σi_ , we perturb the input video sequence with Gaussian noise _ϵ_ to get _x_ ˜ _t_ , the distribution of which is _qσi_ (˜ _x|x_ ) = _N_ (˜ _x|x, σi_<sup>2</sup><sup>_I_).Wesamplethese</sup><sup>_n_perturbed</sup> frames _x_ ˜ _t_ and embed each individual frame into tokens using the uniform frame patch embedding method introduced in ViT [12]. Let _N_ be the number of non-overlapping patches of size _d × d × c_ from each sequence _x_ ˜ _t_ , where _c_ is the number of input channels and _d_ is a hyperparameter that determines _N_ with _n_ . Then, we get a set of tokens _{P_<sup>˜</sup> _j}_<sup>_N_</sup> _j_ =1<sup>_∈_R</sup><sup>_d×d×c_in</sup><sup>_x_˜</sup><sup>_t_and</sup><sup>_{Pj}N_</sup> _j_ =1<sup>_∈_R</sup><sup>_d×d×c_in</sup> _xt_ . The target of our noise-conditioned score transformer _sθ_ ( _P, σ_ ) is to estimate the score functions for all noisy patches with the score matching objective in Eq. (2). To that end, we change the objective to make a patch-wise denoising score matching function, in the sense of solving: 



Our noise-conditioned score transformer is implemented through the latent diffusion transformer (Latte) [35]. We 



Figure 3. A diagram of our NCST block architecture used in Fig. 2. MLP, MHA, and FFP stand for the multi-layer perception layer, the feed-forward neural network, and the multi-head attention layer, respectively. 

train the model conditioned on the diffusion time step _i_ to control the noise level _σi_ (embedded through an MLP). With attention-based architectures, the consecutive supervision of the multiscale patch-wise denoising process can effectively result in a conservative vector field, making the anomalies distinguishable. 

### **3.3. Scene-Conditioned Score Matching** 

Scene-dependent anomalies [44] are catching increasing attention as a characteristic of anomalies in the field of VAD. Reasonably considering scene information of anomalies in videos can effectively enhance the performance of VAD methods [6, 8, 52]. For each video sequence, we assign a scene class label _y_ based on the scene ( _i.e._ , the camera ID) to which it belongs. Then, we combine the scene class label _y_ with the diffusion time step _i_ to get the overall condition _z_ . We apply the scalable adaptive layer normalization (SAdaLN) mechanism [35] to embed and transmit the overall condition _z_ , which is demonstrated to propagate the condition more effectively throughout the model. As shown in Fig. 3, after getting the overall condition _z_ based on the diffusion time step _i_ and scene class label _y_ , we employ a MLP to compute _γz_<sup>1,</sup><sup>_β_</sup> _z_<sup>1,</sup><sup>_α_</sup> _z_<sup>1,</sup><sup>_γ_</sup> _z_<sup>2,</sup><sup>_β_</sup> _z_<sup>2, and</sup><sup>_α_</sup> _z_<sup>2based on</sup><sup>_z_, resulting</sup> in a linear combination as follows: 





where _h_ and _h_<sup>_′_</sup> are hidden outputs within the transformer blocks. MHA and FFP stand for the multi-layer perception layer [53] and the feed-forward neural network, respectively. With this more adaptive manner, we transmit the 

condition information to each transformer block, contributing to superior performance and faster model convergence. 

### **3.4. Key Frame Motion Weighting** 

Unlike the commonly used Fisher score _∇θlogpθ_ ( _x_ ) in statistics, the score considered here is a function of the input sequence _xt_ rather than the model parameter _θ_ . Therefore, we can improve this score based on the characteristics of the input data to achieve better performance. Since the dynamic foreground and static background of video frames are not balanced [44] in the security scenes, the score tends to reflect more static information. To this end, additionally integrating the motion information into the score function is beneficial. Based on the video codec theory [26], we propose to use the key frame difference between the first and last frame of the input sequence to assign weights to the score during denoising score matching. 

For every input sequence _xt_ , we first compute the ab= solute difference between the key frames to get _gt |ft_ + _n−_ 1 _− ft|_ . Then, we divide it into non-overlapping patches _{Gj}_<sup>_N_</sup> _j_ =1<sup>_∈_R</sup><sup>_d×d×c_.Next,we compute the maxi-</sup> mum difference along the _c_ dimension and take the average across channels for each patch, which can be written as: 



Finally, we normalize the value to compute a patch-wise weight and allocate it to our score function: 





Through implementing the motion-weighted denoising score matching process, the score output from our model is able to encapsulate more motion information. 

### **3.5. Autoregressively Anomaly Scoring** 

For inference, we consider the L2-norm of the score: 



In principle, the data density term in the denominator implies that normal events with high likelihoods yield a low norm, while anomalies with low likelihoods result in a high norm. However, the intrinsic limitations of the numerator mentioned in our motivation constrict the effectiveness of the score norm as a good anomaly indicator. To increase 



Figure 4. An overview of the proposed autoregressive denoising score matching process. 

the norm of these local modes while maintaining a small norm for normal events, we propose an autoregressive denoising score matching mechanism for multiple noise levels. As shown in Fig. 4, for each perturbed _x_ ˜ _t_ obtained by adding intensifying noise to the original data, the denoised data _x_ ˆ _t_ = _x_ ˜ _t_ + _σi_<sup>2</sup><sup>_sθ_(˜</sup><sup>_xt, σi_)isderivedthroughjointlyes-</sup> timating the score function. We improve the mechanism by autoregressively applying intensifying noise to the denoised data instead of the original data and estimating the corresponding score function. This modification allows the abnormal context of anomalies to be accumulated, resulting in a more effective score function for VAD. 

Further, since traditional approaches based on unaffected visual pretext tasks have demonstrated fruitful progress, integrating visual information is prospective for complementarity. Therefore, in each denoising process, we utilize the PSNR value between _xt_ and _x_ ˆ _t_ as the denominator of the score norm. When the likelihood and appearance information in the numerator and denominator share consistency, the judgment of normal or abnormal events is highly probable and reliable. In contrast, when they diverge, the regions where the numerator misjudges are suppressed by the denominator, while the parts overlooked by the denominator are compensated by the numerator, resulting in a more robust and balanced score with the appearance gap filled. Finally, for every input sequence _xt_ , we get _L_ scores at the time step _t_ from various noise level _σi_ : 



During inference, we divide the video into _η_ clips with _T_ frames. Then, for each noise level _σi_ , we regard the maximum score as the score of the video clip to widen the discrimination between normal and abnormal events [8, 58]. Next, we normalize each score _i_ ( _t_ ) to obtain anomaly scores _Si_ ( _t_ ) in the range [0 _,_ 1] for quantifying the probability of anomalies occurring [37], which can be written as: 

**Algorithm 1:** Autoregressively anomaly scoring **Input:** the number of noise levels _L_ ; _ϵ ←_ sample _N_ (0 _, I_ ); _σ ←_ sample log-uniform ( _σ_ 1 _, σL_ ); _n_ input frames _xt_ = _{ft, · · · , ft_ + _n−_ 1 _}_ ; the score network _sθ_ **Output:** anomaly score _Si_ ( _t_ ) _xt_ = _{ft, · · · , ft_ + _n−_ 1 _}_ ; _x_ ˙ _t_ = _xt_ ; **for** _i ∈_ [1 _, L_ ] **do** _x_ ˜ _t_ = _x_ ˙ _t_ + _σiϵ_ ; _x_ ˆ _t_ = _x_ ˜ _t_ + _σi_<sup>2</sup><sup>_sθ_(˜</sup><sup>_xt, σi_);</sup> score _i_ ( _t_ ) = _||sθ_ (˜ _xt, σi_ ) _||/PSNR_ (ˆ _xt, xt_ ); _x_ ˙ _t_ = _x_ ˆ _t_ ; **end** _Si_ ( _t_ ) = normalize(score _i_ ( _t_ )); return _Si_ <u>(</u> _t_ <u>)</u> _Si_ ( _t_ ) = normalize(score _i_ ( _t_ )) score _i_ ( _t_ ) _−_ min 1 _≤t≤ηT_<sup>score</sup><sup>_i_(</sup><sup>_t_)</sup> (11) = max min 1 _≤t≤ηT_<sup>score</sup><sup>_i_(</sup><sup>_t_)</sup><sup>_−_</sup> 1 _≤t≤ηT_<sup>score</sup><sup>_i_(</sup><sup>_t_)</sup> 

we weigh and average these anomaly scores _Si_ ( _t_ ) to obtain an ultimate anomaly indicator, which is strong enough for VAD. The whole procedure is outlined in the Algorithm 1. 

## **4. Experiment** 

### **4.1. Experimental Setup** 

**Datasets.** We evaluate the performance of our method on three public datasets widely used in the VAD community. (1) Avenue dataset [30] consists of 16 training videos and 21 testing videos. The abnormal events include running, throwing bags, child skipping, _etc_ , on the sidewalk. (2) ShanghaiTech dataset [32] contains 330 training videos and 107 testing videos. The abnormal events include affray, robbery, fighting, _etc_ , in 13 different scenes. (3) NWPU Campus dataset [6] is the largest semi-supervised VAD dataset with 305 training videos and 242 testing videos. The abnormal events include jaywalking, u-turn, _etc_ , in 43 different scenes. Notably, it is the first VAD dataset that takes the scene-dependent anomaly into account. 

**Evaluation Metric.** We use the area under the curve (AUC) of the receiver operating characteristic (ROC) to evaluate the performance of our method. There are two methods to aggregate the AUC-ROC value over multiple videos in the literature: the micro score and the macro score [1, 4, 16, 45]. For the macro score, the AUC-ROC value is computed separately for each video and then averaged across all videos 

Table 1. Comparison of different methods on the Avenue, ShanghaiTech, and NWPU Campus datasets in AUCs (%) metric. The best result on each dataset is shown in bold. We also list the inputs of the methods, in which method without marking denotes single modal input. “<sup>♢</sup> ” and “<sup>+</sup> ” represent multimodal inputs with optical flow and skeleton, respectively. 

|Method|Ave|nue|Shang|haiTech|NW|PU|
|---|---|---|---|---|---|---|
||Micro|Macro|Micro|Macro|Micro|Macro|
|♢FFP [28]<br>|84.9<br>|-|72.8|-|-|-|
|♢AMMC-Net [5] <br>|86.6|-|73.7|-|64.5|-|
|♢HF2-VAD [29]<br>|91.1|93.5|76.2|83.1|63.7|-|
|♢VABD[24]|86.6|-|78.2|-|-|-|
|+MTP [47]|82.9|-|76.0|-|-|-|
|+STG-NF [19]|-|-|**85.9**|-|-|-|
|+MoCoDAD[14]|89.0|-|77.9|-|-|-|
|MemAE [17]|84.9|-|72.8|-|61.9|-|
|CAE [22]|87.4|90.4|78.7|84.9|-|-|
|MNAD [41]|88.5|-|70.5|-|62.5|-|
|OG-Net [60]|-|-|-|-|62.5|-|
|MPN [34]|89.5|-|73.8|-|64.4|-|
|SSMTL [15]|91.5|91.9|82.4|89.3|-|-|
|LLSH [31]|87.4|88.6|77.6|85.9|62.2|-|
|FBAE [6]|86.8|-|79.2|-|68.2|-|
|FPDM [57]|90.1|-|78.6|-|-|-|
|Two-stream [7]|90.8|93.0|83.7|90.8|-|-|
|SDMAE [46]|91.3|90.9|79.1|84.7|-|-|
|SSAE[8]|90.2|91.5|80.5|88.9|75.6|76.9|
|ADSM(ours)|**91.6**|**94.2**|84.5|**93.2**|**76.9**|**78.1**|



in the test set. The micro score computes the AUC-ROC jointly for all frames of all test videos. We report the results in terms of both metrics. 

**Implementation Details.** The input frames of our model are the regions of 160 _×_ 160 pixels centered on objects that are detected by the pretrained ByteTrack [61] implemented by MMTracking [11]. We expect our model to not only focus on target information but also consider some scene information on its own. If a given frame contains multiple targets, we will process each target separately. Our proposed model _sθ_ receives an input sequence with 8 frames at a time and tokenizes the input sequence into _N_ = 8 _×_ 100 patches _{Pj}_<sup>_N_</sup> _j_ =1<sup>_∈_R</sup><sup>_d×d×c_,wherethepatchsize</sup><sup>_d_is16.Each</sup> model is trained for 100 epochs. The batch size is set to 20 and Adamax [23] is chosen as the optimizer. The initial learning rate is set to 0.0001 and gradually decays following the scheme of cosine annealing. During training, we sample the noise scale used for each batch element from the loguniform distribution on the interval [ _σ_ 1 _, σL_ ] = [0 _._ 001 _,_ 1 _._ 0]. For evaluation, we use _L_ = 20 evenly spaced noise levels between [ _σ_ 1 _, σL_ ]. The scene class label embedding and the diffusion time embedding, both with dimensions of 768, are concatenated for model learning and inference. All the 

experiments are performed on four NVIDIA GeForce RTX 4090 GPUs with pytorch [42] framework. 

### **4.2. VAD Performance Benchmarking** 

The performance comparison between our method and other existing methods in terms of the micro and macro scores on the Avenue, ShanghaiTech, and NWPU Campus datasets is present in Tab. 1. We prefer the macro score, especially in datasets with multiple scenes like ShanghaiTech and NWPU Campus datasets, which can not be comprehensively and fairly judged by the micro score due to the fuzzy boundaries of scenes. Our method outperforms the others on the three datasets in terms of the macro score, all of which contain over 10 classes of abnormal events. Especially in the ShanghaiTech dataset, we achieve an obvious improvement (up to 2 _._ 4%), demonstrating the effectiveness of our method. In the recently released NWPU Campus dataset, we achieve the best performance on both metrics, verifying the scalability and stability of our method. Compared with the works that use multimodal inputs, we only train our model on raw video clips and achieve comparable or even better performance. Compared with the work using the same single type of input, we achieve the best performance on the three datasets in terms of both the micro and the macro scores. The superior performance proves that our proposed autoregressive denoising score matching is a good anomaly detector for the challenging and complex VAD task. Notably, since the score is computed from random noise, it inherently carries some uncertainty. However, the qualitative analysis in Sec. 3.5 confirms the validity of the denoising score matching mechanism, and we have maximally controlled this uncertainty through conditional embedding and the integration of visual information. Furthermore, numerous testing shows that the results exhibit minimal fluctuation (less than 1%), ensuring that the reported outcomes are stable and reliable. 

### **4.3. Study on Latent Space** 

Generative models in latent space have been widely adopted to compress redundant information of the original data and improve computational efficiency to generate high-fidelity images and videos [2, 3, 35, 48]. Unlike traditional generative tasks, the task of VAD is more sensitive to the distribution of input data, which means less redundant information than usual. Under this circumstance, the VAD performance may be limited by a reduced low-dimensional latent space that is unable to preserve essential information of inputs. Moreover, our motivation and experiments also demonstrate the importance of considering the characteristics of original video data. A good VAD performance may be attributed to a good feature extractor rather than a suitable likelihood estimation strategy, which forces us to reconsider the necessity and feasibility of implementing the denoising score 

Table 2. AUCs (%) of our model in latent space and original data space on ShanghaiTech and NWPU Campus datasets. 

|Method|Shangh|aiTech|NW|PU|
|---|---|---|---|---|
||Micro|Macro|Micro|Macro|
|L-ADSM|77.6|87.5|68.1|73.2|
|ADSM|**84.5**|**93.2**|**76.9**|**78.1**|



Table 3. AUCs (%) of different likelihood-based methods in the original data space on ShanghaiTech and NWPU Campus datasets. 

|Method<br>ShanghaiTech<br>NWPU|
|---|
|Micro<br>Macro<br>Micro<br>Macro|
|MSMA [36]<br>81.3<br>87.5<br>72.3<br>75.1|
|MULDE [38]<br>82.6<br>90.4<br>73.1<br>74.5|
|Ours<br>**84.5**<br>**93.2**<br>**76.9**<br>**78.1**|
|matching in the latent space of the pre-trained model.<br>Here, we propose an analysis that assesses the perfor-<br>mance of our method in the latent space. We use the preva-|
|lent variational auto-encoder of Stable Diffusion [48] pre-|
|trained on the CoCo dataset [25] to produce a latent rep-<br>resentation of the uncorrupted input sequence and we per-|
|form our ADSM in this latent space with the NCST as|
|described in Fig. 2.<br>Tab. 2 illustrates the performance<br>in terms of the micro and macro scores on ShanghaiTech<br>and NWPU Campus datasets.<br>We notice that the latent<br>version underperforms the original version, which demon-<br>strates our suspicion that the inconsistent distribution be-<br>tween the features extracted by the pre-trained model and|
|the VAD dataset exacerbates the unstable performance of|
|the latent version. The representation alignment may relieve<br>this problem [59]. Directly implementing our ADSM on the<br>original data space is more intuitive with the advancement<br>of computing resources and model capabilities.|



Since the performance of the VAD method is unstable due to the inconsistent distribution of features extracted by pretrained models. For two likelihood-based baselines related to our method, one of which is implemented in latent space [38] and the other [36] is applied to image anomaly detection. We reproduced them under our settings for a fair comparison on ShanghaiTech and NWPU Campus datasets. As shown in Tab. 3, our method outperforms other related methods, demonstrating that applying denoising score matching in the original data space is more effective. The original data space enables us to take both the efficiency of likelihood estimation and the characteristic of video anomalies into account, supporting our motivation. 

### **4.4. Ablation Study and Analysis** 

In Tab. 4, we show the performance of our model variants on the ShanghaiTech dataset. We sequentially add denoising score matching (DSM), autoregressive denoising score 

Table 4. AUCs(%) of our model variants on the ShanghaiTech dataset. 

||DSM|ADSM|Scene|MotionAearance|Shang|haiTech|
|---|---|---|---|---|---|---|
|||||pp|Micro|Macro|
|1|✓||||73.9|78.5|
|2|✓|✓|||76.4|84.1|
|3|✓|✓|✓||76.9|86.3|
|4|✓|✓||✓|78.6|87.7|
|5|✓|✓||✓|82.1|89.9|
|6|✓|✓|✓|✓<br>✓|**84.5**|**93.2**|



matching (ADSM), scene condition embedding (Scene), key frame motion weighting (Motion), and visual difference aggregating (Appearance) to our noise-conditioned score transformer. As shown in Tab. 4, the autoregressive mechanism benefits the denoising score matching for VAD, as there is an average improvement of 4 _._ 1% in terms of the micro and macro scores. Although the ShanghaiTech dataset does not especially consider the scene-dependent anomalies, there is still an improvement of 2 _._ 2% in terms of the macro score, which indicates that our model is equipped with an enhanced perception of scene information. Taking motion consistency into account also proves useful with an average improvement of 2 _._ 9%, demonstrating that our method reasonably correlates the spatiotemporal information of videos for VAD. Aggregating the difference between the denoised data and the original data helps the most in computing a more distinguishable anomaly indicator based on the score function, resulting in a significant improvement of 5 _._ 7% (5 _._ 8%) in terms of the micro (macro) score. The result demonstrates that aggregating unaffected visual information is complementary. There is an additional improvement of 2 _._ 4% (3 _._ 3%) in terms of the micro (macro) score when we integrate these model variants, indicating that each variant of our model is complementary to each other. 

### **4.5. Running Speed and Model Size** 

We perform all the experiments on four NVIDIA GeForce RTX 4090 GPUs with the pytorch framework. Our method is implemented purely on the raw video clip without an extra feature extractor. Furthermore, we utilize the powerful generative model only for a score rather than high-fidelity images or videos, which greatly simplifies the denoising process. This represents a very small computational cost. The running time of our ADSM is primarily determined by the size of our noise-conditioned score transformer. In our setup described in Sec. 4.1, our ADSM uses an NCST with 130M parameters and takes less than 20 milliseconds to process an input sequence of 8 frames, which enables video anomaly detection at 50 FPS. Even considering the additional consumption of the object detection model [11, 61], our method can still maintain a speed around 35 FPS, suit- 



Figure 5. A diagram of visualization on NWPU Campus dataset. Best viewed in color. 

able for a real-time monitoring system typically around 30 FPS. A more detailed description and comparison are presented in the supplementary material. 

### **4.6. Qualitative Results** 

We visualize the score curves of our method on the NWPU Campus dataset for qualitative studies. We select three videos that contain scene-dependent anomalies (the cyclist on the square in the “D013 ~~0~~ 3”), motion anomalies (the climber in the “D001 ~~0~~ 3”), and appearance anomalies (the puppies in the “D003 ~~0~~ 5”), respectively. As shown in Fig. 5, concerning the scene-dependent cyclist anomaly that is abnormal on the square while normal on the road, our method makes a timely detection without false alarms. As for the abnormal motion of the climber and the abnormal appearance of the puppies, the anomaly score output by our method exhibits a sharp increase upon the occurrence of abnormal events and returns to flat when the abnormal events disappear. These results show that our method captures the characteristics of video anomalies in terms of scene, motion, and appearance. More visualization examples and analyses are provided in the supplementary material. 

## **5. Conclusion** 

In this paper, we propose autoregressive denoising score matching (ADSM), which contains a novel model, a novel score function, and a novel autoregressive mechanism to not only utilize the efficiency of likelihood estimation but also take characteristics of video anomalies into account. We first design a novel noise-conditioned score transformer (NCST), which is the first score-based transformer for VAD, combining the efficiency of denoising score matching with the powerful diffusion probabilistic transformers. Next, we embed the scene condition and assign motion weights based on inputs to form a scene-dependent and motionaware score function. Last but not least, we propose a novel mechanism to autoregressively approximate the score function and combine visual details based on the corresponding denoised data for anomaly accumulation and vision enhancement. Extensive experiments on three datasets verify the effectiveness of our method. 

## **6. Acknowledgments** 

This work is supported by National Natural Science Foundation of China (No.62376217, 62301434), Young Elite Scientists Sponsorship Program by CAST (2023QNRC001), Key R&D Project of Shaanxi Province (No. 2023-YBGY-240), and Young Talent Fund of Association for Science and Technology in Shaanxi, China (No. 20220117) 

## **References** 

- [1] Andra Acsintoae, Andrei Florescu, Mariana-Iuliana Georgescu, Tudor Mare, Paul Sumedrea, Radu Tudor Ionescu, Fahad Shahbaz Khan, and Mubarak Shah. Ub- 

   - normal: New benchmark for supervised open-set video anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 20143–20153, 2022. 6 

- [2] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 22669–22679, 2023. 2, 3, 7 

- [3] Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. One transformer fits all distributions in multi-modal diffusion at scale. In _International Conference on Machine Learning_ , pages 1692–1717. PMLR, 2023. 3, 7 

- [4] Antonio Barbalau, Radu Tudor Ionescu, Mariana-Iuliana Georgescu, Jacob Dueholm, Bharathkumar Ramachandra, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B Moeslund, and Mubarak Shah. Ssmtl++: Revisiting selfsupervised multi-task learning for video anomaly detection. _Computer Vision and Image Understanding_ , 229:103656, 2023. 6 

- [5] Ruichu Cai, Hao Zhang, Wen Liu, Shenghua Gao, and Zhifeng Hao. Appearance-motion memory consistency network for video anomaly detection. In _Proceedings of the AAAI conference on artificial intelligence_ , pages 938–946, 2021. 1, 2, 6 

- [6] Congqi Cao, Yue Lu, Peng Wang, and Yanning Zhang. A new comprehensive benchmark for semi-supervised video anomaly detection and anticipation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 20392–20401, 2023. 1, 2, 4, 6 

- [7] Congqi Cao, Yue Lu, and Yanning Zhang. Context recovery and knowledge retrieval: A novel two-stream framework for video anomaly detection. _IEEE Transactions on Image Processing_ , 2024. 1, 2, 6 

- [8] Congqi Cao, Hanwen Zhang, Yue Lu, Peng Wang, and Yanning Zhang. Scene-dependent prediction in latent space for video anomaly detection and anticipation. _IEEE transactions on pattern analysis and machine intelligence_ , 2024. 1, 2, 4, 5, 6 

- [9] Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. _ACM computing surveys (CSUR)_ , 41(3):1–58, 2009. 1 

- [10] Yunpeng Chang, Zhigang Tu, Wei Xie, and Junsong Yuan. Clustering driven deep autoencoder for video anomaly detection. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16_ , pages 329–345. Springer, 2020. 2 

- [11] MMTracking Contributors. Mmtracking: Openmmlab video perception toolbox and benchmark. In _https://github.com/open-mmlab/mmtracking_ , 2020. 6, 8 

- [12] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 3, 4 

- [13] Zhiwen Fang, Jiafei Liang, Joey Tianyi Zhou, Yang Xiao, and Feng Yang. Anomaly detection with bidirectional consistency in videos. _IEEE transactions on neural networks and learning systems_ , 33(3):1079–1092, 2020. 2 

- [14] Alessandro Flaborea, Luca Collorone, Guido Maria D’Amely Di Melendugno, Stefano D’Arrigo, Bardh Prenkaj, and Fabio Galasso. Multimodal motion conditioned diffusion model for skeleton-based video anomaly detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 10318–10329, 2023. 1, 2, 6 

- [15] Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, and Mubarak Shah. Anomaly detection in video via selfsupervised and multi-task learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 12742–12752, 2021. 1, 2, 6 

- [16] Mariana Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, and Mubarak Shah. A background-agnostic framework with adversarial training for abnormal event detection in video. _IEEE transactions on pattern analysis and machine intelligence_ , 44(9):4505–4523, 2021. 6 

- [17] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1705–1714, 2019. 1, 2, 6 

- [18] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 733–742, 2016. 1, 2 

- [19] Or Hirschorn and Shai Avidan. Normalizing flows for human pose anomaly detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 13545– 13554, 2023. 1, 2, 6 

- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in neural information processing systems_ , 33:6840–6851, 2020. 1, 2 

- [21] Aapo Hyv¨arinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. _Journal of Machine Learning Research_ , 6(4), 2005. 3 

- [22] Radu Tudor Ionescu, Fahad Shahbaz Khan, Mariana-Iuliana Georgescu, and Ling Shao. Object-centric auto-encoders and 

dummy anomalies for abnormal event detection in video. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 7842–7851, 2019. 1, 2, 6 

- [23] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 6 

- [24] Jing Li, Qingwang Huang, Yingjun Du, Xiantong Zhen, Shengyong Chen, and Ling Shao. Variational abnormal behavior detection with motion consistency. _IEEE Transactions on Image Processing_ , 31:275–286, 2021. 1, 2, 6 

- [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pages 740–755. Springer, 2014. 7 

- [26] Guozhu Liu and Junming Zhao. Key frame extraction from mpeg video stream. In _2010 Third International Symposium on Information Processing_ , pages 423–427. IEEE, 2010. 5 

- [27] Qiang Liu, Jason Lee, and Michael Jordan. A kernelized stein discrepancy for goodness-of-fit tests. In _International conference on machine learning_ , pages 276–284. PMLR, 2016. 1 

- [28] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6536–6545, 2018. 1, 2, 6 

- [29] Zhian Liu, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 13588–13597, 2021. 1, 2, 6 

- [30] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In _Proceedings of the IEEE international conference on computer vision_ , pages 2720–2727, 2013. 2, 6 

- [31] Yue Lu, Congqi Cao, Yifan Zhang, and Yanning Zhang. Learnable locality-sensitive hashing for video anomaly detection. _IEEE Transactions on Circuits and Systems for Video Technology_ , 33(2):963–976, 2022. 2, 6 

- [32] Weixin Luo, Wen Liu, and Shenghua Gao. A revisit of sparse coding based anomaly detection in stacked rnn framework. In _Proceedings of the IEEE international conference on computer vision_ , pages 341–349, 2017. 2, 6 

- [33] Weixin Luo, Wen Liu, Dongze Lian, and Shenghua Gao. Future frame prediction network for video anomaly detection. _IEEE transactions on pattern analysis and machine intelligence_ , 44(11):7505–7520, 2021. 2 

- [34] Hui Lv, Chen Chen, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang. Learning normal dynamics in videos with meta prototype network. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 15425–15434, 2021. 1, 2, 6 

- [35] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: 

Latent diffusion transformer for video generation. _arXiv preprint arXiv:2401.03048_ , 2024. 2, 4, 7 

- [36] Ahsan Mahmood, Junier Oliva, and Martin Styner. Multiscale score matching for out-of-distribution detection. _arXiv preprint arXiv:2010.13132_ , 2020. 1, 7 

- [37] Michael Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. _arXiv preprint arXiv:1511.05440_ , 2015. 5 

- [38] Jakub Micorek, Horst Possegger, Dominik Narnhofer, Horst Bischof, and Mateusz Kozinski. Mulde: Multiscale logdensity estimation via denoising score matching for video anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 18868–18877, 2024. 1, 2, 7 

- [39] Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, and Zhenguo Li. Dit-3d: Exploring plain diffusion transformers for 3d shape generation. _Advances in neural information processing systems_ , 36:67960–67971, 2023. 2 

- [40] Trong-Nguyen Nguyen and Jean Meunier. Anomaly detection in video sequence with appearance-motion correspondence. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1273–1283, 2019. 1, 2 

- [41] Hyunjong Park, Jongyoun Noh, and Bumsub Ham. Learning memory-guided normality for anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14372–14381, 2020. 1, 2, 6 

- [42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in Neural Information Processing Systems_ , 32, 2019. 

   - 7 

- [43] William Peebles and Saining Xie. Scalable diffusion models with transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4195–4205, 2023. 2, 3, 4 

- [44] Bharathkumar Ramachandra, Michael J Jones, and Ranga Raju Vatsavai. A survey of single-scene video anomaly detection. _IEEE transactions on pattern analysis and machine intelligence_ , 44(5):2293–2312, 2020. 1, 4, 5 

- [45] Tal Reiss and Yedid Hoshen. Attribute-based representations for accurate and interpretable video anomaly detection. _arXiv preprint arXiv:2212.00789_ , 2022. 6 

- [46] Nicolae-C Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah, et al. Self-distilled masked auto-encoders are efficient video anomaly detectors. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 15984–15995, 2024. 1, 2, 6 

- [47] Royston Rodrigues, Neha Bhargava, Rajbabu Velmurugan, and Subhasis Chaudhuri. Multi-timescale trajectory prediction for abnormal human activity detection. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2626–2634, 2020. 1, 2, 6 

- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image 

synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10684–10695, 2022. 3, 7 

- [49] Hao Song, Che Sun, Xinxiao Wu, Mei Chen, and Yunde Jia. Learning normal patterns via adversarial attention-based autoencoder for abnormal event detection in videos. _IEEE Transactions on Multimedia_ , 22(8):2138–2148, 2019. 2 

- [50] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. _Advances in neural information processing systems_ , 32, 2019. 1, 2, 3, 4 

- [51] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. _arXiv preprint arXiv:2011.13456_ , 2020. 1, 2, 3 

   - [62] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1237–1246, 2019. 2 

   - [63] Shifu Zhou, Wei Shen, Dan Zeng, Mei Fang, Yuanwang Wei, and Zhijiang Zhang. Spatial–temporal convolutional neural networks for anomaly detection and localization in crowded scenes. _Signal Processing: Image Communication_ , 47:358– 368, 2016. 2 

   - [64] Yi Zhu and Shawn Newsam. Motion-aware feature for improved video anomaly detection. _arXiv preprint arXiv:1907.10211_ , 2019. 2 

- [52] Shengyang Sun and Xiaojin Gong. Hierarchical semantic contrast for scene-aware video anomaly detection. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , pages 22846–22856, 2023. 4 

- [53] A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017. 2, 4 

- [54] Pascal Vincent. A connection between score matching and denoising autoencoders. _Neural computation_ , 23(7):1661– 1674, 2011. 3, 4 

- [55] Guodong Wang, Yunhong Wang, Jie Qin, Dongming Zhang, Xiuguo Bao, and Di Huang. Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles. In _European Conference on Computer Vision_ , pages 494–511. Springer, 2022. 1, 2 

- [56] Peng Wu, Jing Liu, and Fang Shen. A deep one-class neural network for anomalous event detection in complex scenes. _IEEE transactions on neural networks and learning systems_ , 31(7):2609–2622, 2019. 2 

- [57] Cheng Yan, Shiyu Zhang, Yang Liu, Guansong Pang, and Wenjun Wang. Feature prediction diffusion model for video anomaly detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 5527–5537, 2023. 1, 2, 6 

- [58] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on keyframes for video anomaly detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14592–14601, 2023. 2, 5 

- [59] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. _arXiv preprint arXiv:2410.06940_ , 2024. 7 

- [60] Muhammad Zaigham Zaheer, Jin-ha Lee, Marcella Astrid, and Seung-Ik Lee. Old is gold: Redefining the adversarially learned one-class classifier training paradigm. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14183–14193, 2020. 1, 2, 6 

- [61] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In _European conference on computer vision_ , pages 1–21. Springer, 2022. 6, 8 

