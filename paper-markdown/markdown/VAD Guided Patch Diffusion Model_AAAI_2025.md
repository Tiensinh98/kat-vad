The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25) 

# **Video Anomaly Detection with Motion and Appearance Guided Patch Diffusion Model** 

## **Hang Zhou, Jiale Cai, Yuteng Ye, Yonghui Feng, Chenxing Gao, Junqing Yu, Zikai Song**<sup>*</sup> **, Wei Yang** 

Huazhong University of Science and Technology, Wuhan, China _{_ henrryzh, jaile ~~c~~ ai, yuteng ~~y~~ e, yhfeng, cxg, yjqing, skyesong, weiyangcs _}_ @hust.edu.cn 

#### **Abstract** 

A recent endeavor in one class of video anomaly detection is to leverage diffusion models and posit the task as a generation problem, where the diffusion model is trained to recover normal patterns exclusively, thus reporting abnormal patterns as outliers. Yet, existing attempts neglect the various formations of anomaly and predict normal samples at the feature level regardless that abnormal objects in surveillance videos are often relatively small. To address this, a novel patch-based diffusion model is proposed, specifically engineered to capture fine-grained local information. We further observe that anomalies in videos manifest themselves as deviations in both appearance and motion. Therefore, we argue that a comprehensive solution must consider both of these aspects simultaneously to achieve accurate frame prediction. To address this, we introduce innovative motion and appearance conditions that are seamlessly integrated into our patch diffusion model. These conditions are designed to guide the model in generating coherent and contextually appropriate predictions for both semantic content and motion relations. Experimental results in four challenging video anomaly detection datasets empirically substantiate the efficacy of our proposed approach, demonstrating that it consistently outperforms most existing methods in detecting abnormal behaviors. 

## **1 Introduction** 

Video anomaly detection (VAD) remains a challenging problem within the security monitoring domain (Lu, Shi, and Jia 2013; Zhou, Yu, and Yang 2023; Wu et al. 2020). Due to the scarcity of anomalous data and restrictions on data collection, training predominantly employs videos depicting normal behavior. To address this imbalance, prevailing strategies have focused on frame prediction and reconstruction methods (Luo, Liu, and Gao 2017; Zhong et al. 2022; Liu et al. 2018; Song et al. 2023). During testing, anomalous frames are identified based on significant discrepancies between predicted and actual frames. In this paper, we adopt the frame prediction approach to generate future video frames (Lee et al. 2021). 

The autoencoder (AE) model is the most widely used technique for VAD (Gong et al. 2019; Chang et al. 2020). 

*indicates corresponding author. 

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 



<!-- Start of picture text -->
Appearance<br>Motion<br>…<br>Continuous Frames<br>…<br>Patch Noise<br>Random  to Clean Frame<br>Corp Frame Noise<br>DDIM<br>…<br>…<br><!-- End of picture text -->

Figure 1: We propose a patch-based diffusion model with a motion and appearance conditions framework for VAD. The conditional frames, which contain motion of temporal difference and appearance information, are cropped into patches and used to estimate noise. These noises are then combined and refined into the frame-level noise. The DDIM reverse step is then applied to recover the clean image. 

However, its remarkable generalization capability can inadvertently lead to the reconstruction of anomalous behavior. To mitigate this overgeneralization, several methods, such as MNAD (Stephen and Menon 2020) and MPN (Lv et al. 2021), introduce learnable and dynamic memory blocks to store various normal patterns. Nevertheless, the placement of the memory unit within the bottleneck of AE, along with the use of skip connections, still permits the reconstruction of abnormal patterns. Alternative approaches, such as AMMC-Net (Cai et al. 2021), aim to diminish the capacity of the model by integrating additional motion information. This method employs a two-stream memory augmentation network, merging motion and appearance prior knowledge to enhance multi-view features for regular events. However, its performance in anomaly detection is re- 

10761 

stricted due to the powerful autoencoder with U-Net architecture (Ronneberger, Fischer, and Brox 2015). An attempt has been made to use the diffusion model to generate video frames without the need for capacity reduction (unlike AE) (Ho, Jain, and Abbeel 2020; Nichol and Dhariwal 2021). FPDM (Yan et al. 2023) propose a feature reconstruction method with a two-stage diffusion model in the latent space. However, it is not able to capture the motion and appearance information of minor irregularities with coarse image features. 

In this paper, we propose a novel Patch-based Diffusion Model with Motion and Appearance Conditions (MAPDM), which is shown in Figure 1. We decompose video frames into separate appearance and motion components. The initial frame encapsulates all visual aspects of appearance, thereby elevating the challenge of video frame prediction, whereas successive motion frames chronicle the temporal dynamics. Consequently, our model distinctively exploits the intrinsic appearance and motion information present in videos as organic conditions, facilitating the generation of frames with enhanced precision and control. To enhance content understanding, we introduce an appearance encoder equipped with a memory block that aids the diffusion model in comprehending appearance information. Drawing inspiration from the Condition Block of ControlNet (Zhang and Agrawala 2023), we use a residual connection to integrate the appearance information with the noise estimation network. Moreover, we propose a prior motion strategy to guide diffusion models, utilizing temporal differences to gain a finer granularity of motion discrepancies. 

In particular, as anomalous objects or behaviors in surveillance videos tend to be small and localized, a comprehensive image sequence analysis with a diffusion model may be suboptimal. To address this, we adopt patch-based techniques, which have proven effective in image anomaly detection to achieve precise localizations (Roth et al. 2022; Song et al. 2022). Following the patch partition approach ( Ozdenizci<sup>¨</sup> and Legenstein 2023), we develop a patch-based diffusion model for precise detection and propose a fusion strategy for the conditions mentioned above. 

In summary, our primary contributions are as follows: 

- We innovate a patch-based condition diffusion framework designed to precisely capture local anomalous activity and ensure the normal area. 

- We introduce a motion and appearance conditioning strategy for the diffusion model to predict normal frames with controllable motion and appearance. 

- We propose a novel fusion method for both conditions, enabling the model to learn and differentiate normal appearance and motion patterns. In this context, the semantic information generated by the patch memory bank is strategically embedded to ensure a significant separation between normal and abnormal segments. 

## **2 Related Work** 

### **2.1 Video Anomaly Detection** 

Due to data collection limitations, existing approaches learn normal patterns to distinguish abnormal behavior that vio- 

lates normality (Ye et al. 2019). These methods can be categorized into two groups: frame-based and object-centerbased methods. 

**Frame-based VAD.** Frame-based methodologies commonly utilize an autoencoder (AE) structure to forecast upcoming frames by leveraging previous ones. For instance, a 3D-AE architecture is presented to capture local spatial and temporal receptive fields. AMC integrates AE with U-Net to produce future frames and optical flow, thereby improving network representation. Researchers have explored the remarkable generalization capabilities of AE models, enabling the accurate prediction of abnormal frames. To enhance the variety of normal patterns, MNAD introduces a more compact and sparse memory module. To optimize storage efficiency and improve relevant features, DLAN-AC introduces a dynamic local aggregation network with an adaptive cluster. While memory-based strategies effectively maintain normality, they are limited by the AE model’s inclination to oversimplify. Recently, diffusion models have emerged with robust and adjustable generation capabilities. FPDM, a feature prediction approach based on a two-stage diffusion model, is suggested to reduce noise in future frame features (Yan et al. 2023). The feature space is coarse for accurately reconstructing future frames, making it less suitable for identifying anomalous regions. Combining motion and appearance details is considered advantageous for diffusion models in predicting normal frames. 

**Object-centered VAD.** Object-oriented techniques initially identify the object of interest in the foreground, then apply the same prediction or reconstruction techniques, significantly enhancing the robustness of VAD. For example, recent works (Yu et al. 2020; Chen et al. 2022) forecast patches related to video events based on detected objects, thus avoiding background interference. However, network performance is restricted by the single prediction or reconstruction task. To improve the embedding performance of the U-Net model, HF2VAD (Liu et al. 2021) combines prediction and reconstruction tasks to incorporate a high correlation between RGB and the optical flow of objects. Furthermore, HSNBM (Bao et al. 2022) argues that real-world objects are associated with the entire background and introduces a hierarchical scene normality-binding model to enhance frame prediction performance. In summary, patchbased methods mainly concentrate on detected objects, but can be time-consuming when there are many objects. We incorporate the image patch way into our diffusion model to focus on local areas and use sliding windows for the testing phase to reconstruct frames. 

### **2.2 Diffusion Model** 

Recently, there has been a growing interest in diffusion models in the field of generative models (Ye et al. 2024; Nichol and Dhariwal 2021; Rombach et al. 2022), which have demonstrated high effectiveness in image synthesis. These models are built on a Markov chain that gradually learns the data distribution by undergoing a forward diffusion process and subsequently learns the reverse diffusion process to reconstruct it. Conditional generative models based on diffusion have achieved impressive results in 

10762 



<!-- Start of picture text -->
Appearance Encoder Network<br> a<br>Image Size : 256<br> a Patch Memory Patch Size :  p =64<br>Slide Step :  r =32 Prediction:<br>… Overlap   0 N  1<br>R Norm<br>1 2<br>Patch Crop<br>(Random) Decoder Semantic 4 Reverse<br>Step<br>Embedding ψ t N  1 Noise<br>Map<br>MA-PDM<br>…  m , f tN  1  NoisePatch   a  m f tN  1  Crop(  tN  1,slide)<br> m  tN  1 Noise Estimation Network<br>MA-PDM(Training Phase) Testing Phase<br><!-- End of picture text -->

Figure 2: Our MA-PDM comprises three components: a patch cropping module for creating patch conditions and noise, an appearance encoder for embedding and retaining the regular pattern, and a noise estimation network for forecasting the noise. During the training stage, the MA-PDM is trained to predict forward noise. During the inference stage, the MA-PDM anticipates the patch noise using conditions and then combines them in a reverse process. 

various tasks, such as synthesizing class-conditional data with the guidance of classifiers (Dhariwal and Nichol 2021), enhancing image resolution (Saharia et al. 2022b), deblurring images ( Ozdenizci and Legenstein 2023; Lugmayr et al.<sup>¨</sup> 2022), generating and editing images based on text descriptions (Rombach et al. 2022; Saharia et al. 2022a; Couairon et al. 2022; Mokady et al. 2023), and performing general image-to-image translation tasks (Preechakul et al. 2022; Lu et al. 2023). These investigations indicate the importance of conditions for the diffusion model to produce high-quality images. Moreover, diffusion models are also being explored for anomaly detection. In the medical field, (Wolleb et al. 2022) introduce the diffusion model for detecting medical anomalies with the guidance of classifiers. (Zhang et al. 2023) propose a one-step diffusion denoising process during inference, achieving excellent performance in industry anomaly detection. Unlike detecting anomalies in images, video anomaly detection focuses on identifying irregular movements or objects. However, due to background noise and small targets, the diffusion model may lose its effectiveness in generating realistic outputs without well-designed conditions. In this study, we introduce a patch-based diffusion model with prior conditions, specifically motion and appearance cues, to address both aspects in videos. 

## **3 Method** 

We introduce a new method for VAD, termed the Motion and Appearance Conditions Patch-based Diffusion Model (MA-PDM). Unlike previous methods that rely on an AE model for frame generation, our approach utilizes a diffusion 

model for forecasting future frames. Illustrated in Figure 2, the MA-PDM comprises three key elements: patch cropping, appearance encoder, and noise estimation network. Our study is structured into training and testing stages. In the subsequent sections, we will initially discuss the problem setup, followed by an explanation of the three aforementioned modules, and conclude with an overview of the training and testing procedures. 

### **3.1 Problem Formation** 

Following previous work, we regard the VAD task as a future frame prediction task. In the training phase, we solely use normal video data and leverage the preceding frames _{F_<sup>1</sup> _, . . . , F_<sup>_N_</sup> _}_ to forecast the subsequent frame _F_<sup>_N_+1</sup> . During the testing stage, we employ the diffusion model to recover the forthcoming frame with Gaussian noise and reconstruct them from patch size to their original resolution. Subsequently, we will elaborate on the specifics of our diffusion model with the prior constraints of videos. 

### **3.2 Diffusion Model with Motion and Appearance Conditions** 

Diffusion models propose a noise estimation function _ϵθ_ ( _xt, t_ ) that takes a noisy image _xt_ in a forward process and then predicts its noise. The model is trained by an MSE loss _||ϵθ_ ( _xt, t_ ) _− ϵ||_ 2<sup>2, where</sup><sup>_ϵ_is the Gaussian noise added to</sup> the origin image _x_ 0 to obtain _xt_ . The same as these diffusion models, we define a Gaussian diffusion process at time step _t_ = 1 _, . . . , T_ and add the noise to our target frame _F_<sup>_N_+1</sup> as: _q_ ( _Ft_<sup>_N_+1</sup> _|Ft_<sup>_N_</sup> _−_<sup>+1</sup> 1<sup>) =</sup><sup>_N_(</sup><sup>_F_</sup> _t_<sup>_N_+1</sup> ; ~~�~~ 1 _− βtFt_<sup>_N_</sup> _−_<sup>+1</sup> 1<sup>_, βt_</sup><sup>**I**)</sup><sup>_,_</sup> (1) 

10763 

where _βt ∈_ (0 _,_ 1) is the variance schedule of each time step. To acquire the data distribution _q_ ( _x_ 0) at time _t_ , we can adapt the Eq.1 as: 



where _αt_ =<sup>�</sup><sup>_t_</sup> _s_ =1<sup>(1</sup><sup>_−βs_).Thenoisesteptransformsour</sup> future frame at any noise step into a simple Gaussian distribution. 

Conditions play a crucial role in the generation of images. The primary characteristics of video frames are motion and appearance, and we leverage both prior knowledge and diffusion conditions to steer the creation of normal patterns. Video frames serve as the fundamental appearance keyframes that we can choose, while there are two options to consider for motion information. We prioritize the temporal difference approach due to its ability to offer more intricate motion details under fixed surveillance perspectives and its quicker processing compared to optical flow. Consequently, we utilize the initial frame _Fa_ = _F_<sup>1</sup> for appearance information and the temporal difference _Fm_ between consecutive frames _F_ = _{F_<sup>1</sup> _, . . . , F_<sup>_N_</sup> _}, N_ = 6, which can be mathematically represented as: 



By incorporating both conditions, we can utilize a noise estimation network to infer the noise in the forward process. The loss function is specified as: 





where _ϵθ_ ( _Ft_<sup>_N_+1</sup> _, t, {Fa, Fm}_ ) is our noise estimation network with appearance _Fa_ and motion _Fm_ conditions. Details of the architecture will be introduced in Sec.3.4. 

### **3.3 Patch-based Diffusion Model** 

The diffusion model is highly efficient in various scenarios, yet it necessitates a substantial amount of computational time for the iterative processing of a complete image. This becomes particularly challenging when working with large images, leading to significant delays in inference time. Moreover, the majority of surveillance images feature a background, posing a challenge for the diffusion model in discerning human objects across the entire image. To tackle this challenge, we introduce a patch-based diffusion model that assesses local patch noise to expedite the computation process while concentrating on more localized regions. In the training phase, we extract patch frames from all images and acquire the corresponding patches for different conditions and noisy images. To enhance the diversity and robustness of the patch approach, random boxes are generated to improve noise learning. This can be formulated as: 





Hence, the forward process will also be carried out in a patch fashion, and the loss function is modified as: 



The training phase of our model predicts the noise of a local patch from the entire image by utilizing random cuts. Subsequently, we employ a sliding-window crop technique to generate mesh patches for the inference stage, which is discussed in Sec.3.5. 

### **3.4 Architecture** 

Figure 2 shows the architecture of our MA-PDM, which is composed of the appearance encoder and noise estimation network. At the initial data processing stage, we combine the patch conditions of appearance _Ca_ and motion _Cm_ . Following previous work on the image-to-image task (Wu et al. 2022; Lugmayr et al. 2022), we join the motion frames with the noisy frame [ _Cm, ft_<sup>_T_+1</sup> ] to form the input and then use the U-Net model to predict the adding noise. The appearance patch _Ca_ contains abundant semantics for the prediction task, and we use the same encoder part of the U-Net framework to extract the appearance and semantic information. Both information will be integrated into the U-Net as additional conditions. 

**Appearance Encoder Network.** As mentioned by DAE (Preechakul et al. 2022), semantics play a crucial role in generating semantic content. By adopting a similar embedding approach and incorporating additional encoder features to handle hierarchical structures (Song et al. 2024), our appearance encoder becomes capable of capturing semantic embeddings and hierarchical attributes from the appearance context. This can be formally represented as: 



where _hi_ represents the feature map of _i_ -th layer of the encoder. _ϕe_ is the appearance encoder network for embedding the input frame and has the same structure as the denoising U-Net. In order to learn the normal semantics for only regular data, we utilize the learnable patch-memory bank _M ∈_ R<sup>_P ×N×D_</sup> to preserve the normal semantics for different partial patches. The addressing operation is characterized as follows: 



where _hn_ represents the output of the Appearance Encoder, _Mp ∈_ R<sup>_N×D_</sup> denotes the _p_ -th memory bank of _M_ , and _p_ corresponds to the patch location within the entire image. For instance, if the coordinates of the random patch _Ca_ are (58 _,_ 58), then _p_ = � <u>5864</u><sup>_−_</sup><sup><u>1</u></sup> 2 � _×_ 4 + � <u>5864</u><sup>_−_</sup><sup><u>1</u></sup> 2 � = 5. Detail framework is shown in Supplementary Material. Depending on the addressing stage using the cross-attention module, the filtered semantic _C_<sup>ˆ</sup> _sem_ is obtained to reconstruct the input patch _Ca_ with a decoder network, and the reconstruction loss is applied to constrain the decoder result _C_<sup>ˆ</sup> _a_ : 



**Noise Estimation Network.** The U-Net structure serves as the foundation for noise estimation. To incorporate motion and appearance conditions into the network, we initially combine the frame differences with the noise frame using [ _Cm, ft_<sup>_N_+1</sup> ] as input frames. Subsequently, the hierarchical 

10764 

feature maps [ _h_ 1 _, . . . , hn_ ] are integrated into the U-Net network, detail framework is shown in Supplementary Material. Filtered semantics _S_ = Pool( _C_<sup>ˆ</sup> _sem_ ) are embedded in the entire network with the time-step embedding _temb_ , the semantics are embedded in each ResBlock of U-Net by the following formulation: 





where _ϕs, ϕb_ are the scale and bias parampters of Adaptive Layer Normalization. 

In the phase of decoding part of the noise estimation network, we incorporate the hierarchical appearance feature of the appearance encoder network into the U-Net decoder, similar to ControlNet (Zhang and Agrawala 2023). It is mathematically represented as [ _h_<sup>_N_</sup> _d_<sup>_, h_</sup> _e_<sup>_N_+</sup><sup>_hA_</sup> _e_<sup>],</sup><sup>_hN_</sup> _d_<sup>_, h_</sup> _e_<sup>_N_are</sup> the noise feature map derived from the encoder and decoder section of the noise estimation network. _h_<sup>_A_</sup> _e_<sup>isthefeature</sup> map _hi_ sourced from the appearance encoder. _h_<sup>_N_</sup> _e_<sup>+</sup><sup>_hA_</sup> _e_<sup>is a</sup> dynamic skip connection to control appearance information. Detail framework is shown in Supplementary Material. 

**Loss function.** At the training stage, we use the above two losses to learn our MA-PDM model: 



### **3.5 Inference Stage** 

During the inference phase, the reverse process of the diffusion model is used to extract random Gaussian noise under specific appearance and motion conditions. In contrast to the random style utilized during the training phase, images are generated from a standard partition. We use a sliding window to obtain various patches _fb_<sup>_N_+1</sup> from the complete Gaussian noise _FT_<sup>_N_+1</sup> , for example, the patch size is _p_ = 64, the sliding step is _s_ = _r, r < p_ , the total image size is _H_ = _W_ = 256, and the position box _b_ is ( _x_ 1 _, y_ 1 _, x_ 2 _, y_ 2). There is an overlap in the patches due to the sampling strategy, and we compute the overlap pixel rate _Ri_ = _n, R ∈_ R<sup>_H×W_</sup> as ( Ozdenizci<sup>¨</sup> and Legenstein 2023). Here, _R_ is the normalization matrix of the current sampling strategy, and _n_ is the number of pixels of overlap. Using these patch samples _ft_<sup>_N_</sup> ; _b_<sup>+1</sup> _∈_ R<sup>_c×p×p_</sup> generated in the position box _b_ , we predict the noise _ϵb_ at step _t_ using our MAPDM network: 



We assemble the patches _{ϵb}_ to build the complete image noise _ψt_<sup>_N_+1</sup> using the subsequent merging technique: 



With the matrix of _R_ , we can always maintain the stable amplitude of the noise. 

The implicit sampling (Nichol and Dhariwal 2021), which exploits a generalized non-Markovian forward process formulation, is applied to our diffusion model to accelerate the sampling speed. The reverse process is defined as: 



where _µθ_ ( _Ft_<sup>_N_+1</sup> _, t, {Fa, Fm}_ ) indicates the estimate of denoising transition mean in the forward process. Hence, a sample step can be generated from the DDIM reverse process on both conditions by: 



Finally, we can obtain the clean image _I_<sup>ˆ</sup> _N_ +1 = _F_<sup>ˆ</sup> 0<sup>_N_+1</sup> after all the denoising steps. Similarly, we can also merge the appearance conditions of the reconstruction _C_<sup>ˆ</sup> _a_ to get _I_<sup>ˆ</sup> 1 according to the above method. We compute the anomaly score using the MSE function between the predicted frame _I_ ˆ _N_ +1 _,_ ˆ _I_ 1 and its ground truth _IN_ +1 _, I_ 1 as follows: 



## **4 Experiments** 

### **4.1 Experimental Setup** 

**Datasets.** We assess the effectiveness of our approach using four standard datasets prevalent in the Visual Anomaly Detection community. The datasets we employ include: 

- Ped2. It comprises 16 training videos and 12 test videos captured in static environments. 

- Avenue. It consists of 16 training videos and 21 testing videos, with a total of 47 abnormal events, including throwing a bag, approaching or moving away from the camera, and running on pavements. 

- ShanghaiTech Campus. It consists of 330 training videos and 107 testing videos, which cover a variety of scenarios. It also contains 130 abnormal events, including brawls, robberies, fights, and other unusual activities. 

- UBnormal. It composes of various virtual scenes created using the Cinema4D software with 29 scenes. We utilize the standard training and testing sets provided to assess the performance of our method for one-class VAD. 

**Evaluation Metric.** In order to evaluate the effectiveness of our approach, we employ the area under the curve (AUC) of the Receiver Operating Characteristic (ROC) curve at the frame-level, a commonly adopted assessment measure in the field of VAD. 

**Training Details.** Each frame is resize to the image size of 256 _×_ 256. The seventh frame is predicted using the previous six frames. The patch memory banks’ capacity for all datasets is established at 16 _×_ 64 _×_ 256. In the training phase, patches are randomly selected and extracted from complete images, with each patch measuring 64 _×_ 64 for all datasets. The Adam optimizer is used to train our MA-PDM, with a learning rate set at 0 _._ 0002. The diffusion process settings are as follows: _β_ 1 = 0 _._ 0001, _β_ 2 = 0 _._ 02, _T_ = 1000, and the linear schedule is implemented in the same way as in DDIM. The weights of the loss function are _λ_ = 0 _._ 1. The training epochs are set to 1000 for Ped2, 300 for Avenue, 30 for Shanghai and 40 for UB. The batch size in all three datasets is 16. During the testing phase, we employ a sliding window 

10765 

||Method|Venue|Ped2|Avenue|Shanghai|UB|
|---|---|---|---|---|---|---|
||Conv-AE|CVPR16|90.0|70.2|60.9|-|
||ConvLSTM-AE|ICME17|88.1|77.0|-|-|
||Stacked RNN|ICCV17|92.2|81.7|68.0|-|
||AMC|ICCV19|96.2|86.9|-|-|
|R.|MemAE|ICCV19|94.1|83.3|71.2|-|
||CDDA|ECCV20|96.5|86.0|73.3|-|
||MNAD|CVPR20|90.2|82.8|69.8|-|
||Zhong et al.|PR22|97.7|88.9|70.7|-|
||FPDM|ICCV23|-|90.1|78.6|62.7|
||FFP|CVPR18|95.4|84.9|72.8|-|
||AnoPCN|ACMMM19|96.8|86.2|73.6|-|
||MNAD|CVPR20|97.0|88.5|70.5|55_._3<sup>_∗_</sup>|
|P.|ROADMAP|TNNLS21|96.3|88.3|76.6|-|
||MPN|CVPR21|96.9|89.5|73.8|-|
||AMMC-Net|AAAI21|96.9|86.6|73.7|-|
||LGN-Net|Arxiv23|97.1|89.3|73.0|57_._4<sup>_∗_</sup>|
||DLAN-AC|ECCV22|97.6|89.9|74.7|-|
||USTN-DSC|CVPR23|98.1|89.9|73.8|-|
||AED-MAE|CVPR24|95.4|**91.3**|79.1|58.5|
||**Ours**|-|**98.6**|**91.3**|**79.2**|**63.4**|



Table 1: Comparison of SOTA methods on four video anomaly datasets. The average AUC score (%) is used as the measurement metric. Here, R. and P. denote reconstructionand prediction-based methods, _∗_ indicates our reproduction. The top two scores are highlighted with bold and underline. 

|PD|A-C|M-C|Ped2|Avenue|Shanghai|UB|
|---|---|---|---|---|---|---|
|✓|||96.8|89.2|73.9|56.3|
||✓||95.8|86.2|72.4|55.3|
||✓|✓|96.7|88.8|76.7|57.8|
|✓|✓||97.4|89.9|77.1|58.4|
|✓|✓|✓|98.6|91.3|79.2|63.4|



Table 2: Comparison of each module’s performance on four video anomaly datasets. PD indicates our patch-based diffusion model, A-C indicates the appearance condition, and M-C indicates the motion condition. 

approach with a stride of 64. The number of reverse steps in DDIM is configured to 5. The values for _α_ are (0 _,_ 0 _._ 2 _,_ 0 _._ 3 _,_ 1) across four datasets. 

### **4.2 Comparison with SOTA Methods** 

In Table 1, we present a comprehensive comparison of our innovative frame prediction-based approach compared to current best-performing methods in the domains of reconstruction and prediction on four benchmark datasets. Our evaluation includes various memory-based methods, particularly MPN (Lv et al. 2021), LGN-Net (Zhao et al. 2022), and DLAN-AC (Yang et al. 2022), known for their advanced anomaly detection capabilities. The results, as shown in the table, indicate that our approach outperforms all other methods assessed. Meanwhile, our MA-PDM model establishes a new standard by improving SOTA performance metrics by 1. 2%, 0. 5%, and 0. 7% in the Avenue, Shanghai, and UB datasets, respectively, compared to the diffusion method: FPDM (Yan et al. 2023). We can also achieve performance that is close to or even better than AED-MAE (Ristea et al. 2024) without any synthetic anomalies. 

|Dataset|32|Rando<br>64|m Crop<br>128|256|Uni<br>32|form C<br>64|rop<br>128|
|---|---|---|---|---|---|---|---|
|Ped2|**98.6**|**98.6**|97.2|96.8|97.7|97.3|96.2|
|Avenue|90.2|**91.3**|89.9|88.7|88.7|89.4|89.1|
|Shanghai|78.7|**79.2**|78.9|74.8|73.3|77.6|76.4|



Table 3: Comparison of the results of different patch sizes under two distinct cropping configurations. 

### **4.3 Ablation Study** 

**Effectiveness of proposed modules.** We present the results of five combinations of three modules to examine their respective influences, as detailed in Table 2. The Patch Diffusion model is efficient from the results of Case 1. It illustrates that we have replicated the MNAD in the A-C scenario for Case 2, which only uses the six subsequent frames to predict the seventh frame using the memory block. As depicted in Case 3, a minor enhancement is observed when the motion condition is incorporated with MNAD. Specifically, in Case 4, when integrated with a patch-based diffusion model, it enhances the performance on four datasets by 1 _._ 6%, 3 _._ 7%, 4 _._ 7%, and 3 _._ 1% compared to Case 2 of MNAD that only uses the appearance condition. This implies that the patch-based diffusion model is successful in handling irregular frames and achieving exceptional results. The final case, utilizing our comprehensive model, shows the effectiveness of motion, allowing us to reach peak performance in detecting anomalies in videos. 

**Effectiveness of patch-base diffusion model.** We conduct an experiment to determine the optimal patch size for our patch-based diffusion model on three datasets. Initially, we evaluated four distinct patch dimensions (32 _,_ 64 _,_ 128 _,_ 256) using a random cropping approach. The results of the AUC score are shown in Table 3. Our technique achieves the highest results on four datasets with a patch size of 64. Intriguingly, in the Ped2 and Avenue datasets, pedestrians are small and the scenes are singular, making the size of 32 more efficient for the Ped2 dataset. With an increased patch size of the complex dataset, the model will discern more data. The comprehensive image-based diffusion model, with a patch size of 256, yields a lower performance compared to the other patch-based diffusion models. The results of the uniform cropping method are then displayed in Table 3. This suggests that a patch size of 32 is appropriate for the Ped2, while a size of 64 is suitable for the other dataset. The random cropping method is shown to be more efficient than the uniform one, as it increases and enlarges the dataset. 

**Effectiveness of patch memory bank model.** To explore the impact of memory block with patch style, we report the results of three types of memory variants in Table 4. Case 1 of “W/O” indicates the results obtained without the integration of any memory blocks. It achieves the approximate level compared with USTN-DSC (Yang et al. 2023). “G-M” represent the global memory without chunking, it has a larger memory size than our patch-based “P-M” method. From the Case-2 and Case-3 results, it indicates the patch memory 

10766 



<!-- Start of picture text -->
(a) A Ped2 video with abnormal events: riding bicycle. (b) A Ped2 video with abnormal events: riding bicycle and driving car<br>(c) A SH video with abnormal events: riding bicycle. (d) A SH video with abnormal events: riding motorcycle.<br>MNAD<br>LGN-Net<br>Ours<br>MNAD<br>LGN-Net<br>Ours<br><!-- End of picture text -->

Figure 3: Four examples of anomaly detection comparison on Ped2 and ShanghaiTech datasets. 

|Memory Variants|Ped2|Avenue|Shanghai|
|---|---|---|---|
|W/O|98.0|89.9|77.9|
|G-M|98.6|90.2|78.7|
|P-M|98.6|91.3|79.2|



Table 4: Comparison of the results of memory module. 

|Motion|Ped2|Avenue|FPS|
|---|---|---|---|
|OF|93.7|88.7|25|
|TD|98.6(+4.9)|91.3(+2.6)|45(+20)|



Table 5: Comparison of the results of temporal difference (TD) and optical flow (OF) conditions. 

### **4.4 Qualitative Results** 

Additionally, the anomaly curves of four test videos are shown in Figure 3, which offers a comparison between MNAD, LGN-Net and our MA-PDM. The curves represent the anomaly scores for each video frame in sequence, facilitating the comparison of the effectiveness of different methods. It’s evident that our MA-PDM outperforms both MNAD and LGN-Net in abnormal sections, yielding higher and more stable anomaly scores. Furthermore, in normal periods, MA-PDM surpasses MNAD and LGN-Net, with our normal scores being more consistent than those of LGN-Net, as shown in Figure 3(a) and Figure 3(c). More visualizations are shown in Supplementary Material. 

## **5 Conclusion** 

bank is more efficent for our patch diffusion model. 

**Effectiveness of motion choice.** We conduct an experiment to compare the performance of optical flow (OF) and temporal difference (TD). Table 5 displays the results of our experiment. We employ RAFT (Teed and Deng 2020) for the rapid and effective extraction of optical flows. Our TD method outperforms the OF technique in terms of results. In a comparison of the frames per second (FPS) for both techniques, our TD method proves to be quicker than the flow approach. Evidently, the OF method requires additional time to extract features and is vulnerable to OF models. 

In this paper, we propose a Patch Diffusion Model with Motion and Appearance Guidance for VAD. To capture finegrained local information, we propose a novel patch-based diffusion model specifically engineered. In addition, we introduce innovative appearance and motion conditions that are seamlessly integrated into our patch diffusion model. Numerous experimental results on four benchmarks demonstrate that our model performs better than previous stateof-the-art methods. Nevertheless, there is considerable room for improvement, particularly in the area of real-time speed for inference. Our future work involves exploring faster frame prediction networks with the MA-PDM model and incorporating richer conditions, such as text prompts. 

10767 

## **Acknowledgements** 

This work is supported by the National Key Research and Development Program of China (No.2020YBF2901202), National Natural Science Foundation of China (NSFC No. 62272184 and No. 62402189), the China Postdoctoral Science Foundation under Grant Number GZC20230894, the China Postdoctoral Science Foundation (Certificate Number: 2024M751012), and the Postdoctor Project of Hubei Province under Grant Number 2024HBBHCXB014. The computation is completed in the HPC Platform of Huazhong University of Science and Technology. 

## **References** 

Bao, Q.; Liu, F.; Liu, Y.; Jiao, L.; Liu, X.; and Li, L. 2022. Hierarchical scene normality-binding modeling for anomaly detection in surveillance videos. In _Proceedings of the 30th ACM international conference on multimedia_ , 6103–6112. Cai, R.; Zhang, H.; Liu, W.; Gao, S.; and Hao, Z. 2021. Appearance-motion memory consistency network for video anomaly detection. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 35, 938–946. 

Chang, Y.; Tu, Z.; Xie, W.; and Yuan, J. 2020. Clustering driven deep autoencoder for video anomaly detection. In _European Conference on Computer Vision_ , 329–345. Springer. Chen, C.; Xie, Y.; Lin, S.; Yao, A.; Jiang, G.; Zhang, W.; Qu, Y.; Qiao, R.; Ren, B.; and Ma, L. 2022. Comprehensive regularization in a bi-directional predictive network for video anomaly detection. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 36, 230–238. 

Couairon, G.; Verbeek, J.; Schwenk, H.; and Cord, M. 2022. Diffedit: Diffusion-based semantic image editing with mask guidance. _arXiv preprint arXiv:2210.11427_ . 

Dhariwal, P.; and Nichol, A. 2021. Diffusion models beat gans on image synthesis. In _Advances in neural information processing systems_ , volume 34, 8780–8794. 

Gong, D.; Liu, L.; Le, V.; Saha, B.; Mansour, M. R.; Venkatesh, S.; and Hengel, A. v. d. 2019. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In _Proceedings of the IEEE international conference on computer vision_ , 1705–1714. 

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. In _Advances in neural information processing systems_ , 6840–6851. 

Lee, S.; Kim, H. G.; Choi, D. H.; Kim, H.-I.; and Ro, Y. M. 2021. Video prediction recalling long-term motion context via memory alignment learning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 3054–3063. 

Liu, W.; Luo, W.; Lian, D.; and Gao, S. 2018. Future frame prediction for anomaly detection–a new baseline. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 6536–6545. 

Liu, Z.; Nie, Y.; Long, C.; Zhang, Q.; and Li, G. 2021. A hybrid video anomaly detection framework via memoryaugmented flow reconstruction and flow-guided frame pre- 

diction. In _Proceedings of the IEEE international conference on computer vision_ , 13588–13597. 

Lu, C.; Shi, J.; and Jia, J. 2013. Abnormal Event Detection at 150 FPS in MATLAB. In _Proceedings of the IEEE international conference on computer vision_ , 2720–2727. 

Lu, Z.; Wu, C.; Chen, X.; Wang, Y.; Qiao, Y.; and Liu, X. 2023. Hierarchical Diffusion Autoencoders and Disentangled Image Manipulation. _arXiv preprint arXiv:2304.11829_ . 

Lugmayr, A.; Danelljan, M.; Romero, A.; Yu, F.; Timofte, R.; and Van Gool, L. 2022. Repaint: Inpainting using denoising diffusion probabilistic models. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 11461–11471. 

Luo, W.; Liu, W.; and Gao, S. 2017. A revisit of sparse coding based anomaly detection in stacked rnn framework. In _Proceedings of the IEEE international conference on computer vision_ , 341–349. 

Lv, H.; Chen, C.; Cui, Z.; Xu, C.; Li, Y.; and Yang, J. 2021. Learning normal dynamics in videos with meta prototype network. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 15425–15434. 

Mokady, R.; Hertz, A.; Aberman, K.; Pritch, Y.; and CohenOr, D. 2023. Null-text inversion for editing real images using guided diffusion models. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 6038–6047. 

Nichol, A. Q.; and Dhariwal, P. 2021. Improved denoising diffusion probabilistic models. In _International conference on machine learning_ , 8162–8171. PMLR. 

¨Ozdenizci, O.; and Legenstein, R. 2023. Restoring vision in adverse weather conditions with patch-based denoising diffusion models. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ . 

Preechakul, K.; Chatthee, N.; Wizadwongsa, S.; and Suwajanakorn, S. 2022. Diffusion autoencoders: Toward a meaningful and decodable representation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 10619–10629. 

Ristea, N.-C.; Croitoru, F.-A.; Ionescu, R. T.; Popescu, M.; Khan, F. S.; Shah, M.; et al. 2024. Self-distilled masked auto-encoders are efficient video anomaly detectors. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 15984–15995. 

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 10684–10695. 

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In _International Conference on Medical image computing and computer-assisted intervention_ , 234–241. Springer. 

Roth, K.; Pemula, L.; Zepeda, J.; Sch¨olkopf, B.; Brox, T.; and Gehler, P. 2022. Towards total recall in industrial anomaly detection. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 14318–14328. 

10768 

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, K.; Gontijo Lopes, R.; Karagol Ayan, B.; Salimans, T.; et al. 2022a. Photorealistic text-to-image diffusion models with deep language understanding. In _Advances in neural information processing systems_ , volume 35, 36479–36494. 

Saharia, C.; Ho, J.; Chan, W.; Salimans, T.; Fleet, D. J.; and Norouzi, M. 2022b. Image super-resolution via iterative refinement. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 45(4): 4713–4726. 

Song, Z.; Luo, R.; Yu, J.; Chen, Y.-P. P.; and Yang, W. 2023. Compact transformer tracker with correlative masked modeling. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 37, 2321–2329. 

Song, Z.; Tang, Y.; Luo, R.; Ma, L.; Yu, J.; Chen, Y.-P. P.; and Yang, W. 2024. Autogenic language embedding for coherent point tracking. In _Proceedings of the 32nd ACM International Conference on Multimedia_ , 2021–2030. 

Song, Z.; Yu, J.; Chen, Y.-P. P.; and Yang, W. 2022. Transformer tracking with cyclic shifting window attention. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 8791–8800. 

Stephen, K.; and Menon, V. 2020. Learning Memory-guided Normality for Anomaly Detection. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 14360–14369. 

Teed, Z.; and Deng, J. 2020. Raft: Recurrent all-pairs field transforms for optical flow. In _European Conference on Computer Vision_ , 402–419. Springer. 

Ye, Y.; Cai, J.; Zhou, H.; Li, G.; Zhang, Y.; Song, Z.; Gao, C.; Yu, J.; and Yang, W. 2024. Progressive Text-to-Image Diffusion with Soft Latent Direction. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, 6693–6701. 

Yu, G.; Wang, S.; Cai, Z.; Zhu, E.; Xu, C.; Yin, J.; and Kloft, M. 2020. Cloze test helps: Effective video anomaly detection via learning to complete video events. In _Proceedings of the 28th ACM international conference on multimedia_ , 583–591. 

Zhang, H.; Wang, Z.; Wu, Z.; and Jiang, Y.-G. 2023. DiffusionAD: Denoising Diffusion for Anomaly Detection. _arXiv preprint arXiv:2303.08730_ . 

Zhang, L.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. _arXiv preprint arXiv:2302.05543_ . 

Zhao, M.; Liu, Y.; Liu, J.; Li, D.; and Zeng, X. 2022. LGNNet: Local-Global Normality Network for Video Anomaly Detection. _arXiv preprint arXiv:2211.07454_ . 

Zhong, Y.; Chen, X.; Jiang, J.; and Ren, F. 2022. A cascade reconstruction model with generalization ability evaluation for anomaly detection in videos. _Pattern Recognition_ , 122: 108336. 

Zhou, H.; Yu, J.; and Yang, W. 2023. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, 3769–3777. 

Wolleb, J.; Bieder, F.; Sandk¨uhler, R.; and Cattin, P. C. 2022. Diffusion models for medical anomaly detection. In _International Conference on Medical image computing and computer-assisted intervention_ , 35–45. Springer. 

Wu, J.; Fang, H.; Zhang, Y.; Yang, Y.; and Xu, Y. 2022. Medsegdiff: Medical image segmentation with diffusion probabilistic model. _arXiv preprint arXiv:2211.00611_ . 

Wu, P.; Liu, J.; Shi, Y.; Shao, F.; Wu, Z.; and Yang, Z. 2020. Not only Look, But Also Listen: Learning Multimodal Violence Detection Under Weak Supervision. In _European Conference on Computer Vision_ , 322–339. 

Yan, C.; Zhang, S.; Liu, Y.; Pang, G.; and Wang, W. 2023. Feature Prediction Diffusion Model for Video Anomaly Detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 5527–5537. 

Yang, Z.; Liu, J.; Wu, Z.; Wu, P.; and Liu, X. 2023. Video Event Restoration Based on Keyframes for Video Anomaly Detection. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 14592–14601. 

Yang, Z.; Wu, P.; Liu, J.; and Liu, X. 2022. Dynamic local aggregation network with adaptive clusterer for anomaly detection. In _European Conference on Computer Vision_ , 404– 421. Springer. 

Ye, M.; Peng, X.; Gan, W.; Wu, W.; and Qiao, Y. 2019. Anopcn: Video anomaly detection via deep predictive coding network. In _Proceedings of the 27th ACM international conference on multimedia_ , 1805–1813. 

10769 

