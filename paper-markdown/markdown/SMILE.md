# **SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning** 

Fida Mohammad Thoker, Letian Jiang, Chen Zhao<sup>_†_</sup> , Bernard Ghanem King Abdullah University of Science and Technology (KAUST) 

_{_ fida.thoker,letian.jiang,chen.zhao,Bernard.Ghanem _}_ @kaust.edu.sa 

## **Abstract** 

_Masked video modeling, such as VideoMAE, is an effective paradigm for video self-supervised learning (SSL). However, they are primarily based on reconstructing pixellevel details on natural videos which have substantial temporal redundancy, limiting their capability for semantic representation and sufficient encoding of motion dynamics. To address these issues, this paper introduces a novel SSL approach for video representation learning, dubbed as SMILE, by infusing both spatial and motion semantics. In SMILE, we leverage image-language pretrained models, such as CLIP, to guide the learning process with their high-level spatial semantics. We enhance the representation of motion by introducing synthetic motion patterns in the training data, allowing the model to capture more complex and dynamic content. Furthermore, using SMILE, we establish a new self-supervised video learning paradigm capable of learning strong video representations without requiring any natural video data. We have carried out extensive experiments on 7 datasets with various downstream scenarios. SMILE surpasses current state-of-the-art SSL methods, showcasing its effectiveness in learning more discriminative and generalizable video representations. Code is available: https://github.com/fmthoker/SMILE_ 

## **1. Introduction** 

Self-supervised learning (SSL) [60, 62, 77, 79] plays a crucial role for many video understanding tasks, such as action recognition [24, 35, 70, 71, 75] and detection [37, 44], significantly reducing the reliance on extensive human annotation. The goal of video SSL is to learn video representations that capture both spatial and temporal semantics effectively. Masked video modeling has emerged as a promising approach with state-of-the-art performance [16, 29, 72, 77]. It involves masking a large portion of video frames and reconstructing the masked regions using the unmasked parts. 

However, this approach has some limitations. _First_ , it relies heavily on pixel reconstruction, which can lead to overfitting low-level details. This limits the model’s abil- 



<!-- Start of picture text -->
UCF-101 UCF-101 FT: Full Finetuning<br>(LP) (LS) LP: Linear Probing<br>LS: Low-Shot<br>Something- Something-<br>Something V2 73 81 Something V2<br>(LP) (FT)<br>20 62 76 70<br>17 68<br>51 71<br>13 66<br>Kinetics-400(LP) 42 27 13 78 80 82 Kinetics-400(FT)<br>20 84<br>33 21<br>24 86<br>27 40 26 88<br>GYM(LP) 47 31 GYM(FT)<br>HMDB-51 GYM<br>(LP) (LS)<br>VideoMAE MME MGMAE MGM SMILE (Ours)<br><!-- End of picture text -->

Figure 1. **Comparison with SOTA masked video modeling methods** . Our method significantly outperforms prior masked video modeling methods across diverse downstream settings. 

ity to learn meaningful high-level semantics essential for discriminative representations, as shown by the suboptimal linear probing performance of current masked video modeling methods [16, 29, 72, 77] in Figure 1. The _second_ key issue in masked video modeling is the substantial temporal redundancy of video data with most videos having little foreground motion, human or object. Consequently, the reconstruction objective can be solved using the temporal correlations between co-located patches or by spatial correlations in tube masking [77], resulting in inadequate motion awareness in the learned representations. This inturn leads to poor generalization on downstream tasks, requiring a finer motion understanding as shown in [75]. 

In this work, we address these limitations by proposing a novel framework for masked video modeling, dubbed as SMILE ( **S** patial and **M** otion semant **I** cs in masked video **LE** arning), to capture high-level spatial and motion semantics leading to generalizable video representations. To improve the spatial semantics in masked video modeling, we propose to explicitly infuse high-level spatial semantics from pretrained image foundation models such as 

CLIP [63]. Such models encapsulate high-level semantics learned through large-scale image-text alignment and have been adapted for video understanding tasks. For example, ActionCLIP [81], ViFiCLIP [65], and X-CLIP [52]) adapt CLIP for action recognition using videos with action labels, whereas ViCLIP [84] and VideoPrism [97]) adapt it for video-text alignment with video-text pairs. We take inspiration from them but aim to learn video representations with unlabeled videos. In particular, we propose replacing the reconstruction of pixels in masked video modeling with that of CLIP embedding, thus providing high-level semantics as the supervision signal. 

To improve motion awareness, we propose to reduce the temporal redundancy in input training videos with synthetic motion patterns. Specifically, we randomly paste objects onto the video frames and move them smoothly along the temporal dimension to simulate variable temporal transitions of motion-intensive videos. This reduces the temporal redundancy in the original videos and results in videos with an additional prominent foreground motion. Furthermore, adding objects with known locations allows us to add a motion focus to the mask-and-predict task by explicitly masking the regions of object motion. This in turn encourages the network to avoid an over-reliance on static spatial information and encode the motion dynamics of the added objects, when solving the reconstruction task. We summarize our contributions as follows. 

- We introduce SMILE, a novel SSL approach for masked video modeling. It encodes spatial and motion semantics using guidance from image-text pretrained models and synthetic-motion augmented videos, leading to more discriminative and generalizable video representation. 

- We present a new strategy to augment video data with diverse synthetic motions, paired with a masking strategy that focuses on reconstructing these added motions to improve masked video modeling pretraining. 

- Through extensive evaluation on 7 datasets, including Kinetics-400 [35], UCF-101 [71], HMDB-51 [38], FineGYM [69], EPIC-Kitchens-100 [12], SomethingSomething V2 [24], and Charades [70], we demonstrate that SMILE significantly outperforms existing state-of-the-art video SSL methods and exhibits better generalization. 

## **2. Related Works** 

**Video Self-Supervised Learning.** Video SSL aims to learn the spatio-temporal dynamics of video data without the need for any supervision. Three main learning paradigms have emerged: transformation prediction, contrastive learning, and masked video modeling. In transformation prediction, models learn video patterns by solving tasks such as space-time puzzles [1, 34, 36], predicting order in video clips [21, 32, 40, 49, 50, 86], or estimating speed [4, 10, 90]. Contrastive learning is based on instance discrimina- 

tion [54] and learning invariances to spatio-temporal augmentations [2, 9, 19, 51, 56, 58, 62, 64, 74, 76]. Masked video modeling methods [16, 29, 72, 77, 88], like VideoMAE [77], use a mask-and-predict task to reconstruct videos from partial data with an encoder-decoder setup similar to MAE [27]. By masking a high proportion of pixels ( _e.g._ , 90%) and reconstructing them such methods learn useful video representations. Our approach aims to improve the spatial and motion semantics of the video representations learned by masked video modeling. 

**CLIP for Video Understanding.** CLIP [63] captures detailed spatial semantics from extensive image-text training, making it a strong foundation for directly modeling video data with added temporal dynamics, _e.g._ , [52, 57, 78, 81, 84, 89]. ViFiCLIP [65] finetunes CLIP on video by adding a temporal pooling layer, ActionCLIP [81] includes a temporal positional embedding and aggregation layer, and X- CLIP [52] integrates cross-frame attention to share information between frames. Most relevant to us are works that also use CLIP for video representation learning [17, 42, 84]. ViCLIP [84] initializes a model with CLIP weights and employs further contrastive pretraining with video-text pairs. UMT [42] uses CLIP as a teacher and aligns its predictions with the CLIP features for unmasked tokens, followed by a 2nd stage of learning from video-text contrastive pairs. TGM [17] relies on CLIP text-encoder features to guide masked video modeling and video-text contrastive learning in a unified framework. In contrast, our SMILE only learns from video data in a self-supervised manner, reconstructing CLIP features of masked tokens and added objects. 

**Motion-aware Video Self-Supervised Learning.** Encoding motion dynamics in video representations is essential for generalizing across various video tasks [75]. As a result, many video SSL methods focus on enhancing the motion sensitivity of learned representations. For example, [22, 26, 29, 53, 85] capture motion using optical flow, [13, 80] do so by removing background or static elements and [15, 30, 76, 79] incorporate synthetic motions. [30, 79] learn by predicting the speed and trajectory of the pseudo motions, and [76] contrasts video with different types of motions. These approaches are specific to transformation prediction and contrastive learning paradigms. We take inspiration from these works and inject synthetic motions into the masked video learning. 

Most relevant to our work are methods like [16, 29, 72, 88], which enhance motion understanding in VideoMAE [77] by modifying certain design choices. MGMAE [29] and MGM [16] improve masking strategies to mask the regions of motion using optical flow and motion vectors. MotionMAE [88] adds frame difference as an additional target for reconstruction, and MME [72] uses optical flow to extract dense object trajectories and then predicts these trajectories and their HOG features. However, these 

methods are limited by the temporal redundancies in the original data, restricting the amount of motion or motionfocused regions available for learning in the reconstruction task. In contrast, we add synthetic motions to the original video data, reducing temporal redundancy and increasing motion-rich regions to learn from. As shown in our experiments, reconstructing these modified videos encourages the model to capture motion dynamics, enhancing the motion awareness of the learned video representations. 

## **3. Methodology** 

### **3.1. Revisiting Masked Video Modeling** 

The goal of masked video modeling [16, 29, 77] is to learn video representations by employing a mask-and-predict task with an asymmetric encoder-decoder architecture. An input video _V ∈_ R<sup>_T ×H×W ×_3</sup> is first divided into nonoverlapped 3D space-time tokens ( _e.g._ , 2 _×_ 16 _×_ 16) as _T_ = _{v_ 1 _, . . . , vN }_ . From _T_ , a subset of tokens _T_<sup>_mask_</sup> are randomly masked with a high masking ratio ( _e.g._ , 90%) and the unmasked tokens _T_<sup>_unmask_</sup> = _T \ T_<sup>_mask_</sup> are fed into a transformer encoder Φ _enc_ . Then, the output of the encoder and the learnable tokens containing the positional information of the masked tokens are passed to a shallow decoder Φ _dec_ to reconstruct the input pixel with the following loss: 



where _vi_ refers to the _pixel values_ of the _i_ -th space-time token in _T_<sup>_mask_</sup> and _Yi_ = Φ _dec_ (Φ _enc_ ( _T_<sup>_unmask_</sup> )) denotes the corresponding reconstructed pixels for _i_ -th token given the set of all unmasked tokens _T_<sup>_unmask_</sup> . 

Such pixel reconstruction drives the model to focus on low-level features, and with high temporal redundancy in video data, it primarily reconstructs static appearance details of video data. However, effective video understanding also requires capturing temporal progression and high-level semantics, such as scene context and object interactions. 

### **3.2. General Scheme of SMILE** 

Our goal is to infuse spatial and motion semantics in masked video modeling. To achieve this, we integrate two key elements into the masked video modeling paradigm. To improve motion semantics, we overlay local motion patterns in input videos in the form of moving objects. To improve spatial semantics, we replace the low-level pixel targets with high-level features extracted from a pretrained CLIP encoder. By combining both, we learn video representations that can encode both spatial and motion dynamics necessary for modeling downstream video tasks. The overview of our method is shown in Figure 2. **Motion Infusion.** Given a input video _V ∈_ R<sup>_T ×H×W ×_3</sup> . We first sample a random object _o_ from a set of predefined 

segmented objects _O_ . Next, we generate a motion trajectory (Section 3.3), along which we overlay this object onto the video _V_ to obtain the motion-augmented video _V_<sup>_′_</sup> . By adding objects with synthetic motions, we reduce temporal redundancy in the original videos and encourage the model to learn from the motion dynamics of the overlaid objects. **Masking.** We employ two separate masking strategies on _V_<sup>_′_</sup> as follows. For the original regions, we employ random tube masking as in VideoMAE [77]. For added object regions, we propose _trajectory-based masking_ . Since we know the location of the object in each frame of _V_<sup>_′_</sup> , we explicitly drop space-time tokens at random along its trajectory. Employing a high masking ratio along the object trajectory ensures that solving the reconstruction task requires encoding the object's motion dynamics, thereby enhancing the motion awareness of learned representations. 

**Architecture.** We employ a teacher-student reconstruction architecture with a pretrained image-encoder as the teacher and an asymmetric encoder-decoder network Φ as the student. The augmented video is patchified into 3D space-time _′ ′_ tokens _T_ = _{v_ 1<sup>_, . . . , v_</sup> _N_<sup>_}_asingeneralvideomaskedau-</sup> toencoders (Section 3.1). The tokens are then masked using tube and trajectory-based masking to generate sets of masked _T_<sup>_mask_</sup> and unmasked _T_<sup>_unmask_</sup> tokens. The unmasked tokens _T_<sup>_unmask_</sup> and the learnable tokens containing the positional information of the masked patches are used as the input to Φ as in Section 3.1. **Reconstruction.** Instead of directly reconstructing pixels, we project the input video _V_<sup>_′_</sup> onto the teacher’s feature space to generate high-level targets. In particular, we choose a CLIP model as the projection network due to its high-level semantics learned from language guidance. Specifically, the input _V_<sup>_′_</sup> is passed frame-by-frame through the CLIP encoder to extract features. The extracted fea- _{_ tures are stacked into a sequence of feature tokens as _f_ 1 _′_<sup>_, . . . , f_</sup> _N′_<sup>_}_.To matchthe number oftokens with</sup><sup>_T_</sup> _F_<sup>,we</sup> = only use the CLIP features for the first time slice of the 3D space-time tokens, thus generating a total of _N_ feature tokens, each corresponding to a space-time token in the input video. The encoder-decoder is then trained with a masked feature reconstruction loss as: 



where _fi′_<sup>isthe</sup><sup>_CLIPfeatureprojection_of</sup><sup>_i_-thspace-time</sup> token _vi′_<sup>in</sup><sup>_Tmask_and</sup><sup>_Yi_=Φ</sup><sup>_dec_(Φ</sup><sup>_enc_(</sup><sup>_Tunmask_))de-</sup> notes the corresponding reconstructed feature for _i_ -th token given the set of all unmasked tokens _T_<sup>_unmask_</sup> . Such formulation first increases the temporal dynamics in otherwise static data, avoiding learning shortcuts for reconstruction. Second, projecting targets onto the CLIP space encourages the network to capture more high-level video semantics. 



<!-- Start of picture text -->
T = 15<br>Motion T = 7<br>Generation<br>ViT ViT<br>Encoder Decoder<br>T = 0<br>Overlay<br>CLIP<br>Visual<br>Encoder<br>Masked<br>Reconstruction<br>: Tube Masking<br>: Masked<br>: Trajectory Masking<br><!-- End of picture text -->

Figure 2. **Overall architecture of our SMILE.** An input video clip _V_ is overlaid with a segmented object along a randomly generated trajectory to generate _V ′_ infusing synthetic object motion in _V_ . _V ′_ is passed frame-by-frame through the CLIP encoder to extract feature tokens _F_ . _V ′_ is patchified into a set of space-time tokens _T_ . We apply two types of masking upon _T_ , tube masking on the space-time tokens of the original video and trajectory-based masking on the tokens of added objects. The unmasked tokens _Tunmask_ are fed into the encoder-decoder network Φ _enc_ Φ _dec_ which is trained to reconstruct the masked feature tokens in _F_ . 

Overall, the key intuition is that CLIP features capture high-level information, such as scene context and object interactions, which guides the network to attend to both the overlaid object motions and the original video semantics. 

### **3.3. Motion Generation** 

To infuse motion, we overlay a segmented object along a randomly generated trajectory onto the original video frames. To introduce more complex dynamics, the object is also transformed _e.g._ rotation, along the trajectory. **Appearance and Trajectory.** To infuse a new synthetic motion onto a video sequence _V_ =[ _v_ 1 _, v_ 2 _, ..., vT_ ] comprising _T_ frames, we begin by selecting an object _o_ from _O_ at random, where _O_ denotes the set of off-the-shelf segmented objects generated using Stable Diffusion [66] and X-Paste[96]. We then sample a scale _P ×Q_ from a predefined set of sizes for _o_ to model different object sizes, where _P ≪ H_ and _Q ≪ W_ . Next, we sample a sequence of locations for the object Traj _o_ = [( _x_<sup>1</sup> _, y_<sup>1</sup> ) _, ..,_ ( _x_<sup>_T_</sup> _, y_<sup>_T_</sup> )] representing the center coordinate ( _x_<sup>_i_</sup> _, y_<sup>_i_</sup> ) of the object _o_ in the corresponding frame. Specifically, we generate a random non-linear path for the trajectory by sampling _M_ 2D points ( _M ≫ T_ ) uniformly from _x ∈_ [0 _, H_ ] and _y ∈_ [0 _, W_ ]. To ensure smoothness in the motion, a 1 _D_ Gaussian filter is applied along _x_ and _y_ dimensions as [( _h_ ( _x_<sup>1</sup> ) _, h_ ( _y_<sup>1</sup> )) _, . . . ,_ ( _h_ ( _x_<sup>_M_</sup> ) _, h_ ( _y_<sup>_M_</sup> ))], where _h_ ( _z_ ) = _~~√~~_ 21 _πκ_<sup>_e−z_2</sup><sup>_/_2</sup><sup>_κ_2and</sup><sup>_κ_isthesmoothingfactor.Theresult-</sup> ing trajectory is then downsampled to _T_ target locations for the object _o_ . This formulation introduces diverse motions 



<!-- Start of picture text -->
Time<br><!-- End of picture text -->

Figure 3. **Motion overlaid videos** showing the position change and transformation of the added objects along the time dimension. 

with varying spatio-temporal dynamics in the form of objects with different shapes, sizes, and temporal progression. **Transformation.** To increase the complexity of the object’s motion, we also apply scale and rotation transformations along the temporal dimension. Specifically, we select 3 frame indices, the first, the last, and a random middle one, _v_ 1, _vm_ , and _vT_ , respectively. For these three frame indices, a rotation angle _αi_ , and a scaling factor _Si_ are selected randomly. The angles and scales for the rest of the frames are linearly interpolated between these keyframes. Finally, the object _o_ is rotated and scaled with the corresponding transformations � cossin _α αii −_ cossin _α αi i_ � and � _S_ 0 _i S_ 0 _i_ �, before overlaying it onto the video frame _vi_ . 

Figure 3 shows two sample videos with our added motions. While these motions may not appear entirely realistic, they represent complex motion dynamics of moving structured pixels, and introducing them into a mask-and-predict framework guides the model to learn from such dynamics. 

This results in video representations with improved downstream performance on motion-aware tasks, as we also validate empirically. Interestingly, we also show in the experiments (Section 5) that overlaying such object motions on black or noise frames can learn strong video representations without requiring any original videos. 

### **3.4. Training Procedure** 

By adding synthetic motions, we have introduced a domain shift in the input for reconstruction, which may affect the performance of downstream tasks. To explore effective learning from both original videos _V_ and motion-overlaid videos _V_<sup>_′_</sup> , we explore two training strategies, _i.e._ **mixed training** and **progressive training.** In the _mixed training_ strategy, we use both _V_ and _V_<sup>_′_</sup> as the input to the network and directly optimize for the reconstruction of both inputs. The idea is to simultaneously encode the spatio-temporal dynamics of the original video data and the motion dynamics of the augmented data, with a limited domain shift. In the _progressive training_ strategy, we employ a two-stage learning approach that first learns from augmented data _V_<sup>_′_</sup> alone, then continues to train the model with the original data _V_ alone. This way we can first learn to encode the spatio-temporal dynamics of the augmented data and then align the network to further encode the dynamics of the original data and its domain. 

## **4. Experiments** 

We evaluate SMILE against prior video SSL methods on action recognition benchmarks: Kinetics-400 (K400) [35], SomethingSomething V2 [24] (SSv2), UCF-101 [71] (UCF), HMDB-51 [38] (HMDB), FineGYM [69] (GYM), and EPIC-Kitchens-100 [12] (EPIC), achieving the best performance in linear probing and full finetuning (Section 4.2). SMILE also shows better generalization on the SEVERE benchmark[75] (Section 4.4). Ablations and further discussion are provided in Section 4.3 and Section 5, respectively. 

### **4.1. Implementation Details** 

**Model.** Following VideoMAE [77], we use ViT-S or ViT-B with joint space-time attention for our student network. We choose CLIP ViT-B [63] as our teacher network. **Motions.** We follow PIN [14] to create the set of segmented objects _O_ with Stable Diffusion [66] and X-Paste [96]. We sample the object size _P ×Q_ uniformly from [32 _×_ 32, 128 _×_ 128] and sample the rotation angles from [ _−_ 90 _,_ 90] and scaling factors from [0 _._ 5 _,_ 1 _._ 5]. 

**Pretraining.** We follow previous masked video modeling works [16, 29, 72, 77] and use Kinetics-400 [35] (K400) or SomethingSomething V2 [24] (SSv2) for pretraining unless specified otherwise. We use the same hyperparameters as in VideoMAE [77]. Unless stated otherwise, we train with our proposed _progressive training_ for a total of 600 epochs, 300 epochs in each stage. More training details are in **supplementary** material. 

Table 1. **Linear probing comparison of masked video modeling methods.** Our SMILE learns highly discriminative video representations useful for both spatial and motion-focused domains. All results are obtained with the ViT-B backbone pretrained on K400 and Top-1 accuracies (%) are reported. 

|**Method**|K400|UCF|HMDB|SSv2|GYM|EPIC|
|---|---|---|---|---|---|---|
|VideoMAE [77]|20.7|58.6|37.7|17.5|23.9|33.2|
|MVD [83]|18.7|49.1|28.6|12.2|22.7|29.7|
|MME [72]|19.1|56.0|37.1|16.6|29.0|32.2|
|MGMAE [29]|24.9|64.4|41.3|16.8|26.1|33.2|
|EVEREST [31]|14.1|51.8|30.3|14.5|23.3|30.5|
|MGM [16]|19.8|62.5|40.3|21.7|25.8|32.4|
|SIGMA [68]|47.5|80.7|52.3|20.8|30.1|34.2|
|SMILE w/o motion|54.6|82.6|51.7|21.6|27.9|32.3|
|SMILE (ours)|**56.2**|**83.8**|**53.4**|**23.7**|**30.2**|**34.4**|



### **4.2. Comparison with State-of-the-Art** 

**Linear Probing.** We freeze the pretrained backbone and only train a linear classifier on the target dataset. When evaluating the prior works for comparison, we utilize their publicly available checkpoints for the ViT-B backbone pretrained on the K400 dataset for 800 epochs except for MVD (2000 epochs) and MME (1600 epochs). We evaluate the downstream task using a common setup for a fair comparison. Details about datasets and experimental settings are in the **supplementary** material. 

Table 1 shows that our method, even without synthetic motion, significantly outperforms all state-of-the-art methods on K400 and UCF, and is among the top performers on HMDB, SSv2, GYM, and EPIC. This highlights that CLIP feature reconstruction offers a stronger supervisory signal than pixel reconstruction, especially in tasks requiring spatial semantics like K400 and UCF. Adding synthetic motions further boosts performance by an average of 1 _._ 8% across datasets, enhancing motion cues through reconstruction. On motion-sensitive datasets like SSv2, GYM, and EPIC, our approach consistently surpasses MGM [16], MGMAE [29], and MME [72], confirming that synthetic motions help overcome the lack of motion diversity in current datasets that may limit these methods. 

**Full Finetuning.** We finetune both the pretrained backbone and the classification head end-to-end following [29, 72, 77] on the downstream datasets K400 and SSv2. Following prior masked video modeling works [16, 29, 77], we show results with self-supervised pretraining on K400 and SSv2 datasets for the VIT-B backbone. We follow VideoMAE [77] for finetuning and evaluation protocols with more details in the **supplementary** . 

Tables 2 and 3 show the results for SSv2 and K400, respectively. On SSv2, we achieve the best performance with both K400 and SSv2 pretraining, highlighting our method’s effectiveness for both in-domain and cross-domain transfer. 

Table 2. **Full finetuning comparison of various self-supervised methods on Something-Something V2** . All results are obtained with a ViT-B backbone. _∗_ denotes results obtained by our evaluation. Our SMILE achieves state-of-the-art performance. Top-1 accuracies (%) are reported. 

|**Method**|**Backbone **|**Epochs **|**Pretrain **|**Top-1**|
|---|---|---|---|---|
|VideoMAE [77]|ViT-B|800|K400|68.5|
|OmniMAE [23]|ViT-B|800|K400|69.0|
|MGMAE<sup>_∗_</sup>[29]|ViT-B|800|K400|68.9|
|MME [72]|ViT-B|800|K400|70.5|
|SIGMA [68]|ViT-B|800|K400|71.1|
|MGM<sup>_∗_</sup>[16]|ViT-B|800|K400|71.1|
|SMILE(ours)|ViT-B|600|K400|**72.1**|
|OmniMAE [23]|ViT-B|800|SSv2|69.5|
|VideoMAE [77]|ViT-B|800|SSv2|69.6|
|MGM [16]|ViT-B|800|SSv2|70.6|
|MME [72]|ViT-B|800|SSv2|70.0|
|SIGMA [68]|ViT-B|800|SSv2|71.2|
|MGMAE [29]|ViT-B|800|SSv2|71.0|
|SMILE (ours)|ViT-B|800|SSv2|**72.5**|



For K400, we exceed the prior top SSL approach by 1.6%. Additionally, our gains over motion-focused methods, such as MGMAE (1.5% on SSv2, 1.9% on K400), MME (2.5% on SSv2, 1.6% on K400), and MGM (1.9% on SSv2, 2.3% on K400), confirm our improved motion sensitivity, even in full-finetuning settings. Notably, our model surpasses VideoMAE, MotionMAE, MME, and MGMAE trained for more than twice the number of epochs (see **supplementary** for the extensive comparison). We attribute this efficiency to CLIP feature reconstruction, which directly captures high-level semantics compared to pixel-level learning. 

### **4.3. Ablations** 

We ablate various components of our SMILE to show their impact on the downstream performance in Table 4. For computational efficiency, we use ViT-S as the backbone and pretrain with a smaller subset of K400, denoted as K400m, containing around 80K samples. For evaluation, we use K400m and SSv2m that denotes a subset of SSv2 with 50% training data. Unless stated otherwise, we use CLIP features as the reconstruction target, add two synthetic objects, mask 90%, use trajectory masking, and train for 400 epochs. **Synthetic motion,** **_<u>yes</u>_ or** **_no_ ?** In Table 4a, we show the impact of adding synthetic motions to the masked video modeling framework for both pixel and feature reconstruction. We can see that adding our synthetic motion patterns significantly improves the downstream performance of both pixel and feature reconstruction. 

**Reconstructing** **_pixels_ or** **_<u>features</u>_ ?** Table 4h compares pixel-based and feature-based reconstruction targets in masked video modeling. Reconstructing in feature space significantly boosts performance over pixel reconstruction, with a +12.1% gain on spatially-focused tasks like K400m and +5.1% on motion-focused tasks like SSv2m. This sug- 

Table 3. **Full finetuning comparison of various self-supervised methods on Kinetics-400** . All results are obtained with a ViTB backbone. _∗_ denotes results obtained by our evaluation. Our SMILE achieves state-of-the-art performance. Top-1 accuracies (%) are reported. 

|**Method**|**Backbone **|**Epochs **|**Pretrain **|**Top-1**|
|---|---|---|---|---|
|SVT [64]|ViT-B|-|K400|78.4|
|VideoMAE [77]|ViT-B|800|K400|80.0|
|BEVT [82]|ViT-B|-|K400|80.6|
|OmniMAE [23]|ViT-B|800|K400|80.8|
|MGM [16]|ViT-B|800|K400|80.8|
|MGMAE [29]|ViT-B|800|K400|81.2|
|SIGMA [68]|ViT-B|800|K400|81.5|
|MME<sup>_∗_</sup>[72]|ViT-B|800|K400|81.5|
|SMILE (ours)|ViT-B|600|K400|**83.1**|



gests that using a high-level feature space as the reconstruction target provides more meaningful supervision for learning transferable spatio-temporal representations. **Reconstructing which features:** **_<u>CLIP</u>_** **<u>,</u>** **_MAE_ or** **_DINO_ ?** Table 4b compares several image foundation models—MAE [27], DINO [7], DINO-v2 [55], and CLIP [63]—as reconstruction targets. CLIP features outperform the others, likely due to CLIP’s image-text alignment training on a large dataset, enhancing its transferability and adaptability to video tasks as shown in [52, 65, 81]. Compared to pixel reconstruction in Table 4h, all featurebased approaches show substantial performance gains. **Trajectory-based masking:** **_<u>yes</u>_ or** **_no?_** Table 4g shows the impact of our trajectory-based masking. Without trajectory masking, all tokens are masked using tube masking, which masks the whole video without distinguishing between object and non-object space-time tokens. In contrast, trajectory masking first masks space-time tokens associated with objects, followed by tube masking on the remaining tokens, maintaining the overall masking ratio. Our trajectory masking improves performance by approximately 1% on both K400m and SSv2m, validating that masking along object trajectories enhances motion-aware representation learning. **How much to mask?** Table 4e shows the impact of masking ratio _m_ on downstream performance. We mask _m_ of space-time tubes within the objects and _m_ of space-time tubes in the whole video with _m_ being 95% _,_ 90% _,_ or 80%. We observe the best performance with _<u>m</u>_ =80%. 

**How many synthetic objects?** In Table 4f, we analyze the impact of adding more object motions to the input video on downstream performance. We evaluate adding 1 _,_ 2 and 3 object motions and compare the results. We observe that adding 2 objects is better than using a single object and the performance saturates when adding more objects. 

**Which object transformations?** In Table 4c, we show the impact of object transformation on downstream performance. Both rotation and scaling improve the performance individually and combining them achieves the best results. 

|Synth.|Target|K400m|SSv2m|Target|K400m|SSv2m|Transform|K400m|SSv2m|Training|K400m|SSv2m|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|w/o|Pixels|66.0|55.1|MAE [27]|77.6|59.5|None|77.0|60.8|Single|78.9|61.6|
|w/|Pixels|**68.1**|**56.8**|DINO [7]|77.6|59.7|Scaling|78.4|61.3|Mix|79.5|62.2|
|w/o|Features|78.1|60.2|DINOv2 [5|5]<br>77.8|59.7|Rotation|78.2|61.4|Progressive|**79.8**|**62.5**|
|w/|Features|**78.9**|**61.6**|CLIP [63]|**78.1**|**60.2**|Both|**78.9**|**61.6**|(d) **Which **|**training **|**strategy?**|
|(a)**Syn**|**thetic moti**|**on,****_yes_ o**|**r****_no_?** Yes.|(b) **Reconst**|**ructing w**|**hich fea-**|(c) **Which o**|**bject** **_tran_**|**_sforma-_**|Progressive t|raining o|utperforms|
|It impro<br>and fea|ves perfor<br>ture targets.|mance for<br>|both pixel|**tures?** CLIP <br>best target. N|features p<br>o motion i|rovide the<br>s added.|**_tions_?** Usin<br>scaling wor|g both rota<br>ks the best.|tion and|both for the sa<br>tions.|me numb|er of itera-|
|Ratio|Target|K400m|SSv2m|# Objects|K400m|SSv2m|Masking|K400|m<br>SSv2m|Target|K400m|SSv2m|
|95%|features|77.7|60.4|1|78.6|61.0|w/o Traject|ory<br>77.|8<br>60.7|Pixels|66.0|55.1|
|90%|features|78.9|61.6|2|**78.9**|**61.6**|w/ Trajecto|ry<br>**78.**|**9**<br>**61.6**|Features|**78.1**|**60.2**|
|80%|features|**80.3**|**62.0**|3|78.6|61.3|(g) **Trajecto**|**ry maskin**|**g,** **_yes_ or**|(h)**Reconst**|**ructing****_p_**|**_ixels_ or****_fea-_**|
|(e) **How**|**much to **|**mask?**|Masking|(f) **How man**|**y** **_objects_?**|Adding|**_no_?** Yes.|Both target|s are fea-|**_tures_?** Reco|nstructin|g features is|
|80% traj|ectory and t|otal toke|ns.|two object mo|tions is th|e best.|tures.|||much better.|No moti|on is added.|



Table 4. **Ablation experiments of our proposed SMILE.** The default setting uses clip reconstruction targets with two synthetic objects and a 90% masking ratio with trajectory-based masking. We also show full-scale ablations with ViT-B and K400 pretraining in **supplementary** . 

**Which training strategy?** Table 4d compares our proposed training strategies. We also compare learning with the single learning strategy, where only augmented videos are used for training. We observe that both mixed and progressive learning are better than learning from augmented videos only. Progressive is better than mixed, most likely due to a reduced domain shift in the second stage where only original videos are used. 

**Qualitative analysis.** To compare the temporal dynamics of video representations learned by different SSL methods, we compute their feature similarity across different frames of input videos (Figure 4). For our model, the features of different frames have larger differences, indicating better temporal awareness. More plots are in the **supplementary** . 



<!-- Start of picture text -->
VideoMAE MGMAE SMILE(Ours)<br><!-- End of picture text -->

Figure 4. **Feature similarity across different frames for different SSL methods.** We compute this on K400 validation videos. 

### **4.4. Generalization Assessment** 

To demonstrate the generalization of our learned video representations to diverse downstream video setups, we evaluate SMILE on the SEVERE benchmark [75], which includes 8 experiments targeting 4 key generalization factors: _domain shift_ , _sample efficiency_ , _action granularity_ , and _task shift_ . Detailed configurations are provided in the **supplementary** . Beyond SEVERE, we also show generalization to more temporal tasks in **supplementary** . 

Table 6 shows the results. For the **Domain Shift** , we test 

cross-domain adaptability on SSv2 and Gym99, which differ from the pretraining dataset (K400). Our method significantly outperforms all others, demonstrating superior adaptability to domain shifts. For **Sample Efficiency** , we evaluate low-shot action recognition on UCF and GYM with only 1,000 samples for finetuning. Our approach surpasses previous methods, achieving a 5% improvement on GYM over the best prior model, MME, showcasing its strong low-shot recognition capability even in motion-sensitive tasks. For the **Action Granularity** , tested on FX-S1 and UB-S1 from FineGym, our model achieves the best results, excelling at distinguishing fine-grained motions within gymnastics routines (e.g., jump variations in FX-S1). It outperforms motion-aware models like MME, MGMAE, and MGM, highlighting superior motion generalizability. For **Task Shift** , evaluated with temporal repetition counting on UCFRC [94] and multi-label recognition on Charades [70], our method matches prior performance on repetition counting and achieves a 9% improvement over MME on Charades, demonstrating broad task adaptability. Overall, SMILE (without synthetic motion) is on par with the prior best with the mean performance of 64.9%. With synthetic motion SMILE, we observe a further 3.0% gain, reinforcing our model’s generalization strength through synthetic motion. 

## **5. Discussions** 

**Comparison with Other CLIP Adaptations.** Table 5 presents a comparison between our SMILE and the CLIP adaptation methods ViCLIP [84] and UMT [42], along with the CLIP baseline. Our method achieves superior performance on motion-sensitive datasets, particularly in linear probing, indicating that our CLIP adaptation learns better video representations than the video-text alignment of ViCLIP and the unmasked feature alignment of UMT, despite using less data. More results are in the **supplementary** . 

|**Pre**|**training**||**Finet**|**uning**||**L**|**inear**|**Probin**|**g**|
|---|---|---|---|---|---|---|---|---|---|
|**Method**|Video Dataset|UCF|GYM|SSv2|EPIC|UCF|GYM|SSv2|EPIC|
|CLIP [63]|-|93.6|88.0|66.7|50.3|77.5|20.7|11.3|25.1|
|ViCLIP [84]|Intervid-10M [84]|95.2|89.7|67.9|55.0|86.7|27.3|18.9|27.3|
|UMT [42]|K700 [8]|96.0|89.9|70.1|50.1|**88.0**|26.4|18.8|28.2|
|SMILE (ours)|K400 [35]|**96.4**|**90.8**|**71.9**|**63.3**|83.8|**30.2**|**23.7**|**34.6**|



Table 5. **Comparision with CLIP adaptations for video representation learning.** Despite using less data, we learn better video representations than other CLIP adaptations, especially for motion-focused downstream domains. 

Table 6. **Generalization assessment on SEVERE benchmark [75]** for recent video SSL methods in terms of domain shift, sample efficiency, action granularity, and task shift. We evaluate prior works with the official ViT-B backbone checkpoints pretrained on the K400 dataset. SMILE shows the best generalization performance across setups. Adding synthetic motions further improves the performance. 

|**Method**|**Dom**|**ain shift**|**Sample ei**|**fficiency (**10<sup>3</sup>**)**|**Action g**|**ranularity**|**Task**|**shift**|**Mean**|
|---|---|---|---|---|---|---|---|---|---|
||SSv2|Gym99|UCF|GYM|FX-S1|UB-S1|UCF-RC_↓_|Charades||
|VideoMAE [77]|68.6|86.6|74.6|25.9|42.8|65.3|0.172|14.4|57.6|
|MVD [83]|70.0|82.5|67.1|17.5|31.3|50.5|0.184|16.1|52.1|
|MGMAE [29]|68.9|87.2|77.2|24.1|33.7|79.5|0.181|17.9|58.8|
|MGM [16]|71.1|89.1|78.4|26.4|38.6|86.9|**0.152**|22.5|62.2|
|SIGMA [68]|70.9|89.7|84.1|28.0|55.1|79.9|0.169|23.1|64.2|
|MME [72]|70.1|89.7|79.2|29.8|**55.5**|87.2|0.155|23.6|65.0|
|SMILE w/o motion|71.6|90.0|85.2|32.0|53.4|74.3|0.175|30.5|64.9|
|SMILE (ours)|**72.1**|**90.8**|**86.4**|**35.1**|55.1|**88.3**|0.170|**32.5**|**67.9**|



Table 7. **Learning without natural videos.** We train a ViT-S for pixel reconstruction. Our synthetic object motions can learn video representations without using any natural videos. 

|**Unnatural Video Scheme**|Pretrain Data|K400m|SSv2m|
|---|---|---|---|
|No Pretraining|-|39.7|28.2|
|Video-frame|K400m-1frm|54.0|43.1|
|Video-frame + Motion|K400m-1frm<br>|63.6<br>|53.2<br>|
|Image|Places-80K|52.1|40.5|
|Image + Motion|Places-80K|60.1|52.1|
|Black image + Motion|-|59.2|50.0|
|Noise image + Motion|-|61.9|51.2|



**Learning without Natural Videos.** In prior experiments, we overlay our object motions onto natural videos from the Kinetics-400 dataset, raising the question of whether downstream improvements stem from original video priors (motion, action, scene) or our added object motions. To investigate this, we construct videos by overlaying object motions onto four different backgrounds: (1) a single frame from natural videos, (2) a single scene image from the image dataset Places [98], (3) a plain black image, and (4) a noise image. Each background image is duplicated across _T_ frames, creating static clips onto which moving objects are placed. For (1), we select one frame per video from K400m (denoted K400m-1frm), preserving action and scene cues without motion priors. For (2), we sample 80K images from Places (Places-80K), matching the dataset size of K400m-1frm, to remove action and motion priors. Approaches (3) and (4) exclude all natural video or image priors entirely. Evaluation involves full finetuning on K400m 

and SSv2m, consistent with our previous ablation studies. 

Table 7 reports results for different augmented video types. Adding object motions consistently enhances downstream performance, whether combined with action and scene cues from natural videos (row 2 vs. row 3) or only scene cues from images (row 4 vs. row 5). Even with black or noisy backgrounds (rows 6 and 7), performance remains robust. Remarkably, using object motions alone yields substantial improvements—22.2% on K400m and 23.0% on SSv2m compared to no pretraining. These findings demonstrate our method’s effectiveness in learning video representations from synthetic motions alone. More results are in the **supplementary** . In summary, our method enables a new paradigm of learning strong video representations with masked video modeling without relying on natural videos. 

## **6. Conclusions** 

In this work, we proposed SMILE, a novel self-supervised approach for video representation learning that addresses key limitations in existing masked video modeling methods by integrating both spatial and motion semantics. Our SMILE, leverages the high-level spatial semantics from image-language pretrained models like CLIP and enhances motion representation through synthetic motion patterns. By capturing complex spatial features and motion dynamics, SMILE achieves superior performance on diverse downstream tasks, demonstrating improved discrimination and generalization over state-of-the-art video SSL methods. Our extensive evaluations demonstrated the advantages of our SMILE, setting a new paradigm for robust and effective video self-supervised learning. 

**Acknowledgements.** This work is supported by the KAUST Center of Excellence for Generative AI under award number 5940. For computing time, this research used Ibex managed by the Supercomputing Core Laboratory at King Abdullah University of Science & Technology (KAUST) in Thuwal, Saudi Arabia. We also thank Michael Dorkenwald for providing the synthetic objects used in this work. 

## **References** 

- [1] Unaiza Ahsan, Rishi Madhok, and Irfan Essa. Video jigsaw: Unsupervised learning of spatiotemporal context for video action recognition. In _Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)_ , 2019. 2 

- [2] Humam Alwassel, Dhruv Mahajan, Bruno Korbar, Lorenzo Torresani, Bernard Ghanem, and Du Tran. Self-supervised learning by cross-modal audio-video clustering. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2020. 2 

- [3] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. _arXiv preprint arXiv:2106.08254_ , 2022. 19 

- [4] Sagie Benaim, Ariel Ephrat, Oran Lang, Inbar Mosseri, William T Freeman, Michael Rubinstein, Michal Irani, and Tali Dekel. Speednet: Learning the speediness in videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2020. 2 

- [5] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? _arXiv preprint arXiv:2102.05095_ , 2021. 14, 15 

- [6] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. ActivityNet: A large-scale video benchmark for human activity understanding. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2015. 15 

- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2021. 6, 7 

- [8] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. _arXiv preprint arXiv:1907.06987_ , 2019. 8, 17 

- [9] Peihao Chen, Deng Huang, Dongliang He, Xiang Long, Runhao Zeng, Shilei Wen, Mingkui Tan, and Chuang Gan. Rspnet: Relative speed perception for unsupervised video representation learning. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2021. 2 

- [10] Hyeon Cho, Taehoon Kim, Hyung Jin Chang, and Wonjun Hwang. Self-supervised visual learning by variable playback speeds prediction of a video. _IEEE Access_ , 9:79562–79571, 2021. 2 

- [11] Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Randaugment: Practical automated data augmentation with a reduced search space. _arXiv preprint arXiv:1909.13719_ , 2019. 19 

- [12] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, , Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for EPIC-KITCHENS-100. _International Journal of Computer Vision (IJCV)_ , 2021. 2, 5, 18, 19 

- [13] Shuangrui Ding, Maomao Li, Tianyu Yang, Rui Qian, Haohang Xu, Qingyi Chen, Jue Wang, and Hongkai Xiong. Motion-aware contrastive video representation learning via foreground-background merging. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 2 

- [14] Michael Dorkenwald, Nimrod Barazani, Cees GM Snoek, and Yuki M Asano. Pin: Positional insert unlocks object localisation abilities in vlms. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2024. 5, 19 

- [15] Hazel Doughty, Fida Mohammad Thoker, and Cees GM Snoek. Locomotion: Learning motion-focused videolanguage representations. In _Proceedings of the Asian Conference on Computer Vision (ACCV)_ , pages 50–70, 2024. 2 

- [16] David Fan, Jue Wang, Shuai Liao, Yi Zhu, Vimal Bhat, Hector Santos-Villalobos, Rohith MV, and Xinyu Li. Motionguided masking for spatiotemporal representation learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023. 1, 2, 3, 5, 6, 8, 14, 15, 16 

- [17] David Fan, Jue Wang, Shuai Liao, Zhikang Zhang, Vimal Bhat, and Xinyu Li. Text-guided video masked autoencoder. In _European Conference on Computer Vision (ECCV)_ , 2024. 2 

- [18] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2021. 15 

- [19] Christoph Feichtenhofer, Haoqi Fan, Bo Xiong, Ross Girshick, and Kaiming He. A large-scale study on unsupervised spatiotemporal representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 2 

- [20] Christoph Feichtenhofer, Yanghao Li, Kaiming He, et al. Masked autoencoders as spatiotemporal learners. _Advances in neural information processing systems_ , 35:35946–35958, 2022. 14, 15 

- [21] Basura Fernando, Hakan Bilen, Efstratios Gavves, and Stephen Gould. Self-supervised video representation learning with odd-one-out networks. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017. 2 

- [22] Kirill Gavrilyuk, Mihir Jain, Ilia Karmanov, and Cees G M Snoek. Motion-augmented self-training for video recognition at smaller scale. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2021. 2 

- [23] Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 

Omnimae: Single model masked pretraining on images and videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 6, 14, 15 

- [24] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The ”something something” video database for learning and evaluating visual common sense. In _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_ , 2017. 1, 2, 5, 18, 19 

- [25] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2019. 19 

- [26] Tengda Han, Weidi Xie, and Andrew Zisserman. Selfsupervised co-training for video representation learning. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2020. 2 

- [27] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 2, 6, 7, 14 

- [28] Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment your batch: Improving generalization through instance repetition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8129–8138, 2020. 19 

- [29] Bingkun Huang, Zhiyu Zhao, Guozhen Zhang, Yu Qiao, and Limin Wang. Mgmae: Motion guided masking for video masked autoencoding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023. 1, 2, 3, 5, 6, 8, 14, 15, 16 

- [30] Ziyuan Huang, Shiwei Zhang, Jianwen Jiang, Mingqian Tang, Rong Jin, and Marcelo H Ang. Self-supervised motion learning from static images. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 2 

- [31] Sunil Hwang, Jaehong Yoon, Youngwan Lee, and Sung Ju Hwang. Everest: Efficient masked video autoencoder by removing redundant spatiotemporal tokens. In _International Conference on Machine Learning (ICML)_ , 2024. 5 

- [32] Simon Jenni, Givi Meishvili, and Paolo Favaro. Video representation learning by recognizing temporal transformations. In _European Conference on Computer Vision (ECCV)_ , 2020. 2 

- [33] Y.-G. Jiang, J. Liu, A. Roshan Zamir, G. Toderici, I. Laptev, M. Shah, and R. Sukthankar. THUMOS challenge: Action recognition with a large number of classes. http: //crcv.ucf.edu/THUMOS14/, 2014. 15 

- [34] Longlong Jing, Xiaodong Yang, Jingen Liu, and Yingli Tian. Self-supervised spatiotemporal feature learning via video rotation prediction. _arXiv preprint arXiv:1811.11387_ , 2018. 2 

- [35] Will Kay, Jo˜ao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, 

and Andrew Zisserman. The kinetics human action video dataset. _arXiv preprint arXiv:1705.06950_ , 2017. 1, 2, 5, 8, 17, 18, 19 

- [36] Dahun Kim, Donghyeon Cho, and In So Kweon. Selfsupervised video representation learning with space-time cubic puzzles. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2019. 2 

- [37] Okan K¨op¨ukl¨u, Xiangyu Wei, and Gerhard Rigoll. You only watch once: A unified CNN architecture for realtime spatiotemporal action localization. _arXiv preprint arXiv:1911.06644_ , 2019. 1 

- [38] Hildegard Kuehne, Hueihan Jhuang, Est´ıbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_ , 2011. 2, 5, 18 

- [39] Harold W Kuhn. The hungarian method for the assignment problem. _Naval research logistics quarterly_ , 2(1-2):83–97, 1955. 14 

- [40] Hsin-Ying Lee, Jia-Bin Huang, Maneesh Singh, and MingHsuan Yang. Unsupervised representation learning by sorting sequences. In _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_ , 2017. 2 

- [41] Kunchang Li, Yali Wang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatiotemporal representation learning. _arXiv preprint arXiv:2201.04676_ , 2022. 14, 15 

- [42] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023. 2, 7, 8, 16, 17 

- [43] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 14, 15 

- [44] Shuming Liu, Chen-Lin Zhang, Chen Zhao, and Bernard Ghanem. End-to-end temporal action detection with 1b parameters across 1000 frames. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2024. 1 

- [45] Shuming Liu, Chen Zhao, Fatimah Zohra, Mattia Soldan, Alejandro Pardo, Mengmeng Xu, Lama Alssum, Merey Ramazanova, Juan Le´on Alc´azar, Anthony Cioppa, Silvio Giancola, Carlos Hinojosa, and Bernard Ghanem. Opentad: A unified framework and comprehensive study of temporal action detection. _arXiv preprint arXiv:2502.20361_ , 2025. 15 

- [46] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 15 

- [47] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2019. 19 

- [48] Chengze Lu, Xiaojie Jin, Zhicheng Huang, Qibin Hou, Ming-Ming Cheng, and Jiashi Feng. CMAE-V: contrastive 

masked autoencoders for video action recognition. _CoRR_ , abs/2301.06018, 2023. 14, 15 

- [49] Dezhao Luo, Chang Liu, Yu Zhou, Dongbao Yang, Can Ma, Qixiang Ye, and Weiping Wang. Video cloze procedure for self-supervised spatio-temporal learning. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , 2020. 

   - 2 

- [50] Ishan Misra, C Lawrence Zitnick, and Martial Hebert. Shuffle and learn: unsupervised learning using temporal order verification. In _European Conference on Computer Vision (ECCV)_ , 2016. 2 

- [51] Pedro Morgado, Nuno Vasconcelos, and Ishan Misra. Audiovisual instance discrimination with cross-modal agreement. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 2 

- [52] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. In _European Conference on Computer Vision (ECCV)_ , 2022. 2, 6, 16, 17 

- [53] Jingcheng Ni, Nan Zhou, Jie Qin, Qian Wu, Junqi Liu, Boxun Li, and Di Huang. Motion sensitive contrastive learning for self-supervised video representation. In _European Conference on Computer Vision (ECCV)_ , 2022. 2 

- [54] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 2 

- [55] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_ , 2023. 6, 7 

- [56] Tian Pan, Yibing Song, Tianyu Yang, Wenhao Jiang, and Wei Liu. Videomoco: Contrastive video representation learning with temporally adversarial examples. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 2 

- [57] Jungin Park, Jiyoung Lee, and Kwanghoon Sohn. Dual-path adaptation from image to video transformers. _arXiv preprint arXiv:2303.09857_ , 2023. 2, 16, 17 

- [58] Mandela Patrick, Yuki M. Asano, Polina Kuznetsova, Ruth Fong, Jo˜ao F. Henriques, Geoffrey Zweig, and Andrea Vedaldi. Multi-modal self-supervision from generalized data transformations. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2021. 2 

- [59] Mandela Patrick, Dylan Campbell, Yuki Asano, Ishan Misra, Florian Metze, Christoph Feichtenhofer, Andrea Vedaldi, and Joao F Henriques. Keeping your eye on the ball: Trajectory attention in video transformers. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2021. 14, 15 

- [60] AJ Piergiovanni, Anelia Angelova, and Michael S Ryoo. Evolving losses for unsupervised video representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2020. 1 

- [61] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. _arXiv preprint arXiv:1704.00675_ , 2017. 14 

- [62] Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. Spatiotemporal contrastive video representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 1, 2 

- [63] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning (ICML)_ , 2021. 2, 5, 6, 7, 8, 17 

- [64] Kanchana Ranasinghe, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan, and Michael S. Ryoo. Self-supervised video transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 2, 6 

- [65] Hanoona Rasheed, Muhammad Uzair Khattak, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Fine-tuned clip models are efficient video learners. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 2, 6, 16, 17 

- [66] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 4, 5, 19 

- [67] Mohammadreza Salehi, Efstratios Gavves, Cees G. M. Snoek, and Yuki M Asano. Time does tell: Self-supervised time-tuning of dense image representations. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023. 14 

- [68] Mohammadreza Salehi, Michael Dorkenwald, Fida Mohammad Thoker, Efstratios Gavves, Cees GM Snoek, and Yuki M Asano. Sigma: Sinkhorn-guided masked video modeling. In _European Conference on Computer Vision (ECCV)_ , 2025. 5, 6, 8, 14, 15, 16 

- [69] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2020. 2, 5, 18, 19 

- [70] Gunnar A. Sigurdsson, G¨ul Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding. In _European Conference on Computer Vision (ECCV)_ , 2016. 1, 2, 7, 19 

- [71] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. _arXiv preprint arXiv:1212.0402_ , 2012. 1, 2, 5, 18, 19 

- [72] Xinyu Sun, Peihao Chen, Liangwei Chen, Changhao Li, Thomas H. Li, Mingkui Tan, and Chuang Gan. Masked motion encoding for self-supervised video representation learn- 

ing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 1, 2, 5, 6, 8, 14, 15 

- [73] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. _arXiv preprint arXiv:1512.00567_ , 2015. 19 

- [74] Fida Mohammad Thoker, Hazel Doughty, and Cees GM Snoek. Skeleton-contrastive 3d action representation learning. In _Proceedings of the ACM International Conference on Multimedia (ACMMM)_ , 2021. 2 

- [75] Fida Mohammad Thoker, Hazel Doughty, Piyush Bagad, and Cees GM Snoek. How severe is benchmark-sensitivity in video self-supervised learning? In _European Conference on Computer Vision (ECCV)_ , 2022. 1, 2, 5, 7, 8, 19, 20 

- [76] Fida Mohammad Thoker, Hazel Doughty, and Cees G. M. Snoek. Tubelet-contrastive self-supervision for videoefficient generalization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023. 2 

- [77] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 1, 2, 3, 5, 6, 8, 14, 15, 16, 17, 19, 20 

- [78] Shuyuan Tu, Qi Dai, Zuxuan Wu, Zhi-Qi Cheng, Han Hu, and Yu-Gang Jiang. Implicit temporal modeling with learnable alignment for video recognition. _arXiv preprint arXiv:2304.10465_ , 2023. 2, 16, 17 

- [79] Guangting Wang, Yizhou Zhou, Chong Luo, Wenxuan Xie, Wenjun Zeng, and Zhiwei Xiong. Unsupervised visual representation learning by tracking patches in video. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 1, 2 

- [80] Jinpeng Wang, Yuting Gao, Ke Li, Yiqi Lin, Andy J Ma, Hao Cheng, Pai Peng, Feiyue Huang, Rongrong Ji, and Xing Sun. Removing the background by adding the background: Towards background robust self-supervised video representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 2 

- [81] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip: A new paradigm for video action recognition. _arXiv preprint arXiv:2109.08472_ , 2021. 2, 6 

- [82] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, and Lu Yuan. Bevt: Bert pretraining of video transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 6, 15 

- [83] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, and Yu-Gang Jiang. Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 5, 8, 14, 15 

- [84] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui 

Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. _arXiv preprint arXiv:2307.06942_ , 2023. 2, 7, 8, 16, 17 

- [85] Fanyi Xiao, Joseph Tighe, and Davide Modolo. Maclr: Motion-aware contrastive learning of representations for videos. In _European Conference on Computer Vision (ECCV)_ , 2022. 2 

- [86] Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and Yueting Zhuang. Self-supervised spatiotemporal learning via video clip order prediction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2019. 2 

- [87] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. _arXiv preprint arXiv:1809.03327_ , 2018. 14 

- [88] Haosen Yang, Deng Huang, Bin Wen, Jiannan Wu, Hongxun Yao, Yi Jiang, Xiatian Zhu, and Zehuan Yuan. Selfsupervised video representation learning with motion-aware masked autoencoders. _arXiv preprint arXiv:2210.04154_ , 2022. 2, 14, 15 

- [89] Taojiannan Yang, Yi Zhu, Yusheng Xie, Aston Zhang, Chen Chen, and Mu Li. Aim: Adapting image models for efficient video action recognition. _arXiv preprint arXiv:2302.03024_ , 2023. 2, 16, 17 

- [90] Yuan Yao, Chang Liu, Dezhao Luo, Yu Zhou, and Qixiang Ye. Video playback rate perception for self-supervised spatio-temporal representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2020. 2 

- [91] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. _arXiv preprint arXiv:1905.04899_ , 2019. 19 

- [92] Chenlin Zhang, Jianxin Wu, and Yin Li. ActionFormer: Localizing moments of actions with transformers. In _European Conference on Computer Vision (ECCV)_ , 2022. 15 

- [93] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. _arXiv preprint arXiv:1710.09412_ , 2018. 19 

- [94] Huaidong Zhang, Xuemiao Xu, Guoqiang Han, and Shengfeng He. Context-aware and scale-insensitive temporal repetition counting. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2020. 7, 19 

- [95] Chen Zhao, Shuming Liu, Karttikeya Mangalam, and Bernard Ghanem. Re<sup>2</sup> TAL: Rewiring pretrained video backbones for reversible temporal action localization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 15 

- [96] Hanqing Zhao, Dianmo Sheng, Jianmin Bao, Dongdong Chen, Dong Chen, Fang Wen, Lu Yuan, Ce Liu, Wenbo Zhou, Qi Chu, et al. X-paste: Revisiting scalable copypaste for instance segmentation using clip and stablediffusion. In _International Conference on Learning Representations (ICLR)_ , 2023. 4, 5, 19 

- [97] Long Zhao, Nitesh B Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, Jennifer J Sun, Luke Friedman, Rui Qian, 

Tobias Weyand, Yue Zhao, et al. Videoprism: A foundational visual encoder for video understanding. _arXiv preprint arXiv:2402.13217_ , 2024. 2 

- [98] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 40(6):1452–1464, 2018. 8 

# **Appendix** 

In this appendix, we present supplementary analysis and detailed experimental validations. Section A provides an extensive state-of-the-art comparison for finetuning performance on K400 and SSv2 datasets. Section B further evaluates the generalization capability of SMILE on additional downstream video tasks. A detailed comparison against prior CLIP adaptation methods is provided in Section C. Section D shows more results for learning without natural videos. In Section E, we include additional experiments, such as performance under fixed training budgets, extended ablation studies with larger datasets, and qualitative visualizations. Finally, Section F specifies dataset details and clearly outlines the training and evaluation procedures used throughout our experiments. 

## **A. Extensive Comparison on K400 and SSv2** 

In the main paper, due to the space limit, we compare only with prior self-supervised methods using the same pretraining setup—specifically, a ViT-B backbone trained for 600 epochs on K400 and 800 on SSv2 datasets. Here, we provide a broader comparison, including self-supervised methods with varying pretraining setups and numerous supervised methods. Results for K400 and SSv2 finetuning are presented in Table 8 and Table 9, respectively. 

Our method not only reaches state of the art among all self-supervised methods but also matches or even surpasses many supervised methods that use specialized backbones, such as the hierarchical 3D transformer in MViTv2 [43] (more than +0.2% on K400 and +1.6% on SSv2) and the multi-head relation aggregation in Uniformer [41] (more than +0.1% on K400, +0.9% on SSv2), highlighting its ability to learn superior representations without heavily customized architectures. Additionally, we significantly outperform supervised methods using the same backbone (ViTB), e.g. TimeSformer [5] by more than +2.4% on K400 and +12.6% on SSv2 and Mformer [59] by more than +3.4% on K400 and +5.4% on SSv2. This demonstrates our method’s strength in capturing better spatio-temporal dynamics than these supervised approaches. 

Our approach significantly improves on VideoMAE [77] baseline, achieving gains of +3.6% on K400 and +4.9% on SSv2 for ViT-S, when trained for 800 epochs. Similarly, for ViT-B, it achieves an improvement of +3.1% on K400 and +3.6% on SSv2, when trained for 600 epochs on K400. Furthermore, our method outperforms all prior selfsupervised approaches under similar pretraining settings, including the same backbone, dataset, and training epochs (e.g., K400 on ViT-B for 600/800 epochs). This includes surpassing methods such as CV-MAE-V [48], which employs contrastive video masked autoencoding, SIGMA [68], which uses Sinkhorn-Guided feature clustering, and Om- 

niMAE [23], which reconstructs from both images and videos. These results underline the effectiveness of our method across diverse self-supervised learning strategies. 

The tables also compare our method with approaches that use significantly longer training schedules, such as MVD [83], ST-MAE [20], and MotionMAE [88]. MVD achieves strong performance but relies on a resourceintensive pipeline, involving 1200 epochs of pretraining for both VideoMAE [77] and MAE [27], followed by 400 epochs of distillation, making direct comparisons challenging. Despite training for only 600 epochs, our method surpasses MVD [83] by 0.4% on K400 finetuning and consistently outperforms other methods trained for 1600 epochs on K400 or 2400 epochs on SSv2. This includes motionaware methods like MotionMAE [88], MME [72], MGMAE [29], and MGM [16], demonstrating the superior training efficiency of our approach. When trained for 1200 epochs SMILE achieves a boost of 0 _._ 3% on both K400 and SSv2 finetuning showing the scalability of our method for longer training schedules. 

To summarize, our method surpasses many supervised methods and achieves state-of-the-art performance among video SSL methods while maintaining training efficiency. 

## **B. Generalization to More Temporal Tasks** 

In the main paper, we show the generalization capability of our method for diverse downstream settings in SEVEREbenchmark. Here we show the generalization of our method to more video understanding tasks. In particular, we evaluate temporally aware tasks: Unsupervised Video Object Segmentation ( **Un-VOS** ) and Temporal Action Localization ( **TAL** ). The goal is to evaluate the motion modeling capability of video representations with **TAL** requiring motion boundary awareness and **Un-VOS** requiring object motion propagation modeling. 

### **B.1. Unsupervised Video Object Segmentation** 

**Setup.** We adopt the evaluation approach from [68] to assess the learned temporal and spatial features via unsupervised video object segmentation using the benchmark introduced by [67]. Unlike conventional action recognition benchmarks that aggregate features into a single global representation, this task examines the encoder’s capability to generate consistent temporal object segmentation maps. Specifically, extracted space-time features are grouped using k-means clustering with a given number of clusters ( _K_ ), then aligned to ground-truth object masks via the Hungarian algorithm [39]. Segmentation accuracy is quantified through mean Intersection over Union (mIoU). The scenario is labeled as clustering when _K_ equals the actual object count and as overclustering when _K_ surpasses this number. We follow the implementation from [68] and report mIoU on **DAVIS** [61] and **YTVOS** [87]. 

Table 8. **Detailed comparison with supervised and selfsupervised pretraining methods for full finetuning on Kinetics400 (K400)** . _∗_ denotes results obtained by our evaluation. Params denote the number of parameters in millions. Our SMILE outperforms many supervised methods, achieves the best performance among self-supervised methods, and demonstrates a faster convergence. 

|**Method**|**Backbone**|**Epochs **|**Pretrain**|**Top-1**|**Params**|
|---|---|---|---|---|---|
|_supervised_||||||
|Mformer [59]|Mformer-B|-|K400|79.7|109|
|VideoSwin [46]|Swin-B|-|K400|80.6|88|
|TimeSformer [5]|ViT-B|-|K400|80.7|430|
|MViTv1 [18]|MViTv1-B|-|K400|80.2|37|
|MViTv2 [43]|MViTv2-B|-|K400|82.9|52|
|Uniformer-B[41]|Uformer-B|-|K400|83.0|50|
|_self-supervised_||||||
|VideoMAE<sup>_∗_</sup>[77]|ViT-S|800|K400|75.9|22|
|VideoMAE [77]|ViT-S|1600|K400|79.0|22|
|SMILE(ours)|ViT-S|800|K400|**79.5**|22|
|VideoMAE [77]|ViT-B|800|K400|80.0|87|
|VideoMAE [77]|ViT-B|1600|K400|81.5|87|
|ST-MAE [20]|ViT-B|1600|K400|81.3|87|
|MVD [83]|ViT-B|1600+400|K400|82.7|87|
|MotionMAE [88]|ViT-B|1600|K400|81.7|87|
|CMAE-V [48]|ViT-B|800|K400|80.2|87|
|CMAE-V [48]|ViT-B|1600|K400|80.9|87|
|BEVT [82]|ViT-B|800+150|K400|80.6|87|
|OmniMAE [23]|ViT-B|800|K400|80.8|87|
|SIGMA [68]|ViT-B|800|K400|81.5|87|
|MGM [16]|ViT-B|800|K400|80.8|87|
|MGM [16]|ViT-B|1600|K400|81.7|87|
|MME<sup>_∗_</sup>[72]|ViT-B|800|K400|81.5|87|
|MME [72]|ViT-B|1600|K400|81.8|87|
|MGMAE [29]|ViT-B|800|K400|81.2|87|
|MGMAE [29]|ViT-B|1600|K400|81.8|87|
|SMILE (ours)|ViT-B|600|K400|**83.1**|87|
|SMILE (ours)|ViT-B|1200|K400|**83.4**|87|



**Results.** As shown in Table 10, SMILE obtains the best segmentation performance across all settings except for DAVIS clustering where it is the second best. In particular, we outperform prior motion modeling methods MGM and MGMAE by 4% and 6% on YTVOS clustering and by 6% and 7% on YTVOS overclustering. Interestingly, we also beat SIGMA which explicitly clusters the reconstructed features via Sinkhornkoop clustering. This demonstrates the superior motion modeling capability of our approach over standard pixel reconstruction, motion-guided pixel reconstruction, and feature clustering reconstruction approaches. 

### **B.2. Temporal Action Localization** 

**Setup.** Temporal action localization (TAL) [45, 92, 95] is a task that aims to identify categories of actions that occur in a video and to locate the start and end timestamps of all action instances. It requires the model to understand not only 

Table 9. **Detailed comparison with supervised and selfsupervised pretraining methods for full finetuning on Something-Something V2 (SSv2)** . _∗_ denotes results obtained by our evaluation. Params denote the number of parameters in millions. Our SMILE outperforms many supervised methods, achieves the best performance among self-supervised methods, and demonstrates a faster convergence. 

|**Method**|**Backbone**|**Epochs  **|**Pretrain**|**Top-1**|**Params**|
|---|---|---|---|---|---|
|_supervised_||||||
|Mformer [59]|Mformer-B|-|K400|66.7|109|
|VideoSwin [46]|Swin-B|-|K400|69.6|88|
|TimeSformer [5]|ViT-B|-|K400|59.5|121|
|MViTv1 [18]|MViTv1-B|-|K400|67.7|37|
|MViTv2 [43]|MViTv2-B|-|K400|70.5|52|
|Uniformer-B[41]|Uformer-B|-|K400|71.2|50|
|_self-supervised_||||||
|OmniMAE [23]|ViT-B|800|SSv2|69.5|87|
|VideoMAE [77]|ViT-B|800|SSv2|69.6|87|
|VideoMAE [77]|ViT-B|2400|SSv2|70.8|87|
|CMAE-V [48]|ViT-B|800|SSv2|69.7|87|
|CMAE-V [48]|ViT-B|1600|SSv2|70.5|87|
|MME [72]|ViT-B|800|SSv2|70.0|87|
|MGM [16]|ViT-B|800|SSv2|70.6|87|
|MGM [16]|ViT-B|2400|SSv2|72.1|87|
|SIGMA [68]|ViT-B|800|SSv2|71.2|87|
|MGMAE [29]|ViT-B|800|SSv2|71.0|87|
|MGMAE [29]|ViT-B|2400|SSv2|72.3|87|
|SMILE(ours)|ViT-B|800|SSv2|**72.5**|87|
|VideoMAE<sup>_∗_</sup>[77]|ViT-S|800|K400|64.2|22|
|SIGMA [68]|ViT-S|800|K400|68.7|22|
|SMILE(ours)|ViT-S|800|K400|**69.1**|22|
|OmniMAE [23]|ViT-B|800|K400|69.0|87|
|VideoMAE [77]|ViT-B|800|K400|68.5|87|
|MVD [83]|ViT-B<br>|1600+400|K400|72.5|87|
|MME [72]|ViT-B|800|K400|70.5|87|
|SIGMA [68]|ViT-B|800|K400|71.1|87|
|MGMAE<sup>_∗_</sup>[29]|ViT-B|800|K400|68.9|87|
|MGM<sup>_∗_</sup>[16]|ViT-B|800|K400|71.1|87|
|SMILE (ours)|ViT-B|600|K400|**72.1**|87|
|SMILE (ours)|ViT-B|1200|K400|**72.4**|87|



the spatial semantics within the frames but also the temporal dynamics across frame sequences to capture the action process. We evaluated our SMILE as well as the comparative methods on two representative TAL benchmarks THUMOS-14 [33] and ActivityNet-v1.3 [6]. We used the pretrained models from each method as the backbones for video spatio-temporal feature extraction and finetuned them with the TAL method ActionFormer [92] on both datasets using the OpenTAD framework [45]. Following the common practice in the TAL community, we report the average mean average precision (mAP) over various temporal intersection of union (tIoU) values, i.e., 10 tIoU values [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] for 

||**Unsuperv**<br>**Cluste**|**ised Video**<br>**ring**|**Object Segm**<br>**Overclu**|**entation**<br>**stering**|**Temporal Act**|**ion Localization**|
|---|---|---|---|---|---|---|
|**Method**|YTVOS|DAVIS|YTVOS|DAVIS|THUMOS-14|ActivityNet-v1.3|
|VideoMAE [77]|34.1|29.5|61.3|56.2|58.5|37.3|
|MGM [16]|36.6|**36.5**|61.2|56.6|62.0|37.6|
|MGMAE [29]|34.5|31.0|60.1|57.5|56.3|37.3|
|SIGMA [68]|37.5|31.5|66.4|58.5|62.7|37.7|
|SMILE (ours)|**40.5**|32.7|**67.0**|**59.5**|**65.6**|**38.0**|



Table 10. **Generalization assessment on Unsupervised Video Object Segmentation and Temporal Action Localization.** All methods are evaluated on the ViT-B backbone pretrained on K400 with their publicly available checkpoints. SMILE outperforms prior masked video modeling works on both tasks demonstrating a better temporal modeling capability for more complex video understanding tasks. 

ActivityNet-v1.3 and 5 tIoU values [0.3, 0.4, 0.5, 0.6, 0.7] for THUMOS-14. 

**Results.** As shown in Table 10, our SMILE achieves the highest average mAPs on both benchmarks. Specifically, on THUMOS-14, SMILE outperforms some methods by a large margin _e.g_ . VideoMAE by 7% and MGMAE by 9%. On the more challenging and largescale dataset ActivityNet-v1.3, it shows 0.3% improvement over the second-best SIGMA. More notably, on ActivityNetv1.3, the performance of SMILE is on par with the state-ofthe-art TAL performance, which relies on fully supervised finetuning with labeled Kinetics-400 videos (not shown in the table). Overall, the results show that SMILE generalizes better to more complex downstream video tasks than the current masked video modeling approaches. 

a superior CLIP adaptation strategy which relies only on the unlabeled videos for adaptation. Moreover, as demonstrated in the main paper, SMILE surpasses adaptation methods that also employ intermediate pretraining, UMT, and ViCLIP. Notably, the performance gains are particularly pronounced on motion-intensive datasets like GYM99, SSv2, and EPIC, underscoring the critical importance of explicit motion modeling an aspect often neglected in existing CLIP adaptations. Our intuition is that CLIP features contain object information like shape, boundaries, and location, which guides the reconstruction task to focus on the overlaid objects as well as the original video semantics. 

In summary, SMILE provides a robust CLIP adaptation by reconstructing masked video inputs directly within the CLIP visual feature space while explicitly integrating synthetic motion cues. 

## **C. More comparisons with CLIP adaptations** 

In this section, we provide further comparisons between our proposed SMILE and existing CLIP adaptation methods for action recognition. Specifically, we consider two distinct categories of CLIP-based adaptations: **_without intermediate pretraining_** and **_with intermediate pretraining_** . The former approaches either jointly finetune the CLIP vision and text encoders using labeled video-text pairs of the target dataset (e.g., X-CLIP [52], ViFi-CLIP [65], ILA [78]) or directly finetune the CLIP vision encoder with added specialized spatio-temporal adaptation modules on labeled videos from the target dataset (e.g., AIM [89], DUAL Path [57]). In contrast, intermediate pretraining methods, such as UMT [42] and ViCLIP [84] use the CLIP model and intermediate video-text pairs for further pretraining to align video and text modality using contrastive learning. Detailed results of these comparisons are presented in Table 11. 

We observe that SMILE significantly outperforms prior adaptation methods that directly finetune the CLIP model on the target dataset. It achieves the best performance on all target datasets except K400, where it achieves comparable results. This highlights the effectiveness of our approach as 

## **D. Learning without Natural Videos** 

In the main paper, we show that our method can learn video representations by overlaying object motions on clips from natural videos, single frames from natural videos, single natural images, or even black and noise images. This raises the question about the effectiveness of using object motions alone and how they compare with learning from natural videos. To answer this we compare the performance of learning from object motions only with learning from natural video data. Specifically, we generate video clips by adding our synthetic object motions to randomly generated noise images. As in the main paper, we take a noise image, duplicate it _T_ times to form a static video clip, and then overlay objects with motion on top of it. We compare its performance with learning from natural videos of the K400 dataset. We use ViT-S and ViT-B backbones for this experiment and results are shown in Table 12. 

We observe that learning with such augmented videos significantly improves on no pretraining. Compared to the VideoMAE baseline which uses 240K natural videos of K400, learning from our object motions only with pixel 

||||**Finet**|**uning**||**L**|**inear**|**Probin**|**g**||
|---|---|---|---|---|---|---|---|---|---|---|
|**Method**|Intermediate Dataset|K400|UCF|GYM|SSv2|EPIC|UCF|GYM|SSv2|EPIC|
|CLIP[63]|-|81.8|93.6|88.0|66.7|50.3|77.5|20.7|11.3|25.1|
|**_Without intermediate pretraining_**|||||||||||
|X-CLIP [52]|-|83.8|92.0|75.2|57.4|52.7|-|-|-|-|
|ViFi-CLIP [65]|-|83.9|94.6|81.5|48.6|48.9|-|-|-|-|
|AIM [89]|-|83.9|94.0|90.3|66.4|58.4|-|-|-|-|
|ILA [78]|-|84.0|94.2|82.7|65.0|60.8|-|-|-|-|
|DUAL-Path[57]|-|**85.4**|-|-|69.6|-|-|-|-|-|
|**_With intermediate pretraining_**|||||||||||
|ViCLIP [84]|Intervid-10M [84]|82.4|95.2|89.7|67.9|55.0|86.7|27.3|18.9|27.3|
|UMT [42]|K700 [8]|81.7|96.0|89.9|70.1|50.1|**88.0**|26.4|18.8|28.2|
|SMILE (ours)|K400 [35]|83.1|**96.4**|**90.8**|**71.9**|**63.3**|83.8|**30.2**|**23.7**|**34.6**|



Table 11. **More comparison with CLIP adaptations.** SMILE learns better video representations than both types of CLIP adaptations: with and without intermediate pretraining. The performance gap is wider on motion-focused domains. 

Table 12. **Learning video representations with only object motions.** We train VideoMAE baseline on Kinetics-400 videos and ours with augmented clips generated by overlaying noise images with object motions. All settings train a ViT-S for 400 and ViT-B for 600 epochs. Our method trained without any natural videos lags only by a small margin compared to the VideoMAE baseline trained with natural videos from the Kinetics-400 dataset. 

|Method|Data|Target|K400|SSv2|GYM|
|---|---|---|---|---|---|
|ViT-S||||||
|No Pretrain.|-|-|65.7|52.2|55.1|
|Ours|Noise + Motion|Pixel|72.7|59.1|72.5|
|Ours|Noise + Motion|Features|73.7|61.0|74.6|
|VideoMAE|K400|Pixel|75.9|62.7|75.1|
|ViT-B||||||
|No Pretrain.|-|-|69.1|49.8|50.0|
|Ours|Noise + Motion|Pixel|74.5|60.0|77.2|
|Ours|Noise + Motion|Features|77.5|64.1|83.0|
|VideoMAE|K400|Pixel|79.0|67.0|86.6|



reconstruction shows a small gap in performance. This highlights the effectiveness of video representations learned only from the proposed object motions via unnatural videos created on the fly. The gap is further reduced when feature reconstruction is used instead of pixel reconstruction, demonstrating the impact of using CLIP feature projections over raw pixels, even for such unnatural data. Overall, our proposed synthetic object motions can act as a strong supervisory signal in a standalone to learn video representations with masked video modeling. We leave the scaling of such learning without natural videos for larger models to future works. 

## **E. Additional Experiments** 

### **E.1. Performance with fixed training budget** 

We now compare our method with VideoMAE [77] baseline for different dataset scales with a fixed training budget, _i.e._ total number of training iterations _n_ = _s × e_ , where _s_ is the dataset size and _e_ is the number of epochs. Using the ViT-S backbone, we pretrain on four subsets of K400, namely, 12.5%, 25%, 50%, and 100% of the full scale. For the whole dataset _s_ = 100%, we pretrain for 200 epochs; smaller subsets are trained for proportionally more epochs _i.e._ , _s_ = 50% for 400 epochs, _s_ = 25% for 800, and _s_ = 12 _._ 5% for 1600. Results in Figure 5 show that our method consistently outperforms the VideoMAE baseline by a large margin for various data scales under the same training budget. This highlights the robustness of our method to dataset size and training duration. 



<!-- Start of picture text -->
80 GYM (Ours)<br>GYM (VideoMAE)<br>SSv2 (Ours)<br>SSv2 (VideoMAE)<br>75<br>70<br>65<br>60<br>12.5% 25% 50% 100%<br>Dataset Scale<br>Top-1<br><!-- End of picture text -->

Figure 5. **Performance comparison with a fixed training budget.** We evaluate on SSv2 and GYM for full finetuning. Our method consistently outperforms VideoMAE [77] across all data scales with the same training budget. 

Table 13. **Full-Scale ablation.** Ablating our main contributions on a larger backbone and a bigger pretraining dataset, i.e., the original K400. Reconstructing features and adding synthetic motions shows consistent improvements for a larger backbone (ViT-B) and scaling to a bigger pretraining dataset. 

|Backbone|Target|Synth.|K400|SSv2|
|---|---|---|---|---|
|ViT-B|Pixels|w/o|78.3|67.2|
|ViT-B|Pixels|w/|78.9|67.7|
|ViT-B|Features|w/o|81.2|70.8|
|ViT-B|Features|w/|**81.7**|**71.2**|



### **E.2. Full-scale ablation** 

All main paper ablations are conducted with the smaller K400m pretraining and a ViT-S backbone. To reinforce the validity of our key contributions—feature target reconstruction and synthetic object motion, we extend the ablations to full-scale K400 pretraining using a larger ViT-B backbone. Full finetuning is performed on the complete K400 and SSv2 datasets. We adopt the best configurations from the small-scale ablations, including 80% masking, trajectory masking, two object overlays, CLIP feature reconstruction, and a 300-epoch training schedule unless stated otherwise. 

Table 13 presents the results for full-scale ablations. Consistent with the small-scale ablations in the main paper, incorporating synthetic motions boosts downstream performance. Specifically, our feature reconstruction improves the downstream performance by 2.9% on K400 and 3.6% on SSv2 over pixel reconstruction. By adding object motions, pixel reconstruction improves by 0.6% on K400 and 0.5% on SSv2, and feature reconstruction sees gains of 0.5% on K400 and 0.4% on SSv2. Our best configuration—feature reconstruction with synthetic motion—achieves the highest performance, reinforcing the robustness and scalability of our method across larger backbones and pretraining datasets. 

### **E.3. Qualitative analysis** 

In Figure 6, we extend the qualitative analysis to compare with more prior video SSL works. As before, we observe that the features of different frames have larger differences for our model, indicating better temporal awareness. In particular, our feature similarity is consistent with moth motion-aware methods like MGM, MGMAE and MME demonstrating the motion focus of our method too. 

## **F. Experimental Details** 

### **F.1. Datasets for main results** 

For linear probing and full finetuning SoTA experiments, we evaluate action recognition task with standard action recognition datasets _i.e._ Kinetics-400 [35] (K400), SomethingSomething V2 [24] (SSv2), UCF-101 [71] (UCF), 



<!-- Start of picture text -->
VideoMAE MME MVD MGM<br>SIGMA MGMAE EVEREST SMILE(Ours)<br><!-- End of picture text -->

Figure 6. **Feature similarity across different frames for different SSL methods.** We compute this on K400 validation videos. 

HMDB-51 [38] (HMDB), FineGYM [69] (GYM), and EPIC-Kitchens-100 [12] (EPIC). More details about the datasets are in Table 15 and following: 

**Kinetics-400 [35]** Kinetics-400 (K400) is a comprehensive benchmark designed for video action recognition tasks. It consists of over 306,000 concise video clips sourced from YouTube, spanning an impressive 400 distinct action categories. As one of the largest and most widely adopted datasets in this field, K400 plays a pivotal role in assessing and advancing cutting-edge models for understanding actions in video content. 

**SomethingSomething V2 [24]** SomethingSomething V2 (SSv2) is a collaboratively sourced dataset consisting of first-person video recordings, specifically crafted to facilitate the development of common-sense reasoning capabilities. In terms of visual composition and perspective, it markedly diverges from Kinetics-400. The dataset comprises 168,913 training samples and 24,777 testing samples, distributed across 174 unique action categories. 

**UCF-101 [71]** UCF-101 is a widely recognized benchmark in video self-supervised learning research. It comprises a diverse set of 9537 training and 3783 testing samples, sourced from YouTube videos grouped into 101 action categories, characterized by coarse granularity. Many of these categories show a substantial overlap with the action types included in Kinetics-400. 

**HMDB-51 [38]** HMDB-51 (HMDB) is a widely-used benchmark for action recognition research. It features a total of 6,766 video clips, carefully selected from a variety of sources, such as films, the Prelinger Archive, YouTube, and Google Videos. The dataset is categorized into 51 unique action classes, with each class comprising no fewer than 100 video samples. 

**FineGYM [69]** FineGYM (GYM) is a benchmark designed for fine-grained action analysis in gymnastics competitions. For our study, we specifically select the Gym-99 subset, which consists of 99 unique action categories. This subset provides 20,484 training samples and 8,521 testing samples. 

Table 14. **SEVERE benchmark.** Details of all the experimental subsets in the benchmark. We follow the configurations from the original work [75]. 

|**Evaluation Setup**|**Experiment **|**Dataset**|**Task**|**#Classes **|**#Finetuning **|**#Testing **|**Eval Metric**|
|---|---|---|---|---|---|---|---|
||Gym99|FineGym [69]|Action Class.|99|20,484|8,521|Top-1 Acc.|
|**Sample Efficiency**|UCF (10<sup>3</sup>)<br>Gym (10<sup>3</sup>)|UCF 101 [71] <br>FineGym [69]|Action Class.<br> Action Class.|101<br>99|1,000<br>1,000|3,783 <br>8,521|Top-1 Acc.<br> Top-1 Acc.|
|**Action Granularit**|<sup>FX-S1</sup>|FineGym [69]|Action Class.|11|1,882|777|Mean-per-class|
|**y**|<br>UB-S1|FineGym [69]|Action Class.|15|3,511|1,471|Mean-per-class|
|**Task Shift**|UCF-RC<br>Charades|UCFRep [94]<br>Charades [70]|Repetition Counting<br> Multi-label Class.|-<br>157|421<br>7,985|105 <br>1,863|Mean Error<br> mAP|



Table 15. **Datasets.** Details of the datasets used for evaluation showing the corresponding number of classes, training, and testing samples for each. 

|Dataset|#Classes|#Train|#Test|
|---|---|---|---|
|K400|400|240K|19K|
|UCF|101|9.5K|3.8K|
|HMDB|51|4.8K|2K|
|SSv2|174|169K|24.8K|
|GYM|99|20.5K|8.5K|
|EPIC|97|67.2K|9.7K|



Table 16. **Linear-Evaluation setting.** 

|config|K400|UCF|HMDB SSv2|GYM|EPIC|
|---|---|---|---|---|---|
|optimizer|||AdamW[47]|||
|base learning rate|||1.e-3|||
|weight decay|||0.05|||
|optimizer momentum||_β_|1_, β_2 = 0_._9_,_0_._9|99||
|layer-wise lr decay [3]|||0.75|||
|batch size|||128|||
|learning rate schedule|||cosine decay|||
|training epochs|30|100|100<br>50|100|100|
|flip augmentation|_yes_|_yes_|_yes_<br>_no_|_yes_|_yes_|



**EPIC-Kitchens-100 [12]** EPIC-Kitchens-100 (EPIC) is a large egocentric dataset capturing daily kitchen activities, annotated with 97 verbs and 300 nouns, where actions are defined as combinations of both. Similar to SS-v2, EK-100 differs significantly from Kinetics-400 in its unique visual style and first-person perspective. Using the standard splits provided in [13], it includes 67,217 training samples and 9,668 for validation, with our study focusing solely on recognizing the 97 verb classes. 

### **F.2. Datasets for SEVERE Benchmark** 

**SEVERE Benchmark[75]** SEVERE-Benchmark spans eight experimental settings across four datasets _i.e._ Something-Something V2, UCF, FineGYM, and Charades [70]. Table 14 provides detailed configurations for each subset. 

Table 17. **Full finetuning evaluation setup.** 

|config|SSv2|K400|SEVERE|
|---|---|---|---|
|optimizer||AdamW||
|base learning rate||1.0e-3||
|weight decay||0.05||
|optimizer momentum|_β_1|_, β_2 = 0_._9_,_0_._9|99|
|layer-wise lr decay[3]||0.75||
|batch size|32|16|16|
|learning rate schedule||cosine decay||
|warmup epochs||5||
|training epochs|40|100|100|
|flip augmentation|_no_|_yes_|_yes_|
|RandAug [11]||(9,0.5)||
|label smoothing[73]<br>mixup [93]||0.1<br>0.8||
|cutmix [91]||1.0||
|drop path||0.1||



### **F.3. Training and Evaluation Details** 

**Pretraining details.** We pretrain on Kinetics-400 (K400) [35] and Something-Something V2 (SSv2) [24] datasets. To generate our segmented object set _O_ , we follow [14] to utilize Stable Diffusion [66] and X-paste [96], generating 60 samples for each of the 1203 categories in the LVIS dataset [25]. Following VideoMAE [77] we use a temporal stride of 2 for SSv2 and a stride of 4 for K400. Each clip contains 16 frames sampled at a resolution of 224 _×_ 224 pixels. Space-time tube embeddings are extracted 

using a 3D convolution layer, treating each 2 _×_ 16 _×_ 16 cube as a token. We use both tube and trajectory masking with a ratio _m_ = 80%. We employ multiple sampling based on [28] during the pretraining which effectively samples two input clips from the same video for reconstruction. This decreases the training time by almost half without any performance drop. We always count epochs as “effective epochs = _No. of epochs × No. of samples per video_ ”, i.e., how many times each video is sampled and processed 

Table 18. **Pretraining details.** 

|config|SSv2<br>K400|
|---|---|
|optimizer|AdamW|
|base learning rate|1.5e-4|
|weight decay|0.05|
|optimizer momentum|_β_1_, β_2 = 0_._9_,_0_._95|
|batch size|256|
|learning rate schedule<br>warmup epochs|cosine decay<br>40|
|flip augmentation|_no_<br>_yes_|
|augmentation|MultiScaleCrop|
|Epochs|800<br>600|



throughout training. We employ our progressive pretraining strategy for 600 epochs on K400 and 800 on SSv2, 300 and 400 in each stage, respectively. Table 18 shows the rest of the configuration. We train our models with 8 NVIDIA V100 GPUs. For downstream evaluation, we only use the student network without its decoder and attach a task-dependent head to the pretrained student encoder e.g., a classification layer for action recognition. 

**Linear Probing details.** Table 16 shows the settings for linear probing. We use 4 NVIDIA V100 GPUs for linear probing. 

**Full Finetuning details.** Table 17 shows the settings for full finetuning, following [77]. We use 4 NVIDIA V100 GPUS for fine-tuning. 

**SEVERE Benchmark evaluation.** We compare our method to recent masked video modeling approaches, using the SEVERE codebase [75] and keeping identical training and evaluation setups for fair comparison. Official models for each comparative method are used. 

